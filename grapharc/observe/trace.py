"""File-first run traces.

Every node execution is recorded as one JSON line: which node ran, in which
run, with which state delta, how long it took, what it cost, and how it ended.
Traces are the replay points and audit trail the production checklist demands —
human-readable, greppable, git-versionable. No hidden state.

Everything else in `grapharc.observe` is derived from this file and nothing
else — `metrics` counts it, `replay` reconstructs a run from it, `cost`
attributes spend from it, `otel` turns it into spans. That is deliberate: a
number no one can find in the audit trail is a number the audit trail can
contradict.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

_MAX_VALUE_CHARS = 2000


class TraceReadError(Exception):
    """A line in a trace file is not a `TraceEvent`.

    Raised instead of skipped on purpose: a partially-read audit trail
    presented as complete would be worse than a refusal. The message names the
    file and the 1-based line so the one bad line — a process killed mid-write,
    a hand edit — can be found without a pydantic field listing.
    """

    def __init__(self, path: Path, line_number: int, cause: Exception) -> None:
        super().__init__(
            f"unreadable trace file: {path}: line {line_number} is not a trace event"
        )
        self.path = path
        self.line_number = line_number
        self.cause = cause


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to something json.dumps accepts, truncating long text."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + f"…[truncated {len(value) - _MAX_VALUE_CHARS} chars]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:_MAX_VALUE_CHARS]


class TraceEvent(BaseModel):
    ts: str
    run_id: str
    thread_id: str | None = None
    attempt: int = 1
    graph: str
    node: str
    # The kernel's node wrapper emits "start" | "end" | "error". A node may emit
    # its own sub-step phases on the same recorder (AgentNode uses "model",
    # "tool", "stop"); readers must treat the set as open, not exhaustive.
    phase: str
    step: int
    state_delta: dict[str, Any] | None = None
    duration_ms: float | None = None
    tokens: int | None = None
    # Added after the original format, both optional and both written only when
    # a producer supplies them: `record` excludes None, so a trace from a
    # producer that does not report cost is byte-identical to one written
    # before these fields existed, and old trace files still parse.
    cost_usd: float | None = None
    model: str | None = None
    error: str | None = None


class TraceRecorder:
    """Append-only JSONL trace writer with a read-back helper for tests and the CLI."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Incremental index behind `thread_summary`: how many bytes of the file
        # have been folded in, and the per-thread maxima folded from them.
        self._indexed_bytes = 0
        self._thread_max: dict[str, tuple[int, int]] = {}

    def record(self, event: TraceEvent) -> None:
        line = json.dumps(_jsonable(event.model_dump(exclude_none=True)), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(
        self,
        *,
        run_id: str,
        graph: str,
        node: str,
        phase: str,
        step: int,
        thread_id: str | None = None,
        attempt: int = 1,
        state_delta: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
        model: str | None = None,
        error: str | None = None,
    ) -> None:
        self.record(
            TraceEvent(
                ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                graph=graph,
                node=node,
                phase=phase,
                step=step,
                state_delta=_jsonable(state_delta) if state_delta is not None else None,
                duration_ms=duration_ms,
                tokens=tokens,
                cost_usd=cost_usd,
                model=model,
                error=error,
            )
        )

    def thread_summary(self, thread_id: str) -> tuple[int, int]:
        """Return (max_step, max_attempt) recorded for a thread — (0, 0) if none.

        Used to seed a resumed run's counters so (thread_id, step) stays unique
        across attempts and the audit trail of one logical thread is stitchable.

        Called once per `invoke`, which is why it is incremental: it folds only
        the bytes appended since the last call, so a long-lived thread costs
        O(events) over its life instead of O(events²). Lines written by another
        process are picked up the same way — the index is keyed on the file's
        length, not on this recorder's own writes.

        The one thing the offset cannot survive is a file replaced in place by
        a *longer* different file: shrinking is detected and rescanned, growing
        is assumed to be an append. Traces are append-only by contract, so this
        is the same bet every log tailer makes.
        """
        with self._lock:
            self._advance_index()
            return self._thread_max.get(thread_id, (0, 0))

    def _advance_index(self) -> None:
        """Fold every complete line appended since the last fold. Caller holds the lock."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            self._indexed_bytes, self._thread_max = 0, {}
            return
        if size < self._indexed_bytes:
            # Truncated or replaced underneath us: the offset and the maxima
            # both describe a file that no longer exists, so start over.
            self._indexed_bytes, self._thread_max = 0, {}
        if size == self._indexed_bytes:
            return
        with self.path.open("rb") as f:
            f.seek(self._indexed_bytes)
            chunk = f.read(size - self._indexed_bytes)
        # Stop at the last newline and leave the remainder unconsumed: a writer
        # in another process may be mid-line, and half a JSON object indexed as
        # if it were whole would corrupt the counters a resume is seeded from.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        for raw in chunk[: cut + 1].splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            thread = record.get("thread_id")
            if thread is None:
                continue
            step, attempt = self._thread_max.get(thread, (0, 0))
            self._thread_max[thread] = (
                max(step, int(record.get("step", 0))),
                max(attempt, int(record.get("attempt", 1))),
            )
        self._indexed_bytes += cut + 1

    def read_events(self, run_id: str | None = None) -> list[TraceEvent]:
        """Every event in the file, in the order it was appended.

        File order, not sort order: the recorder appends under a lock, so the
        sequence on disk is the order the events were observed. Step numbers do
        not reproduce it — a node's `end` carries the same step as its `start`,
        while sub-steps emitted inside that node carry later ones.
        """
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    ev = TraceEvent.model_validate_json(line)
                except ValidationError as exc:
                    raise TraceReadError(self.path, line_number, exc) from exc
                if run_id is None or ev.run_id == run_id:
                    events.append(ev)
        return events

    def run_ids(self) -> list[str]:
        """Distinct run ids in the file, first-seen order."""
        return list(dict.fromkeys(ev.run_id for ev in self.read_events()))


class TailRecorder(TraceRecorder):
    """Read-only recorder over a file another process may be mid-write in.

    `TraceRecorder.read_events` validates every line and raises on a torn one —
    correct for a file the reader owns, wrong for one being appended to right
    now. This override reads bytes, cuts at the last newline, and skips
    anything that does not parse: the same discipline `_advance_index` applies.
    It also skips the parent constructor's mkdir — a reader must not create
    directories for a path that may come from a request, and the file it names
    may simply not exist yet.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._indexed_bytes = 0
        self._thread_max: dict[str, tuple[int, int]] = {}

    def record(self, event: TraceEvent) -> None:
        raise RuntimeError("TailRecorder is read-only")

    def read_events(self, run_id: str | None = None) -> list[TraceEvent]:
        try:
            raw = self.path.read_bytes()
        except OSError:
            return []
        cut = raw.rfind(b"\n")
        if cut < 0:
            return []
        events = []
        for line in raw[: cut + 1].splitlines():
            if not line.strip():
                continue
            try:
                event = TraceEvent.model_validate_json(line)
            except ValueError:
                continue
            if run_id is None or event.run_id == run_id:
                events.append(event)
        return events


def load_events(
    source: TraceRecorder | str | Path, run_id: str | None = None
) -> list[TraceEvent]:
    """Read a trace from a recorder or a path, so callers can pass either."""
    recorder = source if isinstance(source, TraceRecorder) else TraceRecorder(source)
    return recorder.read_events(run_id)


__all__ = ["TailRecorder", "TraceEvent", "TraceReadError", "TraceRecorder", "load_events"]
