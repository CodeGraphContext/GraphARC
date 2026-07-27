"""File-first run traces.

Every node execution is recorded as one JSON line: which node ran, in which
run, with which state delta, how long it took, what it cost, and how it ended.
Traces are the replay points and audit trail the production checklist demands —
human-readable, greppable, git-versionable. No hidden state.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_MAX_VALUE_CHARS = 2000


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
    phase: str  # "start" | "end" | "error"
    step: int
    state_delta: dict[str, Any] | None = None
    duration_ms: float | None = None
    tokens: int | None = None
    error: str | None = None


class TraceRecorder:
    """Append-only JSONL trace writer with a read-back helper for tests and the CLI."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
                error=error,
            )
        )

    def thread_summary(self, thread_id: str) -> tuple[int, int]:
        """Return (max_step, max_attempt) recorded for a thread — (0, 0) if none.

        Used to seed a resumed run's counters so (thread_id, step) stays unique
        across attempts and the audit trail of one logical thread is stitchable.
        """
        max_step = max_attempt = 0
        for ev in self.read_events():
            if ev.thread_id == thread_id:
                max_step = max(max_step, ev.step)
                max_attempt = max(max_attempt, ev.attempt)
        return max_step, max_attempt

    def read_events(self, run_id: str | None = None) -> list[TraceEvent]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev = TraceEvent.model_validate_json(line)
                if run_id is None or ev.run_id == run_id:
                    events.append(ev)
        return events
