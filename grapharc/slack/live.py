"""Tail a trace file while a command runs and narrate it through a callback.

The CLI subprocess appends one JSON line per event to its trace (`TraceRecorder`
writes under a lock, append-only), so the file is complete and readable at any
instant. `LiveTail` polls it from a daemon thread, reconstructs the run with
`replay` — the same tested reconstruction `metrics` and `viz` use, never a
private dialect — and hands a rendered progress message to an `update`
callback whenever something changed.

Everything here is best-effort by contract: the whole loop body runs inside a
`try/except` and an `update` that fails marks the sink dead. A live view that
breaks must degrade to today's behaviour (the final result posted once at the
end), never take the run down with it. This module imports nothing from Slack —
the callback is a plain callable, which is also what makes it testable without
a token.
"""

from __future__ import annotations

import json
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from grapharc.observe.metrics import to_mermaid
from grapharc.observe.replay import NodeExecution, ReplayedRun, replay
from grapharc.observe.status import NodeState, node_states
from grapharc.observe.trace import TailRecorder, TraceEvent
from grapharc.slack.format import fence, mermaid_live_url, truncate

#: How many trailing sub-step events the flat feed shows for a run with no
#: node executions (`grapharc agent` drives an `AgentNode` with no enclosing
#: graph, so everything it does is an orphan sub-event).
FLAT_FEED_LINES = 15


class LiveSink(Protocol):
    """Where live progress goes. `bot.py` backs this with Slack; tests with a list.

    Both methods signal failure with a value, never an exception: `post`
    returns None (the caller falls back to the plain blocking reply), `update`
    returns False (the tailer goes quiet). A sink can be handed an unreliable
    network and the run must not care.
    """

    def post(self, text: str) -> object | None: ...

    def update(self, handle: object, text: str) -> bool: ...


@dataclass(frozen=True)
class LiveSettings:
    """Cadence of the tail loop."""

    # Floor between two `update` calls; chat.update sits in a rate tier.
    update_interval: float = 2.5
    # How often the file's size is polled. Cheap (one stat), so more often
    # than updates: growth is noticed promptly, rendering waits its turn.
    poll_interval: float = 1.0
    # How long __exit__ waits for the thread; a stuck render must not delay
    # the final result by more than this.
    join_timeout: float = 5.0


class LiveTail:
    """Context manager: tail `trace_path` on a daemon thread while the body runs.

    Construct it *before* the subprocess starts: the constructor records the
    file's current size, and only bytes appended after that are read — a
    reused trace file (the agent workspace is constant, so its trace
    accumulates runs) must not replay some earlier run as this one's progress.
    """

    def __init__(
        self,
        trace_path: Path,
        argv: list[str],
        update: Callable[[str], bool],
        settings: LiveSettings | None = None,
        *,
        view_url: str | None = None,
    ) -> None:
        self._path = trace_path
        self._argv = list(argv)
        self._update = update
        self._view_url = view_url
        self._settings = settings or LiveSettings()
        try:
            self._offset = trace_path.stat().st_size
        except OSError:
            self._offset = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._started_at = time.monotonic()

    def __enter__(self) -> LiveTail:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Stop and join *before* the caller posts the final result, so a late
        # tick can never overwrite it.
        self._stop.set()
        self._thread.join(timeout=self._settings.join_timeout)

    def _loop(self) -> None:
        run_id: str | None = None
        last_update = 0.0
        last_text = ""
        dirty = False
        while not self._stop.wait(self._settings.poll_interval):
            try:
                new_id = self._read_new_run_id()
                if new_id is not None:
                    run_id = new_id
                    dirty = True
                if run_id is None or not dirty:
                    continue
                if time.monotonic() - last_update < self._settings.update_interval:
                    continue
                text = self._render(run_id)
                if text is None:
                    continue
                dirty = False
                last_update = time.monotonic()
                if text == last_text:
                    continue
                if not self._update(text):
                    return  # the sink is dead; go quiet, never raise
                last_text = text
            except Exception:
                # Live narration is best-effort; the run and the final result
                # do not depend on it.
                return

    def _render(self, run_id: str) -> str | None:
        # TailRecorder, not TraceRecorder: a read can land mid-write, and the
        # plain reader raises on the torn line where this one skips it.
        recorder = TailRecorder(self._path)
        try:
            run = replay(recorder, run_id)
            diagram = to_mermaid(recorder, run_id)
        except Exception:
            return None  # e.g. no complete events yet; retry next tick
        return render_progress(
            run,
            argv=self._argv,
            elapsed_s=time.monotonic() - self._started_at,
            diagram=diagram,
            view_url=self._view_url,
        )

    def _read_new_run_id(self) -> str | None:
        """Fold newly appended complete lines; return the latest run_id seen.

        Returns None when nothing (complete) was appended — so a non-None
        return doubles as "the file grew".
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            return None
        if size < self._offset:
            # Truncated or replaced underneath us — the offset describes a
            # file that no longer exists. Start over from byte zero (the same
            # bet `TraceRecorder._advance_index` makes); staying put meant
            # permanent silence until the new file outgrew the old offset,
            # then a seek into the middle of a line.
            self._offset = 0
        if size == self._offset:
            return None
        with self._path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        # Stop at the last newline: the writer may be mid-line, and half a
        # JSON object is not an event yet.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return None
        self._offset += cut + 1
        run_id = None
        for raw in chunk[: cut + 1].splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            candidate = record.get("run_id")
            if isinstance(candidate, str) and candidate:
                run_id = candidate
        return run_id


def _mark(execution: NodeExecution) -> str:
    if execution.error is not None:
        return "✗"
    return "✓" if execution.completed else "▸"


def _duration(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _execution_line(execution: NodeExecution) -> str:
    parts = [_mark(execution), f" {execution.node}"]
    if execution.error is not None:
        detail = " ".join(execution.error.split())[:80]
        parts.append(f"  err: {detail}")
    elif not execution.completed:
        parts.append("  running…")
    else:
        if execution.duration_ms is not None:
            parts.append(f"  {_duration(execution.duration_ms)}")
        if execution.tokens:
            parts.append(f"  {execution.tokens} tok")
        if execution.sub_events:
            parts.append(f"  · {len(execution.sub_events)} sub-steps")
    return "".join(parts)


def _sub_event_line(event: TraceEvent) -> str:
    parts = [f"{event.phase}  {event.node}"]
    if event.tokens:
        parts.append(f"  {event.tokens} tok")
    if event.error:
        parts.append(f"  err: {' '.join(event.error.split())[:80]}")
    return "".join(parts)


#: Phases that describe the run rather than doing work; never shown as feed.
_SHAPE_PHASES = frozenset({"topology", "approval_request", "approval_response"})


def _goal(run: ReplayedRun) -> str | None:
    """The goal the loop recorded on its topology/approval events, if any."""
    found: str | None = None
    for event in run.events:
        if event.phase in ("topology", "approval_request"):
            value = (event.state_delta or {}).get("goal")
            if value:
                found = str(value)
    return found


def _pending_approval(run: ReplayedRun) -> TraceEvent | None:
    """The latest approval request no response has answered yet, if any."""
    pending: TraceEvent | None = None
    for event in run.events:
        if event.phase == "approval_request":
            pending = event
        elif event.phase == "approval_response":
            pending = None
    return pending


def _planned_lines(run: ReplayedRun) -> list[str] | None:
    """One line per *declared* node, marked by status — or None without topology.

    The declared graph is what makes a live view honest about scope: a node
    that has not started yet is shown as pending rather than not shown at all.
    Multi-round planner runs list each round's graph in order, and status is
    read from that round's *own* events — keyed by bare name, round 1's
    finished node once wore round 2's still-running state.

    Deltas are merged per graph (the loop's labelled statement plus the
    kernel's bare restatement), so the round label survives execution.
    """
    topologies: dict[str, dict] = {}
    for event in run.events:
        if event.phase == "topology" and event.state_delta:
            topologies.setdefault(event.graph, {}).update(event.state_delta)
    if not topologies:
        return None
    lines: list[str] = []
    for graph, delta in topologies.items():
        if len(topologies) > 1:
            round_no = delta.get("round")
            lines.append(f"round {round_no}:" if round_no else f"{graph}:")
        graph_events = [e for e in run.events if e.graph == graph]
        states = node_states(graph_events)
        for name in delta.get("nodes", []):
            lines.append(_node_status_line(str(name), states.get(str(name))))
    return lines


def _node_status_line(name: str, state: NodeState | None) -> str:
    """One mark per declared node, from the shared `observe.status` rule."""
    status = state.status if state is not None else "pending"
    if status == "errored":
        error = state.last_error.error if state.last_error else None
        detail = " ".join((error or "error").split())[:80]
        return f"✗ {name}  err: {detail}"
    if status == "running":
        return f"▸ {name}  running…"
    if status == "done":
        parts = [f"✓ {name}"]
        last_end = state.last_end
        if last_end is not None:
            if last_end.duration_ms is not None:
                parts.append(f"  {_duration(last_end.duration_ms)}")
            if last_end.tokens:
                parts.append(f"  {last_end.tokens} tok")
        return "".join(parts)
    return f"⬜ {name}  pending"


def render_progress(
    run: ReplayedRun,
    *,
    argv: list[str],
    elapsed_s: float,
    diagram: str | None = None,
    view_url: str | None = None,
) -> str:
    """One Slack message describing the run so far.

    Declared-graph node marks when the trace carries topology; node executions
    when the run has them; otherwise the tail of the flat event feed — an
    agent with no enclosing graph is all flat feed, and an empty box saying
    nothing was the alternative.
    """
    approval = _pending_approval(run)
    if approval is not None:
        header = (
            f"`{shlex.join(['grapharc', *argv])}` — "
            f"⏸ waiting for approval ({elapsed_s:.0f}s)"
        )
    else:
        header = f"`{shlex.join(['grapharc', *argv])}` — running ({elapsed_s:.0f}s)"
    goal = _goal(run)
    if goal:
        header += f" · {goal[:80]}"

    planned = _planned_lines(run)
    if planned is not None:
        lines = planned
    elif run.executions:
        lines = [_execution_line(e) for e in run.executions]
    else:
        feed = [e for e in run.orphan_sub_events if e.phase not in _SHAPE_PHASES]
        hidden = len(feed) - len(feed[-FLAT_FEED_LINES:])
        feed = feed[-FLAT_FEED_LINES:]
        lines = [_sub_event_line(e) for e in feed]
        if hidden > 0:
            lines.insert(0, f"… {hidden} earlier events")

    body, cut = truncate("\n".join(lines))
    parts = [header, fence(body)]
    if cut:
        parts.append(f"_…{cut} earlier characters not shown._")

    done = sum(1 for e in run.executions if e.completed)
    # `run.events` already holds every event, orphans included.
    footer = [f"{len(run.events)} events"]
    if run.executions:
        footer.append(f"{done}/{len(run.executions)} nodes done")
    if run.tokens:
        footer.append(f"{run.tokens} tok")
    # `recorded_cost_usd`, not a raw event sum: an agent's model sub-events
    # and its node's end both carry the same spend, and summing every event
    # reported double the real bill.
    cost = run.recorded_cost_usd
    if cost:
        footer.append(f"${cost:.4f}")
    parts.append(" · ".join(footer))

    if approval is not None:
        trace_arg = _trace_argument(argv)
        if trace_arg:
            where = "live view" if view_url else "diagram"
            parts.append(
                f"planned graph is in the {where} link — approve with "
                f"`/grapharc approve {trace_arg}`, refuse with "
                f"`/grapharc approve {trace_arg} --deny`"
            )
    # The operator's own live view is the primary link; the mermaid.live
    # fragment link is the fallback for a bot with no live server configured.
    if view_url:
        parts.append(f"<{view_url}|open live view>")
    elif diagram:
        parts.append(f"<{mermaid_live_url(diagram)}|current diagram>")
    return "\n".join(parts)


def _trace_argument(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--trace" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--trace="):
            return token.partition("=")[2]
    return None
