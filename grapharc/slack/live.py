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
from typing import Any, Protocol

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

    `update` takes optional Block Kit `blocks` because a parked run's message
    carries Approve/Deny buttons. `None` means "this message is text" — and,
    for a sink editing in place, means *drop* any blocks it had: the edit that
    says a plan was approved must also take its Approve button away.
    """

    def post(self, text: str) -> object | None: ...

    def update(
        self, handle: object, text: str, blocks: list[dict[str, Any]] | None = None
    ) -> bool: ...


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
        update: Callable[..., bool],
        settings: LiveSettings | None = None,
        *,
        view_url: str | None = None,
        workdir: Path | None = None,
        mutating_kinds: frozenset[str] | None = frozenset(),
    ) -> None:
        self._path = trace_path
        self._argv = list(argv)
        self._update = update
        self._view_url = view_url
        self._workdir = workdir or Path.cwd()
        self._mutating_kinds = mutating_kinds
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
                rendered = self._render(run_id)
                if rendered is None:
                    continue
                text, prompt = rendered
                dirty = False
                last_update = time.monotonic()
                if text == last_text:
                    continue
                if not self._update(text, prompt):
                    return  # the sink is dead; go quiet, never raise
                last_text = text
            except Exception:
                # Live narration is best-effort; the run and the final result
                # do not depend on it.
                return

    def _render(self, run_id: str) -> tuple[str, ApprovalPrompt | None] | None:
        # TailRecorder, not TraceRecorder: a read can land mid-write, and the
        # plain reader raises on the torn line where this one skips it.
        recorder = TailRecorder(self._path)
        try:
            run = replay(recorder, run_id)
            diagram = to_mermaid(recorder, run_id)
        except Exception:
            return None  # e.g. no complete events yet; retry next tick
        text = render_progress(
            run,
            argv=self._argv,
            elapsed_s=time.monotonic() - self._started_at,
            diagram=diagram,
            view_url=self._view_url,
            mutating_kinds=self._mutating_kinds,
        )
        return text, pending_approval_prompt(run, self._argv, self._workdir)

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


def plan_lines(
    approval: TraceEvent, *, mutating_kinds: frozenset[str] | None = frozenset()
) -> list[str]:
    """The proposed graph as text: what a human is being asked to approve.

    Read from the `approval_request` event alone, which carries every field
    this needs (`loop._request_approval`). A link is not an answer here — the
    person deciding is on a phone, and "the graph is over there" asks them to
    approve something they have not seen. So the shape goes in the message.

    Kinds, not just names: `fix_it` is a name a planner chose and `apply_change`
    is the thing the gate actually governs. A node whose kind can change files
    is marked ✎, because "will this touch my repo" is the question being asked.

    `mutating_kinds=None` means the registry declared nothing about which of
    its kinds mutate. Every node is then marked, matching the reading `plan`
    already takes for its own plan file: undeclared is not evidence of safety.
    """
    delta = approval.state_delta or {}
    names = [str(n) for n in delta.get("nodes", [])]
    kinds = [str(k) for k in delta.get("kinds", [])]
    lines: list[str] = []
    if mutating_kinds is None and names:
        lines.append("⚠ this registry does not say which kinds change things — assume all do")

    rationale = str(delta.get("rationale") or "").strip()
    if rationale:
        lines.append(f"why: {' '.join(rationale.split())[:240]}")

    if names:
        lines.append(f"{len(names)} node{'s' if len(names) != 1 else ''}:")
        for index, name in enumerate(names):
            kind = kinds[index] if index < len(kinds) else ""
            # A kind equal to the name adds nothing — `ProposedNode` defaults
            # `kind` to `name`, and printing "verify (verify)" is noise.
            label = f"{name} ({kind})" if kind and kind != name else name
            mark = "✎" if mutating_kinds is None or kind in mutating_kinds else "·"
            lines.append(f"  {mark} {label}")

    edges = delta.get("edges", [])
    if edges:
        lines.append(f"{len(edges)} edge{'s' if len(edges) != 1 else ''}:")
        for edge in edges:
            try:
                source, target = edge[0], edge[1]
            except (TypeError, IndexError, KeyError):
                continue
            lines.append(f"  {source} → {target}")

    tokens = delta.get("worst_case_tokens")
    if tokens:
        estimate = f"worst case: {tokens} tok"
        if not delta.get("worst_case_complete", True):
            # The flag exists so nobody reads an incomplete number as complete.
            estimate += " (lower bound — an unregistered kind has no price)"
        lines.append(estimate)

    if not lines:
        # An empty proposal is legal and means "no further work". Saying so is
        # better than an empty box that reads as a rendering failure.
        lines.append("(an empty plan — the planner proposes no further work)")
    return lines


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
    mutating_kinds: frozenset[str] | None = frozenset(),
) -> str:
    """One Slack message describing the run so far.

    A parked run shows the *plan* — the graph it is asking permission for.
    Otherwise: declared-graph node marks when the trace carries topology; node
    executions when the run has them; otherwise the tail of the flat event feed
    — an agent with no enclosing graph is all flat feed, and an empty box
    saying nothing was the alternative.
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
    if approval is not None:
        # The proposed graph, not the node marks: every node of a parked plan
        # is pending by definition, so a column of ⬜ says nothing about what
        # is being authorised. The shape and the kinds are the question.
        lines = plan_lines(approval, mutating_kinds=mutating_kinds)
    elif planned is not None:
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
        parts.append("*nothing above has run yet* — it runs only if you approve.")
        trace_arg = _trace_argument(argv)
        if trace_arg:
            # The typed form stays documented in the message even when the
            # buttons render: a workspace without interactivity enabled, a
            # client that will not draw blocks, and a stale message whose
            # buttons the fingerprint check will refuse all end up here.
            parts.append(
                f"buttons below, or: `/grapharc approve {trace_arg}` · "
                f"`/grapharc approve {trace_arg} --deny`"
            )
    # The operator's own live view is the primary link; the mermaid.live
    # fragment link is the fallback for a bot with no live server configured.
    if view_url:
        parts.append(f"<{view_url}|open live view>")
    elif diagram:
        parts.append(f"<{mermaid_live_url(diagram)}|current diagram>")
    return "\n".join(parts)


@dataclass(frozen=True)
class ApprovalPrompt:
    """What a pending approval needs from the person deciding.

    `directory` is where the file handshake lives — relative to the bot's
    working directory, because that is the only form safe to send to Slack and
    back: an absolute path in a button's value is a path the round trip could
    rewrite into somewhere else. `bot.py` re-confines it on the way in
    regardless; this just keeps the value uninteresting to tamper with.

    `fingerprint` is what makes the button honest. It names the exact proposal
    that was displayed, and `approve` refuses a decision quoting any other — so
    a button on a scrolled-back message cannot approve the plan that replaced
    the one it was drawn for.
    """

    directory: str
    fingerprint: str
    round_number: int = 0


def pending_approval_prompt(
    run: ReplayedRun, argv: list[str], workdir: Path
) -> ApprovalPrompt | None:
    """The prompt for `run`'s unanswered approval request, or None."""
    approval = _pending_approval(run)
    if approval is None:
        return None
    trace_arg = _trace_argument(argv)
    if not trace_arg:
        return None
    delta = approval.state_delta or {}
    fingerprint = str(delta.get("fingerprint") or "")
    if not fingerprint:
        # Without a fingerprint there is nothing to bind a decision to, and an
        # unbound approve button is exactly the thing this design refuses to
        # ship. Fall back to the typed command, which reads the request file.
        return None
    directory = Path(trace_arg).parent
    try:
        relative = directory.as_posix() if not directory.is_absolute() else str(
            directory.relative_to(workdir)
        )
    except ValueError:
        return None
    return ApprovalPrompt(
        directory=relative,
        fingerprint=fingerprint,
        round_number=int(delta.get("round") or 0),
    )


def _trace_argument(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--trace" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--trace="):
            return token.partition("=")[2]
    return None
