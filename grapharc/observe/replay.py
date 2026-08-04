"""Replay: reconstruct a run from its trace, and diff two runs of one graph.

The trace has been called a set of "replay points" since the first commit and
nothing replayed it. This module is what makes the name true — and it is a
*reconstruction*, not a re-execution: it reads the JSONL file and rebuilds the
node sequence, the state deltas, the timing and the failures that were
recorded. Nothing here calls a model, a tool, or a node.

What is reconstructed exactly, and what is not:

- **Sequence** — file order, which is the order the recorder observed. Step
  numbers cannot stand in for it: a node's `end` event carries the same step as
  its `start`, while sub-steps emitted *inside* that node carry later ones, so
  sorting by step would put a node's end before the work it contains.
- **Executions** — `start` and its matching `end`/`error` are paired on
  (node, step, attempt). A node killed by the budget check emits only `error`
  and still becomes an execution, because "it never started" is a fact about
  the run that a reader needs.
- **State** — the fold of the recorded deltas, and no better than the deltas
  are. Two limits, both from the recording side: `_jsonable` truncates a string
  past 2000 chars at write time, so a long value replays truncated; and the
  trace does not record which state fields have reducers, so a field LangGraph
  appended to replays as last-write-wins unless the caller passes a reducer for
  it. `replay_state` says so in its signature rather than in a comment nobody
  reads.
- **Timing** — durations are the recorded `duration_ms`, measured around the
  node body. Wall time spans the first and last event timestamps, which are
  millisecond-resolution, so it includes scheduling gaps and cannot resolve
  anything finer than a millisecond.

`diff_runs` aligns two runs' node sequences with `difflib` and reports where
they diverged: which nodes only one run ran, which shared nodes wrote different
deltas, and how the tokens and durations moved. It is the "did my change alter
behaviour" question answered off the audit trail — `diff_trace` for two runs in
one file, `diff_runs` for two runs reconstructed from wherever.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from grapharc.observe.trace import TraceEvent, TraceRecorder, load_events

# Phases the kernel's node wrapper emits. The set is closed *here* and open on
# the recorder: any other phase was emitted by a node about its own internals
# (AgentNode uses "model"/"tool"/"stop"; the planner "plan"; admission
# "admission") and is attributed *inside* the enclosing execution, never beside
# it — otherwise a sub-step would count as an extra node execution and every
# number derived from the trace would inflate.
NODE_PHASES = frozenset({"start", "end", "error"})
TERMINAL_PHASES = frozenset({"end", "error"})


class ReplayError(Exception):
    """A trace cannot answer the question asked of it."""


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


class NodeExecution(BaseModel):
    """One node execution, rebuilt from its start/end (or error) pair."""

    node: str
    step: int
    attempt: int
    # Position of the terminal event in the trace file: the run's true ordering
    # key, and what `diff_runs` reports so a finding can be grepped back.
    index: int
    # The `start` event's timestamp — or the terminal event's, for a node the
    # budget check refused, which never got a `start` to be stamped from.
    started_at: str
    ended_at: str | None = None
    duration_ms: float | None = None
    tokens: int = 0
    cost_usd: float | None = None
    state_delta: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    # False when the execution ended in `error`, and also when the trace holds
    # no terminal event at all — a run cut off mid-node leaves exactly that.
    ok: bool = False
    completed: bool = False
    sub_events: list[TraceEvent] = Field(default_factory=list)

    @property
    def sub_tokens(self) -> int:
        """Tokens reported by sub-steps inside this node.

        A *breakdown of* `tokens`, not an addition to it: the kernel charges the
        node's meter for model calls made inside it, so the same spend appears
        once here and once in the node total.
        """
        return sum(e.tokens or 0 for e in self.sub_events)


class ReplayedRun(BaseModel):
    """A run reconstructed from its trace."""

    run_id: str
    graph: str
    thread_id: str | None = None
    attempts: list[int] = Field(default_factory=list)
    executions: list[NodeExecution] = Field(default_factory=list)
    # Sub-step events `_attach` could not place under an execution. Surfaced
    # rather than dropped: they are spend and behaviour that happened, and a
    # reader that sees an empty list knows the attribution below is complete.
    orphan_sub_events: list[TraceEvent] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)

    @property
    def path(self) -> list[str]:
        """Node names in the order they ended — the executed path.

        A node that raised is on the path: it ran. `metrics.to_mermaid` draws
        the same set, so the picture and the reconstruction agree. A node the
        trace never shows ending is not, because nothing says it did.
        """
        return [e.node for e in self.executions if e.completed]

    @property
    def tokens(self) -> int:
        """Total tokens, counted the way `metrics.summarize` counts them.

        Node totals plus orphans. A run with no node spans at all — `grapharc
        agent` drives an `AgentNode` with no enclosing graph — holds every
        token it spent in `orphan_sub_events`, and reporting zero for it was
        the audit trail contradicting itself. The two sets are disjoint, so
        nothing is counted twice.

        A node that *failed* counts too. Its terminal `error` event carries what
        it spent, exactly as an `end` does, so excluding it here reported zero
        tokens for a run the budget stopped for spending too many.
        """
        return sum(e.tokens for e in self.executions) + sum(
            e.tokens or 0 for e in self.orphan_sub_events
        )

    @property
    def node_ms(self) -> float:
        """Summed node durations, plus work outside any node. Under fan-out
        this exceeds the wall clock."""
        return round(
            sum(e.duration_ms or 0.0 for e in self.executions if e.ok)
            + sum(e.duration_ms or 0.0 for e in self.orphan_sub_events),
            2,
        )

    @property
    def wall_ms(self) -> float | None:
        """First to last event timestamp, or None if the timestamps are unusable.

        Includes everything between nodes — scheduling, checkpoint writes, a
        resume that happened hours later — which is why it is reported apart
        from `node_ms` rather than instead of it.
        """
        stamps = [t for t in (_parse_ts(e.ts) for e in self.events) if t is not None]
        if len(stamps) < 2:
            return None
        return round((max(stamps) - min(stamps)).total_seconds() * 1000, 2)

    @property
    def recorded_cost_usd(self) -> float | None:
        """Cost as `observe.cost` counts it: node totals plus orphans, once.

        A model call inside an `AgentNode` lands its cost twice on the trace —
        on its own `model` sub-event and again in the node's `end` aggregate —
        so summing every event doubles the bill. Node terminals plus orphan
        sub-events are disjoint by construction; sub-events attributed inside
        a node are the breakdown of its total, never an addition to it.
        """
        amounts = [e.cost_usd for e in self.executions if e.cost_usd is not None]
        amounts += [
            e.cost_usd for e in self.orphan_sub_events if e.cost_usd is not None
        ]
        return round(sum(amounts), 6) if amounts else None

    @property
    def errors(self) -> list[NodeExecution]:
        return [e for e in self.executions if e.error is not None]

    @property
    def termination_reason(self) -> str | None:
        """The last `termination_reason` any node wrote, as `metrics` reads it."""
        for execution in reversed(self.executions):
            if "termination_reason" in execution.state_delta:
                return execution.state_delta["termination_reason"]
        return None

    def replay_state(
        self,
        *,
        upto: int | None = None,
        reducers: Mapping[str, Callable[[Any, Any], Any]] | None = None,
    ) -> dict[str, Any]:
        """Fold the recorded deltas into the state as of `upto` (default: the end).

        `upto` is an execution index into `self.executions`, exclusive. Pass
        `reducers` for any field whose graph annotation carries one
        (`{"log": operator.add}`): the trace records the delta a node returned,
        never the reducer that merged it, so an un-named reduced field replays
        as last-write-wins and would silently under-report an accumulating list.
        """
        merge = dict(reducers or {})
        state: dict[str, Any] = {}
        for execution in self.executions[:upto]:
            if not execution.ok:
                continue  # a failed node's writes never reached the state
            for key, value in execution.state_delta.items():
                if key in merge and key in state:
                    state[key] = merge[key](state[key], value)
                else:
                    state[key] = value
        return state


def _attach(open_executions: list[NodeExecution], event: TraceEvent) -> NodeExecution | None:
    """Pick the execution a sub-step event belongs inside.

    The trace records no parent pointer, so this is inference, and it abstains
    rather than guesses: an `AgentNode` names its sub-steps `"<agent>:<what>"`,
    which identifies the parent whenever the agent's name matches its graph
    node's; failing that, a single open execution is unambiguous; with several
    open (a fan-out) and no name match, None is returned and the caller keeps
    the event in `orphan_sub_events` rather than charging the wrong worker.
    """
    prefix = event.node.split(":", 1)[0]
    for execution in reversed(open_executions):
        if execution.node == prefix:
            return execution
    if len(open_executions) == 1:
        return open_executions[0]
    return None


def _build_executions(
    events: Sequence[TraceEvent],
) -> tuple[list[NodeExecution], list[TraceEvent]]:
    """Return (executions in file order, sub-events with no identifiable parent)."""
    by_key: dict[tuple[str, int, int], NodeExecution] = {}
    order: list[NodeExecution] = []
    open_executions: list[NodeExecution] = []
    unattached: list[TraceEvent] = []

    for index, event in enumerate(events):
        if event.phase not in NODE_PHASES:
            parent = _attach(open_executions, event)
            (parent.sub_events if parent is not None else unattached).append(event)
            continue
        key = (event.node, event.step, event.attempt)
        execution = by_key.get(key)
        if execution is None:
            execution = NodeExecution(
                node=event.node,
                step=event.step,
                attempt=event.attempt,
                index=index,
                started_at=event.ts,
            )
            by_key[key] = execution
            order.append(execution)
            if event.phase == "start":
                open_executions.append(execution)
        if event.phase in TERMINAL_PHASES:
            execution.index = index
            execution.ended_at = event.ts
            execution.duration_ms = event.duration_ms
            execution.tokens = event.tokens or 0
            execution.cost_usd = event.cost_usd
            execution.state_delta = dict(event.state_delta or {})
            execution.error = event.error
            execution.ok = event.phase == "end"
            execution.completed = True
            if execution in open_executions:
                open_executions.remove(execution)

    return order, unattached


def replay(source: TraceRecorder | str | Path, run_id: str) -> ReplayedRun:
    """Reconstruct one run from its trace.

    Raises `ReplayError` if the trace holds no events for `run_id` — an empty
    reconstruction and a missing run are different answers.
    """
    events = load_events(source, run_id)
    if not events:
        raise ReplayError(f"no events for run_id {run_id!r} in the trace")
    executions, orphans = _build_executions(events)
    return ReplayedRun(
        run_id=run_id,
        graph=events[0].graph,
        thread_id=events[0].thread_id,
        attempts=sorted({e.attempt for e in events}),
        executions=executions,
        orphan_sub_events=orphans,
        events=list(events),
    )


def replay_thread(source: TraceRecorder | str | Path, thread_id: str) -> list[ReplayedRun]:
    """Every run recorded against one thread, in first-seen order.

    A thread that was resumed has more than one run id; this is how you read
    the whole logical thread rather than the last attempt at it.
    """
    events = load_events(source)
    run_ids = dict.fromkeys(e.run_id for e in events if e.thread_id == thread_id)
    return [replay(source, run_id) for run_id in run_ids]


class NodeDiff(BaseModel):
    """How one aligned node execution differed between two runs."""

    node: str
    step_a: int
    step_b: int
    changed_keys: list[str] = Field(default_factory=list)
    only_in_a_keys: list[str] = Field(default_factory=list)
    only_in_b_keys: list[str] = Field(default_factory=list)
    tokens_a: int = 0
    tokens_b: int = 0
    duration_ms_a: float | None = None
    duration_ms_b: float | None = None
    error_a: str | None = None
    error_b: str | None = None

    @property
    def duration_delta_ms(self) -> float | None:
        if self.duration_ms_a is None or self.duration_ms_b is None:
            return None
        return round(self.duration_ms_b - self.duration_ms_a, 2)

    @property
    def differs(self) -> bool:
        return bool(
            self.changed_keys
            or self.only_in_a_keys
            or self.only_in_b_keys
            or self.error_a != self.error_b
        )


class PathOp(BaseModel):
    """One `difflib` opcode over the two node sequences."""

    op: str  # "equal" | "replace" | "delete" | "insert"
    a_nodes: list[str] = Field(default_factory=list)
    b_nodes: list[str] = Field(default_factory=list)


class RunDiff(BaseModel):
    """What changed between two runs of the same graph."""

    graph: str
    run_a: str
    run_b: str
    path_a: list[str] = Field(default_factory=list)
    path_b: list[str] = Field(default_factory=list)
    path_ops: list[PathOp] = Field(default_factory=list)
    node_diffs: list[NodeDiff] = Field(default_factory=list)
    state_changes: dict[str, list[Any]] = Field(default_factory=dict)  # key -> [a, b]
    tokens_a: int = 0
    tokens_b: int = 0
    node_ms_a: float = 0.0
    node_ms_b: float = 0.0
    termination_a: str | None = None
    termination_b: str | None = None

    @property
    def same_path(self) -> bool:
        return self.path_a == self.path_b

    @property
    def changed_nodes(self) -> list[NodeDiff]:
        return [d for d in self.node_diffs if d.differs]

    @property
    def identical(self) -> bool:
        """Same path, same deltas, same termination.

        Timing and tokens are deliberately excluded: two runs of a
        deterministic graph differ in milliseconds every time, and a diff that
        is never clean is a diff nobody reads.
        """
        return (
            self.same_path
            and not self.changed_nodes
            and not self.state_changes
            and self.termination_a == self.termination_b
        )

    def summary(self) -> str:
        if self.identical:
            return f"{self.run_a} == {self.run_b}: same path ({len(self.path_a)} nodes), same state"
        parts = []
        if not self.same_path:
            parts.append(f"path {len(self.path_a)} -> {len(self.path_b)} nodes")
        if self.changed_nodes:
            parts.append(f"{len(self.changed_nodes)} node(s) wrote different deltas")
        if self.state_changes:
            parts.append(f"{len(self.state_changes)} state field(s) differ")
        if self.termination_a != self.termination_b:
            parts.append(f"termination {self.termination_a} -> {self.termination_b}")
        return f"{self.run_a} != {self.run_b}: " + "; ".join(parts)


def _delta_diff(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    shared = set(a) & set(b)
    return (
        sorted(k for k in shared if a[k] != b[k]),
        sorted(set(a) - set(b)),
        sorted(set(b) - set(a)),
    )


def diff_runs(
    a: ReplayedRun,
    b: ReplayedRun,
    *,
    reducers: Mapping[str, Callable[[Any, Any], Any]] | None = None,
) -> RunDiff:
    """Diff two reconstructed runs of the same graph.

    Raises `ReplayError` when the graphs differ: aligning `ingest -> report`
    against `plan -> execute` produces a diff that is technically correct and
    means nothing.
    """
    if a.graph != b.graph:
        raise ReplayError(
            f"cannot diff runs of different graphs: {a.graph!r} vs {b.graph!r}"
        )
    path_a, path_b = a.path, b.path
    matcher = difflib.SequenceMatcher(a=path_a, b=path_b, autojunk=False)
    completed_a = [e for e in a.executions if e.completed]
    completed_b = [e for e in b.executions if e.completed]

    ops: list[PathOp] = []
    node_diffs: list[NodeDiff] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        ops.append(PathOp(op=op, a_nodes=path_a[i1:i2], b_nodes=path_b[j1:j2]))
        if op != "equal":
            continue
        for offset in range(i2 - i1):
            ea, eb = completed_a[i1 + offset], completed_b[j1 + offset]
            changed, only_a, only_b = _delta_diff(ea.state_delta, eb.state_delta)
            node_diffs.append(
                NodeDiff(
                    node=ea.node,
                    step_a=ea.step,
                    step_b=eb.step,
                    changed_keys=changed,
                    only_in_a_keys=only_a,
                    only_in_b_keys=only_b,
                    tokens_a=ea.tokens,
                    tokens_b=eb.tokens,
                    duration_ms_a=ea.duration_ms,
                    duration_ms_b=eb.duration_ms,
                    error_a=ea.error,
                    error_b=eb.error,
                )
            )

    state_a = a.replay_state(reducers=reducers)
    state_b = b.replay_state(reducers=reducers)
    changes: dict[str, list[Any]] = {}
    for key in sorted(set(state_a) | set(state_b)):
        left, right = state_a.get(key), state_b.get(key)
        if left != right:
            changes[key] = [left, right]

    return RunDiff(
        graph=a.graph,
        run_a=a.run_id,
        run_b=b.run_id,
        path_a=path_a,
        path_b=path_b,
        path_ops=ops,
        node_diffs=node_diffs,
        state_changes=changes,
        tokens_a=a.tokens,
        tokens_b=b.tokens,
        node_ms_a=a.node_ms,
        node_ms_b=b.node_ms,
        termination_a=a.termination_reason,
        termination_b=b.termination_reason,
    )


def diff_trace(
    source: TraceRecorder | str | Path,
    run_a: str,
    run_b: str,
    *,
    reducers: Mapping[str, Callable[[Any, Any], Any]] | None = None,
) -> RunDiff:
    """`diff_runs` for two run ids in the same trace file."""
    return diff_runs(replay(source, run_a), replay(source, run_b), reducers=reducers)


def format_replay(run: ReplayedRun, *, deltas: bool = True) -> str:
    """A human-readable reconstruction, for a terminal or a bug report."""
    header = f"run {run.run_id} · graph {run.graph}"
    if run.thread_id and run.thread_id != run.run_id:
        header += f" · thread {run.thread_id}"
    if len(run.attempts) > 1:
        header += f" · attempts {run.attempts}"
    lines = [header]
    for execution in run.executions:
        mark = "ok " if execution.ok else ("ERR" if execution.error else "   ")
        timing = f"{execution.duration_ms:.1f}ms" if execution.duration_ms is not None else "-"
        line = f"  {execution.step:>3} {mark} {execution.node} ({timing}"
        if execution.tokens:
            line += f", {execution.tokens} tok"
        lines.append(line + ")")
        if execution.error:
            lines.append(f"        error: {execution.error}")
        if deltas and execution.state_delta:
            for key in sorted(execution.state_delta):
                lines.append(f"        {key} = {execution.state_delta[key]!r}")
        for sub in execution.sub_events:
            lines.append(f"        · {sub.phase} {sub.node}")
    reason = run.termination_reason or "not recorded"
    lines.append(f"  {len(run.path)} nodes · {run.tokens} tokens · stopped: {reason}")
    return "\n".join(lines)


def format_diff(diff: RunDiff) -> str:
    """A human-readable diff, in the same spirit as `format_replay`."""
    lines = [diff.summary()]
    for op in diff.path_ops:
        if op.op == "equal":
            continue
        lines.append(f"  {op.op}: {op.a_nodes} -> {op.b_nodes}")
    for node in diff.changed_nodes:
        detail = []
        if node.changed_keys:
            detail.append(f"changed {node.changed_keys}")
        if node.only_in_a_keys:
            detail.append(f"only in A {node.only_in_a_keys}")
        if node.only_in_b_keys:
            detail.append(f"only in B {node.only_in_b_keys}")
        if node.error_a != node.error_b:
            detail.append(f"error {node.error_a!r} -> {node.error_b!r}")
        lines.append(f"  {node.node}: " + "; ".join(detail))
    for key, (left, right) in diff.state_changes.items():
        lines.append(f"  state {key}: {left!r} -> {right!r}")
    return "\n".join(lines)


__all__ = [
    "NODE_PHASES",
    "NodeDiff",
    "NodeExecution",
    "PathOp",
    "ReplayError",
    "ReplayedRun",
    "RunDiff",
    "diff_runs",
    "diff_trace",
    "format_diff",
    "format_replay",
    "replay",
    "replay_thread",
]
