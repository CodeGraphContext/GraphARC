"""Run metrics derived from traces.

Nothing here is a new source of truth — every number is computed from the
JSONL trace, so the metrics and the audit trail can never disagree.

**Not every run has node spans.** `grapharc agent` drives an `AgentNode`
against a `RunContext` with no enclosing graph, so it emits `model`/`tool`/
`stop` and never `start`/`end`; the planner emits `plan`/`admission`/`round`
for the same reason — a planning round is not a node execution. Counting only
`end` events reported such a run as `tokens: 0, nodes_executed: 0`, which is
the audit trail disagreeing with itself. `replay` already separates events it
could place inside a node (`sub_events`) from those it could not
(`orphan_sub_events`); the two sets are disjoint, so measurements are summed
over node totals *plus* orphans and nothing is counted twice.

`nodes_executed` still counts `end` events only. A run with no kernel node
genuinely executed no kernel nodes, and `per_phase` is what carries its work.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from grapharc.observe.replay import replay
from grapharc.observe.trace import TraceRecorder


class RunMetrics(BaseModel):
    run_id: str
    graph: str
    nodes_executed: int
    errors: int
    tokens: int
    duration_ms: float
    attempts: int
    termination_reason: str | None = None
    per_node: dict[str, int] = {}
    # Every event the run recorded, and how many of each phase. These are what
    # make a run with no node spans legible instead of reading as empty.
    events: int = 0
    per_phase: dict[str, int] = {}


def summarize(recorder: TraceRecorder, run_id: str) -> RunMetrics | None:
    events = recorder.read_events(run_id)
    if not events:
        return None
    ends = [e for e in events if e.phase == "end"]
    errors = [e for e in events if e.phase == "error"]
    # Work the reconstruction could not place inside any node. Disjoint from
    # `ends` by construction, so adding it cannot double-count a node total.
    orphans = replay(recorder, run_id).orphan_sub_events
    measured = [*ends, *orphans]
    reason = None
    # Scanned across every event, not just `end`: an agent writes its
    # `termination_reason` on a `stop` event, and that is still why it stopped.
    for e in reversed(events):
        delta = e.state_delta or {}
        if "termination_reason" in delta:
            reason = delta["termination_reason"]
            break
    return RunMetrics(
        run_id=run_id,
        graph=events[0].graph,
        nodes_executed=len(ends),
        errors=len(errors),
        tokens=sum(e.tokens or 0 for e in measured),
        duration_ms=round(sum(e.duration_ms or 0.0 for e in measured), 2),
        attempts=max((e.attempt for e in events), default=1),
        termination_reason=reason,
        per_node=dict(Counter(e.node for e in ends)),
        events=len(events),
        per_phase=dict(Counter(e.phase for e in events)),
    )


_MERMAID_ESCAPES = {
    '"': "#quot;",
    "[": "#91;",
    "]": "#93;",
    "{": "#123;",
    "}": "#125;",
    "(": "#40;",
    ")": "#41;",
    "<": "#60;",
    ">": "#62;",
    ";": "#59;",
}


def _label(text: str, limit: int = 120) -> str:
    """Make arbitrary text safe inside a Mermaid label.

    Truncate the text *before* escaping and quoting — slicing the composed line
    would cut off the closing delimiter and break the whole diagram, which is
    exactly what happens on the error paths an operator most needs to see.
    """
    flat = " ".join(str(text).split())[:limit]
    return "".join(_MERMAID_ESCAPES.get(ch, ch) for ch in flat)


def to_mermaid(recorder: TraceRecorder, run_id: str) -> str:
    """Render the executed path as a Mermaid flowchart (file-first, git-friendly).

    The path is the run's node executions. A run that has none — `grapharc
    agent`, which drives an `AgentNode` with no enclosing graph — still did
    work, so its recorded events are the path instead of an empty diagram.
    """
    run = replay(recorder, run_id)
    events = [e for e in run.events if e.phase in ("end", "error")]
    if not events:
        events = list(run.orphan_sub_events)
    if not events:
        return 'flowchart TD\n  empty["no events"]'
    lines = ["flowchart TD"]
    # Key by (node, step) so parallel instances of a fan-out worker are distinct
    # nodes rather than one node with a self-loop the graph never had.
    ids: dict[tuple[str, int], str] = {}

    def node_ref(ev) -> str:
        key = (ev.node, ev.step)
        node_id = ids.setdefault(key, f"n{len(ids)}")
        return f'{node_id}["{_label(ev.node)}"]'

    lines.append(f"  start((start)) --> {node_ref(events[0])}")
    for i, e in enumerate(events):
        if e.phase == "error":
            lines.append(
                f'  {node_ref(e)} -.->|error| err{i}{{"{_label(e.error or "error")}"}}'
            )
    for a, b in zip(events, events[1:], strict=False):
        # Anything that is not an error continues the path. On the node-span
        # list this is exactly `phase == "end"`, which is what it used to say;
        # written this way it also chains an agent's model/tool/stop events.
        if a.phase != "error":
            lines.append(f"  {node_ref(a)} --> {node_ref(b)}")
    return "\n".join(dict.fromkeys(lines))
