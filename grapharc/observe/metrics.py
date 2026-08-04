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
from typing import Any

from pydantic import BaseModel

from grapharc.observe.replay import replay
from grapharc.observe.status import node_states
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
    # `errors` are measured too, and for the same reason `ends` are: the kernel
    # stamps a node's terminal event with what that node spent, whichever way it
    # ended. Counting only `end` meant a run *stopped for overspending* reported
    # `tokens: 0` — the audit trail losing precisely the spend that triggered
    # enforcement. Sub-events inside a node are a breakdown of its total rather
    # than an addition to it, so the disjointness that makes `ends + orphans`
    # safe holds here unchanged.
    measured = [*ends, *errors, *orphans]
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


#: Phases that describe the run rather than doing work; the executed-path
#: fallback must never chain them as if they were steps.
_SHAPE_PHASES = frozenset({"topology", "approval_request", "approval_response"})

#: The governed loop's own bookkeeping. A planning round is not a node
#: execution — chaining these drew `plan -> admission -> round1 -> plan ...`
#: as though the planner's paperwork were the orchestration, which is exactly
#: the picture a run whose planning failed used to end on.
_LOOP_PHASES = frozenset({"plan", "admission", "round"})

#: What to draw when a run has no graph to show. Honest about *why* there is
#: nothing: a run that never got a graph admitted and built has no topology,
#: and inventing a chain from its paperwork misrepresents it.
_NO_GRAPH = 'flowchart TD\n  none["no graph ran: no proposal was admitted and built"]'

_STATUS_CLASSES = (
    "  classDef done fill:#d3f2d3,stroke:#2f7d32",
    "  classDef running fill:#fff3cd,stroke:#b8860b",
    "  classDef pending fill:#eeeeee,stroke:#999999,color:#666666",
    "  classDef errored fill:#f8d7da,stroke:#b02a37",
)


def to_mermaid(recorder: TraceRecorder, run_id: str) -> str:
    """Render the run as a Mermaid flowchart (file-first, git-friendly).

    A run whose trace carries `topology` events is drawn as its *declared*
    graph — every node, every edge, branches included — with execution status
    overlaid per node: done, running, errored, or still pending. That is the
    orchestration, not merely the path. A multi-round planner run draws one
    cluster per admitted round graph.

    A run with no topology events keeps the original rendering: the executed
    path in event order. A run with no node executions at all — `grapharc
    agent`, which drives an `AgentNode` with no enclosing graph — still did
    work, so its recorded events are the path instead of an empty diagram.
    """
    run = replay(recorder, run_id)
    topologies = latest_topologies(run.events)
    if topologies:
        return _topology_mermaid(run, topologies)
    events = [e for e in run.events if e.phase in ("end", "error")]
    if not events:
        events = [
            e
            for e in run.orphan_sub_events
            if e.phase not in _SHAPE_PHASES and e.phase not in _LOOP_PHASES
        ]
        # A lone `stop` is a driver saying why it finished, not a path.
        if all(e.phase == "stop" for e in events):
            events = []
        if not events and any(e.phase in _LOOP_PHASES for e in run.events):
            return _NO_GRAPH
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


def latest_topologies(events: list) -> list[tuple[str, dict[str, Any]]]:
    """One merged (graph, delta) per graph, in first-appearance order.

    Two emitters restate the same graph: the loop (whose delta carries
    `round`/`proposal_id`/`fingerprint`) and then the kernel at invoke (whose
    delta does not). Merged rather than last-wins, so the shape is the latest
    statement but the labels an earlier statement carried are never lost —
    last-wins made "round N" labels vanish the moment execution started.
    """
    merged: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.phase == "topology" and event.state_delta:
            merged.setdefault(event.graph, {}).update(event.state_delta)
    return list(merged.items())


def _topology_mermaid(run: Any, topologies: list[tuple[str, dict[str, Any]]]) -> str:
    lines = ["flowchart TD"]
    clustered = len(topologies) > 1
    class_members: dict[str, list[str]] = {}
    error_index = 0

    for graph_index, (graph, delta) in enumerate(topologies):
        # Sentinels are drawn as terminals, never as declared nodes — a delta
        # that lists them (hand-written traces do) must not crash the render.
        nodes = [
            str(n) for n in delta.get("nodes", []) if n not in ("__start__", "__end__")
        ]
        edges = [tuple(edge) for edge in delta.get("edges", [])]
        fanout_sources = [str(s) for s in delta.get("fanout_sources", [])]
        graph_events = [e for e in run.events if e.graph == graph]

        prefix = f"g{graph_index}_" if clustered else ""
        indent = "    " if clustered else "  "
        # Each cluster's lines are collected apart and deduped apart: a global
        # dedup once collapsed every cluster's identical `end` terminator into
        # one, leaving n-1 subgraphs unclosed — invalid Mermaid on every
        # multi-round diagram.
        cluster: list[str] = []

        ids: dict[str, str] = {}

        def ref(name: str, prefix: str = prefix, ids: dict[str, str] = ids) -> str:
            if name == "__start__":
                return f"{prefix}start((start))"
            if name == "__end__":
                return f"{prefix}fin((end))"
            node_id = ids.setdefault(name, f"{prefix}n{len(ids)}")
            return f'{node_id}["{_label(name)}"]'

        # Declare every node up front: a pending node with no edges yet must
        # still appear — the whole point is showing what has not run.
        for name in nodes:
            cluster.append(f"{indent}{ref(name)}")

        # Per-node status from this graph's events, by the shared rule in
        # `observe.status`: parallel instances of one node collapse to the
        # worst-informative status — any error wins, then running, then done.
        states = node_states(graph_events)
        for event in graph_events:
            if event.phase == "error":
                cluster.append(
                    f"{indent}{ref(event.node)} -.->|error| "
                    f'err{error_index}{{"{_label(event.error or "error")}"}}'
                )
                error_index += 1

        for source, target, kind in (e for e in edges if len(e) == 3):
            arrow = "-.->" if kind == "conditional" else "-->"
            cluster.append(f"{indent}{ref(source)} {arrow} {ref(target)}")
        targeted = {e[1] for e in edges if len(e) == 3}
        for source in fanout_sources:
            for name in nodes:
                started = name in states and states[name].starts > 0
                if name != source and name not in targeted and started:
                    cluster.append(f"{indent}{ref(source)} -.-> {ref(name)}")

        for name in nodes:
            node_id = ids[name]
            status = states[name].status if name in states else "pending"
            class_members.setdefault(status, []).append(node_id)

        if clustered:
            label = delta.get("round")
            title = f"round {label}" if label else _label(graph)
            # Cluster ids may not hold ':'; the raw name goes in the label.
            lines.append(f'  subgraph cluster{graph_index}["{_label(title)}"]')
        lines.extend(dict.fromkeys(cluster))
        if clustered:
            lines.append("  end")
            # Each round materializes as its own standalone graph, so its entry
            # really is START — but drawn alone the clusters read as unrelated
            # graphs that happened to share a page. This link says what did
            # happen: the state one round left is what the next one started
            # from. Dotted and labelled, so it is never mistaken for an edge
            # the topology declared.
            if graph_index:
                lines.append(
                    f"  cluster{graph_index - 1} -.->|state| cluster{graph_index}"
                )

    lines.extend(_STATUS_CLASSES)
    for status, members in class_members.items():
        lines.append(f"  class {','.join(members)} {status}")
    return "\n".join(lines)
