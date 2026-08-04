"""A structured view of one run, for renderers that draw rather than count.

`to_mermaid` renders the run as text; this module renders it as data — nodes
with a status and a bill, edges with a kind, clusters for a multi-round
planner run, and a timeline of when each node was open. The live web view
draws its SVG from this; nothing here is a new source of truth, and the three
shapes mirror `to_mermaid`'s three branches exactly:

- a trace with `topology` events becomes the *declared* graph with execution
  status overlaid (kind ``"topology"``),
- a trace with executions but no topology becomes the executed path in event
  order (kind ``"path"``),
- a planner run that admitted nothing becomes an honest empty diagram
  (kind ``"empty"``), with the reason in `note`.

What is deliberately *not* here: `state_delta` contents, event payloads, or
anything else a node wrote. The exposure is the same as the Mermaid path's —
node names, statuses, counts, durations, tokens, recorded cost, and error
labels truncated the way `metrics._label` truncates them. The live server
serializes this model to the browser, so the boundary is enforced by the
model's fields, not by a filter somewhere else.

Geometry fields (`x`/`y`/`w`/`h`, edge `points`, canvas `width`/`height`)
are zeroed here and filled by `observe.layout`, which is a pure function of
the topology — so two snapshots of the same shape carry identical coordinates
and a page can patch statuses in place without anything moving.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from grapharc.observe.metrics import latest_topologies
from grapharc.observe.replay import ReplayedRun
from grapharc.observe.status import NodeStatus, node_states
from grapharc.observe.trace import TraceEvent

#: Same truncation the Mermaid labels use: flatten whitespace, cap the length.
#: An error message is the one free-text string this model carries.
_ERROR_LIMIT = 120

#: Phases that describe the run rather than doing work (mirrors `metrics`).
_SHAPE_PHASES = frozenset({"topology", "approval_request", "approval_response"})
_LOOP_PHASES = frozenset({"plan", "admission", "round"})

_NO_GRAPH_NOTE = "no graph ran: no proposal was admitted and built"


class NodeView(BaseModel):
    """One drawable node: identity, status, and its share of the bill."""

    id: str
    label: str
    cluster: str | None = None
    #: Terminals (`start`/`end` rings) are drawn, not statused.
    role: Literal["node", "start", "end"] = "node"
    status: NodeStatus = "pending"
    executions: int = 0
    tokens: int = 0
    #: Tokens reported by sub-steps of a still-open execution — what a running
    #: node has spent *so far*, before its terminal event exists to say so.
    live_tokens: int = 0
    #: Recorded cost only, summed off terminal events. Never an estimate.
    cost_usd: float | None = None
    duration_ms: float | None = None
    error: str | None = None
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


class EdgeView(BaseModel):
    source: str
    target: str
    kind: Literal["static", "conditional", "fanout", "state"] = "static"
    #: Cubic bezier control points, filled by layout: [[x,y], ×4].
    points: list[list[float]] = Field(default_factory=list)


class ClusterView(BaseModel):
    """One admitted round's frame in a multi-round planner run."""

    id: str
    label: str
    round: int | None = None
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


class TimelineSpan(BaseModel):
    """When one node execution was open, as ms offsets from the run's start.

    Names and times only — this is the replay scrubber's fuel, and it must
    stay as free of payload as the rest of the model.
    """

    node: str
    t0: float
    t1: float | None = None
    #: None while open, True for `end`, False for `error`.
    ok: bool | None = None


class GraphSnapshot(BaseModel):
    kind: Literal["topology", "path", "empty"]
    width: float = 0.0
    height: float = 0.0
    nodes: list[NodeView] = Field(default_factory=list)
    edges: list[EdgeView] = Field(default_factory=list)
    clusters: list[ClusterView] = Field(default_factory=list)
    timeline: list[TimelineSpan] = Field(default_factory=list)
    #: The timeline's horizon: the largest t1, or the run's wall span.
    duration_ms: float | None = None
    note: str | None = None


def _flatten(text: str | None, fallback: str = "error") -> str:
    return " ".join(str(text or fallback).split())[:_ERROR_LIMIT]


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _offsets(run: ReplayedRun) -> dict[str, float]:
    """Event timestamp → ms offset from the run's first parseable stamp."""
    stamps = [t for t in (_parse_ts(e.ts) for e in run.events) if t is not None]
    if not stamps:
        return {}
    origin = min(stamps)
    return {
        e.ts: (t - origin).total_seconds() * 1000
        for e in run.events
        if (t := _parse_ts(e.ts)) is not None
    }


def _timeline(
    run: ReplayedRun, resolve: Callable[[TraceEvent], str | None]
) -> tuple[list[TimelineSpan], float | None]:
    """One span per node execution, paired the way `replay` pairs them.

    `resolve` maps a lifecycle event to the NodeView id its span points at;
    executions of nodes the view does not draw resolve to None and are skipped
    rather than left dangling.
    """
    offsets = _offsets(run)
    open_spans: dict[tuple[str, str, int, int], TimelineSpan] = {}
    spans: list[TimelineSpan] = []
    horizon: float | None = None
    for event in run.events:
        if event.phase not in ("start", "end", "error"):
            continue
        target = resolve(event)
        if target is None or event.ts not in offsets:
            continue
        at = offsets[event.ts]
        horizon = at if horizon is None else max(horizon, at)
        key = (event.graph, event.node, event.step, event.attempt)
        if event.phase == "start":
            span = TimelineSpan(node=target, t0=at)
            open_spans[key] = span
            spans.append(span)
            continue
        span = open_spans.pop(key, None)
        if span is None:
            # A terminal with no start — a budget-refused node. Zero-width
            # span at the moment of refusal; it still scrubs as errored.
            span = TimelineSpan(node=target, t0=at)
            spans.append(span)
        span.t1 = at
        span.ok = event.phase == "end"
    return spans, horizon


def _node_measures(node: NodeView, graph_events: list[TraceEvent], name: str) -> None:
    """Fill tokens/cost/duration off terminal events, live tokens off sub-steps.

    Terminal events (`end` *and* `error`) are the measure, exactly as
    `metrics.summarize` counts a run: the kernel stamps a node's terminal event
    with what it spent, whichever way it ended. Sub-step tokens are a breakdown
    of a total that will exist once the node closes — they are surfaced only
    while it is open (`live_tokens`), never added to `tokens`.
    """
    costs: list[float] = []
    prefix = f"{name}:"
    for event in graph_events:
        if event.node == name and event.phase in ("end", "error"):
            node.tokens += event.tokens or 0
            if event.duration_ms is not None:
                node.duration_ms = (node.duration_ms or 0.0) + event.duration_ms
            if event.cost_usd is not None:
                costs.append(event.cost_usd)
            node.executions += 1
        elif node.status == "running" and (
            event.node == name or event.node.startswith(prefix)
        ):
            if event.phase not in ("start",) and event.tokens:
                node.live_tokens += event.tokens
    if costs:
        node.cost_usd = round(sum(costs), 6)
    if node.duration_ms is not None:
        node.duration_ms = round(node.duration_ms, 2)


def build_graph_view(run: ReplayedRun) -> GraphSnapshot:
    """The structured equivalent of `to_mermaid`, minus geometry."""
    topologies = latest_topologies(run.events)
    if topologies:
        return _topology_view(run, topologies)
    return _path_view(run)


def _topology_view(run: ReplayedRun, topologies: list) -> GraphSnapshot:
    clustered = len(topologies) > 1
    nodes: list[NodeView] = []
    edges: list[EdgeView] = []
    clusters: list[ClusterView] = []
    node_id: dict[tuple[str, str], str] = {}

    for graph_index, (graph, delta) in enumerate(topologies):
        prefix = f"g{graph_index}." if clustered else ""
        cluster_id = f"g{graph_index}" if clustered else None
        declared = [
            str(n) for n in delta.get("nodes", []) if n not in ("__start__", "__end__")
        ]
        raw_edges = [tuple(e) for e in delta.get("edges", []) if len(tuple(e)) == 3]
        fanout_sources = [str(s) for s in delta.get("fanout_sources", [])]
        graph_events = [e for e in run.events if e.graph == graph]
        states = node_states(graph_events)

        if clustered:
            label = delta.get("round")
            clusters.append(
                ClusterView(
                    id=cluster_id,
                    label=f"round {label}" if label else graph,
                    round=label if isinstance(label, int) else None,
                )
            )

        def ref(name: str, prefix: str = prefix, cluster_id: str | None = cluster_id) -> str:
            if name in ("__start__", "__end__"):
                terminal_id = f"{prefix}{name}"
                if all(n.id != terminal_id for n in nodes):
                    nodes.append(
                        NodeView(
                            id=terminal_id,
                            label="start" if name == "__start__" else "end",
                            cluster=cluster_id,
                            role="start" if name == "__start__" else "end",
                        )
                    )
                return terminal_id
            return f"{prefix}{name}"

        for name in declared:
            state = states.get(name)
            node = NodeView(
                id=f"{prefix}{name}",
                label=name,
                cluster=cluster_id,
                status=state.status if state else "pending",
            )
            if state is not None:
                if state.last_error is not None:
                    node.error = _flatten(state.last_error.error)
                _node_measures(node, graph_events, name)
            nodes.append(node)
            node_id[(graph, name)] = node.id

        for source, target, kind in raw_edges:
            edges.append(
                EdgeView(
                    source=ref(str(source)),
                    target=ref(str(target)),
                    kind="conditional" if kind == "conditional" else "static",
                )
            )
        # Fan-out targets are dynamic, so the topology names only the sources;
        # workers are wired to whichever declared nodes actually started —
        # the same rule the Mermaid overlay draws by.
        targeted = {str(e[1]) for e in raw_edges}
        for source in fanout_sources:
            for name in declared:
                started = name in states and states[name].starts > 0
                if name != source and name not in targeted and started:
                    edges.append(
                        EdgeView(source=ref(source), target=ref(name), kind="fanout")
                    )

        if clustered and graph_index:
            edges.append(
                EdgeView(
                    source=f"g{graph_index - 1}",
                    target=f"g{graph_index}",
                    kind="state",
                )
            )

    timeline, horizon = _timeline(run, lambda e: node_id.get((e.graph, e.node)))
    return GraphSnapshot(
        kind="topology",
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        timeline=timeline,
        duration_ms=horizon,
    )


def _feed_view(run: ReplayedRun, feed: list[TraceEvent]) -> GraphSnapshot:
    """A run with no node spans at all: chain its flat feed in event order."""
    nodes: list[NodeView] = []
    seen: dict[tuple[str, int], str] = {}

    def ref(event: TraceEvent) -> str:
        key = (event.node, event.step)
        existing = seen.get(key)
        if existing is not None:
            return existing
        view = NodeView(
            id=f"{event.node}@{event.step}",
            label=event.node,
            status="errored" if event.error else "done",
            tokens=event.tokens or 0,
            cost_usd=event.cost_usd,
            duration_ms=event.duration_ms,
            error=_flatten(event.error) if event.error else None,
            executions=1,
        )
        nodes.append(view)
        seen[key] = view.id
        return view.id

    edges: list[EdgeView] = []
    for a, b in zip(feed, feed[1:], strict=False):
        source, target = ref(a), ref(b)
        if not a.error and source != target:
            edges.append(EdgeView(source=source, target=target))
    ref(feed[-1])

    timeline, horizon = _timeline(run, lambda e: seen.get((e.node, e.step)))
    return GraphSnapshot(
        kind="path",
        nodes=nodes,
        edges=edges,
        timeline=timeline,
        duration_ms=horizon,
    )


def _path_view(run: ReplayedRun) -> GraphSnapshot:
    """The executed path, in event order — `to_mermaid`'s fallback, as data.

    Built from `run.executions`, not from terminal events, because an
    execution that *started and has not finished* is the most important thing
    on a live page — a delegated executor is silent mid-flight, and building
    from `end`/`error` alone rendered exactly that run as "no events" while
    its one node was busy working.
    """
    executions = list(run.executions)
    if not executions:
        feed = [
            e
            for e in run.orphan_sub_events
            if e.phase not in _SHAPE_PHASES and e.phase not in _LOOP_PHASES
        ]
        if all(e.phase == "stop" for e in feed):
            feed = []
        if not feed and any(e.phase in _LOOP_PHASES for e in run.events):
            return GraphSnapshot(kind="empty", note=_NO_GRAPH_NOTE)
        if not feed:
            return GraphSnapshot(kind="empty", note="no events")
        return _feed_view(run, feed)

    nodes: list[NodeView] = []
    seen: dict[tuple[str, int], str] = {}
    for execution in executions:
        key = (execution.node, execution.step)
        if key in seen:
            continue
        if execution.error is not None:
            status = "errored"
        elif execution.completed:
            status = "done"
        else:
            status = "running"
        view = NodeView(
            id=f"{execution.node}@{execution.step}",
            label=execution.node,
            status=status,
            tokens=execution.tokens,
            live_tokens=execution.sub_tokens if status == "running" else 0,
            cost_usd=execution.cost_usd,
            duration_ms=execution.duration_ms,
            error=_flatten(execution.error) if execution.error is not None else None,
            executions=1,
        )
        nodes.append(view)
        seen[key] = view.id

    edges: list[EdgeView] = []
    for a, b in zip(executions, executions[1:], strict=False):
        source, target = seen[(a.node, a.step)], seen[(b.node, b.step)]
        if a.error is None and source != target:
            edges.append(EdgeView(source=source, target=target))

    timeline, horizon = _timeline(run, lambda e: seen.get((e.node, e.step)))
    return GraphSnapshot(
        kind="path",
        nodes=nodes,
        edges=edges,
        timeline=timeline,
        duration_ms=horizon,
    )


__all__ = [
    "ClusterView",
    "EdgeView",
    "GraphSnapshot",
    "NodeView",
    "TimelineSpan",
    "build_graph_view",
]
