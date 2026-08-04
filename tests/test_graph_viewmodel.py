"""The structured graph snapshot the live view draws from.

It must mirror `to_mermaid`'s three shapes, carry the spend the trace
recorded, and — above all — never carry what a node wrote into the state.
"""

from __future__ import annotations

from grapharc.observe.replay import replay
from grapharc.observe.trace import TraceRecorder
from grapharc.observe.viewmodel import build_graph_view

_TOPOLOGY = {
    "nodes": ["plan", "act", "verify"],
    "edges": [
        ["__start__", "plan", "static"],
        ["plan", "act", "static"],
        ["act", "verify", "conditional"],
        ["verify", "__end__", "static"],
    ],
}


def _declared(tmp_path, *, error: bool = False) -> TraceRecorder:
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(
        run_id="r1", graph="g", node="topology", phase="topology", step=0,
        state_delta=_TOPOLOGY,
    )
    trace.event(run_id="r1", graph="g", node="plan", phase="start", step=1)
    trace.event(
        run_id="r1", graph="g", node="plan", phase="end", step=1,
        duration_ms=12.0, tokens=40, cost_usd=0.002,
        state_delta={"secret_finding": "THE-PAYLOAD"},
    )
    trace.event(run_id="r1", graph="g", node="act", phase="start", step=2)
    if error:
        trace.event(
            run_id="r1", graph="g", node="act", phase="error", step=2,
            error="boom went the " + "x" * 200, tokens=9,
        )
    return trace


def test_topology_view_declares_every_node_with_status_and_kinds(tmp_path):
    view = build_graph_view(replay(_declared(tmp_path), "r1"))
    assert view.kind == "topology"
    by_id = {n.id: n for n in view.nodes}
    assert by_id["plan"].status == "done"
    assert by_id["act"].status == "running"
    assert by_id["verify"].status == "pending"
    assert by_id["__start__"].role == "start"
    assert by_id["__end__"].role == "end"
    kinds = {(e.source, e.target): e.kind for e in view.edges}
    assert kinds[("act", "verify")] == "conditional"
    assert kinds[("plan", "act")] == "static"


def test_recorded_spend_reaches_the_node_and_is_never_estimated(tmp_path):
    view = build_graph_view(replay(_declared(tmp_path), "r1"))
    plan = next(n for n in view.nodes if n.id == "plan")
    assert (plan.tokens, plan.cost_usd, plan.duration_ms) == (40, 0.002, 12.0)
    # `act` is still open: no terminal, no bill — and no invented one.
    act = next(n for n in view.nodes if n.id == "act")
    assert (act.tokens, act.cost_usd) == (0, None)


def test_error_text_is_flattened_and_truncated(tmp_path):
    view = build_graph_view(replay(_declared(tmp_path, error=True), "r1"))
    act = next(n for n in view.nodes if n.id == "act")
    assert act.status == "errored"
    assert act.error is not None
    assert len(act.error) <= 120
    # An errored node's terminal spend still counts (the overspend rule).
    assert act.tokens == 9


def test_state_delta_contents_never_reach_the_snapshot(tmp_path):
    view = build_graph_view(replay(_declared(tmp_path), "r1"))
    assert "THE-PAYLOAD" not in view.model_dump_json()
    assert "secret_finding" not in view.model_dump_json()


def test_multi_round_runs_cluster_and_keep_each_rounds_own_status(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    for round_no, graph in ((1, "g1"), (2, "g2")):
        trace.event(
            run_id="r1", graph=graph, node="topology", phase="topology", step=0,
            state_delta={
                "nodes": ["work"],
                "edges": [["__start__", "work", "static"], ["work", "__end__", "static"]],
                "round": round_no,
            },
        )
    trace.event(run_id="r1", graph="g1", node="work", phase="start", step=1)
    trace.event(run_id="r1", graph="g1", node="work", phase="end", step=1)
    trace.event(run_id="r1", graph="g2", node="work", phase="start", step=1)

    view = build_graph_view(replay(trace, "r1"))
    assert [c.label for c in view.clusters] == ["round 1", "round 2"]
    by_id = {n.id: n for n in view.nodes}
    assert by_id["g0.work"].status == "done"
    assert by_id["g1.work"].status == "running"
    state_edges = [e for e in view.edges if e.kind == "state"]
    assert [(e.source, e.target) for e in state_edges] == [("g0", "g1")]


def test_fanout_workers_wire_to_started_nodes_like_the_mermaid_rule(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(
        run_id="r1", graph="g", node="topology", phase="topology", step=0,
        state_delta={
            "nodes": ["split", "worker_a", "worker_b", "join"],
            "edges": [
                ["__start__", "split", "static"],
                ["join", "__end__", "static"],
            ],
            "fanout_sources": ["split"],
        },
    )
    trace.event(run_id="r1", graph="g", node="split", phase="start", step=1)
    trace.event(run_id="r1", graph="g", node="split", phase="end", step=1)
    trace.event(run_id="r1", graph="g", node="worker_a", phase="start", step=2)
    # worker_b never started: it must not be wired.
    view = build_graph_view(replay(trace, "r1"))
    fanout = [(e.source, e.target) for e in view.edges if e.kind == "fanout"]
    assert ("split", "worker_a") in fanout
    assert all(target != "worker_b" for _, target in fanout)


def test_path_fallback_chains_executions_in_event_order(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(run_id="r1", graph="g", node="a", phase="start", step=1)
    trace.event(run_id="r1", graph="g", node="a", phase="end", step=1, tokens=5)
    trace.event(run_id="r1", graph="g", node="b", phase="start", step=2)
    trace.event(run_id="r1", graph="g", node="b", phase="end", step=2)
    view = build_graph_view(replay(trace, "r1"))
    assert view.kind == "path"
    assert [n.id for n in view.nodes] == ["a@1", "b@2"]
    assert [(e.source, e.target) for e in view.edges] == [("a@1", "b@2")]


def test_a_silent_delegated_run_shows_its_open_node_as_running(tmp_path):
    """A delegated executor writes nothing between `start` and its finish —
    the page must show that node running, not claim "no events"."""
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(run_id="r1", graph="cli-agent", node="claude_code", phase="start", step=1)
    view = build_graph_view(replay(trace, "r1"))
    assert view.kind == "path"
    assert [(n.id, n.status) for n in view.nodes] == [("claude_code@1", "running")]


def test_an_open_agent_node_reports_its_live_sub_step_tokens(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(run_id="r1", graph="agent", node="worker", phase="start", step=1)
    trace.event(run_id="r1", graph="agent", node="worker:model", phase="model", step=2, tokens=340)
    view = build_graph_view(replay(trace, "r1"))
    node = view.nodes[0]
    assert node.status == "running"
    assert node.live_tokens == 340


def test_planless_planner_run_is_an_honest_empty(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(run_id="r1", graph="loop", node="planner", phase="plan", step=1)
    trace.event(run_id="r1", graph="loop", node="admission", phase="admission", step=2)
    view = build_graph_view(replay(trace, "r1"))
    assert view.kind == "empty"
    assert view.nodes == []
    assert "no proposal was admitted" in (view.note or "")


def test_timeline_spans_are_monotonic_and_name_only(tmp_path):
    view = build_graph_view(replay(_declared(tmp_path), "r1"))
    assert view.timeline, "expected spans for plan and act"
    for span in view.timeline:
        assert span.t0 >= 0
        if span.t1 is not None:
            assert span.t1 >= span.t0
    names = {s.node for s in view.timeline}
    assert names <= {"plan", "act", "verify"}
    # `act` is open: its span has no end yet.
    act_span = next(s for s in view.timeline if s.node == "act")
    assert act_span.t1 is None and act_span.ok is None
