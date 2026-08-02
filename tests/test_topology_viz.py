"""The topology trace event and the overlay renderer built on it.

The event states the graph's declared shape; the renderer draws that shape
with execution status overlaid. Together they are what makes a diagram show
the *orchestration* — branches not taken included — rather than only the path
that ran.
"""

from __future__ import annotations

from grapharc.observe.metrics import to_mermaid
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    value: int = 0
    route: str = ""


def _touch(state: State) -> dict:
    return {"value": state.value + 1}


# -- the event --------------------------------------------------------------


def test_topology_is_the_first_event_and_carries_the_declared_shape(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    g = GraphARC(State, name="shape", trace=trace)
    g.add_node("a", _touch, writes={"value"})
    g.add_node("b", _touch, writes={"value"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.compile().invoke({}, run_id="r1")

    events = trace.read_events("r1")
    first = events[0]
    assert (first.phase, first.node, first.step) == ("topology", "topology", 0)
    assert first.state_delta == {
        "nodes": ["a", "b"],
        "edges": [
            ["__start__", "a", "static"],
            ["a", "b", "static"],
            ["b", "__end__", "static"],
        ],
    }
    # And it precedes any work: the next event is the first node's start.
    assert events[1].phase == "start"


def test_conditional_routes_are_recorded_with_their_kind(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    g = GraphARC(State, name="branchy", trace=trace)
    g.add_node("decide", lambda s: {"route": "left"}, writes={"route"})
    g.add_node("left", _touch, writes={"value"})
    g.add_node("right", _touch, writes={"value"})
    g.add_edge(START, "decide")
    g.add_conditional_edge(
        "decide", lambda s: s.route, {"left": "left", "right": "right"}
    )
    g.add_edge("left", END)
    g.add_edge("right", END)
    g.compile().invoke({}, run_id="r1")

    delta = trace.read_events("r1")[0].state_delta
    assert ["decide", "left", "conditional"] in delta["edges"]
    assert ["decide", "right", "conditional"] in delta["edges"]


# -- the renderer -----------------------------------------------------------


def _hand_written(tmp_path, *, error: bool = False) -> TraceRecorder:
    """A trace with declared topology and a partial execution over it."""
    trace = TraceRecorder(tmp_path / "hand.jsonl")
    trace.event(
        run_id="r1",
        graph="g",
        node="topology",
        phase="topology",
        step=0,
        state_delta={
            "nodes": ["plan", "act", "verify"],
            "edges": [
                ["__start__", "plan", "static"],
                ["plan", "act", "static"],
                ["act", "verify", "static"],
                ["verify", "__end__", "static"],
            ],
        },
    )
    trace.event(run_id="r1", graph="g", node="plan", phase="start", step=1)
    trace.event(run_id="r1", graph="g", node="plan", phase="end", step=1)
    trace.event(run_id="r1", graph="g", node="act", phase="start", step=2)
    if error:
        trace.event(
            run_id="r1", graph="g", node="act", phase="error", step=2, error="boom"
        )
    return trace


def test_overlay_marks_done_running_and_pending(tmp_path):
    diagram = to_mermaid(_hand_written(tmp_path), "r1")
    assert diagram.startswith("flowchart TD")
    # Declared shape, whether or not it ran:
    assert '["verify"]' in diagram
    # Status classes: plan finished, act is open, verify never started.
    assert "class n0 done" in diagram
    assert "class n1 running" in diagram
    assert "class n2 pending" in diagram


def test_overlay_keeps_the_error_rhombus_and_marks_errored(tmp_path):
    diagram = to_mermaid(_hand_written(tmp_path, error=True), "r1")
    assert '-.->|error| err0{"boom"}' in diagram
    assert "class n1 errored" in diagram


def test_multiple_round_graphs_become_clusters(tmp_path):
    trace = TraceRecorder(tmp_path / "rounds.jsonl")
    for round_no, graph in ((1, "plan:aaa"), (2, "plan:bbb")):
        trace.event(
            run_id="r1",
            graph=graph,
            node="topology",
            phase="topology",
            step=0,
            state_delta={
                "nodes": ["triage"],
                "edges": [["__start__", "triage", "static"]],
                "round": round_no,
            },
        )
    diagram = to_mermaid(trace, "r1")
    assert 'subgraph cluster0["round 1"]' in diagram
    assert 'subgraph cluster1["round 2"]' in diagram
    # Node ids are cluster-prefixed, so the reused name cannot collide.
    assert 'g0_n0["triage"]' in diagram
    assert 'g1_n0["triage"]' in diagram
    # Cluster ids never carry the raw graph name's colon.
    for line in diagram.splitlines():
        if line.strip().startswith("subgraph"):
            assert ":" not in line.split("[")[0]


def test_every_overlay_line_is_balanced(tmp_path):
    """The delimiter property the capstone gate pins, extended to the overlay."""
    diagram = to_mermaid(_hand_written(tmp_path, error=True), "r1")
    for line in diagram.splitlines():
        assert line.count("{") == line.count("}"), line
        assert line.count("[") == line.count("]"), line
        assert line.count('"') % 2 == 0, line


def test_a_trace_without_topology_renders_exactly_as_before(tmp_path):
    trace = TraceRecorder(tmp_path / "old.jsonl")
    trace.event(run_id="r1", graph="g", node="a", phase="start", step=1)
    trace.event(run_id="r1", graph="g", node="a", phase="end", step=1)
    trace.event(run_id="r1", graph="g", node="b", phase="start", step=2)
    trace.event(run_id="r1", graph="g", node="b", phase="end", step=2)
    assert to_mermaid(trace, "r1") == (
        "flowchart TD\n"
        '  start((start)) --> n0["a"]\n'
        '  n0["a"] --> n1["b"]'
    )


def test_fanout_workers_connect_to_their_source_at_render_time(tmp_path):
    trace = TraceRecorder(tmp_path / "fan.jsonl")
    trace.event(
        run_id="r1",
        graph="g",
        node="topology",
        phase="topology",
        step=0,
        state_delta={
            "nodes": ["dispatch", "worker"],
            "edges": [["__start__", "dispatch", "static"]],
            "fanout_sources": ["dispatch"],
        },
    )
    trace.event(run_id="r1", graph="g", node="dispatch", phase="start", step=1)
    trace.event(run_id="r1", graph="g", node="dispatch", phase="end", step=1)
    trace.event(run_id="r1", graph="g", node="worker", phase="start", step=2)
    trace.event(run_id="r1", graph="g", node="worker", phase="end", step=2)
    diagram = to_mermaid(trace, "r1")
    assert 'n0["dispatch"] -.-> n1["worker"]' in diagram


def test_every_cluster_is_closed_and_labels_survive_the_kernel_restatement(tmp_path):
    """Two regressions pinned at once: a global dedup used to collapse every
    cluster's identical `end` into one (invalid Mermaid on every multi-round
    diagram), and the kernel's later topology event — which carries no
    `round` — used to overwrite the loop's labelled one."""
    trace = TraceRecorder(tmp_path / "t.jsonl")
    for round_no, graph in ((1, "plan:aaa"), (2, "plan:bbb")):
        # The loop's statement, with the round label...
        trace.event(
            run_id="r1", graph=graph, node="topology", phase="topology", step=0,
            state_delta={
                "nodes": ["triage"],
                "edges": [["__start__", "triage", "static"]],
                "round": round_no,
            },
        )
        # ...then the kernel's bare restatement at invoke, without it.
        trace.event(
            run_id="r1", graph=graph, node="topology", phase="topology", step=0,
            state_delta={
                "nodes": ["triage"],
                "edges": [["__start__", "triage", "static"]],
            },
        )
        trace.event(run_id="r1", graph=graph, node="triage", phase="start", step=1)
        trace.event(run_id="r1", graph=graph, node="triage", phase="end", step=1)

    diagram = to_mermaid(trace, "r1")
    opened = sum(1 for line in diagram.splitlines() if line.strip().startswith("subgraph"))
    closed = sum(1 for line in diagram.splitlines() if line.strip() == "end")
    assert opened == closed == 2, diagram
    assert 'subgraph cluster0["round 1"]' in diagram
    assert 'subgraph cluster1["round 2"]' in diagram


def test_sentinels_in_a_hand_written_nodes_list_do_not_crash_the_render(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(
        run_id="r1", graph="g", node="topology", phase="topology", step=0,
        state_delta={
            "nodes": ["__start__", "a", "__end__"],
            "edges": [["__start__", "a", "static"], ["a", "__end__", "static"]],
        },
    )
    diagram = to_mermaid(trace, "r1")
    assert "flowchart TD" in diagram and '["a"]' in diagram


def test_a_run_whose_planning_never_produced_a_graph_says_so(tmp_path):
    """The planner's paperwork is not an orchestration. A run that never got a
    proposal admitted-and-built used to render `plan → admission → round1 →
    plan → …` as a linear chain — bookkeeping drawn as though it were the
    graph, which is the picture every failed planning run ended on."""
    trace = TraceRecorder(tmp_path / "t.jsonl")
    for rnd in (1, 2):
        trace.event(run_id="r1", graph="loop", node="p:plan", phase="plan", step=rnd)
        trace.event(run_id="r1", graph="loop", node="a:x", phase="admission", step=rnd)
        trace.event(run_id="r1", graph="loop", node=f"loop:round{rnd}",
                    phase="round", step=rnd)
    trace.event(run_id="r1", graph="loop", node="loop:stop", phase="stop", step=3)

    diagram = to_mermaid(trace, "r1")
    assert "no graph ran" in diagram
    assert "admission" not in diagram and "round" not in diagram


def test_an_agent_run_still_chains_its_own_steps(tmp_path):
    """The bookkeeping filter must not silence a real agent's sequence."""
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(run_id="r1", graph="agent", node="a:model", phase="model", step=1)
    trace.event(run_id="r1", graph="agent", node="a:read_file", phase="tool", step=2)
    trace.event(run_id="r1", graph="agent", node="a:stop", phase="stop", step=3)
    diagram = to_mermaid(trace, "r1")
    assert "a:model" in diagram and "a:read_file" in diagram


def test_rounds_are_linked_by_the_state_that_flowed_between_them(tmp_path):
    """Each round materializes standalone, so its entry really is START —
    but drawn alone the clusters read as unrelated graphs sharing a page.
    A dotted `state` link says what actually happened between them."""
    trace = TraceRecorder(tmp_path / "t.jsonl")
    for rnd, graph in ((1, "plan:aaa"), (2, "plan:bbb")):
        trace.event(
            run_id="r1", graph=graph, node="topology", phase="topology", step=0,
            state_delta={"nodes": ["a"], "edges": [["__start__", "a", "static"]],
                         "round": rnd},
        )
    diagram = to_mermaid(trace, "r1")
    assert "cluster0 -.->|state| cluster1" in diagram
    # And it sits outside both subgraphs, after they close.
    lines = diagram.splitlines()
    assert lines.index("  cluster0 -.->|state| cluster1") > max(
        i for i, line in enumerate(lines) if line.strip() == "end"
    )


def test_a_single_round_gets_no_cluster_link(tmp_path):
    trace = TraceRecorder(tmp_path / "t.jsonl")
    trace.event(
        run_id="r1", graph="g", node="topology", phase="topology", step=0,
        state_delta={"nodes": ["a"], "edges": [["__start__", "a", "static"]]},
    )
    assert "-.->|state|" not in to_mermaid(trace, "r1")
