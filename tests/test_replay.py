"""§10.1 / §10.3 / §10.4 gates — replay, OTel export, cost attribution.

Every test here drives a real graph through `CompiledGraphARC.invoke` and then
reconstructs it from the trace file that run produced. Nothing hand-writes a
trace except where the point *is* a hand-written trace: a field today's runtime
never populates (`cost_usd`, `model`), a torn line from another process, a
resume seeded from a file this process did not write.

The property that matters most is the last section's: what `cost` reports and
what `metrics` reports are the same numbers, because they are read off the same
events. A test that only checked cost against itself would pass with the whole
attribution rewritten.
"""

import json
import operator
import pathlib
import subprocess
import sys
import threading
import types
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from grapharc.harness import (
    AgentNode,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
    ToolSpec,
)
from grapharc.observe.cost import (
    RateCard,
    attribute,
    attribute_thread,
    by_node,
    tokens_by_model,
)
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.otel import (
    ListSpanExporter,
    NullSpanExporter,
    OTelSpanExporter,
    OTelUnavailable,
    Span,
    SpanExporter,
    export_run,
    to_spans,
)
from grapharc.observe.replay import (
    ReplayError,
    diff_runs,
    diff_trace,
    format_diff,
    format_replay,
    replay,
    replay_thread,
)
from grapharc.observe.trace import TraceEvent, TraceRecorder
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import END, START, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class S(GraphARCState):
    question: str = ""
    answer: str = ""
    hops: int = 0
    log: Annotated[list[str], operator.add] = []
    termination_reason: str = ""


def _linear_graph(trace, *, answer="42", boom=False):
    g = GraphARC(S, name="demo", trace=trace)
    g.add_node("load", lambda s: {"question": s.question or "q", "log": ["load"]},
               writes={"question", "log"})

    def think(state):
        if boom:
            raise RuntimeError("think exploded")
        return {"answer": answer, "hops": state.hops + 1, "log": ["think"]}

    g.add_node("think", think, writes={"answer", "hops", "log"})
    g.add_node(
        "report",
        lambda s: {"termination_reason": "target_met", "log": ["report"]},
        writes={"termination_reason", "log"},
    )
    g.add_edge(START, "load")
    g.add_edge("load", "think")
    g.add_edge("think", "report")
    g.add_edge("report", END)
    return g.compile()


# --------------------------------------------------------------------------
# §10.1 replay
# --------------------------------------------------------------------------


def test_replay_reconstructs_the_node_sequence(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")

    assert run.path == ["load", "think", "report"]
    assert run.graph == "demo"
    assert [e.ok for e in run.executions] == [True, True, True]


def test_replay_reconstructs_state_deltas_and_final_state(trace):
    _linear_graph(trace, answer="the answer").invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")

    assert run.executions[1].state_delta["answer"] == "the answer"
    assert run.executions[1].state_delta["hops"] == 1
    state = run.replay_state(reducers={"log": operator.add})
    assert state["answer"] == "the answer"
    assert state["log"] == ["load", "think", "report"]
    assert state["termination_reason"] == "target_met"


def test_replay_state_without_a_reducer_is_last_write_wins(trace):
    """The trace records deltas, never the reducer that merged them."""
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    unreduced = replay(trace, "r1").replay_state()

    assert unreduced["log"] == ["report"]  # the last delta, not the accumulation
    assert replay(trace, "r1").replay_state(reducers={"log": operator.add})["log"] == [
        "load", "think", "report",
    ]


def test_replay_state_upto_stops_where_told(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")

    assert run.replay_state(upto=1) == {"question": "why", "log": ["load"]}
    assert "answer" not in run.replay_state(upto=1)


def test_replay_reconstructs_timing(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")

    assert all(e.duration_ms is not None and e.duration_ms >= 0 for e in run.executions)
    assert run.node_ms == pytest.approx(
        sum(e.duration_ms for e in run.executions), abs=0.01
    )
    assert run.wall_ms is not None and run.wall_ms >= 0
    assert run.termination_reason == "target_met"


def test_replay_keeps_a_failed_node_as_an_execution(trace):
    with pytest.raises(RuntimeError, match="think exploded"):
        _linear_graph(trace, boom=True).invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")

    assert run.path == ["load", "think"]
    failed = run.executions[-1]
    assert failed.node == "think" and not failed.ok
    assert "think exploded" in failed.error
    # A failed node's writes never reached the state, so replay must not apply them.
    assert "answer" not in run.replay_state()


def test_replay_rejects_an_unknown_run(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    with pytest.raises(ReplayError, match="no events for run_id 'nope'"):
        replay(trace, "nope")


def test_replay_accepts_a_path_not_only_a_recorder(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    assert replay(trace.path, "r1").path == ["load", "think", "report"]
    assert replay(str(trace.path), "r1").path == ["load", "think", "report"]


def test_replay_orders_by_file_order_not_step_number(trace):
    """A node's `end` shares its `start`'s step; sub-steps inside it come later.

    Sorting the trace by step would put the node's end before the work it
    contains, so replay reads file order. This asserts the two really differ.
    """
    g = GraphARC(S, name="demo", trace=trace)

    def outer(state, ctx):
        trace.event(
            run_id=ctx.run_id, thread_id=ctx.thread_id, attempt=ctx.attempt,
            graph=ctx.graph, node="outer:model", phase="model", step=ctx.next_step(),
            tokens=11,
        )
        return {"answer": "done"}

    g.add_node("outer", outer, writes={"answer"})
    g.add_node("after", lambda s: {"hops": 1}, writes={"hops"})
    g.add_edge(START, "outer")
    g.add_edge("outer", "after")
    g.add_edge("after", END)
    g.compile().invoke({"question": "q"}, run_id="r1")

    events = trace.read_events("r1")
    steps = [e.step for e in events]
    assert steps != sorted(steps), "the fixture must exercise out-of-order step numbers"
    run = replay(trace, "r1")
    assert run.path == ["outer", "after"]
    assert [e.node for e in run.executions[0].sub_events] == ["outer:model"]


def test_replay_reconstructs_a_fan_out(trace):
    """Parallel workers are separate executions, not one node with a self-loop."""

    class F(GraphARCState):
        shard: int = 0
        found: Annotated[list[str], operator.add] = []

    g = GraphARC(F, name="fan", trace=trace)
    g.add_node("plan", lambda s: None, writes=set())
    g.add_node(
        "worker",
        lambda p: {"found": [f"shard-{p.shard}"]},
        writes={"found"},
        input_schema=F,
    )
    g.add_node("collect", lambda s: None, writes=set())
    g.add_edge(START, "plan")
    g.add_fanout_edge("plan", lambda s: [("worker", F(shard=i)) for i in range(3)])
    g.add_edge("worker", "collect")
    g.add_edge("collect", END)
    g.compile().invoke({"shard": 0}, run_id="r1")

    run = replay(trace, "r1")

    assert run.path.count("worker") == 3
    workers = [e for e in run.executions if e.node == "worker"]
    assert len({e.step for e in workers}) == 3, "each parallel worker is its own execution"
    assert sorted(e.state_delta["found"][0] for e in workers) == [
        "shard-0", "shard-1", "shard-2",
    ]
    # Order across parallel workers is whatever the scheduler produced; the
    # fold must reproduce the set, and one worker must not overwrite another.
    assert sorted(run.replay_state(reducers={"found": operator.add})["found"]) == [
        "shard-0", "shard-1", "shard-2",
    ]


def _fanout_token_run(trace, run_id, *, serial):
    """Three workers, one model call each, overlapping in time."""
    import time

    from grapharc.runtime.budget import Budget
    from grapharc.testing import ScriptedChatModel

    class F(GraphARCState):
        seeds: list[str] = []
        said: Annotated[list[str], operator.add] = []

    class Shard(GraphARCState):
        seed: str = ""

    def worker(shard):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke(
            "hello " + shard.seed
        )
        time.sleep(0.05)  # hold the window open so the siblings overlap
        return {"said": [shard.seed]}

    g = GraphARC(F, name="fan", trace=trace, budget=Budget())
    g.add_node("fan", lambda s: {"seeds": s.seeds}, writes={"seeds"})
    g.add_node("w", worker, writes={"said"}, input_schema=Shard)
    g.add_edge(START, "fan")
    g.add_fanout_edge("fan", lambda s: [("w", Shard(seed=x)) for x in s.seeds])
    g.add_edge("w", END)
    compiled = g.compile()
    compiled.invoke(
        {"seeds": ["a", "b", "c"]},
        run_id=run_id,
        budget=Budget(max_concurrency=1) if serial else Budget(),
    )
    per_worker = [
        e.tokens for e in trace.read_events(run_id) if e.phase == "end" and e.node == "w"
    ]
    return compiled.last_run.meter.tokens, per_worker


def test_a_fanout_worker_is_charged_its_own_tokens_not_its_siblings(trace):
    """A node's `end` event used to report the movement of the run's *shared*
    total while it ran, so overlapping workers each absorbed the others' spend.

    Three workers costing the same each traced as 24/16/8, and `metrics` and
    `cost` both reported three times one worker's spend for the whole run —
    doubling the estimated bill purely because the work ran in parallel.
    """
    real, parallel = _fanout_token_run(trace, "par", serial=False)

    assert len(parallel) == 3
    assert len(set(parallel)) == 1, f"workers absorbed each other's spend: {parallel}"
    assert sum(parallel) == real


def test_fanout_token_attribution_does_not_depend_on_concurrency(trace):
    """The same work must cost the same whether it runs in parallel or serially —
    including the dollar figure `cost` estimates from it."""
    serial_real, serial_per = _fanout_token_run(trace, "ser", serial=True)
    par_real, par_per = _fanout_token_run(trace, "par", serial=False)

    assert serial_real == par_real
    assert sorted(serial_per) == sorted(par_per)

    rates = RateCard(default=3.0)
    assert summarize(trace, "par").tokens == summarize(trace, "ser").tokens == serial_real
    assert (
        attribute(trace, "par", rates=rates).estimated_cost_usd
        == attribute(trace, "ser", rates=rates).estimated_cost_usd
    )


def test_a_hand_charged_token_still_lands_on_the_nodes_end_event(trace):
    """Attribution moved to the meter's per-node scope, which must still capture a
    charge the node makes itself — not only the ones the usage callback saw."""
    from grapharc.runtime.budget import Budget
    from grapharc.runtime.graph import RunContext

    class H(GraphARCState):
        ok: bool = False

    def node(state, ctx: RunContext):
        ctx.meter.charge_tokens(777)
        return {"ok": True}

    g = GraphARC(H, name="hand", trace=trace, budget=Budget())
    g.add_node("n", node, writes={"ok"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    g.compile().invoke({}, run_id="r1")

    ends = [e for e in trace.read_events("r1") if e.phase == "end"]
    assert [e.tokens for e in ends] == [777]


def test_a_named_sub_step_finds_its_worker_among_parallel_nodes(trace):
    trace.event(run_id="r1", graph="demo", node="w1", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="w2", phase="start", step=2)
    trace.event(run_id="r1", graph="demo", node="w2:model", phase="model", step=3, tokens=500)
    trace.event(run_id="r1", graph="demo", node="w1", phase="end", step=1, duration_ms=1.0)
    trace.event(run_id="r1", graph="demo", node="w2", phase="end", step=2, duration_ms=1.0)

    run = replay(trace, "r1")

    assert run.orphan_sub_events == []
    assert [e.node for e in run.executions[1].sub_events] == ["w2:model"]
    assert run.executions[0].sub_events == []


def test_the_sub_step_phase_set_is_open(trace):
    """Replay must not hard-code AgentNode's phases; other nodes emit their own."""
    trace.event(run_id="r1", graph="demo", node="planner", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="planner:plan", phase="plan", step=2,
                tokens=60, state_delta={"proposed": 3})
    trace.event(run_id="r1", graph="demo", node="planner:p-7", phase="admission", step=3,
                state_delta={"status": "rejected"})
    trace.event(run_id="r1", graph="demo", node="planner", phase="end", step=1,
                duration_ms=2.0, tokens=60)

    run = replay(trace, "r1")

    assert run.path == ["planner"]
    assert [e.phase for e in run.executions[0].sub_events] == ["plan", "admission"]
    assert run.orphan_sub_events == []
    # An unknown sub-phase must not be counted as a second node execution.
    assert summarize(trace, "r1").nodes_executed == 1
    assert attribute(trace, "r1").executions == 1
    spans = {s.attributes.get("grapharc.phase"): s for s in to_spans(run)[1:]}
    assert spans["admission"].attributes["grapharc.delta.status"] == "rejected"


def test_an_unidentifiable_sub_step_is_not_charged_to_a_guess(trace):
    """Two workers open and no name match: abstain rather than pick one."""
    trace.event(run_id="r1", graph="demo", node="w1", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="w2", phase="start", step=2)
    trace.event(run_id="r1", graph="demo", node="mystery:model", phase="model", step=3,
                tokens=500)
    trace.event(run_id="r1", graph="demo", node="w1", phase="end", step=1, duration_ms=1.0)
    trace.event(run_id="r1", graph="demo", node="w2", phase="end", step=2, duration_ms=1.0)

    run = replay(trace, "r1")

    assert [e.node for e in run.orphan_sub_events] == ["mystery:model"]
    assert all(e.sub_events == [] for e in run.executions)
    spans = to_spans(run)
    orphan = next(s for s in spans if s.name == "mystery:model")
    assert orphan.parent_id == spans[0].span_id, "parented to the run, not to a worker"
    assert attribute(run).tokens_before_error == 0


def test_replay_thread_returns_every_attempt(trace):
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "a"}, thread_id="t1", run_id="r1")
    compiled.invoke({"question": "b"}, thread_id="t1", run_id="r2")

    runs = replay_thread(trace, "t1")

    assert [r.run_id for r in runs] == ["r1", "r2"]
    assert runs[0].attempts == [1] and runs[1].attempts == [2]


def test_format_replay_names_the_nodes_and_the_reason(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    text = format_replay(replay(trace, "r1"))

    assert "run r1" in text and "graph demo" in text
    for node in ("load", "think", "report"):
        assert node in text
    assert "stopped: target_met" in text


# --------------------------------------------------------------------------
# §10.1 diff
# --------------------------------------------------------------------------


def test_diff_of_identical_runs_is_clean(trace):
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "why"}, run_id="r1")
    compiled.invoke({"question": "why"}, run_id="r2")

    diff = diff_trace(trace, "r1", "r2", reducers={"log": operator.add})

    assert diff.identical
    assert diff.same_path and not diff.changed_nodes and not diff.state_changes
    assert "same path" in diff.summary()


def test_diff_finds_a_changed_state_delta(trace):
    _linear_graph(trace, answer="42").invoke({"question": "why"}, run_id="r1")
    _linear_graph(trace, answer="43").invoke({"question": "why"}, run_id="r2")

    diff = diff_trace(trace, "r1", "r2")

    assert not diff.identical
    assert diff.same_path  # same nodes ran
    changed = {d.node: d for d in diff.changed_nodes}
    assert list(changed) == ["think"]
    assert changed["think"].changed_keys == ["answer"]
    assert diff.state_changes["answer"] == ["42", "43"]
    assert "answer" in format_diff(diff)


def test_diff_finds_a_changed_path(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")
    with pytest.raises(RuntimeError):
        _linear_graph(trace, boom=True).invoke({"question": "why"}, run_id="r2")

    diff = diff_trace(trace, "r1", "r2")

    assert not diff.identical and not diff.same_path
    assert diff.path_a == ["load", "think", "report"]
    assert diff.path_b == ["load", "think"]
    assert [op.op for op in diff.path_ops] == ["equal", "delete"]
    assert diff.path_ops[-1].a_nodes == ["report"]
    think = next(d for d in diff.node_diffs if d.node == "think")
    assert think.error_a is None and "think exploded" in think.error_b
    assert think.differs


def test_diff_reports_token_and_duration_movement(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")
    _linear_graph(trace).invoke({"question": "why"}, run_id="r2")

    diff = diff_trace(trace, "r1", "r2")

    load = next(d for d in diff.node_diffs if d.node == "load")
    assert load.duration_delta_ms is not None
    assert load.duration_delta_ms == pytest.approx(
        load.duration_ms_b - load.duration_ms_a, abs=0.01
    )
    assert diff.node_ms_a > 0 and diff.node_ms_b > 0


def test_diff_refuses_two_different_graphs(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")
    other = GraphARC(S, name="other", trace=trace)
    other.add_node("only", lambda s: {"answer": "x"}, writes={"answer"})
    other.add_edge(START, "only")
    other.add_edge("only", END)
    other.compile().invoke({"question": "q"}, run_id="r2")

    with pytest.raises(ReplayError, match="different graphs"):
        diff_runs(replay(trace, "r1"), replay(trace, "r2"))


def test_diff_aligns_around_an_inserted_node(trace):
    def build(extra):
        g = GraphARC(S, name="demo", trace=trace)
        g.add_node("load", lambda s: {"question": "q"}, writes={"question"})
        g.add_node("think", lambda s: {"answer": "42"}, writes={"answer"})
        g.add_edge(START, "load")
        if extra:
            g.add_node("extra", lambda s: {"hops": 9}, writes={"hops"})
            g.add_edge("load", "extra")
            g.add_edge("extra", "think")
        else:
            g.add_edge("load", "think")
        g.add_edge("think", END)
        return g.compile()

    build(False).invoke({"question": "q"}, run_id="r1")
    build(True).invoke({"question": "q"}, run_id="r2")

    diff = diff_trace(trace, "r1", "r2")

    assert [op.op for op in diff.path_ops] == ["equal", "insert", "equal"]
    assert diff.path_ops[1].b_nodes == ["extra"]
    # `think` is aligned across the insert, so it is compared to itself.
    think = next(d for d in diff.node_diffs if d.node == "think")
    assert not think.differs
    assert diff.state_changes == {"hops": [None, 9]}


# --------------------------------------------------------------------------
# §10.3 OpenTelemetry export
# --------------------------------------------------------------------------


def test_to_spans_builds_a_root_and_one_span_per_node(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    spans = to_spans(trace, "r1")

    root = spans[0]
    assert root.parent_id is None and root.name == "demo"
    assert [s.name for s in spans[1:]] == ["load", "think", "report"]
    assert all(s.parent_id == root.span_id for s in spans[1:])
    assert all(s.end_unix_nano >= s.start_unix_nano for s in spans)
    assert root.start_unix_nano <= min(s.start_unix_nano for s in spans[1:])
    assert root.attributes["grapharc.termination_reason"] == "target_met"


def test_span_duration_comes_from_the_recorded_duration(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    run = replay(trace, "r1")
    spans = {s.name: s for s in to_spans(run)}

    for execution in run.executions:
        expected = int(execution.duration_ms * 1_000_000)
        assert spans[execution.node].duration_ns == expected


def test_sub_step_spans_nest_under_their_node(trace):
    harness, model = _tool_agent()
    g = GraphARC(S, name="demo", trace=trace)
    agent = AgentNode(model, harness, name="agent", trace=trace, task_field="question")
    g.add_node("agent", agent, writes=agent.writes)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    g.compile().invoke({"question": "add them"}, run_id="r1")

    spans = to_spans(trace, "r1")

    by_name = {s.name: s for s in spans}
    node_span = by_name["agent"]
    children = [s for s in spans if s.parent_id == node_span.span_id]
    assert children, "the agent's model/tool sub-steps must nest under the node span"
    assert {s.attributes["grapharc.phase"] for s in children} >= {"model", "tool", "stop"}
    orphans = replay(trace, "r1").orphan_sub_events
    assert [e.phase for e in orphans] == ["topology"]  # shape record, not lost work


def test_error_spans_carry_the_error(trace):
    with pytest.raises(RuntimeError):
        _linear_graph(trace, boom=True).invoke({"question": "why"}, run_id="r1")

    spans = {s.name: s for s in to_spans(trace, "r1")}

    assert spans["think"].status == "error"
    assert "think exploded" in spans["think"].error
    assert spans["demo"].status == "error"


def test_null_exporter_is_the_default_and_swallows_everything(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    spans = export_run(trace, "r1")

    assert len(spans) == 4
    assert NullSpanExporter().export(spans) is None
    assert isinstance(NullSpanExporter(), SpanExporter)


def test_export_run_hands_spans_to_a_protocol_exporter(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")
    sink = ListSpanExporter()

    returned = export_run(trace, "r1", exporter=sink)

    assert isinstance(sink, SpanExporter)
    assert [s.name for s in sink.spans] == ["demo", "load", "think", "report"]
    assert sink.spans == returned


def test_otel_exporter_raises_a_useful_error_without_the_dependency(monkeypatch):
    """opentelemetry is not a GraphARC dependency; asking for it must say so."""
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    with pytest.raises(OTelUnavailable, match="opentelemetry-sdk"):
        OTelSpanExporter()


def test_importing_observe_needs_no_opentelemetry():
    """The claim in `observe/otel.py` is that the dependency is *confined to*
    `OTelSpanExporter`, so the module imports without opentelemetry present.

    This used to assert `"opentelemetry" not in sys.modules`, which tested
    something GraphARC does not control and was wrong in two directions. It was
    vacuous when opentelemetry was absent — true of the dev venv, so it passed
    locally forever — and it failed in CI where `[all]` installs it, because
    `langsmith` (reached transitively through `langchain_core.tracers`) imports
    opentelemetry the moment it is available. Nothing in this package asks for
    it; the assertion was reading a third party's import graph.

    The property that *is* ours: with opentelemetry made unimportable, the module
    still imports. Checked in a subprocess, because a blocker on `sys.meta_path`
    and a half-imported package are not things to leave behind in this one.
    """
    probe = """
import sys

class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("opentelemetry is blocked for this test")
        return None

for name in [m for m in sys.modules if m.split(".")[0] == "opentelemetry"]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocked())

import grapharc.observe.otel as otel

assert hasattr(otel, "OTelSpanExporter"), "the module imported but is not itself"
leaked = [m for m in sys.modules if m.split(".")[0] == "opentelemetry"]
assert not leaked, f"module scope pulled in {leaked}"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180
    )

    assert result.returncode == 0, (
        f"grapharc.observe.otel does not import without opentelemetry:\n{result.stderr}"
    )


class _StubSpan:
    def __init__(self, name, start_time, attributes):
        self.name = name
        self.start_time = start_time
        self.end_time = None
        self.attributes = attributes
        self.status = None

    def set_status(self, status):
        self.status = status

    def end(self, end_time=None):
        self.end_time = end_time


class _StubTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None, start_time=None, attributes=None):
        span = _StubSpan(name, start_time, attributes or {})
        span.parent_context = context
        self.spans.append(span)
        return span


def _install_otel_stub(monkeypatch):
    """A stand-in for `opentelemetry.trace` with the surface the adapter uses.

    Stubbed, not simulated-real: it proves the adapter calls `start_span` /
    `set_status` / `end` with the right times and nesting. It cannot prove the
    adapter works against the actual SDK, which is not installed here.
    """
    module = types.ModuleType("opentelemetry.trace")
    contexts = {}

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class Status:
        def __init__(self, code, description=None):
            self.code = code
            self.description = description

    def set_span_in_context(span, context=None):
        ctx = object()
        contexts[id(ctx)] = span
        module.recorded[id(ctx)] = span
        return ctx

    module.Status = Status
    module.StatusCode = StatusCode
    module.set_span_in_context = set_span_in_context
    module.get_tracer_provider = lambda: None
    module.recorded = {}

    package = types.ModuleType("opentelemetry")
    package.trace = module
    monkeypatch.setitem(sys.modules, "opentelemetry", package)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", module)
    return module


def test_otel_exporter_emits_spans_with_recorded_times_and_parents(trace, monkeypatch):
    module = _install_otel_stub(monkeypatch)
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")
    spans = to_spans(trace, "r1")
    tracer = _StubTracer()

    OTelSpanExporter(tracer=tracer).export(spans)

    assert [s.name for s in tracer.spans] == ["demo", "load", "think", "report"]
    root, *children = tracer.spans
    assert root.parent_context is None
    assert root.start_time == spans[0].start_unix_nano
    assert root.end_time == spans[0].end_unix_nano
    for stub, span in zip(children, spans[1:], strict=True):
        assert module.recorded[id(stub.parent_context)] is root
        assert stub.start_time == span.start_unix_nano
        assert stub.end_time == span.end_unix_nano
        assert stub.status.code == "OK"
        assert stub.attributes["grapharc.node"] == span.name


def test_otel_exporter_marks_a_failed_node_as_error(trace, monkeypatch):
    _install_otel_stub(monkeypatch)
    with pytest.raises(RuntimeError):
        _linear_graph(trace, boom=True).invoke({"question": "why"}, run_id="r1")
    tracer = _StubTracer()

    OTelSpanExporter(tracer=tracer).export(to_spans(trace, "r1"))

    failed = next(s for s in tracer.spans if s.name == "think")
    assert failed.status.code == "ERROR"
    assert "think exploded" in failed.status.description


def test_span_attributes_are_otel_primitives(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    for span in to_spans(trace, "r1"):
        for key, value in span.attributes.items():
            assert isinstance(value, (str, int, float, bool)), f"{key} -> {value!r}"


# --------------------------------------------------------------------------
# §10.4 cost attribution
# --------------------------------------------------------------------------


class _ToolModel(ScriptedChatModel):
    """ScriptedChatModel with one scripted tool-call turn, then a plain answer."""

    tool_call_script: list[list[dict[str, Any]]] = Field(default_factory=list)
    _bound: list[Any] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self._bound.append(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = min(self._cursor, max(len(self.responses) - 1, 0))
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        calls = self.tool_call_script[index] if index < len(self.tool_call_script) else None
        if not calls:
            return result
        base = result.generations[0].message
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=base.content,
                        usage_metadata=base.usage_metadata,
                        tool_calls=[{"type": "tool_call", **c} for c in calls],
                    )
                )
            ]
        )


def _tool_agent():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="add", description="add", fn=lambda a, b: a + b))
    harness = Harness(
        registry=registry,
        policy=PermissionPolicy(rules=[PermissionRule(pattern="add", action="allow")]),
        executor=LocalExecutor(),
    )
    model = _ToolModel(
        responses=["calling add", "the sum is 5"],
        tool_call_script=[[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]],
    )
    return harness, model


def test_cost_totals_match_metrics_exactly(trace):
    """Cost and metrics must be the same numbers off the same events."""
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    cost = attribute(trace, "r1")
    metrics = summarize(trace, "r1")

    assert cost.tokens == metrics.tokens
    assert cost.executions == metrics.nodes_executed
    assert cost.duration_ms == pytest.approx(metrics.duration_ms, abs=0.05)
    assert {n.node: n.executions for n in cost.per_node} == metrics.per_node


def test_cost_attributes_tokens_per_node(trace):
    harness, model = _tool_agent()
    g = GraphARC(S, name="demo", trace=trace, budget=Budget(max_iterations=20))
    agent = AgentNode(model, harness, name="agent", trace=trace, task_field="question")
    g.add_node("agent", agent, writes=agent.writes)
    g.add_node("report", lambda s: {"hops": 1}, writes={"hops"})
    g.add_edge(START, "agent")
    g.add_edge("agent", "report")
    g.add_edge("report", END)
    g.compile().invoke({"question": "add 2 and 3"}, run_id="r1")

    cost = attribute(trace, "r1")

    agent_cost = cost.node("agent")
    report_cost = cost.node("report")
    assert agent_cost.tokens > 0, "the agent's model calls must be attributed to its node"
    assert report_cost.tokens == 0
    assert cost.tokens == agent_cost.tokens
    assert cost.tokens == summarize(trace, "r1").tokens


def test_model_call_breakdown_is_a_subset_of_the_node_total(trace):
    """Sub-step tokens are inside the node's total, never additional to it."""
    harness, model = _tool_agent()
    g = GraphARC(S, name="demo", trace=trace, budget=Budget(max_iterations=20))
    agent = AgentNode(model, harness, name="agent", trace=trace, task_field="question")
    g.add_node("agent", agent, writes=agent.writes)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    g.compile().invoke({"question": "add 2 and 3"}, run_id="r1")

    cost = attribute(trace, "r1")

    assert len(cost.model_calls) == 2  # one tool turn, one answer turn
    call_tokens = sum(c.tokens for c in cost.model_calls)
    assert call_tokens > 0
    assert call_tokens == cost.tokens, "the node total is exactly the calls it made"
    assert all(c.parent_node == "agent" for c in cost.model_calls)


def test_cost_is_none_without_a_rate_and_says_so(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    cost = attribute(trace, "r1")

    assert cost.recorded_cost_usd is None
    assert cost.estimated_cost_usd is None
    assert cost.cost_usd is None
    assert cost.complete, "a run with zero tokens has nothing left unpriced"


def test_estimated_cost_prices_tokens_at_the_card_rate(trace):
    harness, model = _tool_agent()
    g = GraphARC(S, name="demo", trace=trace, budget=Budget(max_iterations=20))
    agent = AgentNode(model, harness, name="agent", trace=trace, task_field="question")
    g.add_node("agent", agent, writes=agent.writes)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    g.compile().invoke({"question": "add 2 and 3"}, run_id="r1")

    cost = attribute(trace, "r1", rates=RateCard(default=2.0))

    assert cost.tokens > 0
    assert cost.estimated_cost_usd == pytest.approx(cost.tokens / 1000 * 2.0)
    assert cost.recorded_cost_usd is None
    assert cost.cost_usd == cost.estimated_cost_usd
    assert cost.unpriced_tokens == 0 and cost.complete


def test_unpriced_tokens_are_reported_not_silently_zero(trace):
    harness, model = _tool_agent()
    g = GraphARC(S, name="demo", trace=trace, budget=Budget(max_iterations=20))
    agent = AgentNode(model, harness, name="agent", trace=trace, task_field="question")
    g.add_node("agent", agent, writes=agent.writes)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    g.compile().invoke({"question": "add 2 and 3"}, run_id="r1")

    # A card that prices a model this run never used.
    cost = attribute(trace, "r1", rates=RateCard(per_model={"openai/gpt-4o": 5.0}))

    assert cost.tokens > 0
    assert cost.cost_usd is None
    assert cost.unpriced_tokens == cost.tokens
    assert not cost.complete
    assert "lower bound" in cost.format()


def test_recorded_cost_beats_the_rate_card(trace):
    """A producer's own cost_usd is used as-is; the card never overrides it."""
    for step, (node, cost_usd) in enumerate(
        [("load", 0.01), ("think", None)], start=1
    ):
        trace.event(run_id="r1", graph="demo", node=node, phase="start", step=step,
                    thread_id="t1")
        trace.event(run_id="r1", graph="demo", node=node, phase="end", step=step,
                    thread_id="t1", duration_ms=1.0, tokens=1000, cost_usd=cost_usd,
                    model="openrouter/anthropic/claude-sonnet-4")

    cost = attribute(trace, "r1", rates=RateCard(default=3.0))

    assert cost.node("load").recorded_cost_usd == 0.01
    assert cost.node("load").estimated_cost_usd is None
    assert cost.node("think").estimated_cost_usd == pytest.approx(3.0)
    assert cost.recorded_cost_usd == 0.01
    assert cost.estimated_cost_usd == pytest.approx(3.0)
    assert cost.cost_usd == pytest.approx(3.01)


def test_two_models_in_one_node_are_priced_call_by_call(trace):
    """Blending a node at whichever model ran first would misprice it."""
    trace.event(run_id="r1", graph="demo", node="agent", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="agent:model", phase="model", step=2,
                tokens=1000, model="cheap/one")
    trace.event(run_id="r1", graph="demo", node="agent:model", phase="model", step=3,
                tokens=1000, model="pricey/two")
    trace.event(run_id="r1", graph="demo", node="agent", phase="end", step=1,
                duration_ms=1.0, tokens=2000)

    card = RateCard(per_model={"cheap/one": 1.0, "pricey/two": 20.0})
    cost = attribute(trace, "r1", rates=card)

    assert cost.node("agent").models == ["cheap/one", "pricey/two"]
    assert cost.estimated_cost_usd == pytest.approx(1.0 + 20.0)
    # The first-model blend would have been 2000/1000 * 1.0 == 2.0.
    assert cost.estimated_cost_usd != pytest.approx(2.0)


def test_a_model_with_no_rate_leaves_only_its_own_tokens_unpriced(trace):
    trace.event(run_id="r1", graph="demo", node="agent", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="agent:model", phase="model", step=2,
                tokens=1000, model="known/one")
    trace.event(run_id="r1", graph="demo", node="agent:model", phase="model", step=3,
                tokens=400, model="unknown/two")
    trace.event(run_id="r1", graph="demo", node="agent", phase="end", step=1,
                duration_ms=1.0, tokens=1400)

    cost = attribute(trace, "r1", rates=RateCard(per_model={"known/one": 2.0}))

    assert cost.estimated_cost_usd == pytest.approx(2.0)
    assert cost.unpriced_tokens == 400
    assert not cost.complete


def test_rate_card_matches_the_longest_prefix(trace):
    card = RateCard(
        per_model={"openrouter/": 1.0, "openrouter/anthropic/": 9.0},
        default=0.5,
    )

    assert card.rate_for("openrouter/anthropic/claude-sonnet-4") == 9.0
    assert card.rate_for("openrouter/openai/gpt-4o") == 1.0
    assert card.rate_for("local/llama") == 0.5
    assert card.rate_for(None) == 0.5
    assert RateCard().rate_for("anything") is None


def test_tokens_spent_inside_a_failed_node_are_reported_separately(trace):
    """The kernel's error event carries no tokens; the spend must not vanish."""
    trace.event(run_id="r1", graph="demo", node="agent", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="agent:model", phase="model", step=2,
                tokens=700)
    trace.event(run_id="r1", graph="demo", node="agent", phase="error", step=1,
                duration_ms=5.0, error="RuntimeError('boom')")

    cost = attribute(trace, "r1")

    assert cost.tokens == 0, "no end event means no node-level tokens; metrics agrees"
    assert summarize(trace, "r1").tokens == 0
    assert cost.tokens_before_error == 700
    assert cost.errors == 1
    assert "spent inside nodes that then failed" in cost.format()


def test_attribute_thread_sums_across_resumes(trace):
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "a"}, thread_id="t1", run_id="r1")
    compiled.invoke({"question": "b"}, thread_id="t1", run_id="r2")

    session = attribute_thread(trace, "t1", rates=RateCard(default=1.0))

    assert [r.run_id for r in session.runs] == ["r1", "r2"]
    assert session.tokens == sum(r.tokens for r in session.runs)
    load = next(n for n in session.per_node if n.node == "load")
    assert load.executions == 2
    assert session.duration_ms == pytest.approx(
        sum(r.duration_ms for r in session.runs), abs=0.05
    )


def test_cost_helpers_are_exported_from_observe():
    from grapharc.observe import by_node as exported_by_node
    from grapharc.observe import tokens_by_model as exported_tokens_by_model

    assert exported_by_node is by_node
    assert exported_tokens_by_model is tokens_by_model


def test_attribute_by_node_ranks_across_every_run(trace):
    trace.event(run_id="r1", graph="demo", node="cheap", phase="start", step=1)
    trace.event(run_id="r1", graph="demo", node="cheap", phase="end", step=1,
                duration_ms=1.0, tokens=10)
    trace.event(run_id="r2", graph="demo", node="pricey", phase="start", step=1)
    trace.event(run_id="r2", graph="demo", node="pricey", phase="end", step=1,
                duration_ms=1.0, tokens=900)

    ranked = by_node(trace, rates=RateCard(default=1.0))

    assert [n.node for n in ranked] == ["pricey", "cheap"]
    assert ranked[0].cost_usd == pytest.approx(0.9)


def test_tokens_by_model_reads_the_model_field(trace):
    trace.event(run_id="r1", graph="demo", node="a:model", phase="model", step=1,
                tokens=100, model="openrouter/anthropic/claude-sonnet-4")
    trace.event(run_id="r1", graph="demo", node="a:model", phase="model", step=2,
                tokens=50, model="openrouter/anthropic/claude-sonnet-4")
    trace.event(run_id="r1", graph="demo", node="b:model", phase="model", step=3,
                tokens=7, model="openai/gpt-4o")

    assert tokens_by_model(trace) == {
        "openrouter/anthropic/claude-sonnet-4": 150,
        "openai/gpt-4o": 7,
    }


def test_attribute_needs_a_run_id_unless_given_a_replayed_run(trace):
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    with pytest.raises(TypeError, match="needs a run_id"):
        attribute(trace)
    assert attribute(replay(trace, "r1")).run_id == "r1"


# --------------------------------------------------------------------------
# trace format: additive fields, and the thread_summary complexity fix
# --------------------------------------------------------------------------


def test_new_fields_are_omitted_when_unset(trace):
    """A trace from a producer that reports no cost is byte-identical to before."""
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    for line in trace.path.read_text().splitlines():
        record = json.loads(line)
        assert "cost_usd" not in record
        assert "model" not in record


def test_old_trace_lines_without_the_new_fields_still_parse(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts": "2026-07-01T00:00:00.000+00:00",
                "run_id": "r1",
                "graph": "demo",
                "node": "load",
                "phase": "end",
                "step": 1,
                "duration_ms": 1.5,
                "tokens": 3,
            }
        )
        + "\n"
    )

    run = replay(path, "r1")

    assert run.path == ["load"]
    assert run.executions[0].cost_usd is None
    assert TraceRecorder(path).read_events()[0].model is None


def test_thread_summary_matches_a_full_rescan(trace):
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "a"}, thread_id="t1", run_id="r1")
    compiled.invoke({"question": "b"}, thread_id="t1", run_id="r2")
    compiled.invoke({"question": "c"}, thread_id="t2", run_id="r3")

    def brute_force(thread_id):
        events = [e for e in trace.read_events() if e.thread_id == thread_id]
        if not events:
            return (0, 0)
        return (max(e.step for e in events), max(e.attempt for e in events))

    for thread_id in ("t1", "t2", "missing"):
        assert trace.thread_summary(thread_id) == brute_force(thread_id)


def test_thread_summary_reads_each_byte_about_once(trace, monkeypatch):
    """§1.7: the old implementation re-read the whole file on every invoke.

    Counted in bytes off the file rather than through a private counter, so the
    assertion holds for any implementation: re-reading the file N times to
    answer N calls cannot come in under a couple of file lengths.
    """
    real_open = pathlib.Path.open
    read = {"bytes": 0}

    class _CountingHandle:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def __iter__(self):
            for line in self._inner:
                read["bytes"] += len(line)
                yield line

        def read(self, *args):
            data = self._inner.read(*args)
            read["bytes"] += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def spy(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        return _CountingHandle(handle) if self == trace.path else handle

    monkeypatch.setattr(pathlib.Path, "open", spy)
    calls = 40
    for i in range(calls):
        trace.event(run_id=f"r{i}", graph="demo", node="n", phase="end", step=i + 1,
                    thread_id="t1")
        trace.thread_summary("t1")

    size = trace.path.stat().st_size
    assert trace.thread_summary("t1") == (calls, 1)
    # Linear: every byte is folded in once. Quadratic would be ~20x the file.
    assert read["bytes"] <= size * 2, f"read {read['bytes']} bytes of a {size}-byte file"


def test_thread_summary_sees_lines_written_by_another_recorder(trace, tmp_path):
    """A resumed run in a fresh process must be seeded from the file on disk."""
    trace.event(run_id="r1", graph="demo", node="n", phase="end", step=7,
                thread_id="t1", attempt=3)
    fresh = TraceRecorder(trace.path)

    assert fresh.thread_summary("t1") == (7, 3)

    trace.event(run_id="r2", graph="demo", node="n", phase="end", step=9,
                thread_id="t1", attempt=4)

    assert fresh.thread_summary("t1") == (9, 4)


def test_thread_summary_ignores_a_partially_written_line(trace):
    trace.event(run_id="r1", graph="demo", node="n", phase="end", step=2, thread_id="t1")
    with trace.path.open("a", encoding="utf-8") as f:
        f.write('{"ts": "2026-07-01T00:00:00.000+00:00", "run_id": "r2", "step": 99')

    assert trace.thread_summary("t1") == (2, 1)

    with trace.path.open("a", encoding="utf-8") as f:
        f.write(', "thread_id": "t1", "attempt": 1, "graph": "demo", '
                '"node": "n", "phase": "end"}\n')

    assert trace.thread_summary("t1") == (99, 1)


def test_thread_summary_recovers_from_a_truncated_file(trace):
    trace.event(run_id="r1", graph="demo", node="n", phase="end", step=5, thread_id="t1")
    assert trace.thread_summary("t1") == (5, 1)

    trace.path.write_text("")  # the file was rotated out from under us
    assert trace.thread_summary("t1") == (0, 0)

    trace.event(run_id="r2", graph="demo", node="n", phase="end", step=2, thread_id="t1")
    assert trace.thread_summary("t1") == (2, 1)


def test_thread_summary_is_safe_under_concurrent_writers(trace):
    def writer(index):
        for step in range(1, 21):
            trace.event(run_id=f"r{index}", graph="demo", node="n", phase="end",
                        step=step, thread_id=f"t{index}")
            trace.thread_summary(f"t{index}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(4):
        assert trace.thread_summary(f"t{i}") == (20, 1)
    assert len(trace.read_events()) == 80


def test_resume_still_seeds_step_and_attempt_from_the_thread(trace):
    """The behaviour the O(n^2) scan existed for must survive the rewrite."""
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "a"}, thread_id="t1", run_id="r1")
    first = [e.step for e in trace.read_events("r1")]

    compiled.invoke({"question": "b"}, thread_id="t1", run_id="r2")
    second = trace.read_events("r2")

    # The topology event always carries step 0 — it states shape, not order —
    # so the monotonicity claim is about the *work* steps.
    steps = [e.step for e in second if e.phase != "topology"]
    assert min(steps) > max(first)
    assert {e.attempt for e in second} == {2}


def test_run_ids_are_listed_in_first_seen_order(trace):
    compiled = _linear_graph(trace)
    compiled.invoke({"question": "a"}, run_id="rA")
    compiled.invoke({"question": "b"}, run_id="rB")

    assert trace.run_ids() == ["rA", "rB"]


def test_spans_are_empty_when_no_timestamp_can_be_parsed(trace):
    """An unparseable `ts` must not become a span at the Unix epoch."""
    trace.record(
        TraceEvent(ts="not-a-timestamp", run_id="r1", graph="demo", node="n",
                   phase="end", step=1, duration_ms=1.0)
    )

    assert to_spans(trace, "r1") == []
    assert replay(trace, "r1").wall_ms is None
    assert Span(name="n", span_id="s", start_unix_nano=1, end_unix_nano=3).duration_ns == 2


# --------------------------------------------------------------------------
# A run with no kernel span: `grapharc agent` drives an AgentNode directly
# against a RunContext, so nothing ever emits "start"/"end". Every event is an
# orphan sub-event, and a reader that only understands node spans reports the
# run as empty — which is what the CLI's own help forbids: "metrics and the
# audit trail cannot disagree: there is only one record."
# --------------------------------------------------------------------------


def _standalone_agent_run(trace, run_id="cli1"):
    """Reproduce `grapharc agent`: an AgentNode driven with no enclosing graph."""
    harness, model = _tool_agent()
    node = AgentNode(model, harness, name="agent", trace=trace)
    ctx = RunContext(run_id=run_id, graph="cli-agent", meter=BudgetMeter(Budget()))
    result = node.run("add 2 and 3", ctx)
    return result


def test_a_run_with_no_kernel_span_still_records_its_events(trace):
    """The premise of the other tests here: the trace itself is not empty."""
    _standalone_agent_run(trace)
    events = trace.read_events("cli1")

    assert [e.phase for e in events] == ["model", "tool", "model", "stop"]
    assert sum(e.tokens or 0 for e in events) > 0
    assert not any(e.phase in ("start", "end") for e in events)


def test_metrics_do_not_report_a_spending_run_as_zero(trace):
    """`tokens: 0` for a run that spent tokens is the audit trail disagreeing."""
    _standalone_agent_run(trace)
    events = trace.read_events("cli1")
    spent = sum(e.tokens or 0 for e in events)

    metrics = summarize(trace, "cli1")

    assert metrics is not None
    assert metrics.tokens == spent
    assert metrics.duration_ms > 0


def test_metrics_report_why_a_spanless_run_stopped(trace):
    """`termination_reason` lives on the agent's "stop" event, not on an "end"."""
    result = _standalone_agent_run(trace)

    metrics = summarize(trace, "cli1")

    assert metrics.termination_reason == result.termination_reason.value


def test_metrics_name_the_work_a_spanless_run_did(trace):
    """`nodes_executed` is 0 with no kernel node — the phase counts carry it."""
    _standalone_agent_run(trace)

    metrics = summarize(trace, "cli1")

    assert metrics.nodes_executed == 0, "no kernel node ran; that stays true"
    assert metrics.events == 4
    assert metrics.per_phase == {"model": 2, "tool": 1, "stop": 1}


def test_viz_renders_the_path_of_a_spanless_run(trace):
    """A run that did work must not render as `no events`."""
    _standalone_agent_run(trace)

    diagram = to_mermaid(trace, "cli1")

    assert "no events" not in diagram
    assert "agent:add" in diagram
    assert diagram.count("-->") >= 3


def test_cost_and_metrics_agree_on_a_run_with_no_kernel_span(trace):
    """The invariant this module exists to protect, on the spanless path."""
    _standalone_agent_run(trace)

    cost = attribute(trace, "cli1")
    metrics = summarize(trace, "cli1")

    assert cost.tokens == metrics.tokens
    assert cost.tokens > 0


def test_orphan_model_calls_are_still_attributed_and_priced(trace):
    """A model call outside any node is still a model call that cost money."""
    _standalone_agent_run(trace)

    cost = attribute(trace, "cli1", rates=RateCard(default=1.0))

    assert [c.tokens for c in cost.model_calls] != []
    assert cost.estimated_cost_usd is not None
    assert cost.estimated_cost_usd > 0


# --------------------------------------------------------------------------
# §10.4 / §12.5 — recorded cost. Both gateways capture the provider's own
# per-call price; until this landed nothing carried it onto a trace event, so
# `recorded_cost_usd` was always None and every money figure `observe.cost`
# reported was a rate-card estimate priced off token counts.
# --------------------------------------------------------------------------


def _priced_graph(trace, cost_usd=0.0025):
    """A one-node graph whose node calls a model that reports a real price."""
    model = ScriptedChatModel(responses=["priced"], cost_usd=cost_usd, model_name="acme/fast")
    g = GraphARC(S, name="demo", trace=trace)
    g.add_node("think", lambda s: {"answer": model.invoke("go").content}, writes={"answer"})
    g.add_edge(START, "think")
    g.add_edge("think", END)
    return g.compile()


def test_a_nodes_end_event_carries_the_price_its_model_calls_reported(trace):
    _priced_graph(trace).invoke({"question": "why"}, run_id="r1")

    ends = [e for e in trace.read_events("r1") if e.phase == "end"]

    assert [e.cost_usd for e in ends] == [pytest.approx(0.0025)]
    assert ends[0].model == "acme/fast"


def test_a_reported_price_is_reported_as_recorded_never_as_an_estimate(trace):
    """The distinction `observe.cost` exists to keep: measured vs. guessed."""
    _priced_graph(trace).invoke({"question": "why"}, run_id="r1")

    cost = attribute(trace, "r1", rates=RateCard(default=999.0))

    assert cost.recorded_cost_usd == pytest.approx(0.0025)
    # A rate card that would price this absurdly is not consulted at all: a
    # recorded price wins over an estimate rather than being averaged with it.
    assert cost.estimated_cost_usd is None
    assert cost.unpriced_tokens == 0
    assert cost.complete


def test_a_backend_that_reports_no_price_still_falls_back_to_an_estimate(trace):
    """The pre-existing path must keep working; None is not zero."""
    _linear_graph(trace).invoke({"question": "why"}, run_id="r1")

    cost = attribute(trace, "r1", rates=RateCard(default=1.0))

    assert cost.recorded_cost_usd is None
    assert cost.estimated_cost_usd is not None


def test_prices_from_several_nodes_add_up_across_the_run(trace):
    model = ScriptedChatModel(
        responses=["a", "b"], on_exhausted="repeat", cost_usd=0.001, model_name="acme/fast"
    )
    g = GraphARC(S, name="demo", trace=trace)
    g.add_node("one", lambda s: {"answer": model.invoke("x").content}, writes={"answer"})
    g.add_node("two", lambda s: {"hops": 1, "answer": model.invoke("y").content},
               writes={"hops", "answer"})
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", END)
    g.compile().invoke({"question": "why"}, run_id="r1")

    cost = attribute(trace, "r1")

    assert cost.recorded_cost_usd == pytest.approx(0.002)
    assert cost.node("one").recorded_cost_usd == pytest.approx(0.001)
    assert cost.node("two").recorded_cost_usd == pytest.approx(0.001)


def test_an_agents_model_events_carry_the_price_of_each_call(trace):
    """The per-call breakdown, not just the node total."""
    harness, model = _tool_agent()
    model.cost_usd = 0.0004
    model.model_name = "acme/fast"
    node = AgentNode(model, harness, name="agent", trace=trace)
    ctx = RunContext(run_id="cli1", graph="cli-agent", meter=BudgetMeter(Budget()))
    node.run("add 2 and 3", ctx)

    models = [e for e in trace.read_events("cli1") if e.phase == "model"]

    assert models, "the agent must record its model calls"
    assert all(e.cost_usd == pytest.approx(0.0004) for e in models)
    assert all(e.model == "acme/fast" for e in models)


def test_a_spanless_agent_run_reports_recorded_cost_not_an_estimate(trace):
    """§12.5 and the spanless-run fix have to hold at the same time."""
    harness, model = _tool_agent()
    model.cost_usd = 0.0004
    node = AgentNode(model, harness, name="agent", trace=trace)
    ctx = RunContext(run_id="cli1", graph="cli-agent", meter=BudgetMeter(Budget()))
    node.run("add 2 and 3", ctx)

    cost = attribute(trace, "cli1", rates=RateCard(default=999.0))

    assert cost.recorded_cost_usd is not None
    assert cost.recorded_cost_usd > 0
