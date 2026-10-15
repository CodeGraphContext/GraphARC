"""Provider-reported prices survive planning and failed governed work (#138)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from grapharc.observe import RateCard, TraceRecorder, attribute, attribute_tenant
from grapharc.observe.cost import attribute_thread
from grapharc.observe.metrics import summarize
from grapharc.planner import PlannerNode
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import (
    END,
    START,
    GraphARC,
    GraphRoutingError,
    RunContext,
    StateTypeError,
    WritePermissionError,
)
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    answer: str = ""
    proposal: object = None
    count: int = 0


def context():
    return RunContext(
        run_id="run", thread_id="thread", tenant="acme", graph="g", meter=BudgetMeter(Budget())
    )


def assert_price(trace, price):
    run = attribute(trace, "run")
    assert run.recorded_cost_usd == pytest.approx(price)
    assert attribute_thread(trace, "thread").recorded_cost_usd == pytest.approx(price)
    assert attribute_tenant(trace, "acme").recorded_cost_usd == pytest.approx(price)
    assert run.unpriced_tokens == 0
    assert run.tokens == summarize(trace, "run").tokens > 0


@pytest.mark.parametrize("reply", ['{"nodes": []}', "not a proposal"])
@pytest.mark.parametrize("price", [None, 0.0, 0.003])
@pytest.mark.parametrize("structured", [False, True])
def test_planner_records_the_call_price_even_when_parsing_fails(tmp_path, reply, price, structured):
    class StructuredModel(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(
                lambda messages: {
                    "raw": self.invoke(messages),
                    "parsed": None,
                    "parsing_error": ValueError("parse failed"),
                }
            )

    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = StructuredModel(responses=[reply], cost_usd=price, model_name="acme/planner")
    ctx = context()
    result = PlannerNode(model, trace=trace, structured=structured).propose("plan", ctx)
    assert result.ok == (reply.startswith("{"))
    event, = trace.read_events()
    assert event.phase == "plan"
    assert event.cost_usd == price
    assert ctx.meter.tokens == result.tokens == event.tokens > 0
    cost = attribute(trace, "run", rates=RateCard(default=2.0))
    if price is None:
        assert cost.recorded_cost_usd is None
        assert cost.estimated_cost_usd == pytest.approx(result.tokens / 1000 * 2)
    else:
        assert event.model == "acme/planner"
        assert_price(trace, price)
        assert cost.estimated_cost_usd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("async_node", [False, True])
@pytest.mark.parametrize("failure", ["body", "write", "type", "routing", "budget", "manual_budget"])
async def test_all_failed_node_terminals_retain_provider_usage(tmp_path, async_node, failure):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=["done"], cost_usd=0.002, model_name="acme/worker")
    graph = GraphARC(State, name="g", trace=trace)

    def finish(ctx):
        if failure == "body":
            raise ValueError("failed after spending")
        if failure == "write":
            return {"undeclared": True}
        if failure == "type":
            return {"count": "not an integer"}
        if failure == "routing":
            return Command(goto="unknown")
        if failure == "manual_budget":
            ctx.meter.charge_tokens(100)
        return {"answer": "done"}

    def body(state, ctx):
        model.invoke("work")
        return finish(ctx)

    async def abody(state, ctx):
        await model.ainvoke("work")
        return finish(ctx)

    graph.add_node("worker", abody if async_node else body, writes={"answer", "count"})
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    compiled = graph.compile()
    budget = Budget(max_tokens=1 if failure == "budget" else 50)
    expected = {
        "body": ValueError,
        "write": WritePermissionError,
        "type": StateTypeError,
        "routing": GraphRoutingError,
        "budget": BudgetExceeded,
        "manual_budget": BudgetExceeded,
    }[failure]
    with pytest.raises(expected):
        await compiled.ainvoke(
            {}, run_id="run", thread_id="thread", tenant="acme", budget=budget
        )
    terminal, = [e for e in trace.read_events() if e.phase == "error"]
    assert terminal.cost_usd == pytest.approx(0.002)
    assert terminal.model == "acme/worker"
    assert terminal.tokens == compiled.last_run.meter.tokens > 0
    assert_price(trace, 0.002)


def test_nested_planner_price_and_tokens_are_counted_once(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=['{"nodes": []}'], cost_usd=0.003)
    planner = PlannerNode(model, trace=trace, structured=False)
    graph = GraphARC(State, name="g", trace=trace)
    graph.add_node("planner", lambda state, ctx: {"proposal": planner.propose("plan", ctx)},
                   writes={"proposal"})
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    compiled = graph.compile()
    compiled.invoke({}, run_id="run", thread_id="thread", tenant="acme")
    plan, = [e for e in trace.read_events() if e.phase == "plan"]
    end, = [e for e in trace.read_events() if e.phase == "end"]
    assert plan.cost_usd == end.cost_usd == pytest.approx(0.003)
    assert plan.tokens == end.tokens == compiled.last_run.meter.tokens
    assert_price(trace, 0.003)


def test_nested_planner_budget_failure_preserves_price_and_stops_more_calls(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=['{"nodes": []}'] * 2, cost_usd=0.003)
    planner = PlannerNode(model, trace=trace, structured=False)
    graph = GraphARC(State, name="g", trace=trace)

    def body(state, ctx):
        planner.propose("plan", ctx)
        planner.propose("more", ctx)
        return {}

    graph.add_node("planner", body, writes=set())
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    with pytest.raises(BudgetExceeded):
        graph.compile().invoke({}, run_id="run", thread_id="thread", tenant="acme",
                               budget=Budget(max_tokens=1))
    assert model.call_count == 1
    assert_price(trace, 0.003)


@pytest.mark.asyncio
async def test_failed_parallel_node_is_not_billed_for_its_siblings(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    graph = GraphARC(State, name="g", trace=trace)
    first = asyncio.Event()
    second = asyncio.Event()
    models = [ScriptedChatModel(responses=["done"], cost_usd=p) for p in (0.002, 0.005)]

    async def bad(state):
        await models[0].ainvoke("first")
        first.set()
        await second.wait()
        raise ValueError("failed after sibling spent")

    async def good(state):
        await first.wait()
        await models[1].ainvoke("second")
        second.set()
        return {}

    for name, body in [("bad", bad), ("good", good)]:
        graph.add_node(name, body, writes=set())
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    with pytest.raises(ValueError):
        await graph.compile().ainvoke({}, run_id="run", thread_id="thread", tenant="acme")
    terminal, = [e for e in trace.read_events() if e.phase == "error"]
    assert terminal.tokens == models[0].last_usage["total_tokens"]
    assert terminal.cost_usd == pytest.approx(0.002)
    assert_price(trace, 0.007)


def test_a_failed_planner_call_cannot_reuse_an_earlier_calls_price(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=['{"nodes": []}'], cost_usd=0.003)
    planner = PlannerNode(model, trace=trace, structured=False)
    ctx = context()
    assert planner.propose("plan", ctx).ok
    assert not planner.propose("no more replies", ctx).ok
    first, failed = trace.read_events()
    assert first.cost_usd == pytest.approx(0.003)
    assert failed.cost_usd is None
    assert failed.model is None
    assert failed.tokens == 0
    assert_price(trace, 0.003)


def test_unpriced_failed_node_remains_unpriced_unless_a_rate_is_supplied(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=["done"])
    graph = GraphARC(State, name="g", trace=trace)

    def body(state):
        model.invoke("work")
        raise ValueError("unpriced failure")

    graph.add_node("worker", body, writes=set())
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    with pytest.raises(ValueError):
        graph.compile().invoke({}, run_id="run")
    terminal, = [e for e in trace.read_events() if e.phase == "error"]
    assert terminal.cost_usd is None
    cost = attribute(trace, "run")
    assert cost.recorded_cost_usd is None
    assert cost.unpriced_tokens == terminal.tokens > 0
    estimate = attribute(trace, "run", rates=RateCard(default=2.0))
    assert estimate.recorded_cost_usd is None
    assert estimate.estimated_cost_usd == pytest.approx(terminal.tokens / 1000 * 2)
