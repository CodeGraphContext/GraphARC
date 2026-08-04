"""The async kernel: ROADMAP §1.1 (async), §1.2 (`Command` returns), §1.3 (state access).

Async is not a second, looser execution path. Every discipline the sync wrapper
enforces — declared writes, deep-copy isolation, typed writes, budgets, the usage
callback, traces, fail-closed entry points — is asserted here against `ainvoke`
and `astream` as well.
"""

import asyncio
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Send
from pydantic import BaseModel

from grapharc.runtime.budget import Budget, BudgetExceeded, NodeDeadlineExceeded
from grapharc.runtime.graph import (
    END,
    START,
    AsyncNodeError,
    GraphARC,
    GraphRoutingError,
    MissingRunContextError,
    StateTypeError,
    WritePermissionError,
)
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class S(GraphARCState):
    a: int = 0
    b: int = 0
    note: str = ""


def _graph(fn, *, writes, name="async", trace=None, budget=None, node="n"):
    g = GraphARC(S, name=name, trace=trace, budget=budget)
    g.add_node(node, fn, writes=writes)
    g.add_edge(START, node)
    g.add_edge(node, END)
    return g.compile()


# -- §1.1 async nodes ------------------------------------------------------


@pytest.mark.asyncio
async def test_an_async_node_runs_and_its_declared_write_lands():
    async def bump(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": state.a + 1}

    result = await _graph(bump, writes={"a"}).ainvoke({"a": 41})
    assert result["a"] == 42


@pytest.mark.asyncio
async def test_an_async_node_receives_the_run_context_when_it_asks_for_one():
    seen = {}

    async def peek(state: S, ctx) -> dict:
        await asyncio.sleep(0)
        seen["run_id"] = ctx.run_id
        seen["budget"] = ctx.meter.budget.max_iterations
        return {"a": 1}

    compiled = _graph(peek, writes={"a"}, budget=Budget(max_iterations=9))
    await compiled.ainvoke({}, run_id="rid-async")
    assert seen == {"run_id": "rid-async", "budget": 9}


@pytest.mark.asyncio
async def test_an_undeclared_write_from_an_async_node_raises_and_is_traced(trace):
    async def sneak(state: S) -> dict:
        await asyncio.sleep(0)
        return {"b": 1}

    compiled = _graph(sneak, writes={"a"}, trace=trace)
    with pytest.raises(WritePermissionError, match="undeclared"):
        await compiled.ainvoke({}, run_id="r-undeclared")
    errors = [e for e in trace.read_events("r-undeclared") if e.phase == "error"]
    assert errors and "undeclared" in errors[0].error


@pytest.mark.asyncio
async def test_an_async_write_is_type_validated_like_a_sync_one():
    async def wrong(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": "not-an-int"}

    with pytest.raises(StateTypeError, match="expected int"):
        await _graph(wrong, writes={"a"}).ainvoke({})


class Inner(BaseModel):
    text: str


class NestedState(GraphARCState):
    items: list[Inner] = []
    seen: str = ""


@pytest.mark.asyncio
async def test_an_async_node_gets_a_deep_copy_not_the_live_state():
    async def tamper(state: NestedState) -> None:
        await asyncio.sleep(0)
        state.items[0].text = "TAMPERED"

    async def observe(state: NestedState) -> dict:
        await asyncio.sleep(0)
        return {"seen": state.items[0].text}

    g = GraphARC(NestedState, name="deep")
    g.add_node("tamper", tamper, writes=set())
    g.add_node("observe", observe, writes={"seen"})
    g.add_edge(START, "tamper")
    g.add_edge("tamper", "observe")
    g.add_edge("observe", END)
    result = await g.compile().ainvoke({"items": [Inner(text="original")]})
    assert result["seen"] == "original"
    assert result["items"][0].text == "original"


@pytest.mark.asyncio
async def test_an_async_node_writes_start_and_end_trace_events(trace):
    async def work(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 3}

    await _graph(work, writes={"a"}, trace=trace).ainvoke({}, run_id="r-trace")
    phases = [(e.node, e.phase) for e in trace.read_events("r-trace")]
    assert phases == [("topology", "topology"), ("n", "start"), ("n", "end")]
    end = trace.read_events("r-trace")[2]
    assert end.state_delta == {"a": 3}
    assert end.duration_ms is not None


@pytest.mark.asyncio
async def test_a_budget_stops_an_async_loop():
    async def loop(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": state.a + 1}

    g = GraphARC(S, name="loop", budget=Budget(max_iterations=5))
    g.add_node("loop", loop, writes={"a"})
    g.add_edge(START, "loop")
    g.add_conditional_edge("loop", lambda s: "again", {"again": "loop", "done": END})
    with pytest.raises(BudgetExceeded, match="max_iterations"):
        await g.compile().ainvoke({"a": 0})


@pytest.mark.asyncio
async def test_a_budget_stops_an_async_stream_too():
    async def loop(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": state.a + 1}

    g = GraphARC(S, name="loop", budget=Budget(max_iterations=3))
    g.add_node("loop", loop, writes={"a"})
    g.add_edge(START, "loop")
    g.add_conditional_edge("loop", lambda s: "again", {"again": "loop", "done": END})
    compiled = g.compile()
    with pytest.raises(BudgetExceeded, match="max_iterations"):
        async for _ in compiled.astream({"a": 0}):
            pass


@pytest.mark.asyncio
async def test_an_async_router_returning_an_unmapped_key_raises_a_grapharc_error():
    """The declaration-time mapping check is shape-blind; this one is not.

    An `async def` router has to be wrapped in an async wrapper or LangGraph
    would await the check's return value instead of the router's.
    """

    async def step(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": state.a + 1}

    async def route(state: S) -> str:
        await asyncio.sleep(0)
        return "dnoe"

    g = GraphARC(S, name="loop")
    g.add_node("step", step, writes={"a"})
    g.add_edge(START, "step")
    g.add_conditional_edge("step", route, {"again": "step", "done": END})
    with pytest.raises(GraphRoutingError, match="'dnoe'"):
        await g.compile().ainvoke({"a": 0})


@pytest.mark.asyncio
async def test_an_async_router_that_maps_still_routes():
    async def step(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": state.a + 1}

    async def route(state: S) -> str:
        await asyncio.sleep(0)
        return "done" if state.a >= 2 else "again"

    g = GraphARC(S, name="loop")
    g.add_node("step", step, writes={"a"})
    g.add_edge(START, "step")
    g.add_conditional_edge("step", route, {"again": "step", "done": END})
    assert (await g.compile().ainvoke({"a": 0}))["a"] == 2


@pytest.mark.asyncio
async def test_astream_yields_one_update_per_node():
    async def one(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 1}

    async def two(state: S) -> dict:
        await asyncio.sleep(0)
        return {"b": 2}

    g = GraphARC(S, name="stream")
    g.add_node("one", one, writes={"a"})
    g.add_node("two", two, writes={"b"})
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", END)
    chunks = [c async for c in g.compile().astream({}, stream_mode="updates")]
    assert [next(iter(c)) for c in chunks] == ["one", "two"]


@pytest.mark.asyncio
async def test_astream_events_carries_the_run_through_langchains_event_stream():
    async def work(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 1}

    compiled = _graph(work, writes={"a"})
    names = set()
    async for event in compiled.astream_events({}, version="v2"):
        if event["event"] == "on_chain_end":
            names.add(event["name"])
    assert "n" in names


@pytest.mark.asyncio
async def test_astream_events_refuses_a_version_it_cannot_iterate():
    compiled = _graph(lambda s: {"a": 1}, writes={"a"})
    with pytest.raises(ValueError, match="v1' or 'v2'"):
        async for _ in compiled.astream_events({}, version="v3"):
            pass


@pytest.mark.asyncio
async def test_two_async_nodes_in_one_superstep_really_run_concurrently():
    """Not a timing heuristic: each node waits for the other at a barrier, so a
    serial executor deadlocks and only genuine concurrency finishes."""
    barrier = asyncio.Barrier(2)

    async def left(state: S) -> dict:
        await barrier.wait()
        return {"a": 1}

    async def right(state: S) -> dict:
        await barrier.wait()
        return {"b": 2}

    g = GraphARC(S, name="parallel")
    g.add_node("left", left, writes={"a"})
    g.add_node("right", right, writes={"b"})
    g.add_edge(START, "left")
    g.add_edge(START, "right")
    g.add_edge("left", END)
    g.add_edge("right", END)
    result = await asyncio.wait_for(g.compile().ainvoke({}), timeout=5)
    assert (result["a"], result["b"]) == (1, 2)


@pytest.mark.asyncio
async def test_sync_nodes_still_run_under_ainvoke():
    g = GraphARC(S, name="mixed")
    g.add_node("sync", lambda s: {"a": 1}, writes={"a"})

    async def later(state: S) -> dict:
        await asyncio.sleep(0)
        return {"b": state.a + 1}

    g.add_node("async", later, writes={"b"})
    g.add_edge(START, "sync")
    g.add_edge("sync", "async")
    g.add_edge("async", END)
    result = await g.compile().ainvoke({})
    assert (result["a"], result["b"]) == (1, 2)


@pytest.mark.asyncio
async def test_a_sync_node_undeclared_write_still_raises_under_ainvoke():
    with pytest.raises(WritePermissionError, match="undeclared"):
        await _graph(lambda s: {"b": 1}, writes={"a"}).ainvoke({})


# -- fail-closed entry points ---------------------------------------------


@pytest.mark.asyncio
async def test_raw_langgraph_ainvoke_fails_closed():
    async def work(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 1}

    compiled = _graph(work, writes={"a"})
    with pytest.raises(MissingRunContextError, match="bypass budgets"):
        await compiled.inner.ainvoke({"a": 0})


@pytest.mark.asyncio
async def test_raw_langgraph_astream_fails_closed():
    async def work(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 1}

    compiled = _graph(work, writes={"a"})
    with pytest.raises(MissingRunContextError, match="bypass budgets"):
        async for _ in compiled.inner.astream({"a": 0}):
            pass


def test_invoke_refuses_a_graph_with_async_nodes_before_anything_runs(trace):
    """A half-run graph is worse than a rejected one: the sync node must not
    have executed by the time the coroutine node is discovered."""

    async def later(state: S) -> dict:
        await asyncio.sleep(0)
        return {"b": 1}

    g = GraphARC(S, name="mixed", trace=trace)
    g.add_node("first", lambda s: {"a": 1}, writes={"a"})
    g.add_node("second", later, writes={"b"})
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    compiled = g.compile()
    with pytest.raises(AsyncNodeError, match="second"):
        compiled.invoke({}, run_id="r-refused")
    assert trace.read_events("r-refused") == []


def test_stream_refuses_a_graph_with_async_nodes():
    async def later(state: S) -> dict:
        await asyncio.sleep(0)
        return {"a": 1}

    compiled = _graph(later, writes={"a"})
    with pytest.raises(AsyncNodeError, match="astream"):
        list(compiled.stream({}))


# -- budget metering inside async nodes ------------------------------------


@pytest.mark.asyncio
async def test_a_model_call_awaited_in_an_async_node_charges_the_run():
    model = ScriptedChatModel(responses=["hello there"])

    async def ask(state: S, ctx) -> dict:
        message = await model.ainvoke("hi")
        return {"note": str(message.content)}

    compiled = _graph(ask, writes={"note"})
    result = await compiled.ainvoke({})
    assert result["note"] == "hello there"
    # Charged by the usage callback, not by the node: `ask` never touches the meter.
    assert compiled.last_run.meter.tokens > 0


@pytest.mark.asyncio
async def test_max_tokens_stops_an_async_run():
    model = ScriptedChatModel(responses=["a fairly long scripted answer"], on_exhausted="repeat")

    async def ask(state: S) -> dict:
        message = await model.ainvoke("hi")
        return {"note": str(message.content)}

    compiled = _graph(ask, writes={"note"}, budget=Budget(max_tokens=2))
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        await compiled.ainvoke({})
    assert compiled.last_run.meter.tokens >= 2


@pytest.mark.asyncio
async def test_tokens_charged_in_an_async_node_reach_the_trace(trace):
    model = ScriptedChatModel(responses=["hello there"])

    async def ask(state: S) -> dict:
        await model.ainvoke("hi")
        return {"a": 1}

    await _graph(ask, writes={"a"}, trace=trace).ainvoke({}, run_id="r-tokens")
    end = [e for e in trace.read_events("r-tokens") if e.phase == "end"][0]
    assert end.tokens and end.tokens > 0


# -- max_seconds against a coroutine ---------------------------------------


@pytest.mark.asyncio
async def test_max_seconds_cuts_off_an_async_node_that_overruns():
    async def slow(state: S) -> dict:
        await asyncio.sleep(30)
        return {"a": 1}

    compiled = _graph(slow, writes={"a"}, budget=Budget(max_seconds=0.1))
    started = time.monotonic()
    with pytest.raises(NodeDeadlineExceeded, match="max_seconds"):
        await compiled.ainvoke({})
    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_an_async_node_that_swallows_one_cancellation_gets_another():
    """One cancellation is one chance to be caught, so the guard re-arms; and
    even a node that survives to the end has its writes rejected."""
    swallowed = []

    async def stubborn(state: S) -> dict:
        for _ in range(10):
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                swallowed.append(1)
        return {"a": 1}

    compiled = _graph(stubborn, writes={"a"}, budget=Budget(max_seconds=0.1))
    with pytest.raises(NodeDeadlineExceeded, match="max_seconds"):
        await compiled.ainvoke({})
    assert len(swallowed) >= 2


@pytest.mark.asyncio
async def test_max_seconds_still_bites_a_sync_node_driven_from_ainvoke():
    """`ainvoke` runs sync nodes on a worker thread, where `deadline_guard`
    falls back to injecting an async exception. That fallback has to keep
    working, or the async entry point would be a way to buy unbounded time."""

    def spin(state: S) -> dict:
        end = time.monotonic() + 3
        while time.monotonic() < end:
            pass
        return {"a": 1}

    compiled = _graph(spin, writes={"a"}, budget=Budget(max_seconds=0.1))
    started = time.monotonic()
    with pytest.raises(NodeDeadlineExceeded, match="max_seconds"):
        await compiled.ainvoke({})
    assert time.monotonic() - started < 2.5


@pytest.mark.asyncio
async def test_an_async_node_inside_the_deadline_is_left_alone():
    async def quick(state: S) -> dict:
        await asyncio.sleep(0.01)
        return {"a": 7}

    compiled = _graph(quick, writes={"a"}, budget=Budget(max_seconds=10))
    assert (await compiled.ainvoke({}))["a"] == 7


# -- §1.2 Command returns --------------------------------------------------


def _command_graph(fn, *, writes, trace=None):
    g = GraphARC(S, name="cmd", trace=trace)
    g.add_node("router", fn, writes=writes)
    g.add_node("landing", lambda s: {"b": s.a * 10}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    return g.compile()


def test_a_command_update_lands_and_its_goto_routes():
    compiled = _command_graph(
        lambda s: Command(update={"a": 4}, goto="landing"), writes={"a"}
    )
    result = compiled.invoke({})
    assert result["a"] == 4
    assert result["b"] == 40


def test_a_command_update_with_an_undeclared_write_raises(trace):
    compiled = _command_graph(
        lambda s: Command(update={"b": 1}, goto="landing"), writes={"a"}, trace=trace
    )
    with pytest.raises(WritePermissionError, match="undeclared"):
        compiled.invoke({}, run_id="r-cmd")
    errors = [e for e in trace.read_events("r-cmd") if e.phase == "error"]
    assert errors and "undeclared" in errors[0].error


def test_a_command_update_is_type_validated():
    compiled = _command_graph(
        lambda s: Command(update={"a": "nope"}, goto="landing"), writes={"a"}
    )
    with pytest.raises(StateTypeError, match="expected int"):
        compiled.invoke({})


def test_the_value_a_command_writes_is_the_validated_one():
    compiled = _command_graph(
        lambda s: Command(update={"a": "5"}, goto="landing"), writes={"a"}
    )
    result = compiled.invoke({})
    assert result["a"] == 5 and isinstance(result["a"], int)


def test_a_command_carrying_a_non_dict_update_is_rejected():
    compiled = _command_graph(
        lambda s: Command(update=[("a", 1)], goto="landing"), writes={"a"}
    )
    with pytest.raises(WritePermissionError, match="requires a dict"):
        compiled.invoke({})


def test_a_command_aimed_at_a_parent_graph_is_rejected():
    """GraphARC cannot check another graph's schema, so it refuses rather than
    let an unchecked update through."""
    compiled = _command_graph(
        lambda s: Command(graph=Command.PARENT, update={"a": 1}), writes={"a"}
    )
    with pytest.raises(WritePermissionError, match="current graph only"):
        compiled.invoke({})


def test_a_command_with_no_update_still_routes():
    compiled = _command_graph(lambda s: Command(goto="landing"), writes={"a"})
    assert compiled.invoke({"a": 2})["b"] == 20


def test_a_command_traces_the_update_it_carried(trace):
    compiled = _command_graph(
        lambda s: Command(update={"a": 4}, goto="landing"), writes={"a"}, trace=trace
    )
    compiled.invoke({}, run_id="r-cmd-trace")
    ends = [e for e in trace.read_events("r-cmd-trace") if e.phase == "end"]
    assert ends[0].node == "router" and ends[0].state_delta == {"a": 4}


@pytest.mark.asyncio
async def test_an_async_node_may_return_a_command_under_the_same_rules():
    async def route(state: S) -> Command:
        await asyncio.sleep(0)
        return Command(update={"a": 3}, goto="landing")

    g = GraphARC(S, name="acmd")
    g.add_node("router", route, writes={"a"})
    g.add_node("landing", lambda s: {"b": s.a * 10}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    result = await g.compile().ainvoke({})
    assert (result["a"], result["b"]) == (3, 30)


@pytest.mark.asyncio
async def test_an_async_command_with_an_undeclared_write_raises():
    async def route(state: S) -> Command:
        await asyncio.sleep(0)
        return Command(update={"b": 9}, goto="landing")

    g = GraphARC(S, name="acmd")
    g.add_node("router", route, writes={"a"})
    g.add_node("landing", lambda s: {"b": 1}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    with pytest.raises(WritePermissionError, match="undeclared"):
        await g.compile().ainvoke({})


def test_a_command_can_route_a_loop_to_end():
    def step(state: S) -> Command:
        if state.a >= 3:
            return Command(update={"note": "done"}, goto=END)
        return Command(update={"a": state.a + 1}, goto="step")

    g = GraphARC(S, name="cmdloop", budget=Budget(max_iterations=10))
    g.add_node("step", step, writes={"a", "note"})
    g.add_edge(START, "step")
    result = g.compile().invoke({})
    assert result["a"] == 3 and result["note"] == "done"


def test_a_command_loop_is_still_bounded_by_the_budget():
    g = GraphARC(S, name="cmdloop", budget=Budget(max_iterations=4))
    g.add_node("step", lambda s: Command(update={"a": s.a + 1}, goto="step"), writes={"a"})
    g.add_edge(START, "step")
    with pytest.raises(BudgetExceeded, match="max_iterations"):
        g.compile().invoke({})


# -- §1.2 the goto half of a Command: a routing decision must not be swallowed --
#
# LangGraph resolves `goto` after the node has returned and drops what it does
# not recognise — a log line, then the run carries on as though the node had
# never routed. That is the same defect as an unpermitted write wearing a
# different hat, so these assert the destination is checked at the boundary.


def _routing_graph(goto, *, name="goto", trace=None):
    """router routes to `goto`; `landing` running is visible as a non-zero `b`."""
    g = GraphARC(S, name=name, trace=trace)
    g.add_node("router", lambda s: Command(update={"a": 1}, goto=goto), writes={"a"})
    g.add_node("landing", lambda s: {"b": s.a * 10}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    return g.compile()


def test_a_goto_naming_a_node_the_graph_does_not_have_raises_instead_of_being_swallowed():
    """The P0: plain LangGraph logs "wrote to unknown channel branch:to:ghost" at
    warning level and finishes the run with `{'a': 1}`, so the handoff simply
    never happened and nobody was told."""
    with pytest.raises(GraphRoutingError) as excinfo:
        _routing_graph("ghost").invoke({})
    message = str(excinfo.value)
    assert "'ghost'" in message
    assert "node 'router'" in message  # who asked for it
    assert "'landing'" in message and "END" in message  # where it could have gone


def test_a_swallowed_goto_is_an_error_the_trace_records(trace):
    with pytest.raises(GraphRoutingError):
        _routing_graph("ghost", trace=trace).invoke({}, run_id="r-goto")
    errors = [e for e in trace.read_events("r-goto") if e.phase == "error"]
    assert errors and "ghost" in errors[0].error
    # And the run stopped there: `landing` never opened a trace event.
    assert {e.node for e in trace.read_events("r-goto")} == {"topology", "router"}


def test_a_send_naming_a_node_the_graph_does_not_have_raises():
    """`Send` is the other fail-open shape: LangGraph logs "Ignoring unknown node
    name ghost in pending sends" and the shard is silently never worked."""
    with pytest.raises(GraphRoutingError, match="Send"):
        _routing_graph(Send("ghost", S(a=1))).invoke({})


def test_a_send_aimed_at_end_raises_rather_than_vanishing():
    """END is a legal `goto` but not a legal `Send` target — a Send has to name a
    node that runs, and LangGraph drops `Send(END, ...)` without a word."""
    with pytest.raises(GraphRoutingError, match="END is not one"):
        _routing_graph(Send(END, S())).invoke({})


def test_one_bad_destination_in_a_list_of_good_ones_still_raises():
    with pytest.raises(GraphRoutingError, match="'ghost'"):
        _routing_graph(["landing", "ghost"]).invoke({})


def test_a_goto_that_is_neither_a_node_name_nor_a_send_raises():
    """LangGraph's control branch skips an element it cannot classify with no log
    line at all — the quietest of the three swallows."""
    with pytest.raises(GraphRoutingError, match="a list"):
        _routing_graph([["landing"]]).invoke({})


def test_start_is_not_a_goto_destination():
    """There is no `branch:to:__start__` channel, so routing to START is a no-op
    LangGraph never mentions."""
    with pytest.raises(GraphRoutingError, match="__start__"):
        _routing_graph(START).invoke({})


def test_goto_none_is_refused_with_a_message_that_says_what_to_do_instead():
    with pytest.raises(GraphRoutingError, match="Omit goto"):
        _routing_graph(None).invoke({})


@pytest.mark.parametrize(
    ("shape", "goto", "lands"),
    [
        ("a node name", "landing", True),
        ("END", END, False),
        ("a list of names", ["landing"], True),
        ("a tuple mixing a name and END", ("landing", END), True),
        ("a set of names", {"landing"}, True),
        ("a Send", Send("landing", S(a=9)), True),
        ("a list mixing a Send and END", [Send("landing", S(a=9)), END], True),
        ("the default empty tuple", (), False),
        ("an empty list", [], False),
        ("a list holding only END", [END], False),
    ],
)
def test_every_legitimate_goto_shape_still_routes(shape, goto, lands):
    """Refusing something LangGraph accepts would be its own bug, so every shape
    `Command.goto` documents is exercised end to end."""
    result = _routing_graph(goto).invoke({})
    assert result["a"] == 1, shape
    # `landing` never running leaves `b` unwritten, so it is absent entirely.
    assert (result.get("b", 0) != 0) is lands, shape


def test_a_one_shot_iterable_goto_is_not_consumed_by_the_check():
    """The validator materialises what it iterated and hands that back; inspecting
    a generator and passing the drained original would turn a legal route into
    exactly the silent no-op this check exists to prevent."""
    result = _routing_graph(node for node in ["landing"]).invoke({})
    assert result["b"] == 10


def test_a_fanout_dispatcher_naming_an_unknown_worker_raises():
    """Same swallow, other producer of Sends: a shard dispatched at a node that
    does not exist is dropped and the work is simply never done."""
    g = GraphARC(S, name="fan")
    g.add_node("plan", lambda s: {"a": 1}, writes={"a"})
    g.add_node("worker", lambda s: {"b": 1}, writes={"b"})
    g.add_edge(START, "plan")
    g.add_fanout_edge("plan", lambda s: [("ghost", S(a=1))])
    g.add_edge("worker", END)
    with pytest.raises(GraphRoutingError, match="fan-out dispatcher on node 'plan'"):
        g.compile().invoke({})


def test_a_fanout_dispatcher_naming_a_real_worker_is_left_alone():
    g = GraphARC(S, name="fan")
    g.add_node("plan", lambda s: {"a": 1}, writes={"a"})
    g.add_node("worker", lambda s: {"b": s.a * 3}, writes={"b"})
    g.add_edge(START, "plan")
    g.add_fanout_edge("plan", lambda s: [("worker", S(a=4))])
    g.add_edge("worker", END)
    assert g.compile().invoke({})["b"] == 12


@pytest.mark.asyncio
async def test_an_async_command_with_an_unknown_goto_raises_too():
    async def route(state: S) -> Command:
        await asyncio.sleep(0)
        return Command(update={"a": 1}, goto="ghost")

    g = GraphARC(S, name="agoto")
    g.add_node("router", route, writes={"a"})
    g.add_node("landing", lambda s: {"b": 1}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    with pytest.raises(GraphRoutingError, match="'ghost'"):
        await g.compile().ainvoke({})


@pytest.mark.asyncio
async def test_an_async_command_with_a_good_goto_still_routes():
    async def route(state: S) -> Command:
        await asyncio.sleep(0)
        return Command(update={"a": 2}, goto=["landing"])

    g = GraphARC(S, name="agoto")
    g.add_node("router", route, writes={"a"})
    g.add_node("landing", lambda s: {"b": s.a * 10}, writes={"b"})
    g.add_edge(START, "router")
    g.add_edge("landing", END)
    assert (await g.compile().ainvoke({}))["b"] == 20


# -- §1.3 state access without reaching into .inner ------------------------


def _checkpointed(saver):
    g = GraphARC(S, name="hitl")
    g.add_node("n", lambda s: {"a": s.a + 1}, writes={"a"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile(checkpointer=saver)


def test_get_state_round_trips_the_thread_that_ran():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    snapshot = compiled.get_state("t1")
    assert snapshot.values["a"] == 2
    assert snapshot.next == ()


def test_get_state_reads_only_the_thread_it_was_asked_for():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    compiled.invoke({"a": 100}, thread_id="t2")
    assert compiled.get_state("t1").values["a"] == 2
    assert compiled.get_state("t2").values["a"] == 101


def test_update_state_round_trips_through_get_state():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    compiled.update_state("t1", {"a": 99}, as_node="n")
    assert compiled.get_state("t1").values["a"] == 99


def test_update_state_enforces_the_declared_writes_of_the_node_it_claims_to_be():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    with pytest.raises(WritePermissionError, match="undeclared"):
        compiled.update_state("t1", {"b": 5}, as_node="n")
    # Absent, not defaulted: a rejected write never reached the checkpoint.
    assert "b" not in compiled.get_state("t1").values


def test_update_state_type_checks_even_without_a_node_to_blame():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    with pytest.raises(StateTypeError, match="expected int"):
        compiled.update_state("t1", {"a": "nope"})


def test_update_state_rejects_a_field_the_schema_does_not_have():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    with pytest.raises(WritePermissionError, match="unknown state fields"):
        compiled.update_state("t1", {"nope": 1})


def test_get_state_history_lists_the_threads_checkpoints():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    history = list(compiled.get_state_history("t1"))
    assert len(history) >= 2
    assert history[0].values["a"] == 2
    assert all(h.config["configurable"]["thread_id"] == "t1" for h in history)


def test_get_state_history_honours_a_limit():
    compiled = _checkpointed(InMemorySaver())
    compiled.invoke({"a": 1}, thread_id="t1")
    assert len(list(compiled.get_state_history("t1", limit=1))) == 1


@pytest.mark.asyncio
async def test_async_state_access_round_trips_too():
    compiled = _checkpointed(InMemorySaver())
    await compiled.ainvoke({"a": 1}, thread_id="t1")
    assert (await compiled.aget_state("t1")).values["a"] == 2
    await compiled.aupdate_state("t1", {"a": 50}, as_node="n")
    assert (await compiled.aget_state("t1")).values["a"] == 50
    history = [s async for s in compiled.aget_state_history("t1")]
    assert len(history) >= 2


@pytest.mark.asyncio
async def test_aupdate_state_enforces_declared_writes():
    compiled = _checkpointed(InMemorySaver())
    await compiled.ainvoke({"a": 1}, thread_id="t1")
    with pytest.raises(WritePermissionError, match="undeclared"):
        await compiled.aupdate_state("t1", {"b": 5}, as_node="n")


# -- sync/async parity: neither wrapper may lose the ending -----------------


# Driven with `asyncio.run` rather than `@pytest.mark.asyncio`: asyncio re-raises
# KeyboardInterrupt and SystemExit out of the task step and into the loop, so they
# leave the run at the runner rather than at the await. Both wrappers are asserted
# in one test so neither can quietly stop matching the other.
@pytest.mark.parametrize("stopper", [KeyboardInterrupt, SystemExit])
def test_both_wrappers_record_a_node_stopped_by_a_baseexception(trace, stopper):
    """Ctrl-C is the commonest way a human ends a long run, and a
    KeyboardInterrupt is not an Exception. The sync wrapper used to let it past
    without a trace line, so the audit trail ended mid-node with no reason and
    `metrics.summarize` reported zero errors for the run."""

    def boom(state: S) -> dict:
        raise stopper("operator hit ^C inside a node")

    async def aboom(state: S) -> dict:
        await asyncio.sleep(0)
        raise stopper("operator hit ^C inside a node")

    with pytest.raises(stopper):
        _graph(boom, writes={"a"}, trace=trace).invoke({}, run_id="r-stop")
    with pytest.raises(stopper):
        asyncio.run(_graph(aboom, writes={"a"}, trace=trace).ainvoke({}, run_id="r-astop"))

    for run_id in ("r-stop", "r-astop"):
        errors = [e for e in trace.read_events(run_id) if e.phase == "error"]
        assert errors, f"{run_id} recorded no terminal error event"
        assert stopper.__name__ in errors[0].error


@pytest.mark.asyncio
async def test_the_async_entry_points_refuse_an_unknown_input_key_too():
    """`invoke`/`stream` fail loudly on a typo'd input key; so must their twins."""
    compiled = _graph(lambda s: {"a": 1}, writes={"a"})
    with pytest.raises(WritePermissionError) as caught:
        await compiled.ainvoke({"aa": 1})
    assert str(caught.value) == "ainvoke() targets unknown state fields: ['aa']"
    with pytest.raises(WritePermissionError) as caught:
        [chunk async for chunk in compiled.astream({"aa": 1})]
    assert str(caught.value) == "astream() targets unknown state fields: ['aa']"
