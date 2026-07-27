"""Budgets that actually bind.

Two limits used to be advisory. `max_tokens` only counted what a node author
remembered to charge by hand — a live capstone run reported zero spend while
costing money. `max_seconds` was checked *between* nodes, so a 1.5s node ran to
completion under a 0.2s budget.

Both now bite *inside* the running node: tokens at `on_llm_end`, time by an
interrupt that keeps coming back until the node lets go. And the meter tells a
re-report of a metered call from a real charge by identity — matching token
counts swallowed charges that were real, which is the direction that costs
money.
"""

import operator
import threading
import time
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from grapharc.runtime.budget import (
    Budget,
    BudgetExceeded,
    BudgetMeter,
    NodeDeadlineExceeded,
    deadline_guard,
)
from grapharc.runtime.graph import END, START, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.runtime.verify import verify_claim
from grapharc.testing import ScriptedChatModel, charge_usage


class S(GraphARCState):
    ok: bool = False
    reply: str = ""


def _single_node_graph(fn, *, writes, budget=None, trace=None, name="t"):
    g = GraphARC(S, name=name, budget=budget, trace=trace)
    g.add_node("n", fn, writes=writes)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


# -- automatic token charging (roadmap 0.4) --------------------------------


def test_a_model_call_inside_a_node_charges_the_run_without_being_asked():
    def node(state):
        model = ScriptedChatModel(responses=["a scripted reply of some length"])
        return {"reply": str(model.invoke([HumanMessage(content="hello")]).content)}

    compiled = _single_node_graph(node, writes={"reply"})
    compiled.invoke({})
    assert compiled.last_run.meter.tokens > 0


def test_charging_reaches_calls_the_node_does_not_make_itself():
    """The live failure was a node that never touched a model directly: it called
    `verify_claim`, which called the reviewer. Depth in the call stack must not
    matter."""
    source = "Budgets place hard ceilings on iterations and tokens."

    def node(state):
        reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "quoted"}'])
        verdict = verify_claim(
            reviewer,
            text="budgets cap tokens",
            citation="hard ceilings",
            source_text=source,
        )
        return {"ok": verdict.accepted}

    compiled = _single_node_graph(node, writes={"ok"})
    assert compiled.invoke({})["ok"] is True
    assert compiled.last_run.meter.tokens > 0


def test_max_tokens_stops_the_run_at_the_node_that_overspent():
    def node(state):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"}, budget=Budget(max_tokens=5))
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        compiled.invoke({})


def test_max_tokens_stops_the_node_at_the_call_that_crossed_the_line():
    """The node boundary is not the first point enforcement is possible, and
    waiting for it is expensive: fifty scripted calls under a ten-token ceiling
    used to be paid for in full. `on_llm_end` is strictly earlier — the same
    shape as the wall-clock interrupt — so overspend is bounded by one call."""
    after_each_call: list[int] = []

    def node(state, ctx: RunContext):
        model = ScriptedChatModel(responses=["a scripted reply"], on_exhausted="repeat")
        for _ in range(50):
            model.invoke("hello")
            after_each_call.append(ctx.meter.tokens)
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"}, budget=Budget(max_tokens=10))
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        compiled.invoke({})

    one_call = after_each_call[0]
    spent = compiled.last_run.meter.tokens
    assert len(after_each_call) < 50, "the node made every call it wanted to"
    assert 10 <= spent <= 10 + one_call


def test_the_ceiling_holds_when_the_spend_is_split_across_nodes():
    def spend(state):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        return None

    budget = Budget(max_tokens=20)
    g = GraphARC(S, name="t", budget=budget)
    g.add_node("a", spend, writes=set())
    g.add_node("b", spend, writes=set())
    g.add_node("c", spend, writes=set())
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    compiled = g.compile()
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        compiled.invoke({})
    assert compiled.last_run.meter.tokens >= 20


def test_hand_metering_with_charge_usage_stays_legal_but_stops_double_counting():
    """Every shipped example still calls `charge_usage`. It must become a no-op,
    not a second charge for the same call."""

    def auto_only(state):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        return {"ok": True}

    def auto_plus_manual(state, ctx: RunContext):
        message = ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        charge_usage(ctx, message)
        return {"ok": True}

    plain = _single_node_graph(auto_only, writes={"ok"})
    plain.invoke({})
    legacy = _single_node_graph(auto_plus_manual, writes={"ok"})
    legacy.invoke({})

    assert plain.last_run.meter.tokens > 0
    assert legacy.last_run.meter.tokens == plain.last_run.meter.tokens


def test_a_manual_charge_the_callback_never_saw_still_counts():
    """Spend the runtime cannot see — a provider billed outside LangChain, say —
    must still land."""

    def node(state, ctx: RunContext):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        auto = ctx.meter.tokens
        ctx.meter.charge_tokens(auto + 1000)
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"})
    compiled.invoke({})
    meter = compiled.last_run.meter
    assert meter.tokens > 1000


def test_an_unseen_charge_counts_even_when_it_equals_a_metered_call_exactly():
    """The collision the old ledger lost money on. It matched charges by token
    count, so a real charge for a provider the callback cannot see vanished as
    soon as some earlier model call happened to cost the same."""

    sizes: dict[str, int] = {}

    def node(state, ctx: RunContext):
        model = ScriptedChatModel(responses=["reply", "reply"])
        model.invoke("hello")
        sizes["call"] = ctx.meter.tokens
        model.invoke("hello")  # a second call of exactly the same size
        assert ctx.meter.tokens == 2 * sizes["call"]
        ctx.meter.charge_tokens(sizes["call"])  # billed by a non-LangChain provider
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"})
    compiled.invoke({})
    # Three charges of the same size: two metered, one not. All three are real.
    assert sizes["call"] > 0
    assert compiled.last_run.meter.tokens == 3 * sizes["call"]


def test_charging_does_not_follow_a_thread_the_node_starts_itself():
    """A documented gap, asserted so it cannot rot into a silent surprise:
    `threading.Thread` does not inherit context variables, so a model called on a
    node-spawned thread is invisible to the meter."""

    def node(state):
        def job():
            ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")

        worker = threading.Thread(target=job)
        worker.start()
        worker.join()
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"})
    compiled.invoke({})
    assert compiled.last_run.meter.tokens == 0


def test_the_trace_reports_what_the_node_actually_spent(trace):
    def quiet(state):
        return None

    def spender(state):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        return {"ok": True}

    g = GraphARC(S, name="spend", trace=trace)
    g.add_node("quiet", quiet, writes=set())
    g.add_node("spender", spender, writes={"ok"})
    g.add_edge(START, "quiet")
    g.add_edge("quiet", "spender")
    g.add_edge("spender", END)
    g.compile().invoke({}, run_id="r1")

    ends = {e.node: e for e in trace.read_events("r1") if e.phase == "end"}
    assert ends["quiet"].tokens == 0
    assert ends["spender"].tokens > 0


def test_streaming_charges_the_same_as_invoke():
    def node(state):
        ScriptedChatModel(responses=["a scripted reply of some length"]).invoke("hello")
        return {"ok": True}

    compiled = _single_node_graph(node, writes={"ok"})
    list(compiled.stream({}))
    assert compiled.last_run.meter.tokens > 0


def _call(run: str) -> AIMessage:
    """A stand-in for one metered model call, identified the way langchain does."""
    return AIMessage(content="x", id=f"lc_run--{run}-0")


def test_a_re_report_that_names_the_call_is_free_exactly_once():
    call = _call("a")
    meter = BudgetMeter(Budget())
    with meter.automatic_scope():
        meter.charge_tokens(30, automatic=True, source=call)
        meter.charge_tokens(30, source=call)  # the node re-reporting that call
        meter.charge_tokens(30, source=call)  # nothing left to re-report
    assert meter.tokens == 60


def test_two_calls_of_the_same_size_are_two_calls():
    """The whole point of keying on identity: matching sizes proves nothing."""
    first, second = _call("a"), _call("b")
    meter = BudgetMeter(Budget())
    with meter.automatic_scope():
        meter.charge_tokens(30, automatic=True, source=first)
        meter.charge_tokens(30, automatic=True, source=second)
        meter.charge_tokens(30, source=first)  # only the first call is re-reported
    assert meter.tokens == 60


def test_a_charge_naming_a_call_nobody_metered_counts():
    meter = BudgetMeter(Budget())
    with meter.automatic_scope():
        meter.charge_tokens(30, automatic=True, source=_call("a"))
        meter.charge_tokens(30, source=_call("elsewhere"))
    assert meter.tokens == 60


def test_a_copy_of_the_message_still_names_the_same_call():
    """Identity is the LLM run, not the object address: a node that re-reports a
    copy of the message — one that crossed a serialisation boundary, say — is
    still re-reporting the call the callback metered."""
    call = _call("a")
    meter = BudgetMeter(Budget())
    with meter.automatic_scope():
        meter.charge_tokens(30, automatic=True, source=call)
        meter.charge_tokens(30, source=call.model_copy(deep=True))
    assert meter.tokens == 30


def test_automatic_scopes_do_not_leak_between_nodes():
    call = _call("a")
    meter = BudgetMeter(Budget())
    with meter.automatic_scope():
        meter.charge_tokens(30, automatic=True, source=call)
    with meter.automatic_scope():
        meter.charge_tokens(30, source=call)  # another node cannot claim it
    assert meter.tokens == 60


# -- max_seconds as a real ceiling (roadmap 0.3) ---------------------------


def test_max_seconds_interrupts_a_node_blocked_in_a_syscall():
    def slow(state):
        time.sleep(5.0)
        return {"ok": True}

    compiled = _single_node_graph(slow, writes={"ok"}, budget=Budget(max_seconds=0.2))
    started = time.perf_counter()
    with pytest.raises(NodeDeadlineExceeded, match="max_seconds"):
        compiled.invoke({})
    assert time.perf_counter() - started < 2.0


def test_the_error_names_the_node_that_ran_long():
    def slow(state):
        time.sleep(5.0)
        return {"ok": True}

    g = GraphARC(S, name="t", budget=Budget(max_seconds=0.2))
    g.add_node("dawdler", slow, writes={"ok"})
    g.add_edge(START, "dawdler")
    g.add_edge("dawdler", END)
    with pytest.raises(NodeDeadlineExceeded, match="node 'dawdler'"):
        g.compile().invoke({})


def test_an_interrupted_node_writes_nothing_even_if_it_swallows_the_interrupt(trace):
    """A node with a broad `except` must not be able to convert a blown deadline
    into a successful write."""

    def stubborn(state):
        try:
            time.sleep(5.0)
        except BaseException:  # noqa: BLE001 — swallowing is the point of the test
            pass
        return {"ok": True}

    compiled = _single_node_graph(
        stubborn, writes={"ok"}, budget=Budget(max_seconds=0.2), trace=trace
    )
    with pytest.raises(NodeDeadlineExceeded):
        compiled.invoke({}, run_id="r1")
    events = trace.read_events("r1")
    assert not [e for e in events if e.phase == "end"]
    assert [e for e in events if e.phase == "error"]


def test_max_seconds_interrupts_a_fanout_worker_on_a_pool_thread():
    """Fan-out workers run off the main thread, where signals are unavailable —
    the async-exception fallback has to carry the ceiling there."""

    class Shard(BaseModel):
        index: int

    class FanState(GraphARCState):
        done: Annotated[list[int], operator.add] = []

    def plan(state):
        return None

    def work(payload: Shard):
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            pass
        return {"done": [payload.index]}

    g = GraphARC(FanState, name="fan", budget=Budget(max_seconds=0.3))
    g.add_node("plan", plan, writes=set())
    g.add_node("work", work, writes={"done"}, input_schema=Shard)
    g.add_edge(START, "plan")
    g.add_fanout_edge("plan", lambda s: [("work", Shard(index=i)) for i in range(3)])
    g.add_edge("work", END)

    started = time.perf_counter()
    with pytest.raises(NodeDeadlineExceeded):
        g.compile().invoke({"done": []})
    assert time.perf_counter() - started < 3.0


def _swallow_interrupts_for(seconds: float, rounds: int = 4) -> tuple[list[float], float]:
    """Run a node that catches every interrupt, and report what it took to stop it.

    One interrupt used to be all there was: the node caught it and carried on
    with nothing left to fire, so a loop that never returns hung the run.
    """
    meter = BudgetMeter(Budget(max_seconds=0.2))
    caught: list[float] = []
    started = time.perf_counter()
    with pytest.raises(NodeDeadlineExceeded):
        with deadline_guard(meter, what="stubborn"):
            for _ in range(rounds):
                try:
                    while time.perf_counter() - started < seconds:
                        pass
                except BaseException:  # noqa: BLE001 — swallowing is the point
                    caught.append(time.perf_counter() - started)
    return caught, time.perf_counter() - started


def test_the_interrupt_is_re_armed_after_a_node_swallows_it():
    caught, ran_for = _swallow_interrupts_for(seconds=5.0)
    assert len(caught) > 1, "the deadline fired once and then gave up"
    assert ran_for < 2.0


def test_the_interrupt_is_re_armed_on_a_worker_thread_too():
    """Off the main thread there is no SIGALRM to repeat, so the async-exception
    fallback has to re-arm itself — and that is the mechanism the whole run uses
    whenever `invoke()` is called from a thread, not just for fan-out."""
    result: dict[str, object] = {}

    def body():
        result["outcome"] = _swallow_interrupts_for(seconds=5.0)

    worker = threading.Thread(target=body)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive()
    caught, ran_for = result["outcome"]  # type: ignore[misc]
    assert len(caught) > 1
    assert ran_for < 2.0


def test_a_node_that_finishes_in_time_is_left_alone():
    def brisk(state):
        time.sleep(0.05)
        return {"ok": True}

    compiled = _single_node_graph(brisk, writes={"ok"}, budget=Budget(max_seconds=5.0))
    assert compiled.invoke({})["ok"] is True


def test_no_interrupt_survives_the_node_that_earned_it():
    """The watchdog queues an exception in another thread. If a queued one could
    outlive its guard it would surface inside whatever ran next on that pooled
    thread — an unattributable crash. Nothing may be left pending."""
    results = {}

    def body():
        meter = BudgetMeter(Budget(max_seconds=0.1))
        with pytest.raises(NodeDeadlineExceeded):
            with deadline_guard(meter, what="worker"):
                time.sleep(0.6)
        total = 0
        for i in range(500_000):
            total += i
        results["after"] = total

    worker = threading.Thread(target=body)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert results["after"] == sum(range(500_000))


def test_deadline_guard_is_inert_without_max_seconds():
    meter = BudgetMeter(Budget(max_iterations=3))
    with deadline_guard(meter, what="node 'n'"):
        time.sleep(0.05)
    assert meter.remaining_seconds() is None


def test_deadline_guard_refuses_to_start_work_past_the_deadline():
    meter = BudgetMeter(Budget(max_seconds=0.01))
    time.sleep(0.05)
    with pytest.raises(NodeDeadlineExceeded, match="max_seconds"):
        with deadline_guard(meter, what="node 'n'"):
            pytest.fail("the guard let a node start after the budget was spent")


def test_a_deadline_exceeded_is_a_budget_exceeded():
    """Callers that already catch BudgetExceeded must keep catching timeouts."""
    assert issubclass(NodeDeadlineExceeded, BudgetExceeded)
