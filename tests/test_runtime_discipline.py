"""Unit tests for the runtime discipline layer: write permissions, DAG mode, budgets, traces."""

import operator
from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import (
    END,
    START,
    GraphARC,
    GraphCycleError,
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


def test_undeclared_write_raises(trace):
    g = GraphARC(S, name="t", trace=trace)
    g.add_node("n", lambda s: {"b": 1}, writes={"a"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    with pytest.raises(WritePermissionError, match="undeclared"):
        g.compile().invoke({"a": 0})
    errors = [e for e in trace.read_events() if e.phase == "error"]
    assert errors and "undeclared" in errors[0].error


def test_declaring_unknown_field_raises():
    g = GraphARC(S, name="t")
    with pytest.raises(WritePermissionError, match="unknown state fields"):
        g.add_node("n", lambda s: None, writes={"nope"})


def test_non_dict_return_raises():
    g = GraphARC(S, name="t")
    g.add_node("n", lambda s: [1], writes={"a"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    with pytest.raises(WritePermissionError, match="dict update or None"):
        g.compile().invoke({"a": 0})


def test_dag_mode_rejects_cycles():
    g = GraphARC(S, name="t", dag=True)
    g.add_node("x", lambda s: None, writes=set())
    g.add_node("y", lambda s: None, writes=set())
    g.add_edge(START, "x")
    g.add_edge("x", "y")
    g.add_edge("y", "x")
    with pytest.raises(GraphCycleError, match="cycle"):
        g.compile()


def test_dag_mode_rejects_conditional_edges():
    g = GraphARC(S, name="t", dag=True)
    g.add_node("x", lambda s: None, writes=set())
    with pytest.raises(GraphCycleError, match="conditional"):
        g.add_conditional_edge("x", lambda s: "x", {"x": "x"})


# -- conditional edges are checked where they are declared ---------------------


def _spinner(name: str = "t") -> GraphARC:
    g = GraphARC(S, name=name)
    g.add_node("spin", lambda s: {"a": s.a + 1}, writes={"a"})
    g.add_edge(START, "spin")
    return g


def test_a_mapping_target_nobody_added_raises_when_the_edge_is_added():
    """The whole point: this used to be a KeyError three nodes into a run."""
    g = _spinner()
    with pytest.raises(GraphRoutingError) as exc:
        g.add_conditional_edge("spin", lambda s: "go", {"go": "sipn"})
    message = str(exc.value)
    assert "'spin'" in message, "the error has to name the edge's source"
    assert "'sipn'" in message, "and the target it cannot reach"
    assert "'go' -> 'sipn'" in message


def test_every_unreachable_target_is_named_at_once():
    g = _spinner()
    with pytest.raises(GraphRoutingError, match="destinations") as exc:
        g.add_conditional_edge("spin", lambda s: "go", {"go": "nope", "stop": "also_nope"})
    assert "'go' -> 'nope'" in str(exc.value)
    assert "'stop' -> 'also_nope'" in str(exc.value)


def test_an_empty_mapping_is_refused():
    g = _spinner()
    with pytest.raises(GraphRoutingError, match="empty mapping"):
        g.add_conditional_edge("spin", lambda s: "go", {})


def test_a_correct_mapping_is_unaffected():
    g = _spinner()
    g.add_conditional_edge(
        "spin", lambda s: "stop" if s.a >= 3 else "again", {"again": "spin", "stop": END}
    )
    assert g.compile().invoke({"a": 0})["a"] == 3


def test_a_router_declaring_a_literal_has_its_members_checked():
    def route(s) -> Literal["again", "stop"]:
        return "again"

    g = _spinner()
    with pytest.raises(GraphRoutingError, match="declaring it returns") as exc:
        g.add_conditional_edge("spin", route, {"again": "spin"})
    assert "'stop' is not a key" in str(exc.value)


def test_a_router_declaring_an_enum_has_its_members_checked():
    class Stop(StrEnum):
        again = "again"
        stop = "stop"

    def route(s) -> Stop:
        return Stop.again

    g = _spinner()
    with pytest.raises(GraphRoutingError, match="declaring it returns"):
        g.add_conditional_edge("spin", route, {Stop.again: "spin"})

    ok = _spinner()
    ok.add_conditional_edge("spin", route, {Stop.again: "spin", Stop.stop: END})


def test_a_router_declaring_a_literal_that_matches_is_accepted():
    def route(s) -> Literal["again", "stop"]:
        return "stop" if s.a >= 2 else "again"

    g = _spinner()
    g.add_conditional_edge("spin", route, {"again": "spin", "stop": END})
    assert g.compile().invoke({"a": 0})["a"] == 2


def test_a_router_that_declares_nothing_is_left_alone():
    """A `str` return says nothing about the mapping; guessing is not checking."""

    def route(s) -> str:
        return "stop" if s.a >= 1 else "again"

    g = _spinner()
    g.add_conditional_edge("spin", route, {"again": "spin", "stop": END})
    assert g.compile().invoke({"a": 0})["a"] == 1


def test_an_unmapped_router_return_is_a_grapharc_error_not_a_keyerror():
    """What cannot be settled at declaration time still must not be a bare KeyError."""
    g = _spinner()
    g.add_conditional_edge("spin", lambda s: "stpo", {"again": "spin", "stop": END})
    with pytest.raises(GraphRoutingError) as exc:
        g.compile().invoke({"a": 0})
    message = str(exc.value)
    assert "'stpo'" in message
    assert "'spin'" in message, "the router that produced it has to be named"
    assert "'again', 'stop'" in message, "and the keys there were"


def test_an_unmapped_key_inside_a_returned_list_is_caught_too():
    g = _spinner()
    g.add_node("other", lambda s: {"b": 1}, writes={"b"})
    g.add_edge("other", END)
    g.add_conditional_edge("spin", lambda s: ["stop", "elsewhere"], {"stop": "other"})
    with pytest.raises(GraphRoutingError, match="elsewhere"):
        g.compile().invoke({"a": 0})


def test_the_wrapped_router_keeps_the_name_langgraph_branches_by():
    """`functools.wraps` is load-bearing: LangGraph names the branch after it."""

    def pick_a_branch(s) -> str:
        return "stop"

    g = _spinner()
    g.add_conditional_edge("spin", pick_a_branch, {"stop": END})
    assert "pick_a_branch" in g._graph.branches["spin"]


def test_budget_hard_ceiling_cannot_be_looped_past():
    g = GraphARC(S, name="t", budget=Budget(max_iterations=5))
    g.add_node("loop", lambda s: {"a": s.a + 1}, writes={"a"})
    g.add_edge(START, "loop")
    g.add_conditional_edge("loop", lambda s: "again", {"again": "loop", "done": END})
    with pytest.raises(BudgetExceeded, match="max_iterations"):
        g.compile().invoke({"a": 0})


def test_budget_meter_token_ceiling():
    meter = BudgetMeter(Budget(max_tokens=10))
    meter.charge_tokens(9)
    meter.check()
    meter.charge_tokens(1)
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        meter.check()


def test_trace_records_state_delta_and_steps(trace):
    g = GraphARC(S, name="traced", trace=trace)
    g.add_node("n1", lambda s: {"a": 1}, writes={"a"})
    g.add_node("n2", lambda s: {"b": 2}, writes={"b"})
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", END)
    compiled = g.compile()
    compiled.invoke({"a": 0}, run_id="run1")
    events = trace.read_events("run1")
    ends = [e for e in events if e.phase == "end"]
    assert [e.node for e in ends] == ["n1", "n2"]
    assert ends[0].state_delta == {"a": 1}
    assert ends[1].state_delta == {"b": 2}
    assert all(e.duration_ms is not None for e in ends)


def test_raw_langgraph_entry_points_fail_closed():
    """Driving the compiled graph via .inner bypasses budgets/traces — so it must raise."""
    g = GraphARC(S, name="t", budget=Budget(max_iterations=3))
    g.add_node("n", lambda s: {"a": s.a + 1}, writes={"a"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    compiled = g.compile()
    with pytest.raises(MissingRunContextError, match="bypass budgets"):
        list(compiled.inner.stream({"a": 0}))


def test_stream_goes_through_the_disciplined_path():
    g = GraphARC(S, name="t", budget=Budget(max_iterations=5))
    g.add_node("loop", lambda s: {"a": s.a + 1}, writes={"a"})
    g.add_edge(START, "loop")
    g.add_conditional_edge("loop", lambda s: "again", {"again": "loop", "done": END})
    with pytest.raises(BudgetExceeded, match="max_iterations"):
        list(g.compile().stream({"a": 0}))


class Inner(BaseModel):
    text: str


class NestedState(GraphARCState):
    items: list[Inner] = []
    downstream_saw: str = ""


def test_in_place_mutation_of_nested_models_is_discarded():
    """The returned dict is the only write channel: mutating state in place
    (even nested models, which Pydantic passes by reference) has no effect."""

    def tamper(state: NestedState) -> None:
        state.items[0].text = "TAMPERED"
        return None

    def observe(state: NestedState) -> dict:
        return {"downstream_saw": state.items[0].text}

    g = GraphARC(NestedState, name="t")
    g.add_node("tamper", tamper, writes=set())
    g.add_node("observe", observe, writes={"downstream_saw"})
    g.add_edge(START, "tamper")
    g.add_edge("tamper", "observe")
    g.add_edge("observe", END)
    result = g.compile().invoke({"items": [Inner(text="original")]})
    assert result["downstream_saw"] == "original"
    assert result["items"][0].text == "original"


# -- fan-out payloads: the same isolation, and the declared input contract ---


class FanState(GraphARCState):
    ran: Annotated[list[str], operator.add] = []


class Shard(BaseModel):
    who: str = ""
    hits: list[int] = []


def _fanout(payloads, *, worker, input_schema=None):
    g = GraphARC(FanState, name="fan")
    g.add_node("plan", lambda s: None, writes=set())
    g.add_node("worker", worker, writes={"ran"}, input_schema=input_schema)
    g.add_edge(START, "plan")
    g.add_fanout_edge("plan", lambda s: [("worker", p) for p in payloads])
    g.add_edge("worker", END)
    return g.compile()


def test_two_workers_fanned_out_with_one_payload_do_not_share_it():
    """The deep copy is not only for BaseModel state. Two Sends built from one
    dict used to hand both workers the same live object — parallel nodes reading
    each other's mutations, through a channel no node declared a write to."""

    def worker(payload: dict) -> dict:
        payload["hits"].append(1)
        return {"ran": [f"hits={len(payload['hits'])}"]}

    shared = {"who": "orig", "hits": []}
    result = _fanout([shared, shared], worker=worker).invoke({})
    assert result["ran"] == ["hits=1", "hits=1"]
    assert shared == {"who": "orig", "hits": []}


def test_a_basemodel_payload_is_still_isolated():
    """The path that already worked has to keep working."""

    def worker(payload: Shard) -> dict:
        payload.hits.append(1)
        return {"ran": [f"hits={len(payload.hits)}"]}

    shard = Shard(who="orig")
    result = _fanout([shard, shard], worker=worker, input_schema=Shard).invoke({})
    assert result["ran"] == ["hits=1", "hits=1"]
    assert shard.hits == []


def test_a_payload_contradicting_input_schema_is_refused_at_dispatch():
    """`input_schema` is a claim about what the worker is handed; LangGraph
    passes Send.arg through untouched, so GraphARC checks it where Send.node is
    checked rather than letting the worker's body discover it."""
    compiled = _fanout(
        [{"who": "orig"}], worker=lambda p: {"ran": [p.who]}, input_schema=Shard
    )
    with pytest.raises(StateTypeError) as caught:
        compiled.invoke({})
    message = str(caught.value)
    assert "fan-out dispatcher on node 'plan'" in message
    assert "node 'worker'" in message
    assert "expected Shard" in message and "got dict" in message


def test_a_payload_of_the_wrong_model_class_is_refused_too():
    """It used to reach the worker and surface as a bare AttributeError."""

    class Other(BaseModel):
        nope: int = 0

    compiled = _fanout(
        [Other()], worker=lambda p: {"ran": [p.who]}, input_schema=Shard
    )
    with pytest.raises(StateTypeError, match="got Other"):
        compiled.invoke({})


def test_a_worker_declaring_no_input_schema_accepts_any_payload():
    """No schema is no claim — the payload is still copied, but not typed."""
    compiled = _fanout([{"who": "a"}, {"who": "b"}], worker=lambda p: {"ran": [p["who"]]})
    assert sorted(compiled.invoke({})["ran"]) == ["a", "b"]


# -- the front door: input keys are checked like any other write ------------


class Front(GraphARCState):
    question: str = ""
    out: str = ""


def _front_door():
    g = GraphARC(Front, name="front")
    g.add_node("n", lambda s: {"out": f"saw:{s.question!r}"}, writes={"out"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def test_invoke_refuses_an_input_key_the_schema_does_not_have():
    """LangGraph filters input down to known channels before the state model is
    constructed, so `extra="forbid"` never saw the typo and the graph ran to
    completion on defaults — a plausible-looking answer to an empty question."""
    with pytest.raises(WritePermissionError) as caught:
        _front_door().invoke({"quesiton": "typo"})
    assert str(caught.value) == "invoke() targets unknown state fields: ['quesiton']"


def test_stream_refuses_it_in_the_same_words():
    with pytest.raises(WritePermissionError) as caught:
        list(_front_door().stream({"quesiton": "typo"}))
    assert str(caught.value) == "stream() targets unknown state fields: ['quesiton']"


def test_a_wrongly_typed_input_value_is_still_pydantics_to_report():
    """Only the unknown-key case changed; a type error was already loud."""
    with pytest.raises(ValidationError):
        _front_door().invoke({"out": 5})


def test_legal_input_reaches_the_graph_untouched():
    assert _front_door().invoke({"question": "q"})["out"] == "saw:'q'"
    assert _front_door().invoke(Front(question="q"))["out"] == "saw:'q'"


class Typed(GraphARCState):
    count: int = 0
    label: str = ""
    positive: Annotated[int, Field(gt=0)] = 1
    items: Annotated[list[Inner], operator.add] = []
    maybe: str | None = None


def _typed_graph(fn, *, writes, trace=None):
    g = GraphARC(Typed, name="typed", trace=trace)
    g.add_node("n", fn, writes=writes)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def test_a_declared_int_field_rejects_a_string():
    """LangGraph drops unknown keys before validating, so a *declared* field
    carrying the wrong value used to reach the graph's output untouched."""
    compiled = _typed_graph(lambda s: {"count": "not-an-int"}, writes={"count"})
    with pytest.raises(StateTypeError) as exc:
        compiled.invoke({})
    message = str(exc.value)
    assert "'n'" in message and "'count'" in message
    assert "expected int" in message and "got str" in message


def test_type_violations_are_traced_like_any_other_contract_break(trace):
    compiled = _typed_graph(lambda s: {"count": "not-an-int"}, writes={"count"}, trace=trace)
    with pytest.raises(StateTypeError):
        compiled.invoke({}, run_id="r1")
    errors = [e for e in trace.read_events("r1") if e.phase == "error"]
    assert errors and "expected int" in errors[0].error


def test_a_well_typed_write_still_flows_through():
    compiled = _typed_graph(lambda s: {"count": 7, "label": "fine"}, writes={"count", "label"})
    result = compiled.invoke({})
    assert result["count"] == 7
    assert result["label"] == "fine"


def test_the_value_that_lands_in_state_is_the_validated_one():
    """`count: int` has to mean the output holds an int, not something that
    merely could become one."""
    result = _typed_graph(lambda s: {"count": "5"}, writes={"count"}).invoke({})
    assert result["count"] == 5
    assert isinstance(result["count"], int)


def test_field_constraints_are_enforced_not_just_the_base_type():
    compiled = _typed_graph(lambda s: {"positive": -3}, writes={"positive"})
    with pytest.raises(StateTypeError, match="greater than 0"):
        compiled.invoke({})


def test_a_reduced_field_validates_the_update_it_is_handed():
    compiled = _typed_graph(lambda s: {"items": [{"text": "ok"}]}, writes={"items"})
    assert compiled.invoke({})["items"][0].text == "ok"

    bad = _typed_graph(lambda s: {"items": [{"wrong": "shape"}]}, writes={"items"})
    with pytest.raises(StateTypeError, match="'items'"):
        bad.invoke({})


def test_none_is_only_accepted_where_the_schema_allows_it():
    assert _typed_graph(lambda s: {"maybe": None}, writes={"maybe"}).invoke({})["maybe"] is None
    with pytest.raises(StateTypeError, match="'count'"):
        _typed_graph(lambda s: {"count": None}, writes={"count"}).invoke({})


def test_scripted_model_reports_usage():
    model = ScriptedChatModel(responses=["hello world"])
    msg = model.invoke("hi")
    assert msg.content == "hello world"
    assert msg.usage_metadata["total_tokens"] > 0
    with pytest.raises(RuntimeError, match="exhausted"):
        model.invoke("again")
