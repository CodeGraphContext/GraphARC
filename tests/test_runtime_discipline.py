"""Unit tests for the runtime discipline layer: write permissions, DAG mode, budgets, traces."""

import operator
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import (
    END,
    START,
    GraphARC,
    GraphCycleError,
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
