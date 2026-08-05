import pytest
from pydantic import BaseModel

from grapharc.runtime.graph import GraphARC, GraphRoutingError


def test_add_conditional_edge_rejects_unknown_target_at_add_time():
    class State(BaseModel):
        value: str = ""

    arc = GraphARC(state_schema=State, name="test")

    arc.add_node("start", lambda state: {"value": "ok"}, writes={"value"})
    arc.add_node("done", lambda state: {"value": "ok"}, writes={"value"})

    with pytest.raises(GraphRoutingError, match="not a node of graph") as exc_info:
        arc.add_conditional_edge("start", lambda _: "go", {"go": "missing"})
    print(f"\nMESSAGE: {exc_info.value}")


def test_add_conditional_edge_accepts_known_target_mapping():
    class State(BaseModel):
        value: str = ""

    arc = GraphARC(state_schema=State, name="test")

    arc.add_node("start", lambda state: {"value": "ok"}, writes={"value"})
    arc.add_node("done", lambda state: {"value": "ok"}, writes={"value"})

    arc.add_conditional_edge("start", lambda _: "go", {"go": "done"})