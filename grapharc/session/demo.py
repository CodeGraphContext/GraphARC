"""A session graph small enough to read, real enough to prove the contract.

Four nodes — ingest, plan, apply, report — with `apply` gated on human
approval. It exists so the session runtime has something concrete to drive
across a process boundary, and so the state contract in `state.py` has a worked
example: `ingest` and `plan` consume `inbox`, and `report` routes on
`state.decision`.

The model is scripted, not live: nothing here is testing a model. What it is
testing is that the same session, resumed in a different interpreter, carries
on from the node it stopped at.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC
from grapharc.session.registry import GraphRegistry, GraphSpec, register_graph
from grapharc.session.state import SessionState
from grapharc.testing import ScriptedChatModel

GRAPH_NAME = "session_demo"
APPROVAL_NODE = "apply"

DEMO_SCRIPT = ("draft the release note", "rewrite the release note", "polish the release note")


class DemoState(SessionState):
    """`received` and `log` are append-only so a resume cannot be faked.

    If a resumed turn silently re-ran earlier nodes, `log` would show their
    names twice. That is the assertion the cross-process test rests on.
    """

    received: list[str] = []
    log: list[str] = []
    plan: str = ""
    outcome: str = ""


def build_demo(
    checkpointer: Any = None,
    *,
    model: BaseChatModel | None = None,
    trace: TraceRecorder | None = None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    """Compile the demo graph. `checkpointer` first, to match `GraphFactory`."""
    chat = model or ScriptedChatModel(responses=list(DEMO_SCRIPT), on_exhausted="repeat")

    def ingest(state: DemoState) -> dict:
        return {
            "received": [*state.received, *state.inbox],
            "inbox": [],
            "log": [*state.log, "ingest"],
        }

    def plan(state: DemoState) -> dict:
        reply = str(chat.invoke([HumanMessage(content=f"plan for: {state.received}")]).content)
        # Anything that arrived after `ingest` ran is steering: it reached the
        # session while this node was still pending, and it still lands.
        steer = state.inbox[-1] if state.inbox else ""
        return {
            "plan": f"{reply} | steer: {steer}" if steer else reply,
            "received": [*state.received, *state.inbox],
            "inbox": [],
            "log": [*state.log, "plan"],
        }

    def apply(state: DemoState) -> dict:
        return {"outcome": f"applied: {state.plan}", "log": [*state.log, "apply"]}

    def report(state: DemoState) -> dict:
        decision = state.decision
        if decision is not None and not decision.approved:
            who = decision.decided_by or "an operator"
            return {
                "outcome": f"refused by {who}: {decision.reason}",
                "log": [*state.log, "report"],
            }
        return {"log": [*state.log, "report"]}

    g = GraphARC(DemoState, name=GRAPH_NAME, budget=budget, trace=trace)
    g.add_node("ingest", ingest, writes={"received", "inbox", "log"})
    g.add_node("plan", plan, writes={"plan", "received", "inbox", "log"})
    g.add_node(APPROVAL_NODE, apply, writes={"outcome", "log"})
    g.add_node("report", report, writes={"outcome", "log"})
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_edge("plan", APPROVAL_NODE)
    g.add_edge(APPROVAL_NODE, "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def _describe(node: str, values: Mapping[str, Any]) -> str:
    return f"{node}: {values.get('plan') or '(no plan yet)'}"


def register_demo(
    *, registry: GraphRegistry | None = None, replace: bool = False
) -> GraphSpec:
    return register_graph(
        GRAPH_NAME,
        build_demo,
        approval_nodes=(APPROVAL_NODE,),
        describe=_describe,
        registry=registry,
        replace=replace,
    )


# Importing this module is how a process declares it can run the demo graph;
# a resume in a process that never imported it fails with UnknownGraphError,
# which is the intended behaviour, not an accident.
SPEC = register_demo(replace=True)

__all__ = [
    "APPROVAL_NODE",
    "DEMO_SCRIPT",
    "GRAPH_NAME",
    "SPEC",
    "DemoState",
    "build_demo",
    "register_demo",
]
