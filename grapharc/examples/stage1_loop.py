"""Stage 1 — earn the loop.

One agent runs discover -> act -> verify -> repeat with real stop conditions:
target met, no progress for K rounds, round cap, budget ceiling. The verifier
is anchored to evidence (the term is actually present in the chunk it names),
not to the model's own prose.

What would a second node add here? Nothing yet: one specialty (term location),
one toolset (substring search), no parallel work, no separate reviewer role.
That is the Stage 1 gate — being able to say exactly that.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.parsing import extract_json
from grapharc.runtime.state import GraphARCState
from grapharc.testing import charge_usage

MAX_NO_PROGRESS_ROUNDS = 2


class Stage1State(GraphARCState):
    chunks: list[str]
    targets: list[str]
    pending: list[str] = []
    found: dict[str, int] = {}
    round: int = 0
    max_rounds: int = 8
    no_progress_rounds: int = 0
    proposal: str = ""
    candidate: int = -1
    termination_reason: str | None = None


def build_stage1(
    model: BaseChatModel,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    def start(state: Stage1State) -> dict:
        return {"pending": [t for t in state.targets if t not in state.found]}

    def plan(state: Stage1State, ctx: RunContext) -> dict:
        prompt = (
            "You are locating terms in document chunks. "
            f"Unresolved terms: {state.pending}. "
            'Reply with JSON: {"term": "<one unresolved term to search next>"}'
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        try:
            proposal = extract_json(message.content)["term"]
        except (KeyError, TypeError):
            proposal = ""
        return {"proposal": proposal, "round": state.round + 1}

    def act(state: Stage1State) -> dict:
        term = state.proposal.lower()
        candidate = -1
        if term:
            for i, chunk in enumerate(state.chunks):
                if term in chunk.lower():
                    candidate = i
                    break
        return {"candidate": candidate}

    def verify(state: Stage1State) -> dict:
        term, idx = state.proposal, state.candidate
        evidence_ok = (
            bool(term)
            and term in state.pending
            and 0 <= idx < len(state.chunks)
            and term.lower() in state.chunks[idx].lower()
        )
        if evidence_ok:
            found = {**state.found, term: idx}
            return {
                "found": found,
                "pending": [t for t in state.pending if t != term],
                "no_progress_rounds": 0,
            }
        return {"no_progress_rounds": state.no_progress_rounds + 1}

    def route(state: Stage1State) -> str:
        if not state.pending:
            return "target_met"
        if state.round >= state.max_rounds:
            return "max_iterations"
        if state.no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
            return "no_progress"
        return "again"

    def finish_reason(reason: StopReason):
        def finish(state: Stage1State) -> dict:
            return {"termination_reason": reason.value}

        return finish

    g = GraphARC(Stage1State, name="stage1_loop", budget=budget, trace=trace)
    g.add_node("start", start, writes={"pending"})
    g.add_node("plan", plan, writes={"proposal", "round"})
    g.add_node("act", act, writes={"candidate"})
    g.add_node("verify", verify, writes={"found", "pending", "no_progress_rounds"})
    for reason in (StopReason.TARGET_MET, StopReason.MAX_ITERATIONS, StopReason.NO_PROGRESS):
        g.add_node(f"finish_{reason.value}", finish_reason(reason), writes={"termination_reason"})
    g.add_edge(START, "start")
    g.add_edge("start", "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "verify")
    g.add_conditional_edge(
        "verify",
        route,
        {
            "again": "plan",
            "target_met": "finish_target_met",
            "max_iterations": "finish_max_iterations",
            "no_progress": "finish_no_progress",
        },
    )
    for reason in (StopReason.TARGET_MET, StopReason.MAX_ITERATIONS, StopReason.NO_PROGRESS):
        g.add_edge(f"finish_{reason.value}", END)
    return g.compile(checkpointer=checkpointer)


DEMO_CHUNKS = [
    "GraphARC is a graph engineering toolkit built on LangGraph.",
    "Budgets put hard ceilings on iterations, tokens, and time.",
    "A verifier must read evidence, not another agent's prose.",
]
