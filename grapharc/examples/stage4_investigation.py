"""Stage 4 — prove cycles stop.

A bounded investigation loop: the model proposes a search query each round,
a deterministic search runs it over the corpus, and findings accumulate with
cross-round dedup. The cycle carries three independent brakes (ProgressGuard:
target / no-progress window / round cap) plus the runtime budget underneath.

The gate: give it an impossible task and it stops early with a recorded
reason — without rediscovering the same dead ends until the budget is gone.
"""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import ProgressGuard
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.parsing import extract_json
from grapharc.runtime.state import GraphARCState
from grapharc.testing import charge_usage


class Stage4State(GraphARCState):
    corpus: list[str]
    goal: str
    target_findings: int | None = None
    max_rounds: int = 6
    max_empty_rounds: int = 2
    round: int = 0
    empty_rounds: int = 0
    query: str = ""
    tried_queries: list[str] = []
    findings: list[str] = []
    termination_reason: str | None = None


def build_stage4(
    model: BaseChatModel,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    def propose(state: Stage4State, ctx: RunContext) -> dict:
        prompt = (
            f"You are investigating: {state.goal}\n"
            f"Queries already tried: {state.tried_queries}\n"
            f"Findings so far: {len(state.findings)}\n"
            'Propose ONE new search query as JSON: {"query": "..."}'
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        try:
            query = str(extract_json(message.content)["query"])
        except (KeyError, TypeError):
            query = ""
        return {
            "query": query,
            "round": state.round + 1,
            "tried_queries": [*state.tried_queries, query],
        }

    def search(state: Stage4State) -> dict:
        terms = [w for w in re.findall(r"[a-zA-Z]+", state.query.lower()) if len(w) > 3]
        matches = [
            doc
            for doc in state.corpus
            if terms and any(t in doc.lower() for t in terms)
        ]
        # Cross-round dedup: a finding is only new if no earlier round recorded it.
        new = [m for m in matches if m not in state.findings]
        if new:
            return {"findings": [*state.findings, *new], "empty_rounds": 0}
        return {"empty_rounds": state.empty_rounds + 1}

    def route(state: Stage4State) -> str:
        guard = ProgressGuard(
            target=state.target_findings,
            max_rounds=state.max_rounds,
            max_empty_rounds=state.max_empty_rounds,
        )
        reason = guard.assess(
            round=state.round,
            total_findings=len(state.findings),
            empty_rounds=state.empty_rounds,
        )
        return reason.value if reason is not None else "again"

    def finish(reason_value: str):
        def _finish(state: Stage4State) -> dict:
            return {"termination_reason": reason_value}

        return _finish

    g = GraphARC(Stage4State, name="stage4_investigation", budget=budget, trace=trace)
    g.add_node("propose", propose, writes={"query", "round", "tried_queries"})
    g.add_node("search", search, writes={"findings", "empty_rounds"})
    for reason in ("target_met", "no_progress", "max_iterations"):
        g.add_node(f"finish_{reason}", finish(reason), writes={"termination_reason"})
    g.add_edge(START, "propose")
    g.add_edge("propose", "search")
    g.add_conditional_edge(
        "search",
        route,
        {
            "again": "propose",
            "target_met": "finish_target_met",
            "no_progress": "finish_no_progress",
            "max_iterations": "finish_max_iterations",
        },
    )
    for reason in ("target_met", "no_progress", "max_iterations"):
        g.add_edge(f"finish_{reason}", END)
    return g.compile(checkpointer=checkpointer)


DEMO_CORPUS = [
    "Budget meters charge tokens and iterations.",
    "Traces record every node execution as a JSON line.",
    "Convergence guards stop cycles for recorded reasons.",
]
