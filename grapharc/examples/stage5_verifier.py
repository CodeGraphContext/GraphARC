"""Stage 5 — verify against reality.

An author model drafts claims with citations; an *independent* verifier judges
them. Independence is enforced, not hoped for: the reviewer must be a different
model instance (correlated agreement is the failure mode), it gets fresh
context (claim + evidence only, never the author's conversation), and its
verdict sits on top of a deterministic anchor — a citation that does not exist
in the source is rejected before any model opinion is consulted.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from grapharc.examples.stage2_claims import Claim
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.parsing import extract_json
from grapharc.runtime.state import GraphARCState
from grapharc.runtime.verify import Verdict, verify_claim
from grapharc.testing import charge_usage


class Stage5State(GraphARCState):
    source_text: str
    claims: list[Claim] = []
    verdicts: list[Verdict] = []
    accepted: list[str] = []
    rejected: list[str] = []
    termination_reason: str | None = None


def build_stage5(
    author: BaseChatModel,
    reviewer: BaseChatModel,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    if author is reviewer:
        raise ValueError(
            "author and reviewer must be different model instances: a model "
            "grading its own homework is the correlated-agreement failure mode"
        )

    def draft(state: Stage5State, ctx: RunContext) -> dict:
        prompt = (
            "Extract factual claims with verbatim citation quotes from the source. "
            'Reply with JSON: {"claims": [{"text": ..., "citation": ...}]}\n'
            f"Source:\n{state.source_text}"
        )
        message = author.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        try:
            raw = extract_json(message.content)["claims"]
            claims = [Claim.model_validate(c) for c in raw]
        except (KeyError, TypeError, ValueError):
            claims = []
        return {"claims": claims}

    def verify(state: Stage5State) -> dict:
        verdicts = [
            verify_claim(
                reviewer,
                text=claim.text,
                citation=claim.citation,
                source_text=state.source_text,
            )
            for claim in state.claims
        ]
        return {"verdicts": verdicts}

    def report(state: Stage5State) -> dict:
        return {
            "accepted": [v.claim_text for v in state.verdicts if v.accepted],
            "rejected": [v.claim_text for v in state.verdicts if not v.accepted],
            "termination_reason": StopReason.TARGET_MET.value,
        }

    g = GraphARC(Stage5State, name="stage5_verifier", budget=budget, trace=trace)
    g.add_node("draft", draft, writes={"claims"})
    g.add_node("verify", verify, writes={"verdicts"})
    g.add_node("report", report, writes={"accepted", "rejected", "termination_reason"})
    g.add_edge(START, "draft")
    g.add_edge("draft", "verify")
    g.add_edge("verify", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


DEMO_SOURCE = """GraphARC wraps LangGraph with typed state contracts.
Budgets put hard ceilings on iterations, tokens, and wall-clock time.
Traces are JSON lines that double as replay points.
"""
