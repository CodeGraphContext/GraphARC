"""Stage 2 — a typed graph with cycles.

ingest a source -> extract claims (LLM, can be wrong) -> verify citations
(deterministic, against the actual source text) -> summarize, with a bounded
retry cycle back to extraction when citations fail.

Discipline on display:
- explicit modes (EXTRACTING / VERIFYING / ...) instead of overlapping booleans,
- routing on validated events computed by deterministic code, never on prose,
- every step traced with its state delta, so any bad output is attributable to
  a node + input + state change, and every step is a checkpoint replay point.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.parsing import extract_json
from grapharc.runtime.state import GraphARCState
from grapharc.testing import charge_usage


class Mode(StrEnum):
    INGESTING = "ingesting"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    SUMMARIZING = "summarizing"
    DONE = "done"


class Claim(BaseModel):
    text: str
    citation: str  # must be a verbatim quote from the source


class Stage2State(GraphARCState):
    source_path: str
    source_text: str = ""
    mode: Mode = Mode.INGESTING
    attempts: int = 0
    max_attempts: int = 3
    claims: list[Claim] = []
    verified: list[Claim] = []
    rejected: list[dict] = []
    feedback: str = ""
    summary: str = ""
    termination_reason: str | None = None


def build_stage2(
    model: BaseChatModel,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    def ingest(state: Stage2State) -> dict:
        return {
            "source_text": Path(state.source_path).read_text(encoding="utf-8"),
            "mode": Mode.EXTRACTING,
        }

    def extract(state: Stage2State, ctx: RunContext) -> dict:
        prompt = (
            "Extract factual claims from the source. Every claim needs a verbatim "
            'citation quote. Reply with JSON: {"claims": [{"text": ..., "citation": ...}]}\n'
            f"Source:\n{state.source_text}"
        )
        if state.feedback:
            prompt += f"\nPrevious attempt was rejected: {state.feedback}"
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        try:
            raw = extract_json(message.content)["claims"]
            claims = [Claim.model_validate(c) for c in raw]
        except (KeyError, TypeError, ValueError):
            claims = []
        return {"claims": claims, "attempts": state.attempts + 1, "mode": Mode.VERIFYING}

    def verify_citations(state: Stage2State) -> dict:
        verified, rejected = [], []
        for claim in state.claims:
            if claim.citation and claim.citation in state.source_text:
                verified.append(claim)
            else:
                rejected.append(
                    {"claim": claim.model_dump(), "reason": "citation not found in source"}
                )
        if not state.claims:
            rejected.append({"claim": None, "reason": "no parseable claims returned"})
        feedback = (
            "; ".join(f"{r['reason']}: {r['claim']}" for r in rejected) if rejected else ""
        )
        return {"verified": verified, "rejected": rejected, "feedback": feedback}

    def route(state: Stage2State) -> str:
        # Validated events computed from typed state — never from model prose.
        if not state.rejected:
            return "all_verified"
        if state.attempts < state.max_attempts:
            return "retry"
        return "attempts_exhausted"

    def summarize(state: Stage2State, ctx: RunContext) -> dict:
        prompt = (
            "Summarize these verified claims in one paragraph. Reply with plain text.\n"
            + "\n".join(f"- {c.text} (evidence: {c.citation!r})" for c in state.verified)
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        reason = StopReason.TARGET_MET if not state.rejected else StopReason.MAX_ITERATIONS
        return {
            "summary": str(message.content),
            "mode": Mode.DONE,
            "termination_reason": reason.value,
        }

    g = GraphARC(Stage2State, name="stage2_claims", budget=budget, trace=trace)
    g.add_node("ingest", ingest, writes={"source_text", "mode"})
    g.add_node("extract", extract, writes={"claims", "attempts", "mode"})
    g.add_node("verify_citations", verify_citations, writes={"verified", "rejected", "feedback"})
    g.add_node(
        "summarize", summarize, writes={"summary", "mode", "termination_reason"}
    )
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "verify_citations")
    g.add_conditional_edge(
        "verify_citations",
        route,
        {"all_verified": "summarize", "retry": "extract", "attempts_exhausted": "summarize"},
    )
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer)


DEMO_SOURCE = """GraphARC wraps LangGraph with typed state contracts.
Budgets put hard ceilings on iterations, tokens, and wall-clock time.
Traces are JSON lines that double as replay points.
"""
