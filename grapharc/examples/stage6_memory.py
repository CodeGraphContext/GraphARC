"""Stage 6 — persist memory across runs.

A graph that recalls before it works, extracts new claims from a source, and
writes them back with provenance. Corrections supersede rather than overwrite.

The gate this exists for: run #51 uses a fact from run #12, sees that run #37
superseded it, and avoids a known dead end.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from grapharc.memory.retrieval import render_context
from grapharc.memory.store import Claim, MemoryStore
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.parsing import extract_json
from grapharc.runtime.state import GraphARCState
from grapharc.testing import charge_usage


class Stage6State(GraphARCState):
    entities: list[str]
    source_name: str = ""
    source_text: str = ""
    recalled: str = ""
    avoided_dead_ends: list[str] = []
    new_claim_ids: list[str] = []
    answer: str = ""
    termination_reason: str | None = None


def build_stage6(
    model: BaseChatModel,
    store: MemoryStore,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    def recall(state: Stage6State) -> dict:
        context = render_context(store, entities=state.entities)
        dead = [
            f"{c.subject} {c.predicate} {c.object}"
            for e in state.entities
            for c in store.dead_ends(e)
        ]
        return {"recalled": context, "avoided_dead_ends": dead}

    def extract(state: Stage6State, ctx: RunContext) -> dict:
        if not state.source_text:
            return {"new_claim_ids": []}
        prompt = (
            "Extract subject-predicate-object facts from the source. "
            "Do not repeat facts already known.\n"
            f"Already known:\n{state.recalled}\n"
            f"Source:\n{state.source_text}\n"
            'Reply with JSON: {"claims": [{"subject":..,"predicate":..,"object":..}]}'
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        try:
            raw = extract_json(message.content)["claims"]
        except (KeyError, TypeError):
            raw = []
        ids = []
        for item in raw:
            try:
                claim = Claim(
                    subject=str(item["subject"]),
                    predicate=str(item["predicate"]),
                    object=str(item["object"]),
                    source=state.source_name or "unknown",
                    run_id=ctx.run_id,
                )
            except (KeyError, TypeError):
                continue  # a malformed claim is dropped, not persisted
            store.add(claim)
            ids.append(claim.id)
        return {"new_claim_ids": ids}

    def answer(state: Stage6State, ctx: RunContext) -> dict:
        prompt = (
            "Answer using the known facts. Cite the source of each fact used.\n"
            f"{render_context(store, entities=state.entities)}"
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        return {
            "answer": str(message.content),
            "termination_reason": StopReason.TARGET_MET.value,
        }

    g = GraphARC(Stage6State, name="stage6_memory", budget=budget, trace=trace)
    g.add_node("recall", recall, writes={"recalled", "avoided_dead_ends"})
    g.add_node("extract", extract, writes={"new_claim_ids"})
    g.add_node("answer", answer, writes={"answer", "termination_reason"})
    g.add_edge(START, "recall")
    g.add_edge("recall", "extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)
