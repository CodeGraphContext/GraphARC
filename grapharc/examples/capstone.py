"""Capstone — every subsystem, one graph.

A research agent that: recalls what earlier runs learned (memory), fans out
bounded workers over a corpus (runtime), verifies each claim against reality
with an independent reviewer (verify), answers with citations, and persists
what it learned with provenance so the next run starts ahead of this one.

    recall -> plan -> [workers…] -> verify -> answer -> remember

Everything is traced; nothing is written to state a node didn't declare.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from grapharc.memory.retrieval import render_context
from grapharc.memory.store import Claim, MemoryStore
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.fanout import WorkerResult, dedupe, run_guarded
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.runtime.verify import Verdict, verify_claim
from grapharc.testing import charge_usage


class ShardPayload(BaseModel):
    worker: str
    question: str
    doc_ids: list[int]
    docs: list[str]
    timeout_seconds: float = 5.0


class CapstoneState(GraphARCState):
    question: str
    corpus: list[str]
    entities: list[str] = []
    n_workers: int = 2
    recalled: str = ""
    # The verifier's anchor source lives in state, not a closure: it must be
    # per-run and checkpointed, or a resumed run verifies against nothing.
    source_text: str = ""
    worker_results: Annotated[list[WorkerResult], operator.add] = []
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    verdicts: list[Verdict] = []
    answer: str = ""
    persisted_claim_ids: list[str] = []
    termination_reason: str | None = None


def build_capstone(
    worker_model: BaseChatModel,
    reviewer_model: BaseChatModel,
    store: MemoryStore,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    if worker_model is reviewer_model:
        raise ValueError("worker and reviewer must be different model instances")

    def recall(state: CapstoneState) -> dict:
        return {"recalled": render_context(store, entities=state.entities or ["research"])}

    def plan(state: CapstoneState) -> dict:
        return {"source_text": "\n".join(state.corpus)}

    def dispatch(state: CapstoneState) -> list[tuple[str, ShardPayload]]:
        shards: list[list[int]] = [[] for _ in range(state.n_workers)]
        for i in range(len(state.corpus)):
            shards[i % state.n_workers].append(i)
        return [
            (
                "search",
                ShardPayload(
                    worker=f"w{w}",
                    question=state.question,
                    doc_ids=shard,
                    docs=[state.corpus[i] for i in shard],
                ),
            )
            for w, shard in enumerate(shards)
        ]

    def search(payload: ShardPayload) -> dict:
        def job() -> list[dict[str, Any]]:
            terms = [t for t in payload.question.lower().split() if len(t) > 3]
            hits = []
            for doc_id, doc in zip(payload.doc_ids, payload.docs, strict=True):
                for sentence in doc.split(". "):
                    if any(t in sentence.lower() for t in terms):
                        hits.append({"doc_id": doc_id, "quote": sentence.strip(". ")})
            return hits

        return {
            "worker_results": [
                run_guarded(job, worker=payload.worker, timeout_seconds=payload.timeout_seconds)
            ]
        }

    def verify(state: CapstoneState) -> dict:
        ok = [r for r in state.worker_results if r.ok]
        failures = [f"{r.worker}: {r.error}" for r in state.worker_results if not r.ok]
        evidence = dedupe(
            (e for r in ok for e in r.evidence), key=lambda e: (e["doc_id"], e["quote"])
        )
        verdicts = [
            verify_claim(
                reviewer_model,
                text=f"{state.question} — {e['quote']}",
                citation=e["quote"],
                source_text=state.source_text,
            )
            for e in evidence
        ]
        return {"evidence": evidence, "failures": failures, "verdicts": verdicts}

    def answer(state: CapstoneState, ctx: RunContext) -> dict:
        accepted = [v for v in state.verdicts if v.accepted]
        if not accepted:
            return {"answer": "", "termination_reason": StopReason.NO_PROGRESS.value}
        prompt = (
            f"Answer the question using ONLY the verified evidence. Cite each fact.\n"
            f"Question: {state.question}\n"
            f"Prior knowledge:\n{state.recalled}\n"
            + "\n".join(f"- {v.citation}" for v in accepted)
        )
        message = worker_model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        return {
            "answer": str(message.content),
            "termination_reason": StopReason.TARGET_MET.value,
        }

    def remember(state: CapstoneState, ctx: RunContext) -> dict:
        ids = []
        for verdict in state.verdicts:
            if not verdict.accepted:
                continue  # only verified facts are ever persisted
            claim = Claim(
                subject=state.entities[0] if state.entities else "research",
                predicate="evidence for",
                object=verdict.citation,
                source=f"{state.question}",
                run_id=ctx.run_id,
            )
            store.add(claim)
            ids.append(claim.id)
        return {"persisted_claim_ids": ids}

    g = GraphARC(CapstoneState, name="capstone", budget=budget, trace=trace)
    g.add_node("recall", recall, writes={"recalled"})
    g.add_node("plan", plan, writes={"source_text"})
    g.add_node("search", search, writes={"worker_results"}, input_schema=ShardPayload)
    g.add_node("verify", verify, writes={"evidence", "failures", "verdicts"})
    g.add_node("answer", answer, writes={"answer", "termination_reason"})
    g.add_node("remember", remember, writes={"persisted_claim_ids"})
    g.add_edge(START, "recall")
    g.add_edge("recall", "plan")
    g.add_fanout_edge("plan", dispatch)
    g.add_edge("search", "verify")
    g.add_edge("verify", "answer")
    g.add_edge("answer", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)


DEMO_CORPUS = [
    "GraphARC enforces typed state contracts on every edge. "
    "Budgets place hard ceilings on iterations and tokens.",
    "Traces are JSON lines that double as replay points. "
    "Verifiers read evidence rather than another agent's prose.",
]
