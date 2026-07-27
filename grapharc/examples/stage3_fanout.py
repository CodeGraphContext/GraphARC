"""Stage 3 — bounded fan-out.

One research question is sharded across several workers dispatched in parallel
(an explicit act via `add_fanout_edge` — the default execution model is serial).
Every worker runs failure-isolated with a wall-clock timeout: a crash or hang
becomes data, not a dead batch. Evidence is deduplicated *before* synthesis so
two workers finding the same chunk don't double the confidence.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.fanout import WorkerResult, dedupe, run_guarded
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.testing import charge_usage

DEFAULT_WORKER_TIMEOUT_SECONDS = 2.0


class WorkerPayload(BaseModel):
    worker: str
    question: str
    chunk_ids: list[int]
    chunks: list[str]
    timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS
    # Test knobs simulating real-world worker failure modes:
    fail: bool = False
    hang_seconds: float = 0.0


class Stage3State(GraphARCState):
    question: str
    chunks: list[str]
    n_workers: int = 3
    overlap: bool = False  # make shards overlap to manufacture duplicate evidence
    worker_specs: dict[str, dict[str, Any]] = {}  # per-worker fail/hang knobs
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS
    worker_results: Annotated[list[WorkerResult], operator.add] = []
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    unique_sources: int = 0
    answer: str = ""
    termination_reason: str | None = None


def _keywords(question: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]+", question.lower()) if len(w) > 3]


def build_stage3(
    model: BaseChatModel,
    *,
    trace: TraceRecorder | None = None,
    checkpointer=None,
    budget: Budget | None = None,
) -> CompiledGraphARC:
    def prepare(state: Stage3State) -> None:
        return None

    def dispatch(state: Stage3State) -> list[tuple[str, WorkerPayload]]:
        shards: list[list[int]] = [[] for _ in range(state.n_workers)]
        for i in range(len(state.chunks)):
            shards[i % state.n_workers].append(i)
        if state.overlap:
            for shard in shards:
                if 0 not in shard:
                    shard.insert(0, 0)
        payloads = []
        for w, shard in enumerate(shards):
            name = f"w{w}"
            spec = state.worker_specs.get(name, {})
            payloads.append(
                (
                    "worker",
                    WorkerPayload(
                        worker=name,
                        question=state.question,
                        chunk_ids=shard,
                        chunks=[state.chunks[i] for i in shard],
                        timeout_seconds=state.worker_timeout_seconds,
                        fail=bool(spec.get("fail", False)),
                        hang_seconds=float(spec.get("hang_seconds", 0.0)),
                    ),
                )
            )
        return payloads

    def worker(payload: WorkerPayload) -> dict:
        def job() -> list[dict[str, Any]]:
            if payload.fail:
                raise RuntimeError(f"worker {payload.worker} exploded")
            if payload.hang_seconds:
                time.sleep(payload.hang_seconds)
            keywords = _keywords(payload.question)
            evidence = []
            for chunk_id, chunk in zip(payload.chunk_ids, payload.chunks, strict=True):
                for sentence in re.split(r"(?<=[.!?])\s+", chunk):
                    if any(k in sentence.lower() for k in keywords):
                        evidence.append(
                            {"chunk_id": chunk_id, "quote": sentence.strip()}
                        )
            return evidence

        result = run_guarded(
            job, worker=payload.worker, timeout_seconds=payload.timeout_seconds
        )
        return {"worker_results": [result]}

    def synthesize(state: Stage3State, ctx: RunContext) -> dict:
        ok = [r for r in state.worker_results if r.ok]
        failures = [f"{r.worker}: {r.error}" for r in state.worker_results if not r.ok]
        # Dedup BEFORE synthesis: duplicate evidence must not inflate confidence.
        evidence = dedupe(
            (e for r in ok for e in r.evidence),
            key=lambda e: (e["chunk_id"], e["quote"]),
        )
        unique_sources = len({e["chunk_id"] for e in evidence})
        if not evidence:
            return {
                "evidence": [],
                "failures": failures,
                "unique_sources": 0,
                "answer": "",
                "termination_reason": StopReason.ERROR.value,
            }
        prompt = (
            f"Answer the question from the deduplicated evidence only.\n"
            f"Question: {state.question}\n"
            + "\n".join(f"- [chunk {e['chunk_id']}] {e['quote']}" for e in evidence)
        )
        message = model.invoke([HumanMessage(content=prompt)])
        charge_usage(ctx, message)
        return {
            "evidence": evidence,
            "failures": failures,
            "unique_sources": unique_sources,
            "answer": str(message.content),
            "termination_reason": StopReason.TARGET_MET.value,
        }

    g = GraphARC(Stage3State, name="stage3_fanout", budget=budget, trace=trace)
    g.add_node("prepare", prepare, writes=set())
    g.add_node("worker", worker, writes={"worker_results"}, input_schema=WorkerPayload)
    g.add_node(
        "synthesize",
        synthesize,
        writes={"evidence", "failures", "unique_sources", "answer", "termination_reason"},
    )
    g.add_edge(START, "prepare")
    g.add_fanout_edge("prepare", dispatch)
    g.add_edge("worker", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)


DEMO_CHUNKS = [
    "Budgets put hard ceilings on iterations and tokens. Budgets are checked before every node.",
    "Traces are JSON lines. Every trace line is a replay point.",
    "Verifiers read evidence, not prose. A verifier needs a deterministic anchor.",
    "Fan-out is explicit. Budgets also cap concurrency during fan-out.",
]
