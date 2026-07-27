"""Capstone gate: question -> fan-out -> verify -> cited answer -> facts
persisted -> a second run reuses them. Every subsystem in one path."""

import pytest

from grapharc.examples.capstone import DEMO_CORPUS, build_capstone
from grapharc.memory import MemoryStore
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.testing import ScriptedChatModel

QUESTION = "How does GraphARC bound work with budgets?"


def _reviewer(n: int) -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=['{"supported": true, "reason": "quote supports it"}'] * n,
        on_exhausted="repeat",
    )


@pytest.mark.timeout(30)
def test_gate_end_to_end_answer_then_reuse(trace):
    store = MemoryStore()

    worker = ScriptedChatModel(responses=["Budgets cap iterations and tokens [doc 0]."])
    compiled = build_capstone(
        worker, _reviewer(8), store, trace=trace, budget=Budget(max_iterations=50)
    )
    first = compiled.invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r1",
    )

    # Answer produced from verified evidence only.
    assert first["termination_reason"] == StopReason.TARGET_MET.value
    assert first["answer"]
    assert first["evidence"]
    assert all(v.anchor_ok for v in first["verdicts"])
    assert first["failures"] == []

    # Facts were persisted with provenance from this run.
    assert first["persisted_claim_ids"]
    persisted = store.get(first["persisted_claim_ids"][0])
    assert persisted.run_id == "r1"
    assert persisted.source == QUESTION

    # A second run recalls what the first learned before doing any work.
    worker2 = ScriptedChatModel(responses=["Same answer, now with prior knowledge."])
    second = build_capstone(worker2, _reviewer(8), store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r2",
    )
    assert "GraphARC" in second["recalled"]
    assert "evidence for" in second["recalled"]  # a fact from run r1
    assert QUESTION in second["recalled"]  # its provenance travelled with it


@pytest.mark.timeout(30)
def test_unverified_evidence_is_never_persisted(trace):
    """A reviewer that rejects everything yields no answer and no memory writes."""
    store = MemoryStore()
    worker = ScriptedChatModel(responses=["unused"], on_exhausted="repeat")
    rejecting = ScriptedChatModel(
        responses=['{"supported": false, "reason": "does not support"}'],
        on_exhausted="repeat",
    )
    result = build_capstone(worker, rejecting, store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r3",
    )
    assert result["termination_reason"] == StopReason.NO_PROGRESS.value
    assert result["answer"] == ""
    assert result.get("persisted_claim_ids", []) == []
    assert store.all_claims() == []


def test_same_model_for_worker_and_reviewer_is_refused():
    model = ScriptedChatModel(responses=["x"])
    with pytest.raises(ValueError, match="different model instances"):
        build_capstone(model, model, MemoryStore())


@pytest.mark.timeout(30)
def test_metrics_and_mermaid_derive_from_the_trace(trace):
    store = MemoryStore()
    worker = ScriptedChatModel(responses=["answer"], on_exhausted="repeat")
    build_capstone(worker, _reviewer(8), store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r4",
    )
    metrics = summarize(trace, "r4")
    assert metrics is not None
    assert metrics.graph == "capstone"
    assert metrics.errors == 0
    assert metrics.nodes_executed >= 6  # recall, plan, 2 workers, verify, answer, remember
    assert metrics.termination_reason == StopReason.TARGET_MET.value
    assert metrics.per_node["search"] == 2  # both fan-out workers ran

    diagram = to_mermaid(trace, "r4")
    assert diagram.startswith("flowchart TD")
    assert "recall" in diagram and "remember" in diagram
