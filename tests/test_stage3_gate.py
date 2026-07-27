"""Stage 3 gates: one worker cannot sink the batch, and duplicate evidence
does not inflate confidence."""

import sys

import pytest

from grapharc.examples.stage3_fanout import DEMO_CHUNKS, build_stage3
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.fanout import run_guarded
from grapharc.testing import ScriptedChatModel

QUESTION = "How do budgets bound work?"


def test_malformed_worker_return_is_a_failure_not_a_crash():
    """A worker that forgets to return a list must not raise in the caller —
    that would take the whole batch down."""
    result = run_guarded(lambda: None, worker="w1", timeout_seconds=5)
    assert result.ok is False
    assert "expected list" in result.error


def test_baseexception_worker_is_recorded_as_failed():
    """SystemExit (or any BaseException) must not be reported as a success."""

    def exits():
        sys.exit(3)

    result = run_guarded(exits, worker="w1", timeout_seconds=5)
    assert result.ok is False
    assert result.error and "SystemExit" in result.error


@pytest.mark.timeout(30)
def test_happy_path_fans_out_and_synthesizes(trace):
    model = ScriptedChatModel(responses=["Budgets cap iterations, tokens, and time."])
    compiled = build_stage3(model, trace=trace)
    result = compiled.invoke(
        {"question": QUESTION, "chunks": DEMO_CHUNKS, "n_workers": 3}
    )
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert len(result["worker_results"]) == 3
    assert result["failures"] == []
    assert result["evidence"]
    assert result["answer"]


@pytest.mark.timeout(30)
def test_gate_failed_and_hung_workers_cannot_sink_the_batch(trace):
    model = ScriptedChatModel(responses=["Partial answer from surviving workers."])
    compiled = build_stage3(model, trace=trace)
    result = compiled.invoke(
        {
            "question": QUESTION,
            "chunks": DEMO_CHUNKS,
            "n_workers": 3,
            "worker_specs": {
                "w1": {"fail": True},
                "w2": {"hang_seconds": 30.0},
            },
            "worker_timeout_seconds": 0.3,
        }
    )
    # The batch completed despite one crash and one hang.
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert len(result["worker_results"]) == 3
    assert len(result["failures"]) == 2
    assert any("exploded" in f for f in result["failures"])
    assert any("timeout" in f for f in result["failures"])
    # Surviving worker's evidence still made it through.
    assert result["evidence"]
    assert result["answer"]


@pytest.mark.timeout(30)
def test_gate_duplicate_evidence_does_not_inflate_confidence(trace):
    model = ScriptedChatModel(responses=["Deduped answer."])
    compiled = build_stage3(model, trace=trace)
    result = compiled.invoke(
        {
            "question": QUESTION,
            "chunks": DEMO_CHUNKS,
            "n_workers": 3,
            "overlap": True,  # every shard also scans chunk 0
        }
    )
    raw_evidence = [
        e for r in result["worker_results"] if r.ok for e in r.evidence
    ]
    deduped = result["evidence"]
    # Overlap manufactured duplicates, and synthesis saw each fact once.
    assert len(raw_evidence) > len(deduped)
    keys = [(e["chunk_id"], e["quote"]) for e in deduped]
    assert len(keys) == len(set(keys))
    # unique_sources counts chunks, not repetitions.
    assert result["unique_sources"] == len({e["chunk_id"] for e in deduped})


@pytest.mark.timeout(30)
def test_concurrency_cap_still_completes(trace):
    model = ScriptedChatModel(responses=["Serialized answer."])
    compiled = build_stage3(model, trace=trace, budget=Budget(max_concurrency=1))
    result = compiled.invoke(
        {"question": QUESTION, "chunks": DEMO_CHUNKS, "n_workers": 3}
    )
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert len(result["worker_results"]) == 3
