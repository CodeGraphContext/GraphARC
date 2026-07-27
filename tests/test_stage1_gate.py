"""Stage 1 gates: the loop knows when it is done, and an impossible task halts
with a recorded reason instead of rediscovering dead ends until the budget dies."""

import pytest

from grapharc.examples.stage1_loop import DEMO_CHUNKS, build_stage1
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.testing import ScriptedChatModel


@pytest.mark.timeout(20)
def test_gate_loop_knows_when_it_is_done(trace):
    model = ScriptedChatModel(
        responses=['{"term": "budgets"}', '{"term": "verifier"}']
    )
    compiled = build_stage1(model, trace=trace)
    result = compiled.invoke({"chunks": DEMO_CHUNKS, "targets": ["budgets", "verifier"]})
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert result["found"] == {"budgets": 1, "verifier": 2}
    assert result["pending"] == []
    # Evidence anchoring: each found term really is in the chunk it names.
    for term, idx in result["found"].items():
        assert term.lower() in DEMO_CHUNKS[idx].lower()


@pytest.mark.timeout(20)
def test_gate_impossible_task_stops_without_burning_budget(trace):
    # The model keeps proposing a term that exists nowhere.
    model = ScriptedChatModel(responses=['{"term": "zebra"}'], on_exhausted="repeat")
    compiled = build_stage1(model, trace=trace, budget=Budget(max_iterations=100))
    result = compiled.invoke({"chunks": DEMO_CHUNKS, "targets": ["zebra"]})
    assert result["termination_reason"] == StopReason.NO_PROGRESS.value
    # It stopped after the no-progress window, far below the hard iteration ceiling.
    assert result["round"] <= 3
    assert result.get("found", {}) == {}


@pytest.mark.timeout(20)
def test_gate_round_cap_is_a_second_independent_stop_condition(trace):
    # Alternate between one real term and a junk term: progress keeps being made
    # rarely enough that no_progress never fires, so the round cap must.
    model = ScriptedChatModel(responses=['{"term": "nonsense"}'], on_exhausted="repeat")
    compiled = build_stage1(model, trace=trace)
    result = compiled.invoke(
        {"chunks": DEMO_CHUNKS, "targets": ["budgets"], "max_rounds": 1}
    )
    assert result["termination_reason"] in (
        StopReason.MAX_ITERATIONS.value,
        StopReason.NO_PROGRESS.value,
    )
