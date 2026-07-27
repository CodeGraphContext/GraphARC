"""Stage 4 gate: an impossible task halts early, for a recorded reason,
without rediscovering dead ends until the budget is gone."""

import pytest

from grapharc.examples.stage4_investigation import DEMO_CORPUS, build_stage4
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import ProgressGuard, StopReason
from grapharc.testing import ScriptedChatModel


@pytest.mark.timeout(20)
def test_happy_path_stops_on_target(trace):
    model = ScriptedChatModel(
        responses=['{"query": "budget meters"}', '{"query": "trace lines"}'],
        on_exhausted="repeat",
    )
    compiled = build_stage4(model, trace=trace)
    result = compiled.invoke(
        {"corpus": DEMO_CORPUS, "goal": "map the runtime", "target_findings": 2}
    )
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert len(result["findings"]) >= 2
    # Cross-round dedup: no finding recorded twice.
    assert len(result["findings"]) == len(set(result["findings"]))


@pytest.mark.timeout(20)
def test_gate_impossible_task_halts_far_below_budget(trace):
    # Nothing in the corpus matches; the model keeps proposing the same dead end.
    model = ScriptedChatModel(responses=['{"query": "quantum zebra"}'], on_exhausted="repeat")
    compiled = build_stage4(model, trace=trace, budget=Budget(max_iterations=100))
    result = compiled.invoke(
        {
            "corpus": DEMO_CORPUS,
            "goal": "find the quantum zebra",
            "target_findings": 5,
            "max_empty_rounds": 2,
        }
    )
    assert result["termination_reason"] == StopReason.NO_PROGRESS.value
    assert result["round"] == 2  # stopped by the no-progress window...
    meter = compiled.last_run.meter
    assert meter.iterations < 10  # ...far below the 100-iteration hard ceiling
    assert result.get("findings", []) == []


@pytest.mark.timeout(20)
def test_gate_round_cap_fires_when_progress_trickles(trace):
    # Each round finds something new-ish? No — corpus has 3 docs; after they're
    # found, rounds go empty. Set a tight round cap and a loose empty window so
    # the cap is the brake that fires.
    model = ScriptedChatModel(
        responses=['{"query": "budget trace convergence"}'], on_exhausted="repeat"
    )
    compiled = build_stage4(model, trace=trace)
    result = compiled.invoke(
        {
            "corpus": DEMO_CORPUS,
            "goal": "map everything",
            "target_findings": None,
            "max_rounds": 3,
            "max_empty_rounds": 99,
        }
    )
    assert result["termination_reason"] == StopReason.MAX_ITERATIONS.value
    assert result["round"] == 3


def test_progress_guard_priority_order():
    guard = ProgressGuard(target=5, max_rounds=10, max_empty_rounds=2)
    # Target met wins even if other conditions also hold.
    assert (
        guard.assess(round=10, total_findings=5, empty_rounds=2)
        == StopReason.TARGET_MET
    )
    assert (
        guard.assess(round=1, total_findings=0, empty_rounds=2)
        == StopReason.NO_PROGRESS
    )
    assert (
        guard.assess(round=10, total_findings=1, empty_rounds=0)
        == StopReason.MAX_ITERATIONS
    )
    assert guard.assess(round=1, total_findings=1, empty_rounds=0) is None
