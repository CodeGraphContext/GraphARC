"""Convergence: cycles must stop for a recorded reason.

A cycle is only allowed when new evidence can change the next action, and every
run must end with an explicit, machine-readable termination reason — never by
silently running out of road.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class StopReason(StrEnum):
    TARGET_MET = "target_met"
    MAX_ITERATIONS = "max_iterations"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HUMAN_STOPPED = "human_stopped"
    ERROR = "error"


class ProgressGuard(BaseModel):
    """Multiple independent stop conditions for a cycle (Stage 4 discipline).

    A cycle needs more than one brake: a target, a no-progress window, and a
    round cap. `assess` returns the first triggered StopReason, or None to
    continue. The runtime budget is the hard ceiling underneath all of these.
    """

    target: int | None = None
    max_rounds: int = 10
    max_empty_rounds: int = 2

    def assess(
        self, *, round: int, total_findings: int, empty_rounds: int
    ) -> StopReason | None:
        if self.target is not None and total_findings >= self.target:
            return StopReason.TARGET_MET
        if empty_rounds >= self.max_empty_rounds:
            return StopReason.NO_PROGRESS
        if round >= self.max_rounds:
            return StopReason.MAX_ITERATIONS
        return None
