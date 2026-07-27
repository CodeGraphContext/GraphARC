"""Bounded work: every graph run carries hard limits on iterations, tokens, and wall-clock time.

The graph-engineering production checklist requires "bounded work": concurrency,
iterations, tokens, and time all need hard limits so a cycling graph cannot burn
money forever. `Budget` declares the limits; `BudgetMeter` is the per-run
accountant that nodes charge against.
"""

from __future__ import annotations

import threading
import time

from pydantic import BaseModel, ConfigDict


class BudgetExceeded(Exception):
    """Raised when a run crosses one of its declared hard limits."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Budget(BaseModel):
    """Declarative hard limits for one graph run. `None` means unlimited."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_concurrency: int | None = None


class BudgetMeter:
    """Thread-safe per-run usage accountant.

    Nodes (or the runtime's node wrapper) call `charge_*` as work happens and
    `check()` before doing more work. Charging never raises; only `check()`
    does, so usage numbers stay accurate even for the call that crossed the line.
    """

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self._lock = threading.Lock()
        self._iterations = 0
        self._tokens = 0
        self._started_at = time.monotonic()

    def charge_iteration(self, n: int = 1) -> None:
        with self._lock:
            self._iterations += n

    def charge_tokens(self, n: int) -> None:
        with self._lock:
            self._tokens += n

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def exceeded(self) -> str | None:
        """Return the first exceeded-limit description, or None if within budget."""
        b = self.budget
        if b.max_iterations is not None and self._iterations >= b.max_iterations:
            return f"max_iterations reached ({self._iterations}/{b.max_iterations})"
        if b.max_tokens is not None and self._tokens >= b.max_tokens:
            return f"max_tokens reached ({self._tokens}/{b.max_tokens})"
        if b.max_seconds is not None and self.elapsed_seconds >= b.max_seconds:
            return f"max_seconds reached ({self.elapsed_seconds:.1f}s/{b.max_seconds}s)"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason is not None:
            raise BudgetExceeded(reason)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "iterations": self._iterations,
            "tokens": self._tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
