"""Cost ceilings — the half of budgeting the gateway captured but never enforced.

Both backends already report `cost_usd` per call. A `SpendMeter` accumulates
that figure and refuses to keep going past a ceiling, so "this graph may spend
five cents" is a property of the run rather than a hope.

Two enforcement points, deliberately:

- **before** a call, `ensure_headroom()` refuses once the meter is already at
  or over its ceiling — that call never reaches the provider;
- **after** a call, `charge()` records the cost and *then* raises if it crossed
  the line, so overspend is bounded by the single call that crossed it rather
  than discovered a node later. This mirrors how the runtime meters tokens.

Where it stops short, stated plainly: a call whose cost the provider does not
report cannot be charged. Those are counted in `unpriced_calls` instead of
being guessed at, so a ceiling on a backend that reports no cost is not an
enforcement — it is a counter. `unpriced_calls > 0` is the signal that the
ceiling saw less than the whole bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grapharc.gateway.errors import CostCeilingExceeded


@dataclass
class SpendMeter:
    """Cumulative USD spend for one model instance, or for a shared group.

    Mutable and unsynchronized: one meter passed to several model objects gives
    them a shared ceiling, but concurrent calls from multiple threads can
    interleave a check with a charge. The bound is then approximate by at most
    the number of calls in flight.
    """

    ceiling_usd: float | None = None
    spent_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0
    per_model_usd: dict[str, float] = field(default_factory=dict)

    def ensure_headroom(self, *, model: str | None = None) -> None:
        """Refuse before spending when the ceiling is already reached."""
        if self.ceiling_usd is None:
            return
        if self.spent_usd >= self.ceiling_usd:
            raise CostCeilingExceeded(
                f"cost ceiling reached before this call: "
                f"${self.spent_usd:.6f} spent of ${self.ceiling_usd:.6f} "
                f"over {self.calls} call(s)"
                + (f" — model {model!r}" if model else ""),
                spent_usd=self.spent_usd,
                ceiling_usd=self.ceiling_usd,
                model=model,
            )

    def charge(self, cost_usd: float | None, *, model: str | None = None) -> float:
        """Record one call's cost; raise if it took cumulative spend past the ceiling.

        The charge lands before the raise, so `spent_usd` after a
        `CostCeilingExceeded` includes the offending call — a resumed run
        knows what it actually spent.
        """
        self.calls += 1
        if cost_usd is None:
            self.unpriced_calls += 1
            return self.spent_usd
        amount = float(cost_usd)
        self.spent_usd += amount
        if model:
            self.per_model_usd[model] = self.per_model_usd.get(model, 0.0) + amount
        if self.ceiling_usd is not None and self.spent_usd > self.ceiling_usd:
            raise CostCeilingExceeded(
                f"cost ceiling exceeded: ${self.spent_usd:.6f} spent of "
                f"${self.ceiling_usd:.6f} after {self.calls} call(s); "
                f"this call cost ${amount:.6f}"
                + (f" — model {model!r}" if model else ""),
                spent_usd=self.spent_usd,
                ceiling_usd=self.ceiling_usd,
                model=model,
            )
        return self.spent_usd

    def snapshot(self) -> dict[str, object]:
        """A plain dict for traces and usage envelopes. Never a live reference."""
        return {
            "spent_usd": self.spent_usd,
            "ceiling_usd": self.ceiling_usd,
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "per_model_usd": dict(self.per_model_usd),
        }
