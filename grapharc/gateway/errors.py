"""Gateway exception hierarchy.

Lives in its own module so `spend` and `resilience` can raise gateway errors
without importing a backend, and so a backend can import them without a cycle.
`GatewayError` is re-exported from `grapharc.gateway.claude_cli`, where it was
originally defined, because callers already import it from there.

The split that matters is `TransientGatewayError`: it is the *only* way a
backend tells the retry driver that re-issuing the same request could plausibly
succeed. Everything else — auth, a malformed request, a refusal — is a
deterministic failure and is raised as a plain `GatewayError`, which the driver
re-raises on the first attempt.
"""

from __future__ import annotations


class GatewayError(Exception):
    """A model backend failed or returned something unusable."""


class TransientGatewayError(GatewayError):
    """A failure that re-issuing the identical request could plausibly fix.

    Raise this only for timeouts, rate limits, and server-side faults. Raising
    it for a deterministic failure turns one wasted call into `max_attempts`.
    """


class CostCeilingExceeded(GatewayError):
    """Recorded spend passed the configured ceiling.

    Carries the numbers rather than only a message so a caller can decide
    whether to raise the ceiling or stop. Never retried: a second identical
    call would cost more and fail the same check.
    """

    def __init__(
        self,
        message: str,
        *,
        spent_usd: float,
        ceiling_usd: float,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        self.model = model
