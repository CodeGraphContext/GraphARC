"""Retries with exponential backoff and jitter, and the rule for what to retry.

A model call fails in two very different ways, and treating them alike is how
a gateway either drops work that would have succeeded or burns three times the
quota proving that a 400 is still a 400:

- **transient** — the request never got a verdict (timeout, connection reset)
  or the provider declined to give one *yet* (429, 5xx). Re-issuing the
  identical request can succeed.
- **deterministic** — the provider gave a verdict and it will not change (401,
  403, 400, 402, a content refusal). Re-issuing wastes time and money.

`is_transient` is the whole policy, and it is conservative: an exception has to
present evidence of transience to be retried. Anything unrecognized is treated
as deterministic and raised on the first attempt.

**Backoff.** Delay before attempt *n+1* is `initial * multiplier**(n-1)`,
capped at `max_backoff_seconds`, then multiplied by a random factor in
`[1 - jitter, 1]`. Shrinking rather than centring the jitter keeps successive
delays strictly increasing — the growth is a property of every draw, not of
the average — as long as `jitter < 1 - 1/multiplier`, which the defaults
satisfy. Above that the windows overlap and only the trend survives.

A `Retry-After` header, when the provider sends one, raises that delay (never
lowers it) but is itself capped by `max_backoff_seconds`: a provider asking for
a five-minute pause gets a bounded wait and then a raised error, rather than a
silently parked process.

Not covered, and worth being explicit about: streaming. Once a chunk has been
handed to the caller the request cannot be re-issued transparently, so
`_stream` is not retried by either backend.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from grapharc.gateway.errors import GatewayError, TransientGatewayError

# Status codes worth another attempt. 402 (out of credit), 401/403 (auth) and
# 400/404/422 (a bad request) are absent on purpose — they are verdicts.
# 529 is Anthropic's "overloaded" and falls out of the 5xx range check.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})

# Exception classes from httpx/openai that carry no status code because the
# request never completed. Matched by name across the MRO so neither library
# has to be importable for the rule to hold.
RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",  # openai: request never reached a verdict
        "APITimeoutError",  # openai
        "TransportError",  # httpx: base of connect/read/write/pool failures
        "TimeoutException",  # httpx
    }
)


def _sleep(seconds: float) -> None:
    """Indirection point: tests replace this instead of the global `time.sleep`."""
    time.sleep(seconds)


async def _asleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to retry a transient failure, and how long to wait.

    `max_attempts` counts the first try, so `max_attempts=1` disables retrying
    and `max_attempts=3` means at most two retries.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 20.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff seconds must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1 or backoff shrinks")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be a fraction in [0, 1]")

    def base_delay(self, attempt: int) -> float:
        """The un-jittered wait after `attempt` (1-based) has failed."""
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        growth = self.initial_backoff_seconds * self.multiplier ** (attempt - 1)
        return min(growth, self.max_backoff_seconds)

    def delay_for(
        self,
        attempt: int,
        *,
        rng: Callable[[], float] | None = None,
        retry_after: float | None = None,
    ) -> float:
        """The actual wait after `attempt` failed, jittered and Retry-After aware."""
        draw = (rng or random.random)()
        delay = self.base_delay(attempt) * (1.0 - self.jitter * draw)
        if retry_after is not None and retry_after > 0:
            delay = max(delay, min(retry_after, self.max_backoff_seconds))
        return delay


DEFAULT_RETRY_POLICY = RetryPolicy()

# A policy that never retries — for callers who want one attempt and the error.
NO_RETRY = RetryPolicy(max_attempts=1)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_transient(exc: BaseException) -> bool:
    """True when re-issuing the identical request could plausibly succeed.

    Deliberately closed rather than open: unrecognized exceptions are *not*
    retried. A missed retry costs one failed call; a wrong retry multiplies a
    deterministic failure by `max_attempts` and can double-charge a side effect.
    """
    if isinstance(exc, TransientGatewayError):
        return True
    if isinstance(exc, GatewayError):
        # Includes CostCeilingExceeded: spending more cannot fix overspending.
        return False
    code = _status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS or 500 <= code < 600
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ in RETRYABLE_EXCEPTION_NAMES for cls in type(exc).__mro__)


def retry_after_seconds(exc: BaseException) -> float | None:
    """The provider's own `Retry-After`, in seconds, when it sent a numeric one.

    HTTP-date form is ignored rather than parsed: the computed backoff is a
    safe fallback, and a misparsed date is not.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except AttributeError:  # pragma: no cover - non-mapping headers
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    is_retryable: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    rng: Callable[[], float] | None = None,
) -> T:
    """Call `fn`, retrying transient failures with backoff. Re-raises the last error.

    The original exception propagates unwrapped, so a caller that already
    handles `GatewayError` or `openai.APIError` keeps working; `on_retry` is
    the hook for recording that attempts happened.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay_for(attempt, rng=rng, retry_after=retry_after_seconds(exc))
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            (sleep or _sleep)(delay)
            attempt += 1


async def acall_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    is_retryable: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], Awaitable[Any]] | None = None,
    rng: Callable[[], float] | None = None,
) -> T:
    """`call_with_retry` for coroutines — same policy, `asyncio.sleep` for waiting."""
    attempt = 1
    while True:
        try:
            return await fn()
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay_for(attempt, rng=rng, retry_after=retry_after_seconds(exc))
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await (sleep or _asleep)(delay)
            attempt += 1
