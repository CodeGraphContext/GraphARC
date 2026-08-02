"""Bounded work: every graph run carries hard limits on iterations, tokens, and wall-clock time.

The graph-engineering production checklist requires "bounded work": concurrency,
iterations, tokens, and time all need hard limits so a cycling graph cannot burn
money forever. `Budget` declares the limits; `BudgetMeter` is the per-run
accountant that nodes charge against.

Time is the one limit that cannot be enforced by looking at a counter between
nodes — a node that runs long is exactly the case a wall-clock ceiling exists
for. `deadline_guard` turns `max_seconds` into an interrupt delivered *into* the
running node; read its docstring for what that does and does not guarantee.
"""

from __future__ import annotations

import contextvars
import ctypes
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict


class BudgetExceeded(Exception):
    """Raised when a run crosses one of its declared hard limits."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NodeDeadlineExceeded(BudgetExceeded):
    """`max_seconds` ran out while a node was still executing.

    The default reason exists because CPython's async-exception slot can only
    carry an exception *class*: the watchdog hands over the type and
    `deadline_guard` re-raises with the run's actual numbers.
    """

    def __init__(self, reason: str = "max_seconds reached while a node was running") -> None:
        super().__init__(reason)


class Budget(BaseModel):
    """Declarative hard limits for one graph run. `None` means unlimited."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_concurrency: int | None = None


def _call_key(source: Any) -> Any:
    """Identify the model call `source` came from.

    langchain stamps every generated message with an `id` derived from the run
    that produced it (`lc_run--<uuid>-<n>`), so the id survives a copy of the
    message and still names one specific call. Anything without a usable string
    id falls back to object identity; the ledger keeps such objects alive for
    the scope so an address cannot be recycled under it.
    """
    ident = getattr(source, "id", None)
    if isinstance(ident, str) and ident:
        return ("id", ident)
    return ("obj", id(source))


class _MeteredCalls:
    """The model calls the usage callback charged in one node scope, by identity.

    Not a count of tokens: a re-report is recognised because it names the same
    call, never because it happens to be the same number of tokens.
    """

    __slots__ = ("_alive", "_unclaimed")

    def __init__(self) -> None:
        self._unclaimed: dict[Any, int] = {}
        self._alive: list[Any] = []

    def record(self, source: Any) -> None:
        key = _call_key(source)
        self._unclaimed[key] = self._unclaimed.get(key, 0) + 1
        if key[0] == "obj":
            # Only the address fallback needs the object kept alive; a run id is
            # a value, so holding the message would just pin a node's history.
            self._alive.append(source)

    def claim(self, source: Any) -> bool:
        """True when `source` names a metered call that has not been re-reported yet."""
        key = _call_key(source)
        remaining = self._unclaimed.get(key, 0)
        if not remaining:
            return False
        self._unclaimed[key] = remaining - 1
        return True


class _NodeScope:
    """One node execution's re-report ledger and its own share of the spend.

    `tokens` exists because the run's meter is shared: subtracting a
    before/after reading off it attributed every *sibling's* concurrent spend to
    whichever fan-out worker happened to be running at the time. This counts only
    the charges made inside this scope, so the same work costs the same whether
    it runs in parallel or one at a time.
    """

    __slots__ = ("ledger", "tokens")

    def __init__(self) -> None:
        self.ledger = _MeteredCalls()
        self.tokens = 0


# The node scope a re-report may claim a metered call from, as (meter, scope).
# Context-scoped rather than thread-scoped: see `BudgetMeter.automatic_scope`.
_METERED_SCOPE: contextvars.ContextVar[tuple[Any, _NodeScope] | None] = (
    contextvars.ContextVar("grapharc_metered_scope", default=None)
)


class BudgetMeter:
    """Thread-safe per-run usage accountant.

    Nodes (or the runtime's node wrapper) call `charge_*` as work happens and
    `check()` before doing more work. Charging never raises; only `check()` and
    `check_tokens()` do, so usage numbers stay accurate even for the call that
    crossed the line.
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

    def credit_seconds(self, seconds: float) -> None:
        """Exclude a stretch of wall clock from `max_seconds`.

        For time the run spent deliberately parked — a human deciding at an
        approval gate — which is nobody's spend. Implemented by moving the
        start mark forward, so `elapsed_seconds` simply never saw the wait.
        Negative or zero credits are ignored rather than trusted.
        """
        if seconds <= 0:
            return
        with self._lock:
            self._started_at += seconds

    def charge_tokens(self, n: int, *, automatic: bool = False, source: Any = None) -> None:
        """Add `n` tokens to the run's total.

        `automatic=True` marks a charge raised by the runtime's usage callback
        (see `grapharc.runtime.usage`), which passes the message it metered as
        `source`. A later manual charge that names the *same* call — same
        `source`, i.e. the same LLM run / message — is the node re-reporting
        spend already counted, and is dropped exactly once. Hand-metering stays
        legal without paying twice, and the match is by identity, so two calls
        of identical size are two distinct calls.

        **A manual charge that does not name its source is always counted**,
        even when it equals a metered call to the token. An integer carries no
        evidence of where it came from: "the node re-reporting the call we just
        saw" and "spend on a provider the callback cannot see" are the same
        number. The previous ledger guessed by matching integers and swallowed
        the second one, so a real charge could vanish. Over-reporting stops a
        run early and is visible in `snapshot()`; under-reporting is invisible
        and arrives on the bill. Pass `source=` to be exact.

        All three shipped re-reporters now name their source — `testing.
        charge_usage`, `AgentNode._charge_tokens` and `planner.proposal._charge`
        each pass `source=message` — so none of them pays twice, and each still
        works outside a graph, where nothing metered the call and the charge must
        land.

        What the source is matched against is a *node scope*, not a thread; see
        `automatic_scope` for why that distinction was load-bearing.
        """
        with self._lock:
            scope = self._current_scope()
            if automatic:
                self._tokens += n
                self._attribute(scope, n)
                if scope is not None and source is not None:
                    scope.ledger.record(source)
                return
            if source is not None and scope is not None and scope.ledger.claim(source):
                return
            self._tokens += n
            self._attribute(scope, n)

    @staticmethod
    def _attribute(scope: _NodeScope | None, n: int) -> None:
        """Credit `n` to the node scope that spent it. Only charges that actually
        landed on the run total get here, so a dropped re-report is not counted."""
        if scope is not None:
            scope.tokens += n

    def _current_scope(self) -> _NodeScope | None:
        """This node execution's scope, or None outside any. Caller holds the lock."""
        entry = _METERED_SCOPE.get()
        # The meter is part of the key so a scope opened by one meter — the
        # planner runs a sub-meter inside the run's — cannot lend its scope to
        # a charge made against another.
        return entry[1] if entry is not None and entry[0] is self else None

    def scope_tokens(self) -> int | None:
        """Tokens charged inside the current node scope, or None outside one.

        What a single node execution actually spent, as opposed to how the run's
        shared total moved while it happened to be running.
        """
        with self._lock:
            scope = self._current_scope()
            return None if scope is None else scope.tokens

    @contextmanager
    def automatic_scope(self) -> Iterator[None]:
        """Bound the window in which a metered call may be re-reported for free.

        Scoped to the calling *context* and opened once per node, so a re-report
        in one node can never claim a call metered by another.

        A `contextvars` scope rather than a thread-keyed one, because the runtime's
        usage callback does not always run on the thread the node body runs on.
        `on_llm_end` is sync, so under `ainvoke`/`astream` LangChain dispatches it
        to a worker thread while the body stays on the event loop. Keyed by thread
        ident, the automatic charge then found no ledger, never recorded the call,
        and the node's *named* re-report — the documented free path — was charged a
        second time: every `async def` node using `charge_usage`, `AgentNode.
        _charge_tokens` or `planner.proposal._charge` reported double its real
        spend and hit `max_tokens` at half its declared allowance. LangChain copies
        the context across that hop, so the callback and the body share one ledger.

        Contexts also nest properly. `reset(token)` restores an enclosing scope's
        ledger, where popping a thread-keyed entry discarded it — so an inner scope
        used to make the outer node's remaining re-reports pay twice.
        """
        token = _METERED_SCOPE.set((self, _NodeScope()))
        try:
            yield
        finally:
            _METERED_SCOPE.reset(token)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def remaining_seconds(self) -> float | None:
        """Seconds left before `max_seconds`, or None when time is unlimited."""
        if self.budget.max_seconds is None:
            return None
        return self.budget.max_seconds - self.elapsed_seconds

    def tokens_exceeded(self) -> str | None:
        b = self.budget
        if b.max_tokens is not None and self._tokens >= b.max_tokens:
            return f"max_tokens reached ({self._tokens}/{b.max_tokens})"
        return None

    def exceeded(self) -> str | None:
        """Return the first exceeded-limit description, or None if within budget."""
        b = self.budget
        if b.max_iterations is not None and self._iterations >= b.max_iterations:
            return f"max_iterations reached ({self._iterations}/{b.max_iterations})"
        tokens = self.tokens_exceeded()
        if tokens is not None:
            return tokens
        if b.max_seconds is not None and self.elapsed_seconds >= b.max_seconds:
            return f"max_seconds reached ({self.elapsed_seconds:.1f}s/{b.max_seconds}s)"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason is not None:
            raise BudgetExceeded(reason)

    def check_tokens(self) -> None:
        """Enforce only the token ceiling.

        Called from two places, and the node boundary is *not* the earlier one:
        `grapharc.runtime.usage.MeterCallbackHandler` calls this from
        `on_llm_end`, so the call that crosses the line is the last one a node
        gets to make. Overspend is bounded by one model call rather than by
        whatever a node does before it returns — a node making fifty scripted
        calls under `Budget(max_tokens=10)` used to spend all fifty.

        The node wrapper calls it again on the way out, which still matters:
        spend charged by hand, or by a provider whose usage never reaches the
        callback, only shows up there.
        """
        reason = self.tokens_exceeded()
        if reason is not None:
            raise BudgetExceeded(reason)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "iterations": self._iterations,
            "tokens": self._tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# -- wall-clock enforcement ------------------------------------------------

# Only one guard at a time may own the process-wide SIGALRM slot.
_SIGNAL_SLOT = threading.Lock()

# How often the interrupt is re-delivered after the deadline passes. One
# interrupt is not a ceiling: a node that catches it — `except BaseException`
# inside a loop is enough — used to carry on running with nothing left to stop
# it. Re-arming means swallowing has to succeed every 50ms, forever, to keep the
# node alive; the cost while a node is being torn down is one timer per 50ms.
_REARM_SECONDS = 0.05

# The longest delay both mechanisms can actually be armed with. `setitimer`
# raises `OverflowError` past the platform's `time_t` (~2**31 seconds), and
# `threading.Timer` accepts a larger value but crashes its own thread once the
# underlying `wait` exceeds `threading.TIMEOUT_MAX`. A `max_seconds` beyond this
# is ~68 years, which no process reaches, so clamping the *armed delay* costs no
# enforcement: the deadline is still computed from the meter, and the guard's
# exit-time check still refuses a node that overran.
_MAX_ARMABLE_SECONDS = min(2.0**31 - 1, threading.TIMEOUT_MAX)


def _async_raise(thread_id: int, exc: type[BaseException] | None) -> None:
    """Queue `exc` in another thread, or clear a queued one when `exc` is None."""
    payload = ctypes.c_void_p(0) if exc is None else ctypes.py_object(exc)
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), payload)


def _signal_slot_available() -> bool:
    """True when this thread may arm SIGALRM without stealing someone else's alarm."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    delay, interval = signal.getitimer(signal.ITIMER_REAL)
    return not delay and not interval


@contextmanager
def deadline_guard(meter: BudgetMeter, *, what: str) -> Iterator[None]:
    """Interrupt the calling thread when the run's `max_seconds` runs out.

    Two mechanisms, strongest first:

    1. **SIGALRM** when the body runs on the main thread and nothing else has
       armed `ITIMER_REAL`. A signal interrupts blocking syscalls, so a node
       parked in `time.sleep` or waiting on a provider's socket is cut off at
       the deadline, not when the call happens to return.
    2. **`PyThreadState_SetAsyncExc`** otherwise (worker threads, non-Unix,
       an alarm already in use). CPython delivers the exception between
       bytecodes, which stops any Python-level loop promptly.

    Either mechanism keeps firing every `_REARM_SECONDS` until the guard is
    released, so catching the interrupt buys a node one more scheduling quantum,
    not the rest of the run.

    What this does *not* guarantee:

    - Mechanism 2 cannot interrupt a thread parked inside a C call: a
      `time.sleep(60)` is not interrupted mid-call, but the guard still raises
      on exit. This is not a fan-out-only weakness. Mechanism 1 needs `invoke()` to be on the
      process's main thread, so *any* run driven from a worker thread — every
      request handler in a threaded server, every `ThreadPoolExecutor` caller —
      falls back to mechanism 2 for the whole run, nodes and fan-out alike.
    - Neither mechanism unwinds code running outside the interpreter (a blocking
      C extension, a subprocess).
    - A node that catches *every* interrupt and never returns cannot be stopped
      from Python at all. Re-arming makes that require unbroken swallowing
      rather than one lucky `except`, but a `while True` around a bare `except
      BaseException` still hangs the run, and no exit-time check can run because
      the node never exits.
    - Like any asynchronous exception, the interrupt lands wherever the node
      happened to be: it is as safe as Ctrl-C, no safer.

    Short of the never-returns case the ceiling is honoured at the node
    boundary: if the deadline passed — whether the node swallowed the exception
    or no interrupt was ever delivered — this guard raises on exit, so the
    node's writes never reach state.
    """
    remaining = meter.remaining_seconds()
    if remaining is None:
        yield
        return

    def detail() -> str:
        return (
            f"max_seconds reached while {what} was running "
            f"({meter.elapsed_seconds:.1f}s/{meter.budget.max_seconds}s)"
        )

    if remaining <= 0:
        raise NodeDeadlineExceeded(detail())

    state: dict[str, Any] = {"armed": True, "fired": False, "timer": None}
    lock = threading.Lock()
    thread_id = threading.get_ident()
    # What the timers are armed with, as opposed to what the deadline *is*.
    armable = min(remaining, _MAX_ARMABLE_SECONDS)

    def arm_signal() -> Callable[[], None] | None:
        """Arm SIGALRM and return its disarm, or return None to use mechanism 2.

        Arming is undone on failure rather than left half-done. `setitimer`
        rejects a `remaining` beyond the platform's `time_t` — `float("inf")`,
        or a plausible "effectively unlimited" like `1e10` — and it raises
        *after* the handler is installed and the slot is taken. Letting that
        propagate leaked both for the life of the process: every later guard
        found the slot held and silently degraded to mechanism 2, which cannot
        unwind a blocking syscall, and a stray SIGALRM anywhere in the program
        would raise `NodeDeadlineExceeded` citing a finished run's meter.
        """
        if not (_signal_slot_available() and _SIGNAL_SLOT.acquire(blocking=False)):
            return None

        def on_alarm(signum: int, frame: object) -> None:
            state["fired"] = True
            raise NodeDeadlineExceeded(detail())

        previous_handler = signal.signal(signal.SIGALRM, on_alarm)

        def disarm() -> None:
            # An alarm landing mid-disarm raises out of `setitimer`, so the
            # restore has to sit in a `finally` or the handler stays installed.
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous_handler)
                _SIGNAL_SLOT.release()

        try:
            # The third argument is the repeat interval: the kernel re-arms the
            # alarm for us, so a swallowed SIGALRM is followed by another one.
            signal.setitimer(signal.ITIMER_REAL, armable, _REARM_SECONDS)
        except (OverflowError, OSError, ValueError):
            disarm()
            return None
        return disarm

    disarm_signal = arm_signal()

    if disarm_signal is not None:
        disarm = disarm_signal
    else:

        def fire() -> None:
            # Arming and clearing both happen under `lock`, so a guard that has
            # already disarmed can never be handed a late exception to leak into
            # the next node scheduled on this (pooled) thread.
            with lock:
                if not state["armed"]:
                    return
                state["fired"] = True
                _async_raise(thread_id, NodeDeadlineExceeded)
                # Re-arm: one async exception is one chance to be caught.
                rearm(_REARM_SECONDS)

        def rearm(delay: float) -> None:
            timer = threading.Timer(delay, fire)
            timer.daemon = True
            state["timer"] = timer
            timer.start()

        with lock:
            rearm(armable)

        def disarm() -> None:
            with lock:
                state["armed"] = False
                state["timer"].cancel()
                if state["fired"]:
                    _async_raise(thread_id, None)

    try:
        try:
            yield
        finally:
            disarm()
    except NodeDeadlineExceeded as exc:
        raise NodeDeadlineExceeded(detail()) from exc

    # Reached only when the node returned normally. It may have swallowed the
    # interrupt, or the deadline may have passed without the timer firing —
    # the timer thread needs the GIL, which a node inside a long C call
    # withholds until it returns; either way its writes must not land.
    left = meter.remaining_seconds()
    if state["fired"] or (left is not None and left <= 0):
        raise NodeDeadlineExceeded(detail())
