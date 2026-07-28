"""What the HTTP layer needs from a session runtime, and an in-process default.

`app.py` talks only to `SessionRuntime`. `InProcessRuntime` implements it: each
session gets a worker thread that drives the graph through a disciplined entry
point on its own event loop — `CompiledGraphARC.astream`, or `.stream` for a
graph whose checkpointer has no async side — and every request handler touches
nothing but in-memory bookkeeping.

A thread per session rather than a task on the server's loop, deliberately: the
run must survive the request that started it, and a graph made of ordinary sync
nodes must not be able to stall the server. This is not the async-native design
the ROADMAP eventually wants — it is one that cannot block the request path.

When the real session layer lands (ROADMAP §6), swapping it in means writing a
second class that satisfies `SessionRuntime` and passing it to
`create_app(runtime=...)`. No route changes.

What this default does and does not do, precisely:

- **Runs** the graph through `CompiledGraphARC.astream(stream_mode="values")`
  by default, so the disciplined entry point is used, budgets and traces apply,
  and a graph containing `async def` nodes works (the sync entry points reject
  those).
- **Falls back to `CompiledGraphARC.stream()`** when the graph was compiled
  with a checkpointer that has no async implementation. `SqliteSaver` — this
  repo's own hard dependency, and what `grapharc.session.runtime` uses — is
  exactly that: it raises `NotImplementedError` the moment `astream` reaches
  it, before a single node runs. The saver is probed once per run, on the run's
  own event loop, and a sync-only one selects the sync driver instead of
  failing the session. The one combination that cannot be served is `async def`
  nodes *and* a sync-only saver; that raises `CheckpointerNotAsyncError`, whose
  message names both ways out.
- **Interrupts** by stopping the consumer at a superstep boundary and closing
  the generator. A node already executing runs to completion; the *next* node
  does not start. There is no in-run cancellation below that granularity.
  Identical on both drivers.
- **Does not deliver** `message` or `approval` events into a running graph. It
  records them, and says so in the ack. Delivery is ROADMAP §6.4 / §6.5.
- **Does not resume across process restart.** Sessions live in this process's
  memory; the trace files outlive it, the sessions do not. No checkpointer is
  wired in either — a builder that wants one passes it to `compile()` itself,
  and either a sync or an async saver will run (see the previous point).
- **Does not evict.** Every session, its result and its trace-event list are
  retained for the life of the process. Fine for a bounded workload, not a
  service left running for a week.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from grapharc.observe.trace import TraceEvent, TraceRecorder

# Imported private on purpose: `_jsonable` *is* the on-disk trace format (the
# 2000-char clip and the repr fallback), so reusing it is what keeps a trace
# written here byte-identical to one written by any other TraceRecorder in the
# repo. A local re-implementation would be free to drift from the readers.
from grapharc.observe.trace import _jsonable as _trace_shape
from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import AsyncNodeError, CompiledGraphARC
from grapharc.server.errors import (
    CheckpointerNotAsyncError,
    InvalidGraphInputError,
    RuntimeClosedError,
    UnknownGraphError,
    UnknownSessionError,
)
from grapharc.server.models import (
    TERMINAL_STATUSES,
    CreateSessionRequest,
    EventAck,
    EventPage,
    QueuedEvent,
    SessionEventRequest,
    SessionStatus,
    SessionView,
)
from grapharc.server.registry import GraphRegistry


@runtime_checkable
class SessionRuntime(Protocol):
    """The whole surface `create_app` depends on.

    Deliberately narrow: eight methods, all synchronous, all cheap enough to
    call from a request handler. Implementations raise `UnknownSessionError` /
    `UnknownGraphError` rather than returning `None`, so the HTTP layer maps
    failures in one place.
    """

    def graph_names(self) -> list[str]:
        """Names a `CreateSessionRequest` may ask for."""
        ...

    def create(self, spec: CreateSessionRequest) -> SessionView:
        """Start a session. Raises `UnknownGraphError` for an unregistered graph."""
        ...

    def get(self, session_id: str) -> SessionView:
        """Current status and result. Raises `UnknownSessionError`."""
        ...

    def list_sessions(self) -> list[SessionView]:
        """Every session this runtime knows about, oldest first."""
        ...

    def submit_event(self, session_id: str, request: SessionEventRequest) -> EventAck:
        """Post a message, approval, or interrupt. The ack says whether it applied."""
        ...

    def events_since(self, session_id: str, cursor: int) -> EventPage:
        """Trace events after `cursor`, with the status read in the same critical section."""
        ...

    def trace_text(self, session_id: str) -> str:
        """The session's JSONL trace, empty before the first event."""
        ...

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting sessions and wind down running ones."""
        ...


_log = logging.getLogger("grapharc.server")

#: Thread id the checkpointer probe reads. No run uses it, so the read finds
#: nothing and writes nothing.
_PROBE_THREAD_ID = "__grapharc_async_probe__"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    """Best-effort JSON shaping for a graph's final state.

    Unlike `grapharc.observe.trace`, nothing is truncated: this is the answer
    the caller asked for, and a clipped result would be a wrong result.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _validate_input(
    graph: str, compiled: CompiledGraphARC, graph_input: Mapping[str, Any]
) -> None:
    """Reject input the graph's state schema would not accept, before the run starts.

    A gate, not a transform: the raw mapping is still what reaches the kernel.

    This is stricter than driving the kernel directly, and deliberately so. A
    `GraphARCState` forbids extra fields, but LangGraph drops keys it does not
    recognise *before* the schema sees them, so `{"quesiton": ...}` would start a
    run against an empty question instead of failing. Here the typo is a 422.
    """
    try:
        compiled.arc.state_schema.model_validate(dict(graph_input))
    except ValidationError as exc:
        errors = [
            {**_jsonable(err), "loc": ["body", "input", *(str(p) for p in err["loc"])]}
            for err in exc.errors(include_url=False)
        ]
        raise InvalidGraphInputError(graph, errors) from exc


class _SyncDriverRequired(Exception):
    """Internal signal: this graph's checkpointer cannot be driven by `astream`.

    Raised by `_drive` before anything executes, caught by `_run`, never seen
    by a caller — `reason` becomes part of the message only if the sync driver
    turns out to be unusable too.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def _async_checkpointer_gap(compiled: CompiledGraphARC) -> str | None:
    """Why `astream` cannot drive this graph's checkpointer — `None` if it can.

    A behavioural probe rather than a list of class names: one read of a thread
    id no run uses. `SqliteSaver` raises `NotImplementedError` from it, and so
    does any saver that inherits `BaseCheckpointSaver`'s unimplemented async
    methods — the same failure the run would hit at its first superstep, except
    that here nothing has executed and the caller can still pick another
    driver. Any *other* exception means the saver tried: a missing table or a
    closed connection is the run's problem to report, not a reason to change
    how it is driven.

    Deliberately run on the loop the graph is about to use. An async saver
    binds its connection to the loop that first touches it, so probing on a
    throwaway loop would break the very configuration this check recommends.
    """
    checkpointer = getattr(compiled.inner, "checkpointer", None)
    if not checkpointer:
        return None
    probe = getattr(checkpointer, "aget_tuple", None)
    if probe is None:
        return f"{type(checkpointer).__name__} has no aget_tuple()"
    try:
        await probe({"configurable": {"thread_id": _PROBE_THREAD_ID}})
    except NotImplementedError as exc:
        first = next((line for line in str(exc).splitlines() if line.strip()), "")
        detail = f": {first.strip()}" if first.strip() else ""
        return f"{type(checkpointer).__name__} does not implement async checkpointing{detail}"
    except Exception:
        return None
    return None


def _trace_payload(event: TraceEvent) -> dict[str, Any]:
    """The one record behind a JSONL line and an SSE frame — shaped exactly once.

    `_trace_shape` is not idempotent: it clips a string at 2000 characters and
    appends a marker, so the result is *longer* than the limit and a second
    pass clips it again. Applying it once, here, and using the returned dict
    for both the file and the subscriber is what makes the two views the same
    record rather than two serializations of one event.
    """
    try:
        dumped = event.model_dump(mode="json", exclude_none=True)
    except Exception:
        # `state_delta` is `dict[str, Any]`, and mode="json" refuses a type
        # pydantic has no serializer for. The python dump keeps the object and
        # `_trace_shape` reprs it below — which is what the trace format does
        # with such a value anyway.
        dumped = event.model_dump(exclude_none=True)
    return _trace_shape(dumped)


class BroadcastRecorder(TraceRecorder):
    """A `TraceRecorder` that hands every record it writes to a live subscriber.

    The subscriber is given *the dict that was written*, not a second
    serialization of the same event. That is the whole mechanism behind
    `GET /trace` and `GET /stream` agreeing: one shaping pass, one `json.dumps`,
    one fan-out, so parsing a line of the trace file yields exactly the object
    the SSE frame carried — long values, clipped values and all.

    Both views are clipped, because the trace format clips: a `state_delta`
    string over 2000 characters (an ordinary model answer) is cut short in the
    file and in the frame alike. The unclipped value is the session's `result`,
    which `_jsonable` never truncates.

    `record` appends the line itself rather than delegating to
    `TraceRecorder.record`, and `event` builds the `TraceEvent` itself rather
    than delegating to `TraceRecorder.event`, for the same reason in both
    cases: each of those shapes the payload again, and two passes over a
    2100-character value produced 2022 characters on the wire against 2021 on
    disk — different strings, both claimed to be the same record.

    The sink runs on the node's own thread, after the line is on disk. Its
    exceptions are counted in `sink_errors` and swallowed: a subscriber must
    never be able to turn a recorded trace event into a node failure. A raising
    sink is therefore the one way the two views can diverge — the event is on
    disk and missing from the stream — and `sink_errors` is how you find out.
    """

    def __init__(self, path: str | Path, sink: Callable[[dict[str, Any]], None]) -> None:
        super().__init__(path)
        self._sink = sink
        #: Sink exceptions swallowed so far. Non-zero means the live stream is
        #: missing an event the trace file has.
        self.sink_errors = 0

    def event(
        self,
        *,
        run_id: str,
        graph: str,
        node: str,
        phase: str,
        step: int,
        thread_id: str | None = None,
        attempt: int = 1,
        state_delta: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
        model: str | None = None,
        error: str | None = None,
    ) -> None:
        """The base contract, minus the base's pre-shaping of `state_delta`.

        `TraceRecorder.event` shapes the delta here and `TraceRecorder.record`
        shapes the whole event again on the way out; `record` below does it
        once, for every field, so a long answer is clipped a single time.
        """
        self.record(
            TraceEvent(
                ts=_now(),
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                graph=graph,
                node=node,
                phase=phase,
                step=step,
                state_delta=state_delta,
                duration_ms=duration_ms,
                tokens=tokens,
                cost_usd=cost_usd,
                model=model,
                error=error,
            )
        )

    def record(self, event: TraceEvent) -> None:
        payload = _trace_payload(event)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            self._sink(payload)
        except Exception:
            with self._lock:  # nodes in a fanout record concurrently
                self.sink_errors += 1
            _log.exception(
                "trace subscriber raised; the event is in %s but not on the live stream",
                self.path,
            )


@dataclass
class _Session:
    id: str
    graph: str
    thread_id: str
    input: dict[str, Any]
    budget: Budget | None
    trace_path: Path
    created_at: str
    compiled: CompiledGraphARC | None = None
    status: SessionStatus = "queued"
    started_at: str | None = None
    ended_at: str | None = None
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    usage: dict[str, float] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    inbox: list[QueuedEvent] = field(default_factory=list)
    interrupt_requested: bool = False
    future: Future[None] | None = None


class InProcessRuntime:
    """Runs each session on a worker thread in this process.

    One `threading.RLock` guards all session bookkeeping. Every critical
    section is a handful of dict and list operations, so a request handler
    never waits on graph execution — the worker thread holds the lock only
    between supersteps, never while a node runs.

    `max_workers` is the real concurrency ceiling: sessions beyond it stay
    `queued` until a worker frees up. `root` is where per-session trace files
    go, defaulting to a fresh `mkdtemp` directory that is *not* cleaned up —
    traces are the audit trail and are meant to outlive the run.

    Note for anyone driving graphs from here: `grapharc.runtime.budget`'s
    `max_seconds` falls back to an asynchronous exception on worker threads
    (SIGALRM needs the main thread), so a node blocked inside a C call is not
    cut off until that call returns. That limit is the kernel's, and it applies
    to any threaded server, not just this one.
    """

    def __init__(
        self,
        registry: GraphRegistry | None = None,
        *,
        root: str | Path | None = None,
        max_workers: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.registry = registry if registry is not None else GraphRegistry()
        if root is None:
            root = tempfile.mkdtemp(prefix="grapharc-server-")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._order: list[str] = []
        self._closed = False
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="grapharc-session"
        )

    # -- SessionRuntime ----------------------------------------------------

    def graph_names(self) -> list[str]:
        return self.registry.names()

    def create(self, spec: CreateSessionRequest) -> SessionView:
        with self._lock:
            if self._closed:
                raise RuntimeClosedError("runtime is shut down; no new sessions")
        if spec.graph not in self.registry:
            # Checked before anything is allocated, so a bad name leaves no
            # half-built session and no trace directory behind.
            raise UnknownGraphError(spec.graph, self.registry.names())

        session_id = uuid.uuid4().hex[:16]
        session = _Session(
            id=session_id,
            graph=spec.graph,
            thread_id=spec.thread_id or session_id,
            input=dict(spec.input),
            budget=spec.budget.to_budget() if spec.budget is not None else None,
            trace_path=self.root / session_id / "trace.jsonl",
            created_at=_now(),
        )
        recorder = BroadcastRecorder(
            session.trace_path, lambda event: self._on_trace_event(session_id, event)
        )
        try:
            session.compiled = self.registry.build(spec.graph, recorder)
            _validate_input(spec.graph, session.compiled, session.input)
        except BaseException:
            # The recorder created the session's directory when it was built. A
            # request that never becomes a session must not leave one behind;
            # nothing has run, so it is still empty and `rmdir` is safe. This
            # covers a builder that raises as well as input the schema refuses
            # — an operator's builder failing is exactly when an orphan
            # directory per attempt is least welcome.
            with suppress(OSError):
                session.trace_path.parent.rmdir()
            raise

        with self._lock:
            if self._closed:
                # shutdown() can land between the check above and here; submitting
                # to a stopped pool would surface as a 500 instead of a 503.
                raise RuntimeClosedError("runtime is shut down; no new sessions")
            self._sessions[session_id] = session
            self._order.append(session_id)
            session.future = self._pool.submit(self._run, session_id)
            return self._view(session)

    def get(self, session_id: str) -> SessionView:
        with self._lock:
            return self._view(self._require(session_id))

    def list_sessions(self) -> list[SessionView]:
        with self._lock:
            return [self._view(self._sessions[sid]) for sid in self._order]

    def submit_event(self, session_id: str, request: SessionEventRequest) -> EventAck:
        with self._lock:
            session = self._require(session_id)
            applied, detail = self._apply_event(session, request)
            event = QueuedEvent(
                id=uuid.uuid4().hex[:12],
                type=request.type,
                data=dict(request.data),
                received_at=_now(),
                applied=applied,
                detail=detail,
            )
            session.inbox.append(event)
            return EventAck(
                session_id=session_id, event=event, session_status=session.status
            )

    def events_since(self, session_id: str, cursor: int) -> EventPage:
        with self._lock:
            session = self._require(session_id)
            start = max(0, cursor)
            return EventPage(
                events=list(session.events[start:]),
                cursor=len(session.events),
                status=session.status,
                terminal=session.status in TERMINAL_STATUSES,
            )

    def trace_text(self, session_id: str) -> str:
        with self._lock:
            path = self._require(session_id).trace_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting sessions, ask running ones to halt, drain the pool.

        With `wait=True` this blocks for as long as the currently executing
        nodes take: the interrupt is honoured at the *next* superstep, so a node
        that never returns holds shutdown open. That is the same ceiling
        `max_seconds` has, and this method does not add one.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for session in self._sessions.values():
                if session.status not in TERMINAL_STATUSES:
                    session.interrupt_requested = True
        # cancel_futures drops sessions that never got a worker; the ones that
        # did stop at their next superstep boundary because of the flag above.
        self._pool.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            for session in self._sessions.values():
                if session.status not in TERMINAL_STATUSES:
                    session.error = session.error or "runtime shut down before the run finished"
                    self._finish(session, "interrupted")

    # -- internals ---------------------------------------------------------

    def _require(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise UnknownSessionError(session_id) from None

    def _apply_event(
        self, session: _Session, request: SessionEventRequest
    ) -> tuple[bool, str]:
        """Return (applied, detail). Caller holds the lock."""
        if request.type == "interrupt":
            if session.status in TERMINAL_STATUSES:
                return False, f"ignored: session already {session.status}"
            session.interrupt_requested = True
            return True, "stop requested; takes effect at the next node boundary"
        return False, (
            f"recorded: this runtime does not deliver {request.type!r} events into a "
            "running graph (ROADMAP §6.4 event queue / §6.5 approval node)"
        )

    def _on_trace_event(self, session_id: str, payload: dict[str, Any]) -> None:
        """Fan one written record out to the pollers. Runs on the node's thread.

        `payload` is the dict `BroadcastRecorder` just wrote to the trace file —
        already shaped, already JSON-safe. Nothing is serialized here on
        purpose: `model_dump(mode="json")` raises `PydanticSerializationError`
        on a state value pydantic has no serializer for, and doing that on this
        thread turned a trace event that was already on disk into a node
        failure. Shaping now happens once, before the write, where a failure
        stops the record rather than the run.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.events.append(payload)

    def _run(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions[session_id]
            if session.interrupt_requested:
                session.started_at = _now()
                self._finish(session, "interrupted")
                return
            session.status = "running"
            session.started_at = _now()
            compiled = session.compiled
            graph_input, budget, thread_id = session.input, session.budget, session.thread_id

        assert compiled is not None  # set in create() before the future is submitted
        last_state: dict[str, Any] | None = None
        stopped = False
        error: str | None = None

        try:
            try:
                last_state, stopped = asyncio.run(
                    self._drive(session, compiled, graph_input, budget, thread_id)
                )
            except _SyncDriverRequired as need:
                # The probe fired before the first superstep, so no node ran,
                # no checkpoint was written and no trace event was recorded:
                # driving the same input again on the sync path is a first
                # attempt, not a retry.
                last_state, stopped = self._drive_sync(
                    session, compiled, graph_input, budget, thread_id, reason=need.reason
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            if compiled.last_run is not None:
                session.run_id = session.run_id or compiled.last_run.run_id
                session.usage = dict(compiled.last_run.meter.snapshot())
            if error is not None:
                session.error = error
                self._finish(session, "failed")
            else:
                session.result = _jsonable(last_state) if last_state is not None else None
                self._finish(session, "interrupted" if stopped else "succeeded")

    async def _drive(
        self,
        session: _Session,
        compiled: CompiledGraphARC,
        graph_input: dict[str, Any],
        budget: Budget | None,
        thread_id: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Consume one run, returning (last state, stopped-by-interrupt).

        `astream` rather than `stream` so a graph containing `async def` nodes
        is runnable: the sync entry points reject those outright, and LangGraph
        puts sync nodes on its own executor either way.

        Unless the graph's checkpointer has no async side, in which case
        `astream` would die at the first superstep with a `NotImplementedError`
        from deep inside the saver. `_async_checkpointer_gap` establishes that
        here, before anything runs, and `_SyncDriverRequired` sends the run to
        `_drive_sync`.

        `self._lock` is a plain threading lock taken on this loop's thread. That
        is safe only because every critical section under it is a few dict and
        list operations — nothing here may ever block while holding it.
        """
        gap = await _async_checkpointer_gap(compiled)
        if gap is not None:
            raise _SyncDriverRequired(gap)

        last_state: dict[str, Any] | None = None
        stopped = False
        stream = compiled.astream(
            graph_input, thread_id=thread_id, budget=budget, stream_mode="values"
        )
        try:
            async for chunk in stream:
                if isinstance(chunk, dict):
                    last_state = chunk
                with self._lock:
                    if session.run_id is None and compiled.last_run is not None:
                        session.run_id = compiled.last_run.run_id
                    if session.interrupt_requested:
                        stopped = True
                        break
        finally:
            # Closing throws GeneratorExit into the kernel's generator, so a
            # broken-out-of run does not sit half-open until GC.
            await stream.aclose()
        return last_state, stopped

    def _drive_sync(
        self,
        session: _Session,
        compiled: CompiledGraphARC,
        graph_input: dict[str, Any],
        budget: Budget | None,
        thread_id: str,
        *,
        reason: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """`_drive` for a graph whose checkpointer only has a sync side.

        Same disciplined entry point, same interrupt granularity, same
        bookkeeping — `stream()` instead of `astream()`, on this worker thread,
        with no event loop for the saver to trip over. Budget semantics do not
        change: a sync node is guarded by `deadline_guard` either way, and this
        thread is not the main thread on either driver.

        A graph of `async def` nodes cannot come this way — `stream()` refuses
        it before the first node — and that refusal is the one configuration
        neither driver can serve, so it is re-raised as a
        `CheckpointerNotAsyncError` that names both fixes instead of an
        `AsyncNodeError` that names neither.
        """
        last_state: dict[str, Any] | None = None
        stopped = False
        stream = compiled.stream(
            graph_input, thread_id=thread_id, budget=budget, stream_mode="values"
        )
        try:
            for chunk in stream:
                if isinstance(chunk, dict):
                    last_state = chunk
                with self._lock:
                    if session.run_id is None and compiled.last_run is not None:
                        session.run_id = compiled.last_run.run_id
                    if session.interrupt_requested:
                        stopped = True
                        break
        except AsyncNodeError as exc:
            raise CheckpointerNotAsyncError(compiled.arc.name, reason, str(exc)) from exc
        finally:
            stream.close()
        return last_state, stopped

    def _finish(self, session: _Session, status: SessionStatus) -> None:
        """Caller holds the lock. Ordering matters: no trace event may follow this."""
        session.status = status
        session.ended_at = _now()

    def _view(self, session: _Session) -> SessionView:
        usage = session.usage
        if usage is None and session.compiled is not None and session.compiled.last_run is not None:
            usage = dict(session.compiled.last_run.meter.snapshot())
        return SessionView(
            id=session.id,
            graph=session.graph,
            thread_id=session.thread_id,
            status=session.status,
            created_at=session.created_at,
            started_at=session.started_at,
            ended_at=session.ended_at,
            run_id=session.run_id,
            result=session.result,
            error=session.error,
            usage=usage,
            event_count=len(session.events),
            inbox=list(session.inbox),
        )


__all__ = [
    "BroadcastRecorder",
    "InProcessRuntime",
    "SessionRuntime",
]
