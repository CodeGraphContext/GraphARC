"""GraphARC: LangGraph with graph-engineering discipline bolted on.

A thin wrapper over `langgraph.graph.StateGraph` that enforces what plain
LangGraph leaves to convention:

- **Typed edges** — state schemas are Pydantic models; every value a node
  returns is validated against the field's declared type before it is written.
  The same schema governs both other boundaries: an entry point's input keys are
  checked rather than silently filtered down to the channels LangGraph knows,
  and a fan-out worker's declared `input_schema` is enforced on the `Send`
  payload it is handed.
- **Write permissions** — every node declares which state fields it may write;
  an undeclared write raises instead of flowing downstream. A node may return a
  plain dict or a `langgraph.types.Command`; the command's `update` is checked
  exactly like a dict, so dynamic `goto` routing costs nothing in discipline.
- **Checked routing** — a `Command(goto=...)` (and a fan-out `Send`) must name a
  node this graph actually has, or `END`. LangGraph drops an unrecognised
  destination with a log line and carries on; GraphARC raises
  `GraphRoutingError`, because a transition that silently does *not* happen is
  the same defect as one that happens unpermitted.
- **Bounded work** — a per-run `BudgetMeter` is checked before every node, model
  calls charge it automatically, and `max_seconds` interrupts the running node;
  the hard ceiling cannot be routed around.
- **Traces** — every node execution is recorded (start/end/error, state delta,
  duration, tokens) as JSONL replay points.
- **DAG mode** — `dag=True` rejects conditional edges and cycles at compile
  time (roadmap Stage 0: earn cycles, don't start with them).

Sync and async nodes get the same treatment. An `async def` node is wrapped in
an async wrapper and driven through `ainvoke()` / `astream()`; the write
allowlist, deep-copy isolation, budget metering, usage callback and traces are
the same code, and the wall-clock ceiling switches to task cancellation because
signalling a shared event-loop thread would land in the wrong coroutine.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from enum import Enum
from typing import Any, Literal, get_args, get_origin, get_type_hints

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, StateSnapshot
from pydantic import BaseModel, TypeAdapter, ValidationError

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import (
    Budget,
    BudgetExceeded,
    BudgetMeter,
    NodeDeadlineExceeded,
    deadline_guard,
)
from grapharc.runtime.usage import charging


class WritePermissionError(Exception):
    """A node wrote a state field it never declared."""


class StateTypeError(Exception):
    """A value contradicted the type its schema declares for it.

    Two boundaries raise this, both for the same reason — a declared type is the
    contract, not documentation:

    - **A node's write.** LangGraph drops unknown keys before validating an
      update, so a declared field carrying the wrong *value* used to sail
      through to the graph's output. GraphARC validates each write against the
      schema at the node boundary instead.
    - **A fan-out `Send` payload.** A worker declaring `input_schema` is stating
      what it is handed; LangGraph passes `Send.arg` to the node untouched, so a
      payload of the wrong class used to surface deep inside the worker as an
      `AttributeError` — or not at all, if the shapes happened to overlap.
      GraphARC checks it at dispatch, next to where `Send.node` is checked.
    """


class GraphCycleError(Exception):
    """A cycle was found in a graph compiled with dag=True."""


class GraphRoutingError(Exception):
    """A node asked the graph to route somewhere the graph cannot reach.

    `Command(goto=...)` is resolved against the compiled graph *after* the node
    returns, and LangGraph drops whatever it does not recognise rather than
    complaining:

    - an unknown node name becomes a write to a channel that does not exist
      (`Task <node> ... wrote to unknown channel branch:to:<x>, ignoring it`);
    - an unknown `Send` target is `Ignoring unknown node name <x> in pending
      sends`;
    - a destination that is neither a string nor a `Send` — a nested list, say —
      is skipped without a word at all.

    All three are log lines, not errors, and the run continues as though the
    node had never routed. A transition that silently does *not* execute is the
    same failure as one that executes unpermitted: the graph did something other
    than what the node asked for, and nobody was told. So GraphARC checks every
    destination at the node boundary, exactly where it checks writes, and fails
    closed.

    Scope, stated rather than implied: this covers destinations GraphARC itself
    hands to LangGraph — a `Command` returned by a node and the `Send`s an
    `add_fanout_edge` dispatcher produces. A `Command` passed as *input* to
    `invoke()`/`stream()` is outside it; those entry points take a dict, a state
    model or `None`, and a `Command` there is unsupported (its `update` would
    reach state unchecked too).
    """


class AsyncNodeError(Exception):
    """A graph holding `async def` nodes was driven through a sync entry point.

    Raised before anything executes, so a graph that mixes sync and async nodes
    cannot half-run — LangGraph would otherwise execute every sync node first
    and fail with a `TypeError` at the first coroutine.
    """


class MissingRunContextError(Exception):
    """A node executed without GraphARC's run context.

    This happens when a compiled graph is driven through raw LangGraph entry
    points (`.inner.stream()`, `.inner.ainvoke()`, ...) instead of
    `CompiledGraphARC.invoke()`/`.stream()`/`.ainvoke()`/`.astream()`. GraphARC
    fails closed: budgets and traces are part of the execution contract, so
    running without them is an error, not a silent downgrade.
    """


class RunContext:
    """Per-run bookkeeping shared with every node via the runnable config."""

    def __init__(
        self,
        *,
        run_id: str,
        graph: str,
        meter: BudgetMeter,
        thread_id: str | None = None,
        attempt: int = 1,
        step_seed: int = 0,
    ) -> None:
        self.run_id = run_id
        self.graph = graph
        self.meter = meter
        self.thread_id = thread_id
        self.attempt = attempt
        self._step = step_seed
        self._lock = threading.Lock()

    def next_step(self) -> int:
        with self._lock:
            self._step += 1
            return self._step


_NOOP_BUDGET = Budget()

# How often an async node's cancellation is re-delivered after the deadline
# passes. Mirrors `budget._REARM_SECONDS`: one cancellation is one chance to be
# caught, and `except BaseException` around an await is enough to catch it.
_ASYNC_REARM_SECONDS = 0.05


def _is_async_callable(fn: Any) -> bool:
    """True for `async def` functions and objects with an `async def __call__`."""
    return inspect.iscoroutinefunction(fn) or (
        callable(fn) and inspect.iscoroutinefunction(fn.__call__)
    )


@asynccontextmanager
async def _async_deadline(meter: BudgetMeter, *, what: str) -> AsyncIterator[None]:
    """Enforce the run's `max_seconds` on a coroutine by cancelling its task.

    `deadline_guard` — the sync mechanism — must not be used here. Both of its
    interrupts target a *thread*, and on an event loop one thread carries every
    coroutine: the exception would land in whichever task happened to be running
    when it was delivered, the loop's own machinery included. Cancellation is the
    asyncio-native equivalent and hits exactly the node that overran.

    Like the sync guard, the cancellation is re-armed every
    `_ASYNC_REARM_SECONDS`, so catching one `CancelledError` buys the node
    another scheduling quantum rather than the rest of the run. Cancellations
    this guard requested are handed back with `Task.uncancel()` on the way out,
    so an undelivered one cannot leak into whatever the task does next.

    What this does *not* guarantee:

    - Cancellation is delivered at an `await`. A coroutine doing blocking work —
      CPU-bound Python, `time.sleep`, a synchronous socket read — is not
      interrupted until it yields to the loop. This is the async analogue of the
      sync guard's inability to unwind a thread parked in a C call.
    - A node that swallows every cancellation and never awaits again cannot be
      stopped from here at all.
    - A cancellation raised by the *caller* is re-raised untouched; only the
      ones this guard requested become `NodeDeadlineExceeded`.

    Short of the never-returns case the ceiling holds at the node boundary: if
    the deadline passed, this raises on exit and the node's writes never reach
    state.
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

    task = asyncio.current_task()
    state: dict[str, Any] = {"fired": 0, "armed": True, "handle": None}
    # A task that was already being cancelled when we arrived: only cancellations
    # above this line are ours to convert and to hand back.
    baseline = task.cancelling() if task is not None else 0

    if task is not None:
        loop = asyncio.get_running_loop()

        def fire() -> None:
            if not state["armed"]:
                return
            state["fired"] += 1
            task.cancel()
            state["handle"] = loop.call_later(_ASYNC_REARM_SECONDS, fire)

        state["handle"] = loop.call_later(remaining, fire)

    def disarm() -> bool:
        """Stop the timer and retract our cancellations; True if the pending one was ours."""
        state["armed"] = False
        if state["handle"] is not None:
            state["handle"].cancel()
        outstanding = baseline
        for _ in range(state["fired"]):
            outstanding = task.uncancel()
        return bool(state["fired"]) and outstanding <= baseline

    try:
        yield
    except asyncio.CancelledError:
        if disarm():
            raise NodeDeadlineExceeded(detail()) from None
        raise
    except BaseException:
        disarm()
        raise
    else:
        disarm()

    # Reached only when the node returned normally. It may have swallowed the
    # cancellation, or the deadline may have passed without the timer running
    # yet; either way its writes must not land.
    left = meter.remaining_seconds()
    if state["fired"] or (left is not None and left <= 0):
        raise NodeDeadlineExceeded(detail())


def _type_label(annotation: Any) -> str:
    """Render an annotation the way an author wrote it: `int`, not `<class 'int'>`."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _short_repr(value: Any, limit: int = 120) -> str:
    # An error message quotes the offending value; a rejected 40k-token string
    # must not become the exception.
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _declared_return_values(router: Callable[..., Any]) -> tuple[Any, ...] | None:
    """The values a router *declares* it returns, or None when it declares none.

    Two annotations say something a mapping can be held against: a `Literal` of
    the event names, and an `Enum` whose members are them. `str`, no annotation
    at all, a forward reference that will not resolve, a callable object with no
    annotations to read — none of those is an opinion about the mapping, so they
    are skipped rather than guessed at. A check that invented a requirement
    would be worse than the gap it closed.
    """
    try:
        annotation = get_type_hints(router).get("return")
    except Exception:  # noqa: BLE001 — an unresolvable annotation is simply no answer
        return None
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(annotation)
    return None


def _in_mapping(key: Any, mapping: dict[str, str]) -> bool:
    """Whether `mapping[key]` would resolve — the lookup LangGraph itself does.

    Hashability is the question, not equality: an unhashable key raises on the
    lookup rather than missing it, and `Enum` hashes by member name while
    `StrEnum` compares by value, so a member can equal a key it does not find.
    Both are answered the way the branch machinery will answer them.
    """
    try:
        return key in mapping
    except TypeError:
        return False


def _first_error(field: str, exc: ValidationError) -> str:
    """Pydantic's first complaint, with the path inside the value it happened at."""
    first = exc.errors()[0]
    path = ".".join(str(part) for part in first["loc"])
    return f"{first['msg']}" + (f" (at {field}.{path})" if path else "")


class GraphARC:
    """Builder for a disciplined graph. Mirrors StateGraph's API where it can."""

    def __init__(
        self,
        state_schema: type[BaseModel],
        *,
        name: str,
        dag: bool = False,
        budget: Budget | None = None,
        trace: TraceRecorder | None = None,
    ) -> None:
        self.name = name
        self.dag = dag
        self.state_schema = state_schema
        self.budget = budget
        self.trace = trace
        self._graph = StateGraph(state_schema)
        self._nodes: dict[str, set[str]] = {}
        self._async_nodes: set[str] = set()
        self._static_edges: list[tuple[str, str]] = []
        # Conditional routes are topology too: the mapping's targets are known
        # statically, and a diagram that omitted them would show a branch node
        # with no way out. Fan-out targets are dynamic, so only the source is
        # recorded and a renderer connects the workers that actually ran.
        self._conditional_edges: list[tuple[str, str]] = []
        self._fanout_sources: list[str] = []
        self._input_schemas: dict[str, type[BaseModel] | None] = {}
        self._adapters: dict[str, TypeAdapter[Any]] = {}

    def add_node(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        writes: Iterable[str],
        input_schema: type[BaseModel] | None = None,
    ) -> GraphARC:
        """Register a node. `input_schema` types a fan-out worker's Send payload
        (defaults to the graph state schema).

        A declared `input_schema` is enforced: a `Send` carrying anything that is
        not an instance of it is refused at dispatch with `StateTypeError`, the
        way an unknown `Send.node` is refused with `GraphRoutingError`. Leaving
        `input_schema` unset keeps an untyped payload legal — the worker is then
        saying nothing about what it accepts — but the payload is deep-copied on
        the way in either way, so parallel workers never share one object.
        """
        writes_set = set(writes)
        unknown = writes_set - set(self.state_schema.model_fields)
        if unknown:
            raise WritePermissionError(
                f"node {name!r} declares writes to unknown state fields: {sorted(unknown)}"
            )
        self._nodes[name] = writes_set
        self._input_schemas[name] = input_schema
        for field in sorted(writes_set):
            self._build_adapter(field)
        self._graph.add_node(
            name, self._wrap(name, fn, writes_set), input_schema=input_schema
        )
        return self

    def add_edge(self, start: str, end: str) -> GraphARC:
        self._graph.add_edge(start, end)
        self._static_edges.append((start, end))
        return self

    def add_conditional_edge(
        self,
        source: str,
        router: Callable[[Any], str],
        mapping: dict[str, str],
    ) -> GraphARC:
        """Route on a validated event name returned by deterministic `router` code.

        The mapping is topology, and topology is checked where it is declared:
        an empty mapping is refused, every target must name a node this graph
        has (or `END`), and a router that says what it returns — a `Literal` or
        an `Enum` return annotation — has those members held against the
        mapping's keys. A router that declares nothing is left alone; guessing
        what a function might return is not a check.

        What cannot be settled here is the key an undeclared router actually
        returns at run time. LangGraph resolves it with a plain `mapping[key]`
        lookup inside its branch machinery, so a typo used to surface as a bare
        `KeyError` several frames from the router that caused it. The router is
        wrapped so it raises `GraphRoutingError` instead, naming the node, the
        key and the keys there are — the same failure the kernel raises for
        every other transition it cannot make.
        """
        if self.dag:
            raise GraphCycleError(
                f"graph {self.name!r} is dag=True: conditional edges are not allowed"
            )
        self._check_mapping(source, router, mapping)
        self._graph.add_conditional_edges(
            source, self._checked_router(source, router, mapping), mapping
        )
        self._conditional_edges.extend(
            (source, target) for target in dict.fromkeys(mapping.values())
        )
        return self

    def add_fanout_edge(
        self,
        source: str,
        dispatcher: Callable[[Any], list[tuple[str, BaseModel]]],
    ) -> GraphARC:
        """Explicit parallelism: `dispatcher` returns (worker_node, payload) pairs,
        each dispatched as a parallel Send. Fan-out is a deliberate act — the
        default execution model stays serial."""
        if self.dag:
            raise GraphCycleError(
                f"graph {self.name!r} is dag=True: fan-out edges are not allowed"
            )

        who = f"the fan-out dispatcher on node {source!r}"

        def dispatch(state: Any) -> list[Send]:
            sends = [Send(node, payload) for node, payload in dispatcher(state)]
            # A Send naming a node this graph does not have is dropped by
            # LangGraph with a log line, so the shard it carried would simply
            # never be worked. Fail closed instead.
            for send in sends:
                self._check_goto_target(who, send)
            return sends

        self._graph.add_conditional_edges(source, dispatch)
        self._fanout_sources.append(source)
        return self

    def compile(self, checkpointer: Any = None) -> CompiledGraphARC:
        if self.dag:
            self._assert_acyclic()
        return CompiledGraphARC(self._graph.compile(checkpointer=checkpointer), self)

    # -- internals ---------------------------------------------------------

    def _build_adapter(self, field: str) -> TypeAdapter[Any]:
        """Compile a validator for one state field, at graph-build time.

        `rebuild_annotation()` keeps the field's Pydantic constraints
        (`Annotated[int, Field(gt=0)]`) while a reducer marker such as
        `operator.add` is inert metadata Pydantic ignores. For a reduced field
        that means the *update* is validated as a whole `T`, which is what
        `operator.add` on a list or a number takes.
        """
        adapter = self._adapters.get(field)
        if adapter is None:
            annotation = self.state_schema.model_fields[field].rebuild_annotation()
            try:
                adapter = TypeAdapter(annotation)
            except Exception as exc:
                raise StateTypeError(
                    f"state field {field!r} of {self.state_schema.__name__} declares "
                    f"{annotation!r}, which cannot be validated standalone: {exc}"
                ) from exc
            self._adapters[field] = adapter
        return adapter

    def _validate_writes(self, who: str, result: dict[str, Any]) -> dict[str, Any]:
        """Return `result` with every value validated against its field's type.

        The validated value replaces the raw one: a schema that says `int` must
        mean the graph's output holds an `int`, not whatever coerced to one.
        `who` names the writer in the error — `node 'x'`, or the entry point that
        wrote outside a node.
        """
        validated: dict[str, Any] = {}
        for field, value in result.items():
            try:
                validated[field] = self._build_adapter(field).validate_python(value)
            except ValidationError as exc:
                # The bare annotation, not `rebuild_annotation()`: a reducer marker
                # in the metadata is noise here, and Pydantic's own message already
                # explains a violated constraint.
                declared = self.state_schema.model_fields[field].annotation
                raise StateTypeError(
                    f"{who} wrote {field!r} with a value the state schema "
                    f"rejects: expected {_type_label(declared)}, got "
                    f"{type(value).__name__} ({_short_repr(value)}); "
                    f"{_first_error(field, exc)}"
                ) from exc
        return validated

    def _check_update(
        self, who: str, writes: set[str], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Enforce the write allowlist, then the declared types, on one update."""
        illegal = set(update) - writes
        if illegal:
            raise WritePermissionError(
                f"{who} wrote undeclared fields {sorted(illegal)}; "
                f"declared writes: {sorted(writes)}"
            )
        return self._validate_writes(who, update)

    def _destinations(self) -> str:
        """The destinations this graph can actually route to, for an error message."""
        return ", ".join([*(repr(name) for name in sorted(self._nodes)), "END"])

    def _check_mapping(
        self, source: str, router: Callable[..., Any], mapping: dict[str, str]
    ) -> None:
        """Everything about a conditional edge that is knowable when it is added."""
        who = f"the conditional edge on node {source!r}"
        if not mapping:
            raise GraphRoutingError(
                f"{who} was given an empty mapping, so there is no key its router "
                f"could return that leads anywhere; LangGraph would accept the edge "
                f"and fail on the first branch taken. Valid destinations: "
                f"{self._destinations()}"
            )
        unreachable = [
            (key, target)
            for key, target in mapping.items()
            if not (target == END or target in self._nodes)
        ]
        if unreachable:
            pairs = ", ".join(
                f"{_short_repr(key)} -> {_short_repr(target)}" for key, target in unreachable
            )
            raise GraphRoutingError(
                f"{who} maps to {'a destination' if len(unreachable) == 1 else 'destinations'} "
                f"graph {self.name!r} does not have: {pairs}. LangGraph resolves a branch "
                f"target when a run reaches it, so this would surface mid-run rather than "
                f"here. Valid destinations: {self._destinations()}"
            )
        declared = _declared_return_values(router) or ()
        unmapped = [value for value in declared if not _in_mapping(value, mapping)]
        if unmapped:
            raise GraphRoutingError(
                f"{who} has a router declaring it returns "
                f"{', '.join(_short_repr(value) for value in declared)}, but "
                f"{', '.join(_short_repr(value) for value in unmapped)} "
                f"{'is' if len(unmapped) == 1 else 'are'} not "
                f"{'a key' if len(unmapped) == 1 else 'keys'} of its mapping; a return "
                f"the mapping has no entry for is a branch the run cannot take. Keys: "
                f"{', '.join(_short_repr(key) for key in mapping)}"
            )

    def _checked_router(
        self, source: str, router: Callable[..., Any], mapping: dict[str, str]
    ) -> Callable[..., Any]:
        """Wrap `router` so an unmapped return is a GraphARC error, not a `KeyError`.

        `functools.wraps` is not decoration here: LangGraph names the branch
        after the callable's `__name__` and reads its annotations to infer the
        branch's input schema, so an anonymous wrapper would quietly rename the
        branch and drop that inference. The signature is `*args`/`**kwargs` for
        the same reason — `wraps` makes `inspect.signature` report the router's
        own parameters, and LangGraph passes `config` to a router that asks for
        one.
        """
        who = f"the router on node {source!r}"

        def check(result: Any) -> Any:
            # LangGraph accepts one key or a sequence of them, and lets a `Send`
            # through to its own dispatch untouched; every other element is
            # resolved against the mapping, which is where the KeyError is.
            for key in result if isinstance(result, (list, tuple)) else [result]:
                if isinstance(key, Send):
                    continue
                if not _in_mapping(key, mapping):
                    raise GraphRoutingError(
                        f"{who} returned {_short_repr(key)}, which is not a key of the "
                        f"mapping the edge was declared with; LangGraph looks a branch "
                        f"key up in that mapping, so this would surface as a bare "
                        f"KeyError from inside the branch machinery instead of naming "
                        f"the router that produced it. Keys: "
                        f"{', '.join(_short_repr(k) for k in mapping)}"
                    )
            return result

        if inspect.iscoroutinefunction(router):

            @functools.wraps(router)
            async def routed(*args: Any, **kwargs: Any) -> Any:
                return check(await router(*args, **kwargs))

        else:

            @functools.wraps(router)
            def routed(*args: Any, **kwargs: Any) -> Any:
                return check(router(*args, **kwargs))

        return routed

    def _check_send_payload(self, who: str, send: Send) -> None:
        """Hold a `Send.arg` to the worker's declared `input_schema`.

        `Send.node` is checked because LangGraph drops an unknown target without
        an error; `Send.arg` is checked for the mirror-image reason — LangGraph
        hands the payload to the worker *exactly* as given, `input_schema` or
        not, so a wrong-class payload is discovered by the worker's own body, as
        an `AttributeError` several frames from the dispatcher that produced it.
        A shard is data with a declared shape; it is refused where it is
        dispatched, not where it is dereferenced.

        No schema declared means no claim was made, so nothing to check — see
        `add_node`. Payload isolation is separate and unconditional: `_enter`
        deep-copies whatever arrives.
        """
        schema = self._input_schemas.get(send.node)
        if schema is None or isinstance(send.arg, schema):
            return
        raise StateTypeError(
            f"{who} sent node {send.node!r} a payload its input_schema rejects: "
            f"expected {schema.__name__}, got {type(send.arg).__name__} "
            f"({_short_repr(send.arg)}); LangGraph hands a Send payload to the "
            f"worker unchecked, so this would surface inside the worker's body "
            f"instead of here"
        )

    def _check_goto_target(self, who: str, target: Any) -> None:
        """Reject one routing destination this graph cannot reach. See `GraphRoutingError`."""
        if isinstance(target, Send):
            if target.node not in self._nodes:
                # `_short_repr` throughout: the offending name is quoted back, and a
                # node name built from a model's output can be arbitrarily long.
                name = _short_repr(target.node)
                raise GraphRoutingError(
                    f"{who} routed to Send(node={name}, ...), but {name} is not a node "
                    f"of graph {self.name!r}; LangGraph drops an unknown Send target "
                    f"and the run continues as if the routing had not happened. Valid "
                    f"Send targets: "
                    f"{', '.join(repr(n) for n in sorted(self._nodes)) or '(none)'} "
                    f"— END is not one, because a Send has to name a node that runs"
                )
            self._check_send_payload(who, target)
            return
        if isinstance(target, str):
            if target == END or target in self._nodes:
                return
            raise GraphRoutingError(
                f"{who} routed to {_short_repr(target)}, which is not a node of graph "
                f"{self.name!r}; LangGraph drops an unknown destination and the run "
                f"continues as if the routing had not happened. Valid destinations: "
                f"{self._destinations()}"
            )
        raise GraphRoutingError(
            f"{who} routed to {_short_repr(target)}, a {type(target).__name__}; a "
            f"destination must be a node name, END, or a langgraph Send — anything "
            f"else is skipped by LangGraph without a word. Valid destinations: "
            f"{self._destinations()}"
        )

    def _check_goto(self, who: str, goto: Any) -> Any:
        """Validate a `Command`'s `goto` and return the value LangGraph should see.

        Every shape LangGraph accepts stays legal: a node name, `END`, a `Send`,
        and any sequence of those (`()` — the default — means "no dynamic
        routing, use the static edges"). A non-string iterable is materialised
        into a tuple and handed back rather than merely inspected, so a one-shot
        iterable (`(n for n in candidates)`) is not consumed by the check and
        then delivered empty to the router.
        """
        if isinstance(goto, (str, Send)):
            self._check_goto_target(who, goto)
            return goto
        if goto is None:
            raise GraphRoutingError(
                f"{who} returned Command(goto=None), which LangGraph cannot iterate. "
                f"Omit goto to fall through to the graph's static edges, or name a "
                f"destination: {self._destinations()}"
            )
        try:
            targets = tuple(goto)
        except TypeError as exc:
            raise GraphRoutingError(
                f"{who} returned Command(goto={_short_repr(goto)}); goto must be a node "
                f"name, END, a Send, or a sequence of those. Valid destinations: "
                f"{self._destinations()}"
            ) from exc
        for target in targets:
            self._check_goto_target(who, target)
        return targets

    def _check_result(
        self, who: str, writes: set[str], result: Any
    ) -> tuple[Any, dict[str, Any] | None]:
        """Return (what LangGraph receives, the state delta it carries).

        A `Command` is accepted so nodes can route dynamically and hand off to
        each other, and its `update` goes through exactly the checks a returned
        dict does — the validated values are put back into the command, so the
        graph applies what the schema approved rather than what the node wrote.

        Its `goto` is checked the same way and for the same reason: LangGraph
        resolves destinations after the node has returned and silently ignores
        the ones it does not recognise, so an unreachable destination is refused
        here instead of quietly becoming a no-op (see `GraphRoutingError`).
        """
        if isinstance(result, Command):
            if result.graph is not None:
                raise WritePermissionError(
                    f"{who} returned Command(graph={result.graph!r}); GraphARC accepts "
                    "commands addressed to the current graph only, because an update "
                    "aimed at another graph's state cannot be checked against this "
                    "graph's declared writes"
                )
            update: dict[str, Any] | None
            if result.update is None:
                update = None
            elif not isinstance(result.update, dict):
                raise WritePermissionError(
                    f"{who} returned Command(update=...) carrying "
                    f"{type(result.update)!r}; GraphARC requires a dict so each field "
                    "can be checked against the node's declared writes"
                )
            else:
                update = self._check_update(who, writes, result.update)
            goto = self._check_goto(who, result.goto)
            if update is None:
                return replace(result, goto=goto), None
            return replace(result, update=update, goto=goto), update
        if not isinstance(result, dict):
            raise WritePermissionError(
                f"{who} must return a dict update or None — or a langgraph Command — "
                f"got {type(result)!r}"
            )
        update = self._check_update(who, writes, result)
        return update, update

    def _assert_acyclic(self) -> None:
        adjacency: dict[str, list[str]] = {}
        for a, b in self._static_edges:
            if a == START or b == END:
                continue
            adjacency.setdefault(a, []).append(b)
        visiting, done = set(), set()

        def visit(node: str, path: list[str]) -> None:
            if node in done:
                return
            if node in visiting:
                cycle = " -> ".join([*path, node])
                raise GraphCycleError(f"graph {self.name!r} is dag=True but has a cycle: {cycle}")
            visiting.add(node)
            for nxt in adjacency.get(node, []):
                visit(nxt, [*path, node])
            visiting.discard(node)
            done.add(node)

        for node in list(adjacency):
            visit(node, [])

    def _enter(
        self, name: str, state: Any, config: RunnableConfig
    ) -> tuple[RunContext, Callable[..., None], Any]:
        """Everything both node wrappers do before the node body runs.

        Fails closed on a missing run context, isolates the state, checks the
        budget, charges the iteration, opens the trace.
        """
        ctx: RunContext | None = config.get("configurable", {}).get("grapharc_ctx")
        if ctx is None:
            raise MissingRunContextError(
                f"node {name!r} executed without a GraphARC run context; drive the "
                "graph via CompiledGraphARC.invoke()/stream()/ainvoke()/astream() — raw "
                "LangGraph entry points would silently bypass budgets and traces"
            )
        step = ctx.next_step()
        trace = self.trace

        def emit(phase: str, **kw: Any) -> None:
            if trace is not None:
                trace.event(
                    run_id=ctx.run_id,
                    thread_id=ctx.thread_id,
                    attempt=ctx.attempt,
                    graph=self.name,
                    node=name,
                    phase=phase,
                    step=step,
                    **kw,
                )

        # Nodes get a deep copy: the returned dict is the *only* write channel.
        # Without this, in-place mutation of nested models would bypass write
        # permissions invisibly (Pydantic passes nested models by reference).
        #
        # *Every* input, not only a BaseModel one: a fan-out `Send` payload can
        # be any object, and two Sends built from one dict used to hand the
        # workers the same live object — parallel nodes mutating shared state,
        # which is a data race whose writes appear in nobody's declared writes.
        # Fan-out is where the isolation matters most, so it cannot be the one
        # path that skips it.
        if isinstance(state, BaseModel):
            state = state.model_copy(deep=True)
        else:
            state = copy.deepcopy(state)

        try:
            ctx.meter.check()
        except BudgetExceeded as exc:
            emit("error", error=f"budget: {exc.reason}")
            raise
        ctx.meter.charge_iteration()

        emit("start")
        return ctx, emit, state

    def _leave(
        self,
        name: str,
        writes: set[str],
        result: Any,
        *,
        ctx: RunContext,
        emit: Callable[..., None],
        duration_ms: float,
        tokens_before: int,
        usage: Any = None,
    ) -> Any:
        """Everything both node wrappers do after the node body returns."""
        delta: dict[str, Any] | None = None
        if result is not None:
            try:
                result, delta = self._check_result(f"node {name!r}", writes, result)
            except (WritePermissionError, StateTypeError, GraphRoutingError) as err:
                emit(
                    "error",
                    duration_ms=duration_ms,
                    error=str(err),
                    tokens=ctx.meter.tokens - tokens_before,
                )
                raise

        # Tokens are charged mid-node by the usage callback, so this is the
        # first boundary at which a spend made inside the node can stop the run.
        try:
            ctx.meter.check_tokens()
        except BudgetExceeded as exc:
            # Stamped with what the node spent, exactly as `end` is. Without it
            # the run stopped *for overspending* and then reported spending
            # nothing, which is the audit trail losing the one number the stop
            # was about.
            emit(
                "error",
                duration_ms=duration_ms,
                error=f"budget: {exc.reason}",
                tokens=ctx.meter.tokens - tokens_before,
            )
            raise

        # The provider's own price for every model call made inside this node,
        # collected by the usage callback. Left None when no backend reported
        # one, so `observe.cost` can keep a measured figure apart from a
        # rate-card estimate instead of recording a guess as a fact.
        models = getattr(usage, "models", ())
        # This node's own spend, from the meter's per-node scope, not the movement
        # of the run's shared total while the node ran. Under fan-out those differ:
        # the workers' windows overlap, so every worker used to be credited with
        # each sibling's concurrent spend — three workers costing 8 tokens each
        # traced as 24/16/8, and `metrics` and `cost` both reported 48 for 24
        # tokens of real work, doubling the estimated bill purely because the work
        # ran in parallel. Falls back to the difference when no scope was open, so
        # a caller driving `_leave` outside `charging()` still gets a figure.
        spent = getattr(usage, "tokens", None)
        emit(
            "end",
            state_delta=delta,
            duration_ms=duration_ms,
            tokens=ctx.meter.tokens - tokens_before if spent is None else spent,
            cost_usd=getattr(usage, "cost_usd", None),
            model=models[0] if len(models) == 1 else None,
        )
        return result

    def _wrap(self, name: str, fn: Callable[..., Any], writes: set[str]) -> Callable[..., Any]:
        wants_ctx = len(inspect.signature(fn).parameters) >= 2

        if _is_async_callable(fn):
            self._async_nodes.add(name)

            async def awrapped(state: Any, config: RunnableConfig) -> Any:
                ctx, emit, state = self._enter(name, state, config)
                tokens_before = ctx.meter.tokens
                t0 = time.perf_counter()
                try:
                    # `charging` is a contextvar, and asyncio copies the context
                    # into every task it spawns, so a model call awaited anywhere
                    # under this node still reports to this run's meter.
                    with charging(ctx.meter) as usage:
                        async with _async_deadline(ctx.meter, what=f"node {name!r}"):
                            result = await (fn(state, ctx) if wants_ctx else fn(state))
                except BaseException as exc:
                    # BaseException, not Exception: an async node is stopped by
                    # cancellation, which is not an Exception, and a stop with no
                    # trace line is a stop nobody can audit afterwards. Carries
                    # the node's spend for the same reason `end` does.
                    emit(
                        "error",
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        error=repr(exc),
                        tokens=ctx.meter.tokens - tokens_before,
                    )
                    raise
                return self._leave(
                    name,
                    writes,
                    result,
                    ctx=ctx,
                    emit=emit,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    tokens_before=tokens_before,
                    usage=usage,
                )

            return awrapped

        def wrapped(state: Any, config: RunnableConfig) -> Any:
            ctx, emit, state = self._enter(name, state, config)
            tokens_before = ctx.meter.tokens
            t0 = time.perf_counter()
            try:
                # `charging` meters model calls the node makes without asking it
                # to; `deadline_guard` makes max_seconds bite inside the node
                # rather than after it.
                with (
                    charging(ctx.meter) as usage,
                    deadline_guard(ctx.meter, what=f"node {name!r}"),
                ):
                    result = fn(state, ctx) if wants_ctx else fn(state)
            except BaseException as exc:
                # BaseException, not Exception, for the reason the async twin
                # gives: a sync node is most often stopped by a human hitting
                # ^C, which is a KeyboardInterrupt and not an Exception, and a
                # stop with no trace line is a stop nobody can audit afterwards.
                # The exception is re-raised untouched; only the record is new.
                emit(
                    "error",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error=repr(exc),
                    tokens=ctx.meter.tokens - tokens_before,
                )
                raise
            return self._leave(
                name,
                writes,
                result,
                ctx=ctx,
                emit=emit,
                duration_ms=(time.perf_counter() - t0) * 1000,
                tokens_before=tokens_before,
                usage=usage,
            )

        return wrapped


def topology_delta(arc: GraphARC) -> dict[str, Any]:
    """The declared shape of a graph, as one trace-event `state_delta`.

    Node order is declaration order (a proposal's order, for materialized
    graphs). Edge triples carry their kind: `static` edges run unconditionally,
    `conditional` targets are a router's statically-known options. Fan-out
    targets are dynamic, so only the sources are named — a renderer connects
    them to the workers that actually ran.
    """
    edges = [[source, target, "static"] for source, target in arc._static_edges]
    edges += [
        [source, target, "conditional"] for source, target in arc._conditional_edges
    ]
    delta: dict[str, Any] = {"nodes": list(arc._nodes), "edges": edges}
    if arc._fanout_sources:
        delta["fanout_sources"] = list(arc._fanout_sources)
    return delta


class CompiledGraphARC:
    """A compiled graph plus GraphARC run semantics (run ids, budgets, resume)."""

    def __init__(self, inner: Any, arc: GraphARC) -> None:
        self.inner = inner
        self.arc = arc
        self.last_run: RunContext | None = None

    def _run_config(
        self, thread_id: str | None, run_id: str | None, budget: Budget | None
    ) -> dict[str, Any]:
        rid = run_id or uuid.uuid4().hex[:12]
        thread = thread_id or rid
        step_seed, attempt = 0, 1
        if self.arc.trace is not None:
            last_step, last_attempt = self.arc.trace.thread_summary(thread)
            step_seed, attempt = last_step, last_attempt + 1
        ctx = RunContext(
            run_id=rid,
            graph=self.arc.name,
            meter=BudgetMeter(budget or self.arc.budget or _NOOP_BUDGET),
            thread_id=thread,
            attempt=attempt,
            step_seed=step_seed,
        )
        self.last_run = ctx
        if self.arc.trace is not None:
            # The graph's declared shape, into the same audit trail its
            # execution lands in. Until this event existed, a diagram could
            # only ever show the path that ran — branches not taken, parallel
            # structure, and nodes that never started were unrecoverable from
            # the trace. Emitted per entry (a resumed attempt re-states it);
            # readers keep the latest per graph. `step=0` sits below every real
            # step, so readers comparing step ranges filter this phase out.
            self.arc.trace.event(
                run_id=rid,
                graph=self.arc.name,
                node="topology",
                phase="topology",
                step=0,
                thread_id=thread,
                attempt=attempt,
                state_delta=topology_delta(self.arc),
            )
        config: dict[str, Any] = {"configurable": {"thread_id": thread, "grapharc_ctx": ctx}}
        limit = (budget or self.arc.budget or _NOOP_BUDGET).max_concurrency
        if limit is not None:
            config["max_concurrency"] = limit
        return config

    def _reject_async_nodes(self, entry: str) -> None:
        """Refuse a sync entry point before any node runs, not at the first coroutine."""
        if self.arc._async_nodes:
            raise AsyncNodeError(
                f"graph {self.arc.name!r} has async nodes "
                f"{sorted(self.arc._async_nodes)}; use a{entry}(), because {entry}() "
                "has no event loop to await them on"
            )

    def _thread_config(self, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if checkpoint_id is not None:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    def _reject_unknown_fields(self, who: str, values: dict[str, Any]) -> None:
        """Refuse keys the state schema does not have. One wording, every door."""
        unknown = set(values) - set(self.arc.state_schema.model_fields)
        if unknown:
            raise WritePermissionError(
                f"{who} targets unknown state fields: {sorted(unknown)}"
            )

    def _checked_input(self, entry: str, input: Any) -> Any:
        """Hold an entry point's `input` to the state schema's field names.

        The state schema forbids extra fields, and `update_state` refuses an
        unknown key — but LangGraph filters a dict input down to the channels it
        knows *before* the state model is ever constructed, so `extra="forbid"`
        never sees the typo and the graph runs to completion on default values.
        A misspelled question field is then a complete, plausible-looking run
        against an empty question, with nothing said to the caller. The front
        door gets the same refusal the side door already gives.

        Only a dict is checked: a state model has already been validated by
        Pydantic, and `None` means "resume from the last checkpoint".
        """
        if isinstance(input, dict):
            self._reject_unknown_fields(f"{entry}()", input)
        return input

    # -- running -----------------------------------------------------------

    def invoke(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> Any:
        """Run the graph. Pass `input=None` with a previous `thread_id` to resume
        from the last checkpoint (requires a checkpointer at compile time).

        Notes: the budget meter is fresh per invoke — limits bound one attempt,
        not the lifetime of a thread. Trace step/attempt counters continue from
        the thread's history so replay points stay unique across resumes.
        """
        self._reject_async_nodes("invoke")
        input = self._checked_input("invoke", input)
        return self.inner.invoke(input, self._run_config(thread_id, run_id, budget))

    def stream(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
        **stream_kwargs: Any,
    ) -> Iterator[Any]:
        """Stream graph execution through the disciplined path (budgets + traces).

        This is the supported streaming entry point; `.inner.stream()` fails
        closed with MissingRunContextError.
        """
        self._reject_async_nodes("stream")
        input = self._checked_input("stream", input)
        yield from self.inner.stream(
            input, self._run_config(thread_id, run_id, budget), **stream_kwargs
        )

    async def ainvoke(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> Any:
        """Async twin of `invoke()`, with the same discipline and the same notes.

        Sync nodes stay legal here — LangGraph runs each on a worker thread, and
        the wrapper's contract does not change. What does change for `async def`
        nodes is how `max_seconds` is delivered: see `_async_deadline`.
        """
        input = self._checked_input("ainvoke", input)
        return await self.inner.ainvoke(input, self._run_config(thread_id, run_id, budget))

    async def astream(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
        **stream_kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async twin of `stream()`; `.inner.astream()` fails closed."""
        input = self._checked_input("astream", input)
        async for chunk in self.inner.astream(
            input, self._run_config(thread_id, run_id, budget), **stream_kwargs
        ):
            yield chunk

    async def astream_events(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
        version: Literal["v1", "v2"] = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """LangChain's event stream (per-node and per-model-call events), budgeted.

        `version="v3"` is not offered: LangGraph returns a stream object rather
        than an async iterator for it, which is a different shape than this
        method's contract.
        """
        if version not in ("v1", "v2"):
            raise ValueError(f"astream_events supports version 'v1' or 'v2', got {version!r}")
        input = self._checked_input("astream_events", input)
        async for event in self.inner.astream_events(
            input, self._run_config(thread_id, run_id, budget), version=version, **kwargs
        ):
            yield event

    # -- state access (human-in-the-loop) ----------------------------------
    #
    # These need a checkpointer at compile time; without one LangGraph raises
    # ValueError("No checkpointer set"). They read and write checkpoints, they
    # do not execute nodes, so they carry no budget and emit no trace events.

    def get_state(
        self, thread_id: str, *, checkpoint_id: str | None = None, subgraphs: bool = False
    ) -> StateSnapshot:
        """The thread's current checkpoint: values, what runs next, pending interrupts."""
        return self.inner.get_state(
            self._thread_config(thread_id, checkpoint_id), subgraphs=subgraphs
        )

    async def aget_state(
        self, thread_id: str, *, checkpoint_id: str | None = None, subgraphs: bool = False
    ) -> StateSnapshot:
        return await self.inner.aget_state(
            self._thread_config(thread_id, checkpoint_id), subgraphs=subgraphs
        )

    def get_state_history(
        self,
        thread_id: str,
        *,
        before_checkpoint_id: str | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[StateSnapshot]:
        """Checkpoints for a thread, newest first — the replay points of a run."""
        return self.inner.get_state_history(
            self._thread_config(thread_id),
            before=(
                self._thread_config(thread_id, before_checkpoint_id)
                if before_checkpoint_id is not None
                else None
            ),
            limit=limit,
            filter=filter,
        )

    async def aget_state_history(
        self,
        thread_id: str,
        *,
        before_checkpoint_id: str | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        async for snapshot in self.inner.aget_state_history(
            self._thread_config(thread_id),
            before=(
                self._thread_config(thread_id, before_checkpoint_id)
                if before_checkpoint_id is not None
                else None
            ),
            limit=limit,
            filter=filter,
        ):
            yield snapshot

    def _checked_values(self, values: Any, as_node: str | None) -> dict[str, Any]:
        """Type-check an out-of-band state update, and its writes when it claims a node.

        A human editing state mid-run is not a node, so there is no allowlist to
        apply by default — but `as_node` says "record this as if that node wrote
        it", and then the node's declared writes are exactly the right contract.
        The residual gap, stated rather than papered over: with `as_node=None`
        LangGraph attributes the update to whichever node last ran, and GraphARC
        does not reproduce that inference, so such an update is type-checked but
        not allowlisted.
        """
        if not isinstance(values, dict):
            raise WritePermissionError(
                f"update_state takes a dict of field updates, got {type(values)!r}"
            )
        self._reject_unknown_fields("update_state", values)
        if as_node is not None and as_node in self.arc._nodes:
            return self.arc._check_update(
                f"update_state(as_node={as_node!r})", self.arc._nodes[as_node], values
            )
        return self.arc._validate_writes("update_state", values)

    def update_state(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        as_node: str | None = None,
        checkpoint_id: str | None = None,
    ) -> RunnableConfig:
        """Write state from outside the graph, as if `as_node` had written it.

        Returns the config of the checkpoint created. See `_checked_values` for
        exactly which checks apply.
        """
        return self.inner.update_state(
            self._thread_config(thread_id, checkpoint_id),
            self._checked_values(values, as_node),
            as_node=as_node,
        )

    async def aupdate_state(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        as_node: str | None = None,
        checkpoint_id: str | None = None,
    ) -> RunnableConfig:
        return await self.inner.aupdate_state(
            self._thread_config(thread_id, checkpoint_id),
            self._checked_values(values, as_node),
            as_node=as_node,
        )


__all__ = [
    "END",
    "START",
    "AsyncNodeError",
    "Command",
    "CompiledGraphARC",
    "GraphARC",
    "GraphCycleError",
    "GraphRoutingError",
    "MissingRunContextError",
    "RunContext",
    "StateTypeError",
    "WritePermissionError",
]
