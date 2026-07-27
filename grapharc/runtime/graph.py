"""ArcGraph: LangGraph with graph-engineering discipline bolted on.

A thin wrapper over `langgraph.graph.StateGraph` that enforces what plain
LangGraph leaves to convention:

- **Typed edges** — state schemas are Pydantic models; node updates are
  validated dicts, not free-form writes.
- **Write permissions** — every node declares which state fields it may write;
  an undeclared write raises instead of flowing downstream.
- **Bounded work** — a per-run `BudgetMeter` is checked before every node; the
  hard ceiling cannot be routed around.
- **Traces** — every node execution is recorded (start/end/error, state delta,
  duration, tokens) as JSONL replay points.
- **DAG mode** — `dag=True` rejects conditional edges and cycles at compile
  time (roadmap Stage 0: earn cycles, don't start with them).
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter


class WritePermissionError(Exception):
    """A node wrote a state field it never declared."""


class GraphCycleError(Exception):
    """A cycle was found in a graph compiled with dag=True."""


class MissingRunContextError(Exception):
    """A node executed without GraphARC's run context.

    This happens when a compiled graph is driven through raw LangGraph entry
    points (`.inner.stream()`, `.inner.ainvoke()`, ...) instead of
    `CompiledArcGraph.invoke()`/`.stream()`. GraphARC fails closed: budgets and
    traces are part of the execution contract, so running without them is an
    error, not a silent downgrade.
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


class ArcGraph:
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
        self._static_edges: list[tuple[str, str]] = []

    def add_node(
        self,
        name: str,
        fn: Callable[..., dict[str, Any] | None],
        *,
        writes: Iterable[str],
        input_schema: type[BaseModel] | None = None,
    ) -> ArcGraph:
        """Register a node. `input_schema` types a fan-out worker's Send payload
        (defaults to the graph state schema)."""
        writes_set = set(writes)
        unknown = writes_set - set(self.state_schema.model_fields)
        if unknown:
            raise WritePermissionError(
                f"node {name!r} declares writes to unknown state fields: {sorted(unknown)}"
            )
        self._nodes[name] = writes_set
        self._graph.add_node(
            name, self._wrap(name, fn, writes_set), input_schema=input_schema
        )
        return self

    def add_edge(self, start: str, end: str) -> ArcGraph:
        self._graph.add_edge(start, end)
        self._static_edges.append((start, end))
        return self

    def add_conditional_edge(
        self,
        source: str,
        router: Callable[[Any], str],
        mapping: dict[str, str],
    ) -> ArcGraph:
        """Route on a validated event name returned by deterministic `router` code."""
        if self.dag:
            raise GraphCycleError(
                f"graph {self.name!r} is dag=True: conditional edges are not allowed"
            )
        self._graph.add_conditional_edges(source, router, mapping)
        return self

    def add_fanout_edge(
        self,
        source: str,
        dispatcher: Callable[[Any], list[tuple[str, BaseModel]]],
    ) -> ArcGraph:
        """Explicit parallelism: `dispatcher` returns (worker_node, payload) pairs,
        each dispatched as a parallel Send. Fan-out is a deliberate act — the
        default execution model stays serial."""
        if self.dag:
            raise GraphCycleError(
                f"graph {self.name!r} is dag=True: fan-out edges are not allowed"
            )

        def dispatch(state: Any) -> list[Send]:
            return [Send(node, payload) for node, payload in dispatcher(state)]

        self._graph.add_conditional_edges(source, dispatch)
        return self

    def compile(self, checkpointer: Any = None) -> CompiledArcGraph:
        if self.dag:
            self._assert_acyclic()
        return CompiledArcGraph(self._graph.compile(checkpointer=checkpointer), self)

    # -- internals ---------------------------------------------------------

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

    def _wrap(
        self, name: str, fn: Callable[..., dict[str, Any] | None], writes: set[str]
    ) -> Callable[..., dict[str, Any] | None]:
        wants_ctx = len(inspect.signature(fn).parameters) >= 2

        def wrapped(state: Any, config: RunnableConfig) -> dict[str, Any] | None:
            ctx: RunContext | None = config.get("configurable", {}).get("arc_ctx")
            if ctx is None:
                raise MissingRunContextError(
                    f"node {name!r} executed without a GraphARC run context; drive the "
                    "graph via CompiledArcGraph.invoke()/stream() — raw LangGraph entry "
                    "points would silently bypass budgets and traces"
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
            if isinstance(state, BaseModel):
                state = state.model_copy(deep=True)

            try:
                ctx.meter.check()
            except BudgetExceeded as exc:
                emit("error", error=f"budget: {exc.reason}")
                raise
            ctx.meter.charge_iteration()

            emit("start")
            tokens_before = ctx.meter.tokens
            t0 = time.perf_counter()
            try:
                result = fn(state, ctx) if wants_ctx else fn(state)
            except Exception as exc:
                emit("error", duration_ms=(time.perf_counter() - t0) * 1000, error=repr(exc))
                raise
            duration_ms = (time.perf_counter() - t0) * 1000

            if result is not None:
                if not isinstance(result, dict):
                    err = WritePermissionError(
                        f"node {name!r} must return a dict update or None, got {type(result)!r}"
                    )
                    emit("error", duration_ms=duration_ms, error=str(err))
                    raise err
                illegal = set(result) - writes
                if illegal:
                    err = WritePermissionError(
                        f"node {name!r} wrote undeclared fields {sorted(illegal)}; "
                        f"declared writes: {sorted(writes)}"
                    )
                    emit("error", duration_ms=duration_ms, error=str(err))
                    raise err

            emit(
                "end",
                state_delta=result,
                duration_ms=duration_ms,
                tokens=ctx.meter.tokens - tokens_before,
            )
            return result

        return wrapped


class CompiledArcGraph:
    """A compiled graph plus GraphARC run semantics (run ids, budgets, resume)."""

    def __init__(self, inner: Any, arc: ArcGraph) -> None:
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
        config: dict[str, Any] = {"configurable": {"thread_id": thread, "arc_ctx": ctx}}
        limit = (budget or self.arc.budget or _NOOP_BUDGET).max_concurrency
        if limit is not None:
            config["max_concurrency"] = limit
        return config

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
        return self.inner.invoke(input, self._run_config(thread_id, run_id, budget))

    def stream(
        self,
        input: dict[str, Any] | BaseModel | None,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
        **stream_kwargs: Any,
    ) -> Any:
        """Stream graph execution through the disciplined path (budgets + traces).

        This is the supported streaming entry point; `.inner.stream()` fails
        closed with MissingRunContextError.
        """
        yield from self.inner.stream(
            input, self._run_config(thread_id, run_id, budget), **stream_kwargs
        )


__all__ = [
    "END",
    "START",
    "ArcGraph",
    "CompiledArcGraph",
    "GraphCycleError",
    "MissingRunContextError",
    "RunContext",
    "WritePermissionError",
]
