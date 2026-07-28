"""The governed loop: plan -> admit -> materialise -> execute -> observe -> replan.

ARCHITECTURE.md §2, driven. One `run()` call carries a goal from a planner's
first proposal to a recorded stop, and every round of it goes through the same
gate:

    goal ──► planner proposes ──► admission checks ──► materialise ──► execute
                    ▲                    │                                 │
                    └── rejection + reason ┘                               │
                    └──────── more work, budget remains ◄──────────────────┘

**The property this exists to hold.** Work discovered mid-run does not bypass
admission. Round 7's proposal is checked by the same `AdmissionChecker`, against
the same registry and edge policy, as round 1's — there is no "already approved"
path and no cached authorisation. "The same registry" means the same *object*,
which is not the same as the same contents: `NodeRegistry` is mutable, and a
node body could widen the allowlist between rounds. Call `registry.freeze()`
before handing it to the checker and that reading collapses into the other one.
An executing node cannot add topology to the
run either: the loop's only source of topology is `planner.propose`, so what a
node discovers reaches the next round as *state*, which the planner may read and
the checker still judges. That is what makes the design general-purpose rather
than a flowchart: the shape is discovered while working, and every piece of it
was still authorised before it ran.

**A rejection is fuel, not an exception.** `AdmissionResult.feedback()` — the
per-check list, with codes and remedies — is handed to the planner as the
feedback for its next turn and kept verbatim in the round record. The loop does
not negotiate on the planner's behalf: it never trims an over-large proposal,
never drops a denied edge, and never argues with a verdict. The checker is
deterministic code, so the only way forward is a different proposal.

**Every stop is a reason.** `run()` returns a `LoopResult` carrying a `LoopStop`
and never simply runs out of road. The reasons are the ones §2 names — goal met,
budget spent, no progress, human halt — plus the three a real driver needs:
`max_rounds`, `admission_refused` (the planner could not produce something legal
within the limit), and `planning_failed` (the model's replies could not be
turned into a proposal at all). What this does *not* promise is that `run()`
never raises: an unusable model *reply* is data and becomes a round record, but
an exception raised by *your own code* propagates. That means the planner object
and the checker, and equally the `goal_reached`, `observe` and `progress`
callbacks — all of them run on the loop's thread and none of them is wrapped.
The loop records stops; it does not launder a broken component into one.

The line is drawn at what a *model* can cause. Anything a planner's reply can
produce — unparseable output, a refused proposal, a name the orchestrator
reserves, a subgraph that will not build, a body that raises — becomes a round
record and a `LoopStop`. Anything your code does is yours.

**Bounds come from `runtime.budget`, not from here.** A single `BudgetMeter` for
the whole run holds the ceilings on tokens, iterations and wall-clock; each
round executes under a `Budget` derived from what that meter has *left*, so the
last round cannot spend what the earlier ones already did, and the loop's own
`max_rounds` is a separate, coarser bound on planning turns. The planner's
tokens are charged to the same meter, so a planner that talks itself out of a
budget stops the run exactly as an expensive graph does.

**The audit trail.** With a `TraceRecorder` every round writes one
`phase="round"` event and the run writes one `phase="stop"` event, on the loop's
own thread. Give the *same* recorder to the planner, the checker and the
materialiser — the loop does not wire them for you — and the planner's
`phase="plan"` events, the checker's `phase="admission"` events and every
materialised node's `start`/`end` events land under the same `run_id`.
Afterwards the three questions ARCHITECTURE §6 asks can be answered from the
file alone: what it did (node events), what it was allowed to do (admission
events, with the fingerprint of the exact proposal), and why it stopped (the
stop event). Each round's execution runs on its own trace thread
(`<run_id>/r<n>`) so per-thread step numbers stay unique.

Three limits, stated rather than implied. The loop drives one planner and one
checker over one shared state object; it is not thread-safe for concurrent
`run()` calls on the same instance, and a `request_halt()` is permanent for the
object that received it. `goal_reached` is *your* code: the loop has no opinion
about what "done" means and will not ask a model, so with no `goal_reached`
supplied a run ends on the planner proposing no further work, or on a bound. And
depth is reported, not measured: this driver plans from outside the graph it
runs, so it tells the checker `parent_depth=0` by default — nest a loop inside a
node body and you must pass the real depth yourself, or the checker's
`max_depth` bounds nothing above one proposal's own nesting.

Not yet here: rounds execute through the kernel's synchronous `invoke()`, so a
registry of `async def` bodies needs an async driver this module does not
provide. `Materializer` builds such a graph correctly — the kernel's async path
and the deadline semantics that go with it are intact — it just has to be driven
through `ainvoke()` by hand.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field

from grapharc.observe.trace import TraceRecorder
from grapharc.planner.admission import (
    AdmissionChecker,
    AdmissionResult,
    Rejection,
    RemainingBudget,
)
from grapharc.planner.materialize import MaterializationError, Materializer
from grapharc.planner.proposal import PlanningOutcome, Subgraph
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import RunContext


class LoopStop(StrEnum):
    """Why a run ended. Machine-readable, and always set.

    Deliberately its own vocabulary rather than
    `grapharc.runtime.convergence.StopReason`: that enum bounds a *cycle inside
    one graph*, and three of the reasons here — a refused proposal, an unusable
    planner reply, a planner with nothing left to propose — have no meaning
    there. Where the two overlap the spelling is shared (`budget_exhausted`,
    `no_progress`, `human_stopped`), and `max_rounds` is deliberately *not*
    spelled `max_iterations`: a round is one planning turn, an iteration is one
    node execution, and the loop bounds both separately.
    """

    GOAL_MET = "goal_met"
    # The planner was admitted while proposing nothing, and no goal check said
    # the goal was met. It has run out of work; that is an answer, not a fault.
    NO_FURTHER_WORK = "no_further_work"
    MAX_ROUNDS = "max_rounds"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    ADMISSION_REFUSED = "admission_refused"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    HUMAN_STOPPED = "human_stopped"


# Stops that mean the run did what it was asked, for the trace's `error` field:
# everything else is recorded as a refusal so an audit can find it by grepping
# for errors rather than by knowing this enum.
_CLEAN_STOPS = frozenset({LoopStop.GOAL_MET, LoopStop.NO_FURTHER_WORK})


class Planner(Protocol):
    """What the loop needs from a planner. `PlannerNode` satisfies it structurally.

    One turn, one outcome, and no exception for a bad model reply — an unusable
    answer must come back as a `PlanningOutcome` carrying the reason, or the
    loop cannot record it and replan from it.
    """

    def propose(
        self, task: str, ctx: RunContext | None = None, *, feedback: str = ""
    ) -> PlanningOutcome: ...


class LoopLimits(BaseModel):
    """Bounds on the *loop*, distinct from the run's resource `Budget`.

    These stop a planner that is failing productively — proposing something
    illegal, unparseable, or ineffective, forever, inside budget. Every counter
    below is consecutive and resets the moment the thing it counts stops
    happening.
    """

    model_config = ConfigDict(frozen=True)

    max_rounds: int = 8
    # Consecutive rounds whose proposal admission refused. The planner is told
    # why each time; this is how many times it may fail to act on that.
    max_consecutive_rejections: int = 3
    # Consecutive rounds whose model reply could not be turned into a proposal.
    max_consecutive_planning_failures: int = 3
    # Consecutive rounds that were admitted but could not be built or raised
    # while running. Materialisation failures count here: they are the front
    # half of execution.
    max_consecutive_execution_failures: int = 2
    # Consecutive executed rounds that left the progress signal unchanged.
    max_stalled_rounds: int = 2


class RoundRecord(BaseModel):
    """One round, kept whole: what was asked, proposed, decided, run and spent."""

    model_config = ConfigDict(frozen=True)

    round: int
    task: str = ""
    proposal: Subgraph | None = None
    admission: AdmissionResult | None = None
    planner_error: str = ""
    materialization_error: str = ""
    execution_error: str = ""
    executed: bool = False
    progressed: bool = False
    tokens: int = 0
    iterations: int = 0
    duration_ms: float = 0.0

    @property
    def admitted(self) -> bool:
        return self.admission is not None and self.admission.admitted

    @property
    def rejections(self) -> tuple[Rejection, ...]:
        return () if self.admission is None else self.admission.rejections


class LoopResult(BaseModel):
    """The whole run: why it stopped, every round, the final state, what it spent."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    stop: LoopStop
    detail: str = ""
    goal: str = ""
    run_id: str = ""
    rounds: tuple[RoundRecord, ...] = ()
    # The state model the last executed round produced (the initial state when
    # nothing ran). Typed by the materialiser's `state_schema`.
    state: Any = None
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True only for `goal_met`. `no_further_work` is a stop, not a success."""
        return self.stop is LoopStop.GOAL_MET

    @property
    def executed_rounds(self) -> tuple[RoundRecord, ...]:
        return tuple(r for r in self.rounds if r.executed)

    @property
    def admitted_rounds(self) -> tuple[RoundRecord, ...]:
        return tuple(r for r in self.rounds if r.admitted)

    def rejections(self) -> tuple[Rejection, ...]:
        """Every rejection from every round, in round order."""
        return tuple(reason for record in self.rounds for reason in record.rejections)


class _Execution(NamedTuple):
    """What one round's execution attempt produced, for the driver to act on."""

    executed: bool = False
    state: Any = None
    materialization_error: str = ""
    execution_error: str = ""
    hard_stop: LoopStop | None = None

    @property
    def failure(self) -> str:
        return self.materialization_error or self.execution_error


def _signature(state: Any) -> str:
    """A content hash of the state, for the default no-progress signal."""
    try:
        payload = state.model_dump_json()
    except Exception:  # noqa: BLE001 - a state that will not serialise still needs a signal
        payload = repr(state)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:16]


class GovernedLoop:
    """Drives the admission cycle. It plans, it gates, it runs, it stops on the record.

    It holds no node bodies of its own: everything that executes was built by
    `Materializer` from the registry, and the planner is called through
    `propose()` and is never given anything to run. The split ARCHITECTURE §2
    rests on therefore survives the loop rather than being re-asserted by it.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        checker: AdmissionChecker,
        materializer: Materializer,
        budget: Budget | None = None,
        limits: LoopLimits | None = None,
        trace: TraceRecorder | None = None,
        name: str = "governed_loop",
        goal_reached: Callable[[Any], bool] | None = None,
        observe: Callable[[Any], str] | None = None,
        progress: Callable[[Any], Any] | None = None,
        checkpointer: Any = None,
        parent_depth: int = 0,
    ) -> None:
        self.planner = planner
        self.checker = checker
        self.materializer = materializer
        self.budget = budget or Budget()
        self.limits = limits or LoopLimits()
        self.trace = trace
        self.name = name
        # Deterministic code, never a model: "are we done" is exactly the kind
        # of question a run must not be able to talk itself into answering yes.
        self.goal_reached = goal_reached
        self.observe = observe
        self.progress = progress
        self.checkpointer = checkpointer
        # How deep this driver already is, reported to the checker on every
        # round. 0 is the truth for a loop planning from outside a graph; a loop
        # embedded in a node body has to say so, because the checker cannot see
        # how it was reached and `max_depth` is only as honest as this number.
        if parent_depth < 0:
            raise ValueError(f"parent_depth must be >= 0, got {parent_depth}")
        self.parent_depth = parent_depth
        self._halt = threading.Event()
        self._halt_reason = ""

    @property
    def state_schema(self) -> type[BaseModel]:
        """The materialiser's schema — one state contract for the whole run."""
        return self.materializer.state_schema

    def request_halt(self, reason: str = "") -> None:
        """Ask the run to stop at the next round boundary.

        Safe to call from another thread. It takes effect between rounds, never
        mid-node: a graph already executing finishes or hits its own budget, so
        a halt never leaves a node half-applied. Permanent for this instance —
        a halted loop stays halted.
        """
        self._halt_reason = reason
        self._halt.set()

    # -- the cycle ------------------------------------------------------------

    def run(
        self,
        goal: str,
        state: Any = None,
        *,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> LoopResult:
        """Carry `goal` from first proposal to recorded stop.

        `state` may be a `state_schema` instance, a dict, or None for the
        schema's defaults. The returned `LoopResult.state` is the last state any
        executed round produced.
        """
        rid = run_id or uuid.uuid4().hex[:12]
        meter = BudgetMeter(self.budget)
        ctx = RunContext(
            run_id=rid, graph=self.name, meter=meter, thread_id=thread_id or rid, attempt=1
        )
        current = self._initial_state(state)
        rounds: list[RoundRecord] = []
        feedback = ""
        note = ""
        rejected_in_a_row = 0
        unplanned_in_a_row = 0
        failed_in_a_row = 0
        stalled_in_a_row = 0

        stop, detail = self._precheck(current)
        round_number = 0
        started = 0.0
        task = ""

        def close(**fields: Any) -> None:
            """End the round: one record, one trace line, whatever happened in it."""
            rounds.append(
                self._record(
                    ctx,
                    round=round_number,
                    task=task,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stop=stop,
                    **fields,
                )
            )

        while stop is None and round_number < self.limits.max_rounds:
            stop, detail = self._boundary(meter)
            if stop is not None:
                break
            round_number += 1
            started = time.perf_counter()

            task = self._task(goal, current, round_number, note, meter)
            note = ""
            try:
                outcome = self.planner.propose(task, ctx, feedback=feedback)
            except BudgetExceeded as exc:
                # The run's hard ceiling, hit while the planner was talking. It
                # is a stop with a reason, not this round's business to retry.
                stop, detail = LoopStop.BUDGET_EXHAUSTED, exc.reason
                close(planner_error=exc.reason)
                break
            feedback = ""

            if not outcome.ok or outcome.proposal is None:
                unplanned_in_a_row += 1
                note = (
                    "Your previous reply could not be used as a proposal: "
                    f"{outcome.error}. Reply with the proposal object and nothing else."
                )
                if unplanned_in_a_row >= self.limits.max_consecutive_planning_failures:
                    stop = LoopStop.PLANNING_FAILED
                    detail = (
                        f"{unplanned_in_a_row} consecutive unusable planner replies; "
                        f"last: {outcome.error}"
                    )
                close(planner_error=outcome.error, tokens=outcome.tokens)
                continue
            unplanned_in_a_row = 0

            proposal = outcome.proposal
            verdict = self.checker.check(
                proposal, meter=meter, ctx=ctx, parent_depth=self.parent_depth
            )
            judged = {"proposal": proposal, "admission": verdict, "tokens": outcome.tokens}

            if not verdict.admitted:
                rejected_in_a_row += 1
                feedback = verdict.feedback()
                if rejected_in_a_row >= self.limits.max_consecutive_rejections:
                    stop = LoopStop.ADMISSION_REFUSED
                    detail = (
                        f"{rejected_in_a_row} consecutive proposals were not admitted; "
                        f"last failed checks: "
                        f"{', '.join(c.value for c in verdict.failed_checks())}"
                    )
                close(**judged)
                continue
            rejected_in_a_row = 0

            if not proposal.nodes:
                # Admitted, and it authorises nothing: the planner is saying
                # there is no further work. Admitting that is the right answer.
                stop = (
                    LoopStop.GOAL_MET if self._goal_met(current) else LoopStop.NO_FURTHER_WORK
                )
                detail = "the planner proposed no further work"
                close(**judged)
                break

            spent = meter.exceeded()
            if spent is not None:
                stop, detail = LoopStop.BUDGET_EXHAUSTED, spent
                close(**judged)
                break

            before = self._progress_of(current)
            attempt = self._execute(
                verdict, proposal, current, meter=meter, ctx=ctx, round_number=round_number
            )
            current = attempt.state
            progressed = attempt.executed and self._progress_of(current) != before

            if attempt.hard_stop is not None:
                stop, detail = attempt.hard_stop, attempt.failure
            elif not attempt.executed:
                failed_in_a_row += 1
                note = (
                    f"The subgraph you proposed did not run: {attempt.failure}. "
                    "Propose something that addresses that."
                )
                if failed_in_a_row >= self.limits.max_consecutive_execution_failures:
                    stop = LoopStop.EXECUTION_FAILED
                    detail = (
                        f"{failed_in_a_row} consecutive admitted proposals failed to "
                        f"run; last: {attempt.failure}"
                    )
            else:
                failed_in_a_row = 0
                stalled_in_a_row = 0 if progressed else stalled_in_a_row + 1
                if self._goal_met(current):
                    stop, detail = LoopStop.GOAL_MET, "the goal check was satisfied"
                elif stalled_in_a_row >= self.limits.max_stalled_rounds:
                    stop = LoopStop.NO_PROGRESS
                    detail = f"{stalled_in_a_row} executed rounds left the state unchanged"

            close(
                **judged,
                materialization_error=attempt.materialization_error,
                execution_error=attempt.execution_error,
                executed=attempt.executed,
                progressed=progressed,
            )

        if stop is None:
            stop = LoopStop.MAX_ROUNDS
            detail = f"reached max_rounds ({self.limits.max_rounds}) without meeting the goal"

        result = LoopResult(
            stop=stop,
            detail=detail,
            goal=goal,
            run_id=rid,
            rounds=tuple(rounds),
            state=current,
            usage=meter.snapshot(),
        )
        self._emit(
            ctx,
            node=f"{self.name}:stop",
            phase="stop",
            state_delta={
                "stop": stop.value,
                "detail": detail,
                "rounds": len(rounds),
                "executed_rounds": len(result.executed_rounds),
                "usage": result.usage,
            },
            error=None if stop in _CLEAN_STOPS else f"{stop.value}: {detail}",
        )
        return result

    # -- pieces of one round --------------------------------------------------

    def _initial_state(self, state: Any) -> Any:
        if state is None:
            return self.state_schema()
        if isinstance(state, self.state_schema):
            return state
        return self.state_schema.model_validate(state)

    def _precheck(self, state: Any) -> tuple[LoopStop | None, str]:
        """Reasons to stop before the planner is ever asked."""
        if self._halt.is_set():
            return LoopStop.HUMAN_STOPPED, self._halt_reason or "a halt was requested"
        if self._goal_met(state):
            return LoopStop.GOAL_MET, "the goal check was already satisfied"
        return None, ""

    def _boundary(self, meter: BudgetMeter) -> tuple[LoopStop | None, str]:
        """The two ceilings checked at the top of every round."""
        if self._halt.is_set():
            return LoopStop.HUMAN_STOPPED, self._halt_reason or "a halt was requested"
        spent = meter.exceeded()
        if spent is not None:
            return LoopStop.BUDGET_EXHAUSTED, spent
        return None, ""

    def _goal_met(self, state: Any) -> bool:
        return bool(self.goal_reached(state)) if self.goal_reached is not None else False

    def _progress_of(self, state: Any) -> Any:
        return _signature(state) if self.progress is None else self.progress(state)

    def _task(
        self, goal: str, state: Any, round_number: int, note: str, meter: BudgetMeter
    ) -> str:
        """What the planner is asked this round: the goal, the world, and the headroom."""
        parts = [goal]
        if self.observe is not None:
            parts.append(f"Current state:\n{self.observe(state)}")
        if note:
            parts.append(note)
        left = RemainingBudget.from_meter(meter)
        parts.append(
            f"Planning round {round_number} of at most {self.limits.max_rounds}. "
            f"Budget remaining — tokens: {_left(left.tokens)}, "
            f"iterations: {_left(left.iterations)}, seconds: {_left(left.seconds)}."
        )
        return "\n\n".join(parts)

    def _execute(
        self,
        verdict: AdmissionResult,
        proposal: Subgraph,
        state: Any,
        *,
        meter: BudgetMeter,
        ctx: RunContext,
        round_number: int,
    ) -> _Execution:
        """Build and run the admitted subgraph. Spend is charged back either way.

        `hard_stop` comes back set only for a failure the loop must not replan
        around — the run's own budget ceiling, which is not this round's
        business to argue with.
        """
        try:
            compiled = self.materializer.materialize(
                verdict, proposal, checkpointer=self.checkpointer
            )
        except MaterializationError as exc:
            return _Execution(state=state, materialization_error=f"could not be built: {exc}")

        budget = self._round_budget(meter)
        try:
            raw = compiled.invoke(
                state,
                run_id=ctx.run_id,
                thread_id=f"{ctx.thread_id}/r{round_number}",
                budget=budget,
            )
        except BudgetExceeded as exc:
            self._charge_back(compiled, meter)
            return _Execution(
                state=state, execution_error=exc.reason, hard_stop=LoopStop.BUDGET_EXHAUSTED
            )
        except Exception as exc:  # noqa: BLE001 - a failed subgraph is a replanning input
            self._charge_back(compiled, meter)
            return _Execution(state=state, execution_error=f"raised {exc!r}")
        self._charge_back(compiled, meter)
        return _Execution(executed=True, state=self._initial_state(raw))

    def _round_budget(self, meter: BudgetMeter) -> Budget:
        """This round's ceiling: exactly what the run has left, per dimension.

        Derived rather than declared, so the rounds share one pool — the last
        round cannot spend what the earlier ones already did.
        """
        left = RemainingBudget.from_meter(meter)
        return Budget(
            max_iterations=left.iterations,
            max_tokens=left.tokens,
            max_seconds=left.seconds,
            max_concurrency=self.budget.max_concurrency,
        )

    @staticmethod
    def _charge_back(compiled: Any, meter: BudgetMeter) -> None:
        """Fold the round's spend into the run's meter, whether or not it finished.

        The sub-run's meter is a separate accountant with its own ceiling; the
        loop's meter never saw those calls, so this charge is new spend rather
        than a re-report and is counted unnamed on purpose.
        """
        sub = getattr(compiled, "last_run", None)
        if sub is None:
            return
        meter.charge_tokens(sub.meter.tokens)
        meter.charge_iteration(sub.meter.iterations)

    # -- recording ------------------------------------------------------------

    def _record(
        self, ctx: RunContext, *, stop: LoopStop | None = None, **fields: Any
    ) -> RoundRecord:
        record = RoundRecord(**fields)
        rejections = [f"{r.check.value}/{r.code}" for r in record.rejections]
        problem = (
            record.planner_error
            or record.materialization_error
            or record.execution_error
            or ("; ".join(rejections) if rejections else "")
        )
        self._emit(
            ctx,
            node=f"{self.name}:round{record.round}",
            phase="round",
            duration_ms=record.duration_ms,
            tokens=record.tokens,
            state_delta={
                "round": record.round,
                "proposal_id": record.proposal.proposal_id if record.proposal else "",
                "fingerprint": record.admission.fingerprint if record.admission else "",
                "status": record.admission.status.value if record.admission else "not_proposed",
                "nodes": record.proposal.node_count() if record.proposal else 0,
                "rejections": rejections,
                "executed": record.executed,
                "progressed": record.progressed,
                "stop": stop.value if stop is not None else "",
            },
            error=problem or None,
        )
        return record

    def _emit(self, ctx: RunContext, *, node: str, phase: str, **fields: Any) -> None:
        # Phases "round" and "stop", never "end": `observe.metrics` counts node
        # executions from "end" events, and a planning round is not one.
        if self.trace is None:
            return
        self.trace.event(
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            attempt=ctx.attempt,
            graph=self.name,
            node=node,
            phase=phase,
            step=ctx.next_step(),
            **fields,
        )


def _left(value: float | int | None) -> str:
    return "unlimited" if value is None else f"{value:g}"


__all__ = [
    "GovernedLoop",
    "LoopLimits",
    "LoopResult",
    "LoopStop",
    "Planner",
    "RoundRecord",
]
