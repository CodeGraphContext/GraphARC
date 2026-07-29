"""The session runtime, sitting over the graph kernel.

A `Session` is a graph thread plus everything the kernel deliberately does not
know about: who is driving it, what was said to it since it last ran, whether a
human still has to sign something off, and whether it should stop.

Three mechanics do the work, and all three are node-boundary mechanics because
that is the only place a graph is in a consistent, checkpointed state:

**Stopping.** A turn drives the graph through `CompiledGraphARC.stream()` and
checks the durable event queue after every superstep. An interrupt closes the
stream. The kernel has already checkpointed the completed nodes, and the node
that was next is still pending, so the next turn resumes exactly there — no
work is repeated and none is skipped.

**Approval.** Gated nodes are named on the graph's `GraphSpec` and passed to
LangGraph as `interrupt_before`, so the graph stops *before* the gated node
runs rather than after. A superstep can put more than one gated node on the
boundary, and *every* one of them is held, because LangGraph resumes a boundary
whole: hold one and approving that one runs all of them. Each is a separate
request with its own id, a decision answers one request, and the graph does not
move while any request is unanswered — so a signature on `send_email` is not a
signature on `delete_records`, and a "no" on one is not a licence to drop the
other. Once every hold is answered the boundary is released in one step:
approved nodes run, and each rejected node has the writes its outgoing *edges*
would have made recorded against its own pending task, which walks the graph
past it without its body ever executing. Per task, not as a state update: an
update attributed to one node of a parallel step clears the other tasks'
triggers and loses them.

**Resuming elsewhere.** Nothing above lives in memory. Status, queue, holds and
the audit trail are rows in a `SessionStore`; graph state is in the kernel's
checkpointer; the graph itself is rebuilt by name from a `GraphRegistry`. A new
process pointed at the same directory can pick a session up by id.

What this does *not* give you, stated plainly because "interruptible" and
"governed" are exactly the words people over-read:

- **An interrupt does not stop a running node.** It is read at the next
  *superstep* boundary — after the whole parallel step has finished and been
  checkpointed, not after each node in it — so a node already inside its body
  runs to completion first, and so does every node running beside it. Cutting
  a node off mid-flight is what `Budget(max_seconds=...)` is for, and its own
  docstring is honest about the limits of that.
- **The approval gate is enforced by this runtime, not by the graph.** Driving
  the same compiled graph directly — `session.graph.invoke(...)` — runs gated
  nodes with nothing holding them. The gate is a property of being driven by a
  session.
- **A turn is synchronous.** `run()` occupies its caller until the session
  stops. The kernel grew `astream` while this was being written; an async turn
  is now buildable and is simply not built yet.
- **One runner at a time is a claim, not a lease.** `SessionStore.transition`
  stops a second runner from claiming a session, and nothing reclaims one whose
  runner died holding it — see that method's docstring.
- **A hold names a node, not one particular task.** When a `Send` fan-out puts
  the *same* gated node on the boundary several times over, each of those tasks
  is held separately and each needs its own decision — so the count is exact
  and nothing runs unapproved — but the requests are indistinguishable, right
  down to the action text, and rejecting one of them skips an arbitrary one of
  that node's tasks rather than a chosen one. Approving *k* of *n* runs *k*;
  which *k* is not yours to pick.
- **One verdict reaches the graph.** `SessionState.approval_decision` is a
  single channel, so when a boundary settles several holds at once the graph
  sees the first hold's verdict. Every verdict is on the session's event log
  and in `TurnResult.decisions`; a node routing on `state.decision` sees one.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableSequence
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel, ConfigDict, Field

from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import CompiledGraphARC
from grapharc.session.approval import (
    DECISION_FIELD,
    ApprovalDecision,
    ApprovalRequest,
)
from grapharc.session.db import open_database
from grapharc.session.errors import (
    ApprovalRequired,
    SessionBusy,
    SessionError,
    SessionTerminated,
)
from grapharc.session.events import EventKind, SessionEvent
from grapharc.session.registry import REGISTRY, GraphRegistry, GraphSpec
from grapharc.session.store import (
    RESUMABLE,
    SessionRecord,
    SessionStatus,
    SessionStore,
)

STORE_FILENAME = "sessions.sqlite"
CHECKPOINT_FILENAME = "checkpoints.sqlite"

#: Keys LangGraph uses for its own stream channels; they are not node names.
_INTERNAL_PREFIX = "__"

#: `updates` gives us the node names; `checkpoints` is the superstep boundary
#: marker. A `checkpoints` chunk means the step finished *and* was written, so
#: closing the stream there loses nothing. `updates` alone cannot mark a
#: boundary: a parallel step emits one chunk per branch, and stopping between
#: two of them tears the step in half.
_STREAM_MODES = ["updates", "checkpoints"]

# LangGraph's own config keys for driving a node's *writers* — the runnables
# that carry its outgoing edges — without its body. Rejecting a node means
# running exactly those. They live in a private module; the literals are the
# wire values, kept as a fallback so an upstream reshuffle degrades to "still
# works" rather than "package will not import".
try:  # pragma: no cover - import shape, exercised by every rejection test
    from langgraph._internal._constants import (
        CONFIG_KEY_READ as _READ_KEY,
    )
    from langgraph._internal._constants import (
        CONFIG_KEY_SEND as _SEND_KEY,
    )
    from langgraph._internal._constants import (
        CONFIG_KEY_TASK_ID as _TASK_ID_KEY,
    )
    from langgraph._internal._constants import (
        NO_WRITES as _NO_WRITES,
    )
except ImportError:  # pragma: no cover
    _READ_KEY, _SEND_KEY, _TASK_ID_KEY = "__pregel_read", "__pregel_send", "__pregel_task_id"
    _NO_WRITES = "__no_writes__"


class TurnResult(BaseModel):
    """What one `run()` did. `nodes` is the honest answer to "what executed"."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    nodes: tuple[str, ...] = ()
    """Nodes whose bodies ran this turn. Rejected nodes are not among them."""
    skipped: tuple[str, ...] = ()
    """Nodes a rejection walked the graph past without executing."""
    pending: tuple[str, ...] = ()
    state: dict[str, Any] = Field(default_factory=dict)
    approvals: tuple[ApprovalRequest, ...] = ()
    """Every hold now open. More than one when a superstep held several gates."""
    interrupted_by: str | None = None
    error: str | None = None
    messages: tuple[str, ...] = ()
    decisions: tuple[ApprovalDecision, ...] = ()
    """Every decision this turn applied, in boundary order."""

    @property
    def approval(self) -> ApprovalRequest | None:
        """The first open hold. `approvals` is the whole answer — see it first."""
        return self.approvals[0] if self.approvals else None

    @property
    def decision(self) -> ApprovalDecision | None:
        """The first decision applied. `decisions` is the whole answer."""
        return self.decisions[0] if self.decisions else None


class Session:
    """One long-lived conversation with one graph thread.

    Construct these through `SessionManager.create()` / `.resume()`: a session
    is only meaningful next to the store and checkpointer it was recorded in.
    """

    def __init__(
        self,
        *,
        record: SessionRecord,
        manager: SessionManager,
        spec: GraphSpec,
        graph: CompiledGraphARC | None = None,
    ) -> None:
        self.id = record.id
        self.graph_name = record.graph
        self.thread_id = record.thread_id
        self.spec = spec
        self._manager = manager
        self._store = manager.store
        # `create()` builds the graph before the record exists, so a graph that
        # fails the session contract does not leave a session behind.
        self.graph: CompiledGraphARC = graph or spec.build(manager.checkpointer)
        self._gates = list(spec.approval_nodes) or None

    def __repr__(self) -> str:
        return f"<Session {self.id} graph={self.graph_name!r} status={self.status.value}>"

    # -- reading -----------------------------------------------------------

    @property
    def record(self) -> SessionRecord:
        return self._store.require(self.id)

    @property
    def status(self) -> SessionStatus:
        return self.record.status

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        """Every gated node this session is holding. Each needs its own decision."""
        return self.record.pending_approvals

    @property
    def pending_approval(self) -> ApprovalRequest | None:
        """The first open hold. Not a "may it run" test — see `pending_approvals`."""
        return self.record.pending_approval

    def _snapshot(self) -> StateSnapshot:
        """The kernel's checkpoint for this thread: values and what runs next."""
        return self.graph.get_state(self.thread_id)

    def state(self) -> dict[str, Any]:
        """The graph state as checkpointed. `{}` for a session that has never run."""
        return dict(self._snapshot().values)

    def pending_nodes(self) -> tuple[str, ...]:
        """Nodes the kernel will run next. Non-empty means the graph is mid-flight."""
        return tuple(self._snapshot().next)

    def events(self) -> list[SessionEvent]:
        """Every event ever queued for this session, consumed or not."""
        return self._store.events(self.id)

    def pending_events(self) -> list[SessionEvent]:
        """Events the session has not acted on yet."""
        return self._store.pending(self.id)

    def history(self):
        """Every status change, with the pid that made it."""
        return self._store.history(self.id)

    # -- writing -----------------------------------------------------------

    def send(self, text: str) -> SessionEvent:
        """Queue a message. Delivered to the graph's `inbox` at the next turn."""
        return self._store.push(self.id, EventKind.MESSAGE, body=text)

    def interrupt(self, reason: str = "") -> SessionEvent:
        """Ask the session to stop at its next node boundary.

        Durable and cross-process: this writes a row, and whichever process is
        driving the session reads it after the next superstep. A turn that is
        not running yet honours a queued interrupt before its first node, so an
        interrupt is never silently lost — it stops the next turn instead.

        Stopping on a boundary where the next node is gated re-opens the hold,
        even if a decision had just been applied: the approved run did not
        happen, so the next one is asked for again. See `_settle`.
        """
        return self._store.push(self.id, EventKind.INTERRUPT, body=reason)

    def decide(
        self,
        *,
        approved: bool,
        decided_by: str = "",
        reason: str = "",
        request_id: str | None = None,
    ) -> SessionEvent:
        """Answer one of the session's outstanding approval requests.

        `request_id` may be omitted only while exactly one node is held. With
        several holds open there is no "the" request, and guessing at one is
        how an operator signs off `send_email` and releases `delete_records`:
        naming the request is required, and `pending_approvals` lists them.
        """
        holds = self.pending_approvals
        rid = request_id
        if rid is None:
            if len(holds) > 1:
                raise SessionError(
                    f"session {self.id!r} is holding {[h.node for h in holds]}; "
                    "pass request_id to say which one this decides — "
                    "see pending_approvals"
                )
            rid = holds[0].id if holds else None
        if rid is None:
            raise SessionError(
                f"session {self.id!r} has no outstanding approval request to decide"
            )
        decision = ApprovalDecision(
            request_id=rid, approved=approved, decided_by=decided_by, reason=reason
        )
        return self._store.push(
            self.id, EventKind.DECISION, body=rid, payload=decision.model_dump()
        )

    def terminate(self, reason: str = "") -> SessionRecord:
        """End the session for good. Idempotent; terminated is a terminal status."""
        record = self.record
        if record.status is SessionStatus.TERMINATED:
            return record
        return self._store.transition(
            self.id,
            SessionStatus.TERMINATED,
            reason=reason or "terminated by caller",
            approval=None,
        )

    # -- running -----------------------------------------------------------

    def run(
        self, input: dict[str, Any] | None = None, *, budget: Budget | None = None
    ) -> TurnResult:
        """Drive the graph until it finishes, is interrupted, or hits a gate.

        `input` seeds a fresh pass through the graph and is only accepted when
        no graph work is pending; mid-flight, use `send()` — the pending node
        will see the message. Queued messages are appended to `inbox` either
        way.

        Ends in one of five statuses: `idle` (nothing pending),
        `awaiting_approval` (at least one gated node is held), `interrupted`
        (stopped because someone asked it to stop — usually, but not always,
        with work pending), `failed` (the turn raised, and `TurnResult.error`
        says how), or `terminated` (someone ended the session while the turn
        was in flight; terminal wins and the turn is reported, not undone). A
        turn that raises is not an exception here: `run()` only raises for
        refusals it makes *before* doing any work.

        Raises `ApprovalRequired` when the session is holding one or more nodes
        and any of those holds has no decision queued against it. Every gated
        node on the boundary is held separately, so a decision on one of them
        releases that one and no other. That refusal is the gate: it is
        enforced here, by the runtime, not by the graph, so a graph author
        cannot forget to check — and so it cannot be relied on by anything that
        drives the graph without going through a session.
        """
        record = self.record
        if record.status is SessionStatus.TERMINATED:
            raise SessionTerminated(f"session {self.id!r} is terminated")

        snapshot = self._snapshot()
        resuming = bool(snapshot.next)
        if resuming and input is not None:
            raise SessionError(
                f"session {self.id!r} has {tuple(snapshot.next)} pending mid-graph; "
                "send() a message instead of passing new input"
            )

        # Checked before anything is claimed or consumed, so a refused run
        # leaves the session exactly as it found it. *Every* hold has to be
        # settled: releasing the boundary because one of them was decided is
        # how an approval for one node runs another.
        holds = record.pending_approvals
        verdicts = self._queued_decisions(holds)
        unsettled = [hold for hold in holds if hold.id not in verdicts]
        if unsettled:
            first = unsettled[0]
            raise ApprovalRequired(self.id, first.node, first.id)

        # Compare-and-set on the status: this is the claim, and it is what a
        # second runner loses.
        self._store.transition(
            self.id,
            SessionStatus.RUNNING,
            expect=RESUMABLE,
            reason="turn started",
            error=None,
        )

        nodes: list[str] = []
        skipped: tuple[str, ...] = ()
        error: str | None = None
        messages: tuple[str, ...] = ()
        decisions: tuple[ApprovalDecision, ...] = ()

        # A stop queued while the session was asleep is honoured before the
        # first node — and before the queue is drained. Consuming input is
        # work: a message swallowed by a turn that then ran nothing would be
        # gone, having reached neither the graph nor the next turn.
        interrupted_by = self._take_interrupt()
        if interrupted_by is None:
            queued = self._store.pending(self.id)
            messages, decisions = self._read_queue(queued, holds)
            try:
                stream_input, skipped = self._prepare(
                    snapshot=snapshot,
                    resuming=resuming,
                    input=input,
                    messages=messages,
                    decisions=decisions,
                    holds=holds,
                )
            except Exception as exc:
                # Fails closed and fails clean. Nothing was consumed and the
                # stream was never opened, so the queue is intact for the next
                # turn, no held node was released, and `_settle` records a
                # failed turn instead of leaving a session wedged in `running`
                # with its input already eaten.
                error = repr(exc)
            else:
                self._consume(queued)
                nodes, interrupted_by, error = self._drive(
                    stream_input, skipped=skipped, budget=budget
                )

        return self._settle(
            nodes=nodes,
            skipped=skipped,
            error=error,
            interrupted_by=interrupted_by,
            messages=messages,
            decisions=decisions,
        )

    def _drive(
        self,
        stream_input: Any,
        *,
        skipped: tuple[str, ...],
        budget: Budget | None,
    ) -> tuple[list[str], str | None, str | None]:
        """Run the graph, stopping at the first superstep boundary that says to.

        The interrupt is read on `checkpoints` chunks only. Those arrive once
        per superstep, after the step's writes have been applied and the
        checkpoint written — closing the stream there is the difference between
        resuming cleanly and losing a parallel branch's result along with the
        boundary that would have scheduled what comes next.
        """
        nodes: list[str] = []
        interrupted_by: str | None = None
        error: str | None = None
        # Rejected nodes are reported by LangGraph like any other task, because
        # their writes are on file. They did not execute, so they are not
        # `nodes`. Counted, not set-tested: a cycle may run the same node for
        # real on a later step.
        unclaimed = Counter(skipped)
        stream = self.graph.stream(
            stream_input,
            thread_id=self.thread_id,
            budget=budget,
            stream_mode=_STREAM_MODES,
            interrupt_before=self._gates,
        )
        try:
            for mode, chunk in stream:
                if mode == "updates":
                    for key in chunk:
                        if key.startswith(_INTERNAL_PREFIX):
                            continue
                        if unclaimed.get(key):
                            unclaimed[key] -= 1
                            continue
                        nodes.append(key)
                    continue
                # A `checkpoints` chunk, and the only place a stop is read: the
                # superstep is finished and on disk.
                interrupted_by = self._take_interrupt()
                if interrupted_by is not None:
                    break
        except Exception as exc:
            # Recorded on the session rather than swallowed: a turn that
            # raised is a stopping point with a reason, and the pending
            # node is still pending, so the next turn picks it up.
            error = repr(exc)
        finally:
            stream.close()
        return nodes, interrupted_by, error

    # -- turn internals ----------------------------------------------------

    def _queued_decisions(
        self, holds: Sequence[ApprovalRequest]
    ) -> dict[str, ApprovalDecision]:
        """Queued verdicts, by request id, for the holds they actually name.

        A decision matches one request and one only. Everything else in the
        queue — including a verdict on a hold that has since been re-opened
        under a new id — is not an answer to any of `holds`.
        """
        wanted = {hold.id for hold in holds}
        found: dict[str, ApprovalDecision] = {}
        for event in self._store.pending(self.id):
            if event.kind is not EventKind.DECISION:
                continue
            candidate = ApprovalDecision.model_validate(event.payload)
            if candidate.request_id in wanted:
                found[candidate.request_id] = candidate
        return found

    def _read_queue(
        self, queued: Sequence[SessionEvent], holds: Sequence[ApprovalRequest]
    ) -> tuple[tuple[str, ...], tuple[ApprovalDecision, ...]]:
        """What this turn will act on. Reads only — `_consume` does the writing.

        Split from consuming so a turn that cannot prepare its writes leaves
        the queue where it found it: input that reached neither the graph nor
        the next turn is input that is gone.

        Decisions come back in hold order, not arrival order, because that is
        the order the boundary will be released in.
        """
        messages = tuple(e.body for e in queued if e.kind is EventKind.MESSAGE)
        by_request: dict[str, ApprovalDecision] = {}
        for event in queued:
            if event.kind is not EventKind.DECISION:
                continue
            candidate = ApprovalDecision.model_validate(event.payload)
            by_request[candidate.request_id] = candidate
        decisions = tuple(
            by_request[hold.id] for hold in holds if hold.id in by_request
        )
        return messages, decisions

    def _consume(self, queued: Iterable[SessionEvent]) -> None:
        """Mark this turn's messages and decisions consumed.

        Decisions that name no live hold are consumed too. They can never apply
        — request ids are unique — and leaving them pending would mean a stale
        verdict sitting in the queue forever. The event row keeps the record.
        """
        self._store.consume(
            e.seq for e in queued if e.kind in (EventKind.MESSAGE, EventKind.DECISION)
        )

    def _prepare(
        self,
        *,
        snapshot: StateSnapshot,
        resuming: bool,
        input: dict[str, Any] | None,
        messages: tuple[str, ...],
        decisions: Sequence[ApprovalDecision],
        holds: Sequence[ApprovalRequest],
    ) -> tuple[Any, tuple[str, ...]]:
        """Settle the boundary and return `(stream input, nodes skipped)`.

        Mid-flight the session-owned writes ride into the resumed superstep as
        `Command(update=...)` rather than being written out of band first. They
        are applied before the pending tasks run, so the node that was about to
        run still runs and still sees them, and — unlike an out-of-band
        `update_state` — nothing about the boundary is disturbed on the way in.

        A rejection then does one thing more. The rejected node's *writers* are
        run — the runnables carrying its outgoing edges, not its body — and the
        writes they produce are filed against that node's own pending task. To
        LangGraph the task then looks already-done: it is not executed, its
        edges fire, and the graph advances past it. Filing them per task rather
        than as a state update is what keeps the *rest* of the superstep alive;
        an `update_state` attributed to one node of a parallel step clears the
        triggers of the others and drops them on the floor.

        Nothing here carries the rejected node's values: the runtime is
        skipping the node, not impersonating it.

        Raises rather than returning half a boundary. The caller treats that as
        a failed turn, which leaves every hold open — a rejection that cannot
        be recorded must never become a resume that runs the node.
        """
        updates: dict[str, Any] = {}
        if messages:
            current = list(snapshot.values.get("inbox") or [])
            if not resuming and input is not None and "inbox" in input:
                current = list(input["inbox"])
            updates["inbox"] = [*current, *messages]
        if decisions:
            # One channel, so one verdict reaches the graph: the first hold's,
            # in boundary order. Every verdict is in the event log and in
            # `TurnResult.decisions`; what a *node* can route on is one.
            updates[DECISION_FIELD] = decisions[0].model_dump()

        if not resuming:
            # A fresh pass: nothing is pending, so there is no boundary to
            # settle and nothing to reject.
            return {**(input or {}), **updates}, ()

        by_request = {decision.request_id: decision for decision in decisions}
        rejected = tuple(
            hold.node
            for hold in holds
            if hold.id in by_request and not by_request[hold.id].approved
        )
        stream_input = (
            # The kernel's own check, the same one `update_state` applies:
            # types are validated, unknown fields refused. `as_node=None`
            # because the session is not a node and claims no allowlist.
            Command(update=self.graph._checked_values(updates, None))
            if updates
            else None
        )
        return stream_input, self._reject(rejected, snapshot)

    def _reject(
        self, nodes: Sequence[str], snapshot: StateSnapshot
    ) -> tuple[str, ...]:
        """File each rejected node's edge writes against its own pending task."""
        if not nodes:
            return ()
        wanted = Counter(nodes)
        checkpointer = self._manager.checkpointer
        values = dict(snapshot.values)
        skipped: list[str] = []
        for task in snapshot.tasks:
            if not wanted.get(task.name):
                continue
            wanted[task.name] -= 1
            checkpointer.put_writes(
                snapshot.config, self._edge_writes(task.name, task.id, values), task.id
            )
            skipped.append(task.name)
        missing = sorted(name for name, left in wanted.items() if left)
        if missing:
            # The hold named a node this boundary is not about to run. Refusing
            # is the only safe answer: carrying on would resume a graph with a
            # rejection that was never applied.
            raise SessionError(
                f"session {self.id!r}: rejected {missing} but the boundary has no "
                f"pending task for them (pending: {list(snapshot.next)})"
            )
        return tuple(skipped)

    def _edge_writes(
        self, node: str, task_id: str, values: Mapping[str, Any]
    ) -> list[tuple[str, Any]]:
        """What `node`'s outgoing edges write, with its body never called.

        These are the node's *writers*: for a static edge, the write that
        schedules the successor; for a conditional edge, the router — run
        against the checkpointed state, since the skipped node contributed
        nothing to it.
        """
        writers = self.graph.inner.nodes[node].flat_writers

        def read(select: str | list[str], fresh: bool = False) -> Any:
            if isinstance(select, str):
                return values.get(select)
            return {key: values[key] for key in select if key in values}

        captured: deque[tuple[str, Any]] = deque()
        run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
        run.invoke(
            {},
            {
                "configurable": {
                    "thread_id": self.thread_id,
                    _SEND_KEY: captured.extend,
                    _TASK_ID_KEY: task_id,
                    _READ_KEY: read,
                }
            },
        )
        # A task with no writes at all is a task LangGraph has not seen run, and
        # it will run it. The sentinel says "done, wrote nothing" — which is
        # what a rejected node with nowhere to go is.
        return list(captured) or [(_NO_WRITES, None)]

    def _take_interrupt(self) -> str | None:
        event = self._store.take(self.id, EventKind.INTERRUPT)
        if event is None:
            return None
        return event.body or "interrupt requested"

    def _settle(
        self,
        *,
        nodes: list[str],
        skipped: tuple[str, ...],
        error: str | None,
        interrupted_by: str | None,
        messages: tuple[str, ...],
        decisions: tuple[ApprovalDecision, ...],
    ) -> TurnResult:
        snapshot = self._snapshot()
        pending = tuple(snapshot.next)
        gates = set(self.spec.approval_nodes)
        # Every gated node on the boundary, not the first one. A hold on one
        # node of a superstep does not hold the superstep: LangGraph resumes
        # the whole of it, so a gate left unrecorded is a gate that runs.
        holds = tuple(
            ApprovalRequest(
                node=node, action=self.spec.describe_action(node, snapshot.values)
            )
            for node in pending
            if node in gates
        )

        if error is not None:
            # The holds still go on the record. A failed turn is retried, and a
            # retry that found nothing held would resume straight into the
            # gated node — `interrupt_before` does not fire twice for a
            # boundary it has already stopped at.
            status = SessionStatus.FAILED
            reason = f"turn raised: {error}"
            if holds:
                reason += f", still holding {[h.node for h in holds]}"
        elif holds:
            # A gate outranks an interrupt that landed on the same boundary,
            # and it re-opens even when this turn already applied a decision.
            # Resuming a thread whose next node is *already* the gated one does
            # not re-trigger LangGraph's `interrupt_before`, so "pending gated
            # node" is the only durable evidence that a hold is needed. Trading
            # a redundant second decision for the chance of running an unheld
            # node is not a close call.
            status = SessionStatus.AWAITING_APPROVAL
            reason = "holding " + "; ".join(f"{h.node!r}: {h.action}" for h in holds)
        elif interrupted_by is not None:
            status, reason = SessionStatus.INTERRUPTED, f"interrupted: {interrupted_by}"
        elif pending:
            status = SessionStatus.INTERRUPTED
            reason = f"graph suspended with {list(pending)} pending"
        else:
            status, reason = SessionStatus.IDLE, "turn complete"

        try:
            record = self._store.transition(
                self.id,
                status,
                expect=(SessionStatus.RUNNING,),
                reason=reason,
                error=error,
                approval=holds or None,
                bump_turn=True,
            )
        except SessionBusy as busy:
            # Someone terminated the session while this turn was in flight.
            # Terminal is terminal: report it rather than resurrect the session.
            if busy.actual != SessionStatus.TERMINATED.value:
                raise
            record = self.record

        return TurnResult(
            session_id=self.id,
            status=record.status,
            nodes=tuple(nodes),
            skipped=skipped,
            pending=pending,
            state=dict(snapshot.values),
            approvals=record.pending_approvals,
            interrupted_by=interrupted_by,
            error=error,
            messages=messages,
            decisions=decisions,
        )


class SessionManager:
    """Sessions rooted at one directory: a session store beside a checkpoint store.

    Both files sit under `root`, so "resume this session in another process"
    is "point another `SessionManager` at the same directory".
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        registry: GraphRegistry | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or REGISTRY
        self.store = SessionStore(self.root / STORE_FILENAME, busy_timeout_ms=busy_timeout_ms)
        self._conn: sqlite3.Connection = open_database(
            self.root / CHECKPOINT_FILENAME, busy_timeout_ms=busy_timeout_ms
        )
        self.checkpointer = SqliteSaver(self._conn)

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()
        self._conn.close()

    def create(
        self,
        graph: str,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Record a new session and bind it to a freshly built graph.

        The graph is resolved and built first: an unknown name or a graph that
        does not meet the session contract must not leave a session record
        pointing at something that cannot run.
        """
        spec = self.registry.get(graph)
        compiled = spec.build(self.checkpointer)
        record = self.store.create(
            session_id=session_id or uuid.uuid4().hex[:16],
            graph=graph,
            thread_id=thread_id,
            metadata=metadata,
        )
        return Session(record=record, manager=self, spec=spec, graph=compiled)

    def resume(self, session_id: str) -> Session:
        """Rehydrate a session by id, in this process, on a rebuilt graph.

        Each session gets its own graph instance: two sessions sharing one
        would share whatever the nodes closed over.
        """
        record = self.store.require(session_id)
        return Session(record=record, manager=self, spec=self.registry.get(record.graph))

    def list(
        self, *, status: SessionStatus | Iterable[SessionStatus] | None = None
    ) -> list[SessionRecord]:
        return self.store.list(status=status)


__all__ = [
    "CHECKPOINT_FILENAME",
    "STORE_FILENAME",
    "Session",
    "SessionManager",
    "TurnResult",
]
