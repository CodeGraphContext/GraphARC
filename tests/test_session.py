"""Session runtime: resumable across processes, interruptible, and gate-able.

Three claims carry this file, and each is tested against the thing that would
disprove it rather than against a convenient proxy:

- **Resume across a restart** is proved with genuinely separate interpreters.
  The demo graph appends its own name to `log` on every execution, so a resumed
  turn that quietly re-ran earlier nodes would show them twice.
- **Interrupt at a boundary** is proved by a *different process* pushing the
  stop while a node is parked on a file barrier. The node after the boundary
  must not appear anywhere in the state.
- **Approval suspends** is proved by the gated node's effects being absent
  while the hold is open, and by a rejection leaving the gated node unexecuted
  rather than executed-and-undone. When a superstep puts two gated nodes on the
  boundary it is proved by the *second* node's effects being absent after the
  first has been approved — one signature must not release two actions.
- **Concurrency** is proved with a barrier six sessions have to reach at the
  same time, because sequential `a.run(); b.run()` proves nothing about it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.session import (
    ApprovalRequired,
    EventKind,
    GraphRegistry,
    SessionBusy,
    SessionContractError,
    SessionError,
    SessionExistsError,
    SessionManager,
    SessionState,
    SessionStatus,
    SessionStore,
    SessionTerminated,
    ThreadInUseError,
    UnknownGraphError,
    UnknownSessionError,
    decision_in,
    register_graph,
)
from grapharc.session.demo import APPROVAL_NODE, DEMO_SCRIPT, GRAPH_NAME
from grapharc.session.errors import InvalidTransition
from grapharc.session.runtime import STORE_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- subprocess plumbing ----------------------------------------------------

_PRELUDE = """\
import json, os, sys, time
from pathlib import Path

from grapharc.session import EventKind, SessionManager, SessionStore, SessionStatus
from grapharc.session.demo import GRAPH_NAME  # importing is what registers the graph

ROOT = sys.argv[1]
ARGS = sys.argv[2:]


def emit(**payload):
    payload["pid"] = os.getpid()
    sys.stdout.write("RESULT " + json.dumps(payload, default=str) + "\\n")

"""


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / f"{name}.py"
    script.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    return script


def _spawn(script: Path, root: Path, *args: str) -> subprocess.Popen:
    # PYTHONPATH rather than an installed package: the child must import the
    # same working tree the test does.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(
        [sys.executable, str(script), str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )


def _result(proc: subprocess.Popen, timeout: int = 120) -> dict:
    out, err = proc.communicate(timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"child failed ({proc.returncode}):\n{err}")
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"child emitted no result:\nstdout={out}\nstderr={err}")


def run_child(tmp_path: Path, name: str, body: str, root: Path, *args: str) -> dict:
    payload = _result(_spawn(_write_script(tmp_path, name, body), root, *args))
    assert payload["pid"] != os.getpid(), "child ran in the test's own process"
    return payload


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def manager(root: Path):
    with SessionManager(root) as m:
        yield m


@pytest.fixture
def registry() -> GraphRegistry:
    """A registry of its own, so test graphs never leak into the process default."""
    return GraphRegistry()


# -- lifecycle --------------------------------------------------------------


def test_a_new_session_is_recorded_and_findable_by_id(manager):
    session = manager.create(GRAPH_NAME, metadata={"tenant": "acme"})

    record = manager.store.require(session.id)
    assert record.status is SessionStatus.CREATED
    assert record.graph == GRAPH_NAME
    assert record.thread_id == session.id
    assert record.metadata == {"tenant": "acme"}
    assert record.turns == 0
    assert [r.id for r in manager.list()] == [session.id]
    assert session.state() == {}  # nothing checkpointed yet


def test_status_transitions_outside_the_lifecycle_are_refused(manager):
    session = manager.create(GRAPH_NAME)
    store = manager.store

    # created -> idle skips `running`; the lifecycle has no such edge.
    with pytest.raises(InvalidTransition) as caught:
        store.transition(session.id, SessionStatus.IDLE)
    assert caught.value.current == "created"
    assert caught.value.requested == "idle"

    store.transition(session.id, SessionStatus.TERMINATED)
    # Terminated is terminal in both directions, including back to itself.
    for target in SessionStatus:
        with pytest.raises(InvalidTransition):
            store.transition(session.id, target)


def test_a_second_runner_cannot_claim_a_running_session(manager):
    session = manager.create(GRAPH_NAME)
    manager.store.transition(session.id, SessionStatus.RUNNING, reason="pretend runner")

    with pytest.raises(SessionBusy) as caught:
        session.run()
    assert caught.value.actual == "running"
    assert manager.store.require(session.id).runner_pid == os.getpid()

    # And nothing reclaims it on its own: a runner that died holding a session
    # leaves it stuck until someone releases it, which is a legal transition.
    manager.store.transition(
        session.id, SessionStatus.IDLE, expect=(SessionStatus.RUNNING,), reason="released"
    )
    assert session.run().status is SessionStatus.AWAITING_APPROVAL


def test_the_gate_belongs_to_the_session_not_to_the_graph(manager):
    """Pinning a known limit rather than implying one that is not there.

    `interrupt_before` is passed by the session's own stream call. The same
    compiled graph invoked directly runs the gated node — the hold is a
    property of being driven by a session, and nothing in the graph enforces
    it. Anything stronger needs the kernel to carry gates through compile
    (ROADMAP §1.4).
    """
    session = manager.create(GRAPH_NAME, session_id="bypass")
    session.run()
    assert session.status is SessionStatus.AWAITING_APPROVAL
    assert APPROVAL_NODE not in session.state()["log"]

    session.graph.invoke(None, thread_id=session.thread_id)

    assert APPROVAL_NODE in session.state()["log"]
    # The session's record still says it is holding: the runtime was bypassed,
    # not satisfied.
    assert session.record.pending_approval is not None


def test_terminated_sessions_refuse_both_work_and_input(manager):
    session = manager.create(GRAPH_NAME)
    session.terminate("done with it")

    with pytest.raises(SessionTerminated):
        session.run()
    with pytest.raises(SessionTerminated):
        session.send("anything")
    # Idempotent: terminating twice is not an error.
    assert session.terminate().status is SessionStatus.TERMINATED


def test_every_status_change_is_recorded_with_a_reason_and_a_pid(manager):
    session = manager.create(GRAPH_NAME)
    session.run()
    session.decide(approved=True, decided_by="ops")
    session.run()

    history = session.history()
    assert [h.to_status for h in history] == [
        SessionStatus.CREATED,
        SessionStatus.RUNNING,
        SessionStatus.AWAITING_APPROVAL,
        SessionStatus.RUNNING,
        SessionStatus.IDLE,
    ]
    assert all(h.pid == os.getpid() for h in history)
    assert any(APPROVAL_NODE in h.reason for h in history)
    assert manager.store.require(session.id).turns == 2


def test_resuming_a_session_id_that_does_not_exist_is_refused(manager):
    with pytest.raises(UnknownSessionError):
        manager.resume("no-such-session")


def test_reusing_a_session_id_is_refused(manager):
    """Two sessions on one id would be two sessions on one checkpoint thread."""
    manager.create(GRAPH_NAME, session_id="taken")
    with pytest.raises(SessionExistsError, match="taken"):
        manager.create(GRAPH_NAME, session_id="taken")
    assert len(manager.list()) == 1


def test_reusing_another_sessions_thread_is_refused(manager):
    """A checkpoint thread belongs to exactly one session, explicit ids included."""
    manager.create(GRAPH_NAME, session_id="first", thread_id="shared")

    with pytest.raises(ThreadInUseError) as caught:
        manager.create(GRAPH_NAME, session_id="second", thread_id="shared")
    # The error names both sides, so a caller can find the session it collided with.
    assert caught.value.thread_id == "shared"
    assert caught.value.session_id == "first"

    # The refusal is atomic: no session row, no transition row left behind.
    assert [r.id for r in manager.list()] == ["first"]
    assert manager.store.get("second") is None
    assert manager.store.history("second") == []


def test_a_second_session_on_one_thread_cannot_run_a_held_gated_node(manager):
    """Regression for the approval-gate bypass a shared thread used to open.

    A second session on the first one's thread got a fresh record with no
    holds, resumed the checkpointed boundary, and — `interrupt_before` does
    not re-fire for a boundary it has already stopped at — ran the gated node
    with no approval ever given, while the first session's record still said
    the hold was open. The refusal at `create()` is what closes it.
    """
    a = manager.create(GRAPH_NAME, session_id="a", thread_id="shared-thread")
    a.run()
    assert a.status is SessionStatus.AWAITING_APPROVAL
    assert [h.node for h in a.record.pending_approvals] == [APPROVAL_NODE]

    with pytest.raises(ThreadInUseError, match="shared-thread"):
        manager.create(GRAPH_NAME, session_id="b", thread_id="shared-thread")

    # The gated node never ran, and the first session's hold is still the truth.
    assert APPROVAL_NODE not in a.state()["log"]
    assert a.status is SessionStatus.AWAITING_APPROVAL


def test_a_session_terminated_mid_turn_is_reported_not_resurrected(manager, registry):
    """Terminal has to survive a turn that is still finishing.

    The runner's closing transition expects `running`. If it just overwrote the
    status, a `terminate()` racing the end of a turn would be undone.
    """
    killer: dict[str, object] = {}

    class S(SessionState):
        log: list[str] = []

    def build(checkpointer=None):
        def only(state: S) -> dict:
            # Stands in for another actor: it goes through the store, on its
            # own connection, exactly as a second process would.
            store = SessionStore(killer["store_path"])
            store.transition(killer["session_id"], SessionStatus.TERMINATED, reason="killed")
            store.close()
            return {"log": [*state.log, "only"]}

        g = GraphARC(S, name="killable")
        g.add_node("only", only, writes={"log"})
        g.add_edge(START, "only")
        g.add_edge("only", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("killable", build, registry=registry)
    with SessionManager(manager.root, registry=registry) as scoped:
        session = scoped.create("killable", session_id="doomed")
        killer["store_path"] = scoped.root / STORE_FILENAME
        killer["session_id"] = session.id

        turn = session.run()

        assert turn.nodes == ("only",)
        assert turn.status is SessionStatus.TERMINATED
        assert session.status is SessionStatus.TERMINATED
        assert [h.to_status for h in session.history()][-1] is SessionStatus.TERMINATED


# -- the graph contract -----------------------------------------------------


def test_a_graph_that_is_not_a_session_graph_is_refused_at_bind_time(manager, registry):
    class Plain(GraphARCState):
        n: int = 0

    def build(checkpointer=None):
        g = GraphARC(Plain, name="plain")
        g.add_node("n", lambda s: {"n": s.n + 1}, writes={"n"})
        g.add_edge(START, "n")
        g.add_edge("n", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("plain", build, registry=registry)
    with SessionManager(manager.root, registry=registry) as scoped:
        with pytest.raises(SessionContractError, match="SessionState subclass"):
            scoped.create("plain")
        # No half-made session was left pointing at a graph that cannot run.
        assert scoped.list() == []


def test_an_approval_node_the_graph_does_not_have_is_refused(manager, registry):
    class S(SessionState):
        n: int = 0

    def build(checkpointer=None):
        g = GraphARC(S, name="typo")
        g.add_node("apply", lambda s: {"n": s.n + 1}, writes={"n"})
        g.add_edge(START, "apply")
        g.add_edge("apply", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("typo", build, approval_nodes=("aply",), registry=registry)
    with SessionManager(manager.root, registry=registry) as scoped:
        # A mistyped gate name is the worst failure available: the gate never
        # fires and the node runs unheld. It has to be caught at bind time.
        with pytest.raises(SessionContractError, match="aply"):
            scoped.create("typo")


def test_a_process_without_the_graph_registered_cannot_resume(manager, registry):
    session = manager.create(GRAPH_NAME)
    with SessionManager(manager.root, registry=registry) as stranger:
        with pytest.raises(UnknownGraphError, match=GRAPH_NAME):
            stranger.resume(session.id)


# -- multi-turn input -------------------------------------------------------


def test_queued_messages_reach_the_graph_in_order(manager):
    session = manager.create(GRAPH_NAME)
    session.send("first")
    session.send("second")

    turn = session.run()

    assert turn.messages == ("first", "second")
    assert turn.state["received"] == ["first", "second"]
    assert turn.state["inbox"] == []  # the graph consumed them
    assert [e.consumed_at is not None for e in session.events()] == [True, True]


def test_a_later_turn_carries_on_the_same_thread(manager):
    session = manager.create(GRAPH_NAME)
    session.send("turn one")
    session.run()
    session.decide(approved=True, decided_by="ops")
    first = session.run()
    assert first.status is SessionStatus.IDLE

    session.send("turn two")
    second = session.run()

    assert second.state["received"] == ["turn one", "turn two"]
    # A second pass through the same graph, on state the first pass left behind.
    assert second.state["log"] == ["ingest", "plan", "apply", "report", "ingest", "plan"]
    assert second.state["plan"] == DEMO_SCRIPT[1]


def test_passing_new_input_mid_graph_is_refused(manager):
    session = manager.create(GRAPH_NAME)
    session.run()
    assert session.pending_nodes() == (APPROVAL_NODE,)

    with pytest.raises(SessionError, match="send"):
        session.run({"received": ["sneaky"]})


# -- approval ---------------------------------------------------------------


def test_approval_suspends_the_graph_before_the_gated_node_runs(manager):
    session = manager.create(GRAPH_NAME)

    turn = session.run()

    assert turn.status is SessionStatus.AWAITING_APPROVAL
    assert turn.nodes == ("ingest", "plan")
    assert turn.pending == (APPROVAL_NODE,)
    assert turn.approval is not None
    assert turn.approval.node == APPROVAL_NODE
    assert turn.approval.action == f"apply: {DEMO_SCRIPT[0]}"
    # The gated node did not run: no log entry, and `outcome` — which only
    # `apply` and `report` ever write — is not in the checkpoint at all.
    assert APPROVAL_NODE not in turn.state["log"]
    assert "outcome" not in turn.state
    assert session.record.pending_approval is not None


def test_resuming_a_held_session_without_a_decision_is_refused(manager):
    session = manager.create(GRAPH_NAME)
    session.run()
    before = session.record

    with pytest.raises(ApprovalRequired) as caught:
        session.run()

    assert caught.value.node == APPROVAL_NODE
    # The refusal changed nothing: not the status, not the turn count.
    after = session.record
    assert after.status is SessionStatus.AWAITING_APPROVAL
    assert after.turns == before.turns
    assert APPROVAL_NODE not in session.state()["log"]


def test_a_decision_naming_a_different_request_does_not_release_the_hold(manager):
    session = manager.create(GRAPH_NAME)
    session.run()

    session.decide(approved=True, decided_by="ops", request_id="some-other-request")
    with pytest.raises(ApprovalRequired):
        session.run()
    assert APPROVAL_NODE not in session.state()["log"]

    # The right decision releases it, and the stale verdict is consumed rather
    # than left in the queue to authorise something later.
    request = session.record.pending_approval
    session.decide(approved=True, decided_by="ops")
    turn = session.run()
    assert turn.decision is not None
    assert turn.decision.request_id == request.id
    decisions = [e for e in session.events() if e.kind is EventKind.DECISION]
    assert [e.body for e in decisions] == ["some-other-request", request.id]
    assert all(e.consumed_at is not None for e in decisions)


def test_approval_lets_the_gated_node_run_exactly_once(manager):
    session = manager.create(GRAPH_NAME)
    session.run()
    request = session.record.pending_approval

    session.decide(approved=True, decided_by="ops", reason="looks right")
    turn = session.run()

    assert turn.status is SessionStatus.IDLE
    assert turn.nodes == (APPROVAL_NODE, "report")
    assert turn.state["log"] == ["ingest", "plan", APPROVAL_NODE, "report"]
    assert turn.state["outcome"] == f"applied: {DEMO_SCRIPT[0]}"
    assert turn.decision is not None
    assert turn.decision.request_id == request.id
    assert turn.decision.decided_by == "ops"
    assert session.record.pending_approval is None

    # The decision is plain JSON in the checkpoint — nothing but JSON goes in —
    # and reads back as the typed model both for callers and for nodes.
    assert isinstance(turn.state["approval_decision"], dict)
    recovered = decision_in(turn.state)
    assert recovered == turn.decision


def test_rejection_skips_the_gated_node_instead_of_undoing_it(manager):
    session = manager.create(GRAPH_NAME)
    session.run()

    session.decide(approved=False, decided_by="ops", reason="not on a friday")
    turn = session.run()

    assert turn.status is SessionStatus.IDLE
    # `apply` is absent from both the executed nodes and the graph's own log:
    # it was never scheduled, not run-and-reverted.
    assert turn.nodes == ("report",)
    assert turn.state["log"] == ["ingest", "plan", "report"]
    assert turn.state["outcome"] == "refused by ops: not on a friday"
    assert session.record.pending_approval is None


def test_deciding_with_nothing_held_is_refused(manager):
    session = manager.create(GRAPH_NAME)
    with pytest.raises(SessionError, match="no outstanding approval"):
        session.decide(approved=True)


# -- approval: more than one gate on one boundary ---------------------------
#
# The worst thing this runtime can do is run an action nobody signed for. A
# superstep can put several gated nodes on the boundary at once, and LangGraph
# resumes a boundary whole: hold one of them and approving that one runs all of
# them. Every test below exists because of that.


class TwoGateState(SessionState):
    emailed: bool = False
    deleted: bool = False
    reported: bool = False


def _register_two_gates(registry: GraphRegistry, ran: list[str] | None = None) -> str:
    """One dispatcher fanning out to two gated nodes, joined by one report.

    Two *different* consequential actions in one superstep, which is the shape
    the bug lives in: "send the email" and "delete the records" are separate
    decisions, and a graph author has no way to make the runtime treat them so.
    """
    log = ran if ran is not None else []

    def build(checkpointer=None):
        def send_email(state: TwoGateState) -> dict:
            log.append("send_email")
            return {"emailed": True}

        def delete_records(state: TwoGateState) -> dict:
            log.append("delete_records")
            return {"deleted": True}

        g = GraphARC(TwoGateState, name="two_gates")
        g.add_node("fan", lambda s: {}, writes=set())
        g.add_node("send_email", send_email, writes={"emailed"})
        g.add_node("delete_records", delete_records, writes={"deleted"})
        g.add_node("report", lambda s: {"reported": True}, writes={"reported"})
        g.add_edge(START, "fan")
        g.add_edge("fan", "send_email")
        g.add_edge("fan", "delete_records")
        g.add_edge("send_email", "report")
        g.add_edge("delete_records", "report")
        g.add_edge("report", END)
        return g.compile(checkpointer=checkpointer)

    register_graph(
        "two_gates",
        build,
        approval_nodes=("send_email", "delete_records"),
        registry=registry,
    )
    return "two_gates"


def test_every_gated_node_in_a_superstep_is_held(root, registry):
    graph = _register_two_gates(registry)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="both")

        turn = session.run()

        assert turn.status is SessionStatus.AWAITING_APPROVAL
        assert turn.pending == ("send_email", "delete_records")
        # One hold per gated node, each with its own request id: a decision
        # names a request, so two nodes sharing one request is two nodes
        # sharing one signature.
        assert [h.node for h in turn.approvals] == ["send_email", "delete_records"]
        assert len({h.id for h in turn.approvals}) == 2
        assert [h.node for h in session.pending_approvals] == [
            "send_email",
            "delete_records",
        ]
        # Neither ran, and neither node's write is anywhere in the checkpoint.
        assert turn.nodes == ("fan",)
        assert "emailed" not in turn.state
        assert "deleted" not in turn.state


def test_approving_one_held_node_does_not_release_the_others(root, registry):
    """The bypass, pinned: one signature must authorise exactly one action."""
    ran: list[str] = []
    graph = _register_two_gates(registry, ran)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="bypass-two")
        turn = session.run()
        holds = {h.node: h.id for h in turn.approvals}

        session.decide(approved=True, decided_by="ops", request_id=holds["send_email"])

        with pytest.raises(ApprovalRequired) as caught:
            session.run()
        assert caught.value.node == "delete_records"
        assert caught.value.request_id == holds["delete_records"]
        # Not one node ran — not even the one that *was* approved. The boundary
        # is released as a whole, so it is settled as a whole.
        assert ran == []
        assert "emailed" not in session.state()
        assert "deleted" not in session.state()
        # The refusal changed nothing: both holds are still on the record and
        # the verdict already given is still queued against its own request.
        assert [h.node for h in session.pending_approvals] == [
            "send_email",
            "delete_records",
        ]
        assert [e.body for e in session.pending_events()] == [holds["send_email"]]


def test_rejecting_one_held_node_does_not_drop_the_others(root, registry):
    ran: list[str] = []
    graph = _register_two_gates(registry, ran)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="reject-one")
        holds = {h.node: h.id for h in session.run().approvals}

        session.decide(
            approved=False, decided_by="ops", request_id=holds["delete_records"]
        )

        # A "no" on one node is not a verdict on the other, and it is not a
        # licence to drop it either: it is still held, still pending, still
        # un-run.
        with pytest.raises(ApprovalRequired) as caught:
            session.run()
        assert caught.value.node == "send_email"
        assert ran == []
        assert session.pending_nodes() == ("send_email", "delete_records")


def test_a_boundary_settles_each_held_node_on_its_own_verdict(root, registry):
    """Approve one, reject the other: exactly one action, and the join still runs."""
    ran: list[str] = []
    graph = _register_two_gates(registry, ran)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="mixed")
        holds = {h.node: h.id for h in session.run().approvals}

        session.decide(approved=True, decided_by="ops", request_id=holds["send_email"])
        session.decide(
            approved=False,
            decided_by="ops",
            reason="not this week",
            request_id=holds["delete_records"],
        )
        turn = session.run()

        assert turn.status is SessionStatus.IDLE
        # The approved node executed; the rejected one was walked past, not
        # executed and undone — its body never ran and its write is absent.
        assert ran == ["send_email"]
        assert turn.nodes == ("send_email", "report")
        assert turn.skipped == ("delete_records",)
        assert turn.state["emailed"] is True
        assert "deleted" not in turn.state
        # Rejecting a branch of a fan-out must not strand what it joins into.
        assert turn.state["reported"] is True
        assert session.pending_approvals == ()
        assert {d.request_id for d in turn.decisions} == set(holds.values())


def test_rejecting_every_held_node_runs_none_of_them(root, registry):
    ran: list[str] = []
    graph = _register_two_gates(registry, ran)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="reject-all")
        holds = {h.node: h.id for h in session.run().approvals}
        for request_id in holds.values():
            session.decide(approved=False, decided_by="ops", request_id=request_id)

        turn = session.run()

        assert turn.status is SessionStatus.IDLE
        assert ran == []
        assert sorted(turn.skipped) == ["delete_records", "send_email"]
        assert "emailed" not in turn.state
        assert "deleted" not in turn.state
        assert turn.state["reported"] is True


def test_deciding_without_naming_a_request_is_refused_while_several_are_held(
    root, registry
):
    """No "the" request when two are open — guessing is how one signature moves two."""
    graph = _register_two_gates(registry)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="ambiguous")
        session.run()

        with pytest.raises(SessionError, match="pass request_id"):
            session.decide(approved=True, decided_by="ops")

        assert len(session.pending_approvals) == 2
        assert session.pending_events() == []


def test_a_turn_that_cannot_settle_a_rejection_fails_without_running_the_node(
    root, registry
):
    """Fail closed. A rejection the runtime cannot record must not become a resume.

    The rejected node's outgoing router is what has to run in its place, and
    here it raises. The turn must stop before the stream opens, leave the hold
    open and the verdict queued, and land the session in `failed` — not leave
    it wedged in `running` having already eaten its input.
    """

    class S(SessionState):
        n: int = 0

    def build(checkpointer=None):
        def router(state) -> str:  # noqa: ANN001 - a local schema breaks hint lookup
            raise RuntimeError("the router is broken")

        g = GraphARC(S, name="badroute")
        g.add_node("gated", lambda s: {"n": s.n + 1}, writes={"n"})
        g.add_node("after", lambda s: {"n": s.n + 10}, writes={"n"})
        g.add_edge(START, "gated")
        g.add_conditional_edge("gated", router, {"go": "after"})
        g.add_edge("after", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("badroute", build, approval_nodes=("gated",), registry=registry)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create("badroute", session_id="badroute")
        turn = session.run()
        assert turn.status is SessionStatus.AWAITING_APPROVAL
        session.send("still queued")
        session.decide(approved=False, decided_by="ops")

        failed = session.run()

        assert failed.status is SessionStatus.FAILED
        assert session.status is SessionStatus.FAILED  # not stuck in `running`
        assert "the router is broken" in failed.error
        assert failed.nodes == ()
        # The gated node did not run, and the hold survived the failure: a
        # retry that found nothing held would resume straight into it, because
        # `interrupt_before` does not stop twice at the same boundary.
        assert session.state().get("n", 0) == 0
        assert [h.node for h in session.pending_approvals] == ["gated"]
        # Nothing was consumed on the way out — the message and the verdict are
        # both still in the queue for whoever picks the session up.
        assert sorted(e.kind for e in session.pending_events()) == [
            EventKind.DECISION,
            EventKind.MESSAGE,
        ]
        # And the gate is still enforced on the next attempt.
        with pytest.raises(ApprovalRequired):
            session.run()


# -- interrupt and steering -------------------------------------------------


class BarrierState(SessionState):
    log: list[str] = []
    steered: str = ""


def _register_barrier(registry: GraphRegistry, entered: Path, release: Path) -> str:
    """A graph whose first node parks until `release` exists.

    The barrier is what makes "interrupt a *running* session" testable without
    guessing at timing: the stop is queued while the node is demonstrably
    inside its body.
    """

    def build(checkpointer=None):
        def first(state: BarrierState) -> dict:
            entered.write_text("in", encoding="utf-8")
            deadline = time.monotonic() + 60
            while not release.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("the release file never appeared")
                time.sleep(0.01)
            return {"log": [*state.log, "first"]}

        def second(state: BarrierState) -> dict:
            return {
                "log": [*state.log, "second"],
                "steered": " ".join(state.inbox),
                "inbox": [],
            }

        g = GraphARC(BarrierState, name="barrier")
        g.add_node("first", first, writes={"log"})
        g.add_node("second", second, writes={"log", "steered", "inbox"})
        g.add_edge(START, "first")
        g.add_edge("first", "second")
        g.add_edge("second", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("barrier", build, registry=registry)
    return "barrier"


@pytest.mark.timeout(120)
def test_another_process_interrupts_a_running_session_at_a_node_boundary(
    tmp_path, root, registry
):
    entered, release = tmp_path / "entered", tmp_path / "release"
    graph = _register_barrier(registry, entered, release)

    with SessionManager(root, registry=registry) as manager:
        session = manager.create(graph, session_id="barrier-session")

        # The child waits until the node is provably running, queues a stop and
        # a steering message, then unblocks it. Nothing in it touches the graph.
        child = _spawn(
            _write_script(
                tmp_path,
                "interrupter",
                """
                entered, release = Path(ARGS[1]), Path(ARGS[2])
                deadline = time.monotonic() + 60
                while not entered.exists():
                    if time.monotonic() > deadline:
                        raise SystemExit("the node never entered the barrier")
                    time.sleep(0.01)
                store = SessionStore(Path(ROOT) / "sessions.sqlite")
                store.push(ARGS[0], EventKind.INTERRUPT, body="stop now")
                store.push(ARGS[0], EventKind.MESSAGE, body="go left")
                store.close()
                release.write_text("go", encoding="utf-8")
                emit(ok=True)
                """,
            ),
            root,
            session.id,
            str(entered),
            str(release),
        )

        turn = session.run()
        interrupter = _result(child)
        assert interrupter["pid"] != os.getpid()

        assert turn.status is SessionStatus.INTERRUPTED
        assert turn.interrupted_by == "stop now"
        # The stop arrived while `first` was inside its body and did not cut it
        # short: boundaries, not pre-emption. It completed and was checkpointed.
        assert turn.nodes == ("first",)
        assert turn.pending == ("second",)
        # The node past the boundary did not run, and its writes are absent
        # from the checkpoint entirely — `steered` has never had a value.
        assert turn.state["log"] == ["first"]
        assert "steered" not in turn.state

        # Steering: the message queued mid-turn is picked up by the node that
        # was pending, which then runs for the first and only time.
        resumed = session.run()
        assert resumed.status is SessionStatus.IDLE
        assert resumed.messages == ("go left",)
        assert resumed.nodes == ("second",)
        assert resumed.state["log"] == ["first", "second"]
        assert resumed.state["steered"] == "go left"


def test_an_interrupt_queued_while_idle_stops_the_next_turn_before_any_node(manager):
    session = manager.create(GRAPH_NAME)
    session.send("do the thing")
    session.interrupt("hold off")

    turn = session.run()

    assert turn.status is SessionStatus.INTERRUPTED
    assert turn.interrupted_by == "hold off"
    assert turn.nodes == ()
    assert session.state() == {}  # not a single node ran

    # `interrupted` does not mean "work is pending": here there is none. The
    # status docstring says so, and this is the case that makes it say so —
    # anything reading `interrupted` as "resumable mid-graph" is wrong.
    assert turn.pending == ()
    assert session.pending_nodes() == ()

    # Consuming input is work too. A turn that ran nothing must not have eaten
    # the queue on its way past — the message is still there for the next one.
    assert turn.messages == ()
    assert [e.body for e in session.pending_events()] == ["do the thing"]

    # The interrupt itself was consumed, so the next turn is not wedged by it.
    resumed = session.run()
    assert resumed.nodes == ("ingest", "plan")
    assert resumed.messages == ("do the thing",)
    assert resumed.state["received"] == ["do the thing"]


class FanOutState(SessionState):
    left: str = ""
    right: str = ""
    joined: str = ""


@pytest.mark.timeout(120)
def test_an_interrupt_during_a_fan_out_keeps_every_branch(root, registry):
    """A stop is read at the *superstep* boundary, not between parallel branches.

    Two branches run in one superstep. The stop is queued while the first has
    finished and the second is still inside its body — the exact instant at
    which a per-branch check tears the step in half. Everything downstream of
    that is what this pins: the second branch's result reaching the checkpoint,
    the node the step was about to schedule still being pending, and the resume
    running that node rather than the whole graph again.
    """
    ran: list[str] = []
    left_entered = threading.Event()
    stop_queued = threading.Event()
    right_released = threading.Event()

    def build(checkpointer=None):
        def left(state: FanOutState) -> dict:
            left_entered.set()
            assert stop_queued.wait(timeout=60), "the stop was never queued"
            ran.append("left")
            return {"left": "L"}

        def right(state: FanOutState) -> dict:
            assert right_released.wait(timeout=60), "the branch was never released"
            ran.append("right")
            return {"right": "R"}

        g = GraphARC(FanOutState, name="fanout")
        g.add_node("fan", lambda s: (ran.append("fan"), {})[1], writes=set())
        g.add_node("left", left, writes={"left"})
        g.add_node("right", right, writes={"right"})
        g.add_node(
            "join",
            lambda s: (ran.append("join"), {"joined": s.left + s.right})[1],
            writes={"joined"},
        )
        g.add_edge(START, "fan")
        g.add_edge("fan", "left")
        g.add_edge("fan", "right")
        g.add_edge("left", "join")
        g.add_edge("right", "join")
        g.add_edge("join", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("fanout", build, registry=registry)
    with SessionManager(root, registry=registry) as manager:
        session = manager.create("fanout", session_id="fanout")

        def stopper() -> None:
            assert left_entered.wait(timeout=60), "the fan-out never started"
            session.interrupt("stop between branches")
            stop_queued.set()
            # `left` returns now; `right` stays parked, so the stop lands with
            # the superstep provably half-finished.
            time.sleep(0.3)
            right_released.set()

        thread = threading.Thread(target=stopper)
        thread.start()
        try:
            turn = session.run()
        finally:
            right_released.set()
            thread.join(timeout=60)

        assert turn.status is SessionStatus.INTERRUPTED
        assert turn.interrupted_by == "stop between branches"
        assert sorted(turn.nodes) == ["fan", "left", "right"]
        # The branch that was still running is not lost: its write is in the
        # checkpoint and the node the step scheduled is pending.
        assert turn.state["left"] == "L"
        assert turn.state["right"] == "R"
        assert turn.pending == ("join",)

        resumed = session.run()

        assert resumed.nodes == ("join",)
        assert resumed.state["joined"] == "LR"
        # Nothing ran twice. A boundary torn mid-step leaves a checkpoint with
        # nothing pending, and the next turn starts the graph over.
        assert ran == ["fan", "left", "right", "join"]


def test_an_interrupt_landing_on_a_gate_boundary_reopens_the_hold(manager):
    """A stop and a gate on the same boundary: the gate wins, and re-opens.

    Resuming a thread whose next node is already the gated one does not
    re-trigger LangGraph's `interrupt_before`. If the runtime reported this
    turn as merely interrupted and dropped the hold, the following turn would
    run the gated node with nothing holding it. So the hold comes back — at the
    price of asking again for a decision that was never acted on.
    """
    session = manager.create(GRAPH_NAME)
    first = session.run()
    session.decide(approved=True, decided_by="ops")
    session.interrupt("stop before applying")

    turn = session.run()

    assert turn.nodes == ()
    assert turn.status is SessionStatus.AWAITING_APPROVAL
    assert turn.pending == (APPROVAL_NODE,)
    assert turn.approval.id != first.approval.id  # a new hold, not the old one
    assert APPROVAL_NODE not in turn.state["log"]

    # The earlier verdict does not carry over to the new hold.
    with pytest.raises(ApprovalRequired):
        session.run()
    session.decide(approved=True, decided_by="ops")
    assert session.run().nodes == (APPROVAL_NODE, "report")


# -- failure ----------------------------------------------------------------


def test_a_turn_that_blows_its_budget_fails_and_is_retryable(manager):
    session = manager.create(GRAPH_NAME)

    turn = session.run(budget=Budget(max_iterations=1))

    assert turn.status is SessionStatus.FAILED
    assert turn.error is not None and "max_iterations" in turn.error
    assert turn.nodes == ("ingest",)
    assert session.record.last_error == turn.error

    # The failure is a stopping point, not a grave: the pending node runs next.
    retried = session.run()
    assert retried.status is SessionStatus.AWAITING_APPROVAL
    assert retried.nodes == ("plan",)
    assert retried.error is None
    assert session.record.last_error is None
    assert retried.state["log"] == ["ingest", "plan"]


# -- cross-process resume ---------------------------------------------------


@pytest.mark.timeout(180)
def test_a_session_survives_the_process_that_started_it(tmp_path, root):
    """The V2 gate: three interpreters, one session, no shared Python object.

    Process A runs to the approval hold and exits. Process B — a different pid,
    a rebuilt graph, an empty registry until it imports the module — resumes by
    id and approves. Process C reads the outcome. `log` proves no node ran
    twice.
    """
    session_id = "cross-process"

    starter = run_child(
        tmp_path,
        "starter",
        """
        with SessionManager(ROOT) as manager:
            session = manager.create(GRAPH_NAME, session_id=ARGS[0])
            session.send("ship the release")
            turn = session.run()
            emit(
                session_id=session.id,
                status=turn.status.value,
                nodes=list(turn.nodes),
                pending=list(turn.pending),
                log=turn.state["log"],
                received=turn.state["received"],
                approval_node=turn.approval.node,
                approval_action=turn.approval.action,
            )
        """,
        root,
        session_id,
    )
    assert starter["status"] == "awaiting_approval"
    assert starter["nodes"] == ["ingest", "plan"]
    assert starter["pending"] == [APPROVAL_NODE]
    assert starter["log"] == ["ingest", "plan"]
    assert starter["received"] == ["ship the release"]
    assert starter["approval_node"] == APPROVAL_NODE

    approver = run_child(
        tmp_path,
        "approver",
        """
        with SessionManager(ROOT) as manager:
            session = manager.resume(ARGS[0])
            record = session.record
            session.decide(approved=True, decided_by="ops", reason="signed off")
            turn = session.run()
            emit(
                status_on_arrival=record.status.value,
                held_node=record.pending_approval.node,
                turns_on_arrival=record.turns,
                status=turn.status.value,
                nodes=list(turn.nodes),
                log=turn.state["log"],
                outcome=turn.state["outcome"],
                received=turn.state["received"],
                plan=turn.state["plan"],
                decision_type=type(turn.state["approval_decision"]).__name__,
                decision=turn.state["approval_decision"],
            )
        """,
        root,
        session_id,
    )
    # The status, the hold and the turn count were read off disk, not memory.
    assert approver["status_on_arrival"] == "awaiting_approval"
    assert approver["held_node"] == APPROVAL_NODE
    assert approver["turns_on_arrival"] == 1
    # Only the nodes that had not run yet ran here.
    assert approver["nodes"] == [APPROVAL_NODE, "report"]
    assert approver["status"] == "idle"
    assert approver["log"] == ["ingest", "plan", APPROVAL_NODE, "report"]
    assert approver["outcome"] == f"applied: {DEMO_SCRIPT[0]}"
    # State written in process A survived into process B untouched.
    assert approver["received"] == ["ship the release"]
    assert approver["plan"] == DEMO_SCRIPT[0]
    # The decision channel holds a plain mapping, not a Pydantic instance:
    # LangGraph warns that unregistered types in a checkpoint "will be blocked
    # in a future version", and a session's whole point is outliving processes.
    assert approver["decision_type"] == "dict"
    assert approver["decision"]["decided_by"] == "ops"

    reader = run_child(
        tmp_path,
        "reader",
        """
        with SessionManager(ROOT) as manager:
            session = manager.resume(ARGS[0])
            emit(
                status=session.status.value,
                turns=session.record.turns,
                pending=list(session.pending_nodes()),
                log=session.state()["log"],
                pids=sorted({h.pid for h in session.history()}),
                reasons=[h.reason for h in session.history()],
            )
        """,
        root,
        session_id,
    )
    assert reader["status"] == "idle"
    assert reader["turns"] == 2
    assert reader["pending"] == []
    assert reader["log"] == ["ingest", "plan", APPROVAL_NODE, "report"]

    pids = {starter["pid"], approver["pid"], reader["pid"], os.getpid()}
    assert len(pids) == 4, "the three children shared a process with each other or the test"
    # The audit trail names both writers, so the resume is visible in the record.
    assert sorted(reader["pids"]) == sorted([starter["pid"], approver["pid"]])
    assert any("holding" in reason for reason in reader["reasons"])


@pytest.mark.timeout(180)
def test_an_interrupted_session_is_resumed_by_a_different_process(tmp_path, root):
    """Interrupt in one interpreter, carry on in another, without repeating work."""
    session_id = "cross-process-interrupt"

    stopper = run_child(
        tmp_path,
        "stopper",
        """
        with SessionManager(ROOT) as manager:
            session = manager.create(GRAPH_NAME, session_id=ARGS[0])
            session.send("first message")
            session.interrupt("not yet")
            turn = session.run()
            emit(status=turn.status.value, nodes=list(turn.nodes), state=turn.state)
        """,
        root,
        session_id,
    )
    assert stopper["status"] == "interrupted"
    assert stopper["nodes"] == []
    assert stopper["state"] == {}

    finisher = run_child(
        tmp_path,
        "finisher",
        """
        with SessionManager(ROOT) as manager:
            session = manager.resume(ARGS[0])
            first = session.run()
            session.decide(approved=True, decided_by="ops")
            second = session.run()
            emit(
                status_on_arrival=first.status.value,
                first_nodes=list(first.nodes),
                messages=list(first.messages),
                status=second.status.value,
                log=second.state["log"],
                received=second.state["received"],
            )
        """,
        root,
        session_id,
    )
    assert finisher["pid"] != stopper["pid"]
    # The message queued before the interrupt was still pending, so the process
    # that picked the session up is the one that delivered it.
    assert finisher["messages"] == ["first message"]
    assert finisher["status_on_arrival"] == "awaiting_approval"
    assert finisher["first_nodes"] == ["ingest", "plan"]
    assert finisher["status"] == "idle"
    assert finisher["log"] == ["ingest", "plan", APPROVAL_NODE, "report"]
    assert finisher["received"] == ["first message"]


@pytest.mark.timeout(180)
def test_the_event_queue_is_durable_across_processes(tmp_path, root):
    """A message queued by a process that then exits still reaches the graph."""
    session_id = "durable-queue"
    with SessionManager(root) as manager:
        manager.create(GRAPH_NAME, session_id=session_id)

    sender = run_child(
        tmp_path,
        "sender",
        """
        store = SessionStore(Path(ROOT) / "sessions.sqlite")
        store.push(ARGS[0], EventKind.MESSAGE, body="from another process")
        pending = [e.body for e in store.pending(ARGS[0])]
        store.close()
        emit(pending=pending)
        """,
        root,
        session_id,
    )
    assert sender["pending"] == ["from another process"]

    with SessionManager(root) as manager:
        session = manager.resume(session_id)
        turn = session.run()
    assert turn.messages == ("from another process",)
    assert turn.state["received"] == ["from another process"]


def test_the_store_is_the_only_place_session_status_lives(root, tmp_path):
    """Two managers over one directory see one session, not two.

    Concurrent sessions are isolated by id, but a session is not isolated from
    itself: the second manager's view is the first manager's writes.
    """
    with SessionManager(root) as first, SessionManager(root) as second:
        session = first.create(GRAPH_NAME, session_id="shared")
        session.run()
        mirror = second.resume("shared")
        assert mirror.status is SessionStatus.AWAITING_APPROVAL
        assert mirror.pending_nodes() == (APPROVAL_NODE,)

        mirror.decide(approved=True, decided_by="the other manager")
        turn = mirror.run()
        assert turn.status is SessionStatus.IDLE
        assert session.status is SessionStatus.IDLE  # same row, not a copy


def test_sequential_sessions_do_not_share_state(manager):
    """Isolation by id, one turn after another. Concurrency is the test below."""
    a = manager.create(GRAPH_NAME, session_id="a")
    b = manager.create(GRAPH_NAME, session_id="b")
    a.send("for a")
    b.send("for b")

    a.run()
    b.run()

    assert a.state()["received"] == ["for a"]
    assert b.state()["received"] == ["for b"]
    assert a.thread_id != b.thread_id
    assert a.graph is not b.graph  # one graph instance per session


@pytest.mark.timeout(180)
def test_sessions_run_concurrently_on_separate_threads(root, registry):
    """§6.6, proved by a barrier rather than by two turns taken in a row.

    `a.run(); b.run()` would pass on a runtime that took a global lock and ran
    every session one at a time — it asserts nothing about concurrency. A
    barrier can only be cleared if all six turns are inside a node at the same
    moment: if anything in the session layer serialised them, the first five
    would sit at the barrier until it times out and the test fails.

    What is being isolated is per-session: six store rows, six checkpoint
    threads and six graph instances, over one `SessionManager` and therefore
    one SQLite connection each for the store and the checkpointer.
    """
    count = 6
    barrier = threading.Barrier(count)
    inside: list[int] = []
    threads: dict[str, int] = {}
    guard = threading.Lock()

    class ConcurrentState(SessionState):
        received: list[str] = []
        rank: int = -1

    def build(checkpointer=None):
        def work(state: ConcurrentState) -> dict:
            # Blocks until all six sessions have reached this line. Raises
            # BrokenBarrierError — and fails the turn — if they never do.
            rank = barrier.wait(timeout=90)
            with guard:
                inside.append(rank)
            return {
                "received": [*state.received, *state.inbox],
                "inbox": [],
                "rank": rank,
            }

        g = GraphARC(ConcurrentState, name="concurrent")
        g.add_node("work", work, writes={"received", "inbox", "rank"})
        g.add_edge(START, "work")
        g.add_edge("work", END)
        return g.compile(checkpointer=checkpointer)

    register_graph("concurrent", build, registry=registry)
    with SessionManager(root, registry=registry) as manager:
        sessions = []
        for index in range(count):
            session = manager.create("concurrent", session_id=f"s{index}")
            session.send(f"for s{index}")
            sessions.append(session)

        def drive(session):
            with guard:
                threads[session.id] = threading.get_ident()
            return session.run()

        with ThreadPoolExecutor(max_workers=count) as pool:
            turns = list(pool.map(drive, sessions))

        # The barrier cleared, so all six were executing at once.
        assert sorted(inside) == list(range(count))
        assert len(set(threads.values())) == count, "the pool reused a thread"
        assert threading.get_ident() not in threads.values()

        for index, (session, turn) in enumerate(zip(sessions, turns, strict=True)):
            assert turn.status is SessionStatus.IDLE
            # Each session saw its own message and nobody else's, and each got
            # its own checkpoint thread and its own graph object.
            assert turn.state["received"] == [f"for s{index}"]
            assert session.state()["received"] == [f"for s{index}"]
            assert session.record.turns == 1
            assert session.record.runner_pid is None
        assert len({s.thread_id for s in sessions}) == count
        assert len({id(s.graph) for s in sessions}) == count
        # Every session's rank is distinct: six barrier slots, six sessions.
        assert sorted(s.state()["rank"] for s in sessions) == list(range(count))


def test_a_hold_written_by_an_older_build_still_reads_back(manager):
    """The column holds a JSON list now; a store on disk outlives the build.

    Rows written before a superstep could hold more than one node carry a bare
    object. Failing to read one would be a session that comes back from disk
    with its gate silently missing.
    """
    session = manager.create(GRAPH_NAME, session_id="legacy")
    session.run()
    hold = session.pending_approval
    assert hold is not None

    with manager.store._transaction() as conn:  # write the old shape by hand
        conn.execute(
            "UPDATE sessions SET pending_approval = ? WHERE id = ?",
            (hold.model_dump_json(), session.id),
        )

    assert session.pending_approvals == (hold,)
    assert session.pending_approval == hold
    with pytest.raises(ApprovalRequired):
        session.run()


def test_the_store_file_lives_where_the_manager_says_it_does(root):
    with SessionManager(root) as manager:
        manager.create(GRAPH_NAME)
    assert (root / STORE_FILENAME).exists()
    store = SessionStore(root / STORE_FILENAME)
    assert len(store.list()) == 1
    store.close()
