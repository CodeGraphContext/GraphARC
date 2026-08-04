"""The approval gate: the loop's pause, the file handshake, the CLI answer.

The chain under test: `GovernedLoop(approval=...)` parks each admitted round;
`file_approval` implements the callback as a request/decision file pair next
to the trace; `grapharc approve` writes the decision. Every hop is exercised
without Slack, a server, or a model backend — the planner is scripted.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from grapharc.cli.main import main
from grapharc.examples.plan_incident import (
    IncidentState,
    build_loop,
    scripted_planner_replies,
)
from grapharc.observe.trace import TraceRecorder
from grapharc.planner import LoopStop
from grapharc.planner.approval_file import (
    DECISION_FILENAME,
    REQUEST_FILENAME,
    file_approval,
    read_request,
    write_decision,
)
from grapharc.testing import ScriptedChatModel


def _loop(tmp_path, approval):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=scripted_planner_replies())
    return build_loop(model, trace=trace, approval=approval), trace


# -- the loop's gate --------------------------------------------------------


def test_an_approved_round_executes(tmp_path):
    calls = []

    def gate(proposal, verdict):
        calls.append(proposal.fingerprint())
        return "approved"

    loop, trace = _loop(tmp_path, gate)
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.GOAL_MET
    assert len(calls) == 1  # round 1 was rejected pre-approval; only round 2 asked
    phases = [e.phase for e in trace.read_events(result.run_id)]
    assert "approval_request" in phases and "approval_response" in phases
    assert "start" in phases  # nodes actually ran


def test_a_denied_round_stops_the_run_before_any_node_starts(tmp_path):
    loop, trace = _loop(tmp_path, lambda p, v: "denied")
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.APPROVAL_DENIED
    events = trace.read_events(result.run_id)
    assert not any(e.phase == "start" for e in events), "a node ran without approval"
    responses = [e for e in events if e.phase == "approval_response"]
    assert responses[-1].state_delta["decision"] == "denied"


def test_an_unanswered_gate_is_a_timeout_stop(tmp_path):
    loop, _ = _loop(tmp_path, lambda p, v: "timeout")
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.APPROVAL_TIMEOUT


def test_a_gate_that_raises_fails_closed_as_denied(tmp_path):
    def broken(proposal, verdict):
        raise RuntimeError("gate exploded")

    loop, trace = _loop(tmp_path, broken)
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.APPROVAL_DENIED
    assert not any(e.phase == "start" for e in trace.read_events(result.run_id))


def test_the_request_event_carries_the_plan_being_asked_about(tmp_path):
    seen = {}

    def gate(proposal, verdict):
        seen["fingerprint"] = proposal.fingerprint()
        return "denied"

    loop, trace = _loop(tmp_path, gate)
    result = loop.run("goal", IncidentState(goal="g"))
    request = next(
        e for e in trace.read_events(result.run_id) if e.phase == "approval_request"
    )
    assert request.state_delta["fingerprint"] == seen["fingerprint"]
    assert request.state_delta["nodes"], "the audited question must name the nodes"


# -- the file handshake -----------------------------------------------------


class _Plan:
    """Just enough proposal shape for the handshake."""

    proposal_id = "p-123"
    rationale = "because"
    nodes = ()
    edges = ()

    @staticmethod
    def fingerprint() -> str:
        return "fp-abc"


def test_file_approval_honours_a_matching_decision(tmp_path):
    gate = file_approval(tmp_path, timeout_seconds=5.0, poll_seconds=0.02)

    def answer():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if (tmp_path / REQUEST_FILENAME).exists():
                write_decision(tmp_path, fingerprint="fp-abc", decision="approved")
                return
            time.sleep(0.02)

    thread = threading.Thread(target=answer)
    thread.start()
    decision = gate(_Plan(), None)
    thread.join()
    assert decision == "approved"
    assert not (tmp_path / REQUEST_FILENAME).exists(), "the handshake must be consumed"
    assert not (tmp_path / DECISION_FILENAME).exists()


def test_a_stale_decision_naming_another_plan_is_discarded(tmp_path):
    # The decision predates the question and names a different fingerprint —
    # exactly the file a previous round could have left behind.
    write_decision(tmp_path, fingerprint="fp-OLD", decision="approved")
    gate = file_approval(tmp_path, timeout_seconds=0.3, poll_seconds=0.02)
    assert gate(_Plan(), None) == "timeout"


def test_a_pre_existing_decision_cannot_pre_approve(tmp_path):
    # Even with the RIGHT fingerprint, a decision written before the request
    # is deleted when the gate opens: the answer must follow the question.
    write_decision(tmp_path, fingerprint="fp-abc", decision="approved")
    gate = file_approval(tmp_path, timeout_seconds=0.3, poll_seconds=0.02)
    assert gate(_Plan(), None) == "timeout"


def test_silence_is_a_timeout(tmp_path):
    gate = file_approval(tmp_path, timeout_seconds=0.2, poll_seconds=0.02)
    assert gate(_Plan(), None) == "timeout"


# -- the CLI answer ---------------------------------------------------------


def _pending(tmp_path) -> Path:
    (tmp_path / "trace.jsonl").touch()
    (tmp_path / REQUEST_FILENAME).write_text(
        json.dumps(
            {
                "proposal_id": "p-9",
                "fingerprint": "fp-9",
                "nodes": ["triage", "verify"],
                "edges": [],
            }
        )
    )
    return tmp_path / "trace.jsonl"


def test_approve_writes_the_decision_quoting_the_request(tmp_path, capsys):
    trace_path = _pending(tmp_path)
    assert main(["approve", str(trace_path)]) == 0
    decision = json.loads((tmp_path / DECISION_FILENAME).read_text())
    assert decision == {"fingerprint": "fp-9", "decision": "approved"}
    out = capsys.readouterr().out
    assert "approved" in out and "triage" in out


def test_approve_deny_writes_a_denial(tmp_path, capsys):
    _pending(tmp_path)
    assert main(["approve", str(tmp_path), "--deny"]) == 0
    decision = json.loads((tmp_path / DECISION_FILENAME).read_text())
    assert decision["decision"] == "denied"


def test_approve_with_nothing_pending_is_a_negative_answer(tmp_path, capsys):
    (tmp_path / "trace.jsonl").touch()
    assert main(["approve", str(tmp_path / "trace.jsonl")]) == 1
    # Failure speaks on stderr: the text-mode contract.
    assert "nothing is waiting" in capsys.readouterr().err


def test_approve_on_a_missing_directory_cannot_run(tmp_path, capsys):
    assert main(["approve", str(tmp_path / "absent" / "trace.jsonl")]) == 2


def test_read_request_tolerates_junk(tmp_path):
    (tmp_path / REQUEST_FILENAME).write_text("not json {")
    assert read_request(tmp_path) is None


# -- the whole handshake, in-process ---------------------------------------


def test_plan_approve_and_answer_end_to_end(tmp_path):
    """The paused loop and the CLI answer, meeting in one directory."""
    trace_dir = tmp_path / "run"
    trace_dir.mkdir()
    gate = file_approval(trace_dir, timeout_seconds=10.0, poll_seconds=0.02)
    loop, trace = _loop(tmp_path, gate)

    def answer_when_asked():
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if (trace_dir / REQUEST_FILENAME).exists():
                assert main(["approve", str(trace_dir)]) == 0
                return
            time.sleep(0.02)

    thread = threading.Thread(target=answer_when_asked)
    thread.start()
    result = loop.run("goal", IncidentState(goal="g"))
    thread.join()
    assert result.stop is LoopStop.GOAL_MET


# -- fixes from the bug hunt ------------------------------------------------


def test_a_halt_during_the_park_stops_the_run_as_human_stopped(tmp_path):
    """`request_halt` used to be inert while parked — and an approval racing
    the halt still executed the round. The halt must win, promptly."""
    loop_holder = {}

    def gate(proposal, verdict, should_stop=None):
        # Simulate the operator pulling the cord mid-park; a halt-aware gate
        # (file_approval is one) polls `should_stop` and returns early.
        loop_holder["loop"].request_halt("operator pulled the cord")
        assert should_stop is not None and should_stop()
        return "approved"  # even a racing approval must not execute now

    loop, trace = _loop(tmp_path, gate)
    loop_holder["loop"] = loop
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.HUMAN_STOPPED
    assert "operator pulled the cord" in result.detail
    assert not any(e.phase == "start" for e in trace.read_events(result.run_id))


def test_file_approval_returns_early_when_should_stop_fires(tmp_path):
    gate = file_approval(tmp_path, timeout_seconds=30.0, poll_seconds=0.02)
    started = time.monotonic()
    decision = gate(_Plan(), None, should_stop=lambda: True)
    assert decision == "timeout"
    assert time.monotonic() - started < 5.0, "the gate waited out its full timeout"


def test_the_approval_wait_does_not_burn_max_seconds(tmp_path):
    """A human who thinks longer than `max_seconds` used to hand the approved
    round a negative budget — approved, then dead before its first node."""
    from grapharc.runtime.budget import Budget

    def slow_yes(proposal, verdict):
        time.sleep(0.4)
        return "approved"

    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=scripted_planner_replies())
    loop = build_loop(
        model, trace=trace, approval=slow_yes, budget=Budget(max_seconds=0.3)
    )
    result = loop.run("goal", IncidentState(goal="g"))
    assert result.stop is LoopStop.GOAL_MET, (result.stop, result.detail)


def test_two_runs_sharing_a_directory_cannot_destroy_each_others_decision(tmp_path):
    """A mismatched-fingerprint decision used to be *unlinked* by the other
    run's poll — the human's approval was consumed by the wrong gate and both
    runs timed out."""
    write_decision(tmp_path, fingerprint="fp-OTHER-RUN", decision="approved")
    gate = file_approval(tmp_path, timeout_seconds=0.3, poll_seconds=0.02)
    assert gate(_Plan(), None) == "timeout"
    # The other run's decision is still there for the other run to consume.
    decision = json.loads((tmp_path / DECISION_FILENAME).read_text())
    assert decision["fingerprint"] == "fp-OTHER-RUN"


def test_handshake_files_are_written_atomically(tmp_path):
    """No reader may ever see half a request: written to a tmp then renamed."""
    from grapharc.planner.approval_file import _write_atomically

    target = tmp_path / "approval-request.json"
    _write_atomically(target, {"fingerprint": "fp"})
    assert json.loads(target.read_text()) == {"fingerprint": "fp"}
    assert not list(tmp_path.glob("*.tmp"))


def test_plan_approve_in_json_mode_emits_one_document(tmp_path, capsys):
    """The park notice used to print ahead of the document, so nothing parsed.

    `--approve` is the flag most likely to be driven unattended: a script starts
    a gated plan, a human answers out of band, the script reads the result. That
    is exactly the combination whose output could not be loaded.
    """
    trace = tmp_path / "run" / "trace.jsonl"

    code = main(
        ["plan", "ship it", "--scripted", "--go", "--approve", "--approval-timeout", "0.2",
         "--trace", str(trace), "--json"]
    )
    captured = capsys.readouterr()

    assert code == 1, "an unanswered gate is a negative answer, not a crash"
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "not approved" in payload["detail"]


def test_plan_approve_in_text_mode_still_announces_how_to_answer(tmp_path, capsys):
    """Silencing the notice in JSON mode must not silence it for a human."""
    trace = tmp_path / "run" / "trace.jsonl"

    main([
        "plan",
        "ship it",
        "--scripted",
        "--go",
        "--approve",
        "--approval-timeout",
        "0.2",
        "--trace",
        str(trace),
    ])

    assert "grapharc approve" in capsys.readouterr().out
