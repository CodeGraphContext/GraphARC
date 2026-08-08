"""Supervised work from Slack: propose a graph, show it, run it only if approved.

The chain under test is the one a person actually uses from a phone:

    /grapharc plan "<goal>" --go
      -> the gate forces --approve, so nothing executes unattended
      -> the run parks and the bot renders the *proposed graph* into the message
      -> Approve / Deny buttons answer the file handshake by fingerprint
      -> the same run then executes what was approved, in one command

Nothing here needs Slack, a token, or a model: the sink is a list, the buttons
are `handle_approval_action` called directly, and the planner is the shipped
scripted stand-in. The end-to-end tests do run the real CLI subprocess, because
the parking and the execution are the behaviour being claimed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from grapharc.observe.replay import replay
from grapharc.observe.trace import TailRecorder, TraceRecorder
from grapharc.planner.approval_file import write_request
from grapharc.slack.bot import (
    APPROVE_ACTION,
    DENY_ACTION,
    approval_blocks,
    handle_approval_action,
    handle_text_live,
)
from grapharc.slack.command import (
    APPROVAL_WAIT_CAP_SECONDS,
    DEFAULT_PLAN_REGISTRY,
    SlackCommandError,
    effective_timeout,
    mutating_kinds_for,
    parse_command,
)
from grapharc.slack.config import SlackBotConfig
from grapharc.slack.live import (
    ApprovalPrompt,
    pending_approval_prompt,
    plan_lines,
    render_progress,
)


def _config(tmp_path, **overrides) -> SlackBotConfig:
    values = dict(
        bot_token="xoxb-x",
        app_token="xapp-x",
        workdir=tmp_path,
        timeout_seconds=60.0,
        work_timeout_seconds=180.0,
        live_interval_seconds=0.05,
        allow_model=True,
        allow_agent=True,
    )
    values.update(overrides)
    return SlackBotConfig(**values)


def _parse(text: str, tmp_path, **kwargs) -> list[str]:
    values = dict(
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
        timeout_seconds=60.0,
        work_timeout_seconds=180.0,
    )
    values.update(kwargs)
    return parse_command(text, **values)


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


# -- the gate: executing from Slack always goes through a human ---------------


def test_go_is_reachable_from_slack_at_all(tmp_path):
    """Without this the supervised loop had no exit.

    A parked `plan --approve` was approved by a human and then returned
    "awaiting `grapharc go`" — into a subcommand this gate does not carry. The
    graph could be proposed and authorised from Slack and never run from there.
    """
    argv = _parse('plan "fix the flaky test" --scripted --go', tmp_path)
    assert "--go" in argv


def test_go_forces_approve_even_when_the_requester_did_not_ask_for_it(tmp_path):
    """The safety rule the rest of this file rests on.

    Anyone in the workspace can type into the bot. `--go` is the difference
    between proposing a graph and running one on the host, so from Slack it
    always parks first — the requester does not get to skip that by omitting
    the flag.
    """
    argv = _parse('plan "do the thing" --scripted --go', tmp_path)
    assert "--approve" in argv


def test_approve_is_forced_for_every_registry_not_only_the_agent_one(tmp_path):
    for registry in (
        "grapharc.examples.plan_incident:build_registry",
        "grapharc.examples.plan_docs:build_registry",
    ):
        argv = _parse(f'plan "g" --scripted --go --registry {registry}', tmp_path)
        assert "--approve" in argv, registry


def test_plan_without_go_is_not_forced_to_park(tmp_path):
    """Proposing is not executing; only `--go` earns the gate."""
    argv = _parse('plan "g" --scripted', tmp_path)
    assert "--approve" not in argv


# -- two budgets: a reader's two minutes is a work run's SIGKILL --------------


def test_a_reader_keeps_the_short_timeout(tmp_path):
    argv = _parse("metrics t.jsonl r1", tmp_path)
    assert effective_timeout(argv, timeout_seconds=60.0, work_timeout_seconds=180.0) == 60.0


def test_a_command_that_executes_gets_the_work_budget(tmp_path):
    for text in ('plan "g" --scripted --go', "agent do-something --model mock/x"):
        argv = _parse(text, tmp_path)
        assert (
            effective_timeout(argv, timeout_seconds=60.0, work_timeout_seconds=180.0) == 180.0
        ), text


def test_a_parked_go_leaves_most_of_the_clock_for_the_work_it_authorises(tmp_path):
    """The human gets a third; the run they approve gets the rest.

    A plan-only park may have half the clock — nothing runs after it. With
    `--go`, saying yes is the *start* of the work, and a wait that ate the
    budget would leave an approved run to be killed before its first node.
    """
    argv = _parse('plan "g" --scripted --go', tmp_path)
    wait = float(_flag_value(argv, "--approval-timeout"))
    assert wait == pytest.approx(60.0)  # 180s work budget / 3
    assert wait < 180.0 - 10.0


def test_a_plan_only_park_may_have_half_the_clock(tmp_path):
    argv = _parse('plan "g" --scripted --approve', tmp_path)
    assert float(_flag_value(argv, "--approval-timeout")) == pytest.approx(30.0)


def test_the_human_wait_is_capped_however_large_the_work_budget_is(tmp_path):
    """A 30-minute work budget must not be spendable entirely on nobody answering."""
    argv = _parse('plan "g" --scripted --go', tmp_path, work_timeout_seconds=7200.0)
    assert float(_flag_value(argv, "--approval-timeout")) == APPROVAL_WAIT_CAP_SECONDS


def test_a_wait_longer_than_the_budget_is_refused_not_run(tmp_path):
    """The injection only fires when the flag is absent, so a supplied one must be checked.

    A park that outlives the runner is not an `approval_timeout` the run can
    report — it is a kill through the middle of the wait, and the requester
    gets a truncated message instead of an answer.
    """
    with pytest.raises(SlackCommandError, match="does not fit this command's budget"):
        _parse('plan "g" --scripted --go --approval-timeout 100000', tmp_path)


def test_a_shorter_wait_than_the_default_is_the_requesters_to_choose(tmp_path):
    argv = _parse('plan "g" --scripted --go --approval-timeout 25', tmp_path)
    assert _flag_value(argv, "--approval-timeout") == "25"


def test_a_wait_that_is_not_a_number_is_a_refusal(tmp_path):
    with pytest.raises(SlackCommandError, match="wants a number of seconds"):
        _parse('plan "g" --scripted --go --approval-timeout soon', tmp_path)


def test_a_delegated_agent_ceiling_comes_from_the_work_budget(tmp_path):
    """`--max-seconds` reports cleanly; the runner's timeout kills. Order matters."""
    argv = _parse("agent do-something --model mock/x", tmp_path)
    assert float(_flag_value(argv, "--max-seconds")) == pytest.approx(170.0)


# -- which kinds change things, for the person deciding ----------------------


def test_the_shipped_registries_report_their_own_mutating_kinds(tmp_path):
    incident = _parse(
        'plan "g" --scripted --registry grapharc.examples.plan_incident:build_registry', tmp_path
    )
    assert mutating_kinds_for(incident, tmp_path) == frozenset({"deploy"})

    stdlib = _parse('plan "g" --registry grapharc.stdlib:build_registry --model mock/x', tmp_path)
    assert mutating_kinds_for(stdlib, tmp_path) == frozenset({"apply_change"})


def test_a_registry_this_process_cannot_vouch_for_is_assumed_to_mutate(tmp_path):
    """`grapharc.toml` may point the CLI at any module; the bot will not import it.

    None means "assume every kind changes things" — the same fail-closed
    reading `plan` takes for its own plan file. Guessing the shipped default
    here would put a reassuring `·` next to a node from a registry nobody in
    this process has read.
    """
    (tmp_path / "grapharc.toml").write_text(
        '[grapharc]\nregistry = "registry.py:build_registry"\n', encoding="utf-8"
    )
    argv = _parse('plan "g" --scripted', tmp_path)
    assert mutating_kinds_for(argv, tmp_path) is None


def test_the_configured_registry_is_read_rather_than_assumed(tmp_path):
    (tmp_path / "grapharc.toml").write_text(
        '[grapharc]\nregistry = "grapharc.examples.plan_docs:build_registry"\n',
        encoding="utf-8",
    )
    argv = _parse('plan "g" --scripted', tmp_path)
    assert mutating_kinds_for(argv, tmp_path) == frozenset()


def test_the_gates_default_registry_matches_the_clis(tmp_path):
    """Two copies of one string; this is the test that keeps them one string."""
    from grapharc.cli.plan import DEFAULT_REGISTRY

    assert DEFAULT_PLAN_REGISTRY == DEFAULT_REGISTRY


# -- the plan a human is shown ------------------------------------------------


def _approval_event(tmp_path, **delta):
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    payload = {
        "round": 1,
        "proposal_id": "p1",
        "fingerprint": "abc123",
        "nodes": ["look", "fix_it", "check"],
        "kinds": ["investigate", "apply_change", "verify"],
        "edges": [["__start__", "look"], ["look", "fix_it"], ["fix_it", "check"]],
        "rationale": "read the failing test, then repair it",
        "worst_case_tokens": 4200,
        "worst_case_complete": True,
        "goal": "fix the flaky test",
    }
    payload.update(delta)
    recorder.event(
        run_id="r1",
        graph="loop",
        node="loop:approval",
        phase="approval_request",
        step=0,
        state_delta=payload,
    )
    return next(
        e for e in recorder.read_events("r1") if e.phase == "approval_request"
    )


def test_the_proposed_graph_is_in_the_message_not_behind_a_link(tmp_path):
    """The person deciding is on a phone. "The graph is over there" is not an answer."""
    lines = plan_lines(_approval_event(tmp_path), mutating_kinds=frozenset({"apply_change"}))
    body = "\n".join(lines)
    assert "look (investigate)" in body
    assert "fix_it (apply_change)" in body
    assert "look → fix_it" in body
    assert "read the failing test" in body
    assert "4200 tok" in body


def test_a_node_that_can_change_files_is_marked(tmp_path):
    """`fix_it` is a name a planner chose; `apply_change` is what the gate governs."""
    lines = plan_lines(_approval_event(tmp_path), mutating_kinds=frozenset({"apply_change"}))
    assert any(line.strip().startswith("✎") and "fix_it" in line for line in lines)
    assert any(line.strip().startswith("·") and "look" in line for line in lines)


def test_an_undeclared_registry_marks_every_node_and_says_why(tmp_path):
    lines = plan_lines(_approval_event(tmp_path), mutating_kinds=None)
    body = "\n".join(lines)
    assert "assume all do" in body
    assert body.count("✎") == 3


def test_a_kind_equal_to_its_name_is_not_repeated(tmp_path):
    event = _approval_event(tmp_path, nodes=["verify"], kinds=["verify"], edges=[])
    assert any(line.strip() == "· verify" for line in plan_lines(event))


def test_an_incomplete_worst_case_says_it_is_a_lower_bound(tmp_path):
    event = _approval_event(tmp_path, worst_case_complete=False)
    assert "lower bound" in "\n".join(plan_lines(event))


def test_an_empty_plan_reads_as_an_empty_plan_not_a_broken_renderer(tmp_path):
    event = _approval_event(
        tmp_path, nodes=[], kinds=[], edges=[], rationale="", worst_case_tokens=0
    )
    assert "no further work" in "\n".join(plan_lines(event))


def test_a_parked_run_shows_the_plan_and_says_nothing_has_run(tmp_path):
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(
        run_id="r1", graph="loop", node="loop:topology", phase="topology", step=0,
        state_delta={"nodes": ["look", "fix_it"], "goal": "fix the flaky test", "round": 1},
    )
    recorder.event(
        run_id="r1", graph="loop", node="loop:approval", phase="approval_request", step=0,
        state_delta={
            "round": 1, "proposal_id": "p1", "fingerprint": "abc123",
            "nodes": ["look", "fix_it"], "kinds": ["investigate", "apply_change"],
            "edges": [["look", "fix_it"]], "rationale": "why not",
            "goal": "fix the flaky test",
        },
    )
    text = render_progress(
        replay(recorder, "r1"),
        argv=["plan", "g", "--trace", "runs/t.jsonl"],
        elapsed_s=3.0,
        mutating_kinds=frozenset({"apply_change"}),
    )
    assert "waiting for approval" in text
    assert "nothing above has run yet" in text
    assert "look → fix_it" in text
    # Not the pending-node column: every node of a parked plan is pending, so a
    # stack of ⬜ says nothing about what is being authorised.
    assert "⬜" not in text


# -- the buttons --------------------------------------------------------------


def _park(tmp_path, fingerprint="abc123") -> Path:
    directory = tmp_path / "slack-runs" / "one"
    directory.mkdir(parents=True)
    write_request(
        directory,
        {"proposal_id": "p1", "fingerprint": fingerprint, "nodes": ["a"], "edges": []},
    )
    return directory


def _value(directory: Path, tmp_path, fingerprint="abc123") -> str:
    return json.dumps(
        {"dir": str(directory.relative_to(tmp_path)), "fp": fingerprint}
    )


def test_a_pending_run_gets_buttons_and_a_finished_one_does_not(tmp_path):
    prompt = ApprovalPrompt(directory="slack-runs/one", fingerprint="abc123")
    blocks = approval_blocks("the plan", prompt)
    action_ids = [
        element["action_id"] for element in blocks[1]["elements"]
    ]
    assert action_ids == [APPROVE_ACTION, DENY_ACTION]
    assert approval_blocks("done", None) is None


def test_approving_writes_the_decision_the_parked_run_is_waiting_for(tmp_path):
    directory = _park(tmp_path)
    answer = handle_approval_action(
        _value(directory, tmp_path), _config(tmp_path), deny=False, actor="U123"
    )
    assert "approved" in answer and "U123" in answer
    decision = json.loads((directory / "approval-decision.json").read_text())
    assert decision == {"fingerprint": "abc123", "decision": "approved"}


def test_denying_writes_a_denial(tmp_path):
    directory = _park(tmp_path)
    handle_approval_action(_value(directory, tmp_path), _config(tmp_path), deny=True)
    decision = json.loads((directory / "approval-decision.json").read_text())
    assert decision["decision"] == "denied"


def test_a_button_from_an_earlier_round_cannot_approve_the_plan_that_replaced_it(tmp_path):
    """The load-bearing check.

    A parked run rewrites its request every round. A message scrolled back to
    from round 1 carries round 1's fingerprint, and clicking Approve on it
    would authorise a graph nobody read.
    """
    directory = _park(tmp_path, fingerprint="round-two")
    answer = handle_approval_action(
        _value(directory, tmp_path, fingerprint="round-one"), _config(tmp_path), deny=False
    )
    assert "earlier plan" in answer
    assert not (directory / "approval-decision.json").exists()


def test_a_button_naming_a_directory_outside_the_workdir_is_refused(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    write_request(outside, {"fingerprint": "abc123"})
    answer = handle_approval_action(
        json.dumps({"dir": "../elsewhere", "fp": "abc123"}), _config(tmp_path), deny=False
    )
    assert "escapes" in answer
    assert not (outside / "approval-decision.json").exists()


def test_a_click_on_a_run_that_already_moved_on_says_so(tmp_path):
    directory = tmp_path / "slack-runs" / "gone"
    directory.mkdir(parents=True)
    answer = handle_approval_action(
        _value(directory, tmp_path), _config(tmp_path), deny=False
    )
    assert "nothing is waiting" in answer


def test_a_malformed_button_value_is_an_answer_not_an_exception(tmp_path):
    for raw in ("", "not json", "{}", '{"dir": "x"}'):
        answer = handle_approval_action(raw, _config(tmp_path), deny=False)
        assert "carried nothing" in answer, raw


def test_the_prompt_is_relative_and_disappears_once_answered(tmp_path):
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    argv = ["plan", "g", "--trace", "slack-runs/one/trace.jsonl"]
    delta = {"round": 1, "proposal_id": "p1", "fingerprint": "abc123", "nodes": [], "edges": []}
    recorder.event(
        run_id="r1", graph="loop", node="loop:approval",
        phase="approval_request", step=0, state_delta=delta,
    )
    prompt = pending_approval_prompt(replay(recorder, "r1"), argv, tmp_path)
    assert prompt == ApprovalPrompt(
        directory="slack-runs/one", fingerprint="abc123", round_number=1
    )

    recorder.event(
        run_id="r1", graph="loop", node="loop:approval",
        phase="approval_response", step=0,
        state_delta={"round": 1, "proposal_id": "p1", "decision": "approved"},
    )
    assert pending_approval_prompt(replay(recorder, "r1"), argv, tmp_path) is None


def test_a_request_without_a_fingerprint_draws_no_button(tmp_path):
    """An unbound Approve button is exactly what this design refuses to ship."""
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(
        run_id="r1", graph="loop", node="loop:approval", phase="approval_request",
        step=0, state_delta={"round": 1, "nodes": [], "edges": []},
    )
    argv = ["plan", "g", "--trace", "slack-runs/one/trace.jsonl"]
    assert pending_approval_prompt(replay(recorder, "r1"), argv, tmp_path) is None


# -- end to end: the whole supervised loop, one Slack message ----------------


class RecordingSink:
    """A `LiveSink` that keeps every edit and the blocks it was drawn with."""

    def __init__(self) -> None:
        self.posted: list[str] = []
        self.updated: list[tuple[str, list | None]] = []
        self._lock = threading.Lock()

    def post(self, text: str):
        self.posted.append(text)
        return ("C1", "1.0")

    def update(self, handle, text: str, blocks=None) -> bool:
        with self._lock:
            self.updated.append((text, blocks))
        return True

    def wait_for_buttons(self, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for _text, blocks in self.updated:
                    if blocks:
                        return blocks
            time.sleep(0.05)
        return None


def test_a_slack_go_parks_shows_the_graph_is_approved_by_button_and_then_runs(tmp_path):
    """The whole claim, through the real CLI subprocess.

    One Slack message proposes a graph; the run parks without executing; the
    message carries the graph and two buttons; a click answers the handshake by
    fingerprint; the same run then executes what was approved.
    """
    sink = RecordingSink()
    config = _config(tmp_path)
    approved: list[str] = []

    def click_when_parked() -> None:
        blocks = sink.wait_for_buttons()
        assert blocks is not None, "the parked run never offered a button"
        value = blocks[1]["elements"][0]["value"]
        approved.append(handle_approval_action(value, config, deny=False, actor="U9"))

    clicker = threading.Thread(target=click_when_parked, daemon=True)
    clicker.start()
    handle_text_live('plan "investigate the checkout outage" --scripted --go', config, sink)
    clicker.join(timeout=10.0)

    assert approved and "approved" in approved[0]

    parked = [text for text, _ in sink.updated if "waiting for approval" in text]
    assert parked, "the run never parked"
    assert "nothing above has run yet" in parked[0]
    assert "→" in parked[0], "the proposed edges were not shown"

    final_text, final_blocks = sink.updated[-1]
    assert final_blocks is None, "a finished run must not keep a live Approve button"

    # The trace is the record: asked, answered, then executed.
    trace = next(tmp_path.glob("slack-runs/*/trace.jsonl"))
    recorder = TailRecorder(trace)
    run_id = recorder.run_ids()[-1]
    phases = [e.phase for e in recorder.read_events(run_id)]
    assert phases.index("approval_request") < phases.index("approval_response")
    assert phases.index("approval_response") < phases.index("start"), (
        "a node started before the approval was answered"
    )


def test_a_denied_slack_go_never_starts_a_node(tmp_path):
    sink = RecordingSink()
    config = _config(tmp_path)

    def click_when_parked() -> None:
        blocks = sink.wait_for_buttons()
        assert blocks is not None
        handle_approval_action(blocks[1]["elements"][1]["value"], config, deny=True)

    clicker = threading.Thread(target=click_when_parked, daemon=True)
    clicker.start()
    handle_text_live('plan "investigate the checkout outage" --scripted --go', config, sink)
    clicker.join(timeout=10.0)

    trace = next(tmp_path.glob("slack-runs/*/trace.jsonl"))
    recorder = TailRecorder(trace)
    run_id = recorder.run_ids()[-1]
    events = list(recorder.read_events(run_id))
    assert any(e.phase == "approval_response" for e in events)
    assert not any(e.phase == "start" for e in events), "a denied plan executed"
