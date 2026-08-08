"""Reviewing a parked plan before answering it — issues #46 and #52.

The approval gate is the product's central trust claim, and two things were
missing from the human half of it. `grapharc approve` decided first and
described the plan afterwards, with no way to look without deciding (#46); and
a parked run announced that it was waiting without saying what would end the
wait (#52), so the failure mode was a run timing out unapproved while its
operator watched a cursor.
"""

from __future__ import annotations

import json

import pytest

from grapharc.cli.approve import approve
from grapharc.planner.approval_file import (
    DECISION_FILENAME,
    describe_request,
    write_request,
)

REQUEST = {
    "proposal_id": "p-1",
    "fingerprint": "abc123def456",
    "nodes": ["collect_context", "investigate", "apply_change"],
    "edges": [["__start__", "collect_context"], ["collect_context", "investigate"]],
    "rationale": "read the failing test, then repair it",
    "timeout_seconds": 300.0,
}


@pytest.fixture
def parked(tmp_path):
    write_request(tmp_path, REQUEST)
    return tmp_path


# -- #46: look before you decide ---------------------------------------------


def test_show_prints_the_plan_and_decides_nothing(parked, capsys):
    code = approve(parked, show=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "abc123def456" in out
    assert "apply_change" in out
    assert "collect_context -> investigate" in out
    assert "read the failing test" in out
    assert not (parked / DECISION_FILENAME).exists(), "--show wrote a decision"


def test_show_names_the_command_that_would_decide(parked, capsys):
    approve(parked, show=True)
    out = capsys.readouterr().out
    assert "--fingerprint abc123def456" in out
    assert "--deny" in out


def test_a_decision_bound_to_a_stale_fingerprint_is_refused(parked, capsys):
    """Between the `--show` and the `approve`, the run may have re-proposed."""
    code = approve(parked, fingerprint="the-one-i-read")
    out = capsys.readouterr().out + capsys.readouterr().err

    assert code == 2
    assert not (parked / DECISION_FILENAME).exists()


def test_the_refusal_names_both_fingerprints(parked, capsys):
    approve(parked, fingerprint="the-one-i-read")
    combined = capsys.readouterr()
    text = combined.out + combined.err
    assert "the-one-i-read" in text and "abc123def456" in text


def test_a_matching_fingerprint_decides_normally(parked):
    assert approve(parked, fingerprint="abc123def456") == 0
    decision = json.loads((parked / DECISION_FILENAME).read_text())
    assert decision == {"fingerprint": "abc123def456", "decision": "approved"}


def test_the_one_shot_form_still_works_unchanged(parked):
    """Automation, and the Slack button path, both rely on this."""
    assert approve(parked) == 0
    decision = json.loads((parked / DECISION_FILENAME).read_text())
    assert decision["decision"] == "approved"


def test_the_plan_is_printed_before_the_decision_line(parked, capsys):
    """The complaint in #46: it decided, then described what it had decided."""
    approve(parked)
    out = capsys.readouterr().out
    assert out.index("abc123def456") < out.index("decision")


def test_show_as_json_carries_the_shape_and_no_decision(parked, capsys):
    approve(parked, show=True, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] is None
    assert payload["nodes"] == REQUEST["nodes"]
    assert payload["fingerprint"] == "abc123def456"


def test_nothing_waiting_is_still_exit_one(tmp_path, capsys):
    assert approve(tmp_path, show=True) == 1


# -- #52: a parked run says what would end the wait ---------------------------


def test_the_park_announcement_is_actionable(tmp_path):
    text = describe_request(REQUEST, tmp_path)

    assert "abc123def456" in text, "the fingerprint a reviewer must quote"
    assert "3 nodes" in text and "apply_change" in text
    assert "300s" in text, "the deadline"
    assert f"grapharc approve {tmp_path}" in text
    assert "--deny" in text
    assert "--show" in text


def test_the_announcement_says_what_the_deadline_means(tmp_path):
    """A silent expiry reads as a planner failure rather than an unanswered question."""
    assert "unapproved" in describe_request(REQUEST, tmp_path)


def test_an_empty_proposal_still_announces_something_answerable(tmp_path):
    text = describe_request({"fingerprint": "f", "nodes": [], "timeout_seconds": 5}, tmp_path)
    assert "0 nodes" in text
    assert f"grapharc approve {tmp_path}" in text


def test_the_announcement_needs_no_cli_imports(tmp_path):
    """`approval_file` is the headless half; the CLI decorates, it does not format.

    A styling import here would make the file handshake — which any process
    that can reach the directory may drive — depend on the CLI package.
    """
    import grapharc.planner.approval_file as module

    source = module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "grapharc.cli" not in text
