"""`grapharc go <dir>` on a plan that has already run.

Bare `go` has always skipped executed plans — `find_unexecuted_plan` passes
over any record carrying an `executed_run_id`. The explicitly-named-directory
form did not make the same check, so a second `go <dir>` re-ran the whole graph
and overwrote the stamp, leaving a `plan.json` that named one run while the
trace correctly held three.

Two separate claims are under test here, and the second is the load-bearing
one. The *record* must be able to name every run that executed the plan. And
the *decision* must not be spent twice: an approval binds to a proposal
fingerprint, which does not change between runs, so a silent re-run of a
`mutating` plan executes on the strength of an earlier human yes.

The planner is scripted throughout — no model backend, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from grapharc.cli.main import main
from grapharc.cli.plan import _executed_run_ids


def _saved_plan(tmp_path, capsys) -> Path:
    """A run directory holding an admitted, unexecuted `plan.json`."""
    trace = tmp_path / "run" / "trace.jsonl"
    assert main(["plan", "investigate", "--scripted", "--trace", str(trace), "--json"]) == 0
    capsys.readouterr()  # drop the plan document
    return trace.parent


def _last_document(text: str) -> dict:
    """The final JSON document in a stream that may carry several."""
    decoder = json.JSONDecoder()
    documents, index = [], 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        try:
            document, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        documents.append(document)
    assert documents, f"no JSON document in: {text[:200]!r}"
    return documents[-1]


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))


def _runs_in_trace(run_dir: Path) -> list[str]:
    seen: list[str] = []
    for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run_id = json.loads(line).get("run_id")
        if run_id and run_id not in seen:
            seen.append(run_id)
    return seen


# -- the refusal ------------------------------------------------------------


def test_a_second_go_on_an_executed_plan_is_refused(tmp_path, capsys):
    """The bug: this used to exit 0 having silently run the whole graph again."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    first = _record(run_dir)["executed_run_id"]
    before = _runs_in_trace(run_dir)
    capsys.readouterr()

    code = main(["go", str(run_dir), "--json"])

    assert code == 2
    payload = _last_document(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["executed_run_id"] == first
    assert "already been executed" in payload["error"]
    assert "--again" in payload["error"]
    # Refused before anything ran: the trace gained no run from the refusal.
    # (It holds two — the planning run, then the one execution.)
    assert _runs_in_trace(run_dir) == before


def test_the_refusal_names_the_run_and_when_it_happened(tmp_path, capsys):
    """A refusal a reader cannot act on is an obstacle, not a gate."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    capsys.readouterr()

    assert main(["go", str(run_dir)]) == 2

    message = capsys.readouterr().err
    assert _record(run_dir)["executed_run_id"] in message
    assert _record(run_dir)["executed_at"] in message


def test_again_executes_it_a_second_time(tmp_path, capsys):
    """Explicit re-runs stay possible; only the silent ones stop."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    first = _record(run_dir)["executed_run_id"]
    capsys.readouterr()

    code = main(["go", str(run_dir), "--again", "--json"])

    assert code == 0
    payload = _last_document(capsys.readouterr().out)
    assert payload["executed"] is True
    assert payload["run_id"] != first
    # The planning run, then both executions.
    assert _runs_in_trace(run_dir)[-2:] == [first, payload["run_id"]]


# -- the record -------------------------------------------------------------


def test_the_record_names_every_run_that_executed_the_plan(tmp_path, capsys):
    """Three executions used to leave a plan.json naming one, while the trace
    — the audit trail, and the one that was right — held all three."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    assert main(["go", str(run_dir), "--again", "--json"]) == 0
    assert main(["go", str(run_dir), "--again", "--json"]) == 0
    capsys.readouterr()

    record = _record(run_dir)
    assert len(record["executed_run_ids"]) == 3
    # The record now agrees with the trace, which was always right. The trace
    # also carries the planning run that produced the plan, hence the slice.
    assert record["executed_run_ids"] == _runs_in_trace(run_dir)[-3:]
    # The scalar stays the newest: `find_unexecuted_plan` and the MCP driver
    # read it, and an older reader must keep working.
    assert record["executed_run_id"] == record["executed_run_ids"][-1]


def test_a_plan_that_never_executed_carries_no_history(tmp_path, capsys):
    """Absent rather than empty: a reader that tests for the key must not see
    one appear merely because the plan was saved."""
    run_dir = _saved_plan(tmp_path, capsys)

    record = _record(run_dir)
    assert "executed_run_id" not in record
    assert "executed_run_ids" not in record
    assert _executed_run_ids(record) == []


# -- compatibility with records written before the list existed -------------


def test_an_old_record_with_only_the_scalar_reports_its_one_run():
    """A `plan.json` written before `executed_run_ids` existed must report the
    run it does know about, not report none."""
    assert _executed_run_ids({"executed_run_id": "abc123"}) == ["abc123"]
    assert _executed_run_ids({}) == []
    # A malformed list is a record for a human to read, not a place to raise.
    assert _executed_run_ids({"executed_run_ids": "not-a-list"}) == []


def test_an_old_record_is_refused_and_then_accumulates_from_its_scalar(tmp_path, capsys):
    """The upgrade path: a pre-existing scalar-only record still refuses a
    silent re-run, and `--again` grows the list from it rather than losing it."""
    run_dir = _saved_plan(tmp_path, capsys)
    record = _record(run_dir)
    record["executed_run_id"] = "old-run-id"
    (run_dir / "plan.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    capsys.readouterr()

    assert main(["go", str(run_dir), "--json"]) == 2
    capsys.readouterr()
    assert main(["go", str(run_dir), "--again", "--json"]) == 0
    capsys.readouterr()

    grown = _record(run_dir)
    assert grown["executed_run_ids"][0] == "old-run-id"
    assert len(grown["executed_run_ids"]) == 2


# -- bare `go` is unchanged -------------------------------------------------


def test_bare_go_still_skips_an_executed_plan(tmp_path, capsys, monkeypatch):
    """`find_unexecuted_plan` already passed over executed plans; the new guard
    must not turn that quiet skip into a refusal."""
    monkeypatch.chdir(tmp_path)
    trace = tmp_path / ".grapharc" / "runs" / "r1" / "trace.jsonl"
    assert main(["plan", "investigate", "--scripted", "--trace", str(trace), "--json"]) == 0
    capsys.readouterr()

    assert main(["go", "--json"]) == 0
    capsys.readouterr()

    # Nothing left unexecuted: the bare form reports that, rather than refusing
    # a directory it was never given.
    assert main(["go", "--json"]) == 1
    payload = _last_document(capsys.readouterr().out)
    assert "no unexecuted plan" in payload["error"]
