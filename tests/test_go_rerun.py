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
import re
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


# -- an attempt that began and never recorded finishing (#113) --------------
#
# `executed_run_id` is stamped after `loop.run()` returns, so a run killed
# partway through — the MCP `execute` work budget expiring, a SIGKILL, an
# OOM-kill — never reaches the stamp. The record then says the plan was never
# executed while the tree may already have been changed, and the guard above
# waves a second run through on the strength of one human approval. The trace
# is written as the run proceeds, so it is the only place that evidence lives.


def _forget_the_stamp(run_dir: Path) -> str:
    """A record as a killed run would leave it: trace written, stamp missing.

    Simulated by removing the stamp rather than by killing a real subprocess,
    because the two leave the same artefacts and only one of them is a test
    that races.
    """
    record = _record(run_dir)
    ran = record.pop("executed_run_id")
    record.pop("executed_run_ids", None)
    record.pop("executed_at", None)
    (run_dir / "plan.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return ran


def test_an_attempt_that_began_and_never_finished_is_refused(tmp_path, capsys):
    """The bug: the record says "never executed", so this used to exit 0 and
    run the plan a second time over a tree the first run had half-changed."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    killed = _forget_the_stamp(run_dir)
    before = _runs_in_trace(run_dir)
    capsys.readouterr()

    code = main(["go", str(run_dir), "--json"])

    assert code == 2
    payload = _last_document(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["unfinished_run_id"] == killed
    assert "never recorded finishing" in payload["error"]
    assert "--again" in payload["error"]
    # Refused before anything ran.
    assert _runs_in_trace(run_dir) == before


def test_the_refusal_points_at_the_trace_that_holds_what_it_did(tmp_path, capsys):
    """The tree may have been changed. A refusal that does not say where to
    look leaves the reader with no way to find out how far it got."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    _forget_the_stamp(run_dir)
    capsys.readouterr()

    assert main(["go", str(run_dir)]) == 2

    message = capsys.readouterr().err
    assert str(run_dir / "trace.jsonl") in message
    assert "may have changed the tree" in message


def test_again_still_runs_an_unfinished_plan(tmp_path, capsys):
    """The escape hatch is the same one the executed-plan guard offers."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert main(["go", str(run_dir), "--json"]) == 0
    _forget_the_stamp(run_dir)
    capsys.readouterr()

    assert main(["go", str(run_dir), "--again", "--json"]) == 0
    assert _last_document(capsys.readouterr().out)["executed"] is True


def test_planning_paperwork_is_not_mistaken_for_a_half_run(tmp_path, capsys):
    """The false positive this guard must not have. `plan` writes its own
    events into the same trace — `plan`, `admission`, `round`, `topology` — and
    a first `go` would be refused forever if those counted as an execution."""
    run_dir = _saved_plan(tmp_path, capsys)
    assert (run_dir / "trace.jsonl").is_file()  # paperwork is already there
    capsys.readouterr()

    assert main(["go", str(run_dir), "--json"]) == 0
    assert _last_document(capsys.readouterr().out)["executed"] is True


# -- the phase vocabulary the guard depends on ------------------------------


def _trace_with(tmp_path: Path, phases: list[str], run_id: str = "killed-run") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00+00:00",
                    "run_id": run_id,
                    "graph": "g",
                    "node": "n",
                    "phase": phase,
                    "step": 1,
                }
            )
            + "\n"
            for phase in phases
        ),
        encoding="utf-8",
    )
    return trace


def test_only_a_node_doing_work_counts_as_having_begun(tmp_path):
    """Pinned deliberately. The loop's bookkeeping, the viewer's shape events
    and a bare `stop` all share the trace with node events; treating any of
    them as an execution would refuse a plan that had never run a node — a
    parked plan a human *denied* being the case that matters most."""
    from grapharc.cli.plan import _unfinished_execution

    paperwork = ["plan", "admission", "round", "topology", "stop"]
    assert _unfinished_execution(_trace_with(tmp_path / "a", paperwork), {}) is None

    denied = ["plan", "admission", "round", "approval_request", "approval_response", "stop"]
    assert _unfinished_execution(_trace_with(tmp_path / "b", denied), {}) is None

    for phase in ("start", "model", "end", "error"):
        trace = _trace_with(tmp_path / phase, ["plan", "admission", phase])
        assert _unfinished_execution(trace, {}) == "killed-run", phase


def test_a_run_the_record_already_names_is_not_unfinished(tmp_path):
    from grapharc.cli.plan import _unfinished_execution

    trace = _trace_with(tmp_path / "known", ["end"], run_id="r1")
    assert _unfinished_execution(trace, {"executed_run_id": "r1"}) is None
    assert _unfinished_execution(trace, {"executed_run_ids": ["r1"]}) is None
    assert _unfinished_execution(trace, {"executed_run_id": "other"}) == "r1"


def test_a_damaged_trace_does_not_wedge_every_plan_beside_it(tmp_path):
    """A torn line is not evidence of a half-run, and refusing on one would
    make an unreadable file block work it says nothing about."""
    from grapharc.cli.plan import _unfinished_execution

    torn = tmp_path / "trace.jsonl"
    torn.parent.mkdir(parents=True, exist_ok=True)
    torn.write_text('{"run_id": "r1", "phase": "en', encoding="utf-8")

    assert _unfinished_execution(torn, {}) is None
    assert _unfinished_execution(tmp_path / "absent.jsonl", {}) is None


# -- the same guard on the documented flow ----------------------------------


def test_bare_go_does_not_re_run_an_unfinished_plan(tmp_path, capsys, monkeypatch):
    """`grapharc plan … && grapharc go` is the flow the README teaches, and
    bare `go` selects its plan by a different route: `find_unexecuted_plan`
    passes over any record carrying an `executed_run_id`, and a killed run
    never wrote one. So the half-finished plan looks *unexecuted* to the
    selector — the newest candidate, chosen first.

    The refusal happens after the selection either way, which is what makes
    this safe, and it is exactly the kind of thing a later refactor of the
    selector could quietly undo. Pinned here for that reason.
    """
    monkeypatch.chdir(tmp_path)
    trace = tmp_path / ".grapharc" / "runs" / "r1" / "trace.jsonl"
    assert main(["plan", "investigate", "--scripted", "--trace", str(trace), "--json"]) == 0
    capsys.readouterr()
    run_dir = trace.parent

    assert main(["go", "--json"]) == 0
    killed = _forget_the_stamp(run_dir)
    before = _runs_in_trace(run_dir)
    capsys.readouterr()

    code = main(["go", "--json"])

    assert code == 2
    payload = _last_document(capsys.readouterr().out)
    assert payload["unfinished_run_id"] == killed
    # The load-bearing assertion: nothing ran a second time.
    assert _runs_in_trace(run_dir) == before


def test_the_phase_vocabulary_has_exactly_one_owner():
    """Four modules needed to know which phases are bookkeeping, and each had
    its own copy — `observe.metrics`, `observe.viewmodel` (whose comment said
    "mirrors `metrics`", which was true and was the problem), `slack.live`, and
    the one `cli.plan` added. A phase classified one way in one file and the
    other way in another is a bug in whichever is wrong, with nothing in the
    tree to say which.

    Identity, not equality: four frozensets that happen to agree today is
    exactly the state this assertion exists to rule out.
    """
    from grapharc.cli import plan as cli_plan
    from grapharc.observe import metrics, trace, viewmodel
    from grapharc.slack import live

    assert metrics._LOOP_PHASES is trace.LOOP_PHASES
    assert metrics._SHAPE_PHASES is trace.SHAPE_PHASES
    assert viewmodel._LOOP_PHASES is trace.LOOP_PHASES
    assert viewmodel._SHAPE_PHASES is trace.SHAPE_PHASES
    assert live._SHAPE_PHASES is trace.SHAPE_PHASES
    assert not hasattr(cli_plan, "_NON_EXECUTION_PHASES"), (
        "cli.plan grew its own copy of the phase vocabulary again"
    )


def test_no_module_redefines_the_phase_vocabulary():
    """The identity check above only covers names it knows to look at, so it
    cannot notice a *fifth* copy appearing under a new name. This reads the
    source instead: `observe.trace` defines the sets, and nothing else does.
    """
    package = Path(__file__).resolve().parents[1] / "grapharc"
    pattern = re.compile(r"^_?(?:LOOP|SHAPE|DRIVER|NON_EXECUTION)_PHASES\s*=\s*frozenset")
    # Files, not file:line — a line number would make this fail on any edit to
    # trace.py, which is noise rather than a finding.
    owners = sorted(
        {
            str(path.relative_to(package))
            for path in package.rglob("*.py")
            if "__pycache__" not in path.parts
            for line in path.read_text(encoding="utf-8").splitlines()
            if pattern.match(line)
        }
    )

    assert owners == ["observe/trace.py"], (
        f"the phase vocabulary is defined outside observe/trace.py: {owners}"
    )


def test_an_unclassified_phase_reads_as_an_execution():
    """The direction the predicate errs in, asserted rather than assumed.

    A bookkeeping phase nobody classified makes `go` refuse a plan it could
    have run, which `--again` recovers from. The opposite would re-run a
    half-finished mutating plan and spend a human approval given once.
    """
    from grapharc.observe import trace

    assert trace.began_execution("a-phase-nobody-has-written-yet") is True
    for phase in trace.NON_EXECUTION_PHASES:
        assert trace.began_execution(phase) is False, phase

    # The three groups partition the bookkeeping set, with nothing dropped.
    assert (
        trace.LOOP_PHASES | trace.SHAPE_PHASES | trace.DRIVER_PHASES
    ) == trace.NON_EXECUTION_PHASES
