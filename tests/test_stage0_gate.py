"""Stage 0 gate: kill the process mid-write, restart, exactly one report copy."""

from grapharc.examples import stage0_dag
from grapharc.examples.stage0_dag import DEMO_DOC, build_stage0, crash_once_before_rename


def _write_doc(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(DEMO_DOC, encoding="utf-8")
    return doc


def test_happy_path_writes_one_report(tmp_path, trace):
    doc = _write_doc(tmp_path)
    report = tmp_path / "report.md"
    compiled = build_stage0(trace=trace)
    result = compiled.invoke({"doc_path": str(doc), "report_path": str(report)})
    assert result["report_written"] is True
    assert report.exists()
    assert "Term report" in report.read_text()


def test_gate_crash_mid_write_then_resume_yields_exactly_one_report(
    tmp_path, trace, checkpointer
):
    doc = _write_doc(tmp_path)
    report = tmp_path / "report.md"
    compiled = build_stage0(trace=trace, checkpointer=checkpointer)

    crash_once_before_rename()
    try:
        compiled.invoke(
            {"doc_path": str(doc), "report_path": str(report)}, thread_id="t1"
        )
        raise AssertionError("expected injected crash")
    except RuntimeError as exc:
        assert "injected crash" in str(exc)
    finally:
        stage0_dag._CRASH_BEFORE_RENAME = False

    # After the crash: zero finished copies (tmp is not a report).
    assert not report.exists()

    # Restart: resume the same thread from its checkpoint.
    result = compiled.invoke(None, thread_id="t1")
    assert result["report_written"] is True

    # Exactly one copy — the report exists, and no orphaned tmp file remains.
    copies = [p for p in report.parent.glob("report.md*") if p.suffix != ".tmp"]
    assert copies == [report]
    assert not list(report.parent.glob("*.tmp"))

    # The crash is on the audit trail.
    errors = [e for e in trace.read_events() if e.phase == "error"]
    assert any("injected crash" in (e.error or "") for e in errors)

    # Replay-point identity survives the resume: both attempts share the
    # thread_id, attempts are numbered, and step numbers never collide.
    events = [
        e
        for e in trace.read_events()
        # Topology events carry step 0 on every attempt — shape, not order.
        if e.thread_id == "t1" and e.phase != "topology"
    ]
    assert events, "trace events must carry thread_id"
    first = [e for e in events if e.attempt == 1]
    second = [e for e in events if e.attempt == 2]
    assert first and second
    assert max(e.step for e in first) < min(e.step for e in second)
    resumed_report = [e for e in second if e.node == "report" and e.phase == "end"]
    assert len(resumed_report) == 1
