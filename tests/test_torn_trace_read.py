"""A trace being written must stay readable to the tools that read it.

`TraceRecorder` appends one JSON line per event, so a reader that opens the
file while a run is in progress can land on a half-written last line. That is
not a corrupt file — it is a file with one more byte coming — and the codebase
already has the answer: `TailRecorder` skips the torn line where the strict
reader raises, which is how the live view and the Slack tailer read a running
run without crashing.

`mcp/driver.graph_status` did not use it. So the supervised agent's own status
tool — the one it calls to find out whether a human has approved yet, while
the run it is asking about is still writing — raised `TraceReadError` out of
the MCP server instead of answering.
"""

from __future__ import annotations

import json

from grapharc.observe.trace import TraceRecorder


def _run_dir(tmp_path, *, torn: bool):
    """A finished plan whose trace has (or has not) a half-written last line."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "plan.json").write_text(
        json.dumps(
            {
                "goal": "investigate",
                "fingerprint": "f1",
                "executed_run_id": "r1",
                "mutating": False,
                "proposal": {"nodes": ["triage"], "edges": [], "rationale": "why"},
            }
        ),
        encoding="utf-8",
    )
    trace = directory / "trace.jsonl"
    recorder = TraceRecorder(trace)
    recorder.event(run_id="r1", graph="g", node="triage", phase="start", step=1)
    recorder.event(
        run_id="r1", graph="g", node="triage", phase="end", step=1, duration_ms=4.0, tokens=11
    )
    if torn:
        # Exactly what a reader sees mid-append: the writer got half a line out.
        with trace.open("a", encoding="utf-8") as handle:
            handle.write('{"run_id": "r1", "graph": "g", "node": "verify", "pha')
    return directory


def test_a_torn_last_line_does_not_crash_the_status_tool(tmp_path):
    """The failure this file exists for.

    A supervised agent polls `show_graph` to learn whether its plan has been
    approved. Polling *while the run writes* is the whole point of polling, so
    the one moment this must not raise is the one moment it did.
    """
    from grapharc.mcp.driver import graph_status

    view = graph_status(_run_dir(tmp_path, torn=True))

    assert view["status"] == "done"
    assert view["metrics"]["tokens"] == 11
    assert "triage" in view["mermaid"]


def test_an_intact_trace_reads_exactly_as_before(tmp_path):
    """The fix must not be "skip lines and hope" — a whole file is unchanged."""
    from grapharc.mcp.driver import graph_status

    view = graph_status(_run_dir(tmp_path, torn=False))

    assert view["status"] == "done"
    assert view["metrics"]["tokens"] == 11
    assert view["metrics"]["nodes_executed"] == 1
