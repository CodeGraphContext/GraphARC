"""Strict trace readers report corrupt encoding at its actual JSONL line."""

from __future__ import annotations

import json

import pytest

from grapharc.cli.main import main
from grapharc.observe.trace import TraceReadError, TraceRecorder

BAD_LINES = [
    pytest.param(b'{"node":"\xff"}\n', id="invalid-byte"),
    pytest.param(b"\xff\xfe\x00binary\n", id="utf16"),
    pytest.param(b'{"node":"\xe2\x82"}\n', id="incomplete-character"),
]
READERS = [
    pytest.param(["trace"], id="trace"),
    pytest.param(["metrics", "r1"], id="metrics"),
    pytest.param(["viz", "r1"], id="viz"),
]


def _trace(tmp_path, bad_line: bytes, line_number: int = 2):
    recorder = TraceRecorder(tmp_path / "bad.jsonl")
    if line_number == 2:
        recorder.event(run_id="r1", graph="g", node="मॉडल", phase="start", step=1)
    with recorder.path.open("ab") as handle:
        handle.write(bad_line)
    return recorder


@pytest.mark.parametrize("bad_line", BAD_LINES)
@pytest.mark.parametrize("line_number", [1, 2])
def test_encoding_errors_use_the_trace_error_and_actual_line(tmp_path, bad_line, line_number):
    recorder = _trace(tmp_path, bad_line, line_number)

    with pytest.raises(TraceReadError) as caught:
        recorder.read_events("r1")

    assert caught.value.path == recorder.path
    assert caught.value.line_number == line_number
    assert isinstance(caught.value.cause, UnicodeDecodeError)


@pytest.mark.parametrize("argv", READERS)
@pytest.mark.parametrize("bad_line", BAD_LINES)
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_cli_reports_non_utf8_trace_without_a_traceback(tmp_path, capsys, argv, bad_line, as_json):
    recorder = _trace(tmp_path, bad_line)
    flags = ["--json"] if as_json else []

    assert main([argv[0], str(recorder.path), *argv[1:], *flags]) == 2

    captured = capsys.readouterr()
    message = f"unreadable trace file: {recorder.path}: line 2 is not a trace event"
    if as_json:
        assert json.loads(captured.out) == {"ok": False, "command": argv[0], "error": message}
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert captured.err == f"error: {message}\n"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
def test_utf8_content_and_blank_lines_keep_their_meaning(tmp_path, newline):
    recorder = TraceRecorder(tmp_path / "valid.jsonl")
    recorder.path.write_text("\u2003\n", encoding="utf-8")
    recorder.event(
        run_id="r1",
        graph="g",
        node="मॉडल",
        phase="end",
        step=1,
        state_delta={"answer": "café ✓"},
    )
    # A valid final event does not require a trailing newline.
    recorder.path.write_bytes(recorder.path.read_bytes().rstrip(b"\n").replace(b"\n", newline))

    events = recorder.read_events("r1")

    assert len(events) == 1
    assert events[0].node == "मॉडल"
    assert events[0].state_delta == {"answer": "café ✓"}
