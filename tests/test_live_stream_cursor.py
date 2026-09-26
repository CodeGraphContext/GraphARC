"""`LiveSnapshot.size` is a cursor, and it has to describe what was read.

`frames()` in `server.live` polls the trace file's size, rebuilds a snapshot
when it changes, and then stores the snapshot's own `size` as "everything up to
here has been sent":

    last_size = snapshot.size if snapshot.size else size

So a `size` larger than the bytes the snapshot actually covers is not a
cosmetic error. It claims events that were never sent, and the next poll sees
an unchanged file size and rebuilds nothing. If the writer has stopped — the
run finished — the file never changes size again, the stream never rebuilds,
and a finished run is streamed as a running one for as long as the page is
open.

`build_snapshot` used to take that number from `path.stat()` *after* reading
the events, which can exceed the bytes read two different ways:

  1. `TailRecorder` cuts at the last newline, so a half-written final line is
     excluded from the events but counted by `st_size`.
  2. The run is appending concurrently — that is the whole premise of a live
     view — so bytes can land between the read and the stat.

Only (2) loses whole events, because the bytes (1) leaves out cannot yet parse
into one. Both are fixed by the same change: the count comes from the read.
"""

from __future__ import annotations

import json
from pathlib import Path

from grapharc.observe.trace import TailRecorder
from grapharc.server.live import build_snapshot


def _event(
    run_id: str = "r1", phase: str = "start", step: int = 1, *, terminal: bool = False
) -> str:
    body: dict = {
        "ts": "2026-01-01T00:00:00+00:00",
        "run_id": run_id,
        "graph": "g",
        "node": "n",
        "phase": phase,
        "step": step,
    }
    if terminal:
        # What `compose_snapshot` actually reads to decide a run is over; a bare
        # `stop` phase on a node is not it.
        body["state_delta"] = {"termination_reason": "completed"}
    return json.dumps(body)


def _trace(tmp_path: Path, *lines: str, tail: str = "") -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(line + "\n" for line in lines) + tail, encoding="utf-8")
    return path


# -- what the recorder reports ----------------------------------------------


def test_the_recorder_reports_the_bytes_its_events_came_from(tmp_path):
    path = _trace(tmp_path, _event(), _event(step=2))
    events, consumed = TailRecorder(path).read_tail()

    assert len(events) == 2
    assert consumed == path.stat().st_size


def test_a_half_written_final_line_is_excluded_from_the_count(tmp_path):
    """The bytes after the last newline cannot parse into an event, so counting
    them would claim an event that does not exist yet."""
    path = _trace(tmp_path, _event(), _event(step=2), tail='{"ts": "2026-01-0')
    events, consumed = TailRecorder(path).read_tail()

    assert len(events) == 2
    assert consumed < path.stat().st_size
    # And the count is exactly the complete prefix.
    assert path.read_bytes()[:consumed].endswith(b"\n")


def test_a_file_with_no_complete_line_reads_as_nothing(tmp_path):
    path = _trace(tmp_path, tail='{"ts": "2026-01-0')

    assert TailRecorder(path).read_tail() == ([], 0)


def test_read_events_still_returns_just_the_events(tmp_path):
    """The old signature is what every other caller uses."""
    path = _trace(tmp_path, _event(), _event(step=2))

    assert TailRecorder(path).read_events() == TailRecorder(path).read_tail()[0]


# -- the cursor the stream stores --------------------------------------------


def test_the_snapshot_size_never_exceeds_what_it_read(tmp_path, monkeypatch):
    """The wedge, demonstrated.

    A writer appending between the read and the stat is the ordinary case for a
    live view, not an exotic one; it is made deterministic here by appending
    from inside the read. With `size` taken from a later `stat()`, the snapshot
    reported 2 events and a cursor past the third, and `frames()` would store
    that cursor and skip the third event for as long as the file stayed that
    size.
    """
    path = _trace(tmp_path, _event(), _event(step=2))
    real_read_tail = TailRecorder.read_tail

    def read_then_append(self, run_id=None):
        result = real_read_tail(self, run_id)
        # The writer lands in the window.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(_event(phase="stop", step=3) + "\n")
        return result

    monkeypatch.setattr(TailRecorder, "read_tail", read_then_append)
    snapshot = build_snapshot(tmp_path, "trace.jsonl", None)

    assert len(snapshot.run_ids) == 1
    # The cursor describes the two events that were read, not the three bytes'
    # worth now on disk.
    assert snapshot.size < path.stat().st_size, (
        "size claims bytes the snapshot never read; frames() will skip them"
    )


def test_the_skipped_event_is_picked_up_on_the_next_read(tmp_path, monkeypatch):
    """The point of the fix: the stream self-corrects.

    Because the cursor stopped short, the file's size now differs from it, so
    the next poll rebuilds and the event arrives. That is the difference
    between a one-poll delay and a stream that never recovers.
    """
    path = _trace(tmp_path, _event(), _event(step=2))
    real_read_tail = TailRecorder.read_tail
    appended = {"done": False}

    def read_then_append_once(self, run_id=None):
        result = real_read_tail(self, run_id)
        if not appended["done"]:
            appended["done"] = True
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(_event(phase="stop", step=3) + "\n")
        return result

    monkeypatch.setattr(TailRecorder, "read_tail", read_then_append_once)
    first = build_snapshot(tmp_path, "trace.jsonl", None)

    monkeypatch.setattr(TailRecorder, "read_tail", real_read_tail)
    second = build_snapshot(tmp_path, "trace.jsonl", None)

    assert first.size < path.stat().st_size  # cursor short, so the poll refires
    assert second.size == path.stat().st_size
    assert second.size > first.size


def test_a_quiet_complete_trace_reports_the_whole_file(tmp_path):
    """The ordinary case must not regress: nothing is being appended, the last
    line is complete, so the cursor is the file size and the stream settles."""
    path = _trace(tmp_path, _event(), _event(step=2), _event(phase="stop", step=3))
    snapshot = build_snapshot(tmp_path, "trace.jsonl", None)

    assert snapshot.size == path.stat().st_size


def test_an_empty_or_missing_trace_is_still_a_waiting_snapshot(tmp_path):
    """The URL is posted before the run starts writing."""
    (tmp_path / "trace.jsonl").write_text("", encoding="utf-8")

    assert build_snapshot(tmp_path, "trace.jsonl", None).size == 0


def test_a_finished_run_stops_reporting_itself_as_running(tmp_path, monkeypatch):
    """The symptom a viewer actually sees, end to end.

    Reproduced against the old code: an append landing in the window left the
    cursor equal to the file's final size while `done` was still False, so
    `frames()` had no reason to rebuild and never learned the run had stopped.
    The page showed a finished run as running for as long as it stayed open.

    With the cursor taken from the read, it falls short of the file, the next
    poll rebuilds, and that rebuild sees the `stop` event.
    """
    path = _trace(tmp_path, _event(), _event(step=2))
    real_read_tail = TailRecorder.read_tail
    fired = {"n": 0}

    def read_then_append_once(self, run_id=None):
        result = real_read_tail(self, run_id)
        fired["n"] += 1
        if fired["n"] == 1:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(_event(phase="stop", step=3, terminal=True) + "\n")
        return result

    monkeypatch.setattr(TailRecorder, "read_tail", read_then_append_once)
    first = build_snapshot(tmp_path, "trace.jsonl", None)
    final_size = path.stat().st_size

    # The old failure was exactly this pair: cursor == final size, done False.
    assert not (first.size == final_size and not first.done), (
        "cursor matches the finished file while done is False: frames() will "
        "never rebuild and the run streams as running forever"
    )

    monkeypatch.setattr(TailRecorder, "read_tail", real_read_tail)
    second = build_snapshot(tmp_path, "trace.jsonl", None)

    assert second.done is True
    assert second.size == final_size
