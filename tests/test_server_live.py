"""The live browser view: file-tailing snapshots over SSE, confined to a root.

The router is mounted on a bare FastAPI app for most tests — it needs nothing
from the session runtime — and driven through `TestClient`, which buffers a
streaming response to completion; stream tests therefore write a
`termination_reason` (with the grace shrunk) so the stream actually ends.
"""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.trace import TailRecorder, TraceRecorder
from grapharc.server import live as live_module
from grapharc.server.app import create_app
from grapharc.server.live import (
    LivePathError,
    build_snapshot,
    live_router,
    resolve_trace,
    scan_traces,
)


def write_run(path, run_id, *, nodes=2, done=False, secret=None):
    recorder = TraceRecorder(path)
    for step in range(1, nodes + 1):
        recorder.event(run_id=run_id, graph="g", node=f"n{step}", phase="start", step=step)
        delta = {"answer": "ok"}
        if secret:
            delta["leaked"] = secret
        recorder.event(
            run_id=run_id,
            graph="g",
            node=f"n{step}",
            phase="end",
            step=step,
            duration_ms=8.0,
            tokens=5,
            cost_usd=0.001,
            state_delta=delta,
        )
    if done:
        recorder.event(
            run_id=run_id,
            graph="g",
            node="finish",
            phase="end",
            step=nodes + 1,
            state_delta={"termination_reason": "completed"},
        )
    return recorder


def live_client(root, **kwargs):
    app = FastAPI()
    app.include_router(live_router(root, poll_seconds=0.01, **kwargs))
    return TestClient(app)


def read_sse(response):
    frames = []
    for block in response.text.split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith(":") or not line.strip():
                continue
            key, _, value = line.partition(": ")
            if key == "event":
                name = value
            elif key == "data":
                data = value
        if name is not None:
            frames.append((name, json.loads(data)))
    return frames


@pytest.fixture(autouse=True)
def fast_grace(monkeypatch):
    monkeypatch.setattr(live_module, "DONE_GRACE_SECONDS", 0.05)


# ---------------------------------------------------------------------------
# TailRecorder and the pure helpers.
# ---------------------------------------------------------------------------


def test_tail_recorder_skips_a_torn_final_line_then_sees_it_complete(tmp_path):
    path = tmp_path / "t.jsonl"
    write_run(path, "r1")
    whole = TailRecorder(path).read_events()
    with path.open("a", encoding="utf-8") as f:
        f.write('{"ts": "2026-01-01T00:00:00.000+00:00", "run_id": "r1", "graph": "g"')

    assert TailRecorder(path).read_events() == whole  # torn tail invisible

    with path.open("a", encoding="utf-8") as f:
        f.write(', "node": "late", "phase": "start", "step": 9}\n')
    completed = TailRecorder(path).read_events()
    assert len(completed) == len(whole) + 1
    assert completed[-1].node == "late"


def test_resolve_trace_confines_gate_style(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "t.jsonl").touch()
    assert resolve_trace(tmp_path, "runs/t.jsonl") == (tmp_path / "runs" / "t.jsonl").resolve()
    # The target need not exist — the run may not have started.
    resolve_trace(tmp_path, "not-yet/t.jsonl")

    for bad in ("../outside.jsonl", "/etc/passwd", "runs/t.txt", ""):
        with pytest.raises(LivePathError):
            resolve_trace(tmp_path, bad)


def test_resolve_trace_refuses_a_symlink_out_of_the_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "t.jsonl").touch()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(LivePathError):
        resolve_trace(root, "link/t.jsonl")


def test_snapshot_agrees_with_viz_and_metrics(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    recorder = TailRecorder(tmp_path / "t.jsonl")
    assert snapshot.mermaid == to_mermaid(recorder, "r1")
    assert snapshot.stats == summarize(recorder, "r1")
    assert snapshot.cost_usd == pytest.approx(0.002)
    assert snapshot.done is True
    assert snapshot.mermaid_live_url and "#pako:" in snapshot.mermaid_live_url


def test_a_missing_file_is_a_waiting_snapshot_not_an_error(tmp_path):
    snapshot = build_snapshot(tmp_path, "not-yet/t.jsonl", None)
    assert snapshot.run_id is None
    assert snapshot.mermaid == ""
    assert snapshot.stats is None
    assert not (tmp_path / "not-yet").exists(), "a read must not create directories"


def test_snapshot_follows_the_latest_run_unless_one_is_named(tmp_path):
    path = tmp_path / "t.jsonl"
    write_run(path, "first")
    write_run(path, "second")
    assert build_snapshot(tmp_path, "t.jsonl", None).run_id == "second"
    assert build_snapshot(tmp_path, "t.jsonl", "first").run_id == "first"
    assert build_snapshot(tmp_path, "t.jsonl", None).run_ids == ["first", "second"]


def test_an_open_node_reads_as_active_even_when_the_file_is_quiet(tmp_path):
    """A delegated run writes nothing between start and finish; that is not idle."""
    import os
    import time

    path = tmp_path / "t.jsonl"
    TraceRecorder(path).event(
        run_id="r1", graph="cli-agent", node="claude_code", phase="start", step=1
    )
    stale = time.time() - 60  # well past the activity window
    os.utime(path, (stale, stale))
    assert build_snapshot(tmp_path, "t.jsonl", None).active is True


def test_scan_traces_lists_newest_first_with_run_ids(tmp_path):
    import os
    import time

    write_run(tmp_path / "a" / "t.jsonl", "ra")
    write_run(tmp_path / "b" / "t.jsonl", "rb")
    old = time.time() - 100
    os.utime(tmp_path / "a" / "t.jsonl", (old, old))
    traces = scan_traces(tmp_path)
    assert [t["trace"] for t in traces] == ["b/t.jsonl", "a/t.jsonl"]
    assert traces[0]["runs"] == ["rb"]


# ---------------------------------------------------------------------------
# The routes.
# ---------------------------------------------------------------------------


def test_stream_snapshots_a_finished_run_and_closes(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path) as client:
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            response.read()
            frames = read_sse(response)
    kinds = [f[0] for f in frames]
    assert kinds[0] == "snapshot" and kinds[-1] == "done"
    assert frames[0][1]["run_id"] == "r1"
    assert frames[0][1]["done"] is True


def test_stream_emits_a_second_snapshot_when_the_file_grows(tmp_path):
    path = tmp_path / "t.jsonl"
    write_run(path, "r1")

    def finish():
        write_run(path, "r1", done=True)

    with live_client(tmp_path) as client:
        threading.Timer(0.2, finish).start()
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            response.read()
            frames = read_sse(response)
    snapshots = [f[1] for f in frames if f[0] == "snapshot"]
    assert len(snapshots) >= 2
    assert snapshots[0]["done"] is False
    assert snapshots[-1]["done"] is True
    assert snapshots[-1]["stats"]["events"] > snapshots[0]["stats"]["events"]


def test_stream_waits_for_a_file_that_does_not_exist_yet(tmp_path):
    def start_run():
        write_run(tmp_path / "later" / "t.jsonl", "r1", done=True)

    with live_client(tmp_path) as client:
        threading.Timer(0.2, start_run).start()
        with client.stream("GET", "/live/api/stream?trace=later/t.jsonl") as response:
            response.read()
            frames = read_sse(response)
    snapshots = [f[1] for f in frames if f[0] == "snapshot"]
    assert snapshots[0]["run_id"] is None  # waiting
    assert snapshots[-1]["run_id"] == "r1"


def test_confinement_failures_are_404_on_every_route(tmp_path):
    with live_client(tmp_path) as client:
        for raw in ("../outside.jsonl", "/etc/passwd", "t.txt"):
            assert client.get(f"/live/view?trace={raw}").status_code == 404
            assert client.get(f"/live/api/stream?trace={raw}").status_code == 404


def test_state_delta_contents_never_reach_a_live_byte(tmp_path):
    sentinel = "SECRET-SENTINEL-a2f9"
    write_run(tmp_path / "t.jsonl", "r1", done=True, secret=sentinel)
    assert sentinel in (tmp_path / "t.jsonl").read_text(), "precondition"
    with live_client(tmp_path) as client:
        for url in ("/live", "/live/api/runs", "/live/view?trace=t.jsonl"):
            assert sentinel not in client.get(url).text
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            response.read()
            assert sentinel not in response.text


def test_a_token_locks_every_live_route(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path, token="s3cret") as client:
        for url in ("/live", "/live/api/runs", "/live/view?trace=t.jsonl"):
            assert client.get(url).status_code == 401
        assert client.get("/live/api/runs?token=s3cret").status_code == 200
        assert (
            client.get(
                "/live/api/runs", headers={"authorization": "Bearer s3cret"}
            ).status_code
            == 200
        )
        assert client.get("/live/api/runs?token=wrong").status_code == 401


def test_the_index_lists_traces_and_links_the_viewer(tmp_path):
    write_run(tmp_path / "runs" / "t.jsonl", "r1")
    with live_client(tmp_path) as client:
        page = client.get("/live").text
        assert "runs/t.jsonl" in page
        assert "r1" in page
        assert client.get("/live/view?trace=runs/t.jsonl").status_code == 200


def test_create_app_mounts_live_only_when_asked(tmp_path):
    with TestClient(create_app()) as client:
        assert client.get("/live").status_code == 404
        assert "/live/api/runs" not in client.get("/openapi.json").json()["paths"]
    with TestClient(create_app(live_root=tmp_path)) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/healthz").status_code == 200  # existing API untouched
