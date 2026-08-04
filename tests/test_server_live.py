"""The live browser view: file-tailing snapshots over SSE, confined to a root.

The router is mounted on a bare FastAPI app for most tests — it needs nothing
from the session runtime — and driven through `TestClient`, which buffers a
streaming response to completion; stream tests therefore write a
`termination_reason` (with the grace shrunk) so the stream actually ends.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.trace import TailRecorder, TraceEvent, TraceRecorder
from grapharc.server import live as live_module
from grapharc.server.app import create_app
from grapharc.server.live import (
    LivePathError,
    build_snapshot,
    live_router,
    replay_schedule,
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


def write_planner_run(path, run_id="p1", *, rounds=2, stop="admission_refused", finish=True):
    """A governed-loop trace: plan/admission/round per round, then the loop's stop.

    Shaped after what `planner.proposal`, `planner.admission` and `planner.loop`
    actually write, including the deltas the live view reads.
    """
    recorder = TraceRecorder(path)
    for number in range(1, rounds + 1):
        recorder.event(
            run_id=run_id, graph="g", node="planner:plan", phase="plan", step=number * 10,
            tokens=20, duration_ms=1500.0,
            state_delta={"structured": True, "nodes": 3, "edges": 2,
                         "proposal_id": f"p{number}"},
        )
        recorder.event(
            run_id=run_id, graph="g", node=f"checker:p{number}", phase="admission",
            step=number * 10 + 1,
            state_delta={"status": "rejected", "fingerprint": "f", "origin": "planner",
                         "nodes": 3, "depth": 1, "checks_run": ["policy", "acyclicity"],
                         "failed_checks": ["policy"], "worst_case": {}},
            error="policy/edge_denied",
        )
        if not finish and number == rounds:
            return recorder  # still mid-round: no `round` event written yet
        recorder.event(
            run_id=run_id, graph="g", node=f"loop:round{number}", phase="round",
            step=number * 10 + 2,
            state_delta={"round": number, "proposal_id": f"p{number}", "status": "rejected",
                         "nodes": 3, "rejections": ["policy/edge_denied"], "executed": False,
                         "progressed": False, "stop": ""},
            error="policy/edge_denied",
        )
    if finish:
        recorder.event(
            run_id=run_id, graph="g", node="loop:stop", phase="stop", step=99,
            state_delta={"stop": stop, "detail": "every round was refused",
                         "rounds": rounds, "executed_rounds": 0},
            error=f"{stop}: every round was refused",
        )
    return recorder


def write_timed_run(path, run_id="r1", *, gap=1.0, nodes=("a", "b")):
    """A topology run whose events carry chosen timestamps, one node at a time.

    The per-node status classes a replay sweeps through only exist in the
    topology rendering, so the trace declares one.
    """
    recorder = TraceRecorder(path)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    def stamp(offset):
        return (base + timedelta(seconds=offset)).isoformat()

    recorder.record(
        TraceEvent(
            ts=stamp(0), run_id=run_id, graph="g", node="topology", phase="topology", step=0,
            state_delta={
                "nodes": list(nodes),
                "edges": [[a, b, "normal"] for a, b in zip(nodes, nodes[1:], strict=False)],
            },
        )
    )
    offset = 0.0
    for step, node in enumerate(nodes, start=1):
        recorder.record(
            TraceEvent(ts=stamp(offset), run_id=run_id, graph="g", node=node,
                       phase="start", step=step)
        )
        offset += gap
        recorder.record(
            TraceEvent(
                ts=stamp(offset), run_id=run_id, graph="g", node=node, phase="end", step=step,
                duration_ms=gap * 1000, tokens=5,
                state_delta={"termination_reason": "completed"} if node == nodes[-1] else None,
            )
        )
    return recorder


def live_client(root, **kwargs):
    app = FastAPI()
    app.include_router(live_router(root, poll_seconds=0.01, **kwargs))
    return TestClient(app)


def read_sse(response):
    return read_sse_text(response.text)


def read_sse_text(text):
    frames = []
    for block in text.split("\n\n"):
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


def test_snapshot_carries_a_positioned_graph_that_agrees_with_the_overlay(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    assert snapshot.graph is not None
    drawn = {n.id: n for n in snapshot.graph.nodes if n.role == "node"}
    # Same statuses the Mermaid class overlay assigns (all done here).
    assert {n.status for n in drawn.values()} == {"done"}
    # Positioned: the layout ran, and the recorded spend reached the node.
    assert all(n.w > 0 and n.h > 0 for n in drawn.values())
    assert snapshot.graph.width > 0 and snapshot.graph.height > 0
    assert any(n.cost_usd for n in drawn.values())
    assert any(n.tokens for n in drawn.values())


def test_the_graph_snapshot_is_in_the_stream_and_replay_frames(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path) as client:
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            response.read()
            live_frames = read_sse(response)
        replayed = client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=500")
    live_snapshot = next(f[1] for f in live_frames if f[0] == "snapshot")
    assert live_snapshot["graph"]["kind"] == "path"
    assert live_snapshot["graph"]["nodes"]
    replay_snapshots = [s for kind, s in read_sse(replayed) if kind == "snapshot"]
    assert all(s["graph"] is not None for s in replay_snapshots)


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


def test_static_assets_serve_with_their_types_and_nothing_else(tmp_path):
    with live_client(tmp_path) as client:
        css = client.get("/live/static/view.css")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        js = client.get("/live/static/view.js")
        assert js.status_code == 200
        assert js.headers["content-type"].startswith("text/javascript")
        # The allowlist is the route: page templates and traversal are 404s.
        for name in ("view.html", "signin.html", "../live.py", "%2e%2e/live.py"):
            assert client.get(f"/live/static/{name}").status_code == 404


def test_the_view_makes_no_external_request(tmp_path):
    """The page must render with the network cable pulled: no CDN, no import
    from another origin, in any byte the live routes serve."""
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path) as client:
        pages = [
            client.get("/live").text,
            client.get("/live/view?trace=t.jsonl").text,
            client.get("/live/static/view.css").text,
            client.get("/live/static/view.js").text,
        ]
    for page in pages:
        assert "cdn.jsdelivr" not in page
        assert "https://" not in page.replace("https://mermaid.live", "")


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
        assert (
            client.get(
                "/live/api/runs", headers={"authorization": "Bearer s3cret"}
            ).status_code
            == 200
        )
        assert client.get("/live/api/runs?token=wrong").status_code == 401
        # The SSE route is the one that has no other way in: `EventSource`
        # cannot set a header.
        assert client.get("/live/api/stream?trace=t.jsonl&token=s3cret").status_code == 200


def test_a_hostile_token_is_a_401_not_a_crash(tmp_path):
    """The gate that refuses strangers must not be crashable by one.

    `secrets.compare_digest` refuses `str` outside ASCII, so a one-character
    guess used to raise `TypeError` through the handler — a 500 that both
    amplifies the log and tells the caller how the token is compared.
    """
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    routes = ("/live", "/live/api/runs", "/live/view?trace=t.jsonl",
              "/live/api/stream?trace=t.jsonl")
    guesses = ("caf%C3%A9", "%C3%A9", "%F0%9F%94%91", "x" * 9000, "")
    with live_client(tmp_path, token="s3cret") as client:
        for route in routes:
            sep = "&" if "?" in route else "?"
            for guess in guesses:
                response = client.get(f"{route}{sep}token={guess}")
                assert response.status_code == 401, (route, guess)
            # Also over the header, where the wire is bytes: starlette decodes
            # them latin-1, so non-ASCII arrives as a non-ASCII `str` too.
            assert client.get(
                route, headers={"authorization": "Bearer café".encode()}
            ).status_code == 401
            # And the right token still gets in, by the means each route allows.
            assert client.get(
                route, headers={"authorization": "Bearer s3cret"}
            ).status_code == 200


def test_a_non_ascii_token_still_authorizes_its_owner(tmp_path):
    """Bytes comparison must widen what is accepted, not only what is refused."""
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path, token="café-🔑") as client:
        # Over the stream route's query parameter, where it is still accepted.
        assert client.get(
            "/live/api/stream?trace=t.jsonl&token=caf%C3%A9-%F0%9F%94%91"
        ).status_code == 200
        assert client.get(
            "/live/api/stream?trace=t.jsonl&token=caf%C3%A9"
        ).status_code == 401
        # And through sign-in, whose cookie is a digest — a header carries
        # latin-1 bytes, so the secret itself could not have ridden in one.
        assert client.post("/live/auth", data={"token": "café"}).status_code == 401
        assert client.post("/live/auth", data={"token": "café-🔑"}).status_code == 200
        assert client.get("/live/api/runs").status_code == 200
        assert "café" not in client.cookies.get("grapharc_live_token", "")


def test_a_nul_byte_in_the_trace_is_404_not_500(tmp_path):
    """`resolve_trace` raises `ValueError`, not `LivePathError`, on a NUL byte."""
    with live_client(tmp_path) as client:
        for raw in ("%00.jsonl", "sub/%00/t.jsonl", "t%00.jsonl"):
            assert client.get(f"/live/view?trace={raw}").status_code == 404
            assert client.get(f"/live/api/stream?trace={raw}").status_code == 404


def test_the_index_hides_a_symlinked_trace_outside_the_root(tmp_path):
    """The index must advertise only what the reader will serve."""
    secret = tmp_path / "OUTSIDE.jsonl"
    write_run(secret, "SECRET-RUN", done=True)
    root = tmp_path / "liveroot"
    root.mkdir()
    write_run(root / "run1.jsonl", "r1", done=True)
    (root / "link_out.jsonl").symlink_to(secret)
    (root / "sub").mkdir()
    (root / "sub" / "link_out.jsonl").symlink_to(secret)
    (root / "linkdir").symlink_to(tmp_path)  # a symlinked *directory* too

    assert [t["trace"] for t in scan_traces(root)] == ["run1.jsonl"]

    with live_client(root) as client:
        listed = client.get("/live/api/runs").json()["traces"]
        assert [t["trace"] for t in listed] == ["run1.jsonl"]
        page = client.get("/live").text
        for leak in ("SECRET-RUN", "link_out.jsonl", "OUTSIDE.jsonl"):
            assert leak not in page
        # And the index still agrees with the reader, which refuses the link.
        assert client.get("/live/api/stream?trace=link_out.jsonl").status_code == 404
        assert client.get("/live/view?trace=link_out.jsonl").status_code == 404


def test_the_index_lists_traces_and_links_the_viewer(tmp_path):
    write_run(tmp_path / "runs" / "t.jsonl", "r1")
    with live_client(tmp_path) as client:
        page = client.get("/live").text
        assert "runs/t.jsonl" in page
        assert "r1" in page
        assert client.get("/live/view?trace=runs/t.jsonl").status_code == 200


# ---------------------------------------------------------------------------
# The token stays out of URLs everywhere it has another way in (#41).
# ---------------------------------------------------------------------------


def test_a_query_token_is_refused_off_the_sse_route(tmp_path):
    """A URL is copied into logs, history and referrers; a header is not."""
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path, token="s3cret") as client:
        for url in ("/live", "/live/api/runs", "/live/view?trace=t.jsonl"):
            sep = "&" if "?" in url else "?"
            response = client.get(f"{url}{sep}token=s3cret")
            assert response.status_code == 401, url
            assert "query string" in response.text, url  # a distinct reason
        # The one route with no alternative keeps it.
        assert client.get("/live/api/stream?trace=t.jsonl&token=s3cret").status_code == 200


def test_a_browser_signs_in_once_and_then_browses_by_cookie(tmp_path):
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path, token="s3cret") as client:
        page = client.get("/live/view?trace=t.jsonl")
        assert page.status_code == 401
        assert "<form" in page.text and "/live/auth" in page.text
        # The form offers to come back to what was asked for — without the
        # token the caller put in the query string.
        assert 'value="/live/view?trace=t.jsonl"' in page.text

        refused = client.post("/live/auth", data={"token": "wrong"})
        assert refused.status_code == 401
        assert client.get("/live").status_code == 401  # no cookie was set

        signed = client.post(
            "/live/auth",
            data={"token": "s3cret", "next": "/live/view?trace=t.jsonl"},
            follow_redirects=False,
        )
        assert signed.status_code == 303
        assert signed.headers["location"] == "/live/view?trace=t.jsonl"
        for url in ("/live", "/live/api/runs", "/live/view?trace=t.jsonl",
                    "/live/api/stream?trace=t.jsonl"):
            assert client.get(url).status_code == 200, url


def test_signing_in_cannot_be_bounced_off_the_site(tmp_path):
    with live_client(tmp_path, token="s3cret") as client:
        for hostile in ("https://evil.example/", "//evil.example/", "/etc/passwd"):
            signed = client.post(
                "/live/auth", data={"token": "s3cret", "next": hostile},
                follow_redirects=False,
            )
            assert signed.headers["location"] == "/live", hostile


def test_the_index_never_writes_the_token_into_a_link(tmp_path):
    write_run(tmp_path / "runs" / "t.jsonl", "r1")
    with live_client(tmp_path, token="s3cret") as client:
        client.post("/live/auth", data={"token": "s3cret"})
        page = client.get("/live")
        assert page.status_code == 200
        assert "runs/t.jsonl" in page.text
        assert "s3cret" not in page.text
        assert "token=" not in page.text


# ---------------------------------------------------------------------------
# Planning is visible before (and without) a topology (#47).
# ---------------------------------------------------------------------------


def test_a_refused_run_shows_its_rounds_and_stop_reason(tmp_path):
    write_planner_run(tmp_path / "t.jsonl", "p1", rounds=2)
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    planning = snapshot.planning
    assert planning is not None
    assert [r.round for r in planning.rounds] == [1, 2]
    assert planning.rounds[0].status == "rejected"
    assert planning.rounds[0].nodes == 3
    assert planning.rounds[0].failed_checks == ["policy"]
    assert planning.rounds[0].rejections == ["policy/edge_denied"]
    assert planning.rounds[0].executed is False
    assert planning.stop == "admission_refused"
    assert planning.stop_detail == "every round was refused"
    assert planning.planner_tokens == 40
    assert planning.has_topology is False


def test_planning_is_visible_while_the_round_is_still_in_flight(tmp_path):
    """`plan` and `admission` are on disk long before the `round` event is."""
    write_planner_run(tmp_path / "t.jsonl", "p1", rounds=2, finish=False)
    planning = build_snapshot(tmp_path, "t.jsonl", None).planning
    assert planning is not None
    assert [r.round for r in planning.rounds] == [1, 2]
    assert planning.rounds[-1].in_flight is True
    assert planning.rounds[-1].nodes == 3
    assert planning.stop is None


def test_a_planner_mid_inference_reads_as_active_not_idle(tmp_path):
    """Nothing is written between the request and the model's reply."""
    import os
    import time

    path = tmp_path / "t.jsonl"
    write_planner_run(path, "p1", rounds=1, finish=False)
    stale = time.time() - 60  # well past the activity window
    os.utime(path, (stale, stale))
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    assert snapshot.active is True
    assert snapshot.done is False


def test_a_run_with_no_planning_carries_no_planning_panel(tmp_path):
    """The execution rendering is untouched for runs that never planned."""
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    assert build_snapshot(tmp_path, "t.jsonl", None).planning is None


def test_the_stream_carries_planning_to_the_page(tmp_path):
    write_planner_run(tmp_path / "t.jsonl", "p1", rounds=1)
    with live_client(tmp_path) as client:
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            response.read()
            frames = read_sse(response)
    snapshot = [f[1] for f in frames if f[0] == "snapshot"][-1]
    assert snapshot["planning"]["stop"] == "admission_refused"
    assert snapshot["planning"]["rounds"][0]["rejections"] == ["policy/edge_denied"]


def test_a_topology_run_still_reports_its_planning(tmp_path):
    """An approved run keeps its round history; `has_topology` says a graph exists."""
    path = tmp_path / "t.jsonl"
    write_planner_run(path, "p1", rounds=1, finish=False)
    write_timed_run(path, "p1")
    planning = build_snapshot(tmp_path, "t.jsonl", None).planning
    assert planning is not None and planning.has_topology is True


# ---------------------------------------------------------------------------
# Replay at recorded speed (#48).
# ---------------------------------------------------------------------------


def test_replay_schedule_scales_and_caps(tmp_path):
    write_timed_run(tmp_path / "t.jsonl", "r1", gap=10.0)  # a 20-second run
    events = TailRecorder(tmp_path / "t.jsonl").read_events()
    plain = replay_schedule(events, 1.0)
    assert [count for count, _ in plain] == list(range(1, len(events) + 1))
    assert plain[0][1] == 0.0
    assert plain[-1][1] == pytest.approx(20.0)
    # A multiplier divides the wall clock…
    assert replay_schedule(events, 4.0)[-1][1] == pytest.approx(5.0)
    # …and the cap wins over a multiplier that would make it longer.
    assert replay_schedule(events, 0.001, max_seconds=2.0)[-1][1] == pytest.approx(2.0)
    # A nonsense speed falls back to recorded speed rather than dividing by zero.
    assert replay_schedule(events, 0.0)[-1][1] == pytest.approx(20.0)


def test_replay_sweeps_a_finished_trace_amber_then_green(tmp_path):
    write_timed_run(tmp_path / "t.jsonl", "r1", gap=1.0, nodes=("a", "b"))
    with live_client(tmp_path) as client:
        response = client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=100")
        assert response.status_code == 200
        frames = read_sse(response)
    snapshots = [f[1] for f in frames if f[0] == "snapshot"]
    assert frames[-1][0] == "done"
    assert len(snapshots) == 5  # one per recorded event
    diagrams = [s["mermaid"] for s in snapshots]
    # The amber frame the finished trace never showed: `a` running, `b` pending.
    assert any("class n0 running" in d and "class n1 pending" in d for d in diagrams)
    assert any("class n1 running" in d for d in diagrams)
    assert "class n0,n1 done" in diagrams[-1]
    assert snapshots[-1]["done"] is True


def test_a_replay_renders_identically_twice(tmp_path):
    write_timed_run(tmp_path / "t.jsonl", "r1", gap=1.0, nodes=("a", "b", "c"))
    with live_client(tmp_path) as client:
        def once():
            response = client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=500")
            return [
                (s["mermaid"], s["stats"]["events"], s["done"])
                for kind, s in read_sse(response) if kind == "snapshot"
            ]

        assert once() == once()


def test_replay_honours_the_speed_multiplier(tmp_path):
    from time import monotonic

    write_timed_run(tmp_path / "t.jsonl", "r1", gap=1.0)  # two seconds recorded
    with live_client(tmp_path) as client:
        started = monotonic()
        client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=4")
        half_speed = monotonic() - started
        started = monotonic()
        client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=200")
        fast = monotonic() - started
    assert half_speed >= 0.4  # 2s at 4x
    assert fast < half_speed


def test_replay_is_off_unless_asked_for(tmp_path):
    """No parameter, no change: one snapshot for a file that is already finished."""
    write_timed_run(tmp_path / "t.jsonl", "r1")
    with live_client(tmp_path) as client:
        with client.stream("GET", "/live/api/stream?trace=t.jsonl") as response:
            response.read()
            live = response.text
        replayed = client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=500").text
    assert len([f for f in read_sse_text(live) if f[0] == "snapshot"]) == 1
    assert len([f for f in read_sse_text(replayed) if f[0] == "snapshot"]) == 5
    assert read_sse_text(live)[-1] == ("done", {"trace": "t.jsonl"})


def test_replay_of_a_trace_that_has_not_started_is_a_waiting_frame(tmp_path):
    with live_client(tmp_path) as client:
        frames = read_sse_text(
            client.get("/live/api/stream?trace=later/t.jsonl&replay=1").text
        )
    assert frames[0][0] == "snapshot" and frames[0][1]["run_id"] is None
    assert frames[-1][0] == "done"


def test_create_app_mounts_live_only_when_asked(tmp_path):
    with TestClient(create_app()) as client:
        assert client.get("/live").status_code == 404
        assert "/live/api/runs" not in client.get("/openapi.json").json()["paths"]
    with TestClient(create_app(live_root=tmp_path)) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/healthz").status_code == 200  # existing API untouched


# ---------------------------------------------------------------------------
# The goal and the approve command reach the page — deliberately.
# ---------------------------------------------------------------------------


def _write_parked_planner_run(path, run_id="r1", *, answered=False, goal="GOAL-SENTINEL"):
    """A loop-shaped trace: labelled topology, then an approval request."""
    recorder = TraceRecorder(path)
    recorder.event(
        run_id=run_id, graph="round-1", node="topology", phase="topology", step=0,
        state_delta={
            "nodes": ["triage", "verify"],
            "edges": [["__start__", "triage", "static"], ["triage", "verify", "static"],
                      ["verify", "__end__", "static"]],
            "round": 1, "proposal_id": "p1", "fingerprint": "f1", "goal": goal,
        },
    )
    recorder.event(
        run_id=run_id, graph="loop", node="loop:approval", phase="approval_request",
        step=0,
        state_delta={"round": 1, "proposal_id": "p1", "fingerprint": "f1",
                     "nodes": ["triage", "verify"], "edges": [], "goal": goal},
    )
    if answered:
        recorder.event(
            run_id=run_id, graph="loop", node="loop:approval", phase="approval_response",
            step=0,
            state_delta={"round": 1, "proposal_id": "p1", "decision": "approved",
                         "detail": ""},
        )
    return recorder


def test_the_goal_is_lifted_and_shown_on_purpose(tmp_path):
    """The goal is the operator's own words about the run, not something a
    node wrote — it is the second state field (after termination_reason)
    shown by convention."""
    _write_parked_planner_run(tmp_path / "t.jsonl")
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    assert snapshot.goal == "GOAL-SENTINEL"
    # What the SSE frame serializes is what the page reads.
    assert snapshot.model_dump(mode="json")["goal"] == "GOAL-SENTINEL"


def test_a_goal_in_an_ordinary_delta_is_not_lifted(tmp_path):
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(run_id="r1", graph="g", node="n1", phase="start", step=1)
    recorder.event(
        run_id="r1", graph="g", node="n1", phase="end", step=1,
        state_delta={"goal": "NODE-WRITTEN", "termination_reason": "completed"},
    )
    snapshot = build_snapshot(tmp_path, "t.jsonl", None)
    assert snapshot.goal is None


def test_the_approve_command_is_shown_only_while_parked(tmp_path):
    parked = tmp_path / "parked"
    _write_parked_planner_run(parked / "t.jsonl")
    snapshot = build_snapshot(tmp_path, "parked/t.jsonl", None)
    assert snapshot.awaiting_approval is True
    assert snapshot.approve_command == f"grapharc approve {parked}"

    answered = tmp_path / "answered"
    _write_parked_planner_run(answered / "t.jsonl", answered=True)
    snapshot = build_snapshot(tmp_path, "answered/t.jsonl", None)
    assert snapshot.awaiting_approval is False
    assert snapshot.approve_command is None


def test_replay_frames_never_carry_the_approve_command(tmp_path):
    _write_parked_planner_run(tmp_path / "t.jsonl", answered=True)
    write_run(tmp_path / "t.jsonl", "r1", done=True)
    with live_client(tmp_path) as client:
        replayed = client.get("/live/api/stream?trace=t.jsonl&replay=1&speed=500")
    for kind, s in read_sse(replayed):
        if kind == "snapshot":
            assert s["approve_command"] is None
