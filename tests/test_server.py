"""HTTP API gate (ROADMAP §9.1).

Every test drives the real app through Starlette's `TestClient` (httpx under
the hood) against real graphs built on `ScriptedChatModel` — no live models, no
mocked runtime. The assertions are about behaviour the server actually
produces: trace events that match the JSONL file byte for byte, an interrupt
that stops a run before its next node, a budget that fails a session.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc.observe.trace import TraceEvent, TraceRecorder

# The trace file's own shaping rule (2000-char clip, repr fallback). Tests here
# derive expectations from it instead of hardcoding the format twice.
from grapharc.observe.trace import _jsonable as trace_shape
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.server import (
    BroadcastRecorder,
    CreateSessionRequest,
    GraphRegistry,
    InProcessRuntime,
    UnknownGraphError,
    create_app,
)
from grapharc.testing import ScriptedChatModel

#: Longer than `grapharc.observe.trace`'s 2000-character clip, which is the
#: normal case for a model answer and the case the trace/stream identity claim
#: used to break on.
LONG_ANSWER = "".join(f"{i % 10}" for i in range(2100))

# A node holds this until a test releases it, so "the session is running" is a
# fact the test established rather than a sleep it hoped was long enough.
GATE = threading.Event()
GATE_ENTERED = threading.Event()

# Two sessions must be inside the `paired` node together for its node to return.
# Replaced per test by the `reset_gates` fixture.
BARRIER = threading.Barrier(2, timeout=5)


class DemoState(GraphARCState):
    question: str
    answer: str = ""
    reviewed: str = ""


def build_demo(trace: TraceRecorder) -> CompiledGraphARC:
    """answer -> review, both scripted model calls."""
    model = ScriptedChatModel(
        responses=["42 is the answer.", "reviewed: sound"], on_exhausted="repeat"
    )

    def answer(state: DemoState) -> dict:
        return {"answer": model.invoke(state.question).content}

    def review(state: DemoState) -> dict:
        return {"reviewed": model.invoke(f"review: {state.answer}").content}

    g = GraphARC(DemoState, name="demo", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_node("review", review, writes={"reviewed"})
    g.add_edge(START, "answer")
    g.add_edge("answer", "review")
    g.add_edge("review", END)
    return g.compile()


def build_gated(trace: TraceRecorder) -> CompiledGraphARC:
    """first blocks on GATE; second must not run once an interrupt is posted."""

    def first(state: DemoState) -> dict:
        GATE_ENTERED.set()
        GATE.wait(timeout=10)
        return {"answer": "first done"}

    def second(state: DemoState) -> dict:
        return {"reviewed": "second done"}

    g = GraphARC(DemoState, name="gated", trace=trace)
    g.add_node("first", first, writes={"answer"})
    g.add_node("second", second, writes={"reviewed"})
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    return g.compile()


def build_boom(trace: TraceRecorder) -> CompiledGraphARC:
    def boom(state: DemoState) -> dict:
        raise ValueError("node exploded")

    g = GraphARC(DemoState, name="boom", trace=trace)
    g.add_node("boom", boom, writes={"answer"})
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    return g.compile()


def build_looper(trace: TraceRecorder) -> CompiledGraphARC:
    """A cycle with no exit but the budget — used to prove budgets reach the API."""

    def step(state: DemoState) -> dict:
        return {"answer": state.answer + "x"}

    def route(state: DemoState) -> str:
        return "again"

    g = GraphARC(DemoState, name="looper", trace=trace)
    g.add_node("step", step, writes={"answer"})
    g.add_edge(START, "step")
    g.add_conditional_edge("step", route, {"again": "step"})
    return g.compile()


def build_async(trace: TraceRecorder) -> CompiledGraphARC:
    """`async def` nodes — rejected by `CompiledGraphARC.stream`, fine under `astream`."""

    async def fetch(state: DemoState) -> dict:
        await asyncio.sleep(0.01)
        return {"answer": f"awaited {state.question}"}

    async def wrap(state: DemoState) -> dict:
        await asyncio.sleep(0.01)
        return {"reviewed": state.answer.upper()}

    g = GraphARC(DemoState, name="async", trace=trace)
    g.add_node("fetch", fetch, writes={"answer"})
    g.add_node("wrap", wrap, writes={"reviewed"})
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "wrap")
    g.add_edge("wrap", END)
    return g.compile()


def build_paired(trace: TraceRecorder) -> CompiledGraphARC:
    """A node that only returns if a second session is inside it at the same time."""

    def meet(state: DemoState) -> dict:
        BARRIER.wait()
        return {"answer": "met"}

    g = GraphARC(DemoState, name="paired", trace=trace)
    g.add_node("meet", meet, writes={"answer"})
    g.add_edge(START, "meet")
    g.add_edge("meet", END)
    return g.compile()


class OpaqueState(GraphARCState):
    """A state whose value pydantic has no JSON serializer for."""

    question: str
    blob: Any = None


class Opaque:
    def __repr__(self) -> str:
        return "Opaque(unserializable)"


def build_long(trace: TraceRecorder) -> CompiledGraphARC:
    """One node whose answer is longer than the trace's 2000-character clip."""

    def answer(state: DemoState) -> dict:
        return {"answer": LONG_ANSWER}

    g = GraphARC(DemoState, name="long", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()


def build_opaque(trace: TraceRecorder) -> CompiledGraphARC:
    """Writes an object `model_dump(mode="json")` refuses to serialize."""

    def stash(state: OpaqueState) -> dict:
        return {"blob": Opaque()}

    g = GraphARC(OpaqueState, name="opaque", trace=trace)
    g.add_node("stash", stash, writes={"blob"})
    g.add_edge(START, "stash")
    g.add_edge("stash", END)
    return g.compile()


@pytest.fixture(autouse=True)
def reset_gates():
    global BARRIER
    GATE.clear()
    GATE_ENTERED.clear()
    BARRIER = threading.Barrier(2, timeout=5)
    try:
        yield
    finally:
        # Safety net for tests that build their own client; the `client` fixture
        # releases earlier, because pool shutdown joins the worker threads.
        GATE.set()
        BARRIER.abort()


@pytest.fixture
def registry() -> GraphRegistry:
    return (
        GraphRegistry()
        .register("demo", build_demo)
        .register("gated", build_gated)
        .register("boom", build_boom)
        .register("looper", build_looper)
        .register("paired", build_paired)
        .register("async", build_async)
        .register("long", build_long)
        .register("opaque", build_opaque)
    )


@pytest.fixture
def client(registry, tmp_path):
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        try:
            yield c
        finally:
            # Must precede the lifespan exit: shutdown() joins the worker pool,
            # and a worker parked on the gate would hold it there.
            GATE.set()
            BARRIER.abort()


def wait_for(client, session_id, statuses, timeout=45.0) -> dict:
    """Poll a session until it reaches one of `statuses`.

    The timeout is generous on purpose. Sessions run on a background thread while
    this polls the synchronous `TestClient` in a tight loop; on a two-core CI
    runner the poller starves the worker, and 10s was not enough for the
    unbudgeted `looper` to reach LangGraph's recursion limit. A ceiling this size
    costs nothing when the session terminates promptly and only bites when it
    genuinely never will.
    """
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/sessions/{session_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.01)
    raise AssertionError(f"session stayed {body.get('status')!r}, wanted one of {statuses}")


def start(client, graph, question="q", **body) -> str:
    """Create a session and return its id, asserting the 201 on the way through."""
    payload = {"graph": graph, "input": {"question": question}, **body}
    response = client.post("/sessions", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def trace_events(client, session_id) -> list[dict]:
    text = client.get(f"/sessions/{session_id}/trace").text
    return [json.loads(line) for line in text.splitlines()]


# -- liveness ---------------------------------------------------------------


def test_healthz_reports_registered_graphs(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["graphs"] == [
        "async",
        "boom",
        "demo",
        "gated",
        "long",
        "looper",
        "opaque",
        "paired",
    ]


def test_healthz_on_an_empty_registry_says_so():
    with TestClient(create_app()) as c:
        assert c.get("/healthz").json()["graphs"] == []
        rejected = c.post("/sessions", json={"graph": "demo", "input": {"question": "q"}})
        assert rejected.status_code == 404
        assert "registered graphs: none" in rejected.json()["detail"]


# -- create / poll ----------------------------------------------------------


def test_create_then_poll_to_completion(client):
    created = client.post(
        "/sessions", json={"graph": "demo", "input": {"question": "meaning?"}}
    )
    assert created.status_code == 201
    view = created.json()
    assert view["graph"] == "demo"
    assert view["status"] in {"queued", "running"}
    assert view["result"] is None
    assert created.headers["location"] == f"/sessions/{view['id']}"

    done = wait_for(client, view["id"], {"succeeded"})
    assert done["result"] == {
        "question": "meaning?",
        "answer": "42 is the answer.",
        "reviewed": "reviewed: sound",
    }
    assert done["error"] is None
    assert done["run_id"]
    assert done["ended_at"] and done["started_at"]
    # The scripted model reports usage, so the meter must have seen both calls.
    assert done["usage"]["iterations"] == 2
    assert done["usage"]["tokens"] > 0
    assert done["event_count"] == 5  # topology, then start+end for each of two nodes


def test_thread_id_defaults_to_the_session_id_and_is_honoured_when_given(client):
    default = client.post(
        "/sessions", json={"graph": "demo", "input": {"question": "a"}}
    ).json()
    assert default["thread_id"] == default["id"]

    named = client.post(
        "/sessions",
        json={"graph": "demo", "input": {"question": "a"}, "thread_id": "conv-7"},
    ).json()
    assert named["thread_id"] == "conv-7"
    wait_for(client, named["id"], {"succeeded"})
    assert all(e["thread_id"] == "conv-7" for e in trace_events(client, named["id"]))


def test_listing_returns_sessions_in_creation_order(client):
    ids = [
        client.post("/sessions", json={"graph": "demo", "input": {"question": str(i)}})
        .json()["id"]
        for i in range(3)
    ]
    listed = [s["id"] for s in client.get("/sessions").json()["sessions"]]
    assert listed == ids


def test_a_failing_node_fails_the_session_with_the_exception(client):
    sid = start(client, "boom")
    done = wait_for(client, sid, {"failed"})
    assert done["error"] == "ValueError: node exploded"
    assert done["result"] is None
    # The failure is in the trace too, not only in the status field.
    assert any(
        e["phase"] == "error" for e in trace_events(client, sid)
    )


def test_a_budget_in_the_request_bounds_the_run(client):
    sid = client.post(
        "/sessions",
        json={
            "graph": "looper",
            "input": {"question": "spin"},
            "budget": {"max_iterations": 3},
        },
    ).json()["id"]
    done = wait_for(client, sid, {"failed"})
    assert "max_iterations reached (3/3)" in done["error"]
    assert done["usage"]["iterations"] == 3


def test_without_a_budget_the_same_graph_is_not_bounded_by_the_server(client):
    """Guards against the budget field being decoration: no budget, no ceiling of 3."""
    sid = client.post(
        "/sessions", json={"graph": "looper", "input": {"question": "spin"}}
    ).json()["id"]
    done = wait_for(client, sid, {"failed"})
    assert "max_iterations" not in (done["error"] or "")
    assert done["usage"]["iterations"] > 3


def test_a_graph_of_async_nodes_runs(client):
    """`CompiledGraphARC.stream` refuses async nodes; the runtime drives `astream`."""
    sid = start(client, "async", question="now")
    done = wait_for(client, sid, {"succeeded"})
    assert done["result"] == {
        "question": "now",
        "answer": "awaited now",
        "reviewed": "AWAITED NOW",
    }
    assert [e["node"] for e in trace_events(client, sid)] == [
        "topology",
        "fetch",
        "fetch",
        "wrap",
        "wrap",
    ]


def test_two_sessions_run_at_the_same_time(client):
    """The `paired` node only returns if both sessions are inside it at once.

    Run serially, the barrier times out and both sessions fail — so the pass is
    real concurrency, not two fast sequential runs.
    """
    a = start(client, "paired")
    b = start(client, "paired")
    for sid in (a, b):
        assert wait_for(client, sid, {"succeeded", "failed"})["status"] == "succeeded"
    assert client.get(f"/sessions/{a}").json()["result"]["answer"] == "met"
    assert client.get(f"/sessions/{a}").json()["run_id"] != client.get(
        f"/sessions/{b}"
    ).json()["run_id"]


def test_worker_pool_bounds_concurrency(registry, tmp_path):
    app = create_app(
        registry=registry, root=tmp_path / "sessions", max_workers=1, poll_seconds=0.005
    )
    with TestClient(app) as c:
        blocking = start(c, "gated")
        assert GATE_ENTERED.wait(timeout=10), "the gated node never started"
        queued = start(c, "demo")
        # One worker, and it is parked: the second session cannot have started.
        assert c.get(f"/sessions/{queued}").json()["status"] == "queued"
        assert c.get(f"/sessions/{queued}").json()["event_count"] == 0
        GATE.set()
        wait_for(c, blocking, {"succeeded"})
        wait_for(c, queued, {"succeeded"})


# -- events -----------------------------------------------------------------


def test_interrupt_stops_the_run_before_the_next_node(client):
    sid = start(client, "gated")
    assert GATE_ENTERED.wait(timeout=10), "the gated node never started"

    ack = client.post(f"/sessions/{sid}/events", json={"type": "interrupt"})
    assert ack.status_code == 202
    body = ack.json()
    assert body["event"]["applied"] is True
    assert "next node boundary" in body["event"]["detail"]

    GATE.set()
    done = wait_for(client, sid, {"interrupted"})
    # "first" completed, "second" never started — that is the whole claim. The
    # result is the last superstep's state, so `reviewed` is absent rather than
    # empty: it was never written.
    assert done["result"] == {"question": "q", "answer": "first done"}
    nodes = {e["node"] for e in trace_events(client, sid)}
    assert nodes == {"topology", "first"}


def test_interrupt_on_a_finished_session_is_reported_as_not_applied(client):
    sid = start(client, "demo")
    wait_for(client, sid, {"succeeded"})
    body = client.post(f"/sessions/{sid}/events", json={"type": "interrupt"}).json()
    assert body["event"]["applied"] is False
    assert "already succeeded" in body["event"]["detail"]
    assert client.get(f"/sessions/{sid}").json()["status"] == "succeeded"


def test_message_and_approval_are_recorded_but_not_delivered(client):
    sid = start(client, "demo")
    wait_for(client, sid, {"succeeded"})
    for kind in ("message", "approval"):
        body = client.post(
            f"/sessions/{sid}/events", json={"type": kind, "data": {"text": "hi"}}
        ).json()
        assert body["event"]["applied"] is False
        assert "does not deliver" in body["event"]["detail"]

    inbox = client.get(f"/sessions/{sid}").json()["inbox"]
    assert [e["type"] for e in inbox] == ["message", "approval"]
    assert inbox[0]["data"] == {"text": "hi"}


# -- streaming --------------------------------------------------------------


def read_sse(response) -> list[tuple[str, dict, str | None]]:
    """Parse an SSE body into (event, data, id) triples, dropping comments."""
    frames = []
    for block in response.text.split("\n\n"):
        name = data = event_id = None
        for line in block.splitlines():
            if line.startswith(":") or not line.strip():
                continue
            key, _, value = line.partition(": ")
            if key == "event":
                name = value
            elif key == "data":
                data = value
            elif key == "id":
                event_id = value
        if name is not None:
            frames.append((name, json.loads(data), event_id))
    return frames


def test_stream_replays_a_finished_run_and_terminates(client):
    sid = start(client, "demo")
    wait_for(client, sid, {"succeeded"})

    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        response.read()
        frames = read_sse(response)

    kinds = [f[0] for f in frames]
    assert kinds == ["trace"] * 5 + ["status", "done"]
    assert [f[2] for f in frames[:5]] == ["1", "2", "3", "4", "5"]
    assert [f[1]["node"] for f in frames[:5]] == [
        "topology",
        "answer",
        "answer",
        "review",
        "review",
    ]
    assert [f[1]["phase"] for f in frames[:5]] == [
        "topology",
        "start",
        "end",
        "start",
        "end",
    ]
    assert frames[-2][1]["status"] == "succeeded"
    assert frames[-2][1]["result"]["answer"] == "42 is the answer."


def test_stream_carries_events_recorded_after_it_opened(client):
    """Not a replay: three of the four events do not exist when the request is made.

    `TestClient` buffers a streaming response to completion, so incremental
    arrival is proved separately in `test_sse_arrives_incrementally_over_a_socket`.
    """
    sid = start(client, "gated")
    assert GATE_ENTERED.wait(timeout=10), "the gated node never started"
    opened_with = client.get(f"/sessions/{sid}").json()
    assert opened_with["status"] == "running"
    assert opened_with["event_count"] == 2  # the topology event, then `first`'s start

    threading.Timer(0.2, GATE.set).start()
    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        response.read()
        frames = read_sse(response)

    assert [f[0] for f in frames] == ["trace"] * 5 + ["status", "done"]
    assert [f[1]["node"] for f in frames[:5]] == [
        "topology",
        "first",
        "first",
        "second",
        "second",
    ]
    assert frames[-2][1]["status"] == "succeeded"


def test_stream_cursor_skips_events_already_seen(client):
    sid = start(client, "demo")
    wait_for(client, sid, {"succeeded"})

    with client.stream("GET", f"/sessions/{sid}/stream?cursor=4") as response:
        response.read()
        frames = read_sse(response)
    assert [f[0] for f in frames] == ["trace", "status", "done"]
    assert frames[0][2] == "5"
    assert frames[0][1]["node"] == "review"
    assert frames[0][1]["phase"] == "end"

    with client.stream(
        "GET", f"/sessions/{sid}/stream", headers={"last-event-id": "4"}
    ) as response:
        response.read()
        resumed = read_sse(response)
    assert [f[0] for f in resumed] == ["trace", "status", "done"]
    assert resumed[0][2] == "5"


def test_stream_events_are_the_trace_file(client):
    """One funnel: the SSE payloads and the JSONL lines must be the same records."""
    sid = start(client, "demo")
    wait_for(client, sid, {"succeeded"})

    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        response.read()
        streamed = [f[1] for f in read_sse(response) if f[0] == "trace"]
    on_disk = trace_events(client, sid)
    assert streamed == on_disk


def raw_sse_data(response) -> list[str]:
    """The `data:` lines exactly as they went over the wire, unparsed."""
    return [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_and_trace_are_the_same_record_for_an_answer_over_2000_chars(client):
    """The headline claim, on the payload that used to falsify it.

    A model answer longer than the trace's 2000-character clip took two
    different serializations to the two views: the SSE frame clipped it once
    (2022 chars) and the JSONL line clipped the already-clipped string again
    (2021 chars). Short ASCII cannot see that, so this test uses the normal
    case. The bytes must match, not merely the parsed objects.
    """
    sid = start(client, "long")
    done = wait_for(client, sid, {"succeeded"})

    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        response.read()
        frames = read_sse(response)
        streamed_lines = raw_sse_data(response)[: len(trace_events(client, sid))]
    streamed = [f[1] for f in frames if f[0] == "trace"]
    disk_lines = client.get(f"/sessions/{sid}/trace").text.splitlines()
    on_disk = [json.loads(line) for line in disk_lines]

    assert len(on_disk) == 3 and len(streamed) == 3, (len(on_disk), len(streamed))
    assert streamed == on_disk
    assert streamed_lines == disk_lines  # byte for byte, not just equal objects

    # Clipped once by the trace format's own rule, not twice. Derived from that
    # format rather than hardcoded, so the assertion is "shaped exactly once"
    # and not "shaped the way it was shaped in July 2026".
    clipped = on_disk[-1]["state_delta"]["answer"]
    once = trace_shape({"answer": LONG_ANSWER})["answer"]
    twice = trace_shape({"answer": once})["answer"]
    assert len(clipped) < len(LONG_ANSWER), "nothing was clipped; the test proves nothing"
    assert clipped == once
    assert clipped != twice  # the two-pass string is what used to land on disk

    # The full answer is still the caller's answer: only the trace is clipped.
    assert done["result"]["answer"] == LONG_ANSWER


def test_a_recorded_line_matches_what_a_plain_trace_recorder_would_write(tmp_path):
    """`BroadcastRecorder` must not invent its own trace format.

    It writes the line itself (delegating would shape the event twice), so this
    pins the output to `TraceRecorder.record`'s for the same event — including
    the clip, which is the part that drifts silently.
    """
    seen: list[dict] = []
    broadcast = BroadcastRecorder(tmp_path / "broadcast.jsonl", seen.append)
    plain = TraceRecorder(tmp_path / "plain.jsonl")
    event = TraceEvent(
        ts="2026-07-28T00:00:00.000+00:00",
        run_id="r1",
        thread_id="t1",
        graph="g",
        node="n",
        phase="end",
        step=2,
        state_delta={"answer": LONG_ANSWER, "n": 3, "ok": True},
        duration_ms=1.5,
    )
    broadcast.record(event)
    plain.record(event)

    assert (tmp_path / "broadcast.jsonl").read_text() == (tmp_path / "plain.jsonl").read_text()
    assert seen == [json.loads((tmp_path / "plain.jsonl").read_text())]


def test_a_sink_exception_cannot_fail_the_node_that_recorded_the_event(tmp_path):
    """A subscriber is not allowed to break the run it is watching."""

    def poisoned(payload: dict) -> None:
        raise RuntimeError("subscriber exploded")

    recorder = BroadcastRecorder(tmp_path / "trace.jsonl", poisoned)
    recorder.event(run_id="r1", graph="g", node="n", phase="start", step=1)

    assert recorder.sink_errors == 1
    # The write happened first, so the audit trail keeps the event the live
    # stream lost — and the counter says the two now disagree.
    assert len(recorder.read_events()) == 1


def test_a_runtime_whose_fanout_raises_still_finishes_the_session(registry, tmp_path):
    """End to end: the same guarantee through the HTTP layer."""

    class PoisonedRuntime(InProcessRuntime):
        def _on_trace_event(self, session_id: str, payload: dict) -> None:
            raise RuntimeError("fan-out exploded")

    runtime = PoisonedRuntime(registry, root=tmp_path / "sessions")
    with TestClient(create_app(runtime=runtime, poll_seconds=0.005)) as c:
        sid = start(c, "demo")
        done = wait_for(c, sid, {"succeeded", "failed"})
        assert done["status"] == "succeeded", done["error"]
        assert done["result"]["answer"] == "42 is the answer."
        # The trace file is complete even though every subscriber call failed.
        assert len(trace_events(c, sid)) == 5
        assert done["event_count"] == 0


def test_a_state_value_json_cannot_hold_is_recorded_on_both_views(client):
    """`model_dump(mode="json")` raises on it; that must not become a node error."""
    sid = client.post(
        "/sessions", json={"graph": "opaque", "input": {"question": "q"}}
    ).json()["id"]
    done = wait_for(client, sid, {"succeeded", "failed"})
    assert done["status"] == "succeeded", done["error"]

    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        response.read()
        streamed = [f[1] for f in read_sse(response) if f[0] == "trace"]
    on_disk = trace_events(client, sid)
    assert streamed == on_disk
    assert on_disk[-1]["state_delta"]["blob"] == "Opaque(unserializable)"


# -- streaming over a real socket -------------------------------------------


@pytest.fixture
def live_server(registry, tmp_path):
    """A real uvicorn server on a loopback port.

    `TestClient` runs the ASGI app to completion and buffers the body, which
    makes it structurally unable to show that SSE frames arrive *before* the run
    ends. This fixture is the only way to assert that.
    """
    import uvicorn

    GATE.clear()
    GATE_ENTERED.clear()
    app = create_app(registry=registry, root=tmp_path / "live", poll_seconds=0.005)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    sock = config.bind_socket()
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        GATE.set()
        server.should_exit = True
        thread.join(timeout=15)


def next_frame(lines) -> tuple[str, dict]:
    """Consume one SSE frame from an iterator of lines."""
    name = data = None
    for line in lines:
        if line == "":
            if name is not None:
                return name, json.loads(data)
            continue
        key, _, value = line.partition(": ")
        if key == "event":
            name = value
        elif key == "data":
            data = value
    raise AssertionError("stream ended before a complete frame arrived")


def test_sse_arrives_incrementally_over_a_socket(live_server):
    import httpx

    with httpx.Client(base_url=live_server, timeout=20.0) as http:
        sid = http.post(
            "/sessions", json={"graph": "gated", "input": {"question": "q"}}
        ).json()["id"]
        assert GATE_ENTERED.wait(timeout=10), "the gated node never started"

        with http.stream("GET", f"/sessions/{sid}/stream") as response:
            assert response.status_code == 200
            lines = response.iter_lines()
            name, event = next_frame(lines)
            assert (name, event["phase"]) == ("trace", "topology")
            name, event = next_frame(lines)
            # This frame reached the client while the graph is still parked in
            # `first` — nothing has finished, so nothing could have been replayed.
            assert (name, event["node"], event["phase"]) == ("trace", "first", "start")
            assert http.get(f"/sessions/{sid}").json()["status"] == "running"

            GATE.set()
            rest = []
            while True:
                name, event = next_frame(lines)
                rest.append((name, event))
                if name == "done":
                    break

    assert [name for name, _ in rest] == ["trace", "trace", "trace", "status", "done"]
    assert [e["node"] for _, e in rest[:3]] == ["first", "second", "second"]
    assert rest[-2][1]["status"] == "succeeded"


# -- trace ------------------------------------------------------------------


def test_trace_is_ndjson_and_empty_before_the_first_event(client):
    sid = start(client, "gated")
    response = client.get(f"/sessions/{sid}/trace")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    GATE.set()
    wait_for(client, sid, {"succeeded"})
    events = trace_events(client, sid)
    assert len(events) == 5
    assert {e["graph"] for e in events} == {"gated"}


def test_sessions_do_not_share_a_trace_file(client):
    a = start(client, "demo", question="a")
    b = start(client, "demo", question="b")
    wait_for(client, a, {"succeeded"})
    wait_for(client, b, {"succeeded"})

    for sid, question in ((a, "a"), (b, "b")):
        events = trace_events(client, sid)
        assert len(events) == 5
        assert {e["run_id"] for e in events} == {client.get(f"/sessions/{sid}").json()["run_id"]}
        assert client.get(f"/sessions/{sid}").json()["result"]["question"] == question


# -- 404 / 422 --------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/sessions/nope"),
        ("GET", "/sessions/nope/trace"),
        ("GET", "/sessions/nope/stream"),
    ],
)
def test_unknown_session_is_404(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 404
    assert "unknown session 'nope'" in response.json()["detail"]


def test_unknown_session_on_events_is_404(client):
    response = client.post("/sessions/nope/events", json={"type": "interrupt"})
    assert response.status_code == 404


def test_unknown_graph_is_404_and_names_what_is_registered(client):
    response = client.post("/sessions", json={"graph": "nosuch", "input": {}})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "unknown graph 'nosuch'" in detail
    assert "'demo'" in detail


@pytest.mark.parametrize(
    "body",
    [
        {},                                              # graph missing
        {"graph": ""},                                   # graph empty
        {"graph": "demo", "input": "not-an-object"},     # input wrong type
        {"graph": "demo", "extra": 1},                   # unknown field
        {"graph": "demo", "budget": {"max_iterations": 0}},      # must be > 0
        {"graph": "demo", "budget": {"max_widgets": 4}},         # unknown budget key
    ],
)
def test_malformed_create_bodies_are_422(client, body):
    assert client.post("/sessions", json=body).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {},                                    # type missing
        {"type": "bogus"},                     # not one of the three
        {"type": "message", "data": "text"},   # data must be an object
        {"type": "message", "extra": 1},       # unknown field
    ],
)
def test_malformed_event_bodies_are_422(client, body):
    sid = start(client, "demo")
    assert client.post(f"/sessions/{sid}/events", json=body).status_code == 422


def test_bad_stream_cursor_is_422(client):
    sid = start(client, "demo")
    assert client.get(f"/sessions/{sid}/stream?cursor=-1").status_code == 422


@pytest.mark.parametrize(
    "graph_input,expected_loc",
    [
        ({}, ["body", "input", "question"]),                       # required field missing
        ({"question": 5}, ["body", "input", "question"]),          # wrong type
        ({"question": "q", "quesiton": "typo"}, ["body", "input", "quesiton"]),
    ],
)
def test_input_is_validated_against_the_graph_state_schema(client, graph_input, expected_loc):
    """A body the graph could not accept is a 422, not a session that fails later.

    The typo case is the one that needs a server: LangGraph drops unrecognised
    input keys before the state schema sees them, so without this check
    `{"quesiton": ...}` would start a run against the wrong question.
    """
    before = len(client.get("/sessions").json()["sessions"])
    response = client.post("/sessions", json={"graph": "demo", "input": graph_input})
    assert response.status_code == 422
    assert any(err["loc"] == expected_loc for err in response.json()["detail"]), response.text
    # Rejected at admission: no session was created to fail afterwards.
    assert len(client.get("/sessions").json()["sessions"]) == before


# -- wiring -----------------------------------------------------------------


def test_openapi_schema_generates_and_covers_every_route(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/healthz",
        "/sessions",
        "/sessions/{session_id}",
        "/sessions/{session_id}/events",
        "/sessions/{session_id}/stream",
        "/sessions/{session_id}/trace",
    }
    assert schema["paths"]["/sessions"]["post"]["responses"]["201"]
    assert schema["paths"]["/sessions/{session_id}/events"]["post"]["responses"]["202"]


@pytest.mark.parametrize(
    "extra", [{"registry": GraphRegistry()}, {"root": "/tmp/x"}, {"max_workers": 2}]
)
def test_create_app_refuses_an_ambiguous_configuration(extra):
    with pytest.raises(ValueError, match="either runtime"):
        create_app(runtime=InProcessRuntime(GraphRegistry()), **extra)


def test_in_process_runtime_satisfies_the_session_runtime_protocol():
    from grapharc.server.runtime import SessionRuntime

    assert isinstance(InProcessRuntime(GraphRegistry()), SessionRuntime)


@pytest.mark.parametrize(
    "body,status",
    [
        ({"graph": "nope", "input": {}}, 404),          # unknown graph
        ({"graph": "demo", "input": {}}, 422),          # input the schema rejects
    ],
)
def test_a_rejected_create_leaves_no_session_and_no_trace_directory(
    registry, tmp_path, body, status
):
    root = tmp_path / "sessions"
    with TestClient(create_app(registry=registry, root=root)) as c:
        assert c.post("/sessions", json=body).status_code == status
        assert c.get("/sessions").json()["sessions"] == []
    assert list(root.iterdir()) == []


def test_a_builder_that_raises_leaves_no_session_and_no_trace_directory(tmp_path):
    """The recorder makes the session directory before the builder runs.

    Only the schema-rejection path used to clean it up, so every failed build —
    a bad API key, a missing file, an operator's typo — left an empty directory
    behind, one per attempt.
    """
    root = tmp_path / "sessions"
    registry = GraphRegistry().register(
        "broken", lambda trace: (_ for _ in ()).throw(RuntimeError("builder exploded"))
    )
    runtime = InProcessRuntime(registry, root=root)
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="builder exploded"):
                runtime.create(CreateSessionRequest(graph="broken", input={"question": "q"}))
        assert runtime.list_sessions() == []
        assert list(root.iterdir()) == []
    finally:
        runtime.shutdown()


# -- checkpointers ----------------------------------------------------------


@pytest.fixture
def sqlite_saver(tmp_path):
    """The repo's own checkpointer: `SqliteSaver`, which has no async side."""
    conn = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    yield saver
    conn.close()


def _checkpointed(builder, checkpointer):
    """Wrap a graph builder so the compiled graph carries `checkpointer`."""

    def build(trace: TraceRecorder) -> CompiledGraphARC:
        compiled = builder(trace)
        return compiled.arc.compile(checkpointer=checkpointer)

    return build


def test_a_sync_only_checkpointer_runs_through_the_http_api(sqlite_saver, tmp_path):
    """`SqliteSaver` is a hard dependency and what `grapharc.session` uses.

    Driven through `astream` it raises "The SqliteSaver does not support async
    methods" before the first node, so the session failed with `event_count` 0
    and nothing said the checkpointer was the reason.
    """
    registry = GraphRegistry().register("saved", _checkpointed(build_demo, sqlite_saver))
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        sid = start(c, "saved", thread_id="conv-1")
        done = wait_for(c, sid, {"succeeded", "failed"})

        assert done["status"] == "succeeded", done["error"]
        assert done["result"]["answer"] == "42 is the answer."
        assert done["event_count"] == 5
        assert len(trace_events(c, sid)) == 5

    # The checkpointer was actually used, not quietly bypassed.
    saved = sqlite_saver.get_tuple({"configurable": {"thread_id": "conv-1"}})
    assert saved is not None
    assert saved.checkpoint["channel_values"]["answer"] == "42 is the answer."


def test_the_sync_driver_honours_an_interrupt_at_the_same_boundary(sqlite_saver, tmp_path):
    """The fallback driver is not a second set of semantics."""
    registry = GraphRegistry().register("saved", _checkpointed(build_gated, sqlite_saver))
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        try:
            sid = start(c, "saved")
            assert GATE_ENTERED.wait(timeout=10), "the gated node never started"
            ack = c.post(f"/sessions/{sid}/events", json={"type": "interrupt"}).json()
            assert ack["event"]["applied"] is True
            GATE.set()
            done = wait_for(c, sid, {"interrupted", "failed", "succeeded"})
            assert done["status"] == "interrupted", done["error"]
            assert done["result"] == {"question": "q", "answer": "first done"}
            assert {e["node"] for e in trace_events(c, sid)} == {"topology", "first"}
        finally:
            GATE.set()


def test_async_nodes_with_a_sync_only_checkpointer_fail_with_a_named_fix(
    sqlite_saver, tmp_path
):
    """The one combination neither driver can serve says so, loudly."""
    registry = GraphRegistry().register("saved", _checkpointed(build_async, sqlite_saver))
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        sid = start(c, "saved", question="now")
        done = wait_for(c, sid, {"succeeded", "failed"})

    assert done["status"] == "failed"
    error = done["error"]
    assert error.startswith("CheckpointerNotAsyncError:")
    assert "SqliteSaver" in error
    assert "AsyncSqliteSaver" in error and "InMemorySaver" in error
    assert "stream()" in error


def test_an_async_capable_checkpointer_still_runs_on_astream(tmp_path):
    """The probe must not push a perfectly good async saver onto the sync path.

    `build_async` is `async def` throughout, so a run that finishes at all
    proves `astream` drove it — the sync entry point refuses those graphs.
    """
    saver = InMemorySaver()
    registry = GraphRegistry().register("saved", _checkpointed(build_async, saver))
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        sid = start(c, "saved", question="now", thread_id="conv-a")
        done = wait_for(c, sid, {"succeeded", "failed"})

    assert done["status"] == "succeeded", done["error"]
    assert done["result"] == {"question": "now", "answer": "awaited now", "reviewed": "AWAITED NOW"}
    assert saver.get_tuple({"configurable": {"thread_id": "conv-a"}}) is not None


def test_the_probe_writes_no_checkpoint_of_its_own(sqlite_saver, tmp_path):
    """A capability probe that left state behind would be a side effect, not a probe."""
    from grapharc.server.runtime import _PROBE_THREAD_ID

    registry = GraphRegistry().register("saved", _checkpointed(build_demo, sqlite_saver))
    app = create_app(registry=registry, root=tmp_path / "sessions", poll_seconds=0.005)
    with TestClient(app) as c:
        wait_for(c, start(c, "saved"), {"succeeded", "failed"})

    assert sqlite_saver.get_tuple({"configurable": {"thread_id": _PROBE_THREAD_ID}}) is None


def test_registry_build_rejects_an_unknown_name(tmp_path):
    registry = GraphRegistry().register("demo", build_demo)
    with pytest.raises(UnknownGraphError) as raised:
        registry.build("nope", TraceRecorder(tmp_path / "t.jsonl"))
    assert raised.value.known == ["demo"]
    assert "unknown graph 'nope'" in str(raised.value)


def test_registry_rejects_duplicate_and_empty_names():
    registry = GraphRegistry().register("demo", build_demo)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", build_demo)
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", build_demo)


def test_registry_can_be_built_from_a_mapping():
    registry = GraphRegistry({"demo": build_demo, "boom": build_boom})
    assert registry.names() == ["boom", "demo"]
    assert len(registry) == 2
    assert list(registry) == ["boom", "demo"]
    assert "demo" in registry and "nope" not in registry


def test_shutdown_stops_accepting_sessions(registry, tmp_path):
    runtime = InProcessRuntime(registry, root=tmp_path / "s")
    app = create_app(runtime=runtime, poll_seconds=0.005)
    body = {"graph": "demo", "input": {"question": "q"}}
    with TestClient(app) as c:
        assert c.post("/sessions", json=body).status_code == 201
    # The lifespan exit shut the runtime down; further creates are 503, not 500.
    with TestClient(app) as c:
        assert c.post("/sessions", json=body).status_code == 503


def test_serve_hands_the_app_to_uvicorn(monkeypatch):
    import uvicorn

    from grapharc.server import serve

    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    app = create_app()
    serve(app, host="0.0.0.0", port=1234)
    assert seen["app"] is app
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 1234)
