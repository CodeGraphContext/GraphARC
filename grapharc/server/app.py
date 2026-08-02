"""GraphARC as an HTTP service (ROADMAP §9.1).

    POST /sessions               create a session          -> 201 SessionView
    GET  /sessions               list sessions             -> 200 SessionList
    GET  /sessions/{id}          status + result           -> 200 SessionView
    POST /sessions/{id}/events   message / approval / stop -> 202 EventAck
    GET  /sessions/{id}/stream   SSE of the run's traces   -> 200 text/event-stream
    GET  /sessions/{id}/trace    the JSONL trace           -> 200 application/x-ndjson
    GET  /healthz                liveness                  -> 200 Health

The app owns no execution logic. It validates, maps `SessionRuntime` errors onto
status codes, and turns the runtime's event log into SSE frames — so replacing
the in-process default with the real session layer (ROADMAP §6) touches nothing
in this module.

A request may name a registered graph and supply input and a budget. It may not
describe a graph: topology comes from `GraphRegistry`, which the operator fills
in Python.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from grapharc import __version__
from grapharc.server.errors import (
    InvalidGraphInputError,
    RuntimeClosedError,
    UnknownGraphError,
    UnknownSessionError,
)
from grapharc.server.models import (
    CreateSessionRequest,
    EventAck,
    Health,
    SessionEventRequest,
    SessionList,
    SessionView,
)
from grapharc.server.registry import GraphRegistry
from grapharc.server.runtime import InProcessRuntime, SessionRuntime

#: How often an open SSE stream re-checks the session for new trace events.
#: A poll rather than a condition variable because the run lives on a worker
#: thread and the reader on the event loop; the read is a list slice.
DEFAULT_POLL_SECONDS = 0.02

#: Idle seconds before the stream emits an SSE comment, so proxies and clients
#: do not time a quiet-but-live run out.
DEFAULT_KEEPALIVE_SECONDS = 15.0


def _sse(event: str, data: Any, *, event_id: int | None = None) -> str:
    """One SSE frame. `default=repr` because a frame must never fail to encode.

    These `json.dumps` arguments match the ones `BroadcastRecorder` writes the
    trace file with, and a trace payload reaches here already shaped, so
    `default` never fires for one: the `data:` line of a `trace` frame is byte
    for byte the JSONL line. Changing either side breaks that, and
    `test_stream_and_trace_are_the_same_record_for_an_answer_over_2000_chars`
    says so.
    """
    payload = json.dumps(data, ensure_ascii=False, default=repr)
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {payload}\n\n"


async def _disconnected(request: Request) -> bool:
    try:
        return await request.is_disconnected()
    except Exception:
        # Disconnect detection is an optimisation; the terminal-status check is
        # what actually ends the stream.
        return False


def create_app(
    *,
    runtime: SessionRuntime | None = None,
    registry: GraphRegistry | None = None,
    root: str | Path | None = None,
    max_workers: int | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    keepalive_seconds: float | None = DEFAULT_KEEPALIVE_SECONDS,
    live_root: str | Path | None = None,
    live_token: str | None = None,
) -> FastAPI:
    """Build the app.

    Pass `registry` to run graphs on the default `InProcessRuntime`, or
    `runtime` to supply your own. Passing both is a configuration error rather
    than a silent precedence rule — `registry`, `root` and `max_workers`
    configure the default runtime and mean nothing next to a supplied one. With
    neither, the registry is empty and every create request gets a 404 naming
    that fact.

    `live_root` mounts the read-only live view (`grapharc.server.live`) over
    the trace files under that directory. It is storage to *read*, not runtime
    configuration, so it composes with either a registry or a supplied runtime
    — the Slack topology is a live root and no registry at all.
    """
    if runtime is not None and any(
        arg is not None for arg in (registry, root, max_workers)
    ):
        raise ValueError(
            "pass either runtime=... or registry=/root=/max_workers= — a supplied "
            "runtime already owns its registry, storage and worker pool"
        )
    session_runtime: SessionRuntime = runtime or InProcessRuntime(
        registry or GraphRegistry(), root=root, max_workers=max_workers or 4
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Running sessions are asked to stop at their next node boundary; the
        # pool is then drained so the process can actually exit.
        await asyncio.to_thread(session_runtime.shutdown)

    app = FastAPI(
        title="GraphARC",
        version=__version__,
        summary="Run disciplined agent graphs as sessions over HTTP.",
        lifespan=lifespan,
    )
    app.state.runtime = session_runtime

    if live_root is not None:
        from grapharc.server.live import live_router

        app.include_router(live_router(Path(live_root), token=live_token))

    @app.exception_handler(UnknownSessionError)
    def _unknown_session(request: Request, exc: UnknownSessionError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnknownGraphError)
    def _unknown_graph(request: Request, exc: UnknownGraphError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidGraphInputError)
    def _invalid_input(request: Request, exc: InvalidGraphInputError) -> JSONResponse:
        # Same 422 shape FastAPI uses for its own body validation, so a client
        # has one error format to parse.
        return JSONResponse(status_code=422, content={"detail": exc.errors})

    @app.exception_handler(RuntimeClosedError)
    def _runtime_closed(request: Request, exc: RuntimeClosedError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        return Health(status="ok", version=__version__, graphs=session_runtime.graph_names())

    @app.post("/sessions", response_model=SessionView, status_code=201)
    def create_session(body: CreateSessionRequest, response: Response) -> SessionView:
        view = session_runtime.create(body)
        response.headers["location"] = f"/sessions/{view.id}"
        return view

    @app.get("/sessions", response_model=SessionList)
    def list_sessions() -> SessionList:
        return SessionList(sessions=session_runtime.list_sessions())

    @app.get("/sessions/{session_id}", response_model=SessionView)
    def get_session(session_id: str) -> SessionView:
        return session_runtime.get(session_id)

    @app.post("/sessions/{session_id}/events", response_model=EventAck, status_code=202)
    def post_event(session_id: str, body: SessionEventRequest) -> EventAck:
        """202 always: the event was accepted for the session.

        Whether it changed the run is `event.applied` in the body — an interrupt
        on a live session applies, an interrupt on a finished one does not, and
        this runtime never applies `message` or `approval`.
        """
        return session_runtime.submit_event(session_id, body)

    @app.get("/sessions/{session_id}/trace")
    def get_trace(session_id: str) -> PlainTextResponse:
        return PlainTextResponse(
            session_runtime.trace_text(session_id), media_type="application/x-ndjson"
        )

    @app.get("/sessions/{session_id}/stream")
    async def stream_session(
        session_id: str,
        request: Request,
        cursor: int = Query(0, ge=0, description="skip this many already-seen events"),
    ) -> StreamingResponse:
        # Resolve the id before the response starts: once the body is streaming
        # a 404 can no longer be sent.
        session_runtime.get(session_id)
        start = cursor
        resume = request.headers.get("last-event-id")
        if not cursor and resume and resume.isdigit():
            start = int(resume)

        async def frames() -> AsyncIterator[str]:
            position = start
            idle = 0.0
            while True:
                page = session_runtime.events_since(session_id, position)
                for event in page.events:
                    position += 1
                    idle = 0.0
                    yield _sse("trace", event, event_id=position)
                if page.terminal:
                    view = session_runtime.get(session_id)
                    yield _sse("status", view.model_dump(mode="json"))
                    yield _sse("done", {"session_id": session_id, "cursor": position})
                    return
                if await _disconnected(request):
                    return
                await asyncio.sleep(poll_seconds)
                idle += poll_seconds
                if keepalive_seconds is not None and idle >= keepalive_seconds:
                    idle = 0.0
                    yield ": keepalive\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    return app


def serve(app: FastAPI, *, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
    """Run `app` under uvicorn. Imported lazily so the app works without it."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, **kwargs)


__all__ = ["DEFAULT_KEEPALIVE_SECONDS", "DEFAULT_POLL_SECONDS", "create_app", "serve"]
