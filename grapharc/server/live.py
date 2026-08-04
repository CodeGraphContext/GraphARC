"""A live browser view over trace files another process is appending.

Mounted into the FastAPI app only when the operator passes `--live-root` to
`grapharc serve`. The routes are GET-only readers over JSONL trace files under
that root — typically the Slack bot's working directory, where the gate gives
every tracing command a trace path it knows — so the topology is: runs happen
in CLI subprocesses, this server only reads their files, and the Slack bot
(which never opens a port) merely posts a URL pointing here.

The server does the recomputation and the page stays dumb: each SSE `snapshot`
frame carries the rendered Mermaid source and summary numbers, never raw trace
events. That is a security decision as much as a simplicity one — `state_delta`
can hold anything a run's nodes wrote, and its *contents* are deliberately not
served. The exposure is the same as the `viz`/`metrics` commands': node names,
error labels, counts, and two state fields shown by convention —
`termination_reason` (a short reason string) and `goal` (the operator's own
question, from the loop's labelled topology/approval events; text the operator
typed, never something a node wrote).

Reachability is the operator's problem by design: bind stays loopback unless
they choose otherwise, and the recommended remote path is a tunnel or tailnet
in front, optionally with the shared-secret `token` (a convenience lock, not a
perimeter). The token travels in a header, or in the cookie `POST /live/auth`
sets for a browser; `?token=` is accepted on `/live/api/stream` alone, because
`EventSource` cannot set headers and that route has no other way in. Everywhere
else a query-string token is refused outright, with its own reason: a URL is
copied into access logs, browser history and referrers, and a secret that
protects read access to every trace under the root should not accrue copies in
places with weaker access control than the traces.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import html
import importlib.resources
import math
import secrets
from datetime import datetime
from pathlib import Path
from time import monotonic, time
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from grapharc.observe.layout import layout_graph
from grapharc.observe.metrics import RunMetrics, summarize, to_mermaid
from grapharc.observe.replay import replay
from grapharc.observe.trace import TailRecorder, TraceEvent
from grapharc.observe.viewmodel import GraphSnapshot, build_graph_view
from grapharc.slack.format import mermaid_live_url

#: How often the stream re-stats the trace file. Coarser than the session
#: stream's 0.02s: every change here costs a full re-read and replay of the
#: file, not a list slice.
LIVE_POLL_SECONDS = 0.5
LIVE_KEEPALIVE_SECONDS = 15.0
#: Keep streaming this long after a termination reason appears, so the final
#: snapshot (and any last events) reach the page before the stream closes.
DONE_GRACE_SECONDS = 5.0
#: A file that grew within this window counts as an active run.
ACTIVE_WINDOW_SECONDS = 10.0
#: How long an open (started, never ended) node keeps a quiet run reading as
#: "running". A delegated executor is silent mid-flight, so quiet ≠ idle — but
#: past this, an open node is a run that died mid-node, not one still working.
OPEN_NODE_GRACE_SECONDS = 900.0
#: The cookie `POST /live/auth` sets, so a browser can authenticate every later
#: navigation without the token ever entering a URL.
LIVE_COOKIE = "grapharc_live_token"
#: Why a token in the query string is refused off the SSE route. Distinct from
#: "missing or wrong token" on purpose: the caller holds the right secret and
#: needs to be told *where* to put it, not that it is wrong.
QUERY_TOKEN_REFUSED = (
    "the token may not travel in the query string on this route — it would be "
    "copied into access logs, browser history and referrers. Send it as an "
    "`Authorization: Bearer` header, or sign in once to set a cookie."
)
#: A replay never runs longer than this however slow the recording was, so a
#: 40-minute run is watchable. A `speed=` multiplier faster than the cap wins.
REPLAY_MAX_SECONDS = 40.0
#: An upper bound on frames per replay: each one re-renders the whole prefix,
#: so a 50k-event trace would otherwise cost quadratic work to watch.
REPLAY_MAX_FRAMES = 400
#: Clamp on `speed=`; past this the replay is instant anyway.
REPLAY_MAX_SPEED = 10_000.0


#: The static assets the router serves, and as what. An allowlist, not a
#: directory walk: any other name under /live/static is a 404, so the route
#: can never be talked into serving a neighbouring file.
STATIC_ASSETS = {
    "view.css": "text/css; charset=utf-8",
    "view.js": "text/javascript; charset=utf-8",
}


@functools.cache
def _static(name: str) -> str:
    """One page or asset from the packaged static directory.

    `importlib.resources`, not a path relative to `__file__`: the files must
    load from an installed wheel, not only from a checkout.
    """
    return (
        importlib.resources.files("grapharc.server")
        .joinpath("static", name)
        .read_text(encoding="utf-8")
    )


class LivePathError(Exception):
    """The requested trace is not one this server will read."""


def resolve_trace(root: Path, raw: str) -> Path:
    """Confine a requested trace path inside the root, gate-style.

    Lexical resolution only — the target need not exist (the run may not have
    started yet). Absolute paths, traversal out of the root (symlinks
    included, via resolve()), and non-`.jsonl` names are all refused.
    """
    if not raw or Path(raw).is_absolute():
        raise LivePathError(f"not a relative trace path: {raw!r}")
    if not raw.endswith(".jsonl"):
        raise LivePathError(f"not a .jsonl trace: {raw!r}")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise LivePathError(f"path escapes the live root: {raw!r}")
    return resolved


class PlanningRound(BaseModel):
    """One governed-loop round, as far as the trace has got with it.

    Counts and labels only — the same exposure the rest of the live view has.
    """

    round: int
    status: str = ""
    nodes: int = 0
    proposals: int = 0
    tokens: int = 0
    failed_checks: list[str] = []
    rejections: list[str] = []
    executed: bool = False
    #: The round has `plan`/`admission` events but no closing `round` event:
    #: the planner is thinking right now, which is exactly the window the page
    #: used to render as "waiting for the run to start…".
    in_flight: bool = False
    error: str | None = None


class PlanningView(BaseModel):
    """What the planner did before (or instead of) producing a graph."""

    rounds: list[PlanningRound] = []
    planner_tokens: int = 0
    stop: str | None = None
    stop_detail: str | None = None
    #: A graph was admitted and materialised, so the diagram is the main event
    #: and this is history. Without it, this *is* the run so far.
    has_topology: bool = False


class LiveSnapshot(BaseModel):
    """Everything one page render needs; recomputed server-side per change."""

    trace: str
    run_id: str | None = None
    run_ids: list[str] = []
    mermaid: str = ""
    mermaid_live_url: str | None = None
    stats: RunMetrics | None = None
    cost_usd: float | None = None
    wall_ms: float | None = None
    last_event_ts: str | None = None
    size: int = 0
    active: bool = False
    done: bool = False
    awaiting_approval: bool = False
    #: None for a run that never planned — an ordinary graph invocation renders
    #: exactly as it did before this field existed.
    planning: PlanningView | None = None
    #: The structured, positioned graph the SVG view draws — same exposure as
    #: `mermaid` (names, statuses, counts, spend, error labels), never deltas.
    graph: GraphSnapshot | None = None
    #: The operator's own question, lifted from the loop's topology/
    #: approval_request deltas the way `termination_reason` is lifted by
    #: `summarize` — the second state field shown by convention, and the only
    #: other one: it is text the operator typed, never something a node wrote.
    goal: str | None = None
    #: The command that answers a parked run, shown only while one is parked.
    #: Reveals the trace's directory to authenticated viewers — accepted: it is
    #: the one string correct from any shell on this machine, the view is
    #: loopback/token/tunnel-gated, and it vanishes once the run is answered.
    approve_command: str | None = None


def _as_int(value: Any, fallback: int = 0) -> int:
    """A count out of a trace file written by someone else. Never raises."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def summarize_planning(run_events: list[TraceEvent]) -> PlanningView | None:
    """Fold `plan`/`admission`/`round` (and the loop's `stop`) into a panel.

    No new trace events: everything here is already on disk while the page is
    showing nothing. A round is opened by the first `plan` or `admission` event
    that follows the last closed one, so an in-flight round — the 30-45 seconds
    of local inference this exists for — is a row too.

    Returns None when the run has no planning at all, which keeps the field
    absent for every non-planner run.
    """
    rounds: list[PlanningRound] = []
    view = PlanningView()
    pending: PlanningRound | None = None

    def slot() -> PlanningRound:
        nonlocal pending
        if pending is None:
            pending = PlanningRound(round=len(rounds) + 1, in_flight=True)
        return pending

    for event in run_events:
        delta = event.state_delta or {}
        if event.phase == "topology":
            view.has_topology = True
        elif event.phase == "plan":
            entry = slot()
            entry.proposals += 1
            entry.nodes = _as_int(delta.get("nodes"), entry.nodes)
            entry.tokens += event.tokens or 0
            entry.error = event.error or entry.error
        elif event.phase == "admission":
            entry = slot()
            entry.status = str(delta.get("status") or entry.status)
            entry.nodes = _as_int(delta.get("nodes"), entry.nodes)
            entry.failed_checks = [str(c) for c in delta.get("failed_checks") or []]
        elif event.phase == "round":
            entry = slot()
            entry.in_flight = False
            entry.round = _as_int(delta.get("round"), entry.round)
            entry.status = str(delta.get("status") or entry.status)
            entry.nodes = _as_int(delta.get("nodes"), entry.nodes)
            entry.rejections = [str(r) for r in delta.get("rejections") or []]
            entry.executed = bool(delta.get("executed"))
            entry.error = event.error or entry.error
            rounds.append(entry)
            pending = None
        elif event.phase == "stop" and "stop" in delta:
            # The governed loop's own terminal event — an agent's `stop` carries
            # a `termination_reason` instead, and is not a planning outcome.
            view.stop = str(delta.get("stop") or "") or None
            view.stop_detail = str(delta.get("detail") or "") or None
    if pending is not None:
        rounds.append(pending)
    if not rounds and view.stop is None:
        return None
    view.rounds = rounds
    view.planner_tokens = sum(r.tokens for r in rounds)
    return view


class _FrozenRecorder(TailRecorder):
    """A reader over a fixed slice of events rather than the file's current one.

    `to_mermaid`, `summarize` and `replay` all take a recorder and call
    `read_events`, so a replay frame is the ordinary snapshot computation
    pointed at a prefix of the trace.
    """

    def __init__(self, path: str | Path, events: list[TraceEvent]) -> None:
        super().__init__(path)
        self._events = list(events)

    def read_events(self, run_id: str | None = None) -> list[TraceEvent]:
        return [e for e in self._events if run_id is None or e.run_id == run_id]


def build_snapshot(root: Path, rel: str, run_id: str | None) -> LiveSnapshot:
    """One consistent read of the trace file. Pure; run it in a thread.

    A missing or empty file is a waiting snapshot, not an error — the URL is
    posted before the run starts writing.
    """
    path = resolve_trace(root, rel)
    recorder = TailRecorder(path)
    events = recorder.read_events()
    if not events:
        return LiveSnapshot(trace=rel)
    try:
        stat = path.stat()
        size, quiet_for = stat.st_size, time() - stat.st_mtime
    except OSError:
        size, quiet_for = 0, float("inf")
    return compose_snapshot(
        rel,
        recorder,
        events,
        run_id,
        size=size,
        quiet_for=quiet_for,
        trace_dir=path.parent,
    )


def compose_snapshot(
    rel: str,
    recorder: TailRecorder,
    events: list[TraceEvent],
    run_id: str | None,
    *,
    size: int,
    quiet_for: float,
    trace_dir: Path | None = None,
) -> LiveSnapshot:
    """The snapshot for one already-read set of events. Pure; run it in a thread.

    Split out of `build_snapshot` so a replay frame — a prefix of a finished
    file, with no file mtime to consult — is computed by the same code that
    computes a live one.
    """
    if not events:
        return LiveSnapshot(trace=rel)
    run_ids = list(dict.fromkeys(e.run_id for e in events))
    chosen = run_id if run_id in run_ids else run_ids[-1]
    run_events = [e for e in events if e.run_id == chosen]
    active = quiet_for < ACTIVE_WINDOW_SECONDS

    mermaid = to_mermaid(recorder, chosen)
    run = replay(recorder, chosen)
    graph = layout_graph(build_graph_view(run))
    # Finished means: something wrote a termination reason, OR a driver wrote
    # its terminal `stop` event — the planner writes the latter and never the
    # former, and keying on `termination_reason` alone held the SSE stream
    # open forever for every planner run.
    done = any(
        e.phase == "stop" or "termination_reason" in (e.state_delta or {})
        for e in run_events
    )
    # A node that started and never ended means someone is inside it right now
    # — a delegated executor writes nothing between its start and its finish,
    # so "the file went quiet" must not read as idle while a node is open.
    # Bounded, though: a run killed mid-node leaves its node open forever, and
    # an hour-quiet open node is a corpse, not work.
    if not done and quiet_for < OPEN_NODE_GRACE_SECONDS and any(
        not e.completed for e in run.executions
    ):
        active = True
    # A planning round that has begun and not closed is the same shape one step
    # earlier: the planner writes nothing between the request and the model's
    # reply, and 30-45 seconds of local inference is longer than the activity
    # window. Bounded by the same grace, for the same reason.
    planning = summarize_planning(run_events)
    if (
        not done
        and quiet_for < OPEN_NODE_GRACE_SECONDS
        and planning is not None
        and any(r.in_flight for r in planning.rounds)
    ):
        active = True
    # An approval request with no later response: the run is deliberately
    # parked, and the page should say "awaiting approval", not "idle".
    awaiting = False
    for event in run_events:
        if event.phase == "approval_request":
            awaiting = True
        elif event.phase == "approval_response":
            awaiting = False
    if awaiting and not done:
        active = True
    # The goal, from the loop's labelled events only. Last one wins — a
    # multi-round run restates it, and the current round's is the answer.
    goal = None
    for event in run_events:
        if event.phase in ("topology", "approval_request"):
            value = (event.state_delta or {}).get("goal")
            if value:
                goal = str(value)
    approve_command = (
        f"grapharc approve {trace_dir}"
        if awaiting and not done and trace_dir is not None
        else None
    )
    return LiveSnapshot(
        trace=rel,
        run_id=chosen,
        run_ids=run_ids,
        mermaid=mermaid,
        mermaid_live_url=mermaid_live_url(mermaid),
        stats=summarize(recorder, chosen),
        cost_usd=run.recorded_cost_usd,
        wall_ms=run.wall_ms,
        last_event_ts=run_events[-1].ts if run_events else None,
        size=size,
        active=active,
        done=done,
        awaiting_approval=awaiting,
        planning=planning,
        graph=graph,
        goal=goal,
        approve_command=approve_command,
    )


def _elapsed_seconds(events: list[TraceEvent]) -> list[float]:
    """Seconds since the first event, per event, from the recorded timestamps.

    Monotonic by construction: an unparseable or out-of-order `ts` — these files
    can be foreign or hand-written — holds the previous offset rather than
    sending the replay backwards or raising.
    """
    offsets: list[float] = []
    first: datetime | None = None
    latest = 0.0
    for event in events:
        try:
            stamp = datetime.fromisoformat(event.ts)
        except (TypeError, ValueError):
            offsets.append(latest)
            continue
        if first is None:
            first = stamp
        try:
            latest = max(latest, (stamp - first).total_seconds())
        except TypeError:  # one naive stamp among aware ones
            pass
        offsets.append(latest)
    return offsets


def replay_schedule(
    events: list[TraceEvent],
    speed: float = 1.0,
    *,
    max_seconds: float = REPLAY_MAX_SECONDS,
    max_frames: int = REPLAY_MAX_FRAMES,
) -> list[tuple[int, float]]:
    """`(events to include, seconds into the replay)` per frame, in order.

    Recorded speed divided by `speed`, then capped: whatever the multiplier, the
    whole replay fits in `max_seconds`, so yesterday's 40-minute incident trace
    is watchable. A nonsense multiplier (zero, negative, NaN) reads as 1.0
    rather than dividing the schedule by it.
    """
    if not events:
        return []
    if not speed > 0 or math.isnan(speed):
        speed = 1.0
    speed = min(speed, REPLAY_MAX_SPEED)
    offsets = _elapsed_seconds(events)
    span = offsets[-1]
    if max_seconds > 0 and span / speed > max_seconds:
        speed = span / max_seconds
    stride = max(1, math.ceil(len(events) / max(1, max_frames)))
    indices = sorted({*range(0, len(events), stride), len(events) - 1})
    return [(index + 1, offsets[index] / speed) for index in indices]


#: How many of the newest traces the index fully parses for run ids. The rest
#: are listed by name and size only: the live root accumulates one directory
#: per run forever, and re-validating every byte of the whole corpus per
#: index refresh made history the server's dominant cost.
SCAN_PARSE_LIMIT = 25


def scan_traces(root: Path) -> list[dict[str, Any]]:
    """Every trace file under the root, newest first.

    Run ids are parsed only for the `SCAN_PARSE_LIMIT` newest files; older
    rows carry an empty `runs` list — their viewer pages still work (the run
    is resolved from the file when the page opens).

    Confined exactly as `resolve_trace` confines the reader: `rglob` matches a
    symlinked file by name, so without this the index advertised names, sizes
    and parsed run ids for files the stream then 404s. Two checks for one
    contract — the shared `resolve_trace` call, and an outright skip of
    symlinks — so a refactor of either cannot quietly reopen the leak.
    """
    found = []
    for path in root.rglob("*.jsonl"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
            resolve_trace(root, rel)
        except (ValueError, LivePathError):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append(
            {
                "trace": rel,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "runs": [],
                "_path": path,
            }
        )
    found.sort(key=lambda item: item["mtime"], reverse=True)
    for item in found[:SCAN_PARSE_LIMIT]:
        try:
            item["runs"] = list(
                dict.fromkeys(
                    e.run_id for e in TailRecorder(item["_path"]).read_events()
                )
            )
        except OSError:
            pass
    for item in found:
        del item["_path"]
    return found


def _safe_next(raw: str) -> str:
    """Where sign-in may send the caller: a path inside `/live`, or nothing.

    The form carries its destination, and a form field is attacker-supplied by
    definition — an absolute URL, a scheme-relative `//host` or a backslash the
    browser normalises would turn the sign-in page into an open redirect.
    """
    if not raw.startswith("/live") or raw.startswith("//"):
        return "/live"
    if any(bad in raw for bad in ("\\", "\n", "\r", "\t")):
        return "/live"
    return raw


def live_router(
    root: str | Path,
    *,
    token: str | None = None,
    poll_seconds: float = LIVE_POLL_SECONDS,
    keepalive_seconds: float | None = LIVE_KEEPALIVE_SECONDS,
) -> APIRouter:
    """GET-only routes under /live, all confined to `root`."""
    root_path = Path(root)
    router = APIRouter(prefix="/live")

    # What the sign-in cookie carries: a digest of the token, never the token.
    # Always ASCII, so a non-ASCII secret survives a header round trip (cookies
    # are latin-1 on the wire), and a stolen cookie is not the secret itself.
    cookie_value = (
        hashlib.sha256(token.encode("utf-8")).hexdigest() if token is not None else ""
    )

    def _matches(supplied: str) -> bool:
        # Compared as bytes: `compare_digest` refuses `str` outside ASCII, so
        # comparing text turned a one-character guess into a 500 — the gate
        # crashing on the strangers it exists to refuse. Encoding keeps the
        # constant-time property, which is the reason it is here at all.
        return token is not None and secrets.compare_digest(
            supplied.encode("utf-8"), token.encode("utf-8")
        )

    def _refusal(request: Request, *, allow_query: bool = False) -> str | None:
        """None when the request may proceed, otherwise why it may not.

        `allow_query` is the SSE route's exemption and nothing else's: a page
        opened by hand, an index, a JSON listing all have a header or the
        sign-in cookie available, and a token they put in the URL is refused
        with its own reason rather than quietly accepted into the logs.
        """
        if token is None:
            return None
        header = request.headers.get("authorization", "")
        candidates = []
        if header.startswith("Bearer "):
            candidates.append(header.removeprefix("Bearer "))
        if allow_query:
            query = request.query_params.get("token")
            if query is not None:
                candidates.append(query)
        cookie = request.cookies.get(LIVE_COOKIE) or ""
        # Every credential presented is tried, not the first one found: a stale
        # cookie must not lock out a request that also carries a good header.
        if any(_matches(candidate) for candidate in candidates) or secrets.compare_digest(
            cookie.encode("utf-8"), cookie_value.encode("utf-8")
        ):
            return None
        if not allow_query and "token" in request.query_params:
            return QUERY_TOKEN_REFUSED
        return "missing or wrong token"

    def _return_to(request: Request) -> str:
        """Where to send the caller after signing in — minus any `?token=`.

        Echoing the refused parameter back into the form would put the secret
        into the next URL, which is the whole thing being fixed.
        """
        kept = [
            (key, value)
            for key, value in request.query_params.multi_items()
            if key != "token"
        ]
        query = urlencode(kept)
        return _safe_next(request.url.path + (f"?{query}" if query else ""))

    def _sign_in(target: str, reason: str) -> HTMLResponse:
        page = (
            _static("signin.html")
            .replace("__REASON__", html.escape(reason))
            .replace("__NEXT__", html.escape(target, quote=True))
        )
        return HTMLResponse(page, status_code=401)

    def _resolved(raw: str) -> str:
        """Validate confinement; 404 on refusal (don't map what exists outside).

        `ValueError` too: a NUL byte in the name reaches the filesystem call
        inside `resolve()`, and a malformed request is a 404 like any other.
        """
        try:
            resolve_trace(root_path, raw)
        except (LivePathError, ValueError):
            raise HTTPException(status_code=404, detail="no such trace") from None
        return raw

    @router.post("/auth", include_in_schema=False)
    async def auth(request: Request) -> Response:
        """Trade the token for a cookie, so no navigation carries it in a URL.

        A form post, not a GET: a GET would put the secret in the query string
        of the very request that exists to keep it out of query strings.
        """
        form = await request.form()
        target = _safe_next(str(form.get("next") or "/live"))
        if token is not None and not _matches(str(form.get("token") or "")):
            return _sign_in(target, "missing or wrong token")
        response = RedirectResponse(target, status_code=303)
        if token is not None:
            response.set_cookie(
                LIVE_COOKIE,
                cookie_value,
                httponly=True,
                samesite="strict",
                path="/live",
                secure=request.url.scheme == "https",
            )
        return response

    @router.get("", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        refusal = _refusal(request)
        if refusal:
            return _sign_in(_return_to(request), refusal)
        rows = []
        for item in scan_traces(root_path):
            # Trace names and run ids come from whoever wrote the files — the
            # Slack gate checks path *containment*, not characters — so every
            # interpolation here is escaped: unescaped, a filename like
            # `x"><script>….jsonl` executes in the operator's browser (and the
            # token sits in this same DOM).
            runs = html.escape(", ".join(item["runs"]) or "—")
            shown = html.escape(item["trace"])
            # No token in the link: the cookie authenticates the page the
            # operator clicks through to, and a link is the thing that ends up
            # in history and in the referrer of anything the page opens.
            href = html.escape(f"/live/view?trace={quote(item['trace'], safe='')}")
            rows.append(
                f'<tr><td><a href="{href}">{shown}</a></td>'
                f"<td>{runs}</td><td>{item['size']}</td></tr>"
            )
        body = "\n".join(rows) or '<tr><td colspan="3">no trace files yet</td></tr>'
        return HTMLResponse(_static("index.html").replace("__ROWS__", body))

    @router.get("/api/runs")
    def runs(request: Request) -> dict[str, Any]:
        refusal = _refusal(request)
        if refusal:
            raise HTTPException(status_code=401, detail=refusal)
        return {"root": str(root_path), "traces": scan_traces(root_path)}

    @router.get("/view", response_class=HTMLResponse, include_in_schema=False)
    def view(request: Request, trace: str = Query(...)) -> HTMLResponse:
        refusal = _refusal(request)
        if refusal:
            return _sign_in(_return_to(request), refusal)
        _resolved(trace)
        return HTMLResponse(_static("view.html"))

    @router.get("/static/{name}", include_in_schema=False)
    def static_asset(name: str) -> Response:
        """CSS/JS for the pages above. Deliberately outside the token gate:
        these are the same bytes anyone can read in the published wheel, and
        the sign-in page itself needs its stylesheet before any credential
        exists. The allowlist is the whole route — no path from here reaches
        the filesystem or the trace root.
        """
        content_type = STATIC_ASSETS.get(name)
        if content_type is None:
            raise HTTPException(status_code=404, detail="no such asset")
        return Response(_static(name), media_type=content_type)

    @router.get("/api/stream")
    async def stream(
        request: Request,
        trace: str = Query(...),
        run: str | None = Query(None),
        replay_recorded: int = Query(0, alias="replay"),
        speed: float = Query(1.0),
    ) -> StreamingResponse:
        # The one route a token may reach by query string: `EventSource` cannot
        # set a header, and the page has nothing else to authenticate with.
        refusal = _refusal(request, allow_query=True)
        if refusal:
            raise HTTPException(status_code=401, detail=refusal)
        rel = _resolved(trace)
        path = resolve_trace(root_path, rel)

        async def replay_frames():
            """The recorded run, re-emitted as the snapshots it would have sent.

            A finished trace renders instantly all-green, so the amber "running"
            frame — the thing the runtime is best at showing — is unreachable
            for every run that is already over. This walks the file's own
            timestamps and rebuilds each intermediate snapshot from a prefix,
            which is the same computation a live stream does, only sourced from
            the past.
            """
            from grapharc.server.app import _disconnected, _sse

            try:
                events = await asyncio.to_thread(TailRecorder(path).read_events)
            except Exception:
                events = []
            if not events:
                yield _sse("snapshot", LiveSnapshot(trace=rel).model_dump(mode="json"))
                yield _sse("done", {"trace": rel, "replay": True})
                return
            run_ids = list(dict.fromkeys(e.run_id for e in events))
            chosen = run if run in run_ids else run_ids[-1]
            run_events = [e for e in events if e.run_id == chosen]
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            started = monotonic()
            for count, offset in replay_schedule(run_events, speed):
                wait = offset - (monotonic() - started)
                if wait > 0:
                    await asyncio.sleep(wait)
                if await _disconnected(request):
                    return
                prefix = run_events[:count]
                try:
                    snapshot = await asyncio.to_thread(
                        compose_snapshot,
                        rel,
                        _FrozenRecorder(path, prefix),
                        prefix,
                        chosen,
                        # A replay has no live file to stat: it is running by
                        # definition until its last frame, and every frame must
                        # be reproducible, so neither depends on the clock.
                        size=size,
                        quiet_for=0.0,
                    )
                except Exception:
                    yield ": snapshot unavailable\n\n"
                    continue
                yield _sse("snapshot", snapshot.model_dump(mode="json"))
            yield _sse("done", {"trace": rel, "replay": True})

        if replay_recorded:
            return StreamingResponse(
                replay_frames(),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )

        async def frames():
            from grapharc.server.app import _disconnected, _sse

            last_size = -1
            done_since: float | None = None
            idle = 0.0
            while True:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                if size != last_size:
                    try:
                        snapshot = await asyncio.to_thread(
                            build_snapshot, root_path, rel, run
                        )
                    except Exception:
                        # A trace this server merely *reads* can be foreign,
                        # torn, or hand-written; one bad file must not tear
                        # the stream down. Mark the size consumed so a
                        # permanently-broken file does not hot-loop.
                        last_size = size
                        yield ": snapshot unavailable\n\n"
                        idle = 0.0
                        await asyncio.sleep(poll_seconds)
                        continue
                    last_size = snapshot.size if snapshot.size else size
                    idle = 0.0
                    yield _sse("snapshot", snapshot.model_dump(mode="json"))
                    if not snapshot.done:
                        done_since = None
                    elif done_since is None:
                        done_since = monotonic()
                if done_since is not None and monotonic() - done_since >= DONE_GRACE_SECONDS:
                    yield _sse("done", {"trace": rel})
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

    return router


# The pages live in `static/` as real files, packaged with the wheel and
# loaded through `_static`. They keep sentinel replacement (`__ROWS__`,
# `__REASON__`, `__NEXT__`), not str.format — JS/CSS braces would fight the
# format machinery. Fully self-contained: no CDN, no external request; the
# mermaid.live link every snapshot carries remains the degraded path.

__all__ = [
    "ACTIVE_WINDOW_SECONDS",
    "DONE_GRACE_SECONDS",
    "LIVE_COOKIE",
    "LIVE_KEEPALIVE_SECONDS",
    "LIVE_POLL_SECONDS",
    "QUERY_TOKEN_REFUSED",
    "REPLAY_MAX_FRAMES",
    "REPLAY_MAX_SECONDS",
    "REPLAY_MAX_SPEED",
    "LivePathError",
    "LiveSnapshot",
    "PlanningRound",
    "PlanningView",
    "TailRecorder",
    "build_snapshot",
    "compose_snapshot",
    "live_router",
    "replay_schedule",
    "resolve_trace",
    "scan_traces",
    "summarize_planning",
]
