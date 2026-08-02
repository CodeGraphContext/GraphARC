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
error labels, counts, and the one state field `summarize` lifts out of the
delta — `termination_reason`, a short reason string by convention.

Reachability is the operator's problem by design: bind stays loopback unless
they choose otherwise, and the recommended remote path is a tunnel or tailnet
in front, optionally with the shared-secret `token` (a convenience lock, not a
perimeter — it rides in the query string because `EventSource` cannot set
headers).
"""

from __future__ import annotations

import asyncio
import html
import secrets
from pathlib import Path
from time import monotonic, time
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from grapharc.observe.metrics import RunMetrics, summarize, to_mermaid
from grapharc.observe.replay import replay
from grapharc.observe.trace import TailRecorder
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

    run_ids = list(dict.fromkeys(e.run_id for e in events))
    chosen = run_id if run_id in run_ids else run_ids[-1]
    run_events = [e for e in events if e.run_id == chosen]

    try:
        size = path.stat().st_size
        quiet_for = time() - path.stat().st_mtime
    except OSError:
        size, quiet_for = 0, float("inf")
    active = quiet_for < ACTIVE_WINDOW_SECONDS

    mermaid = to_mermaid(recorder, chosen)
    run = replay(recorder, chosen)
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
    )


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
    """
    found = []
    for path in root.rglob("*.jsonl"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append(
            {
                "trace": path.relative_to(root).as_posix(),
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

    def _authorized(request: Request) -> None:
        if token is None:
            return
        supplied = request.query_params.get("token")
        header = request.headers.get("authorization", "")
        if header.startswith("Bearer "):
            supplied = supplied or header.removeprefix("Bearer ")
        if supplied is None or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="missing or wrong token")

    def _resolved(raw: str) -> str:
        """Validate confinement; 404 on refusal (don't map what exists outside)."""
        try:
            resolve_trace(root_path, raw)
        except LivePathError:
            raise HTTPException(status_code=404, detail="no such trace") from None
        return raw

    @router.get("", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        _authorized(request)
        keep = f"&token={quote(token, safe='')}" if token else ""
        rows = []
        for item in scan_traces(root_path):
            # Trace names and run ids come from whoever wrote the files — the
            # Slack gate checks path *containment*, not characters — so every
            # interpolation here is escaped: unescaped, a filename like
            # `x"><script>….jsonl` executes in the operator's browser (and the
            # token sits in this same DOM).
            runs = html.escape(", ".join(item["runs"]) or "—")
            shown = html.escape(item["trace"])
            href = html.escape(
                f"/live/view?trace={quote(item['trace'], safe='')}{keep}"
            )
            rows.append(
                f'<tr><td><a href="{href}">{shown}</a></td>'
                f"<td>{runs}</td><td>{item['size']}</td></tr>"
            )
        body = "\n".join(rows) or '<tr><td colspan="3">no trace files yet</td></tr>'
        return HTMLResponse(INDEX_HTML.replace("__ROWS__", body))

    @router.get("/api/runs")
    def runs(request: Request) -> dict[str, Any]:
        _authorized(request)
        return {"root": str(root_path), "traces": scan_traces(root_path)}

    @router.get("/view", response_class=HTMLResponse, include_in_schema=False)
    def view(request: Request, trace: str = Query(...)) -> HTMLResponse:
        _authorized(request)
        _resolved(trace)
        return HTMLResponse(VIEW_HTML)

    @router.get("/api/stream")
    async def stream(
        request: Request,
        trace: str = Query(...),
        run: str | None = Query(None),
    ) -> StreamingResponse:
        _authorized(request)
        rel = _resolved(trace)
        path = resolve_trace(root_path, rel)

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


# The pages are plain strings with sentinel replacement, not str.format —
# JS/CSS braces would fight the format machinery. Self-contained except for
# the pinned mermaid CDN import; without it (offline viewer, blocked CDN) the
# page falls back to the raw Mermaid source plus the mermaid.live link that
# every snapshot carries.

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>grapharc live</title>
<meta http-equiv="refresh" content="5">
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;color:#222}
 table{border-collapse:collapse}
 td,th{padding:.3rem .8rem;border-bottom:1px solid #ddd;text-align:left}
</style></head><body>
<h1>grapharc live</h1>
<p>Trace files under the live root, newest first. This page refreshes itself.</p>
<table><tr><th>trace</th><th>runs</th><th>bytes</th></tr>
__ROWS__
</table></body></html>
"""

VIEW_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>grapharc live view</title>
<style>
 body{font-family:system-ui,sans-serif;margin:1.5rem;color:#222}
 header{display:flex;gap:1rem;align-items:baseline;flex-wrap:wrap}
 #status{font-weight:600}
 #graph{margin:1rem 0;overflow-x:auto}
 #graph svg{max-width:100%}
 pre{background:#f6f6f6;padding:1rem;overflow-x:auto}
 table{border-collapse:collapse}
 td,th{padding:.25rem .7rem;border-bottom:1px solid #ddd;text-align:left}
 .muted{color:#777}
</style></head><body>
<header>
 <span id="status">connecting…</span>
 <span id="run" class="muted"></span>
 <a id="mlive" target="_blank" rel="noopener" hidden>open in mermaid.live</a>
</header>
<div id="graph">waiting for the run to start…</div>
<pre id="src" hidden></pre>
<table id="stats"></table>
<script type="module">
let mermaid = null;
try {
  mermaid = (await import("https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs")).default;
  mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
} catch { /* no CDN: the raw source and the mermaid.live link stay visible */ }

const status = document.getElementById("status");
const runEl = document.getElementById("run");
const graph = document.getElementById("graph");
const src = document.getElementById("src");
const mlive = document.getElementById("mlive");
const stats = document.getElementById("stats");
let seq = 0;

function renderStats(s) {
  const rows = [];
  const add = (k, v) => { if (v !== null && v !== undefined && v !== "") rows.push([k, v]); };
  if (s.stats) {
    add("graph", s.stats.graph);
    add("nodes executed", s.stats.nodes_executed);
    add("errors", s.stats.errors);
    add("tokens", s.stats.tokens);
    add("events", s.stats.events);
    add("termination", s.stats.termination_reason);
  }
  add("cost (USD)", s.cost_usd);
  add("wall ms", s.wall_ms);
  add("last event", s.last_event_ts);
  stats.innerHTML = "";
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = k;
    const td = document.createElement("td"); td.textContent = String(v);
    tr.append(th, td); stats.append(tr);
  }
}

const es = new EventSource("/live/api/stream" + location.search);
es.addEventListener("snapshot", async (e) => {
  const s = JSON.parse(e.data);
  runEl.textContent = s.run_id ? "run " + s.run_id : "";
  status.textContent = s.done ? "finished"
    : (s.awaiting_approval ? "awaiting approval"
    : (s.active ? "running" : (s.run_id ? "idle" : "waiting")));
  renderStats(s);
  if (!s.mermaid) return;
  src.textContent = s.mermaid;
  if (s.mermaid_live_url) { mlive.href = s.mermaid_live_url; mlive.hidden = false; }
  if (mermaid) {
    try {
      const { svg } = await mermaid.render("live" + (seq++), s.mermaid);
      graph.innerHTML = svg;
      src.hidden = true;
    } catch { graph.textContent = ""; src.hidden = false; }
  } else {
    graph.textContent = "";
    src.hidden = false;
  }
});
es.addEventListener("done", () => { es.close(); status.textContent = "finished"; });
es.onerror = () => {
  if (es.readyState === EventSource.CONNECTING) status.textContent = "reconnecting…";
};
</script></body></html>
"""

__all__ = [
    "ACTIVE_WINDOW_SECONDS",
    "DONE_GRACE_SECONDS",
    "LIVE_KEEPALIVE_SECONDS",
    "LIVE_POLL_SECONDS",
    "LivePathError",
    "LiveSnapshot",
    "TailRecorder",
    "build_snapshot",
    "live_router",
    "resolve_trace",
    "scan_traces",
]
