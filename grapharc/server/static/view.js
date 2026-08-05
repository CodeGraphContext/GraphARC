/* GraphARC live view renderer.
 *
 * Draws the positioned graph the server ships in each SSE snapshot. The page
 * stays dumb on purpose: no layout, no counting, no trace parsing — statuses,
 * spend and geometry all arrive computed, and everything written into the DOM
 * goes through textContent (names and error labels come out of a trace file
 * this server merely reads).
 *
 * Flicker-free by construction: the SVG is built once per topology (keyed on
 * node ids + edges + canvas size) and later snapshots only patch classes and
 * text in place. A new planner round changes the key and rebuilds, which is
 * the one moment things may move.
 */

"use strict";

const SVG = "http://www.w3.org/2000/svg";

const el = (id) => document.getElementById(id);
const dot = el("dot"), statusEl = el("status"), runEl = el("run");
const goalEl = el("goal");
const runPick = el("runpick"), badge = el("badge"), panel = el("panel");
const planningEl = el("planning"), drawer = el("drawer");
const srcEl = el("src"), mlive = el("mlive");
const playback = el("playback"), playBtn = el("play");
const speedPick = el("speed"), scrub = el("scrub"), clock = el("clock");
const approvalBar = el("approvalbar"), approveCmd = el("approvecmd");
const copyApprove = el("copyapprove");
const inspector = el("inspector"), iName = el("i-name"), iStatus = el("i-status");
const iProps = el("i-props"), iError = el("i-error");
const iSpans = el("i-spans"), iEdges = el("i-edges"), iClose = el("i-close");

// The node the inspector is pinned to, or null. Survives snapshot patches
// (the panel re-renders from each one) and topology rebuilds when the id
// still exists; cleared when its node leaves the graph.
let selectedId = null;
let lastStats = null;

// Whether the current snapshot is parked at an approval gate. Proposed nodes
// (status pending while parked) render in the approval colour, not plain grey.
let awaiting = false;

const params = new URLSearchParams(location.search);
if (params.get("replay")) badge.hidden = false;

/* ---------------------------------------------------------------- helpers */

function fmtTokens(n) {
  if (n >= 100_000) return Math.round(n / 1000) + "k";
  if (n >= 10_000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

function fmtMs(ms) {
  if (ms == null) return "";
  if (ms < 1000) return Math.round(ms) + "ms";
  if (ms < 90_000) return (ms / 1000).toFixed(1) + "s";
  return Math.round(ms / 60_000) + "m" + Math.round((ms % 60_000) / 1000) + "s";
}

function fmtCost(c) {
  if (c == null) return null;
  return "$" + (c < 0.01 ? c.toFixed(4) : c.toFixed(2));
}

function make(tag, className, parent) {
  const node = document.createElementNS(SVG, tag);
  if (className) node.setAttribute("class", className);
  if (parent) parent.appendChild(node);
  return node;
}

/* ------------------------------------------------------------- stat tiles */

function tile(id, value) {
  const box = el(id);
  if (value === null || value === undefined || value === "") {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.querySelector("b").textContent = String(value);
}

function renderTiles(s) {
  const g = s.graph;
  let nodesDone = null;
  if (g && g.kind === "topology") {
    const drawn = g.nodes.filter((n) => n.role === "node");
    nodesDone = drawn.filter((n) => n.status === "done").length + "/" + drawn.length;
  } else if (s.stats) {
    nodesDone = String(s.stats.nodes_executed);
  }
  tile("t-nodes", nodesDone);
  tile("t-tokens", s.stats && s.stats.tokens ? fmtTokens(s.stats.tokens) : null);
  tile("t-cost", fmtCost(s.cost_usd));
  tile("t-wall", s.wall_ms != null ? fmtMs(s.wall_ms) : null);
  tile("t-errors", s.stats && s.stats.errors ? s.stats.errors : null);
  tile("t-events", s.stats ? s.stats.events : null);
}

/* ------------------------------------------------------------- the graph */

const GLYPH = { pending: "○", running: "▸", done: "✓", errored: "✗", proposed: "◇" };
let topologyKey = null;
let nodeGroups = new Map(); // node id -> {group, rect, name, meta, title}
let edgePaths = new Map(); // "src->dst#i" -> {path, edge}
let lastGraph = null;

function graphKey(g) {
  return [
    g.width, g.height,
    g.nodes.map((n) => n.id).join(","),
    g.edges.map((e) => e.source + ">" + e.target + ":" + e.kind).join(","),
  ].join("|");
}

function nodeMeta(node) {
  if (node.status === "running") {
    const live = node.live_tokens ? fmtTokens(node.live_tokens) + " tok · " : "";
    return live + "running…";
  }
  if (node.status === "errored") {
    return node.error ? node.error.slice(0, 34) : "failed";
  }
  if (node.status === "done") {
    const parts = [];
    if (node.tokens) parts.push(fmtTokens(node.tokens) + " tok");
    const cost = fmtCost(node.cost_usd);
    if (cost) parts.push(cost);
    if (node.duration_ms != null) parts.push(fmtMs(node.duration_ms));
    return parts.join(" · ") || "done";
  }
  return "pending";
}

function patchNode(node, statusOverride) {
  const entry = nodeGroups.get(node.id);
  if (!entry) return;
  const status = statusOverride || node.status;
  // A pending node in a parked run is "proposed": the plan exists and a human
  // has not said yes. Replay scrubbing (statusOverride) is exempt — a finished
  // run is no longer awaiting anything.
  const shown =
    awaiting && status === "pending" && !statusOverride ? "proposed" : status;
  entry.group.setAttribute(
    "class",
    "node st-" + shown + (node.id === selectedId ? " sel" : "")
  );
  entry.name.textContent = GLYPH[shown] + " " + entry.shortLabel;
  entry.meta.textContent =
    shown === "proposed"
      ? "awaiting approval"
      : status === node.status
        ? nodeMeta(node)
        : nodeMeta({ ...node, status, live_tokens: 0 });
  entry.title.textContent =
    node.label + (node.error ? "\n" + node.error : "");
}

function patchEdges(g, statusAt) {
  const running = new Set(
    g.nodes.filter((n) => (statusAt ? statusAt(n.id) : n.status) === "running")
      .map((n) => n.id)
  );
  for (const { path, edge } of edgePaths.values()) {
    const flow = running.has(edge.target) && edge.kind !== "state";
    path.setAttribute(
      "class",
      "edge " + edge.kind + (flow ? " flow" : "")
    );
  }
}

function buildGraph(g) {
  nodeGroups = new Map();
  edgePaths = new Map();
  panel.textContent = "";
  const svg = make("svg", null, panel);
  svg.setAttribute("viewBox", `0 0 ${g.width} ${g.height}`);
  svg.setAttribute("width", g.width);
  svg.style.maxWidth = "100%";
  svg.style.height = "auto";

  const defs = make("defs", null, svg);
  for (const [id, color] of [["arrow", "#3d4250"], ["arrow-flow", "#FBBF24"]]) {
    const marker = make("marker", null, defs);
    marker.setAttribute("id", id);
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "5.5");
    marker.setAttribute("markerHeight", "5.5");
    marker.setAttribute("orient", "auto-start-reverse");
    const tip = make("path", null, marker);
    tip.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    tip.setAttribute("fill", color);
  }

  for (const cluster of g.clusters) {
    const group = make("g", "cluster", svg);
    const rect = make("rect", null, group);
    rect.setAttribute("x", cluster.x);
    rect.setAttribute("y", cluster.y);
    rect.setAttribute("width", cluster.w);
    rect.setAttribute("height", cluster.h);
    const label = make("text", null, group);
    label.setAttribute("x", cluster.x + 10);
    label.setAttribute("y", cluster.y - 6);
    label.textContent = cluster.label;
  }

  g.edges.forEach((edge, index) => {
    if (!edge.points || edge.points.length < 4) return;
    const [p0, p1, p2, p3] = edge.points;
    const path = make("path", "edge " + edge.kind, svg);
    path.setAttribute(
      "d",
      `M ${p0[0]} ${p0[1]} C ${p1[0]} ${p1[1]}, ${p2[0]} ${p2[1]}, ${p3[0]} ${p3[1]}`
    );
    path.setAttribute("marker-end", "url(#arrow)");
    edgePaths.set(edge.source + ">" + edge.target + "#" + index, { path, edge });
  });

  for (const node of g.nodes) {
    if (node.role !== "node") {
      const group = make("g", "term", svg);
      const circle = make("circle", null, group);
      circle.setAttribute("cx", node.x + node.w / 2);
      circle.setAttribute("cy", node.y + node.h / 2);
      circle.setAttribute("r", node.w / 2);
      const label = make("text", null, group);
      label.setAttribute("x", node.x + node.w / 2);
      label.setAttribute("y", node.y + node.h + 11);
      label.setAttribute("text-anchor", "middle");
      label.textContent = node.label;
      continue;
    }
    const group = make("g", "node st-" + node.status, svg);
    const halo = make("rect", "halo", group);
    halo.setAttribute("x", node.x - 3);
    halo.setAttribute("y", node.y - 3);
    halo.setAttribute("width", node.w + 6);
    halo.setAttribute("height", node.h + 6);
    halo.setAttribute("rx", 10);
    const rect = make("rect", null, group);
    rect.setAttribute("x", node.x);
    rect.setAttribute("y", node.y);
    rect.setAttribute("width", node.w);
    rect.setAttribute("height", node.h);
    rect.setAttribute("rx", 8);
    const title = make("title", null, group);
    const name = make("text", "name", group);
    name.setAttribute("x", node.x + 10);
    name.setAttribute("y", node.y + 21);
    const meta = make("text", "meta", group);
    meta.setAttribute("x", node.x + 10);
    meta.setAttribute("y", node.y + 39);
    // Ellipsize by character count; the full name lives in the tooltip.
    const room = Math.floor((node.w - 26) / 7.5);
    const shortLabel =
      node.label.length > room ? node.label.slice(0, room - 1) + "…" : node.label;
    nodeGroups.set(node.id, { group, rect, name, meta, title, shortLabel });
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      select(node.id === selectedId ? null : node.id);
    });
    patchNode(node);
  }
  patchEdges(g);
  // A rebuild means a new topology; keep the pin only if its node survived.
  if (selectedId && !nodeGroups.has(selectedId)) select(null);
}

function renderGraph(g, note) {
  if (!g || !g.nodes.length) {
    topologyKey = null;
    panel.textContent = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = note || (g && g.note) || "waiting for the run to start…";
    panel.appendChild(empty);
    lastGraph = null;
    select(null);
    return;
  }
  const key = graphKey(g);
  if (key !== topologyKey) {
    topologyKey = key;
    lastGraph = g;
    buildGraph(g);
  } else {
    lastGraph = g;
    for (const node of g.nodes) if (node.role === "node") patchNode(node);
    patchEdges(g);
  }
  renderInspector();
}

/* --------------------------------------------------------- node inspector */

function select(id) {
  if (id === selectedId) {
    if (id === null) { inspector.hidden = true; return; }
  }
  const previous = selectedId;
  selectedId = id;
  // Re-stamp both groups' classes so exactly one carries `sel`.
  if (lastGraph) {
    for (const node of lastGraph.nodes) {
      if (node.role === "node" && (node.id === previous || node.id === id)) {
        patchNode(node);
      }
    }
  }
  renderInspector();
}

function prop(term, value, mono) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (mono === false) dd.style.fontFamily = "inherit";
  iProps.append(dt, dd);
}

function section(parent, heading, items) {
  parent.textContent = "";
  if (!items.length) return;
  const h = document.createElement("h3");
  h.textContent = heading;
  parent.append(h);
  const list = document.createElement("ul");
  for (const spans of items) {
    const li = document.createElement("li");
    for (const [text, cls] of spans) {
      const piece = document.createElement("span");
      piece.textContent = text;
      if (cls) piece.className = cls;
      li.append(piece);
    }
    list.append(li);
  }
  parent.append(list);
}

function peerLabel(id) {
  if (!lastGraph) return id;
  const node = lastGraph.nodes.find((n) => n.id === id);
  if (node) return node.label;
  const cluster = lastGraph.clusters.find((c) => c.id === id);
  return cluster ? cluster.label : id;
}

function renderInspector() {
  if (!selectedId || !lastGraph) { inspector.hidden = true; return; }
  const node = lastGraph.nodes.find(
    (n) => n.id === selectedId && n.role === "node"
  );
  if (!node) { inspector.hidden = true; return; }

  const shown = awaiting && node.status === "pending" ? "proposed" : node.status;
  iName.textContent = node.label;
  iStatus.textContent =
    GLYPH[shown] + " " + (shown === "proposed" ? "proposed — awaiting approval" : shown);
  iStatus.className = "st-" + shown;

  iProps.textContent = "";
  if (node.cluster) {
    const cluster = lastGraph.clusters.find((c) => c.id === node.cluster);
    if (cluster) prop("round", cluster.label, false);
  }
  prop("executions", String(node.executions));
  if (node.status === "running" && node.live_tokens) {
    prop("tokens", fmtTokens(node.live_tokens) + " so far");
  } else if (node.tokens) {
    prop("tokens", fmtTokens(node.tokens));
    if (lastStats && lastStats.tokens) {
      prop("share", Math.round((node.tokens / lastStats.tokens) * 100) + "% of run");
    }
  }
  const cost = fmtCost(node.cost_usd);
  if (cost) prop("cost", cost);
  if (node.duration_ms != null) prop("duration", fmtMs(node.duration_ms));
  if (node.id !== node.label) prop("id", node.id);

  iError.hidden = !node.error;
  iError.textContent = node.error || "";

  const spans = (lastGraph.timeline || []).filter((s) => s.node === node.id);
  section(
    iSpans,
    "executions on the timeline",
    spans.map((s) => {
      if (s.t1 == null) {
        return [["▸ started at " + fmtMs(s.t0) + " · still open", "open"]];
      }
      return [
        [s.ok === false ? "✗ " : "✓ ", s.ok === false ? "bad" : "ok"],
        [fmtMs(s.t0) + " → " + fmtMs(s.t1) + " · " + fmtMs(s.t1 - s.t0), ""],
      ];
    })
  );

  const kindTag = (kind) => (kind === "static" ? "" : " · " + kind);
  const wiring = [];
  for (const e of lastGraph.edges) {
    if (e.target === node.id) {
      wiring.push([["← from ", ""], [peerLabel(e.source), "peer"], [kindTag(e.kind), ""]]);
    }
  }
  for (const e of lastGraph.edges) {
    if (e.source === node.id) {
      wiring.push([["→ to ", ""], [peerLabel(e.target), "peer"], [kindTag(e.kind), ""]]);
    }
  }
  section(iEdges, "wiring", wiring);

  inspector.hidden = false;
}

iClose.addEventListener("click", () => select(null));
panel.addEventListener("click", () => select(null));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") select(null);
});

/* -------------------------------------------------------- planning panel */

function cell(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderPlanning(p) {
  planningEl.textContent = "";
  if (!p || (!p.rounds.length && !p.stop)) {
    planningEl.hidden = true;
    return;
  }
  planningEl.append(cell("h2", p.has_topology ? "planning history" : "planning"));
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const h of ["round", "status", "nodes", "why", "tokens"]) head.append(cell("th", h));
  table.append(head);
  for (const r of p.rounds) {
    const tr = document.createElement("tr");
    if (r.in_flight) tr.className = "inflight";
    tr.append(cell("td", "round " + r.round));
    tr.append(cell("td", r.in_flight ? (r.status || "planning…") : (r.status || "—"),
                   r.status === "rejected" ? "rejected" : ""));
    tr.append(cell("td", String(r.nodes || 0)));
    const why = [...(r.rejections || []), ...(r.failed_checks || [])];
    tr.append(cell("td", why.length ? [...new Set(why)].join(", ") : (r.error || "—")));
    tr.append(cell("td", String(r.tokens || 0)));
    table.append(tr);
  }
  planningEl.append(table);
  if (p.stop) {
    const clean =
      p.stop === "goal_met" || p.stop === "no_further_work" || p.stop === "planned";
    planningEl.append(
      cell(
        "p",
        "stopped: " + p.stop + (p.stop_detail ? " — " + p.stop_detail : ""),
        clean ? "stop ok" : "stop"
      )
    );
  }
  planningEl.hidden = false;
}

/* --------------------------------------------------------------- replay */

let horizon = 0;
let playing = null; // {t0Real, t0Scrub} while the animation runs

function statusAtTime(spans, t) {
  // Mirrors the server's status rule (errored > running > done > pending),
  // evaluated against what had happened by time t.
  return (id) => {
    const list = spans.get(id);
    if (!list) return "pending";
    let running = false, done = false, errored = false;
    for (const span of list) {
      if (span.t0 > t) continue;
      if (span.t1 == null || span.t1 > t) running = true;
      else if (span.ok === false) errored = true;
      else done = true;
    }
    if (errored) return "errored";
    if (running) return "running";
    if (done) return "done";
    return "pending";
  };
}

function scrubTo(t) {
  if (!lastGraph || !lastGraph.timeline) return;
  const spans = new Map();
  for (const span of lastGraph.timeline) {
    if (!spans.has(span.node)) spans.set(span.node, []);
    spans.get(span.node).push(span);
  }
  const at = statusAtTime(spans, t);
  for (const node of lastGraph.nodes) {
    if (node.role === "node") patchNode(node, at(node.id));
  }
  patchEdges(lastGraph, at);
  clock.textContent = fmtMs(t);
}

function stopPlaying() {
  playing = null;
  playBtn.textContent = "▶ replay";
}

function tick(now) {
  if (!playing) return;
  const speed = Number(speedPick.value) || 4;
  const t = Math.min(horizon, playing.t0Scrub + (now - playing.t0Real) * speed);
  scrub.value = String(Math.round((t / horizon) * 1000));
  scrubTo(t);
  if (t >= horizon) { stopPlaying(); return; }
  requestAnimationFrame(tick);
}

playBtn.addEventListener("click", () => {
  if (playing) { stopPlaying(); return; }
  const from = Number(scrub.value) >= 1000 ? 0 : (Number(scrub.value) / 1000) * horizon;
  playing = { t0Real: performance.now(), t0Scrub: from };
  playBtn.textContent = "⏸ pause";
  requestAnimationFrame(tick);
});

scrub.addEventListener("input", () => {
  stopPlaying();
  scrubTo((Number(scrub.value) / 1000) * horizon);
});

function offerPlayback(s) {
  const g = s.graph;
  if (!s.done || !g || !g.timeline || !g.timeline.length || !g.duration_ms) {
    playback.hidden = true;
    return;
  }
  horizon = g.duration_ms;
  playback.hidden = false;
  if (Number(scrub.value) >= 1000) clock.textContent = fmtMs(horizon);
}

/* ------------------------------------------------------------ the stream */

let es = null;

function connect() {
  if (es) es.close();
  es = new EventSource("/live/api/stream?" + params.toString());
  es.addEventListener("snapshot", (event) => {
    const s = JSON.parse(event.data);
    const planningOnly = s.planning && !s.planning.has_topology;
    awaiting = Boolean(s.awaiting_approval);
    lastStats = s.stats || null;
    runEl.textContent = s.run_id ? "run " + s.run_id : "";
    goalEl.textContent = s.goal ? "· " + s.goal : "";
    if (awaiting && s.approve_command) {
      approveCmd.textContent = s.approve_command;
      approvalBar.hidden = false;
    } else {
      approvalBar.hidden = true;
    }
    statusEl.textContent = s.done ? "finished"
      : (s.awaiting_approval ? "awaiting approval"
      : (s.active ? (planningOnly ? "planning" : "running")
      : (s.run_id ? "idle" : "waiting")));
    dot.className = "livedot " + (s.done ? "done"
      : (s.awaiting_approval ? "awaiting"
      : (s.active ? (planningOnly ? "planning" : "running") : "")));
    renderTiles(s);
    renderPlanning(s.planning);
    if (s.run_ids && s.run_ids.length > 1) {
      runPick.hidden = false;
      if (runPick.length !== s.run_ids.length) {
        runPick.textContent = "";
        for (const id of s.run_ids) {
          const option = document.createElement("option");
          option.value = id;
          option.textContent = id;
          runPick.append(option);
        }
      }
      runPick.value = s.run_id || "";
    }
    if (planningOnly && !s.done) {
      // The "no graph ran" placeholder is an answer for a finished run, not
      // for one whose planner is still thinking.
      renderGraph(null, "planning…");
      return;
    }
    renderGraph(s.graph);
    offerPlayback(s);
    if (s.mermaid) {
      srcEl.textContent = s.mermaid;
      if (s.mermaid_live_url) { mlive.href = s.mermaid_live_url; mlive.hidden = false; }
    }
  });
  es.addEventListener("done", () => {
    es.close();
    statusEl.textContent = "finished";
    dot.className = "livedot done";
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CONNECTING) statusEl.textContent = "reconnecting…";
  };
}

copyApprove.addEventListener("click", async () => {
  if (!navigator.clipboard) return; // non-secure context: the text is selectable
  try {
    await navigator.clipboard.writeText(approveCmd.textContent);
    copyApprove.textContent = "copied";
    setTimeout(() => { copyApprove.textContent = "copy"; }, 1500);
  } catch { /* the command is right there to select */ }
});

runPick.addEventListener("change", () => {
  params.set("run", runPick.value);
  topologyKey = null;
  connect();
});

connect();
