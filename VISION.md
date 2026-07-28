# GraphARC — a general-purpose agent runtime on governed execution graphs

## Thesis

Every mainstream agent harness — Claude Code, OpenClaw, Hermes — has the same
shape: one implicit loop, plus an ever-growing rind of callbacks, permission
checks, hooks, retry logic, subagent spawns, and special cases. The control
flow is real, but it is *scattered*: it lives in prompt text, in handler
registration order, in the model's judgment, and in whatever the last incident
forced someone to bolt on.

GraphARC's bet is that the control plane should be an explicit, typed,
governed graph.

```
Typical harness                          GraphARC
───────────────                          ────────
one agent loop                           agent behavior IS the graph
model picks next action                  routers make validated transitions
harness accretes callbacks               policies are graph constraints
control flow implicit + scattered        every loop explicit, bounded, traced
"be careful" in the system prompt        enforcement in code, on the edge
```

The differentiator is not "we use a graph library." It is that **no transition
happens that the graph did not permit, and no work happens that the budget did
not authorize** — and that this is true whether the topology was authored up
front or constructed at runtime.

---

## The hard part, stated honestly

There is a real tension in this vision, and pretending otherwise would produce
a system that demos well and dies on contact with open-ended work.

**Explicit graphs are good at known topologies. General-purpose agents face
unknown ones.** You cannot pre-author a graph for "investigate this incident"
or "refactor this repo" — the shape of the work is discovered while doing it.
This is not hypothetical: GPT Researcher migrated *off* LangGraph precisely
because deep-research planning emerges dynamically and a static graph fought
it.

So the naive reading of this vision — "draw the whole agent as a flowchart" —
fails. The resolution is:

> **The graph is constructed at runtime, under governance.**

A planner node doesn't *decide what to do next*; it *proposes a subgraph*,
which is then type-checked, budget-scoped, and policy-filtered before a single
node of it executes. Dynamism lives in topology construction. Governance lives
in admission. That combination is the actual product, and as far as I can tell
nobody ships it.

Concretely, the invariant to hold — and, as of this revision, the invariant
`grapharc/planner/` holds:

- A node may **propose** new nodes and edges, never **execute** them directly.
  A `Subgraph` forbids extra fields, so there is no channel through which a
  proposal could carry a callable; every node body comes from a factory the
  operator registered before any planning happened.
- Every proposed subgraph is admitted or rejected by a deterministic checker:
  do all nodes exist in the registry? are all edges permitted by policy? does
  the budget cover the worst case? is the depth within limits? is it acyclic?
  All five run on every proposal, so a planner gets the complete list of
  objections rather than the first one.
- Admission is traced. A rejected subgraph is a first-class recorded event,
  not a silent fallback — and the reason is handed back to the planner verbatim
  as its next round's feedback.

That is the difference between "an agent that can do anything" and "an agent
that can do anything you authorized, and you can prove which."

One limit belongs here rather than in a footnote, because it is the sentence
above most likely to be over-read: **admission authorises a node's *kind*, not
its arguments.** No rule reaches a proposal's `args`, so a node whose kind is
permitted is admitted whatever it was asked to do with it. Constraining that is
the job of the factory the operator wrote.

---

## What the runtime must provide

Ten subsystems. All ten now exist in some form. Figures re-derived on
2026-07-28 by executing each claim against the tree; [ROADMAP.md](ROADMAP.md)
carries the item-by-item working.

| Subsystem | Purpose | Today |
|---|---|---|
| **Graph kernel** | typed state, declared writes, bounded loops, fan-out, checkpoints, traces | **~85% — async, `Command` returns, state access** |
| **Model gateway** | provider adapters, streaming, tool-calling, structured output, routing, failover, cost ceilings | **~80% — ceilings enforced, retries with backoff** |
| **Tool plane** | registry, permissions, approval gates, hooks, sandboxed fs/shell/browser/network | **~55% — seven core tools, container executor** |
| **Memory plane** | durable claims with provenance, artifacts, retrieval | **~85% — SQLite, BM25F + traversal, artifacts** |
| **Agent node** | the reusable observe→act→verify unit that composes into graphs | **~60% — built and driving tools** |
| **Planner / admission** | propose subgraphs, type-check, budget-scope, policy-filter, record | **~85% — the crux, and the cycle closed** |
| **Session runtime** | long-lived, resumable, interruptible, multi-turn, steerable | **~85% — resumes across processes** |
| **Policy engine** | declarative rules over nodes/edges/tools/spend, human approval routing | **~75% built, 0% wired** |
| **Triggers & surfaces** | CLI, HTTP API, cron, webhooks, chat channels | **~55% — nine CLI commands, HTTP + SSE** |
| **Operations** | metrics, replay, rollback, versioned configs, multi-tenancy | **~60% — replay, diff, OTel spans, cost** |

Honest total: roughly **65% of this product**, up from 15%. Two things keep
that number from meaning what it looks like.

**First, the differentiated part is now the built part.** The old 15% was the
graph kernel — the piece LangGraph already mostly gives you. What has landed
since is the admission gate, the governed loop, the session layer and the
policy engine, which is to say the part of the thesis that had no prior art.
The claim in the section above — a planner proposes, a deterministic checker
admits, and nothing unauthorised runs — is code that executes today, and
ARCHITECTURE.md §7 records the run that demonstrates it.

**Second, four of those percentages count code that nothing calls.** The
governed loop has no CLI command or example driving it. The policy engine has
no importer anywhere in the package, and no bridge from its TOML document to
the `EdgePolicy` the admission checker actually consults. The HTTP API does not
use the durable session layer; it ships its own in-process one that dies with
the process. A subsystem that works and is unreachable is genuinely built and
is worth less to a reader than the number suggests. Those seams are ROADMAP
§12, and they are the whole of the next-five list.

And one claim outside the table, because it is the first thing a reader tests:
**the source is not on the public remote.** `origin/main` holds a licence and a
README, so the documented `git clone && uv sync` fails. Every figure above was
measured against a local tree.

---

## What changed about the existing code

This section used to be a set of predictions. Most of them have happened, so it
is now a record — and the interesting part is which ones did not.

**`harness/` was not orphaned; it had no agent yet.** That was the call, and it
was right. `AgentNode` now drives the registry, the permission engine and the
executor, `grapharc agent` drives the seven core tools, and the `ctypes` hole
and the child environment are both closed. So is the one the first audit
missed: a sandboxed tool can no longer write into `site-packages`, verified by
trying to plant a `.pth` file and getting `SandboxViolation`. The audit hook is
still in-process confinement rather than a kernel boundary, which is why
`ContainerExecutor` now exists beside it.

**`memory/` was the durable half of "two graphs," and it got its disk.** Claims
and artifacts both live in SQLite, verified durable across genuinely separate
processes. Retrieval stopped being an exact-string scan labelled GraphRAG and
became BM25F plus graph traversal plus an optional injected vector channel.
Contradiction detection landed and — correctly — reports rather than resolves.

**`gateway/` became the model plane.** `bind_tools`, `with_structured_output`,
streaming and async all work on OpenRouter; retries carry a real
transient/deterministic split; and cost ceilings are enforced on both sides of
a call rather than merely captured.

**The kernel stopped amputating LangGraph.** `ainvoke` / `astream` /
`astream_events` all run, `async def` nodes execute (and the sync entry points
refuse them loudly rather than failing halfway), `Command` returns route
dynamically with the update still passing the write allowlist, and
`get_state` / `update_state` / `get_state_history` are wrapped rather than
requiring `.inner`. `update_state` is not a bare passthrough: it type-checks and,
given `as_node=`, applies that node's declared writes.

Three predictions that did **not** come true, which is the part worth keeping:

- **The decorator form never happened.** The write-permission idea was supposed
  to become a composable node decorator rather than a wrapper. It is still a
  wrapper. That is now a smaller problem than it looked, because the wrapper
  stopped removing the capabilities that made composition necessary — but the
  ergonomic argument for a decorator stands, and it is ROADMAP §1.6.
- **`.inner` is still an inspection escape hatch, and still fails closed for
  execution.** That was the right call and it has a cost: LangGraph's native
  `interrupt()` suspends a GraphARC graph correctly but there is no supported
  way to resume it, because resuming means passing a `Command` as *input* and
  that path is closed by design. Human-in-the-loop goes through the session
  layer's approval gate instead.
- **Nothing predicted the seams.** The failure mode this document warned about
  was writing the essay before the code. The one it actually hit is subtler:
  building four subsystems that each work and wiring none of them to each
  other. `planner/` and `policy/` are imported by no other module in the
  package. That is not a design flaw and it is not vapour — it is a fifth of
  the remaining work, and it was invisible from here.

---

## Build order

The organizing principle: **one vertical slice through every subsystem beats
ten horizontal layers.** Each milestone must run a real task against a real
model, or it doesn't count — and that last clause is doing real work below.

### V0 — the agent node (the unblocking move) — **passed**

Wire kernel + gateway + harness into a single `AgentNode`: observe → call model
→ request tool → permission check → sandboxed execute → verify → repeat, with
budgets and traces throughout.

*Done when:* a one-node graph edits a file in a sandbox and runs a test,
against a real model, with the tool call permission-gated and the whole thing
inside a token budget. **Gate run and recorded.**

### V1 — dynamic subgraphs under admission — **mechanism done, gate not run**

The planner node and the admission checker. A planner proposes a subgraph; the
checker validates node existence, edge policy, budget coverage, and depth;
admitted graphs execute; rejections are traced.

All of that works. A planner proposing an unregistered kind, a policy-denied
edge, an over-budget branch or a cycle is rejected with a per-check reason
recorded on the trace, and the reason is handed back as the next round's
feedback. `Materializer` will only build a proposal an `AdmissionResult`
authorised, matched by fingerprint.

*Done when:* "refactor this repo and run tests" plans its own fan-out over
files — **against a real model.** That run has not happened. Everything
demonstrated so far used scripted planners, which is exactly the substitution
this document warned against, so the milestone stays open.

### V2 — session runtime — **mechanism done, gate not run**

Long-lived, resumable, interruptible sessions. Human approval as a real graph
node that suspends and resumes. Artifacts on disk.

A session survives a process restart (verified across two interpreters, with
each node appearing exactly once in an append-only log), a human approval holds
every gated node on a superstep boundary separately, and artifacts are durable
in SQLite with content-addressed blobs. What is missing from the original
framing is "async throughout": the kernel is async, and a session turn is still
synchronous.

*Done when:* the same thing happens with a real model driving it.

### V3 — policy engine + surfaces — **half built, unwired**

Declarative policy (who may call what, spend what, touch what), an HTTP API,
cron and webhook triggers.

The policy engine and the HTTP API both exist. The policy engine is called by
nothing, there is no bridge from its document to the admission checker's edge
policy, and there are no cron or webhook triggers.

*Done when:* the incident-response example runs end to end from a webhook,
with remediation gated on approval.

### V4 — operations — **replay works; distribution does not**

Replay, rollback, versioned graph/prompt configs, per-tenant budgets, metrics.

Replay, run-diffing, OpenTelemetry spans and per-node cost attribution all
work off the single trace file. Rollback, versioned configs and per-tenant
budgets do not. Neither does the last clause of the gate: nobody can install
this, because the source is not on the public remote.

---

## What this competes with, and where it wins

Being specific, because "better agent framework" is not a strategy.

- **vs. Claude Code / OpenClaw / Hermes** — they are more capable *today* at
  general tasks and will stay ahead on breadth. GraphARC's claim is
  auditability and enforcement: you can state what an agent is permitted to do
  and prove what it did. That matters for regulated, multi-tenant, and
  unattended work — which is exactly where those harnesses are weakest.
- **vs. LangGraph** — LangGraph is the engine, not the competitor. It gives
  topology and durability; it deliberately leaves policy, tools, memory, and
  operations to you. GraphARC's product is that layer.
- **vs. Temporal / Airflow** — real governance and durability, no notion of
  probabilistic nodes, model budgets, or evidence verification.

The wedge is **unattended agents that touch real systems** — where "the model
usually behaves" is not an acceptable safety argument, and where after an
incident you must answer *what did it do, what was it allowed to do, and why
did it stop.*

---

## Honest scale

This was written as a multi-month build for one person, with the risk
concentrated in V1's admission checker — the piece with no prior art to copy.

**That risk resolved.** The admission checker exists, it is deterministic and
model-free, and the propose → admit → materialise → execute → replan cycle runs
end to end. The thing that had never been built is the thing that got built,
which is the outcome this document was least entitled to expect.

What replaced it is duller and more dangerous: **integration debt**. The
remaining work is not research. It is a CLI command that drives the loop, a
function that turns a policy document into an edge policy, a `SessionRuntime`
implementation that puts the HTTP API on the durable session layer, one extra
argument at each `trace.event(...)` call site, and a `git push`. None of that
is hard, all of it is invisible from a test suite that passes, and it is the
entire distance between "the pieces work" and "a stranger can use this."

The failure mode to avoid is the one this repo already fell into once: writing
the essay before the code, and marking milestones complete because the tests
pass rather than because a real task ran. It has a second form, which V1 and V2
above are now labelled against: **mechanism done is not gate passed.** Both
milestones have working machinery and neither has been pointed at a real task
with a real model. Calling them finished on the strength of a green suite would
be the same mistake in a better disguise.
