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

Concretely, the invariant to hold:

- A node may **propose** new nodes and edges, never **execute** them directly.
- Every proposed subgraph is admitted or rejected by a deterministic checker:
  do all nodes exist in the registry? are all edges permitted by policy? does
  the budget cover the worst case? is the depth within limits?
- Admission is traced. A rejected subgraph is a first-class recorded event,
  not a silent fallback.

That is the difference between "an agent that can do anything" and "an agent
that can do anything you authorized, and you can prove which."

---

## What the runtime must provide

Ten subsystems. Four exist in some form; six do not exist at all.

| Subsystem | Purpose | Today |
|---|---|---|
| **Graph kernel** | typed state, declared writes, bounded loops, fan-out, checkpoints, traces | **~70% — the only mature part** |
| **Model gateway** | provider adapters, streaming, tool-calling, structured output, routing, failover, cost ceilings | **~15% — one CLI adapter, text-only** |
| **Tool plane** | registry, permissions, approval gates, hooks, sandboxed fs/shell/browser/network | **~30% — built, tested, wired to nothing** |
| **Memory plane** | durable claims with provenance, artifacts, retrieval | **~20% — in-process dict, no disk** |
| **Agent node** | the reusable observe→act→verify unit that composes into graphs | **0%** |
| **Planner / admission** | propose subgraphs, type-check, budget-scope, policy-filter, record | **0% — the architectural crux** |
| **Session runtime** | long-lived, resumable, interruptible, multi-turn, steerable | **0%** |
| **Policy engine** | declarative rules over nodes/edges/tools/spend, human approval routing | **0%** |
| **Triggers & surfaces** | CLI, HTTP API, cron, webhooks, chat channels | **~5% — a demo CLI** |
| **Operations** | metrics, replay, rollback, versioned configs, multi-tenancy | **~20% — traces and Mermaid** |

Honest total: the repository is roughly **15% of this product**, and the 15% is
the least differentiated part, because the graph kernel is the piece LangGraph
already mostly gives you.

---

## What changes about the existing code

Your reframing rescues two subsystems I had just recommended cutting.

**`harness/` is not orphaned — it has no agent yet.** A tool registry with a
deny→ask→allow permission engine and a sandboxed executor is exactly what the
tool plane needs. It has zero callers because I built a LangGraph extension,
which has no concept of an agent that calls tools. Under this vision it is
foundational. (It still needs the `ctypes` hole closed and the child
environment scrubbed — see `ASSESSMENT.md`.)

**`memory/` is not decoration — it is the durable half of "two graphs."** The
provenance and supersession model is right; it just needs a disk.

**`gateway/` stops being a curiosity and becomes the model plane.** Right now
it can only do freeform text, which is disqualifying for an agent runtime:
`bind_tools` and `with_structured_output` both raise `NotImplementedError`, so
no agent node can call a tool through it.

What genuinely changes: the **kernel must stop amputating LangGraph**. Today
GraphARC has no `ainvoke`, rejects `Command` returns, and blocks `.inner`
access. A general agent runtime needs async (you cannot serve an HTTP API
without it), needs dynamic routing, and needs `get_state`/`update_state` for
human-in-the-loop. The write-permission idea should become a *composable node
decorator* rather than a wrapper that takes those capabilities away.

---

## Build order

The organizing principle: **one vertical slice through every subsystem beats
ten horizontal layers.** Each milestone must run a real task against a real
model, or it doesn't count.

### V0 — the agent node (the unblocking move)

Wire kernel + gateway + harness into a single `AgentNode`: observe → call model
→ request tool → permission check → sandboxed execute → verify → repeat, with
budgets and traces throughout. Requires `bind_tools` on the gateway.

*Done when:* a one-node graph edits a file in a sandbox and runs a test,
against a real model, with the tool call permission-gated and the whole thing
inside a token budget. This single milestone converts three orphaned subsystems
into a working agent and is the difference between a library and a runtime.

### V1 — dynamic subgraphs under admission

The planner node and the admission checker. A planner proposes a subgraph; the
checker validates node existence, edge policy, budget coverage, and depth;
admitted graphs execute; rejections are traced.

*Done when:* "refactor this repo and run tests" plans its own fan-out over
files, and a planner that proposes an unregistered tool or an over-budget
branch is rejected with a recorded reason rather than silently degraded.

### V2 — session runtime

Long-lived, resumable, interruptible sessions. Async throughout. Human approval
as a real graph node that suspends and resumes. Artifacts on disk.

*Done when:* a session survives a process restart, a human approves a
destructive action mid-run, and an interrupt stops work at a safe boundary.

### V3 — policy engine + surfaces

Declarative policy (who may call what, spend what, touch what), an HTTP API,
cron and webhook triggers.

*Done when:* the incident-response example runs end to end from a webhook,
with remediation gated on approval.

### V4 — operations

Replay, rollback, versioned graph/prompt configs, per-tenant budgets, metrics.

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

This is a multi-month build for one person, not a weekend. V0 is days. V1 is
weeks. V2–V4 are months, and most of the risk lives in V1's admission checker
— that is the piece with no prior art to copy.

The failure mode to avoid is the one this repo already fell into once: writing
the essay before the code, and marking milestones complete because the tests
pass rather than because a real task ran. Every milestone above is defined by a
real task against a real model for exactly that reason.
