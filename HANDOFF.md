# GraphARC — handoff

Everything needed to pick this up in a fresh session. Written 2026-07-28.

---

## The goal

**GraphARC is a general-purpose agent runtime whose control plane is an explicit,
governed graph** — not an implicit loop wrapped in accreting harness logic.

```
Typical harness (Claude Code, OpenClaw, Hermes)   GraphARC
──────────────────────────────────────────────    ────────
one agent loop                                    agent behavior IS the graph
model picks the next action                       routers make validated transitions
harness accretes callbacks over time              policies are graph constraints
control flow implicit and scattered               every loop explicit, bounded, traced
"be careful" in the system prompt                 enforcement in code, on the edge
```

The differentiator is not "we use a graph library." It is that **no transition executes
that the graph did not permit, and no work happens that the budget did not authorize** —
whether the topology was authored up front or constructed at runtime.

The wedge: **unattended agents that touch real systems**, where "the model usually behaves"
is not an acceptable safety argument, and where after an incident you must answer *what did
it do*, *what was it allowed to do*, and *why did it stop*.

Full design: `ARCHITECTURE.md`. Thesis and honest scope: `VISION.md`.

---

## Where things stand

| | |
|---|---|
| Repo | `/home/shashank/Desktop/GraphARC`, branch `main` |
| Remote | `github.com/CodeGraphContext/GraphARC` |
| HEAD | `a9315e5` — "v1: async kernel, admission checker, sessions, HTTP API, tools, policy, ops" |
| Tests | **1,082 passing**, 10 live tests deselected by default |
| Lint | clean (`ruff`) |
| Build | wheel + sdist build; 90 modules import from the wheel in a clean venv |
| Version | `0.1.0a0` — **not yet bumped for release** |
| Python | 3.12+ (dev venv is 3.14) |

**Environment:** `.env` holds an OpenRouter key (gitignored, never committed). The user has a
Claude Max 20x subscription and **no Anthropic API key** — which is why the Claude-CLI gateway
backend exists. Run tests with `.venv/bin/python -m pytest`. **Never run `pytest -m live`
casually — those call paid APIs.**

### Twelve subsystems

```
grapharc/
  runtime/    graph kernel: typed state, declared writes, budgets, async, traces
  planner/    proposals + admission checker      <- THE CRUX, see gap below
  harness/    tool registry, permissions, sandbox, container executor, AgentNode
  gateway/    model plane: Claude CLI + OpenRouter, retries, cost ceilings
  memory/     claims with provenance, SQLite, traversal, contradiction detection
  session/    long-lived, cross-process resume, interrupt, human approval
  server/     FastAPI + SSE
  policy/     TOML rules, approval routing, decision audit
  observe/    JSONL traces, replay, metrics, OTel, cost attribution
  tools/      seven core tools with workspace confinement
  cli/        nine commands, --json on every one
  examples/   stages 0-6, capstone, agent_fixit
```

---

## THE BIGGEST REMAINING GAP

**The propose → admit → execute → replan loop is only half built.**

`grapharc/planner/` can turn a model's output into a typed `Subgraph` proposal
(`PlannerNode`) and can validate it (`AdmissionChecker` — registry, policy, budget, depth,
acyclicity). Verified by grep: the only `.invoke` calls in the package are the planner's own
model calls.

What does **not** exist as of `a9315e5`:
- nothing **materializes** an admitted proposal into a runnable graph
- nothing runs the **execute → observe → replan** cycle

So `ARCHITECTURE.md` Figure 2 is implemented from "propose" to "admit" and stops exactly
where the thesis gets interesting. **This is the single most valuable thing left.**

> **In flight when this was written:** an agent was adding `grapharc/planner/materialize.py`
> and `loop.py` to close it. `materialize.py` exists uncommitted. **Verify what actually
> landed before trusting this section** — check for `loop.py`, then grep the package for a
> replan cycle and for materialization refusing an unadmitted proposal.

The two properties that must hold, and that the verifier was told to attack:
1. Node bodies come **only** from the `NodeRegistry`. A proposal names a *kind*; it can never
   supply a body. That is the boundary.
2. Every replanning round **re-enters admission**. Work discovered mid-run must not bypass
   the gate — that is what makes the design general-purpose rather than a flowchart.

---

## Other work in flight (uncommitted)

- `docs/cookbook/` — six sections, every snippet required to be *executed* and its real
  output pasted in, each backed by `tests/test_cookbook_*.py` so the docs cannot rot. The
  six test files exist uncommitted; the `.md` files were still being written.
- `README.md` / `ROADMAP.md` / `VISION.md` / `ARCHITECTURE.md` — a truth pass. These are
  **stale and now UNDERCLAIM**: they still say async, sessions, the HTTP API, the container
  executor, tools, and policy are "not started" when HEAD contains all of them.
  `ARCHITECTURE.md` §7 "Where we are against this" is the stalest thing in the repo.

Check `git status` first. If these are half-landed, finish or discard them before building
anything new.

---

## How this project works (the part that matters most)

**Every claim must survive execution.** This is not a style preference — it is the reason the
project is worth anything, and it has been earned the hard way.

Three adversarial audits have caught this repo shipping confident prose over guarantees the
code did not provide:

- The README once shipped **five false statements** simultaneously ("validated routing" that
  validated nothing, "durable memory" that was an in-process dict, "crash-safe resume" that
  was entirely LangGraph's).
- The audit-hook sandbox was escaped **twice** — first by three lines of stdlib `ctypes`,
  then, after that fix, by writing a `.pth` into a writable site-packages, which owns every
  later interpreter start.
- Budgets reported `spent: 0 tokens` on a live run that genuinely cost money.
- The approval gate released **two** nodes when one was approved.
- The admission checker was evaded by **renaming** an instance, because policy keyed on the
  planner-chosen name instead of the registry kind.

Every one of those was found by an agent told to *break* the work, not to review it. So:

1. **Build with subagents in parallel on strictly disjoint file sets**, then run a separate
   adversarial verifier per risky subsystem that re-runs the proofs rather than trusting the
   report. The verifiers have disproved builder claims in every single round.
2. **Reproduce a defect before fixing it**, and prove the fix with a test that fails without it.
3. **Document what you could not close.** Known limits live in module docstrings and
   `ROADMAP.md` §0, not smoothed over.
4. Never weaken a test to make something pass.

`ASSESSMENT.md` is the honest self-assessment — three independent reviewers were asked whether
this is a thin LangGraph wrapper, told not to be kind, and three said yes. Read it before
believing anything good about the project.

---

## What is genuinely novel vs. what is not

Be honest about this; it shapes what to build next.

**Mostly not novel.** LangGraph does the graph. LangSmith does tracing better. Graphiti does
memory better. Docker does sandboxing with an actual kernel boundary. LiteLLM does routing at
scale. Each is more mature than the version here.

**Two real exceptions:**
- `ClaudeCodeCLIChatModel` — a LangChain `BaseChatModel` over `claude -p`, so LangGraph runs
  on a Claude subscription with no API key, with the CLI stripped to a pure inference endpoint
  (all tools disallowed, no settings sources, prompt via stdin, argv array never a shell
  string). No drop-in substitute exists. ~139 lines, independent of the rest.
- **The admission checker** — and it only becomes genuinely novel once the loop above is
  closed. Until then it validates topology nobody runs.

---

## Suggested next steps

1. **Verify and finish the loop** (`planner/materialize.py`, `loop.py`). Nothing else moves
   the project as much. Attack it: try to materialize an unadmitted proposal, smuggle a node
   body through a proposal, make round 2 skip admission.
2. **Land the doc truth pass and the cookbook**, then re-read the README as a stranger would.
3. **Decide the version.** `0.1.0a0` today. I would argue for `0.1.0`, not `1.0.0` — a 1.0
   implies API stability and several subsystems are days old.
4. **Publish to PyPI** (`.github/workflows/release.yml` exists, trusted publishing, tag-driven).
5. Then, in rough order: wire the harness into the remaining example graphs; a real
   container-executor story for tools needing subprocesses; per-task approval holds
   (currently a hold names a node, so a `Send` fan-out produces indistinguishable holds);
   admission inspecting node **arguments**, not just kinds.

---

## Known limits, stated plainly

Do not let these get quietly dropped from the docs.

- **The audit-hook sandbox is not a kernel boundary.** It is in-process CPython confinement.
  Four holes are open by construction and documented in `harness/executor.py`: `os.stat`
  metadata reads, raw inherited file descriptors, the forked heap, and trusted runtime
  extensions. `ContainerExecutor` is the real boundary — but a tool must be resolvable inside
  the container, and running `pytest` from a tool means a subprocess, which the audit-hook
  sandbox refuses by design.
- **Admission authorizes a *kind*, not its *arguments*.** `ProposedNode.args` are not inspected.
- **A `Command` passed as graph input** is not validated the way one returned by a node is.
- **Approval holds name a node, not a task**, so a `Send` fan-out creates indistinguishable
  holds and rejecting one skips an arbitrary task of that node.
- **The HTTP server drives `astream`**, so a sync-only `SqliteSaver` raises
  `CheckpointerNotAsyncError`. Use the async saver.
- **No shipped example graph calls a tool** except `agent_fixit`.
- `max_seconds` interrupts a node but a node that catches every interrupt and never returns
  still hangs.

---

## Quick commands

```bash
.venv/bin/python -m pytest              # 1082 tests, live deselected
.venv/bin/python -m ruff check .
uv build                                 # wheel + sdist

grapharc --help                          # run agent serve models replay diff trace metrics viz
grapharc models                          # what backends are configured (redacts secrets)
grapharc run capstone --model openrouter/anthropic/claude-haiku-4.5 \
                      --reviewer-model openrouter/openai/gpt-4o-mini
python -m grapharc.examples.agent_fixit --model openrouter/anthropic/claude-haiku-4.5
```

Key docs: `ARCHITECTURE.md` (target design) · `ROADMAP.md` (numbered backlog) ·
`ASSESSMENT.md` (honest self-critique) · `VISION.md` (thesis) · `docs/cookbook/` (recipes).
