# GraphARC — handoff

Everything needed to pick this up in a fresh session. Rewritten 2026-07-28,
after the governed loop landed.

**How to read this.** Numbers here are snapshots and rot fast — the last version
of this file went stale in under a day. Where a fact is re-derivable, the
command is given next to it; run the command rather than trusting the number.
Where a fact is *not* re-derivable from the tree — why a decision was made, what
was already tried and failed — that is what this document is actually for.

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

The differentiator is not "we use a graph library." It is that **no transition
executes that the graph did not permit, and no work happens that the budget did
not authorize** — whether the topology was authored up front or constructed at
runtime.

The wedge: **unattended agents that touch real systems**, where "the model
usually behaves" is not an acceptable safety argument, and where after an
incident you must answer *what did it do*, *what was it allowed to do*, and *why
did it stop*.

Full design: `ARCHITECTURE.md`. Thesis and honest scope: `VISION.md`.

---

## Where things stand

| | | re-derive with |
|---|---|---|
| Repo | `/home/shashank/Desktop/GraphARC`, branch `main` | |
| Remote | `github.com/CodeGraphContext/GraphARC` | `git remote -v` |
| HEAD | see `git log -1`; it moved four times during the last session | `git log -1` |
| **Unpushed** | **14 commits.** `origin/main` is still at `feef03d` "Initial commit". | `git log origin/main..HEAD` |
| Tests | **1,339 passed, 10 deselected** (live) | `.venv/bin/python -m pytest` |
| Lint | clean | `.venv/bin/python -m ruff check .` |
| Build | wheel + sdist; 93 modules import from the wheel in a clean venv | `uv build` |
| Version | `0.1.0a0` — **not yet bumped for release** | |
| Python | 3.12+ declared; dev venv is 3.14 | |

**Pushing is safe and needs no force.** `origin/main` (`feef03d`) *is* an
ancestor of HEAD, so a plain `git push` fast-forwards. This is worth stating
because the history *was* rewritten (see below) — but only across commits the
remote had never seen, so nothing on the remote is being overwritten. Verify
before pushing: `git merge-base --is-ancestor origin/main HEAD && echo safe`.

**Environment:** `.env` holds an OpenRouter key (gitignored, never committed).
The user has a Claude Max 20x subscription and **no Anthropic API key** — which
is why the Claude-CLI gateway backend exists. Run tests with
`.venv/bin/python -m pytest`. **Never run `pytest -m live` casually — those call
paid APIs and cost real money.**

> ### The `-qq` trap — read this before you believe a test run
>
> `addopts` in `pyproject.toml` is `-q -m 'not live' --strict-markers
> --strict-config`. Two traps live in that one line.
>
> **1. `pytest -q` becomes `-qq`** and pytest then **silently drops the summary
> line**: a fully passing run prints dots and then nothing, which reads exactly
> like a run that died. Cost real time to diagnose, twice.
>
> **2. `-o addopts=""` also clears `-m 'not live'`** — so the "fix" for trap 1
> runs the ten live tests against paid OpenRouter and Claude endpoints. This
> already happened: three verification agents were told to use `-o addopts=""`
> to recover the summary line and spent the user's money doing it. The `skipif`
> guards do not save you, because `config.openrouter_api_key()` reads `.env`
> directly and finds the key.
>
> **Use bare `.venv/bin/python -m pytest`.** If you must override addopts,
> always write `-o addopts="" -m "not live"` — never the first half alone.

> ### This repo has been edited by concurrent agents
>
> During one verification session HEAD moved four times and HANDOFF.md was
> rewritten twice *while agents were auditing it*, so several of their findings
> were already stale when reported. If you fan work out, give the agents
> **disjoint file sets**, and re-derive any number you intend to write down
> rather than quoting one an agent gave you. `git status` before you start.

### History rewrite, already done

The README once said "*inspired by* (never copied from)". The user judged the
parenthetical bad-looking and asked for it to be erased from history, not merely
deleted going forward. Done via `filter-branch --tree-filter`, then
`refs/original/`, the backup tag, the stash and the reflog were purged and
`gc --prune=now` run. Verified unreachable from every ref:

```bash
# Pathspec matters: this file quotes the phrase, so an unrestricted
# search now matches HANDOFF.md's own history instead of the README's.
git log --all -S "not copied from" -- README.md    # must print nothing
```

Attribution to OpenClaw, Hermes, Claude Code and OpenRouter is intact and
should stay — the architecture genuinely draws on them. Only the parenthetical
went.

---

## Twelve subsystems

Source lines excluding tests (`find grapharc/<pkg> -name '*.py' | xargs cat | wc -l`):

```
grapharc/
  planner/   2696  proposals, admission, materialisation, the governed loop
  harness/   2165  tool registry, permissions, sandbox, container executor, AgentNode
  memory/    2165  claims with provenance, SQLite, traversal, contradiction detection
  session/   2084  long-lived, cross-process resume, interrupt, human approval
  runtime/   1921  graph kernel: typed state, declared writes, budgets, async, traces
  observe/   1708  JSONL traces, replay, metrics, OTel, cost attribution
  cli/       1549  nine commands (run agent serve models replay diff trace metrics viz), --json on each
  server/    1320  FastAPI + SSE
  examples/  1244  stages 0-6, capstone, agent_fixit
  gateway/   1215  model plane: Claude CLI + OpenRouter, retries, cost ceilings
  tools/     1054  seven core tools with workspace confinement
  policy/     867  TOML rules, approval routing, decision audit
```

`planner/` is now the **largest** subsystem. One handoff ago it was 1,409 lines
and ranked sixth *smallest* of twelve — not the thinnest, but the one whose
central claim was unbuilt. That inversion is the story of the last session.

38 test files. `tests/test_planner_loop.py` (62 tests) is the one to read first
if you are touching the gate. The seven core tools are `read_file`,
`write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `run_command`
(`grapharc.tools.CORE_TOOL_NAMES`).

---

## What changed since the last handoff

Two commits, both worth reading before you touch anything:

- **`7add137` — the governed loop, and a cookbook that runs.**
- **`857f620` — corrected test count; ARCHITECTURE §7 gained a fifth gap.**

### The loop is closed

The previous handoff's headline was *"THE BIGGEST REMAINING GAP: the propose →
admit → execute → replan loop is only half built."* **It is now built.** Two
modules did it:

`grapharc/planner/materialize.py` — `Materializer.materialize(admitted, proposal)`
is the *only* path from an admitted proposal to a runnable graph. The
authorisation is the first argument; a non-`ADMITTED` result is refused; the
proposal is matched to the result by **content fingerprint**, so "what ran is
what was admitted" is a checked equality rather than a convention. Node bodies
come from the `NodeRegistry` and nowhere else — a proposal names a *kind* and
has no field a body could arrive in.

`grapharc/planner/loop.py` — `GovernedLoop.run()` drives propose → admit →
materialise → execute → observe → replan until a `LoopStop` says why it ended.
Every round goes through the same checker against the same registry and policy.
**There is no already-approved path**, so work discovered mid-run cannot bypass
the gate.

### That claim was attacked, not assumed

Thirteen attempts to get unauthorized work executed: **ten refused, three
succeeded.** The ten first, then the three — which are the more instructive
half, because two were fixable and one redraws the boundary.

Refused:

| attack | refused by |
|---|---|
| raw callable in `args` | not JSON-serialisable → no fingerprint → **never reaches admission at all** |
| `args={"body":…,"factory":…}` on kind `fetch` | registry's `fetch` body built regardless |
| node named `__start__` / `__end__` / `""` | schema `ValidationError` |
| duplicate node names | schema `ValidationError` |
| `AdmissionResult` for P1 used to build P2 | `NotAdmitted` (fingerprint mismatch) |
| mutate the proposal in place after `check()` | `NotAdmitted` |
| body returns `Command(goto="secret")` | `UnadmittedTransition` |
| body writes an undeclared field | `WritePermissionError` |
| a node name in the orchestrator's `__` namespace | `ValidationError`; the loop stops `planning_failed` |
| **through the loop:** behave in round 1, then rename `deploy` to round 1's instance name | refused *every* round; stop = `admission_refused` |

### Three attacks that SUCCEEDED — and what they actually mean

An adversarial verifier beat me on three, and the distinction between them
matters more than the count.

- **Hand-build an `AdmissionResult`** with `status=ADMITTED` and a matching
  fingerprint, for a proposal the real checker rejected → **it materialises and
  runs.** My own version of this attack "passed" only because I used an
  *unregistered kind*, so it died at the registry lookup for an unrelated
  reason. With a registered-but-policy-denied kind, it runs. **This is not
  fixable and is not a hole:** an `AdmissionResult` is ordinary Pydantic data,
  and anyone who can construct one already has the interpreter. The gate is
  between the **planner and the operator**, not between the operator and
  itself — a planner emits JSON and has no channel to build a Python object.
  Now documented in `materialize.py`'s docstring rather than left to be
  over-read.
- **Subclass `Subgraph` and lie about what it contains** → ran a denied
  `deploy`. **Fixed, on the second attempt, and the first attempt is the lesson.**
  Overriding `fingerprint()` was the reported attack; the obvious fix was to call
  the unbound `Subgraph.fingerprint(proposal)`. That is not a fix — the function
  calls `self.model_dump_json()`, so overriding *that* instead defeats it
  identically, and a second reviewer demonstrated `deploy` executing against the
  supposedly-patched tree. Every route to matching a proposal ends in a method
  call on the proposal, and a subclass reaches all of them, so the *type* is now
  refused: `materialize()` takes exactly `Subgraph`, never a subclass. Both
  override paths are parametrised in one test so neither can regress alone.

  Worth internalising: a fix that repels the reported attack while leaving the
  mechanism intact is not a fix, and it is more dangerous than the bug because
  it retires the alarm.
- **Widen the registry between rounds.** "Every round is checked against the
  same registry" meant the same *object*; `register()` stayed callable, so a
  node body could add a kind that round 2 is then admitted against.
  **Mitigation added, not on by default**: `NodeRegistry.freeze()` exists and
  works, `loop.py` points at it — and *nothing in the shipped path calls it*, so
  a registry is mutable unless you freeze it yourself. Saying this is "fixed"
  would be the same overclaim the audits keep catching.

The security claim that survives all of this, stated exactly: **a planner
cannot get unauthorized work executed.** Claims about an operator being unable
to bypass their own gate were never true and are no longer implied.

The last one had no test. It now does —
`test_a_later_round_cannot_rename_its_way_past_a_denial_the_first_round_respected`
— and it was mutation-checked: with the denial removed, `deploy`'s body runs
three times, so the assertion genuinely distinguishes the two worlds.

### The cookbook

`docs/cookbook/`, six pages, 112 recipes:

| page | recipes | lines |
|---|---|---|
| `01-basics.md` | 20 | 1505 |
| `02-models.md` | 13 | 977 |
| `03-agents-and-tools.md` | 17 | 1593 |
| `04-verification-and-memory.md` | 19 | 1114 |
| `05-governance.md` | 22 | 1757 |
| `06-serving-and-ops.md` | 21 | 1695 |

The promise is that **every snippet was executed and its real output pasted**,
never hand-written, and `tests/test_cookbook_*.py` (194 tests) enforce it.
Snippets that need credentials or cost money are exempt, and the page must
visibly say so — a *silent* exemption is a defect, and the tests assert that an
exempt snippet can never acquire an output block.

**Two places that promise is weaker than it sounds** (found by audit; do not
restate the promise without these):

- **`01-basics.md` is not extracted.** `tests/test_cookbook_basics.py` never
  opens the markdown file — its 39 tests *reproduce* each recipe in Python
  rather than parsing the page and byte-comparing. The page says "reproduces",
  which is honest; but it means page 1 alone can drift from its tests silently.
  The other five pages do parse and byte-compare. **Worth closing.**
- **`console` blocks are re-run on one page only.** `test_cookbook_models.py`
  sets `EXECUTABLE_LANGS = {"python", "console"}` and really does execute them;
  the agents and serving extractors pair ```python blocks alone, so the five
  ```console transcripts in `06-serving-and-ops.md` are checked by nothing. That
  page now says so instead of claiming "every snippet below was executed".

Writing those tests immediately caught a real flake: the session-interrupt
example posted a stop from a thread and hoped it beat the next superstep. It
lost under load. It now synchronises, and the prose says plainly that an
interrupt is read at the next boundary *after* it is written and you do not
choose which one that is.

---

## What is actually left

`ARCHITECTURE.md` §7 is the current, re-derived gap analysis — **read it rather
than trusting a copy here**, since a copy is exactly what went stale last time.
In one line each, the five gaps it names:

1. **Nothing shipped drives the loop.** `grapharc.planner` is imported by no
   other module: no CLI command, no example, no session graph. The cycle is a
   library API proven by tests, not something a reader can invoke and watch.
   **This is the highest-value thing left.**
2. **Policy does not reach admission.** `grapharc.policy` parses a TOML document
   that already understands `node`/`edge`/`tool`/`spend` rules;
   `AdmissionChecker` takes an `EdgePolicy` assembled in Python. No bridge.
3. **The HTTP API and the session runtime are two different things.**
   `grapharc/server` has its own `InProcessRuntime` whose sessions die with the
   process. The durable, cross-process `grapharc/session` is not what it uses.
4. **The shipped graphs do not use durable memory.** Every `grapharc run`
   constructs the in-process `MemoryStore()`, though `SQLiteMemoryStore` is
   verified durable across processes.
5. **Admission authorises a kind, not its arguments** — a boundary, not a seam,
   and the one most likely to be over-read. See *Known limits*.

---

## How this project works (the part that matters most)

**Every claim must survive execution.** This is not a style preference — it is
the reason the project is worth anything, and it has been earned the hard way.

Adversarial audits have repeatedly caught this repo shipping confident prose
over guarantees the code did not provide:

- The README once shipped **five false statements** simultaneously ("validated
  routing" that validated nothing, "durable memory" that was an in-process dict,
  "crash-safe resume" that was entirely LangGraph's).
- The audit-hook sandbox was escaped **twice** — first by three lines of stdlib
  `ctypes`, then, after that fix, by writing a `.pth` into a writable
  site-packages, which owns every later interpreter start.
- Budgets reported `spent: 0 tokens` on a live run that genuinely cost money.
- The approval gate released **two** nodes when one was approved.
- The admission checker was evaded by **renaming** an instance, because policy
  keyed on the planner-chosen name instead of the registry kind.
- A live demo rejected **four correct citations** because the source was
  line-wrapped, and failed three correct claims closed because the reviewer
  returned markdown-fenced JSON. Both were only visible by running it.

Every one of those was found by an agent told to *break* the work, not to
review it. So:

1. **Build with subagents in parallel on strictly disjoint file sets**, then run
   a separate adversarial verifier per risky subsystem that **re-runs the
   proofs** rather than trusting the report. The verifiers have disproved
   builder claims in every single round.
2. **Reproduce a defect before fixing it**, and prove the fix with a test that
   fails without it. Mutation-check security tests: remove the guard, confirm
   the test goes red.
3. **Document what you could not close.** Known limits live in module docstrings
   and `ROADMAP.md` §0, not smoothed over.
4. Never weaken a test to make something pass.
5. **A green test is not proof the property holds.** Attack the property
   yourself before believing it. The attacks above were run *after* the suite was
   green — and three of them landed, including one whose first fix was wrong.

`ASSESSMENT.md` is the honest self-assessment — independent reviewers were asked
whether this is a thin LangGraph wrapper, told not to be kind, and three of four
said yes. It describes an **earlier, smaller tree** and is deliberately kept
unedited; the parts it got right are worth more than the parts it has outlived.
Read it before believing anything good about the project.

---

## What is genuinely novel vs. what is not

Be honest about this; it shapes what to build next.

**Mostly not novel.** LangGraph does the graph. LangSmith does tracing better.
Graphiti does memory better. Docker does sandboxing with an actual kernel
boundary. LiteLLM does routing at scale. Each is more mature than the version
here.

**Two real exceptions:**

- **`ClaudeCodeCLIChatModel`** — a LangChain `BaseChatModel` over `claude -p`,
  so LangGraph runs on a Claude subscription with no API key, with the CLI
  stripped to a pure inference endpoint (all tools disallowed, no settings
  sources, prompt via stdin, argv array never a shell string). No drop-in
  substitute exists (306 lines, 215 of them code), and it does not depend on
  the rest of the library — it should probably be its own package.
- **The admission gate, now that the loop is closed.** The previous handoff said
  this "only becomes genuinely novel once the loop is closed. Until then it
  validates topology nobody runs." The loop is closed, so the claim can finally
  be stated properly: a planner proposes topology at runtime, a deterministic
  model-free checker admits or refuses it, materialisation is bound to that
  authorisation by content hash, and *every* replanning round is re-checked.
  The ten refused attacks above are the evidence, and the three that succeeded
  are the boundary.

  **Do not overstate it.** It is novel as a *composition*, not as new
  technology, and it is undercut by gap #1: nothing shipped drives it. Until
  there is an entry point a reader can run, the honest claim is "a governed
  planning loop exists as a library API with adversarial tests," not "GraphARC
  runs governed autonomous agents."

---

## Suggested next steps

1. **Give the loop a surface.** A `grapharc plan` command or an example graph
   that a reader can run and watch. This converts the project's single most
   defensible claim from a test fixture into a demo. Nothing else comes close in
   value. (ROADMAP §12.1)
2. **Decide the version and push.** 14 commits are unpushed, and they are the
   whole project — there is no second copy anywhere. I would argue for
   `0.1.0`, not `1.0.0` — a 1.0 implies API stability and several subsystems are
   days old.
3. **Publish to PyPI.** `.github/workflows/release.yml` is tag-driven
   (`v*`) and uses Trusted Publishing, so **no token exists anywhere**. Before a
   tag can publish, a human must do two things in a browser:
   - create a GitHub **Environment named `pypi`** in repo settings;
   - configure **PyPI Trusted Publishing** naming repo
     `CodeGraphContext/GraphARC`, workflow file `release.yml`, environment
     `pypi`, project `grapharc`.

   Until then the publish step **fails closed** with an OIDC error rather than
   uploading. The tag must equal the pyproject version exactly or the first job
   refuses. Prove the pipeline first with `workflow_dispatch` + `dry_run: true`,
   which builds and verifies but never uploads.
4. Then, in rough order: bridge `policy` → `admission` (gap 2); make the HTTP
   server use the real session runtime (gap 3); hand `grapharc run` the SQLite
   memory store (gap 4); per-task approval holds; admission inspecting node
   **arguments**, not just kinds.

---

## Known limits, stated plainly

Do not let these get quietly dropped from the docs.

- **The audit-hook sandbox is not a kernel boundary.** It is in-process CPython
  confinement. **Five** holes are open by construction and documented at the top
  of `harness/executor.py`: `os.stat` metadata reads; raw inherited file
  descriptors; the `os.readlink` guard being a removable wrapper rather than an
  audit hook; the forked heap; and trusted runtime extensions. *(The previous
  handoff said four — the monkeypatched-guard concession was added later. Count
  them in the source, don't trust this line.)* `ContainerExecutor` is the real
  boundary — but a tool must be resolvable inside the container, and running
  `pytest` from a tool means a subprocess, which the audit-hook sandbox refuses
  by design.
- **Admission authorizes a *kind*, not its *arguments*.** `ProposedNode.args`
  are not inspected. `Materializer` defaults to `forward_args=False` and drops
  them, which is the right default — but `forward_args=True` hands a planner's
  unchecked dictionary straight to a factory, and at that point the gate has
  authorised the verb and not the object. A factory that pulls a callable out of
  `args` and runs it has re-opened the boundary.
- **`GovernedLoop` is synchronous.** It drives rounds through `invoke()`, so a
  registry of `async def` bodies needs an async driver that does not exist yet.
  `Materializer` builds such a graph correctly; it just has to be driven through
  `ainvoke()` by hand.
- **Loop depth is reported, not measured.** The driver tells the checker
  `parent_depth=0`; nest a loop inside a node body and you must pass the real
  depth yourself or `max_depth` bounds nothing.
- **`UnadmittedTransition` covers a `Command` a body returns** — it says nothing
  about what a body does *inside* itself (calling another graph, spawning a
  thread). That is the tool and permission planes' business.
- **A `Command` passed as graph input** is not validated the way one returned by
  a node is.
- **Approval holds name a node, not a task**, so a `Send` fan-out creates
  indistinguishable holds and rejecting one skips an arbitrary task of that node.
- **A sync-only `SqliteSaver` breaks the HTTP server only for async graphs.**
  The server probes the checkpointer once per run and falls back to
  `CompiledGraphARC.stream()`, so a sync graph is fine;
  `CheckpointerNotAsyncError` fires only when `async def` nodes force `astream()`
  *and* the saver cannot support it. *(An earlier handoff stated this as a
  blanket failure. It is not.)*
- **No shipped example graph calls a tool** except `agent_fixit`.
- **`max_seconds` interrupts a node**, but a node that catches every interrupt
  and never returns still hangs.
- **A hard `SIGKILL` loses the last checkpoint.** LangGraph's default
  `durability` is `"async"` and `CompiledGraphARC.invoke()` does not forward the
  parameter, so a killed run replays from the input checkpoint and an
  "expensive" node runs twice. `durability="sync"` would fix it; there is no
  supported path to it today. Demonstrated in `docs/cookbook/01-basics.md`.
- **Nothing ever writes `cost_usd` to a trace.** The field exists on
  `TraceEvent` and the runtime never populates it, so
  `observe.cost.recorded_cost_usd` is always `None` and every money figure is a
  rate-card estimate. `observe/cost.py` says so in its own docstring.

---

## Quick commands

```bash
# `grapharc` is NOT on PATH — it lives only at .venv/bin/grapharc. Either
# activate the venv (`source .venv/bin/activate`) or use the full path, as
# below. Bare `python` is likewise not the venv interpreter.

.venv/bin/python -m pytest              # bare, no extra -q; see the traps above
.venv/bin/python -m ruff check .
uv build                                 # wheel + sdist

.venv/bin/grapharc --help                # run agent serve models replay diff trace metrics viz
.venv/bin/grapharc models                # which backends are configured (redacts the key)

# Run one cookbook page's tests after editing it — five of six extract and
# byte-compare, so a changed output block fails until you paste the real one:
.venv/bin/python -m pytest tests/test_cookbook_governance.py

# These two cost money. Read the live-test trap above first.
.venv/bin/grapharc run capstone --model openrouter/anthropic/claude-haiku-4.5 \
                                --reviewer-model openrouter/openai/gpt-4o-mini
.venv/bin/python -m grapharc.examples.agent_fixit \
                                --model openrouter/anthropic/claude-haiku-4.5
```

If `.venv/` is missing or stale, rebuild it with `uv sync --all-extras --group
dev`. CI pins Python 3.12 in every job while this venv is 3.14, so a passing
local run is not proof CI passes — `.github/workflows/ci.yml` runs four jobs
(`lint`, `live-marker-guard`, `test` on 3.12/3.13/3.14, `build`).

The ten live tests are `test_gateway_openrouter.py` (7),
`test_gateway_gate.py` (1), `test_gateway.py` (1) and `test_v0_gate.py` (1) —
the last being a full agentic run, so it is the expensive one. List them
without running them:
`.venv/bin/python -m pytest -m live --collect-only -o addopts="" -q`.

Key docs: `ARCHITECTURE.md` §7 (**current** gap analysis, re-derived) ·
`ROADMAP.md` (numbered backlog) · `ASSESSMENT.md` (honest self-critique of an
earlier tree) · `VISION.md` (thesis) · `docs/cookbook/` (112 runnable recipes) ·
`CHANGELOG.md`.
