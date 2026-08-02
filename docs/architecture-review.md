# GraphARC Architecture Review

*Compiled 2026-08-01 from a three-track code audit (execution core; observability & delivery
surfaces; planner/gateway/CLI/packaging) plus issues observed first-hand while operating the
Slack bot, the live view, and local/delegated/API agent runs. Version audited: 0.1.1.*

A recurring meta-observation up front: this codebase is **unusually honest with itself** —
most limitations below are already stated in a docstring somewhere. The issues that matter
are the ones where a *documented* limitation is load-bearing for a design decision the doc
does not follow through on. The strengths section at the end lists what should survive any
refactor.

---

## Executive summary — the ten highest-leverage issues

| # | Issue | Layer | Severity |
|---|---|---|---|
| 1 | Two parallel session subsystems: the durable one (`session/`) is unreachable, the exposed one (`server/runtime.py`) can't persist, evict, or gate approvals | server/session | HIGH |
| 2 | The sessions API has **no authentication**, and `/sessions/{id}/trace` serves the raw `state_delta` the live view deliberately redacts | server | HIGH |
| 3 | `Subgraph.fingerprint()` hashes a random `proposal_id` — `--check-only` prints a different fingerprint for the same file every run, so "the plan you reviewed is the plan that runs" cannot be enforced | planner | HIGH |
| 4 | Budgets are per-invoke, in-process objects: resume mints a fresh budget, nested runs escape the outer meter mid-flight (`_charge_back` is a one-call-site patch), nothing survives a process boundary | runtime | HIGH |
| 5 | Slack: subprocess-per-command × 5 bolt worker threads × 120s timeout = ack starvation and a Slack-retry amplification loop at just 5 concurrent commands | slack | HIGH |
| 6 | No cross-process append coordination on trace files; torn lines are silently dropped by readers (no counter, no log) — the audit trail can lose events without saying so | observe | HIGH |
| 7 | Fork-per-tool-call sandbox: `fork` from a multithreaded parent (deprecated in 3.14), a swallowed `setsid` failure that can make the timeout path `killpg` **the host process**, no Windows/macOS path | harness | HIGH |
| 8 | Kernel imports LangGraph *private* internals (`langgraph._internal._constants`, `.nodes[n].flat_writers`) while pinning `langgraph>=0.4` across a 0.x→1.x major boundary | runtime/session | HIGH |
| 9 | `grapharc plan`/`run` hard-import the incident **demo** for their loop factory and default state — the goal check is frozen at `len(state.notes) >= 3` and third-party registries cannot replace it (silent 8-round burn) | planner/cli | HIGH |
| 10 | The delegated executor (`--executor claude-cli`) accepts `--max-tokens` and silently never applies it; `--allow` means different things per executor; its subprocess timeout kills only the direct child | cli/harness | HIGH |

---

## 1. Execution core (`runtime/`, `harness/`)

### 1.1 Budget enforcement is per-invoke and non-compositional — HIGH
The meter is built inside `_run_config` and passed as a live object (`runtime/graph.py:762-771`);
"limits bound one attempt, not the lifetime of a thread" (`graph.py:806-808`). Consequences:
resuming a thread re-mints the full budget (no per-thread ledger); any node driving an inner
`CompiledGraphARC` swaps the global `charging()` contextvar (`runtime/usage.py:123-151`), so
the outer `max_tokens` is unenforced until the inner run returns — the planner hand-rolls
`_charge_back` (`planner/loop.py:632-645`) for exactly one call site and every other
composition loses the spend. The `RunContext` (holding a `threading.Lock`) also cannot cross
a process; today it survives only because LangGraph happens to drop non-scalar
`configurable` values from checkpoint metadata — an upstream behavior nobody pinned.

### 1.2 Check-then-charge is not atomic — MEDIUM
`ctx.meter.check()` then `charge_iteration()` as two separate lock acquisitions
(`graph.py:616-621`; same shape for tokens in `usage.py:117-120`). Under `max_concurrency=8`,
all eight workers can pass `check()` before any charge lands — every limit overshoots by the
concurrency degree, in precisely the regime limits exist for. Fix: one `check_and_charge()`.

### 1.3 Fork-per-tool-call sandbox — HIGH
`harness/executor.py:564-583`. Every tool call forks a fully-loaded interpreter, chdirs,
scrubs env, installs audit hooks, tears down. 24 forks for a modest agent run; sandbox
overhead dominates cheap tools. Sharper problems: `get_context("fork")` from a multithreaded
parent (LangGraph thread pools; Python 3.14 deprecates this) risks child deadlock and has no
Windows/macOS fallback; `os.setsid()` failure is swallowed (`executor.py:528-531`) after
which the timeout path's `os.killpg` (`executor.py:559`) targets the **parent's** process
group — SIGKILLing GraphARC itself; no `try/finally` around `poll()` means a deadline
interrupt leaks the child; a child dying without sending makes `recv()` raise a raw
`EOFError` outside the error taxonomy.

### 1.4 Error taxonomy is executor-dependent — HIGH
The sandbox re-raises child failures as `RuntimeError(repr(exc))` (`executor.py:582`),
destroying exception type; `LocalExecutor` re-raises the real exception; the container
executor overloads `SandboxViolation` for *configuration* problems ("no docker on PATH"),
which `AgentNode` stamps `refused_by="sandbox"` — an audit trail where a missing Docker
install is indistinguishable from an active escape attempt. There is no `GraphARCError`
root across ~30 exception classes in five modules. The tool-argument hint added this session
(`agent.py:611-614`) has to string-match `"TypeError"` in a repr because the type identity
was destroyed at the pipe. Fix direction: a structured error envelope
(`{type, module, msg}`) across the process boundary, and a common exception root.

### 1.5 Kernel↔LangGraph coupling is unbounded — HIGH
`session/runtime.py:125-140` imports `CONFIG_KEY_READ`/`CONFIG_KEY_SEND`/`NO_WRITES` from
`langgraph._internal._constants` with hardcoded string fallbacks and hand-drives
`.nodes[node].flat_writers`; `pyproject.toml` pins `langgraph>=0.4` while 1.2.9 is
installed. The fallback protects against renames, not semantic changes — a Pregel change
makes rejections silently do the wrong thing. The dependency floor should be a narrow range,
and the private-symbol usage isolated behind one adapter module with a version check.

### 1.6 GraphARC graphs don't compose — HIGH/MEDIUM
Fail-closed `MissingRunContextError` (`graph.py:117-125`) makes a `CompiledGraphARC`
unusable as a subgraph of an outer LangGraph (the outer config lacks `grapharc_ctx`), and
`compile(checkpointer=None)` (`graph.py:377-380`) drops every other upstream option
(`interrupt_before/after`, `store`, `cache`, retry policies). The wrapper is currently a
ceiling on LangGraph rather than a layer over it; the planner's separate-top-level-invoke
workaround (see 1.1) exists because of this.

### 1.7 Deep-copy of full state per node execution — MEDIUM
`state.model_copy(deep=True)` on every entry (`graph.py:610-614`): O(state) × nodes ×
fan-out width on the critical path, 32 simultaneous copies under a 32-way fan-out, and an
undocumented "state can hold no live handle" restriction enforced by a `TypeError` from
`deepcopy`.

### 1.8 `AgentNode` monolith: quadratic context, serial tools, sync-only — MEDIUM
726 lines holding schema-gen, prompting, loop, budgets, tracing, rendering, stall detection.
`messages` only grows and is resent whole every iteration (no trimming/summarization seam);
multi-tool turns execute serially (each paying a fork); no async path — under `ainvoke` the
node lands on a worker thread where the deadline guard degrades to an async-exception that
cannot interrupt a blocking socket read, so `max_seconds` is weakest on the one node that
needs it most. Tool schemas are lossy (`list[str]` → `{"type":"array"}` with no `items`;
unions collapse to the first arm; no per-parameter descriptions) and args are never
validated against the signature *before* paying the fork (`inspect.Signature.bind` in
`Harness.call` would make wrong-argument failures free and executor-independent).

### 1.9 Smaller core items
- **Conditional-edge routers are unguarded** (`graph.py:336-348`): `Command`/`Send` are
  checked, a router returning an unknown key is a raw LangGraph `KeyError` — same author
  mistake, three exception types, only one of them GraphARC's.
- **Permissions decide on tool name only** (`permissions.py:42-47`): no argument-scoped
  rules (`write_file` is all-or-nothing), no runtime promotion of an ASK approval, and
  `ApprovalCallback` returns a bare bool — unusable for a real HITL UI.
- **`wants_ctx` arity heuristic** (`graph.py:680`): `def node(state, config)` receives a
  `RunContext` named `config`; defaulted kwargs count wrong.
- **Pre-hooks break on first decision, post-hooks all run** (`core.py:62-77`): a REWRITE
  hook silently short-circuits every security hook after it — ordering is load-bearing.
- **Trace file reopened per event under one process-wide lock** (`observe/trace.py:78-80`):
  the trace becomes a synchronization point inside the parallel execution it observes.

---

## 2. Observability (`observe/`) — the trace as sole source of truth

### 2.1 No cross-process append safety; silent torn-line drops — HIGH
`TraceRecorder.record` locks with a **threading** lock only; two *writers* are uncoordinated
and a >PIPE_BUF line (easy: `_MAX_VALUE_CHARS=2000` is per-value, a six-field delta exceeds
12KB) can tear. Readers then hide the loss: `TailRecorder.read_events` skips unparseable
lines with no counter; `_advance_index` calls bare `json.loads` and *raises* instead. For a
layer whose motto is "a number no one can find in the audit trail is a number the audit
trail can contradict", losing events invisibly is the sharpest self-contradiction.
Fix: `fcntl.flock` on append (or per-writer files + merge), and a visible `skipped_lines`.

### 2.2 Static topology is never serialized — MED-HIGH
`TraceEvent` records the graph as a *name*; `to_mermaid` chains executed events in file
order. Structurally inexpressible: branches not taken, fan-out as fan-out (three parallel
workers render as a chain the graph never had — `metrics.py:133`), nodes that never ran.
Everything downstream (viz, the live view) inherits this. Fix: one `phase:"graph"` event at
run start carrying nodes+edges — cheap, additive, ignored by old readers.

### 2.3 No trace schema version; phase set open on write, closed on read — MED-HIGH
No `v` field on `TraceEvent`; the additive-optional-fields strategy cannot express a
*semantic* change, and `replay` will reconstruct a future producer's file confidently and
wrongly. `NODE_PHASES` is a closed frozenset while the recorder accepts any phase — a future
`phase:"retry"` is silently folded into `sub_events`; a sub-step emitter using `"start"`
mints phantom node executions that inflate every derived number.

### 2.4 Everything is O(file), several things repeatedly — MED-HIGH
No incremental read path exists except `_advance_index` (private to `thread_summary`).
`build_snapshot` performs **five** full parses per file change per open stream
(`server/live.py:90-134`); the Slack `LiveTail._render` does two full parses per tick over
a growing file — **O(n²)** over a run's life; `cost.by_node` is O(runs × file);
`trace_text` loads the whole file into one string. One cursor-based
`read_events(since_bytes=...)` plus an advanceable `ReplayedRun` fixes all four.

### 2.5 `_attach` misattributes under same-name fan-out — MEDIUM
Contrary to its own "abstains rather than guesses" contract, with several open executions
of the *same* node name it charges all sub-events to the last-opened one
(`replay.py:217-219`) — per-node cost attribution and OTel span parenting are wrong exactly
under fan-out. Attribution also silently depends on `AgentNode.name` matching the graph
node name. Fix: abstain (return None) on ambiguous prefix match — the orphan bucket exists.

### 2.6 Three independent implementations of run totals — MED-LOW
`metrics.summarize`, `ReplayedRun.tokens/node_ms`, and `cost._price_run` each re-derive
totals with already-divergent rounding. Compute once on `ReplayedRun`; consume everywhere.

---

## 3. Delivery surfaces (`server/`, `slack/`, `session/`)

### 3.1 Two parallel session subsystems — HIGH (highest-leverage single item)
`grapharc/session/` (~1,600 lines: SQLite CAS transitions, durable event queue, approval
gates, cross-process resume) implements exactly what `server/runtime.py:37-45` lists as its
own missing features — yet `SessionManager` is referenced nowhere outside its package: no
CLI subcommand, no HTTP route. Meanwhile the exposed `InProcessRuntime` cannot persist,
evict, or deliver approvals, and the two `GraphRegistry` classes are incompatible (one takes
a recorder, the other a checkpointer), so migration is re-registration, not adapting. The
`SessionRuntime` protocol and wire models have no approval concept at all, which falsifies
`app.py`'s claim that swapping in the real session layer "touches nothing in this module".

### 3.2 No auth on the sessions API; redaction is inconsistent within one app — HIGH
`create_app` accepts a token for `/live` only. `/sessions*` is wide open: anyone who can
reach the port can enumerate all sessions with full untruncated results, POST new runs
against the operator's credentials, and read the raw trace — `state_delta` included — via
`/sessions/{id}/trace`, undoing the leak-tested redaction the live view enforces. An
operator who tunnels the server for the Slack live view exposes all of it.

### 3.3 Slack ack-starvation amplifier — HIGH
Subprocess-per-command blocks a bolt listener thread for up to `timeout_seconds`; bolt's
default pool is 5 threads; `ack()` happens inside the listener. Five long runs → the sixth
request cannot ack within 3s → Slack marks it failed **and retries** → retries queue behind
the same five. The queue drains at one command per timeout while Slack re-enqueues. Also:
every `metrics`/`viz` read pays a full interpreter + langgraph import. Fix: a bounded queue
with an immediate "queued (N ahead)" ack, or move `run_command` off the listener thread.

### 3.4 Live-view index cost — HIGH
`scan_traces` re-reads and pydantic-validates **every byte of every trace under the root**
to extract run ids, synchronously on the event loop, and the index page meta-refreshes
every 5s per open tab. Coupled with `slack-runs/` never being pruned (retention is policy,
but retention cost and read cost are currently the same knob), this grows monotonically and
stalls every open SSE stream in the process. Fix: run ids in a sidecar index or filename;
listing must be `stat()`-only; move it off the loop.

### 3.5 One 429 permanently kills Slack live narration — MEDIUM
The 2.5s update interval is budgeted per-run; N concurrent runs in one channel multiply it
(3 runs ≈ 72 edits/min, over `chat.update`'s tier). `_ChannelSink.update` returns `False`
on *any* exception and `LiveTail` treats `False` as "sink dead, go quiet forever" — a
transient rate-limit ends narration for the rest of the run. Fix: distinguish retryable
from dead in the sink contract; a process-wide per-channel token bucket.

### 3.6 No requester identity, and nowhere to put one — MEDIUM
"Anyone in the workspace" shares one policy; Slack `user_id` is received and discarded;
`TraceEvent` has no actor/tenant field (`cost.py` declines tenant attribution for this
reason). After an incident the trace reconstructs *what* ran perfectly and cannot say *who
asked*. Fix: additive `actor` field on `TraceEvent`, threaded from the bot via env var.

### 3.7 Other surface items
- **SSE poll (0.02s) contends on the runtime's single global lock** with every session's
  superstep bookkeeping; 20 streams ≈ 1,000 lock acquisitions/s. A condition variable +
  `call_soon_threadsafe` is the standard answer; polling is a choice here, not a constraint.
- **`session.events` grows unboundedly** and the index-based SSE cursor makes eviction a
  breaking change; serving `events_since` from the trace file with a byte-offset cursor
  deletes the RAM copy and stabilizes resume across restarts.
- **The Slack gate duplicates CLI argument knowledge** (`CommandSpec` vs argparse) with no
  drift test; injected defaults (`--deny Bash`, `--max-seconds`) fail *silently* if the CLI
  renames things. A ~30-line test walking the argparse tree closes it.
- **`slack-runs/` and every tempdir trace root accumulate forever** (no rotation anywhere).

---

## 4. Planner, gateway, CLI, project structure

### 4.1 Fingerprint instability — HIGH
`Subgraph.proposal_id` is a fresh `uuid4` per validation and `fingerprint()` hashes the
full dump including it (`planner/proposal.py:190,251`): running `--check-only` twice on an
unchanged file prints two different fingerprints, though the CLI says "the fingerprint is
what a later run is compared against". No CI "reviewed plan == running plan" gate can be
built on it. Fix: exclude `proposal_id`/`origin` from the hash, or add
`content_fingerprint()`.

### 4.2 The demo is load-bearing production code — HIGH
`cli/plan.py:231` and `graphrun.py:154` unconditionally import
`examples.plan_incident.{build_loop, IncidentState}`. The goal check is frozen at
`len(state.notes) >= 3`; a registry whose state lacks `notes` silently burns all 8 rounds
and exits 1 having done the work. `cli/ → examples/` is an inverted dependency; the loop
factory and goal check must be part of the registry-module contract.

### 4.3 The registry-module contract is duck-typed with silent fallbacks — MEDIUM
Six `getattr` lookups, each falling back silently (missing `WRITES` → nodes run and write
nothing; misspelled `STATE_SCHEMA` → writes validated against the demo's schema). Four ways
to get a registry that loads, runs, and does nothing useful — all reported as success. A
validated `RegistryModule` protocol checked at load time converts each into a startup error.

### 4.4 Delegated executor governance gaps — HIGH
`--max-tokens` is accepted and silently unapplied under `--executor claude-cli`
(unbounded spend against the operator's subscription with no error); `--allow '*'` means
"everything" in one executor and "these seven Claude Code tools" in the other; the
wall-clock is `subprocess.run(timeout=...)`, which kills the direct child only — Claude
Code's own spawned shells can survive a timeout (the sandbox uses `setsid`/`killpg` for
exactly this reason). Observability is also coarse by design (start/end/stop only), which
the live view then inherits — mid-run the graph shows one open node. At minimum the
unsupported flags should be *refused*, not ignored.

### 4.5 Replan feedback is one round deep — HIGH (for planner efficacy)
`loop.py` clears feedback each round and rebuilds the planner's messages from scratch:
at round 3 the model cannot see that round 1 was rejected for the identical reason, so
`max_consecutive_rejections` can be exhausted by the same mistake repeated blind.
`RoundRecord` already carries the history; it just isn't fed back. (Observed live in this
session: a weak local model burned its round allowance exactly this way.)

### 4.6 Gateway backend seam doesn't exist — MED-HIGH
Adding a backend touches six hand-maintained lists across five files (`BACKENDS`,
`BACKEND_VENDOR`, the if-chain in `get_model`, the lazy-import table, the probe dict, the
CLI examples dict); `KNOWN_AUTHORS` is an 18-slug allowlist that rots with every new
provider. No entry-point group, no `Backend` protocol — out-of-tree backends require a
fork, which undercuts "the backend is a config change".

### 4.7 `.env` upward search is worked around, not fixed — HIGH
`gateway/config.find_env_file()` walks to `/`; the Slack config refuses to use it (issue
#20) — but the *subprocesses* the bot spawns still resolve OpenRouter keys through the
upward search from the bot's workdir. A `.env` in any ancestor directory is spendable by
any workspace member. Two config loaders in one package have opposite trust models; the
search scope belongs on the loader, not in comments at call sites.

### 4.8 CLI structure — MED-HIGH
`cli/main.py` is a 904-line parser+handler+fixture monolith (adding a flag touches ≥3
places; demo scripts are inline JSON string literals maintained in two files). No top-level
exception handler: an uncaught crash exits 1 — the code reserved for "ran, answer was
negative" — with empty stdout in `--json` mode, and the repo already contains two
one-at-a-time "this used to escape as a traceback" fixes. `--config` reaches only three
subcommands, so `grapharc.toml`'s `model` key is silently ignored by `agent` and `serve`.
Default traces land in unfindable tempdirs with no `grapharc runs` index — the Slack gate
had to build its own run-directory manager (`slack-runs/` + `--trace` injection) to work
around it, and any other integration pays the same cost. Three commands duplicate ~150
lines of run-setup; two divergent `resolve_registry` implementations raise different
exception types for the same user error.

### 4.9 Byte-pinned transcripts freeze output formatting as API — MED-HIGH
Cookbook/README tests byte-compare CLI output; label widths are commented "not up for
revision"; the incident demo's scripted reply sequence is effectively public API; the
version-bump literal `0.1.1` is asserted in a test, so releasing fails the suite until a
doc edit. The intent (docs that cannot rot) is right; there is no seam between "the claim
is still true" and "the bytes are identical", and no snapshot-regeneration tooling.

### 4.10 Admission checks are narrower than the prose — MEDIUM
`parent_depth` is caller-asserted and always 0 in-tree (the recursion bound is decorative);
node `args` ride through admission unchecked into factories; `worst_case` budget ignores
iteration count, so cyclic proposals under-estimate by the loop factor; `NodeRegistry`
mutability is opt-out (`freeze()`), and only the demo freezes.

---

## 5. Cross-cutting patterns

1. **Documented ≠ handled.** The codebase's best habit (stating limitations) repeatedly
   substitutes for closing them: the `.env` search has a warning comment instead of a scope
   parameter; the in-process runtime lists its missing features while the package that has
   them ships unreachable; the sandbox names its holes while the error taxonomy hides which
   ones fired.
2. **Duplicated knowledge with no drift detection.** Slack gate vs argparse; six backend
   lists; three run-total implementations; two `resolve_registry`s; two `GraphRegistry`
   classes; demo hints in two files. None has a test tying the copies together.
3. **Identity is missing at every layer.** Runs have no actor; proposals have no stable
   content hash; checkpoints have no topology fingerprint; trace lines have no schema
   version; graph structure has no serialized form. Each is a small additive field; their
   absence collectively caps the auditability story the project is named for.
4. **O(file) as the universal access pattern.** One incremental reader exists, private, and
   everything else re-parses — the cost lands exactly on the newest features (live views).
5. **The governed path and the convenient path diverge silently.** Delegated executor drops
   flags; `local` executor skips confinement; sub-runs escape budgets. Flags that cannot be
   honored should be refused loudly.

---

## 6. Strengths to preserve

- **Failure-mode honesty as a discipline.** `executor.py`'s five named sandbox holes,
  `budget.py`/`graph.py` on what deadline guards cannot interrupt, `cost.py` on what it
  refuses to attribute, docstrings recording the bug that motivated the code. Spot-checked
  claims held up. This is design rationale future maintainers cannot reconstruct.
- **Fail-closed defaults where they matter**: DENY-by-default policy, absent approval =
  denied, unclassifiable opens treated as writes, container refusing rather than degrading,
  `MissingRunContextError`, two-envelope parses raising rather than picking.
- **The write-allowlist + per-field `TypeAdapter` validation at node boundaries**
  (`graph.py:406-441`) — LangGraph's silent key-dropping becomes a loud contract violation
  at the point of authorship, at zero runtime cost.
- **The single-shaping-pass discipline in `BroadcastRecorder`** (SSE frame and JSONL line
  are the same dict; non-idempotent truncation identified and tested).
- **The propose/admit/materialize split is structurally enforced**, not asserted:
  proposals have no body-shaped field, the materializer re-checks the admission hash, and
  rejections are structured, machine-matchable data with remedies.
- **The Slack gate's admission posture** (allowlist, path confinement, double opt-in,
  executor tempering) is the right shape; its problems are drift risk and missing identity,
  not design.

---

## 7. Suggested fix order

**Now (small, high leverage):**
1. Content-stable `fingerprint()` (§4.1) — small change, unlocks the advertised CI gate.
2. Refuse unsupported flags on the delegated path instead of ignoring them (§4.4).
3. `check_and_charge()` atomicity (§1.2); router wrapping (§1.9); `_attach` abstention (§2.5).
4. Argparse↔gate drift test (§3.7); `skipped_lines` counter on readers (§2.1).
5. `actor` field on `TraceEvent` threaded from Slack (§3.6); `phase:"graph"` topology event (§2.2).
6. Top-level CLI exception handler honoring the exit-code/JSON contract (§4.8).

**Next (structural, medium effort):**
7. One incremental trace reader; retrofit live view, LiveTail, cost, trace_text (§2.4).
8. App-level auth on the whole server; align `/trace` with the live view's redaction (§3.2).
9. Bounded queue + off-listener execution in the Slack bot (§3.3); retryable-vs-dead sink
   contract (§3.5).
10. Registry-module `Protocol` validated at load; move the loop factory/goal check into it,
    breaking `cli → examples` (§4.2, §4.3).
11. Structured error envelope across executor boundaries + `GraphARCError` root (§1.4).

**Eventually (architectural):**
12. Unify the two session runtimes around `session/`'s durable core, redesign the wire
    contract to carry approvals, delete the in-memory event list in favor of file-cursor
    SSE (§3.1).
13. Hierarchical budget meters with a per-thread ledger (§1.1).
14. A sandbox execution service (pooled workers or the container path) replacing
    fork-per-call; structured envelopes come with it (§1.3).
15. An adapter module owning every LangGraph private-symbol touch, with a pinned narrow
    version range and a CI canary against upstream (§1.5, §1.6).

---

*Issues fixed during the session that produced this review: tool-call `TypeError`s now
append the tool's real signature (`harness/agent.py`); torn-line-safe `TailRecorder`
promoted to `observe/trace.py`; the live view no longer reports an open-but-quiet delegated
run as "idle"; the Slack final message keeps its diagram and run-page links.*
