# GraphARC — live build list

Everything between today and a full-fledged general-purpose agent runtime, as
Status is measured, not aspirational.

**Legend:** `[x]` done · `[~]` partial · `[ ]` not started · **B** blocks other
work · **!** known-false claim shipping today

Overall: **~70% of the list below** (72 of 103 enumerated items). Re-derived on 2026-07-28 by executing every
claim against the tree rather than reading the commit log — the percentage is
the fraction of *enumerated items* verified done, section by section, which is
this project's own definition of scope and not the industry's.

At that point: `pytest` → **1,381 passed, 10 deselected** (the live ones);
`ruff check .` clean; the wheel builds and imports all 94 submodules in a clean
virtualenv. Treat the test count as a snapshot rather than a fact about the
project — `pytest` re-derives it in one command, which is the only reason it is
quoted at all.

**The number used to be generous in one specific way, and mostly is not any
more.** Four subsystems were built, tested and reachable by nothing: `planner/`
had no command driving the governed loop, `policy/` had no caller and no bridge
from its TOML document to the `EdgePolicy` admission consults, the CLI handed
graphs the in-process memory store instead of the SQLite one, and no trace event
carried the provider's cost. All four are closed — `grapharc plan`,
`PolicyEngine.edge_policy()`, `grapharc demo --memory`, and `cost_usd` on the
`end` and `model` events. **One seam is left:** `server/` still ships its own
in-process session runtime instead of the durable `session/` one (§12.3). Code
that works and is unreachable scores as built here and is worth less than that
to a reader, which is what §12 exists to count.

---

## Next five things

In order.

1. **Put the HTTP API on the real session layer** (§12.3) — the last seam.
   `session/` is durable and resumes across processes; `server/` uses its own
   `InProcessRuntime` that does neither, and records approvals without
   delivering them. Two session layers, one seam.
2. **Let admission constrain arguments** (§5.6) — **!**. The gap most likely to
   be over-read: a rule reaches a node's *kind* and never its `args`, so
   `args={"path": "/etc/passwd"}` is admitted on the strength of the kind.
   `Materializer` drops args by default, which makes the default safe and the
   opt-in sharp.
3. **Route the tool plane through the document** (§7.5 remainder) — the edge
   side now compiles to the admission gate, but nothing calls
   `permission_policy()`, so `grapharc agent` is still governed by Python
   objects rather than by the TOML file.
4. **~~Publish to PyPI~~** (§11.1) — done, `0.1.0` is live. Next is `0.1.1`,
   to correct the `__version__` the published wheel carries. Build only from a
   clean tree, and let the tag-driven `release.yml` do it rather than a local
   `uv build` — that is exactly how the drift got in.
5. **~~Decide the version.~~** Decided: `0.1.0`. A `1.0` would imply API
   stability that several days-old subsystems do not have.

---

## 0. Correctness debt

Ship-blockers. Every item verified by running code. **Section clear.**

- [x] **0.1 — `ctypes` escape closed**, along with three more found during the
      audit: `sqlite3.connect` (opens files in C, raising no `open` event),
      `_posixsubprocess.fork_exec` (the C entry point under `subprocess`), and
      compiled-extension imports from outside the runtime paths.
- [x] **0.2 — Child environment scrubbed** to an allowlist, so a secret nobody
      thought to name cannot leak just by being new.
- [x] **0.1b — Runtime paths are read-only.** Re-verified by running it: a
      sandboxed tool writing `evil.pth` into `site-packages` raises
      `SandboxViolation`, no file is created, and stdlib *reads* still succeed
      so imports keep working. Reads and mutations use separate grants.
- [x] **0.3 — `max_seconds` interrupts a running node** (SIGALRM on the main
      thread, async-exception injection elsewhere) and re-arms, so a node that
      swallows one interrupt does not run free. Residual limits documented.
- [x] **0.4 — Tokens charge automatically**, via a usage callback that meters
      every model call inside a node, deduplicated by call identity rather than
      by token count. `max_tokens` is enforced at `on_llm_end`, so overspend is
      bounded by the one call that crosses the line rather than discovered a
      node later.
- [x] **0.5 — Types validated at write time.** Remaining gap stated precisely
      rather than papered over: annotation-carried constraints bite, but a state
      model's own `@field_validator` does not run at write time.
- [x] **0.6 — README claims corrected**, each disproof re-run against the tree.
- [x] **0.7 — Neo4j fiction removed**; a real `ClaimStore` protocol now exists.
- [x] **0.8 — `pytest` defaults fixed.** `addopts` carries `-m 'not live'`;
      live tests are opt-in via `pytest -m live`. Verified: a plain run
      deselects 10.
- [x] **0.9 — `LICENSE` copyright and README clone URL corrected.** The URL now
      resolves; what it resolves *to* is §11.7's problem, not this one.
- [x] **0.10 — Gateway tempfile leak fixed.** The scratch directory is an
      `ExitStack`-managed `TemporaryDirectory` that outlives every retry attempt
      and dies with the call, with `ignore_cleanup_errors=True` so an
      undeletable directory cannot fail a call that already cost money.

---

## 1. Graph kernel — `[~] ~85%`

No longer amputates LangGraph. Async, `Command` returns and state access all
came through, which is what unblocked the HTTP API and the session layer.

- [x] Typed state, per-node declared writes, deep-copy isolation
- [x] Budgets (iterations/tokens/seconds/concurrency), convergence guards, DAG
      mode, cycle detection
- [x] JSONL traces, checkpoint resume, fail-closed run context
- [x] Bounded fan-out with worker isolation and dedup
- [x] **1.1 — Async: `ainvoke` / `astream` / `astream_events`.** All three run,
      `async def` nodes execute, and the *sync* entry points now refuse a graph
      containing them with `AsyncNodeError` before anything runs — rather than
      the misleading `WritePermissionError` this line used to describe.
      `astream_events` offers `v1`/`v2`; `v3` is refused because LangGraph
      returns a stream object there, a different shape than the method's
      contract.
- [x] **1.2 — `Command` returns accepted.** A node may return
      `Command(goto=…, update=…)`; the `update` goes through the same write
      allowlist as a dict, and the `goto` is validated against the compiled
      graph at the node boundary. A `Command` passed as *input* to
      `invoke()`/`stream()` remains unsupported, deliberately: its `update`
      would reach state unchecked.
- [x] **1.3 — `get_state` / `update_state` / `get_state_history` passthrough**,
      plus the three `a*` twins. `update_state` is not a bare passthrough: it
      rejects unknown state fields, type-checks the values, and applies the
      node's declared write allowlist when called with `as_node=`. The residual
      gap is stated in the docstring — with `as_node=None` LangGraph attributes
      the update to whichever node last ran, and GraphARC does not reproduce
      that inference, so such an update is type-checked but not allowlisted.
- [x] **1.7 — `TraceRecorder.thread_summary` is incremental.** It folds only the
      bytes appended since the last call and keys its index on file length, so a
      long-lived thread costs O(events) over its life rather than O(events²),
      and lines written by another process are picked up the same way.
- [~] **1.5 — `interrupt()` half-works.** LangGraph's `interrupt()` suspends the
      graph and the `Interrupt` appears on `get_state(...)`, so the *suspend*
      side is real. There is no supported way to **resume**: `invoke()` takes a
      dict, a state model or `None`, and `.inner.invoke(Command(resume=…))`
      fails closed with `MissingRunContextError` by design. It also still writes
      a spurious `phase="error"` trace line carrying the `GraphInterrupt`. Use
      the session layer's approval gate (§6.5) for human-in-the-loop today.
- [ ] **1.4 — Passthrough `retry_policy`, `cache_policy`, `durability`,
      subgraphs.** `add_node` takes `writes` and `input_schema`; `compile` takes
      `checkpointer`. Nothing else reaches LangGraph.
- [ ] **1.6 — Offer a decorator form** so discipline composes with LangGraph
      instead of replacing it.
- [ ] **1.8 — Make deep-copy opt-out-able** for large states. It is currently
      unconditional.

## 2. Model gateway — `[~] ~80%`

- [x] Claude Code CLI adapter (tools disabled, argv array, stdin prompt)
- [x] Correct cache-token accounting
- [x] **2.1 — `bind_tools`** — works on OpenRouter. Still `NotImplementedError`
      on the Claude-CLI backend, which is inherent to `claude -p`.
- [x] **2.2 — `with_structured_output`** — works on OpenRouter.
- [x] **2.3 — `_stream` and `_agenerate`** — both work on OpenRouter.
- [x] **2.5 — Retries, backoff, rate-limit handling.** `RetryPolicy`
      (3 attempts, 0.5s initial, ×2, capped at 20s, 25% shrinking jitter) with
      an explicit transient/deterministic split: a 429 or 5xx or connection
      reset is retried, a 400/401/402/403 is raised on the first attempt, and
      anything unrecognised is treated as deterministic. `Retry-After` raises
      the delay but never lowers it and is itself capped. Streaming is not
      retried — once a chunk reaches the caller the request cannot be re-issued.
- [x] **2.6 — Routing rules** — provider `order` / `sort` / `max_price` /
      `require_parameters`, plus model-level `fallback_models` chains.
- [x] **2.7 — Enforced cost ceilings.** `SpendMeter` refuses before a call once
      the ceiling is reached (`ensure_headroom`) *and* raises after charging the
      call that crossed it, so overspend is bounded by one call. Where it stops:
      a call the provider does not price cannot be charged, and those land in
      `unpriced_calls` rather than being guessed at — `unpriced_calls > 0` means
      the ceiling saw less than the whole bill.
- [x] **2.9 — Backend registry** — `claude-cli`, `openrouter`, `openai`,
      `ollama`, `mock`; a mistyped backend is rejected rather than folded into a
      model name. `openai` is both a backend and an OpenRouter author slug, and
      the backend wins.
- [~] **2.4 — Provider adapters:** OpenRouter, Claude CLI, the OpenAI API and a
      local Ollama server, plus a `mock` backend for tests. The three
      OpenAI-wire backends share one base class, so tool-calling, streaming,
      retries and the usage envelope are identical across them. Direct
      Anthropic API and vLLM pending; the `api` extra exists with nothing
      importing `anthropic`. **What the new backends do not bring:** OpenAI
      returns no per-call price, so a dollar ceiling there counts calls unless
      a `price_per_million=` card is supplied, and Ollama's probe reports
      configuration rather than a running daemon.
      *No model count is quoted here any more.* This line used to say "~340
      models" while the CLI help said "~400" — nobody re-checks a number like
      that and it rots into a contradiction. The CLI settled on wording with no
      count in it and has a test enforcing that; these docs now match.
      `grapharc models --check` reports what this machine can actually reach.
- [ ] **2.8 — Prompt caching support** and per-run model pinning.

## 3. Tool plane — `[~] ~55%`

- [x] Registry, deny→ask→allow permissions, hooks, approval gates
- [x] Audit-hook executor: path confinement, network gating, spawn refusal,
      SIGKILL escalation
- [x] **3.1 — Wired to an agent.** `AgentNode` drives the registry, permissions
      and executor. `grapharc/examples/agent_fixit.py` is a shipped graph that
      calls tools, and `grapharc agent <task>` drives the core toolset.
- [x] **3.2 — Container executor.** `grapharc/harness/container.py` runs each
      tool call in a throwaway container: one bind mount (the workspace),
      `--network none` unless the tool declared `needs_network`, all
      capabilities dropped, `no-new-privileges`, non-root, read-only rootfs,
      memory and pid limits, and no host environment forwarded. Its constraints
      are real and enforced rather than documented away — the tool must be
      importable *inside the image* (a lambda, a `partial` or a bound method is
      refused before a container starts), and arguments and results must be
      JSON. Tests against a live runtime skip themselves when no runtime or
      image is present and never pull one; they were exercised against Docker
      with `python:3.12-slim` during this pass.
- [x] **3.3 — Core tools:** `read_file`, `write_file`, `edit_file`, `list_dir`,
      `glob`, `grep`, `run_command`. Every path argument is resolved and
      confined by *the tool*, independently of the executor — verified that both
      `../../../etc/passwd` and `/etc/passwd` raise `WorkspaceEscape`.
      `run_command` is the deliberate exception and is documented as one: it
      takes an argv **list** and never a shell string, but the child it spawns
      is an ordinary process with the caller's privileges and can read the whole
      filesystem. It also cannot run under `SandboxedExecutor` at all.
- [ ] **3.4 — Browser tool** and HTTP/network tool.
- [ ] **3.5 — MCP client** — the ecosystem standard for third-party tools. The
      `mcp` extra exists with nothing importing it.
- [ ] **3.6 — Progressive disclosure / tool search** for large tool sets.
      `ToolRegistry.visible()` filters by permission, which is a different
      thing.
- [ ] **3.7 — Idempotency keys** for side-effecting tools.
- [ ] **3.8 — Large-output offloading** (write to file, return a preview).
      `ToolLimits` truncates at 20,000 chars; nothing is offloaded.

## 4. Agent node — `[~] ~60%`

- [x] **4.1 — `AgentNode` built**: observe → model → tool request → permission
      check → sandboxed execute → repeat, budgeted and traced. A denied tool is
      fed back to the model rather than killing the run; malformed tool JSON is
      reported back instead of silently reading as success; stall detection keys
      on the tool *result*, so re-running a test suite is not mistaken for a
      loop. Only `target_met` fills the answer field — a run that stopped for
      any other reason keeps its last utterance in `partial_output` where
      nothing can mistake it for an answer.
- [ ] **4.2 — Context management** (compaction, just-in-time retrieval).
- [ ] **4.3 — Subagent spawning** with context isolation and summary-only return.
- [ ] **4.4 — Skills / instruction packs** loaded on demand.
- [ ] **4.5 — Per-node model and effort tiering.**

## 5. Planner & admission — `[~] ~85%` — *the crux, and it closed*

The component with no prior art to copy. It exists, and the cycle runs.

- [x] **5.1 — Planner node** that proposes rather than acts. `PlannerNode.propose()`
      returns a typed `Subgraph` and executes nothing. `Subgraph` and
      `ProposedNode` forbid extra fields, so a proposal has no channel for a
      callable at all — verified, a `body=` or `fn=` key is a `ValidationError`.
- [x] **B 5.2 — Admission checker.** Five checks — registry, policy, budget,
      depth, acyclicity — all of which run on every proposal so a planner gets
      the complete list rather than the first complaint. Nothing executes during
      a check: `NodeSpec.factory` is never called and the budget meter is read,
      not written. Every decision keys on the registry `kind`, never on the
      instance `name`, so renaming a denied kind does not launder it — verified
      by proposing `ProposedNode(name="harmless_helper", kind="deploy")` against
      a rule denying `deploy` and watching it refused. Worst-case cost is summed
      from the registry's own numbers, so a planner cannot buy admission by
      claiming to be cheap.
- [x] **5.3 — Rejections are first-class traced events.** Every decision,
      admitted and rejected alike, writes a `phase="admission"` line carrying
      the status, the proposal fingerprint, the checks run and the failed codes.
      The phase is deliberately not `"end"`, so admission decisions cannot
      inflate the node-execution counts `observe.metrics` reports.
- [x] **5.4 — Replanning with loop protection.** `GovernedLoop.run()` carries a
      goal from first proposal to recorded stop. `AdmissionResult.feedback()` —
      the per-check list with codes and remedies — becomes the planner's next
      feedback, and the loop never trims an over-large proposal, drops a denied
      edge, or retries an identical one. Nine stop reasons, including
      `no_progress`, `max_rounds`, `admission_refused` and `planning_failed`;
      the loop cannot fall out of the bottom.
      *Observed end to end:* a scripted planner whose first proposal named a
      policy-denied `deploy` node → round 1 rejected `policy/edge_denied`, round
      2 admitted and executed, round 3 admitted and executed, stop `goal_met`,
      with 3 admission + 3 round + 2 node + 1 stop events under one `run_id`.
- [x] **5.4b — Materialisation binds to the authorisation.**
      `Materializer.materialize(admitted, proposal)` takes the `AdmissionResult`
      first and matches it to the proposal by fingerprint: a result that
      authorised something else raises `NotAdmitted`, and so does a rejected
      one. There is no overload taking a bare `Subgraph`. The graph is built
      through `GraphARC.add_node`/`add_edge`/`compile`, so declared writes,
      typed state, budgets and traces all apply unchanged, and a body returning
      `Command(goto=…)` is confined to the admitted edge set with
      `UnadmittedTransition`.
- [ ] **5.5 — Decomposition strategies** (map-reduce, specialist fan-out) as
      reusable planner presets.
- [ ] **! 5.6 — Admission cannot constrain arguments.** Stated plainly because
      it is the gap most likely to be over-read: no rule reaches
      `ProposedNode.args`, so a proposal carrying `args={"path": "/etc/passwd"}`
      is admitted on the strength of its kind. `Materializer` drops args by
      default; `forward_args=True` hands the raw dict to a factory with nothing
      having checked it. Admission authorises the verb, not the object.

## 6. Session runtime — `[~] ~85%`

- [x] **6.1 — Long-lived sessions** with a status lifecycle in a `SessionStore`.
- [x] **6.2 — Resume across process restart.** Verified by running it: one
      interpreter created a session, ran `ingest` and `plan`, and stopped
      `awaiting_approval` holding `apply`; a second interpreter resumed by id,
      saw the hold, approved it, and ran `apply` and `report`. The append-only
      log shows each node exactly once — nothing repeated, nothing skipped. The
      resuming process must register the graph in its own `GraphRegistry`, or
      it gets `UnknownGraphError` rather than a guess.
- [x] **6.3 — Interrupt and steering** at superstep boundaries.
- [x] **6.4 — Event queue** for multi-turn input, durable in the store.
- [x] **6.5 — Human approval as a suspending graph node.** Gated nodes are
      passed to LangGraph as `interrupt_before`, so the graph stops *before* the
      gated node runs. Every gated node on a superstep boundary is held
      separately with its own request id, and the graph does not move while any
      is unanswered — a signature on `send_email` is not a signature on
      `delete_records`.
- [x] **6.6 — Concurrent sessions** with isolation, each on its own thread.
- [ ] **6.7 — Async turns.** `run()` is synchronous and occupies its caller
      until the session stops. The kernel grew `astream` while this was being
      written; an async turn is buildable and simply not built.
- [ ] **6.8 — A real runner lease.** `SessionStore.transition` stops a second
      runner from claiming a session, and nothing reclaims one whose runner died
      holding it. That is a claim, not a lease.

## 7. Policy engine — `[~] ~75% built, 0% wired`

Everything here works and nothing calls it.

- [x] **7.1 — Declarative policy config** over nodes, edges, tools and spend.
      TOML in, `PolicyEngine` out; a commented example ships at
      `grapharc/policy/example.toml`. Evaluation is tiered rather than
      positional — every `deny` before every `ask` before every `allow` — so a
      broad deny beats a narrow allow, including one scoped to a single tenant.
      Verified: `delete_*` denied for a tenant that `write_*` is allowed for,
      and an undeclared tenant denied outright.
- [x] **7.2 — Approval routing.** `ask` rules carry a required `approver_role`;
      `engine.approval_router(handlers, tenant=…)` produces the callback a
      `Harness` already obeys, and `engine.permission_policy(tenant=…)` produces
      a real `PermissionPolicy`.
- [x] **7.3 — Policy versioning and decision audit.** Every decision lands in a
      JSONL record naming the resource, subject, tenant, effect, the rule id and
      reason that produced it, the policy version, and a digest of the document
      — so a decision can be tied to the exact policy text that made it.
- [x] **7.4 — Multi-tenant scoping.** A declared tenant list makes a rule scoped
      to an unknown tenant a load error and a request naming one a recorded
      denial.
- [x] **7.5 — The document reaches the gate.** `edge_policy(tenant=…)`
      compiles `edge` rules into the `EdgePolicy` `AdmissionChecker` consults
      and `node_policy(tenant=…)` compiles `node` rules into the `NodePolicy`
      beside it — the node half reached nothing at all until issue #66, so a
      `deny` rule over a kind was text and the kind still ran —
      and `grapharc plan --policy` is a shipped caller, so this package is no
      longer imported by nothing. What the compiled object still cannot carry is
      what `permission_policy()` cannot either: the approver role and the audit
      record, because `EdgePolicy.decide` returns a bare `Decision`. Admission
      treats `ask` as not-yet-permitted. Still open: no call from `AgentNode` or
      `grapharc agent` to `permission_policy()`, so the tool plane is still
      governed by Python objects rather than by the document.

## 8. Memory & artifacts — `[~] ~85%`

- [x] Claims with provenance; supersession instead of overwrite
- [x] Unicode-safe entity normalization
- [x] **8.1 — `SQLiteMemoryStore`**, same `ClaimStore` protocol, verified durable
      across genuinely separate processes.
- [x] **8.2 — Artifact storage.** `SQLiteArtifactStore` points at the same file
      as the claim store. Append-only with versions rather than overwrites,
      mandatory provenance, and content-addressed blobs written before the row
      that references them — so a crash leaves an unreferenced blob (garbage)
      and never a row pointing at content that does not exist (a lie). `name`
      is metadata and never used to build a path.
- [x] **8.3 — Real retrieval.** Okapi BM25F over subject+predicate+object with
      the subject weighted highest, an optional injected vector channel that
      stays silent below a similarity floor, and graph traversal that reads a
      claim's object as an entity so a question about A reaches facts about B,
      each hop decaying the inherited score. Every result list is sorted by a
      total order so both backends rank identically. Nothing here is sublinear
      and nothing here pretends to be.
- [x] **8.4 — Automatic contradiction detection.** `detect_contradictions` and
      `add_and_detect` flag a new claim that shares a normalized
      (subject, predicate) with a stored one and differs in object. Structural,
      not semantic: it will not relate "is fast" to "is slow", will not match a
      rephrased object, and *will* flag a legitimately multi-valued predicate.
      It therefore reports and never resolves — auto-superseding would delete
      half a multi-valued fact inside the one subsystem whose promise is that
      facts are never destroyed.
- [x] **8.5 — Token-budgeted context rendering.** `render_context` takes
      `max_tokens`, `max_dead_ends` and a `count_tokens` callable; the dead-end
      section is no longer uncapped.
- [ ] **8.6 — Per-tenant/user memory scoping.**
- [x] **8.7 — Hand a durable store to the shipped graphs.** `grapharc run
      --memory PATH` gives `stage6` and `capstone` the `SQLiteMemoryStore`; the
      in-process one remains the default so a plain run stays hermetic. The
      `memory` extra that named Neo4j with nothing importing it was removed;
      a Neo4j-backed store would bring its own extra with it.

## 9. Triggers & surfaces — `[~] ~55%`

- [x] CLI (`run` / `trace` / `metrics` / `viz`)
- [x] **9.1 — HTTP API.** Seven routes on FastAPI: create, list, get, post an
      event, SSE stream, NDJSON trace, healthz. Verified end to end — create →
      poll → `succeeded`, `text/event-stream` frames in `trace` / `status` /
      `done` order, the NDJSON trace being the same record the stream carried,
      404 on an unknown session or graph and 422 on input that fails the
      graph's state schema. Behind the `server` extra; importing the rest of
      GraphARC does not import FastAPI.
- [x] **9.2 — Real CLI: ten commands**, `run` / `plan` / `agent` / `serve` / `models` /
      `replay` / `diff` / `trace` / `metrics` / `viz`, every one of them with
      `--json`. In JSON mode the failure is the document rather than a line on
      stderr, and exit codes are part of the interface: `0` did the job, `1` ran
      and the answer was negative, `2` could not run at all.
- [x] **9.6 — Streaming output to clients** via SSE, with a `last-event-id`
      cursor so a reconnect skips what it already saw.
- [ ] **9.3 — Cron schedules** and **9.4 — webhook triggers.**
- [ ] **9.5 — Chat channels** (Slack / Discord).

## 10. Operations — `[~] ~60%`

- [x] JSONL traces, metrics summaries, Mermaid path rendering
- [x] **10.1 — Replay a run from its trace.** A *reconstruction*, not a
      re-execution: `replay()` rebuilds the node sequence, the folded state, the
      timing and the failures off the JSONL, and calls no model, tool or node.
      Two limits inherited from the recording side and stated in the signature —
      strings past 2,000 chars were truncated at write time, and the trace does
      not record which fields have reducers, so a reduced field replays
      last-write-wins unless the caller supplies the reducer. `diff_runs` /
      `diff_trace` align two runs and report where they diverged.
- [x] **10.3 — OpenTelemetry export.** One root span per run, one child per node
      execution, `AgentNode` sub-steps parented by inference (and to the run
      span rather than to a guess when the parent cannot be identified). The
      dependency is confined to `OTelSpanExporter` behind a Protocol, so
      importing the module needs no OTel. Previously documented as unverified
      against the real SDK — verified during this pass against
      `opentelemetry-sdk` 1.44.0, with spans reaching an `InMemorySpanExporter`.
- [~] **10.4 — Cost attribution** per run, thread (session) and node. The
      price is now recorded, not guessed: every gateway publishes a
      `cost_usd` through the same `llm_output` envelope, the runtime's usage
      callback accumulates it per node and writes it onto the `end` event, and
      an `AgentNode` writes the per-call figure onto each `model` event. A
      backend that reports no price still falls back to a `RateCard` estimate,
      and `recorded_cost_usd` and `estimated_cost_usd` never mix — a recorded
      figure wins outright rather than being averaged with a guess. Tokens are
      counted from the same events `metrics.summarize` uses — node `end` events
      *plus* work outside any node span, which is what a `grapharc agent` run
      consists of entirely — and the suite asserts the two agree. **One gap
      left:** there is no tenant on a trace event, so tenant attribution is not
      offered rather than being approximated.
- [ ] **10.2 — Rollback** and versioned graph/prompt configs.
- [ ] **10.5 — Alerting** on budget, failure, and verifier-drift.

## 11. Product & distribution — `[~] ~35%`

- [x] Builds a clean wheel; **1,534 tests**; CI; ruff clean. Verified in a fresh
      virtualenv: a bare wheel install imports most of the 103 submodules and runs
      `grapharc demo stage0` — `gateway.openrouter` and the whole `server`
      package need their extras — and installing `[all]` imports all 93.
- [x] **11.6 — Classifiers, `[project.urls]`, contribution guide.** Every URL
      names a file that exists, and the `all` extra is self-referential so it
      cannot drift out of sync with the others.
- [~] **11.3 — Live-model examples** behind the `live` marker: 10 tests,
      deselected by default. CI wiring still pending (it needs a key in secrets).
- [~] **11.2 — Docs site.** A cookbook is landing under `docs/cookbook/`; no
      published site.
- [x] **11.7 — The source is on the public remote.** `git clone
      https://github.com/CodeGraphContext/GraphARC && uv sync --group dev` works;
      verified by cloning into a scratch directory and finding `pyproject.toml`
      and the `grapharc/` package. This was the single most consequential false
      claim in the tree for most of the project's life, because it was the first
      one a reader hit.

- [x] **11.1 — Publish to PyPI.** `0.1.0` is live; `pip install grapharc`
      verified in a clean virtualenv through to `grapharc demo stage0`. One
      defect shipped with it: the wheel's module carries `__version__ =
      "0.1.0a0"` while its metadata says `0.1.0`. PyPI is immutable, so this
      is corrected by `0.1.1`, not by a re-upload.
- [ ] **11.2 — Ship `0.1.1`** to correct the `__version__` above. Build only
      from a clean tree: `git status` empty, then `uv build`.
- [ ] **11.4 — Benchmarks, including published losses.**
- [ ] **11.5 — External security review** (the audit-hook sandbox is defense in
      depth; §3.2's container executor is the boundary to review).

## 12. Seams — `[~] ~80%`

Each item is two working subsystems that do not know about each other; none of
them is research, and all of them are worth more than another feature. Four of
the five are closed.

- [x] **12.1 — A surface for the governed loop.** `grapharc plan <goal>` drives
      propose → admit → materialise → execute → replan and prints every round,
      its admission status and its rejection codes. Scripted by default, so it
      costs nothing and is deterministic; `--model SPEC` swaps in a real
      backend, `--registry module:attr` swaps in your own kinds (and, via
      `STATE_SCHEMA` and `WRITES` on the same module, the schema they write to).
      The shipped demo registers `deploy` and denies every edge into it, so the
      default run shows round 1 refused on `edge_denied` and round 2 replanning
      without it. `grapharc/examples/plan_incident.py` is the registry;
      `tests/test_cli.py` pins the rounds, the stop reason and the exit codes.
- [x] **12.2 — `PolicyDocument` → `EdgePolicy`.** `PolicyEngine.edge_policy(
      tenant=…)` compiles the document's `edge` rules into the object
      `AdmissionChecker` consults, mirroring `permission_policy()` for the
      planner side; a test pins it to `check_edge` across an edge × tenant
      matrix, and an undeclared tenant compiles to a policy that permits
      nothing. `grapharc plan --policy PATH --tenant NAME` is the shipped
      caller, and the end-to-end test is the one that matters: with a
      `*->deploy` deny rule in the file round 1 is refused, and with the rule
      removed the same run admits it. The document constrains the run.
- [ ] **12.3 — The HTTP API on the real session layer.** `create_app(runtime=…)`
      already takes any `SessionRuntime`; what is missing is the implementation
      backed by `grapharc.session`, which would give the API durable sessions,
      cross-process resume, and approval events that are delivered rather than
      merely recorded.
- [x] **12.4 — A durable store for the shipped graphs.** `grapharc run
      --memory PATH` hands `stage6` and `capstone` a `SQLiteMemoryStore`.
      In-process stays the default, so a run still writes nothing nobody asked
      for. Proved across a real process boundary: two interpreters, one file,
      and the second run recalls what the first persisted.
- [x] **12.5 — `cost_usd` onto trace events.** See §10.4.

---

## Milestones

| | Scope | Gate: a real task against a real model | Status |
|---|---|---|---|
| **V0** | §0 + §2.1 + §4.1 | An agent edits a file and runs tests, permission-gated and budgeted | **passed** |
| **V1** | §5 + §1.1–1.2 | "Refactor this repo and run tests" plans its own fan-out; an over-budget plan is rejected with a recorded reason | **mechanism done, gate not run** — the loop, the rejection and the recorded reason all work on scripted planners; no live-model run of the real task has been recorded |
| **V2** | §6 + §8.1 | A session survives restart; a human approves a destructive action mid-run | **mechanism done, gate not run** — verified across two processes with a scripted model |
| **V3** | §7 + §9 | Incident response runs from a webhook, remediation gated on approval | **blocked on §12.2 and §9.4** — policy and the API exist, the webhook and the wiring do not |
| **V4** | §10 + §11 | Replay any production run; a stranger `pip install`s it | **replay works; nobody can install it** — see §11.7 |

**Read the V1 and V2 rows carefully.** Both say *mechanism done, gate not run*,
and that distinction is the entire point of defining gates this way. The code
does the thing; nobody has yet pointed a real model at the real task and
recorded the result. Marking them passed on the strength of a green suite is
precisely the failure mode below.

**The failure mode to avoid** — this repo already hit it once — is writing the
essay before the code and marking milestones done because tests pass rather
than because a real task ran. It has a second form, which this page is now
guarding against: marking a subsystem done because it is built, when nothing
calls it. §12 exists so that gap has somewhere to be counted.
