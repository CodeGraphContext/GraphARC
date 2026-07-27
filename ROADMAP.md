# GraphARC — live build list

Everything between today and a full-fledged general-purpose agent runtime, as
described in [VISION.md](VISION.md). Status is measured, not aspirational —
see [ASSESSMENT.md](ASSESSMENT.md) for how the current numbers were verified.

**Legend:** `[x]` done · `[~]` partial · `[ ]` not started · **B** blocks other
work · **!** known-false claim shipping today

Overall: **~30% of the product**. Section 0 is now clear: every correctness
defect it listed was fixed and re-verified by running the original repro. The
`AgentNode` exists, memory has a disk, and the sandbox survived a second
adversarial pass that found a full escape the first one missed.

---

## Next five things

In order. Each unblocks more than it costs.

1. **Async through the kernel** (§1.1) — **B** blocks the HTTP API, concurrency,
   and streaming. Cannot be retrofitted cheaply later.
3. **Wire the harness into an example graph** (§3.1) — the tool plane is
   hardened and driveable, and still no shipped graph calls a tool.
4. **Admission checker** (§5.2) — the architectural crux and the only component
   with no prior art to copy.
5. **A container executor** (§3.2) — two adversarial passes have now found real
   escapes in the audit-hook sandbox. It is defense in depth, not a boundary.

---

## 0. Correctness debt

Ship-blockers. Every item verified by running code.

- [x] **0.1 — `ctypes` escape closed**, along with three more found during the
      audit: `sqlite3.connect` (opens files in C, raising no `open` event),
      `_posixsubprocess.fork_exec` (the C entry point under `subprocess`), and
      compiled-extension imports from outside the runtime paths.
- [x] **0.2 — Child environment scrubbed** to an allowlist, so a secret nobody
      thought to name cannot leak just by being new.
- [x] **0.1b — Runtime paths are now read-only.** Found by a reviewer *after*
      the first fix: site-packages was writable, so a tool could drop a `.pth`
      that executes arbitrary Python on every later interpreter start — a full
      escape outliving the run. Reads and mutations now use separate grants.
- [x] **0.3 — `max_seconds` interrupts a running node** (SIGALRM on the main
      thread, async-exception injection elsewhere) and re-arms, so a node that
      swallows one interrupt does not run free. Residual limits documented.
- [x] **0.4 — Tokens charge automatically**, via a usage callback that meters
      every model call inside a node, deduplicated by call identity rather than
      by token count. `max_tokens` is enforced at `on_llm_end`, so overspend is
      bounded by the one call that crosses the line rather than discovered a
      node later. **Proven on the same live command that exposed it:** the
      capstone went from `spent: 0 tokens` to `spent: 220 tokens`.
- [x] **0.5 — Types validated at write time.** Remaining gap stated precisely
      rather than papered over: annotation-carried constraints bite, but a state
      model's own `@field_validator` does not run at write time.
- [x] **0.6 — README claims corrected**, each disproof re-run against the tree.
- [x] **0.7 — Neo4j fiction removed**; a real `ClaimStore` protocol now exists,
      so "the same interface" names something.
- [x] **0.8 — Fix `pytest` defaults.** `addopts` now carries `-m 'not live'`;
      live tests are opt-in via `pytest -m live`.
- [x] **0.9 — `LICENSE` copyright and README clone URL corrected.**
- [ ] **0.10 — Fix the gateway tempfile leak.** One orphaned `mkdtemp` per call.

---

## 1. Graph kernel — `[~] ~70%`

The only mature subsystem. Its problem is what it *removes* from LangGraph.

- [x] Typed state, per-node declared writes, deep-copy isolation
- [x] Budgets (iterations), convergence guards, DAG mode, cycle detection
- [x] JSONL traces, checkpoint resume, fail-closed run context
- [x] Bounded fan-out with worker isolation and dedup
- [ ] **B 1.1 — Async: `ainvoke` / `astream` / `astream_events`.** None exist.
      An `async def` node fails with a misleading `WritePermissionError`.
      Blocks the HTTP API and all concurrency.
- [ ] **B 1.2 — Accept `Command` returns** — needed for dynamic `goto` routing
      and agent handoffs.
- [ ] **1.3 — Passthrough `get_state` / `update_state` / `get_state_history`** —
      needed for human-in-the-loop. The library's own tests reach into `.inner`
      today because there is no supported path.
- [ ] **1.4 — Passthrough `retry_policy`, `cache_policy`, `durability`,
      subgraphs.**
- [ ] **1.5 — Native `interrupt()` support** (currently emits a spurious
      `phase="error"` trace line).
- [ ] **1.6 — Offer a decorator form** so discipline composes with LangGraph
      instead of replacing it — keeps async, `Command`, and state access.
- [ ] **1.7 — Fix `TraceRecorder.thread_summary` O(n²)** — re-parses the whole
      trace file on every `invoke`.
- [ ] **1.8 — Make deep-copy opt-out-able** for large states.

## 2. Model gateway — `[~] ~60%`

Was text-only, which disqualified it for agent work. OpenRouter now brings
tool-calling, structured output, async, and streaming.

- [x] Claude Code CLI adapter (tools disabled, argv array, stdin prompt)
- [x] Correct cache-token accounting
- [x] **2.1 — `bind_tools`** — works on OpenRouter (verified live: a real
      `tool_calls` payload). Still `NotImplementedError` on the Claude-CLI
      backend, which is inherent to `claude -p`.
- [x] **2.2 — `with_structured_output`** — works on OpenRouter, verified live
      returning a parsed Pydantic model.
- [x] **2.3 — `_stream` and `_agenerate`** — both work on OpenRouter.
- [~] **2.4 — Provider adapters:** OpenRouter (~340 models across ~60 providers)
      and Claude CLI done. Direct Anthropic API and local (Ollama/vLLM) pending.
- [ ] **2.5 — Retries, backoff, rate-limit handling** — inherited from the
      OpenAI SDK on OpenRouter; nothing on the CLI backend.
- [x] **2.6 — Routing rules** — provider `order` / `sort` / `max_price` /
      `require_parameters`, plus model-level `fallback_models` chains.
- [ ] **2.7 — Enforced cost ceilings** — `cost_usd` is now captured per call by
      both backends, and still never budgeted against. See §0.4.
- [ ] **2.8 — Prompt caching support** and per-run model pinning.
- [x] **2.9 — Backend registry** — `get_model("openrouter/anthropic/…")`; a
      mistyped backend is rejected rather than folded into a model name.

## 3. Tool plane — `[~] ~45%`

Hardened and now driveable by `AgentNode`; still imported by zero *example* graphs.

- [x] Registry, deny→ask→allow permissions, hooks, approval gates
- [x] Audit-hook executor: path confinement, network gating, spawn refusal,
      SIGKILL escalation
- [x] **3.1 — Wired to an agent.** `AgentNode` drives the registry, permissions
      and executor; `grapharc/examples/agent_fixit.py` is a shipped graph that
      calls tools. Other example graphs still do not.
- [ ] **3.2 — Container executor** — a real kernel boundary. The README claims
      one "slots in behind this interface"; it does not exist.
- [ ] **3.3 — Core tools:** bash, read, write, edit, glob, grep.
- [ ] **3.4 — Browser tool** and HTTP/network tool.
- [ ] **3.5 — MCP client** — the ecosystem standard for third-party tools.
- [ ] **3.6 — Progressive disclosure / tool search** for large tool sets.
- [ ] **3.7 — Idempotency keys** for side-effecting tools.
- [ ] **3.8 — Large-output offloading** (write to file, return a preview).

## 4. Agent node — `[~] ~60%`

The missing unit that makes everything else compose.

- [x] **4.1 — `AgentNode` built**: observe → model → tool request → permission
      check → sandboxed execute → repeat, budgeted and traced. A denied tool is
      fed back to the model rather than killing the run; malformed tool JSON is
      reported back instead of silently reading as success; stall detection keys
      on the tool *result*, so re-running a test suite is not mistaken for a
      loop. **Gate passed:** a real model read a broken `calc.py`, wrote the
      fix, ran the suite, and stopped `target_met` — verified by an independent
      test run, with the denied tool never offered and 3,906 tokens metered.
      See `grapharc/examples/agent_fixit.py` and `pytest -m live`.
- [ ] **4.2 — Context management** (compaction, just-in-time retrieval).
- [ ] **4.3 — Subagent spawning** with context isolation and summary-only return.
- [ ] **4.4 — Skills / instruction packs** loaded on demand.
- [ ] **4.5 — Per-node model and effort tiering.**

## 5. Planner & admission — `[ ] 0%` — *the crux*

Where the vision lives or dies. No prior art to copy.

- [ ] **5.1 — Planner node** that *proposes* a subgraph rather than acting.
- [ ] **B 5.2 — Admission checker:** validate every proposed node against the
      registry, every edge against policy, worst-case cost against budget, and
      depth against limits — *before* any node executes.
- [ ] **5.3 — Rejections as first-class traced events**, never silent fallback.
- [ ] **5.4 — Replanning** on failure, with loop protection.
- [ ] **5.5 — Decomposition strategies** (map-reduce, specialist fan-out).

## 6. Session runtime — `[ ] 0%`

- [ ] **6.1 — Long-lived sessions** with a status lifecycle.
- [ ] **6.2 — Resume across process restart** (kernel checkpoints exist; the
      session layer does not).
- [ ] **6.3 — Interrupt and steering** mid-run at safe boundaries.
- [ ] **6.4 — Event queue** for multi-turn input.
- [ ] **6.5 — Human approval as a suspending graph node.**
- [ ] **6.6 — Concurrent sessions** with isolation.

## 7. Policy engine — `[ ] 0%`

- [ ] **7.1 — Declarative policy config** over nodes, edges, tools, spend.
- [ ] **7.2 — Approval routing** (who approves what, and how they are asked).
- [ ] **7.3 — Policy versioning** and decision audit.
- [ ] **7.4 — Multi-tenant scoping** with per-tenant budgets.

## 8. Memory & artifacts — `[~] ~45%`

- [x] Claims with provenance; supersession instead of overwrite
- [x] Unicode-safe entity normalization
- [x] **8.1 — `SQLiteMemoryStore`**, same `ClaimStore` protocol, proven durable
      across genuinely separate processes rather than merely across objects.
- [ ] **8.2 — Artifact storage** (files an agent produces).
- [ ] **8.3 — Real retrieval:** embeddings + hybrid search. Today it is a linear
      scan with exact string matching, labelled "GraphRAG".
- [ ] **8.4 — Automatic contradiction detection** — `supersede()` currently
      requires the caller to already know the old claim's ID.
- [ ] **8.5 — Token-budgeted context rendering** (the dead-end section is
      uncapped).
- [ ] **8.6 — Per-tenant/user memory scoping.**

## 9. Triggers & surfaces — `[~] ~5%`

- [x] Demo CLI (`run` / `trace` / `metrics` / `viz`)
- [ ] **9.1 — HTTP API** (depends on §1.1 async).
- [x] **9.2 — Real CLI:** `grapharc run <stage> --model <spec>
      [--reviewer-model <spec>]` runs the example graphs against real models,
      and `grapharc models` shows what a spec resolves to.
- [ ] **9.3 — Cron schedules** and **9.4 — webhook triggers.**
- [ ] **9.5 — Chat channels** (Slack / Discord).
- [ ] **9.6 — Streaming output** to clients.

## 10. Operations — `[~] ~20%`

- [x] JSONL traces, metrics summaries, Mermaid path rendering
- [ ] **10.1 — Replay a run from its trace.**
- [ ] **10.2 — Rollback** and versioned graph/prompt configs.
- [ ] **10.3 — OpenTelemetry export** (spans, nested LLM calls).
- [ ] **10.4 — Cost attribution** per tenant, session, and node.
- [ ] **10.5 — Alerting** on budget, failure, and verifier-drift.

## 11. Product & distribution — `[~] ~10%`

- [x] Builds a clean wheel; 101 tests; CI; ruff clean
- [ ] **11.1 — Publish to PyPI.** Not published; `git clone` is the only path.
- [ ] **11.2 — Docs site** with runnable examples.
- [~] **11.3 — Live-model examples** behind the `live` marker: stage 5, the
      capstone, and the V0 agent gate all run end-to-end against real models.
      CI wiring still pending (it needs a key in secrets).
- [ ] **11.4 — Benchmarks, including published losses.**
- [ ] **11.5 — External security review** (after §0 and §3.2).
- [ ] **11.6 — Classifiers, `[project.urls]`, contribution guide.**

---

## Milestones

| | Scope | Gate: a real task against a real model | Rough |
|---|---|---|---|
| **V0** | §0 + §2.1 + §4.1 | An agent edits a file and runs tests, permission-gated and budgeted | days |
| **V1** | §5 + §1.1–1.2 | "Refactor this repo and run tests" plans its own fan-out; an over-budget plan is rejected with a recorded reason | weeks |
| **V2** | §6 + §8.1 | A session survives restart; a human approves a destructive action mid-run | ~1 month |
| **V3** | §7 + §9 | Incident response runs from a webhook, remediation gated on approval | ~1 month |
| **V4** | §10 + §11 | Replay any production run; a stranger `pip install`s it | ~1 month |

**Honest scale:** V0 is days. V1 is weeks. V2–V4 are months. Most of the risk
is concentrated in §5.2.

**The failure mode to avoid** — this repo already hit it once — is writing the
essay before the code and marking milestones done because tests pass rather
than because a real task ran. Every gate above is defined by a real task
against a real model for exactly that reason.
