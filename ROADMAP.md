# GraphARC — live build list

Everything between today and a full-fledged general-purpose agent runtime, as
described in [VISION.md](VISION.md). Status is measured, not aspirational —
see [ASSESSMENT.md](ASSESSMENT.md) for how the current numbers were verified.

**Legend:** `[x]` done · `[~]` partial · `[ ]` not started · **B** blocks other
work · **!** known-false claim shipping today

Overall: **~20% of the product**. The model plane went from text-only to
tool-calling, structured output, async, streaming, and multi-provider routing,
which unblocks the agent node.

---

## Next five things

In order. Each unblocks more than it costs.

1. **Fix the false security claim** (§0.1, §0.2) — a three-line stdlib escape
   disproves a README sentence that is shipping right now.
2. ~~`bind_tools` on the gateway~~ — **done** via OpenRouter, verified live.
   Next in its place: **auto-charge tokens** (§0.4), now proven broken by a live
   run that reported zero spend while costing money.
3. **Async through the kernel** (§1.1) — **B** blocks the HTTP API, concurrency,
   and streaming. Cannot be retrofitted cheaply later.
4. **The `AgentNode`** (§4.1) — converts three orphaned subsystems into one
   working agent. This is the milestone that turns a library into a runtime.
5. **Admission checker** (§5.2) — the architectural crux and the only component
   with no prior art to copy.

---

## 0. Correctness debt

Ship-blockers. Every item verified by running code.

- [ ] **!B 0.1 — Close the `ctypes` sandbox escape.** `ctypes.CDLL("libc.so.6").system(...)`
      runs arbitrary shell from inside a tool and was verified to create a file
      outside the workspace. Hook `ctypes.dlopen` / `ctypes.dlsym` /
      `ctypes.call_function` audit events. Until then the README's *"a tool
      cannot read, list, or delete outside its workspace"* is false.
- [ ] **! 0.2 — Scrub the child environment.** `fork` inherits the parent env; a
      tool returned a planted `sk-ant-…` secret over the result pipe.
- [ ] **! 0.3 — Make `max_seconds` a real ceiling.** Checked *between* nodes, so
      a 1.5 s node completed under a 0.2 s budget. Needs a watchdog.
- [ ] **!B 0.4 — Charge tokens automatically.** `charge_tokens` has one caller:
      `charge_usage()` in `grapharc/testing.py` — the *test-doubles* module,
      imported into all seven production examples. **Confirmed live:** a real
      capstone run over OpenRouter reported `spent: 0 tokens across 7 nodes`
      while actually costing money, because `verify_claim` never charges.
      Move into the gateway via langchain-core's `UsageMetadataCallbackHandler`.
- [ ] **0.5 — Validate types at write time.** A field declared `int` accepted
      `"not-an-int"` and it escaped the graph boundary.
- [ ] **! 0.6 — Delete or correct false README claims:** "validated routing"
      (no validation exists), "crash-safe resume" (100% LangGraph), "durable
      memory" (in-process dict), "independent verifier" (object-identity check
      in an example), "kill the process mid-write" (no process is killed).
- [ ] **0.7 — Remove the Neo4j fiction.** A docstring promises an
      implementation "satisfies the same interface"; no implementation and no
      interface exist. `pyproject.toml` declares an extra nothing imports.
- [x] **0.8 — Fix `pytest` defaults.** `addopts` now carries `-m 'not live'`;
      live tests are opt-in via `pytest -m live`.
- [ ] **0.9 — Fix `LICENSE` copyright** (says "CodeGraphContext") and the
      README clone URL (404s — wrong org).
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

## 3. Tool plane — `[~] ~30%`

Built and tested; **imported by zero graphs**.

- [x] Registry, deny→ask→allow permissions, hooks, approval gates
- [x] Audit-hook executor: path confinement, network gating, spawn refusal,
      SIGKILL escalation
- [ ] **B 3.1 — Wire it to an agent** (see §4.1). Zero callers today.
- [ ] **3.2 — Container executor** — a real kernel boundary. The README claims
      one "slots in behind this interface"; it does not exist.
- [ ] **3.3 — Core tools:** bash, read, write, edit, glob, grep.
- [ ] **3.4 — Browser tool** and HTTP/network tool.
- [ ] **3.5 — MCP client** — the ecosystem standard for third-party tools.
- [ ] **3.6 — Progressive disclosure / tool search** for large tool sets.
- [ ] **3.7 — Idempotency keys** for side-effecting tools.
- [ ] **3.8 — Large-output offloading** (write to file, return a preview).

## 4. Agent node — `[ ] 0%`

The missing unit that makes everything else compose.

- [ ] **B 4.1 — `AgentNode`:** observe → model → tool request → permission
      check → sandboxed execute → verify → repeat, budgeted and traced.
      *Done when:* it edits a file and runs a test against a real model, with
      the tool call gated and the run inside a token budget.
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

## 8. Memory & artifacts — `[~] ~20%`

- [x] Claims with provenance; supersession instead of overwrite
- [x] Unicode-safe entity normalization
- [ ] **B 8.1 — Persist to disk.** SQLite first. Today everything dies with the
      process, which makes the one headline claim untrue.
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
- [~] **11.3 — Live-model examples in CI** behind the `live` marker. Stage 5
      and the capstone now run end-to-end against real cross-vendor models; CI
      wiring still pending.
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
