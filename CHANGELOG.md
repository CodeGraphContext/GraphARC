# Changelog

Notable changes to GraphARC. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[PEP 440](https://peps.python.org/pep-0440/).

Entries describe code that exists in this tree. Where a capability is partial
the entry says which part, and where a declared extra buys nothing yet it is
listed under *Known gaps* rather than omitted. Planned work lives in
[ROADMAP.md](ROADMAP.md) and measured status in [ASSESSMENT.md](ASSESSMENT.md);
neither belongs here.

## [Unreleased]

Nothing yet.

## [0.1.0a0] — unreleased

First packaged version. The distribution builds and installs, and is not on
PyPI: `.github/workflows/release.yml` publishes on a `v*` tag once PyPI Trusted
Publishing is configured for the repository. An alpha because the API is not
settled and several subsystems are partial — see *Known gaps*.

### Graph kernel — `grapharc.runtime`

- `GraphARC` / `CompiledGraphARC`: a LangGraph `StateGraph` wrapper that adds
  typed state contracts, per-node declared writes (`WritePermissionError` on an
  undeclared key), and deep-copy isolation between nodes.
- Budgets: `Budget` and `BudgetMeter` over iterations, wall-clock seconds and
  tokens. `max_seconds` interrupts a node in progress (SIGALRM on the main
  thread, async-exception injection elsewhere) and re-arms. Tokens are metered
  automatically by `MeterCallbackHandler`, deduplicated by call identity, and
  checked at `on_llm_end`.
- Convergence guards and `StopReason`, DAG mode, cycle detection, and bounded
  fan-out with worker isolation and dedup.
- Async entry points: `ainvoke`, `astream`, `astream_events`, `aget_state`,
  `aget_state_history`, `aupdate_state`. `async def` nodes get the same write
  checking and budget treatment as sync ones; driving an async graph through a
  sync entry point raises `AsyncNodeError` rather than misreporting.
- Nodes may return a `langgraph.types.Command`. Its `update` goes through the
  same write permission check, and its `goto` must name a node the compiled
  graph actually has.
- Fail-closed run context: `MissingRunContextError` when a node runs outside
  one, `StateTypeError` on a write that does not match the annotation.

### Model gateway — `grapharc.gateway`

- `get_model("<backend>/<model>")` over three backends: `claude-cli` (the
  Claude Code CLI, argv array, prompt on stdin), `openrouter`, and `mock` for
  deterministic tests. An unknown backend is rejected with
  `UnknownBackendError` instead of being folded into a model name.
- OpenRouter backend: `bind_tools`, `with_structured_output`, `_stream`,
  `_agenerate`, provider routing (`order` / `sort` / `max_price` /
  `require_parameters`) and model-level fallback chains.
- Cross-backend policy that applies whichever adapter serves a node:
  `RetryPolicy` with transient-only classification (`call_with_retry`,
  `is_transient`), and `SpendMeter` / `cost_ceiling_usd` raising
  `CostCeilingExceeded`.
- Cache-token accounting, and `redact` for keys in logged configuration.

### Tool plane — `grapharc.harness`, `grapharc.tools`

- `ToolRegistry` / `ToolSpec`, deny→ask→allow `PermissionPolicy`, pre/post
  hooks, and approval callbacks.
- `SandboxedExecutor`: an audit-hook sandbox with path confinement, network
  gating, spawn refusal and SIGKILL escalation. `LocalExecutor` for the
  unconfined case, named so nobody mistakes it for a boundary.
- `ContainerExecutor` (`grapharc.harness.container`): runs a tool in docker or
  podman with `--network none` unless the tool declares it needs the network,
  `--cap-drop ALL`, `no-new-privileges`, a pids limit, a single workspace mount,
  an environment built from scratch rather than inherited, and optional memory,
  CPU and read-only-rootfs limits.
- `AgentNode`: observe → model → tool request → permission check → sandboxed
  execute → repeat, budgeted and traced. A denied tool is reported back to the
  model rather than ending the run; malformed tool JSON is reported rather than
  read as success; stall detection keys on the tool result.
- `grapharc.tools.register_core_tools` installs seven workspace-confined tools:
  `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`,
  `run_command`. Each resolves and confines its own path arguments
  independently of the executor.

### Planner and admission — `grapharc.planner`

- `PlannerNode.propose` emits a typed `Subgraph` of `ProposedNode` /
  `ProposedEdge` and executes nothing.
- `AdmissionChecker.check` is the deterministic, model-free gate: nodes against
  a `NodeRegistry`, edges against an `EdgePolicy`, worst-case `CostEstimate`
  against remaining budget, and depth/size against `AdmissionLimits`. Rejections
  are structured (`Rejection`, `AdmissionResult.feedback()`) and can be fed back
  to the planner.
- `Materializer.materialize(admitted, proposal)` is the only path from an
  admitted proposal to a runnable graph. It takes the `AdmissionResult` first,
  refuses one that is not `ADMITTED`, and matches it to the proposal by content
  fingerprint, so what runs is what was admitted. Node bodies come from the
  `NodeRegistry` and nowhere else — a proposal names a kind and can never supply
  a body — and a body returning `Command(goto=…)` is confined to destinations
  the admitted proposal declared an edge to (`UnadmittedTransition`).
- `GovernedLoop.run` closes the cycle: propose → admit → materialise → execute →
  observe → replan, until a `LoopStop` says why it ended. Every round goes
  through the same checker against the same registry and edge policy; there is
  no "already approved" path, so work discovered mid-run cannot bypass the gate.
  Rounds share one `BudgetMeter`, and with a `TraceRecorder` each round and the
  stop are recorded as trace events.

### Documentation

- `docs/cookbook/` — six sections, every snippet executed and its real output
  pasted rather than written by hand. `tests/test_cookbook_*.py` re-runs each
  snippet and byte-compares it against the page, so a rotted example fails CI.

### Sessions — `grapharc.session`

- `SessionManager` / `Session`: long-lived sessions with a status lifecycle,
  a SQLite-backed `SessionStore`, and resume across a process restart.
- Human approval as a suspending step (`ApprovalRequired`, `Session.decide`),
  an event queue for multi-turn input, and invalid transitions raised rather
  than absorbed (`InvalidTransition`, `SessionBusy`, `SessionTerminated`).

### Policy — `grapharc.policy`

- `PolicyEngine.from_file` reads a TOML policy document covering nodes, edges,
  tools and spend, with per-tenant scoping.
- Every decision is written to an `AuditLog` naming the rule and the policy
  digest that produced it.
- Composes with the harness: `engine.permission_policy(tenant=...)` returns a
  `PermissionPolicy`, `engine.approval_router(...)` the matching callback. A
  commented example ships at `grapharc/policy/example.toml`.

### Memory — `grapharc.memory`

- Claims with provenance and supersession instead of overwrite, over a
  `ClaimStore` protocol with two implementations: in-memory and
  `SQLiteMemoryStore`, the latter proven durable across separate processes.
- Hybrid retrieval: BM25F lexical scoring over subject/predicate/object, an
  optional injected vector channel, and graph traversal with score decay per
  hop. `HashingEmbedder` is a dependency-free fallback that adds character-level
  robustness, not meaning.
- Contradiction detection (`detect_contradictions`, `supersede_conflicting`) so
  superseding no longer requires the caller to know the old claim's id.
- Artifact storage (`MemoryArtifactStore`, `SQLiteArtifactStore`), with
  `render_artifacts` producing a bounded listing for a prompt.
- `render_context` fills blocks in priority order under a `max_tokens` budget
  and states the one case it can overshoot: a budget smaller than the reserve
  held back for the truncation note.

### Observability — `grapharc.observe`

- JSONL traces are the single writer; metrics, replay, cost and OTel all read
  that one file, so a dashboard cannot contradict the audit trail.
- `replay` / `replay_thread` re-execute a recorded run, and `diff_runs` /
  `diff_trace` compare two of them.
- Cost attribution per node, run and thread against a `RateCard`.
- OpenTelemetry export (`OTelSpanExporter`) with nested node spans, plus
  in-memory and null exporters; `OTelUnavailable` when the extra is missing.
- Run metrics summaries and Mermaid rendering of the executed path.

### HTTP API — `grapharc.server`

- `create_app(registry=...)` builds a FastAPI app over a `GraphRegistry`:
  `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`,
  `POST /sessions/{id}/events`, `GET /sessions/{id}/trace`,
  `GET /sessions/{id}/stream`, `GET /healthz`.
- Importing `grapharc.server` is the only thing that imports FastAPI; the rest
  of the library does not.

### CLI

- `grapharc run | agent | serve | models | replay | diff | trace | metrics | viz`,
  plus `--version`.
- Every command takes `--json` and prints the same payload as one document on
  stdout, errors included. Exit codes are part of the interface: `0` success,
  `1` ran with a negative answer, `2` could not run.

### Packaging

- Trove classifiers, `[project.urls]`, keywords, and optional extras
  (`openrouter`, `server`, `mcp`, `memory`, `otel`, `api`, and `all`).
- Sdist and wheel build with hatchling. `[tool.hatch.build].ignore-vcs` is on:
  hatchling otherwise treats `.gitignore` as a build exclusion, which was
  verified to drop the whole `grapharc/tools` subpackage out of the wheel with
  no error and a build that reported success.
- `MANIFEST.in` mirrors the hatchling sdist allowlist. It is not read by the
  build — hatchling never reads MANIFEST.in — and `tests/test_packaging.py`
  fails if the two lists drift.
- CI (`.github/workflows/ci.yml`): ruff, the test suite on 3.12/3.13/3.14, a
  guard that live tests stay opt-in, and a build job that installs the wheel and
  the sdist into clean environments and imports every module.
- Release (`.github/workflows/release.yml`): builds on a `v*` tag, refuses a tag
  that disagrees with the packaged version, and publishes via PyPI Trusted
  Publishing. No API token exists in the repository.

### Testing

- `pytest` defaults carry `-m 'not live'`, so tests that call a paid API are
  opt-in via `pytest -m live`.
- `--strict-markers` makes a misspelled marker a collection error. Without it a
  typo'd `live` marker was only a warning, so the test escaped the `not live`
  filter and called the API on a plain `pytest`.
- `required_plugins` makes pytest refuse to start without `pytest-asyncio` and
  `pytest-timeout`, so registering their markers cannot mask a missing plugin.

### Known gaps

Declared but not implemented, or implemented less far than the name suggests:

- The `mcp`, `memory` (Neo4j) and `api` (Anthropic SDK) extras install
  dependencies that nothing under `grapharc/` imports yet. They exist so the
  work being built against them has somewhere to declare itself.
- No `py.typed` marker ships, so type checkers treat the package as untyped
  even though the source is annotated.
- Nothing shipped drives the governed loop. `Materializer` and `GovernedLoop`
  close the propose → admit → execute → replan cycle as a library API, and
  `grapharc.planner` is imported by no other module: no CLI command, no example
  graph, no session graph reaches it.
- Admission authorises a *kind*, never its arguments. `Materializer` drops
  `ProposedNode.args` unless built with `forward_args=True`, which hands a
  planner's unchecked dictionary to a factory.
- An `AdmissionResult` is ordinary data, not a signed capability. The gate it
  enforces is between the planner and the operator: a caller who hand-builds one
  with a matching fingerprint materialises anything, and no library check can
  prevent that.
- `HashingEmbedder` is lexical, not semantic. Semantic retrieval requires
  injecting a real embedder.
- `bind_tools` raises `NotImplementedError` on the `claude-cli` backend, which
  is inherent to `claude -p`.
- Retries and cost ceilings are gateway-level. Cost is not yet budgeted against
  in the kernel meter alongside tokens.
- The classifiers name Linux only. The sandboxed executor needs POSIX
  fork/setsid/killpg and Linux is the only platform CI exercises.
- Live-model examples exist behind the `live` marker but are not wired into CI,
  which has no API key.

[Unreleased]: https://github.com/CodeGraphContext/GraphARC/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://github.com/CodeGraphContext/GraphARC/releases/tag/v0.1.0a0
