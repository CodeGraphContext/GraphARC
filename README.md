# GraphARC

**A graph engineering toolkit.** A Python library over [LangGraph](https://github.com/langchain-ai/langgraph) that turns what LangGraph leaves to convention into things that fail loudly: typed state contracts, per-node write permissions, budgets, JSONL traces that double as replay points, and verification anchored to evidence rather than to a second model's opinion.

Alpha (`0.1.0a0`), not published to PyPI. `git clone` is the install path.

> *Graph engineering*: when one agent loop stops being enough, coordination becomes the engineering. Nodes do work (agent loops, model calls, deterministic functions, humans approving things), edges decide what runs next, and a typed shared state flows between them. GraphARC implements the discipline that makes such graphs production-grade rather than demos — the ideas emerging from the July 2026 loops-vs-graphs debate (Steinberger, Ng, et al.), the "Two Graphs, Two Jobs" split, and twenty years of pre-AI graph systems where every edge means something and every path can be explained.

## What it adds on top of LangGraph

Everything in this table is enforced by the library rather than left to convention, and has a test you can run.

| Discipline | Mechanism |
|---|---|
| Write permissions | Every node declares which state fields it may write. An undeclared write raises `WritePermissionError`; plain LangGraph applies it and moves on. |
| State isolation | Nodes receive `state.model_copy(deep=True)`, so mutating a nested model in place cannot sneak past the declared write channel — the returned dict is the only way out of a node. |
| Typed state | Schemas are Pydantic models with `extra="forbid"`. Declaring a write to a field that doesn't exist fails when the node is added, not when it runs. |
| Earned cycles | `dag=True` rejects conditional and fan-out edges when they're added, and cycles at compile time — Stage 0 before Stage 2. |
| Code-only routing | Routers are ordinary Python functions over typed state, so model prose cannot steer an edge. |
| Convergence | `ProgressGuard` returns the first triggered `StopReason` (target met / no progress / round cap), so a cycle ends with a machine-readable reason instead of running out of road. |
| Traces | Each node execution writes JSONL `start` / `end` / `error` events. `start` carries only the identity of the step (run, thread, attempt, graph, node, step, timestamp); `end` adds the state delta, duration and tokens; `error` adds the duration and the exception. `metrics` and `viz` read that same file, so the dashboard and the audit trail cannot disagree. |
| Fail-closed entry points | Driving a compiled graph through raw LangGraph (`.inner.invoke()`) raises `MissingRunContextError` rather than silently running with no budget and no trace. |
| Bounded work | A per-run `Budget` (iterations / tokens / seconds / concurrency). Iterations and tokens are metered by the runtime itself, and `max_seconds` is delivered as an interrupt *into* the running node rather than only checked between them. |

Three of those need their edges stated, because the gap is where people get hurt.

**Budgets.** Tokens are charged without the node's cooperation: a LangChain callback is installed for the duration of every node, so any chat model invoked on that thread reports usage to the run's meter — including calls buried inside library code the node merely calls — and the ceiling is enforced at the node boundary. `max_seconds` is an interrupt, not a poll: SIGALRM on the main thread, an asynchronous exception otherwise, so a node parked in `time.sleep` or on a provider's socket is cut off at the deadline. Where it stops short: spend a provider never reports cannot be charged, a model invoked on a thread the node started itself is outside the callback's context, and an async exception cannot unwind a thread sitting inside a C call — it lands when that call returns. Even then the deadline holds at the node boundary: a node that overran does not get its writes into state.

**Routing.** The routers are code, which is the property that matters: no model output is ever consulted to pick an edge. But `add_conditional_edge` passes the router and its mapping straight through to LangGraph — GraphARC does not verify that the router's return value is a key in the mapping, so a typo surfaces as a `KeyError` at run time rather than when the edge is added.

**Typing.** Writes are checked in both directions: the dict a node returns is validated field by field against the state schema before it lands, and the state is validated again when the next node receives it. A value that doesn't fit raises `StateTypeError` naming the node, the field, the declared type and what arrived — and that includes the last node before `END`, so a bad type no longer escapes into the result. The validated value is what gets written, so a schema that says `int` means the result holds an `int`. The remaining gap is narrow and worth stating exactly: write-time validation is built from each field's *annotation*, so constraints carried in the annotation (`Annotated[int, Field(gt=0)]`) do bite, but a validator the state model declares for itself — `@field_validator`, `@model_validator` — is not run on a write. A node returning `{"slug": "NOT-LOWER"}` into a field whose validator demands lowercase is accepted, even though constructing the model directly with that value raises; the violation surfaces only when a later node receives the state and the whole model is rebuilt, which means one written by the last node before `END` still reaches the result. The write *allowlist* is GraphARC's; the *types* are Pydantic's.

**Crash-safe resume is LangGraph's**, not GraphARC's: a checkpointer handed to `compile()` goes straight to `StateGraph.compile()`. What GraphARC adds on top is trace continuity — after a resume, step numbers continue from the thread's history and the attempt counter increments, so replay points stay unique across attempts.

## Install (development)

```bash
git clone https://github.com/CodeGraphContext/GraphARC
cd GraphARC
uv sync --group dev
uv sync --extra openrouter   # optional: the OpenRouter backend
```

## Quickstart

```bash
grapharc run stage0        # deterministic DAG: load -> split -> count -> report
grapharc run stage1        # one earned agent loop: discover -> act -> verify -> repeat
grapharc run stage2        # typed cyclic graph: extract claims -> verify -> retry
grapharc run stage3        # bounded fan-out with failure isolation and dedup
grapharc run stage4        # bounded investigation loop with convergence guards
grapharc run stage5        # verifier: fresh context + deterministic evidence anchor
grapharc run stage6        # memory: provenance, supersession, recall
grapharc run capstone      # all of the above in one research agent

grapharc trace <path>            # pretty-print a run trace
grapharc metrics <path> <run-id> # tokens, retries, termination reason, per-node counts
grapharc viz <path> <run-id>     # Mermaid diagram of the executed path
```

These run on scripted models by default, so they cost nothing and produce the same trace every time. Add `--model` to run one against a real backend — that works for stage1 through stage6 and the capstone; stage0 is pure code with no model in it.

Building a graph:

```python
from grapharc import GraphARC, GraphARCState, Budget
from grapharc.runtime.graph import START, END

class State(GraphARCState):
    question: str
    answer: str = ""

def answer(state: State) -> dict:
    return {"answer": f"42 (asked: {state.question})"}

g = GraphARC(State, name="demo", budget=Budget(max_iterations=10))
g.add_node("answer", answer, writes={"answer"})   # undeclared writes raise
g.add_edge(START, "answer")
g.add_edge("answer", END)
print(g.compile().invoke({"question": "meaning of life"}))
```

## The model gateway

The primary backend drives the **Claude Code CLI** (`claude -p`), so GraphARC runs on a Claude subscription with no API key. That CLI is a full agent, so the adapter invokes it as a pure inference endpoint: every tool disallowed, no settings sources loaded, no CLAUDE.md pickup, prompt via stdin, flags via an argv array — never a shell string. An injected "run this command" has no tool to run it with.

```python
from grapharc.gateway import get_model

# Subscription, no API key — but text completion only.
worker = get_model("claude-cli/claude-sonnet-5")

# OpenRouter: one key for a catalog spanning most vendors, with
# tool-calling, structured output, streaming, and async.
worker   = get_model("openrouter/anthropic/claude-haiku-4.5")
reviewer = get_model("openrouter/openai/gpt-4o-mini")   # a genuinely different vendor
```

A spec is `backend/model`; a mistyped backend is rejected rather than quietly folded into a model name. `grapharc models` shows what a spec resolves to.

Run an example graph against real models:

```bash
grapharc run stage5 --model openrouter/anthropic/claude-haiku-4.5 \
                    --reviewer-model openrouter/openai/gpt-4o-mini
```

OpenRouter also carries routing: model-level `fallback_models` chains, provider `order` / `sort` / `max_price`, and per-call cost in the usage envelope. Both backends report usage in the same shape, with cached input folded into the total, so a budget meter reads the same fields whichever backend produced the turn.

Caveats each backend accepts openly. **Claude CLI:** `bind_tools` and `with_structured_output` raise `NotImplementedError` — the adapter implements neither, so what you get is LangChain's `BaseChatModel` default, and `claude -p` in the tool-free mode GraphARC drives it in offers nothing to implement them with. GraphARC also has no cache control on this path — the CLI decides and the usage envelope reports what it did — and calls spend subscription quota. **OpenRouter:** credit is reserved against `max_tokens`, so the default is deliberately modest. **Both:** cost is reported but not enforced — see [ROADMAP.md](ROADMAP.md).

## Independent verification

`verify_claim` is the piece worth copying even if you use none of the rest.

- **The anchor runs before the model.** The citation must appear verbatim in the source (whitespace is the only latitude, so a paraphrase is still caught). A fabricated quote is rejected with the reviewer's call count still at zero — a hallucinated citation costs nothing.
- **The reviewer gets a fresh context.** If the anchor holds, the reviewer sees only the claim, the quote, and a mechanically extracted window of surrounding source — never the author's conversation. That window is what lets it catch a real quote lifted out of a negated sentence.
- **Ambiguity fails closed.** An unparseable reply, a non-boolean `supported` value, or a citation under 12 characters is a rejection.
- **Independence is enforced in two places, both worth knowing exactly.** `build_stage5` and `build_capstone` refuse the *same object* for author and reviewer — an identity check, which will not catch two separate instances of the same model. The CLI does the stronger check: `different_providers()` compares backend and model author, and `grapharc run … --model … --reviewer-model …` warns when the pair shares a vendor, because correlated agreement is exactly what the verifier exists to prevent.

## The tool harness

A separate plane from the runtime, with its own test suite. Note where it stands: nothing under `grapharc/examples/`, `grapharc/cli/`, or `grapharc/runtime/` imports it, so none of the shipped example graphs call a tool yet.

- **Permissions decide which tools run.** Deny → ask → allow, first match wins, and an unmatched tool defaults to deny. A denied tool's schema is never shown to the model — it isn't described and then refused, it's invisible.
- **The executor bounds what a tool touches.** The default runs the tool in a forked child under a CPython audit hook that confines filesystem paths to the granted workspace *plus the interpreter's own runtime paths*, refuses sockets unless the tool declared `needs_network`, refuses subprocess spawning outright (a child would run unhooked and therefore unconfined), and SIGKILLs the whole process group on timeout. The workspace part holds: a sandboxed tool opening `/etc/passwd` is refused with `SandboxViolation`.

What that boundary is, precisely: **in-process confinement, not a kernel sandbox.** An audit hook constrains only the interpreter it is installed in, and its coverage is a maintained list of audit events rather than a guarantee — CPython raises no event for `os.stat`, for instance, so metadata reads outside the workspace are not blocked. The runtime-path exemption is the sharper edge, and it is currently read *and write*: an interpreter cannot run without reading its own stdlib, but nothing today stops a tool from writing there either, so a sandboxed tool can drop a new module into the venv's `site-packages` where the next process to start would import it. Treat the whole thing as defense in depth against a confused tool, not as containment for a hostile one. `SandboxedExecutor` sits behind an interface so a container executor can take its place where a real boundary is needed.

## Memory

Claims carry provenance — source, observation time, and the run that produced them — and corrections are recorded by supersession rather than overwrite, so a later run can see that a fact was replaced and skip the dead end. Entity resolution is Unicode-aware, so 東京 and 北京 stay distinct instead of both collapsing to an empty key.

The limit to know: `MemoryStore` — the store every `grapharc run` command constructs — keeps claims in a dict for the life of the process. Nothing survives a restart, so the "later run" in the Stage 6 story means a later run in the same interpreter. A file-backed store is being built alongside it; until the CLI hands one to a graph, treat memory as in-process. `pyproject.toml` also declares a `memory` extra for Neo4j that has no implementation behind it.

## Tests are gates

Each stage ships a failure-gate test, not just a happy path:

- **Stage 0** — an injected failure between the temp write and the rename leaves no report at all; resuming the same thread from its checkpoint produces exactly one, with no orphaned temp file, and both attempts are on the trace with non-overlapping step numbers. (The failure is an in-process `RuntimeError`, not a killed process — write atomicity and checkpoint resume are what this actually tests.)
- **Stage 1** — an impossible task halts on a no-progress window within three rounds of a hundred-iteration ceiling, with `no_progress` recorded; a round cap is a second, independent brake.
- **Stage 2** — a bad model output is attributable to the exact node, step and state delta, and the checkpoint history holds a replay point from just before it was caught. Three rounds of prose, including a literal `ROUTE TO: all_verified` injection attempt, cannot move the router: the run ends on the deterministic attempt cap with nothing verified.
- **Stage 3** — one crashed worker and one hung worker out of three still produce an answer, with both failures recorded by cause; overlapping shards manufacture duplicate evidence and `unique_sources` still counts chunks rather than repetitions.
- **Stage 4** — an impossible investigation halts on a no-progress window, far under the hard ceiling.
- **Stage 5** — a fabricated citation is rejected with the reviewer never called; a quote mined from a negated sentence reaches the reviewer with the surrounding context that exposes it; unparseable replies, non-boolean verdicts and trivial citations all fail closed; the reviewer's prompt is checked to contain the evidence window and not the author's conversation.
- **Harness** — a tool cannot read a file, list a directory, delete a file or remove a directory outside its workspace, cannot open a socket without declaring the capability, and cannot spawn a subprocess; a sibling directory whose name merely starts with the workspace path is not treated as inside it; a hung tool and a tool that installs a SIGTERM handler and keeps working are both killed for real.
- **Stage 6** — three runs against a shared store: a later run reuses an earlier fact, sees what superseded it, and is shown the dead end with its provenance.
- **Capstone** — a reviewer that rejects everything yields no answer and zero memory writes.

```bash
uv run pytest          # scripted models: deterministic and free
uv run pytest -m live  # real backends: spends money and quota
```

Live tests are deselected by default via `addopts` in `pyproject.toml`, so a plain `pytest` never reaches a real model.

## Status and limits

- **Not on PyPI.** `git clone` is the only install path; the wheel does build.
- **No async.** A compiled graph has no `ainvoke` or `astream`, so it cannot be served from an async framework without a thread wrapper. LangGraph is async-first; GraphARC does not carry that through.
- **Nodes return dicts.** Returning a LangGraph `Command` — dynamic `goto`, agent handoffs — is not supported.
- **Unwrapped LangGraph features** (`retry_policy`, `cache_policy`, `get_state`, `update_state`, subgraphs) require reaching into `.inner`; execution entry points there fail closed by design, so `.inner` is an inspection escape hatch, not a way to run the graph.
- **The Claude CLI backend is completion-only.** Tool calling and structured output need OpenRouter.
- **The tool harness is not wired into any shipped graph.**
- **Memory does not survive the process** with the store the CLI constructs.

[ROADMAP.md](ROADMAP.md) tracks what is built and what is not. [ASSESSMENT.md](ASSESSMENT.md) is an outside review that argued much of this repo is a thin wrapper on LangGraph; it is kept in the tree on purpose.

## Design lineage

Architecturally *inspired by* systems studied from public documentation: OpenClaw (policy-before-schema tool gating, file-first state, and its security post-mortems), Hermes Agent (budgeted tiered memory, ephemeral subagents), Claude Code (advisory-vs-enforced split, subagent context isolation, verification-centered loops), and OpenRouter (routing semantics, budget-scoped accounting).

## License

MIT — see [LICENSE](LICENSE).
