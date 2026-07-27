# GraphARC

**A graph engineering toolkit.** Disciplined multi-agent graphs on top of [LangGraph](https://github.com/langchain-ai/langgraph): typed state contracts, per-node write permissions, hard budgets, verification anchored to evidence, replayable traces — and, eventually, a durable memory graph with provenance.

> *Graph engineering*: when one agent loop stops being enough, coordination becomes the engineering. Nodes do work (agent loops, model calls, deterministic functions, humans approving things), edges decide what runs next, and a typed shared state flows between them. GraphARC implements the discipline that makes such graphs production-grade rather than demos — the ideas emerging from the July 2026 loops-vs-graphs debate (Steinberger, Ng, et al.), the "Two Graphs, Two Jobs" split, and twenty years of pre-AI graph systems where every edge means something and every path can be explained.

## What it adds on top of LangGraph

| Discipline | Mechanism |
|---|---|
| Typed edges | State schemas are Pydantic models; unknown fields are rejected at the edge |
| Write permissions | Every node declares which state fields it may write; undeclared writes raise |
| Bounded work | Per-run `Budget` (iterations / tokens / seconds) with a hard ceiling checked before every node |
| Earned cycles | `dag=True` rejects conditional edges and cycles at compile time — Stage 0 before Stage 2 |
| Validated routing | Routers are deterministic code over typed state; model prose cannot steer an edge |
| Convergence | Cycles end with a machine-readable `termination_reason`, never by running out of road |
| Traces as replay points | Every node execution is one JSONL line: state delta, duration, tokens, error |
| Crash-safe resume | SQLite checkpointing; kill a run mid-write and resume produces exactly one output |
| Enforced tool boundary | Permissions decide *which* tools run (deny → ask → allow, default deny); a sandboxed executor bounds *what a tool touches* once it does |
| Durable memory | Claims carry provenance and are superseded, never overwritten — later runs see what changed and skip known dead ends |

The sandbox is a CPython audit hook, which is an in-process boundary, not a kernel one: subprocess spawning is refused outright (a child would run unhooked), and native extensions can bypass it. `os.stat` raises no audit event, so metadata reads outside the workspace are not blocked. For a real boundary, a container executor implements the same interface.

## Install (development)

```bash
git clone https://github.com/shashankshekharsingh1205/GraphARC
cd GraphARC
uv sync --group dev
```

## Quickstart

```bash
grapharc run stage0        # deterministic DAG: load -> split -> count -> report
grapharc run stage1        # one earned agent loop: discover -> act -> verify -> repeat
grapharc run stage2        # typed cyclic graph: extract claims -> verify -> retry
grapharc run stage3        # bounded fan-out with failure isolation and dedup
grapharc run stage4        # bounded investigation loop with convergence guards
grapharc run stage5        # independent verifier: different model + evidence anchor
grapharc run stage6        # durable memory: provenance, supersession, recall
grapharc run capstone      # all of the above in one research agent

grapharc trace <path>            # pretty-print a run trace
grapharc metrics <path> <run-id> # tokens, retries, termination reason, per-node counts
grapharc viz <path> <run-id>     # Mermaid diagram of the executed path
```

Building a graph:

```python
from grapharc import ArcGraph, ArcState, Budget
from grapharc.runtime.graph import START, END

class State(ArcState):
    question: str
    answer: str = ""

def answer(state: State) -> dict:
    return {"answer": f"42 (asked: {state.question})"}

g = ArcGraph(State, name="demo", budget=Budget(max_iterations=10))
g.add_node("answer", answer, writes={"answer"})   # undeclared writes raise
g.add_edge(START, "answer")
g.add_edge("answer", END)
print(g.compile().invoke({"question": "meaning of life"}))
```

## The model gateway

The primary backend drives the **Claude Code CLI** (`claude -p`), so GraphARC runs on a Claude subscription with no API key. That CLI is a full agent, so the adapter invokes it as a pure inference endpoint: every tool disallowed, no settings sources loaded, no CLAUDE.md pickup, prompt via stdin, flags via an argv array — never a shell string. An injected "run this command" has no tool to run it with.

```python
from grapharc.gateway import ClaudeCodeCLIChatModel

worker   = ClaudeCodeCLIChatModel(model="claude-sonnet-5")
reviewer = ClaudeCodeCLIChatModel(model="claude-opus-5")  # different model breaks correlated agreement
```

Caveats this design accepts openly: no prompt caching on this path (keep node contexts lean), subscription quota burn (budget caps are load-bearing), and provider volatility (the backend is a config swap).

## Tests are gates

Each roadmap stage ships with a failure-gate test, not just a happy path:

- **Stage 0** — kill the process mid-write, restart: the report exists exactly once.
- **Stage 1** — an impossible task halts via a no-progress window, far below the budget ceiling, with the reason recorded.
- **Stage 2** — any bad model output is attributable to the exact node, step, and state delta, with a checkpoint replay point; a prompt-injection attempt cannot steer the router.
- **Stage 3** — a crashed worker and a hung worker cannot sink the batch; duplicate evidence cannot inflate confidence.
- **Stage 4** — an impossible investigation halts on a no-progress window, far under the hard ceiling.
- **Stage 5** — the verifier rejects a fabricated citation without consulting the model, catches a misleading one with it, and fails closed on any ambiguous verdict.
- **Harness** — a tool cannot read, list, or delete outside its workspace, cannot open a socket without declaring the capability, and cannot spawn a subprocess to escape the audit hook; a tool that swallows SIGTERM is still killed.
- **Stage 6** — a later run reuses a fact from an earlier run, sees what superseded it, and avoids the dead end.
- **Capstone** — unverified evidence is never persisted to memory.

```bash
uv run pytest          # mock models, deterministic
uv run pytest -m live  # real backends (when configured)
```

## Roadmap

- [x] **M0** — scaffold, mock-model harness, CI
- [x] **M1** — runtime discipline (Stages 0–2): typed state, write permissions, budgets, traces, checkpoint resume
- [x] **M2** — scale & safety (Stages 3–5): bounded fan-out with failure isolation and dedup, convergence guards, independent verifier (different model, fresh context, deterministic anchor)
- [x] **M3** — gateway & harness: Claude Code CLI chat-model adapter (text-completion-only, tools disabled), tool permissions, hooks, sandboxed executor
- [x] **M4** — memory graph (Stage 6): claims with provenance and supersession, GraphRAG retrieval
- [x] **M5** — operate & ship (Stage 7): metrics/viz CLI, capstone research agent

## Design lineage

Architecturally *inspired by* systems studied from public documentation: OpenClaw (policy-before-schema tool gating, file-first state, and its security post-mortems), Hermes Agent (budgeted tiered memory, ephemeral subagents), Claude Code (advisory-vs-enforced split, subagent context isolation, verification-centered loops), and OpenRouter (routing semantics, budget-scoped accounting).

## License

MIT
