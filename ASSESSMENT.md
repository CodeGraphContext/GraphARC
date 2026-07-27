# What GraphARC actually is

An honest assessment, written after three independent reviewers were asked to
answer one question — *"is this a thin wrapper on LangGraph, or an end-to-end
solution?"* — with instructions not to be kind. Every claim below was verified
by running code, not by reading docstrings.

**Three of four assessors said "thin wrapper." They were right.**

---

## The straight answer

GraphARC is **a Python library that wraps LangGraph.** It is not an end-to-end
solution, not a product, and not a framework. It is an alpha (`0.1.0a0`) that
is not published to PyPI.

More precisely, it's three unrelated things sharing a package name:

1. **A discipline layer over LangGraph** (`runtime/`) — genuinely useful, but
   the net new logic is roughly 60–150 lines depending on how you count. The
   rest is signature plumbing over `StateGraph`.
2. **A Claude Code CLI chat model** (`gateway/`) — the single most adoptable
   thing here, and it doesn't depend on the rest of the library at all. It
   should probably be its own package.
3. **Two orphaned subsystems** (`harness/`, `memory/`) — 417 lines that no
   graph, no example, and no CLI command imports. They exist, they're tested,
   and nothing uses them.

### The numbers

| | Lines of real code |
|---|---|
| `runtime/` | 555 |
| `harness/` | 285 *(orphaned)* |
| `cli/` | 195 |
| `observe/` | 166 |
| `gateway/` | 139 *(used by tests only)* |
| `memory/` | 132 *(orphaned)* |
| **Library total** | **1,542** |
| `examples/` | 841 |
| `tests/` | 1,311 |

In `runtime/graph.py` — the heart of the thing — 254 lines of code contain
exactly **9 lines that call LangGraph.** But most of the *public* surface is
pass-through: `invoke` is one line, `stream` is three, `add_edge` is three.
The genuinely new behavior lives in private methods (`_wrap`, `_run_config`,
`_assert_acyclic`) totalling ~130 lines.

One assessor reimplemented write-permissions, budgets, deep-copy isolation and
JSONL tracing directly on raw LangGraph **in 14 lines of glue**, and it caught
the same undeclared write. That's the honest measure of the runtime's value:
the *ideas* are good, the *code* is a decorator, not a library.

---

## Five things it genuinely does

All output below is real, captured from actual runs.

### 1. Catch a node writing state it never declared

LangGraph 1.2.9 **silently drops** an undeclared key. GraphARC raises.

```python
from grapharc import GraphARC, GraphARCState, WritePermissionError
from grapharc.runtime.graph import END, START

class S(GraphARCState):
    plan: str = ""
    answer: str = ""

def planner(state: S) -> dict:
    return {"plan": "outline", "answer": "sneaky write"}   # 'answer' undeclared

g = GraphARC(S, name="demo")
g.add_node("planner", planner, writes={"plan"})
g.add_edge(START, "planner"); g.add_edge("planner", END)
g.compile().invoke({})
```

```
WritePermissionError: node 'planner' wrote undeclared fields ['answer']; declared writes: ['plan']
```

This is the best idea in the repo. The subtler half is that nodes receive
`state.model_copy(deep=True)`, so mutating a nested Pydantic model in place
can't sneak past the declared write channel either — a real correctness
observation most people miss.

### 2. Stop a runaway loop

```python
g = GraphARC(S, name="loop", budget=Budget(max_iterations=5))
g.add_node("tick", lambda s: {"n": s.n + 1}, writes={"n"})
g.add_edge(START, "tick")
g.add_conditional_edge("tick", lambda s: "again", {"again": "tick"})  # never exits
g.compile().invoke({})
```

```
BudgetExceeded: max_iterations reached (5/5)
```

Honest caveat: this is LangGraph's `recursion_limit` with a better exception
name and a lower default. See the overclaims section for what the *other* two
budget dimensions actually do (spoiler: nothing).

### 3. Reject a cycle before the graph ever runs

```python
g = GraphARC(S, name="dag_demo", dag=True)
g.add_node("a", ...); g.add_node("b", ...)
g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", "a")
g.compile()
```

```
GraphCycleError: graph 'dag_demo' is dag=True but has a cycle: a -> b -> a
```

LangGraph has no DAG mode. This is ~30 lines of depth-first search, and it's
real.

### 4. Reject a fabricated citation without paying for a model call

The reviewer here is **primed to approve** (`{"supported": true}`) and the
claim is still rejected, because the quote isn't in the source:

```python
reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "looks right"}'])
verdict = verify_claim(reviewer,
    text="GraphARC guarantees provably optimal routing.",
    citation="GraphARC guarantees provably optimal routing",   # fabricated
    source_text="GraphARC enforces per-node write permissions and bounded work.")
```

```
accepted       = False
anchor_ok      = False
model_accepted = None
reason         = citation does not exist in the source (deterministic anchor)
reviewer.call_count = 0
```

The ordering is the point: mechanical rejection happens *before* the model is
consulted, so a fabrication costs zero tokens. Verified live against real
Claude models too — an invented quote died at the anchor, and a quote lifted
verbatim out of a *negated* sentence was caught by an Opus reviewer using the
surrounding-context window.

### 5. Run LangGraph on a Claude subscription with no API key

```python
from grapharc.gateway import ClaudeCodeCLIChatModel
model = ClaudeCodeCLIChatModel(model="claude-sonnet-5")
model.invoke([HumanMessage(content="Reply with exactly the word: pong")])
```

```
REPLY: 'pong'
USAGE: {'input_tokens': 8979, 'output_tokens': 4, 'total_tokens': 8983,
        'input_token_details': {'cache_creation': 5688, 'cache_read': 3289},
        'uncached_input_tokens': 2, 'cost_usd': 0.0357697}
```

This is the component with no drop-in substitute. LiteLLM and OpenRouter need
an API key; Anthropic's `claude-agent-sdk` runs the CLI *as an agent*, which is
the opposite of what's wanted. The threat model is right — the CLI is a full
agent whose tools GraphARC can't police, so it's stripped to a completion
endpoint (`--disallowedTools *`, `--setting-sources ""`, empty cwd, prompt via
stdin, argv array never a shell string).

Note `uncached_input_tokens: 2` against a real total of `8,983`. Folding cache
tokens into the total is a detail most homegrown adapters get wrong by an
order of magnitude.

---

## What the README overclaims

Each of these was **disproved by running code**, not by inspection.

### The sandbox does not hold. Three lines of standard library walk out.

```python
import ctypes
ctypes.CDLL("libc.so.6").system(b"touch /tmp/grapharc_ESCAPED")
```

```
tool returned 0 (NO violation raised)
marker outside workspace exists? True   <-- ESCAPED
```

CPython raises `ctypes.dlopen` / `ctypes.dlsym` / `ctypes.call_function` audit
events; the executor subscribes to none of them. This defeats both the
workspace confinement and the headline test
`test_gate_tool_cannot_spawn_a_subprocess`. The README's hedge — *"native
extensions can bypass it"* — badly undersells a stdlib-reachable escape.

Second hole, undocumented: `multiprocessing.get_context("fork")` means the
child inherits the parent's environment.

```
tool read from parent env: sk-ant-PLANTED
```

**The README sentence "a tool cannot read, list, or delete outside its
workspace... cannot spawn a subprocess to escape the audit hook" is false as
written.** It should not ship in that form.

### Two of three budget dimensions are decorative

`max_seconds` is checked *between* nodes, so it is not a ceiling:

```
COMPLETED in 1.51s despite max_seconds=0.2
```

`max_tokens` is worse. `BudgetMeter.charge_tokens` has exactly one caller in
the entire repo: `charge_usage()` in **`grapharc/testing.py`** — a module whose
own docstring says "test doubles." All seven example graphs import that test
helper into non-test code and hand-call it after every model call. Forget one
call and `max_tokens` silently becomes infinity. `langchain-core` already ships
`UsageMetadataCallbackHandler`, which would have made this automatic. It isn't
used.

So the two dimensions that actually cost money are advisory bookkeeping —
exactly what the README accuses plain LangGraph of leaving to convention.

### "Typed edges" doesn't type-check anything at write time

```python
class S(GraphARCState):
    b: int = 0
g.add_node("bad", lambda s: {"b": "not-an-int"}, writes={"b"})
```

```
final output: {'b': 'not-an-int'}   b type=str   <-- UNVALIDATED
```

An unvalidated string escaped the graph boundary through a field declared
`int`. The write *allowlist* is real; the *typing* is LangGraph's, and it isn't
enforced at the write.

### "Durable memory graph" is an in-process dict

`MemoryStore` is `dict[str, Claim]` behind a lock. There is no `save`, `load`,
file, or database anywhere in `memory/` — the module imports no `pathlib`, no
`json`, no `sqlite3`, no driver. **Everything dies with the process.** The
Stage 6 gate ("run #51 uses a fact from run #12") passes only because all three
"runs" share one Python object in one process.

`pyproject.toml` declares a `memory = ["neo4j>=5.20"]` extra. There is **zero**
Neo4j implementation. A docstring in `store.py` claims one "satisfies the same
interface" — there is no such interface and no such implementation.

"GraphRAG retrieval" is a linear scan with exact string matching. No traversal,
no edges between claims, no embeddings, no ranking. It is a dict filter.

### "Validated routing" performs no validation

`add_conditional_edge` checks `self.dag`, then forwards the router and mapping
**unchanged** to LangGraph. GraphARC never verifies the router returns a key in
the mapping, nor that it's model-free.

### "Crash-safe resume" is entirely LangGraph's

GraphARC contributes one kwarg passed through to `StateGraph.compile()`. The
gate advertised as *"kill the process mid-write, restart"* kills no process: a
module-level flag raises an in-process `RuntimeError`, then the same object is
re-invoked. Atomicity and resume are genuinely tested; the framing is not what
happens.

### "Independent verifier: different model" is an identity check

The enforcement is `if author is reviewer:` — and it lives in an *example*, not
in `verify.py`. Two `ClaudeCodeCLIChatModel(model="claude-sonnet-5")` instances
are different objects and pass the guard while being the identical model, which
is exactly the correlated-agreement failure it claims to prevent.

### Smaller factual errors

- The README's `git clone` URL 404s (wrong org; the real remote is
  `CodeGraphContext/GraphARC`).
- `LICENSE` says "Copyright (c) 2026 CodeGraphContext" — a different project.
- `uv run pytest # mock models, deterministic` is false: the two `live` tests
  run by default, shelling out to the real `claude` binary. `pyproject.toml`'s
  marker description ("deselected by default") is wrong.
- The roadmap marks M0–M5 all complete for an unpublished alpha whose M3
  subsystems nothing calls.

---

## What it cannot do

- **Async — anything.** There is no `ainvoke` or `astream`. An `async def` node
  fails with a misleading `WritePermissionError: ... got <class 'coroutine'>`.
  You cannot serve this from FastAPI. LangGraph is async-first; GraphARC
  amputates that.
- **Return a `Command`** from a node — so no dynamic `goto` routing and no
  agent handoffs.
- **Reach LangGraph features it never wrapped**: `retry_policy`, `cache_policy`,
  `get_state`, `update_state`, subgraphs. Using them means reaching into
  `.inner` — which the library's own error message tells you not to do, and
  which its own tests do anyway.
- **Persist anything a graph learns.** Second process starts from zero.
- **Be installed by a stranger.** Not on PyPI. `git clone` is the only path.
  (The wheel does build cleanly, so this is a publishing gap, not a packaging
  one.)
- **Run any shipped graph against a real model.** All eight `grapharc run`
  commands hardcode `ScriptedChatModel`. There is no `--model` flag.
- **Call a tool.** `bind_tools` and `with_structured_output` both raise
  `NotImplementedError` on the gateway, and the harness is orphaned. GraphARC
  has no working story for an agent that calls a tool — the central use case of
  the multi-agent graphs it's about.

---

## What's actually worth keeping

Ranked by how much I'd fight to save it:

1. **`ClaudeCodeCLIChatModel`.** Extract it as its own package. It's genuinely
   novel, it's 139 lines, and it has zero dependency on the rest of GraphARC.
   Someone would use this tomorrow.
2. **The write-permission + deep-copy insight.** Ship it as a node decorator
   that composes with LangGraph instead of replacing it — that keeps async,
   `Command`, and `get_state` instead of amputating them.
3. **`verify_claim`'s anchor-before-model ordering.** A good ~60-line
   application pattern. Not library infrastructure, but a pattern worth
   documenting.
4. **The adversarial test suite.** `tests/test_harness_gate.py` is the best
   code in the repo — it's why the ctypes hole is embarrassing rather than
   invisible.

Everything else is either LangGraph's feature listed as GraphARC's, or
well-written prose over ~15 lines of code. The repo is 16% docstrings, and the
essays are consistently more sophisticated than the code beneath them.

---

## If you want this to be real

In rough priority order:

1. **Delete the false claims from the README today.** The sandbox sentence is
   the urgent one — it's a security claim that a three-line stdlib escape
   disproves.
2. **Hook `ctypes.*` audit events and scrub the child's environment**, or
   demote the sandbox to "best-effort, not a boundary" and point at Docker.
3. **Make budgets real**: charge tokens automatically in the gateway via
   `UsageMetadataCallbackHandler`, and enforce `max_seconds` with a watchdog
   rather than a between-nodes check.
4. **Pick one thing and finish it.** The gateway is the strongest candidate:
   add `_stream`, `_agenerate`, `bind_tools`, retries, and publish it.
5. **Wire the harness into an example, or delete it.** 285 tested lines with
   zero callers is not an asset.
6. **Give memory a disk.** SQLite is an afternoon; it would make the one claim
   ("later runs see what changed") actually true.
7. **Run a real model through a shipped graph in CI**, behind the `live`
   marker. Right now nothing is known about how routers, convergence guards, or
   the verifier behave on real output — which is the regime the library claims
   to have engineered for.

The gap between what this repo *says* and what it *does* is the main problem.
The code underneath is smaller than the prose implies, but the parts that are
real are genuinely real — and they'd look better without the overclaims
standing next to them.
