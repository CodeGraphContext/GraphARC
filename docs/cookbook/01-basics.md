# Getting started — the graph kernel

Recipes for the part of GraphARC you touch first: a typed state model, nodes that
declare what they write, edges that decide what runs next, and the machinery that
stops a run before it costs you something.

Every snippet on this page was executed against this repo's `.venv` before it was
written down, and every block labelled *Output* is pasted from that run — not
predicted, not tidied. Exactly one snippet is an exception: the real-model call
near the end, which is labelled as unrun because it would spend money. Where output
would otherwise depend on a clock or a random id, the snippet prints a stable
projection instead and says which fields it dropped.

`tests/test_cookbook_basics.py` reproduces every recipe here and asserts these
exact strings, so the page cannot rot quietly.

Verified against `grapharc 0.1.5`, Python 3.14.6, `langgraph 1.2.9`,
`langchain-core 1.5.1`, `pydantic 2.13.4`.

Each snippet is a complete file. Save it and run it; nothing carries over between
recipes.

---

## How do I install it?

Not on PyPI. `git clone` is the install path.

```bash
git clone https://github.com/CodeGraphContext/GraphARC
cd GraphARC
uv sync --group dev
```

Check that it took:

```bash
uv run grapharc --version
```

Output:

```
grapharc 0.1.5
```

Everything below uses only the base install — no API key, no network, no optional
extras.

---

## How do I build and run my first graph?

Four moving parts: a state schema, a function, edges wiring it between `START` and
`END`, and `compile().invoke()`.

```python
from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    question: str
    answer: str = ""


def answer(state: State) -> dict:
    return {"answer": f"42 (you asked: {state.question})"}


g = GraphARC(State, name="hello")
g.add_node("answer", answer, writes={"answer"})
g.add_edge(START, "answer")
g.add_edge("answer", END)

print(g.compile().invoke({"question": "meaning of life"}))
```

Output:

```
{'question': 'meaning of life', 'answer': '42 (you asked: meaning of life)'}
```

A node takes the state **model** (not a dict) and returns a **dict** of updates.
It never assigns to `state`; the returned dict is the only way anything leaves a
node.

**Where the names live.** `GraphARC`, `GraphARCState`, `Budget`,
`BudgetExceeded`, `BudgetMeter` and `WritePermissionError` are on the top-level
`grapharc` package. `START` and `END` are not — they come from
`grapharc.runtime.graph`, which is also where `StateTypeError`, `GraphCycleError`,
`GraphRoutingError`, `MissingRunContextError`, `AsyncNodeError` and `RunContext`
live. `grapharc.runtime` re-exports most of those, but **not** `START`, `END` or
`GraphCycleError`, so `from grapharc.runtime.graph import ...` is the import that
always works.

---

## Why is a field missing from the result dict?

`invoke()` returns a plain dict of *channel values*, and a field that was never in
the input and never written by a node is not one. Its schema default is not
materialised for you.

```python
from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    question: str
    answer: str = ""
    untouched: str = "default never written"


def answer(state: State) -> dict:
    return {"answer": "42"}


g = GraphARC(State, name="keys")
g.add_node("answer", answer, writes={"answer"})
g.add_edge(START, "answer")
g.add_edge("answer", END)

result = g.compile().invoke({"question": "?"})
print(result)
print(State(**result))
```

Output:

```
{'question': '?', 'answer': '42'}
question='?' answer='42' untouched='default never written'
```

**Why it works this way.** Inside the graph your nodes always see a fully
constructed `State`, defaults and all — the model is rebuilt on the way into every
node. The gap is only at the exit: LangGraph hands back the channel dict, not a
model. `State(**result)` is the one-line fix, and it is worth doing at the
boundary of your program anyway: it is also the moment your state model's own
validators finally run (see the `@field_validator` recipe below).

---

## How do I say what a node is allowed to write?

`writes=` is an allowlist, not documentation. A write outside it raises, and the
update never reaches state.

```python
from grapharc import GraphARC, GraphARCState, WritePermissionError
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    draft: str = ""
    published: str = ""


def write_draft(state: State) -> dict:
    return {"draft": "hello", "published": "hello"}   # published is not declared


g = GraphARC(State, name="perms")
g.add_node("write_draft", write_draft, writes={"draft"})
g.add_edge(START, "write_draft")
g.add_edge("write_draft", END)

try:
    g.compile().invoke({})
except WritePermissionError as exc:
    print(exc)

# A write declared against a field the schema does not have fails earlier still —
# when the node is added, not when it runs.
try:
    GraphARC(State, name="perms").add_node("x", write_draft, writes={"publised"})
except WritePermissionError as exc:
    print(exc)
```

Output:

```
node 'write_draft' wrote undeclared fields ['published']; declared writes: ['draft']
node 'x' declares writes to unknown state fields: ['publised']
```

**Why it works this way.** A node that writes nothing still needs the parameter:
`writes=set()`. The check is on the keys of the returned dict, so returning `None`
is the idiomatic "I wrote nothing" and is always legal.

Note the asymmetry in *when* the two failures land. A typo in the field name is
caught at `add_node` — before you can even build the graph — because the schema is
known then. A node writing a real field it did not declare can only be caught when
it actually returns, so that one is a run-time error. Neither is a warning; both
stop the run.

---

## How do I stop a node from mutating state behind my back?

You do not have to: every node gets a deep copy, so in-place mutation of a nested
model is invisible to everyone else.

```python
from pydantic import BaseModel

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class Report(BaseModel):
    title: str = "untitled"


class State(GraphARCState):
    report: Report = Report()
    seen: str = ""


def sneak(state: State) -> dict:
    state.report.title = "rewritten in place"   # not a declared write
    return {"seen": state.report.title}


g = GraphARC(State, name="isolation")
g.add_node("sneak", sneak, writes={"seen"})
g.add_edge(START, "sneak")
g.add_edge("sneak", END)

result = g.compile().invoke({"report": Report(title="original")})
print("node saw: ", result["seen"])
print("state has:", result["report"].title)
```

Output:

```
node saw:  rewritten in place
state has: original
```

**Why it works this way.** Pydantic passes nested models by reference, so without
the copy a node could rewrite `state.report.title` and have it stick — a write
that no `writes=` declaration ever approved and no trace ever recorded. The copy
makes the returned dict the only channel out. The cost is real: a `model_copy(deep=True)`
per node execution, so a state carrying a 50 MB blob pays for that blob on every
hop. Keep bulk data behind a path or an id in state, not in state.

Note that the mutation is *not* an error — it is just local. If you want the
change to land, return it: `{"report": Report(title="rewritten")}` with `report`
in `writes`.

---

## How do I get a bad value rejected before it reaches the next node?

You already have it. Every value a node returns is validated against the declared
type of its field at the node boundary, and the *validated* value is what gets
written.

```python
from typing import Annotated

from pydantic import Field

from grapharc import GraphARC, GraphARCState
from grapharc.runtime import StateTypeError
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    count: int = 0
    retries: Annotated[int, Field(ge=0)] = 0


def one_node(fn, *, writes):
    g = GraphARC(State, name="types")
    g.add_node("n", fn, writes=writes)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


# 1. The wrong type is refused, named.
try:
    one_node(lambda s: {"count": "seven"}, writes={"count"}).invoke({})
except StateTypeError as exc:
    print(exc)

# 2. A constraint carried in the annotation bites too.
try:
    one_node(lambda s: {"retries": -1}, writes={"retries"}).invoke({})
except StateTypeError as exc:
    print(exc)

# 3. What lands in state is the *validated* value, not what the node returned.
out = one_node(lambda s: {"count": "7"}, writes={"count"}).invoke({})
print(repr(out["count"]))
```

Output:

```
node 'n' wrote 'count' with a value the state schema rejects: expected int, got str ('seven'); Input should be a valid integer, unable to parse string as an integer
node 'n' wrote 'retries' with a value the state schema rejects: expected int, got int (-1); Input should be greater than or equal to 0
7
```

**Why it works this way.** The error names the node, the field, the declared type,
the type that arrived and the value, because when this fires at 3am the question is
always "which node did it". Case 3 is the part people miss: writing `"7"` into an
`int` field is *accepted* — Pydantic's default coercion — but what lands is the
integer `7`, not the string. A schema that says `int` means the result holds an
`int`.

This includes the last node before `END`, so a bad type cannot escape into the
result.

---

## Does my state model's own `@field_validator` run when a node writes?

No. This is the sharpest edge in the kernel, so here is exactly where it bites.

```python
from pydantic import field_validator

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    slug: str = "ok"

    @field_validator("slug")
    @classmethod
    def must_be_lower(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("slug must be lowercase")
        return v


def shout(state: State) -> dict:
    return {"slug": "NOT-LOWER"}


def read_it(state: State) -> dict:
    return {}


# Constructing the model directly does run the validator:
try:
    State(slug="NOT-LOWER")
except Exception as exc:
    print(type(exc).__name__, "on direct construction")

# Writing it from the last node before END does not:
g = GraphARC(State, name="gap")
g.add_node("shout", shout, writes={"slug"})
g.add_edge(START, "shout")
g.add_edge("shout", END)
print("result:", g.compile().invoke({}))

# Add a node downstream and the model is rebuilt on the way in, so it does:
g2 = GraphARC(State, name="gap2")
g2.add_node("shout", shout, writes={"slug"})
g2.add_node("read_it", read_it, writes=set())
g2.add_edge(START, "shout")
g2.add_edge("shout", "read_it")
g2.add_edge("read_it", END)
try:
    g2.compile().invoke({})
except Exception as exc:
    print(type(exc).__name__, "when the next node receives the state")
```

Output:

```
ValidationError on direct construction
result: {'slug': 'NOT-LOWER'}
ValidationError when the next node receives the state
```

**Why it works this way.** Write-time validation is built per field from that
field's *annotation*, and the state model itself is never constructed — only the one
field's value is. A `@field_validator` declared on the state model is a method on
the model, so it has nothing to run against.

So the rule to hold in your head: **`writes=` is GraphARC's; the types are
Pydantic's, and only the annotation half of Pydantic.** Two things follow.

**Prefer `Annotated[...]` constraints for anything you want enforced per write.**
`Field(ge=0)`, `Field(pattern=...)`, `Field(min_length=1)`, `Literal[...]` and
`StringConstraints(to_lower=True)` all live in the annotation and all bite at write
time.

**Or push the invariant into a nested model.** The annotation *is* that model, so
validating the write constructs it — and its validators run:

```python
from pydantic import BaseModel, field_validator

from grapharc import GraphARC, GraphARCState
from grapharc.runtime import StateTypeError
from grapharc.runtime.graph import END, START


class Article(BaseModel):
    slug: str

    @field_validator("slug")
    @classmethod
    def must_be_lower(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("slug must be lowercase")
        return v


class State(GraphARCState):
    article: Article | None = None


def publish(state: State) -> dict:
    return {"article": {"slug": "NOT-LOWER"}}


g = GraphARC(State, name="nested")
g.add_node("publish", publish, writes={"article"})
g.add_edge(START, "publish")
g.add_edge("publish", END)

try:
    g.compile().invoke({})
except StateTypeError as exc:
    print(exc)
```

Output:

```
node 'publish' wrote 'article' with a value the state schema rejects: expected __main__.Article | None, got dict ({'slug': 'NOT-LOWER'}); Value error, slug must be lowercase (at article.slug)
```

(`__main__.Article` is just where the class was defined in the file you ran.)

So the gap is narrower than "validators don't run": it is specifically the *state
model's own* validators. Anything one level down is fine.

If you do keep a validator on the state model, know that a violation written by the
*last* node before `END` reaches your result unchecked — every earlier node's
violation is caught when the next node rebuilds the model. `State(**result)` at your
program's boundary closes that, which is the same line the missing-fields recipe
already recommends.

---

## How do I stop a graph that will not stop itself?

Give it a `Budget`. `max_iterations` is charged once per node execution and checked
before every node, so a router that never says stop still stops.

```python
from grapharc import Budget, BudgetExceeded, GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    spins: int = 0


def spin(state: State) -> dict:
    return {"spins": state.spins + 1}   # never converges


def again(state: State) -> str:
    return "go"                          # the router never says stop


g = GraphARC(State, name="runaway", budget=Budget(max_iterations=5))
g.add_node("spin", spin, writes={"spins"})
g.add_edge(START, "spin")
g.add_conditional_edge("spin", again, {"go": "spin", "stop": END})

compiled = g.compile()
try:
    compiled.invoke({})
except BudgetExceeded as exc:
    print("stopped:", exc.reason)

print("iterations charged:", compiled.last_run.meter.iterations)
```

Output:

```
stopped: max_iterations reached (5/5)
iterations charged: 5
```

**Why it works this way.** Crossing a budget *raises*; it does not return partial
state. That is deliberate — a truncated answer that looks like a real one is worse
than an exception — but it means the caller, not the graph, decides what to do
about it. If you want a partial result you have two options: catch `BudgetExceeded`
and read the checkpoint (see the resume recipe), or build the stop condition into
the graph itself so the run ends through `END` with a recorded reason (see the next
two recipes). The budget is the ceiling, not the plan.

`compiled.last_run` holds the `RunContext` of the most recent `invoke`, which is
where `.meter` lives. It is overwritten by the next `invoke` on the same compiled
graph, so read it before you run again.

---

## How do I cap tokens and wall-clock time?

Same `Budget`. Neither of these needs your nodes to cooperate.

```python
import time

from langchain_core.messages import HumanMessage

from grapharc import Budget, BudgetExceeded, GraphARC, GraphARCState
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    reply: str = ""


def one_node(fn, budget):
    g = GraphARC(State, name="bounded", budget=budget)
    g.add_node("n", fn, writes={"reply"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def talk(state: State) -> dict:
    model = ScriptedChatModel(responses=["a reply long enough to cost something"])
    return {"reply": str(model.invoke([HumanMessage(content="hello")]).content)}


def nap(state: State) -> dict:
    time.sleep(5.0)
    return {"reply": "finished anyway"}


try:
    one_node(talk, Budget(max_tokens=5)).invoke({})
except BudgetExceeded as exc:
    print(exc.reason)

t0 = time.monotonic()
try:
    one_node(nap, Budget(max_seconds=0.25)).invoke({})
except BudgetExceeded as exc:
    print(exc.reason.split(" (")[0])
print(f"gave up after under a second: {time.monotonic() - t0 < 1.0}")
```

Output:

```
Error in MeterCallbackHandler.on_llm_end callback: BudgetExceeded('max_tokens reached (10/5)')
max_tokens reached (10/5)
max_seconds reached while node 'n' was running
gave up after under a second: True
```

That first line is LangChain logging the exception a callback raised. It is not a
second failure — it is the *mechanism*: GraphARC installs a usage callback for the
duration of every node, so any chat model invoked on that thread charges this run's
meter, and the ceiling is enforced at `on_llm_end` rather than waiting for the node
to return. The node never made a second call. (The snippet prints
`exc.reason.split(" (")[0]` for the time case only because the elapsed number in
the full message is a clock reading and would differ on your machine.)

**Why it works this way.** `max_seconds` is delivered as an *interrupt into* the
running node — SIGALRM on the main thread, `PyThreadState_SetAsyncExc` otherwise —
which is why a node parked in `time.sleep(5.0)` dies at 0.25s instead of 5s. Three
limits to know before you rely on it:

- A node running inside a C call cannot be unwound by the second mechanism; the
  exception lands when that call returns. Signals do interrupt blocking syscalls,
  but the signal path needs `invoke()` to be on the process's **main thread**. Drive
  a run from a `ThreadPoolExecutor` or a threaded request handler and the whole run
  silently falls back to the weaker mechanism.
- A node that catches every interrupt and never returns cannot be stopped from
  Python at all. The interrupt is re-armed every 50 ms, so swallowing has to succeed
  forever, but `while True:` around `except BaseException` will still hang you.
- Even when the node survives the interrupt, the deadline holds *at the node
  boundary*: an overrun node's writes never reach state.

`Budget` also carries `max_concurrency`, which caps parallel workers during fan-out;
see the fan-out recipe.

Budgets are per-`invoke`, not per-thread: resuming a thread starts a fresh meter.
You can override the graph's budget for one call with `invoke(..., budget=Budget(...))`.

---

## How do I forbid cycles until I actually need one?

`dag=True`. Cycles are something you earn, not something you start with.

```python
from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START, GraphCycleError


class State(GraphARCState):
    n: int = 0


def bump(state: State) -> dict:
    return {"n": state.n + 1}


g = GraphARC(State, name="pipeline", dag=True)
g.add_node("a", bump, writes={"n"})
g.add_node("b", bump, writes={"n"})
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", "a")        # accepted here...
g.add_edge("b", END)

try:
    g.compile()             # ...and refused here
except GraphCycleError as exc:
    print(exc)

# A conditional edge is refused the moment it is added: it is the mechanism a
# cycle would be built from, so dag mode does not wait for compile().
g2 = GraphARC(State, name="pipeline", dag=True)
g2.add_node("a", bump, writes={"n"})
try:
    g2.add_conditional_edge("a", lambda s: "go", {"go": END})
except GraphCycleError as exc:
    print(exc)
```

Output:

```
graph 'pipeline' is dag=True but has a cycle: a -> b -> a
graph 'pipeline' is dag=True: conditional edges are not allowed
```

**Why it works this way.** The error names the cycle it found, so you get
`a -> b -> a` rather than "graph is cyclic". Static edges can only be judged
together, so those wait for `compile()`; conditional and fan-out edges are refused
at `add_*` time because in dag mode there is no version of them that would be
legal later.

`GraphCycleError` is the one exception that `grapharc.runtime` does not re-export.
Import it from `grapharc.runtime.graph`.

---

## How do I write a loop that ends for a reason instead of running out of road?

Route with code, and record *why* you stopped in state. `ProgressGuard` gives you
three independent brakes — a target, a no-progress window, and a round cap — and
returns the first one that trips.

```python
from grapharc import Budget, GraphARC, GraphARCState
from grapharc.runtime.convergence import ProgressGuard, StopReason
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    round: int = 0
    findings: int = 0
    empty_rounds: int = 0
    termination_reason: str | None = None


HITS = [2, 1, 0, 0, 0]          # the well runs dry after two rounds
GUARD = ProgressGuard(target=10, max_rounds=8, max_empty_rounds=2)


def search(state: State) -> dict:
    hits = HITS[state.round] if state.round < len(HITS) else 0
    return {
        "round": state.round + 1,
        "findings": state.findings + hits,
        "empty_rounds": 0 if hits else state.empty_rounds + 1,
    }


def decide(state: State) -> dict:
    stop = GUARD.assess(
        round=state.round,
        total_findings=state.findings,
        empty_rounds=state.empty_rounds,
    )
    return {"termination_reason": stop.value if stop else None}


def route(state: State) -> str:
    return "stop" if state.termination_reason else "again"


g = GraphARC(State, name="investigate", budget=Budget(max_iterations=50))
g.add_node("search", search, writes={"round", "findings", "empty_rounds"})
g.add_node("decide", decide, writes={"termination_reason"})
g.add_edge(START, "search")
g.add_edge("search", "decide")
g.add_conditional_edge("decide", route, {"again": "search", "stop": END})

compiled = g.compile()
result = compiled.invoke({})
print(result)
print("stopped because:", StopReason(result["termination_reason"]))
print("iterations used:", compiled.last_run.meter.iterations, "of 50")
```

Output:

```
{'round': 4, 'findings': 3, 'empty_rounds': 2, 'termination_reason': 'no_progress'}
stopped because: no_progress
iterations used: 8 of 50
```

The target of 10 findings was never met. The loop stopped anyway, at round 4, with
`no_progress` on the record — and the 50-iteration budget was never touched. That
is the shape to aim for: the budget is the thing that catches your bug, not the
thing that ends your run.

**Why it works this way.** Routers are ordinary Python functions over typed state.
No model output is ever consulted to pick an edge, so no amount of prose in a model
reply can steer the graph — a model that writes `ROUTE TO: all_verified` into a
state field is writing a string, not choosing a branch.

What `add_conditional_edge` checks, and when: the mapping is read at declaration
time, so an empty one is refused, a target naming a node you never added raises
there and then, and a router annotated `-> Literal["again", "stop"]` (or with an
`Enum` return type) has those members held against your mapping's keys. A router
that declares nothing is left alone — the key it returns is only knowable when it
returns one — but that case is no longer a bare `KeyError` from inside LangGraph:
it raises `GraphRoutingError` naming the node, the key and the keys you declared.
`StopReason` is a `StrEnum`, so annotating your router with it moves that last
check to declaration time too.

---

## How do I see what actually happened?

Hand the graph a `TraceRecorder`. Every node execution writes a `start` line and
then either an `end` or an `error` line, as JSONL.

```python
import tempfile
from pathlib import Path

from grapharc import GraphARC, GraphARCState
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    items: list[str] = []
    total: int = 0


def load(state: State) -> dict:
    return {"items": ["a", "b", "c"]}


def count(state: State) -> dict:
    raise ValueError("the counter is not implemented yet")


tmp = Path(tempfile.mkdtemp())
trace = TraceRecorder(tmp / "trace.jsonl")

g = GraphARC(State, name="counter", trace=trace)
g.add_node("load", load, writes={"items"})
g.add_node("count", count, writes={"total"})
g.add_edge(START, "load")
g.add_edge("load", "count")
g.add_edge("count", END)

try:
    g.compile().invoke({}, thread_id="demo")
except ValueError:
    pass

VOLATILE = {"ts", "run_id", "thread_id", "duration_ms"}
for event in trace.read_events():
    print({k: v for k, v in event.model_dump(exclude_none=True).items() if k not in VOLATILE})
```

Output:

```
{'attempt': 1, 'graph': 'counter', 'node': 'topology', 'phase': 'topology', 'step': 0, 'state_delta': {'nodes': ['load', 'count'], 'edges': [['__start__', 'load', 'static'], ['load', 'count', 'static'], ['count', '__end__', 'static']]}}
{'attempt': 1, 'graph': 'counter', 'node': 'load', 'phase': 'start', 'step': 1}
{'attempt': 1, 'graph': 'counter', 'node': 'load', 'phase': 'end', 'step': 1, 'state_delta': {'items': ['a', 'b', 'c']}, 'tokens': 0}
{'attempt': 1, 'graph': 'counter', 'node': 'count', 'phase': 'start', 'step': 2}
{'attempt': 1, 'graph': 'counter', 'node': 'count', 'phase': 'error', 'step': 2, 'tokens': 0, 'error': "ValueError('the counter is not implemented yet')"}
```

The four fields the snippet filtered out are on every line too: `ts` (ISO-8601 UTC),
`run_id`, `thread_id`, and `duration_ms` on `end`/`error`. They are omitted from the
printout only because they differ every run.

So, by phase:

- **`topology`** is written once per entry, before any node runs: the graph's declared
  nodes and edges (conditional routes included, tagged by kind). It is what lets a
  diagram show the whole orchestration — branches not taken included — rather than
  only the path that happened to run. It carries `step: 0` on every attempt: it
  states shape, not order.

- **`start`** carries identity and nothing else: run, thread, attempt, graph, node,
  step, timestamp. It is written *before* the node body, so it exists even when the
  node never returns.
- **`end`** adds `state_delta` (exactly the validated update that was applied),
  `duration_ms`, and `tokens` charged during that node.
- **`error`** adds `duration_ms`, `error` — `repr()` of the exception, so the type is
  preserved — and `tokens`, the spend charged during that node before it failed.
  There is no `state_delta`, because a node that raised wrote nothing. The token
  count is there for the same reason `end` carries one: a run stopped *for*
  overspending used to report having spent nothing, because the only number the
  audit trail read was on the event an interrupted node never writes.

**Why it works this way.** `start` and `end` share a step number; the pair is the
node execution. That means step numbers do not order the file — read events in file
order, which is the order they were observed (the recorder appends under a lock).
Nodes may emit their own phases on the same recorder, so treat the phase set as
open rather than exhaustive.

Long values are truncated at 2000 characters with a `…[truncated N chars]` marker,
and non-JSON-able values fall back to `repr()`. A trace is an audit trail, not a
serialisation format: do not plan to reconstruct a 40k-token string out of it.

The `grapharc trace`, `grapharc metrics` and `grapharc viz` commands read this exact
file, which is why the dashboard and the audit trail cannot disagree.

---

## How do I see what a run spent, from inside a node?

Give your node a second parameter. Any node function with two or more parameters
receives the `RunContext` as the second one.

```python
from langchain_core.messages import HumanMessage

from grapharc import Budget, GraphARC, GraphARCState
from grapharc.runtime.graph import END, START, RunContext
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    reply: str = ""
    spent_so_far: int = 0


def talk(state: State, ctx: RunContext) -> dict:      # two parameters => you get ctx
    model = ScriptedChatModel(responses=["a scripted reply"])
    reply = str(model.invoke([HumanMessage(content="hi")]).content)
    return {"reply": reply, "spent_so_far": ctx.meter.tokens}


g = GraphARC(State, name="ctx", budget=Budget(max_tokens=1000))
g.add_node("talk", talk, writes={"reply", "spent_so_far"})
g.add_edge(START, "talk")
g.add_edge("talk", END)

compiled = g.compile()
print(compiled.invoke({}))
print("iterations:", compiled.last_run.meter.iterations)
print("tokens:    ", compiled.last_run.meter.tokens)
```

Output:

```
{'reply': 'a scripted reply', 'spent_so_far': 5}
iterations: 1
tokens:     5
```

Note that `spent_so_far` is 5 without the node charging anything by hand — the
usage callback did it. `RunContext` also carries `run_id`, `thread_id`, `attempt`
and `graph`, which is what you want when a node writes to something outside the
graph and needs to stamp it with the run that produced it.

**Why it works this way.** The two-parameter rule is detected from the function
signature, so a `def node(state)` and a `def node(state, ctx)` are both valid and
you never register which you wrote. The trap is real though: adding a defaulted
second parameter for your own reasons — `def node(state, retries=3)` — silently
turns it into a ctx parameter, and `retries` will be a `RunContext`. Keep extra
arguments in a closure or in state.

If you charge tokens by hand, pass the message: `ctx.meter.charge_tokens(n, source=message)`.
A bare `charge_tokens(n)` is always counted, even when the callback already metered
that exact call, so hand-metering a call that was already automatic pays twice.

---

## How do I resume a run that died halfway?

Compile with a checkpointer, then call `invoke(None, thread_id=...)` to resume that
thread from its last checkpoint.

```python
import sqlite3
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    url: str
    data: str = ""
    saved: bool = False


fetches: list[str] = []
crash_once = [True]


def fetch(state: State) -> dict:
    fetches.append(state.url)            # the expensive step
    return {"data": f"payload from {state.url}"}


def save(state: State) -> dict:
    if crash_once[0]:
        crash_once[0] = False
        raise RuntimeError("disk full")
    return {"saved": True}


tmp = Path(tempfile.mkdtemp())
conn = sqlite3.connect(tmp / "checkpoints.sqlite", check_same_thread=False)
compiled = (
    GraphARC(State, name="fetch_save")
    .add_node("fetch", fetch, writes={"data"})
    .add_node("save", save, writes={"saved"})
    .add_edge(START, "fetch")
    .add_edge("fetch", "save")
    .add_edge("save", END)
    .compile(checkpointer=SqliteSaver(conn))
)

try:
    compiled.invoke({"url": "https://example.invalid/doc"}, thread_id="t1")
except RuntimeError as exc:
    print("attempt 1 died:", exc)

print("next to run:", compiled.get_state("t1").next)

result = compiled.invoke(None, thread_id="t1")     # input=None means "resume"
print("attempt 2:", result)
print("fetches:", fetches)
conn.close()
```

Output:

```
attempt 1 died: disk full
next to run: ('save',)
attempt 2: {'url': 'https://example.invalid/doc', 'data': 'payload from https://example.invalid/doc', 'saved': True}
fetches: ['https://example.invalid/doc']
```

Exactly one fetch across both attempts: the resume picked up at `save`, which is
what `get_state("t1").next` told you it would do. `SqliteSaver` needs
`check_same_thread=False` because LangGraph runs nodes on worker threads.

**Why it works this way.** Crash-safe resume is LangGraph's, not GraphARC's — the
checkpointer you pass to `compile()` goes straight through. What GraphARC adds is
trace continuity: after a resume, step numbers continue from the thread's history
and `attempt` increments, so `(thread_id, step)` stays unique across attempts and
one logical thread is stitchable across restarts. Hand the same graph a
`TraceRecorder` and you can see it:

```python
import sqlite3
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc import GraphARC, GraphARCState
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    data: str = ""
    saved: bool = False


crash_once = [True]


def fetch(state: State) -> dict:
    return {"data": "payload"}


def save(state: State) -> dict:
    if crash_once[0]:
        crash_once[0] = False
        raise RuntimeError("disk full")
    return {"saved": True}


tmp = Path(tempfile.mkdtemp())
trace = TraceRecorder(tmp / "trace.jsonl")
conn = sqlite3.connect(tmp / "checkpoints.sqlite", check_same_thread=False)

g = GraphARC(State, name="resume", trace=trace)
g.add_node("fetch", fetch, writes={"data"})
g.add_node("save", save, writes={"saved"})
g.add_edge(START, "fetch")
g.add_edge("fetch", "save")
g.add_edge("save", END)
compiled = g.compile(checkpointer=SqliteSaver(conn))

try:
    compiled.invoke({}, thread_id="t1")
except RuntimeError:
    pass
compiled.invoke(None, thread_id="t1")

for e in trace.read_events():
    print(f"attempt {e.attempt}  step {e.step}  {e.node:<6} {e.phase}")
conn.close()
```

Output:

```
attempt 1  step 0  topology topology
attempt 1  step 1  fetch  start
attempt 1  step 1  fetch  end
attempt 1  step 2  save   start
attempt 1  step 2  save   error
attempt 2  step 0  topology topology
attempt 2  step 3  save   start
attempt 2  step 3  save   end
```

The resumed attempt's *work* starts at step 3 rather than restarting the numbering,
and `fetch` has no attempt-2 line because it did not re-run. Each attempt restates
the graph's topology at step 0 — shape, not order — which is why step comparisons
across attempts filter that phase out.

---

## Does that survive a real process kill?

Partly, and the gap is worth knowing before you rely on it. LangGraph's default
checkpoint durability is `"async"` — the checkpoint for a completed step is written
in the background while the next step starts — and `CompiledGraphARC.invoke()` does
not expose the `durability` parameter to change it.

Save as `resume_after_kill.py` and run it:

```python
"""Resume after a hard process kill (SIGKILL), not an in-process exception."""

import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    url: str
    data: str = ""
    saved: bool = False


def build(db: str):
    conn = sqlite3.connect(db, check_same_thread=False)

    def fetch(state: State) -> dict:
        Path(db + ".fetches").open("a").write("fetch\n")
        return {"data": f"payload from {state.url}"}

    def save(state: State) -> dict:
        if os.environ.get("KILL_ME"):
            os.kill(os.getpid(), signal.SIGKILL)
        return {"saved": True}

    return conn, (
        GraphARC(State, name="fetch_save")
        .add_node("fetch", fetch, writes={"data"})
        .add_node("save", save, writes={"saved"})
        .add_edge(START, "fetch")
        .add_edge("fetch", "save")
        .add_edge("save", END)
        .compile(checkpointer=SqliteSaver(conn))
    )


if len(sys.argv) > 1 and sys.argv[1] == "child":
    conn, compiled = build(sys.argv[2])
    compiled.invoke({"url": "https://example.invalid/doc"}, thread_id="t1")
else:
    db = str(Path(tempfile.mkdtemp()) / "checkpoints.sqlite")
    child = subprocess.run(
        [sys.executable, __file__, "child", db], env={**os.environ, "KILL_ME": "1"}
    )
    print("child killed by signal:", -child.returncode)

    conn, compiled = build(db)
    history = list(compiled.get_state_history("t1"))
    print("a checkpoint that resumes at 'save':", any(s.next == ("save",) for s in history))
    print("resumed:", compiled.invoke(None, thread_id="t1")["saved"])
    print("times fetch ran:", Path(db + ".fetches").read_text().count("fetch"))
    conn.close()
```

Output:

```
child killed by signal: 9
a checkpoint that resumes at 'save': False
resumed: True
times fetch ran: 2
```

The run resumed and produced exactly one result — but `fetch` ran **twice**. No
checkpoint reached disk saying "`fetch` is done, start at `save`", so the resume
replayed from the beginning.

(The snippet asks whether such a checkpoint exists rather than printing the whole
history, because *how much* got written before the kill is itself a race — the
step-0 checkpoint sometimes lands and sometimes does not. What is stable across
runs is that the one that would have skipped `fetch` never does.)

**What to do about it.** Make nodes idempotent, or make the expensive step's effect
externally deduplicated (write to a content-addressed path, upsert on a key, check
before you fetch). That is the right answer regardless of durability mode: a node
can also be re-run by a retry or by a resume from an earlier checkpoint you chose
deliberately. Do not treat "it was checkpointed" as "it will not run again". If you
need the stronger guarantee today, the `durability="sync"` option exists on
LangGraph's `Pregel.invoke` and would keep the post-`fetch` checkpoint, but reaching
it means bypassing `CompiledGraphARC.invoke()` — and `.inner.invoke()` fails closed
(next recipe), so there is no supported path to it right now.

---

## Why can't I just call `.inner.invoke()`?

Because a run with no budget and no trace is not a GraphARC run, and failing
silently into one is the bug this prevents.

```python
from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START, MissingRunContextError


class State(GraphARCState):
    n: int = 0


g = GraphARC(State, name="closed")
g.add_node("bump", lambda s: {"n": s.n + 1}, writes={"n"})
g.add_edge(START, "bump")
g.add_edge("bump", END)
compiled = g.compile()

try:
    compiled.inner.invoke({})       # raw LangGraph entry point
except MissingRunContextError as exc:
    print(exc)
```

Output:

```
node 'bump' executed without a GraphARC run context; drive the graph via CompiledGraphARC.invoke()/stream()/ainvoke()/astream() — raw LangGraph entry points would silently bypass budgets and traces
```

**Why it works this way.** `.inner` is an *inspection* escape hatch, not an
execution one. `compiled.inner.get_graph().draw_mermaid()` works fine; so does
anything else that reads. Only the entry points that would run nodes fail closed.
GraphARC already wraps `get_state`, `get_state_history` and `update_state` on the
compiled object, so you rarely need `.inner` at all — and the wrapped
`update_state` type-checks your values and, when you pass `as_node=`, applies that
node's write allowlist.

---

## How do I fan out across workers without one of them sinking the run?

`add_fanout_edge` dispatches parallel `Send`s. `run_guarded` turns a worker's crash
or hang into data. Use both: the first alone gives you parallelism with a shared
fate.

```python
import operator
import time
from typing import Annotated

from pydantic import BaseModel

from grapharc import Budget, GraphARC, GraphARCState
from grapharc.runtime.fanout import WorkerResult, run_guarded
from grapharc.runtime.graph import END, START


class Shard(BaseModel):
    name: str
    words: list[str]
    fail: bool = False
    hang: float = 0.0


class State(GraphARCState):
    words: list[str] = []
    results: Annotated[list[WorkerResult], operator.add] = []
    counted: int = 0
    failures: list[str] = []


def prepare(state: State) -> None:
    return None


def dispatch(state: State) -> list[tuple[str, Shard]]:
    return [
        ("worker", Shard(name="w0", words=state.words[0::3])),
        ("worker", Shard(name="w1", words=state.words[1::3], fail=True)),
        ("worker", Shard(name="w2", words=state.words[2::3], hang=5.0)),
    ]


def worker(shard: Shard) -> dict:
    def job() -> list[dict]:
        if shard.fail:
            raise RuntimeError("shard parser blew up")
        if shard.hang:
            time.sleep(shard.hang)
        return [{"word": w} for w in shard.words]

    return {"results": [run_guarded(job, worker=shard.name, timeout_seconds=0.2)]}


def collect(state: State) -> dict:
    ok = [r for r in state.results if r.ok]
    return {
        "counted": sum(len(r.evidence) for r in ok),
        "failures": [f"{r.worker}: {r.error}" for r in state.results if not r.ok],
    }


g = GraphARC(State, name="fanout", budget=Budget(max_concurrency=2))
g.add_node("prepare", prepare, writes=set())
g.add_node("worker", worker, writes={"results"}, input_schema=Shard)
g.add_node("collect", collect, writes={"counted", "failures"})
g.add_edge(START, "prepare")
g.add_fanout_edge("prepare", dispatch)
g.add_edge("worker", "collect")
g.add_edge("collect", END)

out = g.compile().invoke({"words": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]})
print("counted: ", out["counted"])
for line in out["failures"]:
    print("failed:  ", line)
```

Output:

```
counted:  2
failed:   w1: RuntimeError('shard parser blew up')
failed:   w2: timeout after 0.2s
```

Two of three workers died, one crashing and one hanging, and the run still produced
an answer with both failures attributed by cause.

The pieces that make that work:

- **`input_schema=Shard`** types the payload a worker receives. A fan-out worker is
  called with the `Send` payload, *not* the graph state, so its parameter is a
  `Shard` — which is why `worker` above never reads `state`.
- **A reducer on the target field.** `Annotated[list[WorkerResult], operator.add]`
  is what lets three parallel workers all write `results` without clobbering each
  other. Drop the `Annotated[...]` and LangGraph raises `InvalidUpdateError: At key
  'results': Can receive only one value per step.` — it does not silently pick a
  winner.
- **`run_guarded`** runs the job on a daemon thread joined with a timeout. A raised
  exception becomes `WorkerResult(ok=False, error=repr(exc))`; a hang becomes
  `ok=False, error="timeout after 0.2s"`.
- **`Budget(max_concurrency=2)`** caps how many workers run at once.

**What `run_guarded` does not do.** A hung worker is *abandoned*, not killed — the
daemon thread is still sleeping out its five seconds after the batch moves on, and
a result it produces late is ignored. That is the right trade for correctness (late
evidence never changes a published answer) but it means a worker holding a lock or
a file handle keeps holding it. It is also why the fan-out timeout is per worker and
separate from `Budget(max_seconds=...)`, which is per run.

Two failure modes to see for yourself:

```python
import operator
from typing import Annotated

from pydantic import BaseModel

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START, GraphRoutingError


class Shard(BaseModel):
    name: str
    fail: bool = False


class State(GraphARCState):
    seen: Annotated[list[str], operator.add] = []


def prepare(state: State) -> None:
    return None


def dispatch(state: State) -> list[tuple[str, Shard]]:
    return [("worker", Shard(name="w0")), ("worker", Shard(name="w1", fail=True))]


def worker(shard: Shard) -> dict:
    if shard.fail:
        raise RuntimeError("shard parser blew up")
    return {"seen": [shard.name]}


def build(dispatcher):
    g = GraphARC(State, name="unguarded")
    g.add_node("prepare", prepare, writes=set())
    g.add_node("worker", worker, writes={"seen"}, input_schema=Shard)
    g.add_edge(START, "prepare")
    g.add_fanout_edge("prepare", dispatcher)
    g.add_edge("worker", END)
    return g.compile()


# Without run_guarded, one bad worker takes the whole run with it.
try:
    build(dispatch).invoke({})
except RuntimeError as exc:
    print("run died:", exc)

# A dispatcher naming a node the graph does not have fails closed rather than
# silently dropping the shard.
try:
    build(lambda s: [("wroker", Shard(name="w0"))]).invoke({})
except GraphRoutingError as exc:
    print(exc)
```

Output:

```
run died: shard parser blew up
the fan-out dispatcher on node 'prepare' routed to Send(node='wroker', ...), but 'wroker' is not a node of graph 'unguarded'; LangGraph drops an unknown Send target and the run continues as if the routing had not happened. Valid Send targets: 'prepare', 'worker' — END is not one, because a Send has to name a node that runs
```

The second one is the one that would have cost you a week. Plain LangGraph logs
`Ignoring unknown node name` and carries on, so the shard is simply never worked and
your answer is quietly built from two thirds of the evidence. GraphARC raises.

---

## Can my nodes be `async`?

Yes — and if a graph has any `async def` node you must use `ainvoke`/`astream`, which
it will tell you.

```python
import asyncio

from grapharc import GraphARC, GraphARCState
from grapharc.runtime.graph import END, START, AsyncNodeError


class State(GraphARCState):
    reply: str = ""


async def fetch(state: State) -> dict:
    await asyncio.sleep(0)
    return {"reply": "from an async node"}


g = GraphARC(State, name="async")
g.add_node("fetch", fetch, writes={"reply"})
g.add_edge(START, "fetch")
g.add_edge("fetch", END)
compiled = g.compile()

print(asyncio.run(compiled.ainvoke({})))

try:
    compiled.invoke({})
except AsyncNodeError as exc:
    print(exc)
```

Output:

```
{'reply': 'from an async node'}
graph 'async' has async nodes ['fetch']; use ainvoke(), because invoke() has no event loop to await them on
```

**Why it works this way.** The rejection happens before any node runs, so a graph
mixing sync and async nodes cannot half-execute — plain LangGraph would run every
sync node first and then fail on the first coroutine. Sync nodes stay legal under
`ainvoke()`; LangGraph runs each on a worker thread.

The one behavioural difference: for an `async def` node, `max_seconds` is delivered
as task cancellation rather than a thread interrupt, so it lands at an `await`. An
async node doing blocking work between awaits is not interrupted until it yields.

(If you read `README.md` first and saw "No async" under Status, trust the snippet
above — it ran.)

---

## How do I swap the scripted model for a real one?

Everything above used `grapharc.testing.ScriptedChatModel`, which is a real
`BaseChatModel` that replays a fixed list of strings and reports plausible usage
metadata — so budgets, traces and the usage callback are all exercised for free.
Swapping in a real backend is a one-line change at the point of construction.

**The snippet below was NOT executed.** It calls a paid or subscription-metered
backend, and this cookbook does not run those. It is here for the shape only.

```python
from grapharc.gateway import get_model

# Claude Code CLI: uses your Claude subscription, no API key. Text completion only —
# bind_tools and with_structured_output raise NotImplementedError on this backend.
model = get_model("claude-cli/claude-sonnet-5")

# OpenRouter: needs OPENROUTER_API_KEY and the `openrouter` extra
# (uv sync --extra openrouter). Tool calling, structured output, streaming, async.
model = get_model("openrouter/anthropic/claude-haiku-4.5")

# OpenAI directly: needs OPENAI_API_KEY and the `openai` extra.
model = get_model("openai/gpt-4o-mini")

# Ollama: a model on this machine. No key, no bill — the one real backend you
# can run without an account. Needs the `ollama` extra and a pulled model.
model = get_model("ollama/llama3.1")
```

What you *can* check without spending anything is how a spec resolves:

```python
from grapharc.gateway import describe

print(describe("claude-cli/claude-sonnet-5"))
print(describe("openrouter/anthropic/claude-haiku-4.5"))
```

Output:

```
{'spec': 'claude-cli/claude-sonnet-5', 'backend': 'claude-cli', 'model': 'claude-sonnet-5'}
{'spec': 'openrouter/anthropic/claude-haiku-4.5', 'backend': 'openrouter', 'model': 'anthropic/claude-haiku-4.5'}
```

`describe` never constructs a model and never touches a credential. A mistyped
backend is rejected rather than folded into a model name, so `opnerouter/...` fails
here instead of failing much later as a nonsense Claude CLI call.

Once a real model is in a node, nothing else in this page changes. The usage
callback charges the run's meter whether the model is scripted or real, including
calls made deep inside library code your node merely called — which is the whole
reason `max_tokens` is worth setting.

---

## What this page did not cover

Deliberate omissions, so you know they exist rather than discovering them:

- `stream()` / `astream()` / `astream_events()` — the same discipline, incremental
  output. `.inner.stream()` fails closed exactly like `.inner.invoke()`.
- Returning a `langgraph` `Command` from a node for dynamic `goto` routing. Its
  `update` goes through the same allowlist and type checks a returned dict does, and
  its destination is checked against the graph's nodes.
- `update_state` for human-in-the-loop edits between steps.
- The model gateway's retry policies, spend meters and cost ceilings; verification
  (`verify_claim`); memory; the tool harness. Other pages.

Two limits of the kernel worth carrying with you, both stated where they bite above
but repeated here because they are the ones that surprise people:

1. A state model's own `@field_validator` / `@model_validator` does not run when a
   node writes — only the field's annotation is enforced, which does include a
   nested model's own validators. Rebuild the model at your program's boundary, or
   keep the invariant one level down.
2. `add_conditional_edge` checks its mapping when the edge is added — the targets,
   and a router that annotates what it returns. A router that annotates nothing is
   not second-guessed, so the key it returns is checked when it returns one; that
   is a `GraphRoutingError` naming the router, not a bare `KeyError`.
