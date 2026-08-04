# Serving and operations

Everything up to here was about getting a graph *right*. This section is about
keeping one running: awaiting things without blocking, routing at runtime,
stopping for a human, surviving a process restart, putting the whole thing
behind HTTP, and reading what happened afterwards.

Every Python snippet below was executed exactly as printed, against
`grapharc.testing.ScriptedChatModel`, and `tests/test_cookbook_serving.py`
re-runs each one and byte-compares it against the output block underneath. No
live model is called anywhere in this file; the one snippet that would need a
key says so and was not run.

The `console` transcripts are held to the same standard, marked the way
`02-models.md` marks its own. `<!-- verified: cli -->` above a transcript means
every command in it is re-run on every commit and byte-compared — run ids are
random and durations are wall-clock, so the test maps the former and masks the
latter, and every other character has to match. `<!-- verified: cli varies -->`
means every command is re-run and must succeed, but the output describes one
machine — `serve` binds a port, `models --check` probes the host — so its bytes
are the record of one real run rather than a guarantee about yours. The one
`agent` invocation always needs a live model; it is marked
`<!-- needs-credentials -->` and nothing here claims to have run it.

Three things to know before you start:

- **Async and sync are the same contract.** Write permissions, type checks,
  budgets and traces are one implementation. What changes is which entry point
  you use and how `max_seconds` is delivered.
- **Sessions are the durable layer.** A `CompiledGraphARC` plus a checkpointer
  gets you a resumable *thread*. A `Session` adds the things a thread has no
  opinion about: who is driving it, what was queued for it, whether a human
  still has to sign something off.
- **The trace file is the only record.** `replay`, `diff`, `cost`, `metrics`
  and `viz` all read the same JSONL. There is no second source of truth to
  disagree with it.

---

## How do I run a graph whose node has to `await` something?

Write the node `async def` and drive the graph with `ainvoke()`. Nothing else
changes — the node still returns a dict, still declares its writes, still gets
a deep-copied state.

```python
import asyncio

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    url: str
    body: str = ""


async def fetch(state: State) -> dict:
    await asyncio.sleep(0.01)          # a real client would await here
    return {"body": f"<html>{state.url}</html>"}


g = GraphARC(State, name="fetcher")
g.add_node("fetch", fetch, writes={"body"})
g.add_edge(START, "fetch")
g.add_edge("fetch", END)
graph = g.compile()

result = asyncio.run(graph.ainvoke({"url": "https://example.com"}))
print(result["body"])
```

```
<html>https://example.com</html>
```

Sync nodes stay legal on the async path — LangGraph puts them on its own worker
threads — so a graph may mix the two freely as long as you drive it with
`ainvoke()` / `astream()`.

**Why it works this way.** `max_seconds` is the one thing that genuinely
differs. On a sync node the ceiling is delivered as an interrupt to the thread
running it; on an `async def` node it is delivered as task cancellation,
because a signal aimed at "the thread" on an event loop would land in whichever
coroutine happened to be running. The practical consequence is stated in
`grapharc.runtime.graph._async_deadline`: cancellation arrives at an `await`, so
an async node doing blocking CPU work is not interrupted until it yields.

---

## Why does `invoke()` refuse my graph before it runs anything?

Because it found an `async def` node and there is no event loop to await it on.
The check happens at the entry point, before the first node.

```python
import asyncio

from grapharc.runtime.graph import END, START, AsyncNodeError, GraphARC
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    log: list[str] = []


def cheap(state: State) -> dict:
    print("cheap ran")
    return {"log": [*state.log, "cheap"]}


async def costly(state: State) -> dict:
    await asyncio.sleep(0)
    return {"log": [*state.log, "costly"]}


g = GraphARC(State, name="mixed")
g.add_node("cheap", cheap, writes={"log"})
g.add_node("costly", costly, writes={"log"})
g.add_edge(START, "cheap")
g.add_edge("cheap", "costly")
g.add_edge("costly", END)
graph = g.compile()

try:
    graph.invoke({})
except AsyncNodeError as exc:
    print(f"AsyncNodeError: {exc}")

print(asyncio.run(graph.ainvoke({}))["log"])
```

```
AsyncNodeError: graph 'mixed' has async nodes ['costly']; use ainvoke(), because invoke() has no event loop to await them on
cheap ran
['cheap', 'costly']
```

Read the output order carefully: `cheap ran` prints **once**, from the
`ainvoke()` call. The refused `invoke()` did not execute it. That is the whole
point of a pre-flight check — LangGraph on its own would have run every sync
node first and then died on the first coroutine with a `TypeError`, leaving you
with half a run and a confusing traceback.

`stream()` refuses the same way and names `astream()` in its message.

---

## How do I watch state land as the graph runs?

`astream()`. It takes the same `stream_mode` values LangGraph does; `"updates"`
gives you one chunk per node with just what that node wrote.

```python
import asyncio

from langchain_core.messages import HumanMessage

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    draft: str = ""
    final: str = ""


model = ScriptedChatModel(
    responses=["budgets cap iterations", "budgets cap iterations, tokens and time"]
)


async def draft(state: State) -> dict:
    reply = await model.ainvoke([HumanMessage(content=state.question)])
    return {"draft": str(reply.content)}


async def polish(state: State) -> dict:
    reply = await model.ainvoke([HumanMessage(content=f"polish: {state.draft}")])
    return {"final": str(reply.content)}


g = GraphARC(State, name="writer")
g.add_node("draft", draft, writes={"draft"})
g.add_node("polish", polish, writes={"final"})
g.add_edge(START, "draft")
g.add_edge("draft", "polish")
g.add_edge("polish", END)
graph = g.compile()


async def main() -> None:
    async for chunk in graph.astream(
        {"question": "how do budgets work?"}, stream_mode="updates"
    ):
        for node, delta in chunk.items():
            print(node, "->", delta)


asyncio.run(main())
```

```
draft -> {'draft': 'budgets cap iterations'}
polish -> {'final': 'budgets cap iterations, tokens and time'}
```

A `"updates"` chunk is a `{node_name: delta}` dict, not a `(name, delta)` pair —
unpacking it as two values raises `ValueError: not enough values to unpack`.
Pass a list of modes (`stream_mode=["updates", "checkpoints"]`) and the chunks
become `(mode, payload)` tuples instead. That second shape is what the session
runtime uses, because `checkpoints` is the only chunk that means "the superstep
is finished and on disk".

---

## How do I see every model call, live?

`astream_events()` is LangChain's event stream, driven through GraphARC's
budgeted path. It emits per-node *and* per-model-call events, so it is what you
want behind a "thinking…" UI or a token meter.

```python
import asyncio

from langchain_core.messages import HumanMessage

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    answer: str = ""


model = ScriptedChatModel(responses=["42"])


async def answer(state: State) -> dict:
    reply = await model.ainvoke([HumanMessage(content=state.question)])
    return {"answer": str(reply.content)}


g = GraphARC(State, name="qa")
g.add_node("answer", answer, writes={"answer"})
g.add_edge(START, "answer")
g.add_edge("answer", END)
graph = g.compile()


async def main() -> None:
    async for event in graph.astream_events({"question": "meaning of life?"}):
        if event["event"] == "on_chain_start":
            print("start", event["name"])
        elif event["event"] == "on_chat_model_end":
            usage = event["data"]["output"].usage_metadata
            print("model", event["name"], usage["total_tokens"], "tokens")


asyncio.run(main())
```

```
start LangGraph
start answer
model ScriptedChatModel 5 tokens
```

`version` defaults to `"v2"` and `"v1"` is accepted. `"v3"` is refused with a
`ValueError` rather than passed through, because LangGraph returns a stream
*object* for v3, not an async iterator — a different shape than this method's
contract.

---

## How do I stop someone bypassing the budget?

You do not have to. `CompiledGraphARC.inner` is the LangGraph object underneath
and is reachable, and driving it directly fails closed at the first node.

```python
import asyncio

from grapharc.runtime.graph import END, START, GraphARC, MissingRunContextError
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    x: int = 0


def bump(state: State) -> dict:
    return {"x": state.x + 1}


g = GraphARC(State, name="counter")
g.add_node("bump", bump, writes={"x"})
g.add_edge(START, "bump")
g.add_edge("bump", END)
graph = g.compile()

# The LangGraph object underneath is reachable — and refuses to run unbudgeted.
try:
    graph.inner.invoke({"x": 0})
except MissingRunContextError as exc:
    print("MissingRunContextError:", exc)


async def v3() -> None:
    async for _ in graph.astream_events({"x": 0}, version="v3"):
        pass


try:
    asyncio.run(v3())
except ValueError as exc:
    print("ValueError:", exc)
```

```
MissingRunContextError: node 'bump' executed without a GraphARC run context; drive the graph via CompiledGraphARC.invoke()/stream()/ainvoke()/astream() — raw LangGraph entry points would silently bypass budgets and traces
ValueError: astream_events supports version 'v1' or 'v2', got 'v3'
```

---

## How do I let a node pick the next node?

Return a `langgraph.types.Command` (re-exported from `grapharc.runtime.graph`).
Its `update` goes through exactly the checks a returned dict does, so dynamic
routing costs nothing in discipline.

```python
from grapharc.runtime.graph import END, START, Command, GraphARC
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    text: str
    verdict: str = ""
    outcome: str = ""


def triage(state: State) -> Command:
    if "urgent" in state.text:
        return Command(update={"verdict": "urgent"}, goto="escalate")
    return Command(update={"verdict": "routine"}, goto="archive")


def escalate(state: State) -> dict:
    return {"outcome": "paged the on-call"}


def archive(state: State) -> dict:
    return {"outcome": "filed"}


g = GraphARC(State, name="triage")
g.add_node("triage", triage, writes={"verdict"})
g.add_node("escalate", escalate, writes={"outcome"})
g.add_node("archive", archive, writes={"outcome"})
g.add_edge(START, "triage")
g.add_edge("escalate", END)
g.add_edge("archive", END)
graph = g.compile()

for text in ("urgent: disk full", "weekly digest"):
    result = graph.invoke({"text": text})
    print(f"{text!r} -> {result['verdict']}, {result['outcome']}")
```

```
'urgent: disk full' -> urgent, paged the on-call
'weekly digest' -> routine, filed
```

Note there is no `add_edge(START, ...)` for `escalate` or `archive` and no
conditional edge at all: the `goto` is the transition. The nodes still need
their outgoing edges to `END`.

`Command(update=...)` must be a dict, and `Command(graph=...)` is refused —
an update aimed at another graph's state cannot be checked against this
graph's declared writes.

---

## What happens when a `goto` names a node that does not exist?

It raises. This is the case worth knowing about, because plain LangGraph does
not raise: it logs `wrote to unknown channel branch:to:<x>, ignoring it` and
carries on as if the node had never routed.

```python
from grapharc.runtime.graph import END, START, Command, GraphARC, GraphRoutingError
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    text: str
    outcome: str = ""


def triage(state: State) -> Command:
    return Command(goto="esclate")  # typo


def escalate(state: State) -> dict:
    return {"outcome": "paged the on-call"}


g = GraphARC(State, name="triage")
g.add_node("triage", triage, writes={"outcome"})
g.add_node("escalate", escalate, writes={"outcome"})
g.add_edge(START, "triage")
g.add_edge("escalate", END)
graph = g.compile()

try:
    graph.invoke({"text": "urgent: disk full"})
except GraphRoutingError as exc:
    print(f"GraphRoutingError: {exc}")
```

```
GraphRoutingError: node 'triage' routed to 'esclate', which is not a node of graph 'triage'; LangGraph drops an unknown destination and the run continues as if the routing had not happened. Valid destinations: 'escalate', 'triage', END
```

**Why it works this way.** A transition that silently does *not* happen is the
same defect as one that happens unpermitted: the graph did something other than
what the node asked for and nobody was told. The check covers node names, `END`,
`Send` targets from an `add_fanout_edge` dispatcher, and sequences of those. It
also covers `Command(goto=None)`, which LangGraph cannot iterate at all.

**The stated limit:** this covers destinations GraphARC hands to LangGraph. A
`Command` passed as *input* to `invoke()`/`stream()` is outside it — those entry
points take a dict, a state model or `None`, and a `Command` there is
unsupported.

---

## How do I pause a graph, let a human edit state, and carry on?

Compile with a checkpointer, stream with `interrupt_before`, then use
`get_state` / `update_state` / resume-with-`None`.

```python
from langgraph.checkpoint.memory import InMemorySaver

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    topic: str
    draft: str = ""
    sent: str = ""


def write(state: State) -> dict:
    return {"draft": f"Dear customer, about {state.topic}..."}


def send(state: State) -> dict:
    return {"sent": state.draft}


g = GraphARC(State, name="mailer")
g.add_node("write", write, writes={"draft"})
g.add_node("send", send, writes={"sent"})
g.add_edge(START, "write")
g.add_edge("write", "send")
g.add_edge("send", END)
graph = g.compile(checkpointer=InMemorySaver())

# Run until `send` is next, then stop.
for _ in graph.stream({"topic": "the outage"}, thread_id="t1", interrupt_before=["send"]):
    pass

snapshot = graph.get_state("t1")
print("next    :", snapshot.next)
print("draft   :", snapshot.values["draft"])

# A human edits the draft, attributed to the node that produced it.
edited = "Dear customer, we are sorry about the outage."
graph.update_state("t1", {"draft": edited}, as_node="write")

# Resume: input=None picks up from the checkpoint.
result = graph.invoke(None, thread_id="t1")
print("sent    :", result["sent"])
```

```
next    : ('send',)
draft   : Dear customer, about the outage...
sent    : Dear customer, we are sorry about the outage.
```

**Sharp edge:** `interrupt_before` is a *stream* keyword, not an `invoke`
keyword and not a `compile` keyword. `invoke(...)` takes only
`input`, `thread_id`, `run_id` and `budget`; passing `interrupt_before=` to it
raises `TypeError`. Run the interrupted leg with `stream()` (draining the
iterator, as above) and resume with whichever you prefer.

`get_state` and friends need a checkpointer at compile time; without one
LangGraph raises `ValueError("No checkpointer set")`. They read and write
checkpoints rather than executing nodes, so they carry no budget and emit no
trace events.

---

## What does `update_state` actually check?

Field names and declared types always; the node's write allowlist only when you
say which node the edit is attributed to.

```python
from langgraph.checkpoint.memory import InMemorySaver

from grapharc.runtime.graph import END, START, GraphARC, StateTypeError, WritePermissionError
from grapharc.runtime.state import GraphARCState


class State(GraphARCState):
    topic: str
    draft: str = ""
    sent: str = ""


def write(state: State) -> dict:
    return {"draft": f"Dear customer, about {state.topic}..."}


def send(state: State) -> dict:
    return {"sent": state.draft}


g = GraphARC(State, name="mailer")
g.add_node("write", write, writes={"draft"})
g.add_node("send", send, writes={"sent"})
g.add_edge(START, "write")
g.add_edge("write", "send")
g.add_edge("send", END)
graph = g.compile(checkpointer=InMemorySaver())

for _ in graph.stream({"topic": "the outage"}, thread_id="t1", interrupt_before=["send"]):
    pass

# 1. Unknown field: refused whether or not you claim a node.
try:
    graph.update_state("t1", {"draftt": "oops"})
except WritePermissionError as exc:
    print("1:", exc)

# 2. Wrong type: refused.
try:
    graph.update_state("t1", {"draft": 42})
except StateTypeError as exc:
    print("2:", exc)

# 3. Claiming a node you are not allowed to write as: refused.
try:
    graph.update_state("t1", {"sent": "forged"}, as_node="write")
except WritePermissionError as exc:
    print("3:", exc)

# 4. No as_node: type-checked only, no allowlist to apply.
graph.update_state("t1", {"sent": "written from outside any node"})
print("4:", graph.get_state("t1").values["sent"])
```

```
1: update_state targets unknown state fields: ['draftt']
2: update_state wrote 'draft' with a value the state schema rejects: expected str, got int (42); Input should be a valid string
3: update_state(as_node='write') wrote undeclared fields ['sent']; declared writes: ['draft']
4: written from outside any node
```

**Why case 4 is not an allowlist violation.** A human editing state mid-run is
not a node, so there is no allowlist to apply. `as_node="write"` says "record
this as if `write` had done it", and then `write`'s declared writes are exactly
the right contract. The residual gap, stated rather than papered over: with
`as_node=None` LangGraph attributes the update to whichever node last ran, and
GraphARC does not reproduce that inference — so such an update is type-checked
but not allowlisted. If you want the allowlist, name the node.

---

## How do I run something a human has to approve?

Use a `Session`. The graph's state schema derives from `SessionState`, the
gated nodes are named on the `GraphSpec`, and the runtime holds the graph
*before* the gated node runs.

`grapharc.session.demo` ships a four-node graph (`ingest -> plan -> apply ->
report`) with `apply` gated, so you can see the shape before writing your own.
Importing the module is what registers the graph.

```python
import tempfile

from grapharc.session import SessionManager
from grapharc.session.demo import GRAPH_NAME  # importing registers the graph

root = tempfile.mkdtemp(prefix="cookbook-")

with SessionManager(root) as manager:
    session = manager.create(GRAPH_NAME)
    session.send("summarise the incident")

    first = session.run({})
    print("status  :", first.status.value)
    print("ran     :", first.nodes)
    print("holding :", [(h.node, h.action) for h in first.approvals])

    session.decide(approved=True, decided_by="ops")
    second = session.run()
    print("status  :", second.status.value)
    print("ran     :", second.nodes)
    print("outcome :", second.state["outcome"])
    print("log     :", second.state["log"])
```

```
status  : awaiting_approval
ran     : ('ingest', 'plan')
holding : [('apply', 'apply: draft the release note')]
status  : idle
ran     : ('apply', 'report')
outcome : applied: draft the release note
log     : ['ingest', 'plan', 'apply', 'report']
```

`TurnResult.nodes` is the honest answer to "what executed". `apply` is absent
from the first turn's list because its body never ran.

**Why it works this way.** `SessionManager(root)` puts a session store and a
checkpoint store in one directory, so "resume this session elsewhere" is "point
another `SessionManager` at the same directory". `run()` is synchronous and
occupies its caller until the session stops — the kernel grew `astream` while
the session layer was being written and an async turn is buildable, just not
built.

**The gate belongs to the session, not to the graph.** `session.graph.invoke(...)`
runs gated nodes with nothing holding them. If that matters to you, do not hand
the compiled graph to anything that will not go through a session.

---

## How do I reject a step without unwinding it?

`decide(approved=False)`. Nothing the gated node would have done has happened
yet, so a rejection costs nothing to honour — the graph walks *past* the node
rather than running it and undoing it.

```python
import tempfile

from grapharc.session import ApprovalRequired, SessionManager
from grapharc.session.demo import GRAPH_NAME

with SessionManager(tempfile.mkdtemp(prefix="cookbook-")) as manager:
    session = manager.create(GRAPH_NAME, session_id="incident-42")
    session.run({})

    # Running while a hold is open is refused before any work happens.
    try:
        session.run()
    except ApprovalRequired as exc:
        print("refused :", exc)

    session.decide(approved=False, decided_by="ops", reason="wrong quarter")
    turn = session.run()
    print("status  :", turn.status.value)
    print("ran     :", turn.nodes)
    print("skipped :", turn.skipped)
    print("log     :", turn.state["log"])
    print("outcome :", turn.state["outcome"])
```

```
refused : session 'incident-42' is holding node 'apply' awaiting approval '2f29726ed94f'; call decide() before run()
status  : idle
ran     : ('report',)
skipped : ('apply',)
log     : ['ingest', 'plan', 'report']
outcome : refused by ops: wrong quarter
```

(The approval request id is a fresh random value per hold; yours will differ.)

`log` is append-only in the demo graph, and `'apply'` is not in it — the node's
body never executed. What *did* run is the node's outgoing edges, filed against
its own pending task, which is what advances the graph. `report` then reads
`state.decision` and routes on the verdict.

The `ApprovalRequired` refusal happens before the session claims anything, so a
refused `run()` leaves the queue, the holds and the checkpoint exactly as it
found them.

---

## How do I resume a session after the process restarts?

`SessionManager.resume(session_id)` in the new process. Nothing about a session
lives in memory: status, queue, holds and audit trail are rows in SQLite, graph
state is in the checkpointer, and the graph itself is rebuilt by name from the
registry.

This snippet spawns two genuinely separate interpreters to prove it.

```python
"""Prove a session survives a process restart: two child interpreters, one directory."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="cookbook-"))

FIRST = """
import json, os, sys
from grapharc.session import SessionManager
from grapharc.session.demo import GRAPH_NAME   # importing registers the graph

with SessionManager(sys.argv[1]) as manager:
    session = manager.create(GRAPH_NAME, session_id="incident-42")
    session.send("summarise the incident")
    turn = session.run({})
    print(json.dumps({"pid": os.getpid(), "status": turn.status.value,
                      "ran": turn.nodes, "log": turn.state["log"]}))
"""

SECOND = """
import json, os, sys
from grapharc.session import SessionManager
from grapharc.session.demo import GRAPH_NAME

with SessionManager(sys.argv[1]) as manager:
    session = manager.resume("incident-42")          # rebuilt from the store
    session.decide(approved=True, decided_by="ops")
    turn = session.run()
    print(json.dumps({"pid": os.getpid(), "status": turn.status.value,
                      "ran": turn.nodes, "log": turn.state["log"]}))
"""


def child(script: str) -> dict:
    path = ROOT / "child.py"
    path.write_text(script, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(path), str(ROOT)], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


a, b = child(FIRST), child(SECOND)
print("process A", a["pid"], a["status"], a["ran"], a["log"])
print("process B", b["pid"], b["status"], b["ran"], b["log"])
print("same process?", a["pid"] == b["pid"])
```

```
process A 479013 awaiting_approval ['ingest', 'plan'] ['ingest', 'plan']
process B 479020 idle ['apply', 'report'] ['ingest', 'plan', 'apply', 'report']
same process? False
```

(The pids are whatever the OS handed out; yours will differ.)

`log` is the proof, not the status. The demo graph appends its own node name on
every execution, so a resume that quietly re-ran `ingest` and `plan` would show
them twice. It shows each once.

**Sharp edge:** the resuming process must be able to *build* the graph.
`register_graph` is per-process and a session record only stores a name, so a
resume in a process that never imported the registering module fails with
`UnknownGraphError` — loudly, which is the intended behaviour. That is also why
`from grapharc.session.demo import GRAPH_NAME` appears in both children: the
import is the registration.

---

## How do I stop a session that is already running?

`session.interrupt(reason)`. It is a durable row, not a signal, so it works from
another thread or another process; the process driving the session reads it
after the next superstep.

```python
import tempfile
import threading

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.session import SessionManager, SessionState, register_graph

INSIDE_STEP_ONE = threading.Event()
# `one` waits for this before returning, so the stop is durably recorded
# *before* the superstep boundary that reads it. Posting the interrupt and
# hoping it wins the race against `two` starting is what makes this snippet
# flaky rather than illustrative.
STOP_RECORDED = threading.Event()


class State(SessionState):
    log: list[str] = []


def build(checkpointer=None):
    def one(state: State) -> dict:
        INSIDE_STEP_ONE.set()
        STOP_RECORDED.wait(timeout=10)
        return {"log": [*state.log, "one"]}

    def two(state: State) -> dict:
        return {"log": [*state.log, "two"]}

    def three(state: State) -> dict:
        return {"log": [*state.log, "three"]}

    g = GraphARC(State, name="pipeline")
    for name, fn in (("one", one), ("two", two), ("three", three)):
        g.add_node(name, fn, writes={"log"})
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", "three")
    g.add_edge("three", END)
    return g.compile(checkpointer=checkpointer)


register_graph("pipeline", build, replace=True)

with SessionManager(tempfile.mkdtemp(prefix="cookbook-")) as manager:
    session = manager.create("pipeline", session_id="s1")

    def stopper():
        INSIDE_STEP_ONE.wait(timeout=10)
        session.interrupt("operator asked for a stop")
        STOP_RECORDED.set()

    threading.Thread(target=stopper).start()

    turn = session.run({})
    print("status       :", turn.status.value)
    print("interrupted  :", turn.interrupted_by)
    print("ran          :", turn.nodes)
    print("still pending:", turn.pending)

    resumed = session.run()
    print("after resume :", resumed.nodes, resumed.state["log"])
```

```
status       : interrupted
interrupted  : operator asked for a stop
ran          : ('one',)
still pending: ('two',)
after resume : ('two', 'three') ['one', 'two', 'three']
```

**An interrupt does not stop a running node.** It is read at the next
*superstep* boundary — after the whole parallel step has finished and been
checkpointed, not after each node in it. So a node already inside its body runs
to completion, and so does everything running beside it. In this snippet the
stop is posted while `one` is executing and takes effect before `two` starts —
but only because `one` waits for `STOP_RECORDED` before returning. Without that
handshake the interrupt is racing the superstep boundary, and landing a few
microseconds late means it takes effect after `two` instead. That is a real
property of interrupts, not an artefact of the example: **the boundary an
interrupt is read at is the next one after it is written, and you do not control
which one that is** unless you synchronise. Cutting a node off mid-flight is
what `Budget(max_seconds=...)` is for, and that has its own honest limits.

A stop queued while the session is asleep is honoured before the next turn's
first node — an interrupt is never silently lost.

---

## What does a hold mean when the gated node is fanned out?

It means one task, and you cannot tell which. A `Send` fan-out can put the same
gated node on the boundary several times over; each of those tasks is held
separately and each needs its own decision, so the *count* is exact and nothing
runs unapproved — but the requests are indistinguishable, right down to the
action text.

```python
import operator
import tempfile
from typing import Annotated

from pydantic import BaseModel

from grapharc.runtime.graph import END, START, GraphARC
from grapharc.session import SessionManager, SessionState, register_graph


class State(SessionState):
    recipients: list[str] = []
    sent: Annotated[list[str], operator.add] = []


class Envelope(BaseModel):
    to: str


def build(checkpointer=None):
    def prepare(state: State) -> dict:
        return {}

    def dispatch(state: State) -> list[tuple[str, BaseModel]]:
        return [("send", Envelope(to=who)) for who in state.recipients]

    def send(payload: Envelope) -> dict:
        return {"sent": [payload.to]}

    g = GraphARC(State, name="mailshot")
    g.add_node("prepare", prepare, writes=set())
    g.add_node("send", send, writes={"sent"}, input_schema=Envelope)
    g.add_edge(START, "prepare")
    g.add_fanout_edge("prepare", dispatch)
    g.add_edge("send", END)
    return g.compile(checkpointer=checkpointer)


register_graph("mailshot", build, approval_nodes=("send",), replace=True)

with SessionManager(tempfile.mkdtemp(prefix="cookbook-")) as manager:
    session = manager.create("mailshot", session_id="s1")
    turn = session.run({"recipients": ["ana@x", "bo@x", "cy@x"]})

    print("status :", turn.status.value)
    for hold in turn.approvals:
        print(f"  hold {hold.node!r}: {hold.action!r}")

    # Approve two, reject one. Which recipient loses is not yours to choose.
    holds = session.pending_approvals
    session.decide(approved=True, request_id=holds[0].id)
    session.decide(approved=True, request_id=holds[1].id)
    session.decide(approved=False, request_id=holds[2].id, reason="bounced")

    final = session.run()
    print("ran    :", final.nodes)
    print("skipped:", final.skipped)
    print("sent   :", sorted(final.state["sent"]))
```

```
status : awaiting_approval
  hold 'send': "run node 'send'"
  hold 'send': "run node 'send'"
  hold 'send': "run node 'send'"
ran    : ('send', 'send')
skipped: ('send',)
sent   : ['bo@x', 'cy@x']
```

Three identical holds. Approving *k* of *n* runs *k* of them; **which** *k* is
not yours to pick. Here `ana@x` lost, and that is an artefact of task ordering,
not a guarantee. If the choice matters — "send to ana but not to cy" — do not
model it as one gated node fanned out. Gate a node that reads the decision from
state, or fan out over distinct node names.

Two more things this snippet demonstrates in passing:

- `request_id` is **required** once more than one hold is open. `decide()`
  without it raises rather than guessing, because guessing is how an operator
  signs off `send_email` and releases `delete_records`.
- Only one verdict reaches the graph. `SessionState.approval_decision` is a
  single channel, so a boundary settling several holds shows the graph the first
  hold's verdict. Every verdict is on the event log and in `TurnResult.decisions`.

**Sharp edge:** a `Send` payload that is a Pydantic model goes into the
checkpoint, and LangGraph prints a warning on stderr when it reads one back
(`Deserializing unregistered type <module>.Envelope from checkpoint. This will
be blocked in a future version.`). It is stderr noise today and a future error;
the fix is LangGraph's `allowed_msgpack_modules`, not anything in GraphARC.

---

## How do I put this behind HTTP?

`grapharc.server.create_app(registry=...)` returns a FastAPI app. A request may
*name* a graph the operator registered and supply input and a budget; it may not
*describe* a graph. The snippet below drives the real app through Starlette's
`TestClient`, so it runs with no server process.

```python
import json
import time

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.server import GraphRegistry, create_app
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    answer: str = ""


def build_qa(trace: TraceRecorder):
    model = ScriptedChatModel(responses=["Budgets cap iterations, tokens and time."])

    def answer(state: State) -> dict:
        reply = model.invoke([HumanMessage(content=state.question)])
        return {"answer": str(reply.content)}

    g = GraphARC(State, name="qa", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()


registry = GraphRegistry({"qa": build_qa})
app = create_app(registry=registry)

with TestClient(app) as client:
    print("health :", client.get("/healthz").json())

    created = client.post(
        "/sessions", json={"graph": "qa", "input": {"question": "how do budgets work?"}}
    )
    session_id = created.json()["id"]
    print("created:", created.status_code, created.json()["status"])

    # Poll until the run reaches a terminal status.
    for _ in range(200):
        view = client.get(f"/sessions/{session_id}").json()
        if view["status"] in ("succeeded", "failed", "interrupted"):
            break
        time.sleep(0.01)
    print("status :", view["status"])
    print("answer :", view["result"]["answer"])
    usage = view["usage"]
    print("usage  :", usage["iterations"], "iterations,", usage["tokens"], "tokens")

    ack = client.post(
        f"/sessions/{session_id}/events", json={"type": "message", "data": {}}
    ).json()["event"]
    print("event  :", ack["applied"], ack["detail"])

    # SSE: replay from cursor 0, so a finished run still streams its whole trace.
    frames = [
        line for line in client.get(f"/sessions/{session_id}/stream").text.splitlines() if line
    ]
    print("frames :", [f for f in frames if f.startswith("event:")])

    trace = client.get(f"/sessions/{session_id}/trace").text
    print("trace  :", [json.loads(line)["phase"] for line in trace.splitlines()])
```

```
health : {'status': 'ok', 'version': '0.1.3', 'graphs': ['qa']}
created: 201 queued
status : succeeded
answer : Budgets cap iterations, tokens and time.
usage  : 1.0 iterations, 15.0 tokens
event  : False recorded: this runtime does not deliver 'message' events into a running graph (ROADMAP §6.4 event queue / §6.5 approval node)
frames : ['event: trace', 'event: trace', 'event: trace', 'event: status', 'event: done']
trace  : ['topology', 'start', 'end']
```

The routes:

| Route | What it does |
| --- | --- |
| `POST /sessions` | create and start one → `201` + `location` header |
| `GET /sessions` | list every session this process knows |
| `GET /sessions/{id}` | status, result, live meter reading |
| `POST /sessions/{id}/events` | `message` / `approval` / `interrupt` → `202` |
| `GET /sessions/{id}/stream` | SSE: `trace` frames, then `status`, then `done` |
| `GET /sessions/{id}/trace` | the JSONL trace as `application/x-ndjson` |
| `GET /healthz` | liveness, version, registered graph names |

**Read `applied`, not the status code.** `POST /events` answers `202` for
everything — the event was accepted for the session. Whether it *changed the
run* is `event.applied` in the body, and this runtime only applies `interrupt`.
A `message` or `approval` is recorded with `applied: false` and a `detail`
saying why. An HTTP 2xx here does not mean the run was steered.

**A stream frame and a trace line are the same record.** `BroadcastRecorder`
shapes each event once and hands the same dict to the file and to the
subscriber, so parsing a line of `/trace` yields exactly the object the SSE
frame carried — including the trace format's 2000-character clip on long values.
The unclipped answer is the session's `result`, which is never truncated.

The stream takes a `cursor` query parameter and honours `last-event-id`, so a
dropped SSE connection resumes without replaying frames you already saw.

**What this default runtime does not do:** it does not resume across a process
restart (sessions live in this process's memory; the trace files outlive it, the
sessions do not), and it does not evict — every session and its event list is
retained for the life of the process. For durable sessions, use
`grapharc.session` as shown above.

---

## Which checkpointer does the server need?

An async one, if your nodes are `async def`. The server drives runs through
`astream`, and `SqliteSaver` — this repo's own dependency, and what
`grapharc.session` uses — has no async side.

The runtime probes the saver once per run, on the run's own event loop, before
anything executes. A sync-only saver quietly selects the sync driver instead.
The one combination nothing can serve is **`async def` nodes plus a sync-only
saver**, and that is where you get `CheckpointerNotAsyncError`.

```python
import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.server import GraphRegistry, create_app


class State(GraphARCState):
    question: str
    answer: str = ""


async def answer(state: State) -> dict:
    await asyncio.sleep(0)
    return {"answer": "42"}


def graph_with(checkpointer, trace: TraceRecorder):
    g = GraphARC(State, name="qa", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)


db = Path(tempfile.mkdtemp(prefix="cookbook-")) / "checkpoints.sqlite"
conn = sqlite3.connect(db, check_same_thread=False)

registry = GraphRegistry(
    {
        "sync_saver": lambda trace: graph_with(SqliteSaver(conn), trace),
        "async_ok": lambda trace: graph_with(InMemorySaver(), trace),
    }
)


def run(client, graph):
    sid = client.post("/sessions", json={"graph": graph, "input": {"question": "q"}}).json()["id"]
    for _ in range(500):
        view = client.get(f"/sessions/{sid}").json()
        if view["status"] in ("succeeded", "failed", "interrupted"):
            return view
        time.sleep(0.01)
    raise AssertionError("never finished")


with TestClient(create_app(registry=registry)) as client:
    bad = run(client, "sync_saver")
    print("sync_saver :", bad["status"])
    print("error      :", bad["error"])
    good = run(client, "async_ok")
    print("async_ok   :", good["status"], good["result"])

conn.close()
```

```
sync_saver : failed
error      : CheckpointerNotAsyncError: graph 'qa' needs astream() because it has async nodes, but its checkpointer cannot be driven asynchronously (SqliteSaver does not implement async checkpointing: The SqliteSaver does not support async methods. Consider using AsyncSqliteSaver instead.). Fix either side: give compile() a saver that implements the async API — langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver (needs aiosqlite, and must be constructed on a running loop) or langgraph.checkpoint.memory.InMemorySaver — or make the graph's nodes synchronous, which lets the server drive it through stream() and use the sync saver as it is. [graph 'qa' has async nodes ['answer']; use astream(), because stream() has no event loop to await them on]
async_ok   : succeeded {'question': 'q', 'answer': '42'}
```

So, in a table:

| Nodes | Checkpointer | Result |
| --- | --- | --- |
| sync | none / `SqliteSaver` | runs (sync driver) |
| sync | `InMemorySaver` / `AsyncSqliteSaver` | runs (async driver) |
| `async def` | none / `InMemorySaver` / `AsyncSqliteSaver` | runs (async driver) |
| `async def` | `SqliteSaver` | `CheckpointerNotAsyncError` |

`AsyncSqliteSaver` needs `aiosqlite` and must be constructed on a running loop.
`InMemorySaver` is the zero-dependency answer when durability is not the point.

---

## How do I actually serve it?

`grapharc serve --registry module:attr`, where the attribute is a
`grapharc.server.GraphRegistry` (or a callable returning one). Without
`--registry` the app starts with an empty registry — a legitimate way to check
the process comes up and a useless way to run anything, and the CLI tells you
which of the two you got.

```python
"""mygraphs.py — graphs this server is allowed to run."""

from langchain_core.messages import HumanMessage

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.server import GraphRegistry
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    answer: str = ""


def build_qa(trace: TraceRecorder):
    model = ScriptedChatModel(responses=["Budgets cap iterations, tokens and time."])

    def answer(state: State) -> dict:
        return {"answer": str(model.invoke([HumanMessage(content=state.question)]).content)}

    g = GraphARC(State, name="qa", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()


REGISTRY = GraphRegistry({"qa": build_qa})
```

Then, in a shell. This transcript is a real one: the ids and timestamps differ
per run, and the only editing is the `...` marking where a timestamp or a long
JSON body was cut, plus the session view re-indented for reading — nothing was
reworded. The test re-runs every command here against a fresh server and
requires each to succeed; the bytes below are one run's.

<!-- verified: cli varies -->
```console
$ PYTHONPATH=. grapharc serve --registry mygraphs:REGISTRY --port 8124
serving grapharc.server on http://127.0.0.1:8124
graphs   : qa
ctrl-c to stop

$ curl -s localhost:8124/healthz
{"status":"ok","version":"0.1.3","graphs":["qa"]}

$ curl -s -X POST localhost:8124/sessions -H 'content-type: application/json' \
      -d '{"graph":"qa","input":{"question":"how do budgets work?"}}'
{"id":"bf5ca55bff7b480f","graph":"qa","thread_id":"bf5ca55bff7b480f","status":"queued", ...}

$ curl -s localhost:8124/sessions/bf5ca55bff7b480f
{
    "id": "bf5ca55bff7b480f",
    "graph": "qa",
    "status": "succeeded",
    "run_id": "0d9dce7f61c4",
    "result": {
        "question": "how do budgets work?",
        "answer": "Budgets cap iterations, tokens and time."
    },
    "usage": {"iterations": 1.0, "tokens": 15.0, "elapsed_seconds": 0.003},
    ...
}

$ curl -s localhost:8124/sessions/bf5ca55bff7b480f/trace
{"ts": "...", "run_id": "0d9dce7f61c4", "thread_id": "bf5ca55bff7b480f", "attempt": 1, "graph": "qa", "node": "topology", "phase": "topology", "step": 0, "state_delta": {"nodes": ["answer"], "edges": [["__start__", "answer", "static"], ["answer", "__end__", "static"]]}}
{"ts": "...", "run_id": "0d9dce7f61c4", "thread_id": "bf5ca55bff7b480f", "attempt": 1, "graph": "qa", "node": "answer", "phase": "start", "step": 1}
{"ts": "...", "run_id": "0d9dce7f61c4", "thread_id": "bf5ca55bff7b480f", "attempt": 1, "graph": "qa", "node": "answer", "phase": "end", "step": 1, "state_delta": {"answer": "Budgets cap iterations, tokens and time."}, "duration_ms": 1.0288769999533542, "tokens": 15}
```

The banner is printed *before* the server blocks, so a script watching stdout
for the URL does not have to wait for the process to exit to learn it. `serve`
needs the server extra: `uv sync --extra server`.

To use a real model instead of the scripted one, swap the model in the builder:

```python
from grapharc.gateway import get_model

model = get_model("openrouter/anthropic/claude-haiku-4.5")
# or, on a Claude subscription with the CLI on PATH and no API key:
model = get_model("claude-cli/claude-sonnet-5")
```

**This snippet was not executed** — it needs a paid API key or a Claude
subscription. `grapharc models --check` tells you which backends this machine
could use, without contacting any provider.

---

## How do I watch a run live in a browser?

`grapharc serve --live-root PATH` mounts a read-only live view at `/live` over
the trace files under `PATH` — including files other processes are appending
right now. Traces are append-only JSONL written line-at-a-time under a lock,
so a reader that stops at the last complete newline (`TailRecorder`, in
`grapharc.observe.trace`) can follow a run another process is executing;
that is exactly what the view does.

`GET /live` lists every `*.jsonl` under the root, newest first.
`GET /live/view?trace=REL` is the page: it opens
`GET /live/api/stream?trace=REL` (server-sent events) and receives a fresh
`snapshot` — the run's Mermaid diagram, `metrics`-style numbers, cost, and
status — each time the file grows. The server recomputes the snapshot;
the page only renders it. Add `&run=ID` to pin one run in a file that holds
several; without it the view follows the newest.

A planner run has no graph for as long as it takes the model to propose one,
so the snapshot also carries a `planning` block — round number, proposal size,
admitted or rejected with the checks that failed, planner tokens spent — and
the page renders it as a panel from the `plan`, `admission` and `round` events
already in the trace. A run refused on every round (`admission_refused`) never
produces a topology at all; it shows its rounds and its stop reason instead of
an empty page. A round that has begun and not closed also counts as activity,
because a planner mid-inference writes nothing for a minute at a time.

`GET /live/view?trace=REL&replay=1` replays a finished trace instead of
rendering its final state: the recorded events are walked in timestamp order
and each one emits the snapshot a live run would have sent, so nodes go amber
then green in the order and at the pace they really ran. `&speed=N` divides the
wall clock, and a whole replay is capped at 40 seconds however slow the
recording was, so yesterday's 40-minute incident trace is watchable. Without
the parameter nothing changes: one snapshot per file change, as before.

This composes with the Slack bot, which gives every tracing command a trace
path under its working directory: run `grapharc serve --live-root` over that
same directory, set `GRAPHARC_SLACK_LIVE_URL`, and the bot posts a
"watch live" link when a run starts — the walkthrough is in
[07-slack.md](07-slack.md). It also composes with this page's own server
sessions: point `--live-root` at the session root and each
`<session>/trace.jsonl` gets a page.

The posture is the same as everything else in `grapharc.observe`: the view is
derived from the trace file and nothing else, and it is read-only. Requested
paths are confined inside the root (escapes are 404s), and `state_delta`
contents — arbitrary node writes — are never serialized into any live
response; the exposure is what `viz` already prints. The bind stays
`127.0.0.1` unless you say otherwise; binding wider prints a warning, because
reachability is meant to come from a tunnel or tailnet in front, optionally
with `--live-token TOKEN` (or `GRAPHARC_LIVE_TOKEN`) required on every
`/live` request.

**Where that token is allowed to travel matters.** A URL is copied into places
with much weaker access control than the traces it protects: the uvicorn
request line, an nginx access log, browser history, and the referrer of
anything the page links out to. So the token goes in an
`Authorization: Bearer TOKEN` header, or in the cookie that `POST /live/auth`
sets when you paste it into the sign-in page a browser gets instead of a 401.
`?token=` is accepted on `/live/api/stream` and nowhere else — a browser
`EventSource` cannot set a header, so that one route has no alternative — and
any other `/live` route refuses a query-string token with a 401 that says so
rather than accepting the secret into your logs. The tradeoff that remains:
the SSE request line still carries the token, so if you terminate TLS at nginx
and log request URIs, scrub `token=` from that one path (or log
`$request_method $uri` rather than `$request`). The cookie is a digest of the
token, not the token, is `HttpOnly` and `SameSite=Strict`, and is scoped to
`/live`. Sign-in and the SSE exemption both apply only when a token is
configured at all; without one, nothing about `/live` is authenticated.

The diagram renders with mermaid.js from a pinned CDN; with
no CDN reachable the page falls back to the raw Mermaid source plus the same
mermaid.live fragment link the Slack bot posts.

---

## How do I reconstruct a run after it finished?

`replay(trace, run_id)`. It is a *reconstruction*, not a re-execution: it reads
the JSONL and rebuilds the node sequence, the deltas, the timings and the
failures. Nothing here calls a model, a tool, or a node.

```python
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from grapharc.observe import TraceRecorder, format_replay, replay
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    draft: str = ""
    answer: str = ""


def build(trace: TraceRecorder, responses: list[str]):
    model = ScriptedChatModel(responses=responses, on_exhausted="repeat")

    def draft(state: State) -> dict:
        return {"draft": str(model.invoke([HumanMessage(content=state.question)]).content)}

    def polish(state: State) -> dict:
        return {"answer": state.draft.strip().rstrip(".") + "."}

    g = GraphARC(State, name="writer", trace=trace)
    g.add_node("draft", draft, writes={"draft"})
    g.add_node("polish", polish, writes={"answer"})
    g.add_edge(START, "draft")
    g.add_edge("draft", "polish")
    g.add_edge("polish", END)
    return g.compile()


path = Path(tempfile.mkdtemp(prefix="cookbook-")) / "trace.jsonl"
trace = TraceRecorder(path)

build(trace, ["budgets cap iterations"]).invoke({"question": "budgets?"}, run_id="before")

run = replay(trace, "before")
print("path   :", run.path)
print("tokens :", run.tokens)
print("state  :", run.replay_state())
print()
print(format_replay(run))
```

```
path   : ['draft', 'polish']
tokens : 7
state  : {'draft': 'budgets cap iterations', 'answer': 'budgets cap iterations.'}

run before · graph writer
    1 ok  draft (0.4ms, 7 tok)
        draft = 'budgets cap iterations'
    2 ok  polish (0.0ms)
        answer = 'budgets cap iterations.'
  2 nodes · 7 tokens · stopped: not recorded
```

(Node durations are measured wall-clock; yours will differ.)

Passing `run_id=` to `invoke()` is what makes a run findable later. Without it
you get a random hex id and have to fish it out of the trace file.

**`replay_state()` is only as good as the deltas.** Two limits, both from the
recording side:

- The trace truncates a string past 2000 characters at write time, so a long
  value replays truncated.
- The trace does not record which state fields have reducers. A field LangGraph
  *appended* to replays as last-write-wins unless you say otherwise:
  `run.replay_state(reducers={"log": operator.add})`. Forgetting this silently
  under-reports an accumulating list, which is why the argument is in the
  signature rather than in a comment.

`replay_thread(trace, thread_id)` gives you every run recorded against one
thread, which is how you read a resumed session rather than its last attempt.

---

## How do I tell whether my change altered behaviour?

`diff_trace(trace, run_a, run_b)` aligns the two node sequences with `difflib`
and reports where they diverged.

```python
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from grapharc.observe import TraceRecorder, diff_trace, format_diff
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    draft: str = ""
    answer: str = ""


def build(trace: TraceRecorder, responses: list[str]):
    model = ScriptedChatModel(responses=responses, on_exhausted="repeat")

    def draft(state: State) -> dict:
        return {"draft": str(model.invoke([HumanMessage(content=state.question)]).content)}

    def polish(state: State) -> dict:
        return {"answer": state.draft.strip().rstrip(".") + "."}

    g = GraphARC(State, name="writer", trace=trace)
    g.add_node("draft", draft, writes={"draft"})
    g.add_node("polish", polish, writes={"answer"})
    g.add_edge(START, "draft")
    g.add_edge("draft", "polish")
    g.add_edge("polish", END)
    return g.compile()


path = Path(tempfile.mkdtemp(prefix="cookbook-")) / "trace.jsonl"
trace = TraceRecorder(path)

build(trace, ["budgets cap iterations"]).invoke({"question": "budgets?"}, run_id="before")
build(trace, ["budgets cap iterations, tokens and time"]).invoke(
    {"question": "budgets?"}, run_id="after"
)

diff = diff_trace(trace, "before", "after")
print("identical:", diff.identical)
print(format_diff(diff))
```

```
identical: False
before != after: 2 node(s) wrote different deltas; 2 state field(s) differ
  draft: changed ['draft']
  polish: changed ['answer']
  state answer: 'budgets cap iterations.' -> 'budgets cap iterations, tokens and time.'
  state draft: 'budgets cap iterations' -> 'budgets cap iterations, tokens and time'
```

**`identical` deliberately ignores timing and tokens.** Two runs of a
deterministic graph differ by milliseconds every time, and a diff that is never
clean is a diff nobody reads. It compares path, deltas and termination reason.
The token and duration numbers are still on the `RunDiff` if you want them.

Diffing runs of *different* graphs raises `ReplayError` rather than producing an
alignment that is technically correct and means nothing.

---

## Where did the tokens go?

`attribute(trace, run_id, rates=...)` splits a run's spend per node. A
`RateCard` is USD per 1,000 tokens, **blended** across input and output, because
a trace event records one `tokens` total and no split.

```python
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from grapharc.observe import RateCard, TraceRecorder, attribute, summarize, to_mermaid
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    question: str
    draft: str = ""
    answer: str = ""


path = Path(tempfile.mkdtemp(prefix="cookbook-")) / "trace.jsonl"
trace = TraceRecorder(path)
model = ScriptedChatModel(responses=["budgets cap iterations"], on_exhausted="repeat")


def draft(state: State) -> dict:
    return {"draft": str(model.invoke([HumanMessage(content=state.question)]).content)}


def polish(state: State) -> dict:
    return {"answer": str(model.invoke([HumanMessage(content=state.draft)]).content) + "."}


g = GraphARC(State, name="writer", trace=trace)
g.add_node("draft", draft, writes={"draft"})
g.add_node("polish", polish, writes={"answer"})
g.add_edge(START, "draft")
g.add_edge("draft", "polish")
g.add_edge("polish", END)
g.compile().invoke({"question": "budgets?"}, run_id="r1")

cost = attribute(trace, "r1", rates=RateCard(default=3.0))  # USD per 1k tokens, blended
for node in cost.per_node:
    print(f"{node.node:<8} {node.tokens:>3} tok  ${node.estimated_cost_usd:.6f}")
print("total   ", cost.tokens, "tok  complete:", cost.complete)

metrics = summarize(trace, "r1")
print("metrics :", metrics.nodes_executed, "nodes,", metrics.tokens, "tokens,", metrics.per_node)
print()
print(to_mermaid(trace, "r1"))
```

```
draft      7 tok  $0.021000
polish    10 tok  $0.030000
total    17 tok  complete: True
metrics : 2 nodes, 17 tokens, {'draft': 1, 'polish': 1}

flowchart TD
  n0["draft"]
  n1["polish"]
  start((start)) --> n0["draft"]
  n0["draft"] --> n1["polish"]
  n1["polish"] --> fin((end))
  classDef done fill:#d3f2d3,stroke:#2f7d32
  classDef running fill:#fff3cd,stroke:#b8860b
  classDef pending fill:#eeeeee,stroke:#999999,color:#666666
  classDef errored fill:#f8d7da,stroke:#b02a37
  class n0,n1 done
```

`RunCost.tokens` and `RunMetrics.tokens` agree by construction — both count the
`end` events of node executions, and the test suite asserts they match, because
a cost report and an audit trail that disagree are worse than either alone.

**Three honest limits:**

- **Nothing in GraphARC writes `cost_usd` onto a trace event today.** The field
  exists and `TraceRecorder.event` accepts it, but the kernel does not pass it,
  so `recorded_cost_usd` is always `None` on a trace from today's runtime.
  Everything you see above is `estimated_cost_usd` — tokens × your rate card,
  reported in its own field so nobody mistakes an estimate for an invoice.
- **`complete` is what tells you the total is a total.** Tokens with neither a
  recorded cost nor a matching rate are counted in `unpriced_tokens`, and a
  non-zero count means `cost_usd` is a lower bound. A `RateCard` with no
  `default` and no matching model prices nothing.
- **A node that raised has no token count at node level.** No `end` event was
  written. Where it was an `AgentNode`, its per-call sub-events still hold the
  spend, reported as `tokens_before_error` — kept out of the total so the total
  keeps matching `metrics`.

`to_mermaid` renders the graph's *declared topology* — the `topology` event every
run now writes — with execution status overlaid per node: `done`, `running`,
`errored`, or still `pending`. Branches not taken stay on the diagram in grey,
conditional routes draw dotted, and a multi-round planner run gets one cluster
per admitted round. A trace with no topology event (an `AgentNode` driven with
no enclosing graph, or a file written before the event existed) falls back to
the executed path in event order, keyed by `(node, step)` so parallel instances
of a fan-out worker are distinct boxes. Paste either form into any Markdown
renderer that speaks Mermaid.

`attribute_thread(trace, thread_id)` is the same for a whole session across
resumes, and `by_node(trace)` ranks every node in a file by cost.

---

## The CLI tour

Twelve commands. Every one takes `--json`, which prints the same payload as one
document on stdout — including failures, which become the document rather than a
line on stderr.

| Command | What it is for |
| --- | --- |
| `grapharc demo <example>` | run a built-in example graph (`stage0`…`stage6`, `capstone`) |
| `grapharc run <graph.json>` | run a topology you wrote, through the admission gate; `--check-only` lints it |
| `grapharc plan <goal>` | governed loop: propose → admit → execute → replan; `--approve` parks each admitted round for a human |
| `grapharc approve <trace>` | answer a plan run waiting on its approval gate (`--deny` to refuse) |
| `grapharc agent <task>` | run an agent node with the core tools against a task |
| `grapharc serve` | run the HTTP API |
| `grapharc models [spec]` | what a spec resolves to; `--check` probes this machine |
| `grapharc replay <trace> <run>` | reconstruct a recorded run |
| `grapharc diff <trace> <a> <b>` | compare two runs in one trace |
| `grapharc trace <trace>` | pretty-print a trace file |
| `grapharc metrics <trace> <run>` | summarize one run |
| `grapharc viz <trace> <run>` | render the executed path as Mermaid |

Exit codes are part of the interface: `0` did the job, `1` ran and the answer
was negative (two runs differed, a run id had no events, no backend was usable),
`2` could not run at all (missing file, missing component, unknown model spec).

A whole session, verbatim (run ids are random per run and durations are
wall-clock; the test maps the former, masks the latter, and byte-compares every
other character):

<!-- verified: cli -->
```console
$ grapharc demo stage1 --trace trace.jsonl
...
no_progress_rounds: 0
proposal: verifier
candidate: 2
termination_reason: target_met

trace: trace.jsonl

$ grapharc trace trace.jsonl --json | jq -r '.events[0].run_id'
2a47f18064b7

$ grapharc trace trace.jsonl --run-id 2a47f18064b7 | head -6
[  0] topology             topology Δ{'nodes': ['start', 'plan', 'act', 'verify', 'finish_target_met', 'finish_max_iterations', 'finish_no_progress'], 'edges': [['__start__', 'start', 'static'], ['start', 'plan', 'static'], ['plan', 'act', 'static'], ['act', 'verify', 'static'], ['finish_target_met', '__end__', 'static'], ['finish_max_iterations', '__end__', 'static'], ['finish_no_progress', '__end__', 'static'], ['verify', 'plan', 'conditional'], ['verify', 'finish_target_met', 'conditional'], ['verify', 'finish_max_iterations', 'conditional'], ['verify', 'finish_no_progress', 'conditional']]}
[  1] start                start
[  1] start                end    Δ{'pending': ['budgets', 'verifier']}
[  2] plan                 start
[  2] plan                 end    Δ{'proposal': 'budgets', 'round': 1}
[  3] act                  start

$ grapharc metrics trace.jsonl 2a47f18064b7
run_id: 2a47f18064b7
graph: stage1_loop
nodes_executed: 8
errors: 0
tokens: 81
duration_ms: 0.68
attempts: 1
termination_reason: target_met
per_node: {'start': 1, 'plan': 2, 'act': 2, 'verify': 2, 'finish_target_met': 1}
events: 17
per_phase: {'topology': 1, 'start': 8, 'end': 8}

$ grapharc viz trace.jsonl 2a47f18064b7
flowchart TD
  n0["start"]
  n1["plan"]
  n2["act"]
  n3["verify"]
  n4["finish_target_met"]
  n5["finish_max_iterations"]
  n6["finish_no_progress"]
  start((start)) --> n0["start"]
  n0["start"] --> n1["plan"]
  n1["plan"] --> n2["act"]
  n2["act"] --> n3["verify"]
  n4["finish_target_met"] --> fin((end))
  n5["finish_max_iterations"] --> fin((end))
  n6["finish_no_progress"] --> fin((end))
  n3["verify"] -.-> n1["plan"]
  n3["verify"] -.-> n4["finish_target_met"]
  n3["verify"] -.-> n5["finish_max_iterations"]
  n3["verify"] -.-> n6["finish_no_progress"]
  classDef done fill:#d3f2d3,stroke:#2f7d32
  classDef running fill:#fff3cd,stroke:#b8860b
  classDef pending fill:#eeeeee,stroke:#999999,color:#666666
  classDef errored fill:#f8d7da,stroke:#b02a37
  class n0,n1,n2,n3,n4 done
  class n5,n6 pending

$ grapharc replay trace.jsonl 2a47f18064b7 | tail -4
        pending = []
    8 ok  finish_target_met (0.0ms)
        termination_reason = 'target_met'
  8 nodes · 81 tokens · stopped: target_met

$ grapharc demo stage1 --trace trace.jsonl > /dev/null   # a second run, same file
$ grapharc diff trace.jsonl 2a47f18064b7 3c0d1b4b4b3e; echo "exit $?"
2a47f18064b7 == 3c0d1b4b4b3e: same path (8 nodes), same state
exit 0
```

**Sharp edge:** `grapharc run` does not print the run id. It prints the trace
path, and you get the id out of the file — the `jq` line above, or
`python -c "import json; print(json.loads(open('trace.jsonl').readline())['run_id'])"`.
Everything downstream (`metrics`, `viz`, `replay`, `diff`) wants that id.

`--json` on any of them, and on failures too:

<!-- verified: cli -->
```console
$ grapharc metrics trace.jsonl 2a47f18064b7 --json
{
  "ok": true,
  "command": "metrics",
  "run_id": "2a47f18064b7",
  "graph": "stage1_loop",
  "nodes_executed": 8,
  "errors": 0,
  "tokens": 81,
  "duration_ms": 0.75,
  "attempts": 1,
  "termination_reason": "target_met",
  "per_node": {
    "start": 1,
    "plan": 2,
    "act": 2,
    "verify": 2,
    "finish_target_met": 1
  },
  "events": 17,
  "per_phase": {
    "topology": 1,
    "start": 8,
    "end": 8
  }
}

$ grapharc metrics nope.jsonl abc --json; echo "exit $?"
{
  "ok": false,
  "command": "metrics",
  "error": "no such trace file: nope.jsonl"
}
exit 2
```

`grapharc models` needs no credentials to answer what a spec *resolves* to:

<!-- verified: cli -->
```console
$ grapharc models openrouter/anthropic/claude-haiku-4.5
spec: openrouter/anthropic/claude-haiku-4.5
backend: openrouter
model: anthropic/claude-haiku-4.5
```

`--check` reports the machine it is run on:

<!-- verified: cli varies -->
```console
$ grapharc models --check
claude-cli   usable    'claude' on PATH at /home/you/.local/bin/claude
                       credential: claude subscription login (no API key)
openrouter   unusable  no API key (set OPENROUTER_API_KEY, or add one to .env)
                       credential: <unset>
openai       unusable  no API key (set OPENAI_API_KEY, or add one to .env)
                       credential: <unset>
ollama       usable    local server at http://localhost:11434/v1
                       credential: none needed (local server)
mock         usable    scripted test double; never reaches a provider

local probe only — no provider was contacted, so a configured key
is not a validated one.
```

(Paths in that output are machine-specific; the one above is edited only to
replace a home directory.) `--check` is a *local* probe: it reports that a
credential is configured, never that it is valid, in credit, or entitled to a
model. It exits `1` when no real provider is usable — `mock` alone does not
count.

`grapharc agent` is the one command in this list that always needs a model, so
there is no scripted form of it and nothing here claims to have run it:

<!-- needs-credentials -->
```console
$ grapharc agent "summarise README.md" --workspace ./work --max-turns 6 --json
```

`--allow` / `--deny` / `--ask` are repeatable tool-name globs (`--deny` wins),
`--executor local` drops the sandbox, and `--max-turns` / `--max-tokens` /
`--max-seconds` are the run's budget. See the governance and harness sections
for what those actually enforce.
