"""Executable gate for `docs/cookbook/06-serving-and-ops.md`.

The cookbook's only real promise is that every snippet in it runs exactly as
printed. This file enforces that mechanically rather than by review: each
```python block is extracted, executed, and diffed against the plain block that
follows it. A snippet that stops working — or an output block someone tidied by
hand — fails here with a diff.

Three things are deliberately not left to the differ:

- **Non-deterministic fields are normalised, not ignored.** Approval request
  ids, child pids and node durations change on every run and the doc says so
  inline. `_NORMALISERS` lists exactly which spans are allowed to vary; every
  other character has to match.
- **Snippets are exec'd into a real module object registered in `sys.modules`.**
  A `Send` payload that is a Pydantic model is stored in the checkpoint by
  class path, so a class defined in a namespace no importer can find comes back
  from the checkpoint unusable and the fan-out snippet silently sends nothing.
  Running it as a module is what a reader gets when they save the snippet to a
  file, and it is what the test has to reproduce.
- **The two unpaired snippets are pinned by name.** One is the server registry
  module the shell transcript imports (checked by building it), the other is
  the real-model swap the doc says in words it did not run. A future edit
  cannot quietly add a third unexecuted snippet.

The section's own claims — an async graph refuses the sync entry point before
running anything, an unknown `goto` raises, a rejected node's body never runs,
a hold is per task and not per recipient, a sync-only checkpointer fails the
server — are pinned again below as direct assertions. The differ proves the
printed output is real; these prove it still means what the prose says.
"""

from __future__ import annotations

import asyncio
import io
import re
import sqlite3
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import (
    END,
    START,
    AsyncNodeError,
    Command,
    GraphARC,
    GraphRoutingError,
)
from grapharc.runtime.state import GraphARCState
from grapharc.server import GraphRegistry as ServerRegistry
from grapharc.server import create_app
from grapharc.session import REGISTRY, SessionManager, SessionStatus
from grapharc.session.demo import GRAPH_NAME

DOC = Path(__file__).resolve().parents[1] / "docs" / "cookbook" / "06-serving-and-ops.md"
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)

#: Spans the doc says out loud will differ per run. Anything not listed here
#: has to match character for character.
_NORMALISERS = (
    # `... awaiting approval 'dc4259cee88f'; call decide() ...`
    (re.compile(r"awaiting approval '[0-9a-f]{12}'"), "awaiting approval '<id>'"),
    # `process A 479013 awaiting_approval ...` — child pids
    (re.compile(r"^(process [AB]) \d+", re.MULTILINE), r"\1 <pid>"),
    # `1 ok  draft (0.4ms, 7 tok)` — measured wall clock
    (re.compile(r"\(\d+\.\d+ms"), "(<t>ms"),
)


def _blocks() -> list[tuple[str, str, int]]:
    """(language, body, 1-based line number) for every fenced block, in order."""
    text = DOC.read_text(encoding="utf-8")
    return [
        (m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1)
        for m in FENCE.finditer(text)
    ]


def _snippets() -> tuple[list[dict], list[dict]]:
    """Split python blocks into (runnable, unpaired).

    A python block immediately followed by an unlabelled block is a snippet
    plus its expected stdout. One with anything else after it — prose, a
    ```console transcript, a table — is a fragment the doc does not claim to
    have run, and every one of those is accounted for by name below.
    """
    blocks = _blocks()
    runnable: list[dict] = []
    unpaired: list[dict] = []
    for index, (lang, body, line) in enumerate(blocks):
        if lang != "python":
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if following is not None and following[0] == "":
            runnable.append({"code": body, "expected": following[1], "line": line})
        else:
            unpaired.append({"code": body, "line": line})
    return runnable, unpaired


RUNNABLE, UNPAIRED = _snippets()


def _normalise(text: str) -> str:
    for pattern, replacement in _NORMALISERS:
        text = pattern.sub(replacement, text)
    return text.rstrip("\n")


def _exec_snippet(code: str, name: str) -> str:
    """Run one snippet as a module and return its stdout."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, f"{DOC}:{name}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolated_graph_registry():
    """Snippets register graphs by name; keep that out of the rest of the suite."""
    before = dict(REGISTRY._specs)
    yield
    REGISTRY._specs.clear()
    REGISTRY._specs.update(before)


# -- the differ ---------------------------------------------------------------


def test_the_section_exists_and_has_snippets():
    assert DOC.is_file()
    assert len(RUNNABLE) >= 15


@pytest.mark.parametrize("snippet", RUNNABLE, ids=[f"line{s['line']}" for s in RUNNABLE])
def test_every_snippet_prints_exactly_what_the_doc_says(snippet):
    printed = _exec_snippet(snippet["code"], f"cookbook_serving_L{snippet['line']}")
    assert _normalise(printed) == _normalise(snippet["expected"])


def test_the_unrun_snippets_are_the_two_the_doc_accounts_for():
    assert len(UNPAIRED) == 2
    registry_module, real_model = UNPAIRED
    assert "mygraphs.py" in registry_module["code"]
    assert "get_model(" in real_model["code"]
    compile(real_model["code"], "<fragment>", "exec")  # still has to be valid Python
    assert "**This snippet was not executed**" in DOC.read_text(encoding="utf-8")


def test_the_registry_module_the_shell_transcript_imports_actually_builds():
    """The one unpaired snippet that *can* be run is run, just not for its stdout."""
    code = UNPAIRED[0]["code"]
    module = types.ModuleType("cookbook_mygraphs")
    sys.modules["cookbook_mygraphs"] = module
    try:
        exec(compile(code, "<mygraphs.py>", "exec"), module.__dict__)
        registry = module.REGISTRY
        assert registry.names() == ["qa"]
        with TestClient(create_app(registry=registry)) as client:
            assert client.get("/healthz").json()["graphs"] == ["qa"]
    finally:
        sys.modules.pop("cookbook_mygraphs", None)


def test_no_snippet_reaches_a_live_backend():
    """The scripted double is the only model the executed snippets touch."""
    for snippet in RUNNABLE:
        assert "get_model(" not in snippet["code"]
        assert "ClaudeCodeCLIChatModel" not in snippet["code"]
        assert "OpenRouter" not in snippet["code"]


# -- the claims, pinned independently of the printed output -------------------


class _State(GraphARCState):
    log: list[str] = []


def _mixed_graph():
    ran: list[str] = []

    def cheap(state: _State) -> dict:
        ran.append("cheap")
        return {"log": [*state.log, "cheap"]}

    async def costly(state: _State) -> dict:
        await asyncio.sleep(0)
        return {"log": [*state.log, "costly"]}

    g = GraphARC(_State, name="mixed")
    g.add_node("cheap", cheap, writes={"log"})
    g.add_node("costly", costly, writes={"log"})
    g.add_edge(START, "cheap")
    g.add_edge("cheap", "costly")
    g.add_edge("costly", END)
    return g.compile(), ran


def test_a_sync_entry_point_refuses_before_any_node_runs():
    """The doc's claim that `cheap ran` prints once, not twice."""
    graph, ran = _mixed_graph()
    with pytest.raises(AsyncNodeError):
        graph.invoke({})
    assert ran == [], "a refused invoke() executed a node"

    assert asyncio.run(graph.ainvoke({}))["log"] == ["cheap", "costly"]
    assert ran == ["cheap"]


def test_stream_refuses_async_nodes_and_names_the_async_twin():
    graph, _ = _mixed_graph()
    with pytest.raises(AsyncNodeError) as caught:
        list(graph.stream({}))
    assert "use astream()" in str(caught.value)


@pytest.mark.parametrize("destination", ["esclate", "", "END "])
def test_an_unknown_goto_raises_rather_than_being_dropped(destination):
    def route(state: _State) -> Command:
        return Command(goto=destination)

    g = GraphARC(_State, name="triage")
    g.add_node("route", route, writes={"log"})
    g.add_node("escalate", lambda state: {"log": ["escalated"]}, writes={"log"})
    g.add_edge(START, "route")
    g.add_edge("escalate", END)
    graph = g.compile()

    with pytest.raises(GraphRoutingError) as caught:
        graph.invoke({})
    assert repr(destination) in str(caught.value)


def test_a_goto_that_names_a_real_node_still_works():
    def route(state: _State) -> Command:
        return Command(goto="escalate")

    g = GraphARC(_State, name="triage")
    g.add_node("route", route, writes={"log"})
    g.add_node("escalate", lambda state: {"log": ["escalated"]}, writes={"log"})
    g.add_edge(START, "route")
    g.add_edge("escalate", END)
    assert g.compile().invoke({})["log"] == ["escalated"]


def test_interrupt_before_is_a_stream_keyword_not_an_invoke_one():
    """The doc calls this out as a sharp edge; pin it so the edge stays where it is."""
    g = GraphARC(_State, name="one")
    g.add_node("only", lambda state: {"log": ["ran"]}, writes={"log"})
    g.add_edge(START, "only")
    g.add_edge("only", END)
    graph = g.compile(checkpointer=InMemorySaver())

    with pytest.raises(TypeError, match="interrupt_before"):
        graph.invoke({}, thread_id="t", interrupt_before=["only"])


def test_a_rejected_gated_node_never_executes_its_body(tmp_path):
    with SessionManager(tmp_path / "sessions") as manager:
        session = manager.create(GRAPH_NAME, session_id="s")
        first = session.run({})
        assert first.status is SessionStatus.AWAITING_APPROVAL
        assert "apply" not in first.state["log"]

        session.decide(approved=False, decided_by="ops", reason="no")
        final = session.run()

    assert final.status is SessionStatus.IDLE
    assert final.skipped == ("apply",)
    assert final.nodes == ("report",)
    # Append-only log: the gated node's body left no mark at all.
    assert final.state["log"] == ["ingest", "plan", "report"]
    assert final.state["outcome"].startswith("refused by ops")


def test_a_fanout_hold_is_per_task_and_the_requests_are_indistinguishable():
    """Approving k of n runs k; the doc says which k is not yours to pick."""
    code = next(s["code"] for s in RUNNABLE if "add_fanout_edge" in s["code"])
    printed = _exec_snippet(code, "cookbook_serving_fanout_claim")

    holds = [line for line in printed.splitlines() if line.startswith("  hold ")]
    assert len(holds) == 3
    assert len(set(holds)) == 1, "the doc claims the three holds are indistinguishable"

    sent = printed.splitlines()[-1]
    assert sent.startswith("sent   : [")
    # Two of three recipients, and exactly one task walked past.
    assert sent.count("@x") == 2
    assert "skipped: ('send',)" in printed


class _AsyncState(GraphARCState):
    question: str
    answer: str = ""


def _async_graph(checkpointer, trace: TraceRecorder):
    async def answer(state: _AsyncState) -> dict:
        await asyncio.sleep(0)
        return {"answer": "42"}

    g = GraphARC(_AsyncState, name="qa", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)


def _drive(client, graph: str) -> dict:
    created = client.post("/sessions", json={"graph": graph, "input": {"question": "q"}})
    session_id = created.json()["id"]
    for _ in range(2000):
        view = client.get(f"/sessions/{session_id}").json()
        if view["status"] in ("succeeded", "failed", "interrupted"):
            return view
    raise AssertionError("session never reached a terminal status")


@pytest.mark.parametrize(
    ("saver", "expected"),
    [("sqlite", "failed"), ("memory", "succeeded")],
)
def test_the_server_needs_an_async_checkpointer_for_async_nodes(saver, expected, tmp_path):
    conn = sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn) if saver == "sqlite" else InMemorySaver()
    registry = ServerRegistry({"g": lambda trace: _async_graph(checkpointer, trace)})
    try:
        with TestClient(create_app(registry=registry)) as client:
            view = _drive(client, "g")
    finally:
        conn.close()

    assert view["status"] == expected
    if expected == "failed":
        # The message the doc prints, and both of the ways out it names.
        assert "CheckpointerNotAsyncError" in view["error"]
        assert "AsyncSqliteSaver" in view["error"]
        assert "InMemorySaver" in view["error"]
    else:
        assert view["result"]["answer"] == "42"


def test_a_sync_graph_on_a_sync_saver_is_served_by_the_fallback_driver(tmp_path):
    """The row of the doc's table that would otherwise only be asserted in prose."""

    class State(GraphARCState):
        question: str
        answer: str = ""

    conn = sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)

    def build(trace: TraceRecorder):
        g = GraphARC(State, name="sync", trace=trace)
        g.add_node("answer", lambda state: {"answer": "42"}, writes={"answer"})
        g.add_edge(START, "answer")
        g.add_edge("answer", END)
        return g.compile(checkpointer=SqliteSaver(conn))

    try:
        with TestClient(create_app(registry=ServerRegistry({"g": build}))) as client:
            view = _drive(client, "g")
    finally:
        conn.close()

    assert view["status"] == "succeeded"
    assert view["result"]["answer"] == "42"


def test_posting_a_message_is_accepted_but_not_applied():
    """`202` does not mean the run was steered — the doc leads with this."""

    class State(GraphARCState):
        question: str
        answer: str = ""

    def build(trace: TraceRecorder):
        g = GraphARC(State, name="qa", trace=trace)
        g.add_node("answer", lambda state: {"answer": "42"}, writes={"answer"})
        g.add_edge(START, "answer")
        g.add_edge("answer", END)
        return g.compile()

    with TestClient(create_app(registry=ServerRegistry({"g": build}))) as client:
        view = _drive(client, "g")
        ack = client.post(
            f"/sessions/{view['id']}/events", json={"type": "message", "data": {}}
        )

    assert ack.status_code == 202
    assert ack.json()["event"]["applied"] is False
    assert "does not deliver" in ack.json()["event"]["detail"]


def test_a_session_directory_is_all_a_second_manager_needs(tmp_path):
    """The restart snippet's mechanism, without the subprocess spawn.

    The subprocess proof is in the doc (and in `tests/test_session.py`); this
    pins the part the doc's prose rests on — that a *fresh* manager over the
    same directory picks the thread up where it stopped, with no node re-run.
    """
    root = tmp_path / "sessions"
    with SessionManager(root) as first:
        session = first.create(GRAPH_NAME, session_id="incident-42")
        assert first.list()[0].id == "incident-42"
        assert session.run({}).status is SessionStatus.AWAITING_APPROVAL

    with SessionManager(root) as second:
        resumed = second.resume("incident-42")
        resumed.decide(approved=True, decided_by="ops")
        turn = resumed.run()

    assert turn.nodes == ("apply", "report")
    assert turn.state["log"] == ["ingest", "plan", "apply", "report"]


def test_the_doc_names_a_real_temp_root_pattern():
    """Every session snippet writes under a fresh temp dir, never the repo."""
    for snippet in RUNNABLE:
        if "SessionManager(" not in snippet["code"]:
            continue
        assert "tempfile.mkdtemp" in snippet["code"]


def test_tempdirs_the_snippets_create_are_outside_the_repo():
    assert not Path(tempfile.gettempdir()).is_relative_to(DOC.parents[2])
