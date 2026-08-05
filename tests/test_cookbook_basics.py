"""Executable proof for docs/cookbook/01-basics.md.

Every recipe on that page is reproduced here, and every string the page prints as
"Output" is asserted verbatim. A cookbook snippet that no longer runs, or that
runs and prints something else, is the same failure as a README claiming a
guarantee the code does not provide — so the page cannot drift without this file
going red.

Kept deliberately literal rather than factored: the point is that the code in the
doc is the code that runs. Where a snippet prints something a clock or a random
id would change, the snippet prints a stable projection instead, and this file
asserts that same projection.
"""

import asyncio
import operator
import os
import sqlite3
import subprocess
import sys
import time
from typing import Annotated, Literal

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import InvalidUpdateError
from pydantic import BaseModel, Field, StringConstraints, field_validator

from grapharc import (
    Budget,
    BudgetExceeded,
    GraphARC,
    GraphARCState,
    WritePermissionError,
    __version__,
)
from grapharc.gateway import UnknownBackendError, describe
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.convergence import ProgressGuard, StopReason
from grapharc.runtime.fanout import WorkerResult, run_guarded
from grapharc.runtime.graph import (
    END,
    START,
    AsyncNodeError,
    GraphCycleError,
    GraphRoutingError,
    MissingRunContextError,
    RunContext,
    StateTypeError,
)
from grapharc.testing import ScriptedChatModel

# -- "How do I install it?" ------------------------------------------------


def test_version_matches_the_page():
    assert __version__ == "0.1.5"


# -- "How do I build and run my first graph?" ------------------------------


def test_first_graph():
    class State(GraphARCState):
        question: str
        answer: str = ""

    def answer(state: State) -> dict:
        return {"answer": f"42 (you asked: {state.question})"}

    g = GraphARC(State, name="hello")
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)

    assert g.compile().invoke({"question": "meaning of life"}) == {
        "question": "meaning of life",
        "answer": "42 (you asked: meaning of life)",
    }


def test_the_import_paths_the_page_tells_you_to_use():
    """The page claims START/END/GraphCycleError are only on `runtime.graph`."""
    import grapharc
    import grapharc.runtime as runtime

    for name in ("GraphARC", "GraphARCState", "Budget", "BudgetExceeded", "BudgetMeter"):
        assert hasattr(grapharc, name), name
    assert hasattr(grapharc, "WritePermissionError")

    for name in ("START", "END", "GraphCycleError"):
        assert not hasattr(grapharc, name), name
        assert not hasattr(runtime, name), name

    for name in ("StateTypeError", "GraphRoutingError", "MissingRunContextError"):
        assert hasattr(runtime, name), name


# -- "Why is a field missing from the result dict?" ------------------------


def test_unwritten_defaults_are_absent_from_the_result():
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
    assert result == {"question": "?", "answer": "42"}
    assert State(**result).untouched == "default never written"


# -- "How do I say what a node is allowed to write?" -----------------------


def test_an_undeclared_write_raises_with_the_message_on_the_page():
    class State(GraphARCState):
        draft: str = ""
        published: str = ""

    def write_draft(state: State) -> dict:
        return {"draft": "hello", "published": "hello"}

    g = GraphARC(State, name="perms")
    g.add_node("write_draft", write_draft, writes={"draft"})
    g.add_edge(START, "write_draft")
    g.add_edge("write_draft", END)

    with pytest.raises(WritePermissionError) as caught:
        g.compile().invoke({})
    assert str(caught.value) == (
        "node 'write_draft' wrote undeclared fields ['published']; "
        "declared writes: ['draft']"
    )

    with pytest.raises(WritePermissionError) as caught:
        GraphARC(State, name="perms").add_node("x", write_draft, writes={"publised"})
    assert str(caught.value) == (
        "node 'x' declares writes to unknown state fields: ['publised']"
    )


def test_a_node_that_writes_nothing_declares_nothing_and_may_return_none():
    class State(GraphARCState):
        n: int = 0

    g = GraphARC(State, name="noop")
    g.add_node("noop", lambda s: None, writes=set())
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    assert g.compile().invoke({"n": 1}) == {"n": 1}


# -- "How do I stop a node from mutating state behind my back?" ------------


def test_in_place_mutation_of_a_nested_model_does_not_escape_the_node():
    class Report(BaseModel):
        title: str = "untitled"

    class State(GraphARCState):
        report: Report = Report()
        seen: str = ""

    def sneak(state: State) -> dict:
        state.report.title = "rewritten in place"
        return {"seen": state.report.title}

    g = GraphARC(State, name="isolation")
    g.add_node("sneak", sneak, writes={"seen"})
    g.add_edge(START, "sneak")
    g.add_edge("sneak", END)

    result = g.compile().invoke({"report": Report(title="original")})
    assert result["seen"] == "rewritten in place"
    assert result["report"].title == "original"


# -- "How do I get a bad value rejected before it reaches the next node?" --


class TypedState(GraphARCState):
    count: int = 0
    retries: Annotated[int, Field(ge=0)] = 0
    mode: Literal["fast", "slow"] = "fast"
    slug: Annotated[str, StringConstraints(to_lower=True)] = ""


def _typed_one_node(fn, *, writes):
    g = GraphARC(TypedState, name="types")
    g.add_node("n", fn, writes=writes)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def test_a_wrong_type_is_refused_with_the_message_on_the_page():
    with pytest.raises(StateTypeError) as caught:
        _typed_one_node(lambda s: {"count": "seven"}, writes={"count"}).invoke({})
    assert str(caught.value) == (
        "node 'n' wrote 'count' with a value the state schema rejects: expected int, "
        "got str ('seven'); Input should be a valid integer, unable to parse string "
        "as an integer"
    )


def test_an_annotation_constraint_is_enforced_at_write_time():
    with pytest.raises(StateTypeError) as caught:
        _typed_one_node(lambda s: {"retries": -1}, writes={"retries"}).invoke({})
    assert str(caught.value) == (
        "node 'n' wrote 'retries' with a value the state schema rejects: expected int, "
        "got int (-1); Input should be greater than or equal to 0"
    )


def test_the_validated_value_is_what_lands_in_state():
    out = _typed_one_node(lambda s: {"count": "7"}, writes={"count"}).invoke({})
    assert out["count"] == 7
    assert isinstance(out["count"], int)


def test_the_other_annotation_forms_the_page_recommends_also_bite():
    """The page tells readers to prefer Annotated constraints over validators."""
    with pytest.raises(StateTypeError, match="Literal"):
        _typed_one_node(lambda s: {"mode": "medium"}, writes={"mode"}).invoke({})
    out = _typed_one_node(lambda s: {"slug": "MiXeD"}, writes={"slug"}).invoke({})
    assert out["slug"] == "mixed"


# -- "Does my state model's own @field_validator run when a node writes?" --


class ValidatedState(GraphARCState):
    slug: str = "ok"

    @field_validator("slug")
    @classmethod
    def must_be_lower(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("slug must be lowercase")
        return v


def _shout(state: ValidatedState) -> dict:
    return {"slug": "NOT-LOWER"}


def test_a_field_validator_runs_on_direct_construction():
    with pytest.raises(Exception) as caught:
        ValidatedState(slug="NOT-LOWER")
    assert type(caught.value).__name__ == "ValidationError"


def test_a_field_validator_does_not_run_on_a_write_from_the_last_node():
    g = GraphARC(ValidatedState, name="gap")
    g.add_node("shout", _shout, writes={"slug"})
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    assert g.compile().invoke({}) == {"slug": "NOT-LOWER"}


class Article(BaseModel):
    slug: str

    @field_validator("slug")
    @classmethod
    def must_be_lower(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("slug must be lowercase")
        return v


def test_a_nested_models_validator_does_run_at_write_time():
    """The page's workaround: push the invariant one level down, where the
    annotation is the model and validating the write constructs it."""

    class State(GraphARCState):
        article: Article | None = None

    def publish(state) -> dict:
        return {"article": {"slug": "NOT-LOWER"}}

    g = GraphARC(State, name="nested")
    g.add_node("publish", publish, writes={"article"})
    g.add_edge(START, "publish")
    g.add_edge("publish", END)

    with pytest.raises(StateTypeError) as caught:
        g.compile().invoke({})
    message = str(caught.value)
    assert message.startswith(
        "node 'publish' wrote 'article' with a value the state schema rejects: "
    )
    assert message.endswith(
        "Article | None, got dict ({'slug': 'NOT-LOWER'}); "
        "Value error, slug must be lowercase (at article.slug)"
    )


def test_a_field_validator_does_run_when_the_next_node_receives_the_state():
    g = GraphARC(ValidatedState, name="gap2")
    g.add_node("shout", _shout, writes={"slug"})
    g.add_node("read_it", lambda s: {}, writes=set())
    g.add_edge(START, "shout")
    g.add_edge("shout", "read_it")
    g.add_edge("read_it", END)
    with pytest.raises(Exception) as caught:
        g.compile().invoke({})
    assert type(caught.value).__name__ == "ValidationError"


# -- "How do I stop a graph that will not stop itself?" --------------------


def test_max_iterations_stops_a_runaway_loop():
    class State(GraphARCState):
        spins: int = 0

    def spin(state: State) -> dict:
        return {"spins": state.spins + 1}

    def again(state: State) -> str:
        return "go"

    g = GraphARC(State, name="runaway", budget=Budget(max_iterations=5))
    g.add_node("spin", spin, writes={"spins"})
    g.add_edge(START, "spin")
    g.add_conditional_edge("spin", again, {"go": "spin", "stop": END})

    compiled = g.compile()
    with pytest.raises(BudgetExceeded) as caught:
        compiled.invoke({})
    assert caught.value.reason == "max_iterations reached (5/5)"
    assert compiled.last_run.meter.iterations == 5


def test_a_per_invoke_budget_overrides_the_graphs_own():
    """The page says invoke(..., budget=...) wins for one call."""

    class State(GraphARCState):
        n: int = 0

    g = GraphARC(State, name="override", budget=Budget(max_iterations=100))
    g.add_node("a", lambda s: {"n": s.n + 1}, writes={"n"})
    g.add_edge(START, "a")
    g.add_conditional_edge("a", lambda s: "go", {"go": "a", "stop": END})

    with pytest.raises(BudgetExceeded) as caught:
        g.compile().invoke({}, budget=Budget(max_iterations=3))
    assert caught.value.reason == "max_iterations reached (3/3)"


def test_the_meter_is_fresh_per_invoke():
    class State(GraphARCState):
        n: int = 0

    g = GraphARC(State, name="fresh", budget=Budget(max_iterations=10))
    g.add_node("a", lambda s: {"n": s.n + 1}, writes={"n"})
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()

    compiled.invoke({})
    assert compiled.last_run.meter.iterations == 1
    compiled.invoke({})
    assert compiled.last_run.meter.iterations == 1


# -- "How do I cap tokens and wall-clock time?" ----------------------------


class BoundedState(GraphARCState):
    reply: str = ""


def _bounded(fn, budget):
    g = GraphARC(BoundedState, name="bounded", budget=budget)
    g.add_node("n", fn, writes={"reply"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def test_max_tokens_stops_a_node_that_calls_a_model():
    def talk(state: BoundedState) -> dict:
        model = ScriptedChatModel(responses=["a reply long enough to cost something"])
        return {"reply": str(model.invoke([HumanMessage(content="hello")]).content)}

    with pytest.raises(BudgetExceeded) as caught:
        _bounded(talk, Budget(max_tokens=5)).invoke({})
    assert caught.value.reason == "max_tokens reached (10/5)"


def test_max_seconds_cuts_off_a_node_parked_in_sleep():
    def nap(state: BoundedState) -> dict:
        time.sleep(5.0)
        return {"reply": "finished anyway"}

    t0 = time.monotonic()
    with pytest.raises(BudgetExceeded) as caught:
        _bounded(nap, Budget(max_seconds=0.25)).invoke({})
    elapsed = time.monotonic() - t0
    assert caught.value.reason.split(" (")[0] == (
        "max_seconds reached while node 'n' was running"
    )
    assert elapsed < 1.0, f"the sleep ran to completion ({elapsed:.1f}s)"


# -- "How do I forbid cycles until I actually need one?" -------------------


def test_dag_mode_rejects_a_cycle_at_compile_time():
    class State(GraphARCState):
        n: int = 0

    def bump(state: State) -> dict:
        return {"n": state.n + 1}

    g = GraphARC(State, name="pipeline", dag=True)
    g.add_node("a", bump, writes={"n"})
    g.add_node("b", bump, writes={"n"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("b", END)

    with pytest.raises(GraphCycleError) as caught:
        g.compile()
    assert str(caught.value) == "graph 'pipeline' is dag=True but has a cycle: a -> b -> a"


def test_dag_mode_rejects_a_conditional_edge_when_it_is_added():
    class State(GraphARCState):
        n: int = 0

    with pytest.raises(GraphCycleError) as caught:
        GraphARC(State, name="pipeline", dag=True).add_node(
            "a", lambda s: {"n": s.n + 1}, writes={"n"}
        ).add_conditional_edge("a", lambda s: "go", {"go": END})
    assert str(caught.value) == (
        "graph 'pipeline' is dag=True: conditional edges are not allowed"
    )


# -- "How do I write a loop that ends for a reason?" -----------------------


def test_a_cycle_ends_with_a_recorded_termination_reason():
    class State(GraphARCState):
        round: int = 0
        findings: int = 0
        empty_rounds: int = 0
        termination_reason: str | None = None

    hits = [2, 1, 0, 0, 0]
    guard = ProgressGuard(target=10, max_rounds=8, max_empty_rounds=2)

    def search(state: State) -> dict:
        found = hits[state.round] if state.round < len(hits) else 0
        return {
            "round": state.round + 1,
            "findings": state.findings + found,
            "empty_rounds": 0 if found else state.empty_rounds + 1,
        }

    def decide(state: State) -> dict:
        stop = guard.assess(
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
    assert result == {
        "round": 4,
        "findings": 3,
        "empty_rounds": 2,
        "termination_reason": "no_progress",
    }
    assert StopReason(result["termination_reason"]) is StopReason.NO_PROGRESS
    assert compiled.last_run.meter.iterations == 8


# -- "How do I see what actually happened?" --------------------------------


def test_the_trace_carries_what_the_page_says_each_phase_carries(tmp_path):
    class State(GraphARCState):
        items: list[str] = []
        total: int = 0

    def load(state: State) -> dict:
        return {"items": ["a", "b", "c"]}

    def count(state: State) -> dict:
        raise ValueError("the counter is not implemented yet")

    trace = TraceRecorder(tmp_path / "trace.jsonl")
    g = GraphARC(State, name="counter", trace=trace)
    g.add_node("load", load, writes={"items"})
    g.add_node("count", count, writes={"total"})
    g.add_edge(START, "load")
    g.add_edge("load", "count")
    g.add_edge("count", END)

    with pytest.raises(ValueError):
        g.compile().invoke({}, thread_id="demo")

    volatile = {"ts", "run_id", "thread_id", "duration_ms"}
    printed = [
        {k: v for k, v in e.model_dump(exclude_none=True).items() if k not in volatile}
        for e in trace.read_events()
    ]
    assert printed == [
        {
            "attempt": 1,
            "graph": "counter",
            "node": "topology",
            "phase": "topology",
            "step": 0,
            "state_delta": {
                "nodes": ["load", "count"],
                "edges": [
                    ["__start__", "load", "static"],
                    ["load", "count", "static"],
                    ["count", "__end__", "static"],
                ],
            },
        },
        {"attempt": 1, "graph": "counter", "node": "load", "phase": "start", "step": 1},
        {
            "attempt": 1,
            "graph": "counter",
            "node": "load",
            "phase": "end",
            "step": 1,
            "state_delta": {"items": ["a", "b", "c"]},
            "tokens": 0,
        },
        {"attempt": 1, "graph": "counter", "node": "count", "phase": "start", "step": 2},
        {
            "attempt": 1,
            "graph": "counter",
            "node": "count",
            "phase": "error",
            "step": 2,
            "tokens": 0,
            "error": "ValueError('the counter is not implemented yet')",
        },
    ]

    # The four fields the snippet filters out are on the lines anyway.
    for event in trace.read_events():
        assert event.run_id and event.thread_id == "demo" and event.ts
    assert all(
        e.duration_ms is not None
        for e in trace.read_events()
        if e.phase not in ("start", "topology")
    )


# -- "How do I see what a run spent, from inside a node?" ------------------


def test_a_two_parameter_node_receives_the_run_context():
    class State(GraphARCState):
        reply: str = ""
        spent_so_far: int = 0

    def talk(state: State, ctx: RunContext) -> dict:
        model = ScriptedChatModel(responses=["a scripted reply"])
        reply = str(model.invoke([HumanMessage(content="hi")]).content)
        return {"reply": reply, "spent_so_far": ctx.meter.tokens}

    g = GraphARC(State, name="ctx", budget=Budget(max_tokens=1000))
    g.add_node("talk", talk, writes={"reply", "spent_so_far"})
    g.add_edge(START, "talk")
    g.add_edge("talk", END)

    compiled = g.compile()
    assert compiled.invoke({}) == {"reply": "a scripted reply", "spent_so_far": 5}
    assert compiled.last_run.meter.iterations == 1
    assert compiled.last_run.meter.tokens == 5


def test_the_defaulted_second_parameter_trap_the_page_warns_about():
    class State(GraphARCState):
        got: str = ""

    def node(state, retries=3):
        return {"got": type(retries).__name__}

    g = GraphARC(State, name="ctxtrap")
    g.add_node("n", node, writes={"got"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    assert g.compile().invoke({}) == {"got": "RunContext"}


def test_an_unnamed_manual_charge_pays_twice_and_a_named_one_does_not():
    class State(GraphARCState):
        ledger: list[int] = []

    def talk(state: State, ctx: RunContext) -> dict:
        model = ScriptedChatModel(responses=["a scripted reply"])
        message = model.invoke([HumanMessage(content="hi")])
        total = message.usage_metadata["total_tokens"]
        automatic = ctx.meter.tokens
        ctx.meter.charge_tokens(total)
        unnamed = ctx.meter.tokens
        ctx.meter.charge_tokens(total, source=message)
        return {"ledger": [automatic, unnamed, ctx.meter.tokens]}

    g = GraphARC(State, name="charge")
    g.add_node("t", talk, writes={"ledger"})
    g.add_edge(START, "t")
    g.add_edge("t", END)
    automatic, unnamed, named = g.compile().invoke({})["ledger"]
    assert unnamed == automatic * 2, "an unnamed re-report is counted again"
    assert named == unnamed, "naming the source makes the re-report free"


# -- "How do I resume a run that died halfway?" ----------------------------


def test_resume_reruns_only_what_did_not_finish(tmp_path):
    class State(GraphARCState):
        url: str
        data: str = ""
        saved: bool = False

    fetches: list[str] = []
    crash_once = [True]

    def fetch(state: State) -> dict:
        fetches.append(state.url)
        return {"data": f"payload from {state.url}"}

    def save(state: State) -> dict:
        if crash_once[0]:
            crash_once[0] = False
            raise RuntimeError("disk full")
        return {"saved": True}

    conn = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    try:
        compiled = (
            GraphARC(State, name="fetch_save")
            .add_node("fetch", fetch, writes={"data"})
            .add_node("save", save, writes={"saved"})
            .add_edge(START, "fetch")
            .add_edge("fetch", "save")
            .add_edge("save", END)
            .compile(checkpointer=SqliteSaver(conn))
        )

        with pytest.raises(RuntimeError, match="disk full"):
            compiled.invoke({"url": "https://example.invalid/doc"}, thread_id="t1")

        assert compiled.get_state("t1").next == ("save",)

        assert compiled.invoke(None, thread_id="t1") == {
            "url": "https://example.invalid/doc",
            "data": "payload from https://example.invalid/doc",
            "saved": True,
        }
        assert fetches == ["https://example.invalid/doc"]
    finally:
        conn.close()


def test_trace_step_numbers_continue_across_a_resume(tmp_path):
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

    trace = TraceRecorder(tmp_path / "trace.jsonl")
    conn = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    try:
        g = GraphARC(State, name="resume", trace=trace)
        g.add_node("fetch", fetch, writes={"data"})
        g.add_node("save", save, writes={"saved"})
        g.add_edge(START, "fetch")
        g.add_edge("fetch", "save")
        g.add_edge("save", END)
        compiled = g.compile(checkpointer=SqliteSaver(conn))

        with pytest.raises(RuntimeError):
            compiled.invoke({}, thread_id="t1")
        compiled.invoke(None, thread_id="t1")

        assert [
            (e.attempt, e.step, e.node, e.phase) for e in trace.read_events()
        ] == [
            (1, 0, "topology", "topology"),
            (1, 1, "fetch", "start"),
            (1, 1, "fetch", "end"),
            (1, 2, "save", "start"),
            (1, 2, "save", "error"),
            (2, 0, "topology", "topology"),
            (2, 3, "save", "start"),
            (2, 3, "save", "end"),
        ]
    finally:
        conn.close()


# -- "Does that survive a real process kill?" ------------------------------

_KILL_SCRIPT = '''
import os
import signal
import sqlite3
import sys
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
        Path(db + ".fetches").open("a").write("fetch\\n")
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


if sys.argv[1] == "child":
    conn, compiled = build(sys.argv[2])
    compiled.invoke({"url": "https://example.invalid/doc"}, thread_id="t1")
else:
    db = sys.argv[2]
    conn, compiled = build(db)
    history = list(compiled.get_state_history("t1"))
    print("a checkpoint that resumes at 'save':", any(s.next == ("save",) for s in history))
    print("resumed:", compiled.invoke(None, thread_id="t1")["saved"])
    print("times fetch ran:", Path(db + ".fetches").read_text().count("fetch"))
    conn.close()
'''


@pytest.mark.timeout(120)
def test_a_sigkill_loses_the_last_checkpoint_so_nodes_can_rerun(tmp_path):
    """The page's honest caveat: async durability is LangGraph's default and
    `CompiledGraphARC.invoke()` does not expose `durability=`, so a hard kill
    replays from the input checkpoint. If this ever stops being true, the page
    is wrong and must be updated."""
    script = tmp_path / "resume_after_kill.py"
    script.write_text(_KILL_SCRIPT, encoding="utf-8")
    db = str(tmp_path / "checkpoints.sqlite")

    child = subprocess.run(
        [sys.executable, str(script), "child", db],
        env={**os.environ, "KILL_ME": "1"},
        capture_output=True,
        text=True,
    )
    assert child.returncode == -9, f"child was not SIGKILLed: {child.returncode}"

    parent = subprocess.run(
        [sys.executable, str(script), "parent", db], capture_output=True, text=True
    )
    assert parent.returncode == 0, parent.stderr
    assert parent.stdout.splitlines() == [
        "a checkpoint that resumes at 'save': False",
        "resumed: True",
        "times fetch ran: 2",
    ]


# -- "Why can't I just call .inner.invoke()?" ------------------------------


def test_raw_langgraph_entry_points_fail_closed():
    class State(GraphARCState):
        n: int = 0

    g = GraphARC(State, name="closed")
    g.add_node("bump", lambda s: {"n": s.n + 1}, writes={"n"})
    g.add_edge(START, "bump")
    g.add_edge("bump", END)
    compiled = g.compile()

    with pytest.raises(MissingRunContextError) as caught:
        compiled.inner.invoke({})
    assert str(caught.value) == (
        "node 'bump' executed without a GraphARC run context; drive the graph via "
        "CompiledGraphARC.invoke()/stream()/ainvoke()/astream() — raw LangGraph entry "
        "points would silently bypass budgets and traces"
    )

    # ...but reading through `.inner` is fine, which is what the page calls an
    # inspection escape hatch.
    assert "bump" in compiled.inner.get_graph().draw_mermaid()


# -- "How do I fan out across workers?" ------------------------------------


class Shard(BaseModel):
    name: str
    words: list[str] = []
    fail: bool = False
    hang: float = 0.0


def test_bounded_fanout_survives_a_crashed_worker_and_a_hung_one():
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

    out = g.compile().invoke(
        {"words": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]}
    )
    assert out["counted"] == 2
    assert sorted(out["failures"]) == [
        "w1: RuntimeError('shard parser blew up')",
        "w2: timeout after 0.2s",
    ]


class UnguardedState(GraphARCState):
    seen: Annotated[list[str], operator.add] = []


def _unguarded(dispatcher):
    def worker(shard: Shard) -> dict:
        if shard.fail:
            raise RuntimeError("shard parser blew up")
        return {"seen": [shard.name]}

    g = GraphARC(UnguardedState, name="unguarded")
    g.add_node("prepare", lambda s: None, writes=set())
    g.add_node("worker", worker, writes={"seen"}, input_schema=Shard)
    g.add_edge(START, "prepare")
    g.add_fanout_edge("prepare", dispatcher)
    g.add_edge("worker", END)
    return g.compile()


def test_without_run_guarded_one_bad_worker_takes_the_run_down():
    def dispatch(state):
        return [("worker", Shard(name="w0")), ("worker", Shard(name="w1", fail=True))]

    with pytest.raises(RuntimeError, match="shard parser blew up"):
        _unguarded(dispatch).invoke({})


def test_a_dispatcher_naming_an_unknown_node_fails_closed():
    with pytest.raises(GraphRoutingError) as caught:
        _unguarded(lambda s: [("wroker", Shard(name="w0"))]).invoke({})
    assert str(caught.value) == (
        "the fan-out dispatcher on node 'prepare' routed to Send(node='wroker', ...), "
        "but 'wroker' is not a node of graph 'unguarded'; LangGraph drops an unknown "
        "Send target and the run continues as if the routing had not happened. Valid "
        "Send targets: 'prepare', 'worker' — END is not one, because a Send has to "
        "name a node that runs"
    )


def test_parallel_writes_need_a_reducer():
    """The page names the exact error you get without one."""

    class State(GraphARCState):
        seen: list[str] = []          # no Annotated reducer

    g = GraphARC(State, name="nored")
    g.add_node("prepare", lambda s: None, writes=set())
    g.add_node("worker", lambda shard: {"seen": [shard.name]}, writes={"seen"},
               input_schema=Shard)
    g.add_edge(START, "prepare")
    g.add_fanout_edge(
        "prepare", lambda s: [("worker", Shard(name="a")), ("worker", Shard(name="b"))]
    )
    g.add_edge("worker", END)

    with pytest.raises(InvalidUpdateError) as caught:
        g.compile().invoke({})
    assert "At key 'seen': Can receive only one value per step." in str(caught.value)


# -- "Can my nodes be async?" ----------------------------------------------


class AsyncState(GraphARCState):
    reply: str = ""


def _async_graph():
    async def fetch(state: AsyncState) -> dict:
        await asyncio.sleep(0)
        return {"reply": "from an async node"}

    g = GraphARC(AsyncState, name="async")
    g.add_node("fetch", fetch, writes={"reply"})
    g.add_edge(START, "fetch")
    g.add_edge("fetch", END)
    return g.compile()


@pytest.mark.asyncio
async def test_an_async_node_runs_through_ainvoke():
    assert await _async_graph().ainvoke({}) == {"reply": "from an async node"}


def test_a_sync_entry_point_refuses_a_graph_with_async_nodes():
    with pytest.raises(AsyncNodeError) as caught:
        _async_graph().invoke({})
    assert str(caught.value) == (
        "graph 'async' has async nodes ['fetch']; use ainvoke(), because invoke() has "
        "no event loop to await them on"
    )


# -- "How do I swap the scripted model for a real one?" --------------------
#
# The real-model snippet on the page is explicitly marked as unrun. What is
# checkable without a key is spec resolution, which is what the page shows.


def test_describe_resolves_the_specs_the_page_prints():
    assert describe("claude-cli/claude-sonnet-5") == {
        "spec": "claude-cli/claude-sonnet-5",
        "backend": "claude-cli",
        "model": "claude-sonnet-5",
    }
    assert describe("openrouter/anthropic/claude-haiku-4.5") == {
        "spec": "openrouter/anthropic/claude-haiku-4.5",
        "backend": "openrouter",
        "model": "anthropic/claude-haiku-4.5",
    }


def test_a_mistyped_backend_is_rejected_rather_than_folded_into_a_model_name():
    with pytest.raises(UnknownBackendError, match="unknown backend 'opnerouter'"):
        describe("opnerouter/anthropic/x")
