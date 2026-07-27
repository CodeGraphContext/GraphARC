"""Capstone gate: question -> fan-out -> verify -> cited answer -> facts
persisted -> a second run reuses them. Every subsystem in one path."""

import re

import pytest

from grapharc.examples.capstone import DEMO_CORPUS, build_capstone
from grapharc.memory import MemoryStore
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.runtime.budget import Budget
from grapharc.runtime.convergence import StopReason
from grapharc.testing import ScriptedChatModel

QUESTION = "How does GraphARC bound work with budgets?"


def _reviewer(n: int) -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=['{"supported": true, "reason": "quote supports it"}'] * n,
        on_exhausted="repeat",
    )


@pytest.mark.timeout(30)
def test_gate_end_to_end_answer_then_reuse(trace):
    store = MemoryStore()

    worker = ScriptedChatModel(responses=["Budgets cap iterations and tokens [doc 0]."])
    compiled = build_capstone(
        worker, _reviewer(8), store, trace=trace, budget=Budget(max_iterations=50)
    )
    first = compiled.invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r1",
    )

    # Answer produced from verified evidence only.
    assert first["termination_reason"] == StopReason.TARGET_MET.value
    assert first["answer"]
    assert first["evidence"]
    assert all(v.anchor_ok for v in first["verdicts"])
    assert first["failures"] == []

    # Facts were persisted with provenance from this run.
    assert first["persisted_claim_ids"]
    persisted = store.get(first["persisted_claim_ids"][0])
    assert persisted.run_id == "r1"
    assert persisted.source == QUESTION

    # A second run recalls what the first learned before doing any work.
    worker2 = ScriptedChatModel(responses=["Same answer, now with prior knowledge."])
    second = build_capstone(worker2, _reviewer(8), store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r2",
    )
    assert "GraphARC" in second["recalled"]
    assert "evidence for" in second["recalled"]  # a fact from run r1
    assert QUESTION in second["recalled"]  # its provenance travelled with it


@pytest.mark.timeout(30)
def test_unverified_evidence_is_never_persisted(trace):
    """A reviewer that rejects everything yields no answer and no memory writes."""
    store = MemoryStore()
    worker = ScriptedChatModel(responses=["unused"], on_exhausted="repeat")
    rejecting = ScriptedChatModel(
        responses=['{"supported": false, "reason": "does not support"}'],
        on_exhausted="repeat",
    )
    result = build_capstone(worker, rejecting, store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r3",
    )
    assert result["termination_reason"] == StopReason.NO_PROGRESS.value
    assert result["answer"] == ""
    assert result.get("persisted_claim_ids", []) == []
    assert store.all_claims() == []


@pytest.mark.timeout(30)
def test_gate_anchor_source_survives_a_rebuilt_graph_resume(trace, checkpointer):
    """The verifier's anchor source must live in checkpointed state: a resumed
    run on a rebuilt graph must still verify against the real corpus."""
    store = MemoryStore()
    worker = ScriptedChatModel(responses=["answer"], on_exhausted="repeat")
    first = build_capstone(
        worker, _reviewer(8), store, trace=trace, checkpointer=checkpointer
    )
    result = first.invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        thread_id="t1",
        run_id="ra",
    )
    assert result["source_text"]  # written by `plan`, visible in state

    # A fresh process would rebuild the graph; the closure would be empty here.
    rebuilt = build_capstone(
        ScriptedChatModel(responses=["answer"], on_exhausted="repeat"),
        _reviewer(8),
        store,
        trace=trace,
        checkpointer=checkpointer,
    )
    state = rebuilt.inner.get_state({"configurable": {"thread_id": "t1"}})
    assert state.values["source_text"] == "\n".join(DEMO_CORPUS)


@pytest.mark.timeout(30)
def test_gate_concurrent_runs_do_not_share_an_anchor_source(trace):
    """Two runs of the same compiled graph with disjoint corpora must each
    verify against their own source, not whichever ran last."""
    store = MemoryStore()
    compiled = build_capstone(
        ScriptedChatModel(responses=["a"], on_exhausted="repeat"),
        _reviewer(8),
        store,
        trace=trace,
    )
    corpus_a = ["Budgets cap iterations and tokens in GraphARC."]
    corpus_b = ["Traces are replay points that budgets never touch."]

    ra = compiled.invoke(
        {"question": QUESTION, "corpus": corpus_a, "entities": ["A"]}, run_id="ca"
    )
    rb = compiled.invoke(
        {"question": QUESTION, "corpus": corpus_b, "entities": ["B"]}, run_id="cb"
    )
    assert ra["source_text"] == corpus_a[0]
    assert rb["source_text"] == corpus_b[0]
    # Each run's evidence anchors against its own corpus.
    for result in (ra, rb):
        for verdict in result["verdicts"]:
            assert verdict.anchor_ok, verdict.reason


def test_same_model_for_worker_and_reviewer_is_refused():
    model = ScriptedChatModel(responses=["x"])
    with pytest.raises(ValueError, match="different model instances"):
        build_capstone(model, model, MemoryStore())


@pytest.mark.timeout(30)
def test_metrics_and_mermaid_derive_from_the_trace(trace):
    store = MemoryStore()
    worker = ScriptedChatModel(responses=["answer"], on_exhausted="repeat")
    build_capstone(worker, _reviewer(8), store, trace=trace).invoke(
        {"question": QUESTION, "corpus": DEMO_CORPUS, "entities": ["GraphARC"]},
        run_id="r4",
    )
    metrics = summarize(trace, "r4")
    assert metrics is not None
    assert metrics.graph == "capstone"
    assert metrics.errors == 0
    assert metrics.nodes_executed >= 6  # recall, plan, 2 workers, verify, answer, remember
    assert metrics.termination_reason == StopReason.TARGET_MET.value
    assert metrics.per_node["search"] == 2  # both fan-out workers ran

    diagram = to_mermaid(trace, "r4")
    assert diagram.startswith("flowchart TD")
    assert "recall" in diagram and "remember" in diagram
    # Fan-out workers are distinct nodes, not a self-loop the graph never had.
    assert not any(
        "-->" in line and line.split("-->")[0].strip() == line.split("-->")[1].strip()
        for line in diagram.splitlines()
    )


def test_mermaid_labels_are_escaped_and_balanced(tmp_path, trace):
    """A run that errors is exactly when an operator needs the diagram — so a
    hostile error string must not break its syntax."""
    from grapharc.runtime.graph import END, START, GraphARC
    from grapharc.runtime.state import GraphARCState

    class S(GraphARCState):
        a: int = 0

    nasty = 'boom [brackets] {braces} "quotes" (parens); ' + "x" * 400

    def explode(state: S) -> dict:
        raise RuntimeError(nasty)

    g = GraphARC(S, name="exploder", trace=trace)
    g.add_node("explode", explode, writes={"a"})
    g.add_edge(START, "explode")
    g.add_edge("explode", END)
    with pytest.raises(RuntimeError):
        g.compile().invoke({"a": 0}, run_id="err1")

    diagram = to_mermaid(trace, "err1")
    assert diagram.startswith("flowchart TD")
    body = diagram.split("\n", 1)[1]

    # Delimiters stay balanced on every line — the line is never sliced open.
    for line in body.splitlines():
        assert line.count("{") == line.count("}"), line
        assert line.count("[") == line.count("]"), line
        assert line.count('"') % 2 == 0, line

    # The error text is present but escaped: no raw delimiter survives in it.
    # Strip valid entity references first — those legitimately contain ';'.
    (error_line,) = [ln for ln in body.splitlines() if "err0" in ln]
    label = error_line.split('{"', 1)[1].rsplit('"}', 1)[0]
    assert "RuntimeError" in label
    assert "#quot;" in label and "#91;" in label
    stripped = re.sub(r"#(?:\d+|quot);", "", label)
    for ch in '[]{}()";':
        assert ch not in stripped, f"raw {ch!r} leaked into the label"
    assert len(label) < len(nasty)  # truncated before escaping, not after
