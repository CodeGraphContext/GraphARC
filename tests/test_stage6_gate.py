"""Stage 6 gate: run #51 uses a fact from run #12, sees that run #37 superseded
it, and avoids a known dead end."""

import json

import pytest

from grapharc.examples.stage6_memory import build_stage6
from grapharc.memory import Claim, MemoryStore, render_context, retrieve


def test_supersede_preserves_history_and_hides_the_old_fact():
    store = MemoryStore()
    old = store.add(
        Claim(
            subject="GraphARC",
            predicate="default backend",
            object="Anthropic API",
            source="draft-plan.md",
            run_id="run12",
        )
    )
    new = store.supersede(
        old.id,
        Claim(
            subject="GraphARC",
            predicate="default backend",
            object="Claude Code CLI",
            source="final-plan.md",
            run_id="run37",
        ),
    )

    # Current view shows only the correction...
    current = store.current("GraphARC", "default backend")
    assert [c.object for c in current] == ["Claude Code CLI"]
    # ...but nothing was destroyed: the history is intact and linked.
    history = store.history("GraphARC", "default backend")
    assert [c.object for c in history] == ["Anthropic API", "Claude Code CLI"]
    assert store.get(old.id).superseded_by == new.id
    assert store.get(old.id).superseded_at is not None
    assert store.get(old.id).run_id == "run12"  # provenance survives


def test_double_supersede_is_refused():
    store = MemoryStore()
    old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
    store.supersede(old.id, Claim(subject="a", predicate="p", object="2", source="s"))
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede(old.id, Claim(subject="a", predicate="p", object="3", source="s"))


def test_entity_resolution_is_case_and_punctuation_insensitive():
    store = MemoryStore()
    store.add(Claim(subject="Neo4j", predicate="is a", object="graph db", source="s"))
    assert store.current("neo4j")
    assert store.current("NEO4J")


def test_retrieval_is_bounded():
    store = MemoryStore()
    for i in range(50):
        store.add(Claim(subject="X", predicate=f"p{i}", object=str(i), source="s"))
    assert len(retrieve(store, entities=["X"], max_claims=5)) == 5


def test_render_context_carries_provenance_and_flags_dead_ends():
    store = MemoryStore()
    old = store.add(
        Claim(subject="X", predicate="uses", object="wrong-db", source="old.md")
    )
    store.supersede(
        old.id, Claim(subject="X", predicate="uses", object="right-db", source="new.md")
    )
    context = render_context(store, entities=["X"])
    assert "right-db" in context
    assert "new.md" in context  # provenance travels with the fact
    assert "Superseded" in context and "wrong-db" in context  # dead end is flagged


@pytest.mark.timeout(20)
def test_gate_later_run_reuses_a_fact_and_avoids_a_superseded_dead_end(trace):
    """The Stage 6 gate, end to end across three runs sharing one store."""
    store = MemoryStore()
    from grapharc.testing import ScriptedChatModel

    # Run #12 — learns a fact from a source.
    m12 = ScriptedChatModel(
        responses=[
            json.dumps(
                {
                    "claims": [
                        {
                            "subject": "GraphARC",
                            "predicate": "orchestration runtime",
                            "object": "LangGraph",
                        }
                    ]
                }
            ),
            "GraphARC runs on LangGraph [final-plan.md].",
        ]
    )
    r12 = build_stage6(m12, store, trace=trace).invoke(
        {
            "entities": ["GraphARC"],
            "source_name": "final-plan.md",
            "source_text": "GraphARC's orchestration runtime is LangGraph.",
        },
        run_id="run12",
    )
    assert len(r12["new_claim_ids"]) == 1
    learned_id = r12["new_claim_ids"][0]

    # Run #37 — corrects an earlier belief; the old claim is superseded.
    wrong = store.add(
        Claim(
            subject="GraphARC",
            predicate="memory backend",
            object="a vector store",
            source="early-sketch.md",
            run_id="run12",
        )
    )
    store.supersede(
        wrong.id,
        Claim(
            subject="GraphARC",
            predicate="memory backend",
            object="a provenance graph",
            source="final-plan.md",
            run_id="run37",
        ),
    )

    # Run #51 — recalls, and must see both the surviving fact and the dead end.
    m51 = ScriptedChatModel(
        responses=[
            json.dumps({"claims": []}),  # nothing new in this source
            "GraphARC runs on LangGraph and stores memory in a provenance graph.",
        ]
    )
    r51 = build_stage6(m51, store, trace=trace).invoke(
        {"entities": ["GraphARC"], "source_name": "notes.md", "source_text": "(nothing new)"},
        run_id="run51",
    )

    # (a) uses a fact first learned in run #12
    assert "LangGraph" in r51["recalled"]
    assert store.get(learned_id).run_id == "run12"
    # (b) sees what run #37 superseded, and (c) is told not to re-derive it
    assert any("vector store" in d for d in r51["avoided_dead_ends"])
    assert "Superseded" in r51["recalled"]
    # The superseded value is not presented as current knowledge.
    known_facts = r51["recalled"].split("Superseded")[0]
    assert "vector store" not in known_facts
    assert "provenance graph" in known_facts
