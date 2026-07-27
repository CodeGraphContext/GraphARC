"""Stage 6 gate: run #51 uses a fact from run #12, sees that run #37 superseded
it, and avoids a known dead end.

Every store invariant here runs twice — once against the in-process backend and
once against SQLite — because the two are supposed to be interchangeable. The
SQLite parametrization hands each "run" its own store object over one file, so
the gate can no longer pass just because three runs share a dict.
Cross-process durability is proved in test_memory_persistence.py.
"""

import json
from datetime import UTC, datetime

import pytest

from grapharc.examples.stage6_memory import build_stage6
from grapharc.memory import (
    Claim,
    MemoryStore,
    SQLiteMemoryStore,
    render_context,
    retrieve,
    retrieve_dead_ends,
)
from grapharc.memory import store as store_module
from grapharc.memory.store import _now, validate_supersede


@pytest.fixture(params=["memory", "sqlite"])
def open_store(request, tmp_path):
    """Returns a callable that opens a handle onto one logical store.

    For SQLite each call is a fresh object on the same file, so a caller that
    re-opens per run is testing the stored data rather than a live reference.
    """
    handles = []

    if request.param == "memory":
        shared = MemoryStore()
        yield lambda: shared
        return

    path = tmp_path / "memory.sqlite"

    def _open():
        store = SQLiteMemoryStore(path)
        handles.append(store)
        return store

    yield _open
    for handle in handles:
        handle.close()


def correction_pairs(n: int) -> list[tuple[Claim, Claim]]:
    """`n` (wrong, right) claim pairs, built once so both backends replay the
    identical objects — same ids, same observed_at, no per-backend variation
    left to explain a difference in output."""
    return [
        (
            Claim(subject="X", predicate=f"p{i}", object=f"wrong-{i}", source="s"),
            Claim(subject="X", predicate=f"p{i}", object=f"right-{i}", source="s"),
        )
        for i in range(n)
    ]


def replay(store, pairs: list[tuple[Claim, Claim]]):
    for old, new in pairs:
        store.add(old)
        store.supersede(old.id, new)
    return store


def superseded_objects(context: str) -> list[str]:
    """The object of each rendered dead end, in the order it was rendered.

    Lines look like `- X p39 wrong-39 (superseded by <id>)`; substring matching
    would not do, since "wrong-3" is a prefix of "wrong-37".
    """
    section = context.split("Superseded — do not re-derive these:")[1]
    return [ln.split()[3] for ln in section.splitlines() if ln.startswith("- X ")]


def test_supersede_preserves_history_and_hides_the_old_fact(open_store):
    store = open_store()
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
    assert store.get(old.id).source == "draft-plan.md"
    assert store.get(old.id).observed_at == old.observed_at


def test_double_supersede_is_refused(open_store):
    store = open_store()
    old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
    store.supersede(old.id, Claim(subject="a", predicate="p", object="2", source="s"))
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede(old.id, Claim(subject="a", predicate="p", object="3", source="s"))


def test_superseding_an_unknown_claim_raises(open_store):
    store = open_store()
    with pytest.raises(KeyError):
        store.supersede("nope", Claim(subject="a", predicate="p", object="1", source="s"))


def test_a_refused_supersede_writes_nothing(open_store):
    """The rejected replacement must not land in the store as a loose claim."""
    store = open_store()
    old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
    store.supersede(old.id, Claim(subject="a", predicate="p", object="2", source="s"))
    rejected = Claim(subject="a", predicate="p", object="3", source="s")
    with pytest.raises(ValueError):
        store.supersede(old.id, rejected)
    assert store.get(rejected.id) is None
    assert [c.object for c in store.current("a")] == ["2"]


def test_entity_resolution_is_case_and_punctuation_insensitive(open_store):
    store = open_store()
    store.add(Claim(subject="Neo4j", predicate="is a", object="graph db", source="s"))
    assert store.current("neo4j")
    assert store.current("NEO4J")


def test_non_ascii_entities_stay_distinct(open_store):
    """An ASCII-only normalizer would erase these entirely and merge them into
    one node — cross-contaminating facts about unrelated subjects."""
    store = open_store()
    store.add(Claim(subject="東京", predicate="is in", object="Japan", source="s"))
    store.add(Claim(subject="北京", predicate="is in", object="China", source="s"))

    tokyo = store.current("東京")
    beijing = store.current("北京")
    assert [c.object for c in tokyo] == ["Japan"]
    assert [c.object for c in beijing] == ["China"]


def test_accented_names_do_not_collide_with_their_stripped_form(open_store):
    store = open_store()
    store.add(Claim(subject="Müller", predicate="wrote", object="paper A", source="s"))
    store.add(Claim(subject="M ller", predicate="wrote", object="paper B", source="s"))
    assert [c.object for c in store.current("Müller")] == ["paper A"]


def test_punctuation_only_entity_does_not_become_a_shared_empty_key(open_store):
    store = open_store()
    store.add(Claim(subject="???", predicate="p", object="1", source="s"))
    store.add(Claim(subject="!!!", predicate="p", object="2", source="s"))
    assert [c.object for c in store.current("???")] == ["1"]
    assert [c.object for c in store.current("!!!")] == ["2"]


def test_retrieval_is_bounded(open_store):
    store = open_store()
    for i in range(50):
        store.add(Claim(subject="X", predicate=f"p{i}", object=str(i), source="s"))
    assert len(retrieve(store, entities=["X"], max_claims=5)) == 5


def test_render_context_carries_provenance_and_flags_dead_ends(open_store):
    store = open_store()
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


def test_dead_end_section_is_capped(open_store):
    """A long correction chain must not be able to outgrow the context budget."""
    store = open_store()
    for i in range(40):
        old = store.add(
            Claim(subject="X", predicate=f"p{i}", object=f"wrong-{i}", source="s")
        )
        store.supersede(
            old.id, Claim(subject="X", predicate=f"p{i}", object=f"right-{i}", source="s")
        )

    context = render_context(store, entities=["X"], max_claims=5, max_dead_ends=3)
    superseded_section = context.split("Superseded — do not re-derive these:")[1]
    entries = [
        ln
        for ln in superseded_section.splitlines()
        if ln.startswith("- ") and "superseded by" in ln
    ]
    assert len(entries) == 3
    assert "(+37 older corrections omitted)" in context
    # The cap is what bounds the section, not the number of dead ends stored.
    assert len(store.dead_ends("X")) == 40


def test_dead_ends_kept_are_the_newest_corrections(open_store):
    """*Which* three survive the cap, not just how many.

    Capping discards a tail, so the order decides what a node is actually
    shown — asserting only the count let the section render the three OLDEST
    corrections while claiming "newest first". Forty corrections in a tight
    loop all land inside one clock tick unless the timestamp source is
    strictly increasing, and `list.sort` is stable: it preserves, never
    reverses, a tied run, so `reverse=True` handed them back oldest-first.
    """
    store = replay(open_store(), correction_pairs(40))

    dead, omitted = retrieve_dead_ends(store, entities=["X"], max_dead_ends=3)
    assert [c.object for c in dead] == ["wrong-39", "wrong-38", "wrong-37"]
    assert omitted == 37

    context = render_context(store, entities=["X"], max_claims=5, max_dead_ends=3)
    assert superseded_objects(context) == ["wrong-39", "wrong-38", "wrong-37"]
    assert "wrong-0 " not in context  # the oldest correction is the first to go


def test_both_backends_render_the_same_brief_from_the_same_claims(tmp_path):
    """The two backends are advertised as interchangeable; prove it.

    They were not. `MemoryStore` iterates a dict and `SQLiteMemoryStore`
    orders by rowid, so once timestamps tied, a stable sort simply returned
    whatever each backend's iteration produced: identical claims in, a
    different "Superseded" section out. Nothing in the rendered brief is
    backend-specific and both stores are fed the same `Claim` objects, so the
    two renderings have to match byte for byte.
    """
    pairs = correction_pairs(40)
    mem = replay(MemoryStore(), pairs)

    with SQLiteMemoryStore(tmp_path / "conformance.sqlite") as sql:
        replay(sql, pairs)

        mem_dead, mem_omitted = retrieve_dead_ends(mem, entities=["X"], max_dead_ends=3)
        sql_dead, sql_omitted = retrieve_dead_ends(sql, entities=["X"], max_dead_ends=3)
        assert [c.id for c in mem_dead] == [c.id for c in sql_dead]
        assert [c.object for c in mem_dead] == ["wrong-39", "wrong-38", "wrong-37"]
        assert mem_omitted == sql_omitted == 37

        kwargs = {"entities": ["X"], "max_claims": 5, "max_dead_ends": 3}
        assert render_context(mem, **kwargs) == render_context(sql, **kwargs)


def test_tied_correction_times_still_order_identically(tmp_path):
    """Timestamps can still tie — a coarse platform clock, or two processes
    writing in the same microsecond — so the sort must be a total order over
    the claims themselves. Here the two backends are loaded in OPPOSITE
    insertion orders: anything the sort leaves to iteration order shows up.
    """
    tie = "2026-01-01T00:00:00.000000+00:00"
    claims = [
        Claim(
            id=f"id{i:02d}",
            subject="X",
            predicate=f"p{i}",
            object=f"o{i}",
            source="s",
            observed_at=tie,
            superseded_by=f"new{i}",
            superseded_at=tie,
        )
        for i in range(5)
    ]

    mem = MemoryStore()
    for claim in claims:
        mem.add(claim)
    with SQLiteMemoryStore(tmp_path / "tie.sqlite") as sql:
        for claim in reversed(claims):
            sql.add(claim)

        mem_dead, _ = retrieve_dead_ends(mem, entities=["X"], max_dead_ends=3)
        sql_dead, _ = retrieve_dead_ends(sql, entities=["X"], max_dead_ends=3)
        assert [c.id for c in mem_dead] == ["id04", "id03", "id02"]
        assert [c.id for c in sql_dead] == ["id04", "id03", "id02"]


def test_claim_timestamps_are_strictly_increasing():
    """The mechanism the whole ordering rests on: no two claims written by a
    process can share a timestamp, so sorting by time is a total order."""
    stamps = [_now() for _ in range(1000)]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_a_frozen_clock_cannot_produce_duplicate_timestamps(monkeypatch):
    """Resolution alone is not the fix — it only makes collisions rarer.

    With the clock pinned (the pathological case: a coarse timer, or NTP
    stepping time backwards) the timestamps must still come out distinct and
    increasing, because that is what the guard enforces rather than the host's
    clock granularity happening to be fine enough.
    """

    class _FrozenClock:
        @staticmethod
        def now(tz):
            return datetime(2030, 1, 1, tzinfo=tz)

        fromisoformat = staticmethod(datetime.fromisoformat)

    # monkeypatch restores both module globals afterwards, so the pinned clock
    # cannot leak a year-2030 watermark into the rest of the suite.
    monkeypatch.setattr(store_module, "_last_now", "")
    monkeypatch.setattr(store_module, "datetime", _FrozenClock)

    stamps = [_now() for _ in range(5)]
    assert len(set(stamps)) == 5
    assert stamps == sorted(stamps)
    assert stamps[0] == datetime(2030, 1, 1, tzinfo=UTC).isoformat(timespec="microseconds")


def test_a_claim_cannot_supersede_itself():
    """A replacement carrying the old claim's id is one row asked to replace
    itself, and the backends resolved that differently — `MemoryStore` kept the
    old object (its second dict write clobbered the first), SQLite kept the new
    one (the UPDATE followed the upsert). Both ended with `superseded_by == id`:
    a claim that is its own correction, hidden from `current` and pointing at
    itself in `dead_ends`. Refused before anything is written.

    `MemoryStore` only here — `SQLiteMemoryStore` shares the guard once it
    routes its preconditions through `validate_supersede` too.
    """
    store = MemoryStore()
    old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
    replacement = Claim(id=old.id, subject="a", predicate="p", object="2", source="s")

    with pytest.raises(ValueError, match="cannot supersede itself"):
        store.supersede(old.id, replacement)

    survivor = store.get(old.id)
    assert survivor.object == "1"  # the original was not overwritten
    assert survivor.superseded_by is None  # and is not its own correction
    assert [c.object for c in store.current("a")] == ["1"]
    assert store.dead_ends("a") == []


def test_validate_supersede_is_the_shared_precondition_check():
    """One implementation of the rules, so backends cannot drift on them."""
    live = Claim(id="live", subject="a", predicate="p", object="1", source="s")
    fresh = Claim(subject="a", predicate="p", object="2", source="s")

    with pytest.raises(KeyError):
        validate_supersede(None, "gone", fresh)
    with pytest.raises(ValueError, match="already superseded"):
        validate_supersede(live.model_copy(update={"superseded_by": "other"}), "live", fresh)
    with pytest.raises(ValueError, match="cannot supersede itself"):
        validate_supersede(live, "live", live.model_copy(update={"object": "2"}))

    assert validate_supersede(live, "live", fresh) is live


@pytest.mark.timeout(30)
def test_gate_later_run_reuses_a_fact_and_avoids_a_superseded_dead_end(open_store, trace):
    """The Stage 6 gate, end to end across three runs.

    Each run opens its own store handle: under the SQLite parametrization the
    three runs share a file and nothing else.
    """
    from grapharc.testing import ScriptedChatModel

    # Run #12 — learns a fact from a source.
    store12 = open_store()
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
    r12 = build_stage6(m12, store12, trace=trace).invoke(
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
    store37 = open_store()
    wrong = store37.add(
        Claim(
            subject="GraphARC",
            predicate="memory backend",
            object="a vector store",
            source="early-sketch.md",
            run_id="run12",
        )
    )
    store37.supersede(
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
    store51 = open_store()
    m51 = ScriptedChatModel(
        responses=[
            json.dumps({"claims": []}),  # nothing new in this source
            "GraphARC runs on LangGraph and stores memory in a provenance graph.",
        ]
    )
    r51 = build_stage6(m51, store51, trace=trace).invoke(
        {"entities": ["GraphARC"], "source_name": "notes.md", "source_text": "(nothing new)"},
        run_id="run51",
    )

    # (a) uses a fact first learned in run #12
    assert "LangGraph" in r51["recalled"]
    assert store51.get(learned_id).run_id == "run12"
    # (b) sees what run #37 superseded, and (c) is told not to re-derive it
    assert any("vector store" in d for d in r51["avoided_dead_ends"])
    assert "Superseded" in r51["recalled"]
    # The superseded value is not presented as current knowledge.
    known_facts = r51["recalled"].split("Superseded")[0]
    assert "vector store" not in known_facts
    assert "provenance graph" in known_facts
