"""Retrieval, contradiction detection, and the artifact store.

Three claims are under test here, and each one used to be false:

* **retrieval is retrieval.** It was a linear scan with exact normalized-subject
  matching, labelled "GraphRAG". A fact could not be found by a word in its
  object, ranking was recency, and nothing followed an edge. The tests below
  fail against that implementation rather than merely passing alongside it.
* **contradictions are noticed.** `supersede` needs the old claim's id, so
  nothing detected that a new claim disagreed with a stored one — memory simply
  held both and rendered them as if they agreed.
* **artifacts exist.** Claims were the only durable thing; a file an agent
  produced had nowhere to go, and no provenance if it did.

Every store-level test runs against both backends, because "the same
`ClaimStore` protocol" is a claim about behaviour and not about method names.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from grapharc.memory import (
    Artifact,
    ArtifactStore,
    Claim,
    ClaimIndex,
    ClaimStore,
    HashingEmbedder,
    MemoryArtifactStore,
    MemoryStore,
    SQLiteArtifactStore,
    SQLiteMemoryStore,
    add_and_detect,
    detect_contradictions,
    render_artifacts,
    render_context,
    retrieve,
    search,
    supersede_conflicting,
    tokenize,
    unresolved_contradictions,
)
from grapharc.memory import artifacts as artifacts_module
from grapharc.memory.embeddings import Embedder, cosine
from grapharc.memory.index import MIN_SIMILARITY, claim_text
from grapharc.memory.retrieval import _estimate_tokens
from grapharc.memory.text import fold_plural

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_INTERFACE = (
    "put",
    "get",
    "read",
    "read_text",
    "latest",
    "versions",
    "by_run",
    "all_artifacts",
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """One claim store per backend. Every test using it runs twice."""
    if request.param == "memory":
        yield MemoryStore()
        return
    backend = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    yield backend
    backend.close()


@pytest.fixture(params=["memory", "sqlite"])
def artifacts(request, tmp_path):
    if request.param == "memory":
        yield MemoryArtifactStore()
        return
    backend = SQLiteArtifactStore(tmp_path / "memory.sqlite")
    yield backend
    backend.close()


def fill(target: ClaimStore, triples, source: str = "s") -> list[Claim]:
    """Add (subject, predicate, object) triples in order and return the claims."""
    return [
        target.add(Claim(subject=s, predicate=p, object=o, source=source))
        for s, p, o in triples
    ]


def objects(hits) -> list[str]:
    return [hit.claim.object if hasattr(hit, "claim") else hit.object for hit in hits]


def subjects(hits) -> list[str]:
    return [hit.claim.subject if hasattr(hit, "claim") else hit.subject for hit in hits]


class CountingEmbedder:
    """Wraps an embedder and records how much text it was asked to embed."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.calls = 0
        self.texts = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return self.inner.embed(texts)


class SynonymEmbedder:
    """A stand-in for a real semantic model: synonyms share an axis.

    Crude on purpose. The point is not that it is a good embedder but that it
    knows something no lexical channel can know — that "automobile" and "car"
    are the same thing — so if a search finds the car claim from the word
    "automobile", the injected embedder is genuinely being consulted.
    """

    GROUPS = (("car", "automobile", "vehicle"), ("price", "cost"))

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * (len(self.GROUPS) + 1)
            tokens = set(tokenize(text))
            for position, group in enumerate(self.GROUPS):
                if tokens & set(group):
                    vector[position] = 1.0
            if not any(vector):
                vector[-1] = 1.0
            vectors.append(vector)
        return vectors


# --------------------------------------------------------------------------
# Tokenization — the layer every scoring channel shares
# --------------------------------------------------------------------------


def test_the_tokenizer_folds_case_unicode_and_plurals():
    assert tokenize("Müller's BACKENDS") == ["müller", "s", "backend"]
    # Non-Latin scripts survive as tokens rather than being erased.
    assert tokenize("東京 Neo4j") == ["東京", "neo4j"]
    assert fold_plural("libraries") == "library"
    # Words that merely end in s are not mangled into a different word.
    assert fold_plural("analysis") == "analysis"
    assert fold_plural("status") == "status"
    assert fold_plural("class") == "class"


def test_a_plural_query_term_matches_a_singular_claim(store):
    fill(store, [("GraphARC", "supports", "two memory backend implementations")])
    assert retrieve(store, query="BACKENDS")


# --------------------------------------------------------------------------
# Lexical relevance — BM25F, not an exact match
# --------------------------------------------------------------------------


def test_a_claim_is_findable_by_a_word_in_its_object(store):
    """The defect that made "GraphRAG" a misnomer: only subjects were matched.

    Nothing here names the subject, so an exact-subject scan returns nothing.
    """
    fill(
        store,
        [
            ("GraphARC", "memory backend", "a provenance graph"),
            ("GraphARC", "orchestration runtime", "LangGraph"),
        ],
    )
    found = retrieve(store, query="provenance")
    assert [c.object for c in found] == ["a provenance graph"]


def test_relevance_outranks_recency(store):
    """Recency ordering hands back the newest noise; relevance does not."""
    fill(store, [("GraphARC", "memory backend", "SQLite")])
    fill(store, [("GraphARC", f"note {i}", f"detail {i}") for i in range(10)])

    by_recency = retrieve(store, entities=["GraphARC"], max_claims=3)
    assert by_recency[0].object == "detail 9"

    by_relevance = retrieve(store, query="memory backend", entities=["GraphARC"], max_claims=3)
    assert by_relevance[0].object == "SQLite"


def test_a_rare_term_counts_for_more_than_a_common_one(store):
    """IDF, which is the difference between BM25 and counting matches.

    Both halves of the query match something; only one of them is informative.
    A scorer without IDF ranks by how many query words appear, and the twenty
    identical claims would win on volume.
    """
    fill(store, [(f"doc{i}", "mentions", "python") for i in range(20)])
    fill(store, [("release notes", "mentions", "kubernetes")])

    ranked = search(store, query="python kubernetes", max_claims=5)
    assert subjects(ranked)[0] == "release notes"
    assert ranked[0].score > ranked[1].score * 5


def test_a_subject_match_outranks_the_same_word_in_an_object(store):
    """Field weighting: a claim *about* Redis beats a claim that mentions it."""
    fill(store, [("Redis", "is", "a cache"), ("the app", "uses", "Redis")])
    ranked = search(store, query="Redis")
    assert subjects(ranked) == ["Redis", "the app"]
    assert ranked[0].score > ranked[1].score


def test_a_query_that_matches_nothing_returns_nothing(store):
    fill(store, [("Redis", "is", "a cache")])
    assert search(store, query="quantum chromodynamics") == []


def test_search_with_neither_a_query_nor_an_entity_returns_nothing(store):
    """Not "the whole store, recently first" — that is a random slice of memory
    handed to a node under the name context."""
    fill(store, [("Redis", "is", "a cache"), ("Postgres", "is", "a database")])
    assert search(store) == []
    assert retrieve(store, entities=[]) == []
    assert render_context(store, entities=[]) == "No prior knowledge about these entities."


def test_superseded_claims_stay_out_of_search_results(store):
    old = fill(store, [("GraphARC", "memory backend", "a vector store")])[0]
    store.supersede(
        old.id,
        Claim(subject="GraphARC", predicate="memory backend", object="SQLite", source="s"),
    )
    assert objects(search(store, query="memory backend")) == ["SQLite"]
    # ...unless they are asked for, which is what the dead-end section needs.
    both = search(store, query="memory backend", include_superseded=True)
    assert sorted(objects(both)) == ["SQLite", "a vector store"]


# --------------------------------------------------------------------------
# Graph traversal — the "graph" the name promises
# --------------------------------------------------------------------------


def test_a_question_about_one_entity_surfaces_a_fact_about_what_it_points_at(store):
    """A -> B, and a fact about B. The fact about B mentions neither A nor the
    query, so it is reachable only by following the edge."""
    bridge, downstream = fill(
        store,
        [
            ("GraphARC", "orchestration runtime", "LangGraph"),
            ("LangGraph", "requires", "Python 3.10+"),
        ],
    )

    without = search(store, entities=["GraphARC"], hops=0)
    assert objects(without) == ["LangGraph"]

    with_hop = search(store, entities=["GraphARC"], hops=1)
    assert objects(with_hop) == ["LangGraph", "Python 3.10+"]
    reached = with_hop[1]
    assert reached.claim.id == downstream.id
    assert (reached.hops, reached.via, reached.channel) == (1, bridge.id, "graph")


def test_an_incoming_edge_is_followed_too(store):
    """"Who points at this?" is as much a graph question as "what does this
    point at?" — a query about SQLite should reach the claim that uses it."""
    fill(
        store,
        [
            ("GraphARC", "memory backend", "SQLite"),
            ("SQLite", "is", "an embedded database"),
        ],
    )
    hits = search(store, query="embedded database", hops=1)
    assert objects(hits) == ["an embedded database", "SQLite"]
    assert hits[1].hops == 1


def test_a_traversed_fact_never_outranks_a_direct_match(store):
    """Decay is what keeps a neighbour as evidence rather than as the answer."""
    fill(
        store,
        [
            ("GraphARC", "orchestration runtime", "LangGraph"),
            ("LangGraph", "requires", "Python 3.10+"),
        ],
    )
    hits = search(store, entities=["GraphARC"], hops=1)
    assert hits[0].hops == 0
    assert hits[0].score > hits[1].score


def test_a_second_hop_happens_only_when_asked(store):
    fill(
        store,
        [
            ("GraphARC", "orchestration runtime", "LangGraph"),
            ("LangGraph", "requires", "Python 3.10+"),
            ("Python 3.10+", "released", "2021"),
        ],
    )
    assert objects(search(store, entities=["GraphARC"], hops=1)) == [
        "LangGraph",
        "Python 3.10+",
    ]
    two = search(store, entities=["GraphARC"], hops=2)
    assert objects(two) == ["LangGraph", "Python 3.10+", "2021"]
    assert [hit.hops for hit in two] == [0, 1, 2]


def test_traversal_does_not_cross_a_superseded_claim(store):
    """A retracted fact's edges are retracted with it, or memory would keep
    walking a path the store no longer believes in."""
    bridge = fill(store, [("GraphARC", "orchestration runtime", "LangGraph")])[0]
    fill(store, [("LangGraph", "requires", "Python 3.10+")])
    store.supersede(
        bridge.id,
        Claim(
            subject="GraphARC",
            predicate="orchestration runtime",
            object="a bespoke kernel",
            source="s",
        ),
    )
    hits = search(store, entities=["GraphARC"], hops=2)
    assert objects(hits) == ["a bespoke kernel"]


def test_render_context_says_which_edge_a_traversed_fact_came_from(store):
    fill(
        store,
        [
            ("GraphARC", "orchestration runtime", "LangGraph"),
            ("LangGraph", "requires", "Python 3.10+"),
        ],
    )
    brief = render_context(store, entities=["GraphARC"], hops=1)
    lines = brief.splitlines()
    assert lines[1].endswith(f"observed: {store.current('GraphARC')[0].observed_at}]")
    assert lines[2].endswith("(related via GraphARC)")
    # A direct fact is never annotated — the marker means something.
    assert "related via" not in lines[1]


# --------------------------------------------------------------------------
# The vector channel: pluggable, off by default, honest about what it is
# --------------------------------------------------------------------------


def test_nothing_is_embedded_unless_an_embedder_is_supplied(store):
    fill(store, [("Redis", "is", "a cache")])
    assert ClaimIndex.from_store(store).vectors == {}

    counter = CountingEmbedder(HashingEmbedder())
    search(store, query="cache", embedder=counter)
    assert counter.texts > 0


def test_a_misspelled_query_is_found_only_through_the_vector_channel(store):
    """What the shipped fallback buys: BM25 scores an exact zero on a typo."""
    fill(
        store,
        [
            ("LangGraph", "is", "a graph orchestration library"),
            ("Postgres", "is", "a relational database"),
            ("Redis", "is", "a cache"),
        ],
    )
    assert search(store, query="langraph") == []
    found = search(store, query="langraph", embedder=HashingEmbedder())
    assert subjects(found)[0] == "LangGraph"
    assert found[0].channel == "hybrid"


def test_the_fallback_embedder_is_lexical_and_does_not_pretend_otherwise(store):
    """The docstring says it cannot relate 'car' to 'automobile'. Prove it, so
    nobody reads "embeddings" and assumes semantics that are not there."""
    fill(store, [("Tesla", "sells", "a car"), ("Bakery", "sells", "bread")])
    assert search(store, query="automobile", embedder=HashingEmbedder()) == []


def test_an_injected_embedder_supplies_the_semantics_the_fallback_cannot(store):
    fill(store, [("Tesla", "sells", "a car"), ("Bakery", "sells", "bread")])
    found = search(store, query="automobile", embedder=SynonymEmbedder())
    assert subjects(found) == ["Tesla"]


def test_the_similarity_floor_keeps_the_vector_channel_quiet_on_nonsense(store):
    """Without a floor, normalizing rescales the best of a set of meaningless
    similarities to a confident 1.0 and every query is answered."""
    fill(
        store,
        [
            ("LangGraph", "is", "a graph orchestration library"),
            ("Postgres", "is", "a relational database"),
        ],
    )
    embedder = HashingEmbedder()
    assert search(store, query="quantum chromodynamics", embedder=embedder) == []
    # The floor is a threshold, not a hard-coded refusal: lower it and the same
    # noise comes back.
    noisy = search(
        store, query="quantum chromodynamics", embedder=embedder, min_similarity=0.0
    )
    assert noisy
    assert max(hit.score for hit in noisy) < MIN_SIMILARITY


def test_a_claim_is_embedded_as_its_subject_and_as_its_whole_triple(store):
    """Max-pooling two vectors is what stops a one-word query from being
    diluted to noise by a long document."""
    claims = fill(store, [("Postgres", "is", "a relational database management system")])
    embedder = HashingEmbedder()
    query = embedder.embed(["postgress"])[0]
    subject_only = cosine(query, embedder.embed([claims[0].subject])[0])
    whole = cosine(query, embedder.embed([claim_text(claims[0])])[0])
    assert subject_only > whole

    index = ClaimIndex.from_store(store, embedder=embedder)
    assert index.similarity("postgress")[claims[0].id] == pytest.approx(subject_only)


def test_the_hashing_embedder_is_stable_across_processes(tmp_path):
    """`hash()` is randomized per process, so a `hash()`-based vector would rank
    one way today and another way after a restart — fatal for a store whose
    point is outliving the process."""
    here = HashingEmbedder(dim=32).embed(["GraphARC memory backend"])[0]
    script = tmp_path / "embed.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            from grapharc.memory import HashingEmbedder
            print(json.dumps(list(HashingEmbedder(dim=32).embed(["GraphARC memory backend"])[0])))
            """
        ),
        encoding="utf-8",
    )
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, env=env, check=True
        )
        assert json.loads(out.stdout) == pytest.approx(list(here))


def test_an_embedder_returning_the_wrong_number_of_vectors_is_refused(store):
    class Broken:
        def embed(self, texts):
            return [[1.0, 0.0]]

    fill(store, [("a", "b", "c"), ("d", "e", "f")])
    with pytest.raises(ValueError, match="vectors for"):
        ClaimIndex.from_store(store, embedder=Broken())


# --------------------------------------------------------------------------
# Token-budgeted rendering
# --------------------------------------------------------------------------


def test_the_brief_format_is_unchanged_when_no_budget_is_set(store):
    """Pins the rendering other prompts are built against, provenance included."""
    old = fill(store, [("X", "uses", "wrong-db")], source="old.md")[0]
    new = store.supersede(
        old.id, Claim(subject="X", predicate="uses", object="right-db", source="new.md")
    )
    assert render_context(store, entities=["X"]) == (
        "Known facts (with provenance):\n"
        f"- X uses right-db [source: new.md, observed: {new.observed_at}]\n"
        "\n"
        "Superseded — do not re-derive these:\n"
        f"- X uses wrong-db (superseded by {new.id})"
    )


def test_a_token_budget_drops_whole_lines_and_says_how_many(store):
    fill(store, [("X", f"property number {i}", f"value {i}") for i in range(30)])
    full = render_context(store, entities=["X"], max_claims=30)
    full_cost = sum(_estimate_tokens(line) for line in full.splitlines())
    assert full_cost > 200

    budgeted = render_context(store, entities=["X"], max_claims=30, max_tokens=200)
    lines = budgeted.splitlines()
    assert sum(_estimate_tokens(line) for line in lines) <= 200
    assert len(lines) < len(full.splitlines())
    # Every surviving line is a whole line, never a truncated one.
    assert all(line in full.splitlines() for line in lines[:-1])
    dropped = len(full.splitlines()) - (len(lines) - 1)
    assert lines[-1] == f"(+{dropped} lines omitted to fit a 200-token budget)"


def test_the_supplied_token_counter_is_the_one_that_decides(store):
    """A counter that calls every line enormous must empty the brief — otherwise
    `count_tokens` is decoration and the budget is really the estimator's."""
    fill(store, [("X", "p", "v")])
    assert render_context(store, entities=["X"], max_tokens=500) != ""
    starved = render_context(
        store, entities=["X"], max_tokens=500, count_tokens=lambda line: 1000
    )
    assert starved == "(+1 lines omitted to fit a 500-token budget)"


def test_both_caps_still_apply_underneath_the_token_budget(store):
    for i in range(40):
        old = fill(store, [("X", f"p{i}", f"wrong-{i}")])[0]
        store.supersede(
            old.id, Claim(subject="X", predicate=f"p{i}", object=f"right-{i}", source="s")
        )
    brief = render_context(store, entities=["X"], max_claims=5, max_dead_ends=3)
    facts, superseded = brief.split("Superseded — do not re-derive these:")
    assert len([ln for ln in facts.splitlines() if ln.startswith("- ")]) == 5
    assert "(+37 older corrections omitted)" in superseded


def test_a_query_only_brief_still_warns_about_the_dead_ends_behind_its_answers(store):
    """Corrections are looked up per subject, so a brief driven by a query and
    no entity list used to carry no warnings at all."""
    old = fill(store, [("GraphARC", "memory backend", "a vector store")])[0]
    store.supersede(
        old.id,
        Claim(subject="GraphARC", predicate="memory backend", object="SQLite", source="s"),
    )
    brief = render_context(store, query="memory backend")
    facts, superseded = brief.split("Superseded — do not re-derive these:")
    assert "SQLite" in facts and "a vector store" not in facts
    assert "a vector store" in superseded


def test_a_query_reranks_the_superseded_section_too(store):
    """The cap discards a tail, so which corrections a node is warned about has
    to answer to the question it asked."""
    for i in range(10):
        old = fill(store, [("X", f"p{i}", f"wrong-{i}")])[0]
        store.supersede(
            old.id, Claim(subject="X", predicate=f"p{i}", object=f"right-{i}", source="s")
        )
    interesting = fill(store, [("X", "storage engine", "a bespoke btree")])[0]
    store.supersede(
        interesting.id,
        Claim(subject="X", predicate="storage engine", object="SQLite", source="s"),
    )
    brief = render_context(store, entities=["X"], query="storage engine", max_dead_ends=2)
    section = brief.split("Superseded — do not re-derive these:")[1]
    assert "a bespoke btree" in section.splitlines()[1]


# --------------------------------------------------------------------------
# Contradiction detection
# --------------------------------------------------------------------------


def test_a_claim_that_disagrees_with_a_stored_one_is_detected(store):
    stored = fill(store, [("deploy", "owner", "alice")], source="wiki")[0]
    incoming = Claim(subject="Deploy", predicate="Owner", object="bob", source="pagerduty")

    found = detect_contradictions(store, incoming)
    assert len(found) == 1
    assert found[0].existing.id == stored.id
    assert found[0].incoming is incoming
    # Entity resolution applies to the key, so casing does not hide a conflict.
    assert found[0].key == ("deploy", "owner")
    assert "alice" in found[0].describe() and "wiki" in found[0].describe()


def test_detecting_a_contradiction_writes_nothing(store):
    """Detection is a read. A caller must be able to ask before it commits."""
    fill(store, [("deploy", "owner", "alice")])
    incoming = Claim(subject="deploy", predicate="owner", object="bob", source="s")
    assert detect_contradictions(store, incoming)
    assert store.get(incoming.id) is None
    assert objects(store.current("deploy")) == ["alice"]


def test_a_duplicate_is_not_a_contradiction(store):
    fill(store, [("deploy", "owner", "alice")])
    same = Claim(subject="deploy", predicate="owner", object="  Alice ", source="other")
    assert detect_contradictions(store, same) == []


def test_a_different_predicate_is_not_a_contradiction(store):
    """The limit, asserted rather than described: this is a structural test and
    "is fast" versus "is slow" is not a structure it can see."""
    fill(store, [("cache", "is", "fast")])
    other = Claim(subject="cache", predicate="seems", object="slow", source="s")
    assert detect_contradictions(store, other) == []


def test_add_and_detect_stores_the_claim_and_reports_the_conflict(store):
    fill(store, [("deploy", "owner", "alice")])
    stored, found = add_and_detect(
        store, Claim(subject="deploy", predicate="owner", object="bob", source="s")
    )
    assert store.get(stored.id) is not None
    assert len(found) == 1
    # Reported, not resolved: both are still current until someone decides.
    assert sorted(objects(store.current("deploy"))) == ["alice", "bob"]
    assert store.dead_ends("deploy") == []


def test_supersede_conflicting_resolves_it_and_keeps_the_history(store):
    first = fill(store, [("deploy", "owner", "alice")], source="wiki")[0]
    _, found = add_and_detect(
        store, Claim(subject="deploy", predicate="owner", object="bob", source="pagerduty")
    )
    winner = supersede_conflicting(store, found[0])

    assert objects(store.current("deploy", "owner")) == ["bob"]
    assert objects(store.history("deploy", "owner")) == ["alice", "bob"]
    assert store.get(first.id).superseded_by == winner.id
    # The corrected claim keeps its own provenance.
    assert store.get(first.id).source == "wiki"
    assert unresolved_contradictions(store) == []


def test_resolving_a_stale_report_is_refused(store):
    """Between detection and resolution someone else may have corrected the same
    claim. Two winners is the outcome that must not be possible."""
    fill(store, [("deploy", "owner", "alice")])
    _, found = add_and_detect(
        store, Claim(subject="deploy", predicate="owner", object="bob", source="s")
    )
    supersede_conflicting(store, found[0])
    with pytest.raises(ValueError, match="already superseded"):
        supersede_conflicting(store, found[0])


def test_a_superseded_claim_is_never_reported_as_contradicting(store):
    old = fill(store, [("deploy", "owner", "alice")])[0]
    store.supersede(
        old.id, Claim(subject="deploy", predicate="owner", object="bob", source="s")
    )
    incoming = Claim(subject="deploy", predicate="owner", object="carol", source="s")
    found = detect_contradictions(store, incoming)
    assert [c.existing.object for c in found] == ["bob"]


def test_a_multi_valued_predicate_is_reported_and_left_alone(store):
    """The known false positive, pinned. Three dependencies are not a
    disagreement — which is exactly why detection never resolves anything."""
    fill(
        store,
        [
            ("app", "depends on", "redis"),
            ("app", "depends on", "postgres"),
            ("app", "depends on", "s3"),
        ],
    )
    groups = unresolved_contradictions(store)
    assert len(groups) == 1
    assert sorted(c.object for c in groups[0].claims) == ["postgres", "redis", "s3"]
    assert len(store.current("app", "depends on")) == 3


def test_unresolved_contradictions_scans_the_whole_store_or_named_entities(store):
    fill(
        store,
        [
            ("deploy", "owner", "alice"),
            ("deploy", "owner", "bob"),
            ("build", "owner", "carol"),
        ],
    )
    everything = unresolved_contradictions(store)
    assert [(g.subject, g.predicate) for g in everything] == [("deploy", "owner")]
    assert unresolved_contradictions(store, entities=["build"]) == []
    assert len(unresolved_contradictions(store, entities=["deploy"])) == 1


def test_the_brief_flags_a_contradiction_it_is_showing(store):
    alice, bob = fill(store, [("deploy", "owner", "alice"), ("deploy", "owner", "bob")])
    brief = render_context(store, entities=["deploy"])
    section = brief.split("Contradictions — same subject and predicate, different values:")
    assert len(section) == 2
    line = section[1].strip()
    # The ids are in the line because resolving needs them.
    assert alice.id in line and bob.id in line
    assert "'alice'" in line and "'bob'" in line

    quiet = render_context(store, entities=["deploy"], flag_contradictions=False)
    assert "Contradictions" not in quiet


def test_agreeing_claims_produce_no_contradiction_section(store):
    fill(store, [("deploy", "owner", "alice"), ("deploy", "runs on", "fargate")])
    assert "Contradictions" not in render_context(store, entities=["deploy"])


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_an_artifact_round_trips_bytes_and_text_with_its_provenance(artifacts):
    written = artifacts.put(
        "report.md",
        "# findings\n",
        source="summarizer",
        run_id="run12",
        node="write_report",
        claim_ids=["c1", "c2"],
    )
    assert artifacts.read(written.id) == b"# findings\n"
    assert artifacts.read_text(written.id) == "# findings\n"

    stored = artifacts.get(written.id)
    assert (stored.source, stored.run_id, stored.node) == ("summarizer", "run12", "write_report")
    assert stored.claim_ids == ("c1", "c2")
    assert stored.media_type == "text/plain; charset=utf-8"
    assert stored.size_bytes == 11
    assert artifacts.by_run("run12") == [stored]


def test_binary_content_survives_unchanged(artifacts):
    payload = bytes(range(256))
    written = artifacts.put("core.dump", payload, source="tool")
    assert artifacts.read(written.id) == payload
    assert artifacts.get(written.id).media_type == "application/octet-stream"


def test_writing_a_name_again_versions_it_instead_of_overwriting(artifacts):
    first = artifacts.put("report.md", "draft", source="w", run_id="run1")
    second = artifacts.put("report.md", "final", source="w", run_id="run2")

    assert [a.id for a in artifacts.versions("report.md")] == [first.id, second.id]
    assert artifacts.latest("report.md").id == second.id
    # The earlier bytes are still readable — this is the artifact half of
    # "superseded, never overwritten".
    assert artifacts.read_text(first.id) == "draft"
    assert artifacts.get(first.id).run_id == "run1"


def test_provenance_is_mandatory(artifacts):
    with pytest.raises(ValueError, match="needs a source"):
        artifacts.put("report.md", "x", source="")
    with pytest.raises(ValueError, match="non-empty"):
        artifacts.put("   ", "x", source="w")
    assert artifacts.all_artifacts() == []


def test_identical_content_is_recorded_twice_and_stored_once(artifacts):
    one = artifacts.put("a.txt", "same bytes", source="w", run_id="run1")
    two = artifacts.put("b.txt", "same bytes", source="w", run_id="run2")
    assert one.id != two.id
    assert one.sha256 == two.sha256
    # Two events with two provenances, both readable.
    assert artifacts.read_text(one.id) == artifacts.read_text(two.id) == "same bytes"
    assert [a.run_id for a in artifacts.all_artifacts()] == ["run1", "run2"]


def test_a_repeated_artifact_id_is_refused_by_both_backends(artifacts, monkeypatch):
    """Append-only: a repeated id is a bug, not an update.

    Ids are minted per put, so this is only reachable with the generator pinned
    — but the two backends have to fail the same way when it happens, and one
    of them used to silently replace the row while the other raised a
    `sqlite3.IntegrityError`.
    """

    class OneId:
        @staticmethod
        def uuid4():
            return type("U", (), {"hex": "deadbeefcafe" * 3})()

    monkeypatch.setattr(artifacts_module, "uuid", OneId)
    artifacts.put("a.txt", "one", source="w")
    with pytest.raises(ValueError, match="already exists"):
        artifacts.put("b.txt", "two", source="w")
    assert len(artifacts.all_artifacts()) == 1


def test_artifact_names_are_matched_exactly(artifacts):
    """A filename is not an entity: claim subjects are case-folded, these are not."""
    artifacts.put("Report.md", "upper", source="w")
    artifacts.put("report.md", "lower", source="w")
    assert artifacts.read_text(artifacts.latest("Report.md").id) == "upper"
    assert artifacts.read_text(artifacts.latest("report.md").id) == "lower"


def test_reading_an_unknown_artifact_raises(artifacts):
    with pytest.raises(KeyError):
        artifacts.read("nope")
    assert artifacts.get("nope") is None
    assert artifacts.latest("nope") is None
    assert artifacts.versions("nope") == []


def test_an_artifact_name_is_never_used_as_a_path(tmp_path):
    """A name comes from a model. If it were joined onto the blob directory,
    `../../` would write wherever the process can write."""
    root = tmp_path / "store"
    store = SQLiteArtifactStore(root / "memory.sqlite")
    written = store.put("../../escaped.txt", b"boom", source="tool")

    blob = store.blob_path(written).resolve()
    assert blob.is_relative_to(store.blobs_dir.resolve())
    assert not (tmp_path / "escaped.txt").exists()
    assert not (root / "escaped.txt").exists()
    # The name is kept verbatim as metadata; it is simply never a path.
    assert store.get(written.id).name == "../../escaped.txt"
    assert store.read(written.id) == b"boom"
    store.close()


def test_a_row_whose_content_is_missing_fails_loudly(tmp_path):
    """Returning b"" here would be a memory store lying about what it holds."""
    store = SQLiteArtifactStore(tmp_path / "memory.sqlite")
    written = store.put("report.md", "text", source="w")
    store.blob_path(written).unlink()
    with pytest.raises(FileNotFoundError, match="has a row but no content"):
        store.read(written.id)
    store.close()


def test_artifacts_and_claims_live_in_one_file(tmp_path):
    """"Alongside claims" as a fact about storage, not a figure of speech."""
    path = tmp_path / "memory.sqlite"
    claims = SQLiteMemoryStore(path)
    files = SQLiteArtifactStore(path)
    written = files.put("report.md", "text", source="w", run_id="run12")
    claims.add(
        Claim(
            subject="report.md",
            predicate="produced by",
            object="run12",
            source=f"artifact:{written.id}",
        )
    )
    assert len(claims.all_claims()) == 1
    assert len(files.all_artifacts()) == 1
    claims.close()
    files.close()

    reopened_claims = SQLiteMemoryStore(path)
    reopened_files = SQLiteArtifactStore(path)
    assert reopened_claims.current("report.md")[0].source == f"artifact:{written.id}"
    assert reopened_files.read_text(written.id) == "text"
    reopened_claims.close()
    reopened_files.close()


def test_artifacts_survive_the_process_that_wrote_them(tmp_path):
    """The same standard claims are held to: durability across a real process
    boundary, not across two references to one object."""
    path = tmp_path / "memory.sqlite"
    script = tmp_path / "write.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            from grapharc.memory import SQLiteArtifactStore
            store = SQLiteArtifactStore(sys.argv[1])
            written = store.put("report.md", b"binary\\x00payload", source="agent",
                                run_id="run12", node="writer")
            store.close()
            print(json.dumps({"id": written.id, "pid": os.getpid()}))
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    out = subprocess.run(
        [sys.executable, str(script), str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    written = json.loads(out.stdout)
    assert written["pid"] != os.getpid()

    store = SQLiteArtifactStore(path)
    assert store.read(written["id"]) == b"binary\x00payload"
    assert store.get(written["id"]).source == "agent"
    assert store.get(written["id"]).node == "writer"
    store.close()


def test_the_in_memory_artifact_store_does_not_survive_the_process(tmp_path):
    """The control: what the SQLite backend is actually adding."""
    script = tmp_path / "volatile.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            from grapharc.memory import MemoryArtifactStore
            store = MemoryArtifactStore()
            store.put("report.md", "text", source="w")
            print(json.dumps({"seen": len(store.all_artifacts())}))
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    first = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(first.stdout)["seen"] == 1
    assert MemoryArtifactStore().all_artifacts() == []


def test_render_artifacts_is_capped_and_carries_provenance(artifacts):
    for i in range(15):
        artifacts.put(f"f{i}.txt", f"body {i}", source="tool", run_id="run7", node=f"n{i}")
    brief = render_artifacts(artifacts, max_artifacts=3)
    lines = brief.splitlines()
    assert len(lines) == 5  # header + 3 + omission note
    assert "f14.txt" in lines[1] and "source: tool" in lines[1] and "run: run7" in lines[1]
    assert lines[-1] == "- (+12 more artifacts omitted)"
    assert render_artifacts(MemoryArtifactStore()) == "No artifacts recorded."


def test_render_artifacts_shows_the_latest_version_of_each_name(artifacts):
    artifacts.put("report.md", "draft", source="w")
    newest = artifacts.put("report.md", "final", source="w")
    brief = render_artifacts(artifacts)
    assert brief.count("report.md") == 1
    assert newest.id in brief


# --------------------------------------------------------------------------
# Backend conformance — the two stores are supposed to be interchangeable
# --------------------------------------------------------------------------


def test_both_artifact_backends_satisfy_the_protocol(tmp_path):
    store = SQLiteArtifactStore(tmp_path / "memory.sqlite")
    assert isinstance(MemoryArtifactStore(), ArtifactStore)
    assert isinstance(store, ArtifactStore)
    store.close()


def test_the_two_artifact_backends_agree_on_every_interface_signature():
    """isinstance against a runtime protocol only checks names, so check the
    signatures too — a backend that renamed a parameter would still pass."""
    for name in ARTIFACT_INTERFACE:
        protocol = inspect.signature(getattr(ArtifactStore, name))
        assert inspect.signature(getattr(MemoryArtifactStore, name)) == protocol
        assert inspect.signature(getattr(SQLiteArtifactStore, name)) == protocol


def test_the_two_artifact_backends_record_identical_metadata(tmp_path):
    """Same calls, same rows. Anything backend-specific in the metadata shows up
    here rather than in whichever deployment happened to use the other store."""
    memory = MemoryArtifactStore()
    with SQLiteArtifactStore(tmp_path / "memory.sqlite") as sql:
        for target in (memory, sql):
            target.put("a.txt", "one", source="w", run_id="r1", claim_ids=["c1"])
            target.put("a.txt", b"two", source="w", run_id="r2", parents=["p1", "p2"])
            target.put("b.bin", b"\xff", source="t", run_id="r2", node="n")

        def shape(store):
            return [
                (a.name, a.media_type, a.size_bytes, a.sha256, a.source, a.run_id, a.node,
                 a.parents, a.claim_ids)
                for a in store.all_artifacts()
            ]

        assert shape(memory) == shape(sql)
        assert [a.name for a in memory.versions("a.txt")] == [
            a.name for a in sql.versions("a.txt")
        ]
        assert [a.name for a in memory.by_run("r2")] == [a.name for a in sql.by_run("r2")]
        assert memory.read(memory.latest("a.txt").id) == sql.read(sql.latest("a.txt").id)


def test_both_claim_backends_rank_and_render_a_query_identically(tmp_path):
    """The ranking is arithmetic over stored values, so two backends holding the
    same claims must produce the same order — including the float ties that
    used to be settled by whatever iteration order each backend had."""
    prepared = [
        Claim(subject=s, predicate=p, object=o, source="s")
        for s, p, o in [
            ("GraphARC", "orchestration runtime", "LangGraph"),
            ("LangGraph", "requires", "Python"),
            ("GraphARC", "memory backend", "SQLite"),
            ("SQLite", "is", "an embedded database"),
            ("Redis", "is", "an in-memory database"),
        ]
    ]
    memory = MemoryStore()
    for claim in prepared:
        memory.add(claim)

    with SQLiteMemoryStore(tmp_path / "memory.sqlite") as sql:
        # Loaded in the opposite order: anything decided by iteration order shows.
        for claim in reversed(prepared):
            sql.add(claim)

        for kwargs in (
            {"query": "embedded database"},
            {"entities": ["GraphARC"]},
            {"query": "database", "entities": ["GraphARC"]},
            {"query": "databse", "embedder": HashingEmbedder()},
        ):
            left = [(h.claim.id, round(h.score, 9), h.hops) for h in search(memory, **kwargs)]
            right = [(h.claim.id, round(h.score, 9), h.hops) for h in search(sql, **kwargs)]
            assert left == right, kwargs
            assert render_context(memory, **kwargs) == render_context(sql, **kwargs)


def test_the_claim_store_protocol_is_unchanged_by_this_work():
    """Contradiction detection deliberately added no method to `ClaimStore`: it
    reads through `current`, so a third backend needs to implement nothing new.
    """
    assert isinstance(MemoryStore(), ClaimStore)
    assert {name for name in dir(ClaimStore) if not name.startswith("_")} == {
        "add",
        "get",
        "supersede",
        "current",
        "history",
        "dead_ends",
        "all_claims",
    }
    assert not hasattr(MemoryStore, "contradictions")


def test_an_artifact_is_a_model_with_a_content_address():
    artifact = Artifact(name="a.txt", size_bytes=3, sha256="ab" * 32, source="w")
    assert artifact.blob_key == ("ab", "ab" * 32)
    assert artifact.parents == () and artifact.claim_ids == ()
