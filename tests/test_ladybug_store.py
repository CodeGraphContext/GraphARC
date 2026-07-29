"""The LadybugDB backend, held to the same contract as the other two.

Two things are proved here that the shared conformance tests cannot express.

*Agreement.* Every invariant in `ClaimStore`'s docstring is re-run against this
backend beside `SQLiteMemoryStore`, on the same inputs, asserting the two
return the same claims in the same order. A backend that stores the supersede
link as an edge rather than a column is free to disagree about ordering by
accident, which is exactly what `seq` exists to prevent.

*The limitation.* LadybugDB takes an exclusive lock on the database. That is
documented in the module, and documentation drifts, so it is also tested: a
second process is spawned while a writer holds the database and is asserted to
fail. If a later LadybugDB release relaxes this, that test fails and the
docstring gets corrected rather than quietly becoming false.
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

from grapharc.memory import Claim, LadybugMemoryStore, SQLiteMemoryStore
from grapharc.memory.index import ClaimIndex
from grapharc.memory.ladybug_store import _load_driver
from grapharc.memory.retrieval import search
from grapharc.memory.store import ClaimStore

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def driver():
    return pytest.importorskip("real_ladybug", reason="needs the ladybug extra")


@pytest.fixture
def store(driver, tmp_path):
    store = LadybugMemoryStore(tmp_path / "memory.lbdb")
    yield store
    store.close()


def _claims() -> list[Claim]:
    """A fixed corpus, so both backends are asked exactly the same question."""
    return [
        Claim(subject="GraphARC", predicate="runs on", object="LangGraph", source="README"),
        Claim(subject="GraphARC", predicate="stores claims in", object="SQLite", source="docs"),
        Claim(subject="LangGraph", predicate="is written in", object="Python", source="docs"),
        Claim(subject="東京", predicate="is", object="capital", source="atlas"),
    ]


def _shape(claims: list[Claim]) -> list[tuple]:
    """The parts of a claim two backends must agree on, in list order."""
    return [
        (c.id, c.subject, c.predicate, c.object, c.source, c.confidence, c.superseded_by)
        for c in claims
    ]


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_it_satisfies_the_claim_store_protocol(store):
    assert isinstance(store, ClaimStore)


def test_it_matches_the_protocol_signatures_without_the_driver_installed():
    """The class is importable and checkable with no LadybugDB present, which
    is why the driver import lives in the constructor."""
    for name in ("add", "get", "supersede", "current", "history", "dead_ends", "all_claims"):
        assert inspect.signature(getattr(LadybugMemoryStore, name)) == inspect.signature(
            getattr(ClaimStore, name)
        )


def test_a_claim_survives_a_round_trip_with_its_provenance(store):
    claim = Claim(
        subject="GraphARC",
        predicate="stores claims in",
        object="LadybugDB",
        source="run-37",
        run_id="r37",
        confidence=0.75,
    )
    store.add(claim)
    read = store.get(claim.id)
    assert read is not None
    assert (read.subject, read.predicate, read.object) == (
        "GraphARC",
        "stores claims in",
        "LadybugDB",
    )
    assert (read.source, read.run_id, read.confidence) == ("run-37", "r37", 0.75)
    assert read.observed_at == claim.observed_at
    assert read.is_current


def test_get_returns_none_for_an_unknown_claim(store):
    assert store.get("nosuchclaim") is None


def test_add_is_an_upsert_that_keeps_insertion_order(store):
    first = store.add(Claim(subject="A", predicate="p", object="1", source="s"))
    store.add(Claim(subject="B", predicate="p", object="2", source="s"))
    # Re-adding the same id must update in place, not append.
    store.add(first.model_copy(update={"object": "rewritten"}))
    assert [c.object for c in store.all_claims()] == ["rewritten", "2"]


def test_upsert_moves_the_entity_edge_instead_of_leaving_a_stale_one(store):
    claim = store.add(Claim(subject="OldSubject", predicate="p", object="o", source="s"))
    store.add(claim.model_copy(update={"subject": "NewSubject"}))
    assert [c.subject for c in store.current("NewSubject")] == ["NewSubject"]
    assert store.current("OldSubject") == []
    edges = store.cypher("MATCH (:Claim)-[:ABOUT]->(e:Entity) RETURN e.display")
    assert edges == [["NewSubject"]]


def test_supersede_marks_the_old_claim_and_keeps_its_provenance(store):
    old = store.add(
        Claim(subject="GraphARC", predicate="stores claims in", object="SQLite", source="docs")
    )
    new = Claim(
        subject="GraphARC", predicate="stores claims in", object="LadybugDB", source="run-37"
    )
    store.supersede(old.id, new)

    assert [c.object for c in store.current("GraphARC", "stores claims in")] == ["LadybugDB"]
    dead = store.dead_ends("GraphARC")
    assert [c.object for c in dead] == ["SQLite"]
    # The old claim keeps its own provenance — it is marked, not rewritten.
    assert dead[0].source == "docs"
    assert dead[0].observed_at == old.observed_at
    assert dead[0].superseded_by == new.id
    assert dead[0].superseded_at is not None
    assert not dead[0].is_current


def test_history_returns_every_claim_oldest_first(store):
    old = store.add(Claim(subject="S", predicate="p", object="v1", source="a"))
    mid = Claim(subject="S", predicate="p", object="v2", source="b")
    store.supersede(old.id, mid)
    store.supersede(mid.id, Claim(subject="S", predicate="p", object="v3", source="c"))
    assert [c.object for c in store.history("S", "p")] == ["v1", "v2", "v3"]


@pytest.mark.parametrize(
    "make_bad, expected",
    [
        pytest.param(
            lambda store, old: store.supersede("unknown-id", Claim(
                subject="S", predicate="p", object="x", source="s")),
            KeyError,
            id="unknown-claim",
        ),
        pytest.param(
            lambda store, old: store.supersede(old.id, old),
            ValueError,
            id="self-supersede",
        ),
    ],
)
def test_it_refuses_what_the_shared_validator_refuses(store, make_bad, expected):
    old = store.add(Claim(subject="S", predicate="p", object="v1", source="a"))
    with pytest.raises(expected):
        make_bad(store, old)


def test_double_supersede_is_refused(store):
    old = store.add(Claim(subject="S", predicate="p", object="v1", source="a"))
    store.supersede(old.id, Claim(subject="S", predicate="p", object="v2", source="b"))
    with pytest.raises(ValueError):
        store.supersede(old.id, Claim(subject="S", predicate="p", object="v3", source="c"))


def test_a_refused_supersede_leaves_the_graph_untouched(store):
    """The rollback path: a failed precondition must not leave a half-written
    correction behind."""
    old = store.add(Claim(subject="S", predicate="p", object="v1", source="a"))
    store.supersede(old.id, Claim(subject="S", predicate="p", object="v2", source="b"))
    before = _shape(store.all_claims())
    with pytest.raises(ValueError):
        store.supersede(old.id, Claim(subject="S", predicate="p", object="v3", source="c"))
    assert _shape(store.all_claims()) == before


def test_entity_lookup_is_unicode_aware_not_ascii_casefolded(store):
    store.add(Claim(subject="東京", predicate="is", object="capital", source="atlas"))
    store.add(Claim(subject="北京", predicate="is", object="capital", source="atlas"))
    # An ASCII-only normalizer collapses both to the empty key and merges them.
    assert [c.subject for c in store.current("東京")] == ["東京"]
    assert [c.subject for c in store.current("北京")] == ["北京"]
    store.add(Claim(subject="Ladybug DB", predicate="speaks", object="Cypher", source="docs"))
    assert [c.object for c in store.current("ladybug-db")] == ["Cypher"]


# --------------------------------------------------------------------------
# Agreement with the SQLite backend
# --------------------------------------------------------------------------


def test_it_returns_exactly_what_the_sqlite_backend_returns(store, tmp_path):
    """Same claims, same corrections, same order out of both backends."""
    sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    try:
        for claim in _claims():
            store.add(claim)
            sqlite.add(claim)
        old = store.current("GraphARC", "stores claims in")[0]
        correction = Claim(
            subject="GraphARC",
            predicate="stores claims in",
            object="LadybugDB",
            source="run-37",
        )
        store.supersede(old.id, correction)
        sqlite.supersede(old.id, correction)

        assert _shape(store.all_claims()) == _shape(sqlite.all_claims())
        assert _shape(store.current("GraphARC")) == _shape(sqlite.current("GraphARC"))
        assert _shape(store.current("GraphARC", "runs on")) == _shape(
            sqlite.current("GraphARC", "runs on")
        )
        assert _shape(store.dead_ends("GraphARC")) == _shape(sqlite.dead_ends("GraphARC"))
        assert _shape(store.history("GraphARC", "stores claims in")) == _shape(
            sqlite.history("GraphARC", "stores claims in")
        )
    finally:
        sqlite.close()


def test_retrieval_ranks_it_the_same_way_as_the_sqlite_backend(store, tmp_path):
    """The store is a drop-in for the search stack, not only for the protocol."""
    sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    try:
        for claim in _claims():
            store.add(claim)
            sqlite.add(claim)
        for query in ("LangGraph", "stores claims", "capital"):
            got = search(store, query=query)
            want = search(sqlite, query=query)
            assert [s.claim.id for s in got] == [s.claim.id for s in want], query
        assert len(ClaimIndex.from_store(store)) == len(ClaimIndex.from_store(sqlite))
    finally:
        sqlite.close()


# --------------------------------------------------------------------------
# What this backend is for, and what it costs
# --------------------------------------------------------------------------


def test_cypher_walks_the_correction_chain_as_a_path(store):
    """The reason to pick this backend: provenance as a traversal."""
    old = store.add(
        Claim(subject="GraphARC", predicate="stores claims in", object="SQLite", source="docs")
    )
    store.supersede(
        old.id,
        Claim(
            subject="GraphARC", predicate="stores claims in", object="LadybugDB", source="run-37"
        ),
    )
    assert store.cypher(
        "MATCH (old:Claim)-[:SUPERSEDED_BY]->(new:Claim) "
        "RETURN old.object, new.object, old.source"
    ) == [["SQLite", "LadybugDB", "docs"]]


def test_cypher_reaches_facts_about_b_from_a_question_about_a(store):
    """The object of a claim is a node, so the graph hop happens in the
    database rather than in `ClaimIndex`."""
    for claim in _claims():
        store.add(claim)
    assert store.cypher(
        "MATCH (:Claim {subject: $s})-[:MENTIONS]->(e:Entity)<-[:ABOUT]-(next:Claim) "
        "RETURN next.subject, next.predicate, next.object",
        {"s": "GraphARC"},
    ) == [["LangGraph", "is written in", "Python"]]


def test_cypher_parameters_are_bound_not_interpolated(store):
    """A claim's text is arbitrary, so it must not be able to close a quote."""
    hostile = 'x"} ) DETACH DELETE (c) //'
    store.add(Claim(subject=hostile, predicate="p", object="o", source="s"))
    assert [c.subject for c in store.current(hostile)] == [hostile]
    assert store.cypher(
        "MATCH (c:Claim) WHERE c.subject = $s RETURN count(c)", {"s": hostile}
    ) == [[1]]
    assert len(store.all_claims()) == 1


def test_claims_written_by_one_process_are_read_by_another(driver, tmp_path):
    """Sequential hand-off — the durability claim this backend does satisfy."""
    db = tmp_path / "memory.lbdb"
    written = json.loads(
        _run_child(
            tmp_path,
            "writer",
            """
            store = LadybugMemoryStore(DB)
            claim = store.add(Claim(subject="GraphARC", predicate="learned",
                                    object="a fact", source="run-12"))
            store.close()
            emit(id=claim.id)
            """,
            db,
        )
    )
    read = json.loads(
        _run_child(
            tmp_path,
            "reader",
            """
            store = LadybugMemoryStore(DB)
            found = store.get(ARGS[0])
            store.close()
            emit(object=found.object, source=found.source)
            """,
            db,
            written["id"],
        )
    )
    assert read["object"] == "a fact"
    assert read["source"] == "run-12"
    assert read["pid"] != written["pid"]


def test_a_second_process_cannot_open_the_database_while_a_writer_holds_it(driver, tmp_path):
    """The documented limitation, pinned.

    `SQLiteMemoryStore` allows this; LadybugDB does not, and the module
    docstring says so. If a future release relaxes the lock, this test fails
    and the docstring is corrected rather than silently becoming wrong.
    """
    db = tmp_path / "memory.lbdb"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from grapharc.memory import LadybugMemoryStore
            store = LadybugMemoryStore({str(db)!r})
            print("held", flush=True)
            time.sleep(30)
        """)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        for read_only in (False, True):
            with pytest.raises(Exception) as excinfo:
                LadybugMemoryStore(db, read_only=read_only)
            assert "lock" in str(excinfo.value).lower()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_the_error_names_the_right_distribution_when_the_driver_is_absent(monkeypatch):
    """`pip install ladybug` fetches an unrelated building-science package, so
    the message has to say `real-ladybug` or it sends people to the wrong one."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in ("real_ladybug", "ladybug"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        _load_driver()
    message = str(excinfo.value)
    assert "real-ladybug" in message
    assert "grapharc[ladybug]" in message


def test_a_lookalike_package_named_ladybug_is_not_mistaken_for_the_driver(monkeypatch):
    """Ladybug Tools owns `ladybug` on PyPI. Importing it must not be taken for
    a graph database that happens to be missing some methods."""
    import builtins
    import types

    real_import = builtins.__import__
    impostor = types.ModuleType("ladybug")  # no Database, no Connection

    def fake_import(name, *args, **kwargs):
        if name == "real_ladybug":
            raise ImportError("no module named real_ladybug")
        if name == "ladybug":
            return impostor
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError):
        _load_driver()


# --------------------------------------------------------------------------


_PRELUDE = """\
import json, os, sys
sys.path.insert(0, {root!r})
from grapharc.memory import Claim, LadybugMemoryStore

DB = sys.argv[1]
ARGS = sys.argv[2:]


def emit(**payload):
    payload["pid"] = os.getpid()
    sys.stdout.write("RESULT " + json.dumps(payload) + "\\n")

"""


def _run_child(tmp_path: Path, name: str, body: str, db: Path, *args: str) -> str:
    script = tmp_path / f"{name}.py"
    script.write_text(
        _PRELUDE.format(root=str(REPO_ROOT)) + textwrap.dedent(body), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(db), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        raise AssertionError(f"child failed ({proc.returncode}):\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return line[len("RESULT ") :]
    raise AssertionError(f"child emitted no result:\nstdout={proc.stdout}\n{proc.stderr}")
