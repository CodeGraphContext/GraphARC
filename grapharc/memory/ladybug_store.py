"""LadybugDB backend: the same claim graph, stored as an actual graph.

The other two backends keep claims as rows and rebuild the graph in Python —
`ClaimIndex` scans the corpus and builds the subject/object adjacency in memory
on every construction (see `index.py`). This backend stores the edges, so the
adjacency is the database's job and a traversal is a query rather than a scan.

LadybugDB is an embedded property-graph database with Cypher — a fork of Kuzu,
revived in 2025 after Apple acquired and closed it. Embedded means the same
deal SQLite offers: a path on disk, no server, no daemon to run. What it adds
over the SQLite backend is that `superseded_by` is an *edge* rather than a
foreign key in a column, and the subject and object of every claim are `Entity`
nodes, so "what did run #12 believe that run #37 corrected" is a path, and you
can ask it in Cypher without going through Python at all::

    store = LadybugMemoryStore("memory.lbdb")
    store.cypher(
        "MATCH (old:Claim)-[:SUPERSEDED_BY]->(new:Claim) "
        "RETURN old.object, new.object, old.source"
    )

**Where this backend is weaker than SQLite, stated plainly, because picking it
for the wrong reason will cost you a run.** LadybugDB takes an *exclusive lock
on the database directory*. One process may open it for writing, or several may
open it read-only, and those two groups cannot overlap: while a writer holds
the database, a second process opening it — even with `read_only=True` — fails
outright with an IO exception rather than waiting. `SQLiteMemoryStore` in WAL
mode allows concurrent readers alongside a writer and makes competing writers
queue on `busy_timeout`. So:

* sequential hand-off across processes works — run #12 writes and exits, run
  #37 opens the same path and reads what it left. That is the durability claim
  the memory tests make, and this backend satisfies it.
* *concurrent* multi-process access does not. If two agent processes must write
  the same memory at the same time, use `SQLiteMemoryStore`.

Within one process this store is thread-safe the same way the others are: one
connection behind an `RLock`, and `supersede`'s read-then-write happens inside
a single explicit transaction so two threads cannot both see a claim as
un-superseded and both correct it.

The driver is an optional dependency — nothing here is imported until you
construct the store, so `grapharc.memory` imports fine without it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from grapharc.memory.store import Claim, _normalize, _now, validate_supersede

# Cypher DDL. `seq` is the one column that exists for no reason a graph person
# would recognise: it is the analogue of SQLite's `rowid`, an insertion counter
# so that `current()` and `dead_ends()` can return claims in the order they
# were added. Without it the two backends would disagree on ordering for
# identical input, which is exactly the class of divergence `_now`'s docstring
# describes and the conformance tests exist to catch.
_SCHEMA = (
    """
    CREATE NODE TABLE IF NOT EXISTS Claim(
        id             STRING PRIMARY KEY,
        subject        STRING,
        predicate      STRING,
        object         STRING,
        source         STRING,
        observed_at    STRING,
        run_id         STRING,
        confidence     DOUBLE,
        superseded_at  STRING,
        subject_norm   STRING,
        predicate_norm STRING,
        seq            INT64
    )
    """,
    # An entity is stored under its normalized key, with one human-readable
    # form kept alongside for display. Normalization happens in Python for the
    # same reason it does in the SQLite backend: matching in the query language
    # would fold ASCII only and quietly merge every non-Latin entity.
    """
    CREATE NODE TABLE IF NOT EXISTS Entity(
        name    STRING PRIMARY KEY,
        display STRING
    )
    """,
    # The subject of a claim, and the object read as an entity in its own
    # right — the edge that makes a question about A reach facts about B.
    "CREATE REL TABLE IF NOT EXISTS ABOUT(FROM Claim TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Claim TO Entity)",
    # A correction, as an edge. This is the provenance chain the module exists
    # for, and here it is walkable rather than joinable.
    "CREATE REL TABLE IF NOT EXISTS SUPERSEDED_BY(FROM Claim TO Claim)",
)

# `superseded_by` is not a column — it is reconstructed from the edge, so every
# read pairs its MATCH with this OPTIONAL MATCH and this projection.
_OPTIONAL_SUPERSEDER = "OPTIONAL MATCH (c)-[:SUPERSEDED_BY]->(n:Claim)"
_PROJECTION = """
    c.id AS id, c.subject AS subject, c.predicate AS predicate, c.object AS object,
    c.source AS source, c.observed_at AS observed_at, c.run_id AS run_id,
    c.confidence AS confidence, c.superseded_at AS superseded_at,
    n.id AS superseded_by
"""

_UPSERT = """
MERGE (c:Claim {id: $id})
ON CREATE SET
    c.subject=$subject, c.predicate=$predicate, c.object=$object,
    c.source=$source, c.observed_at=$observed_at, c.run_id=$run_id,
    c.confidence=$confidence, c.superseded_at=$superseded_at,
    c.subject_norm=$subject_norm, c.predicate_norm=$predicate_norm, c.seq=$seq
ON MATCH SET
    c.subject=$subject, c.predicate=$predicate, c.object=$object,
    c.source=$source, c.observed_at=$observed_at, c.run_id=$run_id,
    c.confidence=$confidence, c.superseded_at=$superseded_at,
    c.subject_norm=$subject_norm, c.predicate_norm=$predicate_norm
"""

_DRIVER_MISSING = (
    "LadybugMemoryStore needs the LadybugDB driver, which is an optional "
    "dependency: pip install 'grapharc[ladybug]'.\n"
    "If you install it by hand the distribution is `real-ladybug` — NOT "
    "`ladybug`. `pip install ladybug` fetches the Ladybug Tools building-science "
    "package, which is an unrelated project that happens to own the name on PyPI."
)


def _load_driver() -> Any:
    """Import the LadybugDB driver, and refuse a lookalike.

    The upstream docs say `import ladybug`, but the distribution that owns
    `ladybug` on PyPI is Ladybug Tools — environmental analysis for building
    design, nothing to do with graphs. The driver ships as `real-ladybug` and
    imports as `real_ladybug`. Both names are tried, and whatever is found is
    checked for the API actually used here, so an environment with the
    building-science package installed produces this explanation rather than an
    `AttributeError` from somewhere deep in the first query.
    """
    for name in ("real_ladybug", "ladybug"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        if all(hasattr(module, attr) for attr in ("Database", "Connection")):
            return module
    raise ModuleNotFoundError(_DRIVER_MISSING)


def _params(claim: Claim, seq: int) -> dict[str, Any]:
    """Bind a claim to the upsert. Every value goes through a parameter — a
    claim's object is arbitrary text, and string-formatting it into Cypher is
    the graph-database spelling of SQL injection."""
    return {
        "id": claim.id,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "source": claim.source,
        "observed_at": claim.observed_at,
        "run_id": claim.run_id,
        "confidence": float(claim.confidence),
        "superseded_at": claim.superseded_at,
        "subject_norm": _normalize(claim.subject),
        "predicate_norm": _normalize(claim.predicate),
        "seq": seq,
    }


def _to_claim(row: dict[str, Any]) -> Claim:
    return Claim(**row)


class LadybugMemoryStore:
    """Durable backend implementing `ClaimStore` against a LadybugDB database.

    Single-writer: one process may hold the database for writing, or several
    may hold it read-only, never both at once. See the module docstring — if
    you need concurrent writers across processes, use `SQLiteMemoryStore`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        read_only: bool = False,
    ) -> None:
        driver = _load_driver()
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._db = driver.Database(str(self.path), read_only=read_only)
        self._conn = driver.Connection(self._db)
        # Reentrant: supersede() holds the lock and calls get() inside it.
        self._lock = threading.RLock()
        if not read_only:
            for statement in _SCHEMA:
                self._conn.execute(statement)
        self._seq = self._next_seq()

    def _next_seq(self) -> int:
        """Resume the insertion counter where the last process left it."""
        rows = self._conn.execute("MATCH (c:Claim) RETURN max(c.seq)").get_all()
        highest = rows[0][0] if rows else None
        return 0 if highest is None else int(highest) + 1

    def __enter__(self) -> LadybugMemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the database lock. Until this runs (or the process exits) no
        other process can open the database at all."""
        with self._lock:
            self._conn.close()
            self._db.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """An explicit transaction, so supersede()'s read and write are atomic.

        The lock is held across the whole block for the same reason the SQLite
        backend uses `BEGIN IMMEDIATE`: a read-then-write that two threads can
        interleave is how a claim gets superseded twice.
        """
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, where: str, params: dict[str, Any], order: str = "c.seq") -> list[Claim]:
        cypher = (
            f"MATCH (c:Claim) {where} {_OPTIONAL_SUPERSEDER} "
            f"RETURN {_PROJECTION} ORDER BY {order}"
        )
        with self._lock:
            result = self._conn.execute(cypher, params)
            return [_to_claim(row) for row in result.rows_as_dict()]

    def _write(self, claim: Claim) -> None:
        """Upsert the claim node and rebuild its entity edges.

        The edges are dropped and re-created rather than merged because `add`
        is an upsert: re-adding an existing id with a different subject must not
        leave the claim pointing at the entity it used to be about.
        """
        seq = self._seq
        self._seq += 1
        self._conn.execute(_UPSERT, _params(claim, seq))
        for rel in ("ABOUT", "MENTIONS"):
            self._conn.execute(
                f"MATCH (c:Claim {{id: $id}})-[r:{rel}]->(:Entity) DELETE r",
                {"id": claim.id},
            )
        for rel, text in (("ABOUT", claim.subject), ("MENTIONS", claim.object)):
            name = _normalize(text)
            if not name:
                continue
            self._conn.execute(
                "MERGE (e:Entity {name: $name}) ON CREATE SET e.display = $display",
                {"name": name, "display": text},
            )
            self._conn.execute(
                f"MATCH (c:Claim {{id: $id}}), (e:Entity {{name: $name}}) "
                f"MERGE (c)-[:{rel}]->(e)",
                {"id": claim.id, "name": name},
            )

    def add(self, claim: Claim) -> Claim:
        with self._transaction():
            self._write(claim)
        return claim

    def get(self, claim_id: str) -> Claim | None:
        found = self._query("WHERE c.id = $id", {"id": claim_id})
        return found[0] if found else None

    def supersede(self, old_id: str, new_claim: Claim) -> Claim:
        """Record a correction. The old claim stays — marked, not deleted.

        Preconditions come from the shared helper rather than being restated
        here, so this backend cannot drift from the other two on what it
        refuses.
        """
        with self._transaction():
            validate_supersede(self.get(old_id), old_id, new_claim)
            self._write(new_claim)
            self._conn.execute(
                "MATCH (o:Claim {id: $old}), (n:Claim {id: $new}) "
                "CREATE (o)-[:SUPERSEDED_BY]->(n)",
                {"old": old_id, "new": new_claim.id},
            )
            self._conn.execute(
                "MATCH (o:Claim {id: $old}) SET o.superseded_at = $at",
                {"old": old_id, "at": _now()},
            )
        return new_claim

    def current(self, subject: str, predicate: str | None = None) -> list[Claim]:
        """Live claims about a subject — superseded ones are excluded."""
        where = "WHERE c.subject_norm = $subject"
        params: dict[str, Any] = {"subject": _normalize(subject)}
        if predicate is not None:
            where += " AND c.predicate_norm = $predicate"
            params["predicate"] = _normalize(predicate)
        # The superseded test is on the edge, so it has to follow the OPTIONAL
        # MATCH rather than ride along in the WHERE above.
        cypher = (
            f"MATCH (c:Claim) {where} {_OPTIONAL_SUPERSEDER} "
            f"WITH c, n WHERE n IS NULL "
            f"RETURN {_PROJECTION} ORDER BY c.seq"
        )
        with self._lock:
            result = self._conn.execute(cypher, params)
            return [_to_claim(row) for row in result.rows_as_dict()]

    def history(self, subject: str, predicate: str) -> list[Claim]:
        """Every claim ever made about (subject, predicate), oldest first."""
        return self._query(
            "WHERE c.subject_norm = $subject AND c.predicate_norm = $predicate",
            {"subject": _normalize(subject), "predicate": _normalize(predicate)},
            order="c.observed_at, c.seq",
        )

    def dead_ends(self, subject: str) -> list[Claim]:
        """Superseded claims — what an earlier run believed and was corrected on."""
        cypher = (
            "MATCH (c:Claim)-[:SUPERSEDED_BY]->(n:Claim) "
            "WHERE c.subject_norm = $subject "
            f"RETURN {_PROJECTION} ORDER BY c.seq"
        )
        with self._lock:
            result = self._conn.execute(cypher, {"subject": _normalize(subject)})
            return [_to_claim(row) for row in result.rows_as_dict()]

    def all_claims(self) -> list[Claim]:
        return self._query("", {})

    def cypher(self, query: str, parameters: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run Cypher against the claim graph and return rows.

        The reason to choose this backend. `ClaimStore` is deliberately narrow —
        seven methods that every backend can honour — and this is the escape
        hatch for the questions it does not cover, asked against the stored
        graph rather than a corpus scan in Python.

        Use parameters rather than formatting values into the query string.
        """
        with self._lock:
            return self._conn.execute(query, parameters or {}).get_all()
