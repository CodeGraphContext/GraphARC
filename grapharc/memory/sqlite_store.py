"""SQLite backend: the same claim graph, on disk, readable by the next process.

The in-process `MemoryStore` makes "run #51 sees what run #12 learned" true
only for runs that share a Python object. This backend makes it true for runs
that share nothing but a file path — which is the claim the docs actually make.

Storage is deliberately boring: one row per claim, provenance in columns, no
ORM. The only non-obvious part is entity resolution — see `_norm_pair`.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from grapharc.memory.store import Claim, _normalize, _now, validate_supersede

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id             TEXT PRIMARY KEY,
    subject        TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    object         TEXT NOT NULL,
    source         TEXT NOT NULL,
    observed_at    TEXT NOT NULL,
    run_id         TEXT,
    confidence     REAL NOT NULL,
    superseded_by  TEXT,
    superseded_at  TEXT,
    subject_norm   TEXT NOT NULL,
    predicate_norm TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS claims_subject ON claims (subject_norm);
CREATE INDEX IF NOT EXISTS claims_key ON claims (subject_norm, predicate_norm);
"""

_COLUMNS = (
    "id",
    "subject",
    "predicate",
    "object",
    "source",
    "observed_at",
    "run_id",
    "confidence",
    "superseded_by",
    "superseded_at",
)

_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM claims"

_UPSERT = """
INSERT INTO claims (
    id, subject, predicate, object, source, observed_at, run_id, confidence,
    superseded_by, superseded_at, subject_norm, predicate_norm
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    subject=excluded.subject, predicate=excluded.predicate,
    object=excluded.object, source=excluded.source,
    observed_at=excluded.observed_at, run_id=excluded.run_id,
    confidence=excluded.confidence, superseded_by=excluded.superseded_by,
    superseded_at=excluded.superseded_at, subject_norm=excluded.subject_norm,
    predicate_norm=excluded.predicate_norm
"""


def _norm_pair(claim: Claim) -> tuple[str, str]:
    """Normalized lookup keys, computed in Python and stored as columns.

    SQLite's `lower()` and `LIKE` fold ASCII only, so matching in SQL would
    make "Neo4j"/"NEO4J" work while quietly breaking every entity outside
    Latin-1. The normalized form is written at insert time instead, and every
    query compares against those columns.
    """
    return _normalize(claim.subject), _normalize(claim.predicate)


def _row_values(claim: Claim) -> tuple:
    subject_norm, predicate_norm = _norm_pair(claim)
    return (
        claim.id,
        claim.subject,
        claim.predicate,
        claim.object,
        claim.source,
        claim.observed_at,
        claim.run_id,
        claim.confidence,
        claim.superseded_by,
        claim.superseded_at,
        subject_norm,
        predicate_norm,
    )


def _to_claim(row: sqlite3.Row) -> Claim:
    return Claim(**{name: row[name] for name in _COLUMNS})


def _enable_wal(conn: sqlite3.Connection, timeout_s: float) -> str:
    """Switch to WAL, retrying by hand, and report the mode actually in force.

    Changing the journal mode needs a brief exclusive lock and SQLite does not
    run the busy handler for it, so `busy_timeout` does not cover this one
    statement: two processes opening the same fresh database at the same moment
    would otherwise crash with "database is locked" before writing anything.
    Giving up is safe — every other journal mode is still ACID and still locks
    across processes, just with less reader/writer overlap.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode == "wal" or time.monotonic() >= deadline:
                return mode
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        time.sleep(0.02)


def _create_schema(conn: sqlite3.Connection, timeout_s: float, script: str = _SCHEMA) -> None:
    """Same exclusive-lock problem as `_enable_wal`: `executescript` commits
    first, which drops out from under the busy handler mid-script.

    `script` is a parameter so the artifact store can create its own tables in
    the same file under the same retry — the lock it contends for is the
    database's, not any one table's.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn.executescript(script)
            return
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


@contextmanager
def _immediate_transaction(
    conn: sqlite3.Connection, lock: AbstractContextManager
) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE, not the default deferred transaction.

    A deferred transaction takes the write lock at the first write, so two
    processes can both read a claim as un-superseded and then both write —
    double-superseding it. IMMEDIATE takes the lock up front.

    Module-level so the artifact store shares this exact behaviour instead of
    growing its own near-copy.
    """
    with lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")


class SQLiteMemoryStore:
    """Durable backend implementing `ClaimStore` against a SQLite file.

    Safe to open concurrently from several processes: writers take the database
    write lock and wait out contention (`busy_timeout`) rather than failing.
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit: transactions are opened explicitly so that supersede()'s
        # read and write happen under one BEGIN IMMEDIATE (see _transaction).
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # Reentrant: supersede() holds the lock and calls get() inside it.
        self._lock = threading.RLock()
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.journal_mode = _enable_wal(self._conn, busy_timeout_ms / 1000)
        # The point of this backend is surviving a restart, so pay for it:
        # under WAL, synchronous=NORMAL can lose the most recent commits when
        # the host dies, which is exactly the guarantee being sold here.
        self._conn.execute("PRAGMA synchronous = FULL")
        _create_schema(self._conn, busy_timeout_ms / 1000)

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with _immediate_transaction(self._conn, self._lock) as conn:
            yield conn

    def _query(self, sql: str, params: tuple = ()) -> list[Claim]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_claim(row) for row in rows]

    def add(self, claim: Claim) -> Claim:
        with self._transaction() as conn:
            conn.execute(_UPSERT, _row_values(claim))
        return claim

    def get(self, claim_id: str) -> Claim | None:
        found = self._query(f"{_SELECT} WHERE id = ?", (claim_id,))
        return found[0] if found else None

    def supersede(self, old_id: str, new_claim: Claim) -> Claim:
        """Record a correction. The old claim stays — marked, not deleted.

        Preconditions come from the shared helper rather than being restated
        here: duplicated per-backend checks are how the two stores drifted into
        disagreeing about self-supersede in the first place.
        """
        with self._transaction() as conn:
            validate_supersede(self.get(old_id), old_id, new_claim)
            conn.execute(_UPSERT, _row_values(new_claim))
            conn.execute(
                "UPDATE claims SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                (new_claim.id, _now(), old_id),
            )
        return new_claim

    def current(self, subject: str, predicate: str | None = None) -> list[Claim]:
        """Live claims about a subject — superseded ones are excluded."""
        sql = f"{_SELECT} WHERE superseded_by IS NULL AND subject_norm = ?"
        params: tuple = (_normalize(subject),)
        if predicate is not None:
            sql += " AND predicate_norm = ?"
            params += (_normalize(predicate),)
        return self._query(f"{sql} ORDER BY rowid", params)

    def history(self, subject: str, predicate: str) -> list[Claim]:
        """Every claim ever made about (subject, predicate), oldest first."""
        return self._query(
            f"{_SELECT} WHERE subject_norm = ? AND predicate_norm = ? "
            "ORDER BY observed_at, rowid",
            (_normalize(subject), _normalize(predicate)),
        )

    def dead_ends(self, subject: str) -> list[Claim]:
        """Superseded claims — what an earlier run believed and was corrected on."""
        return self._query(
            f"{_SELECT} WHERE superseded_by IS NOT NULL AND subject_norm = ? ORDER BY rowid",
            (_normalize(subject),),
        )

    def all_claims(self) -> list[Claim]:
        return self._query(f"{_SELECT} ORDER BY rowid")
