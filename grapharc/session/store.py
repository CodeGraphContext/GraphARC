"""Session records on disk: the half of "resumable" that checkpoints do not cover.

The kernel's checkpointer already remembers *graph state* per thread. It does
not remember that a thread is a session, which graph built it, whether anyone
is driving it, what it is waiting for, or what was said to it while it was
asleep. Without that a new process holding a session id has a name and nothing
to do with it.

Three tables:

- `sessions` — one row per session, the authoritative status.
- `session_events` — the durable input queue (see `events.py`).
- `session_transitions` — every status change, with the reason and the pid that
  made it. This is the session-level audit trail: it is how you answer *what
  did it do* and *why did it stop* after the fact, and it is what makes a
  cross-process resume visible rather than merely asserted.

Status changes go through `transition()` and nowhere else. It validates the
edge against the lifecycle and can compare-and-set on the status it expected,
so "claim this session" is one atomic statement rather than a read followed by
a hopeful write.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from grapharc.session.approval import ApprovalRequest
from grapharc.session.db import create_schema, open_database
from grapharc.session.errors import (
    InvalidTransition,
    SessionBusy,
    SessionExistsError,
    SessionTerminated,
    ThreadInUseError,
    UnknownSessionError,
)
from grapharc.session.events import EventKind, SessionEvent


class SessionStatus(StrEnum):
    """Where a session is in its life."""

    CREATED = "created"
    """Recorded, never run."""

    RUNNING = "running"
    """A runner holds it. `runner_pid` names the process that claimed it."""

    IDLE = "idle"
    """A turn finished with nothing left pending. Ready for the next turn."""

    INTERRUPTED = "interrupted"
    """Stopped at a node boundary because someone asked it to stop.

    Usually there is graph work still pending — that is the case a resume
    picks up. It is *not* guaranteed, and callers must not read it that way:
    a turn that honours a queued interrupt before its first node, and a turn
    stopped on the boundary at which the graph happened to finish, both land
    here with nothing pending at all. `Session.pending_nodes()` is the
    authority on whether anything is left to run.
    """

    AWAITING_APPROVAL = "awaiting_approval"
    """Holding one or more nodes that have not run, waiting on decisions.

    Every gated node in the held superstep is recorded, and each needs its own
    decision: see `SessionRecord.pending_approvals`.
    """

    FAILED = "failed"
    """The last turn raised. `last_error` holds it; the session can be re-run."""

    TERMINATED = "terminated"
    """Finished for good."""


#: The lifecycle, as an adjacency table. `transition()` rejects anything not in it.
_ALLOWED: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.CREATED: frozenset({SessionStatus.RUNNING, SessionStatus.TERMINATED}),
    SessionStatus.RUNNING: frozenset(
        {
            SessionStatus.IDLE,
            SessionStatus.INTERRUPTED,
            SessionStatus.AWAITING_APPROVAL,
            SessionStatus.FAILED,
            SessionStatus.TERMINATED,
        }
    ),
    SessionStatus.IDLE: frozenset({SessionStatus.RUNNING, SessionStatus.TERMINATED}),
    SessionStatus.INTERRUPTED: frozenset({SessionStatus.RUNNING, SessionStatus.TERMINATED}),
    SessionStatus.AWAITING_APPROVAL: frozenset(
        {SessionStatus.RUNNING, SessionStatus.TERMINATED}
    ),
    SessionStatus.FAILED: frozenset({SessionStatus.RUNNING, SessionStatus.TERMINATED}),
    SessionStatus.TERMINATED: frozenset(),
}

#: Statuses a runner may claim a session from.
RESUMABLE: frozenset[SessionStatus] = frozenset(
    status for status, targets in _ALLOWED.items() if SessionStatus.RUNNING in targets
)


class _Keep:
    """Sentinel: leave this column as it is (distinct from setting it to NULL)."""


KEEP = _Keep()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    graph            TEXT NOT NULL,
    thread_id        TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    turns            INTEGER NOT NULL DEFAULT 0,
    runner_pid       INTEGER,
    -- JSON list of ApprovalRequest: every gated node the held superstep is
    -- waiting on. NULL means nothing is held. (Rows written by an older build
    -- hold a bare object; `_load_approvals` reads both.)
    pending_approval TEXT,
    last_error       TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sessions_status ON sessions (status);

CREATE TABLE IF NOT EXISTS session_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    body        TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    consumed_at TEXT
);
CREATE INDEX IF NOT EXISTS session_events_pending
    ON session_events (session_id, consumed_at, seq);

CREATE TABLE IF NOT EXISTS session_transitions (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    at          TEXT NOT NULL,
    pid         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS session_transitions_session
    ON session_transitions (session_id, seq);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class SessionRecord(BaseModel):
    """Everything a new process needs to pick a session back up, minus the graph state."""

    model_config = ConfigDict(extra="forbid")

    id: str
    graph: str
    thread_id: str
    status: SessionStatus
    created_at: str
    updated_at: str
    turns: int = 0
    runner_pid: int | None = None
    pending_approvals: tuple[ApprovalRequest, ...] = ()
    """Every hold the session is carrying, in the order the graph would run them.

    A superstep can put more than one gated node on the boundary, and each of
    them is a separate decision — approving one must never release another.
    Holding a *list* rather than a single request is what makes that
    representable at all.
    """
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def pending_approval(self) -> ApprovalRequest | None:
        """The first open hold, or None. Convenience for the common one-gate case.

        Do not use this to decide whether the session may run: a session with
        three holds and one of them settled is still held. `pending_approvals`
        is the whole answer.
        """
        return self.pending_approvals[0] if self.pending_approvals else None


class StatusChange(BaseModel):
    """One row of the session-level audit trail."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    session_id: str
    from_status: SessionStatus
    to_status: SessionStatus
    reason: str
    at: str
    pid: int


def _load_approvals(raw: str | None) -> tuple[ApprovalRequest, ...]:
    """Read the `pending_approval` column, which holds a JSON list of holds.

    A bare object is accepted too: that is how a single hold was written before
    a superstep could hold more than one node, and a session store on disk
    outlives the version that wrote it.
    """
    if raw is None:
        return ()
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]
    return tuple(ApprovalRequest.model_validate(item) for item in parsed)


def _dump_approvals(approval: ApprovalRequest | Sequence[ApprovalRequest] | None) -> str | None:
    """NULL for "nothing held", a JSON list otherwise. An empty list is nothing held."""
    if approval is None:
        return None
    holds = [approval] if isinstance(approval, ApprovalRequest) else list(approval)
    if not holds:
        return None
    return json.dumps([hold.model_dump() for hold in holds])


def _to_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        graph=row["graph"],
        thread_id=row["thread_id"],
        status=SessionStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        turns=row["turns"],
        runner_pid=row["runner_pid"],
        pending_approvals=_load_approvals(row["pending_approval"]),
        last_error=row["last_error"],
        metadata=json.loads(row["metadata"]),
    )


def _to_event(row: sqlite3.Row) -> SessionEvent:
    return SessionEvent(
        seq=row["seq"],
        session_id=row["session_id"],
        kind=EventKind(row["kind"]),
        body=row["body"],
        payload=json.loads(row["payload"]),
        created_at=row["created_at"],
        consumed_at=row["consumed_at"],
    )


class SessionStore:
    """Durable session records. Safe to open from several processes at once."""

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self._conn = open_database(self.path, busy_timeout_ms=busy_timeout_ms)
        # Reentrant: transition() reads the row it is about to write, under the
        # same lock and the same transaction.
        self._lock = threading.RLock()
        create_schema(self._conn, _SCHEMA, busy_timeout_ms=busy_timeout_ms)

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE: the write lock is taken before the read, not after it.

        A deferred transaction would let two processes both read a session as
        claimable and then both claim it.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # -- sessions ----------------------------------------------------------

    def create(
        self,
        *,
        session_id: str,
        graph: str,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Record a new session. The graph's checkpoint thread defaults to the id.

        A checkpoint thread belongs to exactly one session: a `thread_id`
        another session already holds is refused with `ThreadInUseError`,
        checked and inserted under one write lock so two creates racing on one
        thread cannot both land. Ids are unique, so the default path — thread
        id equals session id — can never collide on a fresh id.
        """
        now = _now()
        thread = thread_id or session_id
        with self._transaction() as conn:
            # Under BEGIN IMMEDIATE, so check-then-insert is one atomic claim.
            # A second session on an existing thread would resume the first
            # one's checkpointed boundary with a record carrying no holds, and
            # a gated node the first session is holding would run unapproved —
            # `interrupt_before` does not re-fire for a boundary it has
            # already stopped at.
            holder = conn.execute(
                "SELECT id FROM sessions WHERE thread_id = ?", (thread,)
            ).fetchone()
            # A holder with this very id is an id reuse, not a thread theft:
            # the insert below refuses it as `SessionExistsError`.
            if holder is not None and holder["id"] != session_id:
                raise ThreadInUseError(thread, holder["id"])
            try:
                conn.execute(
                    "INSERT INTO sessions (id, graph, thread_id, status, created_at, "
                    "updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        graph,
                        thread,
                        SessionStatus.CREATED.value,
                        now,
                        now,
                        json.dumps(metadata or {}),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                # Same guarantee, id column: a checkpoint thread belongs to
                # exactly one session, and reusing an id would silently attach
                # a new session to another session's checkpoint thread.
                raise SessionExistsError(
                    f"session {session_id!r} already exists in {self.path}"
                ) from exc
            conn.execute(
                "INSERT INTO session_transitions (session_id, from_status, to_status, "
                "reason, at, pid) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, "", SessionStatus.CREATED.value, "created", now, os.getpid()),
            )
        return self.require(session_id)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return None if row is None else _to_record(row)

    def require(self, session_id: str) -> SessionRecord:
        record = self.get(session_id)
        if record is None:
            raise UnknownSessionError(f"no session {session_id!r} in {self.path}")
        return record

    def list(
        self, *, status: SessionStatus | Iterable[SessionStatus] | None = None
    ) -> list[SessionRecord]:
        sql = "SELECT * FROM sessions"
        params: tuple = ()
        if status is not None:
            wanted = [status] if isinstance(status, SessionStatus) else list(status)
            sql += f" WHERE status IN ({', '.join('?' * len(wanted))})"
            params = tuple(s.value for s in wanted)
        with self._lock:
            rows = self._conn.execute(f"{sql} ORDER BY created_at, id", params).fetchall()
        return [_to_record(row) for row in rows]

    def transition(
        self,
        session_id: str,
        to: SessionStatus,
        *,
        expect: Iterable[SessionStatus] | None = None,
        reason: str = "",
        error: str | None | _Keep = KEEP,
        approval: ApprovalRequest | Sequence[ApprovalRequest] | None | _Keep = KEEP,
        bump_turn: bool = False,
    ) -> SessionRecord:
        """Move a session's status, atomically, and record why.

        `approval` replaces the whole set of holds: pass every hold the session
        still carries, or None to clear them. There is no "add one" — a partial
        write here is a released gate.

        `expect` makes this a compare-and-set: the update only lands if the
        session is still in one of those statuses, and `SessionBusy` names what
        was found instead. That is the whole of the "one runner at a time"
        guard — it stops a second runner from *claiming* a session, and it does
        not detect a runner that died holding one. A session stuck in `running`
        after a crash has to be released deliberately, which is a legal
        `running -> idle` transition.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise UnknownSessionError(f"no session {session_id!r} in {self.path}")
            current = SessionStatus(row["status"])
            # `expect` first: when a caller's assumption about the current
            # status is wrong, "someone else moved it" is the useful answer.
            # An illegal edge from a status the caller *did* expect is a bug in
            # the caller, and gets the other exception.
            if expect is not None:
                allowed = tuple(expect)
                if current not in allowed:
                    raise SessionBusy(
                        session_id, tuple(s.value for s in allowed), current.value
                    )
            if to not in _ALLOWED[current]:
                raise InvalidTransition(session_id, current.value, to.value)

            sets = ["status = ?", "updated_at = ?"]
            params: list[Any] = [to.value, _now()]
            # `runner_pid` means "the process driving this right now", so it is
            # stamped on the way into running and cleared on every way out.
            sets.append("runner_pid = ?")
            params.append(os.getpid() if to is SessionStatus.RUNNING else None)
            if not isinstance(error, _Keep):
                sets.append("last_error = ?")
                params.append(error)
            if not isinstance(approval, _Keep):
                sets.append("pending_approval = ?")
                params.append(_dump_approvals(approval))
            if bump_turn:
                sets.append("turns = turns + 1")
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", tuple(params))
            conn.execute(
                "INSERT INTO session_transitions (session_id, from_status, to_status, "
                "reason, at, pid) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, current.value, to.value, reason, _now(), os.getpid()),
            )
        return self.require(session_id)

    def history(self, session_id: str) -> list[StatusChange]:
        """Every status change this session has made, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_transitions WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            StatusChange(
                seq=row["seq"],
                session_id=row["session_id"],
                from_status=(
                    SessionStatus(row["from_status"])
                    if row["from_status"]
                    else SessionStatus.CREATED
                ),
                to_status=SessionStatus(row["to_status"]),
                reason=row["reason"],
                at=row["at"],
                pid=row["pid"],
            )
            for row in rows
        ]

    # -- events ------------------------------------------------------------

    def push(
        self,
        session_id: str,
        kind: EventKind,
        *,
        body: str = "",
        payload: dict[str, Any] | None = None,
    ) -> SessionEvent:
        """Queue an event. Refused for a terminated session — it will never be read."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise UnknownSessionError(f"no session {session_id!r} in {self.path}")
            if SessionStatus(row["status"]) is SessionStatus.TERMINATED:
                raise SessionTerminated(
                    f"session {session_id!r} is terminated and cannot accept a {kind.value}"
                )
            cursor = conn.execute(
                "INSERT INTO session_events (session_id, kind, body, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, kind.value, body, json.dumps(payload or {}), _now()),
            )
            seq = cursor.lastrowid
        return self.event(int(seq))

    def event(self, seq: int) -> SessionEvent:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_events WHERE seq = ?", (seq,)
            ).fetchone()
        if row is None:
            raise UnknownSessionError(f"no event {seq} in {self.path}")
        return _to_event(row)

    def pending(self, session_id: str) -> list[SessionEvent]:
        """Unconsumed events for a session, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? AND consumed_at IS NULL "
                "ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [_to_event(row) for row in rows]

    def events(self, session_id: str) -> list[SessionEvent]:
        """Every event ever queued, consumed or not — the input-side audit trail."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [_to_event(row) for row in rows]

    def consume(self, seqs: Iterable[int]) -> int:
        """Mark events consumed. Returns how many rows actually changed."""
        wanted = [int(seq) for seq in seqs]
        if not wanted:
            return 0
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE session_events SET consumed_at = ? WHERE consumed_at IS NULL "
                f"AND seq IN ({', '.join('?' * len(wanted))})",
                (_now(), *wanted),
            )
            return cursor.rowcount

    def take(self, session_id: str, kind: EventKind) -> SessionEvent | None:
        """Consume the oldest pending event of `kind`, or return None.

        One statement pair under the write lock, so two runners polling the
        same session cannot both act on one interrupt.
        """
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? AND kind = ? "
                "AND consumed_at IS NULL ORDER BY seq LIMIT 1",
                (session_id, kind.value),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE session_events SET consumed_at = ? WHERE seq = ?",
                (_now(), row["seq"]),
            )
        return self.event(row["seq"])


__all__ = [
    "KEEP",
    "RESUMABLE",
    "SessionRecord",
    "SessionStatus",
    "SessionStore",
    "StatusChange",
]
