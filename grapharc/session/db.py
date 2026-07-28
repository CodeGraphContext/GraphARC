"""One way to open a SQLite file, used by the session store and the checkpointer.

Both files back the same promise — a session survives the process that started
it — so both are opened with the same durability settings. The retry loops
exist because two processes racing to open a database that does not exist yet
both try to take a brief exclusive lock, and SQLite does not run the busy
handler for those statements.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path


def _retry_locked[T](action: Callable[[], T], timeout_s: float) -> T:
    """Run `action`, retrying `database is locked` until `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return action()
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def open_database(
    path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000
) -> sqlite3.Connection:
    """Open `path` in autocommit mode, WAL if it can be had, `synchronous=FULL`.

    Autocommit (`isolation_level=None`) so transactions are opened explicitly:
    a compare-and-set on session status has to hold its read and its write
    under one `BEGIN IMMEDIATE`.

    WAL is best-effort — every other journal mode is still ACID and still locks
    across processes, just with less reader/writer overlap — but `synchronous`
    is not: under WAL, `NORMAL` can lose the most recent commits when the host
    dies, and "the most recent commit" is exactly the session state a restart
    is supposed to find.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(file, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    timeout_s = busy_timeout_ms / 1000
    _retry_locked(lambda: conn.execute("PRAGMA journal_mode = WAL").fetchone(), timeout_s)
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def create_schema(
    conn: sqlite3.Connection, schema: str, *, busy_timeout_ms: int = 5_000
) -> None:
    """`executescript` commits first, dropping out from under the busy handler."""
    _retry_locked(lambda: conn.executescript(schema), busy_timeout_ms / 1000)


__all__ = ["create_schema", "open_database"]
