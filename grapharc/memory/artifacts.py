"""The artifact store: files an agent produced, with the provenance that explains them.

Claims answer "what is true"; artifacts answer "what was produced" — a patch, a
report, a log, a screenshot. Both halves of memory need the same discipline, so
this store follows the claim store's rules rather than inventing its own:

* **append-only.** There is no `delete` and no in-place update. Writing the same
  name again records a new *version*; `latest(name)` returns the newest and
  `versions(name)` returns the lot, oldest first. The claim store refuses to
  overwrite a fact; this refuses to overwrite a file.
* **provenance is mandatory.** `source` cannot be empty. An artifact whose
  origin is unknown is indistinguishable from one an agent hallucinated into
  the workspace, and after an incident that distinction is the whole question.
* **two interchangeable backends.** `MemoryArtifactStore` dies with the
  process; `SQLiteArtifactStore` points at the same file a `SQLiteMemoryStore`
  uses, so claims and artifacts from one run land in one place. Shared
  conformance tests hold them to identical behaviour.

**Where the bytes live.** In SQLite the metadata is a row and the content is a
file under `<db>.blobs/<first two hex>/<sha256>`, written before the row is:
crash between the two and you have an unreferenced blob (garbage), never a row
pointing at content that does not exist (a lie). Content addressing means two
artifacts with identical bytes share one file while keeping their own rows and
their own provenance — the same report produced by two runs is two events.

`name` is metadata and is never used to build a path. An artifact named
`../../etc/passwd` is stored under the hash of its content like everything else.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from grapharc.memory.sqlite_store import _create_schema, _enable_wal, _immediate_transaction
from grapharc.memory.store import _now

DEFAULT_MEDIA_TYPE = "application/octet-stream"
TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    source      TEXT NOT NULL,
    run_id      TEXT,
    node        TEXT,
    parents     TEXT NOT NULL,
    claim_ids   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_name ON artifacts (name, created_at);
CREATE INDEX IF NOT EXISTS artifacts_run  ON artifacts (run_id);
CREATE INDEX IF NOT EXISTS artifacts_sha  ON artifacts (sha256);
"""

_COLUMNS = (
    "id",
    "name",
    "media_type",
    "size_bytes",
    "sha256",
    "created_at",
    "source",
    "run_id",
    "node",
    "parents",
    "claim_ids",
)

_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM artifacts"

_INSERT = f"INSERT INTO artifacts ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})"


class Artifact(BaseModel):
    """A produced file's metadata. The bytes are fetched with `read`.

    Metadata and content are separate because an agent's context can afford a
    listing of thirty artifacts and cannot afford thirty files.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    media_type: str = DEFAULT_MEDIA_TYPE
    size_bytes: int
    sha256: str
    created_at: str = Field(default_factory=_now)
    # Provenance — the whole point, exactly as for a claim.
    source: str
    run_id: str | None = None
    node: str | None = None
    # What this was derived from: other artifacts, and claims it supports or
    # was produced from. Ids, not copies, so nothing here can drift.
    parents: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()

    @property
    def blob_key(self) -> tuple[str, str]:
        """Content-addressed location: a two-hex shard and the full digest."""
        return (self.sha256[:2], self.sha256)


def prepare(
    name: str,
    content: bytes | str,
    *,
    source: str,
    media_type: str | None = None,
    run_id: str | None = None,
    node: str | None = None,
    parents: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
) -> tuple[Artifact, bytes]:
    """Validate and build the metadata for a put, identically for every backend.

    Shared for the same reason `validate_supersede` is: two copies of a rule is
    two rules. The default media type is decided here too, so a `str` written
    to one backend is not `text/plain` in one store and octet-stream in the
    other.
    """
    if not name or not name.strip():
        raise ValueError("artifact name must be a non-empty string")
    if not source or not source.strip():
        raise ValueError(
            f"artifact {name!r} needs a source — an artifact with no provenance "
            "cannot be attributed after the fact"
        )
    if isinstance(content, str):
        payload = content.encode("utf-8")
        media_type = media_type or TEXT_MEDIA_TYPE
    else:
        payload = bytes(content)
        media_type = media_type or DEFAULT_MEDIA_TYPE
    artifact = Artifact(
        name=name,
        media_type=media_type,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        source=source,
        run_id=run_id,
        node=node,
        parents=tuple(parents),
        claim_ids=tuple(claim_ids),
    )
    return artifact, payload


@runtime_checkable
class ArtifactStore(Protocol):
    """The contract every artifact backend implements.

    Invariants the shared conformance tests enforce, since method names alone
    do not capture them:

    * append-only — `put` never replaces an existing artifact, it adds a
      version; nothing removes one;
    * `put` requires a non-empty name and a non-empty source, and raises
      `ValueError` otherwise (`prepare`, which backends call rather than
      re-derive); a repeated artifact id is also a `ValueError` rather than a
      silent replacement;
    * names are matched exactly — unlike a claim's subject, a filename is not
      an entity, and `Report.md` is not `report.md`;
    * `versions(name)` is oldest-first and ordered by `(created_at, id)`, a
      total order both backends produce identically;
    * `latest(name)` is the last element of `versions(name)`, or `None`;
    * `read` returns exactly the bytes that were written, and raises `KeyError`
      for an unknown id;
    * `name` is never interpreted as a filesystem path.

    `isinstance` against this protocol checks method *names* only — passing it
    is not evidence the invariants hold.
    """

    def put(
        self,
        name: str,
        content: bytes | str,
        *,
        source: str,
        media_type: str | None = None,
        run_id: str | None = None,
        node: str | None = None,
        parents: Sequence[str] = (),
        claim_ids: Sequence[str] = (),
    ) -> Artifact: ...

    def get(self, artifact_id: str) -> Artifact | None: ...

    def read(self, artifact_id: str) -> bytes: ...

    def read_text(self, artifact_id: str) -> str: ...

    def latest(self, name: str) -> Artifact | None: ...

    def versions(self, name: str) -> list[Artifact]: ...

    def by_run(self, run_id: str) -> list[Artifact]: ...

    def all_artifacts(self) -> list[Artifact]: ...


def _sorted_versions(artifacts: list[Artifact]) -> list[Artifact]:
    """Oldest first, by `(created_at, id)`.

    Not by insertion order: two processes writing in the same microsecond can
    tie on `created_at`, and the id then decides — the same way for a dict as
    for a rowid, which is what keeps the backends interchangeable.
    """
    return sorted(artifacts, key=lambda a: (a.created_at, a.id))


class MemoryArtifactStore:
    """In-process backend: metadata and bytes in dicts, gone when the process exits.

    The counterpart to `MemoryStore` — same trade, same warning. Content is held
    in RAM, so this is for tests and short runs, not for the 200 MB core dump a
    tool just produced.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._blobs: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(
        self,
        name: str,
        content: bytes | str,
        *,
        source: str,
        media_type: str | None = None,
        run_id: str | None = None,
        node: str | None = None,
        parents: Sequence[str] = (),
        claim_ids: Sequence[str] = (),
    ) -> Artifact:
        artifact, payload = prepare(
            name,
            content,
            source=source,
            media_type=media_type,
            run_id=run_id,
            node=node,
            parents=parents,
            claim_ids=claim_ids,
        )
        with self._lock:
            if artifact.id in self._artifacts:
                raise ValueError(f"artifact id {artifact.id!r} already exists")
            self._blobs[artifact.sha256] = payload
            self._artifacts[artifact.id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def read(self, artifact_id: str) -> bytes:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(f"unknown artifact {artifact_id!r}")
        return self._blobs[artifact.sha256]

    def read_text(self, artifact_id: str) -> str:
        return self.read(artifact_id).decode("utf-8")

    def latest(self, name: str) -> Artifact | None:
        found = self.versions(name)
        return found[-1] if found else None

    def versions(self, name: str) -> list[Artifact]:
        return _sorted_versions([a for a in self._artifacts.values() if a.name == name])

    def by_run(self, run_id: str) -> list[Artifact]:
        return [a for a in self._artifacts.values() if a.run_id == run_id]

    def all_artifacts(self) -> list[Artifact]:
        return list(self._artifacts.values())


class SQLiteArtifactStore:
    """Durable backend: rows in a SQLite file, content in a blob directory beside it.

    Point it at the same path as a `SQLiteMemoryStore` and one run's claims and
    artifacts live in one file (plus its blob directory). Two connections to one
    database is normal SQLite; both take the write lock and wait out contention.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        blobs_dir: str | os.PathLike[str] | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs_dir = Path(blobs_dir) if blobs_dir is not None else Path(f"{self.path}.blobs")
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.journal_mode = _enable_wal(self._conn, busy_timeout_ms / 1000)
        # Same reasoning as the claim store: the point of this backend is
        # surviving a restart, so do not trade that for write throughput.
        self._conn.execute("PRAGMA synchronous = FULL")
        _create_schema(self._conn, busy_timeout_ms / 1000, _SCHEMA)

    def __enter__(self) -> SQLiteArtifactStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def blob_path(self, artifact: Artifact) -> Path:
        shard, digest = artifact.blob_key
        return self.blobs_dir / shard / digest

    def _write_blob(self, artifact: Artifact, payload: bytes) -> None:
        """Durably place the content, then let the caller record the row.

        `os.replace` is atomic within a filesystem, and the fsyncs are what make
        "durable" mean survives-a-power-cut rather than survives-a-crash. The
        temp file carries a uuid so two processes storing the same content at
        the same instant cannot scribble over each other's partial write.
        """
        target = self.blob_path(artifact)
        if target.exists():
            return  # content-addressed: identical bytes are already stored
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _query(self, sql: str, params: tuple = ()) -> list[Artifact]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._to_artifact(row) for row in rows]

    @staticmethod
    def _to_artifact(row: sqlite3.Row) -> Artifact:
        values = {name: row[name] for name in _COLUMNS}
        values["parents"] = tuple(json.loads(values["parents"]))
        values["claim_ids"] = tuple(json.loads(values["claim_ids"]))
        return Artifact(**values)

    @staticmethod
    def _row_values(artifact: Artifact) -> tuple:
        return (
            artifact.id,
            artifact.name,
            artifact.media_type,
            artifact.size_bytes,
            artifact.sha256,
            artifact.created_at,
            artifact.source,
            artifact.run_id,
            artifact.node,
            json.dumps(list(artifact.parents)),
            json.dumps(list(artifact.claim_ids)),
        )

    def put(
        self,
        name: str,
        content: bytes | str,
        *,
        source: str,
        media_type: str | None = None,
        run_id: str | None = None,
        node: str | None = None,
        parents: Sequence[str] = (),
        claim_ids: Sequence[str] = (),
    ) -> Artifact:
        artifact, payload = prepare(
            name,
            content,
            source=source,
            media_type=media_type,
            run_id=run_id,
            node=node,
            parents=parents,
            claim_ids=claim_ids,
        )
        # Blob first, row second: an orphaned blob is garbage, an orphaned row
        # is a record pointing at content that was never written.
        self._write_blob(artifact, payload)
        try:
            with _immediate_transaction(self._conn, self._lock) as conn:
                conn.execute(_INSERT, self._row_values(artifact))
        except sqlite3.IntegrityError as exc:
            # INSERT, not upsert — append-only means a repeated id is a bug, not
            # an update. Raised as the same ValueError the in-process backend
            # raises so the two cannot be told apart by the exception type.
            raise ValueError(f"artifact id {artifact.id!r} already exists") from exc
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        found = self._query(f"{_SELECT} WHERE id = ?", (artifact_id,))
        return found[0] if found else None

    def read(self, artifact_id: str) -> bytes:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"unknown artifact {artifact_id!r}")
        target = self.blob_path(artifact)
        if not target.exists():
            raise FileNotFoundError(
                f"artifact {artifact_id!r} ({artifact.name!r}) has a row but no content "
                f"at {target} — the blob directory was moved, pruned, or never shared"
            )
        return target.read_bytes()

    def read_text(self, artifact_id: str) -> str:
        return self.read(artifact_id).decode("utf-8")

    def latest(self, name: str) -> Artifact | None:
        found = self.versions(name)
        return found[-1] if found else None

    def versions(self, name: str) -> list[Artifact]:
        return self._query(f"{_SELECT} WHERE name = ? ORDER BY created_at, id", (name,))

    def by_run(self, run_id: str) -> list[Artifact]:
        return self._query(f"{_SELECT} WHERE run_id = ? ORDER BY rowid", (run_id,))

    def all_artifacts(self) -> list[Artifact]:
        return self._query(f"{_SELECT} ORDER BY rowid")


def render_artifacts(
    store: ArtifactStore,
    *,
    names: Sequence[str] | None = None,
    run_id: str | None = None,
    max_artifacts: int = 10,
    latest_only: bool = True,
) -> str:
    """A compact listing for a node's prompt: what exists, and where it came from.

    Newest first and capped, for the same reason the claim brief is: a run that
    produced four hundred files must not be able to spend a context window
    listing them. Content is never included — only an id to `read` it with.
    """
    if names is not None:
        found: list[Artifact] = []
        for name in names:
            versions = store.versions(name)
            found += versions[-1:] if latest_only else versions
    elif run_id is not None:
        found = list(store.by_run(run_id))
    else:
        found = list(store.all_artifacts())
        if latest_only:
            newest = {a.name: a for a in _sorted_versions(found)}
            found = list(newest.values())

    ordered = sorted(found, key=lambda a: (a.created_at, a.id), reverse=True)
    shown = ordered[:max_artifacts]
    if not shown:
        return "No artifacts recorded."

    lines = ["Artifacts (id · name · provenance):"]
    for artifact in shown:
        origin = f"source: {artifact.source}"
        if artifact.run_id:
            origin += f", run: {artifact.run_id}"
        if artifact.node:
            origin += f", node: {artifact.node}"
        lines.append(
            f"- {artifact.id} {artifact.name} "
            f"({artifact.media_type}, {artifact.size_bytes} bytes) [{origin}]"
        )
    omitted = len(ordered) - len(shown)
    if omitted:
        lines.append(f"- (+{omitted} more artifacts omitted)")
    return "\n".join(lines)
