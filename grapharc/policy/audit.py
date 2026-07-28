"""The decision audit: what was it allowed to do, and which rule said so.

The product thesis needs three questions answerable from the record alone —
what did it do, what was it *allowed* to do, and why did it stop. Traces
(`grapharc.observe.trace`) answer the first. This answers the second: every
call into `PolicyEngine` appends one record naming the subject, the tenant, the
effect, the rule that caused it, and the policy version *and digest* it was
decided under, so a later reader can tell which bytes of policy were in force
even if the version string was reused.

Records are append-only and, when a path is given, land as JSONL in the same
file-first shape the trace recorder uses — human-readable, greppable,
git-versionable.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class AuditRecord(BaseModel):
    """Base for anything the audit log holds. `kind` is how a reader dispatches."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    ts: str = Field(default_factory=_now)


class AuditLog:
    """Append-only, thread-safe record of policy decisions.

    In memory always; on disk as JSONL when `path` is given. `max_entries` caps
    the in-memory list only — the file keeps everything, so capping without a
    path really does lose records, which is why it defaults to off.
    """

    def __init__(self, path: str | Path | None = None, *, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be at least 1, or None for unbounded")
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> AuditRecord:
        line = json.dumps(entry.model_dump(), default=repr, ensure_ascii=False)
        with self._lock:
            self._entries.append(entry)
            if self.max_entries is not None and len(self._entries) > self.max_entries:
                del self._entries[: len(self._entries) - self.max_entries]
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return entry

    def entries(self, *, kind: str | None = None, tenant: str | None = None) -> list[AuditRecord]:
        with self._lock:
            found: Iterable[AuditRecord] = list(self._entries)
        if kind is not None:
            found = [e for e in found if e.kind == kind]
        if tenant is not None:
            found = [e for e in found if getattr(e, "tenant", None) == tenant]
        return list(found)

    def last(self) -> AuditRecord | None:
        with self._lock:
            return self._entries[-1] if self._entries else None

    def read_jsonl(self) -> list[dict[str, Any]]:
        """Re-read the file as dicts. Empty when no path was configured."""
        if self.path is None or not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
