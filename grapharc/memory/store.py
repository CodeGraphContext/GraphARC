"""The durable memory graph: claims with provenance, superseded not deleted.

This is the second of the "two graphs". The orchestration graph is temporary —
it ends with the run. This one survives: entities, claims, sources, and time.

The pre-AI graph standard applies here more than anywhere else in GraphARC —
every edge means something, every path can be explained. So a claim carries
where it came from, when it was observed, which run produced it, and what (if
anything) later superseded it. Facts are never destructively overwritten: a
correction adds a new claim and marks the old one superseded, so run #51 can
see that run #37 replaced a fact from run #12 and avoid the dead end.
"""

from __future__ import annotations

import re
import threading
import unicodedata
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _normalize(entity: str) -> str:
    """Entity resolution: case- and punctuation-insensitive, Unicode-aware.

    An ASCII-only character class would erase every non-Latin script, silently
    merging unrelated entities (東京 and 北京 both becoming the empty key) — the
    worst failure mode for a store whose job is keeping facts attributable.
    """
    folded = unicodedata.normalize("NFKC", entity).casefold()
    norm = re.sub(r"[\W_]+", " ", folded, flags=re.UNICODE).strip()
    return norm or folded.strip()


class Claim(BaseModel):
    """One fact, with the provenance that makes it auditable."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    subject: str
    predicate: str
    object: str
    # Provenance — the whole point.
    source: str
    observed_at: str = Field(default_factory=_now)
    run_id: str | None = None
    confidence: float = 1.0
    superseded_by: str | None = None
    superseded_at: str | None = None

    @property
    def is_current(self) -> bool:
        return self.superseded_by is None

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.subject), _normalize(self.predicate))


class MemoryStore:
    """In-memory backend (the default, and what tests run against).

    A Neo4j-backed implementation satisfies the same interface; this one keeps
    the gate suite hermetic and the library dependency-free by default.
    """

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        self._lock = threading.Lock()

    def add(self, claim: Claim) -> Claim:
        with self._lock:
            self._claims[claim.id] = claim
        return claim

    def get(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def supersede(self, old_id: str, new_claim: Claim) -> Claim:
        """Record a correction. The old claim stays — marked, not deleted."""
        with self._lock:
            old = self._claims.get(old_id)
            if old is None:
                raise KeyError(f"unknown claim {old_id!r}")
            if old.superseded_by is not None:
                raise ValueError(
                    f"claim {old_id!r} was already superseded by {old.superseded_by!r}"
                )
            self._claims[new_claim.id] = new_claim
            self._claims[old_id] = old.model_copy(
                update={"superseded_by": new_claim.id, "superseded_at": _now()}
            )
        return new_claim

    def current(self, subject: str, predicate: str | None = None) -> list[Claim]:
        """Live claims about a subject — superseded ones are excluded."""
        subj = _normalize(subject)
        pred = _normalize(predicate) if predicate else None
        return [
            c
            for c in self._claims.values()
            if c.is_current
            and _normalize(c.subject) == subj
            and (pred is None or _normalize(c.predicate) == pred)
        ]

    def history(self, subject: str, predicate: str) -> list[Claim]:
        """Every claim ever made about (subject, predicate), oldest first."""
        key = (_normalize(subject), _normalize(predicate))
        return sorted(
            (c for c in self._claims.values() if c.key == key),
            key=lambda c: c.observed_at,
        )

    def dead_ends(self, subject: str) -> list[Claim]:
        """Superseded claims — what an earlier run believed and was corrected on."""
        subj = _normalize(subject)
        return [
            c
            for c in self._claims.values()
            if not c.is_current and _normalize(c.subject) == subj
        ]

    def all_claims(self) -> list[Claim]:
        return list(self._claims.values())
