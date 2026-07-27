"""The memory graph: claims with provenance, superseded not deleted.

This is the second of the "two graphs". The orchestration graph is temporary —
it ends with the run. This one is meant to outlive it: entities, claims,
sources, and time. Which backend you pick decides whether it actually does —
`MemoryStore` here dies with the process; `SQLiteMemoryStore` (sqlite_store.py)
implements the same `ClaimStore` protocol against a file and does not.

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
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

_clock_lock = threading.Lock()
_last_now = ""


def _now() -> str:
    """A strictly increasing UTC timestamp, at microsecond precision.

    Two properties matter here and neither is free.

    *Resolution.* At millisecond precision a loop of corrections all land on
    the same tick. Ties are not harmless: `list.sort` is stable, so
    `reverse=True` does **not** reverse a tied run — it hands it back in
    whatever order the backend happened to iterate. That is what made
    `retrieve_dead_ends` return the oldest corrections from `MemoryStore`
    (dict order) and the newest from `SQLiteMemoryStore` (rowid order) given
    identical inputs.

    *Monotonicity.* Microseconds make ties rare, not impossible — a coarse
    platform clock or NTP stepping the clock backwards brings them straight
    back. So the last value handed out is remembered and the next one is
    forced strictly past it. Every claim a process writes therefore carries a
    distinct, correctly ordered timestamp.

    Both backends import this one function, which is why fixing it here is
    what makes them agree. The format stays fixed-width ISO-8601 UTC, so
    SQLite's lexicographic ordering of the TEXT column is still chronological.
    """
    global _last_now
    with _clock_lock:
        stamp = datetime.now(UTC).isoformat(timespec="microseconds")
        if stamp <= _last_now:
            stamp = (
                datetime.fromisoformat(_last_now) + timedelta(microseconds=1)
            ).isoformat(timespec="microseconds")
        _last_now = stamp
        return stamp


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


def validate_supersede(old: Claim | None, old_id: str, new_claim: Claim) -> Claim:
    """The `supersede` preconditions, in one place so every backend shares them.

    Duplicating these checks per backend is how they drift. The self-supersede
    case is the one that actually did: `supersede(old_id, new_claim)` with
    `new_claim.id == old_id` is a single row being asked to replace itself, and
    each backend resolved that differently — `MemoryStore` kept the old content
    (its second dict write clobbered the first), `SQLiteMemoryStore` kept the
    new content (the UPDATE followed the upsert). Both ended with
    `superseded_by == id`: a claim that is its own correction, invisible to
    `current` and pointing at itself in `dead_ends`. There is no coherent
    reading of that, so it is refused before anything is written.

    Returns the validated `old` claim so callers can use it without a second
    `None` check.
    """
    if old is None:
        raise KeyError(f"unknown claim {old_id!r}")
    if old.superseded_by is not None:
        raise ValueError(f"claim {old_id!r} was already superseded by {old.superseded_by!r}")
    if new_claim.id == old_id:
        raise ValueError(
            f"claim {old_id!r} cannot supersede itself — a correction is a new "
            "claim with its own id, so the old one survives to be pointed at"
        )
    return old


@runtime_checkable
class ClaimStore(Protocol):
    """The contract every memory backend implements.

    Methods alone do not capture it, so the invariants are stated here and
    enforced by the shared conformance tests both backends run against:

    * append-only — `add` and `supersede` never remove a claim;
    * `supersede` marks the old claim and links it to the new one, keeping the
      old claim's own provenance (source, observed_at, run_id) intact;
    * superseding an already-superseded claim raises `ValueError`, superseding
      an unknown claim raises `KeyError`, and superseding a claim with a
      replacement carrying that same id raises `ValueError` — all three are
      `validate_supersede`, which backends call rather than re-derive;
    * timestamps come from `_now`, which is strictly increasing, so any two
      claims are orderable and both backends order them the same way;
    * entity lookup goes through `_normalize`, so it is Unicode-aware rather
      than ASCII case-folding;
    * `current` excludes superseded claims, `dead_ends` returns only those.

    `isinstance` against this protocol checks method *names* only — passing it
    is not evidence the invariants hold.
    """

    def add(self, claim: Claim) -> Claim: ...

    def get(self, claim_id: str) -> Claim | None: ...

    def supersede(self, old_id: str, new_claim: Claim) -> Claim: ...

    def current(self, subject: str, predicate: str | None = None) -> list[Claim]: ...

    def history(self, subject: str, predicate: str) -> list[Claim]: ...

    def dead_ends(self, subject: str) -> list[Claim]: ...

    def all_claims(self) -> list[Claim]: ...


class MemoryStore:
    """In-process backend: a dict behind a lock, gone when the process exits.

    The default because it keeps the gate suite hermetic and the library
    dependency-free. It is not durable: for memory that a later run can read,
    use `SQLiteMemoryStore`, which implements the same `ClaimStore` protocol
    and stores claims in a file.
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
            old = validate_supersede(self._claims.get(old_id), old_id, new_claim)
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
