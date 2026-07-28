"""Noticing that a new claim disagrees with a stored one.

`supersede(old_id, new_claim)` is the correction primitive, and it requires the
caller to already know `old_id`. That is fine for a human editing memory and
useless for an agent, which has a fact in hand and no idea whether the store
already believes something else about the same thing. So nothing was detecting
contradictions — the store would simply hold both, and retrieval would render
them side by side as if they agreed.

**What is detected, exactly.** Two claims contradict when they share a
normalized (subject, predicate) and differ in normalized object. That is a
structural test, not a semantic one, and it is worth being blunt about the
limits:

* it does not catch "is fast" versus "is slow" — different predicates, so
  nothing here relates them;
* it does not catch a rephrased object ("LangGraph" versus "the LangGraph
  library"), because `_normalize` compares strings, not meanings;
* it *will* flag a legitimately multi-valued predicate ("depends on" with three
  different dependencies) as three-way disagreement.

The last one is why detection never resolves anything by itself. It reports,
and a caller supersedes on purpose. Silently auto-superseding on a detected
conflict would delete the second half of a multi-valued fact, and it would do
it inside the one subsystem whose promise is that facts are never destroyed.
"""

from __future__ import annotations

from dataclasses import dataclass

from grapharc.memory.store import Claim, ClaimStore, _normalize


@dataclass(frozen=True)
class Contradiction:
    """Two claims that cannot both be current: same key, different value.

    `incoming` is the claim being offered (or, for a scan of what is already
    stored, the newer of the pair); `existing` is the stored claim it disagrees
    with. Both carry full provenance, which is the point — a caller deciding
    which to keep needs to see where each came from.
    """

    incoming: Claim
    existing: Claim

    @property
    def key(self) -> tuple[str, str]:
        return self.existing.key

    def describe(self) -> str:
        return (
            f"{self.incoming.subject} {self.incoming.predicate}: "
            f"{self.incoming.object!r} (from {self.incoming.source}) contradicts "
            f"{self.existing.object!r} (from {self.existing.source}, "
            f"claim {self.existing.id})"
        )


@dataclass(frozen=True)
class ConflictGroup:
    """Every current claim sharing one (subject, predicate), oldest first.

    A group exists only when the objects actually differ, so a group is always
    evidence of disagreement rather than of a multi-claim key.
    """

    subject: str
    predicate: str
    claims: tuple[Claim, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.subject), _normalize(self.predicate))

    def contradictions(self) -> list[Contradiction]:
        """The newest claim paired against each older one it disagrees with.

        Newest-as-incoming is a convention, not a judgement: recency is not
        correctness, and `supersede_conflicting` will happily be pointed the
        other way.
        """
        newest = self.claims[-1]
        return [
            Contradiction(incoming=newest, existing=claim)
            for claim in self.claims[:-1]
            if _normalize(claim.object) != _normalize(newest.object)
        ]


def detect_contradictions(store: ClaimStore, claim: Claim) -> list[Contradiction]:
    """Current claims that disagree with `claim`, oldest first.

    Reads only through `current(subject, predicate)`, which both backends
    already index, so this costs the same as the lookup an agent would do
    anyway — and there is no new method on `ClaimStore` to keep in sync.

    Works whether or not `claim` has been stored yet: the claim itself is
    excluded by id. An identical claim (same normalized object) is a duplicate,
    not a contradiction, and is not reported.
    """
    rivals = [
        other
        for other in store.current(claim.subject, claim.predicate)
        if other.id != claim.id and _normalize(other.object) != _normalize(claim.object)
    ]
    rivals.sort(key=lambda c: (c.observed_at, c.id))
    return [Contradiction(incoming=claim, existing=other) for other in rivals]


def add_and_detect(store: ClaimStore, claim: Claim) -> tuple[Claim, list[Contradiction]]:
    """Store the claim, then report what it disagrees with.

    The claim is written either way. Refusing to store a contradicting claim
    would be the store deciding which source is right, and it has no basis to;
    holding both and saying so leaves that decision with the caller, who can
    call `supersede_conflicting` when it has one.

    Detection runs after the write so the report reflects the store as it now
    stands, including the case where two runs raced and both landed.
    """
    stored = store.add(claim)
    return stored, detect_contradictions(store, stored)


def supersede_conflicting(store: ClaimStore, contradiction: Contradiction) -> Claim:
    """Resolve one detected contradiction in favour of `incoming`.

    A thin wrapper over `supersede`, and deliberately still a separate call:
    detection reports, resolution writes, and the two stay apart so that
    "memory noticed a disagreement" can never mean "memory quietly picked a
    winner".

    Raises whatever `supersede` raises — in particular `ValueError` if the
    existing claim was already superseded between detection and resolution,
    which is the correct outcome for a stale report rather than an error to
    swallow.
    """
    return store.supersede(contradiction.existing.id, contradiction.incoming)


def group_conflicts(claims: list[Claim]) -> list[ConflictGroup]:
    """Group current claims by key and keep only the keys that disagree.

    Superseded claims are ignored: a corrected fact is not a contradiction, it
    is a resolved one, and `retrieve_dead_ends` already reports those.
    """
    buckets: dict[tuple[str, str], list[Claim]] = {}
    for claim in claims:
        if claim.is_current:
            buckets.setdefault(claim.key, []).append(claim)

    groups: list[ConflictGroup] = []
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda c: (c.observed_at, c.id))
        if len({_normalize(c.object) for c in members}) < 2:
            continue
        groups.append(
            ConflictGroup(
                subject=members[0].subject,
                predicate=members[0].predicate,
                claims=tuple(members),
            )
        )
    return groups


def unresolved_contradictions(
    store: ClaimStore, *, entities: list[str] | None = None
) -> list[ConflictGroup]:
    """Disagreements already sitting in the store, keyed by subject+predicate.

    With `entities`, only claims about those subjects are examined (indexed on
    both backends). Without, the whole corpus is scanned — O(claims), so it is
    a maintenance/report call, not something to run inside a loop.
    """
    if entities is None:
        claims = store.all_claims()
    else:
        seen: set[str] = set()
        claims = []
        for entity in entities:
            for claim in store.current(entity):
                if claim.id not in seen:
                    seen.add(claim.id)
                    claims.append(claim)
    return group_conflicts(claims)
