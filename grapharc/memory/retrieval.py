"""GraphRAG-style retrieval — a bounded context view, not a graph dump.

Hermes' lesson applied: memory is exposed to nodes under a hard token budget,
search-first. A node asks about entities and gets back current claims with
their provenance, plus an explicit note of what was superseded so it doesn't
re-walk a known dead end.
"""

from __future__ import annotations

from grapharc.memory.store import Claim, ClaimStore, _normalize

DEFAULT_MAX_CLAIMS = 20
# Corrections accumulate forever: every supersede adds one more dead end about
# the same subject, so this section grows without bound as a project is worked
# on. It is a hint, not the answer, so it gets a smaller share of the budget.
DEFAULT_MAX_DEAD_ENDS = 10


def retrieve(
    store: ClaimStore,
    *,
    entities: list[str],
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> list[Claim]:
    """Current claims about the given entities, newest first, hard-capped."""
    seen: set[str] = set()
    out: list[Claim] = []
    for entity in entities:
        for claim in store.current(entity):
            if claim.id not in seen:
                seen.add(claim.id)
                out.append(claim)
    out.sort(key=lambda c: (c.observed_at, c.id), reverse=True)
    return out[:max_claims]


def retrieve_dead_ends(
    store: ClaimStore,
    *,
    entities: list[str],
    max_dead_ends: int = DEFAULT_MAX_DEAD_ENDS,
) -> tuple[list[Claim], int]:
    """Superseded claims, most recently corrected first, plus the omitted count.

    Newest first because the oldest corrections are the least likely to be the
    dead end a node is about to walk into. Since the cap discards the tail,
    "newest first" is not cosmetic — it decides *which* corrections a node is
    shown at all, so the order has to be a total one rather than whatever the
    backend's iteration happened to leave behind.

    The sort key is therefore three fields, not one. Correction time is the
    real ordering; observation time then id break any remaining tie with values
    both backends store identically, so `MemoryStore` and `SQLiteMemoryStore`
    render the same "Superseded" section from the same inputs. Ties used to be
    left to `list.sort`'s stability, which preserves — never reverses — the
    input order, so `reverse=True` quietly returned tied groups oldest-first.
    """
    seen: set[str] = set()
    out: list[Claim] = []
    for entity in entities:
        for claim in store.dead_ends(entity):
            if claim.id not in seen:
                seen.add(claim.id)
                out.append(claim)
    out.sort(key=lambda c: (c.superseded_at or c.observed_at, c.observed_at, c.id), reverse=True)
    return out[:max_dead_ends], max(0, len(out) - max_dead_ends)


def render_context(
    store: ClaimStore,
    *,
    entities: list[str],
    max_claims: int = DEFAULT_MAX_CLAIMS,
    max_dead_ends: int = DEFAULT_MAX_DEAD_ENDS,
) -> str:
    """A compact, human-readable memory brief for a node's prompt.

    Provenance travels with each fact — a claim without its source is a rumor.
    Both sections are capped: an uncapped one can outgrow any context budget on
    a subject that has been corrected often.
    """
    claims = retrieve(store, entities=entities, max_claims=max_claims)
    lines: list[str] = []
    if claims:
        lines.append("Known facts (with provenance):")
        lines += [
            f"- {c.subject} {c.predicate} {c.object} "
            f"[source: {c.source}, observed: {c.observed_at}]"
            for c in claims
        ]
    dead, omitted = retrieve_dead_ends(store, entities=entities, max_dead_ends=max_dead_ends)
    if dead:
        lines.append("")
        lines.append("Superseded — do not re-derive these:")
        lines += [
            f"- {c.subject} {c.predicate} {c.object} "
            f"(superseded by {c.superseded_by})"
            for c in dead
        ]
        if omitted:
            lines.append(f"- (+{omitted} older corrections omitted)")
    return "\n".join(lines) if lines else "No prior knowledge about these entities."


def known_entities(store: ClaimStore) -> set[str]:
    return {_normalize(c.subject) for c in store.all_claims()}
