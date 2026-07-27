"""GraphRAG-style retrieval — a bounded context view, not a graph dump.

Hermes' lesson applied: memory is exposed to nodes under a hard token budget,
search-first. A node asks about entities and gets back current claims with
their provenance, plus an explicit note of what was superseded so it doesn't
re-walk a known dead end.
"""

from __future__ import annotations

from grapharc.memory.store import Claim, MemoryStore, _normalize

DEFAULT_MAX_CLAIMS = 20


def retrieve(
    store: MemoryStore,
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
    out.sort(key=lambda c: c.observed_at, reverse=True)
    return out[:max_claims]


def render_context(
    store: MemoryStore, *, entities: list[str], max_claims: int = DEFAULT_MAX_CLAIMS
) -> str:
    """A compact, human-readable memory brief for a node's prompt.

    Provenance travels with each fact — a claim without its source is a rumor.
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
    dead: list[Claim] = []
    for entity in entities:
        dead += [c for c in store.dead_ends(entity) if c not in dead]
    if dead:
        lines.append("")
        lines.append("Superseded — do not re-derive these:")
        lines += [
            f"- {c.subject} {c.predicate} {c.object} "
            f"(superseded by {c.superseded_by})"
            for c in dead
        ]
    return "\n".join(lines) if lines else "No prior knowledge about these entities."


def known_entities(store: MemoryStore) -> set[str]:
    return {_normalize(c.subject) for c in store.all_claims()}
