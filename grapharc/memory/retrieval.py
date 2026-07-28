"""Retrieval over the claim graph: relevance, traversal, and a token budget.

Hermes' lesson applied: memory is exposed to nodes under a hard budget,
search-first. A node asks a question, or names entities, and gets back current
claims with their provenance, plus an explicit note of what was superseded so
it does not re-walk a known dead end.

**What "search" means here**, because the previous version of this module was
labelled GraphRAG while doing an exact-string subject match:

* *lexical* — Okapi BM25F over subject+predicate+object, subject weighted
  highest (`index.py`). Rare terms count for more than common ones, and a
  claim can be found by a word in its object, not only by its subject.
* *vector* — optional and injected. Nothing is embedded unless you pass an
  `Embedder`; `HashingEmbedder` is a dependency-free local fallback that adds
  spelling robustness, not meaning (see `embeddings.py`). Similarities below
  `MIN_SIMILARITY` are discarded, so this channel stays silent on a query the
  corpus has nothing to say about instead of confidently returning its least
  bad row.
* *graph* — the object of a claim is read as an entity, so a question about A
  reaches facts about B when A points at B. Incoming edges are followed too.
  Each hop multiplies the inherited score by `hop_decay`, so a neighbour is
  evidence, never the headline.

**Cost.** Scoring needs corpus statistics, so `search` builds a `ClaimIndex`
over every claim: O(claims) per call. Pass `index=` to build it once and query
it many times. Nothing here is sublinear and nothing here pretends to be.

**Ordering.** Every result list is sorted by a total order — score, then
observation time, then id — so the two backends return the same ranking for
the same claims. Scores are rounded before comparison because two accumulation
orders can differ in the last bits.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from grapharc.memory.contradiction import ConflictGroup, group_conflicts
from grapharc.memory.embeddings import Embedder
from grapharc.memory.index import MIN_SIMILARITY, ClaimIndex, hybrid, rank_key, top_scored
from grapharc.memory.store import Claim, ClaimStore, _normalize

DEFAULT_MAX_CLAIMS = 20
# Corrections accumulate forever: every supersede adds one more dead end about
# the same subject, so this section grows without bound as a project is worked
# on. It is a hint, not the answer, so it gets a smaller share of the budget.
DEFAULT_MAX_DEAD_ENDS = 10

# One hop by default. Two is already a lot of drift on a dense graph, and the
# decay below means a hop-2 claim contributes about a fifth of its seed.
DEFAULT_HOPS = 1
# A neighbour inherits 45% of the score of the claim it was reached through, so
# it lands below that claim and below anything else that matched directly about
# as well — related evidence in the middle of the list rather than at its head.
# It can still outrank a *weak* direct match, which is the intent.
DEFAULT_HOP_DECAY = 0.45
# Lexical carries most of the weight: BM25 is precise when the words match, and
# the shipped fallback embedder is a spelling-robustness channel rather than a
# semantic one. Lower it towards 0 when you inject a real embedder.
DEFAULT_LEXICAL_WEIGHT = 0.6

# Naming an entity is a stronger signal than any amount of text similarity: the
# caller *knows* the subject is relevant. So an entity match contributes a flat
# 1.0 and the text channels contribute at most 0.9 on top, which keeps the two
# bands from overlapping — a claim about a named subject always outranks a claim
# found only by its text, and relevance orders the claims inside each band.
ENTITY_SEED_SCORE = 1.0
TEXT_SEED_SCALE = 0.9

_FACTS_HEADER = "Known facts (with provenance):"
_CONFLICT_HEADER = "Contradictions — same subject and predicate, different values:"
_DEAD_HEADER = "Superseded — do not re-derive these:"
_EMPTY = "No prior knowledge about these entities."
# Held back from the budget for the truncation note. The note is the one line
# that must never be dropped — dropping it turns a truncated brief into a brief
# that looks complete — so a budget below this reserve is overshot by the note
# alone. That is the only case where the rendering exceeds `max_tokens`.
_NOTE_RESERVE = 24


@dataclass(frozen=True)
class ScoredClaim:
    """A retrieved claim and why it was retrieved.

    `hops` is 0 for a direct match and n for a claim reached by n graph edges;
    `via` is the id of the claim it was reached through, so a caller can render
    or audit the path rather than being handed an unexplained fact.
    """

    claim: Claim
    score: float
    hops: int = 0
    via: str | None = None
    channel: str = "entity"


def _estimate_tokens(text: str) -> int:
    """Four characters per token, rounded up — an estimate, not a count.

    No tokenizer ships with this library and adding one for a context brief is
    not worth a dependency. Four is the usual rule of thumb for English prose
    and it is *not* reliable for these lines: ids, hyphenated values and ISO
    timestamps all split into more tokens than their length suggests, so a
    brief that fits this estimate can still overrun a real tokenizer. Pass
    `count_tokens` when the budget has to be exact.
    """
    return max(1, math.ceil(len(text) / 4))


def search(
    store: ClaimStore,
    *,
    query: str | None = None,
    entities: Sequence[str] = (),
    max_claims: int = DEFAULT_MAX_CLAIMS,
    hops: int = DEFAULT_HOPS,
    hop_decay: float = DEFAULT_HOP_DECAY,
    embedder: Embedder | None = None,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    min_similarity: float = MIN_SIMILARITY,
    index: ClaimIndex | None = None,
    include_superseded: bool = False,
) -> list[ScoredClaim]:
    """Rank claims by relevance to a query and/or a set of named entities.

    With neither a query nor an entity there is nothing to be relevant to, so
    the result is empty rather than "the whole store, recently first" — that
    would hand a node a random slice of memory and call it context.

    With entities but no query the scores are all equal, so the ranking is pure
    recency among direct matches, then hop-1 neighbours, then hop-2: the same
    order the previous implementation produced, with neighbours appended.

    With both, the two signals add: a named subject that also answers the query
    leads, then the rest of the named subjects by recency, then everything the
    text channels found.

    Superseded claims are excluded by default, and are not traversed through
    either: a retracted fact's edges are retracted with it.
    """
    named = tuple(entities or ())
    if not query and not named:
        return []

    idx = index if index is not None else ClaimIndex.from_store(store, embedder=embedder)
    if not idx.claims:
        return []

    def keep(claim: Claim) -> bool:
        return include_superseded or claim.is_current

    hits: dict[str, ScoredClaim] = {}

    def offer(claim_id: str, score: float, hop: int, via: str | None, channel: str) -> bool:
        claim = idx.by_id.get(claim_id)
        if claim is None or not keep(claim) or score <= 0.0:
            return False
        previous = hits.get(claim_id)
        if previous is not None and previous.score >= score:
            return False
        hits[claim_id] = ScoredClaim(
            claim=claim, score=score, hops=hop, via=via, channel=channel
        )
        return True

    seeds: dict[str, float] = {}
    channels: dict[str, str] = {}
    for entity in named:
        for claim_id in idx.with_subject(entity):
            seeds[claim_id] = ENTITY_SEED_SCORE
            channels[claim_id] = "entity"

    if query:
        text_channel = "hybrid" if idx.vectors else "lexical"
        blended = hybrid(
            idx.bm25(query),
            idx.similarity(query, min_similarity=min_similarity),
            lexical_weight=lexical_weight,
        )
        for claim_id, score in blended.items():
            if score <= 0.0:
                continue
            seeds[claim_id] = seeds.get(claim_id, 0.0) + TEXT_SEED_SCALE * score
            named_too = channels.get(claim_id) == "entity"
            channels[claim_id] = f"entity+{text_channel}" if named_too else text_channel

    for claim_id, score in seeds.items():
        offer(claim_id, score, 0, None, channels[claim_id])

    # Only the strongest seeds are expanded. Traversal from every weak match
    # would drown the result set in second-hand facts and make the cost
    # quadratic in a dense graph.
    expand_limit = max(1, max_claims)
    frontier = top_scored({cid: hit.score for cid, hit in hits.items()}, idx.by_id, expand_limit)
    for depth in range(1, max(0, hops) + 1):
        promoted: dict[str, float] = {}
        for parent_id in frontier:
            parent = hits[parent_id]
            inherited = parent.score * hop_decay
            for neighbour in idx.neighbours(parent.claim):
                if offer(neighbour, inherited, depth, parent_id, "graph"):
                    promoted[neighbour] = inherited
        if not promoted:
            break
        frontier = top_scored(promoted, idx.by_id, expand_limit)

    ordered = sorted(
        hits.values(), key=lambda hit: rank_key(hit.score, hit.claim), reverse=True
    )
    return ordered[:max_claims]


def retrieve(
    store: ClaimStore,
    *,
    entities: Sequence[str] = (),
    query: str | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    hops: int = DEFAULT_HOPS,
    embedder: Embedder | None = None,
    index: ClaimIndex | None = None,
) -> list[Claim]:
    """The claims from `search`, without the scores. Hard-capped."""
    return [
        hit.claim
        for hit in search(
            store,
            query=query,
            entities=entities,
            max_claims=max_claims,
            hops=hops,
            embedder=embedder,
            index=index,
        )
    ]


def retrieve_dead_ends(
    store: ClaimStore,
    *,
    entities: Sequence[str],
    max_dead_ends: int = DEFAULT_MAX_DEAD_ENDS,
    query: str | None = None,
    embedder: Embedder | None = None,
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

    A `query` re-ranks by relevance first and correction time second, scored
    against the dead ends alone rather than the whole corpus: the question
    being asked is "which of these retracted facts matter to me", so the term
    statistics that matter are the ones inside this set.
    """
    seen: set[str] = set()
    out: list[Claim] = []
    for entity in entities:
        for claim in store.dead_ends(entity):
            if claim.id not in seen:
                seen.add(claim.id)
                out.append(claim)

    if query:
        idx = ClaimIndex(out, embedder=embedder)
        scores = hybrid(
            idx.bm25(query), idx.similarity(query), lexical_weight=DEFAULT_LEXICAL_WEIGHT
        )
        out.sort(
            key=lambda c: (
                round(scores.get(c.id, 0.0), 9),
                c.superseded_at or c.observed_at,
                c.observed_at,
                c.id,
            ),
            reverse=True,
        )
    else:
        out.sort(
            key=lambda c: (c.superseded_at or c.observed_at, c.observed_at, c.id), reverse=True
        )
    return out[:max_dead_ends], max(0, len(out) - max_dead_ends)


def _fact_line(hit: ScoredClaim, index: ClaimIndex | None) -> str:
    claim = hit.claim
    line = (
        f"- {claim.subject} {claim.predicate} {claim.object} "
        f"[source: {claim.source}, observed: {claim.observed_at}]"
    )
    if hit.hops and hit.via and index is not None:
        parent = index.by_id.get(hit.via)
        if parent is not None:
            # A fact nobody asked for needs its edge stated, or the node cannot
            # tell a direct answer from something two steps away.
            line += f" (related via {parent.subject})"
    return line


def _conflict_line(group: ConflictGroup) -> str:
    values = " | ".join(
        f"{claim.object!r} ({claim.id}, {claim.source})" for claim in group.claims
    )
    return f"- {group.subject} {group.predicate}: {values}"


def _dead_line(claim: Claim) -> str:
    return (
        f"- {claim.subject} {claim.predicate} {claim.object} "
        f"(superseded by {claim.superseded_by})"
    )


def _join(blocks: list[tuple[str, list[str]]]) -> list[str]:
    lines: list[str] = []
    for header, entries in blocks:
        if not entries:
            continue
        if lines:
            lines.append("")
        lines.append(header)
        lines += entries
    return lines


def _fit(
    blocks: list[tuple[str, list[str]]],
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Fill blocks in priority order until the budget is gone.

    Whole entries are dropped from the tail, never truncated mid-line: half a
    claim with half its provenance is worse than no claim. A block whose header
    fits but whose first entry does not is dropped entirely, so no section
    header ever appears over nothing.
    """
    budget = max_tokens - _NOTE_RESERVE
    lines: list[str] = []
    spent = 0
    dropped = 0
    for header, entries in blocks:
        if not entries:
            continue
        separator = 1 if lines else 0
        head_cost = count_tokens(header) + separator
        kept: list[str] = []
        cost = head_cost
        for position, entry in enumerate(entries):
            entry_cost = count_tokens(entry)
            if spent + cost + entry_cost > budget:
                dropped += len(entries) - position
                break
            kept.append(entry)
            cost += entry_cost
        if not kept:
            # The loop above already counted every entry as dropped when it
            # broke at position 0; the header simply never gets emitted.
            continue
        if separator:
            lines.append("")
        lines.append(header)
        lines += kept
        spent += cost
    if dropped:
        lines.append(f"(+{dropped} lines omitted to fit a {max_tokens}-token budget)")
    return lines


def render_context(
    store: ClaimStore,
    *,
    entities: Sequence[str] = (),
    query: str | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    max_dead_ends: int = DEFAULT_MAX_DEAD_ENDS,
    hops: int = DEFAULT_HOPS,
    embedder: Embedder | None = None,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    flag_contradictions: bool = True,
) -> str:
    """A compact memory brief for a node's prompt, in priority order.

    Provenance travels with each fact — a claim without its source is a rumor.
    Three sections, most load-bearing first: what is believed, what is disputed,
    what was already tried and retracted. The retracted section covers the
    named entities *and* the subjects of the facts being shown, so a brief
    driven by a query still warns about the dead ends behind its own answers.

    Builds one `ClaimIndex` over the whole store, so this is O(claims) per
    call, the same as `search`.

    Every section is capped by count (`max_claims`, `max_dead_ends`), because an
    uncapped one outgrows any context window on a subject that has been
    corrected often. `max_tokens` caps the brief as a whole on top of that, and
    says in the output how many lines it dropped: a brief that silently omits
    facts is worse than a short one, because the node cannot tell.

    The token figure is `count_tokens` if given and a 4-chars-per-token estimate
    otherwise — so `max_tokens` is a bound on the estimate, not on any
    particular tokenizer. The one case where the output exceeds it is a budget
    too small to hold the truncation note, which is never dropped.

    With `max_tokens` unset the rendering is exactly what it always was, so
    existing prompts do not move.
    """
    idx = ClaimIndex.from_store(store, embedder=embedder)
    hits = search(
        store,
        query=query,
        entities=entities,
        max_claims=max_claims,
        hops=hops,
        embedder=embedder,
        index=idx,
    )
    # Corrections are looked up per subject, so a query-only brief would carry
    # no warnings at all. The subjects of the facts being shown are exactly the
    # ones whose dead ends this brief's reader can walk into.
    warned: list[str] = list(entities)
    seen_subjects = {_normalize(name) for name in warned}
    for hit in hits:
        subject = _normalize(hit.claim.subject)
        if subject not in seen_subjects:
            seen_subjects.add(subject)
            warned.append(hit.claim.subject)
    dead, omitted = retrieve_dead_ends(
        store, entities=warned, max_dead_ends=max_dead_ends, query=query, embedder=embedder
    )

    conflicts: list[ConflictGroup] = []
    if flag_contradictions:
        # Scoped to the facts actually shown. A disagreement the reader cannot
        # see is not this brief's problem; `unresolved_contradictions` scans the
        # store for that.
        conflicts = group_conflicts([hit.claim for hit in hits])

    dead_entries = [_dead_line(claim) for claim in dead]
    if dead_entries and omitted:
        dead_entries.append(f"- (+{omitted} older corrections omitted)")

    blocks = [
        (_FACTS_HEADER, [_fact_line(hit, idx) for hit in hits]),
        (_CONFLICT_HEADER, [_conflict_line(group) for group in conflicts]),
        (_DEAD_HEADER, dead_entries),
    ]

    if max_tokens is None:
        lines = _join(blocks)
    else:
        lines = _fit(blocks, max_tokens, count_tokens or _estimate_tokens)
    return "\n".join(lines) if lines else _EMPTY


def known_entities(store: ClaimStore) -> set[str]:
    return {_normalize(c.subject) for c in store.all_claims()}
