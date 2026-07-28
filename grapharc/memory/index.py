"""The searchable view of a claim corpus: BM25F, adjacency, optional vectors.

Built from `ClaimStore.all_claims()`, which means building one is O(corpus).
That is stated plainly because the previous retrieval was criticised for being
a linear scan, and this does not remove the scan — it makes the scan produce a
ranking instead of an exact-string filter, and lets you pay for it once and
query many times (`search(..., index=...)`).

Three structures live here because they answer three different questions:

* the inverted index answers "which claims mention these words, and how
  unusual are those words in this corpus" — BM25F, so a hit in the subject
  counts for more than a hit in the object;
* the adjacency maps answer "which claims are about the thing this claim
  points at" — the edges that make this a graph rather than a table;
* the vectors answer "which claims look like this text even though they share
  no exact term", and only exist if an embedder was supplied.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from grapharc.memory.embeddings import Embedder, normalize_vector
from grapharc.memory.store import Claim, _normalize
from grapharc.memory.text import tokenize

# BM25 defaults. k1 controls how fast term frequency saturates, b how hard long
# documents are penalized. Claims are three short fields, so the usual Okapi
# values are fine and there is nothing corpus-specific to tune here.
K1 = 1.5
B = 0.75

# BM25F field weights. A query term found in the subject is stronger evidence
# than the same term in the object: "GraphARC" as a subject means the claim is
# *about* GraphARC, as an object it only mentions it. Every weight is a dyadic
# rational so the weighted lengths sum exactly — two backends holding the same
# claims in different insertion orders must produce bit-identical scores.
FIELD_WEIGHTS: dict[str, float] = {"subject": 3.0, "predicate": 1.0, "object": 2.0}

# Cosines below this are noise, not evidence. Without a floor the vector
# channel answers every query: `normalized()` would rescale the best of a set
# of meaningless similarities to a confident-looking 1.0, so "quantum
# chromodynamics" against a corpus about databases would return the databases.
# Measured against `HashingEmbedder`, where unrelated text lands around 0.05
# and a misspelling of a real term lands around 0.2. A semantic embedder has a
# different scale — this is a parameter for that reason.
MIN_SIMILARITY = 0.15


def claim_text(claim: Claim) -> str:
    """The document a claim contributes to the text channels."""
    return f"{claim.subject} {claim.predicate} {claim.object}"


def vector_texts(claim: Claim) -> tuple[str, str]:
    """The two strings a claim is embedded as: its subject, and the whole triple.

    Two rather than one because cosine dilutes: a one-word query against a
    twelve-word document scores like noise even when the word is *the* word.
    Embedding the subject on its own gives a short-to-short comparison, and
    max-pooling the two means a claim is found either by what it is about or by
    what it says. It stops at two rather than embedding every field separately
    because each one is text an injected embedder has to be paid to process.
    """
    return (claim.subject, claim_text(claim))


class ClaimIndex:
    """A snapshot of a corpus, ready to score against.

    There is deliberately no `add`: term statistics change with every write, and
    an index that is a few claims stale does not fail — it silently ranks
    wrongly. Rebuild with `ClaimIndex.from_store` after writing.
    """

    def __init__(
        self,
        claims: Iterable[Claim],
        *,
        embedder: Embedder | None = None,
        field_weights: dict[str, float] | None = None,
        k1: float = K1,
        b: float = B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.field_weights = dict(field_weights or FIELD_WEIGHTS)
        self.embedder = embedder

        by_id: dict[str, Claim] = {}
        for claim in claims:
            by_id.setdefault(claim.id, claim)
        self.by_id = by_id
        self.claims: tuple[Claim, ...] = tuple(by_id.values())

        self._postings: dict[str, dict[str, float]] = {}
        self._length: dict[str, float] = {}
        self._subjects: dict[str, list[str]] = {}
        self._objects: dict[str, list[str]] = {}

        for claim in self.claims:
            self._index_one(claim)

        # avgdl reaches 0 only when not one claim yielded a token, and then
        # `_postings` is empty too, so the division by it in `bm25` is
        # unreachable rather than guarded.
        total = sum(self._length.values())
        self.avgdl = total / len(self.claims) if self.claims else 0.0

        self.vectors: dict[str, tuple[tuple[float, ...], ...]] = {}
        if embedder is not None and self.claims:
            texts: list[str] = []
            for claim in self.claims:
                texts += vector_texts(claim)
            raw = embedder.embed(texts)
            if len(raw) != len(texts):
                raise ValueError(
                    f"embedder returned {len(raw)} vectors for {len(texts)} texts "
                    f"({len(self.claims)} claims)"
                )
            width = len(vector_texts(self.claims[0]))
            self.vectors = {
                claim.id: tuple(
                    normalize_vector(vector) for vector in raw[i * width : (i + 1) * width]
                )
                for i, claim in enumerate(self.claims)
            }

    @classmethod
    def from_store(cls, store, *, embedder: Embedder | None = None) -> ClaimIndex:
        return cls(store.all_claims(), embedder=embedder)

    def _index_one(self, claim: Claim) -> None:
        length = 0.0
        for field, weight in self.field_weights.items():
            tokens = tokenize(getattr(claim, field))
            length += weight * len(tokens)
            for token in tokens:
                bucket = self._postings.setdefault(token, {})
                bucket[claim.id] = bucket.get(claim.id, 0.0) + weight
        self._length[claim.id] = length
        self._subjects.setdefault(_normalize(claim.subject), []).append(claim.id)
        self._objects.setdefault(_normalize(claim.object), []).append(claim.id)

    def __len__(self) -> int:
        return len(self.claims)

    # --- text channels -------------------------------------------------

    def bm25(self, query: str) -> dict[str, float]:
        """Okapi BM25F over subject+predicate+object. Absent terms score nothing.

        Term order in the query drives the accumulation order, so two corpora
        holding the same claims score identically regardless of how they were
        loaded. Repeated query terms are counted once: "backend backend" is not
        twice the evidence of "backend".
        """
        n = len(self.claims)
        if n == 0:
            return {}
        scores: dict[str, float] = {}
        seen: set[str] = set()
        for term in tokenize(query):
            if term in seen:
                continue
            seen.add(term)
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for claim_id, tf in postings.items():
                dl = self._length[claim_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[claim_id] = scores.get(claim_id, 0.0) + idf * tf * (self.k1 + 1.0) / denom
        return scores

    def similarity(self, query: str, *, min_similarity: float = MIN_SIMILARITY) -> dict[str, float]:
        """Best cosine over a claim's vectors, with anything under the floor dropped.

        Empty without an embedder — that is the whole opt-in. The floor is what
        keeps this channel from answering a question the corpus has nothing to
        say about; see `MIN_SIMILARITY`.
        """
        if not self.vectors or self.embedder is None:
            return {}
        query_vector = normalize_vector(self.embedder.embed([query])[0])
        out: dict[str, float] = {}
        for claim_id, vectors in self.vectors.items():
            best = 0.0
            for vector in vectors:
                if len(vector) != len(query_vector):
                    raise ValueError(
                        f"embedder returned {len(query_vector)} dimensions for the query "
                        f"but {len(vector)} for claim {claim_id!r}"
                    )
                best = max(best, sum(a * b for a, b in zip(vector, query_vector, strict=True)))
            if best >= min_similarity:
                out[claim_id] = best
        return out

    # --- graph channel -------------------------------------------------

    def with_subject(self, entity: str) -> tuple[str, ...]:
        """Ids of claims *about* `entity` (outgoing edges start here)."""
        return tuple(self._subjects.get(_normalize(entity), ()))

    def with_object(self, entity: str) -> tuple[str, ...]:
        """Ids of claims that *point at* `entity` (incoming edges)."""
        return tuple(self._objects.get(_normalize(entity), ()))

    def neighbours(self, claim: Claim) -> tuple[str, ...]:
        """One hop from a claim, both directions, deduplicated.

        Outgoing: the claim's object read as an entity, so a fact about A whose
        object is B reaches every fact about B — the traversal the "graph" in
        the name is promising. Incoming: claims whose object is this claim's
        subject, so "who depends on A" is reachable from a question about A.
        """
        out: list[str] = []
        seen = {claim.id}
        for candidate in self.with_subject(claim.object) + self.with_object(claim.subject):
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return tuple(out)


def normalized(scores: dict[str, float]) -> dict[str, float]:
    """Rescale so the best score is 1.0, leaving relative order untouched.

    BM25 is unbounded — its scale depends on corpus size and term rarity — so
    it has to be rescaled before it can be blended with anything. The cost is
    that a normalized score is only meaningful *within one result set*: the top
    hit is 1.0 whether it was a perfect match or the least bad of a bad lot.
    Never compare one across queries, and never read it as a confidence.
    """
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0.0:
        return {}
    return {key: value / top for key, value in scores.items()}


def hybrid(
    lexical: dict[str, float],
    vector: dict[str, float],
    *,
    lexical_weight: float,
) -> dict[str, float]:
    """Blend the two text channels: BM25 rescaled, cosine left alone.

    Only the lexical side is normalized. Cosine is already bounded and already
    comparable across queries, and rescaling it would do real damage — a set of
    0.05 similarities means "nothing here matches", and normalizing turns the
    best of them into a 1.0 that outranks a genuine BM25 hit.
    """
    lex = normalized(lexical)
    vec = dict(vector)
    if not vec:
        return lex
    if not lex:
        return vec
    weight = max(0.0, min(1.0, lexical_weight))
    keys: list[str] = list(lex)
    keys += [key for key in vec if key not in lex]
    return {
        key: weight * lex.get(key, 0.0) + (1.0 - weight) * vec.get(key, 0.0) for key in keys
    }


def rank_key(score: float, claim: Claim) -> tuple[float, str, str]:
    """A total order: score, then recency, then id — descending on all three.

    Scores are rounded before comparison. Two backends can accumulate the same
    sum in a different association order and land a few ULPs apart; without the
    rounding that difference would silently reorder ties and the backends would
    render different briefs from identical claims.
    """
    return (round(score, 9), claim.observed_at, claim.id)


def top_scored(scores: dict[str, float], by_id: dict[str, Claim], limit: int) -> list[str]:
    ordered = sorted(scores, key=lambda cid: rank_key(scores[cid], by_id[cid]), reverse=True)
    return ordered[:limit]
