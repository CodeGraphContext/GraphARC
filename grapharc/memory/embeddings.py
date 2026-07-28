"""The vector channel: a Protocol to inject a real embedder, and a local fallback.

GraphARC does not depend on an embedding model and is not going to start. So
this module defines the one-method interface a real embedder has to satisfy and
ships `HashingEmbedder`, which needs nothing but the standard library.

**What the fallback is, precisely.** It is a hashed bag of word tokens and
character n-grams, L2-normalized. It is *lexical*, not semantic: it cannot
relate "car" to "automobile", and anyone who reads "embeddings" and assumes it
can will be wrong. What it buys over BM25 is character-level overlap — a
misspelled or truncated query ("langraph", "provenanc") still lands near the
right claim, where exact term matching scores a flat zero. For meaning, inject
a real embedder.

Hashing uses BLAKE2b rather than `hash()`: Python randomizes string hashing per
process, so a `hash()`-based vector would rank differently in the next process
— unacceptable for a store whose whole point is outliving the process.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from grapharc.memory.text import tokenize


@runtime_checkable
class Embedder(Protocol):
    """What a vector backend must provide.

    One method, batched. Batching is not cosmetic — a hosted embedder charges
    per request, and the index embeds a whole corpus at once.

    Implementations must return one vector per input text, in order, all of the
    same dimension. Vectors need not be normalized; the index normalizes.
    `isinstance` against this protocol checks the method *name* only.
    """

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]: ...


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize, mapping the zero vector to itself rather than to NaN."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(float(value) for value in vector)
    return tuple(float(value) / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1] against float drift."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    left_n = normalize_vector(left)
    right_n = normalize_vector(right)
    dot = sum(a * b for a, b in zip(left_n, right_n, strict=True))
    return max(-1.0, min(1.0, dot))


class HashingEmbedder:
    """Deterministic local embeddings: hashed word tokens + character n-grams.

    Deterministic across processes and machines, which is what makes ranking
    reproducible and lets the two backends be compared byte for byte.

    Collisions are real and unfixable at this size: two unrelated features can
    land on the same dimension with the same sign, inventing a little
    similarity. `dim` trades memory for how often that happens. This is a
    fallback, not a good embedder.
    """

    def __init__(self, *, dim: int = 256, char_ngram: int = 3, char_weight: float = 0.35) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        # 0 disables the character channel, leaving a pure hashed bag of words —
        # which is BM25 without the IDF, so there is rarely a reason to.
        self.char_ngram = char_ngram
        self.char_weight = char_weight

    def _features(self, text: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        tokens = tokenize(text)
        for token in tokens:
            key = f"w:{token}"
            counts[key] = counts.get(key, 0.0) + 1.0
        if self.char_ngram > 0 and tokens:
            # Padded so that word starts and ends are themselves features:
            # "graph" and "graphs" share more n-grams than their raw strings do.
            padded = " " + " ".join(tokens) + " "
            size = self.char_ngram
            for i in range(len(padded) - size + 1):
                key = f"c:{padded[i : i + size]}"
                counts[key] = counts.get(key, 0.0) + self.char_weight
        return counts

    def _project(self, features: dict[str, float]) -> tuple[float, ...]:
        vector = [0.0] * self.dim
        for feature, weight in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            # A sign bit from the same digest keeps unrelated features that
            # collide on a dimension from always reinforcing each other.
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[value % self.dim] += sign * weight
        return normalize_vector(vector)

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self._project(self._features(text)) for text in texts]
