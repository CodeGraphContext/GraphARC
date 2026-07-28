"""Tokenization shared by every scoring channel.

One tokenizer, used for the inverted index, the query, and the hashing
embedder. If the query were folded differently from the documents, a term that
is present would score zero — the kind of bug that looks like "retrieval is
just bad" rather than like a defect.

This is deliberately *not* `store._normalize`. That function answers "are these
two strings the same entity?" and collapses a whole string to one key; this one
answers "what words are in this text?" and returns many. They agree on the
Unicode handling (NFKC, casefold) because disagreeing there is how "Neo4j" the
entity and "neo4j" the search term stop matching.
"""

from __future__ import annotations

import re
import unicodedata

_SPLIT = re.compile(r"[\W_]+", re.UNICODE)

# Suffixes that are part of the word, not a plural marker. Without these,
# "analysis" folds to "analysi" and "status" to "statu" — harmless for matching
# (queries fold the same way) but they stop matching the singular forms people
# actually type.
_NOT_PLURAL = ("ss", "us", "is", "os")


def fold_plural(token: str) -> str:
    """A minimal, deliberately dumb plural fold.

    Not a stemmer and not linguistically correct — "series" becomes "serie".
    Correctness is not what it buys: it is applied identically to documents and
    to queries, so the only thing that matters is that both sides land on the
    same string. It is here because a memory store accumulates claims like
    "uses backends" that a user then searches for as "backend".
    """
    if len(token) >= 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) >= 4 and token.endswith("s") and not token.endswith(_NOT_PLURAL):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Fold to NFKC + casefold, split on non-word characters, fold plurals.

    Order is preserved and duplicates are kept: BM25 needs term frequencies.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return [fold_plural(tok) for tok in _SPLIT.split(folded) if tok]
