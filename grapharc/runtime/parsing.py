"""Tolerant JSON extraction from model replies.

Models routinely wrap JSON in markdown fences or a sentence of preamble, and a
bare `json.loads` treats that as unparseable. Where the caller fails closed
(the verifier) that silently rejects correct work; where it degrades (claim
extraction) it silently drops every claim. Both were observed against live
models.

This is parsing latitude, not correctness latitude: the JSON that comes back
must still be valid and still say what it says. Nothing here makes a malformed
or missing answer look like a good one — an unrecoverable reply returns None
and the caller's fail-closed path runs exactly as before.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _span_from(text: str, start: int) -> str | None:
    """The balanced {...} or [...] region opening at `start`, or None if unclosed."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _balanced_spans(text: str) -> list[str]:
    """Every balanced {...} or [...] region, longest first.

    Every opener is tried, not just the first one in the text. Taking only the
    first meant any bracket in the model's *prose* hijacked the span and the real
    JSON was never reached: `Based on the context [lines 3-5]: {...}` yielded the
    unparseable `[lines 3-5]` and the reply was rejected, and — worse —
    `Analysis (note [1]): {"supported": false}` yielded a perfectly valid `[1]`,
    substituting a fabricated value for the model's actual answer.

    Longest first is what makes the ranking safe. It prefers a complete structure
    over both a prose fragment that happens to parse and a nested piece of the
    answer itself, so `{"claims": [{...}]}` returns the whole object rather than
    the inner list.
    """
    spans = [
        span
        for i, ch in enumerate(text)
        if ch in "{[" and (span := _span_from(text, i)) is not None
    ]
    return sorted(spans, key=len, reverse=True)


def extract_json(content: Any) -> Any | None:
    """Best-effort JSON from a model reply. None when nothing valid is found."""
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return None

    # Whole reply first, then the fence, then balanced spans: the earlier a
    # candidate is, the more of the model's reply it accounts for, so a fenced
    # top-level array still wins over any span found inside it.
    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.extend(_balanced_spans(text))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
