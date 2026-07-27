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


def _balanced_span(text: str) -> str | None:
    """The first balanced {...} or [...] region, ignoring braces inside strings."""
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None
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


def extract_json(content: Any) -> Any | None:
    """Best-effort JSON from a model reply. None when nothing valid is found."""
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    span = _balanced_span(text)
    if span:
        candidates.append(span)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
