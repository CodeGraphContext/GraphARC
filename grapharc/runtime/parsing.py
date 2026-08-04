"""Tolerant JSON extraction from model replies.

Models routinely wrap JSON in markdown fences or a sentence of preamble, and a
bare `json.loads` treats that as unparseable. Where the caller fails closed
(the verifier) that silently rejects correct work; where it degrades (claim
extraction) it silently drops every claim. Both were observed against live
models.

Reasoning models add a second failure shape: a `<think>…</think>` block that
*contains JSON* — drafts, worked examples, half-answers — ahead of the real
reply. The visible text after stripping those blocks is scanned first, and the
original text is scanned only when the visible text yields nothing parseable.
Second-class rather than deleted, because some chat templates put the *only*
copy of the answer inside the block; first-class would resurrect the bug this
ordering exists to fix — a longer draft inside the think block outranking the
real answer outside it.

This is parsing latitude, not correctness latitude: the JSON that comes back
must still be valid and still say what it says. Nothing here makes a malformed
or missing answer look like a good one — an unrecoverable reply returns None
and the caller's fail-closed path runs exactly as before. The one repair
performed (trailing commas) only ever runs on a candidate that already failed
to parse, and a trailing comma is never valid JSON, so no valid document can
be rewritten.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: The tag names reasoning models wrap their scratch work in. Matched as a
#: pair with a backreference so `<think>` cannot be closed by `</reasoning>`.
_THINK_NAMES = "think|thinking|thought|reasoning|reason"
_THINK = re.compile(
    rf"<({_THINK_NAMES})\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)
#: Some chat templates strip the opening tag and the visible reply *starts*
#: with a bare closer; and a truncated reply can open a block it never closes.
_ORPHAN_CLOSE = re.compile(rf"^\s*</({_THINK_NAMES})\s*>", re.IGNORECASE)
_ORPHAN_OPEN = re.compile(
    rf"<({_THINK_NAMES})\b[^>]*>(?!.*</\1\s*>).*\Z", re.DOTALL | re.IGNORECASE
)


def _without_reasoning(text: str) -> str:
    """The reply minus its reasoning blocks — what the model meant to show."""
    visible = _THINK.sub("", text)
    visible = _ORPHAN_CLOSE.sub("", visible)
    visible = _ORPHAN_OPEN.sub("", visible)
    return visible.strip()


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
    """Every balanced {...} or [...] region: objects longest-first, then arrays
    longest-first.

    Every opener is tried, not just the first one in the text. Taking only the
    first meant any bracket in the model's *prose* hijacked the span and the real
    JSON was never reached: `Based on the context [lines 3-5]: {...}` yielded the
    unparseable `[lines 3-5]` and the reply was rejected, and — worse —
    `Analysis (note [1]): {"supported": false}` yielded a perfectly valid `[1]`,
    substituting a fabricated value for the model's actual answer.

    Length alone was not sufficient to make the ranking safe. It does prefer a
    complete structure over a nested piece of the answer itself — `{"claims":
    [{...}]}` returns the whole object rather than the inner list — but a prose
    fragment *longer* than the answer still won: a citation list like
    `[101, 205, 309, ...]` outranked the real `{"supported": false}` verdict.
    Prose brackets are square (`[1]`, `[lines 3-5]`, citation lists) while the
    model's answer is a top-level object for every shipped caller, so object
    spans rank ahead of array spans, each group longest-first. An array-only
    reply still parses whole or fenced before any span is tried, so top-level
    arrays remain reachable.
    """
    spans = [
        span
        for i, ch in enumerate(text)
        if ch in "{[" and (span := _span_from(text, i)) is not None
    ]
    return sorted(spans, key=lambda span: (span[0] != "{", -len(span)))


def _strip_trailing_commas(candidate: str) -> str:
    """Remove commas whose next non-whitespace char (outside strings) closes a
    scope — the one syntax error local models make that has exactly one honest
    reading. Only ever called on text that already failed `json.loads`."""
    out: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch == ",":
            rest = candidate[i + 1 :].lstrip()
            if rest[:1] in ("}", "]"):
                continue  # the comma before a closer: drop it
        out.append(ch)
    return "".join(out)


def _candidates(text: str) -> list[str]:
    """Whole reply first, then every fence in order, then balanced spans.

    The earlier a candidate is, the more of the model's reply it accounts for,
    so a fenced top-level array still wins over any span found inside it.
    Every fence is tried, not only the first: a model that shows a draft in
    one fence and the answer in the next must not be stranded on the draft.
    """
    found = [text]
    for fence in _FENCE.finditer(text):
        found.append(fence.group(1).strip())
    found.extend(_balanced_spans(text))
    return found


def extract_json(content: Any) -> Any | None:
    """Best-effort JSON from a model reply. None when nothing valid is found."""
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return None

    # Tier 1: the whole reply, untouched. A reply that is already valid JSON
    # must never be rewritten — if it contains a reasoning tag, that tag lives
    # inside a string and is data, not scratch work.
    # Tier 2: the visible text (reasoning blocks stripped), whole/fences/spans
    # — so a draft inside <think> can never outrank the shown answer.
    # Tier 3: fences and spans of the original, for the reply whose only copy
    # of the answer sits inside the reasoning block.
    visible = _without_reasoning(text)
    candidates = [text]
    if visible and visible != text:
        candidates.extend(_candidates(visible))
    candidates.extend(_candidates(text)[1:])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        repaired = _strip_trailing_commas(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None
