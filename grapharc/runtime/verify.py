"""Independent verification against reality (Stage 5 discipline).

A verifier is another fallible node, so it gets three constraints:
1. **A different model** than the one that produced the work — same-model
   agents validate each other's errors ("correlated agreement").
2. **Fresh context** — only the claim and the evidence, never the producing
   agent's conversation.
3. **A deterministic anchor** — the citation must literally exist in the
   source; if it doesn't, the claim is rejected no matter how supportive the
   reviewing model feels. Models judge semantics; reality judges existence.

The anchor proves a quote exists, not that it means what the claim says — a
fragment lifted out of a negated sentence exists verbatim. So the reviewer is
shown a mechanically extracted window around the match (not the author's
conversation, so context isolation holds) and can see the negation.

Every ambiguity resolves toward rejection: an unparseable reply, a
non-boolean verdict, or a trivial citation all fail closed. False accepts and
false rejects are tracked separately — a verifier that rejects everything
looks safe and is useless.
"""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from grapharc.runtime.parsing import extract_json

MIN_CITATION_CHARS = 12
CONTEXT_WINDOW_CHARS = 240


class Verdict(BaseModel):
    claim_text: str
    citation: str
    accepted: bool
    anchor_ok: bool
    model_accepted: bool | None = None  # None: never consulted (anchor already failed)
    reason: str


def _find_span(source_text: str, citation: str) -> re.Match[str] | None:
    """Locate a citation in the source, tolerant of whitespace only.

    Real sources are line-wrapped, so a model that quotes a sentence perfectly
    still writes `declares which` where the source holds `declares\\nwhich`.
    Exact matching rejects that — correct work refused on a formatting artifact,
    which is the false-reject failure mode this verifier exists to avoid.

    Whitespace is the *only* latitude given: every non-space character must
    still match exactly, so a paraphrase or an invented quote is still caught.
    """
    tokens = citation.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    return re.search(pattern, source_text)


def _context_window(source_text: str, span: re.Match[str] | None) -> str:
    """The source span around the citation — mechanical extraction, no author context."""
    if span is None:
        return ""
    start = max(0, span.start() - CONTEXT_WINDOW_CHARS)
    end = min(len(source_text), span.end() + CONTEXT_WINDOW_CHARS)
    return source_text[start:end]


class VerifierStats(BaseModel):
    true_accepts: int = 0
    true_rejects: int = 0
    false_accepts: int = 0
    false_rejects: int = 0


def verify_claim(
    reviewer: BaseChatModel,
    *,
    text: str,
    citation: str,
    source_text: str,
) -> Verdict:
    """Verify one claim with a fresh-context reviewer plus a deterministic anchor."""
    citation = citation.strip()
    if len(citation) < MIN_CITATION_CHARS:
        return Verdict(
            claim_text=text,
            citation=citation,
            accepted=False,
            anchor_ok=False,
            reason=(
                f"citation is too short to anchor anything "
                f"({len(citation)} < {MIN_CITATION_CHARS} chars)"
            ),
        )
    span = _find_span(source_text, citation)
    if span is None:
        return Verdict(
            claim_text=text,
            citation=citation,
            accepted=False,
            anchor_ok=False,
            reason="citation does not exist in the source (deterministic anchor)",
        )

    prompt = (
        "You are verifying a single claim against its cited evidence. "
        "Judge ONLY whether the evidence supports the claim. The surrounding "
        "source context is included: if it negates, reverses, or otherwise "
        "contradicts what the quote alone implies, the claim is NOT supported.\n"
        f"Claim: {text}\n"
        f"Evidence (verbatim from source): {citation}\n"
        f"Surrounding source context: {_context_window(source_text, span)}\n"
        'Reply with JSON: {"supported": true|false, "reason": "..."}'
    )
    message = reviewer.invoke([HumanMessage(content=prompt)])
    parsed = extract_json(message.content)
    try:
        raw_supported = parsed["supported"]
        reason = str(parsed.get("reason", ""))
    except (KeyError, TypeError):
        supported, reason = False, "reviewer reply unparseable — failing closed"
    else:
        # A quoted "false" is truthy in Python; only a real JSON boolean counts.
        if isinstance(raw_supported, bool):
            supported = raw_supported
        else:
            supported = False
            reason = "reviewer 'supported' was not a JSON boolean — failing closed"
    return Verdict(
        claim_text=text,
        citation=citation,
        accepted=supported,
        anchor_ok=True,
        model_accepted=supported,
        reason=reason,
    )


def evaluate_verdicts(
    verdicts: list[Verdict], ground_truth: dict[str, bool]
) -> VerifierStats:
    """Score verdicts against known ground truth (claim_text -> is actually valid)."""
    stats = VerifierStats()
    for v in verdicts:
        truth = ground_truth[v.claim_text]
        if v.accepted and truth:
            stats.true_accepts += 1
        elif not v.accepted and not truth:
            stats.true_rejects += 1
        elif v.accepted and not truth:
            stats.false_accepts += 1
        else:
            stats.false_rejects += 1
    return stats
