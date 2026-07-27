"""Stage 5 gate: the verifier catches fluent-but-unsupported answers and still
accepts correct ones — false accepts and false rejects tracked separately."""

import json

import pytest

from grapharc.examples.stage5_verifier import DEMO_SOURCE, build_stage5
from grapharc.runtime.verify import evaluate_verdicts, verify_claim
from grapharc.testing import ScriptedChatModel

CORRECT = {
    "text": "GraphARC enforces typed state contracts",
    "citation": "typed state contracts",
}
FABRICATED = {
    # Fluent, confident, and citing text that does not exist in the source.
    "text": "GraphARC was benchmarked at 10x faster than raw LangGraph",
    "citation": "benchmarked at 10x faster",
}
MISLEADING = {
    # Citation exists verbatim, but does not support the claim — only the
    # reviewing model can catch this one.
    "text": "Budgets make graphs run faster",
    "citation": "Budgets put hard ceilings on iterations, tokens, and wall-clock time.",
}

GROUND_TRUTH = {
    CORRECT["text"]: True,
    FABRICATED["text"]: False,
    MISLEADING["text"]: False,
}


def test_gate_verifier_catches_unsupported_and_accepts_correct(trace):
    author = ScriptedChatModel(
        responses=[json.dumps({"claims": [CORRECT, FABRICATED, MISLEADING]})]
    )
    # Reviewer is only consulted for anchor-passing claims: CORRECT, MISLEADING.
    reviewer = ScriptedChatModel(
        responses=[
            json.dumps({"supported": True, "reason": "evidence states exactly this"}),
            json.dumps({"supported": False, "reason": "ceilings bound work; nothing about speed"}),
        ]
    )
    compiled = build_stage5(author, reviewer, trace=trace)
    result = compiled.invoke({"source_text": DEMO_SOURCE})

    assert result["accepted"] == [CORRECT["text"]]
    assert sorted(result["rejected"]) == sorted([FABRICATED["text"], MISLEADING["text"]])

    stats = evaluate_verdicts(result["verdicts"], GROUND_TRUTH)
    assert stats.false_accepts == 0
    assert stats.false_rejects == 0
    assert stats.true_accepts == 1
    assert stats.true_rejects == 2

    # The fabricated claim was rejected by the deterministic anchor without
    # consulting the model at all.
    fabricated = next(v for v in result["verdicts"] if v.claim_text == FABRICATED["text"])
    assert fabricated.anchor_ok is False
    assert fabricated.model_accepted is None
    assert reviewer.call_count == 2


def test_gate_reviewer_gets_fresh_context_not_the_authors_conversation(trace):
    """Fresh context means: the claim, its evidence, and a mechanically
    extracted source window — never the author's conversation."""
    author = ScriptedChatModel(responses=[json.dumps({"claims": [CORRECT]})])
    reviewer = ScriptedChatModel(
        responses=[json.dumps({"supported": True, "reason": "ok"})]
    )
    compiled = build_stage5(author, reviewer, trace=trace)
    compiled.invoke({"source_text": DEMO_SOURCE})

    (reviewer_call,) = reviewer.calls
    prompt = str(reviewer_call[0].content)
    assert CORRECT["text"] in prompt
    assert CORRECT["citation"] in prompt
    # The author's instructions and its message history never reach the reviewer.
    assert "Extract factual claims" not in prompt
    assert len(reviewer_call) == 1
    # Source exposure is a bounded window around the match, not the whole corpus.
    assert len(prompt) < len(DEMO_SOURCE) + 1500


def test_reviewer_source_window_is_bounded_for_large_sources():
    filler = "irrelevant background. " * 500
    citation = "typed state contracts are enforced"
    source = f"{filler}GraphARC: {citation}. {filler}"
    reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "ok"}'])
    verify_claim(reviewer, text="claim", citation=citation, source_text=source)

    (call,) = reviewer.calls
    prompt = str(call[0].content)
    assert citation in prompt
    assert len(prompt) < len(source) / 2  # a window, not the whole document


def test_same_model_instance_for_author_and_reviewer_is_refused():
    model = ScriptedChatModel(responses=["x"])
    with pytest.raises(ValueError, match="different model instances"):
        build_stage5(model, model)


def test_unparseable_reviewer_reply_fails_closed():
    reviewer = ScriptedChatModel(responses=["sounds good to me!"])
    verdict = verify_claim(
        reviewer,
        text=CORRECT["text"],
        citation=CORRECT["citation"],
        source_text=DEMO_SOURCE,
    )
    assert verdict.accepted is False
    assert "failing closed" in verdict.reason


@pytest.mark.parametrize("value", ['"false"', '"no"', '"true"', "1", "null"])
def test_non_boolean_verdict_fails_closed(value):
    """A quoted "false" is truthy in Python — only a real JSON boolean accepts."""
    reviewer = ScriptedChatModel(responses=[f'{{"supported": {value}, "reason": "x"}}'])
    verdict = verify_claim(
        reviewer,
        text=CORRECT["text"],
        citation=CORRECT["citation"],
        source_text=DEMO_SOURCE,
    )
    assert verdict.accepted is False
    assert "failing closed" in verdict.reason


def test_trivial_citation_is_rejected_before_the_model_is_consulted():
    reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "sure"}'])
    for citation in (" ", "", "the"):
        verdict = verify_claim(
            reviewer, text="anything", citation=citation, source_text=DEMO_SOURCE
        )
        assert verdict.accepted is False
        assert verdict.anchor_ok is False
        assert "too short" in verdict.reason
    assert reviewer.call_count == 0  # never consulted


def test_line_wrapped_source_does_not_false_reject_a_correct_quote():
    """Found by a live run: real sources are line-wrapped, so a model quoting a
    sentence perfectly writes a space where the source has a newline. Rejecting
    that is a false reject on a formatting artifact."""
    source = (
        "GraphARC is a toolkit built on LangGraph. Every node declares\n"
        "which state fields it may write, and an undeclared write raises."
    )
    citation = "Every node declares which state fields it may write"
    assert citation not in source  # exact matching would reject this

    reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "ok"}'])
    verdict = verify_claim(
        reviewer,
        text="Nodes declare their writes",
        citation=citation,
        source_text=source,
    )
    assert verdict.anchor_ok is True
    assert verdict.accepted is True


def test_whitespace_latitude_does_not_extend_to_paraphrase():
    """Whitespace is the only latitude: every other character must still match."""
    source = "Budgets place hard ceilings on iterations and tokens."
    reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "ok"}'])
    for fake in (
        "Budgets place soft ceilings on iterations and tokens",
        "Budgets  place  hard  limits  on  iterations",
        "Budgets place hard ceilings on iteration and tokens",
    ):
        verdict = verify_claim(
            reviewer, text="claim", citation=fake, source_text=source
        )
        assert verdict.anchor_ok is False, fake
    assert reviewer.call_count == 0


def test_gate_quote_mined_negation_reaches_the_reviewer_with_context():
    """A quote lifted out of a negated sentence exists verbatim, so the anchor
    passes — the reviewer must be shown the surrounding source to catch it."""
    source = (
        "GraphARC was never benchmarked at 10x faster than raw LangGraph; "
        "no such measurement exists."
    )
    citation = "benchmarked at 10x faster than raw LangGraph"
    reviewer = ScriptedChatModel(
        responses=['{"supported": false, "reason": "context negates the quote"}']
    )
    verdict = verify_claim(
        reviewer,
        text="GraphARC was benchmarked at 10x faster than raw LangGraph",
        citation=citation,
        source_text=source,
    )
    assert verdict.anchor_ok is True  # the quote really is in the source
    assert verdict.accepted is False

    (call,) = reviewer.calls
    prompt = str(call[0].content)
    assert "never" in prompt  # the negation is visible to the reviewer
    assert "Surrounding source context" in prompt
