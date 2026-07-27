"""Tolerant JSON extraction — found necessary by live model runs.

Real models fence their JSON and add preamble. A bare `json.loads` turns that
into "unparseable", which fails closed on correct work. Parsing latitude is not
correctness latitude: an actually-missing or malformed answer still returns
None so the caller's fail-closed path runs.
"""

import pytest

from grapharc.runtime.parsing import extract_json


@pytest.mark.parametrize(
    "reply",
    [
        '{"supported": true, "reason": "ok"}',
        '```json\n{"supported": true, "reason": "ok"}\n```',
        '```\n{"supported": true, "reason": "ok"}\n```',
        'Here is my verdict:\n\n{"supported": true, "reason": "ok"}',
        'Sure!\n```json\n{"supported": true, "reason": "ok"}\n```\nHope that helps.',
    ],
)
def test_fenced_and_prefixed_json_is_recovered(reply):
    assert extract_json(reply) == {"supported": True, "reason": "ok"}


def test_nested_braces_and_strings_survive():
    payload = '{"claims": [{"text": "uses {braces}", "citation": "a \\"quote\\" here"}]}'
    parsed = extract_json(f"prose\n```json\n{payload}\n```")
    assert parsed["claims"][0]["text"] == "uses {braces}"
    assert parsed["claims"][0]["citation"] == 'a "quote" here'


def test_top_level_arrays_work():
    assert extract_json('```json\n[1, 2, 3]\n```') == [1, 2, 3]


@pytest.mark.parametrize(
    "reply",
    ["", "   ", "no json at all", "{unclosed: ", '{"a": }', "```json\n{oops\n```"],
)
def test_unrecoverable_replies_return_none(reply):
    """None is what makes the caller fail closed — this must not be lenient."""
    assert extract_json(reply) is None


def test_verifier_still_fails_closed_on_junk():
    from grapharc.runtime.verify import verify_claim
    from grapharc.testing import ScriptedChatModel

    source = "Budgets place hard ceilings on iterations and tokens."
    reviewer = ScriptedChatModel(responses=["I think it's fine, honestly!"])
    verdict = verify_claim(
        reviewer,
        text="claim",
        citation="Budgets place hard ceilings",
        source_text=source,
    )
    assert verdict.accepted is False
    assert "failing closed" in verdict.reason


def test_verifier_accepts_a_fenced_reply():
    from grapharc.runtime.verify import verify_claim
    from grapharc.testing import ScriptedChatModel

    source = "Budgets place hard ceilings on iterations and tokens."
    reviewer = ScriptedChatModel(
        responses=['```json\n{"supported": true, "reason": "stated verbatim"}\n```']
    )
    verdict = verify_claim(
        reviewer,
        text="Budgets bound work",
        citation="Budgets place hard ceilings",
        source_text=source,
    )
    assert verdict.accepted is True
    assert verdict.reason == "stated verbatim"
