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


@pytest.mark.parametrize(
    "reply",
    [
        'Based on the context [lines 3-5]: {"supported": true, "reason": "ok"}',
        'The quote "{" is fine. {"supported": true, "reason": "ok"}',
        'Analysis (note [1]): {"supported": true, "reason": "ok"}',
        'See table[0] and figure{2}: {"supported": true, "reason": "ok"}',
    ],
)
def test_a_bracket_in_the_prose_does_not_hijack_the_answer(reply):
    """Only the *first* opener used to be tried, so any bracket in the preamble
    captured the span and the real JSON was never reached."""
    assert extract_json(reply) == {"supported": True, "reason": "ok"}


def test_a_prose_bracket_is_never_substituted_for_the_answer():
    """The worst shape of the bug above: `[1]` is valid JSON, so it was returned
    as the model's verdict. A wrong answer is worse than no answer."""
    reply = 'Analysis (note [1]): {"supported": false, "reason": "negated"}'
    assert extract_json(reply) == {"supported": False, "reason": "negated"}


@pytest.mark.parametrize("supported", [False, True])
def test_a_long_prose_bracket_does_not_outrank_the_answer(supported):
    """Length-only ranking let a citation list *longer* than the verdict win, so
    the model's actual answer was never returned. Objects rank ahead of arrays:
    prose brackets are square, and every shipped caller expects an object."""
    verdict = "false" if supported is False else "true"
    reply = (
        "Verified against source lines [101, 205, 309, 412, 518, 622, 733, 848]: "
        f'{{"supported": {verdict}}}'
    )
    assert extract_json(reply) == {"supported": supported}


def test_an_array_answer_survives_a_brace_in_the_prose():
    """Ranking objects first must not strand array answers: `figure{2}` does not
    parse, so the balanced-span fallback still reaches the unfenced array."""
    reply = "Matching ids, see figure{2}: [101, 205, 309]"
    assert extract_json(reply) == [101, 205, 309]


def test_the_outermost_structure_wins_over_a_nested_piece_of_it():
    reply = 'Verdict: {"claims": [{"text": "a"}], "n": 1}'
    assert extract_json(reply) == {"claims": [{"text": "a"}], "n": 1}


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


# -- reasoning blocks, multiple fences, and the one permitted repair ---------


def test_a_think_block_is_stripped_before_extraction():
    from grapharc.runtime.parsing import extract_json

    reply = '<think>I should answer with an object.</think>{"answer": 1}'
    assert extract_json(reply) == {"answer": 1}


def test_a_longer_draft_inside_think_does_not_beat_the_real_answer():
    from grapharc.runtime.parsing import extract_json

    # The draft is longer than the answer; before think-stripping it won the
    # objects-longest-first ranking and the model's actual reply was discarded.
    draft = '{"nodes": [{"name": "a"}, {"name": "b"}, {"name": "c"}], "draft": true}'
    reply = f"<think>maybe {draft}</think>\n" '{"nodes": [{"name": "final"}]}'
    assert extract_json(reply) == {"nodes": [{"name": "final"}]}


def test_a_fenced_draft_inside_think_does_not_win_over_the_visible_answer():
    from grapharc.runtime.parsing import extract_json

    reply = (
        '<think>```json\n{"draft": true}\n```</think>\n'
        'Here you go: {"final": true}'
    )
    assert extract_json(reply) == {"final": True}


def test_a_reply_that_is_entirely_think_block_is_still_scanned():
    from grapharc.runtime.parsing import extract_json

    assert extract_json('<think>{"only": "copy"}</think>') == {"only": "copy"}


def test_an_orphan_closing_think_tag_is_ignored():
    from grapharc.runtime.parsing import extract_json

    assert extract_json('</think>\n{"answer": 2}') == {"answer": 2}


def test_an_unclosed_think_block_falls_back_to_the_original_text():
    from grapharc.runtime.parsing import extract_json

    # Truncated reply: the block opens and never closes; the answer inside is
    # still reachable through the fallback tier.
    assert extract_json('<think>so {"answer": 3} is right') == {"answer": 3}


def test_a_think_tag_inside_a_json_string_survives():
    from grapharc.runtime.parsing import extract_json

    reply = '{"note": "models emit <think>stuff</think> sometimes"}'
    assert extract_json(reply) == {"note": "models emit <think>stuff</think> sometimes"}


def test_a_junk_first_fence_does_not_strand_a_later_valid_fence():
    from grapharc.runtime.parsing import extract_json

    reply = (
        "```\nnot json at all\n```\n"
        'and the answer:\n```json\n{"answer": 4}\n```'
    )
    assert extract_json(reply) == {"answer": 4}


def test_a_trailing_comma_is_repaired_only_when_nothing_parses_without_it():
    from grapharc.runtime.parsing import extract_json

    assert extract_json('{"nodes": [{"name": "a"},], "edges": [],}') == {
        "nodes": [{"name": "a"}],
        "edges": [],
    }


def test_repair_never_touches_comma_bracket_sequences_inside_strings():
    from grapharc.runtime.parsing import extract_json

    # The ",]" inside the string is data; only the structural trailing comma
    # outside it may be removed.
    reply = '{"text": "a,] b", "items": [1, 2,]}'
    assert extract_json(reply) == {"text": "a,] b", "items": [1, 2]}
