"""The slim proposal shape and the per-backend structured-output seam.

`Subgraph`'s own JSON schema is wrong for grammar-constrained local decoding
(recursive, all-fields-required, docstring-laden), so backends that declare
`reliable_structured_output = False` get a text path asking for a three-key
shape instead — and everything read that way is re-validated through the real
constructors before admission sees it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from grapharc.planner.proposal import (
    PROPOSAL_EXAMPLE,
    TEXT_FORMAT_INSTRUCTIONS,
    PlannerNode,
    PlanProposal,
    Subgraph,
)
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel


def test_a_slim_proposal_converts_to_the_real_subgraph():
    slim = PlanProposal.model_validate(
        {
            "nodes": [{"name": "a"}, {"name": "b", "kind": "worker"}],
            "edges": [[START, "a"], ["a", "b"], ["b", END]],
            "rationale": "why",
        }
    )
    proposal = slim.to_subgraph()
    assert isinstance(proposal, Subgraph)
    assert [n.name for n in proposal.nodes] == ["a", "b"]
    assert proposal.nodes[0].kind == "a"  # kind defaults to name, as ever
    assert proposal.nodes[1].kind == "worker"
    assert proposal.edges[0].source == START
    assert proposal.rationale == "why"


def test_slim_edges_accept_pair_object_and_from_to_forms():
    slim = PlanProposal.model_validate(
        {
            "nodes": [{"name": "a"}, {"name": "b"}],
            "edges": [
                ["__start__", "a"],
                {"source": "a", "target": "b"},
                {"from": "b", "to": "__end__"},
            ],
        }
    )
    rendered = [e.render() for e in slim.to_subgraph().edges]
    assert rendered == ["__start__ -> a", "a -> b", "b -> __end__"]


def test_extra_model_invented_fields_are_ignored_not_fatal():
    slim = PlanProposal.model_validate(
        {
            "nodes": [{"name": "a", "confidence": 0.9}],
            "edges": [["__start__", "a"]],
            "thoughts": "I am a helpful model",
        }
    )
    assert [n.name for n in slim.to_subgraph().nodes] == ["a"]


def test_a_bad_name_in_a_slim_proposal_still_fails_with_the_named_reason():
    slim = PlanProposal.model_validate(
        {"nodes": [{"name": "__start__"}], "edges": []}
    )
    with pytest.raises(ValidationError, match="reserved"):
        slim.to_subgraph()


def _planner(model, **kwargs) -> PlannerNode:
    return PlannerNode(model, catalog={"a": "does a", "b": "does b"}, **kwargs)


def test_a_backend_that_disclaims_structured_output_gets_the_text_path():
    class Disclaiming(ScriptedChatModel):
        reliable_structured_output: bool = False

    reply = {"nodes": [{"name": "a"}], "edges": [["__start__", "a"], ["a", "__end__"]]}
    model = Disclaiming(responses=[json.dumps(reply)])
    outcome = _planner(model).propose("go")
    assert outcome.structured is False
    assert outcome.proposal is not None
    assert [n.name for n in outcome.proposal.nodes] == ["a"]


def test_the_text_path_system_prompt_carries_the_worked_example():
    planner = _planner(ScriptedChatModel(responses=["{}"]))
    messages = planner._messages("go", "", structured=False)
    assert TEXT_FORMAT_INSTRUCTIONS in messages[0].content
    assert PROPOSAL_EXAMPLE in messages[0].content
    # And the structured path carries none of it: the schema is the contract.
    structured = planner._messages("go", "", structured=True)
    assert TEXT_FORMAT_INSTRUCTIONS not in structured[0].content


def test_the_operator_override_beats_the_backend_attribute():
    class Disclaiming(ScriptedChatModel):
        reliable_structured_output: bool = False

    # structured=True on a disclaiming backend: with_structured_output raises
    # NotImplementedError on the scripted double, so it still lands on text —
    # but the decision path went through the override, not the attribute.
    planner = _planner(Disclaiming(responses=["{}"]), structured=True)
    _, structured = planner._runnable()
    assert structured is False  # scripted model cannot bind tools

    # structured=False on a capable-looking backend skips even the attempt.
    planner = _planner(ScriptedChatModel(responses=["{}"]), structured=False)
    _, structured = planner._runnable()
    assert structured is False


def test_a_full_subgraph_reply_on_the_text_path_keeps_its_args():
    reply = json.dumps(
        {
            "nodes": [{"name": "a", "args": {"path": "x"}}],
            "edges": [{"source": "__start__", "target": "a"}],
        }
    )
    outcome = _planner(ScriptedChatModel(responses=[reply])).propose("go")
    assert outcome.proposal is not None
    assert outcome.proposal.nodes[0].args == {"path": "x"}


def test_ollama_declares_the_seam():
    from grapharc.gateway.ollama import OllamaChatModel

    assert OllamaChatModel.reliable_structured_output is False


def test_the_parse_failure_note_shows_the_offending_reply_and_an_example():
    from grapharc.examples.plan_incident import build_loop

    # Two junk replies then a valid plan: the loop must recover, and the note
    # it fed back must have carried the snippet and the worked example.
    junk = "I am afraid I cannot do that." * 3
    good = json.dumps(
        {
            "nodes": [{"name": "triage"}, {"name": "patch"}, {"name": "verify"}],
            "edges": [
                ["__start__", "triage"],
                ["triage", "patch"],
                ["patch", "verify"],
                ["verify", "__end__"],
            ],
        }
    )
    model = ScriptedChatModel(responses=[junk, good])
    captured: list[str] = []

    loop = build_loop(model)
    original = loop.planner.propose

    def spying_propose(task, ctx=None, *, feedback=""):
        captured.append(task)
        return original(task, ctx, feedback=feedback)

    loop.planner.propose = spying_propose
    result = loop.run("fix it")
    assert result.succeeded
    # The second round's task carried the snippet of the junk reply plus the shape.
    retry_prompt = captured[1]
    assert "could not be used as a proposal" in retry_prompt
    assert "I am afraid I cannot do that." in retry_prompt
    assert '"nodes"' in retry_prompt and '"edges"' in retry_prompt


def test_the_parse_failure_snippet_is_truncated():
    from grapharc.examples.plan_incident import build_loop
    from grapharc.planner.loop import _SNIPPET_LIMIT

    junk = "x" * (_SNIPPET_LIMIT * 3)
    good = json.dumps(
        {
            "nodes": [{"name": "triage"}, {"name": "patch"}, {"name": "verify"}],
            "edges": [
                ["__start__", "triage"],
                ["triage", "patch"],
                ["patch", "verify"],
                ["verify", "__end__"],
            ],
        }
    )
    model = ScriptedChatModel(responses=[junk, good])
    captured: list[str] = []
    loop = build_loop(model)
    original = loop.planner.propose

    def spying_propose(task, ctx=None, *, feedback=""):
        captured.append(task)
        return original(task, ctx, feedback=feedback)

    loop.planner.propose = spying_propose
    assert loop.run("fix it").succeeded
    assert "…[truncated]" in captured[1]
    assert "x" * (_SNIPPET_LIMIT + 1) not in captured[1]


def test_a_missing_model_is_unreachable_not_a_retry():
    """A 404 on the model name (`ollama/qwen3:8` for `qwen3:8b`) is
    deterministic; retrying it burned every allowed round on one typo."""
    from grapharc.planner.proposal import _is_unreachable

    class NotFoundError(Exception):
        pass

    assert _is_unreachable(NotFoundError("model 'qwen3:8' not found"))
