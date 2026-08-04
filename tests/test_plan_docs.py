"""The docs registry: node bodies that really read, confined to the cwd.

`plan_incident` proves governance with stub bodies; `plan_docs` is the
shipped answer to "summarise the docs in this workspace" being either a
hollow success or an honest negative. These tests pin the three properties
that make it safe to expose through the Slack gate: bodies read real content,
they read only under the working directory, and the full governed loop
reaches `goal_met` with substantive notes rather than "x ran" markers.
"""

from __future__ import annotations

import json

import pytest

from grapharc.examples import plan_docs
from grapharc.examples.plan_incident import build_loop
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel


@pytest.fixture()
def docs_dir(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Widget\n\nA widget that frobs.\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n\nStart by frobbing.\n")
    (tmp_path / "code.py").write_text("print('not documentation')\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_survey_lists_only_documentation_under_cwd(docs_dir):
    notes = plan_docs._survey_body(plan_docs.DocsState())["notes"]
    assert len(notes) == 1
    assert "2 documentation file(s)" in notes[0]
    assert "README.md" in notes[0]
    assert "code.py" not in notes[0]


def test_read_produces_real_content_not_ran_markers(docs_dir):
    notes = plan_docs._read_body(plan_docs.DocsState())["notes"]
    assert any("A widget that frobs." in note for note in notes)
    assert not any(note.endswith(" ran") for note in notes)


def test_a_symlink_pointing_outside_the_cwd_is_never_read(docs_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.md"
    outside.write_text("# Secret\n\nnot yours\n")
    (docs_dir / "link.md").symlink_to(outside)
    notes = plan_docs._read_body(plan_docs.DocsState())["notes"]
    assert not any("not yours" in note for note in notes)


def test_the_cli_scripted_path_uses_this_registrys_replies(docs_dir, capsys):
    """No `--model`, docs registry: the free path must reach the goal.

    Fails without `_model_for` reading `scripted_planner_replies` off the
    registry module — the incident replies propose kinds this registry does
    not have, and every round dies `unregistered_node`.
    """
    from grapharc.cli.main import main

    code = main(
        [
            "plan",
            "summarise the docs",
            "--scripted",
            "--go",
            "--registry",
            "grapharc.examples.plan_docs:build_registry",
            "--trace",
            str(docs_dir / "t.jsonl"),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "goal_met" in printed
    assert "A widget that frobs." in printed


class _FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """A real-model stand-in: has `invoke`, is not the scripted type."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _FakeReply:
        self.prompts.append(prompt)
        return _FakeReply("merge 07 and 08 into one Slack page; keep the walkthrough")


def test_propose_hands_the_collected_notes_to_the_model(docs_dir):
    model = _FakeModel()
    body = plan_docs._propose_body_for(model)
    state = plan_docs.DocsState(
        goal="plan a merge", notes=["read README.md: Widget — A widget that frobs."]
    )
    notes = body(state)["notes"]
    assert any("proposal (model-authored): merge 07 and 08" in n for n in notes)
    assert "A widget that frobs." in model.prompts[0]
    assert "plan a merge" in model.prompts[0]


def test_propose_without_a_real_model_declines_honestly(docs_dir):
    from grapharc.testing import ScriptedChatModel

    for model in (None, ScriptedChatModel(responses=["{}"])):
        notes = plan_docs._propose_body_for(model)(plan_docs.DocsState())["notes"]
        assert any("needs a real model" in n for n in notes)


def test_a_failing_model_becomes_a_note_not_a_crash(docs_dir):
    class Exploding:
        def invoke(self, prompt):
            raise RuntimeError("backend fell over")

    notes = plan_docs._propose_body_for(Exploding())(plan_docs.DocsState())["notes"]
    assert any("model call failed" in n and "backend fell over" in n for n in notes)


def test_the_governed_loop_reaches_goal_met_with_substantive_notes(docs_dir, tmp_path):
    # Instance names differ from kinds on purpose: a real planner invents
    # names ("docs_survey" of kind "survey"), and behaviour must key on the
    # kind — this is the shape that catches a factory keyed on the name.
    chain = ["survey", "read", "summarise", "propose"]
    names = [f"docs_{kind}" for kind in chain]
    endpoints = [START, *names, END]
    reply = json.dumps(
        {
            "nodes": [
                {"name": name, "kind": kind}
                for name, kind in zip(names, chain, strict=True)
            ],
            "edges": [
                {"source": a, "target": b}
                for a, b in zip(endpoints, endpoints[1:], strict=False)
            ],
        }
    )
    loop = build_loop(
        ScriptedChatModel(responses=[reply]),
        # The registry carries a "real" model for `propose` while the planner
        # stays scripted — the two roles are deliberately separable.
        registry=plan_docs.build_registry(_FakeModel()),
        state_schema=plan_docs.DocsState,
        writes=plan_docs.WRITES,
        edge_policy=plan_docs.default_edge_policy(),
    )
    result = loop.run(
        "summarise the docs", plan_docs.DocsState(goal="summarise the docs")
    )
    assert result.succeeded
    assert any("summary of 2 file(s)" in note for note in result.state.notes)
    assert any("Widget" in note for note in result.state.notes)
    assert any("proposal (model-authored):" in note for note in result.state.notes)
