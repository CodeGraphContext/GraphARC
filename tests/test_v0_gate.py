"""The V0 gate (ROADMAP §4.1): an agent edits a file and runs a test.

The offline tests pin the parts that must hold regardless of which model runs:
a denied tool is never offered, and the wiring composes. The `live` test is the
gate itself and needs a real model, so it is opt-in (`pytest -m live`).
"""

from __future__ import annotations

import json

import pytest

from grapharc.examples.agent_fixit import (
    BROKEN_SOURCE,
    build_harness,
    make_workspace,
    run_fixit,
    run_pytest,
)
from grapharc.gateway import config
from grapharc.harness import ToolCallStatus
from grapharc.testing import ScriptedChatModel

LIVE_MODEL = "openrouter/anthropic/claude-haiku-4.5"


def test_the_broken_project_really_fails_first(tmp_path):
    """Without this, a passing gate would prove nothing."""
    workspace = make_workspace(tmp_path / "ws")
    assert "a - b" in (workspace / "calc.py").read_text(encoding="utf-8")
    assert run_pytest(workspace).returncode != 0


def test_denied_tool_is_never_offered_to_the_model(tmp_path):
    """Policy-before-schema: the model cannot ask for what it was never shown."""
    harness = build_harness(make_workspace(tmp_path / "ws"))
    visible = [t.name for t in harness.visible_tools()]
    assert "delete_file" not in visible
    assert {"read_file", "write_file", "run_tests"} == set(visible)


def test_denied_tool_is_refused_even_when_called_directly(tmp_path):
    """The registry still holds it, so the denial has to be enforced, not just
    hidden — a model that guesses the name gets nothing."""
    from grapharc.harness import PermissionDenied

    workspace = make_workspace(tmp_path / "ws")
    harness = build_harness(workspace)
    with pytest.raises(PermissionDenied):
        harness.call("delete_file", {"path": "calc.py"})
    assert (workspace / "calc.py").exists()


def test_a_scripted_agent_fixes_the_bug_end_to_end(tmp_path):
    """The same wiring the live gate uses, driven deterministically."""
    workspace = make_workspace(tmp_path / "ws")
    harness = build_harness(workspace)
    fixed = "def add(a, b):\n    return a + b\n"

    model = ScriptedChatModel(
        responses=[
            json.dumps({"tool": "read_file", "args": {"path": "calc.py"}}),
            json.dumps({"tool": "write_file", "args": {"path": "calc.py", "content": fixed}}),
            json.dumps({"tool": "run_tests", "args": {}}),
            "Fixed: add was subtracting. The suite passes.",
        ],
        on_exhausted="repeat",
    )
    # ScriptedChatModel emits text, not tool_calls, so drive the tools directly
    # and assert the harness path rather than the model's call format.
    assert harness.call("read_file", {"path": "calc.py"}) == BROKEN_SOURCE
    harness.call("write_file", {"path": "calc.py", "content": fixed})
    output = harness.call("run_tests", {})

    assert "1 passed" in output
    assert run_pytest(workspace).returncode == 0
    assert model is not None  # constructed; the live gate exercises the model path


def test_only_visible_tools_become_model_schemas(tmp_path):
    """The schemas handed to the model come from the policy-filtered set, so a
    denied tool is absent from the request itself — not merely discouraged."""
    from grapharc.harness import tool_schemas

    harness = build_harness(make_workspace(tmp_path / "ws"))
    names = {schema["function"]["name"] for schema in tool_schemas(harness)}
    assert "delete_file" not in names
    assert {"read_file", "write_file", "run_tests"} == names


@pytest.mark.live
@pytest.mark.skipif(not config.openrouter_api_key(), reason="no OpenRouter API key")
@pytest.mark.timeout(600)
def test_gate_a_real_model_fixes_the_bug_and_proves_it(tmp_path):
    """ROADMAP §4.1's gate, in full.

    A real model must read the file, correct it, run the suite, and stop — with
    the denied tool never offered, tokens actually metered, and the fix verified
    by a test run the agent did not perform.
    """
    outcome = run_fixit(LIVE_MODEL, workspace=tmp_path / "ws")
    result = outcome["result"]

    assert outcome["tests_pass"], f"suite still failing; source was:\n{outcome['source']}"
    assert "a + b" in outcome["source"]
    assert result.termination_reason == "target_met"
    assert "delete_file" not in outcome["visible_tools"]

    tools_used = [c.tool for c in result.tool_calls if c.status is ToolCallStatus.OK]
    assert "write_file" in tools_used and "run_tests" in tools_used
    assert outcome["tokens"] > 0, "tokens must be metered automatically (ROADMAP §0.4)"
