"""The delegated tier ladder — `AgentNode` on the Claude CLI backend.

The default tier changed from unconfined to allowlisted, and these tests pin
the ladder's rungs: the default argv carries `--allowedTools` mapped from the
node's own tools and no `bypassPermissions`; the bypass tier is unreachable
without naming it; a token ceiling the delegated path cannot enforce is
refused rather than silently unapplied; and a timed-out delegate's
grandchildren die with it instead of surviving as orphans.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from grapharc.cli import delegate
from grapharc.cli.delegate import CLAUDE_TOOL_FOR, claude_allowlist_for
from grapharc.harness.agent import AgentConfigError, AgentNode, DelegatedToolUseWarning
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import RunContext
from grapharc.stdlib import WRITE_TOOLS, default_harness
from grapharc.testing import ScriptedChatModel


class ClaudeCliDouble(ScriptedChatModel):
    """Looks like the Claude CLI backend to `_is_claude_cli`, runs nothing."""

    @property
    def _llm_type(self) -> str:
        return "grapharc-claude-cli"


SUCCESS_REPORT = json.dumps(
    {
        "subtype": "success",
        "is_error": False,
        "result": "did the task",
        "num_turns": 2,
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "total_cost_usd": 0.01,
        "session_id": "s-1",
    }
)


@pytest.fixture
def spawn_capture(monkeypatch, tmp_path):
    """Intercept the CLI spawn; record argv, return a canned success report."""
    calls: list[list[str]] = []

    def fake_spawn(argv, *, cwd, timeout, stdin_text=None):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, SUCCESS_REPORT, "")

    monkeypatch.setattr(delegate, "_spawn", fake_spawn)
    monkeypatch.setattr(delegate.shutil, "which", lambda name: "/usr/bin/claude")
    return calls


def _node(workspace: Path, **kwargs) -> AgentNode:
    class _Workspaced:
        def __init__(self, root: Path) -> None:
            self.workspace = str(root)

        def run(self, spec, args):  # pragma: no cover - never called when delegated
            raise AssertionError("delegated node ran the local executor")

    harness = default_harness(WRITE_TOOLS, workspace)
    harness.executor = _Workspaced(workspace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DelegatedToolUseWarning)
        return AgentNode(ClaudeCliDouble(responses=[]), harness, name="fixer", **kwargs)


def _ctx() -> RunContext:
    return RunContext(run_id="r-1", graph="g", meter=BudgetMeter(Budget()))


def test_the_default_tier_is_the_allowlist_and_not_bypass(tmp_path, spawn_capture):
    """One operator declaration governs both tiers: the argv pre-approves
    exactly the node's own tools, mapped, and carries no bypassPermissions."""
    result = _node(tmp_path).run("fix it", _ctx())

    assert result.termination_reason.value == "target_met"
    argv = spawn_capture[0]
    allowed = argv[argv.index("--allowedTools") + 1].split(",")
    assert set(allowed) == set(claude_allowlist_for(WRITE_TOOLS))
    assert "Bash" not in allowed  # WRITE_TOOLS has no run_command
    assert "--permission-mode" not in argv


def test_bypass_is_unreachable_without_naming_it(tmp_path, spawn_capture):
    result = _node(tmp_path, delegated_mode="bypass").run("fix it", _ctx())

    assert result.termination_reason.value == "target_met"
    argv = spawn_capture[0]
    assert "--permission-mode" in argv
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--allowedTools" not in argv


def test_an_unknown_tier_is_refused_at_construction(tmp_path):
    with pytest.raises(AgentConfigError) as refusal:
        _node(tmp_path, delegated_mode="everything")
    assert "bypass" in str(refusal.value)


def test_the_construction_warning_names_the_tier_and_its_allowlist(tmp_path):
    harness = default_harness(WRITE_TOOLS, tmp_path)
    with pytest.warns(DelegatedToolUseWarning, match="allowlist"):
        AgentNode(ClaudeCliDouble(responses=[]), harness, name="fixer")
    with pytest.warns(DelegatedToolUseWarning, match="bypassPermissions"):
        AgentNode(
            ClaudeCliDouble(responses=[]), harness, name="fixer", delegated_mode="bypass"
        )


def test_the_mapping_covers_every_core_tool_exactly_once():
    from grapharc.tools import CORE_TOOL_NAMES

    assert set(CLAUDE_TOOL_FOR) == set(CORE_TOOL_NAMES)
    assert len(set(CLAUDE_TOOL_FOR.values())) == len(CLAUDE_TOOL_FOR)
    # Unmapped names are dropped, not guessed at.
    assert claude_allowlist_for(["read_file", "not_a_tool"]) == ["Read"]


def test_a_token_ceiling_the_delegate_cannot_enforce_is_refused(tmp_path, capsys):
    """Accepted-and-unapplied was a limit that existed only in the invocation."""
    from grapharc.cli.agent import run_agent

    code = run_agent(
        "task",
        workspace=tmp_path,
        executor="claude-cli",
        max_tokens=5_000,
        as_json=True,
    )
    assert code != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "max-tokens" in payload["error"]


def test_a_timed_out_delegate_takes_its_process_group_with_it(tmp_path):
    """The deadline kills the group, not just the direct child — Claude Code's
    own spawned shells must not survive as orphans."""
    pidfile = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time, pathlib\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        delegate._spawn([sys.executable, "-c", script], cwd=tmp_path, timeout=2)

    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            import os

            os.kill(grandchild, 0)
        except ProcessLookupError:
            return  # dead, as required
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} survived the group kill")
