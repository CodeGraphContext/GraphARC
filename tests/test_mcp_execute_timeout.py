"""The MCP `execute` tool's two budgets, and the kill that enforces them.

One timeout used to cover both halves of a mutating `execute`: the park, where
a human is being asked, and the work, where an approved graph runs. It was
`approval_timeout + 120s`, so a human who answered near the end of the park
left roughly two minutes for the run itself — and a governed run's agent
phases delegate to Claude Code, which reads files, edits them and verifies.
The likely outcome was not a wedged process being cleaned up; it was a
human-approved, tree-mutating run being SIGKILLed partway through mutating the
tree. The non-mutating branch, meanwhile, passed `timeout=None` and so was not
bounded at all.

The tool then told the calling agent, in its own docstring, that "a timeout
leaves the plan unexecuted and this call safe to reissue" — which is true of a
park that expired unanswered and false of everything else.

Issue #113. The park and the work have separate budgets now, the kill reaches
the whole process group rather than the direct child, and the docstring no
longer promises something the mutating path cannot honour.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import sys

import pytest

from grapharc.mcp import build_server
from grapharc.mcp.driver import (
    DEFAULT_WORK_TIMEOUT,
    DriverError,
    _kill_tree,
    run_cli,
    work_timeout,
)


def _unwrap(raw) -> dict:
    """FastMCP's call_tool result as the tool's own dict, across SDK shapes.

    Duplicated from test_mcp_gate rather than imported: `tests/` is not a
    package, so a cross-module import here would depend on sys.path shape.
    """
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        structured = raw[1]
        return structured.get("result", structured)
    blocks = raw[0] if isinstance(raw, tuple) else raw
    text = "".join(getattr(block, "text", "") for block in blocks)
    return json.loads(text)


def _mark_mutating(root, run_dir: str):
    """Flip the record's verdict, resolving the CLI's root-relative run_dir."""
    from pathlib import Path

    resolved = Path(run_dir) if Path(run_dir).is_absolute() else root / run_dir
    plan_file = resolved / "plan.json"
    record = json.loads(plan_file.read_text())
    record["mutating"] = True
    plan_file.write_text(json.dumps(record, indent=2) + "\n")
    return resolved


def _capture_timeout(monkeypatch) -> dict:
    """Stand in for `run_cli` and record the bound it was handed.

    Installed *after* the plan under test exists: the `plan` tool goes through
    the same `run_cli`, so patching earlier would stub out the very call that
    writes the run directory.
    """
    seen: dict = {}

    async def fake_run_cli(argv, *, cwd, timeout=None):
        seen["argv"] = argv
        seen["timeout"] = timeout
        return 0, json.dumps({"ok": True, "executed": True, "run_id": "r"}), ""

    from grapharc.mcp import driver

    monkeypatch.setattr(driver, "run_cli", fake_run_cli)
    return seen


# -- the two budgets --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_work_budget_is_separate_from_the_park(tmp_path, monkeypatch):
    """The bug: this was `approval_timeout + 120`, so approving late left two
    minutes to do the work in."""
    server = build_server(tmp_path)
    planned = _unwrap(await server.call_tool("plan", {"goal": "fix", "scripted": True}))
    run_dir = _mark_mutating(tmp_path, planned["run_dir"])
    seen = _capture_timeout(monkeypatch)

    await server.call_tool("execute", {"run_dir": str(run_dir), "approval_timeout": 240.0})

    # The park keeps its own budget; the work gets a whole one beside it.
    assert seen["timeout"] == 240.0 + DEFAULT_WORK_TIMEOUT
    assert seen["timeout"] != 240.0 + 120.0


@pytest.mark.asyncio
async def test_a_non_mutating_execute_is_bounded_at_all(tmp_path, monkeypatch):
    """The other half of the bug: this branch passed `timeout=None`, so a
    wedged run held the tool call open with nothing to end it."""
    server = build_server(tmp_path)
    planned = _unwrap(await server.call_tool("plan", {"goal": "fix", "scripted": True}))
    seen = _capture_timeout(monkeypatch)

    await server.call_tool("execute", {"run_dir": planned["run_dir"]})

    assert seen["timeout"] is not None
    # No park to wait through, so the work budget is the whole bound.
    assert seen["timeout"] == DEFAULT_WORK_TIMEOUT


# -- the operator's ceiling -------------------------------------------------


def test_the_work_budget_defaults_and_honours_the_environment():
    assert work_timeout({}) == DEFAULT_WORK_TIMEOUT
    assert work_timeout({"GRAPHARC_MCP_WORK_TIMEOUT": "60"}) == 60.0
    # Unset and empty both mean "operator said nothing", not "zero".
    assert work_timeout({"GRAPHARC_MCP_WORK_TIMEOUT": ""}) == DEFAULT_WORK_TIMEOUT


@pytest.mark.parametrize("bad", ["soon", "-1", "0"])
def test_a_nonsense_work_budget_is_refused_rather_than_replaced(bad):
    """Substituting the default for an unreadable override is how a run gets
    killed at a limit nobody chose."""
    with pytest.raises(DriverError) as caught:
        work_timeout({"GRAPHARC_MCP_WORK_TIMEOUT": bad})
    assert "GRAPHARC_MCP_WORK_TIMEOUT" in str(caught.value)


@pytest.mark.asyncio
async def test_the_env_override_reaches_the_call(tmp_path, monkeypatch):
    server = build_server(tmp_path)
    planned = _unwrap(await server.call_tool("plan", {"goal": "fix", "scripted": True}))
    seen = _capture_timeout(monkeypatch)
    monkeypatch.setenv("GRAPHARC_MCP_WORK_TIMEOUT", "45")

    await server.call_tool("execute", {"run_dir": planned["run_dir"]})

    assert seen["timeout"] == 45.0


# -- the kill reaches the whole tree ----------------------------------------


@pytest.mark.asyncio
async def test_the_child_is_started_in_its_own_session(tmp_path, monkeypatch):
    """`killpg` needs a group to aim at, and the child only has its own if it
    was started with one."""
    seen: dict = {}
    real = asyncio.create_subprocess_exec

    async def spy(*argv, **kwargs):
        seen.update(kwargs)
        return await real(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    await run_cli(["--version"], cwd=tmp_path, timeout=30)

    assert seen.get("start_new_session") is True


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_kill_reaches_a_grandchild(tmp_path):
    """The defect this closes: `process.kill()` signals the direct child only,
    so a delegated Claude Code process outlived the run that spawned it and
    kept holding the workspace.

    A real tree, not a mock: the child spawns a grandchild that writes its pid
    and sleeps, and the grandchild must be gone once the group is killed.
    """
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import os, subprocess, sys, time\n"
        f"kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(kid.pid))\n"
        "time.sleep(120)\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, cwd=tmp_path, start_new_session=True
    )
    for _ in range(200):  # wait for the grandchild to exist
        if pid_file.is_file() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    grandchild = int(pid_file.read_text().strip())
    os.kill(grandchild, 0)  # alive, or this raises

    _kill_tree(process)
    await process.wait()

    for _ in range(200):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:  # pragma: no cover — the failure this test exists to catch
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail(f"grandchild {grandchild} survived the kill")


def test_killing_an_already_dead_group_is_not_an_error():
    """The success case arriving early must not raise out of the timeout path."""

    class Gone:
        pid = 2**22  # far above any live pid on a normal machine

        def kill(self):
            raise ProcessLookupError

    _kill_tree(Gone())  # no exception


# -- what the agent is told -------------------------------------------------


@pytest.mark.asyncio
async def test_the_tool_no_longer_promises_a_timeout_is_safe_to_reissue(tmp_path):
    """The docstring is the agent's whole briefing. It said a timeout left the
    plan unexecuted and the call safe to reissue, which is true of an
    unanswered park and false of a run killed partway through its work."""
    server = build_server(tmp_path)
    doc = next(t.description for t in await server.list_tools() if t.name == "execute")

    assert "safe to reissue" not in doc
    # And it must say what to do instead of reissuing blindly.
    assert "show_graph" in doc


@pytest.mark.asyncio
async def test_the_timeout_error_sends_the_reader_to_the_run_directory(tmp_path):
    """`DriverError` is what the agent sees when the bound is hit, so it is the
    other half of the same briefing."""
    with pytest.raises(DriverError) as caught:
        await run_cli(["plan", "wait", "--scripted"], cwd=tmp_path, timeout=0.001)

    message = str(caught.value)
    assert "show_graph" in message
    assert "did not finish" in message


def test_the_default_work_budget_matches_the_slack_path():
    """Both exist for the same reason — an approved run needs minutes, not two
    of them — so a reader comparing the two surfaces should find one number.
    Read off the module rather than a cwd-relative path."""
    from grapharc.slack import config as slack_config

    assert DEFAULT_WORK_TIMEOUT == 1800.0
    assert '"GRAPHARC_SLACK_WORK_TIMEOUT", "1800"' in inspect.getsource(slack_config)
