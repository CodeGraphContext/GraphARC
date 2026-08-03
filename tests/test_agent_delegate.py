"""The delegated executor: `agent --executor claude-cli` without a real Claude Code.

A fake `claude` on PATH records the argv it was given and prints a canned JSON
report, so these tests pin the contract — what is forwarded, what is refused,
what lands in the trace — without a subscription or a network. The fake is the
point: the executor's job is framing and faithful reporting, and both are
checkable against a stand-in.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from grapharc.cli.main import main

REPORT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Two markdown files; both describe the widget.",
    "num_turns": 3,
    "session_id": "sess-1",
    "total_cost_usd": 0.0,
    "usage": {"input_tokens": 120, "output_tokens": 45},
}


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    """A `claude` that logs argv to argv.json and prints REPORT."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv.json"
    report = tmp_path / "report.json"
    report.write_text(json.dumps(REPORT))
    script = bindir / "claude"
    dump = f"import json,sys; json.dump(sys.argv[1:], open({str(argv_log)!r},'w'))"
    script.write_text(f'#!/bin/sh\npython3 -c "{dump}" "$@"\ncat {report}\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return argv_log


def _run(tmp_path, *extra):
    return main(
        [
            "agent",
            "summarise the docs",
            "--executor",
            "claude-cli",
            "--workspace",
            str(tmp_path / "ws"),
            *extra,
        ]
    )


def test_a_successful_delegated_run_reports_and_traces(fake_claude, tmp_path, capsys):
    code = _run(tmp_path)
    printed = capsys.readouterr().out
    assert code == 0
    assert "delegated" in printed
    assert "Two markdown files" in printed

    events = [
        json.loads(line)
        for line in (tmp_path / "ws" / "trace.jsonl").read_text().splitlines()
    ]
    phases = [e["phase"] for e in events]
    assert phases == ["start", "end", "stop"]
    assert events[1]["state_delta"]["tokens_reported"] == 165
    assert events[2]["state_delta"]["termination_reason"] == "target_met"


def test_default_tools_are_forwarded_and_deny_maps_to_disallowed(fake_claude, tmp_path):
    assert _run(tmp_path, "--deny", "Bash") == 0
    argv = json.loads(fake_claude.read_text())
    allowed = argv[argv.index("--allowedTools") + 1]
    assert "Read" in allowed and "Bash" in allowed
    assert argv[argv.index("--disallowedTools") + 1] == "Bash"
    assert argv[argv.index("--max-turns") + 1] == "12"


def test_an_explicit_allow_replaces_the_default_set(fake_claude, tmp_path):
    assert _run(tmp_path, "--allow", "Read", "--allow", "Grep") == 0
    argv = json.loads(fake_claude.read_text())
    assert argv[argv.index("--allowedTools") + 1] == "Read,Grep"


def test_a_claude_cli_model_spec_forwards_its_tail(fake_claude, tmp_path):
    assert _run(tmp_path, "--model", "claude-cli/claude-sonnet-5") == 0
    argv = json.loads(fake_claude.read_text())
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"


def test_a_foreign_model_spec_is_refused(fake_claude, tmp_path, capsys):
    # The openrouter *default* spec is indistinguishable from --model being
    # omitted (argparse fills the same string), so it is treated as omitted;
    # any other foreign backend is an explicit choice and is refused.
    code = _run(tmp_path, "--model", "openai/gpt-4o-mini")
    assert code == 2
    assert "claude-cli/<name> or omitted" in capsys.readouterr().err


def test_ask_globs_are_refused_headless(fake_claude, tmp_path, capsys):
    code = _run(tmp_path, "--ask", "Bash")
    assert code == 2
    assert "headless" in capsys.readouterr().err


def test_a_missing_binary_is_exit_2_with_the_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code = _run(tmp_path)
    assert code == 2
    assert "not on PATH" in capsys.readouterr().err


# ---- the same delegation, reached from an AgentNode ---------------------------
#
# `AgentNode` used to refuse the Claude CLI outright: it has no tool-calling wire
# format, so GraphARC cannot run its own gated loop over it. It now delegates the
# whole loop to Claude Code instead, which is a genuine widening of the trust
# boundary — so what these gates pin is that the widening is *visible*, at
# construction and afterwards in the trace.


def _node(workspace, trace=None, name="worker"):
    from grapharc.gateway import get_model
    from grapharc.harness import Harness, PermissionPolicy, PermissionRule, ToolRegistry
    from grapharc.harness.agent import AgentNode

    harness = Harness(
        ToolRegistry(),
        PermissionPolicy(rules=[PermissionRule(action="allow", pattern="*")]),
        workspace=str(workspace),
    )
    with pytest.warns(Warning):
        return AgentNode(get_model("claude-cli"), harness, name=name, trace=trace)


def test_a_claude_cli_agent_node_warns_loudly_at_construction(tmp_path, fake_claude):
    """A silent switch from "refuses" to "runs with every tool and no checks"
    is the one thing this must not be. The warning names each thing given up.
    """
    from grapharc.gateway import get_model
    from grapharc.harness import Harness, PermissionPolicy, PermissionRule, ToolRegistry
    from grapharc.harness.agent import AgentNode, DelegatedToolUseWarning

    harness = Harness(
        ToolRegistry(),
        PermissionPolicy(rules=[PermissionRule(action="allow", pattern="*")]),
        workspace=str(tmp_path),
    )
    with pytest.warns(DelegatedToolUseWarning) as caught:
        node = AgentNode(get_model("claude-cli"), harness, name="worker")

    assert node.delegated is True
    text = str(caught[0].message)
    for claim in ("EVERY tool", "NOT checked", "NOT confined", "bypassPermissions"):
        assert claim in text, f"the warning does not mention {claim!r}: {text}"


def test_a_tool_calling_backend_is_not_delegated_and_does_not_warn(tmp_path):
    """The mock double must keep running GraphARC's own loop.

    Detection is on `_llm_type`, not on "does this model lack bind_tools" —
    `ScriptedChatModel` lacks it too, and matching that way would have silently
    delegated every mocked agent in the suite to a real subprocess.
    """
    import warnings as _warnings

    from grapharc.gateway import get_model
    from grapharc.harness import Harness, PermissionPolicy, PermissionRule, ToolRegistry
    from grapharc.harness.agent import AgentNode

    harness = Harness(
        ToolRegistry(),
        PermissionPolicy(rules=[PermissionRule(action="allow", pattern="*")]),
        workspace=str(tmp_path),
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning at all fails this
        node = AgentNode(get_model("mock/x", responses=["hi"]), harness, name="m")
    assert node.delegated is False


def test_the_delegated_node_asks_for_every_tool_and_bypasses_the_prompt(
    tmp_path, fake_claude
):
    """Two axes, and conflating them was a real bug found by running it.

    Omitting `--allowedTools` does not mean "every tool" — it leaves Claude
    Code's own gating on, and headless there is nobody to approve a Write, so
    the sub-agent came back reporting it could not create the file. Only
    `--permission-mode bypassPermissions` means what "everything Claude Code
    has" was chosen to mean.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _node(workspace).run("do a thing")

    argv = json.loads(fake_claude.read_text())
    assert "--allowedTools" not in argv, "an allowlist would narrow the tool set"
    assert "--permission-mode" in argv
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"


def test_every_delegated_trace_event_says_it_was_delegated(tmp_path, fake_claude):
    """The construction warning is gone by the time anyone reads the run back.

    Without this marking, a JSONL reader six months later sees an agent node
    that completed and has no way to know its tool calls never reached this
    graph's permission policy — which is exactly the claim the project makes
    about its traces.
    """
    from grapharc.observe.trace import TraceRecorder

    workspace = tmp_path / "ws"
    workspace.mkdir()
    trace_path = tmp_path / "t.jsonl"
    _node(workspace, trace=TraceRecorder(trace_path)).run("do a thing")

    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    assert events, "the delegated run recorded nothing"
    for event in events:
        delta = event.get("state_delta") or {}
        assert delta.get("executor") == "delegated", event

    opening = events[0]["state_delta"]
    assert opening["permission_mode"] == "bypassPermissions"
    assert "not this graph's policy" in opening["governed_by"]


def test_a_delegated_run_charges_the_meter_what_the_sub_agent_reported(tmp_path, fake_claude):
    """A budget must not be simply blind to a delegated node — but the figure is
    the sub-agent's own, and every name it surfaces under says so.
    """
    from grapharc.runtime.budget import Budget, BudgetMeter
    from grapharc.runtime.graph import RunContext

    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = RunContext(run_id="r", graph="g", meter=BudgetMeter(Budget()))
    result = _node(workspace).run("do a thing", ctx)

    assert ctx.meter.tokens == REPORT["usage"]["input_tokens"] + REPORT["usage"]["output_tokens"]
    assert "tokens reported by the sub-agent" in result.note
    assert "not checked by this graph's policy" in result.note
