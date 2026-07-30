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
