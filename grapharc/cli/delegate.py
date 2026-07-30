"""`grapharc agent --executor claude-cli` — delegate the whole loop to Claude Code.

The harness executors run grapharc's own tool loop: the model is a raw
ingredient, grapharc's gate approves every call, grapharc's meter charges it.
This executor is the other trade: hand the task, the workspace and a tool
policy to the `claude` CLI in headless mode and let *its* agent loop do the
work on the operator's subscription. What grapharc keeps is the frame — the
workspace boundary, the wall-clock ceiling enforced from outside, the tool
allow/deny handed down, and a trace of what came back.

Named honestly in the output as `delegated`: the tools are Claude Code's, the
permission granularity is Claude Code's, and the token figure is what the
sub-agent *reports*, not what grapharc metered inline. Coarser governance,
bought deliberately, for the backend that cannot be driven as a raw model
(`ClaudeCodeCLIChatModel` has no `bind_tools` — the CLI exposes a finished
agent, not a tool-calling completion API).

Tool names here are Claude Code's (`Read`, `Glob`, `Grep`, `Edit`, `Write`,
`Bash`, …), not grapharc's seven — they are what `--allowedTools` understands.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

from grapharc.cli import style
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail

#: What a bare run may use, mirroring the harness default of "the core tools,
#: shell included". An explicit `--allow` replaces this outright.
DEFAULT_DELEGATED_TOOLS = ("Read", "Glob", "Grep", "LS", "Edit", "Write", "Bash")


def run_delegated(
    task: str,
    *,
    model_spec: str | None,
    workspace: Path,
    trace_path: Path | None,
    allow: list[str] | None,
    deny: list[str] | None,
    ask: list[str] | None,
    max_turns: int,
    max_seconds: float | None,
    system_prompt: str | None,
    run_id: str | None,
    as_json: bool,
) -> int:
    """One `claude -p` run inside the workspace. Returns the exit code."""
    from grapharc.observe.trace import TraceRecorder

    if ask:
        return fail(
            "--ask needs a human at a prompt; the delegated executor is headless "
            "by construction — use --allow/--deny",
            as_json=as_json,
            command="agent",
        )

    binary = shutil.which("claude")
    if binary is None:
        return fail(
            "the delegated executor shells out to `claude`, which is not on PATH; "
            "install Claude Code or use --executor sandbox with a tool-calling backend",
            as_json=as_json,
            command="agent",
        )

    model_arg: list[str] = []
    if model_spec and model_spec.startswith("claude-cli/"):
        model_arg = ["--model", model_spec.removeprefix("claude-cli/")]
    elif model_spec:
        return fail(
            f"--executor claude-cli runs the Claude Code CLI; --model must be "
            f"claude-cli/<name> or omitted, got {model_spec!r}",
            as_json=as_json,
            command="agent",
        )

    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    trace_path = Path(trace_path) if trace_path else workspace / "trace.jsonl"
    run_id = run_id or f"agent-{uuid.uuid4().hex[:8]}"

    allowed = list(allow) if allow and allow != ["*"] else list(DEFAULT_DELEGATED_TOOLS)
    argv = [
        binary,
        "-p",
        task,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
        "--allowedTools",
        ",".join(allowed),
        *model_arg,
    ]
    if deny:
        argv += ["--disallowedTools", ",".join(deny)]
    if system_prompt:
        argv += ["--append-system-prompt", system_prompt]

    trace = TraceRecorder(trace_path)
    trace.event(
        run_id=run_id,
        graph="cli-agent",
        node="claude_code",
        phase="start",
        step=1,
        state_delta={"executor": "delegated", "allowed": allowed, "denied": deny or []},
    )

    try:
        completed = subprocess.run(
            argv,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=max_seconds,
        )
    except subprocess.TimeoutExpired:
        trace.event(
            run_id=run_id, graph="cli-agent", node="claude_code", phase="stop", step=1,
            state_delta={"termination_reason": "deadline_exceeded"},
        )
        return fail(
            f"max_seconds ({max_seconds:g}) reached; the delegated run was stopped",
            as_json=as_json,
            command="agent",
            code=EXIT_FAILED,
            run_id=run_id,
            trace=str(trace_path),
        )

    try:
        report = json.loads(completed.stdout)
    except (json.JSONDecodeError, ValueError):
        trace.event(
            run_id=run_id, graph="cli-agent", node="claude_code", phase="stop", step=1,
            state_delta={"termination_reason": "unreadable_report"},
        )
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        return fail(
            f"claude exited {completed.returncode} without a readable JSON report: {detail}",
            as_json=as_json,
            command="agent",
            code=EXIT_FAILED,
            run_id=run_id,
            trace=str(trace_path),
        )

    usage = report.get("usage") or {}
    tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    turns = int(report.get("num_turns") or 0)
    met = report.get("subtype") == "success" and not report.get("is_error", False)
    reason = "target_met" if met else str(report.get("subtype") or "error")
    answer = str(report.get("result") or "").strip()

    trace.event(
        run_id=run_id, graph="cli-agent", node="claude_code", phase="end", step=1,
        state_delta={
            "turns": turns,
            "tokens_reported": tokens,
            "cost_usd": report.get("total_cost_usd"),
            "session_id": report.get("session_id"),
        },
    )
    trace.event(
        run_id=run_id, graph="cli-agent", node="claude_code", phase="stop", step=1,
        state_delta={"termination_reason": reason},
    )

    payload = {
        "ok": met,
        "command": "agent",
        "task": task,
        "model": model_arg[1] if model_arg else "claude-cli default",
        "run_id": run_id,
        "workspace": str(workspace),
        "trace": str(trace_path),
        "executor": "delegated",
        "policy": {"allow": allowed, "deny": deny or []},
        "termination_reason": reason,
        "turns": turns,
        "tokens_reported": tokens,
        "cost_usd": report.get("total_cost_usd"),
        "answer": answer,
    }

    width = style.LABEL_WIDTH
    lines = [
        style.kv("task", task, width=width),
        style.kv("executor", "delegated (Claude Code's own loop and tools)", width=width),
        style.kv("model", payload["model"], width=width, tint=style.accent),
        style.kv("workspace", str(workspace), width=width, tint=style.accent),
        style.kv(
            "policy",
            f"{style.dim('allow=')}{allowed} {style.dim('deny=')}{deny or []}",
            width=width,
        ),
        "",
        style.kv("stopped", (style.ok if met else style.warn)(reason), width=width),
        style.kv(
            "turns",
            f"{turns}   {style.dim('tokens (reported):')} {tokens:,}",
            width=width,
        ),
        "",
        style.kv("answer", answer or "(empty)", width=width),
        style.kv("trace", str(trace_path), width=width, tint=style.accent),
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK if met else EXIT_FAILED


__all__ = ["DEFAULT_DELEGATED_TOOLS", "run_delegated"]
