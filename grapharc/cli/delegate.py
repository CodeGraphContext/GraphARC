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
from dataclasses import dataclass
from pathlib import Path

from grapharc.cli import style
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.cli.runid import refuse_reused_run_id

#: What a bare run may use, mirroring the harness default of "the core tools,
#: shell included". An explicit `--allow` replaces this outright.
DEFAULT_DELEGATED_TOOLS = ("Read", "Glob", "Grep", "LS", "Edit", "Write", "Bash")



# ---------------------------------------------------------------------------
# The reusable core. `run_delegated` below is the CLI's presentation of it, and
# `grapharc.harness.agent.AgentNode` calls it directly when its backend is the
# Claude CLI — one implementation, so the two paths cannot drift on what
# actually gets spawned.


class DelegationError(Exception):
    """A delegated run could not be started, or came back unreadable.

    `reason` is the machine-readable form, written to the trace so a run that
    failed here is distinguishable afterwards from one that ran and refused.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DelegatedRun:
    """What Claude Code reported back. Every number here is *its* figure.

    `tokens_reported` is named for what it is: the sub-agent's own count, not
    something GraphARC metered call by call. Nothing in this object was
    observed by the permission engine.
    """

    ok: bool
    answer: str
    reason: str
    turns: int
    tokens_reported: int
    cost_usd: float | None
    session_id: str | None
    allowed: list[str] | None  # None == no --allowedTools restriction at all
    denied: list[str]


def delegate_task(
    task: str,
    *,
    workspace: Path,
    model: str | None = None,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    max_turns: int = 20,
    max_seconds: float | None = None,
    system_prompt: str | None = None,
    permission_mode: str | None = None,
) -> DelegatedRun:
    """Run one headless `claude -p` agent loop in `workspace`.

    Two separate axes, and conflating them is a trap worth naming. `allow`
    controls *which* tools exist; `permission_mode` controls whether the ones
    that mutate anything are allowed to run without a human answering a prompt.
    Omitting `--allowedTools` does **not** mean "every tool": it means Claude
    Code's own default gating, and headless there is nobody to approve a Write,
    so the sub-agent reports back that it could not create the file. Measured,
    not assumed. `permission_mode="bypassPermissions"` is what actually means
    "everything", and it means it literally — no checks at all.

    Either way the caller is responsible for having said so out loud;
    `AgentNode` warns at construction and marks every trace event.
    """
    binary = shutil.which("claude")
    if binary is None:
        raise DelegationError(
            "the delegated executor shells out to `claude`, which is not on PATH; "
            "install Claude Code or use a tool-calling backend",
            reason="claude_not_found",
        )

    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    argv = [binary, "-p", task, "--output-format", "json", "--max-turns", str(max_turns)]
    if allow is not None:
        argv += ["--allowedTools", ",".join(allow)]
    if deny:
        argv += ["--disallowedTools", ",".join(deny)]
    if permission_mode:
        argv += ["--permission-mode", permission_mode]
    if model:
        argv += ["--model", model]
    if system_prompt:
        argv += ["--append-system-prompt", system_prompt]

    try:
        completed = subprocess.run(
            argv, cwd=workspace, capture_output=True, text=True, timeout=max_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise DelegationError(
            f"max_seconds ({max_seconds}) reached; the delegated run was stopped",
            reason="deadline_exceeded",
        ) from exc

    try:
        report = json.loads(completed.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise DelegationError(
            f"claude exited {completed.returncode} without a readable JSON report: {detail}",
            reason="unreadable_report",
        ) from exc

    usage = report.get("usage") or {}
    met = report.get("subtype") == "success" and not report.get("is_error", False)
    return DelegatedRun(
        ok=met,
        answer=str(report.get("result") or "").strip(),
        reason="target_met" if met else str(report.get("subtype") or "error"),
        turns=int(report.get("num_turns") or 0),
        tokens_reported=int(usage.get("input_tokens") or 0)
        + int(usage.get("output_tokens") or 0),
        cost_usd=report.get("total_cost_usd"),
        session_id=report.get("session_id"),
        allowed=list(allow) if allow is not None else None,
        denied=list(deny or []),
    )


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
    # `--executor claude-cli` resolves its own trace path, so it owns the same
    # guard `run_agent` applies to the sandboxed path.
    reused = refuse_reused_run_id(trace_path, run_id, command="agent", as_json=as_json, task=task)
    if reused is not None:
        return reused
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


__all__ = [
    "DEFAULT_DELEGATED_TOOLS",
    "DelegatedRun",
    "DelegationError",
    "delegate_task",
    "run_delegated",
]
