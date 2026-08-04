"""`grapharc agent` — run one agent node against a real task from the shell.

This is the first CLI command that is not a demo: the task comes from the
caller, the tools come from `grapharc.tools`, and what the agent was permitted
to do comes from flags rather than from a hard-coded example. The output is
built so a run answers the three questions the architecture is graded on — what
it did (`tool_calls`), what it was allowed to do (`policy`, `tools_visible`),
and why it stopped (`termination_reason`, `note`).

`grapharc.tools` is imported at call time, not at module scope: a CLI whose
`--help` fails because one optional package is absent is worse than one that
says which package it wanted when you ask it to do the thing that needs it.
"""

from __future__ import annotations

import inspect
import sys
import uuid
from pathlib import Path
from typing import Any

from grapharc.cli import optional, style
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.cli.runid import refuse_reused_run_id

# Entry points accepted from `grapharc.tools`, in preference order: a registrar
# that fills a registry, and a factory that returns specs. Both are supported
# because both are natural to write and the CLI does not own that module.
CORE_TOOL_ENTRY_POINTS = ("register_core_tools", "core_tools")

TOOLS_HINT = (
    "The core toolset (ROADMAP §3.3) is not in this checkout; "
    "`grapharc.examples.agent_fixit` shows the same wiring with its own tools."
)

DEFAULT_MODEL = "openrouter/anthropic/claude-haiku-4.5"
DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_TOKENS = 100_000
DEFAULT_MAX_SECONDS = 300.0


def _accepts(fn: Any, param: str) -> bool:
    """Whether `fn` can be called with `param=`; unknown signatures say no."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return False
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return param in parameters


def build_registry(workspace: Path) -> tuple[Any, str]:
    """Fill a `ToolRegistry` from `grapharc.tools`, whichever shape it exposes.

    Returns the registry and the entry point it came from, so the run records
    which function produced its tools rather than only that some did. A shape
    that is neither a registry, `None`, nor an iterable of `ToolSpec` raises
    `Unavailable` — guessing at an unknown return value is how a CLI reports a
    toolset it never actually loaded.
    """
    from grapharc.harness import ToolRegistry, ToolSpec

    module = optional.load("grapharc.tools", needed_for="grapharc agent", hint=TOOLS_HINT)
    name, fn = optional.pick(module, CORE_TOOL_ENTRY_POINTS, needed_for="grapharc agent")

    registry = ToolRegistry()
    kwargs: dict[str, Any] = {"workspace": workspace} if _accepts(fn, "workspace") else {}
    returned = fn(registry, **kwargs) if name == "register_core_tools" else fn(**kwargs)

    if isinstance(returned, ToolRegistry):
        return returned, name
    if returned is None:
        return registry, name
    try:
        specs = list(returned)
    except TypeError:
        specs = []
    if specs and all(isinstance(spec, ToolSpec) for spec in specs):
        for spec in specs:
            registry.register(spec)
        return registry, name
    raise optional.Unavailable(
        f"grapharc.tools.{name} returned {type(returned).__name__}; grapharc agent "
        "expects a ToolRegistry, None, or an iterable of ToolSpec"
    )


def build_policy(allow: list[str], deny: list[str], ask: list[str]) -> Any:
    """Flags to a `PermissionPolicy`. Unmatched tools keep the policy's DENY default."""
    from grapharc.harness import Decision, PermissionPolicy, PermissionRule

    rules = [PermissionRule(action=Decision.DENY, pattern=p) for p in deny]
    rules += [PermissionRule(action=Decision.ASK, pattern=p) for p in ask]
    rules += [PermissionRule(action=Decision.ALLOW, pattern=p) for p in allow]
    return PermissionPolicy(rules=rules)


def _approval(as_json: bool, *, stream: Any = None) -> Any:
    """Prompt a human for an ASK tool, and refuse when there is nobody to ask.

    Fail closed on both non-interactive paths: `--json` output means a script is
    reading stdout, and a redirected stdin has no human behind it. Either way
    the call is denied and the denial is recorded — a prompt written into a pipe
    would hang the run or, worse, read the next line of piped data as consent.
    """

    def ask(tool_name: str, args: dict[str, Any]) -> bool:
        source = stream or sys.stdin
        if as_json or not getattr(source, "isatty", lambda: False)():
            return False
        answer = input(f"allow {tool_name}({args})? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    return ask


def run_agent(
    task: str,
    *,
    model_spec: str = DEFAULT_MODEL,
    workspace: Path,
    trace_path: Path | None = None,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    ask: list[str] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    executor: str = "sandbox",
    system_prompt: str | None = None,
    run_id: str | None = None,
    as_json: bool = False,
) -> int:
    """Run one agent loop and report it. Returns the process exit code.

    Only `target_met` exits 0. Every other termination — the turn cap, a stall,
    an exhausted budget, an error — exits 1, because a script that ran an agent
    needs to know the task was not finished without parsing the reason first.
    """
    if executor == "claude-cli":
        # The whole loop is Claude Code's; nothing below (registry, harness,
        # gateway model) applies. `--model` semantics shift too: the delegated
        # run cannot use the openrouter default, so only an explicit
        # claude-cli/<name> is forwarded.
        from grapharc.cli.delegate import run_delegated

        return run_delegated(
            task,
            model_spec=None if model_spec == DEFAULT_MODEL else model_spec,
            workspace=workspace,
            trace_path=trace_path,
            allow=allow,
            deny=deny,
            ask=ask,
            max_turns=max_turns,
            max_seconds=max_seconds,
            system_prompt=system_prompt,
            run_id=run_id,
            as_json=as_json,
        )

    from grapharc.harness import AgentConfigError, AgentNode, Harness, LocalExecutor
    from grapharc.harness.agent import DEFAULT_SYSTEM_PROMPT
    from grapharc.observe.trace import TraceRecorder
    from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter, deadline_guard
    from grapharc.runtime.graph import RunContext

    allow = allow or ["*"]
    deny = deny or []
    ask = ask or []
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    trace_path = Path(trace_path) if trace_path else workspace / "trace.jsonl"
    # While `run_id` still says whether the operator chose one: the generated
    # id below is fresh by construction and has nothing to collide with.
    reused = refuse_reused_run_id(trace_path, run_id, command="agent", as_json=as_json, task=task)
    if reused is not None:
        return reused
    run_id = run_id or f"agent-{uuid.uuid4().hex[:8]}"

    try:
        registry, entry_point = build_registry(workspace)
    except optional.Unavailable as exc:
        return fail(str(exc), as_json=as_json, command="agent")

    policy = build_policy(allow, deny, ask)
    harness = Harness(
        registry,
        policy,
        executor=LocalExecutor() if executor == "local" else None,
        workspace=str(workspace),
        approval=_approval(as_json),
    )
    visible = [spec.name for spec in harness.visible_tools()]

    try:
        from grapharc.gateway import get_model

        model = get_model(model_spec, temperature=0)
    except Exception as exc:  # noqa: BLE001 — every backend has its own error type
        return fail(
            f"could not build model {model_spec!r}: {exc}", as_json=as_json, command="agent"
        )

    trace = TraceRecorder(trace_path)
    # The loop's own turn cap bounds iterations, so the meter is left to bound
    # the two things it alone can see: spend and wall clock. Setting both would
    # make an ordinary turn-limited stop report as `budget_exhausted`.
    meter = BudgetMeter(Budget(max_tokens=max_tokens, max_seconds=max_seconds))
    ctx = RunContext(run_id=run_id, graph="cli-agent", meter=meter)
    node = AgentNode(
        model=model,
        harness=harness,
        max_iterations=max_turns,
        trace=trace,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )

    # Everything known before the run. Repeated into a failure payload so a
    # command that died still records what it was configured to do; `command`
    # is not in here because `fail` supplies it.
    common = {
        "task": task,
        "model": model_spec,
        "run_id": run_id,
        "workspace": str(workspace),
        "trace": str(trace_path),
        "executor": "local" if executor == "local" else "sandbox",
        "tools_from": f"grapharc.tools.{entry_point}",
        "policy": {"allow": allow, "ask": ask, "deny": deny},
        "tools_visible": visible,
    }

    try:
        # `deadline_guard` is what turns `--max-seconds` into an interrupt
        # delivered into a call already in flight; the meter alone only notices
        # between turns, which a single long provider call sails past.
        with deadline_guard(meter, what=f"cli agent run {run_id}"):
            result = node.run(task, ctx)
    except AgentConfigError as exc:
        return fail(str(exc), as_json=as_json, command="agent", **common)
    except BudgetExceeded as exc:
        return fail(
            f"budget exceeded: {exc}",
            as_json=as_json,
            command="agent",
            code=EXIT_FAILED,
            tokens=meter.tokens,
            **common,
        )

    reason = result.termination_reason.value
    met = reason == "target_met"
    payload = {
        "ok": met,
        "command": "agent",
        **common,
        "termination_reason": reason,
        "note": result.note,
        "turns": result.iterations,
        "tokens": meter.tokens,
        "answer": result.output,
        "partial_output": result.partial_output,
        "tool_calls": [call.model_dump() for call in result.tool_calls],
        "denied": len(result.denied),
        "refused": len(result.refused),
    }

    width = style.LABEL_WIDTH
    note = f"  {style.dim(f'({result.note})')}" if result.note else ""

    def count(number: int) -> str:
        """A count, red once it is not zero.

        Zero refusals is not news; one is the reason to read the tool-call rows
        underneath it. The digits are the same either way when colour is off.
        """
        return style.err(str(number)) if number else str(number)

    lines = [
        style.kv("task", task, width=width),
        style.kv("model", model_spec, width=width, tint=style.accent),
        style.kv("workspace", str(workspace), width=width, tint=style.accent),
        style.kv(
            "tools",
            ", ".join(visible) or "(none visible under this policy)",
            width=width,
        ),
        style.kv(
            "policy",
            f"{style.dim('allow=')}{allow} {style.dim('ask=')}{ask} {style.dim('deny=')}{deny}",
            width=width,
        ),
        "",
        style.kv(
            "stopped",
            f"{(style.ok if met else style.warn)(reason)}{note}",
            width=width,
        ),
        style.kv(
            "turns",
            f"{result.iterations}   {style.dim('tool calls:')} {len(result.tool_calls)}   "
            f"{style.dim('denied:')} {count(len(result.denied))}   "
            f"{style.dim('refused:')} {count(len(result.refused))}",
            width=width,
        ),
        style.kv("tokens", f"{meter.tokens:,}", width=width),
    ]
    for call in result.tool_calls:
        suffix = f" {style.dim(f'[{call.refused_by}]')}" if call.refused_by else ""
        # `ToolCallStatus` is ok / denied / error; anything a later version adds
        # lands on amber rather than being quietly called a success.
        verdict = {"ok": True, "denied": False, "error": False}.get(call.status.value)
        lines.append(
            f"   {style.cell(call.status.value, 8, tint=style.tint_for(verdict))} "
            f"{style.accent(call.tool)}{suffix}"
        )
    lines.append("")
    lines.append(
        style.kv("answer", str(result.output), width=width)
        if met
        else style.kv("partial", str(result.partial_output), width=width, tint=style.dim)
    )
    lines.append(style.kv("trace", str(trace_path), width=width, tint=style.accent))

    emit(payload, lines, as_json=as_json)
    return EXIT_OK if met else EXIT_FAILED


__all__ = [
    "CORE_TOOL_ENTRY_POINTS",
    "build_policy",
    "build_registry",
    "run_agent",
]
