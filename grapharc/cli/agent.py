"""`grapharc agent` — run one agent node against a real task from the shell.

This is the first CLI command that is not a demo: the task comes from the
caller, the tools come from `grapharc.tools`, and what the agent was permitted
to do comes from a policy document and flags. The output is
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
from grapharc.cli.config import ConfigError
from grapharc.cli.config import load as load_settings
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

#: Sibling of the run's trace: the policy-document enforcement behind an
#: `agent --policy` run lands here as JSONL, via `grapharc.policy.audit` —
#: denials, and the decision-plus-outcome pair behind every approval request.
#: Allowances leave no record: the compiled policy grants them without
#: consulting the engine, so there is no document decision to write down.
POLICY_AUDIT_FILENAME = "policy-audit.jsonl"


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


def build_policy(
    allow: list[str], deny: list[str], ask: list[str], *, default: Any = None
) -> Any:
    """Flags to a `PermissionPolicy`.

    Unmatched tools keep the `PermissionPolicy` DENY default — unless a
    caller passes `default=ALLOW`, which is what the policy-document path
    does: beside a document the flags are refinements, and an unmatched tool
    is "no flag opinion", decided by the document alone. The default is `None`
    meaning DENY rather than a `Decision` so this module keeps its lazy
    `grapharc.harness` import.
    """
    from grapharc.harness import Decision, PermissionPolicy, PermissionRule

    rules = [PermissionRule(action=Decision.DENY, pattern=p) for p in deny]
    rules += [PermissionRule(action=Decision.ASK, pattern=p) for p in ask]
    rules += [PermissionRule(action=Decision.ALLOW, pattern=p) for p in allow]
    unmatched = Decision.DENY if default is None else default
    return PermissionPolicy(rules=rules, default=unmatched)


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


def _load_agent_document(
    policy_path: Path, *, tenant: str | None, audit_path: Path
) -> tuple[Any, str]:
    """Load and validate the document governing an agent run.

    Returns `(engine, tenant_name)`. Raises `PolicyError` — the shape a
    missing or malformed file already fails with — for the two agent-specific
    refusals as well: a document declaring no tool rules cannot govern tools,
    and a tenant the document does not declare would deny every tool. Both
    are refused before the audit log is built, so a refused run writes
    nothing, not even an empty audit file.
    """
    from grapharc.policy import AuditLog, PolicyEngine, PolicyError, load_document
    from grapharc.policy.document import DEFAULT_TENANT, ResourceKind

    document = load_document(policy_path)
    if not document.rules_for(ResourceKind.TOOL):
        raise PolicyError(
            f"policy document {policy_path} declares no tool rules, so it cannot "
            "govern an agent run — it constrains nothing this command does"
        )
    tenant_name = DEFAULT_TENANT if tenant is None else tenant
    if not document.declares_tenant(tenant_name):
        if tenant is None:
            hint = "pass --tenant to name one"
        else:
            hint = "check the spelling of --tenant"
        raise PolicyError(
            f"tenant {tenant_name!r} is not declared by policy "
            f"{document.version!r}; declared: {document.tenants!r} — {hint}"
        )
    return PolicyEngine(document, audit=AuditLog(audit_path)), tenant_name


def combined_approval(
    *,
    engine: Any,
    doc_policy: Any,
    flag_policy: Any,
    tenant: str,
    as_json: bool,
    stream: Any = None,
    context: dict[str, Any] | None = None,
) -> Any:
    """Approval honoring both a policy document and CLI `--ask` flags.

    Called by `Harness` when the combined policy answers ASK — so at least
    one side asks and neither denies — and granted only when every asking
    side grants: the document's approver role through `ApprovalRouter`, the
    flags through the same terminal prompt `--ask` has always used. A role
    handler that is missing, refuses, fails, or has nobody behind it (JSON
    mode, redirected stdin) denies, exactly as the router already fails
    closed; the role, rule and reason travel into the prompt so the human
    approves a named thing, not a bare tool call.
    """
    from grapharc.harness import Decision
    from grapharc.policy.document import ResourceKind

    roles = sorted(
        {
            rule.approver_role
            for rule in engine.document.rules_for(ResourceKind.TOOL)
            if rule.effect is Decision.ASK and rule.approver_role
        }
    )

    def handle_role(role: str) -> Any:
        def handle(request: Any) -> bool:
            # Same fail-closed guard `_approval` stands on: a prompt has no
            # business in a pipe, and piped bytes are not consent.
            source = stream or sys.stdin
            if as_json or not getattr(source, "isatty", lambda: False)():
                return False
            answer = input(
                f"allow {request.subject} as {role} "
                f"(rule {request.rule_id}: {request.reason})? [y/N] "
            ).strip().lower()
            return answer in ("y", "yes")

        return handle

    router = engine.approval_router(
        {role: handle_role(role) for role in roles}, tenant=tenant
    )
    flag_prompt = _approval(as_json, stream=stream)

    def approve(tool_name: str, args: dict[str, Any]) -> bool:
        doc_asks = doc_policy.decide(tool_name) is Decision.ASK
        flag_asks = flag_policy.decide(tool_name) is Decision.ASK
        granted = True
        if doc_asks:
            decision = engine.check_tool(tool_name, tenant=tenant, context=context)
            granted = router.route(decision, args=args).granted
        if granted and flag_asks:
            granted = flag_prompt(tool_name, args)
        # Both asking sides granted; neither asking at all is a refusal, not
        # an approval — unreachable from Harness, which only calls back on ASK.
        return granted and (doc_asks or flag_asks)

    return approve


def document_denial_recorder(
    *, engine: Any, doc_policy: Any, tenant: str, context: dict[str, Any] | None = None
) -> Any:
    """An `on_denial` hook recording document-side denials, and only those.

    `Harness` fires the hook for every policy denial, including ones the CLI
    flags caused. Recording a flag-side denial through the engine would write
    ALLOW beside a refusal whenever the document permits the tool — a
    contradiction in the audit log — so those stay trace-only, exactly as
    before this command learned about documents.
    """
    from grapharc.harness import Decision

    def record(tool_name: str) -> None:
        if doc_policy.decide(tool_name) is not Decision.DENY:
            return
        engine.check_tool(tool_name, tenant=tenant, context=context)

    return record


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
    max_tokens: int | None = None,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    executor: str = "sandbox",
    system_prompt: str | None = None,
    run_id: str | None = None,
    policy_path: Path | None = None,
    tenant: str | None = None,
    config_path: Path | None = None,
    as_json: bool = False,
) -> int:
    """Run one agent loop and report it. Returns the process exit code.

    Only `target_met` exits 0. Every other termination — the turn cap, a stall,
    an exhausted budget, an error — exits 1, because a script that ran an agent
    needs to know the task was not finished without parsing the reason first.

    `policy_path` names a TOML policy document whose tool rules govern the run
    beside the CLI flags: the document is the ceiling and the flags can only
    narrow it — most restrictive wins, so a document DENY beats a flag allow
    and a document ASK stays asked. Denials the document causes are recorded
    to a policy audit file next to the trace; `--tenant` compiles the
    document for one tenant and is refused without `--policy`, as `--policy`
    itself is refused for delegated execution — `--executor claude-cli` or a
    `claude-cli/*` model — which cannot enforce it.

    `max_tokens=None` means the default ceiling on the governed path — and is
    the only value the delegated path accepts, because a ceiling it cannot
    enforce must be refused rather than silently unapplied.

    Policy and tenant resolve from flags, environment, `grapharc.toml`, then
    defaults, using the same path anchoring and provenance as the planner.
    Other agent options retain their flag/Python-argument defaults.
    """
    try:
        settings = load_settings(config_path)
        policy_path = settings.resolve_path("policy", policy_path)
        tenant = settings.resolve("tenant", tenant)
    except ConfigError as exc:
        return fail(str(exc), as_json=as_json, command="agent", task=task)

    if policy_path is not None and executor == "claude-cli":
        # The delegated loop runs inside Claude Code, outside this process's
        # policy, approval routing and audit — mapping a document onto CLI
        # flags would claim an enforcement that is not there. Refused, like
        # the token ceiling the same path cannot honor below.
        return fail(
            "--policy cannot be enforced under --executor claude-cli: the delegated "
            "loop runs outside this process's policy and audit. Drop --executor "
            "claude-cli for a governed run",
            as_json=as_json,
            command="agent",
        )
    if tenant is not None and policy_path is None:
        return fail(
            "--tenant names whose rules a policy document enforces, so it needs "
            "--policy to mean anything",
            as_json=as_json,
            command="agent",
        )
    if executor == "claude-cli":
        # The whole loop is Claude Code's; nothing below (registry, harness,
        # gateway model) applies. `--model` semantics shift too: the delegated
        # run cannot use the openrouter default, so only an explicit
        # claude-cli/<name> is forwarded.
        from grapharc.cli.delegate import run_delegated

        if max_tokens is not None:
            # Claude Code reports tokens after the fact; there is no inline
            # meter to stop the call that crosses a ceiling. Accepting the
            # flag and not applying it would be a limit that exists only in
            # the invocation.
            return fail(
                "--max-tokens cannot be enforced under --executor claude-cli: "
                "the delegated loop reports its tokens after the fact. Drop "
                "the flag, or use a tool-calling backend for a metered run",
                as_json=as_json,
                command="agent",
            )
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

    if policy_path is None:
        allow = allow or ["*"]
    else:
        # No implicit allow-all beside a document: the document's default
        # governs unmatched tools, and an implicit `*` would permit what a
        # default-deny document refuses. Explicit --allow still votes allow —
        # it just cannot outvote the document (see CombinedPolicy).
        allow = allow or []
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

    governed: tuple[Any, str, Path] | None = None
    if policy_path is not None:
        from grapharc.policy import PolicyError

        try:
            audit_path = trace_path.parent / POLICY_AUDIT_FILENAME
            engine, tenant_name = _load_agent_document(
                Path(policy_path),
                tenant=tenant,
                audit_path=audit_path,
            )
        except PolicyError as exc:
            return fail(str(exc), as_json=as_json, command="agent", task=task)
        governed = (engine, tenant_name, audit_path)

    if governed is None:
        policy = build_policy(allow, deny, ask)
        approval: Any = _approval(as_json)
        on_denial: Any = None
    else:
        from grapharc.harness import CombinedPolicy, Decision

        engine, tenant_name, _ = governed
        doc_policy = engine.permission_policy(tenant=tenant_name)
        # Flags beside a document vote no opinion by default (ALLOW): the
        # document decides unmatched tools, and --deny/--ask narrow it.
        flag_policy = build_policy(allow, deny, ask, default=Decision.ALLOW)
        policy = CombinedPolicy(policies=[doc_policy, flag_policy])
        audit_context = {"command": "agent", "run_id": run_id}
        approval = combined_approval(
            engine=engine,
            doc_policy=doc_policy,
            flag_policy=flag_policy,
            tenant=tenant_name,
            as_json=as_json,
            context=audit_context,
        )
        on_denial = document_denial_recorder(
            engine=engine, doc_policy=doc_policy, tenant=tenant_name, context=audit_context
        )
    harness = Harness(
        registry,
        policy,
        executor=LocalExecutor() if executor == "local" else None,
        workspace=str(workspace),
        approval=approval,
        on_denial=on_denial,
    )
    visible = [spec.name for spec in harness.visible_tools()]
    # Present only when a document governed the run, with the same provenance
    # used to resolve it. Runs without a document retain their existing payload.
    governance_extra: dict[str, Any] = {}
    if governed is not None:
        from grapharc.policy.document import ResourceKind

        _engine, _tenant_name, _audit_path = governed
        _document = _engine.document
        governance_extra = {
            "policy_source": settings.sources["policy"],
            **settings.provenance(),
            "policy_document": {
                "source": settings.sources["policy"],
                "path": str(policy_path),
                "version": _document.version,
                "digest": _engine.digest,
                "tenant": _tenant_name,
                "tool_rules": len(_document.rules_for(ResourceKind.TOOL)),
            },
            "policy_audit": str(_audit_path),
        }

    try:
        from grapharc.gateway import get_model

        model = get_model(model_spec, temperature=0)
    except Exception as exc:  # noqa: BLE001 — every backend has its own error type
        return fail(
            f"could not build model {model_spec!r}: {exc}", as_json=as_json, command="agent"
        )

    if governed is not None:
        from grapharc.harness.agent import is_claude_cli

        # Delegation triggers on the model, not on --executor: a claude-cli
        # model would hand the loop to a subprocess outside the harness this
        # policy was just compiled for. Same predicate AgentNode delegates on,
        # so the two cannot disagree.
        if is_claude_cli(model):
            return fail(
                "--policy cannot be enforced for a claude-cli model: the loop "
                "runs inside Claude Code, outside this process's policy and "
                "audit. Use a tool-calling backend for a governed run",
                as_json=as_json,
                command="agent",
            )

    trace = TraceRecorder(trace_path)
    # The loop's own turn cap bounds iterations, so the meter is left to bound
    # the two things it alone can see: spend and wall clock. Setting both would
    # make an ordinary turn-limited stop report as `budget_exhausted`.
    # None means "the default ceiling", resolved here so the delegated branch
    # above could tell an explicit flag from an untouched one.
    meter = BudgetMeter(
        Budget(
            max_tokens=DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            max_seconds=max_seconds,
        )
    )
    ctx = RunContext(run_id=run_id, graph="cli-agent", meter=meter, tenant=tenant)
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
        **governance_extra,
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

    policy_value = (
        f"{style.dim('allow=')}{allow} {style.dim('ask=')}{ask} {style.dim('deny=')}{deny}"
    )
    if governed is not None:
        _, _tenant_name, _audit_path = governed
        policy_value += (
            f" {style.dim('document=')}{policy_path}"
            f" {style.dim('source=')}{settings.sources['policy']}"
            f" {style.dim('tenant=')}{_tenant_name}"
            f" {style.dim('audit=')}{_audit_path}"
        )
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
            policy_value,
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
    "combined_approval",
    "document_denial_recorder",
    "run_agent",
]
