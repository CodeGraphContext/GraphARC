"""`grapharc plan` — drive the governed loop from the command line (§12.1).

The crux of the architecture was, until this existed, unreachable without
writing Python: `grapharc.planner` closed the propose → admit → materialise →
execute → replan cycle and no shipped code path drove it. This is the surface.

    grapharc plan "investigate the checkout outage"

Scripted by default, so it costs nothing and prints the same thing every time —
the shipped planner proposes a `deploy` in round 1, admission refuses the edge,
and round 2 replans without it. `--model` swaps in a real backend and changes
nothing about the enforcement, because none of it is prompt text.

Two things an operator supplies, and neither can come from a model:

- `--registry module:attr` — the kinds a planner may propose. Absence is
  refusal; there is no wildcard. Defaults to the shipped incident demo.
- `--policy PATH [--tenant NAME]` — a TOML document whose `edge` rules are
  compiled to the `EdgePolicy` admission consults, via
  `PolicyEngine.edge_policy()`. This is the path that makes declarative
  governance constrain a run rather than answer questions about one.

Exit codes follow the CLI's convention: `0` the goal was met, `1` the run
stopped short for a recorded reason (refused, out of budget, out of rounds),
`2` it could not start.
"""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grapharc.cli import style
from grapharc.cli.config import ConfigError, Settings
from grapharc.cli.config import load as load_settings
from grapharc.cli.generate import resolve_or_generate_policy
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail

DEFAULT_REGISTRY = "grapharc.examples.plan_incident:build_registry"


class PlanSetupError(Exception):
    """Raised before anything runs, so a bad flag never half-executes a plan."""


@dataclass
class RegistryBundle:
    """Everything a registry module supplies, travelling together.

    Separated fields would let a registry be paired with someone else's schema,
    write map or policy — each of which fails quietly rather than loudly.
    """

    registry: Any
    state_schema: Any = None
    writes: dict[str, set[str]] | None = None
    #: The module's own `default_edge_policy()`, used when no policy is named
    #: and none was generated. A policy written for other kinds would deny names
    #: that do not exist and permit ones that do.
    default_policy: Any = None
    #: Kinds the module considers dangerous, from its `MUTATING_KINDS`. Handed to
    #: the policy generator so it knows what to deny; empty means it denies
    #: nothing, which is why a module that can change things should say so.
    mutating: tuple[str, ...] = ()
    #: The module's own `build_loop`, when it ships one. This is how a registry
    #: owns its goal check and observer instead of inheriting the incident
    #: demo's (`len(notes) >= 3`) — a registry whose state never accumulates
    #: three notes would otherwise burn every round and report failure after
    #: doing the work. Absent, the incident builder remains the fallback.
    build_loop: Any = None


def resolve_registry(target: str, model: Any = None) -> RegistryBundle:
    """Import `module:attr` and return `(registry, state_schema, writes)`.

    A callable attribute is called, so both `mypkg:build_registry` and
    `mypkg:REGISTRY` work; a `NodeRegistry` is not callable, so the two cannot
    be confused. Mirrors `grapharc.cli.serve.resolve_registry` deliberately —
    two flags that look the same should behave the same.

    **A factory that accepts an argument is given the model.** Any registry with
    agent-backed kinds needs one to build them, and there is nowhere else for it
    to come from; `grapharc.stdlib:build_registry` relies on this, and leaves its
    agent kinds out entirely when the model is None. A zero-argument factory is
    called bare, so a registry of plain functions never has to care.

    A registry alone is not enough to run a loop: the kinds have to agree with a
    state schema, each kind needs a declared write set or it may write nothing,
    and *something* has to say which transitions between those kinds are sane
    when no policy file is given. So the module may also export `STATE_SCHEMA`,
    `WRITES` and `default_edge_policy`, and all four travel together — a registry
    silently paired with a schema its nodes cannot write to, or with a policy
    written for someone else's kinds, is worse than one that refuses to load.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator or not attribute:
        raise PlanSetupError(f"--registry expects module:attr, got {target!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PlanSetupError(f"--registry {target!r}: {exc}") from exc
    registry = getattr(module, attribute, None)
    if registry is None:
        raise PlanSetupError(f"--registry {target!r}: {module_name} has no {attribute!r}")
    if callable(registry):
        registry = registry(model) if _accepts_an_argument(registry) else registry()
    default_policy = getattr(module, "default_edge_policy", None)
    return RegistryBundle(
        registry=registry,
        state_schema=getattr(module, "STATE_SCHEMA", None),
        writes=getattr(module, "WRITES", None),
        default_policy=default_policy() if callable(default_policy) else default_policy,
        mutating=tuple(getattr(module, "MUTATING_KINDS", ())),
        build_loop=getattr(module, "build_loop", None),
    )


def _accepts_an_argument(factory: Any) -> bool:
    """Whether `factory` takes a positional parameter we can hand the model to."""
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # a builtin or C callable
        return False
    return any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def resolve_edge_policy(policy_path: Path | None, *, tenant: str) -> tuple[Any, str]:
    """Compile a policy document's edge rules, or fall back to the demo's.

    Returns `(edge_policy, description)`. The description is printed, because
    which policy a run was subject to is the first thing anyone asks afterwards.
    """
    if policy_path is None:
        from grapharc.examples.plan_incident import default_edge_policy

        return default_edge_policy(), "built-in demo (deny -> deploy, allow otherwise)"
    from grapharc.policy import PolicyEngine, PolicyError

    try:
        engine = PolicyEngine.from_file(policy_path)
    except (OSError, PolicyError) as exc:
        raise PlanSetupError(f"--policy {str(policy_path)!r}: {exc}") from exc
    policy = engine.edge_policy(tenant=tenant)
    return policy, f"{policy_path} (tenant {tenant!r}, {len(policy.rules)} edge rule(s))"


def _model_for(spec: str | None, registry_target: str = DEFAULT_REGISTRY) -> tuple[Any, str]:
    """The scripted planner by default; a real backend when asked for one.

    The scripted replies come from the registry module when it supplies
    `scripted_planner_replies`, because a script that proposes one registry's
    kinds against another registry's catalog is rejected every round — the
    incident replies against the docs registry produced five rounds of
    `unregistered_node` and a `planning_failed`. The incident module's replies
    stay the fallback for modules that ship none.
    """
    if spec is None:
        from grapharc.testing import ScriptedChatModel

        module_name = registry_target.split(":", 1)[0]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PlanSetupError(f"--registry {registry_target!r}: {exc}") from exc
        replies = getattr(module, "scripted_planner_replies", None)
        if replies is None:
            from grapharc.examples.plan_incident import scripted_planner_replies as replies

        return ScriptedChatModel(responses=replies()), "scripted"
    from grapharc.gateway import get_model

    return get_model(spec), spec


def plan(
    goal: str,
    *,
    model_spec: str | None = None,
    registry_target: str | None = None,
    policy_path: Path | None = None,
    tenant: str | None = None,
    trace_path: Path | None = None,
    run_id: str | None = None,
    max_rounds: int | None = None,
    max_tokens: int | None = None,
    config_path: Path | None = None,
    settings: Settings | None = None,
    approve: bool = False,
    approval_timeout: float | None = None,
    as_json: bool = False,
) -> int:
    """Run one governed planning loop against `goal`. Returns the exit code."""
    from grapharc.observe.trace import TraceRecorder
    from grapharc.planner import LoopLimits
    from grapharc.runtime.budget import Budget

    trace_path = trace_path or Path(tempfile.mkdtemp(prefix="grapharc-plan-")) / "trace.jsonl"

    # Everything that can be wrong about the setup is decided before a model is
    # asked anything, so a bad flag cannot half-execute a plan.
    try:
        if settings is None:
            settings = load_settings(config_path)
        model_spec = settings.resolve("model", model_spec)
        registry_target = settings.resolve("registry", registry_target, DEFAULT_REGISTRY)
        policy_path = settings.resolve_path("policy", policy_path)
        tenant = settings.resolve("tenant", tenant, "default")
        max_rounds = settings.resolve("max_rounds", max_rounds, 8)
        max_tokens = settings.resolve("max_tokens", max_tokens, 100_000)
        model, model_description = _model_for(model_spec, registry_target)
        bundle = resolve_registry(registry_target, model)
        registry, state_schema, writes = bundle.registry, bundle.state_schema, bundle.writes
        edge_policy, policy_description, policy_source = resolve_or_generate_policy(
            policy_path,
            tenant=tenant,
            # Only a *real* backend generates. The scripted planner has no
            # opinion worth asking for, and a deterministic demo that quietly
            # produced a different policy each run would not be one.
            model=model if model_spec else None,
            goal=goal,
            catalog=registry.catalog(),
            mutating=bundle.mutating,
            fallback=bundle.default_policy,
            fallback_label=f"{registry_target} default",
        )
    except (ConfigError, PlanSetupError) as exc:
        return fail(str(exc), as_json=as_json, command="plan", goal=goal)
    except Exception as exc:  # noqa: BLE001 — a backend that will not load is a setup failure
        return fail(f"could not build the plan: {exc}", as_json=as_json, command="plan", goal=goal)

    from grapharc.examples.plan_incident import IncidentState
    from grapharc.examples.plan_incident import build_loop as incident_build_loop

    schema = state_schema or IncidentState
    trace = TraceRecorder(trace_path)
    approval = None
    if approve:
        import sys

        from grapharc.planner.approval_file import DEFAULT_TIMEOUT_SECONDS, file_approval

        def _announce(message: str) -> None:
            # Printed *and flushed* before the run parks: a terminal user (or a
            # log tailer) must learn how to answer without waiting for the exit.
            print(message, flush=True, file=sys.stdout)

        approval = file_approval(
            trace_path.parent,
            timeout_seconds=approval_timeout or DEFAULT_TIMEOUT_SECONDS,
            announce=_announce,
        )
    # The registry module's own loop builder wins; the incident demo's is the
    # fallback that keeps the default path byte-identical.
    build_loop = bundle.build_loop or incident_build_loop
    loop = build_loop(
        model,
        edge_policy=edge_policy,
        trace=trace,
        budget=Budget(max_tokens=max_tokens),
        limits=LoopLimits(max_rounds=max_rounds),
        registry=registry,
        state_schema=schema,
        writes=writes,
        approval=approval,
    )
    # `goal` is set when the schema has somewhere to put it; a custom schema is
    # not required to carry one, and the planner is told the goal regardless.
    initial = schema(goal=goal) if "goal" in schema.model_fields else schema()
    result = loop.run(goal, initial, run_id=run_id)

    rounds = [
        {
            "round": record.round,
            "status": record.admission.status.value if record.admission else "not_proposed",
            "nodes": record.proposal.node_count() if record.proposal else 0,
            "executed": record.executed,
            "rejections": [r.code for r in (record.rejections or ())],
        }
        for record in result.rounds
    ]
    payload = {
        "ok": result.succeeded,
        "command": "plan",
        "goal": goal,
        "model": model_description,
        "registry": registry_target,
        "kinds": sorted(registry.names()),
        "policy": policy_description,
        "policy_source": policy_source,
        "trace": str(trace_path),
        **settings.provenance(policy_source=policy_source),
        "stop": result.stop.value,
        "detail": result.detail,
        "rounds": rounds,
        "rejections": [r.code for r in result.rejections()],
        "state": result.state.model_dump() if hasattr(result.state, "model_dump") else result.state,
    }

    # The plain text of every line below is exactly what it was before colour
    # existed — the labels are still ten characters wide and the round rows still
    # pad the status to nine — because this block is the transcript printed in
    # `README.md`. On a terminal the colour carries the verdict: green admitted,
    # red rejected, and the stop reason in the same green only if the goal was met.
    stop_tint = style.ok if result.succeeded else style.warn
    lines = [
        style.kv("goal", goal, width=style.LABEL_WIDTH),
        style.kv("model", model_description, width=style.LABEL_WIDTH, tint=style.accent),
        style.kv("registry", registry_target, width=style.LABEL_WIDTH, tint=style.accent),
        style.kv(
            "kinds",
            ", ".join(sorted(registry.names())) or "(none)",
            width=style.LABEL_WIDTH,
        ),
        style.kv(
            "policy",
            f"{policy_description}  {style.dim(f'[{policy_source}]')}",
            width=style.LABEL_WIDTH,
        ),
        style.kv("config", settings.describe(), width=style.LABEL_WIDTH, tint=style.dim),
        "",
        style.kv(
            "stopped",
            f"{stop_tint(result.stop.value)}  {style.dim(f'({result.detail})')}",
            width=style.LABEL_WIDTH,
        ),
        style.kv("rounds", f"{len(rounds)} of max {max_rounds}", width=style.LABEL_WIDTH),
    ]
    for record in rounds:
        codes = ", ".join(record["rejections"])
        note = f"  {style.dim('rejected:')} {style.err(codes)}" if codes else ""
        # A round nobody proposed is neither: amber, not a green it did not earn.
        verdict = {"admitted": True, "rejected": False}.get(str(record["status"]))
        lines.append(
            f"   round {record['round']}: "
            f"{style.cell(str(record['status']), 9, tint=style.tint_for(verdict))} "
            f"{style.dim('nodes=')}{record['nodes']} "
            f"{style.dim('executed=')}{record['executed']}{note}"
        )
    lines += [
        "",
        style.kv("state", str(result.state), width=style.LABEL_WIDTH),
        style.kv("trace", str(trace_path), width=style.LABEL_WIDTH, tint=style.accent),
    ]

    emit(payload, lines, as_json=as_json)
    return EXIT_OK if result.succeeded else EXIT_FAILED


__all__ = ["DEFAULT_REGISTRY", "PlanSetupError", "plan", "resolve_edge_policy", "resolve_registry"]
