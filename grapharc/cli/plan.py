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
from pathlib import Path
from typing import Any

from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail

DEFAULT_REGISTRY = "grapharc.examples.plan_incident:build_registry"


class PlanSetupError(Exception):
    """Raised before anything runs, so a bad flag never half-executes a plan."""


def resolve_registry(target: str) -> tuple[Any, Any, dict[str, set[str]] | None]:
    """Import `module:attr` and return `(registry, state_schema, writes)`.

    A callable attribute is called, so both `mypkg:build_registry` and
    `mypkg:REGISTRY` work; a `NodeRegistry` is not callable, so the two cannot
    be confused. Mirrors `grapharc.cli.serve.resolve_registry` deliberately —
    two flags that look the same should behave the same.

    A registry alone is not enough to run a loop: the kinds have to agree with a
    state schema, and each kind needs a declared write set or it may write
    nothing. So the module may also export `STATE_SCHEMA` and `WRITES`, and a
    module that exports neither gets the demo's — which is right only when its
    kinds write `notes`. Returning them together keeps a custom registry from
    being silently paired with a schema that has no field its nodes can reach.
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
    registry = registry() if callable(registry) else registry
    return registry, getattr(module, "STATE_SCHEMA", None), getattr(module, "WRITES", None)


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


def _model_for(spec: str | None) -> tuple[Any, str]:
    """The scripted planner by default; a real backend when asked for one."""
    if spec is None:
        from grapharc.examples.plan_incident import scripted_planner_replies
        from grapharc.testing import ScriptedChatModel

        return ScriptedChatModel(responses=scripted_planner_replies()), "scripted"
    from grapharc.gateway import get_model

    return get_model(spec), spec


def plan(
    goal: str,
    *,
    model_spec: str | None = None,
    registry_target: str = DEFAULT_REGISTRY,
    policy_path: Path | None = None,
    tenant: str = "default",
    trace_path: Path | None = None,
    run_id: str | None = None,
    max_rounds: int = 8,
    max_tokens: int = 100_000,
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
        registry, state_schema, writes = resolve_registry(registry_target)
        edge_policy, policy_description = resolve_edge_policy(policy_path, tenant=tenant)
        model, model_description = _model_for(model_spec)
    except PlanSetupError as exc:
        return fail(str(exc), as_json=as_json, command="plan", goal=goal)
    except Exception as exc:  # noqa: BLE001 — a backend that will not load is a setup failure
        return fail(f"could not build the plan: {exc}", as_json=as_json, command="plan", goal=goal)

    from grapharc.examples.plan_incident import IncidentState, build_loop

    schema = state_schema or IncidentState
    trace = TraceRecorder(trace_path)
    loop = build_loop(
        model,
        edge_policy=edge_policy,
        trace=trace,
        budget=Budget(max_tokens=max_tokens),
        limits=LoopLimits(max_rounds=max_rounds),
        registry=registry,
        state_schema=schema,
        writes=writes,
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
        "trace": str(trace_path),
        "stop": result.stop.value,
        "detail": result.detail,
        "rounds": rounds,
        "rejections": [r.code for r in result.rejections()],
        "state": result.state.model_dump() if hasattr(result.state, "model_dump") else result.state,
    }

    lines = [
        f"goal      : {goal}",
        f"model     : {model_description}",
        f"registry  : {registry_target}",
        f"kinds     : {', '.join(sorted(registry.names())) or '(none)'}",
        f"policy    : {policy_description}",
        "",
        f"stopped   : {result.stop.value}  ({result.detail})",
        f"rounds    : {len(rounds)} of max {max_rounds}",
    ]
    for record in rounds:
        note = f"  rejected: {', '.join(record['rejections'])}" if record["rejections"] else ""
        lines.append(
            f"   round {record['round']}: {record['status']:<9} "
            f"nodes={record['nodes']} executed={record['executed']}{note}"
        )
    lines += [
        "",
        f"state     : {result.state}",
        f"trace     : {trace_path}",
    ]

    emit(payload, lines, as_json=as_json)
    return EXIT_OK if result.succeeded else EXIT_FAILED


__all__ = ["DEFAULT_REGISTRY", "PlanSetupError", "plan", "resolve_edge_policy", "resolve_registry"]
