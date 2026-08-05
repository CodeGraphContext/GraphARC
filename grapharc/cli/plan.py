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
- `--policy PATH [--tenant NAME]` — a TOML document whose `edge` and `node`
  rules are compiled to the `EdgePolicy` and `NodePolicy` admission consults,
  via `PolicyEngine.edge_policy()` and `PolicyEngine.node_policy()`. This is the
  path that makes declarative governance constrain a run rather than answer
  questions about one.

Exit codes follow the CLI's convention: `0` the goal was met, `1` the run
stopped short for a recorded reason (refused, out of budget, out of rounds),
`2` it could not start.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grapharc.cli import style
from grapharc.cli.config import ConfigError, Settings
from grapharc.cli.config import load as load_settings
from grapharc.cli.generate import resolve_or_generate_policy
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.cli.runid import refuse_reused_run_id

DEFAULT_REGISTRY = "grapharc.examples.plan_incident:build_registry"
STDLIB_REGISTRY = "grapharc.stdlib:build_registry"
PLAN_FILENAME = "plan.json"


def resolve_registry_target(
    configured: str | None, *, scripted: bool, use_default: bool
) -> str:
    """Which registry governs this run, by one visible chain.

    Flag/config first, always. Then, unconfigured: a `registry.py` in this
    directory is *yours* and wins — that is what `grapharc init` scaffolds
    and what an operator expects to be running. `--default` forces the
    built-in general-purpose kinds past all of that. With nothing at all:
    the built-ins for real runs; the shipped rehearsal registry for
    `--scripted`, whose canned replies exist to demonstrate a refusal.
    """
    if use_default:
        return STDLIB_REGISTRY
    if configured:
        return configured
    if Path("registry.py").is_file():
        return "registry.py:build_registry"
    if scripted:
        return DEFAULT_REGISTRY
    return STDLIB_REGISTRY


def default_trace_path() -> Path:
    """`.grapharc/runs/<utc-stamp>-<6 hex>/trace.jsonl` under cwd.

    The same shape the Slack gate mints, and — deliberately — inside the
    directory `grapharc serve --live-root .grapharc/runs` serves, so a default
    run is watchable without anyone passing `--trace`. The random suffix keeps
    two plans started in the same second apart. Falls back to the old tempdir
    when cwd is unwritable: a read-only checkout must not make `plan` exit 2
    over a file the operator never asked for.
    """
    import uuid
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(".grapharc") / "runs" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(tempfile.mkdtemp(prefix="grapharc-plan-")) / "trace.jsonl"
    return run_dir / "trace.jsonl"


def watch_url(trace_path: Path, *, run_id: str | None = None, timeout: float = 0.25) -> str | None:
    """The live-view URL for this trace, when a server is up to serve it.

    `.grapharc/live-server.json` (written by `grapharc serve --live-root`)
    names the server; the URL is returned only when (a) the trace resolves
    under the marker's live root and (b) one loopback TCP connect — never
    slower than `timeout` — answers, so a marker left by a crashed server is
    harmless. None on every other path; the caller decides what hint to print
    instead, because a guessed URL is worse than an honest instruction.
    """
    import json
    import socket
    from urllib.parse import quote

    marker = Path(".grapharc") / "live-server.json"
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        # TypeError is in the net for the marker shapes JSON allows but this
        # code does not: a non-object document (indexing a list raises it) and
        # a null port (`int(None)`). A malformed marker must degrade to the
        # hint, never escape as a traceback from a command that only wanted to
        # print a courtesy URL.
        root = Path(record["live_root"])
        base = str(record["url"])
        host, port = str(record["host"]), int(record["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        # The marker's root is resolved by the serve that wrote it; resolve it
        # again here so a hand-edited or symlinked root still matches the same
        # directory instead of failing the lexical comparison.
        rel = trace_path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return None
    url = f"{base}/live/view?trace={quote(rel.as_posix(), safe='')}"
    if run_id:
        url += f"&run={quote(run_id)}"
    return url


def _marker_base() -> str:
    """The last-known server base URL, from the marker; the default otherwise.

    A marker that exists but whose server stopped answering still names the
    host and port the operator actually uses — an instruction quoting a
    different port than their `grapharc serve` command is a wrong instruction.
    Read with the same tolerance as `watch_url`: any defect means the default.
    """
    import json

    try:
        record = json.loads(
            (Path(".grapharc") / "live-server.json").read_text(encoding="utf-8")
        )
        return str(record["url"]).rstrip("/")
    except (OSError, ValueError, KeyError, TypeError):
        return "http://127.0.0.1:8000"


def watch_hint(trace_path: Path) -> str:
    """What to print when no live server answers: the command, then the URL.

    The user asked for the link to always exist — so when it cannot be exact,
    it is an instruction that produces the exact one.
    """
    base = _marker_base()
    try:
        rel = trace_path.resolve().relative_to((Path(".grapharc") / "runs").resolve())
        from urllib.parse import quote

        would_be = f"{base}/live/view?trace={quote(rel.as_posix(), safe='')}"
        return f"run `grapharc serve --live-root .grapharc/runs` then open {would_be}"
    except ValueError:
        return (
            f"run `grapharc serve --live-root {trace_path.parent}` "
            f"then open {base}/live"
        )


class PlanSetupError(Exception):
    """Raised before anything runs, so a bad flag never half-executes a plan."""


def _write_plan_file(run_dir: Path, *, goal, registry_target, model_spec, result) -> None:
    """Persist the admitted-but-unexecuted plan next to its trace.

    What `grapharc go` reads. The proposal is stored whole and re-judged by
    admission at execution time — a hand-edited plan.json is a new proposal,
    not a pre-approved one.
    """
    import json
    from datetime import UTC, datetime

    admitted = next(
        (r for r in reversed(result.rounds) if r.proposal is not None and r.admission
         and r.admission.status.value == "admitted"),
        None,
    )
    if admitted is None:
        return
    (run_dir / PLAN_FILENAME).write_text(
        json.dumps(
            {
                "goal": goal,
                "registry": registry_target,
                "model": model_spec,
                "fingerprint": admitted.proposal.fingerprint(),
                "proposal": admitted.proposal.model_dump(mode="json"),
                "planned_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def find_unexecuted_plan(runs_root: Path | None = None) -> Path | None:
    """The newest saved plan `go` has not executed yet, or None."""
    import json

    root = runs_root or Path(".grapharc") / "runs"
    candidates = sorted(root.glob(f"*/{PLAN_FILENAME}"), reverse=True)
    for candidate in candidates:
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not record.get("executed_run_id"):
            return candidate
    return None


def execute_plan(
    target: str | None,
    *,
    model_spec: str | None = None,
    model_arg_pairs: list[str] | None = None,
    workspace: Path | None = None,
    policy_path: Path | None = None,
    tenant: str | None = None,
    run_id: str | None = None,
    max_tokens: int | None = None,
    config_path: Path | None = None,
    as_json: bool = False,
) -> int:
    """`grapharc go [<run-dir>]` — execute a plan `grapharc plan` saved.

    The stored proposal is replayed through the full governed loop — a
    scripted planner whose one reply *is* the plan — so admission judges it
    again on the way in: what runs is what the gate admits now, not what a
    file claims was admitted before. Executing is the human approval; there
    is no second gate.
    """
    import json

    from grapharc.planner import LoopLimits
    from grapharc.runtime.budget import Budget
    from grapharc.testing import ScriptedChatModel

    if target is None:
        plan_file = find_unexecuted_plan()
        if plan_file is None:
            return fail(
                "no unexecuted plan under .grapharc/runs — run `grapharc plan "
                '"<goal>"` first, or pass a run directory',
                as_json=as_json,
                command="go",
                code=EXIT_FAILED,
            )
    else:
        candidate = Path(target)
        plan_file = candidate if candidate.name == PLAN_FILENAME else candidate / PLAN_FILENAME
        if not plan_file.is_file():
            return fail(
                f"no {PLAN_FILENAME} in {candidate} — `grapharc plan` writes one "
                "next to its trace",
                as_json=as_json,
                command="go",
            )

    try:
        record = json.loads(plan_file.read_text(encoding="utf-8"))
        from grapharc.planner import Subgraph

        proposal = Subgraph.model_validate(record["proposal"])
        goal = str(record.get("goal", ""))
        registry_target = str(record["registry"])
    except (OSError, ValueError, KeyError) as exc:
        return fail(f"unreadable plan file {plan_file}: {exc}", as_json=as_json, command="go")

    run_dir = plan_file.parent
    trace_path = run_dir / "trace.jsonl"

    try:
        settings = load_settings(config_path)
        model_spec = settings.resolve("model", model_spec, record.get("model"))
        tenant = settings.resolve("tenant", tenant, "default")
        max_tokens = settings.resolve("max_tokens", max_tokens, 100_000)
        model_args = _parse_model_args(model_arg_pairs)
        # The registry gets the real model (agent-backed kinds need one); the
        # PLANNER gets a stand-in whose only reply is the saved plan.
        model = None
        model_description = "none (deterministic bodies only)"
        if model_spec:
            from grapharc.gateway import get_model

            model = get_model(model_spec, **model_args)
            model_description = model_spec
        bundle = resolve_registry(registry_target, model, workspace=workspace)
        gate_policy, policy_description, policy_source = resolve_or_generate_policy(
            policy_path,
            tenant=tenant,
            model=None,  # executing a saved plan never generates policy
            goal=goal,
            catalog=bundle.registry.catalog(),
            mutating=bundle.mutating,
            fallback=bundle.default_policy,
            fallback_label=f"{registry_target} default",
            registry_target=registry_target,
        )
    except (ConfigError, PlanSetupError) as exc:
        return fail(str(exc), as_json=as_json, command="go", goal=goal)
    except Exception as exc:  # noqa: BLE001 — a backend that will not load is a setup failure
        return fail(f"could not load the plan's setup: {exc}", as_json=as_json, command="go")

    from grapharc.examples.plan_incident import IncidentState
    from grapharc.examples.plan_incident import build_loop as incident_build_loop
    from grapharc.observe.trace import TraceRecorder

    replay = {
        "nodes": [n.model_dump(mode="json", exclude={"subgraph"}) for n in proposal.nodes],
        "edges": [e.model_dump(mode="json") for e in proposal.edges],
        "rationale": proposal.rationale,
    }
    planner_model = ScriptedChatModel(responses=[json.dumps(replay)])
    schema = bundle.state_schema or IncidentState
    trace = TraceRecorder(trace_path)
    build_loop = bundle.build_loop or incident_build_loop
    loop = build_loop(
        planner_model,
        edge_policy=gate_policy.edge,
        node_policy=gate_policy.node,
        trace=trace,
        budget=Budget(max_tokens=max_tokens),
        limits=LoopLimits(max_rounds=1),
        registry=bundle.registry,
        state_schema=schema,
        writes=bundle.writes,
        approval=None,
    )
    initial = schema(goal=goal) if "goal" in schema.model_fields else schema()
    result = loop.run(goal, initial, run_id=run_id)

    executed = any(r.executed for r in result.rounds)
    if executed:
        record["executed_run_id"] = result.run_id
        plan_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    url = watch_url(trace_path, run_id=result.run_id)
    payload = {
        "ok": executed,
        "command": "go",
        "goal": goal,
        "plan": str(plan_file),
        "registry": registry_target,
        "model": model_description,
        "policy": policy_description,
        "policy_source": policy_source,
        "stop": result.stop.value,
        "detail": result.detail,
        "executed": executed,
        "run_id": result.run_id,
        "trace": str(trace_path),
        "watch_url": url,
        "state": result.state.model_dump() if hasattr(result.state, "model_dump") else result.state,
    }
    stop_tint = style.ok if executed else style.warn
    lines = [
        style.kv("plan", str(plan_file), width=style.LABEL_WIDTH),
        style.kv("goal", goal, width=style.LABEL_WIDTH),
        style.kv("model", model_description, width=style.LABEL_WIDTH, tint=style.accent),
        style.kv("registry", registry_target, width=style.LABEL_WIDTH, tint=style.accent),
        style.kv(
            "policy",
            f"{policy_description}  {style.dim(f'[{policy_source}]')}",
            width=style.LABEL_WIDTH,
        ),
        "",
        style.kv(
            "stopped",
            f"{stop_tint(result.stop.value)}  {style.dim(f'({result.detail})')}",
            width=style.LABEL_WIDTH,
        ),
        style.kv("state", str(result.state), width=style.LABEL_WIDTH),
        style.kv("trace", str(trace_path), width=style.LABEL_WIDTH, tint=style.accent),
        style.kv(
            "watch",
            url if url else style.dim(watch_hint(trace_path)),
            width=style.LABEL_WIDTH,
            tint=style.accent if url else None,
        ),
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK if executed else EXIT_FAILED


def _registry_module(target: str) -> tuple[Any, str]:
    """The module a `--registry` target names, and the attribute after the colon.

    Two forms, one rule — before the colon: an importable module name, or a
    `.py` file; after it: the attribute. The path form (detected by a `.py`
    suffix or a path separator) is what `grapharc init`'s scaffold uses:
    `registry.py:build_registry`, cwd-relative like everything else here.

    Loaded via `spec_from_file_location` under a deterministic private name
    derived from the resolved path, registered in `sys.modules` *before*
    `exec_module` (pydantic model creation looks the module up by name
    mid-exec), and reused from `sys.modules` on a second call — which is what
    keeps `resolve_registry` and `_model_for` looking at one module object.
    `sys.path` is never touched: the file may import installed packages but
    not unlisted siblings; a package of kinds should use `module:attr`.
    """
    import hashlib
    import importlib.util
    import os
    import sys

    module_name, separator, attribute = target.partition(":")
    if not separator or not attribute:
        raise PlanSetupError(
            f"--registry expects module:attr or path/to/file.py:attr, got {target!r}"
        )
    if not (module_name.endswith(".py") or os.sep in module_name):
        try:
            return importlib.import_module(module_name), attribute
        except ImportError as exc:
            raise PlanSetupError(f"--registry {target!r}: {exc}") from exc

    path = Path(module_name)
    if not path.is_file():
        raise PlanSetupError(f"--registry {target!r}: no such file: {path}")
    resolved = path.resolve()
    private = f"_grapharc_registry_{hashlib.sha1(str(resolved).encode()).hexdigest()[:12]}"
    cached = sys.modules.get(private)
    if cached is not None:
        return cached, attribute
    spec = importlib.util.spec_from_file_location(private, resolved)
    if spec is None or spec.loader is None:
        raise PlanSetupError(f"--registry {target!r}: could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — user code failing at import is a setup failure
        sys.modules.pop(private, None)
        raise PlanSetupError(f"--registry {target!r}: {exc}") from exc
    return module, attribute


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


def resolve_registry(
    target: str, model: Any = None, *, workspace: Path | None = None
) -> RegistryBundle:
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
    module, attribute = _registry_module(target)
    registry = getattr(module, attribute, None)
    if registry is None:
        raise PlanSetupError(f"--registry {target!r}: {module.__name__} has no {attribute!r}")
    if callable(registry):
        kwargs: dict[str, Any] = {}
        if workspace is not None:
            # Fail closed, never silently un-confined: a factory that has no
            # `workspace` parameter cannot honour the flag, and running anyway
            # would leave the tools on the process cwd the operator asked to
            # leave behind.
            if not _accepts_keyword(registry, "workspace"):
                raise PlanSetupError(
                    f"--workspace: registry {target!r} does not accept a workspace"
                )
            kwargs["workspace"] = workspace
        registry = (
            registry(model, **kwargs) if _accepts_an_argument(registry) else registry(**kwargs)
        )
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


def _accepts_keyword(factory: Any, name: str) -> bool:
    """Whether `factory` can take `name` as a keyword argument."""
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def _parse_model_args(pairs: list[str] | None) -> dict[str, Any]:
    """`--model-arg KEY=VALUE` pairs into backend-constructor kwargs.

    Values are read as JSON first (`temperature=0` is the number zero, not the
    string), falling back to the raw string. A pair with no `=` is a setup
    error before anything runs.
    """
    import json as _json

    kwargs: dict[str, Any] = {}
    for pair in pairs or []:
        key, separator, raw = pair.partition("=")
        if not separator or not key:
            raise PlanSetupError(f"--model-arg expects KEY=VALUE, got {pair!r}")
        try:
            kwargs[key] = _json.loads(raw)
        except ValueError:
            kwargs[key] = raw
    return kwargs


@dataclass(frozen=True)
class GatePolicy:
    """The gate objects one policy document compiles to, travelling together.

    Two halves of the same file: `edge` says what may be wired, `node` says
    which kinds may run at all. They travel as one object because a run gated by
    one half of a document is exactly the failure this exists to fix — the node
    half used to be compiled by nobody, so a `deny` rule over a node kind was
    text (issue #66).

    `node` is `None` when the document declares no `node` rules. That is not the
    same as an empty `NodePolicy`, which denies every kind: a document that says
    nothing about nodes leaves them to the registry, which is itself an
    allowlist with no wildcard, while a document that mentions nodes at all is
    taken at its word, default included.
    """

    edge: Any
    node: Any = None


def resolve_policy(policy_path: Path | None, *, tenant: str) -> tuple[GatePolicy, str]:
    """Compile a policy document's edge and node rules, or fall back to the demo's.

    Returns `(gate_policy, description)`. The description is printed, because
    which policy a run was subject to is the first thing anyone asks afterwards —
    and it counts both kinds of rule, so a document whose node rules do not
    constrain this run says so in the count rather than by omission.
    """
    if policy_path is None:
        from grapharc.examples.plan_incident import default_edge_policy

        return (
            GatePolicy(edge=default_edge_policy()),
            "built-in demo (deny -> deploy, allow otherwise)",
        )
    from grapharc.policy import PolicyEngine, PolicyError

    try:
        engine = PolicyEngine.from_file(policy_path)
    except (OSError, PolicyError) as exc:
        raise PlanSetupError(f"--policy {str(policy_path)!r}: {exc}") from exc
    policy = compile_policy(engine, tenant=tenant)
    # Both counts are of the rules that survived tenant scoping, which is what
    # this run is actually subject to.
    description = (
        f"{policy_path} (tenant {tenant!r}, {len(policy.edge.rules)} edge rule(s), "
        f"{0 if policy.node is None else len(policy.node.rules)} node rule(s))"
    )
    return policy, description


def compile_policy(engine: Any, *, tenant: str) -> GatePolicy:
    """Compile a loaded `PolicyEngine`'s planner-side rules into both gates.

    The one place a document becomes admission's objects, so a run started from
    a file and a run started from a document generated in memory are subject to
    the same reading of the same rules.

    The node half is compiled **only when the document declares node rules**.
    `PolicyEngine.node_policy()` is faithful to `check_node`, which means a
    document with no node rules and a `deny` default denies every kind — a true
    reading of the engine, and not what an operator who wrote only edge rules
    meant to say. Saying nothing about nodes leaves them where they were: the
    registry, an allowlist with no wildcard.
    """
    from grapharc.policy import ResourceKind

    node_rules = engine.document.rules_for(ResourceKind.NODE)
    return GatePolicy(
        edge=engine.edge_policy(tenant=tenant),
        node=engine.node_policy(tenant=tenant) if node_rules else None,
    )


def _model_for(
    spec: str | None,
    registry_target: str = DEFAULT_REGISTRY,
    model_args: dict[str, Any] | None = None,
) -> tuple[Any, str]:
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

        module, _ = _registry_module(registry_target)
        replies = getattr(module, "scripted_planner_replies", None)
        if replies is None:
            from grapharc.examples.plan_incident import scripted_planner_replies as replies

        return ScriptedChatModel(responses=replies()), "scripted stand-in (--scripted)"
    from grapharc.gateway import get_model

    return get_model(spec, **(model_args or {})), spec


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
    max_planning_failures: int | None = None,
    workspace: Path | None = None,
    model_arg_pairs: list[str] | None = None,
    config_path: Path | None = None,
    settings: Settings | None = None,
    approve: bool = False,
    approval_timeout: float | None = None,
    as_json: bool = False,
    command: str = "plan",
    scripted: bool = False,
    go_after: bool = False,
    use_default_registry: bool = False,
) -> int:
    """Run one governed planning loop against `goal`. Returns the exit code.

    A real command runs a real model, full stop: with no `--model` (and none
    in `grapharc.toml`), both `plan` and `go` refuse before anything happens.
    The one exception is opt-in and named out loud — `plan --scripted` runs
    the registry's canned planner replies so the machinery (admission,
    refusal, approval, the live page) can be rehearsed for free. `go` has no
    such flag: go means do.
    """
    from grapharc.observe.trace import TraceRecorder
    from grapharc.planner import LoopLimits, LoopStop
    from grapharc.runtime.budget import Budget

    trace_path = trace_path or default_trace_path()

    # Before the setup, because this one is about the file the setup would start
    # writing into: a run id already in that file merges this plan with an
    # earlier one under a single name.
    reused = refuse_reused_run_id(
        trace_path, run_id, command=command, as_json=as_json, goal=goal
    )
    if reused is not None:
        return reused

    # Everything that can be wrong about the setup is decided before a model is
    # asked anything, so a bad flag cannot half-execute a plan.
    try:
        if settings is None:
            settings = load_settings(config_path)
        model_spec = settings.resolve("model", model_spec)
        registry_target = resolve_registry_target(
            settings.resolve("registry", registry_target, None),
            scripted=scripted,
            use_default=use_default_registry,
        )
        if scripted and model_spec:
            raise PlanSetupError(
                "--scripted and --model contradict each other: the scripted "
                "rehearsal never calls a model — drop one of the two"
            )
        if not model_spec and not scripted:
            if command == "go":
                raise PlanSetupError(
                    "go plans with a real model and there is no scripted "
                    "fallback: pass --model SPEC (see `grapharc models --check` "
                    "for what this machine can use)"
                )
            raise PlanSetupError(
                "plan needs a real model: pass --model SPEC (see `grapharc "
                "models --check`), set `model` in grapharc.toml — or add "
                "--scripted to rehearse the machinery with the registry's "
                "built-in stand-in planner (free, deterministic, no AI)"
            )
        policy_path = settings.resolve_path("policy", policy_path)
        tenant = settings.resolve("tenant", tenant, "default")
        max_rounds = settings.resolve("max_rounds", max_rounds, 8)
        max_tokens = settings.resolve("max_tokens", max_tokens, 100_000)
        max_planning_failures = settings.resolve(
            "max_planning_failures", max_planning_failures, 3
        )
        if workspace is not None:
            workspace = Path(workspace).resolve()
            if not workspace.is_dir():
                raise PlanSetupError(f"--workspace: not a directory: {workspace}")
        model_args = _parse_model_args(model_arg_pairs)
        model, model_description = _model_for(model_spec, registry_target, model_args)
        bundle = resolve_registry(registry_target, model, workspace=workspace)
        registry, state_schema, writes = bundle.registry, bundle.state_schema, bundle.writes
        gate_policy, policy_description, policy_source = resolve_or_generate_policy(
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
            registry_target=registry_target,
        )
    except (ConfigError, PlanSetupError) as exc:
        return fail(str(exc), as_json=as_json, command=command, goal=goal)
    except Exception as exc:  # noqa: BLE001 — a backend that will not load is a setup failure
        return fail(f"could not build the plan: {exc}", as_json=as_json, command=command, goal=goal)

    from grapharc.examples.plan_incident import IncidentState
    from grapharc.examples.plan_incident import build_loop as incident_build_loop

    schema = state_schema or IncidentState
    trace = TraceRecorder(trace_path)
    approval = None
    if approve:
        import sys

        from grapharc.planner.approval_file import DEFAULT_TIMEOUT_SECONDS, file_approval

        watch_shown = False

        def _announce(message: str) -> None:
            # Printed *and flushed* before the run parks: a terminal user (or a
            # log tailer) must learn how to answer without waiting for the exit.
            # Silent in JSON mode: stdout there carries exactly one document, and
            # a notice printed ahead of it makes the whole output unparseable.
            if as_json:
                return
            # The live link first, once: the page is where the parked proposal
            # is drawn, and it should be open while the human decides.
            nonlocal watch_shown
            if not watch_shown:
                watch_shown = True
                url = watch_url(trace_path, run_id=run_id)
                if url:
                    print(
                        style.kv("watch", url, width=style.LABEL_WIDTH, tint=style.accent),
                        flush=True,
                        file=sys.stdout,
                    )
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
        edge_policy=gate_policy.edge,
        node_policy=gate_policy.node,
        trace=trace,
        budget=Budget(max_tokens=max_tokens),
        limits=LoopLimits(
            max_rounds=max_rounds,
            max_consecutive_planning_failures=max_planning_failures,
        ),
        registry=registry,
        state_schema=schema,
        writes=writes,
        approval=approval,
    )
    # `goal` is set when the schema has somewhere to put it; a custom schema is
    # not required to carry one, and the planner is told the goal regardless.
    initial = schema(goal=goal) if "goal" in schema.model_fields else schema()
    # `plan` plans; `go` (and `plan --go`) executes. The attribute rather
    # than a ctor param keeps every registry module's build_loop signature.
    loop.plan_only = command == "plan" and not go_after
    result = loop.run(goal, initial, run_id=run_id)

    if result.stop is LoopStop.PLANNED:
        _write_plan_file(
            trace_path.parent,
            goal=goal,
            registry_target=registry_target,
            model_spec=model_spec,
            result=result,
        )

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
        "ok": result.succeeded or result.stop is LoopStop.PLANNED,
        "command": command,
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
    # The live link, always: the exact URL when a serve is discoverable and
    # reachable, otherwise the one-line instruction that produces it. Filtered
    # out (with the trace line) by the README byte-comparison, which is why it
    # may vary per machine.
    url = watch_url(trace_path, run_id=run_id)
    payload["watch_url"] = url
    lines += [
        "",
        style.kv("state", str(result.state), width=style.LABEL_WIDTH),
        style.kv("trace", str(trace_path), width=style.LABEL_WIDTH, tint=style.accent),
        style.kv(
            "watch",
            url if url else style.dim(watch_hint(trace_path)),
            width=style.LABEL_WIDTH,
            tint=style.accent if url else None,
        ),
    ]
    if result.stop is LoopStop.PLANNED:
        payload["plan_file"] = str(trace_path.parent / PLAN_FILENAME)
        lines += [
            style.kv(
                "execute",
                f"grapharc go   (or: grapharc go {trace_path.parent})",
                width=style.LABEL_WIDTH,
                tint=style.accent,
            ),
        ]

    emit(payload, lines, as_json=as_json)
    done = result.succeeded or result.stop is LoopStop.PLANNED
    return EXIT_OK if done else EXIT_FAILED


__all__ = [
    "DEFAULT_REGISTRY",
    "GatePolicy",
    "PlanSetupError",
    "compile_policy",
    "plan",
    "resolve_policy",
    "resolve_registry",
]
