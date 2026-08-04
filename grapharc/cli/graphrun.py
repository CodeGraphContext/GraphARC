"""`grapharc run` — execute a topology *you* wrote, through the same gate.

The deterministic sibling of `grapharc plan`. `plan` asks a model for a
topology; this reads one from a file. Everything after that is identical: the
same `AdmissionChecker`, the same registry, the same policy, the same trace. A
hand-written graph earns no exemption — if your file names an unregistered kind
or an edge the policy denies, it is refused exactly as a planner's proposal
would be.

Two reasons this exists rather than being a footnote on `plan`:

- **A graph you wrote is the common case.** Most work has a shape you already
  know. Needing a model to state it was an accident of which command got built
  first.
- **`--check-only` turns admission into a linter.** Validate a topology against
  your policy without running it — the rejection list is already structured
  data with codes and remedies, so this is nearly free and it is the fastest way
  to find out that a workflow is illegal before anyone depends on it.

The file is JSON or TOML, and holds *only* topology:

    {"nodes": [{"name": "triage"}, {"name": "fix", "kind": "patch"}],
     "edges": [{"source": "__start__", "target": "triage"},
               {"source": "triage", "target": "fix"},
               {"source": "fix", "target": "__end__"}]}

It cannot hold a node body — `ProposedNode` forbids extra fields, and bodies
come from the registry. That is the same boundary a model is held to, for the
same reason: a file is authored by whoever can write to the directory.
"""

from __future__ import annotations

import json
import tempfile
import tomllib
import uuid
from pathlib import Path
from typing import Any

from grapharc.cli import style
from grapharc.cli.config import ConfigError
from grapharc.cli.config import load as load_settings
from grapharc.cli.generate import resolve_or_generate_policy
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.cli.plan import PlanSetupError, resolve_registry
from grapharc.cli.runid import refuse_reused_run_id

#: Stage names `demo` owns. Kept so `grapharc run stage0` — which worked before
#: the split — fails with a redirection rather than an argparse complaint about
#: a path that does not exist.
DEMO_HINTS = (
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "capstone",
)


def load_topology(path: Path) -> dict[str, Any]:
    """Read a topology file. JSON or TOML, chosen by suffix.

    TOML is offered because the rest of the project's operator-authored files
    are TOML and comments matter in a file that encodes a decision. JSON is
    offered because it is what a planner emits, so a proposal captured from a
    trace can be replayed as a file without translation.
    """
    if not path.is_file():
        raise PlanSetupError(f"no such graph file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # A topology saved as UTF-16, or truncated in transit, is a file we
        # cannot run — not a crash. `UnicodeDecodeError` is a `ValueError`, so
        # neither decoder below would ever have caught it.
        raise PlanSetupError(f"{path}: {exc}") from exc
    try:
        if path.suffix.lower() == ".toml":
            return tomllib.loads(text)
        return json.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PlanSetupError(f"{path}: {exc}") from exc


def build_proposal(document: dict[str, Any]) -> Any:
    """Turn the file's contents into a `Subgraph`, or say why it is not one.

    Validation is Pydantic's, not ours: `Subgraph` already forbids extra fields,
    refuses reserved names and rejects duplicates. Re-implementing any of that
    here would give a hand-written file a second, weaker set of rules.
    """
    from pydantic import ValidationError

    from grapharc.planner import Subgraph

    try:
        return Subgraph.model_validate(document)
    except ValidationError as exc:
        raise PlanSetupError(f"not a valid topology: {exc}") from exc


def run_graph(
    graph_path: str,
    *,
    registry_target: str | None = None,
    policy_path: Path | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
    max_seconds: float | None = None,
    max_concurrency: int | None = None,
    tenant: str | None = None,
    trace_path: Path | None = None,
    run_id: str | None = None,
    check_only: bool = False,
    config_path: Path | None = None,
    as_json: bool = False,
) -> int:
    """Admit and execute one hand-written topology. Returns the exit code."""
    from grapharc.observe.trace import TraceRecorder
    from grapharc.planner import MaterializationError, Materializer
    from grapharc.runtime.budget import Budget, BudgetMeter
    from grapharc.runtime.graph import RunContext

    if graph_path in DEMO_HINTS:
        return fail(
            f"{graph_path!r} is a built-in example, not a graph file — "
            f"run `grapharc demo {graph_path}` instead. `grapharc run` takes a "
            "topology file you wrote; see `grapharc run --help`.",
            as_json=as_json,
            command="run",
        )

    from grapharc.cli.plan import DEFAULT_REGISTRY

    try:
        settings = load_settings(config_path)
        registry_target = settings.resolve("registry", registry_target, DEFAULT_REGISTRY)
        policy_path = settings.resolve_path("policy", policy_path)
        tenant = settings.resolve("tenant", tenant, "default")
        max_tokens = settings.resolve("max_tokens", max_tokens)
        max_iterations = settings.resolve("max_iterations", max_iterations)
        max_seconds = settings.resolve("max_seconds", max_seconds)
        max_concurrency = settings.resolve("max_concurrency", max_concurrency)
        document = load_topology(Path(graph_path))
        proposal = build_proposal(document)
        bundle = resolve_registry(registry_target)
        registry, state_schema, writes = bundle.registry, bundle.state_schema, bundle.writes
        gate_policy, policy_description, policy_source = resolve_or_generate_policy(
            policy_path,
            tenant=tenant,
            fallback=bundle.default_policy,
            fallback_label=f"{registry_target} default",
        )
    except (ConfigError, PlanSetupError) as exc:
        return fail(str(exc), as_json=as_json, command="run", graph=graph_path)

    from grapharc.examples.plan_incident import IncidentState
    from grapharc.planner import AdmissionChecker

    schema = state_schema or IncidentState
    trace_path = trace_path or Path(tempfile.mkdtemp(prefix="grapharc-run-")) / "trace.jsonl"
    # Ahead of the recorder, and ahead of the admission check that writes the
    # first event: an id already in this file would merge this run with an
    # earlier one, and `--check-only` writes its verdict under that id too.
    reused = refuse_reused_run_id(
        trace_path, run_id, command="run", as_json=as_json, graph=graph_path
    )
    if reused is not None:
        return reused
    trace = TraceRecorder(trace_path)
    checker = AdmissionChecker(
        registry=registry,
        edge_policy=gate_policy.edge,
        node_policy=gate_policy.node,
        trace=trace,
    )

    # `Budget()` is genuinely unlimited on every dimension, so with no
    # ceilings the budget check passes anything. The meter is real only when
    # at least one dimension is set.
    budget = Budget(
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        max_seconds=max_seconds,
        max_concurrency=max_concurrency,
    )
    meter = BudgetMeter(budget)
    # D2: the admission event has to carry the run id the operator chose, or
    # `grapharc metrics <trace> <run-id>` finds nothing under that id.
    rid = run_id or uuid.uuid4().hex[:12]
    verdict = checker.check(
        proposal, meter=meter, ctx=RunContext(run_id=rid, graph="cli-run", meter=meter)
    )

    common = {
        "command": "run",
        "graph": graph_path,
        "registry": registry_target,
        "policy": policy_description,
        "policy_source": policy_source,
        "nodes": proposal.node_count(),
        "admitted": verdict.admitted,
        "fingerprint": verdict.fingerprint,
        "rejections": [
            {"code": r.code, "subject": r.subject, "detail": r.detail, "remedy": r.remedy}
            for r in verdict.rejections
        ],
        "run_id": rid,
        "max_tokens": max_tokens,
        "max_iterations": max_iterations,
        "max_seconds": max_seconds,
        "max_concurrency": max_concurrency,
        "trace": str(trace_path),
        **settings.provenance(policy_source=policy_source),
    }

    # `REFUSED` and `ADMITTED` are the whole answer, so on a terminal they carry
    # the only two colours that matter here. The words, the widths and the order
    # are untouched: `--check-only` is what CI runs, and CI reads text.
    width = style.LABEL_WIDTH
    header = [
        style.kv("graph", graph_path, width=width, tint=style.accent),
        style.kv("policy", policy_description, width=width),
    ]
    trace_line = style.kv("trace", str(trace_path), width=width, tint=style.accent)

    if not verdict.admitted:
        lines = [
            *header,
            "",
            style.kv(
                "REFUSED",
                f"{len(verdict.rejections)} objection(s)",
                width=width,
                key_tint=style.err,
                tint=style.err,
            ),
        ]
        lines += [
            f"   {style.cell(r.code, 18, tint=style.err)} {r.detail}" for r in verdict.rejections
        ]
        lines += ["", trace_line]
        emit({"ok": False, **common}, lines, as_json=as_json)
        return EXIT_FAILED

    # Built for BOTH paths, and before `--check-only` answers. Admission checks
    # the proposal; materialisation checks that it can become a graph at all —
    # no edge out of START, no nodes, a node unreachable from START all pass the
    # gate and fail here. A linter that says ADMITTED and then crashes on the
    # same file is worse than no linter.
    materializer = Materializer(
        registry=registry,
        state_schema=schema,
        writes=writes if writes is not None else None,
        trace=trace,
    )
    try:
        compiled = materializer.materialize(verdict, proposal)
    except MaterializationError as exc:
        lines = [
            *header,
            "",
            style.err("ADMITTED, BUT CANNOT BE BUILT"),
            f"   {exc}",
            "",
            trace_line,
        ]
        emit(
            {"ok": False, "buildable": False, "error": str(exc), **common},
            lines,
            as_json=as_json,
        )
        return EXIT_FAILED

    if check_only:
        lines = [
            *header,
            style.kv("nodes", str(proposal.node_count()), width=width),
            "",
            style.ok("ADMITTED") + style.dim(" and buildable. Nothing was run."),
            # Wider than the label column on purpose, and always has been: the
            # fingerprint is what a later run is compared against.
            style.kv("fingerprint", verdict.fingerprint, width=width, tint=style.accent),
        ]
        emit(
            {"ok": True, "checked_only": True, "buildable": True, **common},
            lines,
            as_json=as_json,
        )
        return EXIT_OK

    state = compiled.invoke(schema().model_dump(), run_id=rid)

    payload = {"ok": True, "checked_only": False, **common, "state": state}
    lines = [
        *header,
        style.kv("nodes", str(proposal.node_count()), width=width),
        "",
        style.ok("ADMITTED") + style.dim(" and executed."),
        style.kv("state", str(state), width=width),
        trace_line,
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK


__all__ = ["DEMO_HINTS", "build_proposal", "load_topology", "run_graph"]
