"""`grapharc init` — scaffold a working directory an operator can grow.

Three artifacts, all in cwd: a heavily-commented `registry.py` (the authoring
surface — node kinds, bodies, write permissions, edge policy, goal check), a
`grapharc.toml` pointing `plan`/`go` at it, and the `.grapharc/runs/` directory
the live server serves. The templates are string constants rather than packaged
data files, so nothing new can fall out of the wheel.

Refusal semantics: never overwrite, no `--force`. The two files are *authored*
surfaces — the registry is code that will execute — and a scaffold that can
silently replace an operator's kinds is the one failure mode this command must
not have. `.grapharc/runs/` is ensured unconditionally: it is run output, not
authorship, and re-running `init` after a `serve` must not fail on it.
"""

from __future__ import annotations

from pathlib import Path

from grapharc.cli import style
from grapharc.cli.output import EXIT_OK, EXIT_UNAVAILABLE, emit

REGISTRY_FILENAME = "registry.py"
CONFIG_FILENAME = "grapharc.toml"
RUNS_DIR = Path(".grapharc") / "runs"

REGISTRY_TEMPLATE = '''\
"""Your GraphARC registry — the node kinds a planner may propose.

Read this top to bottom once; it is the whole authoring surface.

The contract, in one paragraph: a *registry* is an allowlist of node kinds.
Each kind owns its body (a Python function YOU wrote), a description the
planning model reads, and a worst-case cost the admission gate budgets
against. When you run `grapharc plan "some goal"`, a model proposes a graph —
which kinds, wired how — and a deterministic checker admits or refuses it
against this file and your policy. A proposal names kinds; it carries no
code, no arguments, no paths. Absence from this list is refusal; there is no
wildcard.

Try it now, before editing anything (free — `--scripted` swaps the model
for this file's own canned planner replies, so no AI is involved):

    grapharc plan "review what is in this directory" --scripted

Round 1 proposes the `apply` kind below and is refused (the edge policy
denies it); round 2 replans without it and is admitted — then everything
STOPS, planned but unexecuted. Look at it, then execute it:

    grapharc go

That refusal is the point of the whole tool; keep one dangerous kind around
so you can watch the gate work.

Then make it yours: rename the kinds, rewrite the bodies, grow the State.
`grapharc.toml` next door points `plan` at this file.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel

from grapharc.harness.permissions import Decision
from grapharc.observe.trace import TraceRecorder
from grapharc.planner import (
    AdmissionChecker,
    AdmissionLimits,
    CostEstimate,
    EdgePolicy,
    EdgeRule,
    GovernedLoop,
    LoopLimits,
    Materializer,
    NodeRegistry,
    NodeSpec,
    PlannerNode,
)
from grapharc.runtime.budget import Budget

# ── 1. State ────────────────────────────────────────────────────────────────
# One typed contract for the whole run, however the topology changes between
# rounds. Every field a node writes must be declared here AND granted in
# WRITES below — a kind nobody declared writes for may write nothing.


class State(BaseModel):
    goal: str = ""          # filled from the CLI argument; the planner reads it
    # `notes` is a REDUCER (Annotated + operator.add): each writer returns just
    # its own lines and LangGraph merges them, so two nodes — or two
    # planner-named instances of ONE kind — may write it in the same parallel
    # step. A plain `list[str]` here crashes the first time a planner runs two
    # writers concurrently (InvalidUpdateError); keep the pattern for any field
    # more than one node may write.
    notes: Annotated[list[str], operator.add] = []
    report: str = ""        # the deliverable; the goal check below watches it


# ── 2. Bodies ───────────────────────────────────────────────────────────────
# Operator code. The planner never sees these and can never supply one — the
# registry is the only source of a body. Bodies read `state` and return a
# dict of the fields they write.


def _gather(state: State) -> dict:
    """Collect raw material — really. Lists this directory so the report node
    has actual evidence; replace with your own reading/searching code."""
    from pathlib import Path as _Path

    root = _Path.cwd()
    entries = sorted(p for p in root.iterdir() if not p.name.startswith("."))
    dirs = [p.name + "/" for p in entries if p.is_dir()]
    files = [p.name for p in entries if p.is_file()]
    note = (
        f"gather: {root.name}/ holds {len(dirs)} director(y/ies) "
        f"({', '.join(dirs[:12]) or 'none'}) and {len(files)} file(s) "
        f"({', '.join(files[:12]) or 'none'})"
    )
    return {"notes": [note]}


def _analyse(state: State) -> dict:
    """Work over what gather found. Deterministic on purpose: cheap, testable."""
    from pathlib import Path as _Path

    suffixes: dict[str, int] = {}
    for p in _Path.cwd().rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            suffixes[p.suffix or "(none)"] = suffixes.get(p.suffix or "(none)", 0) + 1
    top = sorted(suffixes.items(), key=lambda kv: -kv[1])[:6]
    note = "analyse: file types by count — " + ", ".join(
        f"{ext} x{count}" for ext, count in top
    )
    return {"notes": [note]}


def _report_for(model: Any):
    """The one body with a model in it, built per-run so it closes over the
    run's own backend. With no real model (scripted runs) it says so honestly
    instead of pretending — and still writes `report`, so the goal check
    passes and a free run completes end to end."""

    def body(state: State) -> dict:
        # A scripted double is the free path's planner, not a writer — its
        # replies are plans, and asking it for prose exhausts the script.
        scripted = "Scripted" in type(model).__name__
        if model is None or scripted or not hasattr(model, "invoke"):
            return {
                "report": "report: run with --model SPEC for a model-written report",
                "notes": ["report: written without a model"],
            }
        try:
            reply = model.invoke(
                f"Goal: {state.goal!r}. Evidence:\\n" + "\\n".join(state.notes)
                + "\\n\\nWrite one short report that satisfies the goal, grounded "
                "only in the evidence above."
            )
            text = str(getattr(reply, "content", reply)).strip()[:2000]
        except Exception as exc:  # a failed call is a note, not a crash
            text = f"report: model call failed ({exc}); notes stand"
        return {"report": text, "notes": ["report: written"]}

    return body


def _apply(state: State) -> dict:
    """The deliberately dangerous kind: in your registry this would change
    things — write files, call APIs, deploy. It is REGISTERED (a planner may
    propose it) and DENIED by the edge policy below (no admitted graph may
    reach it) until you decide otherwise. Keep the pattern even after you
    rename it: a gate with nothing to refuse proves nothing."""
    return {"notes": ["apply: this should not have run"]}


# ── 3. Write permissions ────────────────────────────────────────────────────
# Enforced by the materializer, not trusted to the bodies. A body that writes
# an unlisted field is refused at build time.

WRITES: dict[str, set[str]] = {
    "gather": {"notes"},
    "analyse": {"notes"},
    "report": {"report", "notes"},
    "apply": {"notes"},
}

# ── 4. The registry ─────────────────────────────────────────────────────────
# `description` is what the planning model reads when choosing kinds — write
# it for the model. `worst_case` is what admission budgets against — estimate
# it honestly; a lowball here is a budget the gate cannot keep.


def build_registry(model: Any = None) -> NodeRegistry:
    """Accepting `model` is the CLI's contract: a factory with a positional
    parameter is handed the run's backend."""

    def factory(build: Any) -> Any:
        # `build.kind` is what this registry licensed; `build.name` is the
        # instance name the planner chose. Behaviour keys on the kind.
        if build.kind == "report":
            return _report_for(model)
        return {"gather": _gather, "analyse": _analyse, "apply": _apply}[build.kind]

    return NodeRegistry(
        [
            NodeSpec(
                name="gather",
                description="collect raw material for the goal",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=400),
            ),
            NodeSpec(
                name="analyse",
                description="work over what gather collected",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=800),
            ),
            NodeSpec(
                name="report",
                description="write the deliverable from the notes; the run is "
                "complete once this has run",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=2500),
            ),
            NodeSpec(
                name="apply",
                description="act on the report (changes things)",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=500),
            ),
        ]
    )


# ── 5. Edge policy ──────────────────────────────────────────────────────────
# What may be wired to what when no --policy document is given. Deny beats
# allow. The one rule below is the demo refusal; delete it the day you mean it.


def default_edge_policy() -> EdgePolicy:
    return EdgePolicy(
        rules=(
            EdgeRule(action=Decision.DENY, target="apply"),  # nothing reaches apply
            EdgeRule(action=Decision.ALLOW),                 # everything else may flow
        )
    )


# ── 6. Dangerous kinds ──────────────────────────────────────────────────────
# Read by the policy generator on a first --model run so a generated policy
# knows what is worth denying. Empty means "nothing here mutates" — say so
# only if it is true.

MUTATING_KINDS: tuple[str, ...] = ("apply",)

# ── 7. The scripted planner ─────────────────────────────────────────────────
# What `grapharc plan` uses when no --model is given, so the free path
# exercises this registry — refusal included — and spends nothing.


def scripted_planner_replies() -> list[str]:
    import json

    from grapharc.runtime.graph import END, START

    def chain(*kinds: str) -> str:
        stops = [START, *kinds, END]
        return json.dumps(
            {
                "nodes": [{"name": k} for k in kinds],
                "edges": [
                    {"source": a, "target": b}
                    for a, b in zip(stops, stops[1:], strict=False)
                ],
            }
        )

    # Round 1 reaches for `apply` and is refused; round 2 replans without it.
    return [chain("gather", "apply"), chain("gather", "analyse", "report")]


# ── 8. The loop ─────────────────────────────────────────────────────────────
# Owning build_loop is what gives this file its own goal check. Without one,
# `grapharc plan` falls back to the incident demo's check (len(notes) >= 3) —
# a documented trap for any state that counts differently.


def build_loop(
    model,
    *,
    edge_policy=None,
    node_policy=None,
    trace: TraceRecorder | None = None,
    budget: Budget | None = None,
    limits: LoopLimits | None = None,
    registry: NodeRegistry | None = None,
    state_schema=None,
    writes=None,
    approval=None,
) -> GovernedLoop:
    registry = registry or build_registry(model)
    registry.freeze()  # the same object every round
    edge_policy = edge_policy or default_edge_policy()
    return GovernedLoop(
        planner=PlannerNode(
            model,
            name="planner",
            catalog=registry.catalog(),
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
        ),
        checker=AdmissionChecker(
            registry=registry,
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
            limits=AdmissionLimits(require_entry=True),
        ),
        materializer=Materializer(
            registry=registry,
            state_schema=state_schema or State,
            writes=writes if writes is not None else WRITES,
            trace=trace,
        ),
        budget=budget,
        limits=limits,
        trace=trace,
        name="my_loop",
        # Deterministic code, never a model: "am I done" is exactly the
        # question a run must not talk itself into answering yes.
        goal_reached=lambda state: bool(getattr(state, "report", "")),
        approval=approval,
    )


STATE_SCHEMA = State
'''

CONFIG_TEMPLATE = '''\
# grapharc.toml — defaults for the flags nobody wants to retype.
# Read from THIS directory only; parent directories are never searched, so a
# run cannot be governed by a file you did not know about. Flags beat
# environment (GRAPHARC_*) beat this file beat built-ins.

[grapharc]
# The node kinds a planner may propose: a module:attr, or a .py file next to
# this config (cwd-relative — run grapharc from this directory).
registry = "registry.py:build_registry"

# Uncomment to plan with a real model instead of the free scripted planner.
# `grapharc models --check` shows what this machine can use.
# model = "ollama/qwen3:8b"

# Uncomment to pin a policy document instead of the registry's own default
# (or the one a first --model run generates into .grapharc/).
# policy = "policy.toml"

# max_rounds = 8
# max_tokens = 100000
'''


def init(*, as_json: bool = False) -> int:
    """Scaffold a working directory: registry.py, grapharc.toml, .grapharc/runs/."""
    registry = Path(REGISTRY_FILENAME)
    config = Path(CONFIG_FILENAME)

    existing = [str(p) for p in (registry, config) if p.exists()]
    if existing:
        message = (
            f"refusing to overwrite: {', '.join(existing)} — move it aside, "
            "or start in an empty directory"
        )
        if as_json:
            emit({"ok": False, "command": "init", "error": message}, [], as_json=True)
        else:
            import sys

            print(f"error: {message}", file=sys.stderr)
        return EXIT_UNAVAILABLE  # exit 2: could not run at all

    registry.write_text(REGISTRY_TEMPLATE, encoding="utf-8")
    config.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "ok": True,
        "command": "init",
        "registry": str(registry),
        "config": str(config),
        "runs": str(RUNS_DIR),
    }
    width = style.LABEL_WIDTH
    lines = [
        style.kv(
            "wrote",
            f"{registry}  (your node kinds — open it, it is heavily commented)",
            width=width,
            tint=style.accent,
        ),
        style.kv(
            "wrote",
            f"{config}  (points plan at registry.py:build_registry)",
            width=width,
            tint=style.accent,
        ),
        style.kv(
            "created",
            f"{RUNS_DIR}/  (traces land here; serve it as the live root)",
            width=width,
        ),
        "",
        style.kv(
            "next",
            'grapharc plan "review what is in this directory" --scripted',
            width=width,
        ),
        style.kv("", "grapharc go   (executes the plan it saved)", width=width),
        style.kv("", f"grapharc serve --live-root {RUNS_DIR}   (second terminal)", width=width),
        style.kv("", 'grapharc go "your real goal" --model ollama/qwen3:8b', width=width),
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK


__all__ = ["CONFIG_TEMPLATE", "REGISTRY_TEMPLATE", "init"]
