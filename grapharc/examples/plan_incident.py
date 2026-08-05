"""A governed planning loop a reader can run — ROADMAP §12.1.

`grapharc/planner` closed the propose → admit → materialise → execute → replan
cycle and, until this module existed, nothing shipped drove it: the crux of the
architecture was a library API proven by tests, reachable only by importing it.
This is the registry and the scripted planner behind `grapharc plan`.

**What the demo is built to show.** `deploy` is a registered kind — an operator
declared it, so a planner is allowed to *propose* it — and the default edge
policy denies every transition into it. The scripted planner proposes a deploy
in round 1 and is refused, then replans without it. That is the whole thesis in
two rounds: the model wanted to ship to production, the graph did not permit it,
and the refusal came back as a reason the planner could act on.

Nothing here holds a node body a planner supplied. The registry owns every
factory; a proposal names a *kind*. Swap the scripted model for a real one with
`--model` and the enforcement is identical, because none of it is prompt text.
"""

from __future__ import annotations

import json
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
from grapharc.runtime.graph import END, START


class IncidentState(BaseModel):
    """One state contract for the whole run, however the topology changes.

    `notes` is a reducer (`Annotated` + `operator.add`): each writer returns
    only its own lines and LangGraph merges them, so a planner that runs two
    writers in the same parallel step — including two instances of one kind —
    composes instead of colliding. A plain `list[str]` here raises
    `InvalidUpdateError` the first time that happens.
    """

    goal: str = ""
    notes: Annotated[list[str], operator.add] = []


def _step_factory(spec: NodeSpec) -> Any:
    """Build one node body from its registry entry, and nowhere else.

    The registry is the only source of a body. This factory is what
    `Materializer` calls after admission has already said yes; a proposal has
    no field a body could arrive in, so there is nothing to smuggle.
    """

    def body(state: IncidentState) -> dict:
        return {"notes": [f"{spec.name} ran"]}

    body.writes = {"notes"}
    return body


#: What each kind is permitted to write. A kind nobody declared writes for may
#: write nothing — `Materializer` enforces that, not the node. `grapharc plan
#: --registry` reads this name, and `STATE_SCHEMA`, off the module it is given.
WRITES: dict[str, set[str]] = {
    "triage": {"notes"},
    "patch": {"notes"},
    "verify": {"notes"},
    "deploy": {"notes"},
}


def build_registry() -> NodeRegistry:
    """The kinds a planner may propose. Absence is refusal; there is no wildcard."""
    return NodeRegistry(
        [
            NodeSpec(
                name="triage",
                description="classify the incident and gather context",
                factory=_step_factory,
                worst_case=CostEstimate(iterations=1, tokens=400),
            ),
            NodeSpec(
                name="patch",
                description="write a fix for the cause",
                factory=_step_factory,
                worst_case=CostEstimate(iterations=1, tokens=2000),
            ),
            NodeSpec(
                name="verify",
                description="prove the fix works against evidence",
                factory=_step_factory,
                worst_case=CostEstimate(iterations=1, tokens=800),
            ),
            NodeSpec(
                name="deploy",
                description="push to production",
                factory=_step_factory,
                worst_case=CostEstimate(iterations=1, tokens=200),
            ),
        ]
    )


def default_edge_policy() -> EdgePolicy:
    """Deny anything into `deploy`; allow the rest. Unmatched is deny.

    Registering `deploy` and denying every edge into it is the point: the kind
    is proposable and unreachable, so a refusal is a policy decision about a
    transition rather than a gap in the allowlist.
    """
    return EdgePolicy(
        rules=(
            EdgeRule(action=Decision.DENY, target="deploy"),
            EdgeRule(action=Decision.ALLOW),
        )
    )


def _plan(*kinds: str) -> str:
    """One planner reply: a chain from START through `kinds` to END."""
    endpoints = [START, *kinds, END]
    return json.dumps(
        {
            "nodes": [{"name": kind} for kind in kinds],
            "edges": [
                {"source": a, "target": b}
                for a, b in zip(endpoints, endpoints[1:], strict=False)
            ],
        }
    )


def scripted_planner_replies() -> list[str]:
    """Round 1 proposes a deploy and is refused; round 2 replans without it."""
    return [_plan("triage", "deploy"), _plan("triage", "patch", "verify")]


def build_loop(
    model: Any,
    *,
    edge_policy: EdgePolicy | None = None,
    node_policy: Any = None,
    trace: TraceRecorder | None = None,
    budget: Budget | None = None,
    limits: LoopLimits | None = None,
    registry: NodeRegistry | None = None,
    state_schema: type[BaseModel] | None = None,
    writes: dict[str, set[str]] | None = None,
    approval: Any = None,
) -> GovernedLoop:
    """Assemble the loop from operator-owned parts.

    Everything a run is subject to is decided here, before a model is asked
    anything: which kinds exist, which transitions are permitted, what the run
    may spend, and how many rounds it gets.
    """
    registry = registry or build_registry()
    # Frozen: a driver that checks "against the same registry" every round means
    # the same object, and a node body could otherwise widen it between rounds.
    registry.freeze()
    # Resolved once so the planner is *told* about exactly the policy the checker
    # will *apply*. Two calls to `default_edge_policy()` would be two objects,
    # and a disclosure describing a different object than the gate enforces is
    # worse than no disclosure at all.
    edge_policy = edge_policy or default_edge_policy()
    return GovernedLoop(
        # The planner and the materializer get the recorder too. Without it the
        # run's own trace held only `admission`/`round`/`stop`: no `plan` event
        # saying what was proposed and what it cost, and — because the built
        # subgraph inherits the materializer's recorder — no `start`/`end` pair
        # for any node the loop actually executed. README's "the executed nodes'
        # own start/end pairs" was true of a hand-wired loop and false of the
        # shipped one, which is the one `grapharc plan` drives.
        planner=PlannerNode(
            model,
            name="incident",
            catalog=registry.catalog(),
            # Disclosure, not enforcement: the planner is shown the deny rules so
            # it need not learn them one refusal at a time. The scripted planner
            # below proposes a `deploy` anyway, and round 1 is still refused —
            # which is the demo's whole point, and stays true with a real model.
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
        ),
        checker=AdmissionChecker(
            registry=registry,
            edge_policy=edge_policy,
            # There is no default node policy: this demo's registry *is* its
            # node allowlist. One arrives only when a policy document declares
            # node rules, and then it gates every kind the planner proposes.
            node_policy=node_policy,
            trace=trace,
            # This loop materializes each admitted round as a standalone
            # graph, so structural runnability is admission's business:
            # otherwise a plan with no entry is admitted and then fails to
            # build, which the planner cannot replan against.
            limits=AdmissionLimits(require_entry=True),
        ),
        materializer=Materializer(
            registry=registry,
            state_schema=state_schema or IncidentState,
            writes=writes if writes is not None else WRITES,
            trace=trace,
        ),
        budget=budget,
        limits=limits,
        trace=trace,
        name="incident_loop",
        # Deterministic code, never a model: "am I done" is exactly the question
        # a run must not be able to talk itself into answering yes.
        # Three steps done. Reads `notes` defensively so a custom state schema
        # supplied through `--registry` cannot turn "am I done" into an
        # AttributeError halfway through a run.
        goal_reached=lambda state: len(getattr(state, "notes", ()) or ()) >= 3,
        approval=approval,
    )


#: Read by `grapharc plan --registry` so a custom module can supply its own.
STATE_SCHEMA = IncidentState

#: The kind this demo treats as dangerous. Read by the policy generator so a
#: generated policy knows what is worth denying; without it, generation would
#: produce a document that denies nothing.
MUTATING_KINDS = ("deploy",)

__all__ = [
    "STATE_SCHEMA",
    "WRITES",
    "IncidentState",
    "build_loop",
    "build_registry",
    "default_edge_policy",
    "scripted_planner_replies",
]
