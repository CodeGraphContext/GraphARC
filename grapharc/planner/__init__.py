"""Planner and admission: propose freely, run only what a gate authorised.

The split this package exists for is that a node may
*propose* nodes and edges and may never *execute* them. `proposal` holds the
typed thing a planner emits; `admission` holds the deterministic, model-free
checker that is the only way one becomes work; `materialize` turns an admitted
proposal into a runnable graph and refuses everything else; `loop` drives the
whole cycle to a recorded stop.

    planner  = PlannerNode(model, catalog=registry.catalog())
    checker  = AdmissionChecker(registry=registry, edge_policy=policy)

    outcome  = planner.propose(task, ctx)          # produces, runs nothing
    result   = checker.check(outcome.proposal, meter=ctx.meter)
    if not result.admitted:
        outcome = planner.propose(task, ctx, feedback=result.feedback())

    graph = Materializer(registry=registry, state_schema=State).materialize(
        result, outcome.proposal      # the result first: it is the authorisation
    )

or the same four steps as a bounded, traced, replanning run:

    loop = GovernedLoop(planner=planner, checker=checker, materializer=materializer)
    done = loop.run("investigate the incident", State())
    done.stop                                      # a LoopStop, always set

The line dividing planner from gate is which strings are trusted. A planner
chooses an instance `name`; an operator registers a `kind`. Admission decides on
the kind in every check that grants anything — registration, edge policy and
cost — so the same node is refused however it is spelled, and an endpoint whose
kind the checker cannot establish is refused rather than assumed safe. Names
survive as labels: they resolve which node an edge means, and they appear in
rejections and traces so a decision can be found again.

Materialisation keeps that line on the far side of the gate. A node's body comes
from the registry's `NodeSpec.factory` and from nowhere else, the graph is built
through the ordinary `GraphARC` builder so writes, budgets, typed state and
traces all still apply, and the entry point takes the `AdmissionResult` — a
proposal on its own has no authorisation to offer.
"""

from grapharc.planner.admission import (
    AdmissionChecker,
    AdmissionLimits,
    AdmissionRejected,
    AdmissionResult,
    AdmissionStatus,
    Check,
    CostEstimate,
    EdgePolicy,
    EdgeRule,
    NodePolicy,
    NodeRegistry,
    NodeRule,
    NodeSpec,
    Rejection,
    RemainingBudget,
)
from grapharc.planner.loop import (
    GovernedLoop,
    LoopLimits,
    LoopResult,
    LoopStop,
    Planner,
    RoundRecord,
)
from grapharc.planner.materialize import (
    MaterializationError,
    Materializer,
    NodeBuild,
    NotAdmitted,
    UnadmittedTransition,
)
from grapharc.planner.proposal import (
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    PROPOSAL_EXAMPLE,
    PlannerConfigError,
    PlannerNode,
    PlanningOutcome,
    PlanProposal,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

__all__ = [
    "DEFAULT_PLANNER_SYSTEM_PROMPT",
    "PROPOSAL_EXAMPLE",
    "AdmissionChecker",
    "AdmissionLimits",
    "AdmissionRejected",
    "AdmissionResult",
    "AdmissionStatus",
    "Check",
    "CostEstimate",
    "EdgePolicy",
    "EdgeRule",
    "GovernedLoop",
    "LoopLimits",
    "LoopResult",
    "LoopStop",
    "MaterializationError",
    "Materializer",
    "NodeBuild",
    "NodePolicy",
    "NodeRegistry",
    "NodeRule",
    "NodeSpec",
    "NotAdmitted",
    "Planner",
    "PlannerConfigError",
    "PlannerNode",
    "PlanProposal",
    "PlanningOutcome",
    "ProposedEdge",
    "ProposedNode",
    "Rejection",
    "RemainingBudget",
    "RoundRecord",
    "Subgraph",
    "UnadmittedTransition",
]
