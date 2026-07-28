"""Planner and admission: propose freely, run only what a gate authorised.

The split this package exists for — ARCHITECTURE.md §2 — is that a node may
*propose* nodes and edges and may never *execute* them. `proposal` holds the
typed thing a planner emits; `admission` holds the deterministic, model-free
checker that is the only way one becomes work.

    planner  = PlannerNode(model, catalog=registry.catalog())
    checker  = AdmissionChecker(registry=registry, edge_policy=policy)

    outcome  = planner.propose(task, ctx)          # produces, runs nothing
    result   = checker.check(outcome.proposal, meter=ctx.meter)
    if not result.admitted:
        outcome = planner.propose(task, ctx, feedback=result.feedback())

The line dividing the two is which strings are trusted. A planner chooses an
instance `name`; an operator registers a `kind`. Admission decides on the kind
in every check that grants anything — registration, edge policy and cost — so
the same node is refused however it is spelled, and an endpoint whose kind the
checker cannot establish is refused rather than assumed safe. Names survive as
labels: they resolve which node an edge means, and they appear in rejections
and traces so a decision can be found again.

Materialising an admitted proposal into a live graph is not implemented here;
admission authorises a shape and stops.
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
    NodeRegistry,
    NodeSpec,
    Rejection,
    RemainingBudget,
)
from grapharc.planner.proposal import (
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    PlannerConfigError,
    PlannerNode,
    PlanningOutcome,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

__all__ = [
    "DEFAULT_PLANNER_SYSTEM_PROMPT",
    "AdmissionChecker",
    "AdmissionLimits",
    "AdmissionRejected",
    "AdmissionResult",
    "AdmissionStatus",
    "Check",
    "CostEstimate",
    "EdgePolicy",
    "EdgeRule",
    "NodeRegistry",
    "NodeSpec",
    "PlannerConfigError",
    "PlannerNode",
    "PlanningOutcome",
    "ProposedEdge",
    "ProposedNode",
    "Rejection",
    "RemainingBudget",
    "Subgraph",
]
