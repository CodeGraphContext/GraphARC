"""A governed incident-research plan with visible parallel decomposition.

``plan_incident`` demonstrates refusal and replanning with a short chain. This
registry demonstrates the topology a real investigation needs: four evidence
collectors fan out from ``START``, join at correlation, fork into hypothesis
and impact analysis, then fan in to one final report.

Every kind writes one distinct state field. That makes parallel updates safe
without reducers and lets ``Materializer`` enforce the boundary independently
of the node body. ``page_oncall`` is registered so a planner can propose it,
but the default edge policy denies every transition into it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

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
    NodeBuild,
    NodeRegistry,
    NodeSpec,
    PlannerNode,
)
from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import END, START


class ResearchState(BaseModel):
    """One independent field per investigative kind, plus the operator's goal."""

    goal: str = ""
    incident_logs: str = ""
    service_metrics: str = ""
    recent_deployments: str = ""
    support_tickets: str = ""
    correlation: str = ""
    hypothesis: str = ""
    impact: str = ""
    report: str = ""
    oncall_page: str = ""


def _pull_logs(state: ResearchState) -> dict[str, str]:
    return {"incident_logs": f"logs: checkout timeouts increased for {state.goal}"}


def _pull_metrics(state: ResearchState) -> dict[str, str]:
    return {"service_metrics": "metrics: p95 latency and error rate rose together"}


def _pull_deploys(state: ResearchState) -> dict[str, str]:
    return {"recent_deployments": "deploys: checkout-api release preceded the increase"}


def _pull_tickets(state: ResearchState) -> dict[str, str]:
    return {"support_tickets": "tickets: customers report payment confirmation timeouts"}


def _correlate(state: ResearchState) -> dict[str, str]:
    evidence = (
        state.incident_logs,
        state.service_metrics,
        state.recent_deployments,
        state.support_tickets,
    )
    return {"correlation": "correlation: " + " | ".join(evidence)}


def _test_hypothesis(state: ResearchState) -> dict[str, str]:
    return {
        "hypothesis": (
            "hypothesis: the checkout-api release caused the latency regression; "
            f"evidence considered: {state.correlation}"
        )
    }


def _estimate_impact(state: ResearchState) -> dict[str, str]:
    return {
        "impact": (
            "impact: checkout completion is degraded for affected customers; "
            f"ticket signal: {state.support_tickets}"
        )
    }


def _write_report(state: ResearchState) -> dict[str, str]:
    return {
        "report": (
            f"final report: {state.hypothesis}. {state.impact}. "
            "Recommended next step: verify a rollback in a controlled environment."
        )
    }


def _page_oncall(state: ResearchState) -> dict[str, str]:
    return {"oncall_page": f"page requested for: {state.goal}"}


_BODIES: dict[str, Callable[[ResearchState], dict[str, str]]] = {
    "pull_logs": _pull_logs,
    "pull_metrics": _pull_metrics,
    "pull_deploys": _pull_deploys,
    "pull_tickets": _pull_tickets,
    "correlate": _correlate,
    "test_hypothesis": _test_hypothesis,
    "estimate_impact": _estimate_impact,
    "write_report": _write_report,
    "page_oncall": _page_oncall,
}

WRITES: dict[str, set[str]] = {
    "pull_logs": {"incident_logs"},
    "pull_metrics": {"service_metrics"},
    "pull_deploys": {"recent_deployments"},
    "pull_tickets": {"support_tickets"},
    "correlate": {"correlation"},
    "test_hypothesis": {"hypothesis"},
    "estimate_impact": {"impact"},
    "write_report": {"report"},
    "page_oncall": {"oncall_page"},
}

_DESCRIPTIONS = {
    "pull_logs": "collect relevant application logs",
    "pull_metrics": "collect service latency and error metrics",
    "pull_deploys": "collect recent deployment history",
    "pull_tickets": "collect related customer-support reports",
    "correlate": "join the four evidence streams and correlate their timing",
    "test_hypothesis": "test the most likely causal hypothesis",
    "estimate_impact": "estimate customer and service impact",
    "write_report": "fan in the analyses and write the final incident report",
    "page_oncall": "page the human on-call responder",
}


def _factory(build: NodeBuild) -> Callable[[ResearchState], dict[str, str]]:
    body = _BODIES[build.kind]
    body.writes = WRITES[build.kind]  # type: ignore[attr-defined]
    return body


def build_registry() -> NodeRegistry:
    """Nine bounded kinds; a proposal can name them but cannot supply bodies."""
    return NodeRegistry(
        NodeSpec(
            name=kind,
            description=description,
            factory=_factory,
            worst_case=CostEstimate(iterations=1, tokens=200),
        )
        for kind, description in _DESCRIPTIONS.items()
    )


def default_edge_policy() -> EdgePolicy:
    """Keep paging visible to the planner while making it unreachable."""
    return EdgePolicy(
        rules=(
            EdgeRule(
                action=Decision.DENY,
                target="page_oncall",
                reason="paging a human requires a separate approval path",
            ),
            EdgeRule(action=Decision.ALLOW),
        )
    )


def _reply(nodes: list[str], edges: list[tuple[str, str]], rationale: str) -> str:
    return json.dumps(
        {
            "nodes": [{"name": name} for name in nodes],
            "edges": [{"source": source, "target": target} for source, target in edges],
            "rationale": rationale,
        }
    )


def scripted_planner_replies() -> list[str]:
    """Refuse paging, then run a deterministic parallel research topology."""
    collectors = ["pull_logs", "pull_metrics", "pull_deploys", "pull_tickets"]
    research_nodes = [
        *collectors,
        "correlate",
        "test_hypothesis",
        "estimate_impact",
        "write_report",
    ]
    research_edges = [
        *((START, kind) for kind in collectors),
        *((kind, "correlate") for kind in collectors),
        ("correlate", "test_hypothesis"),
        ("correlate", "estimate_impact"),
        ("test_hypothesis", "write_report"),
        ("estimate_impact", "write_report"),
        ("write_report", END),
    ]
    return [
        _reply(
            ["page_oncall"],
            [(START, "page_oncall"), ("page_oncall", END)],
            "page a human immediately",
        ),
        _reply(
            research_nodes,
            research_edges,
            "collect evidence in parallel, join it, analyse in parallel, then report",
        ),
    ]


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
    """Assemble the research loop from operator-owned policy and bodies."""
    registry = registry or build_registry()
    registry.freeze()
    edge_policy = edge_policy or default_edge_policy()
    return GovernedLoop(
        planner=PlannerNode(
            model,
            name="research",
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
            state_schema=state_schema or ResearchState,
            writes=writes if writes is not None else WRITES,
            trace=trace,
        ),
        budget=budget,
        limits=limits,
        trace=trace,
        name="research_loop",
        goal_reached=lambda state: bool(getattr(state, "report", "").strip()),
        approval=approval,
    )


STATE_SCHEMA = ResearchState
MUTATING_KINDS = ("page_oncall",)

__all__ = [
    "MUTATING_KINDS",
    "STATE_SCHEMA",
    "WRITES",
    "ResearchState",
    "build_loop",
    "build_registry",
    "default_edge_policy",
    "scripted_planner_replies",
]
