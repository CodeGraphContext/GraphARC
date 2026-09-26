"""The research registry demonstrates governed parallel decomposition."""

from __future__ import annotations

import json

from grapharc.cli.main import main
from grapharc.examples import plan_research
from grapharc.planner import AdmissionChecker, NodeBuild, Subgraph
from grapharc.runtime.graph import END, START

INVESTIGATIVE_KINDS = {
    "pull_logs",
    "pull_metrics",
    "pull_deploys",
    "pull_tickets",
    "correlate",
    "test_hypothesis",
    "estimate_impact",
    "write_report",
    "page_oncall",
}


def _proposal(reply: str) -> Subgraph:
    return Subgraph.model_validate_json(reply)


def test_every_kind_has_one_distinct_state_field_and_only_writes_it():
    assert set(plan_research.WRITES) == INVESTIGATIVE_KINDS
    fields = [next(iter(writes)) for writes in plan_research.WRITES.values()]
    assert all(len(writes) == 1 for writes in plan_research.WRITES.values())
    assert len(fields) == len(set(fields))
    assert set(fields) <= set(plan_research.ResearchState.model_fields)

    registry = plan_research.build_registry()
    state = plan_research.ResearchState(goal="explain elevated checkout latency")
    for kind, declared in plan_research.WRITES.items():
        spec = registry.get(kind)
        assert spec is not None and spec.factory is not None
        body = spec.factory(
            NodeBuild(
                name=f"example_{kind}",
                kind=kind,
                proposal_id="test-plan",
                fingerprint="test-fingerprint",
            )
        )
        assert set(body(state)) == declared


def test_page_oncall_is_registered_but_policy_denied():
    proposal = _proposal(plan_research.scripted_planner_replies()[0])
    registry = plan_research.build_registry()
    verdict = AdmissionChecker(
        registry=registry,
        edge_policy=plan_research.default_edge_policy(),
    ).check(proposal)

    assert not verdict.admitted
    assert {rejection.code for rejection in verdict.rejections} == {"edge_denied"}


def test_scripted_research_plan_fans_out_joins_forks_and_fans_in():
    proposal = _proposal(plan_research.scripted_planner_replies()[1])
    edges = {(edge.source, edge.target) for edge in proposal.edges}

    assert {(START, kind) for kind in {
        "pull_logs",
        "pull_metrics",
        "pull_deploys",
        "pull_tickets",
    }} <= edges
    assert {(kind, "correlate") for kind in {
        "pull_logs",
        "pull_metrics",
        "pull_deploys",
        "pull_tickets",
    }} <= edges
    assert {
        ("correlate", "test_hypothesis"),
        ("correlate", "estimate_impact"),
        ("test_hypothesis", "write_report"),
        ("estimate_impact", "write_report"),
        ("write_report", END),
    } <= edges


def test_cli_runs_parallel_research_plan_to_goal_and_traces_topology(tmp_path, capsys):
    trace_path = tmp_path / "research.jsonl"
    code = main(
        [
            "plan",
            "explain elevated checkout latency",
            "--scripted",
            "--go",
            "--registry",
            "grapharc.examples.plan_research:build_registry",
            "--trace",
            str(trace_path),
        ]
    )

    printed = capsys.readouterr().out
    assert code == 0
    assert "goal_met" in printed
    assert "report=" in printed

    events = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    topology_edges = {
        tuple(edge[:2])
        for event in events
        if event["phase"] == "topology"
        for edge in (event.get("state_delta") or {}).get("edges", [])
    }
    assert {
        (START, "pull_logs"),
        (START, "pull_metrics"),
        (START, "pull_deploys"),
        (START, "pull_tickets"),
        ("pull_logs", "correlate"),
        ("pull_metrics", "correlate"),
        ("pull_deploys", "correlate"),
        ("pull_tickets", "correlate"),
        ("correlate", "test_hypothesis"),
        ("correlate", "estimate_impact"),
        ("test_hypothesis", "write_report"),
        ("estimate_impact", "write_report"),
        ("write_report", END),
    } <= topology_edges
