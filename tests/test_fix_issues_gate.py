"""The listener/fixer registry — `grapharc.registries.fix_issues`.

The claim this module makes is that a fan-out of autonomous fixers stays
governed: the width of a round is decided by the planner, and every widening
re-enters the admission gate. The tests below pin the refusals that make the
claim true rather than decorative:

- an eager fixer is refused by the default policy before anything runs, and
  the rehearsal still ends with an honest report;
- a round of fixers whose registry worst case exceeds what remains is refused
  with the recorded reason, spend-free;
- with no model the kinds do not exist, so naming one fails at the gate as
  `unregistered_node` rather than at materialisation;
- two fixers finishing together merge instead of colliding.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from grapharc.harness.permissions import Decision
from grapharc.planner import EdgePolicy, EdgeRule, LoopStop
from grapharc.registries import fix_issues
from grapharc.registries.fix_issues import FixState, _split_issues, unfixed
from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel

ALLOW_EVERYTHING = EdgePolicy(rules=(EdgeRule(action=Decision.ALLOW),))


def _plan(nodes: list[dict], edges: list[tuple[str, str]], rationale: str) -> str:
    return json.dumps(
        {
            "nodes": nodes,
            "edges": [{"source": s, "target": t} for s, t in edges],
            "rationale": rationale,
        }
    )


NO_FURTHER_WORK = _plan([], [], "no further work")


def test_the_eager_fix_is_refused_and_the_replan_finishes_honestly():
    """Round 1 wires an edge into `fix_one` and never executes; the rehearsal
    still ends `goal_met`, with zero fixes and a report that says why."""
    model = ScriptedChatModel(responses=fix_issues.scripted_planner_replies())
    loop = fix_issues.build_loop(model)
    result = loop.run("fix the issues in this repo", FixState(goal="fix the issues"))

    assert result.stop is LoopStop.GOAL_MET
    first = result.rounds[0]
    assert not first.admitted and not first.executed
    assert "edge_denied" in [r.code for r in result.rejections()]
    assert result.state.fixes == []
    assert len(result.state.issues) == 3
    assert len(result.state.failures) == 3  # every issue verified as unfixed
    assert "not admitted" in result.state.notes[0]


def test_a_round_of_fixers_beyond_the_budget_is_rejected_with_the_recorded_reason():
    """Three fixers cost 18k tokens worst case against a 10k budget: the round
    is refused before anything runs, and the reason names the shortfall."""
    replies = [
        _plan(
            [{"name": "scan_issues"}],
            [(START, "scan_issues"), ("scan_issues", END)],
            "scan first",
        ),
        _plan(
            [
                {"name": f"fix_{i}", "kind": "fix_one", "args": {"issue": f"issue {i}"}}
                for i in (1, 2, 3)
            ],
            [(START, f"fix_{i}") for i in (1, 2, 3)]
            + [(f"fix_{i}", END) for i in (1, 2, 3)],
            "one fixer per issue",
        ),
        NO_FURTHER_WORK,
        NO_FURTHER_WORK,  # the empty plan gets one nudge before it is believed
    ]
    loop = fix_issues.build_loop(
        ScriptedChatModel(responses=replies),
        edge_policy=ALLOW_EVERYTHING,
        budget=Budget(max_tokens=10_000),
    )
    result = loop.run("fix everything", FixState(goal="fix everything"))

    codes = [r.code for r in result.rejections()]
    assert "over_token_budget" in codes
    rejection = next(r for r in result.rejections() if r.code == "over_token_budget")
    assert "remain" in rejection.detail
    assert result.state.fixes == []  # the refused round bought nothing


def test_naming_a_kind_without_a_model_is_refused_at_the_gate():
    """With no model the registry is empty, so a proposal naming the listener
    fails admission as `unregistered_node` — not at materialisation."""
    registry = fix_issues.build_registry(None)
    assert registry.catalog() == {}

    replies = [
        _plan(
            [{"name": "scan_issues"}],
            [(START, "scan_issues"), ("scan_issues", END)],
            "scan",
        ),
        NO_FURTHER_WORK,
        NO_FURTHER_WORK,
    ]
    loop = fix_issues.build_loop(ScriptedChatModel(responses=replies), registry=registry)
    result = loop.run("fix the issues", FixState(goal="fix the issues"))

    assert "unregistered_node" in [r.code for r in result.rejections()]
    assert not result.rounds[0].executed


def test_parallel_fixers_merge_instead_of_colliding():
    """Two fixers finish in one superstep; the reducer appends both entries.
    A plain list field dies here with `InvalidUpdateError`."""
    replies = [
        _plan(
            [{"name": "scan_issues"}],
            [(START, "scan_issues"), ("scan_issues", END)],
            "scan",
        ),
        _plan(
            [
                {
                    "name": "fix_1",
                    "kind": "fix_one",
                    "args": {"issue": fix_issues._SCRIPTED_ISSUES[0]},
                },
                {
                    "name": "fix_2",
                    "kind": "fix_one",
                    "args": {"issue": fix_issues._SCRIPTED_ISSUES[1]},
                },
            ],
            [(START, "fix_1"), (START, "fix_2"), ("fix_1", END), ("fix_2", END)],
            "two fixers in parallel, each with its assignment",
        ),
        _plan(
            [{"name": "verify_fixes"}, {"name": "report"}],
            [(START, "verify_fixes"), ("verify_fixes", "report"), ("report", END)],
            "verify and report",
        ),
    ]
    loop = fix_issues.build_loop(
        ScriptedChatModel(responses=replies), edge_policy=ALLOW_EVERYTHING
    )
    result = loop.run("fix the issues", FixState(goal="fix the issues"))

    assert result.stop is LoopStop.GOAL_MET
    assert len(result.state.fixes) == 2
    # Each fixer took exactly its admission-checked assignment, so the third
    # issue is the one still outstanding.
    assert set(result.state.fixes) == {
        f"fixed: {fix_issues._SCRIPTED_ISSUES[0]}",
        f"fixed: {fix_issues._SCRIPTED_ISSUES[1]}",
    }
    assert unfixed(result.state) == [fix_issues._SCRIPTED_ISSUES[2]]
    assert len(result.state.notes) == 1


def test_a_fixer_without_its_assignment_is_rejected_at_the_gate():
    """`fix_one` declares `FixAssignment`, so a fixer proposal with no args is
    refused at admission with the field named — not built and hoped about."""
    replies = [
        _plan(
            [{"name": "fix_1", "kind": "fix_one"}],
            [(START, "fix_1"), ("fix_1", END)],
            "an unassigned fixer",
        ),
        NO_FURTHER_WORK,
        NO_FURTHER_WORK,
    ]
    loop = fix_issues.build_loop(
        ScriptedChatModel(responses=replies), edge_policy=ALLOW_EVERYTHING
    )
    result = loop.run("fix the issues", FixState(goal="fix the issues"))

    assert "args_schema_violation" in [r.code for r in result.rejections()]
    assert not result.rounds[0].executed


def test_an_assignment_edited_after_admission_refuses_to_build():
    """`ProposedNode` is frozen but its args dict is mutable in place — the
    documented gap. The fingerprint is what closes it: an edited assignment is
    a different proposal, and the materialiser refuses it."""
    import pytest

    from grapharc.planner import (
        AdmissionChecker,
        Materializer,
        NotAdmitted,
        ProposedEdge,
        ProposedNode,
        Subgraph,
    )

    registry = fix_issues.build_registry(ScriptedChatModel(responses=[])).freeze()
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="fix_1", kind="fix_one", args={"issue": "issue: a"}),
        ),
        edges=(
            ProposedEdge(source=START, target="fix_1"),
            ProposedEdge(source="fix_1", target=END),
        ),
        rationale="one assigned fixer",
    )
    checker = AdmissionChecker(registry=registry, edge_policy=ALLOW_EVERYTHING)
    result = checker.check(proposal)
    assert result.admitted

    proposal.nodes[0].args["issue"] = "issue: something else entirely"

    materializer = Materializer(
        registry=registry, state_schema=FixState, writes=fix_issues.WRITES
    )
    with pytest.raises(NotAdmitted):
        materializer.materialize(result, proposal)


def test_the_module_ships_the_full_registry_contract():
    """Everything `RegistryBundle` reads travels together, and the write map
    covers every kind — a kind absent from it may write nothing."""
    assert fix_issues.STATE_SCHEMA is FixState
    assert set(fix_issues.WRITES) == set(fix_issues.AGENT_KINDS)
    assert set(fix_issues.TOOLS_FOR) == set(fix_issues.AGENT_KINDS)
    assert set(fix_issues.MUTATING_KINDS) <= set(fix_issues.AGENT_KINDS)
    assert callable(fix_issues.default_edge_policy)
    assert callable(fix_issues.build_loop)
    assert callable(fix_issues.scripted_planner_replies)
    # The completion rule is defensive: a foreign state cannot turn "am I
    # done" into an AttributeError.
    assert fix_issues.goal_met(object()) is False
    assert fix_issues.goal_met(SimpleNamespace(notes=["done"])) is True


def test_the_listener_report_splits_into_one_issue_per_entry():
    """The planner fans out over entries, so a blob report would collapse the
    run to one fixer. Markers stripped, blanks dropped, duplicates folded."""
    text = "- issue: a thing\n\n* Issue: A THING\n  issue: another thing\n"
    assert _split_issues(text) == ["issue: a thing", "issue: another thing"]


def test_the_unfixed_diff_matches_by_the_issue_text_a_fixer_was_told_to_take():
    state = FixState(
        issues=["issue: a", "issue: b"],
        fixes=["fixed: issue: a — changed one line"],
    )
    assert unfixed(state) == ["issue: b"]
