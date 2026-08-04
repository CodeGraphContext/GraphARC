"""Gates for the planner/admission split (ROADMAP §5).

The invariant under test: a node may *propose* nodes and edges and may never
execute them. Every test here is written so that deleting the thing it covers
makes it fail — the registry tests use factories that raise if anything calls
them, the budget test asserts the meter was never charged, and the "valid"
test asserts the full set of checks actually ran.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from grapharc.harness.permissions import Decision
from grapharc.planner import (
    AdmissionChecker,
    AdmissionLimits,
    AdmissionRejected,
    AdmissionStatus,
    Check,
    CostEstimate,
    EdgePolicy,
    EdgeRule,
    NodePolicy,
    NodeRegistry,
    NodeRule,
    NodeSpec,
    PlannerNode,
    ProposedEdge,
    ProposedNode,
    RemainingBudget,
    Subgraph,
)
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel

ALLOW_ALL = EdgePolicy(rules=(EdgeRule(action="allow"),))


def _explode(*args, **kwargs):  # pragma: no cover - calling it is the failure
    raise AssertionError("admission executed a node factory")


def registry(*kinds: str, **costs: CostEstimate) -> NodeRegistry:
    """A registry whose every node blows up if anything tries to run it."""
    return NodeRegistry(
        NodeSpec(
            name=kind,
            description=f"{kind} description",
            worst_case=costs.get(kind, CostEstimate(iterations=1)),
            factory=_explode,
        )
        for kind in kinds
    )


def linear(*names: str, kind: str | None = None) -> Subgraph:
    """START -> n1 -> n2 -> ... -> END, one node per name."""
    nodes = tuple(ProposedNode(name=n, kind=kind or n) for n in names)
    endpoints = [START, *names, END]
    edges = tuple(
        ProposedEdge(source=a, target=b)
        for a, b in zip(endpoints, endpoints[1:], strict=False)
    )
    return Subgraph(nodes=nodes, edges=edges, rationale="linear")


def checker(reg: NodeRegistry, **kwargs) -> AdmissionChecker:
    kwargs.setdefault("edge_policy", ALLOW_ALL)
    return AdmissionChecker(registry=reg, **kwargs)


# -- the happy path ------------------------------------------------------------


def test_a_valid_proposal_is_admitted_and_every_check_ran():
    # `require_entry` is opt-in (a proposal may attach to a live graph, where
    # the entry is outside it), so a checker that runs *every* gate has to ask.
    result = checker(
        registry("fetch", "summarise"),
        limits=AdmissionLimits(require_entry=True),
    ).check(linear("fetch", "summarise"))

    assert result.admitted
    assert result.status is AdmissionStatus.ADMITTED
    assert result.rejections == ()
    assert result.failed_checks() == ()
    # A gate that silently skipped a check would still look "admitted"; this is
    # what distinguishes admission from a pass-through.
    assert set(result.checks_run) == set(Check)
    assert result.node_count == 2
    assert result.depth == 1
    assert result.worst_case_complete


def test_admit_returns_the_proposal_it_authorised():
    proposal = linear("fetch")
    assert checker(registry("fetch")).admit(proposal) is proposal


def test_an_empty_proposal_authorises_nothing_and_is_admitted():
    result = checker(registry("fetch")).check(Subgraph())
    assert result.admitted
    assert result.node_count == 0
    assert result.depth == 0


# -- REGISTRY ------------------------------------------------------------------


def test_an_unregistered_node_is_rejected_naming_the_registry_check():
    proposal = linear("fetch", "rm_rf")
    result = checker(registry("fetch")).check(proposal)

    assert not result.admitted
    assert result.status is AdmissionStatus.REJECTED
    assert result.failed_checks() == (Check.REGISTRY,)
    (reason,) = result.reasons(Check.REGISTRY)
    assert reason.code == "unregistered_node"
    assert reason.subject == "rm_rf"
    assert "rm_rf" in reason.detail
    assert "fetch" in reason.remedy  # the planner is told what it may use


def test_an_empty_registry_admits_nothing():
    result = checker(NodeRegistry()).check(linear("fetch"))
    assert not result.admitted
    assert [r.code for r in result.reasons(Check.REGISTRY)] == ["unregistered_node"]


def test_kind_defaults_to_name_so_a_one_off_node_needs_no_alias():
    node = ProposedNode(name="fetch")
    assert node.kind == "fetch"


def test_many_instances_of_one_registered_kind_are_admitted():
    proposal = Subgraph(
        nodes=tuple(ProposedNode(name=f"w{i}", kind="worker") for i in range(3)),
        edges=tuple(ProposedEdge(source=START, target=f"w{i}") for i in range(3)),
    )
    assert checker(registry("worker")).check(proposal).admitted


def test_an_edge_to_a_node_that_does_not_exist_is_rejected():
    proposal = Subgraph(
        nodes=(ProposedNode(name="fetch"),),
        edges=(ProposedEdge(source="fetch", target="ghost"),),
    )
    result = checker(registry("fetch")).check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.REGISTRY)
    assert reason.code == "unknown_edge_endpoint"
    assert reason.subject == "fetch -> ghost"
    assert "target" in reason.detail


def test_an_edge_may_reference_a_node_already_in_the_graph():
    proposal = Subgraph(
        nodes=(ProposedNode(name="fetch"),),
        edges=(ProposedEdge(source="triage", target="fetch"),),
    )
    # The live node is declared as `{name: kind}`: policy decides over kinds, so
    # the caller has to say what its own nodes are.
    gate = checker(registry("fetch"), known_nodes={"triage": "triage_agent"})
    assert gate.check(proposal).admitted


def test_a_proposed_name_may_not_shadow_a_live_node():
    result = checker(registry("fetch"), known_nodes={"fetch"}).check(linear("fetch"))

    assert not result.admitted
    codes = [r.code for r in result.reasons(Check.REGISTRY)]
    assert "name_collides_with_existing_node" in codes


# -- POLICY --------------------------------------------------------------------


def test_a_policy_forbidden_edge_is_rejected_naming_the_policy_check():
    policy = EdgePolicy(
        rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
    )
    proposal = linear("build", "deploy")
    result = checker(registry("build", "deploy"), edge_policy=policy).check(proposal)

    assert not result.admitted
    assert result.failed_checks() == (Check.POLICY,)
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "edge_denied"
    assert reason.subject == "build -> deploy"


def test_an_unmatched_edge_defaults_to_deny():
    proposal = linear("fetch")
    result = checker(registry("fetch"), edge_policy=EdgePolicy()).check(proposal)

    assert not result.admitted
    # START -> fetch and fetch -> END, both refused by the default.
    assert {r.subject for r in result.reasons(Check.POLICY)} == {
        f"{START} -> fetch",
        f"fetch -> {END}",
    }
    assert all(r.code == "edge_denied" for r in result.reasons(Check.POLICY))


def test_a_broad_deny_beats_a_narrower_allow():
    """No allowlist exception hides inside a denial — the harness rule, on edges."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="allow", source="build", target="deploy"),
            EdgeRule(action="deny", target="deploy"),
        )
    )
    assert policy.decide("build", "deploy") is Decision.DENY
    assert policy.decide("build", "test") is Decision.DENY  # unmatched: fail closed

    result = checker(registry("build", "deploy"), edge_policy=policy).check(
        linear("build", "deploy")
    )
    assert not result.admitted


def test_an_edge_needing_approval_is_not_admitted():
    policy = EdgePolicy(rules=(EdgeRule(action="ask", target="deploy"), EdgeRule(action="allow")))
    result = checker(registry("build", "deploy"), edge_policy=policy).check(
        linear("build", "deploy")
    )

    assert not result.admitted
    assert result.needs_approval
    assert result.status is AdmissionStatus.NEEDS_APPROVAL
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "edge_needs_approval"
    assert reason.subject == "build -> deploy"


def test_a_denial_outranks_a_pending_approval():
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="deny", target="deploy"),
            EdgeRule(action="ask", target="build"),
            EdgeRule(action="allow"),
        )
    )
    result = checker(registry("build", "deploy"), edge_policy=policy).check(
        linear("build", "deploy")
    )
    assert result.status is AdmissionStatus.REJECTED


# -- POLICY over node kinds ----------------------------------------------------
# The registry says a kind exists and what it costs. A `NodePolicy` — compiled
# from a document an operator can edit without touching the code that builds the
# registry — says whether it may run here. A kind has to pass both, and a
# proposal is refused with a code the planner can replan against.


def test_a_policy_forbidden_node_kind_is_rejected_naming_the_policy_check():
    policy = NodePolicy(
        rules=(NodeRule(action="deny", match="deploy"), NodeRule(action="allow"))
    )
    result = checker(registry("build", "deploy"), node_policy=policy).check(
        linear("build", "deploy")
    )

    assert not result.admitted
    assert result.failed_checks() == (Check.POLICY,)
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "node_denied"
    assert reason.subject == "deploy"


def test_a_denied_node_kind_cannot_be_renamed_out_of_its_denial():
    """The same rule the edge half follows: a name grants and borrows nothing."""
    policy = NodePolicy(
        rules=(NodeRule(action="deny", match="deploy"), NodeRule(action="allow"))
    )
    gate = checker(registry("build", "deploy"), node_policy=policy)

    renamed = gate.check(
        Subgraph(nodes=(ProposedNode(name="totally_fine", kind="deploy"),))
    )
    borrowed = gate.check(
        Subgraph(nodes=(ProposedNode(name="deploy", kind="build"),))
    )

    assert [r.code for r in renamed.rejections] == ["node_denied"]
    assert "kind 'deploy'" in renamed.rejections[0].detail
    assert "totally_fine" in renamed.rejections[0].detail
    assert borrowed.admitted


def test_a_node_denial_quotes_the_reason_the_operator_wrote():
    policy = NodePolicy(
        rules=(
            NodeRule(action="deny", match="shell_*", reason="a shell node is unbounded"),
            NodeRule(action="allow"),
        )
    )
    result = checker(registry("shell_worker"), node_policy=policy).check(
        Subgraph(nodes=(ProposedNode(name="shell_worker"),))
    )

    assert result.rejections[0].detail.endswith("a shell node is unbounded")


def test_a_node_kind_needing_approval_is_not_admitted():
    policy = NodePolicy(
        rules=(NodeRule(action="ask", match="deploy"), NodeRule(action="allow"))
    )
    result = checker(registry("build", "deploy"), node_policy=policy).check(
        linear("build", "deploy")
    )

    assert not result.admitted
    assert result.needs_approval
    assert result.status is AdmissionStatus.NEEDS_APPROVAL
    assert [r.code for r in result.reasons(Check.POLICY)] == ["node_needs_approval"]


def test_an_unmatched_node_kind_defaults_to_deny():
    """`NodePolicy()` is empty, not permissive — the same shape `EdgePolicy` has."""
    result = checker(registry("fetch"), node_policy=NodePolicy()).check(
        Subgraph(nodes=(ProposedNode(name="fetch"),))
    )

    assert [r.code for r in result.rejections] == ["node_denied"]


def test_a_broad_node_deny_beats_a_narrower_allow():
    policy = NodePolicy(
        rules=(
            NodeRule(action="allow", match="deploy"),
            NodeRule(action="deny", match="dep*"),
        )
    )
    assert policy.decide("deploy") is Decision.DENY


def test_no_node_policy_leaves_the_registry_as_the_only_node_gate():
    """The behaviour every existing caller has: absence is not a wildcard.

    A checker built without a node policy decides node kinds exactly as it did
    before one existed — registered kinds pass, unregistered ones are refused by
    REGISTRY.
    """
    gate = checker(registry("fetch"))

    assert gate.check(Subgraph(nodes=(ProposedNode(name="fetch"),))).admitted
    refused = gate.check(Subgraph(nodes=(ProposedNode(name="shell"),)))
    assert [r.code for r in refused.rejections] == ["unregistered_node"]


def test_a_denied_node_kind_hidden_in_a_nested_scope_is_refused_too():
    policy = NodePolicy(
        rules=(NodeRule(action="deny", match="deploy"), NodeRule(action="allow"))
    )
    nested = Subgraph(
        nodes=(
            ProposedNode(
                name="outer",
                kind="build",
                subgraph=Subgraph(nodes=(ProposedNode(name="inner", kind="deploy"),)),
            ),
        )
    )
    result = checker(
        registry("build", "deploy"),
        node_policy=policy,
        limits=AdmissionLimits(max_depth=2),
    ).check(nested)

    assert [r.code for r in result.rejections] == ["node_denied"]
    assert result.rejections[0].subject == "outer/inner"


def test_a_node_denial_and_an_edge_denial_are_both_reported():
    """One POLICY check, both halves — a planner gets the whole list."""
    result = checker(
        registry("build", "deploy"),
        edge_policy=EdgePolicy(rules=(EdgeRule(action="deny", target="deploy"),)),
        node_policy=NodePolicy(
            rules=(NodeRule(action="deny", match="deploy"), NodeRule(action="allow"))
        ),
    ).check(
        Subgraph(
            nodes=(ProposedNode(name="build"), ProposedNode(name="deploy")),
            edges=(ProposedEdge(source="build", target="deploy"),),
        )
    )

    assert {r.code for r in result.rejections} == {"node_denied", "edge_denied"}
    assert result.failed_checks() == (Check.POLICY,)


# -- POLICY decides on the KIND, never on the name the planner chose -----------
#
# The gate is only a gate if what it matches is a fact about the node. `name` is
# a string the planner invents; `kind` is the key of something an operator
# registered. Every test below is the same claim from a different angle: a
# policy rule is about what a node *is*.


DENY_DEPLOY = EdgePolicy(rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow")))


def wired(*pairs: tuple[str, str]) -> Subgraph:
    """START -> ... -> END over explicit `(name, kind)` pairs."""
    nodes = tuple(ProposedNode(name=name, kind=kind) for name, kind in pairs)
    endpoints = [START, *(name for name, _ in pairs), END]
    edges = tuple(
        ProposedEdge(source=a, target=b)
        for a, b in zip(endpoints, endpoints[1:], strict=False)
    )
    return Subgraph(nodes=nodes, edges=edges, rationale="wired")


def test_renaming_a_denied_kind_does_not_evade_the_policy():
    """The rename evasion: the operator denied a kind, not a spelling."""
    gate = checker(registry("build", "deploy"), edge_policy=DENY_DEPLOY)

    honest = gate.check(wired(("build", "build"), ("deploy", "deploy")))
    renamed = gate.check(wired(("build", "build"), ("helper", "deploy")))

    assert not honest.admitted
    assert not renamed.admitted, "renaming the instance evaded the edge policy"
    assert renamed.failed_checks() == (Check.POLICY,)
    (reason,) = renamed.reasons(Check.POLICY)
    assert reason.code == "edge_denied"
    # The edge is located by instance name; the *reason* names the kind, so an
    # audit reading the rejection can see which authorisation was refused.
    assert reason.subject == "build -> helper"
    assert "deploy" in reason.detail


def test_a_denial_holds_whatever_the_planner_calls_the_instance():
    gate = checker(registry("build", "deploy"), edge_policy=DENY_DEPLOY)

    for alias in ("helper", "step2", "safe", "build_helper", "x"):
        result = gate.check(wired(("build", "build"), (alias, "deploy")))
        assert not result.admitted, f"alias {alias!r} evaded the denial"
        assert [r.code for r in result.reasons(Check.POLICY)] == ["edge_denied"]


def test_the_rejection_names_the_kind_so_the_denial_is_auditable():
    gate = checker(registry("build", "deploy"), edge_policy=DENY_DEPLOY)
    result = gate.check(wired(("build", "build"), ("helper", "deploy")))

    text = result.feedback()
    assert "deploy" in text  # what was refused
    assert "helper" in text  # where to find it


def test_naming_an_instance_after_a_permitted_kind_does_not_borrow_permission():
    """The inverse evasion: a permitted *name* is not a permitted *kind*."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="allow", target="safe"),
            EdgeRule(action="allow", source="safe", target=END),
        )
    )
    gate = checker(registry("safe", "danger"), edge_policy=policy)

    assert gate.check(wired(("anything", "safe"))).admitted

    stolen = gate.check(wired(("safe", "danger"),))
    assert not stolen.admitted, "an instance named after a permitted kind borrowed its rule"
    assert stolen.failed_checks() == (Check.POLICY,)


def test_an_allow_rule_for_a_kind_admits_every_instance_of_that_kind():
    """The fix is keying, not blanket refusal: a fan-out of one kind still passes."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="allow", source=START, target="worker"),
            EdgeRule(action="allow", source="worker", target=END),
        )
    )
    proposal = Subgraph(
        nodes=tuple(ProposedNode(name=f"w{i}", kind="worker") for i in range(3)),
        edges=tuple(
            edge
            for i in range(3)
            for edge in (
                ProposedEdge(source=START, target=f"w{i}"),
                ProposedEdge(source=f"w{i}", target=END),
            )
        ),
    )
    assert checker(registry("worker"), edge_policy=policy).check(proposal).admitted


def test_a_rename_evasion_hidden_in_a_nested_scope_is_refused_too():
    inner = Subgraph(
        nodes=(ProposedNode(name="a", kind="build"), ProposedNode(name="helper", kind="deploy")),
        edges=(ProposedEdge(source="a", target="helper"),),
    )
    proposal = Subgraph(nodes=(ProposedNode(name="outer", kind="build", subgraph=inner),))
    gate = checker(
        registry("build", "deploy"),
        edge_policy=DENY_DEPLOY,
        limits=AdmissionLimits(max_depth=2),
    )

    result = gate.check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "edge_denied"
    assert reason.subject == "outer/a -> helper"
    assert "deploy" in reason.detail


def test_a_live_nodes_declared_kind_is_what_the_policy_matches():
    """A node already in the graph is matched on the kind its owner declared."""
    gate = checker(
        registry("build"),
        edge_policy=DENY_DEPLOY,
        known_nodes={"shipper": "deploy"},
    )
    proposal = Subgraph(
        nodes=(ProposedNode(name="build"),),
        edges=(ProposedEdge(source="build", target="shipper"),),
    )

    result = gate.check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "edge_denied"
    assert "deploy" in reason.detail


def test_a_live_node_listed_without_a_kind_cannot_be_wired_at_all():
    """Fail closed: no rule can be evaluated against a node nobody identified."""
    gate = checker(registry("fetch"), known_nodes={"triage"})  # names only, no kinds
    proposal = Subgraph(
        nodes=(ProposedNode(name="fetch"),),
        edges=(ProposedEdge(source="triage", target="fetch"),),
    )

    result = gate.check(proposal)

    assert not result.admitted  # even though the policy here is allow-all
    (reason,) = result.reasons(Check.POLICY)
    assert reason.code == "unresolved_endpoint_kind"
    assert "triage" in reason.detail
    assert "known_nodes" in reason.remedy


def test_the_kind_is_charset_validated_like_a_name():
    """The policy key cannot carry fnmatch metacharacters or edge-render syntax."""
    for bad in ("*", "dep loy", "a -> b", "[a]", "?", "de*ploy"):
        with pytest.raises(ValidationError):
            ProposedNode(name="ok", kind=bad)
    # An omitted kind is the documented default, not a hole: it becomes the
    # name, which the registry check then has to recognise like any other kind.
    assert ProposedNode(name="ok", kind="").kind == "ok"


def test_the_worst_case_cost_is_charged_to_the_kind_not_the_name():
    """The same confusion, on the budget axis: a cheap name is not a cheap node."""
    reg = registry("cheap", "expensive", expensive=CostEstimate(tokens=9_000))
    proposal = Subgraph(nodes=(ProposedNode(name="cheap", kind="expensive"),))

    result = checker(reg).check(proposal, remaining=RemainingBudget(tokens=1_000))

    assert not result.admitted
    assert result.worst_case.tokens == 9_000
    assert [r.code for r in result.reasons(Check.BUDGET)] == ["over_token_budget"]


# -- BUDGET --------------------------------------------------------------------


def test_an_over_budget_proposal_is_rejected_before_anything_runs():
    reg = registry("crawl", crawl=CostEstimate(tokens=5_000, iterations=1))
    meter = BudgetMeter(Budget(max_tokens=1_000, max_iterations=50))
    proposal = Subgraph(
        nodes=tuple(ProposedNode(name=f"c{i}", kind="crawl") for i in range(3)),
        edges=(),
    )

    result = checker(reg).check(proposal, meter=meter)

    assert not result.admitted
    assert result.failed_checks() == (Check.BUDGET,)
    (reason,) = result.reasons(Check.BUDGET)
    assert reason.code == "over_token_budget"
    assert reason.subject == "tokens"
    assert "15000" in reason.detail and "1000" in reason.detail
    # Nothing ran and nothing was charged: the gate is before the work, not
    # after it. `_explode` on every registered factory covers the first half.
    assert meter.tokens == 0
    assert meter.iterations == 0


def test_the_budget_check_uses_what_is_left_not_the_total():
    reg = registry("step", step=CostEstimate(tokens=600))
    meter = BudgetMeter(Budget(max_tokens=1_000))
    proposal = Subgraph(nodes=(ProposedNode(name="s", kind="step"),))

    assert checker(reg).check(proposal, meter=meter).admitted

    meter.charge_tokens(500)
    later = checker(reg).check(proposal, meter=meter)
    assert not later.admitted
    assert later.remaining.tokens == 500
    assert [r.code for r in later.reasons(Check.BUDGET)] == ["over_token_budget"]


def test_iteration_and_time_ceilings_reject_independently():
    reg = registry("slow", slow=CostEstimate(iterations=4, seconds=30.0))
    proposal = Subgraph(nodes=(ProposedNode(name="s", kind="slow"),))
    remaining = RemainingBudget(tokens=None, iterations=2, seconds=10.0)

    result = checker(reg).check(proposal, remaining=remaining)
    assert {r.code for r in result.reasons(Check.BUDGET)} == {
        "over_iteration_budget",
        "over_time_budget",
    }


def test_no_budget_at_all_means_unlimited():
    reg = registry("huge", huge=CostEstimate(tokens=10**9, iterations=10**6))
    assert checker(reg).check(Subgraph(nodes=(ProposedNode(name="h", kind="huge"),))).admitted


def test_worst_case_comes_from_the_registry_not_the_proposal():
    reg = registry("step", step=CostEstimate(tokens=750))
    proposal = Subgraph(
        nodes=(ProposedNode(name="s", kind="step", args={"tokens": 1, "cost": 0}),)
    )
    result = checker(reg).check(proposal, remaining=RemainingBudget(tokens=10_000))
    assert result.worst_case.tokens == 750


def test_worst_case_is_flagged_incomplete_when_a_kind_is_unknown():
    reg = registry("known", known=CostEstimate(tokens=10))
    proposal = Subgraph(
        nodes=(ProposedNode(name="a", kind="known"), ProposedNode(name="b", kind="mystery"))
    )
    result = checker(reg).check(proposal, remaining=RemainingBudget(tokens=10_000))

    assert not result.admitted  # REGISTRY caught it regardless
    assert result.worst_case.tokens == 10
    assert result.worst_case_complete is False


def test_an_already_overspent_run_admits_nothing_that_costs():
    reg = registry("step", step=CostEstimate(tokens=1))
    meter = BudgetMeter(Budget(max_tokens=100))
    meter.charge_tokens(500)
    result = checker(reg).check(
        Subgraph(nodes=(ProposedNode(name="s", kind="step"),)), meter=meter
    )
    assert not result.admitted
    assert result.remaining.tokens == -400


def test_worst_case_counts_iterations_for_unequal_arm_joins_in_acyclic_plans():
    reg = registry("step", step=CostEstimate(tokens=100, iterations=1))
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="a", kind="step"),
            ProposedNode(name="b", kind="step"),
            ProposedNode(name="c", kind="step"),
        ),
        edges=(
            ProposedEdge(source=START, target="a"),
            ProposedEdge(source="a", target="b"),
            ProposedEdge(source="b", target="c"),
            ProposedEdge(source="a", target="c"),
            ProposedEdge(source="c", target=END),
        ),
    )
    result = checker(reg).check(proposal)
    assert result.worst_case.iterations == 4
    assert result.worst_case.tokens == 400


# -- DEPTH ---------------------------------------------------------------------


def nested(depth: int) -> Subgraph:
    """A chain of `depth` nested scopes, each holding one node."""
    inner: Subgraph | None = None
    for level in reversed(range(depth)):
        inner = Subgraph(nodes=(ProposedNode(name=f"n{level}", kind="step", subgraph=inner),))
    assert inner is not None
    return inner


def test_a_too_deep_proposal_is_rejected():
    result = checker(registry("step")).check(nested(3))

    assert not result.admitted
    assert result.failed_checks() == (Check.DEPTH,)
    (reason,) = result.reasons(Check.DEPTH)
    assert reason.code == "too_deep"
    assert "3" in reason.detail
    assert result.depth == 3


def test_nesting_is_admitted_when_the_limit_allows_it():
    limits = AdmissionLimits(max_depth=3)
    assert checker(registry("step"), limits=limits).check(nested(3)).admitted


def test_the_depth_a_proposal_arrives_at_counts_toward_the_limit():
    """A planner running inside an admitted subgraph cannot recurse for free."""
    reg = registry("step")
    flat = linear("step")

    assert checker(reg).check(flat, parent_depth=0).admitted

    deeper = checker(reg).check(flat, parent_depth=1)
    assert not deeper.admitted
    assert deeper.depth == 2
    assert deeper.reasons(Check.DEPTH)[0].code == "too_deep"


def test_nested_scopes_are_checked_too():
    """An unregistered kind hidden one level down is still caught."""
    inner = Subgraph(nodes=(ProposedNode(name="sneaky", kind="rm_rf"),))
    proposal = Subgraph(nodes=(ProposedNode(name="outer", kind="step", subgraph=inner),))
    result = checker(registry("step"), limits=AdmissionLimits(max_depth=2)).check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.REGISTRY)
    assert reason.code == "unregistered_node"
    assert reason.subject == "outer/sneaky"  # scoped, so an audit can find it


def test_an_empty_nested_subgraph_adds_no_depth():
    proposal = Subgraph(nodes=(ProposedNode(name="outer", kind="step", subgraph=Subgraph()),))
    result = checker(registry("step")).check(proposal)
    assert result.depth == 1
    assert result.admitted


def test_negative_parent_depth_is_a_programming_error():
    with pytest.raises(ValueError, match="parent_depth"):
        checker(registry("step")).check(linear("step"), parent_depth=-1)


# -- ACYCLICITY ----------------------------------------------------------------


def cyclic() -> Subgraph:
    return Subgraph(
        nodes=(ProposedNode(name="a", kind="step"), ProposedNode(name="b", kind="step")),
        edges=(
            ProposedEdge(source=START, target="a"),
            ProposedEdge(source="a", target="b"),
            ProposedEdge(source="b", target="a"),
        ),
    )


def test_a_cyclic_proposal_is_rejected_when_acyclicity_is_required():
    result = checker(registry("step")).check(cyclic())

    assert not result.admitted
    assert result.failed_checks() == (Check.ACYCLICITY,)
    (reason,) = result.reasons(Check.ACYCLICITY)
    assert reason.code == "cycle"
    assert reason.subject in ("a -> b -> a", "b -> a -> b")


def test_a_self_loop_is_a_cycle():
    proposal = Subgraph(
        nodes=(ProposedNode(name="a", kind="step"),),
        edges=(ProposedEdge(source="a", target="a"),),
    )
    result = checker(registry("step")).check(proposal)
    assert not result.admitted
    assert result.reasons(Check.ACYCLICITY)[0].subject == "a -> a"


def test_a_cycle_is_admitted_when_acyclicity_is_not_required():
    limits = AdmissionLimits(require_acyclic=False)
    result = checker(registry("step"), limits=limits).check(cyclic())

    assert result.admitted
    # The check is skipped, and says so rather than silently passing.
    assert Check.ACYCLICITY not in result.checks_run


def test_a_cycle_inside_a_nested_scope_is_found():
    inner = Subgraph(
        nodes=(ProposedNode(name="x", kind="step"), ProposedNode(name="y", kind="step")),
        edges=(
            ProposedEdge(source="x", target="y"),
            ProposedEdge(source="y", target="x"),
        ),
    )
    proposal = Subgraph(nodes=(ProposedNode(name="outer", kind="step", subgraph=inner),))
    result = checker(registry("step"), limits=AdmissionLimits(max_depth=2)).check(proposal)

    assert not result.admitted
    assert result.reasons(Check.ACYCLICITY)[0].subject.startswith("outer/")


def test_a_diamond_is_not_a_cycle():
    proposal = Subgraph(
        nodes=tuple(ProposedNode(name=n, kind="step") for n in "abcd"),
        edges=(
            ProposedEdge(source="a", target="b"),
            ProposedEdge(source="a", target="c"),
            ProposedEdge(source="b", target="d"),
            ProposedEdge(source="c", target="d"),
        ),
    )
    assert checker(registry("step")).check(proposal).admitted


# -- rejections are complete, machine-readable, and recorded -------------------


def all_checks_failing() -> Subgraph:
    """One proposal that trips every gate at once.

    Registry (an unregistered kind), policy (a deny-all policy), budget (a
    costly kind against a tiny remainder), depth (a nested subgraph),
    acyclicity (a -> b -> a), and reachability (nothing leaves START).
    """
    inner = Subgraph(nodes=(ProposedNode(name="inner", kind="step"),))
    return Subgraph(
        nodes=(
            ProposedNode(name="a", kind="step", subgraph=inner),
            ProposedNode(name="b", kind="unregistered"),
        ),
        edges=(
            ProposedEdge(source="a", target="b"),
            ProposedEdge(source="b", target="a"),
        ),
    )


def test_every_failed_check_is_reported_not_just_the_first():
    reg = registry("step", step=CostEstimate(tokens=10_000))
    result = checker(
        reg, edge_policy=EdgePolicy(), limits=AdmissionLimits(require_entry=True)
    ).check(all_checks_failing(), remaining=RemainingBudget(tokens=5))

    assert not result.admitted
    assert set(result.failed_checks()) == set(Check)


def test_every_rejection_names_a_check_a_code_and_a_subject():
    reg = registry("step", step=CostEstimate(tokens=10_000))
    result = checker(
        reg, edge_policy=EdgePolicy(), limits=AdmissionLimits(require_entry=True)
    ).check(all_checks_failing(), remaining=RemainingBudget(tokens=5))

    for reason in result.rejections:
        assert isinstance(reason.check, Check)
        assert reason.code and " " not in reason.code
        assert reason.subject
        assert reason.detail


def test_feedback_is_a_planner_readable_list_of_every_failure():
    reg = registry("step", step=CostEstimate(tokens=10_000))
    result = checker(
        reg, edge_policy=EdgePolicy(), limits=AdmissionLimits(require_entry=True)
    ).check(all_checks_failing(), remaining=RemainingBudget(tokens=5))
    text = result.feedback()

    assert result.proposal_id in text
    for reason in result.rejections:
        assert f"{reason.check.value}/{reason.code}" in text
        assert reason.subject in text


def test_admit_raises_carrying_the_result():
    with pytest.raises(AdmissionRejected) as caught:
        checker(registry("fetch")).admit(linear("nope"))

    assert caught.value.result.failed_checks() == (Check.REGISTRY,)
    assert "unregistered_node" in str(caught.value)


def test_a_rejection_is_written_to_the_trace(trace):
    gate = checker(registry("fetch"), trace=trace)
    proposal = linear("nope")

    result = gate.check(proposal)

    events = [e for e in trace.read_events() if e.phase == "admission"]
    assert len(events) == 1
    assert events[0].node.endswith(proposal.proposal_id)
    assert "registry/unregistered_node" in (events[0].error or "")
    assert events[0].state_delta["status"] == "rejected"
    assert events[0].state_delta["failed_checks"] == ["registry"]
    assert events[0].state_delta["fingerprint"] == result.fingerprint


def test_an_admission_is_recorded_too(trace):
    checker(registry("fetch"), trace=trace).check(linear("fetch"))

    (event,) = [e for e in trace.read_events() if e.phase == "admission"]
    assert event.state_delta["status"] == "admitted"
    assert event.error is None


def test_admission_events_do_not_pollute_run_metrics(trace):
    """`observe.metrics` counts node executions from "end" events."""
    checker(registry("fetch"), trace=trace).check(linear("fetch"))
    assert [e.phase for e in trace.read_events()] == ["admission"]


def test_the_fingerprint_pins_the_exact_proposal():
    proposal = linear("fetch")
    same = Subgraph(
        nodes=proposal.nodes,
        edges=proposal.edges,
        rationale=proposal.rationale,
        proposal_id=proposal.proposal_id,
    )
    changed = Subgraph(
        nodes=(*proposal.nodes, ProposedNode(name="extra", kind="fetch")),
        edges=proposal.edges,
        rationale=proposal.rationale,
        proposal_id=proposal.proposal_id,
    )
    assert proposal.fingerprint() == same.fingerprint()
    assert proposal.fingerprint() != changed.fingerprint()


# -- proposals are well-formed by construction ---------------------------------


def test_duplicate_node_names_cannot_be_constructed():
    with pytest.raises(ValidationError, match="duplicate node names"):
        Subgraph(nodes=(ProposedNode(name="a"), ProposedNode(name="a", kind="b")))


def test_a_node_may_not_be_named_after_the_graph_sentinels():
    for reserved in (START, END):
        with pytest.raises(ValidationError, match="reserved"):
            ProposedNode(name=reserved, kind="step")


def test_an_edge_may_reference_the_sentinels():
    edge = ProposedEdge(source=START, target=END)
    assert edge.render() == f"{START} -> {END}"


def test_a_name_outside_the_charset_is_rejected():
    for bad in ("a -> b", "a b", "", "9lives", "drop table"):
        with pytest.raises(ValidationError):
            ProposedNode(name=bad, kind="step")


def test_an_unknown_field_in_a_proposal_is_rejected():
    with pytest.raises(ValidationError):
        Subgraph.model_validate({"nodes": [], "edges": [], "run_this": "yes"})


def test_a_proposal_cannot_be_reassigned_after_admission():
    proposal = linear("fetch")
    with pytest.raises(ValidationError):
        proposal.nodes = ()


# -- the planner proposes and does not execute ---------------------------------


class StructuredScriptedModel(ScriptedChatModel):
    """A scripted model that also implements the structured-output path.

    Real structured output needs a tool-calling backend, which no scripted
    double has and which a test may not call. Overriding the method directly is
    what lets the planner's structured branch be exercised offline: the script
    still supplies the JSON and the raw `AIMessage` still carries usage, so the
    token-metering path on that branch is covered too.
    """

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        def run(messages):
            raw = self.invoke(messages)
            try:
                parsed = schema.model_validate_json(str(raw.content))
                error = None
            except Exception as exc:  # noqa: BLE001 - mirrors langchain's envelope
                parsed, error = None, exc
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": error}
            return parsed

        return RunnableLambda(run)


PLAN_JSON = {
    "rationale": "read, then summarise",
    "nodes": [{"name": "read", "kind": "fetch"}, {"name": "write", "kind": "summarise"}],
    "edges": [
        {"source": START, "target": "read"},
        {"source": "read", "target": "write"},
        {"source": "write", "target": END},
    ],
}


def test_the_planner_turns_model_text_into_a_typed_proposal():
    model = ScriptedChatModel(responses=[f"Sure:\n```json\n{json.dumps(PLAN_JSON)}\n```"])
    outcome = PlannerNode(model, catalog=registry("fetch", "summarise").catalog()).propose("go")

    assert outcome.ok
    assert outcome.structured is False
    assert [n.name for n in outcome.proposal.nodes] == ["read", "write"]
    assert outcome.tokens > 0


def test_the_planner_uses_structured_output_when_the_backend_has_it():
    model = StructuredScriptedModel(responses=[json.dumps(PLAN_JSON)])
    outcome = PlannerNode(model).propose("go")

    assert outcome.ok
    assert outcome.structured is True
    assert outcome.tokens > 0


def test_a_structured_parse_failure_falls_back_to_the_text_path():
    fenced = f"here is the plan\n```json\n{json.dumps(PLAN_JSON)}\n```"
    # Precondition: the strict structured parse really does fail on this reply.
    with pytest.raises(ValidationError):
        Subgraph.model_validate_json(fenced)

    outcome = PlannerNode(StructuredScriptedModel(responses=[fenced])).propose("go")

    assert outcome.structured is True
    assert outcome.ok  # the envelope's parse failed; extract_json recovered it
    assert outcome.error == ""
    assert [n.name for n in outcome.proposal.nodes] == ["read", "write"]


def test_the_planner_never_runs_what_it_proposes():
    """Every registered kind's factory raises; a full plan-then-admit is silent."""
    reg = registry("fetch", "summarise")
    model = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog())

    outcome = planner.propose("go")
    result = checker(reg).check(outcome.proposal)

    assert result.admitted
    assert reg.get("fetch").factory is _explode  # it was there to be called


RENAMED_PLAN_JSON = {
    "rationale": "deploy is denied, so this one is called `helper`",
    "nodes": [{"name": "build", "kind": "build"}, {"name": "helper", "kind": "deploy"}],
    "edges": [
        {"source": START, "target": "build"},
        {"source": "build", "target": "helper"},
        {"source": "helper", "target": END},
    ],
}


def test_a_planner_cannot_rename_its_way_past_a_denial_end_to_end():
    """Model -> proposal -> gate, with the model choosing the instance names."""
    reg = registry("build", "deploy")
    model = ScriptedChatModel(responses=[json.dumps(RENAMED_PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog())
    gate = checker(reg, edge_policy=DENY_DEPLOY)

    outcome = planner.propose("ship it")
    assert outcome.ok  # the proposal is well-formed; the gate is what refuses it

    result = gate.check(outcome.proposal)

    assert not result.admitted
    assert result.failed_checks() == (Check.POLICY,)
    assert "deploy" in result.feedback()
    assert reg.get("deploy").factory is _explode  # it was there to be called


def test_the_planner_stamps_its_own_provenance_over_the_models():
    payload = {**PLAN_JSON, "proposal_id": "attacker-chosen", "origin": "trusted"}
    model = ScriptedChatModel(responses=[json.dumps(payload)])
    outcome = PlannerNode(model, name="triage").propose("go")

    assert outcome.proposal.origin == "planner:triage"
    assert outcome.proposal.proposal_id != "attacker-chosen"


def test_an_unparseable_reply_is_an_outcome_not_an_exception():
    model = ScriptedChatModel(responses=["I would rather not."])
    outcome = PlannerNode(model).propose("go")

    assert not outcome.ok
    assert outcome.proposal is None
    assert "no JSON" in outcome.error
    assert outcome.raw == "I would rather not."


def test_a_structurally_invalid_proposal_is_reported_with_its_reason():
    model = ScriptedChatModel(
        responses=[json.dumps({"nodes": [{"name": "a"}, {"name": "a"}], "edges": []})]
    )
    outcome = PlannerNode(model).propose("go")

    assert not outcome.ok
    assert "duplicate node names" in outcome.error


def test_a_model_that_raises_is_reported_not_propagated():
    model = ScriptedChatModel(responses=[])  # exhausted on the first call
    outcome = PlannerNode(model).propose("go")

    assert not outcome.ok
    assert "planner model call failed" in outcome.error


def test_the_catalog_is_put_in_front_of_the_model():
    reg = registry("fetch", "summarise")
    model = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    PlannerNode(model, catalog=reg.catalog()).propose("go")

    system = str(model.calls[0][0].content)
    assert "fetch: fetch description" in system
    assert "summarise: summarise description" in system


# -- issue #45: the policy is disclosed to the planner, and only disclosed -----
#
# The catalog said which kinds exist and nothing said which transitions were
# denied, so a model proposed an edge into a denied kind every round until the
# loop gave up: `edge_denied` names the check, not the rule, and "no edge may
# enter deploy, ever" is not inferable from it. These tests pin both halves —
# the disclosure is in the prompt, and it is *only* a disclosure.


def _system_prompt(planner: PlannerNode, model: ScriptedChatModel, task: str = "go") -> str:
    planner.propose(task)
    return str(model.calls[0][0].content)


def test_the_edge_policys_denials_are_put_in_front_of_the_model():
    reg = registry("build", "deploy")
    model = ScriptedChatModel(responses=[json.dumps(RENAMED_PLAN_JSON)])

    system = _system_prompt(
        PlannerNode(model, catalog=reg.catalog(), edge_policy=DENY_DEPLOY), model, "ship it"
    )

    assert "edges into 'deploy' are denied by policy — do not propose them" in system
    # And the catalog still lists the kind: it is registered, an operator did
    # allow it to be proposed, and it is the *edge* that is refused.
    assert "deploy: deploy description" in system


def test_a_disclosed_denial_carries_the_operators_own_words():
    policy = EdgePolicy(
        rules=(
            EdgeRule(
                action="deny", target="deploy", reason="a deploy is the operator's decision"
            ),
            EdgeRule(action="allow"),
        )
    )
    model = ScriptedChatModel(responses=[json.dumps(RENAMED_PLAN_JSON)])

    system = _system_prompt(
        PlannerNode(model, catalog=registry("build", "deploy").catalog(), edge_policy=policy),
        model,
        "ship it",
    )

    assert (
        "edges into 'deploy' are denied by policy — do not propose them: "
        "a deploy is the operator's decision" in system
    )


def test_denied_node_kinds_are_disclosed_beside_the_denied_edges():
    node_policy = NodePolicy(
        rules=(
            NodeRule(action="deny", match="deploy", reason="not from a plan"),
            NodeRule(action="allow"),
        )
    )
    model = ScriptedChatModel(responses=[json.dumps(RENAMED_PLAN_JSON)])

    system = _system_prompt(
        PlannerNode(
            model,
            catalog=registry("build", "deploy").catalog(),
            edge_policy=DENY_DEPLOY,
            node_policy=node_policy,
        ),
        model,
        "ship it",
    )

    assert (
        "nodes of kind 'deploy' are denied by policy — do not propose them: not from a plan"
        in system
    )
    assert "edges into 'deploy' are denied by policy" in system


def test_a_policy_that_denies_nothing_adds_nothing_to_the_prompt():
    """Allow rules and the default are the catalog's business, not a warning's."""
    reg = registry("fetch", "summarise")
    told = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    untold = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])

    with_policy = _system_prompt(
        PlannerNode(told, catalog=reg.catalog(), edge_policy=ALLOW_ALL), told
    )
    without = _system_prompt(PlannerNode(untold, catalog=reg.catalog()), untold)

    assert "denied by policy" not in with_policy
    assert with_policy == without  # the unpolicied prompt is byte-identical to before


def test_disclosure_does_not_move_the_enforcement_into_the_prompt():
    """A planner told about the denial and proposing it anyway is refused identically.

    The constraint the fix is subject to: enforcement stays in the checker. Two
    planners, one shown the deny rule and one not, produce the same proposal
    here — and the gate has to return the same rejections for both, down to the
    text, or the disclosure has started deciding something.
    """
    reg = registry("build", "deploy")
    reply = json.dumps(RENAMED_PLAN_JSON)
    told = PlannerNode(
        ScriptedChatModel(responses=[reply]), catalog=reg.catalog(), edge_policy=DENY_DEPLOY
    )
    untold = PlannerNode(ScriptedChatModel(responses=[reply]), catalog=reg.catalog())
    gate = checker(reg, edge_policy=DENY_DEPLOY)

    disclosed = gate.check(told.propose("ship it").proposal)
    blind = gate.check(untold.propose("ship it").proposal)

    assert not disclosed.admitted
    assert disclosed.failed_checks() == (Check.POLICY,)
    assert [r.model_dump() for r in disclosed.rejections] == [
        r.model_dump() for r in blind.rejections
    ]
    assert reg.get("deploy").factory is _explode  # it was there to be called


def test_a_denied_edge_is_refused_with_the_rules_reason_not_only_its_code():
    """`edge_denied` is what happened; the reason is why, and the planner needs both."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="deny", target="deploy", reason="Deploy changes are dangerous"),
            EdgeRule(action="allow"),
        )
    )

    result = checker(registry("build", "deploy"), edge_policy=policy).check(
        linear("build", "deploy")
    )

    reason = result.reasons(Check.POLICY)[0]
    assert reason.code == "edge_denied"
    assert "Deploy changes are dangerous" in reason.detail
    assert "Deploy changes are dangerous" in result.feedback()


def test_a_rule_without_a_reason_still_refuses_and_says_only_what_it_knows():
    result = checker(registry("build", "deploy"), edge_policy=DENY_DEPLOY).check(
        linear("build", "deploy")
    )

    reason = result.reasons(Check.POLICY)[0]
    assert reason.code == "edge_denied"
    assert reason.detail.endswith("kind 'deploy' (proposed as 'deploy')")


def test_a_disclosure_describes_the_rule_it_was_compiled_from():
    """Every shape of deny rule renders as something a model can act on."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="deny", target="deploy"),
            EdgeRule(action="deny", source="deploy"),
            EdgeRule(action="deny", source="triage", target="verify"),
            EdgeRule(action="deny", target="risky_*"),
            EdgeRule(action="ask", target="patch"),
            EdgeRule(action="allow"),
        )
    )

    assert policy.disclosure() == (
        "edges into 'deploy' are denied by policy — do not propose them",
        "edges out of 'deploy' are denied by policy — do not propose them",
        "edges from 'triage' to 'verify' are denied by policy — do not propose them",
        "edges into kinds matching 'risky_*' are denied by policy — do not propose them",
    )
    # An `ask` is not a "do not propose": its remedy is an approval, not another
    # proposal. And a policy with nothing denied has nothing to disclose.
    assert ALLOW_ALL.disclosure() == ()
    assert NodePolicy(rules=(NodeRule(action="deny", match="*"),)).disclosure() == (
        "all node kinds are denied by policy — do not propose them",
    )


def test_a_repeated_denial_is_disclosed_once():
    """Two rules, one sentence: a per-tenant document renders duplicates."""
    policy = EdgePolicy(
        rules=(
            EdgeRule(action="deny", target="deploy"),
            EdgeRule(action="deny", target="deploy"),
            EdgeRule(action="allow"),
        )
    )

    assert len(policy.disclosure()) == 1


def test_a_rejection_can_be_fed_back_for_a_second_attempt():
    reg = registry("fetch", "summarise")
    bad = {
        "nodes": [{"name": "shell", "kind": "run_shell"}],
        "edges": [{"source": START, "target": "shell"}],
    }
    model = ScriptedChatModel(responses=[json.dumps(bad), json.dumps(PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog())
    gate = checker(reg)

    first = planner.propose("go")
    rejected = gate.check(first.proposal)
    assert not rejected.admitted

    second = planner.propose("go", feedback=rejected.feedback())
    assert gate.check(second.proposal).admitted

    replan_prompt = "\n".join(str(m.content) for m in model.calls[1])
    assert "registry/unregistered_node" in replan_prompt
    assert "run_shell" in replan_prompt


def test_the_planner_records_the_turn_on_the_trace(trace):
    model = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    PlannerNode(model, trace=trace).propose("go")

    (event,) = [e for e in trace.read_events() if e.phase == "plan"]
    assert event.state_delta["nodes"] == 2
    assert event.state_delta["edges"] == 3
    assert event.tokens > 0


# -- it composes as a GraphARC node --------------------------------------------


class PlanState(GraphARCState):
    task: str
    proposal: Subgraph | None = None
    admitted: bool = False
    rejection: str = ""


def test_the_planner_composes_as_a_node_and_admission_gates_its_output(trace):
    reg = registry("fetch", "summarise")
    model = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog(), trace=trace)
    gate = checker(reg, trace=trace)

    def admit(state: PlanState, ctx) -> dict:
        result = gate.check(state.proposal, meter=ctx.meter, ctx=ctx)
        return {"admitted": result.admitted, "rejection": result.feedback()}

    g = GraphARC(PlanState, name="plan", budget=Budget(max_iterations=10), trace=trace)
    g.add_node("planner", planner, writes=planner.writes)
    g.add_node("admission", admit, writes={"admitted", "rejection"})
    g.add_edge(START, "planner")
    g.add_edge("planner", "admission")
    g.add_edge("admission", END)

    out = g.compile().invoke({"task": "read then summarise"})

    assert out["admitted"] is True
    assert isinstance(out["proposal"], Subgraph)
    assert [n.name for n in out["proposal"].nodes] == ["read", "write"]
    # The proposal reached state; none of its nodes ever ran.
    executed = {e.node for e in trace.read_events() if e.phase == "end"}
    assert executed == {"planner", "admission"}


def test_an_over_budget_plan_is_refused_inside_a_graph_run(trace):
    reg = registry(
        "fetch",
        "summarise",
        fetch=CostEstimate(tokens=100_000),
        summarise=CostEstimate(tokens=100_000),
    )
    model = ScriptedChatModel(responses=[json.dumps(PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog(), trace=trace)
    gate = checker(reg, trace=trace)

    def admit(state: PlanState, ctx) -> dict:
        result = gate.check(state.proposal, meter=ctx.meter, ctx=ctx)
        return {"admitted": result.admitted, "rejection": result.feedback()}

    g = GraphARC(PlanState, name="plan", budget=Budget(max_tokens=50_000), trace=trace)
    g.add_node("planner", planner, writes=planner.writes)
    g.add_node("admission", admit, writes={"admitted", "rejection"})
    g.add_edge(START, "planner")
    g.add_edge("planner", "admission")
    g.add_edge("admission", END)

    out = g.compile().invoke({"task": "read then summarise"})

    assert out["admitted"] is False
    assert "budget/over_token_budget" in out["rejection"]
    admission_events = [e for e in trace.read_events() if e.phase == "admission"]
    assert admission_events and "budget/over_token_budget" in (admission_events[0].error or "")


def test_a_renamed_denied_kind_is_refused_inside_a_graph_run(trace):
    """The reviewer's end-to-end path: a live model names the node, the gate still holds."""
    reg = registry("build", "deploy")
    model = ScriptedChatModel(responses=[json.dumps(RENAMED_PLAN_JSON)])
    planner = PlannerNode(model, catalog=reg.catalog(), trace=trace)
    gate = checker(reg, edge_policy=DENY_DEPLOY, trace=trace)

    def admit(state: PlanState, ctx) -> dict:
        result = gate.check(state.proposal, meter=ctx.meter, ctx=ctx)
        return {"admitted": result.admitted, "rejection": result.feedback()}

    g = GraphARC(PlanState, name="plan", budget=Budget(max_iterations=10), trace=trace)
    g.add_node("planner", planner, writes=planner.writes)
    g.add_node("admission", admit, writes={"admitted", "rejection"})
    g.add_edge(START, "planner")
    g.add_edge("planner", "admission")
    g.add_edge("admission", END)

    out = g.compile().invoke({"task": "ship it"})

    assert out["admitted"] is False
    assert "policy/edge_denied" in out["rejection"]
    assert "deploy" in out["rejection"]
    admission_events = [e for e in trace.read_events() if e.phase == "admission"]
    assert admission_events and "policy/edge_denied" in (admission_events[0].error or "")
    executed = {e.node for e in trace.read_events() if e.phase == "end"}
    assert executed == {"planner", "admission"}  # `helper` never became a node


# -- reachability: structural runnability, opt-in ------------------------------


def test_a_proposal_with_no_entry_edge_is_rejected_when_entry_is_required():
    """The materializer always refused this; admission used to say yes first,
    so the loop learned "could not be built" — a build failure it cannot
    replan against — instead of a rejection carrying a remedy."""
    proposal = Subgraph(
        nodes=(ProposedNode(name="a", kind="step"), ProposedNode(name="b", kind="step")),
        edges=(ProposedEdge(source="a", target="b"),),
    )
    result = checker(
        registry("step"), limits=AdmissionLimits(require_entry=True)
    ).check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.REACHABILITY)
    assert reason.code == "no_entry_edge"
    assert START in reason.remedy


def test_a_node_nothing_reaches_is_rejected_by_name():
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="a", kind="step"),
            ProposedNode(name="orphan", kind="step"),
        ),
        edges=(
            ProposedEdge(source=START, target="a"),
            ProposedEdge(source="a", target=END),
        ),
    )
    result = checker(
        registry("step"), limits=AdmissionLimits(require_entry=True)
    ).check(proposal)

    assert not result.admitted
    (reason,) = result.reasons(Check.REACHABILITY)
    assert reason.code == "unreachable_node"
    assert reason.subject == "orphan"


def test_a_fan_out_and_join_is_reachable():
    """Parallel branches converging is the shape planners propose most; it
    must not read as unreachable."""
    proposal = Subgraph(
        nodes=tuple(
            ProposedNode(name=n, kind="step") for n in ("seed", "l", "r", "join")
        ),
        edges=(
            ProposedEdge(source=START, target="seed"),
            ProposedEdge(source="seed", target="l"),
            ProposedEdge(source="seed", target="r"),
            ProposedEdge(source="l", target="join"),
            ProposedEdge(source="r", target="join"),
            ProposedEdge(source="join", target=END),
        ),
    )
    assert (
        checker(registry("step"), limits=AdmissionLimits(require_entry=True))
        .check(proposal)
        .admitted
    )


def test_entry_is_not_required_by_default_so_a_live_graph_can_be_extended():
    """Admission stays the broader gate: a proposal may name nodes of a graph
    already running, where the entry lives outside the proposal entirely."""
    proposal = Subgraph(
        nodes=(ProposedNode(name="added", kind="step"),),
        edges=(ProposedEdge(source="live", target="added"),),
    )
    result = checker(registry("step"), known_nodes={"live": "step"}).check(proposal)
    assert result.admitted
    assert Check.REACHABILITY not in result.checks_run