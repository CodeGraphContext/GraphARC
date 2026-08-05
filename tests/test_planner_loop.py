"""Gates for materialisation and the governed loop (ROADMAP §5.4).

Admission decides what *may* run. These two modules are what makes that decision
bite: `materialize` is the only path from an admitted proposal to a runnable
graph, and `loop` drives plan -> admit -> materialise -> execute -> observe ->
replan until something machine-readable says stop.

Every test is written so deleting the thing it covers makes it fail. The
registry's bodies append their instance name to a shared list, so "the gate was
bypassed" is observable rather than inferred; the proposals that must not run
name kinds whose bodies would be *visible* if they ran; and the budget tests
assert which node was the last one to execute, not merely that something raised.

No live models: every planner here is a real `PlannerNode` driven by a
`ScriptedChatModel`, so the parse-and-stamp path is exercised for real while the
answers stay fixed.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any

import pytest
from langgraph.types import Command
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from grapharc.harness.permissions import Decision
from grapharc.observe.trace import TraceRecorder
from grapharc.planner import (
    AdmissionChecker,
    AdmissionLimits,
    AdmissionStatus,
    Check,
    CostEstimate,
    EdgePolicy,
    EdgeRule,
    GovernedLoop,
    LoopLimits,
    LoopStop,
    MaterializationError,
    Materializer,
    NodeRegistry,
    NodeSpec,
    NotAdmitted,
    PlannerNode,
    ProposedEdge,
    ProposedNode,
    Subgraph,
    UnadmittedTransition,
)
from grapharc.runtime.budget import Budget, BudgetExceeded
from grapharc.runtime.graph import (
    END,
    START,
    AsyncNodeError,
    CompiledGraphARC,
    WritePermissionError,
)
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel

ALLOW_ALL = EdgePolicy(rules=(EdgeRule(action=Decision.ALLOW),))
DENY_DEPLOY = EdgePolicy(
    rules=(
        EdgeRule(action=Decision.DENY, target="deploy"),
        EdgeRule(action=Decision.DENY, source="deploy"),
        EdgeRule(action=Decision.ALLOW),
    )
)
FREE = CostEstimate()  # declares no cost, so the BUDGET check never fires


class LoopState(GraphARCState):
    """One state contract for the whole run, shared by every round's subgraph."""

    steps: Annotated[list[str], operator.add] = []
    findings: Annotated[list[str], operator.add] = []
    done: bool = False


# -- registry doubles ----------------------------------------------------------


class Bodies:
    """Factories whose bodies record that they actually executed.

    `ran` is the evidence every "this must not run" assertion in this file rests
    on: a bypassed gate shows up as a name in the list, not as a missing log line.
    """

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.built: list[Any] = []

    def step(self, **extra: Any):
        """A body that appends its instance name to `steps` and writes `extra`."""

        def factory(build):
            self.built.append(build)

            def body(state: LoopState) -> dict:
                self.ran.append(build.name)
                return {"steps": [build.name], **extra}

            body.writes = {"steps", *extra}
            return body

        return factory

    def inert(self):
        """A body that runs and changes nothing — the no-progress case."""

        def factory(build):
            def body(state: LoopState) -> None:
                self.ran.append(build.name)
                return None

            body.writes = set()
            return body

        return factory

    def boom(self, message: str):
        def factory(build):
            def body(state: LoopState) -> dict:
                self.ran.append(build.name)
                raise ValueError(message)

            body.writes = {"steps"}
            return body

        return factory

    def routes_to(self, target: str):
        def factory(build):
            def body(state: LoopState):
                self.ran.append(build.name)
                return Command(update={"steps": [build.name]}, goto=target)

            body.writes = {"steps"}
            return body

        return factory


def registry(bodies: Bodies, *, cost: CostEstimate | None = None) -> NodeRegistry:
    """The catalogue every test starts from: cheap, visible, and operator-owned."""
    worst = cost if cost is not None else CostEstimate(iterations=1)
    return NodeRegistry(
        [
            NodeSpec(name="fetch", description="read a source", worst_case=worst,
                     factory=bodies.step()),
            NodeSpec(name="summarise", description="write the answer", worst_case=worst,
                     factory=bodies.step(done=True)),
            NodeSpec(name="triage", description="look for more work", worst_case=worst,
                     factory=bodies.step()),
            NodeSpec(name="deploy", description="ship it", worst_case=worst,
                     factory=bodies.step()),
        ]
    )


# -- proposals as a model would emit them --------------------------------------


def plan(*specs: tuple[str, str], rationale: str = "a plan") -> str:
    """JSON for a linear proposal: START -> n1 -> ... -> END."""
    names = [name for name, _ in specs]
    endpoints = [START, *names, END]
    return json.dumps(
        {
            "rationale": rationale,
            "nodes": [{"name": name, "kind": kind} for name, kind in specs],
            "edges": [
                {"source": a, "target": b}
                for a, b in zip(endpoints, endpoints[1:], strict=False)
            ],
        }
    )


NOTHING_MORE = json.dumps({"rationale": "no further work", "nodes": [], "edges": []})


def subgraph(*specs: tuple[str, str]) -> Subgraph:
    """The same linear shape, built directly rather than through a model."""
    names = [name for name, _ in specs]
    endpoints = [START, *names, END]
    return Subgraph(
        nodes=tuple(ProposedNode(name=name, kind=kind) for name, kind in specs),
        edges=tuple(
            ProposedEdge(source=a, target=b)
            for a, b in zip(endpoints, endpoints[1:], strict=False)
        ),
        rationale="a plan",
    )


def gate(reg: NodeRegistry, *, policy: EdgePolicy = ALLOW_ALL, **kwargs) -> AdmissionChecker:
    return AdmissionChecker(registry=reg, edge_policy=policy, **kwargs)


class CountingChecker(AdmissionChecker):
    """An `AdmissionChecker` that remembers every proposal it was asked about."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checked: list[str] = []

    def check(self, proposal, **kwargs):
        self.checked.append(proposal.proposal_id)
        return super().check(proposal, **kwargs)


def admitted(reg: NodeRegistry, proposal: Subgraph, *, policy: EdgePolicy = ALLOW_ALL):
    result = gate(reg, policy=policy).check(proposal)
    assert result.admitted, result.feedback()
    return result


def build_loop(
    responses: list[str],
    *,
    bodies: Bodies | None = None,
    reg: NodeRegistry | None = None,
    policy: EdgePolicy = ALLOW_ALL,
    budget: Budget | None = None,
    limits: LoopLimits | None = None,
    trace: TraceRecorder | None = None,
    goal_reached=None,
    on_exhausted: str = "raise",
    checker: AdmissionChecker | None = None,
    disclose: bool = False,
    **loop_kwargs,
):
    """A whole cycle wired the way an operator would wire it."""
    bodies = bodies if bodies is not None else Bodies()
    reg = reg if reg is not None else registry(bodies)
    model = ScriptedChatModel(responses=responses, on_exhausted=on_exhausted)
    planner = PlannerNode(
        model,
        catalog=reg.catalog(),
        # Off unless a test asks, so every prompt assertion above stays about
        # what it was written about. The shipped builders pass the policy here;
        # `disclose=True` is that wiring, and it changes the prompt only.
        edge_policy=policy if disclose else None,
        trace=trace,
    )
    loop = GovernedLoop(
        planner=planner,
        checker=checker if checker is not None else gate(reg, policy=policy, trace=trace),
        materializer=Materializer(registry=reg, state_schema=LoopState, trace=trace),
        budget=budget,
        limits=limits,
        trace=trace,
        goal_reached=goal_reached,
        **loop_kwargs,
    )
    return loop, model, bodies


def goal_is_done(state: LoopState) -> bool:
    return state.done


# ==============================================================================
# MATERIALISE — the authorisation is the input, the registry is the only body
# ==============================================================================


def test_an_admitted_proposal_becomes_a_graph_that_runs():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = subgraph(("read", "fetch"), ("write", "summarise"))

    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    assert isinstance(graph, CompiledGraphARC)
    out = graph.invoke(LoopState())
    assert out["steps"] == ["read", "write"]
    assert out["done"] is True
    assert bodies.ran == ["read", "write"]


def test_materialising_a_rejected_proposal_raises():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = subgraph(("read", "fetch"), ("wipe", "rm_rf"))
    verdict = gate(reg).check(proposal)
    assert verdict.status is AdmissionStatus.REJECTED

    with pytest.raises(NotAdmitted) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(verdict, proposal)

    # The refusal carries the reason, so a caller does not have to re-run the gate.
    assert "was not admitted" in str(excinfo.value)
    assert "unregistered_node" in str(excinfo.value)
    assert bodies.ran == []


def test_a_pending_approval_is_not_a_licence_to_build():
    """NEEDS_APPROVAL is a stop, not a slow yes — materialisation must agree."""
    bodies = Bodies()
    reg = registry(bodies)
    ask = EdgePolicy(rules=(EdgeRule(action=Decision.ASK),))
    proposal = subgraph(("read", "fetch"))
    verdict = gate(reg, policy=ask).check(proposal)
    assert verdict.status is AdmissionStatus.NEEDS_APPROVAL

    with pytest.raises(NotAdmitted):
        Materializer(registry=reg, state_schema=LoopState).materialize(verdict, proposal)
    assert bodies.ran == []


def test_an_admission_result_cannot_authorise_a_different_proposal():
    """A genuine approval plus an edited proposal is refused by content hash."""
    bodies = Bodies()
    reg = registry(bodies)
    approved = subgraph(("read", "fetch"))
    verdict = admitted(reg, approved)
    swapped = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch"), ProposedNode(name="ship", kind="deploy")),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target="ship"),
            ProposedEdge(source="ship", target=END),
        ),
        proposal_id=approved.proposal_id,  # even the id is copied
    )

    with pytest.raises(NotAdmitted) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(verdict, swapped)

    assert "what runs must be what was admitted" in str(excinfo.value)
    assert bodies.ran == []


def test_a_proposal_on_its_own_is_not_an_argument():
    """There is no entry point that takes a bare Subgraph; passing one says so."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = subgraph(("read", "fetch"))

    with pytest.raises(NotAdmitted) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            proposal, admitted(reg, proposal)
        )

    assert "takes the AdmissionResult first" in str(excinfo.value)


def test_a_node_body_cannot_come_from_the_proposal():
    """Whatever `args` says a node is, the registry's body for its kind is what runs."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(
            ProposedNode(
                name="read",
                kind="fetch",
                # Every field name a materialiser might be tempted to honour.
                args={"body": "deploy", "kind": "deploy", "factory": "deploy", "run": "deploy"},
            ),
        ),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target=END),
        ),
    )

    graph = Materializer(
        registry=reg, state_schema=LoopState, forward_args=True
    ).materialize(admitted(reg, proposal), proposal)
    out = graph.invoke(LoopState())

    assert bodies.built[-1].kind == "fetch"  # the proposal's `kind` field, not its args
    assert bodies.ran == ["read"]
    assert out["steps"] == ["read"]
    assert out["done"] is False  # `deploy`'s body was never anywhere near this


def test_a_callable_cannot_be_smuggled_through_args_because_the_gate_hashes_them():
    """`args` is JSON a model wrote; a Python callable does not survive fingerprinting."""
    bodies = Bodies()
    reg = registry(bodies)

    def evil(state):  # pragma: no cover - running it would be the failure
        return {"steps": ["evil"]}

    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch", args={"body": evil}),),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target=END),
        ),
    )

    # Every admission decision fingerprints the proposal, so this never returns
    # an AdmissionResult at all — and materialisation demands one.
    with pytest.raises(PydanticSerializationError):
        gate(reg).check(proposal)
    assert bodies.ran == []


def test_a_proposed_node_has_no_field_a_body_could_arrive_in():
    """The other half of the boundary: the type forbids the field entirely."""
    for field in ("body", "factory", "fn", "run"):
        with pytest.raises(ValidationError):
            ProposedNode.model_validate({"name": "read", "kind": "fetch", field: "anything"})


def test_a_planners_args_do_not_reach_a_factory_unless_the_operator_opts_in():
    """Admission does not inspect `args`, so forwarding them is opt-in here."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch", args={"depth": 3}),),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target=END),
        ),
    )
    verdict = admitted(reg, proposal)

    Materializer(registry=reg, state_schema=LoopState).materialize(verdict, proposal)
    assert bodies.built[-1].args == {}

    Materializer(registry=reg, state_schema=LoopState, forward_args=True).materialize(
        verdict, proposal
    )
    assert bodies.built[-1].args == {"depth": 3}


def test_a_factory_is_told_which_instance_it_is_building():
    """One kind, three instances: the fan-out case the name/kind split exists for."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=tuple(ProposedNode(name=n, kind="fetch") for n in ("a", "b", "c")),
        edges=(
            ProposedEdge(source=START, target="a"),
            ProposedEdge(source="a", target="b"),
            ProposedEdge(source="b", target="c"),
            ProposedEdge(source="c", target=END),
        ),
    )

    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    assert [b.name for b in bodies.built] == ["a", "b", "c"]
    assert {b.kind for b in bodies.built} == {"fetch"}
    assert {b.fingerprint for b in bodies.built} == {proposal.fingerprint()}
    assert graph.invoke(LoopState())["steps"] == ["a", "b", "c"]


def test_a_kind_without_a_factory_has_no_body_to_offer():
    reg = NodeRegistry([NodeSpec(name="fetch", factory=None)])
    proposal = subgraph(("read", "fetch"))

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )

    assert "carries no factory" in str(excinfo.value)


def test_a_kind_dropped_from_the_registry_after_admission_is_refused():
    """The decision and the build are separate moments; the build re-checks."""
    bodies = Bodies()
    proposal = subgraph(("read", "fetch"))
    verdict = admitted(registry(bodies), proposal)
    emptied = NodeRegistry([NodeSpec(name="summarise", factory=bodies.step())])

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=emptied, state_schema=LoopState).materialize(verdict, proposal)

    assert "not in the node registry" in str(excinfo.value)


def test_a_factory_that_returns_a_non_callable_is_refused():
    reg = NodeRegistry([NodeSpec(name="fetch", factory=lambda build: "not a node")])
    proposal = subgraph(("read", "fetch"))

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )

    assert "must return the node's callable body" in str(excinfo.value)


# -- built through the kernel, not around it -----------------------------------


def test_the_graph_goes_through_the_kernel_so_write_permissions_still_apply():
    """A body that writes what the operator did not declare is stopped by GraphARC."""

    def overreaching(build):
        def body(state: LoopState) -> dict:
            return {"steps": ["read"], "done": True}

        body.writes = {"steps"}  # `done` is not declared
        return body

    reg = NodeRegistry([NodeSpec(name="fetch", factory=overreaching)])
    proposal = subgraph(("read", "fetch"))
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    with pytest.raises(WritePermissionError):
        graph.invoke(LoopState())


def test_a_kind_nobody_declared_writes_for_may_write_nothing():
    def undeclared(build):
        def body(state: LoopState) -> dict:
            return {"steps": ["read"]}

        return body  # no `writes` attribute, and no writes table entry

    reg = NodeRegistry([NodeSpec(name="fetch", factory=undeclared)])
    proposal = subgraph(("read", "fetch"))
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    with pytest.raises(WritePermissionError):
        graph.invoke(LoopState())


def test_a_body_that_wants_the_run_context_still_gets_it():
    """The confinement wrapper preserves arity, which is how the kernel decides."""
    seen: list[str] = []

    def wants_ctx(build):
        def body(state: LoopState, ctx) -> dict:
            seen.append(ctx.run_id)
            return {"steps": [build.name]}

        body.writes = {"steps"}
        return body

    reg = NodeRegistry([NodeSpec(name="fetch", factory=wants_ctx)])
    proposal = subgraph(("read", "fetch"))
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    graph.invoke(LoopState(), run_id="run-1234")
    assert seen == ["run-1234"]


@pytest.mark.asyncio
async def test_an_async_body_stays_async_all_the_way_to_the_kernel():
    bodies = Bodies()

    def factory(build):
        async def body(state: LoopState) -> dict:
            bodies.ran.append(build.name)
            return {"steps": [build.name]}

        body.writes = {"steps"}
        return body

    reg = NodeRegistry([NodeSpec(name="fetch", factory=factory)])
    proposal = subgraph(("read", "fetch"))
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    # The kernel classified it as async, which it can only do if the wrapper
    # kept the body a coroutine function.
    with pytest.raises(AsyncNodeError):
        graph.invoke(LoopState())

    out = await graph.ainvoke(LoopState())
    assert out["steps"] == ["read"]
    assert bodies.ran == ["read"]


def test_the_writes_table_is_keyed_on_the_kind_an_operator_registered():
    def undeclared(build):
        def body(state: LoopState) -> dict:
            return {"steps": [build.name]}

        return body

    reg = NodeRegistry([NodeSpec(name="fetch", factory=undeclared)])
    proposal = subgraph(("read", "fetch"))
    graph = Materializer(
        registry=reg, state_schema=LoopState, writes={"fetch": {"steps"}}
    ).materialize(admitted(reg, proposal), proposal)

    assert graph.invoke(LoopState())["steps"] == ["read"]


def test_the_run_is_budgeted_and_traced_like_any_other_graph(trace):
    bodies = Bodies()
    reg = registry(bodies)
    proposal = subgraph(("read", "fetch"), ("write", "summarise"))
    graph = Materializer(registry=reg, state_schema=LoopState, trace=trace).materialize(
        admitted(reg, proposal), proposal
    )

    with pytest.raises(BudgetExceeded) as excinfo:
        graph.invoke(LoopState(), budget=Budget(max_iterations=1))

    assert "max_iterations" in str(excinfo.value)
    assert bodies.ran == ["read"]  # the ceiling bit before the second node
    phases = {(e.node, e.phase) for e in trace.read_events()}
    assert ("read", "start") in phases and ("read", "end") in phases


# -- topology ------------------------------------------------------------------


def test_fan_out_and_join_come_from_edge_multiplicity():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="split", kind="fetch"),
            ProposedNode(name="left", kind="triage"),
            ProposedNode(name="right", kind="triage"),
            ProposedNode(name="join", kind="summarise"),
        ),
        edges=(
            ProposedEdge(source=START, target="split"),
            ProposedEdge(source="split", target="left"),
            ProposedEdge(source="split", target="right"),
            ProposedEdge(source="left", target="join"),
            ProposedEdge(source="right", target="join"),
            ProposedEdge(source="join", target=END),
        ),
    )

    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )
    out = graph.invoke(LoopState())

    assert out["steps"][0] == "split"
    assert set(out["steps"][1:3]) == {"left", "right"}
    assert out["steps"][3] == "join"  # the join waited for both branches


def test_dynamic_routing_stays_inside_the_admitted_edges():
    """A registry body may still route — but only along an edge the gate approved."""
    bodies = Bodies()
    reg = NodeRegistry(
        [
            NodeSpec(name="hop", factory=bodies.routes_to("second")),
            NodeSpec(name="plain", factory=bodies.step()),
        ]
    )
    proposal = Subgraph(
        nodes=(ProposedNode(name="first", kind="hop"), ProposedNode(name="second", kind="plain")),
        edges=(
            ProposedEdge(source=START, target="first"),
            ProposedEdge(source="first", target="second"),
            ProposedEdge(source="second", target=END),
        ),
    )
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    assert graph.invoke(LoopState())["steps"] == ["first", "second"]


def test_a_body_cannot_route_along_an_edge_the_proposal_never_declared():
    bodies = Bodies()
    reg = NodeRegistry(
        [
            NodeSpec(name="plain", factory=bodies.step()),
            NodeSpec(name="back", factory=bodies.routes_to("first")),
        ]
    )
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="first", kind="plain"),
            ProposedNode(name="second", kind="back"),
            ProposedNode(name="third", kind="plain"),
        ),
        edges=(
            ProposedEdge(source=START, target="first"),
            ProposedEdge(source="first", target="second"),
            ProposedEdge(source="second", target="third"),
            ProposedEdge(source="third", target=END),
        ),
    )
    graph = Materializer(registry=reg, state_schema=LoopState).materialize(
        admitted(reg, proposal), proposal
    )

    with pytest.raises(UnadmittedTransition) as excinfo:
        graph.invoke(LoopState())

    assert "declared no edge to" in str(excinfo.value)
    assert bodies.ran == ["first", "second"]  # third was never reached


def test_an_edge_to_a_node_outside_the_proposal_is_refused():
    """Admission can approve an edge to a live node; this builds a standalone graph."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch"),),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target="already_running"),
        ),
    )
    checker = AdmissionChecker(
        registry=reg,
        edge_policy=ALLOW_ALL,
        known_nodes={"already_running": "summarise"},
    )
    verdict = checker.check(proposal)
    assert verdict.admitted

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(verdict, proposal)

    assert "already_running" in str(excinfo.value)


def test_a_node_start_cannot_reach_is_refused_rather_than_silently_skipped():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch"), ProposedNode(name="orphan", kind="triage")),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target=END),
        ),
    )

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )

    assert "orphan" in str(excinfo.value)
    assert "cannot be reached from START" in str(excinfo.value)


def test_a_proposal_with_no_entry_point_is_refused():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch"),),
        edges=(ProposedEdge(source="read", target=END),),
    )

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )

    assert "no edge out of START" in str(excinfo.value)


def test_an_empty_proposal_authorises_nothing_and_builds_nothing():
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(rationale="no further work")

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )

    assert "declares no nodes" in str(excinfo.value)


def test_a_nested_subgraph_is_refused_rather_than_silently_dropped():
    """A checker configured to allow depth 2 can admit one; this builds flat graphs."""
    bodies = Bodies()
    reg = registry(bodies)
    proposal = Subgraph(
        nodes=(ProposedNode(name="read", kind="fetch", subgraph=subgraph(("write", "summarise"))),),
        edges=(
            ProposedEdge(source=START, target="read"),
            ProposedEdge(source="read", target=END),
        ),
    )
    verdict = gate(reg, limits=AdmissionLimits(max_depth=2)).check(proposal)
    assert verdict.admitted

    with pytest.raises(MaterializationError) as excinfo:
        Materializer(registry=reg, state_schema=LoopState).materialize(verdict, proposal)

    assert "nested subgraph" in str(excinfo.value)
    assert bodies.ran == []


# ==============================================================================
# THE GOVERNED LOOP
# ==============================================================================


def test_a_run_plans_admits_executes_and_reaches_the_goal(trace):
    loop, model, bodies = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        trace=trace,
        goal_reached=goal_is_done,
    )

    result = loop.run("summarise the incident", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert result.succeeded
    assert model.call_count == 1
    assert bodies.ran == ["read", "write"]
    assert result.state.steps == ["read", "write"]
    assert result.state.done is True
    assert len(result.rounds) == 1
    assert result.rounds[0].admitted and result.rounds[0].executed
    assert result.usage["iterations"] == 2  # both node executions folded back in
    assert result.usage["tokens"] > 0


def test_a_round_records_the_iterations_it_spent(trace):
    """`RoundRecord.iterations` was declared and never assigned, so every round
    reported 0 while the run's meter counted the same work."""
    loop, _model, _bodies = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        trace=trace,
        goal_reached=goal_is_done,
    )

    result = loop.run("summarise the incident", LoopState())

    assert [r.iterations for r in result.rounds] == [2]
    assert sum(r.iterations for r in result.rounds) == result.usage["iterations"]


def test_the_planners_tokens_are_counted_once_not_once_per_round(trace):
    """A `round` event is an envelope, not a measurement.

    Its tokens are the planner's, already reported by the `plan` event, and both
    land in the set `metrics`/`cost` add on top of node totals — so the planner's
    spend was charged to the report twice.
    """
    from grapharc.observe import cost, metrics

    loop, _model, _bodies = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        trace=trace,
        goal_reached=goal_is_done,
    )
    result = loop.run("summarise the incident", LoopState(), run_id="r1")

    real = result.usage["tokens"]
    assert real > 0
    assert metrics.summarize(trace, "r1").tokens == real
    assert cost.attribute(trace, "r1").tokens == real


def test_a_round_event_still_reports_what_the_round_itself_spent(trace):
    """Removing the double count must not remove the information."""
    loop, _model, _bodies = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        trace=trace,
        goal_reached=goal_is_done,
    )
    loop.run("summarise the incident", LoopState(), run_id="r1")

    rounds = [e for e in trace.read_events("r1") if e.phase == "round"]
    assert rounds, "the round event went missing"
    for event in rounds:
        # Not in the typed fields any reader sums...
        assert event.tokens is None
        assert event.duration_ms is None
        # ...but still answerable from the file.
        assert event.state_delta["round_tokens"] > 0
        assert event.state_delta["round_iterations"] == 2
        assert event.state_delta["round_duration_ms"] >= 0


def test_a_rejection_reaches_the_planner_and_the_next_proposal_completes_the_run():
    """The replanning edge of ARCHITECTURE §2: rejected + reason -> propose again."""
    loop, model, bodies = build_loop(
        [
            plan(("wipe", "rm_rf")),  # not in the registry
            plan(("read", "fetch"), ("write", "summarise")),
        ],
        goal_reached=goal_is_done,
    )

    result = loop.run("summarise the incident", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert len(result.rounds) == 2
    first, second = result.rounds
    assert not first.admitted and not first.executed
    assert [r.code for r in first.rejections] == ["unregistered_node"]
    assert second.admitted and second.executed
    # Nothing from the refused round ran, in either round.
    assert bodies.ran == ["read", "write"]

    replanning_turn = " ".join(str(m.content) for m in model.calls[1])
    assert "registry/unregistered_node" in replanning_turn
    assert "rm_rf" in replanning_turn


def test_discovered_work_re_enters_admission_on_the_second_round():
    """Round 2's proposal is gated exactly like round 1's — no 'already approved' path."""
    bodies = Bodies()
    reg = registry(bodies)
    counting = CountingChecker(registry=reg, edge_policy=DENY_DEPLOY)
    loop, model, bodies = build_loop(
        [
            plan(("look", "triage")),  # admitted: discovers work
            plan(("ship", "deploy")),  # discovered mid-run, and denied by policy
            plan(("write", "summarise")),  # legal follow-up
        ],
        bodies=bodies,
        reg=reg,
        checker=counting,
        goal_reached=goal_is_done,
    )

    result = loop.run("investigate and fix", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert len(counting.checked) == 3  # every round went through the gate
    assert [r.admitted for r in result.rounds] == [True, False, True]
    assert result.rounds[1].admission.failed_checks() == (Check.POLICY,)
    # The proof: the denied kind's body never executed, on round 2 or any other.
    assert bodies.ran == ["look", "write"]


def test_a_later_round_cannot_rename_its_way_past_a_denial_the_first_round_respected():
    """The rename evasion, composed with the loop: behaving once buys nothing later.

    `test_discovered_work_re_enters_admission_on_the_second_round` proves round 2
    is checked; this proves it is checked *on the kind*. A planner that proposes
    a legal graph first and then dresses `deploy` up under a name the earlier,
    admitted round used is refused every time — so "the gate keys on the registry
    kind" survives being composed with "the gate runs every round", which is the
    pair of claims an escalating planner would have to break together.
    """
    bodies = Bodies()
    reg = registry(bodies)
    loop, model, bodies = build_loop(
        [
            plan(("look", "triage")),  # honest round: admitted, and it runs
            plan(("look", "deploy")),  # same instance name, denied kind underneath
        ],
        bodies=bodies,
        reg=reg,
        policy=DENY_DEPLOY,
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=8, max_consecutive_rejections=3),
        goal_reached=goal_is_done,
    )

    result = loop.run("investigate, then ship it", LoopState())

    assert result.stop is LoopStop.ADMISSION_REFUSED
    assert [r.admitted for r in result.rounds] == [True, False, False, False]
    # Denied on the kind, and the rejection says so rather than naming "look".
    denials = {r.code for r in result.rejections()}
    assert denials == {"edge_denied"}
    assert "deploy" in result.rounds[1].rejections[0].detail
    # The evidence: `deploy`'s body never executed, and the one admitted round did.
    assert bodies.ran == ["look"]
    assert result.state.steps == ["look"]


@pytest.mark.parametrize(
    "reserved", ["__interrupt__", "__pregel_pull", "__pregel_push", "__copy__", "__anything__"]
)
def test_a_name_in_the_orchestrators_namespace_is_refused_by_the_schema(reserved):
    """A model-chosen string must not be able to reach LangGraph's reserved set.

    `__start__` and `__end__` were refused; the rest of the dunder space was not.
    A proposal naming `__interrupt__` was *admitted* — names are deliberately not
    governed by policy — and then LangGraph raised a bare `ValueError` from
    inside `add_node`, which is neither a `MaterializationError` nor anything the
    loop catches. The whole prefix is refused rather than the members enumerated,
    because the membership list is LangGraph's private constant and grows.
    """
    with pytest.raises(ValidationError, match="orchestrator's own namespace"):
        ProposedNode(name=reserved, kind="fetch")


def test_an_unusable_name_stops_the_run_for_a_reason_instead_of_raising():
    """The loop's "every stop is a reason" contract, against the crash above."""
    loop, model, bodies = build_loop(
        [plan(("__interrupt__", "fetch"))],
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=9, max_consecutive_planning_failures=3),
    )

    result = loop.run("go", LoopState())

    assert result.stop is LoopStop.PLANNING_FAILED
    assert len(result.rounds) == 3
    assert bodies.ran == []
    assert "orchestrator's own namespace" in result.rounds[0].planner_error


def test_a_writes_table_naming_an_unknown_field_is_refused_before_anything_is_registered():
    """`MaterializationError`'s docstring promises this happens before registration.

    It did not: the kernel raised `WritePermissionError` at `add_node`, a class
    this module does not document and `GovernedLoop._execute` does not catch, so
    an operator typo escaped `run()` as a crash.
    """
    bodies = Bodies()
    reg = registry(bodies)
    proposal = subgraph(("read", "fetch"))

    with pytest.raises(MaterializationError, match="LoopState has no field for"):
        Materializer(
            registry=reg, state_schema=LoopState, writes={"fetch": {"stpes"}}
        ).materialize(admitted(reg, proposal), proposal)

    assert bodies.built == []  # refused before a single factory was called


def test_no_foreign_exception_escapes_materialisation():
    """A body's own `writes` is not known until its factory has run.

    That case cannot be checked up front, so it is converted instead: whatever
    the kernel or LangGraph raises during assembly comes out as
    `MaterializationError`, which is the class the loop catches and records.
    """

    def factory(build):
        def body(state: LoopState) -> dict:
            return {}

        body.writes = {"not_a_field_on_LoopState"}
        return body

    reg = NodeRegistry([NodeSpec(name="fetch", worst_case=FREE, factory=factory)])
    proposal = subgraph(("read", "fetch"))

    with pytest.raises(MaterializationError, match="could not be assembled into a graph"):
        Materializer(registry=reg, state_schema=LoopState).materialize(
            admitted(reg, proposal), proposal
        )


@pytest.mark.parametrize("overridden", ["fingerprint", "model_dump_json"])
def test_a_subclass_cannot_lie_about_what_it_contains(overridden):
    """Matching a proposal to its authorisation ends in a call *on the proposal*.

    Two attacks, one root cause. Overriding `fingerprint()` to return an
    admitted proposal's hash ran a policy-denied `deploy`. Calling the unbound
    `Subgraph.fingerprint(proposal)` looked like the fix and was not: that
    function calls `self.model_dump_json()`, so overriding *that* instead
    defeats it identically — verified, `deploy` executed again.

    There is no method here a subclass cannot reach, so the subclass is refused
    rather than the methods hardened one at a time. Both variants are
    parametrised so neither can be fixed without the other.

    This still does not make an `AdmissionResult` unforgeable — a caller who
    builds one by hand with a matching fingerprint materialises anything, and no
    library check stops code that already has the interpreter. See the
    trust-boundary note in the module docstring.
    """
    bodies = Bodies()
    reg = registry(bodies)
    honest = subgraph(("read", "fetch"))
    denied = subgraph(("ship", "deploy"))
    approval = gate(reg, policy=DENY_DEPLOY).check(honest)
    assert approval.admitted

    if overridden == "fingerprint":

        class Liar(Subgraph):
            def fingerprint(self) -> str:
                return honest.fingerprint()

    else:

        class Liar(Subgraph):  # type: ignore[no-redef]
            def model_dump_json(self, **kwargs) -> str:
                return honest.model_dump_json(**kwargs)

    smuggled = Liar(
        nodes=denied.nodes,
        edges=denied.edges,
        rationale="looks like the admitted one",
        proposal_id=honest.proposal_id,
    )

    with pytest.raises(NotAdmitted, match="A subclass is refused rather than trusted"):
        Materializer(registry=reg, state_schema=LoopState).materialize(approval, smuggled)
    assert bodies.ran == []


def test_a_frozen_registry_cannot_be_widened_underneath_a_decision():
    """`GovernedLoop` re-checks every round against the same registry *object*.

    Same object is not same contents while `register()` stays callable, so a
    node body could add a kind between rounds and have round 2 admitted against
    an allowlist round 1 never saw. `freeze()` is the mitigation the loop
    docstring points at.
    """
    reg = registry(Bodies())
    reg.register(NodeSpec(name="late", worst_case=FREE))  # fine before freezing
    assert "late" in reg.names()

    reg.freeze()
    assert reg.frozen
    with pytest.raises(ValueError, match="this registry is frozen"):
        reg.register(NodeSpec(name="later_still", worst_case=FREE))
    assert "later_still" not in reg.names()


def test_a_planner_that_keeps_proposing_an_unregistered_node_stops_for_a_reason():
    loop, model, bodies = build_loop(
        [plan(("wipe", "rm_rf"))],
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=20, max_consecutive_rejections=3),
        goal_reached=goal_is_done,
    )

    result = loop.run("delete everything", LoopState())

    assert result.stop is LoopStop.ADMISSION_REFUSED
    assert "3 consecutive proposals were not admitted" in result.detail
    assert len(result.rounds) == 3  # it stopped at the limit, not at max_rounds
    assert model.call_count == 3
    assert bodies.ran == []
    assert {r.code for r in result.rejections()} == {"unregistered_node"}


def test_budget_exhaustion_stops_the_loop_before_the_next_round_is_planned():
    loop, model, bodies = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        budget=Budget(max_iterations=2),
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10),
    )

    result = loop.run("keep going forever", LoopState())

    assert result.stop is LoopStop.BUDGET_EXHAUSTED
    assert "max_iterations reached (2/2)" in result.detail
    assert len(result.executed_rounds) == 1
    assert bodies.ran == ["read", "write"]
    assert model.call_count == 1  # round 2 was never even planned


def test_a_round_executes_under_what_the_run_has_left_not_a_fresh_budget():
    """The per-round ceiling is derived from the loop's meter, so rounds share one pool."""
    bodies = Bodies()
    loop, model, _ = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        bodies=bodies,
        reg=registry(bodies, cost=FREE),  # admission's BUDGET check cannot fire
        budget=Budget(max_iterations=1),
        on_exhausted="repeat",
    )

    result = loop.run("do two things on a one-node budget", LoopState())

    assert result.stop is LoopStop.BUDGET_EXHAUSTED
    assert bodies.ran == ["read"]  # the ceiling bit inside the round
    assert result.executed_rounds == ()


def test_the_gate_refuses_work_the_remaining_budget_cannot_cover():
    """Budget flows into admission: the worst case is checked against what is left."""
    bodies = Bodies()
    loop, model, _ = build_loop(
        [plan(("read", "fetch"), ("write", "summarise"))],
        bodies=bodies,
        budget=Budget(max_iterations=3),
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10, max_consecutive_rejections=1),
    )

    result = loop.run("keep going", LoopState())

    assert result.stop is LoopStop.ADMISSION_REFUSED
    assert [r.code for r in result.rejections()] == ["over_iteration_budget"]
    assert bodies.ran == ["read", "write"]  # only the first round's work


def test_no_progress_stops_the_loop():
    bodies = Bodies()
    reg = NodeRegistry([NodeSpec(name="fetch", factory=bodies.inert())])
    loop, model, _ = build_loop(
        [plan(("read", "fetch"))],
        bodies=bodies,
        reg=reg,
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10, max_stalled_rounds=2),
    )

    result = loop.run("change something", LoopState())

    assert result.stop is LoopStop.NO_PROGRESS
    assert len(result.executed_rounds) == 2
    assert all(not r.progressed for r in result.executed_rounds)


def test_max_rounds_stops_the_loop():
    loop, model, bodies = build_loop(
        [plan(("read", "fetch"))],
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=2, max_stalled_rounds=99),
    )

    result = loop.run("never finish", LoopState())

    assert result.stop is LoopStop.MAX_ROUNDS
    assert len(result.rounds) == 2
    assert bodies.ran == ["read", "read"]


def test_a_human_halt_stops_the_loop_at_the_next_round_boundary():
    holder: dict[str, GovernedLoop] = {}
    bodies = Bodies()

    def pull_the_cord(build):
        def body(state: LoopState) -> dict:
            bodies.ran.append(build.name)
            holder["loop"].request_halt("operator pulled the cord")
            return {"steps": [build.name]}

        body.writes = {"steps"}
        return body

    reg = NodeRegistry([NodeSpec(name="fetch", factory=pull_the_cord)])
    loop, model, _ = build_loop(
        [plan(("read", "fetch"))],
        bodies=bodies,
        reg=reg,
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10),
    )
    holder["loop"] = loop

    result = loop.run("run until told otherwise", LoopState())

    assert result.stop is LoopStop.HUMAN_STOPPED
    assert result.detail == "operator pulled the cord"
    assert bodies.ran == ["read"]  # the running round finished; no second one started
    assert model.call_count == 1


def test_a_halt_requested_before_the_run_prevents_any_planning():
    loop, model, bodies = build_loop([plan(("read", "fetch"))])
    loop.request_halt("maintenance window")

    result = loop.run("do the thing", LoopState())

    assert result.stop is LoopStop.HUMAN_STOPPED
    assert result.rounds == ()
    assert model.call_count == 0


def test_an_admitted_empty_proposal_ends_the_run_with_no_further_work():
    loop, model, bodies = build_loop([NOTHING_MORE])

    result = loop.run("nothing needs doing", LoopState())

    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert not result.succeeded
    assert result.rounds[-1].admitted
    assert bodies.ran == []


def test_a_goal_already_met_is_not_a_reason_to_plan():
    loop, model, bodies = build_loop([plan(("read", "fetch"))], goal_reached=goal_is_done)

    result = loop.run("already finished", LoopState(done=True))

    assert result.stop is LoopStop.GOAL_MET
    assert model.call_count == 0
    assert result.rounds == ()


def test_unusable_planner_replies_stop_the_run_for_a_recorded_reason():
    loop, model, bodies = build_loop(
        ["I would rather not."],
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10, max_consecutive_planning_failures=3),
    )

    result = loop.run("do something", LoopState())

    assert result.stop is LoopStop.PLANNING_FAILED
    assert len(result.rounds) == 3
    assert all(r.planner_error for r in result.rounds)
    assert bodies.ran == []


def test_a_subgraph_that_raises_is_replanned_from_and_then_given_up_on():
    bodies = Bodies()
    reg = NodeRegistry(
        [
            NodeSpec(name="fetch", factory=bodies.boom("the source is down")),
            NodeSpec(name="summarise", factory=bodies.step(done=True)),
        ]
    )
    loop, model, _ = build_loop(
        [plan(("read", "fetch"))],
        bodies=bodies,
        reg=reg,
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10, max_consecutive_execution_failures=2),
    )

    result = loop.run("read the source", LoopState())

    assert result.stop is LoopStop.EXECUTION_FAILED
    assert len(result.rounds) == 2
    assert all(r.admitted and not r.executed for r in result.rounds)
    assert "the source is down" in result.rounds[0].execution_error
    # The failure was fed back so the planner had a chance to route around it.
    assert "did not run" in " ".join(str(m.content) for m in model.calls[1])


def test_a_proposal_that_cannot_be_built_is_a_replanning_input_not_a_crash():
    bodies = Bodies()
    orphan = json.dumps(
        {
            "rationale": "an unreachable node",
            "nodes": [{"name": "read", "kind": "fetch"}, {"name": "lost", "kind": "triage"}],
            "edges": [{"source": START, "target": "read"}, {"source": "read", "target": END}],
        }
    )
    loop, model, bodies = build_loop(
        [orphan, plan(("read", "fetch"), ("write", "summarise"))],
        bodies=bodies,
        goal_reached=goal_is_done,
        limits=LoopLimits(max_rounds=10, max_consecutive_execution_failures=2),
    )

    result = loop.run("do the work", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert result.rounds[0].admitted and not result.rounds[0].executed
    assert "cannot be reached from START" in result.rounds[0].materialization_error
    assert bodies.ran == ["read", "write"]


def test_state_accumulates_across_rounds():
    loop, model, bodies = build_loop(
        [plan(("read", "fetch")), plan(("write", "summarise"))],
        goal_reached=goal_is_done,
    )

    result = loop.run("two rounds of work", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert result.state.steps == ["read", "write"]  # round 2 saw round 1's writes


def test_the_planner_is_told_what_the_state_and_the_budget_look_like():
    """`observe outcome + remaining budget` is an input to the next round, not decoration."""
    loop, model, bodies = build_loop(
        [plan(("read", "fetch")), NOTHING_MORE],
        budget=Budget(max_iterations=9),
        observe=lambda state: f"steps so far: {state.steps}",
    )

    result = loop.run("summarise", LoopState())
    seen = [" ".join(str(m.content) for m in call) for call in model.calls]

    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert "steps so far: []" in seen[0]
    assert "steps so far: ['read']" in seen[1]
    assert "iterations: 9" in seen[0]
    assert "iterations: 8" in seen[1]  # round one's node execution was charged


# -- the audit trail -----------------------------------------------------------


def test_every_round_is_auditable_from_the_trace_alone(trace):
    loop, model, bodies = build_loop(
        [plan(("wipe", "rm_rf")), plan(("read", "fetch"), ("write", "summarise"))],
        trace=trace,
        goal_reached=goal_is_done,
    )

    result = loop.run("summarise the incident", LoopState())
    events = trace.read_events(result.run_id)

    assert events, "the run left no trace at all"
    assert {e.run_id for e in events} == {result.run_id}

    by_phase: dict[str, list] = {}
    for event in events:
        by_phase.setdefault(event.phase, []).append(event)

    # What it was allowed to do: one admission verdict per proposal, carrying the
    # fingerprint of the exact proposal that was judged.
    verdicts = by_phase["admission"]
    assert [e.state_delta["status"] for e in verdicts] == ["rejected", "admitted"]
    assert verdicts[1].state_delta["fingerprint"] == result.rounds[1].proposal.fingerprint()
    assert verdicts[0].error and "registry/unregistered_node" in verdicts[0].error

    # What it did: the planner's turns and the nodes that actually executed.
    assert len(by_phase["plan"]) == 2
    assert {e.node for e in by_phase["end"]} == {"read", "write"}

    # Why it stopped, and a summary line per round.
    assert [e.state_delta["round"] for e in by_phase["round"]] == [1, 2]
    assert by_phase["round"][0].state_delta["rejections"] == ["registry/unregistered_node"]
    (stop,) = by_phase["stop"]
    assert stop.state_delta["stop"] == "goal_met"
    assert stop.error is None


def test_a_refused_run_records_the_refusal_as_an_error_on_the_trace(trace):
    loop, model, bodies = build_loop(
        [plan(("wipe", "rm_rf"))],
        trace=trace,
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=10, max_consecutive_rejections=2),
    )

    result = loop.run("delete everything", LoopState())
    (stop,) = [e for e in trace.read_events(result.run_id) if e.phase == "stop"]

    assert result.stop is LoopStop.ADMISSION_REFUSED
    assert stop.error.startswith("admission_refused:")
    assert stop.state_delta["executed_rounds"] == 0


def test_the_planner_hitting_the_runs_ceiling_is_a_stop_not_an_exception():
    """`PlannerNode.propose` lets `BudgetExceeded` through on purpose; the loop lands it."""

    class BrokePlanner:
        """A minimal `Planner`: structural typing, no `PlannerNode` involved."""

        def propose(self, task, ctx=None, *, feedback=""):
            raise BudgetExceeded("max_tokens reached (120/100)")

    bodies = Bodies()
    reg = registry(bodies)
    loop = GovernedLoop(
        planner=BrokePlanner(),
        checker=gate(reg),
        materializer=Materializer(registry=reg, state_schema=LoopState),
    )

    result = loop.run("plan something expensive", LoopState())

    assert result.stop is LoopStop.BUDGET_EXHAUSTED
    assert result.detail == "max_tokens reached (120/100)"
    assert len(result.rounds) == 1
    assert bodies.ran == []


def test_a_loop_nested_inside_a_node_has_to_declare_its_depth():
    """`max_depth` is only as honest as the `parent_depth` its caller reports."""
    loop, model, bodies = build_loop(
        [plan(("read", "fetch"))],
        parent_depth=1,  # this driver is already one level down
        on_exhausted="repeat",
        limits=LoopLimits(max_rounds=5, max_consecutive_rejections=1),
    )

    result = loop.run("recurse", LoopState())

    assert result.stop is LoopStop.ADMISSION_REFUSED
    assert [r.code for r in result.rejections()] == ["too_deep"]
    assert bodies.ran == []


def test_a_negative_parent_depth_is_refused_at_construction():
    bodies = Bodies()
    reg = registry(bodies)
    with pytest.raises(ValueError):
        GovernedLoop(
            planner=PlannerNode(ScriptedChatModel(responses=[])),
            checker=gate(reg),
            materializer=Materializer(registry=reg, state_schema=LoopState),
            parent_depth=-1,
        )


def test_stop_reasons_are_stable_machine_readable_strings():
    """Invariant 4 is only useful if the reason is matchable; renaming one breaks callers."""
    assert {reason.value for reason in LoopStop} == {
        "goal_met",
        "no_further_work",
        "planned",
        "max_rounds",
        "budget_exhausted",
        "no_progress",
        "admission_refused",
        "planning_failed",
        "execution_failed",
        "human_stopped",
        "approval_denied",
        "approval_timeout",
    }


def test_rejection_feedback_accumulates_across_rounds():
    """Round 3's prompt shows rounds 1 AND 2 — a model shown only the last
    refusal re-proposes the round-before-last's mistake with no way to see
    the pattern, and exhausts the rejection allowance repeating itself."""
    denied = plan(("ship", "deploy"))
    loop, model, _ = build_loop(
        [denied, denied, denied],
        policy=DENY_DEPLOY,
    )
    result = loop.run("goal")
    assert result.stop is LoopStop.ADMISSION_REFUSED

    third_turn = " ".join(str(m.content) for m in model.calls[2])
    assert "Round 1:" in third_turn
    assert "Round 2:" in third_turn


def test_accumulated_feedback_is_bounded_to_the_last_three_rounds():
    denied = plan(("ship", "deploy"))
    loop, model, _ = build_loop(
        [denied] * 5,
        policy=DENY_DEPLOY,
        limits=LoopLimits(max_consecutive_rejections=5, max_rounds=6),
    )
    loop.run("goal")
    fifth_turn = " ".join(str(m.content) for m in model.calls[4])
    assert "Round 1:" not in fifth_turn  # dropped: the prompt must not grow forever
    assert "Round 2:" in fifth_turn and "Round 4:" in fifth_turn


def test_an_unreachable_backend_stops_on_the_first_round():
    """A dead server is not a planning problem: rephrasing cannot reach it.
    The loop used to burn its whole planning allowance re-dialing the same
    socket, then report a planning failure for an infrastructure one."""

    class DeadBackend:
        def __init__(self):
            self.calls = 0

        def propose(self, task, ctx=None, *, feedback=""):
            self.calls += 1
            from grapharc.planner.proposal import PlanningOutcome

            return PlanningOutcome(
                error="planner model call failed: APIConnectionError('Connection error.')",
                unreachable=True,
            )

    planner = DeadBackend()
    bodies = Bodies()
    reg = registry(bodies)
    loop = GovernedLoop(
        planner=planner,
        checker=gate(reg),
        materializer=Materializer(registry=reg, state_schema=LoopState),
    )
    result = loop.run("goal")

    assert result.stop is LoopStop.PLANNING_FAILED
    assert "could not be reached" in result.detail
    assert planner.calls == 1, "the loop retried an unreachable backend"
    assert bodies.ran == []


def test_a_merely_bad_reply_still_gets_its_retries():
    """The unreachable shortcut must not swallow ordinary planning failures."""
    loop, model, _ = build_loop(["not json at all"], on_exhausted="repeat")
    result = loop.run("goal")
    assert result.stop is LoopStop.PLANNING_FAILED
    assert len(result.rounds) == 3  # the full allowance, as before


# -- issue #45: the loop's planner is told the policy before round 1 ----------


def test_the_shipped_loop_discloses_its_edge_policy_to_the_planner():
    """The observed failure, wired exactly as `grapharc plan` wires it.

    A real model burned three rounds proposing an edge into `deploy` — denied
    by the policy every time — because the prompt listed `deploy` in the catalog
    and never said no edge may enter it. The deny rule now travels with the
    catalog. The scripted planner still proposes the deploy (it is a fixed
    script), and round 1 is still refused, which is the point: the disclosure is
    a courtesy and the checker is the gate.
    """
    from grapharc.examples import plan_incident

    model = ScriptedChatModel(responses=plan_incident.scripted_planner_replies())
    result = plan_incident.build_loop(model).run(
        "find the cause", plan_incident.IncidentState(goal="g")
    )

    system = str(model.calls[0][0].content)
    assert "edges into 'deploy' are denied by policy — do not propose them" in system
    assert "deploy: push to production" in system  # the catalog is unchanged

    # Enforcement is where it was: round 1 proposed the denied edge anyway and
    # was refused by the checker, round 2 replanned without it and ran.
    assert [r.code for r in result.rejections()] == ["edge_denied"]
    assert result.stop is LoopStop.GOAL_MET


def test_the_disclosure_is_not_what_refuses_the_edge():
    """Strip the disclosure and the verdict is byte-identical.

    The planner is handed the same policy object the checker holds, so the only
    thing that could differ between these two runs is the prompt. If the
    rejections ever diverge, enforcement has leaked into prompt text.
    """
    denied = [plan(("ship", "deploy"))] * 3
    told, model_told, _ = build_loop(denied, policy=DENY_DEPLOY, disclose=True)
    untold, model_untold, _ = build_loop(denied, policy=DENY_DEPLOY)

    with_disclosure = told.run("goal")
    without = untold.run("goal")

    assert "denied by policy" in str(model_told.calls[0][0].content)
    assert "denied by policy" not in str(model_untold.calls[0][0].content)
    assert [r.model_dump() for r in with_disclosure.rejections()] == [
        r.model_dump() for r in without.rejections()
    ]


def test_the_incident_example_state_merges_parallel_writers():
    """Three kinds writing `notes` in one superstep compose via the reducer.

    The shipped example's state used a plain `list[str]`, so the first plan
    that fanned kinds out of START died on LangGraph's InvalidUpdateError.
    `IncidentState.notes` is a reducer now; this run is the shape that broke.
    """
    from grapharc.examples.plan_incident import IncidentState
    from grapharc.examples.plan_incident import build_loop as build_incident_loop

    fan_out = json.dumps(
        {
            "nodes": [{"name": "triage"}, {"name": "patch"}, {"name": "verify"}],
            "edges": [
                {"source": "__start__", "target": "triage"},
                {"source": "__start__", "target": "patch"},
                {"source": "__start__", "target": "verify"},
                {"source": "triage", "target": "__end__"},
                {"source": "patch", "target": "__end__"},
                {"source": "verify", "target": "__end__"},
            ],
        }
    )
    loop = build_incident_loop(ScriptedChatModel(responses=[fan_out]))
    result = loop.run("triage, patch and verify at once", IncidentState())
    assert result.stop.value == "goal_met"
    assert sorted(result.state.notes) == ["patch ran", "triage ran", "verify ran"]


def test_an_empty_proposal_against_an_unmet_goal_gets_one_nudge():
    """Empty plan, goal check unsatisfied: the contradiction is named once.

    The planner is told the goal check is not met and asked to either propose
    the remaining work or repeat the empty proposal. Here it proposes the
    work, and the run finishes on the goal — where before the nudge existed,
    round 1's empty reply ended the run as `no_further_work` with the goal
    never mentioned to the model.
    """
    loop, model, bodies = build_loop(
        [NOTHING_MORE, plan(("write", "summarise"))], goal_reached=goal_is_done
    )

    result = loop.run("summarise the findings", LoopState())

    assert result.stop is LoopStop.GOAL_MET
    assert bodies.ran == ["write"]
    assert len(result.rounds) == 2
    assert not result.rounds[0].executed and result.rounds[0].admitted
    assert any(
        "goal check is not yet satisfied" in str(message.content)
        for message in model.calls[1]
    )


def test_a_second_empty_proposal_is_believed():
    """The nudge is one round, not a counter: a repeat empty plan is an answer."""
    loop, model, bodies = build_loop(
        [NOTHING_MORE, NOTHING_MORE], goal_reached=goal_is_done
    )

    result = loop.run("summarise the findings", LoopState())

    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert "confirmed no further work" in result.detail
    assert bodies.ran == []
    assert len(result.rounds) == 2
    assert model.call_count == 2


def test_an_unusable_reply_after_the_nudge_confirms_no_further_work():
    """A planner with nothing left to say after the nudge is not a failure.

    The scripted stand-ins end their reply lists with an empty proposal; the
    nudge asks one more question than the script answers. Exhaustion there
    must read as the confirmation it is — never as `planning_failed` burning
    the failure allowance on a planner that already said "nothing more".
    """
    loop, model, bodies = build_loop([NOTHING_MORE], goal_reached=goal_is_done)

    result = loop.run("summarise the findings", LoopState())

    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert "nothing usable" in result.detail
    assert bodies.ran == []
    assert result.rounds[-1].planner_error


def test_an_empty_proposal_with_no_goal_check_stops_without_a_nudge():
    """No goal check means nothing to contradict: one round, one clean stop."""
    loop, model, bodies = build_loop([NOTHING_MORE])

    result = loop.run("nothing needs doing", LoopState())

    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert len(result.rounds) == 1
    assert model.call_count == 1
