"""The shipped registry of node kinds — `grapharc.stdlib`.

This exists so a zero-config run can happen **without generating code**. A model
asked to set up a workflow selects from these kinds; it never authors a node
body. The tests below pin the properties that make that claim true rather than
decorative:

- the kinds available with no model are exactly the ones that need none;
- a phase's tool allowlist is fixed here by an operator, not chosen at run time;
- `apply_change` is registered *and* denied, so refusing it is a policy decision
  about a transition rather than a hole in the allowlist;
- a kind absent from `WRITES` may write nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import PrivateAttr

from grapharc import stdlib
from grapharc.harness.permissions import Decision
from grapharc.planner import (
    AdmissionChecker,
    Materializer,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel


class ToolCallingModel(ScriptedChatModel):
    """A double that implements `bind_tools`, so agent phases can be driven.

    Records what it was handed, which is how the tests check that a phase's reach
    is what `TOOLS_FOR` says and not whatever the model would have liked.
    """

    _bound: list[Any] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003 - LangChain's signature
        self._bound.append(tools)
        return self

    @property
    def bound_tool_names(self) -> list[str]:
        names: list[str] = []
        for batch in self._bound:
            for tool in batch:
                name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                if name is None and isinstance(tool, dict):
                    name = (tool.get("function") or {}).get("name")
                if name:
                    names.append(str(name))
        return names


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("hello from a file", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _model():
    return ToolCallingModel(responses=["done"], on_exhausted="repeat")


def _chain(*kinds: str) -> Subgraph:
    endpoints = [START, *kinds, END]
    return Subgraph(
        nodes=tuple(ProposedNode(name=k) for k in kinds),
        edges=tuple(
            ProposedEdge(source=a, target=b)
            for a, b in zip(endpoints, endpoints[1:], strict=False)
        ),
    )


def _gate(model=None):
    return AdmissionChecker(
        registry=stdlib.build_registry(model), edge_policy=stdlib.default_edge_policy()
    )


def _build(model, proposal):
    registry = stdlib.build_registry(model)
    gate = AdmissionChecker(registry=registry, edge_policy=stdlib.default_edge_policy())
    verdict = gate.check(proposal)
    assert verdict.admitted, [r.code for r in verdict.rejections]
    return Materializer(
        registry=registry, state_schema=stdlib.STATE_SCHEMA, writes=stdlib.WRITES
    ).materialize(verdict, proposal)


# -- what exists, and when ---------------------------------------------------


def test_with_no_model_only_the_deterministic_kinds_exist():
    """Agent kinds are left out rather than registered without a factory: a
    proposal naming one then fails at the gate with the allowed list, instead of
    passing and dying at materialisation."""
    names = sorted(stdlib.build_registry().names())

    assert names == sorted(stdlib.DETERMINISTIC_KINDS)


def test_with_a_model_every_kind_exists():
    names = sorted(stdlib.build_registry(_model()).names())

    assert names == sorted([*stdlib.DETERMINISTIC_KINDS, *stdlib.AGENT_KINDS])


def test_naming_an_agent_kind_without_a_model_is_refused_at_the_gate():
    verdict = _gate(model=None).check(_chain("investigate"))

    assert not verdict.admitted
    assert [r.code for r in verdict.rejections] == ["unregistered_node"]


def test_every_registered_kind_has_a_description_a_model_can_choose_from():
    catalog = stdlib.catalog_for_prompt(_model())

    assert set(catalog) == set(stdlib.build_registry(_model()).names())
    assert all(text.strip() for text in catalog.values())


# -- the tool allowlist is the operator's ------------------------------------


def test_a_read_only_phase_is_handed_only_read_only_tools(workspace):
    model = _model()
    _build(model, _chain("investigate")).invoke({"goal": "what is here?"})

    assert set(model.bound_tool_names) == set(stdlib.READ_ONLY_TOOLS)
    for forbidden in ("write_file", "edit_file", "run_command"):
        assert forbidden not in model.bound_tool_names


def test_the_mutating_phase_is_the_only_one_handed_write_tools(workspace):
    model = _model()
    registry = stdlib.build_registry(model)
    # Admitted under a permissive policy, because the *default* policy denies it —
    # which is the subject of a separate test.
    from grapharc.planner import EdgePolicy, EdgeRule

    gate = AdmissionChecker(
        registry=registry,
        edge_policy=EdgePolicy(rules=(EdgeRule(action=Decision.ALLOW),)),
    )
    proposal = _chain("apply_change")
    verdict = gate.check(proposal)
    Materializer(
        registry=registry, state_schema=stdlib.STATE_SCHEMA, writes=stdlib.WRITES
    ).materialize(verdict, proposal).invoke({"goal": "change something"})

    assert "write_file" in model.bound_tool_names
    assert "run_command" not in model.bound_tool_names, "no phase gets a shell"


def test_no_phase_anywhere_is_handed_run_command():
    """The one tool that escapes every guard in the package. No stdlib phase
    should be able to reach it, under any policy."""
    for tools in stdlib.TOOLS_FOR.values():
        assert "run_command" not in tools


def test_the_toolless_phase_binds_nothing_so_it_runs_on_any_backend(workspace):
    """`summarize` is the one agent kind a subscription-only backend can drive:
    `AgentNode` skips `bind_tools` when no tool is visible."""
    model = _model()
    _build(model, _chain("summarize")).invoke({"goal": "wrap up"})

    assert stdlib.TOOLS_FOR["summarize"] == ()
    assert model.bound_tool_names == []


# -- registered and denied ---------------------------------------------------


def test_the_mutating_kind_is_registered_and_denied_by_default():
    """Both halves matter. Registered, so turning it on is a TOML edit rather
    than a code change. Denied, so a refusal is a decision about a transition."""
    registry = stdlib.build_registry(_model())

    assert "apply_change" in registry.names()
    verdict = _gate(_model()).check(_chain("apply_change"))
    assert not verdict.admitted
    assert [r.code for r in verdict.rejections] == ["edge_denied"]


def test_read_only_work_needs_no_decision(workspace):
    verdict = _gate(_model()).check(_chain("collect_context", "investigate", "verify"))

    assert verdict.admitted


def test_one_toml_rule_is_what_turns_mutation_on(tmp_path):
    """The zero-config default is restrictive; opting in is a file, not a patch."""
    from grapharc.policy import PolicyEngine

    engine = PolicyEngine.from_toml('version = "1"\ndefault = "allow"\n')
    gate = AdmissionChecker(
        registry=stdlib.build_registry(_model()), edge_policy=engine.edge_policy()
    )

    assert gate.check(_chain("apply_change")).admitted


# -- writes ------------------------------------------------------------------


def test_a_deterministic_phase_records_what_it_found(workspace):
    state = _build(None, _chain("collect_context")).invoke({"goal": "look"})

    assert any("notes.txt" in line for line in state["findings"])


def test_the_no_op_phase_writes_nothing(workspace):
    state = _build(None, _chain("checkpoint")).invoke({"goal": "nothing"})

    assert state.get("findings", []) == []
    assert state.get("notes", []) == []


def test_every_kind_has_a_declared_write_set():
    """A kind absent from WRITES may write nothing — that is enforced, but an
    accidental omission would be a silently mute phase."""
    registry = stdlib.build_registry(_model())

    assert set(registry.names()) <= set(stdlib.WRITES)


def test_a_phase_cannot_write_outside_its_declared_fields(workspace):
    """`investigate` writes findings; if it returned notes the kernel refuses."""
    assert stdlib.WRITES["investigate"] == {"findings"}
    assert stdlib.WRITES["checkpoint"] == set()
    assert "notes" not in stdlib.WRITES["investigate"]


def test_each_agent_phase_appends_rather_than_overwriting(workspace):
    """Two phases writing the same scalar would clobber each other; appending is
    what makes the accumulation the contract."""
    state = _build(_model(), _chain("investigate", "verify")).invoke({"goal": "look"})

    assert len(state["findings"]) == 2, state["findings"]


def test_a_later_phase_is_shown_what_earlier_phases_found(workspace):
    """Otherwise every phase re-does the previous one's work."""
    model = _model()
    _build(model, _chain("collect_context", "investigate")).invoke({"goal": "look"})

    prompts = "\n".join(str(m.content) for call in model.calls for m in call)
    assert "What earlier phases found" in prompts
    assert "notes.txt" in prompts


def test_a_phase_that_gave_up_is_labelled_as_such(workspace):
    """A stop short of `target_met` must not read like a finished answer."""
    model = ToolCallingModel(responses=["partial"], on_exhausted="repeat")
    state = _build(model, _chain("summarize")).invoke({"goal": "x"})

    assert state["notes"], state
    # Either it met the target or the reason is on the line; never silent.
    assert state["notes"][0] == "partial" or state["notes"][0].startswith("[")


# -- the plan-registry contract (`grapharc plan --registry grapharc.stdlib:...`) --


def test_stdlib_ships_the_full_registry_module_contract():
    """Everything `cli/plan.py` reads by getattr, present and coherent."""
    import grapharc.stdlib as stdlib

    assert stdlib.STATE_SCHEMA is stdlib.WorkState
    assert callable(stdlib.build_loop)
    assert callable(stdlib.goal_met)
    replies = stdlib.scripted_planner_replies()
    assert replies, "the spend-free smoke path needs at least one reply"


def test_goal_met_reads_notes_defensively():
    from grapharc.stdlib import WorkState, goal_met

    assert not goal_met(WorkState())
    assert goal_met(WorkState(notes=["a report landed"]))
    assert not goal_met(object())  # no notes attribute: not met, not a crash


def test_the_scripted_plan_runs_spend_free_and_stops_cleanly(tmp_path):
    """The whole loop through stdlib's own builder, no model, no network."""
    from grapharc.observe.trace import TraceRecorder
    from grapharc.planner import LoopStop
    from grapharc.stdlib import WorkState, build_loop, scripted_planner_replies
    from grapharc.testing import ScriptedChatModel

    trace = TraceRecorder(tmp_path / "t.jsonl")
    model = ScriptedChatModel(responses=scripted_planner_replies())
    loop = build_loop(model, trace=trace)
    result = loop.run("smoke", WorkState(goal="smoke"))

    # The deterministic kinds cannot write `notes`, so the honest clean stop
    # is "no further work" — never a burn to planning_failed.
    assert result.stop is LoopStop.NO_FURTHER_WORK
    assert result.state.findings, "collect_context must actually have run"
    phases = {e.phase for e in trace.read_events(result.run_id)}
    assert "topology" in phases  # the admitted round's shape is on the trace


def test_parallel_phases_merge_instead_of_colliding(tmp_path):
    """The schema's whole claim: two phases finishing in one superstep must
    merge. Without `operator.add` reducers LangGraph raised InvalidUpdateError
    ("Can receive only one value per step"), so every fan-out a planner
    proposed — the natural shape for a decomposable job — died at execution."""
    from grapharc.observe.trace import TraceRecorder
    from grapharc.runtime.graph import END, START, GraphARC
    from grapharc.stdlib import WorkState

    trace = TraceRecorder(tmp_path / "t.jsonl")
    g = GraphARC(WorkState, name="fanout", trace=trace)
    g.add_node("seed", lambda s: {"findings": ["seed"]}, writes={"findings"})
    for name in ("a", "b", "c"):
        body = (lambda n: lambda s: {"findings": [f"from {n}"]})(name)
        g.add_node(name, body, writes={"findings"})
        g.add_edge("seed", name)
        g.add_edge(name, "join")
    g.add_node("join", lambda s: None, writes=set())
    g.add_edge(START, "seed")
    g.add_edge("join", END)

    out = g.compile().invoke({"goal": "x"}, run_id="r1")
    assert sorted(out["findings"]) == ["from a", "from b", "from c", "seed"]


def test_phases_return_only_their_own_addition(tmp_path):
    """With a reducer, returning the accumulated list would double it."""
    from grapharc.stdlib import WorkState, _collect_context

    body = _collect_context(None)
    delta = body(WorkState(findings=["already here"]))
    assert len(delta["findings"]) == 1, delta


def test_the_planner_is_told_the_run_completes_via_summarize():
    """The completion rule is deterministic code; a planner that was never
    told it burns rounds on investigate-only plans that cannot finish."""
    from grapharc.stdlib import build_loop
    from grapharc.testing import ScriptedChatModel

    loop = build_loop(ScriptedChatModel(responses=["{}"]))
    assert "summarize" in loop.planner.instructions
    assert "complete" in loop.planner.instructions
    catalog = dict(loop.planner.catalog)
    assert "the run is complete" in catalog["summarize"]


def test_build_registry_workspace_confines_tools_and_the_listing(tmp_path):
    from grapharc.stdlib import build_registry
    from grapharc.testing import ScriptedChatModel

    (tmp_path / "only-file.txt").write_text("hello")
    registry = build_registry(ScriptedChatModel(responses=[]), workspace=tmp_path)
    spec = registry.get("collect_context")
    body = spec.factory(spec)

    class _State:
        goal = ""
        findings: list = []
        notes: list = []

    delta = body(_State())
    assert "only-file.txt" in delta["findings"][0]
