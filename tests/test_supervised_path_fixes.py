"""Regressions for the five bugs on the supervised-Claude-Code path.

Each of these was reachable from one Slack message — propose a graph, approve
it, let Claude Code do the work — and each failed quietly rather than loudly:
a run billed at zero, a phase's output dropped, a live page that closed while
the graph was still running, a generated policy that permitted the one kind
worth denying, and a delegated agent spawned only to be killed.
"""

from __future__ import annotations

import pytest

from grapharc.observe.cost import attribute
from grapharc.observe.replay import replay
from grapharc.observe.trace import TraceRecorder

# -- a delegated run's bill ---------------------------------------------------


def test_a_delegated_runs_cost_is_billed_not_dropped(tmp_path):
    """`--executor claude-cli` reports its spend on a `stop` event, not a `model` one.

    `grapharc agent` drives an `AgentNode` with no enclosing graph, so every
    event it writes is an orphan. Attributing orphan cost only from `model`
    events billed the whole run at $0.00 — and left `unpriced_tokens` at zero
    too, so nothing said the figure was incomplete.
    """
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(
        run_id="r1", graph="agent", node="fix", phase="model", step=1,
        state_delta={"executor": "delegated"},
    )
    recorder.event(
        run_id="r1", graph="agent", node="fix", phase="stop", step=1,
        tokens=4200, cost_usd=0.0731,
        state_delta={"executor": "delegated", "turns": 6},
    )
    run = replay(recorder, "r1")
    cost = attribute(run)
    assert cost.recorded_cost_usd == pytest.approx(0.0731)
    assert cost.tokens == 4200
    # And the two readers of one trace now agree.
    assert cost.recorded_cost_usd == pytest.approx(run.recorded_cost_usd)


def test_a_stop_event_is_not_counted_as_a_model_call(tmp_path):
    """Billing it must not turn it into a row in the model-call breakdown."""
    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(
        run_id="r1", graph="agent", node="fix", phase="stop", step=1,
        tokens=10, cost_usd=0.5,
    )
    cost = attribute(replay(recorder, "r1"))
    assert cost.model_calls == []
    assert cost.recorded_cost_usd == pytest.approx(0.5)


# -- a curtailed phase's output ----------------------------------------------


def test_a_curtailed_phase_reports_what_it_managed_to_say(tmp_path):
    """`AgentResult.output` is empty by contract for every reason but TARGET_MET.

    Reading it for a curtailed phase formatted "[budget_exhausted] " with
    nothing after it: the work the phase did manage was dropped, and a
    downstream goal check saw an empty note rather than a truncated one.
    """
    from grapharc.harness.agent import AgentResult, StopReason
    from grapharc.stdlib import _agent_factory

    class _Node:
        def run(self, _prompt, _ctx):
            return AgentResult(
                output="",
                partial_output="I read tests/test_x.py and found the seed is unset",
                termination_reason=StopReason.BUDGET_EXHAUSTED,
                note="max_tokens reached (900/900)",
            )

    factory = _agent_factory(object(), lambda tools: object(), "investigate")
    import grapharc.harness as harness_module

    original = harness_module.AgentNode
    harness_module.AgentNode = lambda *a, **k: _Node()  # type: ignore[assignment]
    try:
        body = factory(None)
        from grapharc.stdlib import WorkState

        written = body(WorkState(goal="g"), None)
    finally:
        harness_module.AgentNode = original  # type: ignore[assignment]

    line = written["findings"][0]
    assert "budget_exhausted" in line
    assert "the seed is unset" in line, "the phase's own text was dropped"


def test_a_curtailed_phase_with_no_prose_falls_back_to_its_reason(tmp_path):
    from grapharc.harness.agent import AgentResult, StopReason
    from grapharc.stdlib import WorkState, _agent_factory

    class _Node:
        def run(self, _prompt, _ctx):
            return AgentResult(
                termination_reason=StopReason.ERROR, note="tool raised RuntimeError"
            )

    import grapharc.harness as harness_module

    original = harness_module.AgentNode
    harness_module.AgentNode = lambda *a, **k: _Node()  # type: ignore[assignment]
    try:
        written = _agent_factory(object(), lambda tools: object(), "verify")(None)(
            WorkState(goal="g"), None
        )
    finally:
        harness_module.AgentNode = original  # type: ignore[assignment]
    assert "RuntimeError" in written["findings"][0]


def test_a_run_whose_only_note_is_an_error_has_not_met_its_goal(tmp_path):
    """Issue #96, the half that made the other half invisible.

    `goal_met` counted notes, so a run whose entire output was `['[error] ']`
    stopped `goal_met` — reporting success for a run that produced nothing.
    An agent node's failure is folded into state as prose, and a length check
    cannot tell a report from an apology.
    """
    from grapharc.stdlib import WorkState, curtailed_note, goal_met

    failed = WorkState(goal="g", notes=[curtailed_note("error", "the tool exploded")])
    assert not goal_met(failed)

    # The empty-message form the issue actually observed.
    assert not goal_met(WorkState(goal="g", notes=[curtailed_note("error", "")]))


def test_a_real_report_still_meets_the_goal(tmp_path):
    from grapharc.stdlib import WorkState, goal_met

    assert goal_met(WorkState(goal="g", notes=["here is what happened"]))
    # One good note among failures is still an outcome.
    assert goal_met(WorkState(goal="g", notes=["[error] nope", "the real summary"]))


def test_curtailed_is_read_off_the_stop_reasons_not_guessed(tmp_path):
    """Prose that merely starts with a bracket is not a stop reason."""
    from grapharc.stdlib import WorkState, goal_met, is_curtailed

    assert is_curtailed("[budget_exhausted] ran out")
    assert not is_curtailed("[NOTE] the config is fine")
    assert goal_met(WorkState(goal="g", notes=["[NOTE] the config is fine"]))


# -- the live page during a multi-phase run ----------------------------------


def _phase(recorder, node, step, *, sub_stop=True):
    """One graph node that internally drives an agent loop."""
    recorder.event(run_id="r1", graph="work", node=node, phase="start", step=step)
    if sub_stop:
        # The AgentNode's own terminal event, inside the node's span.
        recorder.event(
            run_id="r1", graph="work", node=node, phase="stop", step=step,
            state_delta={"executor": "delegated"},
        )
    recorder.event(
        run_id="r1", graph="work", node=node, phase="end", step=step, duration_ms=5.0
    )


def test_the_live_page_stays_open_while_later_phases_still_have_to_run(tmp_path):
    """Every agent phase writes a `stop`; only the driver's ends the run.

    Keying on any `stop` declared a three-phase graph finished the moment the
    first phase ended, and closed the SSE stream on a page with two nodes left.
    """
    from grapharc.server.live import build_snapshot

    recorder = TraceRecorder(tmp_path / "t.jsonl")
    _phase(recorder, "investigate", 1)
    recorder.event(run_id="r1", graph="work", node="apply_change", phase="start", step=2)

    snapshot = build_snapshot(tmp_path, "t.jsonl", "r1")
    assert not snapshot.done, "the run was declared finished with a node still open"


def test_a_run_with_no_graph_still_finishes_on_its_own_stop(tmp_path):
    """`grapharc agent` has no node spans, so its stop is an orphan — and terminal."""
    from grapharc.server.live import build_snapshot

    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(run_id="r1", graph="agent", node="fix", phase="model", step=1)
    recorder.event(run_id="r1", graph="agent", node="fix", phase="stop", step=1)

    snapshot = build_snapshot(tmp_path, "t.jsonl", "r1")
    assert snapshot.done


def test_a_termination_reason_still_ends_the_run(tmp_path):
    from grapharc.server.live import build_snapshot

    recorder = TraceRecorder(tmp_path / "t.jsonl")
    recorder.event(run_id="r1", graph="work", node="n", phase="start", step=1)
    recorder.event(
        run_id="r1", graph="work", node="n", phase="end", step=1,
        state_delta={"termination_reason": "goal_met"},
    )
    snapshot = build_snapshot(tmp_path, "t.jsonl", "r1")
    assert snapshot.done


# -- the generated policy ----------------------------------------------------


class _CapturingModel:
    """Records the prompt it was asked, answers with a permissive document."""

    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return "[[rules]]\naction = 'allow'\ntarget = '*'\n"


def test_an_undeclared_registry_has_every_kind_treated_as_dangerous(tmp_path):
    """`None` means nobody said, which is not evidence of safety.

    Substituting stdlib's `MUTATING_KINDS` named `apply_change` — a kind this
    registry does not have — while its real mutating kind went unnamed, and the
    generated policy was then cached and governed every later run.
    """
    from grapharc.cli.generate import resolve_or_generate_policy

    model = _CapturingModel()
    resolve_or_generate_policy(
        None,
        tenant="default",
        model=model,
        goal="ship it",
        catalog={"survey": "read things", "publish": "push to prod"},
        mutating=None,
        workdir=tmp_path,
        write=False,
    )
    dangerous = model.prompt.split("treat as dangerous):")[1].splitlines()[0]
    assert "publish" in dangerous and "survey" in dangerous
    assert "apply_change" not in dangerous, "a kind from some other registry was named"


def test_a_registry_that_declares_nothing_mutates_is_believed(tmp_path):
    """`()` is a declaration, not an absence — it must not be overwritten."""
    from grapharc.cli.generate import resolve_or_generate_policy

    model = _CapturingModel()
    resolve_or_generate_policy(
        None,
        tenant="default",
        model=model,
        goal="read the docs",
        catalog={"survey": "read things", "summarise": "write a summary"},
        mutating=(),
        workdir=tmp_path,
        write=False,
    )
    dangerous = model.prompt.split("treat as dangerous):")[1].splitlines()[0]
    assert "(none)" in dangerous


def test_a_declared_mutating_tuple_is_passed_through(tmp_path):
    from grapharc.cli.generate import resolve_or_generate_policy

    model = _CapturingModel()
    resolve_or_generate_policy(
        None,
        tenant="default",
        model=model,
        goal="fix it",
        catalog={"survey": "read", "deploy": "ship"},
        mutating=("deploy",),
        workdir=tmp_path,
        write=False,
    )
    dangerous = model.prompt.split("treat as dangerous):")[1].splitlines()[0]
    assert "deploy" in dangerous and "survey" not in dangerous


# -- delegating with no time left --------------------------------------------


# -- the workspace a delegated run is given ----------------------------------


def test_a_harness_keeps_the_workspace_it_was_given(tmp_path):
    """`workspace=` used to be discarded whenever an explicit executor was passed.

    It existed only to construct the default `SandboxedExecutor`, so
    `Harness(..., executor=LocalExecutor(), workspace=ws)` — which several call
    sites in this repo write — silently dropped `ws`.
    """
    from grapharc.harness import Harness, LocalExecutor, PermissionPolicy, ToolRegistry

    harness = Harness(
        ToolRegistry(),
        PermissionPolicy(rules=[]),
        executor=LocalExecutor(),
        workspace=str(tmp_path),
    )
    assert harness.workspace == str(tmp_path)


def test_a_sandboxed_harness_still_reports_its_executors_workspace(tmp_path):
    from grapharc.harness import Harness, PermissionPolicy, ToolRegistry

    harness = Harness(ToolRegistry(), PermissionPolicy(rules=[]), workspace=str(tmp_path))
    assert harness.workspace == str(tmp_path)
    assert harness.executor.workspace == str(tmp_path)


def test_the_stdlib_harness_names_a_workspace_so_claude_code_can_be_delegated_to(tmp_path):
    """The stdlib registry driven by the Claude CLI *is* a delegated run.

    Every agent kind in this registry delegates its whole loop to Claude Code
    when the backend has no tool-calling wire format — that is what the module
    docstring describes. Its harness used `LocalExecutor`, which has no
    workspace, so `_run_delegated` refused before spawning anything and every
    agent phase of every such run failed with "does not expose one".
    """
    from grapharc.stdlib import default_harness

    harness = default_harness(("read_file",), workspace=tmp_path)
    assert harness.workspace == str(tmp_path)


def test_an_exhausted_deadline_does_not_spawn_claude_code(tmp_path):
    """`remaining_seconds()` goes negative once the budget is spent.

    `subprocess.run` accepts a negative timeout: it starts the child and kills
    it on the first wait. So an over-budget run launched Claude Code, tore it
    down mid-startup, and reported a max_seconds nobody had set.
    """
    from grapharc.cli.delegate import DelegationError, delegate_task

    for remaining in (0.0, -3.2):
        with pytest.raises(DelegationError) as caught:
            delegate_task("do a thing", workspace=tmp_path, max_seconds=remaining)
        assert caught.value.reason == "deadline_exceeded"
        assert "no time left" in str(caught.value)
