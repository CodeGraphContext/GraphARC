"""The shared per-node status rule every renderer reads.

`metrics` (Mermaid overlay), `slack.live` (narration marks) and `viewmodel`
(the structured graph snapshot) all fold trace events into one status per
node. The rule lives once, here-under-test, so the three surfaces cannot
drift apart again.
"""

from __future__ import annotations

from grapharc.observe.status import node_states
from grapharc.observe.trace import TraceEvent


def _event(node: str, phase: str, *, graph: str = "g", step: int = 1, **kw) -> TraceEvent:
    return TraceEvent(
        ts="2026-08-05T00:00:00+00:00",
        run_id="r1",
        graph=graph,
        node=node,
        phase=phase,
        step=step,
        **kw,
    )


def test_precedence_is_errored_then_running_then_done_then_pending():
    events = [
        # `a` failed once and later succeeded: the failure still wins.
        _event("a", "start"),
        _event("a", "error", error="boom"),
        _event("a", "start"),
        _event("a", "end"),
        # `b` is mid-flight.
        _event("b", "start"),
        # `c` finished cleanly.
        _event("c", "start"),
        _event("c", "end"),
    ]
    states = node_states(events)
    assert states["a"].status == "errored"
    assert states["b"].status == "running"
    assert states["c"].status == "done"
    # A declared node with no events has no entry — the caller reads pending.
    assert "d" not in states


def test_fanout_instances_collapse_to_one_running_name():
    # Three parallel instances of one worker; one is still open.
    events = [
        _event("worker", "start", step=3),
        _event("worker", "start", step=4),
        _event("worker", "start", step=5),
        _event("worker", "end", step=3),
        _event("worker", "end", step=4),
    ]
    state = node_states(events)["worker"]
    assert (state.starts, state.ends) == (3, 2)
    assert state.status == "running"


def test_last_terminal_events_are_kept_whole():
    events = [
        _event("a", "start"),
        _event("a", "end", duration_ms=12.5, tokens=40),
        _event("a", "start"),
        _event("a", "end", duration_ms=99.0, tokens=7),
        _event("b", "error", error="first"),
        _event("b", "error", error="second"),
    ]
    states = node_states(events)
    assert states["a"].last_end is not None
    assert states["a"].last_end.duration_ms == 99.0
    assert states["b"].last_error is not None
    assert states["b"].last_error.error == "second"


def test_sub_step_phases_never_touch_a_nodes_lifecycle():
    # model/tool/stop describe work inside a node; the planner's paperwork
    # describes the run. Neither is a lifecycle event.
    events = [
        _event("agent", "model", tokens=100),
        _event("agent", "tool"),
        _event("agent", "stop"),
        _event("planner", "plan"),
        _event("planner", "admission"),
    ]
    assert node_states(events) == {}


def test_a_budget_refused_node_is_errored_without_ever_starting():
    # The kernel emits only `error` for a node the budget check refused.
    states = node_states([_event("expensive", "error", error="over budget")])
    assert states["expensive"].status == "errored"
    assert states["expensive"].starts == 0


def test_scoping_is_the_callers_job_one_graph_at_a_time():
    # The same node name in two round graphs: fold each graph's events apart
    # and the statuses stay each round's own.
    round1 = [_event("work", "start", graph="g1"), _event("work", "end", graph="g1")]
    round2 = [_event("work", "start", graph="g2")]
    assert node_states(round1)["work"].status == "done"
    assert node_states(round2)["work"].status == "running"
