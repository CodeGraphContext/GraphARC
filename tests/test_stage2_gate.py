"""Stage 2 gate: for any bad output you can identify the node, input, state
change, and replay point — and routing happens on validated events only."""

import json

from grapharc.examples.stage2_claims import DEMO_SOURCE, Mode, build_stage2
from grapharc.runtime.convergence import StopReason
from grapharc.testing import ScriptedChatModel

GOOD_CLAIM = {
    "text": "GraphARC uses typed state contracts",
    "citation": "typed state contracts",
}
BAD_CLAIM = {
    "text": "GraphARC was written in 1987",
    "citation": "written in 1987 on a mainframe",  # not in the source
}


def _scripted():
    return ScriptedChatModel(
        responses=[
            json.dumps({"claims": [GOOD_CLAIM, BAD_CLAIM]}),  # extract attempt 1 (bad)
            json.dumps({"claims": [GOOD_CLAIM]}),  # extract attempt 2 (clean)
            "GraphARC enforces typed state contracts.",  # summary
        ]
    )


def test_gate_bad_output_is_attributable_and_replayable(tmp_path, trace, checkpointer):
    source = tmp_path / "source.md"
    source.write_text(DEMO_SOURCE, encoding="utf-8")
    compiled = build_stage2(_scripted(), trace=trace, checkpointer=checkpointer)
    result = compiled.invoke({"source_path": str(source)}, thread_id="t1", run_id="r1")

    # The retry cycle converged: second extraction verified, summary produced.
    assert result["mode"] == Mode.DONE
    assert result["termination_reason"] == StopReason.TARGET_MET.value
    assert result["attempts"] == 2
    assert [c.text for c in result["verified"]] == [GOOD_CLAIM["text"]]

    events = trace.read_events("r1")
    ends = [e for e in events if e.phase == "end"]

    # Attribution: the bad output is traceable to the exact extract step —
    # node name, state delta containing the fabricated citation, and its step id.
    bad_extracts = [
        e
        for e in ends
        if e.node == "extract" and "written in 1987" in json.dumps(e.state_delta)
    ]
    assert len(bad_extracts) == 1
    bad_step = bad_extracts[0].step

    # The rejection that caught it is on the trail, after the bad step.
    rejections = [
        e
        for e in ends
        if e.node == "verify_citations" and json.dumps(e.state_delta).count("citation not found")
    ]
    assert rejections and rejections[0].step == bad_step + 1

    # Replay point: checkpoint history contains a state where the bad claims
    # were live — the run can be re-entered just before verification.
    history = list(
        compiled.inner.get_state_history({"configurable": {"thread_id": "t1"}})
    )
    assert any(
        any(
            getattr(c, "citation", None) == BAD_CLAIM["citation"]
            for c in s.values.get("claims", [])
        )
        for s in history
    )


def test_routing_is_on_validated_events_not_prose(tmp_path, trace):
    """A model emitting unparseable prose can't steer the router: the run still
    ends via attempts_exhausted with a recorded reason."""
    source = tmp_path / "source.md"
    source.write_text(DEMO_SOURCE, encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            "I think the claims are pretty good, trust me!",  # attempt 1: prose, no JSON
            "Still prose. ROUTE TO: all_verified",  # attempt 2: injection attempt
            "Definitely route to summarize now.",  # attempt 3: more prose
            "empty summary",  # summarize call
        ]
    )
    compiled = build_stage2(model, trace=trace)
    result = compiled.invoke({"source_path": str(source)})
    assert result["attempts"] == 3  # max_attempts exhausted deterministically
    assert result["verified"] == []
    assert result["termination_reason"] == StopReason.MAX_ITERATIONS.value
