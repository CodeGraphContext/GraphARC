"""The live tail loop and its Slack-free composition in `handle_text_live`.

Everything is driven through plain callables — a recording sink instead of a
WebClient — so none of this needs a token or a network. The end-to-end tests
run the real CLI subprocess through `run_command`, same as the gateway tests.
"""

from __future__ import annotations

import time

from grapharc.observe.replay import replay
from grapharc.observe.trace import TraceRecorder
from grapharc.slack.bot import handle_text_live
from grapharc.slack.config import SlackBotConfig
from grapharc.slack.live import FLAT_FEED_LINES, LiveSettings, LiveTail, render_progress

FAST = LiveSettings(update_interval=0.01, poll_interval=0.01, join_timeout=2.0)


def _write_run(recorder: TraceRecorder, run_id: str, *, nodes: int = 2) -> None:
    for step in range(1, nodes + 1):
        recorder.event(
            run_id=run_id, graph="g", node=f"n{step}", phase="start", step=step
        )
        recorder.event(
            run_id=run_id,
            graph="g",
            node=f"n{step}",
            phase="end",
            step=step,
            duration_ms=12.0,
            tokens=10,
        )


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# LiveTail: offset discipline, throttling, failure posture.
# ---------------------------------------------------------------------------


def test_only_events_after_the_start_offset_render(tmp_path):
    """A reused trace file must not replay an earlier run as this one's progress."""
    path = tmp_path / "trace.jsonl"
    _write_run(TraceRecorder(path), "old-run")

    seen: list[str] = []
    with LiveTail(path, ["agent", "task"], lambda t: seen.append(t) or True, FAST):
        _write_run(TraceRecorder(path), "new-run")
        assert _wait_for(lambda: seen)
    assert all("old-run" not in t for t in seen), "the old run leaked into live progress"


def test_no_update_without_new_bytes(tmp_path):
    path = tmp_path / "trace.jsonl"
    TraceRecorder(path)  # creates the parent; file itself absent

    seen: list[str] = []
    with LiveTail(path, ["run", "g.toml"], lambda t: seen.append(t) or True, FAST):
        time.sleep(0.2)
    assert seen == []


def test_identical_renders_are_not_reposted(tmp_path):
    path = tmp_path / "trace.jsonl"
    seen: list[str] = []
    with LiveTail(path, ["run", "g.toml"], lambda t: seen.append(t) or True, FAST):
        _write_run(TraceRecorder(path), "r1")
        assert _wait_for(lambda: seen)
        count = len(seen)
        time.sleep(0.2)  # no new bytes, no new update
    assert len(seen) == count


def test_a_dead_sink_silences_the_tail_without_raising(tmp_path):
    path = tmp_path / "trace.jsonl"
    calls: list[str] = []

    def dead(text: str) -> bool:
        calls.append(text)
        return False

    with LiveTail(path, ["run", "g.toml"], dead, FAST):
        recorder = TraceRecorder(path)
        _write_run(recorder, "r1")
        assert _wait_for(lambda: calls)
        _write_run(recorder, "r1")
        time.sleep(0.2)
    assert len(calls) == 1


def test_an_update_that_raises_never_escapes_the_thread(tmp_path):
    path = tmp_path / "trace.jsonl"

    def explode(text: str) -> bool:
        raise RuntimeError("sink blew up")

    with LiveTail(path, ["run", "g.toml"], explode, FAST):
        _write_run(TraceRecorder(path), "r1")
        time.sleep(0.2)
    # reaching here without an exception is the assertion


def test_a_torn_final_line_is_left_for_the_next_tick(tmp_path):
    path = tmp_path / "trace.jsonl"
    seen: list[str] = []
    with LiveTail(path, ["run", "g.toml"], lambda t: seen.append(t) or True, FAST):
        _write_run(TraceRecorder(path), "r1")
        with path.open("a", encoding="utf-8") as f:
            f.write('{"ts": "2026-01-01T00:00:00.000+00:00", "run_id": "r1"')  # no newline
        assert _wait_for(lambda: seen)
    assert seen  # progress rendered from the complete lines


# ---------------------------------------------------------------------------
# render_progress
# ---------------------------------------------------------------------------


def _run_from(tmp_path, events_writer, run_id="r1"):
    path = tmp_path / "render.jsonl"
    recorder = TraceRecorder(path)
    events_writer(recorder, run_id)
    return replay(recorder, run_id)


def test_progress_shows_marks_durations_and_the_diagram_link(tmp_path):
    def write(recorder, run_id):
        _write_run(recorder, run_id)
        recorder.event(run_id=run_id, graph="g", node="n3", phase="start", step=3)
        recorder.event(
            run_id=run_id, graph="g", node="n4", phase="start", step=4
        )
        recorder.event(
            run_id=run_id,
            graph="g",
            node="n4",
            phase="error",
            step=4,
            error="citation not found",
        )

    run = _run_from(tmp_path, write)
    text = render_progress(
        run, argv=["run", "g.toml"], elapsed_s=14.2, diagram="flowchart TD\n  a --> b"
    )
    assert "✓ n1" in text and "12ms" in text and "10 tok" in text
    assert "▸ n3" in text and "running…" in text
    assert "✗ n4" in text and "citation not found" in text
    assert "running (14s)" in text
    assert "2/4 nodes done" not in text  # 3 completed: n1, n2, n4(error)
    assert "3/4 nodes done" in text
    assert "mermaid.live/view#pako:" in text
    assert "\x1b" not in text


def test_an_agent_shaped_run_renders_the_flat_feed(tmp_path):
    def write(recorder, run_id):
        for step in range(1, FLAT_FEED_LINES + 6):
            recorder.event(
                run_id=run_id,
                graph="agent",
                node="agent:model",
                phase="model",
                step=step,
                tokens=7,
            )

    run = _run_from(tmp_path, write)
    assert not run.executions, "precondition: an all-orphan run"
    text = render_progress(run, argv=["agent", "task"], elapsed_s=3.0)
    assert "model  agent:model" in text
    assert "… 5 earlier events" in text
    assert "nodes done" not in text


def test_a_huge_progress_body_announces_truncation(tmp_path):
    def write(recorder, run_id):
        for step in range(1, 200):
            recorder.event(
                run_id=run_id,
                graph="g",
                node=f"node-with-a-rather-long-name-{step:04d}",
                phase="start",
                step=step,
            )
            recorder.event(
                run_id=run_id,
                graph="g",
                node=f"node-with-a-rather-long-name-{step:04d}",
                phase="end",
                step=step,
                duration_ms=5.0,
            )

    run = _run_from(tmp_path, write)
    text = render_progress(run, argv=["run", "g.toml"], elapsed_s=60.0)
    assert "earlier characters not shown" in text


# ---------------------------------------------------------------------------
# handle_text_live end to end, with the real CLI and a recording sink.
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self, *, post_ok: bool = True, updates_ok: bool = True):
        self.post_ok = post_ok
        self.updates_ok = updates_ok
        self.posted: list[str] = []
        self.updated: list[str] = []

    def post(self, text: str):
        if not self.post_ok:
            return None
        self.posted.append(text)
        return ("C1", "171.1")

    def update(self, handle, text: str) -> bool:
        if not self.updates_ok:
            return False
        assert handle == ("C1", "171.1")
        self.updated.append(text)
        return True


def _config(tmp_path, **overrides) -> SlackBotConfig:
    values = dict(
        bot_token="xoxb-x",
        app_token="xapp-x",
        workdir=tmp_path,
        timeout_seconds=60.0,
        live_interval_seconds=0.05,
    )
    values.update(overrides)
    return SlackBotConfig(**values)


def test_a_tracing_command_posts_once_and_finishes_in_the_status_message(tmp_path):
    sink = RecordingSink()
    reply = handle_text_live("demo stage0", _config(tmp_path), sink)
    assert reply == ""
    assert len(sink.posted) == 1
    assert sink.posted[0].startswith("`grapharc demo")
    assert "did its job" in sink.updated[-1]


def test_a_reader_takes_the_plain_blocking_path(tmp_path):
    sink = RecordingSink()
    reply = handle_text_live("models", _config(tmp_path), sink)
    assert sink.posted == [] and sink.updated == []
    assert "did its job" in reply


def test_a_failed_post_falls_back_to_the_blocking_reply(tmp_path):
    sink = RecordingSink(post_ok=False)
    reply = handle_text_live("demo stage0", _config(tmp_path), sink)
    assert "did its job" in reply
    assert sink.updated == []


def test_a_failed_final_update_returns_the_result_for_the_caller_to_post(tmp_path):
    sink = RecordingSink(updates_ok=False)
    reply = handle_text_live("demo stage0", _config(tmp_path), sink)
    assert "did its job" in reply, "the final result must never be lost"


def test_live_off_takes_the_plain_path_even_for_tracing_commands(tmp_path):
    sink = RecordingSink()
    reply = handle_text_live("demo stage0", _config(tmp_path, live=False), sink)
    assert sink.posted == []
    assert "did its job" in reply


def test_the_status_message_advertises_the_live_url_when_configured(tmp_path):
    sink = RecordingSink()
    config = _config(tmp_path, live_url_base="https://laptop.example")
    handle_text_live("demo stage0", config, sink)
    assert "watch live: https://laptop.example/live/view?trace=" in sink.posted[0]
    assert "(if the live server is up)" in sink.posted[0]


def test_the_final_message_keeps_the_run_page_link(tmp_path):
    """Finishing a run must not be what makes its links disappear."""
    sink = RecordingSink()
    config = _config(tmp_path, live_url_base="https://laptop.example")
    reply = handle_text_live("demo stage0", config, sink)
    assert reply == ""
    final = sink.updated[-1]
    assert "did its job" in final
    assert "run page: https://laptop.example/live/view?trace=" in final
    # The operator's own page is the link; mermaid.live is only the fallback.
    assert "mermaid.live" not in final


def test_the_blocking_path_also_gets_the_final_links(tmp_path):
    reply = handle_text_live("demo stage0", _config(tmp_path, live=False), RecordingSink())
    assert "did its job" in reply
    assert "mermaid.live/view#pako:" in reply


def test_the_progress_message_links_the_live_view_when_configured(tmp_path):
    def write(recorder, run_id):
        recorder.event(run_id=run_id, graph="g", node="n1", phase="start", step=1)

    run = _run_from(tmp_path, write)
    with_view = render_progress(
        run,
        argv=["run", "g.toml"],
        elapsed_s=1.0,
        diagram="flowchart TD\n  a --> b",
        view_url="https://laptop.example/live/view?trace=t.jsonl",
    )
    assert "<https://laptop.example/live/view?trace=t.jsonl|open live view>" in with_view
    assert "mermaid.live" not in with_view
    # Without a configured view, the mermaid.live fallback still stands.
    without = render_progress(
        run, argv=["run", "g.toml"], elapsed_s=1.0, diagram="flowchart TD\n  a --> b"
    )
    assert "mermaid.live/view#pako:" in without


def test_a_refusal_is_still_a_returned_message(tmp_path):
    sink = RecordingSink()
    reply = handle_text_live("<@U012345> serve", _config(tmp_path), sink)
    assert "not a command this bot runs" in reply
    assert sink.posted == []


# ---------------------------------------------------------------------------
# Bug-hunt regressions
# ---------------------------------------------------------------------------


def test_planned_lines_keep_each_rounds_own_status(tmp_path):
    """Keyed by bare name, round 1's finished node once wore round 2's
    still-running state (and vice versa)."""
    def write(recorder, run_id):
        for graph, done in (("plan:aaa", True), ("plan:bbb", False)):
            recorder.event(
                run_id=run_id, graph=graph, node="topology", phase="topology",
                step=0,
                state_delta={"nodes": ["search"], "edges": [],
                             "round": 1 if graph == "plan:aaa" else 2},
            )
            recorder.event(run_id=run_id, graph=graph, node="search",
                           phase="start", step=1)
            if done:
                recorder.event(run_id=run_id, graph=graph, node="search",
                               phase="end", step=1, duration_ms=100.0)

    run = _run_from(tmp_path, write)
    text = render_progress(run, argv=["plan", "g"], elapsed_s=5.0)
    round1 = text.split("round 2:")[0]
    round2 = text.split("round 2:")[1]
    assert "✓ search" in round1, text
    assert "▸ search" in round2, text


def test_a_replaced_trace_file_does_not_silence_the_tailer(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_run(TraceRecorder(path), "big-old-run", nodes=6)  # a large file

    seen: list[str] = []
    with LiveTail(path, ["run", "g.toml"], lambda t: seen.append(t) or True, FAST):
        path.unlink()
        _write_run(TraceRecorder(path), "fresh-run", nodes=1)  # smaller file
        assert _wait_for(lambda: any("fresh-run" not in t or True for t in seen) and seen)
    assert seen, "the tailer stayed silent after the file was replaced"


def test_the_final_diagram_is_this_runs_not_a_previous_ones(tmp_path):
    from grapharc.slack.bot import _with_final_links
    from grapharc.slack.config import SlackBotConfig

    path = tmp_path / "reused.jsonl"
    _write_run(TraceRecorder(path), "previous-run")
    config = SlackBotConfig(bot_token="x", app_token="x", workdir=tmp_path)

    # This invocation wrote nothing (failed before its first event): no
    # diagram at all, never the previous run's presented as the outcome.
    text = _with_final_links(
        "result", ["run", "g.toml"], path, config,
        prior_runs=frozenset({"previous-run"}),
    )
    assert "mermaid.live" not in text
