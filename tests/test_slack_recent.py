"""Bare Slack reader commands select and disclose a confined recent run."""

from __future__ import annotations

import os

import pytest

from grapharc.observe.trace import TraceRecorder
from grapharc.slack.bot import handle_text, handle_text_live
from grapharc.slack.config import SlackBotConfig


def trace_at(root, directory, run_id):
    trace = TraceRecorder(root / "slack-runs" / directory / "trace.jsonl")
    trace.event(run_id=run_id, graph="g", node="work", phase="start", step=1)
    trace.event(run_id=run_id, graph="g", node="work", phase="end", step=1, tokens=7)
    return trace.path


@pytest.mark.parametrize("command", ["metrics", "viz", "replay", "trace"])
def test_bare_reader_names_the_selected_run(tmp_path, command):
    path = trace_at(tmp_path, "one", "r1")
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text(command, config)
    assert f"{command} for `slack-runs/one/trace.jsonl`" in reply
    assert "run `r1`" in reply
    assert "workdir fallback" in reply
    assert "arguments are required" not in reply
    assert path.read_text()


def test_demo_then_bare_metrics_works_without_copying_generated_identifiers(tmp_path):
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    assert "did its job" in handle_text("demo stage0", config)
    reply = handle_text("metrics", config)
    assert "metrics for `slack-runs/" in reply
    assert "run `" in reply
    assert "nodes_executed" in reply


@pytest.mark.parametrize("command", ["metrics", "viz", "replay", "trace"])
def test_no_prior_run_is_actionable_not_an_argparse_dump(tmp_path, command):
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text(command, config)
    assert "No prior Slack run" in reply
    assert "demo stage0" in reply
    assert "arguments are required" not in reply


def test_explicit_reader_arguments_are_unchanged(tmp_path):
    trace_at(tmp_path, "one", "r1")
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text("metrics slack-runs/one/trace.jsonl r1", config)
    assert "metrics for" not in reply
    assert "nodes_executed" in reply


def test_latest_file_wins_for_a_workdir_fallback(tmp_path):
    old = trace_at(tmp_path, "old", "old-run")
    new = trace_at(tmp_path, "new", "new-run")
    os.utime(old, ns=(1, 1))
    os.utime(new, ns=(2, 2))
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text("metrics", config)
    assert "run `new-run`" in reply


def test_ambiguous_latest_trace_requires_explicit_selection(tmp_path):
    path = trace_at(tmp_path, "two", "r1")
    TraceRecorder(path).event(run_id="r2", graph="g", node="n", phase="end", step=1)
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text("metrics", config)
    assert "contains 2 runs" in reply
    assert "explicit path and run id" in reply


def test_recent_runs_are_confined_before_reading(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    trace_at(outside, "secret", "private")
    (workdir / "slack-runs").symlink_to(outside / "slack-runs", target_is_directory=True)
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=workdir)
    assert "escapes" in handle_text("metrics", config)


def test_channel_history_precedes_workdir_history_and_survives_missing_paths(tmp_path):
    from grapharc.slack.recent import RecentRuns

    history = RecentRuns()
    one = trace_at(tmp_path, "one", "r1")
    two = trace_at(tmp_path, "two", "r2")
    history.remember(tmp_path, "C1", one)
    history.remember(tmp_path, "C2", two)
    for channel, run_id in [("C1", "r1"), ("C2", "r2")]:
        argv, prefix = history.resolve(["metrics"], tmp_path, channel)
        assert argv[-1] == run_id
        assert "this channel" in prefix
    one.unlink()
    argv, prefix = history.resolve(["metrics"], tmp_path, "C1")
    assert argv[-1] == "r2"
    assert "fallback" in prefix


def test_live_handler_remembers_the_channel_after_running(tmp_path):
    from grapharc.slack.recent import RecentRuns

    history = RecentRuns()
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    handle_text_live("demo stage0", config, None, history=history, channel="C1")
    reply = handle_text_live("metrics", config, None, history=history, channel="C1")
    assert "this channel" in reply
    assert "nodes_executed" in reply


def test_corrupt_latest_trace_is_reported_instead_of_answering_about_an_older_run(tmp_path):
    old = trace_at(tmp_path, "old", "r1")
    new = trace_at(tmp_path, "new", "r2")
    os.utime(old, ns=(1, 1))
    new.write_bytes(b"\xff\n")
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=tmp_path)
    reply = handle_text("metrics", config)
    assert "cannot read the latest Slack trace" in reply
    assert "slack-runs/new/trace.jsonl" in reply


def test_escaping_candidate_is_skipped_before_it_is_read(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    trace_at(work, "one", "r1")
    outside = tmp_path / "private.jsonl"
    outside.write_bytes(b"\xff\n")
    link = work / "slack-runs" / "evil"
    link.mkdir()
    (link / "trace.jsonl").symlink_to(outside)
    config = SlackBotConfig(bot_token="unused", app_token="unused", workdir=work)
    assert "run `r1`" in handle_text("metrics", config)
