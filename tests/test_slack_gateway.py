"""The Slack gate, runner and formatter — everything except Slack itself.

The layering under test: `command.py` decides what Slack text may become an
argv, `runner.py` runs it against this interpreter's grapharc, `format.py`
turns the result into one message. `bot.py` is glue; its one behaviour worth a
test here (the missing-extra error) is tested by faking the import failure,
so none of this file needs a Slack token or a network.
"""

from __future__ import annotations

import sys

import pytest

from grapharc.slack.command import SlackCommandError, parse_command, usage_text
from grapharc.slack.config import SlackBotConfig, SlackConfigError
from grapharc.slack.format import MAX_FENCE_CHARS, format_result
from grapharc.slack.runner import CommandResult, run_command

# ---------------------------------------------------------------------------
# The gate: what text is allowed to become an argv.
# ---------------------------------------------------------------------------


def test_a_reading_command_passes_through_verbatim(tmp_path):
    argv = parse_command("metrics t.jsonl r1", workdir=tmp_path)
    assert argv == ["metrics", "t.jsonl", "r1"]


def test_a_leading_grapharc_token_is_tolerated(tmp_path):
    assert parse_command("grapharc models", workdir=tmp_path) == ["models"]


def test_agent_and_serve_are_refused(tmp_path):
    for name in ("agent", "serve"):
        with pytest.raises(SlackCommandError, match="not a command this bot runs"):
            parse_command(f"{name} whatever", workdir=tmp_path)


def test_registry_config_and_json_are_refused(tmp_path):
    for flag in ("--registry mod:attr", "--config g.toml", "--json"):
        with pytest.raises(SlackCommandError, match="not allowed"):
            parse_command(f"run graph.toml {flag}", workdir=tmp_path)


def test_model_is_refused_by_default_and_admitted_on_opt_in(tmp_path):
    with pytest.raises(SlackCommandError, match="paid backend"):
        parse_command("plan 'a goal' --model mock/x", workdir=tmp_path)
    argv = parse_command("plan 'a goal' --model mock/x", workdir=tmp_path, allow_model=True)
    assert argv == ["plan", "a goal", "--model", "mock/x"]


def test_a_path_positional_may_not_escape_the_workdir(tmp_path):
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("trace ../outside.jsonl", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("trace /etc/passwd", workdir=tmp_path)


def test_a_path_flag_value_may_not_escape_the_workdir_either_form(tmp_path):
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("plan goal --trace ../t.jsonl", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("plan goal --trace=../t.jsonl", workdir=tmp_path)


def test_a_path_inside_the_workdir_is_admitted_even_absolute(tmp_path):
    inside = tmp_path / "runs" / "t.jsonl"
    argv = parse_command(f"trace {inside}", workdir=tmp_path)
    assert argv == ["trace", str(inside)]


def test_a_quoted_goal_survives_as_one_argument(tmp_path):
    argv = parse_command('plan "investigate the checkout outage"', workdir=tmp_path)
    assert argv == ["plan", "investigate the checkout outage"]


def test_empty_text_answers_with_usage_not_a_traceback(tmp_path):
    with pytest.raises(SlackCommandError) as excinfo:
        parse_command("", workdir=tmp_path)
    assert "Allowed here" in str(excinfo.value)
    assert "agent" in usage_text()


def test_a_value_flag_missing_its_value_is_refused(tmp_path):
    with pytest.raises(SlackCommandError, match="needs a value"):
        parse_command("trace t.jsonl --run-id", workdir=tmp_path)


# ---------------------------------------------------------------------------
# Runner and formatter, end to end against the real CLI.
# ---------------------------------------------------------------------------


def test_models_runs_and_formats_as_success(tmp_path):
    result = run_command(["models"], workdir=tmp_path, timeout_seconds=60)
    assert result.exit_code == 0
    message = format_result(result)
    assert "did its job" in message
    assert "```" in message
    assert "\x1b" not in message, "an escape reached a Slack message"


def test_a_missing_graph_formats_as_could_not_run(tmp_path):
    result = run_command(
        ["run", str(tmp_path / "missing.toml"), "--trace", str(tmp_path / "t.jsonl")],
        workdir=tmp_path,
        timeout_seconds=60,
    )
    assert result.exit_code == 2
    message = format_result(result)
    assert "could not run" in message
    assert "stderr:" in message


def test_the_timeout_kills_the_process_and_says_so(tmp_path):
    # Interpreter startup alone exceeds this, so the timeout always fires.
    result = run_command(["models"], workdir=tmp_path, timeout_seconds=0.05)
    assert result.exit_code is None
    assert "was stopped" in format_result(result)


def test_truncation_is_announced_never_silent():
    result = CommandResult(
        argv=["trace", "t.jsonl"],
        exit_code=0,
        stdout="x" * (MAX_FENCE_CHARS + 500),
        stderr="",
        duration_seconds=0.1,
        timeout_seconds=60,
    )
    message = format_result(result)
    assert "500 more characters not shown" in message


def test_a_fence_in_the_output_cannot_break_out():
    result = CommandResult(
        argv=["trace", "t.jsonl"],
        exit_code=0,
        stdout="before\n```\nafter",
        stderr="",
        duration_seconds=0.1,
        timeout_seconds=60,
    )
    body = format_result(result).split("```", 1)[1]
    assert "\n```\n" not in body.rsplit("```", 1)[0]


# ---------------------------------------------------------------------------
# Config and the bot's import posture.
# ---------------------------------------------------------------------------


def test_missing_tokens_name_every_missing_variable():
    with pytest.raises(SlackConfigError, match="SLACK_BOT_TOKEN and SLACK_APP_TOKEN"):
        SlackBotConfig.from_env({})


def test_a_non_numeric_timeout_is_a_named_error_not_a_traceback(tmp_path):
    env = {
        "SLACK_BOT_TOKEN": "xoxb-x",
        "SLACK_APP_TOKEN": "xapp-x",
        "GRAPHARC_SLACK_TIMEOUT": "forever",
    }
    with pytest.raises(SlackConfigError, match="GRAPHARC_SLACK_TIMEOUT"):
        SlackBotConfig.from_env(env)


def test_config_reads_workdir_timeout_and_model_opt_in(tmp_path):
    config = SlackBotConfig.from_env(
        {
            "SLACK_BOT_TOKEN": "xoxb-x",
            "SLACK_APP_TOKEN": "xapp-x",
            "GRAPHARC_SLACK_WORKDIR": str(tmp_path),
            "GRAPHARC_SLACK_TIMEOUT": "5",
            "GRAPHARC_SLACK_ALLOW_MODEL": "1",
        }
    )
    assert config.workdir == tmp_path
    assert config.timeout_seconds == 5.0
    assert config.allow_model


def test_handle_text_turns_a_refusal_into_a_message_not_an_exception(tmp_path):
    from grapharc.slack.bot import handle_text

    config = SlackBotConfig(bot_token="xoxb-x", app_token="xapp-x", workdir=tmp_path)
    reply = handle_text("<@U012345> agent rm -rf /", config)
    assert "not a command this bot runs" in reply


def test_a_missing_slack_extra_is_an_install_hint_not_an_import_error(monkeypatch, tmp_path):
    from grapharc.slack import bot

    monkeypatch.setitem(sys.modules, "slack_bolt", None)
    config = SlackBotConfig(bot_token="xoxb-x", app_token="xapp-x", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="slack"):
        bot.build_app(config)
