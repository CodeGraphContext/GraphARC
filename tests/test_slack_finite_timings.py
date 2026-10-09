"""Non-finite time settings must not reach Slack's timers or command runner."""

from __future__ import annotations

import pytest

from grapharc.slack.command import SlackCommandError, parse_command
from grapharc.slack.config import SlackBotConfig, SlackConfigError


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e309"])
@pytest.mark.parametrize(
    "key", ["GRAPHARC_SLACK_TIMEOUT", "GRAPHARC_SLACK_WORK_TIMEOUT", "GRAPHARC_SLACK_LIVE_INTERVAL"]
)
def test_environment_timing_values_must_be_finite(tmp_path, key, value):
    env = {
        "SLACK_BOT_TOKEN": "test-bot",
        "SLACK_APP_TOKEN": "test-app",
        "GRAPHARC_SLACK_WORKDIR": str(tmp_path),
        key: value,
    }

    with pytest.raises(SlackConfigError, match=key):
        SlackBotConfig.from_env(env)


@pytest.mark.parametrize("form", ["--approval-timeout nan", "--approval-timeout=NaN"])
def test_a_nan_approval_wait_is_refused(tmp_path, form):
    with pytest.raises(SlackCommandError, match="does not fit this command's budget"):
        parse_command(
            f"plan goal --scripted --go {form}",
            workdir=tmp_path,
            timeout_seconds=60,
            work_timeout_seconds=180,
        )


def test_finite_fractional_timings_remain_configurable(tmp_path):
    config = SlackBotConfig.from_env(
        {
            "SLACK_BOT_TOKEN": "test-bot",
            "SLACK_APP_TOKEN": "test-app",
            "GRAPHARC_SLACK_WORKDIR": str(tmp_path),
            "GRAPHARC_SLACK_TIMEOUT": "12.5",
            "GRAPHARC_SLACK_WORK_TIMEOUT": "25.5",
            "GRAPHARC_SLACK_LIVE_INTERVAL": "0.25",
        }
    )

    assert config.timeout_seconds == 12.5
    assert config.work_timeout_seconds == 25.5
    assert config.live_interval_seconds == 0.25
    argv = parse_command(
        "plan goal --scripted --approval-timeout 1.5",
        workdir=tmp_path,
        timeout_seconds=config.timeout_seconds,
        work_timeout_seconds=config.work_timeout_seconds,
    )
    assert argv[argv.index("--approval-timeout") + 1] == "1.5"
