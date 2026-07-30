"""What the bot needs from its environment, read once at startup.

Tokens come from process environment variables only. The gateway's `.env`
loader is deliberately not used here: it searches parent directories upward
(the subject of issue #20), and a bot that anyone in a Slack workspace can
drive must not pick up credentials from a file the operator did not point it
at. `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` are exported in the shell that
starts the bot, and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class SlackConfigError(Exception):
    """The environment does not describe a runnable bot."""


@dataclass(frozen=True)
class SlackBotConfig:
    """Everything `bot.py` and the gate need, resolved and validated."""

    bot_token: str
    app_token: str
    # Every path a Slack user names must resolve inside this directory.
    workdir: Path = field(default_factory=Path.cwd)
    # One command's wall clock. Slack acks immediately, so this bounds how
    # long a runaway command can hold one of the bot's worker threads.
    timeout_seconds: float = 120.0
    # Opt-in: allow `--model` / `--reviewer-model`, which reach paid backends.
    allow_model: bool = False
    # Second opt-in: allow `agent`, which executes tools on the host. Only
    # effective together with allow_model — an agent cannot run spend-free.
    allow_agent: bool = False
    slash_command: str = "/grapharc"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> SlackBotConfig:
        env = os.environ if environ is None else environ
        bot_token = env.get("SLACK_BOT_TOKEN", "")
        app_token = env.get("SLACK_APP_TOKEN", "")
        missing = [
            name
            for name, value in (
                ("SLACK_BOT_TOKEN", bot_token),
                ("SLACK_APP_TOKEN", app_token),
            )
            if not value
        ]
        if missing:
            raise SlackConfigError(
                f"{' and '.join(missing)} must be set in the environment that starts the bot"
            )

        workdir = Path(env.get("GRAPHARC_SLACK_WORKDIR", ".")).resolve()
        if not workdir.is_dir():
            raise SlackConfigError(f"GRAPHARC_SLACK_WORKDIR is not a directory: {workdir}")

        raw_timeout = env.get("GRAPHARC_SLACK_TIMEOUT", "120")
        try:
            timeout = float(raw_timeout)
        except ValueError:
            raise SlackConfigError(
                f"GRAPHARC_SLACK_TIMEOUT must be a number of seconds, got {raw_timeout!r}"
            ) from None
        if timeout <= 0:
            raise SlackConfigError("GRAPHARC_SLACK_TIMEOUT must be positive")

        return cls(
            bot_token=bot_token,
            app_token=app_token,
            workdir=workdir,
            timeout_seconds=timeout,
            allow_model=env.get("GRAPHARC_SLACK_ALLOW_MODEL", "") == "1",
            allow_agent=env.get("GRAPHARC_SLACK_ALLOW_AGENT", "") == "1",
            slash_command=env.get("GRAPHARC_SLACK_COMMAND", "/grapharc"),
        )
