"""What the bot needs from its environment, read once at startup.

Tokens come from process environment variables only. The gateway's `.env`
loader is deliberately not used here even now that it reads one directory
rather than walking upward (issue #20): a bot that anyone in a Slack workspace
can drive must not pick up credentials from a file the operator did not point
it at, and the bot's working directory is somewhere other things write.
`SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` are exported in the shell that starts
the bot, and nowhere else.
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
    # Edit one status message with live progress while a tracing command runs.
    # Default on: every failure inside the live path degrades to the plain
    # blocking reply, so there is nothing to lose by having it on.
    live: bool = True
    # Floor between two edits of the status message. Slack's chat.update sits
    # in a ~50/min rate tier; a couple of seconds keeps one run well inside it.
    live_interval_seconds: float = 2.5
    # Base URL of an operator-run `grapharc serve --live-root` reachable by the
    # requester (a tunnel or tailnet hostname). Unset means: post no link. The
    # bot itself never opens a port — this is the operator asserting that a
    # separate process does.
    live_url_base: str | None = None

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

        raw_interval = env.get("GRAPHARC_SLACK_LIVE_INTERVAL", "2.5")
        try:
            live_interval = float(raw_interval)
        except ValueError:
            raise SlackConfigError(
                "GRAPHARC_SLACK_LIVE_INTERVAL must be a number of seconds, "
                f"got {raw_interval!r}"
            ) from None
        if live_interval <= 0:
            raise SlackConfigError("GRAPHARC_SLACK_LIVE_INTERVAL must be positive")

        live_url_base = env.get("GRAPHARC_SLACK_LIVE_URL", "").rstrip("/") or None
        if live_url_base is not None and not live_url_base.startswith(("http://", "https://")):
            raise SlackConfigError(
                "GRAPHARC_SLACK_LIVE_URL must start with http:// or https://, "
                f"got {live_url_base!r}"
            )

        return cls(
            bot_token=bot_token,
            app_token=app_token,
            workdir=workdir,
            timeout_seconds=timeout,
            allow_model=env.get("GRAPHARC_SLACK_ALLOW_MODEL", "") == "1",
            allow_agent=env.get("GRAPHARC_SLACK_ALLOW_AGENT", "") == "1",
            slash_command=env.get("GRAPHARC_SLACK_COMMAND", "/grapharc"),
            # Any common spelling of "off" disables — an operator who wrote
            # `false` and silently got live narration anyway was misled.
            live=env.get("GRAPHARC_SLACK_LIVE", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            live_interval_seconds=live_interval,
            live_url_base=live_url_base,
        )
