"""slack-bolt wiring: the only module that touches Slack itself.

Everything with behaviour lives in `command`/`runner`/`format`; what remains
here is `handle_text` (their composition, still import-safe without slack-bolt
and tested that way) and the listener glue. Slack requires an ack within three
seconds, so each listener acks with "running…" first and posts the result when
the command finishes — bolt runs listeners on worker threads, so a slow
command blocks neither the socket nor other requests.

`slack_bolt` is imported inside `build_app`, not at module top: the wheel-check
imports every module in an environment without the extra, and a user who never
runs the bot should never need it installed.
"""

from __future__ import annotations

import re
from typing import Any

from grapharc.slack.command import SlackCommandError, parse_command, usage_text
from grapharc.slack.config import SlackBotConfig
from grapharc.slack.format import format_result
from grapharc.slack.runner import run_command

# An app_mention's text arrives as "<@U0BOTID> metrics t.jsonl r1".
_MENTION = re.compile(r"<@[A-Z0-9]+>\s*")


def handle_text(text: str, config: SlackBotConfig) -> str:
    """Gate, run, format: the whole request path, with Slack stripped away."""
    stripped = _MENTION.sub("", text).strip()
    try:
        argv = parse_command(
            stripped, workdir=config.workdir, allow_model=config.allow_model
        )
    except SlackCommandError as exc:
        return str(exc)
    result = run_command(
        argv, workdir=config.workdir, timeout_seconds=config.timeout_seconds
    )
    return format_result(result)


def build_app(config: SlackBotConfig) -> Any:
    """A configured `slack_bolt.App`; raises with the install hint if the extra is absent."""
    try:
        from slack_bolt import App
    except ImportError:
        raise SlackCommandError(
            "the Slack bot needs the `slack` extra: uv sync --extra slack "
            "(or: pip install 'grapharc[slack]')"
        ) from None

    app = App(token=config.bot_token)

    @app.command(config.slash_command)
    def _slash(ack: Any, respond: Any, command: dict[str, Any]) -> None:
        text = command.get("text", "").strip()
        if not text:
            ack(usage_text(allow_model=config.allow_model))
            return
        ack(f"running `grapharc {text}`…")
        respond(handle_text(text, config))

    @app.event("app_mention")
    def _mention(event: dict[str, Any], say: Any) -> None:
        say(
            handle_text(event.get("text", ""), config),
            thread_ts=event.get("thread_ts") or event.get("ts"),
        )

    return app


def serve(config: SlackBotConfig) -> None:
    """Open the Socket Mode connection and block until interrupted."""
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    SocketModeHandler(build_app(config), config.app_token).start()
