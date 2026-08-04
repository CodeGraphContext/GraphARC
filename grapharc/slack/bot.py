"""slack-bolt wiring: the only module that touches Slack itself.

Everything with behaviour lives in `command`/`runner`/`format`/`live`; what
remains here is `handle_text_live` (their composition, still import-safe
without slack-bolt and tested that way with a fake sink) and the listener
glue. Slack requires an ack within three seconds, so each listener acks first
— bolt runs listeners on worker threads, so a slow command blocks neither the
socket nor other requests.

A tracing command gets a live status message: one `chat.postMessage` when it
starts, edited in place (`chat.update`) as trace events arrive, and edited one
last time into the final result. Note the visibility change this brings to the
slash path: `respond()` output was ephemeral by default, a posted status
message is visible to the channel — deliberate, since a live run is channel
activity, and the ephemeral behaviour survives wherever the bot cannot post
(not in the channel, API error): every failure in the live path falls back to
today's blocking reply through `respond()`/`say()`, and the final result is
never lost to a failed edit.

`slack_bolt` is imported inside `build_app`, not at module top: the wheel-check
imports every module in an environment without the extra, and a user who never
runs the bot should never need it installed.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from grapharc.observe.metrics import to_mermaid
from grapharc.observe.trace import TailRecorder
from grapharc.slack.command import SlackCommandError, parse_command, trace_path, usage_text
from grapharc.slack.config import SlackBotConfig
from grapharc.slack.format import format_result, live_view_url, mermaid_live_url
from grapharc.slack.live import LiveSettings, LiveSink, LiveTail
from grapharc.slack.runner import run_command

# An app_mention's text arrives as "<@U0BOTID> metrics t.jsonl r1".
_MENTION = re.compile(r"<@[A-Z0-9]+>\s*")


def handle_text(text: str, config: SlackBotConfig) -> str:
    """Gate, run, format: the whole request path, with Slack stripped away."""
    return handle_text_live(text, config, sink=None)


def handle_text_live(text: str, config: SlackBotConfig, sink: LiveSink | None) -> str:
    """Like `handle_text`, but narrates a tracing command through `sink`.

    Returns the message the caller must still post — `""` when the sink
    already delivered the final result by editing the status message. Any
    failure to post or edit degrades to the plain blocking path; the final
    result always reaches the requester through one route or the other.
    """
    stripped = _MENTION.sub("", text).strip()
    try:
        argv = parse_command(
            stripped,
            workdir=config.workdir,
            allow_model=config.allow_model,
            allow_agent=config.allow_agent,
            timeout_seconds=config.timeout_seconds,
        )
    except SlackCommandError as exc:
        return str(exc)

    tpath = trace_path(argv, config.workdir)
    # Run ids already in the file, noted before the run: on a reused trace,
    # the final diagram must be *this* run's, and if this run wrote nothing
    # (failed before its first event) there must be no diagram at all — not a
    # previous run's presented as the outcome.
    prior_runs: frozenset[str] = frozenset()
    if tpath is not None and tpath.exists():
        try:
            prior_runs = frozenset(TailRecorder(tpath).run_ids())
        except Exception:
            pass
    handle: object | None = None
    if config.live and sink is not None and tpath is not None:
        try:
            handle = sink.post(_starting_text(argv, config))
        except Exception:
            handle = None

    if handle is None:
        result = run_command(
            argv, workdir=config.workdir, timeout_seconds=config.timeout_seconds
        )
        return _with_final_links(format_result(result), argv, tpath, config, prior_runs)

    def _update(message: str) -> bool:
        try:
            return sink.update(handle, message)
        except Exception:
            return False

    settings = LiveSettings(update_interval=config.live_interval_seconds)
    view_url = live_view_url(argv, base=config.live_url_base, workdir=config.workdir)
    with LiveTail(tpath, argv, _update, settings, view_url=view_url):
        result = run_command(
            argv, workdir=config.workdir, timeout_seconds=config.timeout_seconds
        )
    final = _with_final_links(format_result(result), argv, tpath, config, prior_runs)
    if _update(final):
        return ""
    return final


def _with_final_links(
    final: str,
    argv: list[str],
    tpath,
    config: SlackBotConfig,
    prior_runs: frozenset[str] = frozenset(),
) -> str:
    """Keep the run-page (or diagram) link on the *final* message.

    The final result edits over the live status message, which is where the
    "watch live" link lived — without this, finishing a run is what makes its
    links disappear. The operator's own run page is the primary link; the
    mermaid.live fragment link is the fallback for a bot with no live server
    configured. Both are best-effort: a link that cannot be computed is
    simply absent.
    """
    if tpath is None:
        return final
    lines = [final]
    url = live_view_url(argv, base=config.live_url_base, workdir=config.workdir)
    if url:
        lines.append(f"run page: {url}")
    else:
        try:
            recorder = TailRecorder(tpath)
            new_runs = [r for r in recorder.run_ids() if r not in prior_runs]
            if new_runs:
                diagram = to_mermaid(recorder, new_runs[-1])
                lines.append(f"<{mermaid_live_url(diagram)}|final diagram>")
        except Exception:
            pass
    return "\n".join(lines)


def _starting_text(argv: list[str], config: SlackBotConfig) -> str:
    lines = [f"`{shlex.join(['grapharc', *argv])}` — starting…"]
    url = live_view_url(argv, base=config.live_url_base, workdir=config.workdir)
    if url:
        lines.append(f"watch live: {url} (if the live server is up)")
    return "\n".join(lines)


class _ChannelSink:
    """A `LiveSink` over a slack-sdk WebClient; failures are values, not raises."""

    def __init__(self, client: Any, channel: str, thread_ts: str | None = None) -> None:
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts

    def post(self, text: str) -> object | None:
        if not self._channel:
            return None
        try:
            kwargs: dict[str, Any] = {"channel": self._channel, "text": text}
            if self._thread_ts:
                kwargs["thread_ts"] = self._thread_ts
            response = self._client.chat_postMessage(**kwargs)
            return (response["channel"], response["ts"])
        except Exception:
            return None

    def update(self, handle: object, text: str) -> bool:
        try:
            channel, ts = handle  # type: ignore[misc]
            self._client.chat_update(channel=channel, ts=ts, text=text)
            return True
        except Exception:
            return False


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
    def _slash(ack: Any, respond: Any, command: dict[str, Any], client: Any) -> None:
        text = command.get("text", "").strip()
        if not text:
            ack(usage_text(allow_model=config.allow_model, allow_agent=config.allow_agent))
            return
        ack(f"running `grapharc {text}`…")
        sink = _ChannelSink(client, command.get("channel_id", ""))
        reply = handle_text_live(text, config, sink)
        if reply:
            respond(reply)

    @app.event("app_mention")
    def _mention(event: dict[str, Any], say: Any, client: Any) -> None:
        thread_ts = event.get("thread_ts") or event.get("ts")
        sink = _ChannelSink(client, event.get("channel", ""), thread_ts=thread_ts)
        reply = handle_text_live(event.get("text", ""), config, sink)
        if reply:
            say(reply, thread_ts=thread_ts)

    return app


def serve(config: SlackBotConfig) -> None:
    """Open the Socket Mode connection and block until interrupted."""
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    SocketModeHandler(build_app(config), config.app_token).start()
