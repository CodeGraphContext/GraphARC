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

import json
import re
import shlex
from typing import Any

from grapharc.observe.metrics import to_mermaid
from grapharc.observe.trace import TailRecorder
from grapharc.planner.approval_file import read_request, write_decision
from grapharc.slack.command import (
    SlackCommandError,
    confine_path,
    effective_timeout,
    mutating_kinds_for,
    parse_command,
    trace_path,
    usage_text,
)
from grapharc.slack.config import SlackBotConfig
from grapharc.slack.format import format_result, live_view_url, mermaid_live_url
from grapharc.slack.live import ApprovalPrompt, LiveSettings, LiveSink, LiveTail
from grapharc.slack.runner import run_command

#: `action_id`s for the two approval buttons. Stable strings, because a bolt
#: listener is registered against them and a message posted by an older bot
#: keeps whatever it was drawn with.
APPROVE_ACTION = "grapharc_approve"
DENY_ACTION = "grapharc_deny"

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
            work_timeout_seconds=config.work_timeout_seconds,
        )
    except SlackCommandError as exc:
        return str(exc)

    # Recomputed from the admitted argv, so the runner's kill and the ceilings
    # the gate injected into that same argv are derived from one number.
    timeout = effective_timeout(
        argv,
        timeout_seconds=config.timeout_seconds,
        work_timeout_seconds=config.work_timeout_seconds,
    )
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
        result = run_command(argv, workdir=config.workdir, timeout_seconds=timeout)
        return _with_final_links(format_result(result), argv, tpath, config, prior_runs)

    def _update(message: str, prompt: ApprovalPrompt | None = None) -> bool:
        try:
            return sink.update(handle, message, approval_blocks(message, prompt))
        except Exception:
            return False

    settings = LiveSettings(update_interval=config.live_interval_seconds)
    view_url = live_view_url(argv, base=config.live_url_base, workdir=config.workdir)
    with LiveTail(
        tpath,
        argv,
        _update,
        settings,
        view_url=view_url,
        workdir=config.workdir,
        mutating_kinds=mutating_kinds_for(argv, config.workdir),
    ):
        result = run_command(argv, workdir=config.workdir, timeout_seconds=timeout)
    final = _with_final_links(format_result(result), argv, tpath, config, prior_runs)
    if _update(final):
        return ""
    return final


def approval_blocks(text: str, prompt: ApprovalPrompt | None) -> list[dict[str, Any]] | None:
    """Block Kit for a parked run: the message, then Approve / Deny.

    None when nothing is pending — which is also how the buttons *leave*. The
    status message is edited in place all run long, so the same edit that
    reports "approved, running" has to drop the blocks; a finished run still
    showing a live Approve button is a lie a later click would act on.
    """
    if prompt is None:
        return None
    value = json.dumps({"dir": prompt.directory, "fp": prompt.fingerprint})
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": APPROVE_ACTION,
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "value": value,
                },
                {
                    "type": "button",
                    "action_id": DENY_ACTION,
                    "style": "danger",
                    "text": {"type": "plain_text", "text": "Deny"},
                    "value": value,
                },
            ],
        },
    ]


def handle_approval_action(
    raw_value: str, config: SlackBotConfig, *, deny: bool, actor: str = ""
) -> str:
    """Answer a parked run from a button click. Returns what to say about it.

    Slack signs the payloads bolt hands us, so the value is not attacker-typed
    the way a slash command is. It is still re-checked here, because "signed by
    Slack" says the click is genuine, not that the button was drawn by a
    version of this bot that meant the same thing by it:

    - the directory is re-confined inside the working directory, exactly as the
      command gate confines a typed path;
    - the fingerprint must match the request currently on disk. This is the
      load-bearing check. A parked run rewrites its request every round, so a
      button scrolled back to from round 1 names a proposal that is no longer
      the one waiting — and approving *that* would be approving a graph nobody
      read. Mismatch is refused here and, belt and braces, would be discarded
      by `file_approval` too.
    """
    try:
        payload = json.loads(raw_value)
        directory = str(payload["dir"])
        fingerprint = str(payload["fp"])
    except (ValueError, KeyError, TypeError):
        return "that button carried nothing I can act on."

    try:
        confine_path(directory, config.workdir)
    except SlackCommandError as exc:
        return str(exc)

    resolved = (config.workdir / directory).resolve()
    request = read_request(resolved)
    if request is None:
        # The common, boring case: the run already moved on — approved by
        # someone else, denied, or timed out while the message sat there.
        return "nothing is waiting for approval there any more."
    if str(request.get("fingerprint", "")) != fingerprint:
        return (
            "that button was drawn for an earlier plan; the run has since "
            "proposed a different one. Scroll down to the current message."
        )

    decision = "denied" if deny else "approved"
    write_decision(resolved, fingerprint=fingerprint, decision=decision)
    who = f" by <@{actor}>" if actor else ""
    return f"{decision}{who} — plan `{fingerprint}`"


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

    def update(
        self, handle: object, text: str, blocks: list[dict[str, Any]] | None = None
    ) -> bool:
        try:
            channel, ts = handle  # type: ignore[misc]
            # `blocks=[]` rather than omitting the field: Slack keeps a
            # message's existing blocks when `chat.update` does not mention
            # them, so leaving it out would leave a finished run's Approve
            # button on screen for someone to click.
            self._client.chat_update(
                channel=channel, ts=ts, text=text, blocks=blocks if blocks else []
            )
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

    def _decide(ack: Any, body: dict[str, Any], client: Any, *, deny: bool) -> None:
        """One Approve/Deny click. Ack first — Slack wants it within 3 seconds."""
        ack()
        actions = body.get("actions") or [{}]
        answer = handle_approval_action(
            str(actions[0].get("value", "")),
            config,
            deny=deny,
            actor=str((body.get("user") or {}).get("id", "")),
        )
        # Posted as a new message in the run's thread rather than edited over
        # the status message: the tailer owns that message and would overwrite
        # this within the second. A reply is also the honest shape — who
        # decided, and when, is a fact about the conversation, and the trace
        # (which has no actor field) cannot record it.
        channel = (body.get("channel") or {}).get("id", "")
        message_ts = (body.get("message") or {}).get("ts")
        if not channel:
            return
        try:
            client.chat_postMessage(channel=channel, text=answer, thread_ts=message_ts)
        except Exception:
            pass

    @app.action(APPROVE_ACTION)
    def _approve(ack: Any, body: dict[str, Any], client: Any) -> None:
        _decide(ack, body, client, deny=False)

    @app.action(DENY_ACTION)
    def _deny(ack: Any, body: dict[str, Any], client: Any) -> None:
        _decide(ack, body, client, deny=True)

    return app


def serve(config: SlackBotConfig) -> None:
    """Open the Socket Mode connection and block until interrupted."""
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    SocketModeHandler(build_app(config), config.app_token).start()
