"""A Slack front door for the `grapharc` CLI, over Socket Mode.

The bot holds one outbound WebSocket to Slack, so it runs anywhere with
internet — a laptop behind NAT included. No public URL, no open port. A
workspace member types `/grapharc metrics t.jsonl r1` (or mentions the bot)
from any Slack client, the bot runs the command on the host, and posts the
piped-mode output back in the thread. The piped bytes are already the CLI's
machine interface — colourless, byte-compared against the docs — which is why
the bot posts them verbatim inside a code fence instead of inventing a third
output format.

The module is layered so everything with behaviour is testable without Slack:

    command.py   what Slack text is allowed to become an argv (the gate)
    runner.py    run an argv against this interpreter's grapharc, with a timeout
    format.py    turn an exit code and captured output into one Slack message
    live.py      tail a running command's trace and narrate it via a callback
    bot.py       slack-bolt wiring; the only file that imports slack_bolt
    config.py    tokens and limits from the environment, nothing else

Only `bot.py` needs the `slack` extra, and it imports it lazily — every other
module (and this package) needs nothing beyond grapharc's own core
dependencies, so a wheel without the extra still imports.

A tracing command (`demo`, `run`, `plan`, `agent`) is narrated live: the gate
gives it a trace path the bot knows, `live.py` tails that file while the
subprocess runs, and the bot edits one status message in place — a node
progress line per event batch plus a refreshed mermaid.live diagram link.
Every failure in that path degrades to the plain blocking reply.

The gate's default is deliberately spend-free: `serve` is refused, `agent` and
`--model` are refused unless the operator opts in (`agent` needs two switches:
it acts on the host *and* it spends), and every path argument must resolve
inside the bot's working directory. Anyone in the workspace can talk to the
bot; the gate is what makes that safe to allow.
"""

from grapharc.slack.command import (
    ALLOWED_COMMANDS,
    SlackCommandError,
    parse_command,
)
from grapharc.slack.config import SlackBotConfig, SlackConfigError
from grapharc.slack.format import format_result
from grapharc.slack.runner import CommandResult, run_command

__all__ = [
    "ALLOWED_COMMANDS",
    "CommandResult",
    "SlackBotConfig",
    "SlackCommandError",
    "SlackConfigError",
    "format_result",
    "parse_command",
    "run_command",
]
