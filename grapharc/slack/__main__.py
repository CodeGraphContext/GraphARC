"""`python -m grapharc.slack` — start the bot, or say exactly why it cannot.

A deliberate module entry rather than a `grapharc slack` subcommand: the CLI's
help text is printed verbatim in README.md and byte-compared by the test
suite, and a long-running daemon does not belong in a parser whose every other
command terminates. Exit codes keep the CLI's meaning: 0 on a clean shutdown,
2 when the environment does not describe a runnable bot.
"""

from __future__ import annotations

import sys

from grapharc.slack.command import SlackCommandError
from grapharc.slack.config import SlackBotConfig, SlackConfigError


def main() -> int:
    try:
        config = SlackBotConfig.from_env()
    except SlackConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        from grapharc.slack.bot import serve

        print(
            f"grapharc slack bot: workdir {config.workdir}, "
            f"timeout {config.timeout_seconds:.0f}s, "
            f"model flags {'on' if config.allow_model else 'off'}, "
            f"agent {'on' if config.allow_agent and config.allow_model else 'off'}",
            file=sys.stderr,
        )
        serve(config)
    except SlackCommandError as exc:  # the missing-extra message from build_app
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
