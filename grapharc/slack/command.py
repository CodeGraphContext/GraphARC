"""The gate between Slack text and an argv: what the bot will and will not run.

Anyone in the workspace can talk to the bot, so this is an admission decision,
not a convenience parser — the same posture as the CLI's own policy layer. The
rules, and why each exists:

- **Subcommands are allowlisted.** `agent` (arbitrary tool execution on the
  host) and `serve` (holds a worker thread forever) are not in the list.
- **Flags are allowlisted per subcommand.** `--registry MODULE:ATTR` imports
  an arbitrary module on the host, `--config PATH` swaps the governing file,
  and `--json`/`--no-color` fight the bot's own output handling — none are
  reachable from Slack.
- **`--model` is refused unless the operator opted in**, because it reaches a
  paid backend. Without it every allowed command runs the scripted, spend-free
  path; the default answer to "can Slack cost me money?" is no.
- **Every path must resolve inside the bot's working directory.** `trace
  ../../.env` is refused before a process is spawned, whether it arrives as a
  positional or as a flag value.

The output is an argv list for `runner.py`, never a shell string — nothing a
user types is ever interpreted by a shell.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


class SlackCommandError(Exception):
    """The text is not something the bot will run; the message says why."""


@dataclass(frozen=True)
class CommandSpec:
    """What one subcommand may be given from Slack."""

    # flag -> True when its value is a path that must stay inside the workdir
    value_flags: dict[str, bool] = field(default_factory=dict)
    bool_flags: frozenset[str] = frozenset()
    # positional indices (0-based, after the subcommand) that are paths
    path_positionals: frozenset[int] = frozenset()
    # value flags that reach a paid backend; admitted only with allow_model
    model_flags: frozenset[str] = frozenset()


_BUDGET = {"--max-tokens": False, "--max-iterations": False, "--max-seconds": False}
_NAMED_RUN = {"--trace": True, "--run-id": False}

ALLOWED_COMMANDS: dict[str, CommandSpec] = {
    "demo": CommandSpec(
        value_flags={"--trace": True, "--memory": True, "--memory-backend": False},
        model_flags=frozenset({"--model", "--reviewer-model"}),
    ),
    "run": CommandSpec(
        value_flags={
            **_NAMED_RUN,
            "--policy": True,
            "--tenant": False,
            **_BUDGET,
            "--max-concurrency": False,
        },
        bool_flags=frozenset({"--check-only"}),
        path_positionals=frozenset({0}),
    ),
    "plan": CommandSpec(
        value_flags={
            **_NAMED_RUN,
            "--policy": True,
            "--tenant": False,
            "--max-rounds": False,
            "--max-tokens": False,
        },
        model_flags=frozenset({"--model"}),
    ),
    "models": CommandSpec(bool_flags=frozenset({"--check"})),
    "replay": CommandSpec(path_positionals=frozenset({0})),
    "diff": CommandSpec(path_positionals=frozenset({0})),
    "trace": CommandSpec(value_flags={"--run-id": False}, path_positionals=frozenset({0})),
    "metrics": CommandSpec(path_positionals=frozenset({0})),
    "viz": CommandSpec(path_positionals=frozenset({0})),
}


def usage_text(*, allow_model: bool = False) -> str:
    """One short message for an empty or unrecognised request."""
    lines = ["I run `grapharc` commands. Allowed here:"]
    for name in sorted(ALLOWED_COMMANDS):
        lines.append(f"• `{name}`")
    lines.append("`agent` and `serve` are not reachable from Slack, nor is `--registry`.")
    if not allow_model:
        lines.append(
            "`--model` is off; the operator can enable it with GRAPHARC_SLACK_ALLOW_MODEL=1."
        )
    return "\n".join(lines)


def _confined(raw: str, workdir: Path) -> None:
    """Refuse a path that escapes the working directory, before anything runs.

    Lexical resolution only — the target need not exist yet (`--trace` names a
    file the run will create).
    """
    resolved = (workdir / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not resolved.is_relative_to(workdir.resolve()):
        raise SlackCommandError(f"path escapes the bot's working directory: `{raw}`")


def parse_command(text: str, *, workdir: Path, allow_model: bool = False) -> list[str]:
    """Turn Slack text into the argv the bot may run, or raise with the reason."""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise SlackCommandError(f"could not parse that: {exc}") from None

    if tokens and tokens[0] == "grapharc":
        tokens = tokens[1:]
    if not tokens:
        raise SlackCommandError(usage_text(allow_model=allow_model))

    name, rest = tokens[0], tokens[1:]
    spec = ALLOWED_COMMANDS.get(name)
    if spec is None:
        raise SlackCommandError(
            f"`{name}` is not a command this bot runs.\n" + usage_text(allow_model=allow_model)
        )

    argv = [name]
    positional_index = 0
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("--"):
            flag, eq, inline_value = token.partition("=")
            if flag in spec.bool_flags:
                if eq:
                    raise SlackCommandError(f"`{flag}` takes no value")
                argv.append(flag)
                index += 1
                continue
            if flag in spec.model_flags:
                if not allow_model:
                    raise SlackCommandError(
                        f"`{flag}` reaches a paid backend and is off by default; "
                        "the operator enables it with GRAPHARC_SLACK_ALLOW_MODEL=1"
                    )
                is_path = False
            elif flag in spec.value_flags:
                is_path = spec.value_flags[flag]
            else:
                raise SlackCommandError(f"`{flag}` is not allowed on `{name}` from Slack")
            if eq:
                value = inline_value
                index += 1
            else:
                if index + 1 >= len(rest):
                    raise SlackCommandError(f"`{flag}` needs a value")
                value = rest[index + 1]
                index += 2
            if is_path:
                _confined(value, workdir)
            argv.extend([flag, value])
            continue
        if positional_index in spec.path_positionals:
            _confined(token, workdir)
        argv.append(token)
        positional_index += 1
        index += 1

    return argv
