"""The gate between Slack text and an argv: what the bot will and will not run.

Anyone in the workspace can talk to the bot, so this is an admission decision,
not a convenience parser — the same posture as the CLI's own policy layer. The
rules, and why each exists:

- **Subcommands are allowlisted.** `serve` (holds a worker thread forever) is
  not in the list. `agent` (tool execution on the host) is behind a double
  opt-in: GRAPHARC_SLACK_ALLOW_AGENT *and* GRAPHARC_SLACK_ALLOW_MODEL, because
  it acts on the host and cannot run without a paid backend. Even then its
  executor stays `sandbox` (`--executor` is not admitted), `--system-prompt`
  is unreachable, and its workspace defaults into the bot's working directory.
- **Flags are allowlisted per subcommand.** `--registry MODULE:ATTR` imports
  an arbitrary module on the host, `--config PATH` swaps the governing file,
  and `--json`/`--no-color` fight the bot's own output handling — none are
  reachable from Slack, with one carve-out: `plan --registry` accepts exactly
  the registry modules this package ships (`PLAN_REGISTRIES`), because code
  the wheel itself carries is the operator's, not the requester's.
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
    # flag -> the exact values it may take. How `--registry` stays shut against
    # arbitrary imports while the registries this package ships stay reachable.
    choice_flags: dict[str, frozenset[str]] = field(default_factory=dict)


_BUDGET = {"--max-tokens": False, "--max-iterations": False, "--max-seconds": False}
_NAMED_RUN = {"--trace": True, "--run-id": False}

#: The only `--registry` values `plan` accepts from Slack: the two registries
#: this package ships. The flag stays refused everywhere else — its value is
#: an arbitrary `module:attr` import, which is exactly what the gate exists to
#: prevent — but a registry the wheel itself carries is the operator's code.
PLAN_REGISTRIES = frozenset(
    {
        "grapharc.examples.plan_incident:build_registry",
        "grapharc.examples.plan_docs:build_registry",
    }
)

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
        choice_flags={"--registry": PLAN_REGISTRIES},
    ),
    "models": CommandSpec(bool_flags=frozenset({"--check"})),
    "agent": CommandSpec(
        value_flags={
            "--workspace": True,
            "--trace": True,
            "--run-id": False,
            "--allow": False,
            "--deny": False,
            "--max-turns": False,
            "--max-tokens": False,
            "--max-seconds": False,
        },
        model_flags=frozenset({"--model"}),
        # `local` (no confinement) stays unreachable; `claude-cli` delegates to
        # Claude Code's own sandboxed loop, which the injection below tempers.
        choice_flags={"--executor": frozenset({"sandbox", "claude-cli"})},
    ),
    "replay": CommandSpec(path_positionals=frozenset({0})),
    "diff": CommandSpec(path_positionals=frozenset({0})),
    "trace": CommandSpec(value_flags={"--run-id": False}, path_positionals=frozenset({0})),
    "metrics": CommandSpec(path_positionals=frozenset({0})),
    "viz": CommandSpec(path_positionals=frozenset({0})),
}


def usage_text(*, allow_model: bool = False, allow_agent: bool = False) -> str:
    """One short message for an empty or unrecognised request."""
    agent_on = allow_agent and allow_model
    lines = ["I run `grapharc` commands. Allowed here:"]
    for name in sorted(ALLOWED_COMMANDS):
        if name == "agent" and not agent_on:
            continue
        lines.append(f"• `{name}`")
    lines.append("`serve` is not reachable from Slack, nor is `--registry`.")
    if not agent_on:
        lines.append(
            "`agent` is off; it needs both GRAPHARC_SLACK_ALLOW_AGENT=1 "
            "and GRAPHARC_SLACK_ALLOW_MODEL=1 in the shell that starts the bot."
        )
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


def parse_command(
    text: str,
    *,
    workdir: Path,
    allow_model: bool = False,
    allow_agent: bool = False,
    timeout_seconds: float | None = None,
) -> list[str]:
    """Turn Slack text into the argv the bot may run, or raise with the reason."""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise SlackCommandError(f"could not parse that: {exc}") from None

    if tokens and tokens[0] == "grapharc":
        tokens = tokens[1:]
    if not tokens:
        raise SlackCommandError(usage_text(allow_model=allow_model, allow_agent=allow_agent))

    name, rest = tokens[0], tokens[1:]
    spec = ALLOWED_COMMANDS.get(name)
    if spec is None:
        raise SlackCommandError(
            f"`{name}` is not a command this bot runs.\n"
            + usage_text(allow_model=allow_model, allow_agent=allow_agent)
        )
    if name == "agent" and not (allow_agent and allow_model):
        # A double opt-in: `agent` both executes tools on the host and cannot
        # run without a real (paid) backend, so it needs the agent switch AND
        # the spend switch. One without the other stays off.
        raise SlackCommandError(
            "`agent` executes tools on the host and is off by default; the operator "
            "enables it with both GRAPHARC_SLACK_ALLOW_AGENT=1 and "
            "GRAPHARC_SLACK_ALLOW_MODEL=1 in the shell that starts the bot"
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
            elif flag in spec.choice_flags or flag in spec.value_flags:
                is_path = spec.value_flags.get(flag, False)
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
            if flag in spec.choice_flags and value not in spec.choice_flags[flag]:
                allowed = ", ".join(f"`{v}`" for v in sorted(spec.choice_flags[flag]))
                raise SlackCommandError(f"`{flag}` from Slack accepts only: {allowed}")
            if is_path:
                _confined(value, workdir)
            argv.extend([flag, value])
            continue
        if positional_index in spec.path_positionals:
            _confined(token, workdir)
        argv.append(token)
        positional_index += 1
        index += 1

    if name == "agent":
        # The CLI's default workspace is a fresh temp dir — *outside* the
        # bot's world, where nothing written there could be read back from
        # Slack. Default it to a subdirectory instead (the CLI mkdirs it);
        # `--workspace` can still choose any confined path.
        if "--workspace" not in argv:
            argv.extend(["--workspace", "agent"])
        # The CLI's max_seconds interrupts the run cleanly and reports; the
        # bot's timeout kills the process mid-sentence. Default the ceiling
        # to just under the timeout so the graceful mechanism fires first.
        if "--max-seconds" not in argv and timeout_seconds is not None:
            argv.extend(["--max-seconds", str(max(5.0, timeout_seconds - 10.0))])
        # A delegated run uses Claude Code's tools, and its Bash is a real
        # shell on the host with no grapharc sandbox around it. From Slack
        # that defaults off; a requester who set explicit globs made a
        # deliberate policy and keeps it (deny still beats allow downstream).
        delegated = "--executor" in argv and argv[argv.index("--executor") + 1] == "claude-cli"
        if delegated and "--allow" not in argv and "--deny" not in argv:
            argv.extend(["--deny", "Bash"])

    return argv
