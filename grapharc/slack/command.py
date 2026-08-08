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
- **A flag may not be repeated.** The gate admits a command by reading a flag's
  value, and argparse's `store` action then runs the *last* occurrence — so any
  reader that takes a different one is a bypass, and `--registry <shipped demo>
  --registry <stdlib>` was exactly that: admitted against the benign value,
  executed against the agent registry, with the forced `--approve` skipped in
  the same step. Refusing the repeat is the fail-closed reading and it closes
  the whole first-vs-last family at once, rather than the one flag that showed
  it. The carve-out is `repeatable_flags`: options the CLI itself accumulates
  (`agent --allow/--deny`, argparse `action="append"`), where every occurrence
  reaches the run and there is no "other" value to diverge from.
- **Every path must resolve inside the bot's working directory.** `trace
  ../../.env` is refused before a process is spawned, whether it arrives as a
  positional or as a flag value.

The output is an argv list for `runner.py`, never a shell string — nothing a
user types is ever interpreted by a shell.
"""

from __future__ import annotations

import importlib
import shlex
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

#: What the CLI plans with when neither a flag nor grapharc.toml says
#: otherwise. A second copy of `cli.plan.DEFAULT_REGISTRY`, kept here so this
#: module stays importable with nothing but the standard library — a test
#: asserts the two agree.
DEFAULT_PLAN_REGISTRY = "grapharc.examples.plan_incident:build_registry"


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
    # flags the CLI accumulates (argparse `action="append"`), so a second
    # occurrence adds to the run rather than replacing what the gate read.
    # Every other flag is refused on its second occurrence.
    repeatable_flags: frozenset[str] = frozenset()


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
        "grapharc.stdlib:build_registry",
    }
)

#: The one plan registry whose kinds execute agent tools on the host (under
#: `LocalExecutor`, path-confined to the workspace but unsandboxed). From Slack
#: it needs the same double opt-in as `agent`, and every run of it is parked on
#: the human approval gate before anything executes.
AGENT_PLAN_REGISTRY = "grapharc.stdlib:build_registry"

#: Subcommands whose Slack run *does* something rather than reading a file
#: back. They get `work_timeout_seconds` instead of the reader timeout: a
#: delegated Claude Code phase takes minutes, and the 120s that is generous for
#: `metrics` is a SIGKILL through the middle of an approved run.
WORK_COMMANDS = frozenset({"agent"})

#: The longest a parked plan may hold its wall clock waiting for a human. The
#: rest of the budget belongs to the work the human is approving; without a cap
#: a 30-minute work budget could be spent entirely on nobody answering.
APPROVAL_WAIT_CAP_SECONDS = 900.0

#: Subcommands that write a trace while they run. The gate gives each of them
#: a trace path it knows (unless the requester named one), so the bot can tail
#: the file for live progress and readers (`metrics`, `viz`, `replay`) can be
#: pointed at it from Slack afterwards. The CLI's own defaults are tempdirs
#: outside the bot's world (`agent` excepted), where nothing can be read back.
LIVE_COMMANDS = frozenset({"demo", "run", "plan", "agent"})

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
            "--approval-timeout": False,
        },
        # `--go` executes the admitted graph in the same run instead of
        # stopping at PLANNED. Admitted from Slack because without it the
        # supervised loop had nowhere to go: a parked plan was approved by a
        # human and then returned "awaiting `grapharc go`" — into a subcommand
        # this gate does not carry. It is safe to admit only because of the
        # rule below, which makes `--go` imply `--approve`.
        bool_flags=frozenset({"--approve", "--scripted", "--go"}),
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
        # Tool-name globs: the CLI appends them, so `--deny 'shell*' --deny
        # 'net*'` denies both. Repeating them narrows the run, never widens it.
        repeatable_flags=frozenset({"--allow", "--deny"}),
    ),
    "approve": CommandSpec(
        # `--show` decides nothing and `--fingerprint` binds a decision to the
        # plan that was read, so both make the Slack path safer rather than
        # wider: from a phone, "look at it first" and "only if it is still
        # this one" are exactly the two things a reviewer needs.
        value_flags={"--fingerprint": False},
        bool_flags=frozenset({"--deny", "--show"}),
        path_positionals=frozenset({0}),
    ),
    "replay": CommandSpec(path_positionals=frozenset({0})),
    "diff": CommandSpec(path_positionals=frozenset({0})),
    "trace": CommandSpec(value_flags={"--run-id": False}, path_positionals=frozenset({0})),
    "metrics": CommandSpec(path_positionals=frozenset({0})),
    "viz": CommandSpec(path_positionals=frozenset({0})),
}


def effective_timeout(
    argv: list[str], *, timeout_seconds: float, work_timeout_seconds: float
) -> float:
    """The wall clock this argv gets before the runner kills it.

    Two budgets, because two very different things are being bounded. A reader
    (`metrics`, `viz`, `trace`) that has not answered in two minutes is stuck,
    and holding a bot worker thread longer helps nobody. A command that plans
    and then executes is *supposed* to take minutes — a delegated Claude Code
    phase reads files and edits them — and killing it at 120s does not protect
    anything, it just severs an approved run partway through its work.

    `bot.py` calls this on the argv the gate returned, so the runner and the
    gate's own `--approval-timeout` / `--max-seconds` injections agree on which
    budget is in play.
    """
    if not argv:
        return timeout_seconds
    if argv[0] in WORK_COMMANDS:
        return work_timeout_seconds
    if argv[0] == "plan" and _has_flag(argv, "--go"):
        return work_timeout_seconds
    return timeout_seconds


def mutating_kinds_for(argv: list[str], workdir: Path | None = None) -> frozenset[str] | None:
    """Which kinds of this argv's registry can change things, for the approver.

    None means "assume every kind does" — the same fail-closed reading `plan`
    takes when a registry declares no `MUTATING_KINDS`. It is returned whenever
    this cannot *prove* which registry will be in play, because the mark exists
    to warn and a mark that is wrong in the reassuring direction is worse than
    no mark at all.

    Only modules in `PLAN_REGISTRIES` are ever imported. A `--registry` outside
    that set was already refused by the gate; a `grapharc.toml` may point the
    CLI at any module at all (including a `.py` file beside it), and that one is
    reported as unknown rather than imported here — this runs inside the bot
    process, and the bot does not execute the working directory's code.
    """
    if not argv or argv[0] != "plan":
        return frozenset()
    target = _flag_value(argv, "--registry")
    if target is None:
        # No flag: the CLI will resolve `registry` from grapharc.toml in the
        # working directory (which is the runner's cwd), else its own default.
        target = _configured_registry(workdir) or DEFAULT_PLAN_REGISTRY
    if target not in PLAN_REGISTRIES:
        return None
    module_name = target.partition(":")[0]
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    declared = getattr(module, "MUTATING_KINDS", None)
    if declared is None:
        return None
    return frozenset(str(kind) for kind in declared)


def _configured_registry(workdir: Path | None) -> str | None:
    """`registry` from the working directory's grapharc.toml, if it sets one."""
    if workdir is None:
        return None
    path = workdir / "grapharc.toml"
    try:
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    section = loaded.get("grapharc")
    if not isinstance(section, dict):
        return None
    value = section.get("registry")
    return value if isinstance(value, str) else None


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


def confine_path(raw: str, workdir: Path) -> None:
    """Refuse a path that escapes the working directory, before anything runs.

    Lexical resolution only — the target need not exist yet (`--trace` names a
    file the run will create).

    Every refusal here leaves as a `SlackCommandError`, including the ones the
    filesystem raises: `Path.resolve()` throws `ValueError` on a NUL byte, and
    the bolt listeners catch only `SlackCommandError`, so anything else escapes
    the handler and the requester gets no reply at all — silence being the one
    answer a chat bot must never give, since it is indistinguishable from being
    down. `parse_command` screens NUL bytes out of the whole request before it
    gets here; this keeps the guarantee for any other caller.
    """
    if "\x00" in raw:
        # `repr`, not the raw string: a NUL echoed back into a Slack message
        # is invisible, and an invisible character is exactly what the reader
        # needs to see named.
        raise SlackCommandError(f"path contains a NUL byte, which cannot name a file: {raw!r}")
    try:
        resolved = (
            (workdir / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        )
    except (ValueError, OSError) as exc:
        raise SlackCommandError(f"that is not a usable path: {exc}") from None
    if not resolved.is_relative_to(workdir.resolve()):
        raise SlackCommandError(f"path escapes the bot's working directory: `{raw}`")


def parse_command(
    text: str,
    *,
    workdir: Path,
    allow_model: bool = False,
    allow_agent: bool = False,
    timeout_seconds: float | None = None,
    work_timeout_seconds: float | None = None,
) -> list[str]:
    """Turn Slack text into the argv the bot may run, or raise with the reason.

    `work_timeout_seconds` is the budget a command that *executes* gets (see
    `effective_timeout`); the injected `--approval-timeout` and `--max-seconds`
    are derived from whichever of the two will actually apply, so the CLI's own
    graceful limits fire before the runner's kill rather than after it.
    """
    # People copy commands out of code-formatted Slack messages, and the
    # backticks come along for the ride: "`approve x/trace.jsonl`" arrives
    # with a backtick glued to the first and last token. No admissible
    # command starts or ends with one, so wrapping backticks are noise.
    text = text.strip().strip("`").strip()
    # A NUL byte cannot name a file, cannot cross into `subprocess`, and makes
    # `Path.resolve()` raise `ValueError` — an exception type the bolt handlers
    # do not catch, so it would surface as no reply rather than as a refusal.
    # Screen it out of the whole request, not just the path-shaped parts: it is
    # never meaningful anywhere in a command, and this is the one place that
    # sees the text before anything tries to use it.
    if "\x00" in text:
        raise SlackCommandError("that contains a NUL byte, which cannot name a file or a command")
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
    seen_flags: set[str] = set()
    while index < len(rest):
        token = rest[index]
        # Any leading dash is a flag, not a positional. The allowlist is meant
        # to be exhaustive, and testing for `--` let short options through it:
        # `trace -h` was admitted and spent the path positional on `-h`. The
        # CLI has one short option and no positional that begins with a dash,
        # so treating the whole shape as a flag costs nothing and leaves the
        # allowlist the only way in.
        if token.startswith("-"):
            flag, eq, inline_value = token.partition("=")
            # Repeats are refused before the flag is read, because reading one
            # of several occurrences is what the gate cannot safely do — see
            # the module docstring. An inadmissible flag was already refused on
            # its first occurrence, so anything reaching here was admitted once.
            if flag in seen_flags and flag not in spec.repeatable_flags:
                raise SlackCommandError(
                    f"`{flag}` was given twice; from Slack a flag may appear only once, "
                    "because the gate and the CLI would not necessarily read the same one"
                )
            seen_flags.add(flag)
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
                confine_path(value, workdir)
            argv.extend([flag, value])
            continue
        if positional_index in spec.path_positionals:
            confine_path(token, workdir)
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
        # to just under the timeout so the graceful mechanism fires first —
        # `agent` is a work command, so that is the work budget, not the
        # reader's two minutes.
        budget = work_timeout_seconds if work_timeout_seconds is not None else timeout_seconds
        if "--max-seconds" not in argv and budget is not None:
            argv.extend(["--max-seconds", str(max(5.0, budget - 10.0))])
        # A delegated run uses Claude Code's tools, and its Bash is a real
        # shell on the host with no grapharc sandbox around it. From Slack
        # that defaults off; a requester who set explicit globs made a
        # deliberate policy and keeps it (deny still beats allow downstream).
        delegated = "--executor" in argv and argv[argv.index("--executor") + 1] == "claude-cli"
        if delegated and "--allow" not in argv and "--deny" not in argv:
            argv.extend(["--deny", "Bash"])

    if name == "plan" and _flag_value(argv, "--registry") == AGENT_PLAN_REGISTRY:
        # The stdlib registry materializes agent nodes that run tools on the
        # host, so it inherits `agent`'s double opt-in — and, opted in or not,
        # a Slack-launched agent plan always parks on the human approval gate.
        # The gate is answered with `/grapharc approve <trace>` — by ANY
        # workspace member, not only the requester: the handshake is bound to
        # the run's directory, and the workspace is the trust boundary here
        # exactly as it is for every other command the bot runs. A human saw
        # the graph and said yes; *which* human is not recorded (the trace has
        # no actor field — see the architecture review).
        if not (allow_agent and allow_model):
            raise SlackCommandError(
                "the stdlib plan registry runs agent tools on the host and is off "
                "by default; the operator enables it with both "
                "GRAPHARC_SLACK_ALLOW_AGENT=1 and GRAPHARC_SLACK_ALLOW_MODEL=1 "
                "in the shell that starts the bot"
            )
        if not _has_flag(argv, "--approve"):
            argv.append("--approve")

    if name == "plan" and _has_flag(argv, "--go") and not _has_flag(argv, "--approve"):
        # `--go` is the difference between proposing a graph and running one,
        # and from Slack that difference always goes through a person. Any
        # workspace member can type into this bot; without this rule a single
        # message would take a model's proposal straight to execution on the
        # host, with the graph visible only afterwards.
        #
        # It is forced rather than refused so that the useful command stays one
        # message: `plan "…" --go` means propose it, show it to me, and run it
        # if I say yes. The stdlib rule above already forces `--approve` for the
        # agent registry; this extends the same reading to every registry, since
        # what makes execution worth gating is that it executes.
        argv.append("--approve")

    # ANY parked plan — stdlib-injected, `--go`-forced, or a requester's own
    # `--approve` on a demo registry — must time its wait under the runner's
    # kill: the CLI default (300s) exceeds the bot's reader default (120s), and
    # a park that outlives the runner is a hard kill mid-wait instead of a clean
    # approval_timeout.
    #
    # How much of the budget the human gets depends on what happens after they
    # answer. A plan-only park has nothing left to do, so the wait may have half
    # the clock. With `--go`, saying yes is the *start* of the work — give the
    # human a third, capped, and leave the rest for the run they authorised.
    if name == "plan" and _has_flag(argv, "--approve") and timeout_seconds is not None:
        budget = effective_timeout(
            argv,
            timeout_seconds=timeout_seconds,
            work_timeout_seconds=(
                timeout_seconds if work_timeout_seconds is None else work_timeout_seconds
            ),
        )
        ceiling = max(10.0, budget - 10.0)
        supplied = _flag_value(argv, "--approval-timeout")
        if supplied is None:
            share = budget / 3 if _has_flag(argv, "--go") else budget / 2
            wait = max(10.0, min(ceiling, share, APPROVAL_WAIT_CAP_SECONDS))
            argv.extend(["--approval-timeout", str(wait)])
        else:
            # A requester may ask for a shorter wait, never a longer one. The
            # injection above only fires when the flag is absent, so without
            # this check `--approval-timeout 100000` sailed past it and the
            # park outlived the runner — which is not an `approval_timeout`
            # the run can report, it is a SIGKILL through the middle of the
            # wait. Refused rather than clamped: silently running a different
            # command from the one someone typed is how a gate loses its
            # meaning.
            try:
                asked = float(supplied)
            except ValueError:
                raise SlackCommandError(
                    f"`--approval-timeout` wants a number of seconds, got {supplied!r}"
                ) from None
            if asked <= 0 or asked > ceiling:
                raise SlackCommandError(
                    f"`--approval-timeout {supplied}` does not fit this command's "
                    f"budget: the wait must be between 1 and {ceiling:.0f} seconds, "
                    "so that a run nobody answers ends by reporting a timeout "
                    "rather than by being killed mid-wait"
                )

    # A tracing command whose trace the requester did not place gets one the
    # bot can find: unique per invocation (a reused file would make a tailer's
    # first line some other run's), relative to the workdir the runner uses as
    # cwd. A requester-named `--trace` was already confined above and wins.
    if name in LIVE_COMMANDS and not _has_flag(argv, "--trace"):
        argv.extend(["--trace", _default_trace()])

    return argv


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The value the CLI will act on: the **last** occurrence, as argparse.

    `parse_command` already refuses a repeated flag, so there is only ever one
    here. Reading it argparse's way anyway is the cheap half of the belt: this
    function returning the *first* occurrence while `store` kept the last is
    precisely how a second `--registry` walked past the agent opt-in, and a
    future caller that assembles an argv some other way should not be able to
    reopen that gap.
    """
    value: str | None = None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(f"{flag}="):
            value = token.partition("=")[2]
    return value


def _default_trace() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"slack-runs/{stamp}-{uuid.uuid4().hex[:8]}/trace.jsonl"


def trace_path(argv: list[str], workdir: Path) -> Path | None:
    """The trace file an admitted argv will write, or None for a reader.

    The gate injects `--trace` into every `LIVE_COMMANDS` argv it admits, so
    for those this always resolves; it is how `bot.py` learns where to tail
    without re-deriving the gate's decisions.
    """
    if not argv or argv[0] not in LIVE_COMMANDS:
        return None
    raw: str | None = None
    for index, token in enumerate(argv):
        if token == "--trace" and index + 1 < len(argv):
            raw = argv[index + 1]
        elif token.startswith("--trace="):
            raw = token.partition("=")[2]
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else workdir / path
