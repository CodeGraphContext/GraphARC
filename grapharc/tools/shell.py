"""run_command — an argv list, a timeout, and a scrubbed environment.

Three things this deliberately is not, said here rather than discovered later:

* **It is not a shell.** `argv` is a list and `shell=False`, always. A single
  string is refused rather than split: splitting reintroduces quoting rules the
  caller never asked for, and `shell=True` makes every model-supplied substring
  an injection site. A pipeline is spelled `["bash", "-lc", "..."]` — explicit,
  visible in the tool-call record, and the caller's decision to permit.
* **It is not confined.** The child runs outside every guard in this package.
  Its cwd is a workspace-resolved directory and its environment is an
  allowlist; past that it is an ordinary process with the caller's privileges
  and can reach the whole filesystem. What limits `run_command` is the
  permission policy deciding whether it may run at all, and whatever real
  boundary the executor provides — not this function. Registering it is a
  decision of a different size from registering `read_file`.
* **It cannot run under `SandboxedExecutor`.** That executor's audit hook
  refuses `subprocess.Popen` outright, because a child process runs without the
  hook. Executing `run_command` through the default executor raises
  `SandboxViolation` every time — in practice at the `/dev/null` open that
  `stdin=DEVNULL` performs a moment before the spawn, which is the same refusal
  arriving one event earlier. It needs `LocalExecutor` today, or ROADMAP §3.2's
  container executor once that exists.

What it does provide is the discipline a subprocess call needs to be auditable:
a deadline that kills the whole process group rather than only the child it
started, an environment allowlist so a provider key is not handed to whatever
was run, and both streams captured and bounded separately.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from grapharc.harness.tools import ToolSpec
from grapharc.tools.workspace import ToolError, ToolLimits, Workspace, truncate_text

# The child's whole environment. An allowlist, not a blocklist: a secret nobody
# thought to name cannot leak merely by being new. Anything the command needs
# beyond this — a token, a config path — is the caller's to pass in argv.
SAFE_ENV_KEYS = frozenset(
    {"HOME", "LANG", "LOGNAME", "PATH", "SHELL", "TERM", "TZ", "USER"}
)
SAFE_ENV_PREFIXES = ("LC_",)

# Grace after a SIGKILL to the process group before giving up on the pipes.
_REAP_SECONDS = 5.0

RUN_COMMAND_DESCRIPTION = (
    "Run a program in the workspace and return its exit code, stdout and "
    "stderr. Use this for builds, test suites, linters, git and anything else "
    "with a command-line interface; use read_file, glob and grep for inspecting "
    "files, which is cheaper and cannot have side effects. 'argv' is a LIST of "
    "strings — the program followed by its arguments, e.g. ['pytest', '-q'] — "
    "and is never passed through a shell, so pipes, globs, redirects and '&&' "
    "do not work: run ['bash', '-lc', '<command line>'] if you actually need "
    "shell syntax. 'cwd' is a directory inside the workspace and defaults to "
    "its root. 'timeout_seconds' bounds the run; on expiry the process group "
    "is killed and whatever it printed first is still returned. Output is "
    "truncated with a marker if it is long."
)


def _check_argv(argv: Any) -> list[str]:
    """`argv` as a list of strings, or a refusal explaining the list.

    A string is refused rather than split. With `shell=False`, `subprocess`
    reads `"rm -rf /tmp/x"` as the name of a program with spaces in it and
    fails with a confusing ENOENT; splitting it here would quietly apply shell
    word-splitting that nothing in this function honours afterwards. Saying so
    is the only answer that leaves the caller with a correct next step.
    """
    if isinstance(argv, (str, bytes)):
        raise ToolError(
            "argv must be a list of strings, not one string: run_command never "
            'invokes a shell. Use ["ls", "-la"] — or ["bash", "-lc", "ls -la | head"] '
            "if you genuinely need shell syntax."
        )
    if not isinstance(argv, (list, tuple)):
        raise ToolError(f"argv must be a list of strings, got {type(argv).__name__}")
    if not argv:
        raise ToolError("argv is empty; its first element is the program to run")
    command: list[str] = []
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            raise ToolError(f"argv[{index}] must be a string, got {type(item).__name__}")
        if "\x00" in item:
            raise ToolError(f"argv[{index}] contains a NUL byte")
        command.append(item)
    return command


def _check_timeout(requested: Any, limits: ToolLimits) -> float:
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        raise ToolError(
            f"timeout_seconds must be a number, got {type(requested).__name__}"
        )
    if requested <= 0:
        raise ToolError("timeout_seconds must be greater than zero")
    if requested > limits.max_command_timeout:
        raise ToolError(
            f"timeout_seconds {requested} exceeds this toolset's ceiling of "
            f"{limits.max_command_timeout}s"
        )
    return float(requested)


def _child_environment(cwd: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_ENV_KEYS or key.startswith(SAFE_ENV_PREFIXES)
    }
    environment["PWD"] = str(cwd)
    return environment


def _kill_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL the child's whole session, not just the child.

    `start_new_session=True` put the child in its own process group whose id is
    its pid. Killing the child alone leaves any grandchildren — the actual
    compiler, the actual test run — holding the inherited pipe open, and
    `communicate()` then blocks past the deadline it was supposed to enforce.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def run_command_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def run_command(
        argv: list[str], cwd: str = ".", timeout_seconds: float = limits.command_timeout
    ) -> str:
        """Run `argv` (no shell) inside the workspace and return its result."""
        command = _check_argv(argv)
        timeout = _check_timeout(timeout_seconds, limits)
        directory = workspace.resolve(cwd, argument="cwd")
        if not directory.is_dir():
            raise ToolError(f"cwd {workspace.display(directory)} is not a directory")

        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(directory),
                env=_child_environment(directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                errors="replace",
                shell=False,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"command not found: {command[0]!r}") from exc
        except PermissionError as exc:
            raise ToolError(f"not executable: {command[0]!r}") from exc
        except OSError as exc:
            raise ToolError(f"could not start {command[0]!r}: {exc}") from exc

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(process)
            try:
                stdout, stderr = process.communicate(timeout=_REAP_SECONDS)
            except subprocess.TimeoutExpired:
                # A descendant survived SIGKILL to the group (uninterruptible
                # sleep, most likely) and still holds the pipe. Report the run
                # rather than blocking the agent on it forever.
                stdout, stderr = "", ""
        elapsed = time.perf_counter() - started

        # Halved, so a flood on one stream cannot crowd the other out entirely:
        # the error message is usually the reason the command was run twice.
        per_stream = max(1, limits.max_output_chars // 2)
        status = f"exit code {process.returncode} after {elapsed:.2f}s"
        if timed_out:
            status += f" (timed out at {timeout:g}s; the process group was killed)"
        return "\n".join(
            [
                status,
                f"stdout:\n{truncate_text(stdout, per_stream) if stdout else '(empty)'}",
                f"stderr:\n{truncate_text(stderr, per_stream) if stderr else '(empty)'}",
            ]
        )

    return ToolSpec(
        name="run_command",
        description=RUN_COMMAND_DESCRIPTION,
        fn=run_command,
        # The tool enforces its own deadline; the executor's is the backstop for
        # a run_command that never returns at all, so it has to sit above the
        # largest deadline any single call may ask for.
        timeout_seconds=limits.max_command_timeout + _REAP_SECONDS + 10.0,
    )


__all__ = [
    "RUN_COMMAND_DESCRIPTION",
    "SAFE_ENV_KEYS",
    "SAFE_ENV_PREFIXES",
    "run_command_tool",
]
