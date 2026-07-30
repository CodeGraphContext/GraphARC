"""Run one admitted argv against this interpreter's grapharc, and say what happened.

`sys.executable -m grapharc.cli.main` rather than a `grapharc` found on PATH, so the
bot always runs the code it was installed with — a PATH pointing at some other
environment cannot swap the CLI out from under it. The subprocess gets the
bot's working directory as cwd, which is the same directory the gate confined
every path argument to.

stdout here is a pipe, so by the CLI's own contract the output is the plain
byte-stable form with no colour — exactly what belongs in a Slack code fence.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """One finished (or timed-out) command, everything the formatter needs."""

    argv: list[str]
    exit_code: int | None  # None: the timeout fired and the process was killed
    stdout: str
    stderr: str
    duration_seconds: float
    timeout_seconds: float


def run_command(argv: list[str], *, workdir: Path, timeout_seconds: float) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "grapharc.cli.main", *argv],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        def _decode(stream: bytes | str | None) -> str:
            if stream is None:
                return ""
            return stream.decode(errors="replace") if isinstance(stream, bytes) else stream

        return CommandResult(
            argv=argv,
            exit_code=None,
            stdout=_decode(exc.stdout),
            stderr=_decode(exc.stderr),
            duration_seconds=time.monotonic() - started,
            timeout_seconds=timeout_seconds,
        )
    return CommandResult(
        argv=argv,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.monotonic() - started,
        timeout_seconds=timeout_seconds,
    )
