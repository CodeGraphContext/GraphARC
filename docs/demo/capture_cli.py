"""Run a scripted terminal session for real and record what it printed.

Each step is a shell command run in a **pseudo-terminal**, so the CLI takes its
tty branch and emits the colour a person actually sees — piping it would give
the byte-stable colourless form instead, which is a different (also real, also
tested) output and not the one a demo is about.

Nothing is faked: the commands run, in order, in a working directory you name,
and what lands in the recording is their bytes and their wall-clock duration.
A step may declare `expect_exit` so a demo that is *supposed* to show a failure
(a crash, a refusal) does not read as the capture going wrong.

    python docs/demo/capture_cli.py docs/demo/scenarios/gate.py --out session.json

A scenario module defines `WORKDIR`, `TITLE`, and `STEPS` — see
`docs/demo/scenarios/` for the three this repo ships.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pty
import selectors
import subprocess
import time
from pathlib import Path

#: The terminal width every recording is made at, so frames line up with the
#: renderer's grid and the CLI wraps where the video shows it wrapping.
COLUMNS = 100


def run_step(
    command: str, *, cwd: Path, timeout: float, venv_bin: Path | None = None
) -> dict:
    """One command in a pty. Returns its bytes, exit code and duration."""
    master, slave = pty.openpty()
    started = time.monotonic()
    environment = {
        **os.environ,
        # A real terminal width, so the CLI's rules and tables wrap the way
        # they would on someone's screen rather than at whatever COLUMNS the
        # capture process happened to inherit.
        "COLUMNS": str(COLUMNS),
        "TERM": "xterm-256color",
    }
    if venv_bin is not None:
        # So the recording shows `grapharc …`, which is what a reader will
        # type, rather than `python -m grapharc.cli.main …`, which is what a
        # checkout without an activated environment would need.
        environment["PATH"] = f"{venv_bin}{os.pathsep}{environment.get('PATH', '')}"
    process = subprocess.Popen(
        ["bash", "--noprofile", "--norc", "-c", command],
        cwd=cwd,
        stdout=slave,
        stderr=slave,
        stdin=subprocess.DEVNULL,
        env=environment,
    )
    os.close(slave)
    chunks: list[bytes] = []
    selector = selectors.DefaultSelector()
    selector.register(master, selectors.EVENT_READ)
    deadline = started + timeout
    while True:
        if time.monotonic() > deadline:
            process.kill()
            chunks.append(b"\n[capture] timed out\n")
            break
        for _key, _mask in selector.select(timeout=0.2):
            try:
                data = os.read(master, 65536)
            except OSError:
                data = b""
            if data:
                chunks.append(data)
        if process.poll() is not None:
            # Drain whatever the child wrote just before exiting.
            while True:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            break
    selector.close()
    os.close(master)
    process.wait()
    return {
        "command": command,
        "output": b"".join(chunks).decode("utf-8", errors="replace"),
        "exit_code": process.returncode,
        "seconds": round(time.monotonic() - started, 2),
    }


def load_scenario(path: Path):
    spec = importlib.util.spec_from_file_location("scenario", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--venv-bin",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".venv" / "bin",
        help="prepended to PATH so `grapharc` resolves without an activated venv",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    workdir = Path(scenario.WORKDIR).resolve()
    setup = getattr(scenario, "setup", None)
    if setup is not None:
        setup(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    steps = []
    for step in scenario.STEPS:
        command = step["run"]
        print(f"$ {command}", flush=True)
        result = run_step(
            command,
            cwd=workdir,
            timeout=step.get("timeout", args.timeout),
            venv_bin=args.venv_bin,
        )
        expected = step.get("expect_exit")
        if expected is not None and result["exit_code"] != expected:
            # Loud, not fatal: a demo whose refusal stopped refusing is a
            # finding about the code, and the recording is the evidence.
            print(
                f"  ! expected exit {expected}, got {result['exit_code']}",
                flush=True,
            )
        result["caption"] = step.get("caption", "")
        result["expect_exit"] = expected
        result["hold"] = step.get("hold")
        steps.append(result)
        print(f"  exit {result['exit_code']} in {result['seconds']}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "title": scenario.TITLE,
                "subtitle": getattr(scenario, "SUBTITLE", ""),
                "workdir": str(workdir),
                "steps": steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(steps)} steps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
