"""The MCP server's hands: build argv, spawn the CLI, read the record back.

Everything here drives `grapharc` **as a subprocess**, never in process, for
three reasons the server module repeats: the stdio transport owns stdout and
`emit()` prints there; the `--json` payloads and exit codes are the tested
interface, so this stays a thin shim over a contract that already has a
suite; and an async subprocess keeps the event loop free, so `show_graph`
answers while an `execute` is parked.

Argv is **built, never parsed**: no tool accepts a registry, policy or model
argument, because those resolve from the operator's `grapharc.toml` in the
root directory — the requester's call must not be able to widen what the
operator configured. Run directories are confined to the root the server was
started in, the same `is_relative_to` posture as the Slack gate's paths.

Only fastapi-free modules are imported here (`observe.trace`,
`observe.metrics`, `planner.approval_file`): this package ships behind the
`mcp` extra and must not drag the `server` extra in with it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

PLAN_FILENAME = "plan.json"

#: How long a parked `execute` waits for a human to answer. A parked call
#: lives inside one MCP call and hosts time tool calls out, so the default
#: stays under typical host limits. This bounds *the wait*, nothing else:
#: a park that expires without an answer leaves the plan unexecuted, and
#: only that case is safe to reissue.
DEFAULT_APPROVAL_TIMEOUT = 240.0

#: How long the *work* gets, once a plan is admitted and (if mutating) a
#: human has said yes. Separate from the park on purpose. These used to be
#: one budget of `approval_timeout + 120s` covering both, which meant a human
#: approving near the end of the park left roughly two minutes for the run
#: itself — and a governed run's agent phases delegate to Claude Code, which
#: reads files, edits them and verifies. Killing an approved run at 120s does
#: not protect anything; it severs an approved run partway through its work,
#: which for a mutating plan means partway through mutating the tree.
#:
#: 1800s matches `GRAPHARC_SLACK_WORK_TIMEOUT`, which the Slack path carved
#: out of one shared timeout for this exact reason. Overridable because the
#: right ceiling is a property of the operator's machine, not of this file.
DEFAULT_WORK_TIMEOUT = 1800.0


def work_timeout(env: dict[str, str] | None = None) -> float:
    """The work budget, from `GRAPHARC_MCP_WORK_TIMEOUT` or the default.

    Refuses a non-numeric or non-positive override rather than falling back
    to the default: an operator who sets this has a ceiling in mind, and
    silently substituting a different one is how a run gets killed at a
    limit nobody chose.
    """
    source = os.environ if env is None else env
    raw = source.get("GRAPHARC_MCP_WORK_TIMEOUT")
    if raw is None or raw == "":
        return DEFAULT_WORK_TIMEOUT
    try:
        seconds = float(raw)
    except ValueError:
        raise DriverError(
            f"GRAPHARC_MCP_WORK_TIMEOUT must be a number of seconds, not {raw!r}"
        ) from None
    if seconds <= 0:
        raise DriverError(
            f"GRAPHARC_MCP_WORK_TIMEOUT must be positive, not {seconds!r}"
        )
    return seconds


class DriverError(Exception):
    """A tool call that cannot proceed, with the reason as the message."""


def confine_run_dir(root: Path, run_dir: str) -> Path:
    """Resolve `run_dir` and refuse anything outside the server's root.

    The server reads plan.json and trace.jsonl from whatever directory a
    client names; without this, a client could point it at any readable
    path on the machine.
    """
    resolved = (root / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(
        run_dir
    ).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DriverError(
            f"run_dir {run_dir!r} is outside the directory this server was "
            f"started in ({root}); a run directory is always under it"
        )
    return resolved


def _kill_tree(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group, falling back to the child.

    The group is the point: see `start_new_session` in `run_cli`. A group
    that has already exited raises `ProcessLookupError`, which is the
    success case arriving early, and a platform without `killpg` still gets
    the old single-process behaviour rather than an exception.
    """
    # `ProcessLookupError` and `PermissionError` are `OSError`; `AttributeError`
    # is a platform with no `killpg` at all.
    with contextlib.suppress(AttributeError, OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


async def run_cli(
    argv: list[str], *, cwd: Path, timeout: float | None = None
) -> tuple[int, str, str]:
    """One `grapharc` subprocess, the Slack runner's spawn pattern made async."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "grapharc.cli.main",
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Its own process group, so the timeout below can kill the whole tree.
        # `process.kill()` signals the direct child only, and a governed run
        # delegates agent phases to Claude Code — so a killed run used to
        # leave that grandchild alive, still holding the workspace the next
        # call would run in. The CLI's own `deadline_guard` kills its group
        # deliberately, but that is the CLI's guard, not this outer one.
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        _kill_tree(process)
        await process.wait()
        raise DriverError(
            f"grapharc {argv[0]} did not finish within {timeout}s and was stopped. "
            "Whether the plan ran is recorded in the run directory, not here: "
            "read it back with show_graph before reissuing anything."
        ) from None
    return process.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def parse_document(stdout: str, *, command: str) -> dict[str, Any]:
    """The one JSON document a `--json` command prints, or a named refusal."""
    try:
        document = json.loads(stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DriverError(
            f"grapharc {command} --json did not print a readable document: "
            f"{stdout.strip()[-300:] or '(empty)'}"
        ) from exc
    if not isinstance(document, dict):
        raise DriverError(f"grapharc {command} --json printed {type(document).__name__}")
    return document


def read_plan_record(run_dir: Path) -> dict[str, Any]:
    plan_file = run_dir / PLAN_FILENAME
    if not plan_file.is_file():
        raise DriverError(
            f"no {PLAN_FILENAME} in {run_dir} — call plan first; execute and "
            "show_graph work on a directory plan created"
        )
    try:
        return json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DriverError(f"unreadable {plan_file}: {exc}") from exc


def plan_is_mutating(record: dict[str, Any]) -> bool:
    """The plan's own verdict — absent reads as mutating, never as safe."""
    value = record.get("mutating")
    return True if not isinstance(value, bool) else value


def graph_status(run_dir: Path) -> dict[str, Any]:
    """Compose the read-only view of one run directory.

    plan.json for the admitted shape, the trace for what has happened since,
    the approval-request file for whether a human is being asked right now.
    Rendered summaries only — the raw `state_delta` a node wrote is served by
    nothing here, the same posture as the live view.

    `TailRecorder`, not `TraceRecorder`, for the same reason the live view and
    the Slack tailer use it: this reads a file something else is still writing.
    An agent polls `show_graph` precisely *while* its run is in progress, so a
    read that lands mid-append is the normal case, not the edge case — and the
    strict reader raises `TraceReadError` on the half-written last line, which
    came out of the MCP server as a crash instead of an answer.
    """
    from grapharc.observe.metrics import summarize, to_mermaid
    from grapharc.observe.trace import TailRecorder
    from grapharc.planner.approval_file import read_request

    record = read_plan_record(run_dir)
    view: dict[str, Any] = {
        "run_dir": str(run_dir),
        "goal": record.get("goal", ""),
        "fingerprint": record.get("fingerprint", ""),
        "mutating": plan_is_mutating(record),
        "proposal": {
            "nodes": (record.get("proposal") or {}).get("nodes", []),
            "edges": (record.get("proposal") or {}).get("edges", []),
            "rationale": (record.get("proposal") or {}).get("rationale", ""),
        },
        "executed_run_id": record.get("executed_run_id"),
    }

    request = read_request(run_dir)
    view["awaiting_approval"] = request is not None
    view["approve_command"] = (
        f"grapharc approve {run_dir}" if request is not None else None
    )

    trace_path = run_dir / "trace.jsonl"
    view["status"] = "planned"
    if request is not None:
        view["status"] = "awaiting_approval"
    if record.get("executed_run_id") and trace_path.is_file():
        recorder = TailRecorder(trace_path)
        run_id = str(record["executed_run_id"])
        metrics = summarize(recorder, run_id)
        if metrics is not None:
            view["status"] = "done"
            view["metrics"] = metrics.model_dump(mode="json")
            view["mermaid"] = to_mermaid(recorder, run_id)
    return view


def build_plan_argv(goal: str, *, scripted: bool, max_rounds: int | None) -> list[str]:
    argv = ["plan", goal, "--json"]
    if scripted:
        argv.append("--scripted")
    if max_rounds is not None:
        argv += ["--max-rounds", str(int(max_rounds))]
    return argv


def build_execute_argv(
    run_dir: Path, *, mutating: bool, approval_timeout: float
) -> list[str]:
    """`go <dir>`, parked on the file handshake exactly when the plan mutates.

    The tiering is the maintainer's decision made mechanical: an all-read-only
    plan executes on the host's own prompt; anything that can change files
    parks for an out-of-band human. The verdict comes from the plan record,
    where a missing field already read as mutating.
    """
    argv = ["go", str(run_dir), "--json"]
    if mutating:
        argv += ["--approve", "--approval-timeout", str(float(approval_timeout))]
    return argv


__all__ = [
    "DEFAULT_APPROVAL_TIMEOUT",
    "PLAN_FILENAME",
    "DriverError",
    "build_execute_argv",
    "build_plan_argv",
    "confine_run_dir",
    "graph_status",
    "parse_document",
    "plan_is_mutating",
    "read_plan_record",
    "run_cli",
]
