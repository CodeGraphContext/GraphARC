"""One guard, shared by every command that writes a trace under a chosen id.

`TraceRecorder` opens its file in append mode, and that is right: several runs
in one file is the pattern `grapharc diff` is built on. What append mode cannot
notice is that the id being appended under is already in the file. Two runs then
merge under one name, and every reader downstream presents the blend as a single
coherent run — `metrics` sums both runs' tokens and nodes, `viz` welds the second
path onto the end of the first, `replay` reconstructs a chimera. The trace is
documented as the one record everything else agrees with; a reused id is what
makes that one record lie.

So the file staying appendable is correct and the id being reused is the defect,
which puts the refusal here, at the start of a run, rather than in the recorder.
Only an id the operator actually typed is checked: a generated one is fresh by
construction and must not make every run pay for a file scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grapharc.cli.output import fail


def count_events(path: Path, run_id: str) -> int:
    """How many events in `path` already carry `run_id`.

    Reads the lines itself instead of going through `TraceRecorder.read_events`:
    this runs before a guarded command starts, so it reads one field per line
    rather than validating a whole event model, and a line it cannot parse is
    skipped rather than raised over. The guard's job is to spot a collision, not
    to be the file's validator — the readers still report a torn trace, and an
    unreadable file is a failure the run's own writer will report in its own
    words.
    """
    if not path.is_file():
        return 0
    seen = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("run_id") == run_id:
                    seen += 1
    except OSError:
        return 0
    return seen


def refuse_reused_run_id(
    trace_path: Path | str | None,
    run_id: str | None,
    *,
    command: str,
    as_json: bool,
    **extra: Any,
) -> int | None:
    """The exit code to return, or None when this run may go ahead.

    `run_id is None` means the caller will generate one, which cannot collide.
    """
    if run_id is None or trace_path is None:
        return None
    path = Path(trace_path)
    events = count_events(path, run_id)
    if not events:
        return None
    return fail(
        f"run id {run_id!r} already has {events} event{'' if events == 1 else 's'} in "
        f"{path}; pick a new --run-id or a new --trace file. Appending a second run "
        f"under one id merges the two, and metrics, replay and viz would then report "
        f"the blend as one run",
        as_json=as_json,
        command=command,
        run_id=run_id,
        trace=str(path),
        **extra,
    )


__all__ = ["count_events", "refuse_reused_run_id"]
