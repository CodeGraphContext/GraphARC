"""`grapharc replay` and `grapharc diff` — get at a run after it finished.

Both delegate to `grapharc.observe.replay`, which owns the reconstruction and
the comparison; this module finds it, calls it, and renders what comes back. It
is imported at call time so the rest of the CLI keeps working in a checkout
without it, and a missing engine is reported as a message with an exit code
rather than an ImportError traceback.

Nothing here re-implements a diff. A command that compared traces by hand while
the engine was unavailable would answer a question the engine answers
differently, and two answers to "did these runs differ" is worse than one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grapharc.cli import optional
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail, jsonable
from grapharc.observe.trace import TraceRecorder

REPLAY_MODULE = "grapharc.observe.replay"
REPLAY_ENTRY_POINTS = ("replay", "replay_run")
# `diff_trace` takes a trace and two run ids. `diff_runs` takes two already
# reconstructed runs, so it is deliberately not a candidate here.
DIFF_ENTRY_POINTS = ("diff_trace",)

REPLAY_HINT = "Replay (ROADMAP §10.1) is not in this checkout."


def _render(result: Any, formatter: Any, fallback: str) -> list[str]:
    """Text view of an engine result, preferring the engine's own formatter."""
    if formatter is not None:
        try:
            return str(formatter(result)).splitlines()
        except Exception:  # noqa: BLE001 — a formatter fault must not lose the result
            pass
    summary = getattr(result, "summary", None)
    if callable(summary):
        return [fallback, str(summary())]
    return [fallback, str(jsonable(result))]


def replay(path: Path, run_id: str, *, as_json: bool = False) -> int:
    if not path.exists():
        return fail(f"no such trace file: {path}", as_json=as_json, command="replay")
    try:
        module = optional.load(REPLAY_MODULE, needed_for="grapharc replay", hint=REPLAY_HINT)
        name, fn = optional.pick(module, REPLAY_ENTRY_POINTS, needed_for="grapharc replay")
    except optional.Unavailable as exc:
        return fail(str(exc), as_json=as_json, command="replay")

    try:
        result = fn(TraceRecorder(path), run_id)
    except Exception as exc:  # noqa: BLE001 — a failed replay is an outcome, not a crash
        return fail(
            f"replay failed: {exc}",
            as_json=as_json,
            command="replay",
            code=EXIT_FAILED,
            path=str(path),
            run_id=run_id,
        )

    via = f"{REPLAY_MODULE}.{name}"
    payload = {
        "ok": True,
        "command": "replay",
        "path": str(path),
        "run_id": run_id,
        "via": via,
        "result": jsonable(result),
    }
    emit(
        payload,
        _render(result, getattr(module, "format_replay", None), f"replayed {run_id} via {via}"),
        as_json=as_json,
    )
    return EXIT_OK


def diff(path: Path, run_a: str, run_b: str, *, as_json: bool = False) -> int:
    """Compare two runs. Exits 1 when they differ, following `diff(1)`."""
    if not path.exists():
        return fail(f"no such trace file: {path}", as_json=as_json, command="diff")
    try:
        module = optional.load(REPLAY_MODULE, needed_for="grapharc diff", hint=REPLAY_HINT)
        name, fn = optional.pick(module, DIFF_ENTRY_POINTS, needed_for="grapharc diff")
    except optional.Unavailable as exc:
        return fail(str(exc), as_json=as_json, command="diff")

    try:
        result = fn(TraceRecorder(path), run_a, run_b)
    except Exception as exc:  # noqa: BLE001 — reported, not raised out of the CLI
        return fail(
            f"diff failed: {exc}",
            as_json=as_json,
            command="diff",
            code=EXIT_FAILED,
            path=str(path),
            run_a=run_a,
            run_b=run_b,
        )

    via = f"{REPLAY_MODULE}.{name}"
    # An engine that reports no verdict is not evidence the runs matched, so an
    # absent `identical` reads as "differs" and exits non-zero.
    identical = bool(getattr(result, "identical", False))
    payload = {
        "ok": True,
        "command": "diff",
        "path": str(path),
        "run_a": run_a,
        "run_b": run_b,
        "via": via,
        "identical": identical,
        "result": jsonable(result),
    }
    emit(
        payload,
        _render(result, getattr(module, "format_diff", None), f"diff {run_a} -> {run_b}"),
        as_json=as_json,
    )
    return EXIT_OK if identical else EXIT_FAILED


__all__ = ["diff", "replay"]
