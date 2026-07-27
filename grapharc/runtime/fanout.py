"""Bounded fan-out: parallel workers that cannot sink the batch.

Stage 3 discipline: cap concurrency, bound each worker's wall-clock time,
convert worker failures into data instead of run-killing exceptions, and
deduplicate evidence before synthesis so duplicates don't inflate confidence.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel


class WorkerResult(BaseModel):
    """One worker's outcome, ok or not — failures are data, not exceptions."""

    worker: str
    ok: bool
    evidence: list[dict[str, Any]] = []
    error: str | None = None


def run_guarded(
    fn: Callable[[], list[dict[str, Any]]], *, worker: str, timeout_seconds: float | None
) -> WorkerResult:
    """Run one worker's payload function with failure isolation.

    A raised exception or a hang past `timeout_seconds` becomes a
    WorkerResult(ok=False) — the batch continues without it. The job runs on a
    daemon thread joined with a timeout: a hung worker is abandoned, and a
    result arriving after the deadline is ignored — late evidence does not
    change a published answer.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        # BaseException, not Exception: a worker killed by SystemExit or a
        # cancellation error is a failed worker, not a silent success.
        try:
            box["evidence"] = fn()
        except BaseException as exc:  # noqa: BLE001 — isolation is the point
            box["error"] = repr(exc)

    thread = threading.Thread(target=target, daemon=True, name=f"grapharc-{worker}")
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return WorkerResult(worker=worker, ok=False, error=f"timeout after {timeout_seconds}s")
    if "error" in box:
        return WorkerResult(worker=worker, ok=False, error=box["error"])
    if "evidence" not in box:
        return WorkerResult(
            worker=worker, ok=False, error="worker terminated without producing a result"
        )
    # Classify inside the boundary: a malformed return must not raise in the
    # caller, or one bad worker takes the batch down with it.
    evidence = box["evidence"]
    if not isinstance(evidence, list):
        return WorkerResult(
            worker=worker,
            ok=False,
            error=f"worker returned {type(evidence).__name__}, expected list",
        )
    return WorkerResult(worker=worker, ok=True, evidence=evidence)


def dedupe[T](items: Iterable[T], key: Callable[[T], Any]) -> list[T]:
    """Order-preserving dedup by key — run before synthesis, never after."""
    seen: set[Any] = set()
    out: list[T] = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out
