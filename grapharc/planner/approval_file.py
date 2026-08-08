"""A file-handshake approval gate: ask on disk, answer with `grapharc approve`.

The governed loop's `approval` callback may be anything; this is the one the
CLI wires when `--approve` is passed, chosen because it needs no server, no
socket and no state in any bot: the paused run writes a request file next to
its trace, and *any* process that can reach that directory — a terminal, the
Slack bot running `grapharc approve` — answers by writing the decision file.
Both files are transient (consumed on read); the durable record is the pair of
`approval_request` / `approval_response` events the loop writes to the trace.

The decision must quote the request's fingerprint. That is what makes "approve
what was shown" enforceable: a decision file left over from an earlier round —
or written against a plan that has since been re-proposed — names the wrong
fingerprint and is discarded rather than trusted.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REQUEST_FILENAME = "approval-request.json"
DECISION_FILENAME = "approval-decision.json"

DEFAULT_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5


def _write_atomically(path: Path, payload: dict[str, Any]) -> None:
    """tmp + rename: a reader must never see half a handshake file.

    `Path.write_text` truncates then writes — a `grapharc approve` that reads
    in that window sees torn JSON, reports "nothing is waiting", and a
    one-shot caller loses its chance to answer.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_request(directory: Path, request: dict[str, Any]) -> Path:
    path = directory / REQUEST_FILENAME
    _write_atomically(path, request)
    return path


def read_request(directory: Path) -> dict[str, Any] | None:
    """The pending request, or None when nothing is waiting."""
    path = directory / REQUEST_FILENAME
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def describe_request(request: dict[str, Any], directory: Path) -> str:
    """The parked question as something a human can act on without prior knowledge.

    A run that parks and then says nothing but "waiting" depends on the reader
    already knowing that `grapharc approve` exists, that `--deny` is the other
    half, and that the wait expires. That is tribal knowledge, and the failure
    mode is a run timing out unapproved while its operator watches a cursor —
    which then reads as a planner failure rather than an unanswered question
    (issue #52). So the block names the plan, its fingerprint, the deadline,
    and both commands that end the wait.

    Deliberately plain text: this module is the headless half of the handshake
    and must stay importable without the CLI's styling. The caller decorates.
    """
    nodes = [str(node) for node in request.get("nodes", ())]
    shape = f"{len(nodes)} node{'s' if len(nodes) != 1 else ''}"
    if nodes:
        shape += ": " + ", ".join(nodes)
    lines = [
        f"awaiting approval — {shape}",
        f"  fingerprint : {request.get('fingerprint', '?')}",
        f"  waiting     : up to {float(request.get('timeout_seconds') or 0):.0f}s,"
        " then the round counts as unapproved",
        f"  approve     : grapharc approve {directory}",
        f"  refuse      : grapharc approve {directory} --deny",
        f"  look first  : grapharc approve {directory} --show",
    ]
    return "\n".join(lines)


def write_decision(directory: Path, *, fingerprint: str, decision: str) -> Path:
    path = directory / DECISION_FILENAME
    _write_atomically(path, {"fingerprint": fingerprint, "decision": decision})
    return path


def file_approval(
    directory: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = POLL_SECONDS,
    announce: Callable[[str], None] | None = None,
) -> Callable[[Any, Any], str]:
    """The loop-shaped gate: blocks until a decision, a timeout, or a halt.

    Returns "approved", "denied" or "timeout" — the vocabulary
    `GovernedLoop._request_approval` expects.
    """

    def gate(
        proposal: Any,
        verdict: Any,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        request = {
            "proposal_id": proposal.proposal_id,
            "fingerprint": proposal.fingerprint(),
            "nodes": [n.name for n in proposal.nodes],
            "edges": [[e.source, e.target] for e in proposal.edges],
            "rationale": proposal.rationale,
            "timeout_seconds": timeout_seconds,
        }
        request_path = directory / REQUEST_FILENAME
        decision_path = directory / DECISION_FILENAME
        deadline = time.monotonic() + timeout_seconds
        # Everything after the question is on disk sits under the finally —
        # including the announce: a broken stdout pipe must not leak a stale
        # request file for a later `grapharc approve` to "answer".
        try:
            # A matching decision that predates the question must never
            # answer it; a mismatched one is some other run's business and is
            # left alone — unlinking it once destroyed a neighbour's approval.
            _discard_own_stale_decision(decision_path, request["fingerprint"])
            write_request(directory, request)
            if announce is not None:
                try:
                    announce(describe_request(request, directory))
                except Exception:  # noqa: BLE001 — narration must not decide
                    pass
            while time.monotonic() < deadline:
                if should_stop is not None and should_stop():
                    return "timeout"  # the loop's halt check names the reason
                decision = _read_decision(decision_path)
                if decision is not None:
                    if decision.get("fingerprint") != request["fingerprint"]:
                        # Stale or another run's: ignore, never delete.
                        time.sleep(poll_seconds)
                        continue
                    return (
                        "approved"
                        if decision.get("decision") == "approved"
                        else "denied"
                    )
                time.sleep(poll_seconds)
            return "timeout"
        finally:
            request_path.unlink(missing_ok=True)
            _discard_own_stale_decision(decision_path, request["fingerprint"])

    return gate


def _read_decision(path: Path) -> dict[str, Any] | None:
    """The decision if a complete one is on disk; None on absent/torn/racing."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # Torn mid-write, or unlinked between our check and read: both mean
        # "not answered yet", never an error a gate should convert to denial.
        return None
    return loaded if isinstance(loaded, dict) else None


def _discard_own_stale_decision(path: Path, fingerprint: str) -> None:
    """Delete a decision naming *this* fingerprint; leave any other alone."""
    decision = _read_decision(path)
    if decision is not None and decision.get("fingerprint") == fingerprint:
        path.unlink(missing_ok=True)


__all__ = [
    "DECISION_FILENAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "REQUEST_FILENAME",
    "describe_request",
    "file_approval",
    "read_request",
    "write_decision",
    "write_request",
]
