"""`grapharc approve` — answer a plan run parked on its approval gate.

The paused run (a `grapharc plan --approve`) wrote `approval-request.json`
next to its trace and is polling for `approval-decision.json`. This command
reads the request, writes the decision quoting the request's fingerprint — the
run ignores a decision naming any other plan — and exits. Exit codes follow
the CLI contract: 0 the decision was delivered, 1 nothing is waiting for one,
2 the path does not lead anywhere a request could be.
"""

from __future__ import annotations

from pathlib import Path

from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.planner.approval_file import read_request, write_decision


def approve(path: str | Path, *, deny: bool = False, as_json: bool = False) -> int:
    """Deliver a decision to the approval request in `path`'s directory.

    `path` may be the trace file the run printed, or the directory holding it —
    both name the same handshake directory.
    """
    target = Path(path)
    directory = target if target.is_dir() else target.parent
    if not directory.is_dir():
        return fail(
            f"no such directory: {directory}", as_json=as_json, command="approve"
        )

    request = read_request(directory)
    if request is None:
        return fail(
            f"nothing is waiting for approval in {directory}",
            as_json=as_json,
            command="approve",
            code=EXIT_FAILED,
        )

    decision = "denied" if deny else "approved"
    fingerprint = str(request.get("fingerprint", ""))
    write_decision(directory, fingerprint=fingerprint, decision=decision)
    payload = {
        "ok": True,
        "command": "approve",
        "decision": decision,
        "fingerprint": fingerprint,
        "proposal_id": request.get("proposal_id", ""),
        "nodes": request.get("nodes", []),
    }
    nodes = ", ".join(str(n) for n in request.get("nodes", [])) or "(none)"
    lines = [
        f"{decision}: plan {request.get('proposal_id', '?')} ({nodes})",
        f"fingerprint {fingerprint}",
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK


__all__ = ["approve"]
