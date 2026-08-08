"""`grapharc approve` — answer a plan run parked on its approval gate.

The paused run (a `grapharc plan --approve`) wrote `approval-request.json`
next to its trace and is polling for `approval-decision.json`. This command
reads the request, writes the decision quoting the request's fingerprint — the
run ignores a decision naming any other plan — and exits. Exit codes follow
the CLI contract: 0 the decision was delivered, 1 nothing is waiting for one,
2 the path does not lead anywhere a request could be.

Two flags exist because an approval nobody read is not an approval (issue #46):

- `--show` prints the parked plan and exits **without deciding**, so a
  reviewer can look before answering. It is the only mode that writes nothing.
- `--fingerprint FP` binds the decision to the plan the reviewer actually
  read. A parked run rewrites its request every round, so between a `--show`
  and the `approve` that follows it the plan may have been replaced; quoting
  the fingerprint turns that race into a refusal (exit 2, both fingerprints
  named) instead of a yes to a graph nobody saw.

The deciding path prints the plan *before* it writes the decision, so even the
one-shot form leaves a record of what was approved rather than reporting it
after the fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grapharc.cli.output import EXIT_FAILED, EXIT_OK, EXIT_UNAVAILABLE, emit, fail
from grapharc.planner.approval_file import read_request, write_decision


def _plan_lines(request: dict[str, Any]) -> list[str]:
    """The parked proposal as text: nodes, edges, rationale, fingerprint."""
    nodes = [str(node) for node in request.get("nodes", ())]
    lines = [
        f"plan        : {request.get('proposal_id', '?')}",
        f"fingerprint : {request.get('fingerprint', '?')}",
        f"nodes       : {', '.join(nodes) if nodes else '(none)'}",
    ]
    edges = request.get("edges") or ()
    for index, edge in enumerate(edges):
        try:
            source, target = edge[0], edge[1]
        except (TypeError, IndexError, KeyError):
            continue
        label = "edges       : " if index == 0 else "              "
        lines.append(f"{label}{source} -> {target}")
    rationale = str(request.get("rationale") or "").strip()
    if rationale:
        lines.append(f"rationale   : {' '.join(rationale.split())}")
    return lines


def approve(
    path: str | Path,
    *,
    deny: bool = False,
    show: bool = False,
    fingerprint: str | None = None,
    as_json: bool = False,
) -> int:
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

    parked = str(request.get("fingerprint", ""))
    plan_lines = _plan_lines(request)

    if show:
        # The one mode that decides nothing. Printing the fingerprint is what
        # makes the follow-up command bindable.
        emit(
            {
                "ok": True,
                "command": "approve",
                "decision": None,
                "fingerprint": parked,
                "proposal_id": request.get("proposal_id", ""),
                "nodes": request.get("nodes", []),
                "edges": request.get("edges", []),
                "rationale": request.get("rationale", ""),
            },
            [
                *plan_lines,
                "",
                f"approve : grapharc approve {directory} --fingerprint {parked}",
                f"refuse  : grapharc approve {directory} --fingerprint {parked} --deny",
            ],
            as_json=as_json,
        )
        return EXIT_OK

    if fingerprint is not None and fingerprint != parked:
        # Not a stale-file problem — `file_approval` already discards a decision
        # naming the wrong plan. This is the reviewer being told *why* nothing
        # was written: what they read is not what is waiting.
        return fail(
            "that is not the plan now waiting: you reviewed "
            f"{fingerprint}, the parked plan is {parked}. "
            f"Re-read it with `grapharc approve {directory} --show`.",
            as_json=as_json,
            command="approve",
            # Exit 2: by this CLI's contract that is "the request you named
            # is not there", which is exactly what a superseded plan is.
            code=EXIT_UNAVAILABLE,
        )

    decision = "denied" if deny else "approved"
    write_decision(directory, fingerprint=parked, decision=decision)
    payload = {
        "ok": True,
        "command": "approve",
        "decision": decision,
        "fingerprint": parked,
        "proposal_id": request.get("proposal_id", ""),
        "nodes": request.get("nodes", []),
    }
    emit(payload, [*plan_lines, "", f"decision    : {decision}"], as_json=as_json)
    return EXIT_OK


__all__ = ["approve"]
