"""Run metrics derived from traces.

Nothing here is a new source of truth — every number is computed from the
JSONL trace, so the metrics and the audit trail can never disagree.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from grapharc.observe.trace import TraceRecorder


class RunMetrics(BaseModel):
    run_id: str
    graph: str
    nodes_executed: int
    errors: int
    tokens: int
    duration_ms: float
    attempts: int
    termination_reason: str | None = None
    per_node: dict[str, int] = {}


def summarize(recorder: TraceRecorder, run_id: str) -> RunMetrics | None:
    events = recorder.read_events(run_id)
    if not events:
        return None
    ends = [e for e in events if e.phase == "end"]
    errors = [e for e in events if e.phase == "error"]
    reason = None
    for e in reversed(ends):
        delta = e.state_delta or {}
        if "termination_reason" in delta:
            reason = delta["termination_reason"]
            break
    return RunMetrics(
        run_id=run_id,
        graph=events[0].graph,
        nodes_executed=len(ends),
        errors=len(errors),
        tokens=sum(e.tokens or 0 for e in ends),
        duration_ms=round(sum(e.duration_ms or 0.0 for e in ends), 2),
        attempts=max((e.attempt for e in events), default=1),
        termination_reason=reason,
        per_node=dict(Counter(e.node for e in ends)),
    )


def to_mermaid(recorder: TraceRecorder, run_id: str) -> str:
    """Render the executed path as a Mermaid flowchart (file-first, git-friendly)."""
    events = [e for e in recorder.read_events(run_id) if e.phase in ("end", "error")]
    if not events:
        return "flowchart TD\n  empty[no events]"
    lines = ["flowchart TD"]
    seen: dict[str, str] = {}
    for i, e in enumerate(events):
        node_id = seen.setdefault(e.node, f"n{len(seen)}")
        if i == 0:
            lines.append(f"  start((start)) --> {node_id}[{e.node}]")
        if e.phase == "error":
            lines.append(f"  {node_id} -.->|error| err{i}{{{e.error or 'error'}}}"[:200])
    for a, b in zip(events, events[1:], strict=False):
        if a.phase == "end":
            lines.append(f"  {seen[a.node]}[{a.node}] --> {seen[b.node]}[{b.node}]")
    return "\n".join(dict.fromkeys(lines))
