"""One rule for what a node's events say about it.

Three renderers derive a per-node status from the same trace — the Mermaid
overlay in `metrics`, the Slack narration in `slack.live`, and the structured
graph snapshot in `viewmodel` — and for a while each carried its own copy of
the counting. The copies agreed by luck, not by construction. This module is
the single statement of the rule they all share:

    errored > running > done > pending

Any error wins: a node that failed and was retried to success still shows the
failure, because "it failed at some point" is what an operator scanning a live
view needs to notice first. `running` means more starts than ends — under
fan-out several instances of one node may be open at once and one still-open
instance keeps the name running. `done` means at least one end. A declared
node with no events at all is `pending`, which is the whole point of drawing
the declared graph rather than the executed path.

Statuses are computed from `start`/`end`/`error` events only. Sub-step phases
(`model`, `tool`, `stop`, the planner's paperwork) describe work *inside* a
node, never the node's own lifecycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from grapharc.observe.trace import TraceEvent

NodeStatus = Literal["pending", "running", "done", "errored"]


class NodeState(BaseModel):
    """What one node's own events add up to."""

    node: str
    starts: int = 0
    ends: int = 0
    errors: int = 0
    # The most recent terminal events, kept whole: the Slack line wants the
    # last end's duration and tokens, the snapshot wants the last error's text.
    last_end: TraceEvent | None = None
    last_error: TraceEvent | None = None

    @property
    def status(self) -> NodeStatus:
        if self.errors:
            return "errored"
        if self.starts > self.ends:
            return "running"
        if self.ends:
            return "done"
        return "pending"


def node_states(events: Iterable[TraceEvent]) -> dict[str, NodeState]:
    """Fold events into per-node states, keyed by node name.

    The caller chooses the scope: pass one graph's events and statuses are that
    graph's own — a multi-round planner run must not let round 1's finished
    node wear round 2's still-running state.
    """
    states: dict[str, NodeState] = {}
    for event in events:
        if event.phase not in ("start", "end", "error"):
            continue
        state = states.get(event.node)
        if state is None:
            state = states[event.node] = NodeState(node=event.node)
        if event.phase == "start":
            state.starts += 1
        elif event.phase == "end":
            state.ends += 1
            state.last_end = event
        else:
            state.errors += 1
            state.last_error = event

    return states


__all__ = ["NodeState", "NodeStatus", "node_states"]
