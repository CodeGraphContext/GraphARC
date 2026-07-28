"""The state contract between a graph and the session that drives it.

A session persists nothing about a graph's own fields — those live in the
kernel's checkpoint. What it does need is two channels it owns:

- `inbox`, where queued messages are delivered. The runtime *appends*; the
  graph is responsible for consuming and clearing, because only the graph knows
  what "consumed" means for it.
- `approval_decision`, written by the runtime just before a held node is
  released, so the graph can route on the verdict.

Both are declared here rather than discovered by name, and a graph is bound to
a session only if its schema derives from `SessionState` — so a graph that
cannot receive either one is refused before a session record exists, not three
supersteps into a run.
"""

from __future__ import annotations

from typing import Any

from grapharc.runtime.state import GraphARCState
from grapharc.session.approval import DECISION_FIELD, ApprovalDecision, decision_in

#: Fields the session runtime writes. A session graph must declare all of them.
SESSION_FIELDS: tuple[str, ...] = ("inbox", DECISION_FIELD)


class SessionState(GraphARCState):
    """Base state schema for a graph a `Session` can drive."""

    inbox: list[str] = []

    approval_decision: dict[str, Any] | None = None
    """The last verdict, as a plain mapping. Read it through `decision`.

    Deliberately not typed as `ApprovalDecision`: the kernel validates state
    writes into their declared type, so a model-typed channel would put a
    Pydantic instance in the checkpoint, and LangGraph serializes those as a
    msgpack extension it already warns it will stop deserializing. A session's
    whole promise is that the checkpoint outlives the process that wrote it, so
    the channel holds JSON and the typing lives in the accessor.
    """

    @property
    def decision(self) -> ApprovalDecision | None:
        """`approval_decision`, validated — None when nothing has been decided."""
        return decision_in({DECISION_FIELD: self.approval_decision})


__all__ = ["SESSION_FIELDS", "SessionState"]
