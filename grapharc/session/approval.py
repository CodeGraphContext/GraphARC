"""Human approval as a suspension, not a prompt.

An approval request is not something a node asks for and then proceeds past.
It is a *held node*: the session runtime stops the graph before the gated node
executes, records what is waiting, and will not schedule that node again until
a decision exists. Nothing the gated node would have done has happened yet, so
a rejection costs nothing to honour.

Requests live in the session store, not in graph state: they are runtime
bookkeeping about a graph, and keeping them out of the checkpoint keeps the
checkpoint's payload plain JSON. The *decision* does travel in graph state —
the graph has to be able to route on it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: State field the session runtime writes a decision into. Nodes read it typed
#: via `SessionState.decision`.
DECISION_FIELD = "approval_decision"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ApprovalRequest(BaseModel):
    """A named node the session is holding, and why anyone should care."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    node: str
    action: str = ""
    requested_at: str = Field(default_factory=_now)


class ApprovalDecision(BaseModel):
    """A verdict on one request. `request_id` is what binds it to that request.

    Matching by id rather than by "the session is awaiting something" is what
    stops a decision made about an earlier hold from silently authorising a
    later one.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    approved: bool
    decided_by: str = ""
    reason: str = ""
    decided_at: str = Field(default_factory=_now)


def decision_in(values: Mapping[str, Any]) -> ApprovalDecision | None:
    """Read the decision out of a checkpointed state mapping, typed.

    The channel holds a plain dict so nothing but JSON ends up in the
    checkpoint (see `SessionState.approval_decision`). This is the one place
    that turns it back into a model — nodes reach it via `SessionState.decision`
    and callers holding raw checkpoint values call this directly.
    """
    raw = values.get(DECISION_FIELD)
    if raw is None:
        return None
    if isinstance(raw, ApprovalDecision):
        return raw
    return ApprovalDecision.model_validate(raw)


__all__ = ["DECISION_FIELD", "ApprovalDecision", "ApprovalRequest", "decision_in"]
