"""The session's input channel: durable, ordered, and readable by any process.

A long-lived session is driven by things that happen while it is not running —
a follow-up message, a stop request, an approval decision. Holding those in
memory would make "interrupt the session" mean "interrupt it from the process
that happens to own the object". They are rows instead, so any process pointed
at the same store can queue one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventKind(StrEnum):
    """What the runtime does with a queued event when it next reaches a boundary."""

    MESSAGE = "message"
    """Turn input. Appended to the graph's `inbox` before work continues."""

    INTERRUPT = "interrupt"
    """Stop at the next node boundary. Consumed by the run that honours it."""

    DECISION = "decision"
    """An `ApprovalDecision` payload. Applied only to the request it names."""


class SessionEvent(BaseModel):
    """One queued input. `seq` is the store's ordering, monotonic per session."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    session_id: str
    kind: EventKind
    body: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds")
    )
    consumed_at: str | None = None


__all__ = ["EventKind", "SessionEvent"]
