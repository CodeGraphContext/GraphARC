"""The wire contract: request bodies, the session view, and event acks.

Every request model sets `extra="forbid"` so a misspelled field is a 422 rather
than a silently ignored instruction — the same reason `GraphARCState` forbids
extras.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from grapharc.runtime.budget import Budget

SessionStatus = Literal["queued", "running", "succeeded", "failed", "interrupted"]

#: Statuses after which no further trace event is recorded for a session.
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "interrupted"})

EventType = Literal["message", "approval", "interrupt"]


class BudgetSpec(BaseModel):
    """Per-session hard limits, mapped onto `grapharc.runtime.budget.Budget`."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_seconds: float | None = Field(default=None, gt=0)
    max_concurrency: int | None = Field(default=None, gt=0)

    def to_budget(self) -> Budget:
        return Budget(**self.model_dump(exclude_none=True))


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    budget: BudgetSpec | None = None


class SessionEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class QueuedEvent(BaseModel):
    """A posted event and what the runtime actually did with it.

    `applied` is the honest bit: an event the runtime recorded but could not
    deliver reports `applied=False` with `detail` saying why, rather than
    letting an HTTP 2xx imply the run was steered.
    """

    id: str
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
    received_at: str
    applied: bool
    detail: str


class SessionView(BaseModel):
    """A session as the API reports it."""

    id: str
    graph: str
    thread_id: str
    status: SessionStatus
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    run_id: str | None = None
    #: Final state for `succeeded`, last completed superstep for `interrupted`,
    #: None for `failed` and for a run that has not produced one yet.
    result: dict[str, Any] | None = None
    error: str | None = None
    #: Live meter reading while running; the final one afterwards.
    usage: dict[str, float] | None = None
    event_count: int = 0
    inbox: list[QueuedEvent] = Field(default_factory=list)


class EventAck(BaseModel):
    session_id: str
    event: QueuedEvent
    session_status: SessionStatus


class EventPage(BaseModel):
    """Trace events after a cursor, plus the status read in the same critical section.

    Reading both together is what makes the SSE loop safe to end: `terminal` is
    only true once no further event can be appended, so a stream that stops on
    it cannot have dropped one.
    """

    events: list[dict[str, Any]] = Field(default_factory=list)
    cursor: int
    status: SessionStatus
    terminal: bool


class SessionList(BaseModel):
    sessions: list[SessionView] = Field(default_factory=list)


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    graphs: list[str]


__all__ = [
    "TERMINAL_STATUSES",
    "BudgetSpec",
    "CreateSessionRequest",
    "EventAck",
    "EventPage",
    "EventType",
    "Health",
    "QueuedEvent",
    "SessionEventRequest",
    "SessionList",
    "SessionStatus",
    "SessionView",
]
