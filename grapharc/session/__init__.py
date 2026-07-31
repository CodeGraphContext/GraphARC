"""Long-lived, resumable sessions over the GraphARC kernel.

    from grapharc.session import SessionManager
    from grapharc.session.demo import GRAPH_NAME   # registers the graph

    with SessionManager("/var/lib/grapharc") as manager:
        session = manager.create(GRAPH_NAME)
        turn = session.run({"received": []})
        # ... another process, later ...
        session = manager.resume(session.id)
        session.decide(approved=True, decided_by="ops")
        session.run()

`runtime.py` documents what each mechanic does and does not guarantee.
"""

from grapharc.session.approval import ApprovalDecision, ApprovalRequest, decision_in
from grapharc.session.errors import (
    ApprovalRequired,
    InvalidTransition,
    SessionBusy,
    SessionContractError,
    SessionError,
    SessionExistsError,
    SessionTerminated,
    ThreadInUseError,
    UnknownGraphError,
    UnknownSessionError,
)
from grapharc.session.events import EventKind, SessionEvent
from grapharc.session.registry import (
    REGISTRY,
    GraphRegistry,
    GraphSpec,
    get_graph_spec,
    register_graph,
)
from grapharc.session.runtime import Session, SessionManager, TurnResult
from grapharc.session.state import SESSION_FIELDS, SessionState
from grapharc.session.store import (
    RESUMABLE,
    SessionRecord,
    SessionStatus,
    SessionStore,
    StatusChange,
)

__all__ = [
    "REGISTRY",
    "RESUMABLE",
    "SESSION_FIELDS",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalRequired",
    "EventKind",
    "GraphRegistry",
    "GraphSpec",
    "InvalidTransition",
    "Session",
    "SessionBusy",
    "SessionContractError",
    "SessionError",
    "SessionEvent",
    "SessionExistsError",
    "SessionManager",
    "SessionRecord",
    "SessionState",
    "SessionStatus",
    "SessionStore",
    "SessionTerminated",
    "StatusChange",
    "ThreadInUseError",
    "TurnResult",
    "UnknownGraphError",
    "UnknownSessionError",
    "decision_in",
    "get_graph_spec",
    "register_graph",
]
