"""HTTP surface for GraphARC (ROADMAP §9.1).

```python
from grapharc.server import GraphRegistry, create_app, serve

registry = GraphRegistry().register("triage", build_triage)   # builder(trace) -> CompiledGraphARC
serve(create_app(registry=registry))
```

Requires the `server` extra (`fastapi`, `uvicorn`). Importing this package
imports FastAPI; the rest of GraphARC does not.
"""

from grapharc.server.app import create_app, serve
from grapharc.server.errors import (
    CheckpointerNotAsyncError,
    InvalidGraphInputError,
    RuntimeClosedError,
    ServerError,
    UnknownGraphError,
    UnknownSessionError,
)
from grapharc.server.models import (
    TERMINAL_STATUSES,
    BudgetSpec,
    CreateSessionRequest,
    EventAck,
    EventPage,
    QueuedEvent,
    SessionEventRequest,
    SessionStatus,
    SessionView,
)
from grapharc.server.registry import GraphBuilder, GraphRegistry
from grapharc.server.runtime import BroadcastRecorder, InProcessRuntime, SessionRuntime

__all__ = [
    "TERMINAL_STATUSES",
    "BroadcastRecorder",
    "BudgetSpec",
    "CheckpointerNotAsyncError",
    "CreateSessionRequest",
    "EventAck",
    "EventPage",
    "GraphBuilder",
    "GraphRegistry",
    "InProcessRuntime",
    "InvalidGraphInputError",
    "QueuedEvent",
    "RuntimeClosedError",
    "ServerError",
    "SessionEventRequest",
    "SessionRuntime",
    "SessionStatus",
    "SessionView",
    "UnknownGraphError",
    "UnknownSessionError",
    "create_app",
    "serve",
]
