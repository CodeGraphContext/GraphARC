"""Runtime-layer errors the HTTP layer turns into status codes.

They live here rather than in `app.py` so a `SessionRuntime` implementation can
raise them without importing FastAPI.
"""

from __future__ import annotations

from collections.abc import Iterable


class ServerError(Exception):
    """Base for errors `create_app` knows how to map onto a response."""


class UnknownSessionError(ServerError):
    """No session with this id — mapped to 404."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"unknown session {session_id!r}")
        self.session_id = session_id


class UnknownGraphError(ServerError):
    """The requested graph name is not registered — mapped to 404.

    404 rather than 422: the body is well-formed, it names a resource the
    server does not have. The message lists what is registered, because an
    empty registry is the most likely cause and is otherwise invisible.
    """

    def __init__(self, name: str, known: Iterable[str] = ()) -> None:
        registered = sorted(known)
        listing = ", ".join(repr(n) for n in registered) if registered else "none"
        super().__init__(f"unknown graph {name!r}; registered graphs: {listing}")
        self.name = name
        self.known = registered


class InvalidGraphInputError(ServerError):
    """`input` does not satisfy the graph's state schema — mapped to 422.

    Checked before the session starts, so a bad body is a rejected request
    rather than a session that fails a moment later. `errors` carries Pydantic's
    own complaints in FastAPI's `detail` shape.
    """

    def __init__(self, graph: str, errors: list[dict]) -> None:
        super().__init__(f"input does not satisfy the state schema of graph {graph!r}")
        self.graph = graph
        self.errors = errors


class RuntimeClosedError(ServerError):
    """A session was requested after the runtime shut down — mapped to 503."""


class CheckpointerNotAsyncError(ServerError):
    """A graph of `async def` nodes was compiled with a sync-only checkpointer.

    Not mapped to a status code: the runtime cannot know a saver has no async
    side until it asks one, and it asks on the run's own event loop, which
    exists only after the session has started. So this fails the session rather
    than the create — but with the two ways out spelled out, because the raw
    failure (`NotImplementedError` from inside the saver, with `event_count`
    still 0) says nothing about either.

    `InProcessRuntime` drives every other graph a sync-only saver appears in
    through `CompiledGraphARC.stream()`, which such a saver supports. Only
    `async def` nodes force `astream()`, and only then is the combination
    unserviceable.
    """

    def __init__(self, graph: str, reason: str, detail: str = "") -> None:
        super().__init__(
            f"graph {graph!r} needs astream() because it has async nodes, but its "
            f"checkpointer cannot be driven asynchronously ({reason}). Fix either "
            "side: give compile() a saver that implements the async API — "
            "langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver (needs aiosqlite, and "
            "must be constructed on a running loop) or "
            "langgraph.checkpoint.memory.InMemorySaver — or make the graph's nodes "
            "synchronous, which lets the server drive it through stream() and use "
            "the sync saver as it is."
            + (f" [{detail}]" if detail else "")
        )
        self.graph = graph
        self.reason = reason
        self.detail = detail


__all__ = [
    "CheckpointerNotAsyncError",
    "InvalidGraphInputError",
    "RuntimeClosedError",
    "ServerError",
    "UnknownGraphError",
    "UnknownSessionError",
]
