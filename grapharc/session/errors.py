"""Every refusal the session layer can make, as a distinct type.

The invariant that every rejection is recorded with a reason
starts with the refusals being nameable. A caller that cannot tell "someone
else is running this session" from "this session is finished" from "this
session is waiting on a human" cannot do anything sensible about either.
"""

from __future__ import annotations


class SessionError(Exception):
    """Base class for every refusal from the session layer."""


class UnknownSessionError(SessionError):
    """No session with that id exists in the store."""


class SessionExistsError(SessionError):
    """A session with that id is already recorded in this store."""


class ThreadInUseError(SessionError):
    """The checkpoint thread already belongs to another session in this store.

    A checkpoint thread belongs to exactly one session. A second session on the
    same thread would resume the first one's checkpointed boundary with a fresh
    record carrying no holds — and a gated node the first session is holding
    would run with no approval ever given.
    """

    def __init__(self, thread_id: str, session_id: str) -> None:
        super().__init__(
            f"thread {thread_id!r} already belongs to session {session_id!r}"
        )
        self.thread_id = thread_id
        self.session_id = session_id


class UnknownGraphError(SessionError):
    """The session names a graph this process has not registered.

    The common cause is a resume in a process that never imported the module
    holding the graph's `register_graph` call. Failing here is the point: the
    alternative is rehydrating a session against a *different* graph.
    """


class InvalidTransition(SessionError):
    """The status lifecycle has no edge from the current status to the requested one."""

    def __init__(self, session_id: str, current: str, requested: str) -> None:
        super().__init__(
            f"session {session_id!r}: {current} -> {requested} is not a legal transition"
        )
        self.session_id = session_id
        self.current = current
        self.requested = requested


class SessionBusy(SessionError):
    """A compare-and-set on session status lost: someone else moved it first.

    `actual` is the status found instead, so a caller can decide whether it was
    beaten to the claim or the session was terminated out from under it.
    """

    def __init__(self, session_id: str, expected: tuple[str, ...], actual: str) -> None:
        super().__init__(
            f"session {session_id!r} is {actual}; expected one of {list(expected)}"
        )
        self.session_id = session_id
        self.expected = expected
        self.actual = actual


class SessionTerminated(SessionError):
    """The session is finished. Terminated is terminal — there is no way back."""


class ApprovalRequired(SessionError):
    """The session is holding a node and no decision for that request has arrived.

    Raised instead of resuming, because resuming is exactly what must not
    happen: the held node has not run, and running it is the thing a human was
    asked about.
    """

    def __init__(self, session_id: str, node: str, request_id: str) -> None:
        super().__init__(
            f"session {session_id!r} is holding node {node!r} awaiting approval "
            f"{request_id!r}; call decide() before run()"
        )
        self.session_id = session_id
        self.node = node
        self.request_id = request_id


class SessionContractError(SessionError):
    """A graph does not meet the contract a session needs to drive it."""


__all__ = [
    "ApprovalRequired",
    "InvalidTransition",
    "SessionBusy",
    "SessionContractError",
    "SessionError",
    "SessionExistsError",
    "SessionTerminated",
    "ThreadInUseError",
    "UnknownGraphError",
    "UnknownSessionError",
]
