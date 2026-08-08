"""GraphARC as an MCP server — supervision for agents that already exist.

Ships behind the `mcp` extra (`pip install 'grapharc[mcp]'`). The package
imports nothing from `grapharc.server`, which needs the `server` extra: the
read-side primitives here are the fastapi-free ones (`observe`, the approval
file handshake, `plan.json`).

`grapharc mcp` is the entry point; `build_server` is the library surface a
test drives directly.

**The extra is imported lazily**, the same posture `grapharc.slack` takes: only
`server.py` needs the MCP SDK, `driver.py` needs nothing beyond grapharc's core
dependencies, and importing this package must not require either. Eagerly
importing `server` here made `grapharc.mcp` unimportable in any environment
without the extra — which is every environment the packaging check builds, so a
whole subpackage read as missing from the wheel it was actually in.

`build_server` and `serve_stdio` therefore resolve on first attribute access
(PEP 562), and `from grapharc.mcp import build_server` still works.
"""

from typing import Any

from grapharc.mcp.driver import DriverError

__all__ = ["FORBIDDEN_TOOL_WORDS", "DriverError", "build_server", "serve_stdio"]

_DEFERRED = frozenset({"FORBIDDEN_TOOL_WORDS", "build_server", "serve_stdio"})


def __getattr__(name: str) -> Any:
    """Resolve the SDK-backed names on first use, not at import."""
    if name in _DEFERRED:
        from grapharc.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
