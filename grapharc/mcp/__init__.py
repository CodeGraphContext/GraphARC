"""GraphARC as an MCP server — supervision for agents that already exist.

Ships behind the `mcp` extra (`pip install 'grapharc[mcp]'`). The package
imports nothing from `grapharc.server`, which needs the `server` extra: the
read-side primitives here are the fastapi-free ones (`observe`, the approval
file handshake, `plan.json`).

`grapharc mcp` is the entry point; `build_server` is the library surface a
test drives directly.
"""

from grapharc.mcp.driver import DriverError
from grapharc.mcp.server import FORBIDDEN_TOOL_WORDS, build_server, serve_stdio

__all__ = ["FORBIDDEN_TOOL_WORDS", "DriverError", "build_server", "serve_stdio"]
