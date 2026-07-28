"""The core toolset — file, search and shell tools every real agent needs.

`ToolSpec` objects bound to one workspace, ready for a `ToolRegistry`:

    from grapharc.harness import Harness, LocalExecutor, ToolRegistry
    from grapharc.tools import register_core_tools

    registry = ToolRegistry()
    register_core_tools(registry, workspace)

Seven tools: `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`,
`run_command`. What they share is that every path argument is resolved and
confined to the workspace by the tool itself, independently of the executor —
`LocalExecutor` confines nothing, and a tool is the last thing standing between
a model-supplied `../../.ssh/id_rsa` and the file. See `workspace.py` for what
that check does and does not cover, and `shell.py` for why `run_command` is a
larger decision than the rest.
"""

from grapharc.tools.core import CORE_TOOL_NAMES, core_tools, register_core_tools
from grapharc.tools.files import AmbiguousEdit
from grapharc.tools.search import SKIPPED_DIRECTORIES
from grapharc.tools.workspace import ToolError, ToolLimits, Workspace, WorkspaceEscape

__all__ = [
    "CORE_TOOL_NAMES",
    "SKIPPED_DIRECTORIES",
    "AmbiguousEdit",
    "ToolError",
    "ToolLimits",
    "Workspace",
    "WorkspaceEscape",
    "core_tools",
    "register_core_tools",
]
