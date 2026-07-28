"""The core toolset: the seven tools an agent needs before it needs anything else.

`core_tools` builds them bound to one workspace; `register_core_tools` puts them
in a `ToolRegistry`. Nothing here decides whether a tool may *run* — that is the
permission policy's job, and it is a separate decision on purpose. Building the
toolset and authorising it are different acts:

    registry = ToolRegistry()
    register_core_tools(registry, workspace, exclude=["run_command"])
    policy = PermissionPolicy(rules=[
        PermissionRule(action=Decision.ALLOW, pattern="read_file"),
        PermissionRule(action=Decision.ALLOW, pattern="grep"),
        PermissionRule(action=Decision.ASK, pattern="write_file"),
    ])
    harness = Harness(registry, policy, executor=LocalExecutor(), workspace=str(ws))

`include`/`exclude` are the coarse knob — a name never registered can never be
called, however the policy is later edited — and the policy is the fine one.
Use both: excluding `run_command` from a toolset that has no business shelling
out is a stronger statement than denying it, because there is nothing left to
deny.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from grapharc.harness.tools import ToolRegistry, ToolSpec
from grapharc.tools.files import edit_file_tool, read_file_tool, write_file_tool
from grapharc.tools.search import glob_tool, grep_tool, list_dir_tool
from grapharc.tools.shell import run_command_tool
from grapharc.tools.workspace import ToolLimits, Workspace

_BUILDERS: dict[str, Callable[[Workspace, ToolLimits], ToolSpec]] = {
    "read_file": read_file_tool,
    "write_file": write_file_tool,
    "edit_file": edit_file_tool,
    "list_dir": list_dir_tool,
    "glob": glob_tool,
    "grep": grep_tool,
    "run_command": run_command_tool,
}

CORE_TOOL_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def _known(names: Iterable[str], argument: str) -> set[str]:
    """Validate a selection. An unrecognised name raises rather than being ignored.

    A typo in `exclude` is the failure that matters: silently ignoring
    `run_comand` leaves the caller believing the shell tool is off while it is
    registered and callable. Raising costs a typo; ignoring costs the property.
    """
    if isinstance(names, str):
        raise ValueError(
            f"{argument} must be a collection of tool names, not a single string "
            f"(a string iterates as characters); pass [{names!r}]"
        )
    requested = set(names)
    unknown = sorted(requested - set(CORE_TOOL_NAMES))
    if unknown:
        raise ValueError(
            f"{argument} names unknown core tools {unknown}; "
            f"the core toolset is {list(CORE_TOOL_NAMES)}"
        )
    return requested


def core_tools(
    workspace: str | os.PathLike[str],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    limits: ToolLimits | None = None,
) -> list[ToolSpec]:
    """Build the core toolset bound to `workspace`.

    Every returned tool confines every path argument to `workspace`, which must
    already exist. `include` defaults to all of `CORE_TOOL_NAMES`; `exclude` is
    applied after it, so passing both is a narrowing rather than a conflict.
    Unknown names in either raise `ValueError`.

    The order is `CORE_TOOL_NAMES` order, not the caller's, so two callers
    asking for the same set get byte-identical schema lists — the tool list goes
    into a model prompt, where ordering changes cache keys and outputs.
    """
    selected = set(CORE_TOOL_NAMES) if include is None else _known(include, "include")
    if exclude is not None:
        selected -= _known(exclude, "exclude")
    root = Workspace(workspace)
    bounds = limits or ToolLimits()
    return [_BUILDERS[name](root, bounds) for name in CORE_TOOL_NAMES if name in selected]


def register_core_tools(
    registry: ToolRegistry,
    workspace: str | os.PathLike[str],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    limits: ToolLimits | None = None,
) -> ToolRegistry:
    """Register the core toolset on `registry` and return it, for chaining.

    `ToolRegistry.register` refuses a duplicate name, so this cannot quietly
    replace a tool already registered under one of these names — the collision
    raises and the registry is left holding whichever tools were added before
    it. Register the core set first if you also register your own.
    """
    for spec in core_tools(workspace, include=include, exclude=exclude, limits=limits):
        registry.register(spec)
    return registry


__all__ = ["CORE_TOOL_NAMES", "core_tools", "register_core_tools"]
