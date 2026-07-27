"""Hooks: a deterministic control plane around tool calls.

Pre-hooks run before a permitted tool executes and can deny the call or
rewrite its arguments; post-hooks can transform the result. Hooks are code,
not prompt text — they fire every time, whatever the model says.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class HookAction(StrEnum):
    DENY = "deny"
    REWRITE = "rewrite"


class HookDecision(BaseModel):
    action: HookAction
    args: dict[str, Any] | None = None  # for REWRITE
    reason: str = ""


# Return None for "no opinion"; the first non-None decision wins.
PreHook = Callable[[str, dict[str, Any]], HookDecision | None]
# (tool_name, args, result) -> transformed result
PostHook = Callable[[str, dict[str, Any], Any], Any]
