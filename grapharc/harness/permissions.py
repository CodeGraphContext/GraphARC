"""Tool permissions: deny → ask → allow, first match wins, fail closed.

Permissions gate *which* tools a model may call. They are enforced by the
harness, never by prompts — instructions are advisory, this is not. A broad
deny always beats a narrower allow (no allowlist exceptions inside a deny),
matching the semantics that survived contact with reality in Claude Code.
"""

from __future__ import annotations

from enum import StrEnum
from fnmatch import fnmatch

from pydantic import BaseModel


class Decision(StrEnum):
    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"


class PermissionDenied(Exception):
    """A tool call was refused by policy (or by an absent/negative approval)."""


class PermissionRule(BaseModel):
    action: Decision
    pattern: str  # fnmatch pattern over the tool name


class PermissionPolicy(BaseModel):
    """Rules evaluated by tier: every deny rule, then ask, then allow.

    The default for an unmatched tool is DENY — a tool nobody thought about
    is a tool that doesn't run.
    """

    rules: list[PermissionRule] = []
    default: Decision = Decision.DENY

    def decide(self, tool_name: str) -> Decision:
        for tier in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            for rule in self.rules:
                if rule.action == tier and fnmatch(tool_name, rule.pattern):
                    return tier
        return self.default
