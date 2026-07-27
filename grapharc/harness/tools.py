"""Tool registry with policy-before-schema filtering.

A tool declares its capabilities up front (does it need the network? may it
write inside its workspace?). The registry can render the *visible* tool set
for a given policy — denied tools' schemas are never shown to a model, so the
model cannot even ask for what it may not have.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from grapharc.harness.permissions import Decision, PermissionPolicy


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    fn: Callable[..., Any]
    needs_network: bool = False
    timeout_seconds: float = 30.0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolRegistry:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec
        return self

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def visible(self, policy: PermissionPolicy) -> list[ToolSpec]:
        """Policy-before-schema: only tools the policy could ever permit."""
        return [
            spec
            for name, spec in sorted(self._tools.items())
            if policy.decide(name) is not Decision.DENY
        ]
