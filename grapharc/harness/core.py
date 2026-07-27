"""The harness: registry + permissions + hooks + executor, composed.

The security boundary lives here, in code. Prompts and skills shape what a
model *wants* to do; the harness decides what actually *happens*:

    call -> permission (deny/ask/allow) -> approval gate -> pre-hooks
         -> sandboxed executor -> post-hooks -> result
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from grapharc.harness.executor import SandboxedExecutor
from grapharc.harness.hooks import HookAction, PostHook, PreHook
from grapharc.harness.permissions import Decision, PermissionDenied, PermissionPolicy
from grapharc.harness.tools import ToolRegistry, ToolSpec

# (tool_name, args) -> approved? Human checkpoints implement this.
ApprovalCallback = Callable[[str, dict[str, Any]], bool]


class Harness:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        *,
        executor: Any = None,
        pre_hooks: tuple[PreHook, ...] = (),
        post_hooks: tuple[PostHook, ...] = (),
        approval: ApprovalCallback | None = None,
        workspace: str | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.executor = executor or SandboxedExecutor(workspace)
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks
        self.approval = approval

    def visible_tools(self) -> list[ToolSpec]:
        """The tool schemas a model may see — policy-filtered before exposure."""
        return self.registry.visible(self.policy)

    def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        spec = self.registry.get(tool_name)
        if spec is None:
            raise PermissionDenied(f"unknown tool {tool_name!r}")

        decision = self.policy.decide(tool_name)
        if decision is Decision.DENY:
            raise PermissionDenied(f"tool {tool_name!r} denied by policy")
        if decision is Decision.ASK:
            # Fail closed: no approval channel means no approval.
            if self.approval is None or not self.approval(tool_name, dict(args)):
                raise PermissionDenied(
                    f"tool {tool_name!r} requires approval and none was granted"
                )

        for hook in self.pre_hooks:
            outcome = hook(tool_name, dict(args))
            if outcome is None:
                continue
            if outcome.action is HookAction.DENY:
                raise PermissionDenied(
                    f"tool {tool_name!r} blocked by hook: {outcome.reason}"
                )
            if outcome.action is HookAction.REWRITE and outcome.args is not None:
                args = outcome.args
            break

        result = self.executor.run(spec, args)

        for post in self.post_hooks:
            result = post(tool_name, dict(args), result)
        return result
