"""The harness: registry + permissions + hooks + executor, composed.

The security boundary lives here, in code. Prompts and skills shape what a
model *wants* to do; the harness decides what actually *happens*:

    call -> permission (deny/ask/allow) -> approval gate -> pre-hooks
         -> sandboxed executor -> post-hooks -> result
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from grapharc.harness.executor import SandboxedExecutor
from grapharc.harness.hooks import HookAction, PostHook, PreHook
from grapharc.harness.permissions import Decision, PermissionDenied, PermissionPolicy
from grapharc.harness.tools import ToolRegistry, ToolSpec

_log = logging.getLogger(__name__)

# (tool_name, args) -> approved? Human checkpoints implement this.
ApprovalCallback = Callable[[str, dict[str, Any]], bool]
# (tool_name) -> None. Fired when the policy denies a call, so a caller that
# records enforcement elsewhere (a policy document's audit log) hears about
# the denials this object otherwise answers silently.
DenialCallback = Callable[[str], None]


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
        on_denial: DenialCallback | None = None,
        workspace: str | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.executor = executor or SandboxedExecutor(workspace)
        # The directory this harness works in, kept on the harness itself.
        #
        # It used to exist only to build the default `SandboxedExecutor`, so
        # `Harness(..., executor=LocalExecutor(), workspace=str(ws))` — which
        # three call sites in this repo write — silently discarded `ws`. That
        # was invisible until something asked where the harness was working:
        # a delegated `AgentNode` reads the workspace to give Claude Code a
        # cwd, found only `SandboxedExecutor` had one, and refused to run at
        # all under `LocalExecutor`. Recording it here makes the argument mean
        # what it reads as, whichever executor is in play.
        self.workspace = workspace or getattr(self.executor, "workspace", None)
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks
        self.approval = approval
        self.on_denial = on_denial
        #: Denial-hook exceptions swallowed so far. Non-zero means enforcement
        #: records are missing somewhere downstream — see `_notify_denial`.
        self.denial_hook_errors = 0
        self._denial_lock = threading.Lock()

    def visible_tools(self) -> list[ToolSpec]:
        """The tool schemas a model may see — policy-filtered before exposure."""
        return self.registry.visible(self.policy)

    def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        spec = self.registry.get(tool_name)
        if spec is None:
            raise PermissionDenied(f"unknown tool {tool_name!r}")

        decision = self.policy.decide(tool_name)
        if decision is Decision.DENY:
            self._notify_denial(tool_name)
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

    def _notify_denial(self, tool_name: str) -> None:
        """Tell `on_denial` about a refused call, without fail.

        The hook runs before the `PermissionDenied` it annotates, and its
        exceptions are counted in `denial_hook_errors` and swallowed: a
        recorder must never turn a denied call into a tool error, which is
        what an exception here would become one frame up in `AgentNode`.
        Only the policy-DENY branch notifies — an approval refused by a
        human is the approval path's record to write, and an unknown tool
        is not a policy decision at all.
        """
        if self.on_denial is None:
            return
        try:
            self.on_denial(tool_name)
        except Exception:
            with self._denial_lock:
                self.denial_hook_errors += 1
            _log.exception(
                "denial hook raised for tool %r; the denial stands", tool_name
            )
