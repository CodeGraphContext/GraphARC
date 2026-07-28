"""Approval routing: an `ask` decision goes to the role the rule named.

A rule that requires approval also says *who* approves it (`approver_role`),
which is the half `PermissionPolicy` has no room for — it can return ASK but
cannot say to whom. `ApprovalRouter` closes that: it re-asks the engine, reads
the role off the decision, and hands the request to the handler registered for
that role.

Every path fails closed, and every path is recorded:

- no handler for the role                → denied
- a handler that raises                  → denied (a broken channel is not consent)
- an ASK decision carrying no role       → denied
- a DENY decision                        → denied, and no handler is ever called

The router is also callable with the `(tool_name, args) -> bool` shape
`grapharc.harness.core.ApprovalCallback` expects, so it drops straight into a
`Harness`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from grapharc.harness.permissions import Decision
from grapharc.policy.audit import AuditRecord
from grapharc.policy.document import DEFAULT_TENANT, ResourceKind

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from grapharc.policy.engine import PolicyDecision, PolicyEngine


class ApprovalRequest(BaseModel):
    """What an approver is shown. Carries the rule so a human can see the why."""

    model_config = ConfigDict(extra="forbid")

    resource: ResourceKind
    subject: str
    tenant: str
    approver_role: str
    rule_id: str | None
    reason: str
    policy_version: str
    args: dict[str, Any] = {}


# Return True to approve. Anything else — False, or an exception — is a refusal.
ApprovalHandler = Callable[[ApprovalRequest], bool]


class ApprovalRecord(AuditRecord):
    """The outcome of routing one approval, appended to the same audit log."""

    kind: Literal["approval"] = "approval"
    resource: ResourceKind
    subject: str
    tenant: str
    approver_role: str | None
    rule_id: str | None
    granted: bool
    reason: str
    policy_version: str
    policy_digest: str


class ApprovalRouter:
    """Routes ASK decisions to per-role handlers. Bound to one tenant."""

    def __init__(
        self,
        engine: PolicyEngine,
        handlers: Mapping[str, ApprovalHandler],
        *,
        tenant: str = DEFAULT_TENANT,
    ) -> None:
        self._engine = engine
        self._handlers = dict(handlers)
        self.tenant = tenant

    @property
    def roles(self) -> list[str]:
        return sorted(self._handlers)

    def route(
        self, decision: PolicyDecision, *, args: Mapping[str, Any] | None = None
    ) -> ApprovalRecord:
        """Resolve `decision` to a granted/refused record, asking a human if needed."""
        granted, reason = self._resolve(decision, dict(args or {}))
        record = ApprovalRecord(
            resource=decision.resource,
            subject=decision.subject,
            tenant=decision.tenant,
            approver_role=decision.approver_role,
            rule_id=decision.rule_id,
            granted=granted,
            reason=reason,
            policy_version=decision.policy_version,
            policy_digest=decision.policy_digest,
        )
        self._engine.audit.record(record)
        return record

    def _resolve(self, decision: PolicyDecision, args: dict[str, Any]) -> tuple[bool, str]:
        if decision.effect is Decision.DENY:
            return False, f"denied by policy: {decision.reason}"
        if decision.effect is Decision.ALLOW:
            return True, f"allowed by policy: {decision.reason}"

        role = decision.approver_role
        if not role:
            return False, "approval required but the decision names no approver role"
        handler = self._handlers.get(role)
        if handler is None:
            return False, f"no approval handler registered for role {role!r}"

        request = ApprovalRequest(
            resource=decision.resource,
            subject=decision.subject,
            tenant=decision.tenant,
            approver_role=role,
            rule_id=decision.rule_id,
            reason=decision.reason,
            policy_version=decision.policy_version,
            args=args,
        )
        try:
            granted = bool(handler(request))
        except Exception as exc:  # a broken approval channel is not an approval
            return False, f"approval handler for role {role!r} failed: {exc!r}"
        verb = "granted" if granted else "refused"
        return granted, f"{role} {verb} approval"

    def __call__(self, tool_name: str, args: dict[str, Any]) -> bool:
        """`grapharc.harness.core.ApprovalCallback` shape, for use with `Harness`."""
        decision = self._engine.check_tool(tool_name, tenant=self.tenant)
        return self.route(decision, args=args).granted
