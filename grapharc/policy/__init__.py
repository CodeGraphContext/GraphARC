"""Declarative policy over nodes, edges, tools and spend (ROADMAP §7).

Permissions elsewhere in the repo are Python objects built in code. This
package is the declarative layer above them: a TOML document states the rules,
a `PolicyEngine` answers decisions against it, and every decision lands in an
audit log naming the rule and the policy digest that produced it.

    from grapharc.policy import PolicyEngine

    engine = PolicyEngine.from_file("policy.toml")
    decision = engine.check_tool("deploy_prod", tenant="acme")
    if decision.requires_approval:
        ask(decision.approver_role)

It composes with the harness rather than replacing it:
`engine.permission_policy(tenant=...)` produces a `PermissionPolicy` a
`Harness` already obeys, and `engine.approval_router(handlers, tenant=...)`
produces the approval callback that goes with it.

For the planner plane the pair is `engine.node_policy(tenant=...)` and
`engine.edge_policy(tenant=...)`, which produce the two objects
`grapharc.planner.admission.AdmissionChecker` consults — so the document
decides which kinds may run and which transitions may be wired, rather than
only answering questions about them.

A shipped, commented example document lives at `grapharc/policy/example.toml`.
"""

from grapharc.policy.approvals import (
    ApprovalHandler,
    ApprovalRecord,
    ApprovalRequest,
    ApprovalRouter,
)
from grapharc.policy.audit import AuditLog, AuditRecord
from grapharc.policy.document import (
    DEFAULT_TENANT,
    GLOBAL_TENANT,
    PolicyDocument,
    PolicyError,
    PolicyRule,
    ResourceKind,
    SpendBasis,
    load_document,
    parse_document,
)
from grapharc.policy.engine import PolicyDecision, PolicyEngine

__all__ = [
    "DEFAULT_TENANT",
    "GLOBAL_TENANT",
    "ApprovalHandler",
    "ApprovalRecord",
    "ApprovalRequest",
    "ApprovalRouter",
    "AuditLog",
    "AuditRecord",
    "PolicyDecision",
    "PolicyDocument",
    "PolicyEngine",
    "PolicyError",
    "PolicyRule",
    "ResourceKind",
    "SpendBasis",
    "load_document",
    "parse_document",
]
