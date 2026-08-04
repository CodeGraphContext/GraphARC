"""The policy engine: a loaded document, and the decisions it answers.

Four questions, one evaluation path:

    may this tool run?      engine.check_tool("deploy_prod", tenant="acme")
    may this edge be taken? engine.check_edge("plan", "deploy_prod", tenant="acme")
    is this spend allowed?  engine.check_spend(12.50, tenant="acme")
    may this node run?      engine.check_node("shell_worker", tenant="acme")

Each returns a `PolicyDecision` carrying the effect, the id of the rule that
caused it, the approver role when one is required, and the policy version and
digest it was decided under — and each is appended to the audit log on the way
out. Asking the engine always leaves a record; that is the point of the layer.

**Where this sits.** `grapharc.harness.permissions.PermissionPolicy` remains
the enforcement point for tool calls; this engine is a layer above it, not a
replacement. `permission_policy()` compiles the tool rules for one tenant
*down* to a `PermissionPolicy` the existing `Harness` already obeys — both tier
deny → ask → allow over the same ordered rules, and a test pins the two to the
same answer across a tool × tenant matrix.

That compiled object is also the one unrecorded path, deliberately: a `Harness`
holding it decides without consulting the engine, and so writes no audit
record. Pair it with `approval_router()`, which comes back through here.

`node_policy()` and `edge_policy()` do the same job for the *planner* plane,
compiling this document's `node` and `edge` rules into the two objects
`grapharc.planner.admission.AdmissionChecker` consults. Both halves are
compiled: a document's node rules used to reach nothing, so a `deny` rule over
a node kind was text rather than a rule (issue #66).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from grapharc.harness.permissions import Decision, PermissionPolicy, PermissionRule
from grapharc.policy.approvals import ApprovalHandler, ApprovalRouter
from grapharc.policy.audit import AuditLog, AuditRecord
from grapharc.policy.document import (
    DEFAULT_TENANT,
    EDGE_ARROW,
    PolicyDocument,
    PolicyRule,
    ResourceKind,
    SpendBasis,
    load_document,
    parse_document,
)


class PolicyDecision(AuditRecord):
    """One answer, and everything needed to defend it after the fact."""

    kind: Literal["decision"] = "decision"
    resource: ResourceKind
    subject: str
    tenant: str
    effect: Decision
    rule_id: str | None  # None when no rule matched and the document default applied
    reason: str
    approver_role: str | None = None
    policy_version: str
    policy_digest: str
    request: dict[str, Any] = {}  # what was asked (amounts, endpoints)
    context: dict[str, Any] = {}  # caller-supplied: run id, node, session

    @property
    def allowed(self) -> bool:
        return self.effect is Decision.ALLOW

    @property
    def denied(self) -> bool:
        return self.effect is Decision.DENY

    @property
    def requires_approval(self) -> bool:
        return self.effect is Decision.ASK


class PolicyEngine:
    """Evaluates a `PolicyDocument` and records every decision it makes.

    Thread-safe: the document is frozen, and the spend ledger and audit log are
    each guarded.
    """

    def __init__(self, document: PolicyDocument, *, audit: AuditLog | None = None) -> None:
        self._document = document
        self._digest = document.digest  # hashed once; it is stamped on every record
        self._by_resource = {kind: document.rules_for(kind) for kind in ResourceKind}
        self._audit = audit if audit is not None else AuditLog()
        self._lock = threading.Lock()
        self._committed: dict[str, float] = {}

    @classmethod
    def from_toml(cls, text: str, *, audit: AuditLog | None = None) -> PolicyEngine:
        return cls(parse_document(text), audit=audit)

    @classmethod
    def from_file(cls, path: str | Path, *, audit: AuditLog | None = None) -> PolicyEngine:
        return cls(load_document(path), audit=audit)

    @property
    def document(self) -> PolicyDocument:
        return self._document

    @property
    def version(self) -> str:
        return self._document.version

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def audit(self) -> AuditLog:
        return self._audit

    # -- decisions ---------------------------------------------------------

    def check_tool(
        self,
        name: str,
        *,
        tenant: str = DEFAULT_TENANT,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        return self._evaluate(ResourceKind.TOOL, name, tenant, context=context)

    def check_node(
        self,
        name: str,
        *,
        tenant: str = DEFAULT_TENANT,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        return self._evaluate(ResourceKind.NODE, name, tenant, context=context)

    def check_edge(
        self,
        source: str,
        target: str,
        *,
        tenant: str = DEFAULT_TENANT,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        return self._evaluate(
            ResourceKind.EDGE, f"{source}{EDGE_ARROW}{target}", tenant, context=context
        )

    def check_spend(
        self,
        amount_usd: float,
        *,
        tenant: str = DEFAULT_TENANT,
        subject: str = "*",
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Decide one spend of `amount_usd`.

        Checking is not committing. A `cumulative` rule compares its threshold
        against `committed_usd(tenant) + amount_usd`, and that committed total
        only moves when `record_spend` is called — so a caller that checks and
        never records will pass a cumulative cap forever.
        """
        if amount_usd < 0:
            raise ValueError("amount_usd must not be negative")
        return self._evaluate(
            ResourceKind.SPEND, subject, tenant, amount_usd=amount_usd, context=context
        )

    # -- spend ledger ------------------------------------------------------

    def record_spend(self, amount_usd: float, *, tenant: str = DEFAULT_TENANT) -> float:
        """Commit `amount_usd` against `tenant` and return the new total."""
        if amount_usd < 0:
            raise ValueError("amount_usd must not be negative")
        with self._lock:
            total = self._committed.get(tenant, 0.0) + amount_usd
            self._committed[tenant] = total
            return total

    def committed_usd(self, tenant: str = DEFAULT_TENANT) -> float:
        with self._lock:
            return self._committed.get(tenant, 0.0)

    # -- composition -------------------------------------------------------

    def permission_policy(self, *, tenant: str = DEFAULT_TENANT) -> PermissionPolicy:
        """Compile this document's tool rules for `tenant` into a `PermissionPolicy`.

        The result answers exactly as `check_tool` does for the same tenant —
        both tier deny → ask → allow over the same ordered rule list — so the
        existing `Harness` enforces the document without knowing it exists.

        What the compiled object cannot carry: the approver role, the rule id,
        and the audit record. A `Harness` holding one will ASK without knowing
        who to ask and will decide without leaving a policy record. Pair it with
        `approval_router()`, which asks the engine again and does both.
        """
        if not self._document.declares_tenant(tenant):
            # `check_tool` denies an undeclared tenant outright; the compiled
            # policy has to agree, or the two answers diverge.
            return PermissionPolicy(rules=[], default=Decision.DENY)
        rules = [
            PermissionRule(action=rule.effect, pattern=rule.match)
            for rule in self._by_resource[ResourceKind.TOOL]
            if rule.matches_tenant(tenant)
        ]
        return PermissionPolicy(rules=rules, default=self._document.default)

    def node_policy(self, *, tenant: str = DEFAULT_TENANT) -> Any:
        """Compile this document's `node` rules for `tenant` into a `NodePolicy`.

        The other half of what `edge_policy()` started. A document's `node`
        rules used to reach nothing at all: `AdmissionChecker` gated kinds on
        registry membership alone, so `no-shell-nodes` in a policy file was
        text, not a rule (issue #66). This is the object the checker consults,
        and it answers exactly as `check_node` does for the same tenant — a test
        pins the two together across a kind × tenant matrix.

        Same two losses as the other compiled objects: no approver role and no
        audit record, because `NodePolicy.decide` returns a bare `Decision`.
        Admission treats `ASK` as not-yet-permitted.

        **A document with no `node` rules compiles to a policy that denies every
        kind**, because an unmatched subject gets the document default and that
        default is usually `deny` — the same answer `check_node` gives. That is
        faithful rather than convenient, which is why the CLI applies this only
        to a document that declares at least one `node` rule: saying nothing
        about nodes is not the same statement as denying all of them, and a
        caller compiling this by hand should decide which it meant.
        """
        from grapharc.planner.admission import NodePolicy, NodeRule

        if not self._document.declares_tenant(tenant):
            # `check_node` denies an undeclared tenant outright; the compiled
            # policy has to agree, or the two answers diverge.
            return NodePolicy(rules=(), default=Decision.DENY)
        rules = [
            NodeRule(action=rule.effect, match=rule.match, reason=rule.reason)
            for rule in self._by_resource[ResourceKind.NODE]
            if rule.matches_tenant(tenant)
        ]
        return NodePolicy(rules=tuple(rules), default=self._document.default)

    def edge_policy(self, *, tenant: str = DEFAULT_TENANT) -> Any:
        """Compile this document's `edge` rules for `tenant` into an `EdgePolicy`.

        The counterpart of `permission_policy()` for the *planner* side: it
        turns the declarative document into the object `AdmissionChecker`
        actually consults, so a TOML file can constrain what a planner is
        allowed to wire instead of only answering questions about it. Without
        this, "declarative governance" stopped at the tool plane and every
        admission gate had its `EdgePolicy` hand-built in Python.

        The result answers exactly as `check_edge` does for the same tenant, and
        a test pins the two together across an edge × tenant matrix. It carries
        the same two losses `permission_policy()` does: no approver role and no
        audit record, because `EdgePolicy.decide` returns a bare `Decision`.
        Admission treats `ASK` as not-yet-permitted; route the approval through
        the session layer, then re-check.

        Kinds, not instance names. An `edge` rule's `match` is
        `"<source><arrow><target>"` and each side becomes an fnmatch pattern
        over a **registry kind**, which is what the checker resolves before
        deciding — so renaming an instance cannot walk around a denial.
        """
        # Imported here rather than at module scope: `policy` is the declarative
        # layer and may depend on the enforcement objects, but paying for the
        # planner package on every `import grapharc.policy` is not worth it.
        from grapharc.planner.admission import EdgePolicy, EdgeRule

        if not self._document.declares_tenant(tenant):
            # `check_edge` denies an undeclared tenant outright; the compiled
            # policy has to agree, or the two answers diverge.
            return EdgePolicy(rules=(), default=Decision.DENY)
        rules = []
        for rule in self._by_resource[ResourceKind.EDGE]:
            if not rule.matches_tenant(tenant):
                continue
            source, _, target = rule.match.partition(EDGE_ARROW)
            rules.append(
                EdgeRule(
                    action=rule.effect,
                    source=source.strip(),
                    target=target.strip(),
                    # Carried, as `node_policy()` carries it: the document's own
                    # words are what a refusal quotes and what a planner is told
                    # up front, and a rule that arrives without them leaves both
                    # saying only `edge_denied`.
                    reason=rule.reason,
                )
            )
        return EdgePolicy(rules=tuple(rules), default=self._document.default)

    def approval_router(
        self,
        handlers: Mapping[str, ApprovalHandler],
        *,
        tenant: str = DEFAULT_TENANT,
    ) -> ApprovalRouter:
        """An approval callback for `Harness`, routed by the rule's approver role.

        Bound to one tenant because the harness's callback signature carries
        none: a `Harness` is a single-tenant object.
        """
        return ApprovalRouter(self, handlers, tenant=tenant)

    # -- evaluation --------------------------------------------------------

    def _evaluate(
        self,
        resource: ResourceKind,
        subject: str,
        tenant: str,
        *,
        amount_usd: float | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        request: dict[str, Any] = {}
        if amount_usd is not None:
            request["amount_usd"] = amount_usd
            request["committed_usd"] = self.committed_usd(tenant)

        if not self._document.declares_tenant(tenant):
            # A tenant name arrives with the request, i.e. from data, so a bad
            # one is refused and recorded rather than raised. A bad tenant in a
            # *rule* comes from the document author and does fail loud, at load.
            return self._record(
                resource=resource,
                subject=subject,
                tenant=tenant,
                effect=Decision.DENY,
                rule=None,
                reason=(
                    f"tenant {tenant!r} is not declared by policy "
                    f"{self._document.version!r}; declared: {self._document.tenants!r}"
                ),
                request=request,
                context=context,
            )

        for tier in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            for rule in self._by_resource[resource]:
                if rule.effect is not tier:
                    continue
                if not rule.matches_tenant(tenant):
                    continue
                if not rule.matches_subject(subject):
                    continue
                if amount_usd is not None:
                    considered = self._amount_for(rule, tenant, amount_usd)
                    if not rule.matches_amount(considered):
                        continue
                    request["basis"] = rule.spend_basis.value
                    request["considered_usd"] = considered
                return self._record(
                    resource=resource,
                    subject=subject,
                    tenant=tenant,
                    effect=rule.effect,
                    rule=rule,
                    reason=rule.reason or self._default_reason(rule, resource, subject, tenant),
                    request=request,
                    context=context,
                )

        return self._record(
            resource=resource,
            subject=subject,
            tenant=tenant,
            effect=self._document.default,
            rule=None,
            reason=(
                f"no {resource.value} rule matched {subject!r} for tenant {tenant!r}; "
                f"document default is {self._document.default.value!r}"
            ),
            request=request,
            context=context,
        )

    def _amount_for(self, rule: PolicyRule, tenant: str, amount_usd: float) -> float:
        if rule.spend_basis is SpendBasis.CUMULATIVE:
            return self.committed_usd(tenant) + amount_usd
        return amount_usd

    @staticmethod
    def _default_reason(
        rule: PolicyRule, resource: ResourceKind, subject: str, tenant: str
    ) -> str:
        return (
            f"rule {rule.id!r} ({rule.effect.value}) matched {resource.value} "
            f"{subject!r} for tenant {tenant!r}"
        )

    def _record(
        self,
        *,
        resource: ResourceKind,
        subject: str,
        tenant: str,
        effect: Decision,
        rule: PolicyRule | None,
        reason: str,
        request: dict[str, Any],
        context: Mapping[str, Any] | None,
    ) -> PolicyDecision:
        decision = PolicyDecision(
            resource=resource,
            subject=subject,
            tenant=tenant,
            effect=effect,
            rule_id=rule.id if rule is not None else None,
            reason=reason,
            approver_role=rule.approver_role if rule is not None else None,
            policy_version=self._document.version,
            policy_digest=self._digest,
            request=dict(request),
            context=dict(context or {}),
        )
        self._audit.record(decision)
        return decision
