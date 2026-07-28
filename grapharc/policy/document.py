"""The policy document: TOML in, validated rules out.

**Why TOML.** The document is edited by whoever owns the policy, so loading it
must not cost a dependency: `tomllib` is in the standard library from 3.11 and
this project already requires 3.12. PyYAML is not installed here, and a config
format is a poor reason to add a runtime dependency. TOML's array-of-tables
(`[[rule]]`) also keeps rule order visible in the file, which matters because
order decides *which* rule of a tier is reported as the cause of a decision.

`tomllib` reads and cannot write, so a document never round-trips through this
module. Policies are authored by hand and versioned in git; nothing here
rewrites one.

A document declares rules over four resource kinds — `tool`, `node`, `edge`,
`spend` — each with an effect drawn from the harness's own three-valued
vocabulary (`deny` / `ask` / `allow`), so a document composes with
`PermissionPolicy` rather than inventing a parallel one.

Evaluation semantics, which the loader's validation exists to protect:

- **Tiered, not positional.** Every `deny` rule is considered before every
  `ask`, and every `ask` before every `allow`. A broad deny therefore beats a
  narrow allow, *including* a narrow allow scoped to one tenant. There is no
  way to write an exception inside a deny; that is deliberate, and it is the
  same rule `grapharc.harness.permissions` follows.
- **First match within a tier wins**, in document order, and that rule's id is
  what the audit record names.
- **Unmatched means the document default**, which is `deny` unless the document
  says otherwise. A resource kind with no rules at all denies everything.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from grapharc.harness.permissions import Decision

# The tenant a request carries when the caller names none.
DEFAULT_TENANT = "default"
# The tenant pattern a rule carries when it is not scoped to anyone.
GLOBAL_TENANT = "*"
EDGE_ARROW = "->"


class PolicyError(Exception):
    """A policy document is malformed, or declares something it cannot mean."""


class ResourceKind(StrEnum):
    TOOL = "tool"
    NODE = "node"
    EDGE = "edge"
    SPEND = "spend"


class SpendBasis(StrEnum):
    """What amount a spend rule's threshold is compared against."""

    CALL = "call"  # this one request
    CUMULATIVE = "cumulative"  # the tenant's committed spend plus this request


class PolicyRule(BaseModel):
    """One rule. Immutable, and validated hard enough that a typo cannot be silent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    resource: ResourceKind
    effect: Decision
    # fnmatch pattern over the subject. For an edge the pattern is
    # "<source>-><target>" and each side is matched separately, so a `*` on one
    # side cannot swallow the arrow and match the other.
    match: str = "*"
    tenant: str = GLOBAL_TENANT  # fnmatch pattern over the requesting tenant
    approver_role: str | None = None  # required on an `ask` rule, forbidden elsewhere
    reason: str = ""  # carried verbatim into the audit record
    over_usd: float | None = None  # spend only; None matches any amount
    basis: SpendBasis | None = None  # spend only; defaults to `call`

    @model_validator(mode="after")
    def _check(self) -> PolicyRule:
        if not self.id.strip():
            raise ValueError("a rule needs a non-empty id: the audit record names it")
        if self.effect is Decision.ASK and not (self.approver_role or "").strip():
            raise ValueError(
                f"rule {self.id!r}: an 'ask' rule must name an approver_role — "
                "an approval nobody is asked for can only fail closed"
            )
        if self.effect is not Decision.ASK and self.approver_role is not None:
            raise ValueError(
                f"rule {self.id!r}: approver_role is only meaningful on an 'ask' rule "
                f"(this one is {self.effect.value!r})"
            )
        spend = self.resource is ResourceKind.SPEND
        if not spend and (self.over_usd is not None or self.basis is not None):
            raise ValueError(
                f"rule {self.id!r}: over_usd/basis apply to a 'spend' rule, "
                f"not {self.resource.value!r}"
            )
        if self.over_usd is not None and self.over_usd < 0:
            raise ValueError(f"rule {self.id!r}: over_usd must not be negative")
        if self.resource is ResourceKind.EDGE:
            source, arrow, target = self.match.partition(EDGE_ARROW)
            if not arrow or not source.strip() or not target.strip():
                raise ValueError(
                    f"rule {self.id!r}: an edge match must be '<source>-><target>', "
                    f"got {self.match!r}"
                )
        return self

    @property
    def spend_basis(self) -> SpendBasis:
        return self.basis or SpendBasis.CALL

    def matches_tenant(self, tenant: str) -> bool:
        return fnmatch(tenant, self.tenant)

    def matches_subject(self, subject: str) -> bool:
        if self.resource is ResourceKind.EDGE:
            source_pattern, _, target_pattern = self.match.partition(EDGE_ARROW)
            source, _, target = subject.partition(EDGE_ARROW)
            return fnmatch(source, source_pattern.strip()) and fnmatch(
                target, target_pattern.strip()
            )
        return fnmatch(subject, self.match)

    def matches_amount(self, amount_usd: float) -> bool:
        """True when this rule's threshold is crossed. No threshold matches any amount."""
        return self.over_usd is None or amount_usd > self.over_usd


class PolicyDocument(BaseModel):
    """A loaded, validated policy.

    Immutable in both directions that matter: `frozen=True` blocks rebinding a
    field, and the collections are tuples, so `doc.rules.append(...)` is not a
    way around it either. A decision is therefore made against exactly the
    document its digest describes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    version: str
    name: str = ""
    description: str = ""
    default: Decision = Decision.DENY
    # Optional. When non-empty the loader checks every rule's tenant pattern
    # against it, and the engine refuses requests naming a tenant not on it.
    tenants: tuple[str, ...] = ()
    rules: tuple[PolicyRule, ...] = Field(default=(), alias="rule")

    @model_validator(mode="after")
    def _check(self) -> PolicyDocument:
        if not self.version.strip():
            raise ValueError("a policy document must carry a non-empty version")
        if self.default is Decision.ASK:
            raise ValueError(
                "default = 'ask' is not allowed: an unmatched request matches no rule and "
                "therefore names no approver, so it could only fail closed. Write an "
                "explicit catch-all 'ask' rule with an approver_role instead."
            )
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(
                    f"duplicate rule id {rule.id!r}: an audit record must name one rule"
                )
            seen.add(rule.id)
        if len(set(self.tenants)) != len(self.tenants):
            raise ValueError("duplicate entry in `tenants`")
        if any(not tenant.strip() for tenant in self.tenants):
            raise ValueError("an empty tenant name in `tenants` can never be requested")
        if self.tenants:
            for rule in self.rules:
                if rule.tenant == GLOBAL_TENANT:
                    continue
                if not any(fnmatch(tenant, rule.tenant) for tenant in self.tenants):
                    raise ValueError(
                        f"rule {rule.id!r} is scoped to tenant pattern {rule.tenant!r}, which "
                        f"matches none of the declared tenants {self.tenants!r} — it could "
                        "never fire"
                    )
        return self

    @property
    def digest(self) -> str:
        """SHA-256 over the parsed document.

        The `version` string is a claim by the author; this is evidence. Two
        documents that disagree about a single rule while carrying the same
        version have different digests, so an audit record pins which bytes
        actually decided. Comments and formatting are not in it — the digest is
        taken over the parsed model, not the file.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def rules_for(self, resource: ResourceKind) -> list[PolicyRule]:
        return [rule for rule in self.rules if rule.resource is resource]

    def declares_tenant(self, tenant: str) -> bool:
        """True when `tenant` is usable: either the document lists it, or it lists none.

        An exact-match check, not a pattern one: `tenants` enumerates real
        tenants, while a *rule's* `tenant` field is the pattern side.
        """
        return not self.tenants or tenant in self.tenants


def parse_document(text: str, *, origin: str = "<string>") -> PolicyDocument:
    """Parse TOML `text` into a validated document, or raise `PolicyError`."""
    try:
        raw: dict[str, Any] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"{origin}: not valid TOML: {exc}") from exc
    try:
        return PolicyDocument.model_validate(raw)
    except ValidationError as exc:
        raise PolicyError(f"{origin}: invalid policy document: {exc}") from exc


def load_document(path: str | Path) -> PolicyDocument:
    """Read and validate the policy document at `path`, or raise `PolicyError`."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read policy document {target}: {exc}") from exc
    return parse_document(text, origin=str(target))
