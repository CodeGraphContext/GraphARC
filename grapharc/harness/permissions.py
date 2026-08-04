"""Tool permissions: deny → ask → allow, first match wins, fail closed.

Permissions gate *which* tools a model may call. They are enforced by the
harness, never by prompts — instructions are advisory, this is not. A broad
deny always beats a narrower allow (no allowlist exceptions inside a deny),
matching the semantics that survived contact with reality in Claude Code.

A pattern is a glob, but a DENY or ASK rule also fires on an exact literal
match, so a rule that simply names a tool refuses it even when the name carries
fnmatch metacharacters (`exfil[all]`). ALLOW keeps glob-only matching — see
`PermissionRule.matches` for why the two tiers differ, and
`PermissionRule.literal` for naming one tool exactly at any tier.
"""

from __future__ import annotations

import glob
from enum import StrEnum
from fnmatch import fnmatch

from pydantic import BaseModel


class Decision(StrEnum):
    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"


class PermissionDenied(Exception):
    """A tool call was refused by policy (or by an absent/negative approval)."""


#: Tiers where an exact literal name is also honoured as a match, on top of the
#: glob. Restricting a tool can only ever *narrow* what runs, so widening these
#: two is safe in the direction this module already fails; see `matches`.
_LITERAL_TIERS = frozenset({Decision.DENY, Decision.ASK})


class PermissionRule(BaseModel):
    action: Decision
    pattern: str  # fnmatch pattern over the tool name; DENY/ASK also match literally

    @classmethod
    def literal(cls, action: Decision, name: str) -> PermissionRule:
        """A rule matching exactly one tool name, whatever characters it holds.

        `pattern` is a glob, so a name containing `*`, `?` or `[` is not the
        rule that names it — `exfil[all]` reads as a character class. This
        escapes the name (`glob.escape`) so the rule means the tool and nothing
        else. Use it whenever the name comes from a registry rather than from an
        operator writing a pattern by hand, and especially for ALLOW, where the
        literal fallback in `matches` deliberately does not apply.
        """
        return cls(action=action, pattern=glob.escape(name))

    def matches(self, tool_name: str) -> bool:
        """Does this rule fire for `tool_name`?

        The glob, plus — for DENY and ASK only — an exact string equality. A
        tool whose name contains fnmatch metacharacters (`mcp__srv__do[all]`)
        would otherwise slip past the rule that names it exactly, and the
        evaluation would fall through to a broader ALLOW: the one place in this
        tree where a *deny* failed open.

        Bound to DENY/ASK on purpose. Equality can only add rules that refuse or
        gate a call, never ones that permit it, so it cannot loosen a policy;
        the same widening on ALLOW could grant a tool the operator never allowed.
        For an ALLOW rule naming a metacharacter-bearing tool, write it with
        `PermissionRule.literal`, which escapes rather than widens.
        """
        return fnmatch(tool_name, self.pattern) or (
            self.action in _LITERAL_TIERS and tool_name == self.pattern
        )


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
                if rule.action == tier and rule.matches(tool_name):
                    return tier
        return self.default
