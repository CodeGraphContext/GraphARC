"""Zero-config policy: generate one, write it down, and say so.

A first run with no `--policy`, no `grapharc.toml` and no `.grapharc/` should
still work. The alternative is a blank page, and a governance tool that demands
one before it will do anything gets evaluated by nobody.

**What is generated, and what is not.** Policy only. Policy is *data* — a set of
TOML rules, readable, and the worst case is bad rules. A registry holds
*functions*, so generating one would mean an LLM writing code that then executes,
and the gate would be checking a list the gated thing wrote. There is no registry
generation here and there should not be: `grapharc.stdlib` ships the node kinds,
and a planner selects from them at proposal time, which is the existing
mechanism. Selecting is safe; authoring is not.

**Generation is a one-time state, by design.** The result is written to
`.grapharc/generated-policy.toml`, so the *second* run reads it from disk as an
ordinary policy file. By then the user is subject to something they can open,
diff and edit — and promoting it to a committed policy is a `mv`. That is what
turns a zero-config default into an onboarding ramp instead of a permanent hole.

**Disclosure is not optional here.** With no flag on the command line, a
generated run and an authored one look identical. So the source is printed *and*
carried in the payload as `policy_source`, which is the only place an incident
review would find it.

**With no model there is nothing to generate**, and that is fine: the fallback is
`grapharc.stdlib.default_edge_policy()` — allow read-only work, deny anything
into a kind that can change files. Restrictive, shipped, and honest about being a
default rather than a decision.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

GENERATED_DIR = ".grapharc"
GENERATED_POLICY = "generated-policy.toml"

#: Seeded restrictively on purpose. An LLM asked to draft permissions with no
#: steer produces something plausible and permissive, and plausible is the
#: failure mode rather than the safeguard.
PROMPT = """\
You are writing a GraphARC edge policy as a TOML document. It decides which
transitions between node kinds an automated planner may propose.

The goal the operator gave is:
{goal}

The node kinds that exist are:
{catalog}

Kinds that can CHANGE FILES (treat as dangerous): {mutating}

Rules for your output:
- Emit ONLY TOML. No prose, no code fences.
- Start with: version = "1"
- Set `default = "allow"` so read-only work needs no rule.
- Add one `[[rule]]` with `effect = "deny"` for every dangerous kind, UNLESS the
  goal clearly and explicitly asks for changes to be made. Investigating,
  reading, diagnosing, summarising and checking are NOT asking for changes.
- Every rule needs: id, resource = "edge", match = "*->KIND", effect, reason.
- The `reason` must say why, in one short sentence. It is read by a human during
  an incident review.

Err towards denying. A denial is one line for the operator to remove; an
unwanted change is not.
"""


def describe_policy(policy: Any) -> str:
    """Say what a policy *does*, read back from its own rules.

    A label like `myco:build default` says where a policy came from, which is
    not the question anyone asks first. Deriving the description from the rules
    means it cannot go stale the way a hand-written one would.
    """
    from grapharc.harness.permissions import Decision

    def targets(action: Decision) -> list[str]:
        return list(
            dict.fromkeys(
                r.target for r in policy.rules if r.action is action and r.target != "*"
            )
        )

    parts = []
    if denied := targets(Decision.DENY):
        parts.append("deny -> " + ", ".join(denied))
    if asked := targets(Decision.ASK):
        parts.append("ask -> " + ", ".join(asked))
    # The fallthrough is asked of the policy rather than read off `default`,
    # because a catch-all `allow` *rule* overrides a denying default and quoting
    # the field would describe the opposite of what the policy does.
    unmatched = "\x00unmatched"
    parts.append(f"otherwise {policy.decide(unmatched, unmatched).value}")
    return ", ".join(parts)


def generated_policy_path(workdir: Path | None = None, *, registry: str = "") -> Path:
    """Where a generated policy lives. Not created until something is written.

    Keyed by the registry target that generated it. A policy is a statement
    about one registry's kinds — a file generated for the incident demo's
    `deploy` said nothing about the stdlib's `apply_change`, yet the un-keyed
    cache served it anyway, silently overriding the stdlib registry's own
    deny of its mutating kind. The un-keyed name is kept only so callers can
    recognise (and refuse) legacy files; nothing writes it anymore.
    """
    base = Path(workdir or Path.cwd()) / GENERATED_DIR
    if not registry:
        return base / GENERATED_POLICY
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", registry)
    return base / f"generated-policy.{slug}.toml"


def build_policy_toml(
    model: Any, goal: str, catalog: dict[str, str], mutating: tuple[str, ...]
) -> str:
    """Ask the model for a policy document. Returns TOML text, never a policy.

    Kept separate from validation so a model that emits prose or fences is a
    parse failure at a named boundary rather than a confusing policy error.
    """
    listing = "\n".join(f"- {name}: {text}" for name, text in sorted(catalog.items()))
    prompt = PROMPT.format(
        goal=goal or "(none given)",
        catalog=listing or "(none)",
        mutating=", ".join(mutating) or "(none)",
    )
    message = model.invoke(prompt)
    text = getattr(message, "content", message)
    return _strip_fences(str(text))


def _strip_fences(text: str) -> str:
    """Remove a markdown fence if the model added one despite being told not to.

    Not defensive clutter: a fenced reply is the single most common way a model
    breaks a "emit only X" instruction, and refusing the whole run over three
    backticks would make the fallback fire for no reason.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def resolve_or_generate_policy(
    policy_path: Path | None,
    *,
    tenant: str,
    model: Any = None,
    goal: str = "",
    workdir: Path | None = None,
    write: bool = True,
    catalog: dict[str, str] | None = None,
    mutating: tuple[str, ...] = (),
    fallback: Any = None,
    fallback_label: str = "",
    registry_target: str = "",
) -> tuple[Any, str, str]:
    """Return `(gate_policy, description, source)`.

    The policy is a `grapharc.cli.plan.GatePolicy` carrying both halves a
    document compiles to — the edge rules and the node rules — because a run
    gated by only one half of a document is the bug this pair exists to prevent.
    Every document goes through `compile_policy`, generated ones included, so
    what a freshly generated policy means and what the same file means when it
    is read back off disk next run cannot differ. A fallback has no node half at
    all: it is an `EdgePolicy` written in Python, not a document.

    `source` is one of `flag-or-config`, `registry-default`, `generated-cached`,
    `generated`, `builtin-default`. Callers put it in the payload verbatim: it is
    the whole disclosure mechanism when nothing on the command line names a
    policy.

    Order: a named policy, then one generated earlier, then a fresh generation
    when a model is available, then the registry's own default, then the
    built-in. Generation sits *above* the registry default because the default
    cannot read the goal — "investigate the outage" and "fix the outage" deserve
    different answers about whether mutation is permitted, and only the model
    asked at run time knows which one was said.

    `catalog` and `mutating` come from the registry that will actually be used.
    Generating against some other registry's kinds would deny names that do not
    exist and permit ones that do.
    """
    from grapharc import stdlib
    from grapharc.cli.plan import GatePolicy, compile_policy, resolve_policy

    if policy_path is not None:
        policy, description = resolve_policy(policy_path, tenant=tenant)
        return policy, description, "flag-or-config"

    cached = generated_policy_path(workdir, registry=registry_target)
    if cached.is_file():
        # Read back as an ordinary document, node rules included: a generated
        # file the operator has since edited is theirs, not the generator's.
        policy, description = resolve_policy(cached, tenant=tenant)
        return policy, f"{description}  [previously generated]", "generated-cached"
    # A legacy un-keyed file cannot prove which registry it was generated for,
    # so it governs nothing implicitly — fail closed to generation or the
    # registry's own default, and say so. The operator's edits survive on disk
    # and are one `--policy` flag away.
    legacy = generated_policy_path(workdir)
    legacy_note = (
        f"  (ignoring un-keyed {legacy} — pass --policy to use it)"
        if registry_target and legacy.is_file()
        else ""
    )

    def _settled() -> tuple[Any, str, str]:
        if fallback is not None:
            label = fallback_label or "registry default"
            return (
                GatePolicy(edge=fallback),
                f"{label} ({describe_policy(fallback)}){legacy_note}",
                "registry-default",
            )
        builtin = stdlib.default_edge_policy()
        return (
            GatePolicy(edge=builtin),
            f"built-in default ({describe_policy(builtin)}){legacy_note}",
            "builtin-default",
        )

    if model is None:
        return _settled()

    try:
        toml_text = build_policy_toml(
            model,
            goal,
            catalog if catalog is not None else stdlib.catalog_for_prompt(model),
            mutating or stdlib.MUTATING_KINDS,
        )
        from grapharc.policy import PolicyEngine

        engine = PolicyEngine.from_toml(toml_text)
        # Compiled exactly as the file will be on the next run — the text below
        # is written to disk and read back as an ordinary document, so the two
        # readings must not differ.
        policy = compile_policy(engine, tenant=tenant)
    except Exception:  # noqa: BLE001 — any failure falls back rather than breaking the run
        policy, description, source = _settled()
        return policy, f"{description}  [generation failed]", source

    if write:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(
            "# Generated by `grapharc` because no policy was specified.\n"
            "# REVIEW THIS. It decides what an automated planner may wire together.\n"
            "# Edit it, or move it somewhere and pass --policy, to make it yours.\n\n"
            + toml_text
            + "\n",
            encoding="utf-8",
        )
    return policy, f"generated -> {cached}  (review it)", "generated"


__all__ = [
    "GENERATED_DIR",
    "GENERATED_POLICY",
    "PROMPT",
    "build_policy_toml",
    "describe_policy",
    "generated_policy_path",
    "resolve_or_generate_policy",
]
