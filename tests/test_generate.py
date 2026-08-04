"""Zero-config policy generation — and the disclosure that has to come with it.

The risk this feature carries is not that a generated policy is bad. It is that
a generated run and an authored one look *identical* afterwards. There is no
flag on the command line to signal it, so the banner and `policy_source` are the
entire disclosure mechanism, and the tests below treat them as load-bearing
rather than cosmetic.

The other property worth pinning: generation is a **one-time** state. The result
is written to `.grapharc/`, so the second run reads it as an ordinary file and
reports `generated-cached`. By then the operator is subject to something they can
open and edit, which is what makes this an onboarding ramp rather than a
permanent hole.
"""

from __future__ import annotations

import json

import pytest

from grapharc import stdlib
from grapharc.cli.generate import (
    GENERATED_DIR,
    build_policy_toml,
    describe_policy,
    generated_policy_path,
    resolve_or_generate_policy,
)
from grapharc.cli.main import main
from grapharc.testing import ScriptedChatModel

GOOD_POLICY = """version = "1"
default = "allow"

[[rule]]
id = "no-file-changes"
resource = "edge"
match = "*->apply_change"
effect = "deny"
reason = "the goal asked to investigate, not to change anything"
"""


def _model(reply: str) -> ScriptedChatModel:
    return ScriptedChatModel(responses=[reply], on_exhausted="repeat")


def _generate(tmp_path, reply=GOOD_POLICY, **kwargs):
    return resolve_or_generate_policy(
        None,
        tenant="default",
        model=_model(reply),
        goal=kwargs.pop("goal", "investigate the outage"),
        workdir=tmp_path,
        catalog=stdlib.catalog_for_prompt(_model("x")),
        mutating=stdlib.MUTATING_KINDS,
        **kwargs,
    )


# -- it generates, and the result is real ------------------------------------


def test_a_first_run_with_nothing_specified_generates_a_policy(tmp_path):
    policy, description, source = _generate(tmp_path)

    assert source == "generated"
    assert "review it" in description
    assert policy.edge.rules, "a generated policy with no rules would gate nothing"


def test_the_generated_policy_actually_governs(tmp_path):
    """Not just parsed — consulted."""
    from grapharc.harness.permissions import Decision

    policy, _, _ = _generate(tmp_path)

    assert policy.edge.decide("investigate", "apply_change") is Decision.DENY
    assert policy.edge.decide("investigate", "verify") is Decision.ALLOW


# -- disclosure --------------------------------------------------------------


def test_the_source_is_reported_so_a_reader_can_tell(tmp_path):
    """With no flag on the command line, this is the only signal there is."""
    _, _, source = _generate(tmp_path)

    assert source == "generated"


def test_a_named_policy_is_never_reported_as_generated(tmp_path):
    named = tmp_path / "mine.toml"
    named.write_text(GOOD_POLICY, encoding="utf-8")

    _, _, source = resolve_or_generate_policy(named, tenant="default", workdir=tmp_path)

    assert source == "flag-or-config"


def test_the_command_puts_the_source_in_its_payload(tmp_path, monkeypatch, capsys):
    """The trace is what an incident review reads; a banner is seen once."""
    monkeypatch.chdir(tmp_path)

    main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert "policy_source" in payload
    assert payload["policy_source"] in {
        "flag-or-config",
        "registry-default",
        "generated",
        "generated-cached",
        "builtin-default",
    }


def test_the_human_view_names_the_source_too(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    main(["plan", "look into it", "--scripted"])
    out = capsys.readouterr().out

    assert "policy    :" in out
    assert "[" in out and "]" in out, "the source is bracketed onto the policy line"


# -- generation is a one-time state ------------------------------------------


def test_the_generated_policy_is_written_where_a_human_can_read_it(tmp_path):
    _generate(tmp_path)

    written = generated_policy_path(tmp_path)
    assert written.is_file()
    assert written.parent.name == GENERATED_DIR
    text = written.read_text(encoding="utf-8")
    assert "REVIEW THIS" in text, "a file nobody is told to read is not disclosure"
    assert "no-file-changes" in text


def test_the_second_run_reads_it_off_disk_rather_than_regenerating(tmp_path):
    _generate(tmp_path)

    _, description, source = _generate(tmp_path)

    assert source == "generated-cached"
    assert "previously generated" in description


def test_node_rules_the_operator_added_to_the_cached_file_are_honoured(tmp_path):
    """The cached file is an ordinary document, and editing it is the point.

    Reading it back through the same compiler as `--policy` is what stops the
    generated path being a place where `resource = "node"` rules go to die
    (issue #66).
    """
    from grapharc.harness.permissions import Decision

    _generate(tmp_path)
    path = generated_policy_path(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[[rule]]\nid = "no-shell-nodes"\nresource = "node"\n'
        'match = "shell_*"\neffect = "deny"\n'
        '\n[[rule]]\nid = "others"\nresource = "node"\nmatch = "*"\neffect = "allow"\n',
        encoding="utf-8",
    )

    policy, _description, source = _generate(tmp_path)

    assert source == "generated-cached"
    assert policy.node is not None
    assert policy.node.decide("shell_exec") is Decision.DENY
    assert policy.node.decide("investigate") is Decision.ALLOW


def test_a_cached_policy_beats_generating_a_new_one_even_with_a_model(tmp_path):
    """Otherwise the policy changes under the operator on every run."""
    _generate(tmp_path)
    first = generated_policy_path(tmp_path).read_text(encoding="utf-8")

    _generate(tmp_path, reply='version = "1"\ndefault = "deny"\n')

    assert generated_policy_path(tmp_path).read_text(encoding="utf-8") == first


def test_write_can_be_turned_off_for_a_caller_that_does_not_want_a_file(tmp_path):
    _, _, source = _generate(tmp_path, write=False)

    assert source == "generated"
    assert not generated_policy_path(tmp_path).exists()


# -- failing safely ----------------------------------------------------------


def test_a_model_that_returns_prose_falls_back_rather_than_crashing(tmp_path):
    _, description, source = _generate(tmp_path, reply="I'm sorry, I can't do that.")

    assert source == "builtin-default"
    assert "generation failed" in description


def test_a_fenced_reply_is_still_accepted(tmp_path):
    """The commonest way a model breaks "emit only TOML". Refusing the run over
    three backticks would fire the fallback for no reason."""
    _, _, source = _generate(tmp_path, reply=f"```toml\n{GOOD_POLICY}```")

    assert source == "generated"


def test_with_no_model_nothing_is_generated(tmp_path):
    policy, description, source = resolve_or_generate_policy(
        None, tenant="default", model=None, workdir=tmp_path
    )

    assert source == "builtin-default"
    assert not generated_policy_path(tmp_path).exists()
    assert policy.edge.rules
    # The description says what the policy *does*, read back from its rules, so
    # it cannot drift from them.
    assert "deny -> apply_change" in description
    assert "otherwise allow" in description


def test_a_registrys_own_default_beats_the_builtin(tmp_path):
    """A policy written for other kinds would deny names that do not exist."""
    sentinel = stdlib.default_edge_policy()

    policy, description, source = resolve_or_generate_policy(
        None,
        tenant="default",
        model=None,
        workdir=tmp_path,
        fallback=sentinel,
        fallback_label="myco:build default",
    )

    assert source == "registry-default"
    assert description.startswith("myco:build default")
    assert "deny -> apply_change" in description
    assert policy.edge is sentinel


# -- what goes into the prompt -----------------------------------------------


def test_the_prompt_names_the_dangerous_kinds_and_the_goal():
    model = _model(GOOD_POLICY)

    build_policy_toml(model, "fix the outage", {"patch": "write a fix"}, ("deploy",))

    prompt = str(model.calls[0][0].content)
    assert "fix the outage" in prompt
    assert "deploy" in prompt
    assert "patch: write a fix" in prompt
    assert "Err towards denying" in prompt, "an unseeded prompt drafts permissive rules"


@pytest.mark.parametrize("kind", stdlib.MUTATING_KINDS)
def test_every_mutating_kind_is_offered_to_the_generator(kind):
    model = _model(GOOD_POLICY)

    build_policy_toml(model, "g", stdlib.catalog_for_prompt(model), stdlib.MUTATING_KINDS)

    assert kind in str(model.calls[0][0].content)


def test_a_scripted_run_never_generates(tmp_path, monkeypatch, capsys):
    """`grapharc plan` with no `--model` must stay deterministic and free: a demo
    that produced a different policy each run would not be one."""
    monkeypatch.chdir(tmp_path)

    main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["policy_source"] != "generated"
    # `.grapharc/` itself now exists (the default trace lands under runs/);
    # what a scripted run must never do is write a *policy* there.
    generated = list((tmp_path / GENERATED_DIR).glob("generated-policy*.toml"))
    assert generated == []


# -- the description is derived, not written ---------------------------------


def test_a_description_reads_the_fallthrough_off_the_policy_not_the_field():
    """A catch-all `allow` rule overrides a denying `default`. Quoting the field
    would describe the opposite of what the policy actually does."""
    from grapharc.harness.permissions import Decision
    from grapharc.planner import EdgePolicy, EdgeRule

    policy = EdgePolicy(
        rules=(
            EdgeRule(action=Decision.DENY, target="deploy"),
            EdgeRule(action=Decision.ALLOW),
        ),
        default=Decision.DENY,
    )

    assert policy.default is Decision.DENY
    assert describe_policy(policy) == "deny -> deploy, otherwise allow"


def test_a_description_names_every_tier_that_has_rules():
    from grapharc.harness.permissions import Decision
    from grapharc.planner import EdgePolicy, EdgeRule

    policy = EdgePolicy(
        rules=(
            EdgeRule(action=Decision.DENY, target="deploy"),
            EdgeRule(action=Decision.ASK, target="patch"),
        )
    )

    described = describe_policy(policy)
    assert "deny -> deploy" in described
    assert "ask -> patch" in described
    assert "otherwise deny" in described


# -- the cache is keyed by the registry that generated it ---------------------


STALE_INCIDENT_POLICY = """\
version = "1"
default = "allow"

[[rule]]
id = "deny-deploy"
resource = "edge"
match = "*->deploy"
effect = "deny"
reason = "generated for the incident registry, knows nothing of stdlib"
"""


def test_a_cached_policy_is_keyed_to_the_registry_that_generated_it(tmp_path):
    first = _generate(tmp_path, registry_target="pkg.one:build_registry")
    keyed = generated_policy_path(tmp_path, registry="pkg.one:build_registry")
    assert keyed.is_file()
    assert "pkg.one-build_registry" in keyed.name
    # A different registry's run does not read it.
    _, _, source = _generate(tmp_path, registry_target="pkg.two:build_registry")
    assert source == "generated"
    assert generated_policy_path(tmp_path, registry="pkg.two:build_registry").is_file()
    # The same registry's second run does.
    _, _, source = _generate(tmp_path, registry_target="pkg.one:build_registry")
    assert source == "generated-cached"
    assert first  # the first result existed; silences the unused warning


def test_a_stale_incident_policy_cannot_neuter_the_stdlib_mutating_deny(tmp_path):
    """The proof test for the reproduced safety bug: an un-keyed cache file
    generated for another registry must not override this registry's own
    default deny of its mutating kind."""
    from grapharc.harness.permissions import Decision

    legacy = generated_policy_path(tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(STALE_INCIDENT_POLICY, encoding="utf-8")

    policy, description, source = resolve_or_generate_policy(
        None,
        tenant="default",
        model=None,  # scripted run: no generation, fallback must win
        workdir=tmp_path,
        fallback=stdlib.default_edge_policy(),
        fallback_label="grapharc.stdlib:build_registry default",
        registry_target="grapharc.stdlib:build_registry",
    )
    assert source == "registry-default"
    assert policy.edge.decide("investigate", "apply_change") is Decision.DENY
    assert "ignoring un-keyed" in description
    # The operator's file survives untouched for an explicit --policy.
    assert legacy.read_text(encoding="utf-8") == STALE_INCIDENT_POLICY


def test_a_legacy_unkeyed_cache_is_still_honoured_without_a_registry_target(tmp_path):
    """Direct callers that never pass a target keep the old behavior — the
    refusal is scoped to runs that *can* say which registry they are."""
    legacy = generated_policy_path(tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(STALE_INCIDENT_POLICY, encoding="utf-8")
    _, _, source = resolve_or_generate_policy(
        None, tenant="default", workdir=tmp_path
    )
    assert source == "generated-cached"
