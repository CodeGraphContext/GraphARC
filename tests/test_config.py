"""`grapharc.toml` — the four precedence layers, and the one convention broken.

Two properties here are load-bearing rather than convenient.

**No parent-directory search.** git, npm and cargo all walk upward. This does
not, and the test below pins it: a run must never be governed by a policy file
in a directory the operator did not know about. Convenience is the wrong trade
when the file decides what an agent is permitted to do.

**Provenance on the payload.** A config file makes "which policy was I subject
to" *less* visible on the command line, so every resolved value records the layer
that supplied it and the commands print it. Without that, a governed run and a
defaulted one look identical afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grapharc.cli.config import CONFIG_NAME, ConfigError, Settings, load
from grapharc.cli.main import main

DENY_DEPLOY = """version = "1"
default = "allow"

[[rule]]
id = "no-deploy"
resource = "edge"
match = "*->deploy"
effect = "deny"
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A scratch cwd holding a config file and the policy it names."""
    (tmp_path / "deny.toml").write_text(DENY_DEPLOY, encoding="utf-8")
    (tmp_path / CONFIG_NAME).write_text(
        '[grapharc]\npolicy = "deny.toml"\ntenant = "acme"\nmax_rounds = 3\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _plan_payload(capsys, *args):
    code = main(["plan", "a goal", "--scripted", "--json", *args])
    return code, json.loads(capsys.readouterr().out)


# -- the four layers ---------------------------------------------------------


def test_a_config_file_supplies_what_no_flag_did(project, capsys):
    code, payload = _plan_payload(capsys)

    assert code == 0
    assert payload["sources"]["policy"] == "config"
    assert payload["sources"]["tenant"] == "config"
    assert "deny.toml" in payload["policy"]
    assert payload["config_file"] == str(project / CONFIG_NAME)


def test_an_environment_variable_beats_the_config_file(project, capsys, monkeypatch):
    monkeypatch.setenv("GRAPHARC_TENANT", "globex")

    _, payload = _plan_payload(capsys)

    assert payload["sources"]["tenant"] == "env"
    assert "tenant 'globex'" in payload["policy"]


def test_a_flag_beats_an_environment_variable(project, capsys, monkeypatch):
    monkeypatch.setenv("GRAPHARC_TENANT", "globex")

    _, payload = _plan_payload(capsys, "--tenant", "from-flag")

    assert payload["sources"]["tenant"] == "flag"
    assert "tenant 'from-flag'" in payload["policy"]


def test_a_default_is_recorded_as_a_default(project, capsys):
    """`max_tokens` is in neither the file nor a flag."""
    _, payload = _plan_payload(capsys)

    assert payload["sources"]["max_tokens"] == "default"


def test_with_no_config_file_everything_is_a_default(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    _, payload = _plan_payload(capsys)

    assert payload["config_file"] is None
    # The property is that no *config* layer was consulted — not that every
    # label reads "default". `policy` legitimately reports where the policy came
    # from (the registry's own), which is a different question.
    assert "config" not in payload["sources"].values()
    assert "env" not in payload["sources"].values()


# -- the convention this deliberately breaks ---------------------------------


def test_parent_directories_are_never_searched(project, monkeypatch, capsys):
    """The property, stated as a test: a run from a subdirectory is NOT governed
    by the config above it. Convenience loses to knowing what you are subject to."""
    nested = project / "sub" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    _, payload = _plan_payload(capsys)

    assert payload["config_file"] is None, "a parent's config must not reach this run"
    assert "config" not in payload["sources"].values(), "nothing may come from a parent"


def test_config_can_still_be_named_explicitly_from_anywhere(project, monkeypatch, capsys):
    """Not searching upward must not mean the file is unreachable — it means
    reaching it is a decision the command line records."""
    nested = project / "sub"
    nested.mkdir()
    monkeypatch.chdir(nested)

    _, payload = _plan_payload(capsys, "--config", str(project / CONFIG_NAME))

    assert payload["sources"]["policy"] == "config"
    assert "deny.toml" in payload["policy"]


# -- refusing to guess -------------------------------------------------------


def test_an_unknown_key_is_an_error_not_a_silent_no_op(tmp_path):
    """A typo in a governance file that is quietly ignored is the same failure
    as a policy rule nobody reads."""
    path = tmp_path / CONFIG_NAME
    path.write_text('[grapharc]\npollicy = "deny.toml"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load(path)


def test_a_wrong_type_is_named_rather_than_coerced(tmp_path):
    path = tmp_path / CONFIG_NAME
    path.write_text("[grapharc]\nmax_rounds = true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="max_rounds must be int"):
        load(path)


def test_a_bool_is_not_accepted_where_a_count_belongs(tmp_path):
    """`bool` is an `int` subclass; `true` must not become 1."""
    path = tmp_path / CONFIG_NAME
    path.write_text("[grapharc]\nmax_tokens = true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="got bool"):
        load(path)


def test_a_non_coercible_env_value_names_the_variable(monkeypatch):
    """The file layer already names this mistake; the env layer used to let it
    escape as a raw ValueError. An exported variable is invisible on the
    command line, which is exactly why the error must say which one it was."""
    monkeypatch.setenv("GRAPHARC_MAX_TOKENS", "unlimited")

    with pytest.raises(ConfigError, match="GRAPHARC_MAX_TOKENS must be int, got 'unlimited'"):
        Settings().resolve("max_tokens", None)


def test_run_under_a_bad_env_value_fails_as_one_document(tmp_path, monkeypatch, capsys):
    """The `--json` contract, held even for this failure: exit 2, one parseable
    document on stdout, nothing on stderr. It used to be exit 1, an empty
    stdout and a traceback — a CI gate read that as "topology refused"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPHARC_MAX_TOKENS", "unlimited")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"name": "triage"}],
                "edges": [
                    {"source": "__start__", "target": "triage"},
                    {"source": "triage", "target": "__end__"},
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["run", str(graph), "--check-only", "--json"])
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert code == 2
    assert payload["ok"] is False
    assert "GRAPHARC_MAX_TOKENS must be int, got 'unlimited'" in payload["error"]
    assert err == ""


def test_plan_reports_the_env_error_not_a_planner_one(tmp_path, monkeypatch, capsys):
    """The broad handler used to relabel this "could not build the plan:
    invalid literal…" — pointing the reader at the model, not their shell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPHARC_MAX_ROUNDS", "abc")

    code, payload = _plan_payload(capsys)

    assert code == 2
    assert "GRAPHARC_MAX_ROUNDS must be int, got 'abc'" in payload["error"]
    assert "could not build the plan" not in payload["error"]


def test_an_explicit_config_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="no such file"):
        load(tmp_path / "absent.toml")


def test_an_absent_default_config_is_not_an_error(tmp_path):
    """Most runs have no config; inventing one would be worse than having none."""
    settings = load(cwd=tmp_path)

    assert settings.path is None
    assert settings.values == {}


def test_malformed_toml_names_the_file(tmp_path):
    path = tmp_path / CONFIG_NAME
    path.write_text("[grapharc\nbroken", encoding="utf-8")

    with pytest.raises(ConfigError, match=CONFIG_NAME):
        load(path)


def test_a_config_that_is_not_utf8_names_the_file(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`, so it was in neither except tuple.

    This file is picked up implicitly from the working directory, so a stray
    binary `grapharc.toml` used to traceback out of every configurable command.
    """
    path = tmp_path / CONFIG_NAME
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError, match=CONFIG_NAME):
        load(path)


# -- the resolver itself -----------------------------------------------------


def test_resolve_reports_every_layer_it_used():
    settings = Settings(path=Path("x.toml"), values={"tenant": "acme"})

    assert settings.resolve("tenant", None) == "acme"
    assert settings.resolve("model", "flagged") == "flagged"
    assert settings.resolve("policy", None, "fallback") == "fallback"
    assert settings.sources == {"tenant": "config", "model": "flag", "policy": "default"}


def test_describe_names_what_the_file_sets():
    settings = Settings(path=Path("grapharc.toml"), values={"policy": "p.toml", "tenant": "a"})

    assert "policy" in settings.describe()
    assert "tenant" in settings.describe()
    assert Settings().describe() == "no grapharc.toml (flags and defaults only)"


def test_run_reads_the_same_config_as_plan(project, capsys):
    """One resolver, both commands — or the two drift."""
    graph = project / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"name": "triage"}, {"name": "ship", "kind": "deploy"}],
                "edges": [
                    {"source": "__start__", "target": "triage"},
                    {"source": "triage", "target": "ship"},
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["run", str(graph), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1, "the config's deny rule has to reach this command too"
    assert payload["sources"]["policy"] == "config"
    assert [r["code"] for r in payload["rejections"]] == ["edge_denied"]


def test_a_relative_config_path_anchors_to_the_config_not_the_cwd(
    project, monkeypatch, capsys
):
    """The bug this caught: `policy = "deny.toml"` resolved against cwd, so the
    same file named a different policy depending on where you stood — and
    `--config` from another directory could not work at all."""
    nested = project / "elsewhere"
    nested.mkdir()
    monkeypatch.chdir(nested)

    _, payload = _plan_payload(capsys, "--config", str(project / CONFIG_NAME))

    assert payload["sources"]["policy"] == "config"
    assert str(project / "deny.toml") in payload["policy"]


def test_a_path_from_a_flag_still_resolves_against_the_cwd(project, monkeypatch, capsys):
    """Someone typing a path means the one relative to where they typed it."""
    nested = project / "here"
    nested.mkdir()
    (nested / "local.toml").write_text(DENY_DEPLOY, encoding="utf-8")
    monkeypatch.chdir(nested)

    _, payload = _plan_payload(capsys, "--policy", "local.toml")

    assert payload["sources"]["policy"] == "flag"
    assert payload["policy"].startswith("local.toml")


# -- defects an adversarial pass found -------------------------------------


def test_a_key_beside_the_table_is_an_error_not_a_silent_drop(tmp_path):
    """`policy` at the top level, outside `[grapharc]`, used to be dropped in
    silence — so a run intended to be strict quietly used the permissive
    default. This module's own docstring forbids exactly that."""
    path = tmp_path / CONFIG_NAME
    path.write_text('policy = "strict.toml"\n[grapharc]\ntenant = "x"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="sits beside the"):
        load(path)


def test_a_table_free_document_is_still_accepted(tmp_path):
    """Keys at the top level with no `[grapharc]` table at all are the other
    supported shape and must keep working."""
    path = tmp_path / CONFIG_NAME
    path.write_text('tenant = "acme"\n', encoding="utf-8")

    assert load(path).values == {"tenant": "acme"}


def test_sources_agrees_with_policy_source(tmp_path, monkeypatch, capsys):
    """They contradicted each other: a policy read from `.grapharc/` reported
    `policy_source: generated-cached` while `sources.policy` still said
    `default`. A reviewer reading `sources` alone would conclude no file was
    involved."""
    monkeypatch.chdir(tmp_path)
    from grapharc.cli.generate import generated_policy_path
    from grapharc.cli.plan import DEFAULT_REGISTRY

    # Keyed to the registry the run will use: an un-keyed file is a statement
    # about an unknown registry and is deliberately ignored now.
    cached = generated_policy_path(tmp_path, registry=DEFAULT_REGISTRY)
    cached.parent.mkdir()
    cached.write_text(DENY_DEPLOY, encoding="utf-8")

    _, payload = _plan_payload(capsys)

    assert payload["policy_source"] == "generated-cached"
    assert payload["sources"]["policy"] == "generated-cached"


def test_an_explicit_policy_still_reports_its_own_layer(project, capsys):
    """The override must not clobber a real flag/config answer."""
    _, payload = _plan_payload(capsys, "--policy", "deny.toml")

    assert payload["sources"]["policy"] == "flag"


def test_the_demo_command_reads_the_config_too(tmp_path, monkeypatch, capsys):
    """`memory` and `reviewer_model` were declared in KEYS and read by nothing,
    so the same file made `plan` fail on a bad model and left `demo` scripted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        '[grapharc]\nmemory = "./claims.sqlite"\n', encoding="utf-8"
    )

    main(["demo", "capstone", "--json", "--trace", str(tmp_path / "t.jsonl")])
    first = json.loads(capsys.readouterr().out)
    main(["demo", "capstone", "--json", "--trace", str(tmp_path / "t2.jsonl")])
    second = json.loads(capsys.readouterr().out)

    assert (tmp_path / "claims.sqlite").exists(), "the config's memory path was ignored"
    assert "No prior knowledge" in first["result"]["recalled"]
    assert "No prior knowledge" not in second["result"]["recalled"]
