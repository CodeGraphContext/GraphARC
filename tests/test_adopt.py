"""`grapharc init --claude-code` — the adoption files, and their contract.

Two properties: the command never overwrites what an operator already owns,
and the skill it writes carries the never-self-approve clause in so many
words — the MCP surface refuses the approval verb on its own connection, and
the skill states the same boundary for the host agent's other hands.
"""

from __future__ import annotations

import json

import pytest

from grapharc.cli.adopt import MCP_CONFIG_FILENAME, SKILL_PATH, SKILL_TEMPLATE
from grapharc.cli.main import main


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_adopt_writes_the_server_config_and_the_skill(in_tmp, capsys):
    assert main(["init", "--claude-code", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    config = json.loads((in_tmp / MCP_CONFIG_FILENAME).read_text())
    assert config["mcpServers"]["grapharc"]["command"] == "grapharc"
    assert config["mcpServers"]["grapharc"]["args"] == ["mcp"]
    assert (in_tmp / SKILL_PATH).is_file()


def test_adopt_refuses_to_overwrite_and_names_the_files(in_tmp, capsys):
    (in_tmp / MCP_CONFIG_FILENAME).write_text("{}")

    assert main(["init", "--claude-code", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert MCP_CONFIG_FILENAME in payload["error"]
    # The refusal changed nothing: the operator's file is intact and the
    # skill was not half-written beside it.
    assert (in_tmp / MCP_CONFIG_FILENAME).read_text() == "{}"
    assert not (in_tmp / SKILL_PATH).exists()


def test_the_skill_carries_the_never_self_approve_clause(in_tmp):
    """The one sentence the packaging must not lose, asserted against the
    exact bytes a user receives."""
    assert main(["init", "--claude-code"]) == 0
    text = (in_tmp / SKILL_PATH).read_text()

    assert text == SKILL_TEMPLATE
    assert "Never run `grapharc approve`" in text
    assert "approval-decision.json" in text
    assert "A timeout means ask, not retry" in text
    assert "The decision belongs to the user" in text


def test_plain_init_is_untouched_by_the_new_flag(in_tmp):
    """`init` without the flag still scaffolds the registry pair, and the
    adoption files are not part of that scaffold."""
    assert main(["init"]) == 0
    assert (in_tmp / "registry.py").is_file()
    assert (in_tmp / "grapharc.toml").is_file()
    assert not (in_tmp / MCP_CONFIG_FILENAME).exists()
    assert not (in_tmp / SKILL_PATH).exists()
