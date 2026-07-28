"""The README's runnable blocks, executed against this tree.

The page's `## The admission gate` section is the first code a visitor reads and
the project's central claim. It used to be illustrative only — it named
`make_search`, `make_edit`, `model`, `trace` and `State` without defining any of
them, imported no `Budget`, and nothing executed it — so it could drift from the
API indefinitely without a test noticing. The cookbook already has this
discipline (`tests/test_cookbook_*.py`); the README did not.

Both blocks in that section are checked here: the shell one against a real
`grapharc plan` run, the Python one by executing it and byte-comparing stdout
with the output block that follows it on the page.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from grapharc.cli.main import main

README = Path(__file__).resolve().parents[1] / "README.md"
SECTION = "The admission gate"
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```", re.M | re.S)


def _section(title: str) -> str:
    for chunk in re.split(r"^## ", README.read_text(encoding="utf-8"), flags=re.M):
        if chunk.startswith(title):
            return chunk
    raise AssertionError(f"README has no '## {title}' section")


def _blocks(title: str) -> list[tuple[str, str]]:
    return [(lang, body) for lang, body in FENCE.findall(_section(title))]


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def test_the_section_still_holds_the_two_blocks_this_file_checks():
    """A guard on the guard: a rewrite that drops a block must not pass silently."""
    langs = [lang for lang, _ in _blocks(SECTION)]

    assert langs == ["bash", "", "python", ""], langs


def test_the_shell_block_is_the_command_and_the_output_it_really_prints(capsys):
    shell, expected, *_ = (body for _, body in _blocks(SECTION))
    command = shell.strip()

    assert command.startswith("grapharc plan "), command
    goal = command.removeprefix("grapharc plan ").strip().strip('"')

    code = main(["plan", goal])
    printed = capsys.readouterr().out

    assert code == 0
    # The trace path is a fresh temp dir on every run, so the page does not
    # quote it. Everything above it is fixed and is compared exactly.
    kept = [line for line in printed.splitlines() if not line.startswith("trace     :")]
    assert _normalise("\n".join(kept)) == _normalise(expected)


def test_the_python_block_prints_exactly_what_the_page_says():
    _, _, code, expected = (body for _, body in _blocks(SECTION))

    buffer = io.StringIO()
    namespace: dict = {"__name__": "__readme__"}
    with redirect_stdout(buffer):
        exec(compile(code, f"{README}:admission-gate", "exec"), namespace)

    assert _normalise(buffer.getvalue()) == _normalise(expected)


def test_the_python_block_reaches_no_live_backend():
    """A README snippet must never be able to spend money when someone runs it."""
    _, _, code, _ = (body for _, body in _blocks(SECTION))

    for forbidden in ("get_model(", "openrouter", "ClaudeCodeCLIChatModel", "claude-cli"):
        assert forbidden not in code, forbidden


@pytest.mark.parametrize(
    "path", ["docs/diagrams/01-lifecycle.png", "docs/diagrams/03-agent-node.png"]
)
def test_every_image_the_readme_embeds_exists(path):
    """A README that renders a broken image is a broken README."""
    text = README.read_text(encoding="utf-8")
    assert f"]({path})" in text, f"README no longer embeds {path}"
    assert (README.parent / path).is_file(), f"{path} is referenced and missing"
