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


def test_the_shell_block_is_the_command_and_the_output_it_really_prints(
    capsys, tmp_path, monkeypatch
):
    shell, expected, *_ = (body for _, body in _blocks(SECTION))
    command = shell.strip()

    assert command.startswith("grapharc plan "), command
    import shlex

    argv = shlex.split(command)[1:]  # drop the program name; keep flags intact

    # A scratch cwd: run output must not depend on the developer's checkout —
    # a leftover `.grapharc/` policy cache or a running live server would flip
    # the policy/watch lines and fail the byte comparison for the wrong reason.
    monkeypatch.chdir(tmp_path)
    code = main(argv)
    printed = capsys.readouterr().out

    assert code == 0
    # The trace path (and the watch line derived from it) varies per run and
    # per machine, so the page does not quote either. Everything else is fixed
    # and is compared exactly.
    kept = [
        line
        for line in printed.splitlines()
        if not line.startswith(("trace     :", "watch     :"))
    ]
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


def test_the_quick_start_block_actually_runs_against_this_tree():
    """The page's *first* code block, executed rather than admired.

    It was previously LangGraph's API rather than GraphARC's — it opened with
    `from grapharc.runtime import StateGraph`, a name this package has never
    exported, so the very first line raised `ImportError`. The two calls after
    it were wrong in their own right: `add_node` without `writes=` raises
    `TypeError` (the argument is required, and per-node write permissions are
    the project's headline claim), and `add_edge("START", ...)` raises
    `ValueError` because `START` is a sentinel, not the string `"START"`.

    Nothing caught it because nothing ran it. The admission-gate section has
    been executed by this file since it was written; the Quick Start had the
    same standing on the page and none of the same discipline, which is exactly
    the drift this module's docstring describes. So: same treatment. The
    trailing comment on the page states the printed result, and it is compared
    against what the block really prints.
    """
    # The section leads with the CLI tour (a bash block) and follows it with one
    # Python graph, so this selects by language rather than by position — the
    # earlier `== ["python"]` assertion encoded the old layout, in which the
    # Python snippet was the section's only content and sat 100 lines above the
    # commands that actually demonstrate the project.
    python_blocks = [code for lang, code in _blocks("Quick start") if lang == "python"]
    assert len(python_blocks) == 1, "the quick start must carry exactly one Python block"
    code = python_blocks[0]

    # The expectation is written on the page as a trailing `# {...}` comment,
    # so the snippet stays copy-pasteable instead of carrying a second block.
    expected = [
        line.lstrip("# ").strip() for line in code.splitlines() if line.startswith("# {")
    ]
    assert len(expected) == 1, "the quick start must state its printed result"

    buffer = io.StringIO()
    namespace: dict = {"__name__": "__readme__"}
    with redirect_stdout(buffer):
        exec(compile(code, f"{README}:quick-start", "exec"), namespace)

    assert _normalise(buffer.getvalue()) == _normalise(expected[0])


def test_the_quick_start_python_block_reaches_no_live_backend():
    """The first snippet a visitor copies must not be able to spend money."""
    code = [c for lang, c in _blocks("Quick start") if lang == "python"][0]

    for forbidden in ("get_model(", "openrouter", "ClaudeCodeCLIChatModel", "claude-cli"):
        assert forbidden not in code, forbidden


def test_every_in_page_link_resolves_to_a_real_heading():
    """A dead `#anchor` does nothing visible on GitHub, so neither reading the
    page nor running the suite used to surface one.

    This began life checking only the hand-maintained contents list, which had
    two invented entries (`#usage`, `#documentation`) pointing at sections that
    never existed. That list is gone — GitHub renders its own outline, and a
    second one in a different order from the document was a maintenance burden
    that had already drifted — so the check now covers *every* in-page link on
    the page, which is strictly more than it covered before and no longer
    depends on a particular section existing.
    """
    text = README.read_text(encoding="utf-8")

    anchors = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", title.lower())
        anchors.add(re.sub(r"\s+", "-", slug.strip()))

    linked = re.findall(r"\]\(#([a-z0-9-]+)\)", text)
    assert linked, "the README has no in-page links at all"

    dead = sorted(set(linked) - anchors)
    assert not dead, f"README links to non-existent sections: {dead}"


def _embedded_images() -> list[str]:
    """Every local image the README embeds, read off the README itself.

    Derived rather than hardcoded: a list of paths spelled out here only ever
    tests that someone remembered to update the list. Remote images are skipped
    — this asserts about files in the tree.
    """
    text = README.read_text(encoding="utf-8")
    return [m for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) if "://" not in m]


def test_the_readme_embeds_at_least_one_image():
    """Guards the regex above: a silent zero-match would make the next test vacuous."""
    assert _embedded_images()


@pytest.mark.parametrize("path", _embedded_images())
def test_every_image_the_readme_embeds_exists(path):
    """A README that renders a broken image is a broken README."""
    assert (README.parent / path).is_file(), f"{path} is referenced and missing"
