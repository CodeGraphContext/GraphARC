"""The deep dive's **Verified this pass** paragraph, held against reality.

The paragraph's whole value is that its numbers are real. Nothing checked them,
so they drifted: it read "1,533 passed … 103 submodules" while the tree it
described had grown to 1,754 tests and 116 submodules, and it read "1,985
passed, 12 deselected" against a tree with 2,132 selected and 13 live. A reader
who spots one stale figure discounts every other verified claim on the page —
including the ones the suite genuinely enforces.

This is the discipline the cookbook pages and the README's runnable blocks
already have (`tests/test_cookbook_*.py`, `tests/test_readme.py` byte-compare
those against real output): prose that states a checkable fact gets a check.

The figures are therefore quoted as what one command re-derives — how many
tests `pytest` selects, and how many it holds back as `live` — rather than as
a pass count, which cannot be re-derived without running the suite from inside
itself. A green suite is asserted by the suite being green.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEEP_DIVE = ROOT / "docs" / "deep-dive.md"
MARKER = "**Verified this pass:**"

# The recount runs pytest in a subprocess rather than calling `pytest.main`
# in-process: this module is itself collected by the session doing the asking,
# and re-entering the collector from inside it is not a supported thing to do.
_RECOUNT = textwrap.dedent(
    """
    import pytest


    class Capture:
        def pytest_collection_finish(self, session):
            selected = live = 0
            for item in session.items:
                if item.get_closest_marker("live"):
                    live += 1
                else:
                    selected += 1
            print(f"COUNTS {selected} {live}")


    # `-m ""` clears the `-m 'not live'` that pyproject's addopts supplies, so
    # one collection pass yields both figures instead of two passes yielding one
    # each. The marker is read off each item rather than inferred from a second
    # selection.
    raise SystemExit(
        pytest.main(
            ["--collect-only", "-q", "-m", "", "-p", "no:cacheprovider"],
            plugins=[Capture()],
        )
    )
    """
)


def _paragraph() -> str:
    for line in DEEP_DIVE.read_text(encoding="utf-8").splitlines():
        if line.startswith(MARKER):
            return line
    raise AssertionError(f"{DEEP_DIVE.name} has no line starting with {MARKER!r}")


@pytest.fixture(scope="module")
def recount() -> tuple[int, int]:
    """(selected, deselected-as-live), re-derived from this tree."""
    proc = subprocess.run(
        [sys.executable, "-c", _RECOUNT],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^COUNTS (\d+) (\d+)$", proc.stdout, re.M)
    assert match, (
        f"collection did not report counts (exit {proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    return int(match.group(1)), int(match.group(2))


def _quoted(pattern: str) -> str:
    line = _paragraph()
    match = re.search(pattern, line)
    assert match, f"the paragraph no longer quotes {pattern!r}:\n{line}"
    return match.group(1)


# -- the figures ------------------------------------------------------------


def test_the_quoted_selection_is_what_pytest_selects(recount):
    selected, _ = recount
    quoted = int(_quoted(r"([\d,]+) selected").replace(",", ""))

    assert quoted == selected, (
        f"update the **Verified this pass** paragraph in {DEEP_DIVE.name}: it "
        f"says {quoted:,} selected, this tree has {selected:,}"
    )


def test_the_quoted_deselection_is_what_pytest_holds_back(recount):
    _, live = recount
    quoted = int(_quoted(r"([\d,]+) deselected").replace(",", ""))

    assert quoted == live, (
        f"update the **Verified this pass** paragraph in {DEEP_DIVE.name}: it "
        f"says {quoted:,} deselected, this tree marks {live:,} `live`"
    )


def test_the_quoted_published_version_is_the_packaged_one():
    """The paragraph names the version it says is on PyPI. `ci.yml` already
    refuses a `grapharc.__version__` that disagrees with pyproject; a release
    note naming a third number is the same failure with no check on it."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        packaged = tomllib.load(fh)["project"]["version"]
    quoted = _quoted(r"`(\d+\.\d+\.\d+)` on PyPI")

    assert quoted == packaged, (
        f"update the **Verified this pass** paragraph in {DEEP_DIVE.name}: it "
        f"says {quoted} is on PyPI, pyproject says {packaged}"
    )


# -- a guard on the guard ---------------------------------------------------


def test_the_paragraph_still_quotes_every_figure_this_file_checks():
    """A rewrite that drops a figure must not pass by leaving nothing to check.

    Without this, deleting "2,145 selected" from the sentence would make the
    test above vacuous rather than red — the same trap the README's
    `test_the_section_still_holds_the_two_blocks_this_file_checks` closes.
    """
    line = _paragraph()

    assert re.search(r"[\d,]+ selected", line), line
    assert re.search(r"[\d,]+ deselected", line), line
    assert re.search(r"`\d+\.\d+\.\d+` on PyPI", line), line


def test_the_paragraph_quotes_no_figure_that_nothing_re_derives():
    """The rule the issue settled on: a number on this line is either
    re-derived by a test in this file, or it does not belong on the line.

    `pass`/`fail` counts are the specific thing being kept off it — they cannot
    be re-derived without running the suite from inside itself, which is how
    the old "1,985 passed" figure came to be unowned in the first place.
    """
    line = _paragraph()
    checked = re.sub(r"[\d,]+ (?:selected|deselected)", "", line)
    checked = re.sub(r"`\d+\.\d+\.\d+` on PyPI", "", checked)
    # Version numbers inside command names and prose ordinals are not figures;
    # what this catches is a bare count with a unit, e.g. "1,985 passed".
    stray = re.findall(r"[\d,]{3,} \w+", checked)

    assert not stray, (
        f"these figures on the **Verified this pass** line are re-derived by "
        f"nothing: {stray}. Either add a check for them here or take them off "
        f"the line — that is the rot this file exists to stop."
    )
