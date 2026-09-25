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

**Keeping the figure honest must not be a newcomer's problem.** Any PR that
adds or removes a test moves these counts, so this file failed *every* such
branch until someone hand-edited a number in a docs file they had no reason to
know existed. That is what happened to PR #115: an outside contributor's first
change sat red for a month over "2,151 selected", and nothing in the failure
pointed at a fix they could run. The check is unchanged and still strict --
`GRAPHARC_UPDATE_FIGURES=1 pytest tests/test_deep_dive.py` now re-derives the
line and writes it back, and the failure message says so. CI never sets that
variable, so a stale figure still fails there, which is the whole point.
"""

from __future__ import annotations

import os
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

#: Opt in to rewriting the line instead of failing on it. Deliberately an
#: environment variable and not a pytest flag: `--strict-config` means an
#: unknown flag is an error, and a contributor reading a failure message can
#: paste an env var in front of the command they already ran.
UPDATE_ENV = "GRAPHARC_UPDATE_FIGURES"


def _updating() -> bool:
    """Whether this run may rewrite the paragraph.

    Off for unset, empty, "0" and "false", so a leftover `=0` in a shell
    profile cannot quietly turn the guard into a no-op. CI sets nothing, which
    is what keeps a stale figure red there.
    """
    return os.environ.get(UPDATE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _with_figure(line: str, pattern: str, value: int) -> str:
    """`line` with the one figure `pattern` captures replaced by `value`.

    Pure, so the rewrite is tested without touching the real document: only
    group 1's span changes, which keeps the surrounding prose and the comma
    grouping the page uses byte-identical everywhere else.
    """
    match = re.search(pattern, line)
    assert match, f"cannot rewrite {pattern!r}: it does not match:\n{line}"
    start, end = match.span(1)
    return line[:start] + f"{value:,}" + line[end:]


def _rewrite(pattern: str, value: int) -> None:
    """Write `value` into the marker line in place."""
    lines = DEEP_DIVE.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, raw in enumerate(lines):
        if raw.startswith(MARKER):
            lines[index] = _with_figure(raw, pattern, value)
            DEEP_DIVE.write_text("".join(lines), encoding="utf-8")
            return
    raise AssertionError(f"{DEEP_DIVE.name} has no line starting with {MARKER!r}")


def _remedy(pattern: str, value: int) -> str:
    """The failure message's second half: what to run, or what to edit."""
    return (
        f"\n\nRe-derive it: {UPDATE_ENV}=1 pytest tests/test_deep_dive.py\n"
        f"or edit the line by hand — the figure should read {value:,}."
    )

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
    pattern = r"([\d,]+) selected"
    quoted = int(_quoted(pattern).replace(",", ""))

    if quoted != selected and _updating():
        _rewrite(pattern, selected)
        pytest.skip(f"refreshed: {quoted:,} -> {selected:,} selected. Re-run to verify.")

    assert quoted == selected, (
        f"update the **Verified this pass** paragraph in {DEEP_DIVE.name}: it "
        f"says {quoted:,} selected, this tree has {selected:,}"
        + _remedy(pattern, selected)
    )


def test_the_quoted_deselection_is_what_pytest_holds_back(recount):
    _, live = recount
    pattern = r"([\d,]+) deselected"
    quoted = int(_quoted(pattern).replace(",", ""))

    if quoted != live and _updating():
        _rewrite(pattern, live)
        pytest.skip(f"refreshed: {quoted:,} -> {live:,} deselected. Re-run to verify.")

    assert quoted == live, (
        f"update the **Verified this pass** paragraph in {DEEP_DIVE.name}: it "
        f"says {quoted:,} deselected, this tree marks {live:,} `live`"
        + _remedy(pattern, live)
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
        f"says {quoted} is on PyPI, pyproject says {packaged}. Deliberately "
        f"not rewritten by {UPDATE_ENV}: whether a version is *published* is "
        f"not something this tree can re-derive, and a release note that "
        f"claims it should be written by whoever released it."
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


# -- the update mode --------------------------------------------------------
#
# The guard's value is that it is strict; its cost was that a contributor could
# not tell what to do about it. These cover both halves: the rewrite is correct,
# and it cannot happen unless someone asked for it.

_SAMPLE = (
    "**Verified this pass:** `pytest` -> green, 2,151 selected and 13 deselected "
    "(the live ones); `ruff check .` clean; `0.1.7` on PyPI is that wheel.\n"
)


def test_the_rewrite_changes_the_figure_and_nothing_else():
    """Comma grouping and every surrounding word survive, because the span of
    one capture group is all that is replaced."""
    updated = _with_figure(_SAMPLE, r"([\d,]+) selected", 2171)

    assert "2,171 selected" in updated
    assert "2,151" not in updated
    # Untouched: the other figure, the prose, the trailing newline.
    assert "13 deselected" in updated
    assert "`ruff check .` clean" in updated
    assert updated.endswith("\n")
    assert updated.replace("2,171", "2,151") == _SAMPLE


def test_the_rewrite_groups_thousands_like_the_page_does():
    """A bare "2171" beside "2,151" would read as a typo, and the next reader
    would 'fix' it back."""
    assert "10,000 selected" in _with_figure(_SAMPLE, r"([\d,]+) selected", 10_000)
    # Under a thousand takes no separator.
    assert "999 selected" in _with_figure(_SAMPLE, r"([\d,]+) selected", 999)


def test_the_rewrite_refuses_a_line_it_cannot_find_the_figure_in():
    """Silently writing nothing would leave a stale figure looking refreshed."""
    with pytest.raises(AssertionError):
        _with_figure("no figures here\n", r"([\d,]+) selected", 5)


def test_the_rewrite_reaches_the_real_document(tmp_path, monkeypatch):
    """`_rewrite` finds the marker line among others and leaves them alone."""
    document = tmp_path / "deep-dive.md"
    document.write_text("# Title\n\nsome prose\n\n" + _SAMPLE + "\nafter\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "DEEP_DIVE", document)

    _rewrite(r"([\d,]+) selected", 2171)

    written = document.read_text(encoding="utf-8")
    assert "2,171 selected" in written
    assert written.startswith("# Title\n\nsome prose\n")
    assert written.endswith("\nafter\n")


def test_nothing_is_rewritten_unless_it_was_asked_for(monkeypatch):
    """The load-bearing half. CI sets nothing, so the guard must be strict on an
    unset variable — a self-healing check in CI would assert nothing at all."""
    monkeypatch.delenv(UPDATE_ENV, raising=False)
    assert _updating() is False

    # A leftover `=0` or `=false` in a shell profile must not disarm it either.
    for off in ("", "0", "false", "FALSE", "no", "  "):
        monkeypatch.setenv(UPDATE_ENV, off)
        assert _updating() is False, off

    for on in ("1", "true", "yes", "please"):
        monkeypatch.setenv(UPDATE_ENV, on)
        assert _updating() is True, on
