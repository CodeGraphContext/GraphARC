"""Guards on the things that decide what ships and what a plain `pytest` runs.

Two defects were reproduced here before these tests existed, both silent:

1. An unregistered marker was only a warning, so `@pytest.mark.<typo>` on a paid
   test did not match `-m 'not live'`, was not deselected, and called a real API
   on a plain `pytest`. The CI `live-marker-guard` does not catch it either: a
   typo'd test is missing from *both* the default and the `-m live` selection,
   so the intersection it compares stays empty and the job passes.
2. Hatchling treats `.gitignore` as a build exclusion by default. Adding
   `grapharc/tools/` to `.gitignore` dropped the whole subpackage out of the
   wheel with no error, no warning, and a build that reported success.

Everything below is cheap except `test_wheel_ships_every_file_in_the_package`,
which does a real `uv build` (~0.3s warm).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
PACKAGE = REPO / "grapharc"

# Shipped by pytest itself or by a plugin's own registration; a test may use
# these without them appearing in the `markers` table.
BUILTIN_MARKERS = frozenset(
    {
        "filterwarnings",
        "parametrize",
        "skip",
        "skipif",
        "usefixtures",
        "xfail",
    }
)

# Junk that must never reach an artifact. Anything matching these is expected to
# be absent from the wheel and is not compared against the source tree.
JUNK_SUFFIXES = (".pyc", ".pyo", ".pyd", ".sqlite", ".jsonl")


def config() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def pytest_options() -> dict:
    return config()["tool"]["pytest"]["ini_options"]


def registered_markers() -> set[str]:
    return {entry.split(":", 1)[0].strip() for entry in pytest_options()["markers"]}


def shipped_files() -> set[str]:
    """Every file under `grapharc/` that a wheel is expected to carry."""
    found = set()
    for path in PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.name.endswith(JUNK_SUFFIXES):
            continue
        found.add(path.relative_to(REPO).as_posix())
    return found


# --------------------------------------------------------------------------
# The live marker: a broken guard here spends real money.
# --------------------------------------------------------------------------


def test_live_tests_are_deselected_by_default():
    """`addopts` must keep filtering out the marker that costs money."""
    assert "-m 'not live'" in pytest_options()["addopts"]
    assert "live" in registered_markers()


def test_strict_markers_is_on_so_a_typo_cannot_smuggle_a_live_test_through():
    assert "--strict-markers" in pytest_options()["addopts"]


def test_a_misspelled_marker_is_a_collection_error(tmp_path):
    """The behavioural half of the guard above, run against the real config.

    Without `--strict-markers` this test file collects, the bogus marker is a
    `PytestUnknownMarkWarning`, and the test *runs* under `-m 'not live'`.
    """
    bogus = "l" + "vie"  # assembled so the marker scan below cannot see it
    assert bogus not in registered_markers()
    (tmp_path / "test_typo.py").write_text(
        "import pytest\n\n\n"
        f"@pytest.mark.{bogus}\n"
        "def test_spends_money():\n"
        "    raise AssertionError('a paid API was called')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYPROJECT),
            "--rootdir",
            str(tmp_path),
            "-p",
            "no:cacheprovider",
            str(tmp_path / "test_typo.py"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "a paid API was called" not in result.stdout
    assert bogus in result.stdout


def test_every_marker_the_tree_uses_is_registered():
    """A marker nobody registered is a marker `-m` cannot filter on."""
    used: dict[str, set[str]] = {}
    for path in sorted([*(REPO / "tests").rglob("*.py"), *PACKAGE.rglob("*.py")]):
        if "__pycache__" in path.parts:
            continue
        for name in re.findall(r"@pytest\.mark\.(\w+)", path.read_text(encoding="utf-8")):
            used.setdefault(name, set()).add(path.relative_to(REPO).as_posix())

    known = registered_markers() | BUILTIN_MARKERS
    unknown = {name: sorted(files) for name, files in used.items() if name not in known}
    assert not unknown, f"unregistered markers in use: {unknown}"


def test_plugins_that_give_registered_markers_meaning_are_required():
    """Registering `timeout`/`asyncio` here would otherwise mask a missing plugin.

    With them in the `markers` table, `--strict-markers` stops complaining — so
    without `pytest-timeout` installed, `@pytest.mark.timeout(5)` would be
    accepted and enforce nothing. `required_plugins` makes pytest refuse to
    start instead.
    """
    required = {
        entry.split(">")[0].split("=")[0].strip()
        for entry in pytest_options()["required_plugins"]
    }
    assert {"pytest-asyncio", "pytest-timeout"} <= required


# --------------------------------------------------------------------------
# What ships.
# --------------------------------------------------------------------------


def test_ignore_vcs_is_on_so_gitignore_cannot_drop_a_subpackage():
    """Reproduced: `grapharc/tools/` in `.gitignore` emptied it out of the wheel.

    Hatchling's default is `ignore-vcs = false`, which means VCS ignore rules
    double as build exclusions. The build still reports success, so the first
    sign of trouble is an `ImportError` in somebody else's environment.
    """
    assert config()["tool"]["hatch"]["build"]["ignore-vcs"] is True


def test_no_shipped_source_file_is_git_ignored():
    """A second, independent guard, for a failure `ignore-vcs` does not cover.

    A git-ignored source file is not committed, so it is missing from a clone
    and from GitHub's tarball even though a local build happens to include it.
    """
    if shutil.which("git") is None:  # pragma: no cover - git is present in CI
        pytest.skip("git is not installed")
    paths = sorted(shipped_files())
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, f"these shipped files are git-ignored: {ignored}"


def test_wheel_packages_config_covers_the_whole_package():
    wheel = config()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["grapharc"]
    # A target-level key of either name would silently override the recursive
    # `packages` above and could drop a subpackage.
    assert "only-include" not in wheel
    assert "exclude" not in wheel


@pytest.mark.timeout(300)
def test_wheel_ships_every_file_in_the_package(tmp_path):
    """Build for real and compare against the tree.

    Skipped when `uv` is absent; the `build` job in `.github/workflows/ci.yml`
    runs the equivalent comparison and cannot skip, so CI still enforces it.
    """
    if shutil.which("uv") is None:  # pragma: no cover - uv is present in CI
        pytest.skip("uv is not installed; CI's build job covers this")

    out = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", str(REPO), "--out-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=280,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    (built,) = out.glob("*.whl")
    names = set(zipfile.ZipFile(built).namelist())
    packaged = {name for name in names if name.startswith("grapharc/")}

    missing = sorted(shipped_files() - packaged)
    assert not missing, f"in the tree but not in the wheel: {missing}"

    junk = sorted(n for n in packaged if "__pycache__" in n or n.endswith(JUNK_SUFFIXES))
    assert not junk, f"junk in the wheel: {junk}"

    # Named explicitly: these landed after the packaging config was written and
    # are the subpackages a stale `packages` list would have missed.
    for subpackage in ("planner", "policy", "server", "session", "tools"):
        assert f"grapharc/{subpackage}/__init__.py" in packaged

    # Non-Python package data: the live view's frontend is real files, and a
    # wheel that drops them serves a 500 where the dashboard should be.
    for asset in ("view.html", "view.css", "view.js", "index.html", "signin.html"):
        assert f"grapharc/server/static/{asset}" in packaged


def test_manifest_in_mirrors_the_hatch_sdist_allowlist():
    """MANIFEST.in is not read by hatchling, so nothing else stops it lying."""
    build = config()["tool"]["hatch"]["build"]
    expected = set()
    for entry in build["targets"]["sdist"]["include"]:
        relative = entry.lstrip("/")
        verb = "graft" if (REPO / relative).is_dir() else "include"
        expected.add(f"{verb} {relative}")
    for pattern in build["exclude"]:
        expected.add(f"global-exclude {pattern.removeprefix('**/')}")

    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    directives = {
        " ".join(line.split())
        for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert directives == expected


def test_project_urls_name_files_that_exist():
    """A dead link in this table is a dead link on the PyPI project page."""
    for label, url in config()["project"]["urls"].items():
        _, _, tail = url.partition("/blob/main/")
        if not tail:
            continue
        assert (REPO / tail).exists(), f"[project.urls] {label} points at missing {tail}"


def test_the_all_extra_names_every_other_extra():
    extras = config()["project"]["optional-dependencies"]
    named = set(re.findall(r"\[([^\]]+)\]", extras["all"][0])[0].split(","))
    assert named == set(extras) - {"all"}


def test_the_declared_version_matches_the_package():
    import grapharc

    assert grapharc.__version__ == config()["project"]["version"]


def test_the_sdist_ships_the_docs_the_tests_read():
    """`tests/` shipped and `docs/` did not, so six `test_cookbook_*` modules —
    which open their page at import time — turned `pytest` inside an unpacked
    sdist into five collection errors. The suite could not run from a release."""
    include = config()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert "/docs" in include
    assert "graft docs" in (REPO / "MANIFEST.in").read_text(encoding="utf-8")
