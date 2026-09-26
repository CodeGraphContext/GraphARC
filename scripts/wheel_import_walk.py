"""Every module in the source tree is in the installed wheel, and imports.

Run against a *clean* environment that has `grapharc` installed from a built
wheel, from a working directory outside the checkout — otherwise `import
grapharc` can fall back to the source tree and pass for the wrong reason. This
file lives in `scripts/`, which contains no `grapharc` package, so running it by
path does not put the checkout on `sys.path` either.

Two jobs need exactly this check, which is why it is a file rather than a
heredoc:

- `build`, against the dependencies `uv.lock` pins — the shipped artifact is
  importable;
- `upstream-drift`, against dependencies re-resolved from `pyproject.toml` with
  no ceiling — the shipped artifact is *still* importable once upstream moves.
  That is the half issue #103 asked for: `uv.lock` hides a new major from every
  other job, because keeping development reproducible is the lockfile's whole
  job.

The comparison is against the checkout rather than a magic number. A `walked >
N` check cannot notice a whole subpackage going missing, and one did go missing
in testing: hatchling treats `.gitignore` as a build exclusion unless
`ignore-vcs` is set, and the build still succeeds.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-tree",
        required=True,
        type=Path,
        help="the checkout to compare against (the directory holding grapharc/)",
    )
    parser.add_argument(
        "--expect-prefix",
        required=True,
        help="a path fragment the imported package must come from, e.g. /tmp/wheelcheck/",
    )
    args = parser.parse_args()

    import grapharc

    if args.expect_prefix not in grapharc.__file__:
        return _fail(
            f"imported grapharc from {grapharc.__file__}, which is not under "
            f"{args.expect_prefix!r} — the installed wheel is not what was imported"
        )

    # The public entry points, named explicitly: these are what a reader of the
    # README types first, so a wheel that walks cleanly and cannot do these is
    # still broken.
    from grapharc import Budget, GraphARC, GraphARCState  # noqa: F401
    from grapharc.gateway import get_model  # noqa: F401
    from grapharc.harness import Harness  # noqa: F401

    source = args.source_tree / "grapharc"
    if not source.is_dir():
        return _fail(f"no grapharc package under {args.source_tree}")

    expected = {
        ".".join(("grapharc", *path.relative_to(source).parts))[: -len(".py")].removesuffix(
            ".__init__"
        )
        for path in source.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    installed = {module.name for module in pkgutil.walk_packages(grapharc.__path__, "grapharc.")}
    installed.add("grapharc")

    missing = sorted(expected - installed)
    if missing:
        return _fail(f"in the source tree but not in the wheel: {missing}")

    for name in sorted(installed):
        importlib.import_module(name)
    print(f"ok: {len(installed)} modules imported from wheel {grapharc.__version__}")
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
