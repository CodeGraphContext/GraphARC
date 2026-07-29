# Contributing to GraphARC

Thanks for looking. This file covers the three things that are easy to get
wrong here: how to run the tests, the rule that keeps live tests from spending
your money, and the standard this project holds documentation to.

## Setup

The toolchain is [uv](https://docs.astral.sh/uv/). `uv.lock` is committed, so
everyone resolves to the same versions.

```bash
git clone https://github.com/CodeGraphContext/GraphARC
cd GraphARC
uv sync --all-extras --group dev
```

`--all-extras` matters. Several tests skip themselves when an optional
dependency is missing, so syncing only the dev group silently shrinks the suite
instead of failing.

## Running the tests

```bash
uv run pytest                       # the whole suite
uv run pytest tests/test_server.py  # one file
uv run pytest -k retrieval          # one pattern
uv run pytest -x -q                 # stop at the first failure
```

Lint and autofix:

```bash
uv run ruff check . --fix
```

Both are what CI runs (`.github/workflows/ci.yml`), across Python 3.12, 3.13
and 3.14.

### The live-marker rule

**A test marked `live` calls a real model backend and spends real money.** Never
run `pytest -m live` casually, never in CI, and never because a failing test
looked like it might pass against a real model.

```bash
pytest            # 'not live' — every paid test is deselected
pytest -m live    # opt in, deliberately, with your own key
```

The mechanics, so you can tell when they are broken:

- `addopts` in `pyproject.toml` carries `-m 'not live'`. It must stay there.
  If you have a reason to change `addopts`, keep that filter.
- `addopts` also carries `--strict-markers`, and that is part of the same
  guarantee rather than a style preference. Without it a misspelled
  `@pytest.mark.lvie` is only a warning — the test does not match `not live`,
  is not deselected, and calls the API on a plain `pytest`. With it, the typo
  is a collection error.
- Every marker must therefore be registered in the `markers` table in
  `pyproject.toml` before you use it. `tests/test_packaging.py` fails if the
  tree uses one that is not.
- `required_plugins` names `pytest-asyncio` and `pytest-timeout`. Registering
  their markers locally would otherwise make a missing plugin silent — tests
  would collect with no timeout enforced at all.
- CI has a `live-marker-guard` job that compares the default selection against
  the `-m live` selection and fails if anything is in both.

If you add a test that talks to a real model:

```python
@pytest.mark.live
def test_something_against_a_real_backend():
    ...
```

Mark it, keep it out of any fixture the default suite touches, and say in the
test what it costs to run.

## Writing a change

- **One concern per PR.** A fix and the refactor around it are two PRs.
- **A bug fix comes with a test that fails without it.** Prove it: revert the
  fix, watch the new test go red, put the fix back. A test that passes either
  way is documentation, not a regression guard.
- **Never weaken an existing test to make a change pass.** If an assertion is
  genuinely wrong, say so in the PR and explain why in the same breath as
  changing it.
- **Keep `ruff check .` clean.** Line length is 100.
- **Note behaviour changes in the commit message** when you change behaviour,
  the public API, or what ships.

## The honesty rule for documentation

This is the one that gets enforced hardest, because this repo has broken it
before and the ROADMAP still carries a `!` legend entry for claims that shipped
while being false.

**A docstring, a README line, or a comment must not claim a guarantee the code
does not provide.** Not "aspirationally", not "once the TODO lands", not
because the happy path happens to hold.

Concretely:

- If a function is confined, sandboxed, budgeted, atomic, ordered or durable
  **only under conditions**, name the conditions in the docstring. `render_context`
  documents the single case where it overshoots `max_tokens`. `LocalExecutor` is
  named for what it does not do.
- If something is defense in depth rather than a boundary, say which.
  `SandboxedExecutor` is an audit-hook sandbox, not a kernel boundary, and its
  docstring says so.
- If a name oversells the thing — the way an exact-string match was once
  labelled "GraphRAG" — rename it or write down precisely what it does.
  `HashingEmbedder`'s docstring states that it is lexical, not semantic, and
  that anyone who assumes otherwise will be wrong.
- If an extra, a config key or a parameter exists but nothing implements it
  yet, say that where a reader will hit it. The optional-dependency table in
  `pyproject.toml` states, per extra, whether anything under `grapharc/`
  imports it today.
- When you fix something, fix the prose in the same commit. A stale docstring
  that used to be true is exactly as harmful as one that was never true.

The failure mode to avoid is confident prose over a guarantee the code does not
provide. A missing feature is fine. A feature that is documented as working and
is not will cost somebody a debugging session, or worse, will be trusted with
something it cannot hold.

If you cannot close a gap, write the gap down. `ROADMAP.md` has a
*Known gaps* section for exactly that, and `ROADMAP.md` tracks the rest.

## Packaging changes

`pyproject.toml` is the authority for what ships:

- `[tool.hatch.build.targets.sdist].include` is an allowlist, so a new top-level
  file that should ship has to be added there. `MANIFEST.in` mirrors that list
  for readers and tools; hatchling never reads it, and
  `tests/test_packaging.py` fails if the two drift.
- `[tool.hatch.build].ignore-vcs` is on deliberately. With hatchling's default
  a `.gitignore` entry doubles as a build exclusion — verified to drop the whole
  `grapharc/tools` subpackage out of the wheel with no error and a successful
  build. Junk you want kept out belongs in `[tool.hatch.build].exclude`.
- After a packaging change, build and install into a throwaway environment
  rather than trusting the build's exit code:

```bash
uv build
uv venv /tmp/check && uv pip install --python /tmp/check/bin/python "$(echo dist/*.whl)[all]"
cd /tmp && /tmp/check/bin/python -c "import grapharc, pkgutil, importlib
[importlib.import_module(m.name) for m in pkgutil.walk_packages(grapharc.__path__, 'grapharc.')]"
/tmp/check/bin/grapharc --version
```

CI does the same on every PR, for both the wheel and the sdist.

## Releasing

Tag-driven. `.github/workflows/release.yml` refuses a tag that disagrees with
the version in `pyproject.toml`, builds, verifies the artifacts in clean
environments, and publishes through PyPI Trusted Publishing. There is no API
token in this repository or in its secrets.

1. Bump `version` in `pyproject.toml` **and** `__version__` in
   `grapharc/__init__.py`. CI fails if the two disagree.
2. Tag the release; the commit log is the record of what changed.
3. Tag `vX.Y.Z` and push it.
