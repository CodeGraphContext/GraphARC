"""Executable checks for `docs/cookbook/03-agents-and-tools.md`.

The cookbook's one promise is that every snippet on the page runs exactly as
printed. That is a property of the docs, so it is enforced the same way every
other property in this repo is: by running them.

Each snippet is preceded in the markdown by an HTML comment — invisible when
rendered — naming the file it should be saved as:

    <!-- cookbook name=04-agent-loop.py -->

Attributes: `name=` the filename, `shell` to run the block with bash instead of
Python, `requires=docker` to skip when no container runtime answers, `compare=off`
to run the snippet but not diff its output (used where the output legitimately
depends on the machine), and `norun="reason"` for a block that is never executed
and is labelled as such on the page.

Every snippet is written into one shared directory before any of them runs, so
`from cookbook_model import ToolScriptedModel` resolves for the reader exactly as
it does here.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from grapharc.harness.container import runtime_available

DOC = Path(__file__).resolve().parent.parent / "docs" / "cookbook" / "03-agents-and-tools.md"

# `<!-- cookbook ... -->`, the fenced snippet, and optionally the `Output:` block
# that follows it. Non-greedy bodies: a fenced block never contains a fence.
_BLOCK = re.compile(
    r"<!-- cookbook (?P<attrs>[^>]*?) -->\n"
    r"```(?P<lang>[a-z]*)\n(?P<code>.*?)\n```"
    r"(?:\n\nOutput:\n\n```\n(?P<output>.*?)\n```)?",
    re.DOTALL,
)
# Wall-clock durations that `run_command` reports. The only volatile fragment
# left in any recorded output, and it is masked rather than hidden.
_DURATION = re.compile(r"after \d+\.\d+s")


class Snippet:
    def __init__(self, attrs: dict[str, str], lang: str, code: str, output: str | None) -> None:
        self.attrs = attrs
        self.lang = lang
        self.code = code
        self.output = output

    @property
    def name(self) -> str:
        return self.attrs.get("name", "")

    @property
    def id(self) -> str:
        return self.name or f"norun:{self.attrs.get('norun', '?')}"

    def __repr__(self) -> str:  # pragma: no cover - pytest ids use `id`
        return f"Snippet({self.id})"


def _parse_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for token in shlex.split(raw):
        key, _, value = token.partition("=")
        attrs[key] = value
    return attrs


def _snippets() -> list[Snippet]:
    text = DOC.read_text(encoding="utf-8")
    return [
        Snippet(_parse_attrs(m.group("attrs")), m.group("lang"), m.group("code"), m.group("output"))
        for m in _BLOCK.finditer(text)
    ]


SNIPPETS = _snippets()
EXECUTABLE = [s for s in SNIPPETS if "norun" not in s.attrs]
NOT_RUN = [s for s in SNIPPETS if "norun" in s.attrs]


def _normalise(text: str) -> str:
    """Strip trailing whitespace and mask the one volatile fragment on the page."""
    lines = [line.rstrip() for line in _DURATION.sub("after <t>s", text).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


@pytest.fixture(scope="module")
def cookbook_dir(tmp_path_factory) -> Path:
    """Every snippet saved side by side, as a reader following the page would."""
    directory = tmp_path_factory.mktemp("cookbook-agents")
    for snippet in EXECUTABLE:
        (directory / snippet.name).write_text(snippet.code + "\n", encoding="utf-8")
    return directory


def _run(snippet: Snippet, directory: Path) -> subprocess.CompletedProcess[str]:
    if "shell" in snippet.attrs:
        # The page says `grapharc ...`; make the console script from this
        # interpreter's environment the one that resolves.
        env = dict(os.environ)
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        argv = ["bash", str(directory / snippet.name)]
    else:
        env = dict(os.environ)
        argv = [sys.executable, snippet.name]
    return subprocess.run(
        argv, cwd=directory, capture_output=True, text=True, timeout=600, env=env, check=False
    )


# -- the page's own claims ----------------------------------------------------


def test_every_fenced_snippet_is_declared():
    """No python or bash block may sit on the page without a cookbook marker.

    Without this, adding an unverified snippet later is a one-line edit that
    nothing catches — which is exactly the failure this file exists to prevent.
    Bare ``` blocks are recorded output and illustrations, and are not code.
    """
    text = DOC.read_text(encoding="utf-8")
    fenced = re.findall(r"^```(python|bash)$", text, re.MULTILINE)
    assert len(fenced) == len(SNIPPETS), (
        f"{len(fenced)} python/bash blocks on the page but {len(SNIPPETS)} carry a "
        "`<!-- cookbook ... -->` marker; every one must be declared"
    )


def test_unrun_snippets_are_labelled():
    """A block nobody ran has to say why, and the page has to say how many."""
    assert NOT_RUN, "the page claims two unrun snippets; the markers say none"
    for snippet in NOT_RUN:
        assert snippet.attrs["norun"], f"{snippet.id} is marked norun with no reason"
        assert snippet.output is None, f"{snippet.id} was never run but records an output block"
    assert len(NOT_RUN) == 2, (
        f"the page's preamble says two snippets are not run; the markers say {len(NOT_RUN)}"
    )


def test_executable_snippets_are_named():
    for snippet in EXECUTABLE:
        assert snippet.name, f"executable snippet has no name= attribute: {snippet.code[:60]!r}"
    names = [s.name for s in EXECUTABLE]
    assert len(names) == len(set(names)), f"duplicate snippet filenames: {names}"


# -- the snippets themselves --------------------------------------------------


@pytest.mark.parametrize("snippet", EXECUTABLE, ids=[s.id for s in EXECUTABLE])
def test_snippet_runs_as_printed(snippet: Snippet, cookbook_dir: Path):
    if snippet.attrs.get("requires") == "docker" and not runtime_available():
        pytest.skip("no container runtime answers on this machine")
    if "shell" in snippet.attrs and not (Path(sys.executable).parent / "grapharc").exists():
        pytest.skip("the `grapharc` console script is not installed in this environment")

    completed = _run(snippet, cookbook_dir)
    # `exit=` is for the one snippet whose whole point is a non-zero exit code;
    # everything else has to succeed.
    expected = int(snippet.attrs.get("exit", 0))
    assert completed.returncode == expected, (
        f"{snippet.name} exited {completed.returncode}, expected {expected}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    if snippet.output is None or snippet.attrs.get("compare") == "off":
        return
    assert _normalise(completed.stdout) == _normalise(snippet.output), (
        f"{snippet.name} printed something other than the output recorded on the page\n"
        f"--- stderr ---\n{completed.stderr}"
    )
