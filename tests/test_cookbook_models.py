"""The models cookbook has to survive execution.

`docs/cookbook/02-models.md` makes claims by showing code and its output. A doc
that drifts from the library is the same failure as prose that overstates a
guarantee, so this file executes the page rather than trusting it: every marked
snippet is run in a subprocess and its captured output is compared, byte for
byte, against the block printed underneath it.

The contract is carried by HTML comments in the markdown, which are invisible
when rendered:

    <!-- verified -->            run this block; next ```text block must match exactly
    <!-- verified: varies -->    run it; it must exit 0, but the output is not compared
    <!-- verified: cli -->       a ```console block: `$ grapharc …`, output compared
    <!-- verified: cli varies -->  run it; machine-dependent output, not compared
    <!-- needs-credentials -->   never executed — costs money or quota

`test_every_executable_block_is_marked` is the load-bearing one: a ```python or
```console block added without a marker fails the suite, so a future edit cannot
quietly slip an unverified snippet onto the page. ```bash blocks are illustrative
commands (installs, live runs that cost money) and are deliberately exempt;
```text blocks are output.

Snippets run with the process environment's OpenRouter credentials removed and a
cwd outside the repo, so a machine that happens to have a key configured runs the
same page as a machine that does not.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from grapharc.gateway import DEFAULT_RETRY_POLICY, NO_RETRY
from grapharc.gateway.config import (
    OLLAMA_BASE_URL_KEYS,
    OLLAMA_KEYS,
    OPENAI_BASE_URL_KEYS,
    OPENAI_KEYS,
    OPENROUTER_KEYS,
)
from grapharc.gateway.registry import BACKENDS, DEFAULT_BACKEND
from grapharc.gateway.resilience import RETRYABLE_STATUS

DOC = Path(__file__).resolve().parents[1] / "docs" / "cookbook" / "02-models.md"

RUN_MARKERS = {
    "verified",
    "verified: varies",
    "verified: cli",
    "verified: cli varies",
    "needs-credentials",
}
# Languages whose blocks are executable claims and therefore need a marker.
EXECUTABLE_LANGS = {"python", "console"}

_FENCE = re.compile(r"^```(\w*)\s*$")
_COMMENT = re.compile(r"^<!--\s*(.*?)\s*-->$")

_HAS_OPENROUTER = importlib.util.find_spec("langchain_openai") is not None

# Stripped from every snippet's environment. A machine that happens to have a
# key — or an OLLAMA_HOST pointing somewhere unusual — must run the same page
# as a machine with nothing configured, or `grapharc models` prints a
# fingerprint where the page shows `<unset>`.
_CREDENTIAL_ENV = frozenset(
    (*OPENROUTER_KEYS, *OPENAI_KEYS, *OPENAI_BASE_URL_KEYS, *OLLAMA_KEYS, *OLLAMA_BASE_URL_KEYS)
)


@dataclass(frozen=True)
class Block:
    lang: str
    body: str
    line: int
    marker: str | None

    @property
    def ident(self) -> str:
        return f"{self.lang}@L{self.line}"


def _parse(text: str) -> list[Block]:
    """Fenced blocks in order, each carrying the marker comment above it."""
    blocks: list[Block] = []
    pending: str | None = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        comment = _COMMENT.match(line.strip())
        if comment:
            pending = comment.group(1)
            index += 1
            continue
        fence = _FENCE.match(line)
        if fence:
            lang = fence.group(1)
            start = index + 1
            end = start
            while end < len(lines) and not lines[end].startswith("```"):
                end += 1
            blocks.append(Block(lang, "\n".join(lines[start:end]), start + 1, pending))
            pending = None
            index = end + 1
            continue
        if line.strip():
            pending = None
        index += 1
    return blocks


BLOCKS = _parse(DOC.read_text(encoding="utf-8"))


def _expected_output(position: int) -> str:
    """The ```text block that follows the snippet at `position`."""
    following = BLOCKS[position + 1]
    assert following.lang == "text", (
        f"{BLOCKS[position].ident} is marked verified but is followed by a "
        f"```{following.lang} block, not ```text"
    )
    return following.body


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_ENV}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_python(body: str, workdir: Path) -> subprocess.CompletedProcess[str]:
    """Run a snippet the way a reader would, with stderr on the same stream.

    One snippet's real output includes a warning LangChain logs to stderr, and
    the page shows it in place. Merging at the file-descriptor level (rather
    than concatenating two captures) is what reproduces the interleaving; `-u`
    is what makes it deterministic.
    """
    script = workdir / "snippet.py"
    script.write_text(body, encoding="utf-8")
    return subprocess.run(  # noqa: S603 — argv array, no shell
        [sys.executable, "-u", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workdir,
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        timeout=180,
    )


def _run_console(body: str, workdir: Path) -> tuple[str, subprocess.CompletedProcess[str]]:
    """A ```console block: first line is `$ grapharc …`, the rest is its output."""
    first, _, expected = body.partition("\n")
    assert first.startswith("$ "), f"console block must start with '$ ': {first!r}"
    argv = shlex.split(first[2:])
    assert argv[0] == "grapharc", f"only the grapharc CLI is run from docs: {argv[0]!r}"
    proc = subprocess.run(  # noqa: S603 — argv array, no shell
        [sys.executable, "-m", "grapharc.cli.main", *argv[1:]],
        capture_output=True,
        text=True,
        cwd=workdir,
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        timeout=180,
    )
    return expected, proc


def _needs_openrouter(body: str) -> bool:
    """True for a snippet that cannot run without the langchain-openai extra.

    All three OpenAI-wire backends need it, not just OpenRouter — a snippet
    building `openai/…` or `ollama/…` fails the same way on a bare install.
    """
    return any(
        marker in body
        for marker in ("openrouter", "langchain_openai", '"openai/', '"ollama/')
    )


# --------------------------------------------------------------- the contract


def test_the_page_exists_and_has_blocks():
    assert BLOCKS, f"no fenced blocks found in {DOC}"


def test_every_executable_block_is_marked():
    """An unmarked python/console block would be an unverified claim on the page."""
    unmarked = [b.ident for b in BLOCKS if b.lang in EXECUTABLE_LANGS and b.marker is None]
    assert not unmarked, f"add a marker comment above: {unmarked}"


def test_no_block_carries_an_unknown_marker():
    bad = [(b.ident, b.marker) for b in BLOCKS if b.marker and b.marker not in RUN_MARKERS]
    assert not bad, f"unknown marker(s): {bad}"


def test_credentialed_snippets_claim_no_output():
    """A snippet nobody ran must not be followed by an output block."""
    for position, block in enumerate(BLOCKS):
        if block.marker != "needs-credentials":
            continue
        following = BLOCKS[position + 1] if position + 1 < len(BLOCKS) else None
        assert following is None or following.lang != "text", (
            f"{block.ident} needs credentials but shows an output block"
        )


# -------------------------------------------------------------- the snippets

_PY_EXACT = [i for i, b in enumerate(BLOCKS) if b.marker == "verified" and b.lang == "python"]
_PY_VARIES = [
    i for i, b in enumerate(BLOCKS) if b.marker == "verified: varies" and b.lang == "python"
]
_CLI_EXACT = [i for i, b in enumerate(BLOCKS) if b.marker == "verified: cli"]
_CLI_VARIES = [i for i, b in enumerate(BLOCKS) if b.marker == "verified: cli varies"]


def test_the_page_actually_verifies_something():
    """Guards against a parser change that silently selects nothing."""
    assert len(_PY_EXACT) >= 8
    assert _CLI_EXACT and _PY_VARIES


@pytest.mark.timeout(180)
@pytest.mark.parametrize("position", _PY_EXACT, ids=[BLOCKS[i].ident for i in _PY_EXACT])
def test_python_snippet_output_matches_the_page(position, tmp_path):
    block = BLOCKS[position]
    if _needs_openrouter(block.body) and not _HAS_OPENROUTER:
        pytest.skip("needs the openrouter extra")
    proc = _run_python(block.body, tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.rstrip("\n") == _expected_output(position).rstrip("\n")


@pytest.mark.timeout(180)
@pytest.mark.parametrize("position", _PY_VARIES, ids=[BLOCKS[i].ident for i in _PY_VARIES])
def test_python_snippet_with_varying_output_still_runs(position, tmp_path):
    block = BLOCKS[position]
    if _needs_openrouter(block.body) and not _HAS_OPENROUTER:
        pytest.skip("needs the openrouter extra")
    proc = _run_python(block.body, tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("position", _CLI_EXACT, ids=[BLOCKS[i].ident for i in _CLI_EXACT])
def test_cli_output_matches_the_page(position, tmp_path):
    expected, proc = _run_console(BLOCKS[position].body, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.rstrip("\n") == expected.rstrip("\n")


@pytest.mark.timeout(180)
@pytest.mark.parametrize("position", _CLI_VARIES, ids=[BLOCKS[i].ident for i in _CLI_VARIES])
def test_cli_command_with_machine_dependent_output_runs(position, tmp_path):
    """`models --check` reports this machine, so only its shape is asserted."""
    _, proc = _run_console(BLOCKS[position].body, tmp_path)
    assert "Traceback" not in proc.stderr
    assert proc.stdout.strip()


# --------------------------------------------------- prose that could rot ----


def test_the_backend_list_the_page_prints_is_the_real_one():
    page = DOC.read_text(encoding="utf-8")
    assert BACKENDS == ("claude-cli", "openrouter", "openai", "ollama", "mock")
    assert DEFAULT_BACKEND == "claude-cli"
    assert "There are five backends:" in page
    # Every backend has a row in the page's table. A backend added to the
    # gateway and not to the page fails here rather than going undocumented.
    for backend in BACKENDS:
        assert f"| `{backend}` |" in page, f"{backend} has no row in the table"


def test_the_ollama_default_endpoint_quoted_on_the_page_is_the_real_one():
    from grapharc.gateway.config import DEFAULT_OLLAMA_BASE_URL

    assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434/v1"
    assert f"base_url: {DEFAULT_OLLAMA_BASE_URL}" in DOC.read_text(encoding="utf-8")


def test_the_retry_defaults_quoted_on_the_page_are_the_real_ones():
    """The page says: 3 attempts, 0.5s initial, 20s cap, x2, 25% jitter."""
    assert DEFAULT_RETRY_POLICY.max_attempts == 3
    assert DEFAULT_RETRY_POLICY.initial_backoff_seconds == 0.5
    assert DEFAULT_RETRY_POLICY.max_backoff_seconds == 20.0
    assert DEFAULT_RETRY_POLICY.multiplier == 2.0
    assert DEFAULT_RETRY_POLICY.jitter == 0.25
    assert NO_RETRY.max_attempts == 1
    assert "3 attempts, 0.5s" in DOC.read_text(encoding="utf-8")


def test_the_retryable_status_codes_quoted_on_the_page_are_the_real_ones():
    """The page says: 408, 409, 425, 429, any 5xx."""
    assert RETRYABLE_STATUS == frozenset({408, 409, 425, 429})
    assert "408, 409, 425, 429" in DOC.read_text(encoding="utf-8")


def test_the_openrouter_defaults_quoted_on_the_page_are_the_real_ones():
    pytest.importorskip("langchain_openai", reason="needs the openrouter extra")
    from grapharc.gateway.openrouter import DEFAULT_MAX_TOKENS

    assert DEFAULT_MAX_TOKENS == 4096
    assert "defaults to **4096**" in DOC.read_text(encoding="utf-8")


def test_the_cli_timeout_default_quoted_on_the_page_is_the_real_one():
    from grapharc.gateway import ClaudeCodeCLIChatModel

    assert ClaudeCodeCLIChatModel().timeout_seconds == 600.0
    assert "`timeout_seconds` defaults\nto 600" in DOC.read_text(encoding="utf-8")
