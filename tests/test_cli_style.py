"""The CLI is styled on a terminal and byte-identical when piped.

That second half is load-bearing rather than cosmetic. `tests/test_readme.py` and
`tests/test_cookbook_models.py` byte-compare command output against fenced blocks
in `README.md` and `docs/cookbook/02-models.md`, and both harnesses present a
non-tty stdout — so those pages only stay true while piped output carries no
styling at all. Nothing else asserts that, which means a single call site that
painted unconditionally, or a helper that stopped consulting `isatty()`, would
leak escapes into the docs' own evidence and no test would notice.

The gate here is the strong form of the property: strip the escapes from what a
terminal receives and it must equal what a pipe receives, exactly. That catches a
leak in either direction — an escape reaching a pipe, and a tty-only change to
*layout* rather than colour, which would make the pages describe output no user
sees.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "openpty"), reason="needs a pty, which Windows has no equivalent for"
)

# Commands with styled human-mode output. Exit codes are deliberately not pinned
# here: `models --check` exits 1 when the host can reach no real provider, which
# is correct and is what a machine with no credentials does.
STYLED = [
    pytest.param(["plan", "investigate the checkout outage", "--scripted"], id="plan"),
    pytest.param(["models"], id="models"),
    pytest.param(["models", "--check"], id="models-check"),
    pytest.param(["demo", "stage0"], id="demo-stage0"),
]

# The subset whose output is reproducible enough to compare byte-for-byte across
# two separate invocations. `models --check` is excluded on purpose: it probes the
# host — a `claude` binary on PATH, a key in the environment, a socket to a local
# ollama — so two runs are not guaranteed to agree, and comparing them would buy a
# flake rather than a guarantee. `docs/cookbook/02-models.md` marks that command as
# varying for the same reason; this follows that judgement rather than contradicting it.
COMPARABLE = [param for param in STYLED if param.id != "models-check"]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Each run mints its own scratch directory, so the path is not a property of the
# output. Normalised rather than excluded, so a *missing* path still fails.
# Built from `gettempdir()` rather than a literal "/tmp", because a runner that
# sets TMPDIR elsewhere would otherwise leave the paths unnormalised and the
# comparison below would fail for a reason that has nothing to do with styling.
_TMPDIR = re.compile(re.escape(tempfile.gettempdir()) + r"/grapharc-[A-Za-z0-9_.-]+")
# The default trace directory stamp: two invocations of one command are two
# runs with two stamps, and the comparison is about styling, not clocks.
_RUNDIR = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")


def _env(**extra: str) -> dict[str, str]:
    """A parent-independent environment, so a developer's own TERM or NO_COLOR
    cannot decide what this test proves."""
    env = {k: v for k, v in os.environ.items() if k not in {"NO_COLOR", "FORCE_COLOR"}}
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = "100"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def _argv(args: list[str]) -> list[str]:
    # `-m` rather than the console script: the entry point is not guaranteed to be
    # on PATH in a bare checkout, and the module path is what the other CLI tests use.
    return [sys.executable, "-m", "grapharc.cli.main", *args]


def _piped(args: list[str], **extra: str) -> tuple[str, str, int]:
    proc = subprocess.run(  # noqa: S603 — argv array, no shell
        _argv(args),
        capture_output=True,
        text=True,
        env=_env(**extra),
        stdin=subprocess.DEVNULL,
        timeout=180,
    )
    return proc.stdout, proc.stderr, proc.returncode


def _on_pty(args: list[str], **extra: str) -> tuple[str, int]:
    """Run with stdout *and* stderr on one pty and return everything written.

    Both on the same pty because that is the shape a person sees, and because it
    catches an escape written to stderr while stdout happens to be a terminal.
    """
    primary, secondary = os.openpty()
    proc = subprocess.Popen(  # noqa: S603 — argv array, no shell
        _argv(args),
        stdout=secondary,
        stderr=secondary,
        stdin=subprocess.DEVNULL,
        env=_env(**extra),
    )
    os.close(secondary)
    chunks: list[bytes] = []
    selector = selectors.DefaultSelector()
    selector.register(primary, selectors.EVENT_READ)
    try:
        while selector.select(timeout=180):
            try:
                data = os.read(primary, 65536)
            except OSError:  # the child closed its end
                break
            if not data:
                break
            chunks.append(data)
    finally:
        selector.close()
        proc.wait(timeout=180)
        os.close(primary)
    # A pty turns "\n" into "\r\n"; that is the terminal's doing, not the CLI's.
    text = b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n")
    return text, proc.returncode


def _normalise(text: str) -> str:
    text = _TMPDIR.sub("/tmp/grapharc-NORMALISED", text)
    return _RUNDIR.sub("RUNDIR-NORMALISED", text)


@pytest.mark.parametrize("args", STYLED)
def test_a_terminal_gets_escapes_and_a_pipe_gets_none(args):
    """Both halves in one test: styling must be real, and confined to a terminal."""
    on_pty, pty_code = _on_pty(args)
    out, err, piped_code = _piped(args)

    # Not a fixed value — whether the command *succeeds* is the host's business
    # (`models --check` exits 1 where no real provider is reachable). What must
    # hold is that being watched by a terminal does not change the answer.
    assert pty_code == piped_code, "styling changed the exit code"
    assert _ANSI.search(on_pty), "a terminal received no styling at all"
    assert "\x1b" not in out, "an escape reached piped stdout"
    assert "\x1b" not in err, "an escape reached piped stderr"


@pytest.mark.parametrize("args", COMPARABLE)
def test_stripping_the_escapes_reproduces_the_piped_output_exactly(args):
    """The property the byte-compared doc pages depend on.

    Colour must be the *only* difference between what a terminal shows and what a
    pipe carries. A tty-only change to spacing, alignment or line count would pass
    the leak check above and still make README describe output nobody sees.
    """
    on_pty, _ = _on_pty(args)
    out, err, _ = _piped(args)

    stripped = _normalise(_ANSI.sub("", on_pty))
    assert stripped == _normalise(out + err)


@pytest.mark.parametrize(
    ("label", "args", "extra"),
    [
        ("NO_COLOR", ["plan", "x", "--scripted"], {"NO_COLOR": "1"}),
        ("TERM=dumb", ["plan", "x", "--scripted"], {"TERM": "dumb"}),
        ("--no-color", ["plan", "x", "--scripted", "--no-color"], {}),
        ("--json", ["plan", "x", "--scripted", "--json"], {}),
    ],
)
def test_every_opt_out_silences_styling_even_on_a_terminal(label, args, extra):
    on_pty, code = _on_pty(args, **extra)

    assert code == 0
    assert "\x1b" not in on_pty, f"{label} did not silence styling"


def test_json_on_a_terminal_is_still_one_clean_document():
    """`--json` promises a parseable document on stdout and nothing on stderr.

    Asserted on a *pty* specifically: that is the one condition under which a
    renderer might decide it is allowed to decorate.
    """
    import json

    out, err, _ = _piped(["plan", "x", "--scripted", "--json"])
    assert err == ""
    assert json.loads(out)["ok"] is True

    on_pty, _ = _on_pty(["plan", "x", "--scripted", "--json"])
    assert "\x1b" not in on_pty
    assert json.loads(on_pty)["ok"] is True


def test_viz_on_a_terminal_stays_pasteable_mermaid():
    """Being pasteable into a Mermaid renderer is the only reason `viz` exists, so
    it is the one command that must not be styled even on a terminal."""
    out, _, _ = _piped(["demo", "stage0"])
    trace = next(
        line.split(": ", 1)[1].strip() for line in out.splitlines() if line.startswith("trace: ")
    )
    with open(trace, encoding="utf-8") as handle:
        run_id = __import__("json").loads(handle.readline())["run_id"]

    on_pty, code = _on_pty(["viz", trace, run_id])

    assert code == 0
    assert "\x1b" not in on_pty
    assert on_pty.lstrip().startswith("flowchart TD")


def test_a_text_mode_failure_keeps_stdout_empty_and_says_error_on_stderr():
    """The failure contract, checked with styling live on stderr's own terminal."""
    out, err, code = _piped(["metrics", "/nope/none.jsonl", "r1"])

    assert code == 2
    assert out == ""
    assert err.startswith("error: ")
    assert "no such trace file" in err

    on_pty, pty_code = _on_pty(["metrics", "/nope/none.jsonl", "r1"])
    assert pty_code == 2
    # Styled here, but the message body must survive intact for the substring
    # assertions elsewhere in the suite.
    assert "no such trace file" in _ANSI.sub("", on_pty)
