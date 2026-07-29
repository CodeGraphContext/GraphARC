"""Colour for the only reader who can see it: a person at a terminal.

Every helper here is the identity function unless the stream it is writing to is
an interactive terminal. That is not politeness, it is the design: this CLI's
human output is *pinned*. `tests/test_readme.py` byte-compares `grapharc plan`
against a fenced block in `README.md`, and `tests/test_cookbook_models.py` runs
`grapharc models` in a subprocess and byte-compares it against
`docs/cookbook/02-models.md`. Both read a pipe. So a pipe gets exactly the bytes
it got before this module existed, a terminal gets colour, and the two cannot
drift apart — because they are the same string with escape sequences added.

Four rules the helpers keep, so a call site cannot break that by accident:

- **Colour off is identity.** `ok("admitted") == "admitted"`, `kv("goal", g,
  width=10) == f"{'goal':<10}: {g}"`. A styled line and the line it replaced are
  the same object to `==`, to `grep`, and to a doc test.
- **Padding is never dyed.** `cell()` and `kv()` wrap the *text* in escapes and
  leave the spaces outside them, so a column is the same number of characters
  wide whether or not it is coloured, and a dyed cell cannot swallow its own
  padding into a colour a terminal renders differently.
- **Only text is coloured, never structure.** Every call site in `grapharc.cli`
  keeps the lines, words, glyphs and column widths it printed before — a
  terminal and a pipe differ in colour and in nothing else, so `README.md` is
  still a transcript of what you actually see. `rule()` and `bullet()` are the
  two helpers that would *draw* something; nothing emits them today, and a call
  site that starts to should emit them only when `enabled()`.
- **The terminal is measured only when colour is on.** A width read from
  `shutil.get_terminal_size()` in piped output would make a pinned test depend
  on the window that ran it.

Stdlib only, and free to import: `grapharc.cli.main` imports this at module
scope, and `tests/test_cli.py::test_building_the_parser_imports_no_optional_package`
pins that building the parser pulls in no optional package.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Iterable
from typing import TextIO

#: A tint: `(256-colour index, basic SGR parameter)`. Both are carried because a
#: 256-colour escape on a 16-colour terminal renders as nothing at all on some
#: emulators, and guessing wrong there costs the reader the text, not the colour.
Tone = tuple[int, str]

_RESET = "\x1b[0m"

# Mid-tone on purpose. A palette tuned for a dark background disappears on a
# light one, and plenty of terminals are light; these five stay legible on both.
# The warm orange is the accent the rest of the project is branded in.
ACCENT: Tone = (173, "33")  # warm orange — names, paths, the thing you asked for
OK: Tone = (71, "32")  # medium green — it worked, it was admitted, it is usable
WARN: Tone = (172, "33")  # amber — it ran and the answer is unwelcome
ERROR: Tone = (167, "31")  # soft red — it was refused, it failed, it is unusable
MUTED: Tone = (245, "2")  # mid grey — labels, units, prose about the output

#: The label column `plan`, `agent` and `run` share: they all print
#: `f"{key:<10}: value"`, and lining their blocks up with each other is the whole
#: reason the number is the same in three files. It is a constant and not a
#: measurement of the terminal because `README.md` and the cookbook print these
#: blocks verbatim and `tests/test_readme.py` byte-compares them.
LABEL_WIDTH = 10

#: Used by `rule()` when colour is off, instead of the terminal's width.
RULE_WIDTH = 60
#: The widest rule drawn on a terminal, however wide the window is: a line that
#: runs the full width of a 200-column window reads as a mistake.
RULE_MAX = 72

_RULE_GLYPH = "─"
_BULLET_GLYPH = "·"

#: `None` means "decide per stream"; `False` is `--no-color` (or `--json`, which
#: has no human reader). Set by `configure()`, read by `enabled()`.
_override: bool | None = None


# -- the decision -------------------------------------------------------------


def configure(*, no_color: bool = False) -> None:
    """Apply the colour decision the parsed flags made.

    Assigned on every call rather than latched, so one `main()` cannot decide the
    colour of the next one in the same interpreter — the test suite calls `main()`
    hundreds of times in one process, and a sticky flag set by whichever test ran
    first is a failure that only reproduces in a full run.
    """
    global _override
    _override = False if no_color else None


def enabled(stream: TextIO | None = None) -> bool:
    """Whether `stream` (default stdout) should be written with colour.

    Asked at print time, not at import time: under `pytest`'s `capsys` and under
    `contextlib.redirect_stdout` the answer changes after this module is loaded,
    and it has to be the current stream's answer or the pinned tests see escapes.

    `TERM=dumb` and `NO_COLOR` beat `FORCE_COLOR`, because refusing to colour is
    never the destructive mistake.
    """
    if _override is not None:
        return _override
    if os.environ.get("TERM") == "dumb" or "NO_COLOR" in os.environ:
        return False
    forced = os.environ.get("FORCE_COLOR")
    if forced is not None and forced != "0":
        return True
    try:
        return bool((sys.stdout if stream is None else stream).isatty())
    except (AttributeError, ValueError):
        # A stream that is closed, or is not a stream at all. Nobody is reading
        # it interactively; plain text is the safe answer.
        return False


def _depth() -> int:
    """256 when the terminal advertises it, else 16."""
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        return 256
    return 256 if "256" in os.environ.get("TERM", "") else 16


def _paint(text: str, tone: Tone, stream: TextIO | None = None) -> str:
    """`text` in `tone`, or `text`. Empty strings are never wrapped in escapes.

    That last part is load-bearing: `cell("", 12)` is how a continuation line
    keeps a column empty, and escapes around nothing would be escapes printed
    for nothing.
    """
    if not text or not enabled(stream):
        return text
    code = f"38;5;{tone[0]}" if _depth() == 256 else tone[1]
    return f"\x1b[{code}m{text}{_RESET}"


def _drawable(text: str, fallback: str, stream: TextIO | None = None) -> str:
    """`text` if the stream can encode it, else `fallback`.

    A box-drawing character is worth nothing if printing it raises
    `UnicodeEncodeError` and takes the whole command's output with it.
    """
    encoding = getattr(stream or sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return fallback
    return text


# -- semantic helpers ---------------------------------------------------------
#
# Named for what the text *means*, not for a colour: the call sites say `ok(…)`
# and `err(…)`, so the palette can be re-tuned here and nowhere else. Every one
# of them returns its argument unchanged when colour is off.


def accent(text: str, *, stream: TextIO | None = None) -> str:
    """A name the reader came for: a model spec, a path, a tool, a URL."""
    return _paint(text, ACCENT, stream)


def ok(text: str, *, stream: TextIO | None = None) -> str:
    """An affirmative outcome: admitted, usable, executed, the goal was met."""
    return _paint(text, OK, stream)


def warn(text: str, *, stream: TextIO | None = None) -> str:
    """It ran, and the answer is one the reader will not like."""
    return _paint(text, WARN, stream)


def err(text: str, *, stream: TextIO | None = None) -> str:
    """A refusal or a failure. Also the `error:` prefix on stderr."""
    return _paint(text, ERROR, stream)


def dim(text: str, *, stream: TextIO | None = None) -> str:
    """Secondary text: units, provenance, the sentence explaining the output."""
    return _paint(text, MUTED, stream)


def label(text: str, *, stream: TextIO | None = None) -> str:
    """The key of a `key: value` line — structure, not content."""
    return _paint(text, MUTED, stream)


def value(text: str, *, stream: TextIO | None = None) -> str:
    """The payload of a `key: value` line, deliberately left alone.

    It exists so a call site can say which half is the answer, and it returns the
    text unchanged in *both* modes on purpose: the payload is what the reader is
    actually here to read, and every colour available for it is a colour chosen
    against somebody's terminal theme. The default foreground is the one colour
    guaranteed to be legible.

    It takes (and ignores) `stream` so that it has the same signature as every
    other tint and can be handed to `kv(tint=…)` like one — which is exactly what
    `kv` does with it by default.
    """
    return text


def bold(text: str, *, stream: TextIO | None = None) -> str:
    """Weight rather than colour, for the rare line that has to win outright."""
    if not text or not enabled(stream):
        return text
    return f"\x1b[1m{text}{_RESET}"


def heading(text: str, *, stream: TextIO | None = None) -> str:
    """A section title inside an otherwise flat block, e.g. `examples:`."""
    if not text or not enabled(stream):
        return text
    code = f"1;38;5;{ACCENT[0]}" if _depth() == 256 else f"1;{ACCENT[1]}"
    return f"\x1b[{code}m{text}{_RESET}"


def tint_for(good: bool | None) -> Callable[..., str]:
    """The tint for a yes / no / nobody-checked outcome.

    One place decides that "usable" and "admitted" are the same green and that an
    unprobed backend is amber rather than red — three call sites each choosing
    for themselves is how a CLI ends up with two greens.
    """
    return {True: ok, False: err, None: warn}[good]


def rule(width: int | None = None, *, stream: TextIO | None = None) -> str:
    """A horizontal rule.

    The width comes from the terminal only when colour is on. Piped output gets
    `RULE_WIDTH`, a constant, because a rule whose length depends on the window
    that produced it is a diff waiting to happen in anything that captures it.
    """
    if width is None:
        if enabled(stream):
            width = min(shutil.get_terminal_size((RULE_WIDTH, 24)).columns, RULE_MAX)
        else:
            width = RULE_WIDTH
    return dim(_drawable(_RULE_GLYPH, "-", stream) * max(0, width), stream=stream)


def bullet(text: str, *, marker: str = _BULLET_GLYPH, stream: TextIO | None = None) -> str:
    """A list item: a dimmed marker, a space, then the text."""
    return f"{dim(_drawable(marker, '-', stream), stream=stream)} {text}"


# -- fixed-width blocks -------------------------------------------------------


def cell(
    text: str,
    width: int,
    *,
    tint: Callable[..., str] | None = None,
    stream: TextIO | None = None,
) -> str:
    """`text` padded to `width`, exactly as `f"{text:<width}"` would pad it.

    The padding is appended *after* the tint, never inside it, so the cell is
    `width` characters of text whether or not there are escapes around them.
    """
    painted = text if tint is None else tint(text, stream=stream)
    return f"{painted}{' ' * max(0, width - len(text))}"


def kv(
    key: str,
    text: str,
    *,
    width: int = 0,
    sep: str = ": ",
    tint: Callable[..., str] | None = None,
    key_tint: Callable[..., str] | None = None,
    stream: TextIO | None = None,
) -> str:
    """One `key: value` line whose plain form is `f"{key:<width}{sep}{text}"`.

    `width` is the caller's, not this module's: `plan`, `agent` and `run` line
    their labels up at ten characters, `serve` and a live `demo` at nine, and
    `demo`, `models` and `metrics` do not pad at all. Those widths are printed in
    the README and the cookbook, so they belong to the call site and this only
    reproduces them.
    """
    painted = (key_tint or label)(key, stream=stream)
    body = (tint or value)(text, stream=stream)
    return f"{painted}{' ' * max(0, width - len(key))}{sep}{body}"


def kv_block(
    pairs: Iterable[tuple[str, str]],
    *,
    width: int = 0,
    sep: str = ": ",
    stream: TextIO | None = None,
) -> list[str]:
    """`kv()` over a mapping's items, in order."""
    return [kv(k, v, width=width, sep=sep, stream=stream) for k, v in pairs]


def error_prefix(stream: TextIO | None = None) -> str:
    """The literal `error: ` every text-mode failure starts with, dyed on a tty.

    Gated on **stderr**, which is where it goes: `grapharc … 2> log` on a
    terminal must not put escape sequences in the log, and the substring
    assertions in `tests/test_cli.py` read a captured (non-tty) stderr.
    """
    return f"{err('error:', stream=stream or sys.stderr)} "


__all__ = [
    "ACCENT",
    "ERROR",
    "LABEL_WIDTH",
    "MUTED",
    "OK",
    "RULE_MAX",
    "RULE_WIDTH",
    "WARN",
    "Tone",
    "accent",
    "bold",
    "bullet",
    "cell",
    "configure",
    "dim",
    "enabled",
    "err",
    "error_prefix",
    "heading",
    "kv",
    "kv_block",
    "label",
    "ok",
    "rule",
    "tint_for",
    "value",
    "warn",
]
