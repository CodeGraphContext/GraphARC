"""One payload, two renderings: a person at a terminal and a script parsing stdout.

Every command builds one result dict and hands it here together with a text
view of that same dict. The JSON *is* the payload; the text is a view over it,
usually a shorter one — so the two can differ in how much they show and cannot
differ about what happened. Two independently maintained renderings drift.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from grapharc.cli import style

EXIT_OK = 0
# The command ran and the answer is negative: an agent that stopped short of its
# target, a run id with no events, a probe that found no usable backend.
EXIT_FAILED = 1
# The command could not run at all: a component that is not installed, an
# unreadable trace, a model spec that names no backend.
EXIT_UNAVAILABLE = 2


def jsonable(value: Any) -> Any:
    """Convert to something `json.dumps` accepts.

    Unknown objects degrade to their `repr` rather than raising: a CLI that
    crashes while printing a result has destroyed the result it was reporting.
    """
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int, float)):
        return value
    return repr(value)


def dumps(payload: Any) -> str:
    return json.dumps(jsonable(payload), indent=2, ensure_ascii=False)


def emit(
    payload: Mapping[str, Any],
    lines: Iterable[str] = (),
    *,
    as_json: bool,
    stream: TextIO | None = None,
) -> None:
    """Print the payload as JSON, or the text view of it."""
    out = stream or sys.stdout
    if as_json:
        print(dumps(payload), file=out)
        return
    for line in lines:
        print(line, file=out)


def fail(
    message: str,
    *,
    as_json: bool,
    command: str,
    code: int = EXIT_UNAVAILABLE,
    **extra: Any,
) -> int:
    """Report a command that produced no result, and return its exit code.

    Text errors go to stderr so `grapharc … > out` still shows the failure;
    JSON errors go to stdout so the one document a script parses is the one
    that says what went wrong.

    The message body is never wrapped, dyed or re-flowed — only the `error: `
    prefix is, and only on a terminal. Callers grep stderr for phrases the
    message contains (`tests/test_cli.py` asserts on `"not a valid topology"`,
    `"no such trace file"`, `"grapharc demo stage0"`), and a hard break inserted
    at column 80 would cut one of them in half.
    """
    payload = {"ok": False, "command": command, "error": message, **extra}
    if as_json:
        print(dumps(payload))
    else:
        print(f"{style.error_prefix()}{message}", file=sys.stderr)
    return code


__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "dumps",
    "emit",
    "fail",
    "jsonable",
]
