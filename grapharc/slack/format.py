"""One `CommandResult` becomes one Slack message.

The CLI's exit codes are part of its interface (0 did its job, 1 ran and the
answer was negative, 2 could not run), so the header states which of the three
happened instead of a bare number. The captured output is posted verbatim in a
code fence — piped-mode bytes are the CLI's stable form, and inventing a Slack
rendering of them would be a third dialect the docs never promised.

Slack rejects messages past 40,000 characters; the fence is truncated well
below that, from the top, with a line saying how much was cut. Truncation is
announced, never silent — the reader must know they are not seeing everything.
"""

from __future__ import annotations

import shlex

from grapharc.slack.runner import CommandResult

# Leaves generous room for the header and the truncation notice.
MAX_FENCE_CHARS = 3500

_VERDICTS = {
    0: "did its job",
    1: "ran; the answer was negative",
    2: "could not run",
}


def _fence(body: str) -> str:
    # A ``` inside the body would end the fence early and spill the rest as
    # prose; a zero-width space between the backticks defuses it.
    return "```" + body.replace("```", "`​``") + "```"


def _truncate(body: str) -> tuple[str, int]:
    if len(body) <= MAX_FENCE_CHARS:
        return body, 0
    return body[:MAX_FENCE_CHARS], len(body) - MAX_FENCE_CHARS


def format_result(result: CommandResult) -> str:
    shown = shlex.join(["grapharc", *result.argv])
    if result.exit_code is None:
        header = (
            f"`{shown}` was still running after {result.timeout_seconds:.0f}s "
            "and was stopped."
        )
    else:
        verdict = _VERDICTS.get(result.exit_code, f"exited {result.exit_code}")
        header = f"`{shown}` — {verdict} ({result.duration_seconds:.1f}s)."

    # Text-mode contract: success speaks on stdout, failure on stderr with an
    # empty stdout. Show whichever stream carries the answer; both if both do.
    parts = [header]
    for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not stream.strip():
            continue
        body, cut = _truncate(stream)
        if len(parts) > 1 or label == "stderr":
            parts.append(f"{label}:")
        parts.append(_fence(body))
        if cut:
            parts.append(f"_…{cut} more characters not shown._")
    if len(parts) == 1:
        parts.append("_(no output)_")
    return "\n".join(parts)
