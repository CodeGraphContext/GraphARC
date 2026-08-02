"""One `CommandResult` becomes one Slack message.

The CLI's exit codes are part of its interface (0 did its job, 1 ran and the
answer was negative, 2 could not run), so the header states which of the three
happened instead of a bare number. The captured output is posted verbatim in a
code fence — piped-mode bytes are the CLI's stable form, and inventing a Slack
rendering of them would be a third dialect the docs never promised.

Slack rejects messages past 40,000 characters; the fence is truncated well
below that, keeping the tail (where a traceback's actual error lives), with a
line saying how much was cut. Truncation is
announced, never silent — the reader must know they are not seeing everything.
"""

from __future__ import annotations

import base64
import json
import shlex
import zlib
from pathlib import Path
from urllib.parse import quote

from grapharc.slack.command import trace_path
from grapharc.slack.runner import CommandResult

# Leaves generous room for the header and the truncation notice.
MAX_FENCE_CHARS = 3500

_VERDICTS = {
    0: "did its job",
    1: "ran; the answer was negative",
    2: "could not run",
}


def fence(body: str) -> str:
    # A ``` inside the body would end the fence early and spill the rest as
    # prose; a zero-width space between the backticks defuses it.
    return "```" + body.replace("```", "`​``") + "```"


def truncate(body: str) -> tuple[str, int]:
    """Clip to the fence budget, keeping the TAIL.

    The end is where the information lives: a traceback's actual error is its
    last line, and a live feed's most recent events are at the bottom. Keeping
    the head cut exactly the part the reader needed.
    """
    if len(body) <= MAX_FENCE_CHARS:
        return body, 0
    return body[-MAX_FENCE_CHARS:], len(body) - MAX_FENCE_CHARS


def mermaid_live_url(code: str) -> str:
    """A mermaid.live link with the whole diagram compressed into the fragment.

    The editor's `#pako:` form: zlib-deflated JSON, base64url. Everything after
    the `#` is a URL fragment, which a browser never sends to the server — the
    site's JavaScript renders the diagram locally, so following the link ships
    the diagram to no one.
    """
    payload = json.dumps({"code": code, "mermaid": {"theme": "default"}})
    packed = base64.urlsafe_b64encode(zlib.compress(payload.encode())).decode()
    return f"https://mermaid.live/view#pako:{packed}"


def live_view_url(argv: list[str], *, base: str | None, workdir: Path) -> str | None:
    """The live-view page for an admitted argv, or None.

    None when the operator configured no base URL, or the argv is not a
    tracing command. The gate injects `--trace` into every tracing argv it
    admits, so for those the path is always present; it is keyed relative to
    the workdir because that is the root the operator serves
    (`grapharc serve --live-root "$GRAPHARC_SLACK_WORKDIR"`).

    The bot never verifies the server is up or reachable — the base URL is
    the operator's assertion, and the message wording says so.
    """
    if not base:
        return None
    path = trace_path(argv, workdir)
    if path is None:
        return None
    try:
        rel = path.relative_to(workdir)
    except ValueError:
        return None
    url = f"{base}/live/view?trace={quote(rel.as_posix(), safe='')}"
    for index, token in enumerate(argv):
        if token == "--run-id" and index + 1 < len(argv):
            url += f"&run={quote(argv[index + 1])}"
        elif token.startswith("--run-id="):
            url += f"&run={quote(token.partition('=')[2])}"
    return url


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
        body, cut = truncate(stream)
        if len(parts) > 1 or label == "stderr":
            parts.append(f"{label}:")
        parts.append(fence(body))
        if cut:
            parts.append(f"_…{cut} earlier characters not shown._")
    if len(parts) == 1:
        parts.append("_(no output)_")
    # `viz` prints raw Mermaid, which Slack shows as text. One extra line makes
    # it a diagram: a link that renders it in the browser, locally.
    if result.argv[:1] == ["viz"] and result.exit_code == 0 and result.stdout.strip():
        parts.append(f"<{mermaid_live_url(result.stdout.strip())}|render this diagram>")
    return "\n".join(parts)
