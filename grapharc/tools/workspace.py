"""Workspace confinement and output bounds — what every core tool does first.

Two things have to happen before a core tool starts its own job, and both live
here because both are easy to forget once per tool:

* **Confine the path.** Every path argument arrives from a model. The harness's
  permission layer decides *which* tool runs, never *what path* it is handed,
  so a permitted `read_file` called with `../../.ssh/id_rsa` is a permitted
  read of `~/.ssh/id_rsa` unless the tool itself refuses. `SandboxedExecutor`
  confines paths too, but only inside the interpreter its audit hook is
  installed in: `LocalExecutor` installs nothing, and `run_command` leaves that
  interpreter entirely. So the check is implemented again here, independently,
  and every core tool routes every path argument through `Workspace.resolve`.
* **Bound the output.** A tool result becomes model context. `grep` over a
  vendored tree, or `read_file` on a 50 MB log, is a context blowout — so
  results are cut at a ceiling and the cut is announced rather than silent.

What confinement does *not* cover, stated rather than hidden:

* **Resolution and use are not atomic.** `resolve` follows every symlink and
  checks where the path lands; the tool then opens that resolved path. A
  symlink swapped in between the two is a race this does not close. Reads and
  writes pass `O_NOFOLLOW` on the final component, which narrows the window to
  the parent directories, and nothing here narrows it further.
* **`run_command`'s child is not confined at all.** Its cwd and environment are
  set and that is the end of this package's reach; see `shell.py`.
* **A refusal names the path it refused**, resolved, so a model that asks for
  `../../etc/passwd` is told the absolute path it would have reached and can
  infer where the workspace sits on the host. That is deliberate — a refusal
  the caller cannot act on gets retried — but it is a disclosure, and it is the
  reason every *successful* result is rendered workspace-relative instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Enough of a file to decide "this is not text". A UTF-8 text file has no NUL
# bytes; ELF headers, object files and images have one within a few dozen.
BINARY_SNIFF_BYTES = 8192

# Opening the *resolved* path with O_NOFOLLOW asserts the final component is
# not a symlink at open time, not merely at resolve time. 0 on platforms
# without it leaves the flag set unchanged rather than failing the open.
_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)


class ToolError(Exception):
    """A core tool refused a request or could not complete it.

    Raised rather than returned: `AgentNode` turns an exception out of a tool
    into a `TOOL_ERROR` result that goes back to the model, so these messages
    are written for that reader and say what to do differently.
    """


class WorkspaceEscape(ToolError):
    """A path argument resolved to somewhere outside the workspace root."""


@dataclass(frozen=True)
class ToolLimits:
    """Ceilings the core tools apply to their own output and runtime.

    Frozen because a `ToolSpec`'s closure captures one: a limits object mutated
    after `core_tools` returned would change the behaviour of already-registered
    tools, with nothing in the registry recording that it had happened.
    """

    max_output_chars: int = 20_000
    max_read_bytes: int = 2_000_000
    max_entries: int = 200  # list_dir rows, glob paths
    max_matches: int = 100  # grep result lines
    max_files_scanned: int = 5_000  # grep, before it gives up on the tree
    max_line_chars: int = 300  # grep, per matched line
    command_timeout: float = 60.0  # run_command default
    max_command_timeout: float = 600.0  # run_command ceiling, not overridable per call


def truncate_text(text: str, limit: int) -> str:
    """Cut `text` at `limit` characters, saying so. `limit <= 0` means no cut."""
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n"
        f"[truncated: {len(text) - limit} of {len(text)} characters omitted]"
    )


def truncate_items(items: list[str], limit: int, noun: str) -> list[str]:
    """Cut a list of output rows at `limit`, appending a row that says so."""
    if limit <= 0 or len(items) <= limit:
        return items
    return [
        *items[:limit],
        f"[truncated: {len(items) - limit} of {len(items)} {noun} omitted]",
    ]


def read_bytes(real: Path, limit: int) -> tuple[bytes, bool]:
    """Read at most `limit` bytes from an already-resolved path.

    Returns the bytes and whether the file was longer than the limit, so a
    caller can tell "the whole file" from "as much of it as was allowed" —
    `edit_file` refuses to rewrite a file it only partly read, and getting that
    distinction wrong truncates the user's file.
    """
    descriptor = os.open(real, os.O_RDONLY | _NO_FOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        return data[:limit], True
    return data, False


def write_bytes(real: Path, data: bytes) -> None:
    """Create or truncate an already-resolved path and write `data` to it."""
    descriptor = os.open(real, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NO_FOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


class Workspace:
    """A resolved directory root, and the only way a core tool names a path."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        resolved = Path(os.path.realpath(root))
        if not resolved.is_dir():
            raise ValueError(
                f"workspace {resolved} is not an existing directory; core tools are "
                "built around a root that already exists"
            )
        self.root = resolved

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    def contains(self, real: Path) -> bool:
        """Component-wise containment: `<root>-evil` is a sibling, not a child.

        String prefixes get this wrong — `"/ws-evil".startswith("/ws")` is true
        — and the separator patch (`root + os.sep`) then gets `/` wrong, since
        no absolute path starts with `//`. Comparing `parts` has neither
        failure. `real` must already be absolute and symlink-free; `resolve` is
        what makes it so.
        """
        root = self.root.parts
        return real.parts[: len(root)] == root

    def resolve(self, path: str | os.PathLike[str], *, argument: str = "path") -> Path:
        """A model-supplied path as an absolute, symlink-free path inside the root.

        Absolute inputs, `..` segments and symlinks are not three special cases
        with three rules — they are resolved first and then judged by one rule:
        the thing the OS would actually open has to be inside the root. That is
        why `..` is not rejected by spelling. `docs/../README.md` is a perfectly
        ordinary in-workspace path, and a blocklist that refuses it while a
        symlink walks straight past is the shape of every path-traversal CVE.
        """
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        if not isinstance(path, str):
            raise ToolError(f"{argument} must be a string, got {type(path).__name__}")
        if "\x00" in path:
            raise ToolError(f"{argument} contains a NUL byte, which cannot name a file")
        candidate = Path(path) if os.path.isabs(path) else self.root / path
        real = Path(os.path.realpath(candidate))
        if not self.contains(real):
            raise WorkspaceEscape(
                f"{argument} {path!r} resolves to {real}, which is outside the "
                f"workspace {self.root}. Core tools only reach paths inside the "
                "workspace; use a path relative to its root."
            )
        return real

    def display(self, real: Path) -> str:
        """Workspace-relative name for messages, so results do not leak host layout."""
        try:
            relative = real.relative_to(self.root)
        except ValueError:
            return str(real)
        return str(relative)


__all__ = [
    "BINARY_SNIFF_BYTES",
    "ToolError",
    "ToolLimits",
    "Workspace",
    "WorkspaceEscape",
    "read_bytes",
    "truncate_items",
    "truncate_text",
    "write_bytes",
]
