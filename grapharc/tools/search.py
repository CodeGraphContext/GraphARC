"""list_dir, glob, grep — finding things in the workspace without reading it all.

Every path these produce is checked on the way *out*, not only on the way in.
Confining the starting point is not enough: `os.walk` and `Path.glob` can hand
back a name that leaves the tree through a symlink they followed on their own
terms, and a result is as much a disclosure as a read. So each candidate is
re-resolved and dropped if it lands outside — a symlinked file pointing at
`/etc/passwd` is simply not in the results, and `list_dir` names it as pointing
outside rather than saying where.

`grep` skips the directories that make a repository scan meaningless — `.git`,
`__pycache__`, `node_modules`, virtualenvs — and says so in its description, so
a model that expected a hit in one of them can tell why it did not get one.
"""

from __future__ import annotations

import os
import re
from fnmatch import fnmatch
from pathlib import Path

from grapharc.harness.tools import ToolSpec
from grapharc.tools.workspace import (
    BINARY_SNIFF_BYTES,
    ToolError,
    ToolLimits,
    Workspace,
    read_bytes,
    truncate_items,
    truncate_text,
)

# Directories whose contents are machine-generated, vendored, or a copy of the
# history. Searching them buries the answer under thousands of irrelevant hits.
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
    }
)

LIST_DIR_DESCRIPTION = (
    "List the immediate contents of a workspace directory, sorted by name. "
    "Use this to orient yourself before reading or globbing — it is the "
    "cheapest way to find out what is actually there. Directories are shown "
    "with a trailing '/', files with their size, and symlinks with a trailing "
    "'@' and where they point (a link out of the workspace is reported as "
    "'outside the workspace' and cannot be read). 'path' defaults to the "
    "workspace root and may not resolve outside it."
)

GLOB_DESCRIPTION = (
    "Find workspace files by shell glob pattern, e.g. '*.py' for one directory "
    "or '**/*.test.ts' to recurse. Use this when you know roughly what a file "
    "is called; use grep when you know what is inside it. 'path' is the "
    "directory to search from and defaults to the workspace root. The pattern "
    "is relative to it — absolute patterns and '..' are refused, and any match "
    "that resolves outside the workspace is dropped from the results. Returns "
    "workspace-relative paths, sorted."
)

GREP_DESCRIPTION = (
    "Search workspace file contents with a Python regular expression, "
    "returning 'path:line: text' for each match. Use this to find where "
    "something is defined or used when you do not know the filename. 'path' "
    "may be a directory (searched recursively) or a single file, and defaults "
    "to the workspace root; 'include' is a filename glob such as '*.py' to "
    "narrow the sweep; 'ignore_case' makes the match case-insensitive. Binary "
    "files are skipped, and so are generated and vendored directories (.git, "
    "__pycache__, node_modules, .venv and similar), so a hit inside one of "
    "those will not be reported. Results are capped and the cap is announced."
)


def _describe(workspace: Workspace, entry: os.DirEntry[str]) -> str:
    """One `list_dir` row, naming a link's destination only if it is reachable."""
    if entry.is_symlink():
        target = Path(os.path.realpath(entry.path))
        if not workspace.contains(target):
            return f"{entry.name}@ -> outside the workspace"
        return f"{entry.name}@ -> {workspace.display(target)}"
    if entry.is_dir(follow_symlinks=False):
        return f"{entry.name}/"
    try:
        size = entry.stat(follow_symlinks=False).st_size
    except OSError:
        return entry.name
    return f"{entry.name}  {size}B"


def list_dir_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def list_dir(path: str = ".") -> str:
        """List one workspace directory, one entry per line."""
        real = workspace.resolve(path)
        name = workspace.display(real)
        if not real.exists():
            raise ToolError(f"{name} does not exist")
        if not real.is_dir():
            raise ToolError(f"{name} is a file, not a directory; use read_file")

        with os.scandir(real) as entries:
            rows = [_describe(workspace, e) for e in sorted(entries, key=lambda e: e.name)]
        if not rows:
            return f"{name} is empty"
        return "\n".join([f"{name}:", *truncate_items(rows, limits.max_entries, "entries")])

    return ToolSpec(name="list_dir", description=LIST_DIR_DESCRIPTION, fn=list_dir)


def glob_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def glob(pattern: str, path: str = ".") -> str:
        """Match workspace paths against a glob pattern, one result per line."""
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty glob string, such as '**/*.py'")
        if os.path.isabs(pattern):
            raise ToolError(
                "pattern must be relative to 'path'; an absolute pattern is refused. "
                "Set 'path' to the directory you want to search from."
            )
        if ".." in Path(pattern).parts:
            raise ToolError("pattern must not contain '..'; glob stays inside the workspace")

        root = workspace.resolve(path)
        if not root.is_dir():
            raise ToolError(f"{workspace.display(root)} is not a directory")
        try:
            found = list(root.glob(pattern))
        except (ValueError, OSError) as exc:
            # Python 3.12 rejects patterns such as 'a/**b' with a ValueError
            # that 3.13+ accepts; a model should get the message, not a stack.
            raise ToolError(f"invalid glob pattern {pattern!r}: {exc}") from exc

        # Re-resolved, not trusted: a symlinked directory glob walked into can
        # name a path outside the tree it was told to search.
        inside = sorted(
            {
                workspace.display(match)
                for match in found
                if workspace.contains(Path(os.path.realpath(match)))
            }
        )
        if not inside:
            return f"no paths match {pattern!r} under {workspace.display(root)}"
        return "\n".join(truncate_items(inside, limits.max_entries, "paths"))

    return ToolSpec(name="glob", description=GLOB_DESCRIPTION, fn=glob)


def _collect_files(
    workspace: Workspace, root: Path, include: str, limits: ToolLimits
) -> tuple[list[Path], bool]:
    """Files under `root` worth searching, and whether the file cap cut it short."""
    files: list[Path] = []
    # followlinks=False keeps the walk out of symlinked directories; the
    # per-file check below catches a symlinked *file* naming a target outside.
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        subdirectories[:] = sorted(d for d in subdirectories if d not in SKIPPED_DIRECTORIES)
        for filename in sorted(filenames):
            if include and not fnmatch(filename, include):
                continue
            candidate = Path(directory) / filename
            if not workspace.contains(Path(os.path.realpath(candidate))):
                continue
            files.append(candidate)
            if len(files) >= limits.max_files_scanned:
                return files, True
    return files, False


def _clip_line(line: str, limit: int) -> str:
    """Bound one matched line *inline*, so grep keeps one match per output line."""
    if len(line) <= limit:
        return line
    return f"{line[:limit]}…[truncated {len(line) - limit} chars]"


def _searchable_lines(path: Path, limits: ToolLimits) -> list[str] | None:
    """Text lines of `path`, or None if it is binary or unreadable."""
    try:
        data, _ = read_bytes(path, limits.max_read_bytes)
    except OSError:
        return None
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def grep_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def grep(
        pattern: str, path: str = ".", include: str = "*", ignore_case: bool = False
    ) -> str:
        """Search workspace file contents for a regular expression."""
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty regular expression")
        if not isinstance(include, str) or not include:
            raise ToolError("include must be a non-empty filename glob, such as '*.py'")
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression {pattern!r}: {exc}") from exc

        root = workspace.resolve(path)
        if not root.exists():
            raise ToolError(f"{workspace.display(root)} does not exist")
        if root.is_file():
            targets, file_cap = [root], False
        else:
            targets, file_cap = _collect_files(workspace, root, include, limits)

        matches: list[str] = []
        match_cap = False
        # Counted rather than taken from len(targets): the match cap can stop
        # the sweep early, and binary files are skipped without being read, so
        # "files searched" has to mean the files this call actually opened.
        searched = 0
        for target in targets:
            lines = _searchable_lines(target, limits)
            if lines is None:
                continue
            searched += 1
            name = workspace.display(target)
            for number, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                text = _clip_line(line.strip(), limits.max_line_chars)
                matches.append(f"{name}:{number}: {text}")
                if len(matches) >= limits.max_matches:
                    match_cap = True
                    break
            if match_cap:
                break

        where = workspace.display(root)
        if not matches:
            return (
                f"no matches for {pattern!r} in {where} "
                f"({searched} file(s) searched, include={include!r})"
            )
        header = f"{len(matches)} match(es) in {where} ({searched} file(s) searched)"
        if match_cap:
            header += f"; truncated at the {limits.max_matches}-match cap"
        if file_cap:
            header += f"; truncated at the {limits.max_files_scanned}-file cap"
        return truncate_text("\n".join([header, *matches]), limits.max_output_chars)

    return ToolSpec(name="grep", description=GREP_DESCRIPTION, fn=grep)


__all__ = [
    "GLOB_DESCRIPTION",
    "GREP_DESCRIPTION",
    "LIST_DIR_DESCRIPTION",
    "SKIPPED_DIRECTORIES",
    "glob_tool",
    "grep_tool",
    "list_dir_tool",
]
