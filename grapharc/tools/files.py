"""read_file, write_file, edit_file — reading and changing text in the workspace.

`edit_file` carries the only real design decision in this module: it replaces
an *exact* string, and it refuses when that string occurs zero times or more
than once. Both refusals are the point. A model that cannot find its anchor has
misremembered the file and needs to re-read it; a model whose anchor matches
four places has not said which one it meant, and picking the first is how an
agent edits the wrong function and reports success. `replace_all=True` exists
for the case where "all of them" is genuinely the instruction — that is a
stated intent, not a guess.

Reads are deliberately verbatim: no line numbers, no gutter. The output of
`read_file` is the input `edit_file` matches against, and any decoration would
mean the model has to strip it back off by hand, byte for byte, before an edit
can land.
"""

from __future__ import annotations

from grapharc.harness.tools import ToolSpec
from grapharc.tools.workspace import (
    BINARY_SNIFF_BYTES,
    ToolError,
    ToolLimits,
    Workspace,
    read_bytes,
    truncate_text,
    write_bytes,
)


class AmbiguousEdit(ToolError):
    """`edit_file`'s anchor matched more than one place and no choice was stated."""


READ_FILE_DESCRIPTION = (
    "Read a UTF-8 text file from the workspace. Use this before editing "
    "anything: edit_file matches an exact string, and read_file returns the "
    "file verbatim — no line numbers or other decoration — so text copied out "
    "of this output can be pasted straight into edit_file's old_string. "
    "'path' is relative to the workspace root; a path that resolves outside it "
    "is refused. For a large file, 'offset' is the 1-based line to start at "
    "and 'limit' the maximum number of lines to return (0 means no line "
    "limit). Very long output is cut with a '[truncated: ...]' marker, and "
    "binary files are refused rather than returned as noise."
)

WRITE_FILE_DESCRIPTION = (
    "Create a file in the workspace, or replace an existing one wholesale. "
    "Use this for a new file or a full rewrite; to change part of a file that "
    "already exists, prefer edit_file, which cannot silently discard the parts "
    "you did not mean to touch. Missing parent directories are created. "
    "'path' is relative to the workspace root; a path resolving outside it is "
    "refused. Returns whether the file was created or overwritten, and its "
    "new size."
)
EDIT_FILE_DESCRIPTION = (
    "Replace an exact string in a workspace file. Use this for every change to "
    "a file that already exists — unlike write_file it cannot lose the parts "
    "you did not mention. 'old_string' must match the file byte for byte, "
    "including indentation and trailing whitespace, so read the file first and "
    "copy the text out of read_file's output. The edit is refused if "
    "old_string appears nowhere (you have misremembered the file — read it "
    "again) or in more than one place (ambiguous: extend old_string with "
    "surrounding lines until it is unique). Set 'replace_all' to true only "
    "when you genuinely mean every occurrence, such as renaming a symbol. "
    "Nothing is written when an edit is refused."
)


def read_file_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
        """Return a workspace text file, optionally a line window of it."""
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise ToolError("offset is a 1-based line number, so it must be at least 1")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ToolError("limit must be 0 (every line) or a positive line count")

        real = workspace.resolve(path)
        name = workspace.display(real)
        if not real.exists():
            raise ToolError(f"{name} does not exist")
        if real.is_dir():
            raise ToolError(f"{name} is a directory, not a file; use list_dir")

        data, clipped = read_bytes(real, limits.max_read_bytes)
        if b"\x00" in data[:BINARY_SNIFF_BYTES]:
            raise ToolError(
                f"{name} looks like a binary file (NUL byte within its first "
                f"{BINARY_SNIFF_BYTES} bytes); read_file only returns text"
            )
        # errors="replace", not strict: a mostly-text file with one bad byte is
        # still worth reading. `edit_file` decodes strictly for the opposite
        # reason — it writes the result back.
        text = data.decode("utf-8", errors="replace")

        notes: list[str] = []
        if clipped:
            notes.append(
                f"only the first {limits.max_read_bytes} bytes were read; "
                "the file is longer"
            )
        if offset > 1 or limit:
            lines = text.splitlines(keepends=True)
            end = offset - 1 + limit if limit else None
            window = lines[offset - 1 : end]
            text = "".join(window)
            if window:
                last = offset + len(window) - 1
                notes.append(f"lines {offset}-{last} of {len(lines)}")
            else:
                notes.append(f"offset {offset} is past the last line ({len(lines)})")
        elif not text:
            notes.append("file is empty")

        text = truncate_text(text, limits.max_output_chars)
        if notes:
            text = f"{text}\n[{'; '.join(notes)}]"
        return text

    return ToolSpec(name="read_file", description=READ_FILE_DESCRIPTION, fn=read_file)


def write_file_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def write_file(path: str, content: str) -> str:
        """Create or wholly replace a workspace file."""
        if not isinstance(content, str):
            raise ToolError(f"content must be a string, got {type(content).__name__}")
        real = workspace.resolve(path)
        name = workspace.display(real)
        # Catches the workspace root itself as well as any existing directory:
        # both would otherwise fail deep inside os.open with an errno.
        if real.is_dir():
            raise ToolError(f"{name} is a directory; write_file writes files")

        existed = real.exists()
        data = content.encode("utf-8")
        try:
            real.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(real, data)
        except OSError as exc:
            # Most often a component of the path is an existing *file*, which
            # surfaces from mkdir as a bare errno with no clue which component.
            raise ToolError(f"could not write {name}: {exc.strerror or exc}") from exc
        return f"{'overwrote' if existed else 'created'} {name} ({len(data)} bytes)"

    # `limits` is unused: write_file's result is one line, so there is nothing
    # to bound. The signature stays uniform so `core.py` can build any tool the
    # same way.
    return ToolSpec(name="write_file", description=WRITE_FILE_DESCRIPTION, fn=write_file)


def edit_file_tool(workspace: Workspace, limits: ToolLimits) -> ToolSpec:
    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace an exact string in a workspace file, or refuse and change nothing."""
        for argument, value in (("old_string", old_string), ("new_string", new_string)):
            if not isinstance(value, str):
                raise ToolError(
                    f"{argument} must be a string, got {type(value).__name__}"
                )
        if not old_string:
            raise ToolError(
                "old_string is empty; it must be the exact text to replace. To "
                "create a file or rewrite one from scratch, use write_file."
            )
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical; nothing to do")

        real = workspace.resolve(path)
        name = workspace.display(real)
        if not real.exists():
            raise ToolError(f"{name} does not exist; use write_file to create it")
        if real.is_dir():
            raise ToolError(f"{name} is a directory, not a file")

        data, clipped = read_bytes(real, limits.max_read_bytes)
        if clipped:
            # Writing back what was read would delete everything past the cut.
            raise ToolError(
                f"{name} is larger than {limits.max_read_bytes} bytes; edit_file "
                "will not rewrite a file it could not read whole"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Strict, unlike read_file: a lossy decode here would be written
            # back, silently replacing every undecodable byte in the user's file.
            raise ToolError(
                f"{name} is not valid UTF-8, so editing it would corrupt the "
                f"bytes that could not be decoded ({exc})"
            ) from exc

        count = text.count(old_string)
        if count == 0:
            raise ToolError(
                f"old_string was not found in {name}. The match is exact, "
                "including indentation and trailing whitespace — read the file "
                "and copy the text verbatim out of read_file's output."
            )
        if count > 1 and not replace_all:
            raise AmbiguousEdit(
                f"old_string matches {count} places in {name}, and edit_file will "
                "not guess which one you meant. Extend old_string with the lines "
                "around it until it is unique, or pass replace_all=true if you "
                "mean to change every occurrence."
            )

        write_bytes(real, text.replace(old_string, new_string).encode("utf-8"))
        plural = "" if count == 1 else "s"
        return f"edited {name}: replaced {count} occurrence{plural}"

    return ToolSpec(name="edit_file", description=EDIT_FILE_DESCRIPTION, fn=edit_file)


__all__ = [
    "EDIT_FILE_DESCRIPTION",
    "READ_FILE_DESCRIPTION",
    "WRITE_FILE_DESCRIPTION",
    "AmbiguousEdit",
    "edit_file_tool",
    "read_file_tool",
    "write_file_tool",
]
