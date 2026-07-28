"""ROADMAP §3.3 gates — the core toolset.

Three properties are worth more than the rest, so they are tested first and
hardest:

(a) **Confinement.** Every tool, every path argument, every route out —
    relative traversal, an absolute path, a symlinked file, a symlinked
    directory, and a sibling directory whose name merely shares a prefix with
    the workspace. The refusal is checked *and* the absence of the side effect
    is checked, because "it raised" and "it did not write the file" are
    different claims.
(b) **`edit_file` refuses to guess.** Zero matches and two matches are both
    errors, and the file on disk is unchanged after each.
(c) **`run_command` is not a shell.** A string argv is refused rather than
    split, its deadline kills the whole process group, and its child does not
    inherit the parent's secrets.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from grapharc.harness import (
    Harness,
    LocalExecutor,
    PermissionDenied,
    PermissionPolicy,
    PermissionRule,
    SandboxedExecutor,
    SandboxViolation,
    ToolRegistry,
)
from grapharc.harness.agent import tool_schema
from grapharc.tools import (
    CORE_TOOL_NAMES,
    AmbiguousEdit,
    ToolError,
    ToolLimits,
    Workspace,
    WorkspaceEscape,
    core_tools,
    register_core_tools,
)

SECRET = "SECRET\n"


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def outside(tmp_path):
    """A directory beside the workspace, holding something worth stealing."""
    root = tmp_path / "outside"
    root.mkdir()
    (root / "secret.txt").write_text(SECRET, encoding="utf-8")
    return root


def _tools(root, **kwargs) -> dict:
    return {spec.name: spec.fn for spec in core_tools(root, **kwargs)}


@pytest.fixture
def tools(workspace):
    return _tools(workspace)


# -- (a) confinement -----------------------------------------------------------

# Every tool, called with an argument that leaves the workspace one directory up.
RELATIVE_ESCAPES = [
    ("read_file", lambda t, out: t["read_file"]("../outside/secret.txt")),
    ("write_file", lambda t, out: t["write_file"]("../outside/pwned.txt", "x")),
    ("edit_file", lambda t, out: t["edit_file"]("../outside/secret.txt", "SECRET", "x")),
    ("list_dir", lambda t, out: t["list_dir"]("../outside")),
    ("glob", lambda t, out: t["glob"]("*", "../outside")),
    ("grep", lambda t, out: t["grep"]("SECRET", "../outside")),
    ("run_command", lambda t, out: t["run_command"](["true"], "../outside")),
]

# The same tools handed an absolute path to the same place.
ABSOLUTE_ESCAPES = [
    ("read_file", lambda t, out: t["read_file"](str(out / "secret.txt"))),
    ("write_file", lambda t, out: t["write_file"](str(out / "pwned.txt"), "x")),
    ("edit_file", lambda t, out: t["edit_file"](str(out / "secret.txt"), "SECRET", "x")),
    ("list_dir", lambda t, out: t["list_dir"](str(out))),
    ("glob", lambda t, out: t["glob"]("*", str(out))),
    ("grep", lambda t, out: t["grep"]("SECRET", str(out))),
    ("run_command", lambda t, out: t["run_command"](["true"], str(out))),
]


def _assert_untouched(outside):
    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET
    assert not (outside / "pwned.txt").exists()


@pytest.mark.parametrize(("name", "call"), RELATIVE_ESCAPES, ids=[n for n, _ in RELATIVE_ESCAPES])
def test_relative_traversal_is_refused_by_every_tool(tools, outside, name, call):
    with pytest.raises(WorkspaceEscape):
        call(tools, outside)
    _assert_untouched(outside)


@pytest.mark.parametrize(("name", "call"), ABSOLUTE_ESCAPES, ids=[n for n, _ in ABSOLUTE_ESCAPES])
def test_absolute_path_outside_is_refused_by_every_tool(tools, outside, name, call):
    with pytest.raises(WorkspaceEscape):
        call(tools, outside)
    _assert_untouched(outside)


@pytest.mark.parametrize(("name", "call"), RELATIVE_ESCAPES, ids=[n for n, _ in RELATIVE_ESCAPES])
def test_symlink_out_of_the_workspace_is_refused_by_every_tool(
    workspace, outside, tools, name, call
):
    """The same calls again, but routed through links that live *inside* the tree.

    `../outside` becomes `escape/`, so nothing in the argument looks like an
    escape: the traversal happens in the filesystem, not in the string.
    """
    (workspace / "outside").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceEscape):
        call(tools, outside)
    _assert_untouched(outside)


def test_symlinked_file_out_is_refused_for_read_write_and_edit(workspace, outside, tools):
    (workspace / "link.txt").symlink_to(outside / "secret.txt")
    for call in (
        lambda: tools["read_file"]("link.txt"),
        lambda: tools["write_file"]("link.txt", "pwned"),
        lambda: tools["edit_file"]("link.txt", "SECRET", "pwned"),
    ):
        with pytest.raises(WorkspaceEscape):
            call()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET


def test_grep_skips_a_symlinked_file_whose_target_is_outside(workspace, outside, tools):
    """A walk can reach a link the caller never named; the result is checked too."""
    (workspace / "link.txt").symlink_to(outside / "secret.txt")
    (workspace / "own.txt").write_text("SECRET lives here\n", encoding="utf-8")
    result = tools["grep"]("SECRET")
    assert "own.txt:1:" in result
    assert "link.txt" not in result


def test_glob_drops_matches_that_resolve_outside(workspace, outside, tools):
    (workspace / "link.txt").symlink_to(outside / "secret.txt")
    (workspace / "own.txt").write_text("x", encoding="utf-8")
    assert tools["glob"]("*.txt") == "own.txt"


def test_list_dir_names_an_outward_link_without_naming_its_target(workspace, outside, tools):
    (workspace / "link.txt").symlink_to(outside / "secret.txt")
    result = tools["list_dir"]()
    assert "link.txt@ -> outside the workspace" in result
    assert str(outside) not in result


def test_sibling_sharing_a_name_prefix_is_not_inside_the_workspace(tmp_path):
    """`<ws>-evil` is the case a `startswith` containment check gets wrong."""
    root = tmp_path / "work"
    root.mkdir()
    evil = tmp_path / "work-evil"
    evil.mkdir()
    (evil / "secret.txt").write_text(SECRET, encoding="utf-8")

    space = Workspace(root)
    assert not space.contains(evil)
    assert not space.contains(evil / "secret.txt")
    assert space.contains(root)
    assert space.contains(root / "a" / "b")

    read_file = _tools(root)["read_file"]
    with pytest.raises(WorkspaceEscape):
        read_file(str(evil / "secret.txt"))
    assert (evil / "secret.txt").read_text(encoding="utf-8") == SECRET


def test_containment_is_correct_for_a_filesystem_root():
    """The other case a `root + os.sep` prefix check gets wrong: no path starts '//'.

    Pure computation on an existing directory — nothing is read or written.
    """
    space = Workspace("/")
    assert space.contains(Path("/etc"))
    assert space.contains(Path("/"))


def test_dotdot_that_stays_inside_the_workspace_is_allowed(workspace, tools):
    """Confinement is where the path lands, not how it is spelled."""
    (workspace / "docs").mkdir()
    (workspace / "README.md").write_text("hi", encoding="utf-8")
    assert tools["read_file"]("docs/../README.md") == "hi"


def test_symlink_inside_the_workspace_still_works(workspace, tools):
    (workspace / "real.txt").write_text("original", encoding="utf-8")
    (workspace / "link.txt").symlink_to(workspace / "real.txt")
    assert tools["read_file"]("link.txt") == "original"
    tools["write_file"]("link.txt", "replaced")
    assert (workspace / "real.txt").read_text(encoding="utf-8") == "replaced"


def test_nul_byte_in_a_path_is_refused(tools):
    with pytest.raises(ToolError, match="NUL byte"):
        tools["read_file"]("a\x00b")


def test_non_string_path_is_refused(tools):
    with pytest.raises(ToolError, match="must be a string"):
        tools["read_file"](17)


def test_workspace_root_must_already_exist(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        Workspace(tmp_path / "missing")
    with pytest.raises(ValueError, match="not an existing directory"):
        core_tools(tmp_path / "missing")


def test_workspace_root_is_resolved_so_a_symlinked_root_still_matches(tmp_path):
    """A workspace reached through a link must not reject its own contents."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("inside", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert _tools(link)["read_file"]("a.txt") == "inside"


# -- read_file -----------------------------------------------------------------


def test_read_file_returns_content_verbatim(workspace, tools):
    body = "def add(a, b):\n    return a + b\n"
    (workspace / "calc.py").write_text(body, encoding="utf-8")
    # Byte-for-byte: edit_file matches exactly, so any gutter or line number
    # added here would have to be stripped back off before an edit could land.
    assert tools["read_file"]("calc.py") == body


def test_read_file_windows_lines_with_offset_and_limit(workspace, tools):
    (workspace / "n.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")
    result = tools["read_file"]("n.txt", offset=3, limit=2)
    assert result.startswith("line3\nline4\n")
    assert "line5" not in result
    assert "lines 3-4 of 10" in result


def test_read_file_reports_an_offset_past_the_end(workspace, tools):
    (workspace / "n.txt").write_text("one\ntwo\n", encoding="utf-8")
    assert "past the last line (2)" in tools["read_file"]("n.txt", offset=9)


def test_read_file_rejects_a_zero_or_negative_offset(workspace, tools):
    (workspace / "n.txt").write_text("one\n", encoding="utf-8")
    with pytest.raises(ToolError, match="1-based"):
        tools["read_file"]("n.txt", offset=0)


def test_read_file_marks_an_empty_file(workspace, tools):
    (workspace / "empty.txt").write_text("", encoding="utf-8")
    assert "file is empty" in tools["read_file"]("empty.txt")


def test_read_file_truncates_long_output_with_a_marker(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_output_chars=20))
    (workspace / "big.txt").write_text("x" * 500, encoding="utf-8")
    result = root_tools["read_file"]("big.txt")
    assert result.startswith("x" * 20)
    assert "truncated: 480 of 500 characters omitted" in result
    assert len(result) < 200


def test_read_file_notes_when_it_could_not_read_the_whole_file(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_read_bytes=8))
    (workspace / "big.txt").write_text("y" * 100, encoding="utf-8")
    result = root_tools["read_file"]("big.txt")
    assert "only the first 8 bytes were read" in result


def test_read_file_refuses_a_binary_file(workspace, tools):
    (workspace / "blob.bin").write_bytes(b"\x7fELF\x00\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        tools["read_file"]("blob.bin")


def test_read_file_refuses_a_directory_and_a_missing_file(workspace, tools):
    (workspace / "sub").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        tools["read_file"]("sub")
    with pytest.raises(ToolError, match="does not exist"):
        tools["read_file"]("nope.txt")


# -- write_file ----------------------------------------------------------------


def test_write_file_creates_parents_and_reports_the_size(workspace, tools):
    result = tools["write_file"]("a/b/c.txt", "hello")
    assert (workspace / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "hello"
    assert result == "created a/b/c.txt (5 bytes)"


def test_write_file_reports_an_overwrite_as_one(workspace, tools):
    tools["write_file"]("f.txt", "one")
    assert tools["write_file"]("f.txt", "two") == "overwrote f.txt (3 bytes)"
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "two"


def test_write_file_refuses_a_directory_including_the_root(workspace, tools):
    (workspace / "sub").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        tools["write_file"]("sub", "x")
    with pytest.raises(ToolError, match="is a directory"):
        tools["write_file"](".", "x")


def test_write_file_refuses_non_string_content(tools):
    with pytest.raises(ToolError, match="content must be a string"):
        tools["write_file"]("f.txt", 5)


def test_write_file_reports_an_unwritable_path_as_a_tool_error(workspace, tools):
    """A file standing where a directory is needed surfaces as prose, not an errno."""
    (workspace / "f.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="could not write f.txt/g.txt"):
        tools["write_file"]("f.txt/g.txt", "y")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "x"


# -- (b) edit_file refuses to guess --------------------------------------------


def test_edit_file_replaces_a_unique_string(workspace, tools):
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    result = tools["edit_file"]("calc.py", "a - b", "a + b")
    assert result == "edited calc.py: replaced 1 occurrence"
    body = (workspace / "calc.py").read_text(encoding="utf-8")
    assert body == "def add(a, b):\n    return a + b\n"


def test_edit_file_refuses_an_anchor_that_matches_nothing(workspace, tools):
    (workspace / "f.txt").write_text("alpha\n", encoding="utf-8")
    with pytest.raises(ToolError, match="was not found"):
        tools["edit_file"]("f.txt", "beta", "gamma")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "alpha\n"


def test_edit_file_refuses_an_ambiguous_anchor_and_changes_nothing(workspace, tools):
    body = "x = 1\ny = 1\nz = 1\n"
    (workspace / "f.py").write_text(body, encoding="utf-8")
    with pytest.raises(AmbiguousEdit, match="matches 3 places"):
        tools["edit_file"]("f.py", "= 1", "= 2")
    assert (workspace / "f.py").read_text(encoding="utf-8") == body


def test_edit_file_replaces_every_occurrence_only_when_told_to(workspace, tools):
    (workspace / "f.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
    result = tools["edit_file"]("f.py", "= 1", "= 2", replace_all=True)
    assert result == "edited f.py: replaced 2 occurrences"
    assert (workspace / "f.py").read_text(encoding="utf-8") == "a = 2\nb = 2\n"


def test_edit_file_still_refuses_a_missing_anchor_under_replace_all(workspace, tools):
    (workspace / "f.py").write_text("a = 1\n", encoding="utf-8")
    with pytest.raises(ToolError, match="was not found"):
        tools["edit_file"]("f.py", "nope", "x", replace_all=True)


def test_edit_file_refuses_an_empty_or_identical_anchor(workspace, tools):
    (workspace / "f.txt").write_text("alpha\n", encoding="utf-8")
    with pytest.raises(ToolError, match="old_string is empty"):
        tools["edit_file"]("f.txt", "", "x")
    with pytest.raises(ToolError, match="identical"):
        tools["edit_file"]("f.txt", "alpha", "alpha")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "alpha\n"


def test_edit_file_refuses_a_file_it_could_not_read_whole(workspace):
    """Rewriting a partial read would delete everything past the cut."""
    root_tools = _tools(workspace, limits=ToolLimits(max_read_bytes=4))
    body = "alpha beta gamma\n"
    (workspace / "f.txt").write_text(body, encoding="utf-8")
    with pytest.raises(ToolError, match="could not read whole"):
        root_tools["edit_file"]("f.txt", "alph", "x")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == body


def test_edit_file_refuses_a_file_that_is_not_valid_utf8(workspace, tools):
    body = b"caf\xe9 latte\n"
    (workspace / "f.txt").write_bytes(body)
    with pytest.raises(ToolError, match="not valid UTF-8"):
        tools["edit_file"]("f.txt", "latte", "mocha")
    assert (workspace / "f.txt").read_bytes() == body


def test_edit_file_refuses_a_missing_file(tools):
    with pytest.raises(ToolError, match="does not exist"):
        tools["edit_file"]("nope.txt", "a", "b")


# -- list_dir ------------------------------------------------------------------


def test_list_dir_sorts_and_marks_directories_and_sizes(workspace, tools):
    (workspace / "zeta").mkdir()
    (workspace / "alpha.txt").write_text("1234", encoding="utf-8")
    result = tools["list_dir"]().splitlines()
    assert result[1:] == ["alpha.txt  4B", "zeta/"]


def test_list_dir_reports_an_empty_directory(workspace, tools):
    (workspace / "sub").mkdir()
    assert tools["list_dir"]("sub") == "sub is empty"


def test_list_dir_refuses_a_file(workspace, tools):
    (workspace / "f.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="is a file"):
        tools["list_dir"]("f.txt")


def test_list_dir_truncates_a_large_directory(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_entries=3))
    for index in range(10):
        (workspace / f"f{index}.txt").write_text("x", encoding="utf-8")
    rows = root_tools["list_dir"]().splitlines()
    assert rows[-1] == "[truncated: 7 of 10 entries omitted]"
    assert len(rows) == 5  # header + 3 entries + marker


# -- glob ----------------------------------------------------------------------


def test_glob_matches_recursively_and_returns_relative_paths(workspace, tools):
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "mod.py").write_text("", encoding="utf-8")
    (workspace / "top.py").write_text("", encoding="utf-8")
    (workspace / "notes.md").write_text("", encoding="utf-8")
    assert tools["glob"]("**/*.py").splitlines() == ["pkg/mod.py", "top.py"]


def test_glob_refuses_an_absolute_pattern_or_a_dotdot_pattern(tools):
    with pytest.raises(ToolError, match="absolute pattern is refused"):
        tools["glob"]("/etc/*")
    with pytest.raises(ToolError, match="must not contain"):
        tools["glob"]("../*")
    with pytest.raises(ToolError, match="non-empty"):
        tools["glob"]("")


def test_glob_says_so_when_nothing_matches(tools):
    assert tools["glob"]("*.rs").startswith("no paths match '*.rs'")


def test_glob_truncates_a_large_result_set(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_entries=2))
    for index in range(6):
        (workspace / f"f{index}.py").write_text("", encoding="utf-8")
    rows = root_tools["glob"]("*.py").splitlines()
    assert rows == ["f0.py", "f1.py", "[truncated: 4 of 6 paths omitted]"]


# -- grep ----------------------------------------------------------------------


def test_grep_reports_path_line_and_text(workspace, tools):
    (workspace / "a.py").write_text("import os\nvalue = 1\n", encoding="utf-8")
    result = tools["grep"](r"^value")
    assert "a.py:2: value = 1" in result
    assert result.startswith("1 match(es)")


def test_grep_honours_the_include_filter(workspace, tools):
    (workspace / "a.py").write_text("needle\n", encoding="utf-8")
    (workspace / "b.md").write_text("needle\n", encoding="utf-8")
    result = tools["grep"]("needle", include="*.py")
    assert "a.py:1:" in result
    assert "b.md" not in result


def test_grep_ignore_case_is_off_by_default(workspace, tools):
    (workspace / "a.txt").write_text("Needle\n", encoding="utf-8")
    assert tools["grep"]("needle").startswith("no matches")
    assert "a.txt:1: Needle" in tools["grep"]("needle", ignore_case=True)


def test_grep_skips_generated_directories(workspace, tools):
    (workspace / ".git").mkdir()
    (workspace / ".git" / "log.txt").write_text("needle\n", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "x.txt").write_text("needle\n", encoding="utf-8")
    (workspace / "src.py").write_text("needle\n", encoding="utf-8")
    result = tools["grep"]("needle")
    assert "src.py:1:" in result
    assert ".git" not in result
    assert "__pycache__" not in result


def test_grep_skips_binary_files(workspace, tools):
    (workspace / "blob.bin").write_bytes(b"needle\x00\x01padding")
    (workspace / "text.txt").write_text("needle\n", encoding="utf-8")
    result = tools["grep"]("needle")
    assert "text.txt:1:" in result
    assert "blob.bin" not in result
    # The count is files actually read, not files considered.
    assert "(1 file(s) searched)" in result


def test_grep_searches_a_single_file_when_given_one(workspace, tools):
    (workspace / "a.txt").write_text("needle\n", encoding="utf-8")
    (workspace / "b.txt").write_text("needle\n", encoding="utf-8")
    result = tools["grep"]("needle", path="a.txt")
    assert "a.txt:1:" in result
    assert "b.txt" not in result


def test_grep_announces_its_match_cap(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_matches=3))
    (workspace / "a.txt").write_text("hit\n" * 20, encoding="utf-8")
    result = root_tools["grep"]("hit")
    assert "truncated at the 3-match cap" in result
    assert len(result.splitlines()) == 4  # header + 3 matches


def test_grep_clips_a_very_long_matched_line_inline(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_line_chars=10))
    (workspace / "a.txt").write_text("needle" + "z" * 100 + "\n", encoding="utf-8")
    result = root_tools["grep"]("needle")
    assert "…[truncated 96 chars]" in result
    assert len(result.splitlines()) == 2  # the clip stays on one line


def test_grep_refuses_an_invalid_regular_expression(tools):
    with pytest.raises(ToolError, match="invalid regular expression"):
        tools["grep"]("(unclosed")


def test_grep_says_so_when_nothing_matches(tools):
    assert tools["grep"]("needle").startswith("no matches for 'needle'")


# -- (c) run_command is not a shell --------------------------------------------


def test_run_command_runs_an_argv_list(workspace, tools):
    (workspace / "f.txt").write_text("body\n", encoding="utf-8")
    result = tools["run_command"](["cat", "f.txt"])
    assert result.startswith("exit code 0")
    assert "stdout:\nbody" in result


def test_run_command_refuses_a_string_and_does_not_split_it(workspace, tools):
    """The refusal has to be a refusal: nothing may run, not even the first word."""
    with pytest.raises(ToolError, match="never\ninvokes a shell|never invokes a shell"):
        tools["run_command"]("touch pwned.txt")
    assert not (workspace / "pwned.txt").exists()
    assert sorted(p.name for p in workspace.iterdir()) == []


def test_run_command_does_not_interpret_shell_metacharacters(workspace, tools):
    """`;` and `>` are arguments to echo, not operators — nothing is created."""
    result = tools["run_command"](["echo", "hi ; touch pwned.txt > out.txt"])
    assert "hi ; touch pwned.txt > out.txt" in result
    assert not (workspace / "pwned.txt").exists()
    assert not (workspace / "out.txt").exists()


def test_run_command_refuses_an_empty_or_badly_typed_argv(tools):
    with pytest.raises(ToolError, match="argv is empty"):
        tools["run_command"]([])
    with pytest.raises(ToolError, match=r"argv\[1\] must be a string"):
        tools["run_command"](["echo", 5])
    with pytest.raises(ToolError, match="argv must be a list"):
        tools["run_command"]({"cmd": "echo"})


def test_run_command_reports_a_non_zero_exit_and_stderr(tools):
    result = tools["run_command"](["sh", "-c", "echo oops >&2; exit 3"])
    assert result.startswith("exit code 3")
    assert "stderr:\noops" in result


def test_run_command_refuses_a_command_that_does_not_exist(tools):
    with pytest.raises(ToolError, match="command not found"):
        tools["run_command"](["definitely-not-a-real-program-xyz"])


def test_run_command_times_out_and_kills_the_whole_process_group(tools):
    """A backgrounded grandchild holds the pipe; killing only the child hangs."""
    assert shutil.which("bash"), "bash is required for the process-group test"
    started = time.perf_counter()
    result = tools["run_command"](["bash", "-c", "sleep 30 & wait"], timeout_seconds=1)
    elapsed = time.perf_counter() - started
    assert "timed out at 1s" in result
    assert "the process group was killed" in result
    assert elapsed < 10, f"kill did not release the pipes ({elapsed:.1f}s)"


def test_run_command_keeps_partial_output_from_a_timed_out_run(tools):
    result = tools["run_command"](["sh", "-c", "echo early; sleep 30"], timeout_seconds=1)
    assert "timed out" in result
    assert "early" in result


def test_run_command_rejects_a_timeout_outside_its_bounds(tools):
    with pytest.raises(ToolError, match="greater than zero"):
        tools["run_command"](["true"], timeout_seconds=0)
    with pytest.raises(ToolError, match="exceeds this toolset's ceiling"):
        tools["run_command"](["true"], timeout_seconds=10_000)


def test_run_command_does_not_hand_the_parents_secrets_to_the_child(tools, monkeypatch):
    monkeypatch.setenv("GRAPHARC_TEST_API_KEY", "hunter2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "hunter2")
    result = tools["run_command"](["env"])
    assert "hunter2" not in result
    assert "GRAPHARC_TEST_API_KEY" not in result
    assert "PATH=" in result  # the allowlist is not empty


def test_run_command_runs_in_the_requested_workspace_directory(workspace, tools):
    (workspace / "sub").mkdir()
    (workspace / "sub" / "marker.txt").write_text("", encoding="utf-8")
    result = tools["run_command"](["ls"], cwd="sub")
    assert "marker.txt" in result


def test_run_command_defaults_to_the_workspace_root(workspace, tools):
    (workspace / "root-marker.txt").write_text("", encoding="utf-8")
    assert "root-marker.txt" in tools["run_command"](["ls"])


def test_run_command_truncates_a_flood_on_each_stream_separately(workspace):
    root_tools = _tools(workspace, limits=ToolLimits(max_output_chars=200))
    result = root_tools["run_command"](
        ["sh", "-c", "head -c 5000 /dev/zero | tr '\\0' 'a'; echo err >&2"]
    )
    assert "truncated:" in result
    assert "stderr:\nerr" in result  # the flood did not crowd stderr out


# -- the factory and the registry ----------------------------------------------


def test_core_tools_builds_every_tool_in_a_stable_order(workspace):
    specs = core_tools(workspace)
    assert [spec.name for spec in specs] == list(CORE_TOOL_NAMES)
    assert [spec.name for spec in core_tools(workspace)] == [spec.name for spec in specs]


def test_core_tools_include_and_exclude_narrow_the_set(workspace):
    assert [s.name for s in core_tools(workspace, include=["grep", "read_file"])] == [
        "read_file",
        "grep",
    ]  # CORE_TOOL_NAMES order, not the caller's
    assert "run_command" not in [s.name for s in core_tools(workspace, exclude=["run_command"])]
    assert [
        s.name for s in core_tools(workspace, include=["read_file", "grep"], exclude=["grep"])
    ] == ["read_file"]


def test_core_tools_rejects_an_unknown_name_rather_than_ignoring_it(workspace):
    """A typo in `exclude` must not leave the caller thinking a tool is off."""
    with pytest.raises(ValueError, match="unknown core tools"):
        core_tools(workspace, exclude=["run_comand"])
    with pytest.raises(ValueError, match="unknown core tools"):
        core_tools(workspace, include=["bash"])


def test_core_tools_rejects_a_bare_string_selection(workspace):
    with pytest.raises(ValueError, match="not a single string"):
        core_tools(workspace, include="grep")


def test_register_core_tools_fills_a_registry_and_chains(workspace):
    registry = ToolRegistry()
    returned = register_core_tools(registry, workspace)
    assert returned is registry
    for name in CORE_TOOL_NAMES:
        assert registry.get(name) is not None


def test_register_core_tools_refuses_to_shadow_an_existing_tool(workspace):
    registry = ToolRegistry()
    register_core_tools(registry, workspace, include=["grep"])
    with pytest.raises(ValueError, match="already registered"):
        register_core_tools(registry, workspace, include=["grep"])


def test_every_spec_carries_a_description_a_model_can_act_on(workspace):
    descriptions = {spec.name: spec.description for spec in core_tools(workspace)}
    for name, description in descriptions.items():
        assert len(description) > 120, name  # a label is not a description
        assert "Use this" in description, name  # say *when* to reach for it
        assert "workspace" in description, name  # and that it cannot leave it
    assert "LIST" in descriptions["run_command"]  # argv is a list, said loudly
    assert "never passed through a shell" in descriptions["run_command"]
    assert "exact string" in descriptions["edit_file"]
    assert "refused" in descriptions["read_file"]


def test_specs_render_to_tool_schemas_the_model_can_call(workspace):
    schemas = {
        s.name: tool_schema(s)["function"] for s in core_tools(workspace)
    }
    assert schemas["read_file"]["parameters"]["required"] == ["path"]
    assert schemas["edit_file"]["parameters"]["required"] == ["path", "old_string", "new_string"]
    assert schemas["edit_file"]["parameters"]["properties"]["replace_all"]["type"] == "boolean"
    assert schemas["run_command"]["parameters"]["properties"]["argv"]["type"] == "array"
    assert schemas["list_dir"]["parameters"].get("required") is None  # path defaults


# -- composed with the harness -------------------------------------------------


def _harness(workspace, rules, executor):
    registry = ToolRegistry()
    register_core_tools(registry, workspace)
    policy = PermissionPolicy(rules=[PermissionRule(**rule) for rule in rules])
    return Harness(registry, policy, executor=executor, workspace=str(workspace))


def test_core_tools_run_through_the_harness_under_policy(workspace):
    (workspace / "f.txt").write_text("body\n", encoding="utf-8")
    # No wildcard deny: a broad deny outranks a narrow allow in this policy, so
    # `deny *` would take read_file with it. An unmatched tool defaults to deny.
    harness = _harness(workspace, [{"action": "allow", "pattern": "read_file"}], LocalExecutor())
    assert harness.call("read_file", {"path": "f.txt"}) == "body\n"
    assert [spec.name for spec in harness.visible_tools()] == ["read_file"]
    with pytest.raises(PermissionDenied):
        harness.call("write_file", {"path": "f.txt", "content": "x"})


def test_confinement_holds_when_the_executor_provides_none(workspace, outside):
    """LocalExecutor installs no audit hook: the tool is the only thing left."""
    harness = _harness(workspace, [{"action": "allow", "pattern": "*"}], LocalExecutor())
    with pytest.raises(WorkspaceEscape):
        harness.call("read_file", {"path": str(outside / "secret.txt")})
    with pytest.raises(WorkspaceEscape):
        harness.call("write_file", {"path": str(outside / "pwned.txt"), "content": "x"})
    _assert_untouched(outside)


def test_file_tools_also_work_under_the_sandboxed_executor(workspace):
    (workspace / "f.txt").write_text("sandboxed\n", encoding="utf-8")
    harness = _harness(
        workspace, [{"action": "allow", "pattern": "*"}], SandboxedExecutor(str(workspace))
    )
    assert harness.call("read_file", {"path": "f.txt"}) == "sandboxed\n"
    written = harness.call("write_file", {"path": "g.txt", "content": "ok"})
    assert written == "created g.txt (2 bytes)"
    assert (workspace / "g.txt").read_text(encoding="utf-8") == "ok"


def test_run_command_cannot_run_under_the_sandboxed_executor(workspace):
    """`shell.py`'s claim, checked: an audit hook cannot follow a subprocess.

    The refusal arrives at the `/dev/null` open `stdin=DEVNULL` performs just
    before the spawn, so the message names that rather than `subprocess.Popen`.
    Either way nothing ran — which is what the assertion on disk checks.
    """
    harness = _harness(
        workspace, [{"action": "allow", "pattern": "*"}], SandboxedExecutor(str(workspace))
    )
    with pytest.raises(SandboxViolation, match="run_command"):
        harness.call("run_command", {"argv": ["touch", "pwned.txt"]})
    assert not (workspace / "pwned.txt").exists()


def test_the_package_does_not_shadow_the_harness_tools_module():
    """`grapharc.tools` (package) and `grapharc.harness.tools` (module) coexist."""
    from grapharc.harness.tools import ToolSpec as HarnessToolSpec

    assert all(isinstance(spec, HarnessToolSpec) for spec in core_tools(Path(os.getcwd())))
