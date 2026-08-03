"""M3 gates.

(a) Permissions: deny → ask → allow, first match wins, fail closed; denied
    tools never appear in the visible schema set.
(b) Sandbox: a tool cannot read outside its granted workspace, open a network
    connection without declaring the capability, spawn a process, reach native
    code, or read the parent process's environment.
"""

# ctypes and sqlite3 are imported here rather than in the tool bodies below.
# A first-time import inside a sandboxed child has to scan sys.path, and
# pytest's entries sit outside the granted workspace; the forked child inherits
# the parent's sys.modules, so these resolve without any scan at all.
import ctypes  # noqa: F401
import os
import shutil
import sqlite3  # noqa: F401
import sys
import sysconfig
import time
import uuid

import pytest

from grapharc.harness import (
    Decision,
    Harness,
    HookAction,
    HookDecision,
    PermissionDenied,
    PermissionPolicy,
    PermissionRule,
    SandboxViolation,
    ToolRegistry,
    ToolSpec,
)


def _echo(**kwargs):
    return kwargs


def _policy(rules):
    return PermissionPolicy(rules=[PermissionRule(**r) for r in rules])


def test_deny_beats_allow_first_match_wins():
    policy = _policy(
        [
            {"action": "allow", "pattern": "*"},
            {"action": "deny", "pattern": "dangerous_*"},
        ]
    )
    assert policy.decide("safe_read") is Decision.ALLOW
    assert policy.decide("dangerous_delete") is Decision.DENY  # deny tier evaluated first


def test_unmatched_tool_defaults_to_deny():
    assert PermissionPolicy().decide("anything") is Decision.DENY


def test_denied_tools_are_never_visible():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="read", description="", fn=_echo))
    reg.register(ToolSpec(name="rm", description="", fn=_echo))
    policy = _policy([{"action": "allow", "pattern": "read"}, {"action": "deny", "pattern": "rm"}])
    visible = [t.name for t in reg.visible(policy)]
    assert visible == ["read"]  # rm's schema is never exposed


def test_deny_by_literal_name_beats_a_broad_allow_when_the_name_is_a_glob():
    """A DENY rule that *is* the tool's name refuses it, metacharacters and all.

    `exfil[all]` reads as a character class to fnmatch, so the rule naming it
    used to miss, evaluation fell through to `ALLOW "*"`, and the tool both
    appeared in `visible()` and ran. This is the one place a deny failed open.
    """
    name = "exfil[all]"
    reg = ToolRegistry()
    reg.register(ToolSpec(name=name, description="dangerous", fn=_echo))
    policy = _policy([{"action": "deny", "pattern": name}, {"action": "allow", "pattern": "*"}])

    assert policy.decide(name) is Decision.DENY
    assert [t.name for t in reg.visible(policy)] == []  # never offered to the model
    with pytest.raises(PermissionDenied):
        Harness(reg, policy).call(name, {})  # and never runs if it asks anyway


def test_ask_by_literal_name_gates_a_glob_shaped_name():
    """The same widening on ASK: a gate that reads right is a gate that holds."""
    name = "mcp__srv__do[all]"
    reg = ToolRegistry()
    reg.register(ToolSpec(name=name, description="", fn=_echo))
    policy = _policy([{"action": "ask", "pattern": name}, {"action": "allow", "pattern": "*"}])

    assert policy.decide(name) is Decision.ASK
    assert [t.name for t in reg.visible(policy)] == [name]  # ASK is not DENY
    with pytest.raises(PermissionDenied, match="requires approval"):
        Harness(reg, policy).call(name, {})  # no approval callback


def test_allow_stays_glob_only_and_literal_escapes_instead():
    """Literal equality is bound to DENY/ASK — it may only ever refuse more.

    Widening ALLOW the same way would grant a tool on a pattern the operator
    wrote as a glob, so `PermissionRule.literal` escapes the name instead.
    """
    name = "exfil[all]"
    glob_rule = PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern=name)])
    assert glob_rule.decide(name) is Decision.DENY  # the DENY default still holds

    literal_rule = PermissionPolicy(rules=[PermissionRule.literal(Decision.ALLOW, name)])
    assert literal_rule.decide(name) is Decision.ALLOW
    assert literal_rule.decide("exfila") is Decision.DENY  # and nothing the class covers


def test_glob_matching_is_unchanged_by_the_literal_fallback():
    """Regression: every existing glob semantic, at every tier."""
    policy = _policy([{"action": "deny", "pattern": "rm*"}, {"action": "allow", "pattern": "*"}])
    assert policy.decide("rmdir") is Decision.DENY  # prefix glob still spans
    assert policy.decide("rm") is Decision.DENY
    assert policy.decide("read_file") is Decision.ALLOW  # "*" still matches everything

    # A glob still matches through every tier, and never only its own text.
    for action in ("deny", "ask", "allow"):
        tier = _policy([{"action": action, "pattern": "danger_?"}])
        assert tier.decide("danger_1") is Decision(action)
        assert tier.decide("danger_1x") is Decision.DENY  # unmatched -> default
    # deny -> ask -> allow ordering, independent of rule order
    ordered = _policy(
        [
            {"action": "allow", "pattern": "x_*"},
            {"action": "ask", "pattern": "x_a*"},
            {"action": "deny", "pattern": "x_ab*"},
        ]
    )
    assert (ordered.decide("x_abc"), ordered.decide("x_ax"), ordered.decide("x_b")) == (
        Decision.DENY,
        Decision.ASK,
        Decision.ALLOW,
    )


def test_ask_without_approval_fails_closed():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="send", description="", fn=_echo))
    policy = _policy([{"action": "ask", "pattern": "send"}])
    harness = Harness(reg, policy)  # no approval callback
    with pytest.raises(PermissionDenied, match="requires approval"):
        harness.call("send", {"x": 1})


def test_ask_with_denial_is_refused():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="send", description="", fn=_echo))
    policy = _policy([{"action": "ask", "pattern": "send"}])
    harness = Harness(reg, policy, approval=lambda name, args: False)
    with pytest.raises(PermissionDenied):
        harness.call("send", {"x": 1})


def test_pre_hook_can_deny_and_rewrite():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="run", description="", fn=_echo))
    policy = _policy([{"action": "allow", "pattern": "run"}])

    def block_rm(name, args):
        if "rm" in str(args.get("cmd", "")):
            return HookDecision(action=HookAction.DENY, reason="rm blocked")
        return None

    harness = Harness(reg, policy, pre_hooks=(block_rm,))
    with pytest.raises(PermissionDenied, match="rm blocked"):
        harness.call("run", {"cmd": "rm -rf /"})

    def force_flag(name, args):
        return HookDecision(action=HookAction.REWRITE, args={**args, "safe": True})

    harness2 = Harness(reg, policy, pre_hooks=(force_flag,))
    assert harness2.call("run", {"cmd": "ls"}) == {"cmd": "ls", "safe": True}


# ---- sandbox gates ----------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
def test_gate_tool_cannot_read_outside_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("password123", encoding="utf-8")

    def read_secret(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    reg = ToolRegistry()
    reg.register(ToolSpec(name="read_secret", description="", fn=read_secret))
    policy = _policy([{"action": "allow", "pattern": "read_secret"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("read_secret", {"path": str(secret)})


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
def test_gate_tool_can_read_inside_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    allowed = workspace / "data.txt"
    allowed.write_text("ok", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    reg = ToolRegistry()
    reg.register(ToolSpec(name="read_file", description="", fn=read_file))
    policy = _policy([{"action": "allow", "pattern": "read_file"}])
    harness = Harness(reg, policy, workspace=str(workspace))
    assert harness.call("read_file", {"path": str(allowed)}) == "ok"


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
def test_gate_tool_cannot_open_network_without_capability(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def phone_home(host: str) -> str:
        import socket

        socket.getaddrinfo(host, 80)
        return "connected"

    reg = ToolRegistry()
    reg.register(ToolSpec(name="phone_home", description="", fn=phone_home, needs_network=False))
    policy = _policy([{"action": "allow", "pattern": "phone_home"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="network access"):
        harness.call("phone_home", {"host": "example.com"})


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
@pytest.mark.parametrize(
    "spawn",
    [
        "os.system('cat /etc/passwd')",
        "__import__('subprocess').run(['cat', '/etc/passwd'])",
    ],
)
def test_gate_tool_cannot_spawn_a_subprocess(tmp_path, spawn):
    """A child process runs without the audit hook, so it can't be confined —
    spawning one is refused outright rather than silently escaping."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def escape(code: str) -> str:
        import os  # noqa: F401 — in scope for the hostile eval below

        return str(eval(code))  # noqa: S307 — deliberately hostile tool body

    reg = ToolRegistry()
    reg.register(ToolSpec(name="escape", description="", fn=escape))
    policy = _policy([{"action": "allow", "pattern": "escape"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="spawn a process"):
        harness.call("escape", {"code": spawn})


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
@pytest.mark.parametrize("op", ["os.listdir", "os.scandir", "os.remove", "os.rmdir"])
def test_gate_non_open_filesystem_calls_are_confined(tmp_path, op):
    """os.listdir/remove/rmdir take paths too — model-supplied arguments reach
    them directly, so gating `open` alone was never enough."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    def touch(operation: str, path: str) -> str:
        import os as _os

        fn = {
            "os.listdir": _os.listdir,
            "os.scandir": lambda p: list(_os.scandir(p)),
            "os.remove": _os.remove,
            "os.rmdir": _os.rmdir,
        }[operation]
        return str(fn(path))

    reg = ToolRegistry()
    reg.register(ToolSpec(name="touch", description="", fn=touch))
    policy = _policy([{"action": "allow", "pattern": "touch"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    directory_ops = ("os.listdir", "os.scandir", "os.rmdir")
    target = str(outside.parent if op in directory_ops else outside)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("touch", {"operation": op, "path": target})
    assert outside.exists()  # the destructive ops never happened


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
def test_gate_sibling_directory_is_not_inside_the_workspace(tmp_path):
    """`<ws>-evil` shares a string prefix with `<ws>` but is not inside it."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sibling = tmp_path / "ws-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("SIBLING-SECRET", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    reg = ToolRegistry()
    reg.register(ToolSpec(name="read_file", description="", fn=read_file))
    policy = _policy([{"action": "allow", "pattern": "read_file"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("read_file", {"path": str(sibling / "secret.txt")})


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
def test_gate_hung_tool_is_killed(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def hang() -> None:
        import time

        time.sleep(30)

    reg = ToolRegistry()
    reg.register(ToolSpec(name="hang", description="", fn=hang, timeout_seconds=0.5))
    policy = _policy([{"action": "allow", "pattern": "hang"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="timeout"):
        harness.call("hang", {})


@pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)
@pytest.mark.timeout(30)
def test_gate_sigterm_handling_tool_is_still_killed(tmp_path):
    """"Killed" must mean killed: a tool that swallows SIGTERM must not keep
    running after the harness has already reported a timeout."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    marker = workspace / "still-alive.txt"

    def stubborn(marker_path: str) -> None:
        import signal as _signal
        import time as _time

        _signal.signal(_signal.SIGTERM, lambda *_: None)  # ignore SIGTERM
        for _ in range(60):
            _time.sleep(0.1)
            with open(marker_path, "a", encoding="utf-8") as f:
                f.write("tick\n")

    reg = ToolRegistry()
    reg.register(ToolSpec(name="stubborn", description="", fn=stubborn, timeout_seconds=0.5))
    policy = _policy([{"action": "allow", "pattern": "stubborn"}])
    harness = Harness(reg, policy, workspace=str(workspace))

    with pytest.raises(SandboxViolation, match="timeout"):
        harness.call("stubborn", {"marker_path": str(marker)})

    ticks_at_kill = marker.read_text().count("tick") if marker.exists() else 0
    time.sleep(1.5)
    ticks_later = marker.read_text().count("tick") if marker.exists() else 0
    assert ticks_later == ticks_at_kill, "tool kept running after the timeout"


# ---- native-code gates ------------------------------------------------------

_posix = pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)


def _sandbox(workspace, **tools):
    reg = ToolRegistry()
    for name, fn in tools.items():
        reg.register(ToolSpec(name=name, description="", fn=fn))
    policy = _policy([{"action": "allow", "pattern": "*"}])
    return Harness(reg, policy, workspace=str(workspace))


@_posix
@pytest.mark.parametrize(
    "code",
    [
        'ctypes.CDLL("libc.so.6").system(b"touch {marker}")',  # dlopen a library
        "ctypes.CDLL(None).getpid()",  # dlsym on an already-open handle
        "ctypes.c_long.from_address(id(1) + 16).value",  # cdata: read raw memory
        'ctypes.string_at(id(b"abc") + 32, 3)',  # call_function
    ],
)
def test_gate_ctypes_cannot_reach_native_code(tmp_path, code):
    """ctypes hands a tool the C ABI. Machine code called through it runs where
    no audit hook can see it, so every gate below would become advisory —
    `CDLL("libc.so.6").system(...)` was a working shell out of the sandbox."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    marker = tmp_path / "escaped-via-ctypes"

    def escape(source: str) -> str:
        # `ctypes` resolves from this module's globals, which the forked child
        # inherits — see the note at the top of the file.
        return str(eval(source))  # noqa: S307 — deliberately hostile tool body

    harness = _sandbox(workspace, escape=escape)
    with pytest.raises(SandboxViolation, match="native code"):
        harness.call("escape", {"source": code.format(marker=marker)})
    time.sleep(0.3)
    assert not marker.exists()


@_posix
def test_gate_importing_ctypes_still_works(tmp_path):
    """The one ctypes event that is allowed is `dlopen(NULL)`, which loads no
    library — it names the running process image, and ctypes does it at module
    init. Refusing it would break `import ctypes` for every tool while denying
    an attacker nothing, since dlsym and call_function stay blocked (above)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fresh_import() -> int:
        import sys as _sys
        import sysconfig as _sysconfig

        for name in [m for m in _sys.modules if m == "ctypes" or m.startswith("ctypes.")]:
            del _sys.modules[name]
        # Trim the path to the stdlib so the finder's directory scan stays
        # inside the runtime prefixes the sandbox grants.
        _sys.path[:] = [_sysconfig.get_paths()["stdlib"]]
        import ctypes  # noqa: F811 — re-importing from scratch is the point

        return ctypes.c_int(7).value

    assert _sandbox(workspace, fresh_import=fresh_import).call("fresh_import", {}) == 7


@_posix
def test_gate_raw_fork_exec_is_refused(tmp_path):
    """`subprocess.Popen` is a Python wrapper over `_posixsubprocess.fork_exec`,
    which is importable and callable on its own — gating the wrapper alone left
    a working `/bin/sh` behind it."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    marker = tmp_path / "escaped-via-fork-exec"

    def raw_spawn(marker_path: str) -> int:
        import _posixsubprocess
        import os as _os
        import re as _re

        read_fd, write_fd = _os.pipe()
        # Argument list taken from CPython's own subprocess.Popen call site.
        call_args = [
            [b"/bin/sh", b"-c", f"touch {marker_path}".encode()],
            (b"/bin/sh",),
            True,
            (write_fd,),
            None,
            None,
            -1, -1, -1, -1, -1, -1,
            read_fd,
            write_fd,
            True,
            False,
            -1, None, None, None, -1,
            None,
        ]
        # `fork_exec` is a C function with no introspectable signature, and its
        # arity is NOT stable across interpreters: 3.12 and 3.13 take 23
        # positionals, 3.14 takes 22 because `preexec_fn` was removed. Pinning
        # the list to one version made this security test fail on the others
        # with a TypeError *before* the sandbox was ever consulted — a green
        # local run and a red CI matrix. The arity is read off the interpreter's
        # own complaint, which cannot go stale the way a version table does.
        try:
            _posixsubprocess.fork_exec()
        except TypeError as exc:
            match = _re.search(r"expected (\d+) arguments", str(exc))
            wanted = int(match.group(1)) if match else len(call_args)
        else:  # pragma: no cover - fork_exec never accepts zero arguments
            wanted = len(call_args)
        # The removed parameter sat second-to-last, before `allow_vfork`.
        while len(call_args) < wanted:
            call_args.insert(-1, None)
        return _posixsubprocess.fork_exec(*call_args[:wanted])

    harness = _sandbox(workspace, raw_spawn=raw_spawn)
    with pytest.raises(SandboxViolation, match="spawn a process"):
        harness.call("raw_spawn", {"marker_path": str(marker)})
    time.sleep(0.5)
    assert not marker.exists()


@_posix
def test_gate_native_extension_cannot_be_imported_from_the_workspace(tmp_path):
    """A compiled module executes its init function as machine code, so nothing
    it goes on to do is visible to the hook. A tool can write bytes into its own
    workspace, so importing an extension from there has to be refused."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "planted.so").write_bytes(b"\x7fELF not a real extension")

    def load() -> str:
        import importlib
        import os as _os
        import sys as _sys

        _sys.path.insert(0, _os.getcwd())
        importlib.import_module("planted")
        return "loaded"

    harness = _sandbox(workspace, load=load)
    with pytest.raises(SandboxViolation, match="compiled extension"):
        harness.call("load", {})


@_posix
def test_gate_sqlite_cannot_reach_a_database_outside_the_workspace(tmp_path):
    """sqlite3 opens its database file in C, so CPython raises no `open` event
    for it: connecting was a straight read/write hole out of the workspace."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "escaped.db"

    def make_db(path: str) -> str:
        import sqlite3 as _sqlite3

        con = _sqlite3.connect(path)
        con.execute("create table t (x)")
        con.commit()
        con.close()
        return "created"

    harness = _sandbox(workspace, make_db=make_db)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("make_db", {"path": str(outside)})
    assert not outside.exists()

    assert harness.call("make_db", {"path": "inside.db"}) == "created"
    assert (workspace / "inside.db").exists()


@_posix
def test_gate_sqlite_uri_names_are_refused(tmp_path):
    """A `file:` name is decoded by sqlite itself — percent-escapes, authority,
    query string — so `realpath` would be vetting a path sqlite never opens."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "secret.db"

    def open_uri(uri: str) -> str:
        import sqlite3 as _sqlite3

        con = _sqlite3.connect(uri, uri=True)
        con.close()
        return "opened"

    harness = _sandbox(workspace, open_uri=open_uri)
    with pytest.raises(SandboxViolation, match="sqlite URI"):
        harness.call("open_uri", {"uri": f"file:{outside}?mode=rwc"})
    assert not outside.exists()


@_posix
@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason="sqlite3 built without extension loading",
)
def test_gate_sqlite_extension_loading_is_refused(tmp_path):
    """A sqlite extension is a shared library: loading one runs native code."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def enable() -> str:
        import sqlite3 as _sqlite3

        con = _sqlite3.connect(":memory:")
        con.enable_load_extension(True)
        return "enabled"

    def in_memory() -> tuple:
        import sqlite3 as _sqlite3

        con = _sqlite3.connect(":memory:")
        con.execute("create table t (x)")
        con.execute("insert into t values (42)")
        row = con.execute("select x from t").fetchone()
        con.close()
        return row

    harness = _sandbox(workspace, enable=enable, in_memory=in_memory)
    with pytest.raises(SandboxViolation, match="native extension"):
        harness.call("enable", {})
    assert harness.call("in_memory", {}) == (42,)  # ordinary sqlite still works


# ---- environment and blast-radius gates -------------------------------------


@_posix
def test_gate_child_environment_is_scrubbed(tmp_path, monkeypatch):
    """`fork` hands the child the parent's environment, so a tool body that just
    returns it exfiltrates every key the agent process holds. The keys below are
    invented for this test: no blocklist could know them, so their absence is
    what proves the scrub is an allowlist."""
    monkeypatch.setenv("GRAPHARC_TEST_SECRET_XYZ", "must-not-leak")
    monkeypatch.setenv("SOME_VENDOR_API_KEY", "sk-must-not-leak")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def dump_env() -> dict:
        import os as _os

        # os.environb is a second mapping over the same block; read both, or a
        # scrub that missed one would still hand the secrets back.
        seen = dict(_os.environ)
        seen.update({k.decode(): v.decode() for k, v in _os.environb.items()})
        return seen

    env = _sandbox(workspace, dump_env=dump_env).call("dump_env", {})
    assert "GRAPHARC_TEST_SECRET_XYZ" not in env
    assert "SOME_VENDOR_API_KEY" not in env
    assert "must-not-leak" not in "".join(env.values())
    assert env["PATH"] == os.environ["PATH"]  # still usable
    assert env["HOME"] == os.environ["HOME"]


@_posix
@pytest.mark.skipif(
    not os.path.exists("/proc/self/environ"), reason="needs Linux /proc/self/environ"
)
def test_gate_proc_environ_is_not_readable(tmp_path, monkeypatch):
    """`unsetenv` does not rewrite /proc/self/environ — the process's original
    block is still there — so scrubbing os.environ only holds while that file
    stays outside the grant."""
    monkeypatch.setenv("GRAPHARC_TEST_SECRET_XYZ", "must-not-leak")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def read_proc_environ() -> str:
        with open("/proc/self/environ", encoding="utf-8", errors="replace") as f:
            return f.read()

    harness = _sandbox(workspace, read_proc_environ=read_proc_environ)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("read_proc_environ", {})


@_posix
def test_gate_tempfile_lands_inside_the_workspace(tmp_path):
    """Scrubbing the environment must not disarm the tool: the workspace is the
    only directory it may write to, so TMPDIR has to point there."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def make_temp() -> str:
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("hi")
            return f.name

    path = _sandbox(workspace, make_temp=make_temp).call("make_temp", {})
    assert os.path.dirname(os.path.realpath(path)) == os.path.realpath(workspace)
    assert os.path.exists(path)


@_posix
@pytest.mark.parametrize("op", ["os.utime", "os.chown"])
def test_gate_metadata_mutations_are_confined(tmp_path, op):
    """utime and chown change a file without ever raising an `open` event, so
    gating the read paths alone left them writable from inside the sandbox."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "victim.txt"
    outside.write_text("untouched", encoding="utf-8")
    before = outside.stat().st_mtime_ns

    def mutate(operation: str, path: str) -> str:
        import os as _os

        {"os.utime": lambda p: _os.utime(p, (0, 0)), "os.chown": lambda p: _os.chown(p, -1, -1)}[
            operation
        ](path)
        return "mutated"

    harness = _sandbox(workspace, mutate=mutate)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("mutate", {"operation": op, "path": str(outside)})
    assert outside.stat().st_mtime_ns == before


# ---- runtime paths are readable, never writable -----------------------------
#
# `_runtime_prefixes()` (sys.prefix, purelib, /usr/lib, ...) used to sit in the
# same grant as the workspace, and that grant governed writes and deletes as
# well as reads. Everything below pins the split.

_SITE_PACKAGES = sysconfig.get_paths()["purelib"]


def _scrub(prefix):
    """Delete everything in site-packages whose name starts with `prefix`."""
    for name in os.listdir(_SITE_PACKAGES):
        if not name.startswith(prefix):
            continue
        target = os.path.join(_SITE_PACKAGES, name)
        if os.path.islink(target) or os.path.isfile(target):
            os.remove(target)
        else:
            shutil.rmtree(target, ignore_errors=True)


@pytest.fixture
def site_packages_probe():
    """A real file inside site-packages, cleaned up however the test ends.

    Several gates below need a target that actually exists — you cannot show a
    delete is refused against a path that was never there. The name is unique
    per test so parallel test processes cannot collide on it. Cleanup sweeps
    every derived name too (`.new`, `.dir`, `.link`, `.hard`): if one of these
    gates ever regresses, the tool under test really does plant a file in
    site-packages, and a leftover there is exactly the harm under test — it
    must not be left behind for the next interpreter to load.
    """
    stem = f"grapharc_probe_{uuid.uuid4().hex}"
    path = os.path.join(_SITE_PACKAGES, stem + ".txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("probe")
    except OSError:
        pytest.skip("site-packages is not writable from the test process")
    try:
        yield path
    finally:
        _scrub(stem)


@_posix
def test_gate_dot_pth_cannot_be_planted_in_site_packages(tmp_path, site_packages_probe):
    """The reported escape, run verbatim: a tool wrote `grapharc_pwned.pth` into
    the venv's site-packages and then deleted it, and both were allowed.

    A `.pth` file in site-packages is executed by `site` on *every* subsequent
    start of that interpreter. That is arbitrary Python in a different process,
    after this run is over, with no audit hook installed — the hook here cannot
    reach it and never could. Write and delete are asserted separately because
    they are separate holes: the delete arrives on `os.remove`, not on `open`.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pth = os.path.join(_SITE_PACKAGES, "grapharc_pwned.pth")

    def plant(target: str) -> str:
        with open(target, "w", encoding="utf-8") as f:
            f.write("import os; os.system('touch /tmp/grapharc-pwned')\n")
        return "planted"

    def unplant(target: str) -> str:
        os.remove(target)
        return "removed"

    harness = _sandbox(workspace, plant=plant, unplant=unplant)
    try:
        with pytest.raises(SandboxViolation, match="outside its workspace"):
            harness.call("plant", {"target": pth})
        assert not os.path.exists(pth), "a .pth file was planted in site-packages"
    finally:
        _scrub("grapharc_pwned.pth")  # a regression really does leave one behind

    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("unplant", {"target": site_packages_probe})
    assert os.path.exists(site_packages_probe), "a site-packages file was deleted from the sandbox"


@_posix
@pytest.mark.parametrize(
    "op",
    ["os.open", "os.mkdir", "os.remove", "os.rename"],
)
def test_gate_dot_pth_cannot_be_planted_via_dir_fd(tmp_path, op, site_packages_probe):
    """The same escape as `test_gate_dot_pth_cannot_be_planted_in_site_packages`,
    reached through `dir_fd=` instead of a plain path.

    `check_path` resolves the path argument of a call against the process's
    cwd (pinned to the workspace). Every one of these calls also accepts a
    directory-descriptor keyword that makes the real syscall resolve the same
    string against an *open file descriptor* instead — cwd never enters into
    it. A tool legitimately opens site-packages for reading (it is in the read
    grant, or the import machinery could not work), gets a descriptor for it,
    and then `os.open("evil.pth", O_CREAT | O_WRONLY, dir_fd=that_fd)` writes
    there while `check_path` validates a path with nothing to do with where
    the write actually lands. Verified end to end before this test existed:
    the planted `.pth` ran on the very next interpreter start in that
    environment — this is not a theoretical gap.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pth_name = f"grapharc_pwned_dirfd_{uuid.uuid4().hex}.pth"

    def plant_via_dir_fd(operation: str, name: str, victim: str) -> str:
        import os as _os

        dfd = _os.open(_SITE_PACKAGES, _os.O_RDONLY)
        try:
            if operation == "os.open":
                fd = _os.open(name, _os.O_CREAT | _os.O_WRONLY, dir_fd=dfd)
                _os.close(fd)
            elif operation == "os.mkdir":
                _os.mkdir(name, dir_fd=dfd)
            elif operation == "os.remove":
                _os.remove(_os.path.basename(victim), dir_fd=dfd)
            elif operation == "os.rename":
                _os.rename(
                    _os.path.basename(victim), name, src_dir_fd=dfd, dst_dir_fd=dfd
                )
            return "escaped"
        finally:
            _os.close(dfd)

    harness = _sandbox(workspace, plant_via_dir_fd=plant_via_dir_fd)
    try:
        with pytest.raises(SandboxViolation, match="directory descriptor"):
            harness.call(
                "plant_via_dir_fd",
                {"operation": op, "name": pth_name, "victim": site_packages_probe},
            )
        assert not os.path.exists(
            os.path.join(_SITE_PACKAGES, pth_name)
        ), "a file was planted in site-packages via dir_fd"
        if op in ("os.remove", "os.rename"):
            assert os.path.exists(
                site_packages_probe
            ), "a site-packages file was removed/renamed via dir_fd"
    finally:
        _scrub("grapharc_pwned_dirfd_")  # a regression really does leave one behind


@_posix
@pytest.mark.parametrize(
    "op",
    ["os.mkdir", "os.chmod", "os.utime", "os.truncate", "os.rename", "os.symlink", "os.link"],
)
def test_gate_runtime_paths_cannot_be_mutated(tmp_path, op, site_packages_probe):
    """Every mutating route into the runtime paths, not just `open` and
    `os.remove`. A shadowing module dropped next to the stdlib, a chmod on
    site-packages, a rename over an installed package — each one plants code or
    opens the way for it in an interpreter this hook will never see."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "payload.txt").write_text("payload", encoding="utf-8")
    victim = site_packages_probe
    before = os.stat(victim)

    def mutate(operation: str, target: str, source: str) -> str:
        import os as _os

        {
            "os.mkdir": lambda: _os.mkdir(target + ".dir"),
            "os.chmod": lambda: _os.chmod(target, 0o777),
            "os.utime": lambda: _os.utime(target, (0, 0)),
            "os.truncate": lambda: _os.truncate(target, 0),
            "os.rename": lambda: _os.rename(source, target),
            "os.symlink": lambda: _os.symlink(source, target + ".link"),
            "os.link": lambda: _os.link(source, target + ".hard"),
        }[operation]()
        return "mutated"

    harness = _sandbox(workspace, mutate=mutate)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call(
            "mutate",
            {"operation": op, "target": victim, "source": str(workspace / "payload.txt")},
        )
    after = os.stat(victim)
    assert (after.st_mode, after.st_mtime_ns, after.st_size) == (
        before.st_mode,
        before.st_mtime_ns,
        before.st_size,
    )
    for suffix in (".dir", ".link", ".hard"):
        assert not os.path.lexists(victim + suffix)


@_posix
def test_gate_runtime_paths_are_still_readable(tmp_path, site_packages_probe):
    """The other half of the split, and the reason it is a split at all: a tool
    has to import, so the stdlib and site-packages have to stay readable. A fix
    that simply dropped them from the grant would pass every gate above and
    leave the sandbox unable to run anything."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    probe = site_packages_probe

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def list_dir(path: str) -> int:
        return len(os.listdir(path))

    def copy_in(source: str, dest: str) -> str:
        import shutil as _shutil

        _shutil.copyfile(source, dest)
        with open(dest, encoding="utf-8") as f:
            return f.read()

    harness = _sandbox(workspace, read_file=read_file, list_dir=list_dir, copy_in=copy_in)
    assert harness.call("read_file", {"path": probe}) == "probe"
    assert harness.call("list_dir", {"path": _SITE_PACKAGES}) > 0
    # A two-path event whose source is a read and whose destination is a write:
    # copying *out of* the runtime paths *into* the workspace is exactly what
    # the per-argument classification has to permit.
    assert harness.call("copy_in", {"source": probe, "dest": "copied.txt"}) == "probe"


@_posix
@pytest.mark.parametrize(
    ("mode", "writes"),
    [("r", False), ("rb", False), ("w", True), ("a", True), ("r+", True), ("x", True)],
)
def test_gate_open_mode_decides_read_from_write(tmp_path, mode, writes, site_packages_probe):
    """`open`'s audit event is `open(path, mode, flags)` — the mode is right
    there in the arguments, so the same event can be a read or a write and has
    to be classified per call. Reading site-packages is fine; opening the same
    file `a` or `r+` is a write to a runtime path and is not."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    probe = site_packages_probe
    target = probe + ".new" if mode == "x" else probe

    def touch(path: str, how: str) -> str:
        with open(path, how) as f:
            if "w" in how or "a" in how or "x" in how or "+" in how:
                return "wrote"
            data = f.read()
            return data.decode() if isinstance(data, bytes) else data

    harness = _sandbox(workspace, touch=touch)
    if writes:
        with pytest.raises(SandboxViolation, match="outside its workspace"):
            harness.call("touch", {"path": target, "how": mode})
        assert not os.path.exists(probe + ".new")
        assert os.path.getsize(probe) == len("probe")
    else:
        assert harness.call("touch", {"path": target, "how": mode}) == "probe"


@_posix
@pytest.mark.parametrize("flag", ["O_RDONLY", "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC"])
def test_gate_os_open_flags_decide_read_from_write(tmp_path, flag, site_packages_probe):
    """`os.open` leaves the mode argument None and reports only the OS flags, so
    the flags are the only signal there — reading them wrongly would classify
    `os.open(pth, O_WRONLY|O_CREAT)` as a read and hand back the escape."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    probe = site_packages_probe

    def raw_open(path: str, flag_name: str) -> str:
        import os as _os

        fd = _os.open(path, getattr(_os, flag_name))
        _os.close(fd)
        return "opened"

    harness = _sandbox(workspace, raw_open=raw_open)
    if flag == "O_RDONLY":
        assert harness.call("raw_open", {"path": probe, "flag_name": flag}) == "opened"
    else:
        with pytest.raises(SandboxViolation, match="outside its workspace"):
            harness.call("raw_open", {"path": probe, "flag_name": flag})
        assert os.path.getsize(probe) == len("probe")


@_posix
def test_gate_hard_link_cannot_launder_an_outside_file(tmp_path):
    """`os.link` gives an outside file a second name, and a name inside the
    workspace resolves to itself — `realpath` sees nothing wrong with it. The
    source of a hard link therefore has to be held to the write grant too, or
    every read gate becomes a two-step."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("password123", encoding="utf-8")

    def launder(source: str, dest: str) -> str:
        os.link(source, dest)
        with open(dest, encoding="utf-8") as f:
            return f.read()

    harness = _sandbox(workspace, launder=launder)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("launder", {"source": str(secret), "dest": str(workspace / "copy")})
    assert not os.path.lexists(workspace / "copy")


@_posix
def test_gate_child_does_not_write_bytecode(tmp_path):
    """Read-only runtime paths make one ordinary thing illegal: the `.pyc` an
    import writes next to a stdlib source on a cold cache. `importlib` catches
    only `OSError` around that write, so a `SandboxViolation` would escape the
    import itself and fail a tool for a caching side effect it never asked for.
    Verified by flipping this flag off, which turns a fresh stdlib import into a
    violation naming `.../__pycache__/*.pyc`."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def bytecode_flag() -> bool:
        import sys as _sys

        return _sys.dont_write_bytecode

    assert _sandbox(workspace, bytecode_flag=bytecode_flag).call("bytecode_flag", {}) is True


# ---- symlink targets --------------------------------------------------------


@_posix
@pytest.mark.parametrize("route", ["os", "posix", "pathlib", "realpath"])
def test_gate_readlink_cannot_leak_a_target_outside_the_workspace(tmp_path, route):
    """A symlink's target is a path, and CPython raises no audit event for
    `os.readlink` — so a link planted in the workspace read back any name the
    tool asked about. Following the link was already refused; naming it was not.

    The four routes are the point of the parametrize. `os.readlink` *is*
    `posix.readlink` (the same object, re-exported), `pathlib.Path.readlink`
    calls it, and `os.path.realpath` walks through it — a guard installed on
    `os` alone leaves `import posix` as a one-line bypass."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("password123", encoding="utf-8")
    os.symlink(secret, workspace / "leak")
    inside = workspace / "real.txt"
    inside.write_text("fine", encoding="utf-8")
    os.symlink(inside, workspace / "ok")

    def read_link(path: str, how: str) -> str:
        import os as _os
        import pathlib as _pathlib
        import posix as _posix_mod

        return str(
            {
                "os": lambda: _os.readlink(path),
                "posix": lambda: _posix_mod.readlink(path),
                "pathlib": lambda: _pathlib.Path(path).readlink(),
                "realpath": lambda: _os.path.realpath(path),
            }[how]()
        )

    def follow(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    harness = _sandbox(workspace, read_link=read_link, follow=follow)
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("read_link", {"path": str(workspace / "leak"), "how": route})
    with pytest.raises(SandboxViolation, match="outside its workspace"):
        harness.call("follow", {"path": str(workspace / "leak")})
    # A link that stays inside the workspace is still readable, so the guard is
    # confining targets rather than banning the call.
    assert harness.call("read_link", {"path": str(workspace / "ok"), "how": route}) == str(
        os.path.realpath(inside)
    )
    assert harness.call("follow", {"path": str(workspace / "ok")}) == "fine"


@_posix
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs Linux /proc/self/fd")
def test_gate_readlink_cannot_enumerate_inherited_descriptors(tmp_path):
    """`fork` hands the child every descriptor the parent had open, and those
    cannot be closed here (the timeout kill path needs multiprocessing's own
    sentinel pipe). `/proc/self/fd/N` is a symlink to whatever each one points
    at, which turned the guess into a directory listing."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "inherited.txt"
    outside.write_text("inherited", encoding="utf-8")
    handle = open(outside, encoding="utf-8")  # noqa: SIM115 — must stay open across the fork

    def enumerate_fds() -> list:
        names = [f"/proc/self/fd/{n}" for n in range(3, 40)]
        return [os.readlink(p) for p in names if os.path.lexists(p)]

    try:
        harness = _sandbox(workspace, enumerate_fds=enumerate_fds)
        with pytest.raises(SandboxViolation, match="outside its workspace"):
            harness.call("enumerate_fds", {})
    finally:
        handle.close()


@_posix
def test_gate_tool_cannot_signal_another_process(tmp_path):
    """A sandboxed tool cannot spawn anything, so every process it can name
    belongs to someone else — including the harness that called it."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def signal_pid(pid: int) -> str:
        import os as _os

        _os.kill(pid, 0)
        return "signalled"

    def signal_self() -> str:
        import os as _os

        _os.kill(_os.getpid(), 0)
        return "signalled"

    harness = _sandbox(workspace, signal_pid=signal_pid, signal_self=signal_self)
    with pytest.raises(SandboxViolation, match="signal another process"):
        harness.call("signal_pid", {"pid": os.getpid()})
    assert harness.call("signal_self", {}) == "signalled"
