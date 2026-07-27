"""M3 gates.

(a) Permissions: deny → ask → allow, first match wins, fail closed; denied
    tools never appear in the visible schema set.
(b) Sandbox: a tool cannot read outside its granted workspace or open a
    network connection without declaring the capability.
"""

import sys
import time

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
