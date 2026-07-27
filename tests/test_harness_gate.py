"""M3 gates.

(a) Permissions: deny → ask → allow, first match wins, fail closed; denied
    tools never appear in the visible schema set.
(b) Sandbox: a tool cannot read outside its granted workspace or open a
    network connection without declaring the capability.
"""

import sys

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
