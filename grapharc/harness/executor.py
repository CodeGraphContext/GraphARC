"""Tool executors — sandboxed by default, local by explicit opt-in.

Permissions gate *which* tools may run; the executor bounds *what a tool can
touch once called*. The default executor runs the tool in a forked child
process with a Python audit hook that confines filesystem access to the
granted workspace (plus the interpreter's own runtime paths) and blocks
network access unless the tool declared `needs_network`. A hang past the
tool's timeout kills the child.

Honest limitation, stated rather than hidden: an audit hook constrains the
CPython interpreter it is installed in. That is the OpenClaw lesson applied
("sandbox by default"), **not** a kernel sandbox. Spawning a subprocess is
therefore blocked outright rather than confined — a child program runs without
the hook — and malicious native extensions can bypass the hook entirely. When
you need a real boundary, a container executor slots in behind this interface.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import sysconfig
import tempfile
from typing import Any

from grapharc.harness.tools import ToolSpec

# Filesystem audit events whose first argument is a path to confine. Only
# `open` was covered originally, which left os.listdir/os.remove/os.rename —
# all reachable from model-supplied arguments — completely unchecked.
#
# Known gap, since claiming coverage you don't have is worse than a documented
# hole: CPython raises no audit event for `os.stat`, so metadata *reads* (size,
# mtime, existence) outside the workspace cannot be blocked here. Content reads
# and every mutation below are.
_PATH_EVENTS = {
    "open",
    "os.listdir",
    "os.scandir",
    "os.remove",
    "os.rmdir",
    "os.mkdir",
    "os.truncate",
    "os.rename",
    "os.replace",
    "os.link",
    "os.symlink",
    "os.chmod",
    "shutil.rmtree",
    "pathlib.Path.glob",
}
# Events taking two paths — both ends must be inside the grant.
_TWO_PATH_EVENTS = {"os.rename", "os.replace", "os.link", "os.symlink"}
# Spawning a process escapes the interpreter, and the audit hook with it.
# There is no way to confine the child from here, so refuse to start one.
_SPAWN_EVENTS = {
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "subprocess.Popen",
}
_NETWORK_EVENTS = {
    "socket.__new__",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "urllib.Request",
}


class SandboxViolation(Exception):
    """A tool touched something outside its grant (path, network, or time)."""


def _runtime_prefixes() -> tuple[str, ...]:
    paths = {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sysconfig.get_paths().get("stdlib", ""),
        sysconfig.get_paths().get("purelib", ""),
        "/usr/lib",
        "/usr/local/lib",
        "/dev/null",
        "/dev/urandom",
        "/dev/random",
    }
    return tuple(os.path.realpath(p) for p in paths if p)


def _within(real: str, roots: tuple[str, ...]) -> bool:
    """Component-wise containment — a `<ws>-evil` sibling is not inside `<ws>`."""
    return any(real == root or real.startswith(root + os.sep) for root in roots)


def _child_main(conn: Any, spec: ToolSpec, args: dict[str, Any], workspace: str) -> None:
    allowed = (os.path.realpath(workspace), *_runtime_prefixes())

    def check_path(event: str, path: Any) -> None:
        if isinstance(path, bytes):
            path = path.decode(errors="replace")
        if not isinstance(path, (str, os.PathLike)):
            return  # fd-based call: no path to confine
        real = os.path.realpath(path)
        if not _within(real, allowed):
            raise SandboxViolation(
                f"tool {spec.name!r} touched {real!r} outside its workspace ({event})"
            )

    def hook(event: str, hook_args: tuple[Any, ...]) -> None:
        if event in _SPAWN_EVENTS or event.startswith(("os.exec", "os.spawn")):
            # A child process runs without this hook, so it cannot be confined.
            raise SandboxViolation(
                f"tool {spec.name!r} tried to spawn a process ({event}); "
                "subprocesses escape the sandbox and are refused"
            )
        if event in _NETWORK_EVENTS and not spec.needs_network:
            raise SandboxViolation(
                f"tool {spec.name!r} attempted network access ({event}) "
                "without declaring needs_network"
            )
        if event in _PATH_EVENTS and hook_args:
            check_path(event, hook_args[0])
            if event in _TWO_PATH_EVENTS and len(hook_args) > 1:
                check_path(event, hook_args[1])

    os.chdir(workspace)
    # Own process group, so the parent can kill any descendants on timeout.
    try:
        os.setsid()
    except OSError:
        pass
    sys.addaudithook(hook)
    try:
        result = spec.fn(**args)
        conn.send(("ok", result))
    except SandboxViolation as exc:
        conn.send(("violation", str(exc)))
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


class SandboxedExecutor:
    """Default executor: forked child + audit-hook confinement + hard timeout."""

    def __init__(self, workspace: str | None = None) -> None:
        self.workspace = workspace or tempfile.mkdtemp(prefix="grapharc-ws-")

    @staticmethod
    def _kill(proc: Any) -> None:
        """SIGTERM, then SIGKILL the whole process group — a handler must not survive."""
        proc.terminate()
        proc.join(1)
        if proc.is_alive():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()
            proc.join(2)

    def run(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(child_conn, spec, args, self.workspace), daemon=True
        )
        proc.start()
        child_conn.close()
        if not parent_conn.poll(spec.timeout_seconds):
            self._kill(proc)
            raise SandboxViolation(
                f"tool {spec.name!r} exceeded its {spec.timeout_seconds}s timeout"
            )
        kind, payload = parent_conn.recv()
        proc.join(5)
        if kind == "violation":
            raise SandboxViolation(payload)
        if kind == "error":
            raise RuntimeError(f"tool {spec.name!r} failed: {payload}")
        return payload


class LocalExecutor:
    """Direct in-process execution. Explicit opt-in only — no confinement."""

    def run(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        return spec.fn(**args)
