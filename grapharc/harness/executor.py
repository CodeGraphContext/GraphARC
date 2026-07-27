"""Tool executors — sandboxed by default, local by explicit opt-in.

Permissions gate *which* tools may run; the executor bounds *what a tool can
touch once called*. The default executor runs the tool in a forked child
process with a Python audit hook that confines filesystem access to the
granted workspace (plus the interpreter's own runtime paths) and blocks
network access unless the tool declared `needs_network`. A hang past the
tool's timeout kills the child.

Honest limitation, stated rather than hidden: an audit hook constrains Python
code paths — it is the OpenClaw lesson applied ("sandbox by default"), not a
kernel sandbox. Malicious native extensions can bypass it; a container
executor slots in behind the same interface when stronger isolation is needed.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import sysconfig
import tempfile
from typing import Any

from grapharc.harness.tools import ToolSpec


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


def _child_main(conn: Any, spec: ToolSpec, args: dict[str, Any], workspace: str) -> None:
    allowed = (os.path.realpath(workspace), *_runtime_prefixes())
    blocked_socket_events = (
        "socket.connect",
        "socket.bind",
        "socket.sendto",
        "socket.getaddrinfo",
    )

    def hook(event: str, hook_args: tuple[Any, ...]) -> None:
        if event == "open":
            path = hook_args[0]
            if isinstance(path, bytes):
                path = path.decode(errors="replace")
            if not isinstance(path, str):
                return  # fd-based open
            real = os.path.realpath(path)
            if not real.startswith(allowed):
                raise SandboxViolation(
                    f"tool {spec.name!r} opened {real!r} outside its workspace"
                )
        elif event in blocked_socket_events and not spec.needs_network:
            raise SandboxViolation(
                f"tool {spec.name!r} attempted network access ({event}) "
                "without declaring needs_network"
            )

    os.chdir(workspace)
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

    def run(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(child_conn, spec, args, self.workspace), daemon=True
        )
        proc.start()
        child_conn.close()
        if not parent_conn.poll(spec.timeout_seconds):
            proc.terminate()
            proc.join(1)
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
