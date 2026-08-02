"""Tool executors — sandboxed by default, local by explicit opt-in.

Permissions gate *which* tools may run; the executor bounds *what a tool can
touch once called*. The default executor forks a child process, scrubs its
environment down to an allowlist, and installs a Python audit hook that:

* confines filesystem access to two grants, not one. **Reads** may reach the
  granted workspace *and* the interpreter's own runtime paths — stdlib,
  site-packages, `/usr/lib` — because a tool has to be able to import.
  **Mutations** — creating, writing, appending, truncating, deleting,
  renaming, linking, chmod/chown/utime, xattr sets, the `shutil` write family
  — are confined to the workspace and nothing else. The split is the whole
  point: while the runtime paths were writable, a tool could drop a `.pth`
  file into site-packages, and a `.pth` file executes arbitrary Python on
  every later start of that interpreter — a complete escape from this hook,
  in another process, outliving the run. `open` is classified per call from
  the mode in its own audit arguments; `sqlite3`, which opens its database in
  C behind CPython's back, is treated as a mutation because `connect` opens
  read/write and the audit event carries no flag saying otherwise;
* confines `os.readlink`, which CPython raises no audit event for, so a
  symlink cannot be used to hand back a path outside the read grant (a link
  to `/etc/passwd`, or `/proc/self/fd/N` naming every descriptor `fork` left
  open). This one is a wrapper on the function rather than a hook;
* blocks network access unless the tool declared `needs_network` (a tool that
  declares it gets the whole network — there is no per-host filtering);
* refuses to spawn a process, since a child runs without the hook;
* refuses every route to native code — `ctypes`, `sqlite3` extension loading,
  and importing a compiled extension from outside the interpreter's runtime
  paths — since machine code runs without the hook too;
* refuses to signal any process but itself.

A hang past the tool's timeout kills the child's whole process group.

What this is *not*, stated rather than hidden: an audit hook constrains the
CPython interpreter it is installed in. That is the OpenClaw lesson applied
("sandbox by default"), **not** a kernel sandbox. Five holes are known and
still open — some because CPython raises no audit event that would close
them, the rest by deliberate concession:

* **Metadata reads.** There is no event for `os.stat`, so size, mtime and
  existence of files outside the workspace remain readable. Content reads and
  mutations are not.
* **Raw file descriptors.** `os.read`, `os.write`, `os.dup` and `mmap` on an
  already-open descriptor carry no path to check, and `fork` hands the child
  every descriptor the parent had open. A tool that guesses an fd number can
  read through it. The descriptors cannot simply be closed here: the child
  cannot tell `multiprocessing`'s own sentinel pipe from an inherited one,
  and closing that breaks the timeout kill path. Naming them is now refused
  (`/proc/self/fd/N` goes through the `os.readlink` guard), which costs an
  attacker the enumeration step but not the guess.
* **Monkeypatched guards.** The `os.readlink` guard is a wrapper, not an
  audit hook, so unlike everything else here it is in principle removable
  from inside — the original is reachable through the wrapper's closure. An
  audit hook cannot be unregistered; this can. It is worth having anyway: it
  closes an accidental leak, and it does not pretend to stop a tool that has
  already gone looking for closure cells.
* **The forked heap.** `fork` copies the parent's memory. `os.environ` is
  scrubbed, but a secret a parent module already holds in a global stays
  reachable through `sys.modules`. Only a non-`fork` start method would fix
  this, and that would require every tool function to be picklable.
* **Extensions inside the runtime paths.** Compiled modules shipped with the
  interpreter and its site-packages are trusted, necessarily — importing any
  of them executes native code.

When you need a real boundary, a container executor slots in behind this
interface.
"""

from __future__ import annotations

import importlib.machinery
import multiprocessing
import os
import signal
import sys
import sysconfig
import tempfile
import threading
from typing import Any

from grapharc.harness.tools import ToolSpec

# Filesystem audit events whose first argument is a path to confine. Only
# `open` was covered originally, which left os.listdir/os.remove/os.rename —
# all reachable from model-supplied arguments — completely unchecked.
#
# The set is split by what the event *does*, because reads and mutations get
# different grants. Reads reach the workspace plus the interpreter's runtime
# paths; mutations reach the workspace alone. See `_grant_for` and the module
# docstring for why a single combined grant was a sandbox escape.
_READ_PATH_EVENTS = {
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.fwalk",
    # chdir moves the process, not a file. Every check below resolves through
    # the cwd, so a tool that chdirs into the stdlib still cannot write there.
    "os.chdir",
    "os.getxattr",
    "os.listxattr",
    "glob.glob",
    "pathlib.Path.glob",
}
_WRITE_PATH_EVENTS = {
    "os.remove",
    "os.rmdir",
    "os.mkdir",
    "os.truncate",
    "os.rename",
    "os.replace",
    "os.link",
    "os.symlink",
    "os.chmod",
    "os.chown",
    "os.utime",
    "os.setxattr",
    "os.removexattr",
    "shutil.rmtree",
    "shutil.chown",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "shutil.unpack_archive",
    # sqlite3 opens the database file in C, so no `open` event is raised for
    # it — connecting was a straight read/write hole out of the workspace.
    # `connect` opens read/write and the event carries no flag to say
    # otherwise, so it counts as a mutation and stays inside the workspace.
    "sqlite3.connect",
}
# `open` belongs to neither set: its own audit arguments carry the mode, so it
# is classified per call by `_open_writes`.
_PATH_EVENTS = _READ_PATH_EVENTS | _WRITE_PATH_EVENTS | {"open"}
# Events taking two paths — both ends must be inside the grant.
_TWO_PATH_EVENTS = {
    "os.rename",
    "os.replace",
    "os.link",
    "os.symlink",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "shutil.unpack_archive",
}
# Of those, the ones that mutate at *both* ends, so both are held to the write
# grant. Renaming unlinks the source; `os.link` makes an outside file readable
# through a workspace name, which would launder a read straight past the check;
# `os.symlink`'s first argument is the target the new link will point at. The
# rest are `shutil`'s source-then-destination pairs, where the source is only
# read — and the copy's actual bytes still go through `open` and `os.*` events
# of their own, each checked on its own terms.
_MUTATES_BOTH_PATHS = {"os.rename", "os.replace", "os.link", "os.symlink"}
# Flags that make an `open` a write. O_RDONLY is 0, so a read-only open masks
# to nothing here; O_CLOEXEC, which CPython sets on every open, is not listed.
_OPEN_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_EXCL
)
# Spawning a process escapes the interpreter, and the audit hook with it.
# There is no way to confine the child from here, so refuse to start one.
# `_posixsubprocess.fork_exec` is the C entry point `subprocess` calls; it is
# importable and callable directly, so gating `subprocess.Popen` alone left
# the door open.
_SPAWN_EVENTS = {
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "subprocess.Popen",
    "_posixsubprocess.fork_exec",
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
# Loading a shared library into this process runs code the hook never sees.
# `ctypes.*` is handled by prefix (see `hook`); these are the named rest.
_NATIVE_EVENTS = {
    "sqlite3.enable_load_extension",
    "sqlite3.load_extension",
}
# A sandboxed tool cannot spawn anything, so the only process it has business
# signalling is itself; anything else reaches out of the sandbox.
_SIGNAL_EVENTS = {"os.kill", "os.killpg"}
_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)

# Environment variables the child keeps. `fork` hands it the parent's whole
# environment, API keys included, and a tool body returning `dict(os.environ)`
# ships that straight back over the result pipe. This is an allowlist, not a
# blocklist, so a secret nobody thought to name cannot leak just by being new.
_SAFE_ENV_KEYS = frozenset(
    {"HOME", "LANG", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TZ", "USER"}
)
_SAFE_ENV_PREFIXES = ("LC_",)


class SandboxViolation(Exception):
    """A tool touched something outside its grant (path, network, or time)."""


def _runtime_prefixes() -> tuple[str, ...]:
    """Paths a tool may **read** but never mutate.

    A tool has to import, and importing reads the stdlib and site-packages, so
    these have to be readable. They must not be writable: site-packages is on
    `sys.path`, and anything a tool leaves there — a `.pth` file, a shadowing
    module, a replaced `.py` — runs in every later interpreter that uses this
    environment, where no audit hook is watching. Callers pass this tuple to
    the read grant only; the write grant is the workspace by itself.

    That applies to the device nodes here too, so `open(os.devnull, 'w')` is
    refused. Writing to `/dev/null` is harmless in itself, but the rule is
    worth more than the idiom: "the workspace and nothing else" is a sentence
    with no exceptions to audit, and a tool that wants a bit bucket can make
    one in its workspace.
    """
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


def _open_writes(hook_args: tuple[Any, ...]) -> bool:
    """Does this `open` audit event describe a write?

    The event's real signature is `open(path, mode, flags)`. Builtin `open`
    fills `mode` with its mode string and leaves `flags` as the OS flags it
    derived; `os.open` and CPython's own C-level opens pass `mode=None` and
    only set `flags`. Both were checked against CPython 3.14 before this was
    relied on::

        open(p, 'r')            -> (p, 'r',  O_CLOEXEC)
        open(p, 'w')            -> (p, 'w',  O_CLOEXEC|O_WRONLY|O_CREAT|O_TRUNC)
        open(p, 'a')            -> (p, 'a',  O_CLOEXEC|O_WRONLY|O_CREAT|O_APPEND)
        open(p, 'r+')           -> (p, 'r+', O_CLOEXEC|O_RDWR)
        os.open(p, O_RDONLY)    -> (p, None, O_CLOEXEC)
        os.open(p, O_WRONLY)    -> (p, None, O_CLOEXEC|O_WRONLY)

    An event carrying neither a usable mode nor usable flags counts as a
    write: the expensive mistake is to call a write a read.
    """
    mode = hook_args[1] if len(hook_args) > 1 else None
    if isinstance(mode, bytes):
        mode = mode.decode(errors="replace")
    if isinstance(mode, str) and mode:
        # 'x' creates, '+' adds writing to any base mode, 'w'/'a' are writes.
        return any(char in mode for char in "wax+")
    flags = hook_args[2] if len(hook_args) > 2 else None
    if isinstance(flags, int):
        return bool(flags & _OPEN_WRITE_FLAGS)
    return True


# `os.path.realpath` calls `os.readlink` on every component it walks, and the
# guard installed by `_install_readlink_guard` wraps that same name. This flag
# marks the sandbox's own resolution so the guard does not check the paths the
# check itself is built from. It is thread-local because a tool may be
# threaded; it is only ever true while the sandbox, not the tool, is running.
_resolving = threading.local()


def _realpath(path: Any) -> str:
    """`os.path.realpath`, with the readlink guard suspended."""
    previous = getattr(_resolving, "active", False)
    _resolving.active = True
    try:
        return os.path.realpath(path)
    finally:
        _resolving.active = previous


def _install_readlink_guard(check: Any) -> None:
    """Confine `os.readlink`, which raises no audit event of its own.

    A symlink's target *is* a path, so reading one is a read — and an
    unchecked one. A link inside the workspace pointing at `/etc/passwd` hands
    back that name, and `/proc/self/fd/N` names every descriptor `fork` left
    open, turning the inherited-descriptor hole from a guessing game into an
    enumeration. CPython raises no `os.readlink` event, so there is nothing to
    hook; this wraps the function instead, on `os` and on the `posix` builtin
    it is re-exported from (they are the same object, and a tool that imports
    `posix` directly must not get the unwrapped one).

    The *result* is checked, not the argument. `realpath` calls `readlink`
    speculatively on every component of a path, so vetting the argument would
    refuse `os.path.realpath('/usr/lib/x')` at the `/` step. A relative target
    is resolved against the link's own directory before checking.

    `dir_fd` is refused outright: the path is then relative to an inherited
    descriptor that cannot be named, so the target cannot be resolved and
    there is nothing to check against. Nothing in the stdlib's import or copy
    paths passes it.
    """
    original = os.readlink

    def readlink(path: Any, *, dir_fd: Any = None) -> Any:
        if dir_fd is not None and not getattr(_resolving, "active", False):
            raise SandboxViolation(
                "os.readlink through a directory descriptor cannot be resolved "
                "to a path, so the target cannot be checked and it is refused"
            )
        target = original(path, dir_fd=dir_fd)
        if getattr(_resolving, "active", False):
            return target
        resolved = os.fsdecode(target)
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.path.dirname(os.fsdecode(path)), resolved)
        check("os.readlink", resolved)
        return target

    readlink.__name__ = "readlink"
    readlink.__qualname__ = "readlink"
    os.readlink = readlink
    posix = sys.modules.get("posix") or sys.modules.get("nt")
    if posix is not None and hasattr(posix, "readlink"):
        posix.readlink = readlink


def _install_fork_exec_guard(spec_name: str) -> None:
    """Refuse `_posixsubprocess.fork_exec` on interpreters that do not audit it.

    CPython raises the `_posixsubprocess.fork_exec` audit event only from 3.14.
    On 3.12 and 3.13 — and 3.12 is this project's declared minimum — the call
    raises **no event at all**, so the hook above never sees it, the fork
    succeeds, and `/bin/sh` runs outside the sandbox. Verified by running it:
    the spawned process created its marker file on 3.12 and was refused on 3.14.

    The hole was hidden by a test whose argument list matched one interpreter's
    arity. `fork_exec` takes 23 positionals on 3.12 and 22 on 3.14 (`preexec_fn`
    was removed), so on 3.12 the call died with a `TypeError` before reaching
    the fork. That read like a version-skew annoyance in the test and was
    actually the gate never firing.

    Like the `os.readlink` guard this is a wrapper rather than a hook, so it
    carries the same documented caveat: the original is reachable through the
    closure and a determined tool inside the process can unwrap it. It closes
    the accident, not the adversary. On 3.14 the audit event fires first and
    this never runs.
    """
    module = sys.modules.get("_posixsubprocess")
    if module is None:  # pragma: no cover - present wherever subprocess is
        try:
            import _posixsubprocess as module  # type: ignore[no-redef]
        except ImportError:
            return
    original = getattr(module, "fork_exec", None)
    if original is None:  # pragma: no cover - non-POSIX
        return

    def fork_exec(*args: Any, **kwargs: Any) -> Any:
        raise SandboxViolation(
            f"tool {spec_name!r} tried to spawn a process "
            "(_posixsubprocess.fork_exec); subprocesses escape the sandbox and "
            "are refused"
        )

    fork_exec.__name__ = "fork_exec"
    fork_exec.__qualname__ = "fork_exec"
    module.fork_exec = fork_exec


#: Every stdlib function that accepts a directory-descriptor keyword uses one
#: of exactly three spellings (`dir_fd`, `src_dir_fd`, `dst_dir_fd`), so
#: `_install_dir_fd_guard` below checks all three by name rather than needing
#: a table of which function takes which — a table would go stale the way the
#: `fork_exec` arity above already did once. `readlink` is excluded: it has
#: its own dedicated guard above, for a different reason (its *result*, not
#: just its argument, has to be resolved and checked).
_DIR_FD_MUTATORS = (
    "chmod",
    "chown",
    "link",
    "mkdir",
    "mkfifo",
    "mknod",
    "open",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "symlink",
    "unlink",
    "utime",
)


def _install_dir_fd_guard() -> None:
    """Refuse `dir_fd=`/`src_dir_fd=`/`dst_dir_fd=` on every mutating call that
    accepts one — a real, verified escape from the workspace grant.

    `check_path` resolves a call's path argument against the *process's cwd*,
    pinned to the workspace by the `os.chdir` below. A directory-descriptor
    keyword makes the underlying syscall resolve that same string against an
    open file descriptor instead; cwd never enters into it. A tool legitimately
    reading a directory inside the read grant — site-packages, say — gets a
    descriptor for it, and `os.open("evil.pth", O_CREAT | O_WRONLY,
    dir_fd=that_fd)` then writes there while `check_path` is validating a path
    that has nothing to do with where the write actually lands. Confirmed by
    running it end to end: the planted file executes on the next interpreter
    start in that environment — the exact site-packages `.pth`-drop escape the
    read/write grant split exists to close (see the module docstring above),
    reopened through a syscall shape the audit-event path check never
    considered a path argument for at all.

    Like the `os.readlink` guard, this refuses the capability outright rather
    than trying to resolve and check it: there is no reliable way to turn a
    directory descriptor back into the path it names, so there is nothing to
    validate against the grant. It is a wrapper, not an audit hook, and carries
    the same documented caveat for the same reason — the original stays
    reachable through the closure, so it closes the accident, not a
    determined adversary already unwrapping functions from inside.
    """
    posix = sys.modules.get("posix") or sys.modules.get("nt")

    def make_wrapper(original: Any, name: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for kw in ("dir_fd", "src_dir_fd", "dst_dir_fd"):
                if kwargs.get(kw) is not None:
                    raise SandboxViolation(
                        f"os.{name} through a directory descriptor ({kw}) resolves "
                        "outside the process's cwd and cannot be checked against the "
                        "workspace grant, so it is refused"
                    )
            return original(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__qualname__ = name
        return wrapper

    for name in _DIR_FD_MUTATORS:
        original = getattr(os, name, None)
        if original is None:  # pragma: no cover - platform-dependent (mkfifo/mknod on Windows)
            continue
        wrapped = make_wrapper(original, name)
        setattr(os, name, wrapped)
        if posix is not None and hasattr(posix, name):
            setattr(posix, name, wrapped)


def _is_sqlite_uri(database: Any) -> bool:
    """sqlite re-reads a `file:` name itself — percent-decoding it and honouring
    an authority and query string — so `realpath` does not name the file that
    actually gets opened, and the path check below would be vetting a fiction.

    The audit event does not carry `connect`'s `uri` flag, so there is no way to
    tell a real URI from a literal filename that merely starts with `file:`.
    Both are refused; failing closed costs a caller nothing but the prefix.
    """
    if isinstance(database, bytes):
        database = database.decode(errors="replace")
    return isinstance(database, str) and database.startswith("file:")


def _scrub_environ(workspace: str) -> None:
    """Reduce the forked child's environment to what a tool can safely be told."""
    kept = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENV_KEYS or key.startswith(_SAFE_ENV_PREFIXES)
    }
    kept["PWD"] = workspace
    # The workspace is the only place the tool may write, so temp files have to
    # land there or every `tempfile` call would trip the path check. The module
    # caches gettempdir() on first use and the parent has usually resolved it
    # already, so the env var alone would not take effect.
    kept["TMPDIR"] = kept["TEMP"] = kept["TMP"] = workspace
    os.environ.clear()
    os.environ.update(kept)
    tempfile.tempdir = workspace


def _child_main(conn: Any, spec: ToolSpec, args: dict[str, Any], workspace: str) -> None:
    runtime = _runtime_prefixes()
    # Two grants, deliberately different sizes. Reads reach the runtime paths
    # so a tool can import; writes reach the workspace and nothing else, so a
    # tool cannot leave anything behind that a later interpreter will execute.
    readable = (os.path.realpath(workspace), *runtime)
    writable = (os.path.realpath(workspace),)

    def check_path(event: str, path: Any, *, mutating: bool = False) -> None:
        if isinstance(path, bytes):
            path = path.decode(errors="replace")
        if not isinstance(path, (str, os.PathLike)):
            return  # fd-based call: no path to confine
        real = _realpath(path)
        if _within(real, writable):
            return
        if mutating:
            raise SandboxViolation(
                f"tool {spec.name!r} tried to modify {real!r} outside its workspace "
                f"({event}); the runtime paths are readable but never writable — "
                "code left in them runs in later interpreters, where no hook is watching"
            )
        if not _within(real, readable):
            raise SandboxViolation(
                f"tool {spec.name!r} touched {real!r} outside its workspace ({event})"
            )

    def check_read(event: str, path: Any) -> None:
        check_path(event, path, mutating=False)

    def hook(event: str, hook_args: tuple[Any, ...]) -> None:
        if event.startswith("ctypes."):
            # `dlopen(None)` is the one exception: it loads no library, it only
            # names the running process image, and `import ctypes` performs it
            # at module init. Refusing it would break the import while buying
            # nothing — dlsym, call_function, cdata and friends stay refused,
            # so the handle it yields cannot be used for anything.
            if event == "ctypes.dlopen" and hook_args and hook_args[0] is None:
                return
            raise SandboxViolation(
                f"tool {spec.name!r} tried to reach native code through ctypes "
                f"({event}); native calls run outside the audit hook and are refused"
            )
        if event in _NATIVE_EVENTS:
            raise SandboxViolation(
                f"tool {spec.name!r} tried to load a native extension ({event}); "
                "loaded code runs outside the audit hook and is refused"
            )
        if event in _SPAWN_EVENTS or event.startswith(("os.exec", "os.spawn")):
            # A child process runs without this hook, so it cannot be confined.
            raise SandboxViolation(
                f"tool {spec.name!r} tried to spawn a process ({event}); "
                "subprocesses escape the sandbox and are refused"
            )
        if event == "import" and len(hook_args) > 1:
            # Extensions shipped with the interpreter are trusted by necessity;
            # one planted in the workspace is arbitrary machine code.
            origin = hook_args[1]
            if (
                isinstance(origin, str)
                and origin.endswith(_EXTENSION_SUFFIXES)
                and not _within(_realpath(origin), runtime)
            ):
                raise SandboxViolation(
                    f"tool {spec.name!r} tried to import the compiled extension "
                    f"{origin!r}; extension code runs outside the audit hook, so "
                    "only those shipped with the interpreter may be loaded"
                )
            return
        if event in _SIGNAL_EVENTS:
            if not hook_args or hook_args[0] != os.getpid():
                raise SandboxViolation(
                    f"tool {spec.name!r} tried to signal another process ({event}); "
                    "processes outside the sandbox are not the tool's to touch"
                )
            return
        if event in _NETWORK_EVENTS and not spec.needs_network:
            raise SandboxViolation(
                f"tool {spec.name!r} attempted network access ({event}) "
                "without declaring needs_network"
            )
        if event in _PATH_EVENTS and hook_args:
            if event == "sqlite3.connect" and _is_sqlite_uri(hook_args[0]):
                raise SandboxViolation(
                    f"tool {spec.name!r} opened the sqlite URI {hook_args[0]!r}; "
                    "sqlite decodes URI names itself, so the target cannot be "
                    "checked here and the connection is refused"
                )
            mutating = event in _WRITE_PATH_EVENTS or (
                event == "open" and _open_writes(hook_args)
            )
            if event in _TWO_PATH_EVENTS:
                # The first path is the source: a mutation only where the
                # event mutates both ends, otherwise a read. The second is the
                # destination, which every two-path event here writes.
                check_path(event, hook_args[0], mutating=event in _MUTATES_BOTH_PATHS)
                if len(hook_args) > 1:
                    check_path(event, hook_args[1], mutating=True)
            else:
                check_path(event, hook_args[0], mutating=mutating)

    os.chdir(workspace)
    _scrub_environ(workspace)
    # An import of a not-yet-cached module writes a `.pyc` next to the source —
    # inside the runtime paths, which are now read-only. `importlib` catches
    # only `OSError` around that write, so a `SandboxViolation` would escape
    # the import and fail the tool for a caching side effect it never asked
    # for. Not writing the cache at all costs a few milliseconds per import.
    sys.dont_write_bytecode = True
    # Own process group, so the parent can kill any descendants on timeout.
    try:
        os.setsid()
    except OSError:
        pass
    _install_readlink_guard(check_read)
    _install_fork_exec_guard(spec.name)
    _install_dir_fd_guard()
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
