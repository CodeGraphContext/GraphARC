"""Container-backed tool execution — the boundary the audit hook is not.

`SandboxedExecutor` confines a tool inside the interpreter that runs it: an
audit hook, a hand-maintained list of events, and five documented holes. Two
adversarial passes found real escapes behind it — `ctypes`, then a writable
site-packages where a dropped `.pth` file owns every later start of that
interpreter. `ContainerExecutor` moves the tool out of this process entirely,
so the confinement is the kernel's (namespaces, cgroups, capability bounding
set) rather than CPython's.

Same interface as `SandboxedExecutor`: `run(spec, args)` takes a `ToolSpec`
and returns the tool's result, raising the same `SandboxViolation` on refusal
or timeout, so callers never branch on which executor they hold.

**What the container gets.** One bind mount: the workspace, read-write, at
`/workspace`, which is also the working directory. Nothing else from the host
is mounted — but the container is not empty, it also has the *image's* own
filesystem, which is why the image is part of the security decision and is the
caller's to choose. Network is `--network none` unless the tool declared
`needs_network`. All capabilities are dropped, `no-new-privileges` is set, the
process runs as a non-root uid unless `user=` says otherwise, the rootfs is
read-only by default, and memory and pid limits are set. No host environment
variable is forwarded: the child sees only `HOME`, `TMPDIR`, two `PYTHON*`
flags, and whatever the caller named in `env=`.

**What cannot cross the boundary.** This is the honest part, and it is a real
constraint, not a footnote:

* **The callable itself.** A container is a different filesystem and a
  different interpreter; `spec.fn` is a live object in *this* one. What
  crosses is an import path — `module:qualname`, derived from `spec.fn` — and
  the tool is resolved by importing that module *inside* the container. So the
  tool's code must already be installed in the image. The default image is a
  stock `python:3.12-slim`, which contains no GraphARC and none of your code;
  running your own tools means building an image that has them.
* **Anything without an import path.** Refused with `SandboxViolation` before
  a container starts, in two steps because two different things go wrong. A
  lambda or a closure carries `<lambda>`/`<locals>` in its qualname, and a
  `functools.partial` or a callable object has no `__qualname__` at all: there
  is nothing to derive. A bound method *does* derive a well-formed path — but
  it names the underlying function, not the bound object, and `self` cannot
  cross either way, so it is refused on the shape of the object alone, before
  any name is looked up.

  Past that, the derived path is compared with `spec.fn` by identity **only
  when the named module is already in this process's `sys.modules`** — a dict
  lookup, never an import. Deciding how to containerize a tool must not run
  host code: importing a module here to check a name would execute its
  top-level code in the *unsandboxed* parent, which is the failure this class
  exists to avoid. (`importlib.util.find_spec` is no way out — for a dotted
  name it imports every parent package to locate the child.) So when the
  module is *not* already loaded here, this check is **undecided**, and it is
  the container that answers it: the import happens there, contained, and a
  failure comes back as `resolve_error` — a `SandboxViolation`, not a result.
  The honest statement of the guarantee is therefore: a derived path that
  provably names something else is refused here; one that cannot be checked
  without running host code is refused *there*.

  A function defined in `__main__` is refused outright, since that name means
  a different file there. Callers who know better can override the derived
  path per tool with `entrypoints={...}` or by setting
  `__grapharc_container_entrypoint__` on the function; an override is taken on
  trust and not resolved locally, since it names something in the image.
* **Non-JSON arguments and results.** Arguments go in as a JSON object on
  stdin and become keyword arguments; the result comes back as JSON on
  stdout. Objects, file handles, sockets, generators and callables do not
  survive that, and are refused rather than silently mangled. Two things JSON
  converts rather than refuses, so they are stated instead of hidden: a tuple
  returns as a list, and non-string dict keys arrive as strings.
* **Host paths.** A path in `args` must be a *container* path. The host
  workspace is `/workspace` inside; `container_path()` does that translation.
* **Anything printed.** Before the tool runs, the bootstrap duplicates the real
  stdout onto a private, non-inheritable descriptor and points fd 1 itself at
  stderr. Redirecting `sys.stdout` alone would not be enough: a child process
  the tool spawns inherits fd 1, and one that writes *after* the envelope would
  otherwise be the last sentinel line the parent reads — a print that forges a
  result. At the fd level the tool's prints and its children's both land on
  stderr, and more than one envelope on the result stream is a refusal rather
  than a choice between them. Neither stream is returned to the caller. Only
  the return value crosses; a tool that prints its answer instead of returning
  it hands back `None`. What this does not defend against, because nothing at
  this layer can: the tool shares an address space with the bootstrap, so code
  that goes hunting for that descriptor can write to it. A tool's return value
  is the tool's to choose in any case — the guarantee is that *printing*, by
  the tool or anything it spawns, cannot forge one.

**Limits worth stating rather than hiding.**

* The container runtime is trusted. Docker's daemon runs as root; this is a
  kernel boundary, not a hypervisor one, and a container escape defeats it.
* Network is all-or-nothing. `needs_network` grants the whole default network
  — no per-host filtering, exactly as coarse as the audit-hook executor.
* `spec.timeout_seconds` is wall-clock from process launch, so container
  startup is charged against the tool's budget.
* The image's contents are not verified. Everything in it runs.
* The read-only rootfs does not cover the workspace — it is read-write by
  design — and no disk quota is imposed, so a tool can still fill it.
* Killing a timed-out container is best-effort (`kill`, then `rm -f`). If both
  fail, the container may linger; the name is in the raised message.
* `--user` means different things under docker and rootless podman. The
  default is the host `uid:gid`, because that is what makes the bind-mounted
  workspace writable under docker. Rootless podman maps container uids through
  a subuid range instead, so the same flag lands on a host uid that owns
  nothing — pass `extra_run_args=("--userns=keep-id",)` there, or set `user=`
  to match.
* If no runtime is found, `run()` refuses. It never falls back to running the
  tool locally — a silent downgrade from "contained" to "not contained" is the
  bug this class exists to prevent.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from grapharc.harness.executor import SandboxViolation
from grapharc.harness.tools import ToolSpec

# Where the workspace is bound inside the container. Fixed, not configurable:
# it is half of the contract the tool is written against.
WORKSPACE_MOUNT = "/workspace"
DEFAULT_IMAGE = "python:3.12-slim"
# Set on a tool function to name its import path inside the image explicitly,
# for the cases where one cannot be derived (see the module docstring).
ENTRYPOINT_ATTR = "__grapharc_container_entrypoint__"
# The in-container bootstrap prints exactly one line starting with this. The
# tool's own stdout is redirected to stderr, so nothing else can produce it,
# and the parser still scans from the end in case something did.
RESULT_SENTINEL = "__GRAPHARC_CONTAINER_RESULT__"
_DEFAULT_KILL_TIMEOUT = 15.0
# uid/gid of `nobody` on every mainstream distro. Only used when the host user
# is root, where copying the host uid would put root in the container.
_NOBODY = "65534:65534"
_DOTTED_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
# Exit codes at or above this mean "killed by signal N-128" — the runtime
# enforcing a limit (OOM kill is 137), not the tool failing on its own terms.
_SIGNAL_EXIT_BASE = 128


@dataclass(frozen=True)
class CommandResult:
    """What a `CommandRunner` returns: one finished container-runtime command."""

    returncode: int
    stdout: str
    stderr: str


# (argv, stdin_text, timeout_seconds) -> CommandResult. Must raise
# `subprocess.TimeoutExpired` when the timeout elapses; `ContainerExecutor`
# catches exactly that to trigger the kill path. Injectable so the launch
# logic can be tested on a machine with no container runtime.
CommandRunner = Callable[[Sequence[str], str, float], CommandResult]


def _subprocess_runner(argv: Sequence[str], stdin_text: str, timeout: float) -> CommandResult:
    # argv list, never a shell string: nothing here is shell-interpreted.
    completed = subprocess.run(
        list(argv),
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def detect_runtime(candidates: Sequence[str] = ("docker", "podman")) -> str | None:
    """First container runtime on `PATH`, as an absolute path, or `None`.

    Presence of the binary only. `runtime_available()` is the stronger check —
    a `docker` binary with no daemon behind it is on `PATH` all the same.
    """
    for name in candidates:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


@lru_cache(maxsize=8)
def runtime_available(runtime: str | None = None, timeout: float = 20.0) -> bool:
    """Is there a container runtime that actually answers?

    Runs `<runtime> ps --quiet`, which is cheap and fails when the daemon is
    unreachable (docker) or the user cannot talk to it. Cached, because tests
    use it as a skip condition and the probe costs real time.
    """
    resolved = runtime or detect_runtime()
    if resolved is None:
        return False
    try:
        completed = subprocess.run(
            [resolved, "ps", "--quiet"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _default_user() -> str:
    """`uid:gid` for the container process — the host's, unless that is root.

    Matching the host uid is what makes the bind-mounted workspace writable
    under docker. When the host user *is* root there is nothing to match that
    would also be non-root, so this falls back to `nobody` and keeps the
    non-root promise; the workspace may then be unwritable, which is why
    `user=` exists.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return _NOBODY
    uid, gid = getuid(), getgid()
    return _NOBODY if uid == 0 else f"{uid}:{gid}"


# Runs inside the container under `python -c`. Reads one JSON request from
# stdin, resolves `module:qualname` by import, calls it with the arguments as
# keywords, and writes exactly one sentinel-prefixed JSON envelope to the real
# stdout. Every failure mode is reported *through* the envelope, so the parent
# can tell "the tool raised" from "the container died".
_RUNNER_TEMPLATE = '''\
import importlib
import json
import os
import sys

SENTINEL = "__SENTINEL__"
# The result stream is a private duplicate of fd 1, and fd 1 itself is pointed
# at stderr before the tool gets control. Reassigning `sys.stdout` alone would
# only redirect *this* interpreter's prints: a child process the tool spawns
# inherits the real fd 1, and a child that writes after the envelope would be
# the last sentinel line the parent parses. `os.dup` returns a non-inheritable
# descriptor (PEP 446), so no child inherits the result stream either.
_out_fd = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def _emit(kind, payload):
    envelope = json.dumps({"kind": kind, "payload": payload})
    data = ("\\n" + SENTINEL + " " + envelope + "\\n").encode("utf-8")
    while data:
        # A short write would truncate the envelope into unparseable garbage.
        data = data[os.write(_out_fd, data):]
    raise SystemExit(0)


try:
    _request = json.loads(sys.stdin.read())
    _module_name, _, _attr_path = _request["target"].partition(":")
    _obj = importlib.import_module(_module_name)
    for _part in _attr_path.split("."):
        _obj = getattr(_obj, _part)
except BaseException as _exc:
    _emit("resolve_error", repr(_exc))

try:
    _result = _obj(**_request["args"])
except BaseException as _exc:
    _emit("error", repr(_exc))

try:
    _encoded = json.dumps(_result)
except BaseException as _exc:
    _emit("unserializable", repr(_exc) + " from " + type(_result).__name__)

_emit("ok", json.loads(_encoded))
'''
RUNNER_SOURCE = _RUNNER_TEMPLATE.replace("__SENTINEL__", RESULT_SENTINEL)


def parse_envelope(stdout: str) -> dict[str, Any] | None:
    """The one well-formed result envelope in `stdout`, or `None` if there is none.

    Scans from the end, so noise *before* the envelope — including a malformed
    sentinel line — does not hide it. Two rules keep that from becoming a
    search for the most convenient answer:

    * If the last sentinel line is not a well-formed envelope, that is the
      result: `None`. Nothing behind it is reached for.
    * The bootstrap writes exactly one envelope and then exits, so a second
      well-formed one means something other than the bootstrap reached the
      result stream. That is not a case to pick a winner in — it raises
      `SandboxViolation` rather than return either.
    """
    envelopes: list[dict[str, Any]] = []
    nearest = True
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith(RESULT_SENTINEL):
            continue
        try:
            envelope = json.loads(stripped[len(RESULT_SENTINEL) :])
        except json.JSONDecodeError:
            envelope = None
        well_formed = isinstance(envelope, dict) and "kind" in envelope
        if nearest and not well_formed:
            return None
        nearest = False
        if well_formed:
            envelopes.append(envelope)  # type: ignore[arg-type]
    if len(envelopes) > 1:
        raise SandboxViolation(
            f"the container wrote {len(envelopes)} result envelopes; exactly one comes "
            "from the bootstrap, so the result stream was written to by something else "
            "and no envelope on it can be trusted"
        )
    return envelopes[0] if envelopes else None


def _binding_problem(fn: Any) -> str | None:
    """Why `module:qualname` cannot name `fn` in another process — or `None`.

    Decided from the object's own type, so the answer does not depend on which
    modules this process happens to have imported. Bound callables are the
    case that matters: the derived path is perfectly well formed and names the
    *underlying* function, while the object it is bound to stays here.
    """
    if inspect.ismethod(fn):
        return (
            "a bound method's import path names the underlying function, and the object "
            "it is bound to cannot cross the boundary"
        )
    bound_to = getattr(fn, "__self__", None)
    if bound_to is not None and not isinstance(bound_to, types.ModuleType):
        return (
            f"it is bound to a {type(bound_to).__name__}, which cannot cross the boundary"
        )
    return None


def _resolves_in_loaded_modules(entrypoint: str, fn: Any) -> bool | None:
    """Does `module:qualname` name exactly `fn` here — *without importing anything*?

    Tri-state, and the third state is the whole point:

    * `True` — the module is already in `sys.modules` and the path leads back
      to `fn` by identity.
    * `False` — the module is already loaded and the path leads somewhere else,
      or nowhere. Refuse: it would name something else in the container too.
    * `None` — the module is not loaded in this process, so the question cannot
      be answered without importing it. This runs in the *host*, during setup
      for an isolated execution; importing here would execute that module's
      top-level code unsandboxed, which is exactly what `ContainerExecutor`
      exists to stop. So it stays undecided and the container decides it, where
      the import is contained and a failure returns as `resolve_error`.

    `importlib.util.find_spec` is not a way around it: for a dotted name it
    imports every parent package in order to locate the child.
    """
    module_name, _, attr_path = entrypoint.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        return None
    obj: Any = module
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            # Module-level `__getattr__` (PEP 562) and descriptors are ordinary
            # code. Anything it raises means "this name does not lead to fn",
            # not "crash the caller with something that is not a refusal".
            return False
    return obj is fn


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


class ContainerExecutor:
    """Executor that runs each tool call in a fresh, throwaway container.

    Read the module docstring before using this: the tool has to be resolvable
    *inside the image*, and arguments and results have to be JSON. Both are
    enforced, not assumed — a tool that cannot cross the boundary is refused
    with `SandboxViolation` rather than run somewhere weaker.
    """

    def __init__(
        self,
        workspace: str | None = None,
        *,
        image: str = DEFAULT_IMAGE,
        runtime: str | None = None,
        python_bin: str = "python",
        memory: str | None = "512m",
        pids_limit: int = 128,
        cpus: str | float | None = None,
        read_only: bool = True,
        user: str | None = None,
        network: str = "bridge",
        mount_flags: str = "rw",
        env: Mapping[str, str] | None = None,
        entrypoints: Mapping[str, str] | None = None,
        extra_run_args: Sequence[str] = (),
        runner: CommandRunner | None = None,
        kill_timeout: float = _DEFAULT_KILL_TIMEOUT,
    ) -> None:
        self.workspace = os.path.realpath(workspace or tempfile.mkdtemp(prefix="grapharc-ws-"))
        os.makedirs(self.workspace, exist_ok=True)
        # The image is the first *positional* argument to `<runtime> run`, so a
        # value starting with "-" is parsed as another flag and the next word
        # becomes the image — `image="--privileged"` would undo the hardening
        # above it. Every other caller-supplied string lands in a flag's value
        # slot or after the image, where it cannot become a runtime flag.
        if not image or image.startswith("-"):
            raise ValueError(
                f"image {image!r} must be non-empty and must not start with '-': it is "
                "passed positionally to the container runtime, where a leading dash "
                "would be parsed as a run flag rather than an image name"
            )
        self.image = image
        # An explicit runtime is used verbatim as argv[0]; only the default is
        # discovered. "docker" meaning *this* docker is the caller's call.
        self.runtime = runtime
        self.python_bin = python_bin
        self.memory = memory
        self.pids_limit = pids_limit
        self.cpus = cpus
        self.read_only = read_only
        self.user = user or _default_user()
        self.network = network
        self.mount_flags = mount_flags
        self.env = dict(env or {})
        self.entrypoints = dict(entrypoints or {})
        # Appended after every hardening flag and before the image, so a
        # caller can add flags — including, deliberately, more mounts. The
        # "one mount" property above is the default's, not an invariant this
        # class can enforce against its own caller.
        self.extra_run_args = tuple(extra_run_args)
        self._runner: CommandRunner = runner or _subprocess_runner
        self.kill_timeout = kill_timeout

    # -- boundary translation ------------------------------------------------

    def container_path(self, host_path: str | os.PathLike[str]) -> str:
        """Translate a host path under the workspace to its path in the container.

        Refuses anything outside the workspace: it is not mounted, so naming it
        in a tool argument would silently mean a different file (or none).
        """
        real = os.path.realpath(host_path)
        if real == self.workspace:
            return WORKSPACE_MOUNT
        if not real.startswith(self.workspace + os.sep):
            raise SandboxViolation(
                f"{real!r} is outside the workspace {self.workspace!r}, so it is not "
                "mounted into the container and cannot be named in a tool argument"
            )
        relative = os.path.relpath(real, self.workspace).replace(os.sep, "/")
        return f"{WORKSPACE_MOUNT}/{relative}"

    def entrypoint_for(self, spec: ToolSpec) -> str:
        """The `module:qualname` the container will import for this tool.

        Refuses, rather than guesses, when the function has no importable name.
        Imports nothing: deciding how to containerize a tool must not run host
        code. See `_resolves_in_loaded_modules` for what that costs — a derived
        path naming a module this process has not already imported is checked
        inside the container instead of here.
        """
        override = self.entrypoints.get(spec.name) or getattr(spec.fn, ENTRYPOINT_ATTR, None)
        if override is not None:
            return self._validated(spec, override)

        module = getattr(spec.fn, "__module__", None)
        qualname = getattr(spec.fn, "__qualname__", None)
        if not module or not qualname:
            raise SandboxViolation(
                f"tool {spec.name!r} is a {type(spec.fn).__name__} with no importable "
                f"name, so it cannot be resolved inside the container; {self._hint()}"
            )
        if module in {"__main__", "__mp_main__"}:
            raise SandboxViolation(
                f"tool {spec.name!r} is defined in {module!r}, which names a different "
                f"file inside the container than it does here; {self._hint()}"
            )
        if "<" in qualname:
            # "<lambda>" and "...<locals>.f" — both unimportable by name.
            raise SandboxViolation(
                f"tool {spec.name!r} is a lambda or a nested function ({qualname!r}); "
                f"there is no import path to resolve inside the container; {self._hint()}"
            )
        derived = self._validated(spec, f"{module}:{qualname}")
        # A partial, a bound method and a callable object all inherit
        # `__module__`/`__qualname__` from their *class* or their underlying
        # function, so the derived path is syntactically fine and names the
        # wrong callable. Shape first — it holds regardless of what is loaded —
        # then identity, but only against modules already in `sys.modules`.
        reason = _binding_problem(spec.fn)
        if reason is None and _resolves_in_loaded_modules(derived, spec.fn) is False:
            reason = "here that name is bound to a different object"
        if reason is not None:
            raise SandboxViolation(
                f"tool {spec.name!r} derives the import path {derived!r}, which does not "
                f"resolve back to this {type(spec.fn).__name__} ({reason}), so it would "
                f"name something else inside the container; {self._hint()}"
            )
        return derived

    @staticmethod
    def _hint() -> str:
        return (
            f"give it an import path with entrypoints={{...}} or by setting "
            f"{ENTRYPOINT_ATTR} on the function"
        )

    def _validated(self, spec: ToolSpec, entrypoint: str) -> str:
        module_name, separator, attr_path = entrypoint.partition(":")
        if (
            not separator
            or not _DOTTED_NAME.match(module_name)
            or not _DOTTED_NAME.match(attr_path)
        ):
            raise SandboxViolation(
                f"tool {spec.name!r} has the entrypoint {entrypoint!r}, which is not of "
                "the form 'module.path:qualified.name'"
            )
        return entrypoint

    def _encode_request(self, spec: ToolSpec, target: str, args: dict[str, Any]) -> str:
        try:
            return json.dumps({"target": target, "args": dict(args)})
        except (TypeError, ValueError) as exc:
            raise SandboxViolation(
                f"tool {spec.name!r} was called with arguments that cannot cross the "
                f"container boundary as JSON: {exc}"
            ) from exc

    # -- launch --------------------------------------------------------------

    def _resolve_runtime(self, spec: ToolSpec) -> str:
        if self.runtime is not None:
            return self.runtime
        detected = detect_runtime()
        if detected is None:
            raise SandboxViolation(
                f"tool {spec.name!r} cannot run: no container runtime (docker or podman) "
                "was found on PATH, and ContainerExecutor will not fall back to running "
                "a tool outside a container"
            )
        return detected

    def _run_argv(self, runtime: str, spec: ToolSpec, name: str) -> list[str]:
        argv = [
            runtime,
            "run",
            "--rm",
            "--interactive",
            "--name",
            name,
            # Off unless declared. All-or-nothing: see the module docstring.
            "--network",
            self.network if spec.needs_network else "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            self.user,
            "--pids-limit",
            str(self.pids_limit),
            "--workdir",
            WORKSPACE_MOUNT,
            # The one mount. Everything else the container sees is the image.
            "--volume",
            f"{self.workspace}:{WORKSPACE_MOUNT}:{self.mount_flags}",
        ]
        if self.memory:
            # Equal memory and memory-swap is how you say "no swap" to both
            # runtimes; without it the limit is soft in the way that matters.
            argv += ["--memory", self.memory, "--memory-swap", self.memory]
        if self.cpus is not None:
            argv += ["--cpus", str(self.cpus)]
        if self.read_only:
            argv += ["--read-only"]
        for key, value in self._environment().items():
            argv += ["--env", f"{key}={value}"]
        argv += list(self.extra_run_args)
        argv += [self.image, self.python_bin, "-c", RUNNER_SOURCE]
        return argv

    def _environment(self) -> dict[str, str]:
        """The child's whole environment. Nothing is inherited from the host.

        `TMPDIR` and `HOME` point at the workspace because it is the only
        writable path when the rootfs is read-only, and bytecode writing is off
        for the same reason.
        """
        environment = {
            "HOME": WORKSPACE_MOUNT,
            "TMPDIR": WORKSPACE_MOUNT,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        environment.update(self.env)
        return environment

    def _kill(self, runtime: str, name: str) -> None:
        """Best-effort teardown. Killing the client does not stop the container."""
        for argv in ([runtime, "kill", name], [runtime, "rm", "--force", name]):
            try:
                self._runner(argv, "", self.kill_timeout)
            except Exception:
                # Teardown is not allowed to mask the timeout that caused it,
                # and the container may already be gone by the time we ask.
                continue

    def run(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        """Run one tool call in a container. Same contract as `SandboxedExecutor.run`."""
        target = self.entrypoint_for(spec)
        request = self._encode_request(spec, target, args)
        runtime = self._resolve_runtime(spec)
        name = f"grapharc-{uuid.uuid4().hex[:16]}"
        argv = self._run_argv(runtime, spec, name)

        try:
            result = self._runner(argv, request, spec.timeout_seconds)
        except subprocess.TimeoutExpired:
            self._kill(runtime, name)
            raise SandboxViolation(
                f"tool {spec.name!r} exceeded its {spec.timeout_seconds}s timeout; "
                f"container {name} was killed"
            ) from None
        except OSError as exc:
            raise SandboxViolation(
                f"tool {spec.name!r} cannot run: the container runtime {runtime!r} "
                f"could not be executed ({exc})"
            ) from exc

        envelope = parse_envelope(result.stdout)
        if envelope is None:
            raise self._missing_envelope_error(spec, name, result)
        return self._unwrap(spec, target, envelope)

    def _missing_envelope_error(
        self, spec: ToolSpec, name: str, result: CommandResult
    ) -> Exception:
        detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
        if result.returncode >= _SIGNAL_EXIT_BASE or result.returncode < 0:
            return SandboxViolation(
                f"tool {spec.name!r} was killed inside container {name} "
                f"(exit {result.returncode}); the memory and pid limits are enforced "
                f"by the runtime: {detail}"
            )
        return RuntimeError(
            f"tool {spec.name!r} produced no result envelope (exit {result.returncode}); "
            f"the image may not have a usable {self.python_bin!r}: {detail}"
        )

    def _unwrap(self, spec: ToolSpec, target: str, envelope: dict[str, Any]) -> Any:
        kind = envelope.get("kind")
        payload = envelope.get("payload")
        if kind == "ok":
            return payload
        if kind == "error":
            # The tool ran and raised. Same shape SandboxedExecutor reports.
            raise RuntimeError(f"tool {spec.name!r} failed: {payload}")
        if kind == "resolve_error":
            raise SandboxViolation(
                f"tool {spec.name!r} could not be resolved inside the image "
                f"{self.image!r} as {target!r}: {payload}. The tool's code has to be "
                "installed in the image — a container cannot import from this process"
            )
        if kind == "unserializable":
            raise SandboxViolation(
                f"tool {spec.name!r} returned a value that cannot cross the container "
                f"boundary as JSON: {payload}"
            )
        raise RuntimeError(f"tool {spec.name!r} returned an unrecognised envelope: {envelope!r}")
