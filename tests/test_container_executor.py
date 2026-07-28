"""§3.2 gates — the container executor.

Three layers, because a container runtime may not exist on the machine running
these tests and the logic still has to be proven:

(a) **Launch logic**, with an injected fake runner: what flags are actually
    passed, that only the workspace is mounted, that the network is off unless
    declared, that a timeout kills the named container, and that a tool with no
    import path is refused *before* anything starts.
(b) **The in-container bootstrap**, run for real under this interpreter rather
    than mocked: argument marshalling, result marshalling, a raising tool, an
    unresolvable target, an unserializable result, and stdout noise that must
    not corrupt the envelope.
(c) **Real containers**, skipped cleanly when no runtime answers or the test
    image is not already local. These never pull an image.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import os.path
import re
import subprocess
import sys
import time
import uuid

import pytest

from grapharc.harness import (
    Harness,
    PermissionPolicy,
    PermissionRule,
    SandboxedExecutor,
    SandboxViolation,
    ToolRegistry,
    ToolSpec,
)
from grapharc.harness.container import (
    RESULT_SENTINEL,
    RUNNER_SOURCE,
    WORKSPACE_MOUNT,
    CommandResult,
    ContainerExecutor,
    detect_runtime,
    parse_envelope,
    runtime_available,
)

# ---------------------------------------------------------------------------
# fixtures and fakes
# ---------------------------------------------------------------------------


def sample_tool(**kwargs):
    """A module-level function, so it has an import path to derive."""
    return kwargs


# Proof that a refusal is a refusal: if anything ever runs this in-process, the
# executor degraded from "contained" to "not contained".
LOCAL_CALLS: list[dict] = []


def tracked_tool(**kwargs):
    LOCAL_CALLS.append(kwargs)
    return "ran"


def envelope_line(kind, payload):
    return f'{RESULT_SENTINEL} {json.dumps({"kind": kind, "payload": payload})}\n'


class FakeRunner:
    """Records every runtime invocation instead of making one.

    `run` invocations return `result`; teardown invocations (`kill`, `rm`)
    always succeed. Set `timeout=True` to make the `run` invocation behave the
    way `subprocess.run` does when the deadline passes.
    """

    def __init__(self, result=None, timeout=False):
        self.result = result if result is not None else CommandResult(0, "", "")
        self.timeout = timeout
        self.calls: list[tuple[list[str], str, float]] = []

    def __call__(self, argv, stdin_text, timeout):
        argv = list(argv)
        self.calls.append((argv, stdin_text, timeout))
        if argv[1] != "run":
            return CommandResult(0, "", "")
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
        return self.result

    @property
    def subcommands(self):
        return [call[0][1] for call in self.calls]

    def argv_of(self, subcommand):
        for argv, _, _ in self.calls:
            if argv[1] == subcommand:
                return argv
        raise AssertionError(f"no {subcommand!r} invocation was made")


def make_executor(workspace, runner, **kwargs):
    # runtime is given explicitly so these tests never depend on a real
    # docker/podman being installed.
    kwargs.setdefault("runtime", "docker")
    return ContainerExecutor(str(workspace), runner=runner, **kwargs)


def flag_values(argv, flag):
    return [argv[i + 1] for i, item in enumerate(argv) if item == flag and i + 1 < len(argv)]


# A module whose *import* is observable. It stands in for every real module
# that opens a connection, reads a credential, starts a thread or mutates
# global state at import time — which is to say, most of them.
_SIDE_EFFECT_MODULE = '''\
import os

with open(os.environ["GRAPHARC_IMPORT_MARKER"], "w") as handle:
    handle.write("host code ran\\n")


def tool(**kwargs):
    return kwargs
'''


@pytest.fixture
def unimported_module(tmp_path, monkeypatch):
    """A real, importable module that this process has deliberately not imported.

    Yields `(module_name, marker_path)`; the marker exists if and only if
    something imported the module. The whole point of `ContainerExecutor` is
    that setting up an isolated execution does not run host code, so nothing
    in `entrypoint_for` may make that marker appear.
    """
    name = f"grapharc_import_probe_{uuid.uuid4().hex[:8]}"
    marker = tmp_path / "imported.marker"
    (tmp_path / f"{name}.py").write_text(_SIDE_EFFECT_MODULE)
    monkeypatch.setenv("GRAPHARC_IMPORT_MARKER", str(marker))
    monkeypatch.syspath_prepend(str(tmp_path))
    assert name not in sys.modules
    try:
        yield name, marker
    finally:
        # If a regression does import it, do not leave it loaded for the
        # next test — that would hide the regression the second time round.
        sys.modules.pop(name, None)


class ImportTripwire:
    """A meta-path finder that records every import attempt without blocking it."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        self.attempts.append(fullname)
        return None  # let the real finders answer; we only want the record


# ---------------------------------------------------------------------------
# (a) launch logic — no container runtime required
# ---------------------------------------------------------------------------


def test_interface_matches_the_sandboxed_executor():
    """A caller must be able to swap executors without changing the call."""
    assert list(inspect.signature(ContainerExecutor.run).parameters) == list(
        inspect.signature(SandboxedExecutor.run).parameters
    )


def test_only_the_workspace_is_mounted(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    workspace = tmp_path / "ws"
    executor = make_executor(workspace, runner)
    executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})

    argv = runner.argv_of("run")
    assert flag_values(argv, "--volume") == [f"{os.path.realpath(workspace)}:{WORKSPACE_MOUNT}:rw"]
    # Anything else that could expose host state is absent entirely.
    for forbidden in ("-v", "--mount", "--tmpfs", "--volumes-from", "--env-file"):
        assert forbidden not in argv


def test_no_host_environment_is_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHARC_TEST_SECRET", "hunter2")
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner)
    executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})

    argv = runner.argv_of("run")
    assert "hunter2" not in " ".join(argv)
    assert set(flag_values(argv, "--env")) == {
        f"HOME={WORKSPACE_MOUNT}",
        f"TMPDIR={WORKSPACE_MOUNT}",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
    }


def test_network_is_off_unless_the_tool_declares_it(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner)

    executor.run(ToolSpec(name="quiet", description="", fn=sample_tool), {})
    assert flag_values(runner.argv_of("run"), "--network") == ["none"]

    runner.calls.clear()
    executor.run(
        ToolSpec(name="loud", description="", fn=sample_tool, needs_network=True),
        {},
    )
    assert flag_values(runner.argv_of("run"), "--network") == ["bridge"]


def test_hardening_flags_are_all_present(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner, memory="256m", pids_limit=32)
    executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})

    argv = runner.argv_of("run")
    assert flag_values(argv, "--cap-drop") == ["ALL"]
    assert flag_values(argv, "--security-opt") == ["no-new-privileges"]
    assert flag_values(argv, "--memory") == ["256m"]
    assert flag_values(argv, "--memory-swap") == ["256m"]  # equal == no swap
    assert flag_values(argv, "--pids-limit") == ["32"]
    assert flag_values(argv, "--workdir") == [WORKSPACE_MOUNT]
    assert "--read-only" in argv
    assert "--rm" in argv
    assert "--privileged" not in argv


def test_container_user_is_never_root(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner)
    executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})

    (user,) = flag_values(runner.argv_of("run"), "--user")
    uid = user.split(":")[0]
    assert uid.isdigit() and int(uid) != 0


def test_extra_run_args_land_before_the_image(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(
        tmp_path / "ws", runner, image="my/image:1", extra_run_args=("--userns=keep-id",)
    )
    executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})

    argv = runner.argv_of("run")
    assert argv.index("--userns=keep-id") < argv.index("my/image:1")
    # The image is followed by the command, and nothing else.
    assert argv[argv.index("my/image:1") + 1 : argv.index("my/image:1") + 3] == ["python", "-c"]


def test_arguments_cross_as_json_on_stdin(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", None), ""))
    executor = make_executor(tmp_path / "ws", runner)
    executor.run(
        ToolSpec(name="t", description="", fn=sample_tool),
        {"path": "/workspace/x.txt", "n": 3},
    )

    _, stdin_text, _ = runner.calls[0]
    assert json.loads(stdin_text) == {
        "target": f"{sample_tool.__module__}:sample_tool",
        "args": {"path": "/workspace/x.txt", "n": 3},
    }


def test_timeout_kills_the_named_container(tmp_path):
    runner = FakeRunner(timeout=True)
    executor = make_executor(tmp_path / "ws", runner)
    spec = ToolSpec(name="slow", description="", fn=sample_tool, timeout_seconds=4.0)

    with pytest.raises(SandboxViolation, match="exceeded its 4.0s timeout"):
        executor.run(spec, {})

    # The deadline is handed to the runner, and the container the run named is
    # the container the teardown kills — killing the client would not stop it.
    (name,) = flag_values(runner.argv_of("run"), "--name")
    assert runner.calls[0][2] == 4.0
    assert runner.subcommands == ["run", "kill", "rm"]
    assert runner.argv_of("kill") == ["docker", "kill", name]
    assert runner.argv_of("rm") == ["docker", "rm", "--force", name]


def test_each_run_gets_a_distinct_container_name(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner)
    spec = ToolSpec(name="t", description="", fn=sample_tool)
    executor.run(spec, {})
    executor.run(spec, {})

    names = [flag_values(argv, "--name")[0] for argv, _, _ in runner.calls]
    assert len(set(names)) == 2


def test_result_is_unmarshalled_from_the_envelope(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", {"rows": [1, 2]}), ""))
    executor = make_executor(tmp_path / "ws", runner)
    assert executor.run(ToolSpec(name="t", description="", fn=sample_tool), {}) == {"rows": [1, 2]}


def test_tool_output_before_the_envelope_is_ignored(tmp_path):
    noisy = "not json\n" + RESULT_SENTINEL + " garbage\n" + envelope_line("ok", "real")
    runner = FakeRunner(CommandResult(0, noisy, ""))
    executor = make_executor(tmp_path / "ws", runner)
    assert executor.run(ToolSpec(name="t", description="", fn=sample_tool), {}) == "real"


def test_tool_exception_becomes_a_runtime_error(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("error", "ValueError('nope')"), ""))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(RuntimeError, match=r"tool 't' failed: ValueError\('nope'\)") as exc:
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})
    assert not isinstance(exc.value, SandboxViolation)


def test_unresolvable_target_inside_the_image_is_a_violation(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("resolve_error", "ModuleNotFound"), ""))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="installed in the image"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})


def test_unserializable_result_is_a_violation(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("unserializable", "TypeError"), ""))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="cannot cross the container boundary"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})


def test_container_killed_by_a_limit_is_a_violation(tmp_path):
    # 137 == 128 + SIGKILL, which is what an OOM kill looks like from outside.
    runner = FakeRunner(CommandResult(137, "", "Killed"))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="memory and pid limits"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})


def test_broken_image_is_a_runtime_error_not_a_violation(tmp_path):
    # 127 == command not found: a misconfigured image, not a refused tool.
    runner = FakeRunner(CommandResult(127, "", "python: not found"))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(RuntimeError, match="no result envelope") as exc:
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})
    assert not isinstance(exc.value, SandboxViolation)


def test_missing_runtime_refuses_rather_than_running_locally(tmp_path, monkeypatch):
    LOCAL_CALLS.clear()
    monkeypatch.setattr("grapharc.harness.container.detect_runtime", lambda *a, **k: None)
    executor = ContainerExecutor(str(tmp_path / "ws"))  # no explicit runtime
    with pytest.raises(SandboxViolation, match="no container runtime"):
        executor.run(ToolSpec(name="t", description="", fn=tracked_tool), {})
    assert LOCAL_CALLS == []  # never degraded to running the tool in this process


def test_unexecutable_runtime_is_a_violation(tmp_path):
    def explode(argv, stdin_text, timeout):
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    executor = make_executor(tmp_path / "ws", explode, runtime="/nonexistent/docker")
    with pytest.raises(SandboxViolation, match="could not be executed"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})


# -- refusals that happen before any container starts ------------------------


class CallableObject:
    def __call__(self, **kwargs):
        return kwargs


class WithMethod:
    def method(self, **kwargs):
        return kwargs


@pytest.mark.parametrize(
    ("fn", "pattern"),
    [
        # A lambda has no name to import.
        (lambda **kw: kw, "lambda or a nested function"),
        # A partial and a callable object inherit __module__ from their class
        # but have no __qualname__ at all, so there is nothing to derive.
        (functools.partial(sample_tool, x=1), "no importable name"),
        (CallableObject(), "no importable name"),
        # A bound method has both, and they are well formed — they name the
        # underlying *function*, which is not this object. `self` cannot cross
        # the boundary, so only the resolve-back check catches this one.
        (WithMethod().method, "does not resolve back"),
    ],
)
def test_tools_without_an_import_path_are_refused_before_launch(tmp_path, fn, pattern):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", 1), ""))
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match=pattern):
        executor.run(ToolSpec(name="t", description="", fn=fn), {})
    assert runner.calls == []  # nothing was launched


def test_a_module_level_function_does_resolve(tmp_path):
    """The counterweight: the refusals above must not be refusing everything."""
    executor = make_executor(tmp_path / "ws", FakeRunner())
    spec = ToolSpec(name="t", description="", fn=sample_tool)
    assert executor.entrypoint_for(spec) == f"{__name__}:sample_tool"


# -- deciding how to containerize must not run host code ---------------------


def test_entrypoint_never_imports_the_tools_module_in_the_host(tmp_path, unimported_module):
    """The check that a tool resolves must not *execute* the host to find out.

    `entrypoint_for` runs in the unsandboxed parent. Importing a module there
    to verify a name runs that module's top-level code — arbitrary host code,
    during the setup for an isolated execution. Resolvability is verified from
    `sys.modules` alone, and deferred to the container when that cannot answer.
    """
    name, marker = unimported_module

    def fn(**kwargs):
        return kwargs

    # A tool whose module this process has not needed yet: a lazily-imported
    # plugin, a re-export, or anything a decorator has relabelled.
    fn.__module__ = name
    fn.__qualname__ = "tool"
    executor = make_executor(tmp_path / "ws", FakeRunner())

    entrypoint = executor.entrypoint_for(ToolSpec(name="t", description="", fn=fn))

    assert entrypoint == f"{name}:tool"  # derived, and left for the container
    assert not marker.exists(), "entrypoint_for executed the module in the host process"
    assert name not in sys.modules


def test_entrypoint_attempts_no_import_at_all(tmp_path, unimported_module):
    """The general form: not one import attempt, whatever the module turns out to be."""
    name, _ = unimported_module

    def fn(**kwargs):
        return kwargs

    fn.__module__ = name
    fn.__qualname__ = "tool"
    executor = make_executor(tmp_path / "ws", FakeRunner())

    tripwire = ImportTripwire()
    sys.meta_path.insert(0, tripwire)
    try:
        executor.entrypoint_for(ToolSpec(name="t", description="", fn=fn))
    finally:
        sys.meta_path.remove(tripwire)
    assert tripwire.attempts == [], f"the host tried to import {tripwire.attempts}"


def test_a_bound_method_is_refused_even_when_its_module_is_not_loaded(
    tmp_path, unimported_module
):
    """The shape check must not be a side effect of the module happening to be loaded.

    Deferring the *identity* check to the container is only safe because the
    cases that can never work are refused on the object's own type first. A
    bound method's derived path names the underlying function, and `self` stays
    here — true whether or not this process has imported the module.
    """
    name, marker = unimported_module

    class Holder:
        def method(self, **kwargs):
            return kwargs

    # No "<locals>": this is about binding, not about being nested.
    Holder.method.__qualname__ = "Holder.method"
    Holder.method.__module__ = name

    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="does not resolve back"):
        executor.run(ToolSpec(name="t", description="", fn=Holder().method), {})
    assert runner.calls == []
    assert not marker.exists()


def test_a_loaded_module_still_catches_a_path_that_names_something_else(tmp_path):
    """The other half: when the module *is* loaded, identity is still checked."""

    def impostor(**kwargs):
        return kwargs

    # Well-formed, importable, and names a different object in a module this
    # process has definitely loaded.
    impostor.__module__ = __name__
    impostor.__qualname__ = "sample_tool"

    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="does not resolve back"):
        executor.run(ToolSpec(name="t", description="", fn=impostor), {})
    assert runner.calls == []


def test_closure_is_refused(tmp_path):
    def outer():
        def inner(**kwargs):
            return kwargs

        return inner

    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="lambda or a nested function"):
        executor.run(ToolSpec(name="t", description="", fn=outer()), {})
    assert runner.calls == []


def test_main_module_tool_is_refused(tmp_path):
    def defined_here(**kwargs):
        return kwargs

    # A qualname without "<locals>" but a module that names a different file
    # inside the container. Both halves of the check have to bite.
    defined_here.__qualname__ = "defined_here"
    defined_here.__module__ = "__main__"
    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="__main__"):
        executor.run(ToolSpec(name="t", description="", fn=defined_here), {})
    assert runner.calls == []


def test_explicit_entrypoint_overrides_a_refusal(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", "done"), ""))
    executor = make_executor(
        tmp_path / "ws", runner, entrypoints={"t": "mypkg.tools:handler"}
    )
    assert executor.run(ToolSpec(name="t", description="", fn=lambda: None), {}) == "done"
    assert json.loads(runner.calls[0][1])["target"] == "mypkg.tools:handler"


def test_entrypoint_attribute_on_the_function_is_honoured(tmp_path):
    def marked(**kwargs):
        return kwargs

    marked.__grapharc_container_entrypoint__ = "mypkg.tools:marked"
    runner = FakeRunner(CommandResult(0, envelope_line("ok", None), ""))
    executor = make_executor(tmp_path / "ws", runner)
    executor.run(ToolSpec(name="t", description="", fn=marked), {})
    assert json.loads(runner.calls[0][1])["target"] == "mypkg.tools:marked"


@pytest.mark.parametrize(
    "bad", ["no-colon", "mod:", ":name", "mod:name; rm -rf /", "../etc:passwd"]
)
def test_malformed_entrypoints_are_refused(tmp_path, bad):
    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner, entrypoints={"t": bad})
    with pytest.raises(SandboxViolation, match="not of the form"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {})
    assert runner.calls == []


@pytest.mark.parametrize("bad", ["--privileged", "-v", "--user=0:0", ""])
def test_an_image_that_would_parse_as_a_run_flag_is_refused(tmp_path, bad):
    """The image is positional, so a leading dash makes it a flag instead.

    `image="--privileged"` would otherwise be handed to the runtime as a run
    flag and undo every hardening flag before it.
    """
    with pytest.raises(ValueError, match="must not start with"):
        ContainerExecutor(str(tmp_path / "ws"), image=bad)


def test_nonserializable_arguments_are_refused_before_launch(tmp_path):
    runner = FakeRunner()
    executor = make_executor(tmp_path / "ws", runner)
    with pytest.raises(SandboxViolation, match="cannot cross the container boundary"):
        executor.run(ToolSpec(name="t", description="", fn=sample_tool), {"handle": object()})
    assert runner.calls == []


# -- path translation --------------------------------------------------------


def test_container_path_translates_inside_the_workspace(tmp_path):
    executor = make_executor(tmp_path / "ws", FakeRunner())
    nested = tmp_path / "ws" / "a" / "b.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert executor.container_path(nested) == f"{WORKSPACE_MOUNT}/a/b.txt"
    assert executor.container_path(tmp_path / "ws") == WORKSPACE_MOUNT


def test_container_path_refuses_outside_the_workspace(tmp_path):
    executor = make_executor(tmp_path / "ws", FakeRunner())
    with pytest.raises(SandboxViolation, match="outside the workspace"):
        executor.container_path("/etc/passwd")


def test_sibling_directory_is_not_inside_the_workspace(tmp_path):
    executor = make_executor(tmp_path / "ws", FakeRunner())
    with pytest.raises(SandboxViolation):
        executor.container_path(str(tmp_path / "ws-evil") + "/secret")


# -- composed through the harness --------------------------------------------


def test_harness_drives_the_container_executor(tmp_path):
    runner = FakeRunner(CommandResult(0, envelope_line("ok", {"ok": True}), ""))
    registry = ToolRegistry().register(ToolSpec(name="probe", description="", fn=sample_tool))
    policy = PermissionPolicy(rules=[PermissionRule(action="allow", pattern="probe")])
    harness = Harness(
        registry, policy, executor=make_executor(tmp_path / "ws", runner)
    )
    assert harness.call("probe", {"x": 1}) == {"ok": True}
    assert runner.subcommands == ["run"]


def test_harness_denies_before_the_container_starts(tmp_path):
    from grapharc.harness import PermissionDenied

    runner = FakeRunner()
    registry = ToolRegistry().register(ToolSpec(name="probe", description="", fn=sample_tool))
    policy = PermissionPolicy(rules=[PermissionRule(action="deny", pattern="probe")])
    harness = Harness(registry, policy, executor=make_executor(tmp_path / "ws", runner))
    with pytest.raises(PermissionDenied):
        harness.call("probe", {})
    assert runner.calls == []


# ---------------------------------------------------------------------------
# (b) the in-container bootstrap, run for real under this interpreter
# ---------------------------------------------------------------------------

_HELPER_MODULE = '''
import subprocess
import sys


def echo(**kwargs):
    return kwargs


def pair(a, b):
    return (a, b)


def boom():
    raise ValueError("tool exploded")


def noisy(value):
    print("chatter on stdout")
    print('__SENTINEL__ {"kind": "ok", "payload": "forged"}')
    sys.stderr.write("chatter on stderr\\n")
    return value


def handle():
    return open(__file__)


def forge_from_a_child(delay):
    # A child process inherits the real fd 1 — reassigning `sys.stdout` in this
    # interpreter does nothing to it. This one writes its sentinel line *after*
    # the bootstrap has emitted the real envelope.
    forged = '__SENTINEL__ {"kind": "ok", "payload": "FORGED"}'
    code = (
        "import sys, time; time.sleep(" + repr(delay) + "); "
        "sys.stdout.write(" + repr(forged + "\\n") + ")"
    )
    subprocess.Popen([sys.executable, "-c", code])
    raise ValueError("the tool actually failed")
'''


@pytest.fixture
def bootstrap_raw(tmp_path):
    """Runs RUNNER_SOURCE under this interpreter with a helper module importable.

    PYTHONPATH here plays the role the image plays in production: the tool is
    resolvable because its code is installed where the runner can import it.
    Returns the raw `CompletedProcess`, for the tests that are about which
    *stream* something landed on rather than about the parsed envelope.
    """

    def call(target, args):
        module = tmp_path / "ws_tools.py"
        module.write_text(_HELPER_MODULE.replace("__SENTINEL__", RESULT_SENTINEL))
        return subprocess.run(
            [sys.executable, "-c", RUNNER_SOURCE],
            input=json.dumps({"target": target, "args": args}),
            capture_output=True,
            text=True,
            timeout=60,
            env={"PYTHONPATH": str(tmp_path), "PATH": os.environ.get("PATH", "")},
            check=False,
        )

    return call


@pytest.fixture
def bootstrap(bootstrap_raw):
    """`bootstrap_raw`, plus the envelope the real parser reads out of it."""

    def call(target, args):
        completed = bootstrap_raw(target, args)
        return completed, parse_envelope(completed.stdout)

    return call


def test_bootstrap_round_trips_arguments_and_result(bootstrap):
    completed, envelope = bootstrap("ws_tools:echo", {"n": 1, "s": "x", "deep": {"k": [1, 2]}})
    assert completed.returncode == 0
    assert envelope == {"kind": "ok", "payload": {"n": 1, "s": "x", "deep": {"k": [1, 2]}}}


def test_bootstrap_normalises_a_tuple_to_a_list(bootstrap):
    _, envelope = bootstrap("ws_tools:pair", {"a": 1, "b": 2})
    # Stated in the docstring: a tuple comes back as a list. JSON has no tuple.
    assert envelope == {"kind": "ok", "payload": [1, 2]}


def test_bootstrap_reports_a_raising_tool_as_error(bootstrap):
    _, envelope = bootstrap("ws_tools:boom", {})
    assert envelope["kind"] == "error"
    assert "ValueError" in envelope["payload"] and "tool exploded" in envelope["payload"]


def test_bootstrap_reports_an_unimportable_target(bootstrap):
    _, envelope = bootstrap("no_such_module_xyz:thing", {})
    assert envelope["kind"] == "resolve_error"
    assert "ModuleNotFoundError" in envelope["payload"]


def test_bootstrap_reports_a_missing_attribute(bootstrap):
    _, envelope = bootstrap("ws_tools:absent", {})
    assert envelope["kind"] == "resolve_error"
    assert "AttributeError" in envelope["payload"]


def test_bootstrap_reports_a_bad_keyword(bootstrap):
    _, envelope = bootstrap("ws_tools:pair", {"a": 1, "nope": 2})
    assert envelope["kind"] == "error"
    assert "TypeError" in envelope["payload"]


def test_bootstrap_refuses_an_unserializable_result(bootstrap):
    _, envelope = bootstrap("ws_tools:handle", {})
    assert envelope["kind"] == "unserializable"
    assert "TextIOWrapper" in envelope["payload"]


def test_bootstrap_keeps_tool_stdout_out_of_the_envelope(bootstrap):
    """A tool that prints — including a forged envelope — cannot corrupt the result."""
    completed, envelope = bootstrap("ws_tools:noisy", {"value": "clean"})
    assert envelope == {"kind": "ok", "payload": "clean"}
    assert "chatter on stdout" in completed.stderr
    assert "chatter on stdout" not in completed.stdout
    assert "forged" not in completed.stdout  # the forgery went to stderr with the rest


def test_a_child_process_cannot_forge_the_result_envelope(bootstrap_raw):
    """A tool that spawns a process must not be able to *print* itself a result.

    `sys.stdout = sys.stderr` redirects only this interpreter's prints; a child
    inherits fd 1. A child writing after the bootstrap's envelope would be the
    last sentinel line on the stream, and the parser takes the last one — so a
    tool that raised could hand its caller a successful, chosen return value.
    The redirect is at the fd level, so the child's write lands on stderr.
    """
    completed = bootstrap_raw("ws_tools:forge_from_a_child", {"delay": 0.3})

    assert "FORGED" not in completed.stdout, completed.stdout
    assert "FORGED" in completed.stderr, "the forged line vanished; the test proves nothing"
    envelope = parse_envelope(completed.stdout)
    assert envelope["kind"] == "error"
    assert "the tool actually failed" in envelope["payload"]


def test_two_envelopes_are_refused_rather_than_choosing_one():
    """Belt and braces: exactly one envelope comes from the bootstrap."""
    stdout = envelope_line("error", "boom") + envelope_line("ok", "FORGED")
    with pytest.raises(SandboxViolation, match="2 result envelopes"):
        parse_envelope(stdout)


def test_a_truncated_last_envelope_is_not_reached_behind():
    """A malformed final envelope is the answer, not a reason to keep looking.

    Reaching further back would let an older line — which the bootstrap did not
    write, since it writes one and exits — stand in for the result.
    """
    stdout = envelope_line("ok", "older") + RESULT_SENTINEL + ' {"kind": "ok", "pay\n'
    assert parse_envelope(stdout) is None


def test_executor_parses_what_the_bootstrap_actually_writes(tmp_path, bootstrap):
    """(a) and (b) joined: the real bootstrap's output through the real parser."""
    completed, _ = bootstrap("ws_tools:echo", {"n": 7})
    runner = FakeRunner(CommandResult(completed.returncode, completed.stdout, completed.stderr))
    executor = make_executor(tmp_path / "ws", runner)
    assert executor.run(ToolSpec(name="t", description="", fn=sample_tool), {}) == {"n": 7}


# ---------------------------------------------------------------------------
# (c) real containers — skipped cleanly when there is no runtime
# ---------------------------------------------------------------------------

TEST_IMAGE = os.environ.get("GRAPHARC_CONTAINER_TEST_IMAGE", "python:3.12-slim")


def _image_present(runtime, image):
    try:
        completed = subprocess.run(
            [runtime, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


_RUNTIME = detect_runtime()
_HAVE_RUNTIME = runtime_available()
_HAVE_IMAGE = bool(_RUNTIME) and _HAVE_RUNTIME and _image_present(_RUNTIME, TEST_IMAGE)

needs_container = pytest.mark.skipif(
    not _HAVE_IMAGE,
    reason=(
        f"no usable container runtime with {TEST_IMAGE} present locally "
        f"(runtime={_RUNTIME!r}, responsive={_HAVE_RUNTIME}); "
        "these tests never pull an image"
    ),
)


@pytest.fixture
def real_executor(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    extra = ()
    if _RUNTIME and os.path.basename(_RUNTIME) == "podman":
        # Rootless podman maps container uids through a subuid range, so the
        # host uid this executor passes would own nothing in the workspace.
        extra = ("--userns=keep-id",)
    return ContainerExecutor(str(workspace), image=TEST_IMAGE, extra_run_args=extra)


@needs_container
def test_real_container_round_trips_a_result(real_executor):
    spec = ToolSpec(name="dumps", description="", fn=json.dumps, timeout_seconds=120)
    assert real_executor.run(spec, {"obj": {"a": [1, 2]}}) == '{"a": [1, 2]}'


@needs_container
def test_real_container_sees_the_workspace(real_executor, tmp_path):
    (tmp_path / "ws" / "hello.txt").write_text("from the host\n")
    import linecache

    spec = ToolSpec(name="getline", description="", fn=linecache.getline, timeout_seconds=120)
    container_file = real_executor.container_path(tmp_path / "ws" / "hello.txt")
    assert container_file == f"{WORKSPACE_MOUNT}/hello.txt"
    assert real_executor.run(spec, {"filename": container_file, "lineno": 1}) == "from the host\n"


@needs_container
def test_real_container_cannot_see_the_host_filesystem(real_executor, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token")
    spec = ToolSpec(name="exists", description="", fn=os.path.exists, timeout_seconds=120)
    # The file is on the host and demonstrably readable from this process.
    assert os.path.exists(secret)
    # It is not mounted, so inside the container that path names nothing.
    assert real_executor.run(spec, {"path": str(secret)}) is False


@needs_container
def test_real_container_runs_as_non_root(real_executor):
    spec = ToolSpec(name="getuid", description="", fn=sample_tool, timeout_seconds=120)
    real_executor.entrypoints["getuid"] = "os:getuid"
    assert real_executor.run(spec, {}) != 0


@needs_container
def test_real_container_drops_capabilities_and_new_privileges(real_executor):
    """The kernel's own view, not the flags we passed: /proc/self/status."""
    import linecache

    spec = ToolSpec(name="status", description="", fn=linecache.getlines, timeout_seconds=120)
    fields = {}
    for line in real_executor.run(spec, {"filename": "/proc/self/status"}):
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    assert set(fields["CapEff"]) == {"0"}, f"effective capabilities: {fields['CapEff']}"
    assert set(fields["CapBnd"]) == {"0"}, f"bounding set: {fields['CapBnd']}"
    assert fields["NoNewPrivs"] == "1"
    assert fields["Uid"].split()[0] != "0"


@needs_container
def test_real_container_cannot_write_outside_the_workspace(real_executor, tmp_path):
    """The escape that broke the audit-hook sandbox, refused by the kernel instead.

    A `.pth` dropped in site-packages executes on every later interpreter
    start. Under the audit hook that took a second adversarial pass to close;
    here the rootfs is simply read-only.
    """
    import shutil

    (tmp_path / "ws" / "payload.pth").write_text("import os\n")
    spec = ToolSpec(name="copy", description="", fn=shutil.copyfile, timeout_seconds=120)
    source = f"{WORKSPACE_MOUNT}/payload.pth"

    with pytest.raises(RuntimeError) as exc:
        real_executor.run(
            spec,
            {"src": source, "dst": "/usr/local/lib/python3.12/site-packages/payload.pth"},
        )
    assert "Read-only file system" in str(exc.value), str(exc.value)

    # Control: the same copy inside the workspace succeeds, so the refusal
    # above is the rootfs being read-only and not the tool being broken.
    assert real_executor.run(spec, {"src": source, "dst": f"{WORKSPACE_MOUNT}/copy.pth"})
    assert (tmp_path / "ws" / "copy.pth").read_text() == "import os\n"


@needs_container
def test_real_container_has_no_network_by_default(real_executor):
    spec = ToolSpec(name="interfaces", description="", fn=os.listdir, timeout_seconds=120)
    real_executor.entrypoints["interfaces"] = "os:listdir"
    # --network none leaves exactly one interface: loopback.
    assert sorted(real_executor.run(spec, {"path": "/sys/class/net"})) == ["lo"]


@needs_container
def test_real_container_gets_a_network_when_declared(real_executor):
    spec = ToolSpec(
        name="interfaces", description="", fn=os.listdir, needs_network=True, timeout_seconds=120
    )
    real_executor.entrypoints["interfaces"] = "os:listdir"
    interfaces = set(real_executor.run(spec, {"path": "/sys/class/net"}))
    assert interfaces != {"lo"}, f"expected a network interface beyond loopback, got {interfaces}"


@needs_container
def test_real_container_is_killed_at_the_timeout(real_executor):
    spec = ToolSpec(name="sleep", description="", fn=subprocess.run, timeout_seconds=5.0)
    started = time.monotonic()
    with pytest.raises(SandboxViolation, match="exceeded its 5.0s timeout") as exc:
        real_executor.run(spec, {"args": ["sleep", "120"]})
    elapsed = time.monotonic() - started
    assert elapsed < 60, "the timeout did not cut the run short"

    match = re.search(r"container (\S+) was killed", str(exc.value))
    assert match, str(exc.value)
    name = match.group(1)
    # The container itself is gone, not merely the client that started it.
    for _ in range(20):
        listed = subprocess.run(
            [_RUNTIME, "ps", "--all", "--filter", f"name={name}", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if not listed.stdout.strip():
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"container {name} survived the timeout kill")


@needs_container
def test_real_container_refuses_an_unserializable_result(real_executor):
    import io

    spec = ToolSpec(name="buffer", description="", fn=io.StringIO, timeout_seconds=120)
    with pytest.raises(SandboxViolation, match="cannot cross the container boundary"):
        real_executor.run(spec, {})


@needs_container
def test_real_container_decides_the_resolution_the_host_deferred(
    real_executor, unimported_module
):
    """The deferred half of the resolvability check, answered where it is contained.

    The host refuses to import the module to find out whether the path is
    good. The container imports it — that is what a container is for — and the
    failure comes back as a refusal, not as a result. Fail-closed, and the
    module still never ran here.
    """
    name, marker = unimported_module

    def fn(**kwargs):
        return kwargs

    fn.__module__ = name
    fn.__qualname__ = "tool"
    spec = ToolSpec(name="deferred", description="", fn=fn, timeout_seconds=120)

    with pytest.raises(SandboxViolation, match="installed in the image"):
        real_executor.run(spec, {})
    assert not marker.exists()
    assert name not in sys.modules


@needs_container
def test_real_container_reports_a_tool_that_raises(real_executor):
    spec = ToolSpec(name="loads", description="", fn=json.loads, timeout_seconds=120)
    with pytest.raises(RuntimeError, match="tool 'loads' failed") as exc:
        real_executor.run(spec, {"s": "{not json"})
    assert not isinstance(exc.value, SandboxViolation)
