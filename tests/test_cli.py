"""Tests for the `grapharc` command line.

Every command is driven through `main(argv)` rather than a subprocess. The exit
code and the emitted document are the interface; spawning a process would test
the shell's argument splitting and hide the assertion behind a pipe.

Two things these tests will not do. They never call a model: `grapharc agent`
resolves its model through `grapharc.gateway.get_model`, which is replaced with
a scripted tool-calling double, so no test here can reach a provider. And they
never bind a socket: `serve` is checked by giving it a server package whose
runner records its arguments instead of running them.

The three packages the CLI delegates to — `grapharc.tools`, `grapharc.server`,
`grapharc.observe.replay` — are exercised as installed where they exist. Their
"not in this checkout" paths are exercised by putting `None` in `sys.modules`,
which is what CPython does to a module whose import is halted, so the command
takes the same branch it would on a machine that never had them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from grapharc.cli.agent import _approval
from grapharc.cli.main import build_parser, main
from grapharc.examples.stage0_dag import DEMO_DOC, build_stage0
from grapharc.harness import ToolSpec
from grapharc.observe.trace import TraceRecorder

# -- harness ------------------------------------------------------------------


def call(argv: list[str], capsys) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def call_json(argv: list[str], capsys) -> tuple[int, dict, str]:
    code, out, err = call([*argv, "--json"], capsys)
    return code, json.loads(out), err


class ToolCallingModel:
    """A scripted tool-calling backend.

    `AgentNode` needs a model that implements `bind_tools` and can emit
    `tool_calls`; `ScriptedChatModel` does neither. Recording what it was bound
    with is the point of the double as much as replaying turns: the schemas it
    receives are the evidence that the policy filtered the tool set before the
    model ever saw it.
    """

    def __init__(self, turns: list[dict]) -> None:
        self.turns = list(turns)
        self.bound_tools: list[str] | None = None
        self.turn_count = 0

    def bind_tools(self, schemas):
        self.bound_tools = [s["function"]["name"] for s in schemas]
        return self

    def invoke(self, messages):
        self.turn_count += 1
        turn = self.turns.pop(0) if self.turns else {"content": "done"}
        calls = [
            {
                "name": name,
                "args": args,
                "id": f"call-{self.turn_count}-{index}",
                "type": "tool_call",
            }
            for index, (name, args) in enumerate(turn.get("tools", ()))
        ]
        return AIMessage(
            content=turn.get("content", ""),
            tool_calls=calls,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


@pytest.fixture
def scripted_model(monkeypatch):
    def install(turns: list[dict]) -> ToolCallingModel:
        model = ToolCallingModel(turns)
        monkeypatch.setattr("grapharc.gateway.get_model", lambda spec, **kw: model)
        return model

    return install


@pytest.fixture
def stub_tools(monkeypatch):
    """A stand-in for `grapharc.tools`.

    The core toolset is another agent's package, and these tests have to hold
    whether or not it is installed — so the default toolset here is a stub
    registering the same shape of `ToolSpec`. The tools are real functions
    against the real workspace, so what they prove (a write happened, a denial
    stopped one) is real; only their provenance is stubbed.
    `test_agent_drives_the_real_core_toolset` covers the shipped one.
    """

    def install(*, entry: str = "register_core_tools") -> ModuleType:
        module = ModuleType("grapharc.tools")

        def build(workspace) -> list[ToolSpec]:
            root = Path(workspace)

            def write_note(path: str, content: str) -> str:
                """Write a note into the workspace."""
                (root / path).write_text(content, encoding="utf-8")
                return f"wrote {path}"

            def list_notes() -> str:
                """List the workspace."""
                return ",".join(sorted(p.name for p in root.iterdir()))

            def run_command(argv: str) -> str:
                """Stand-in for the shell tool, so there is something to deny."""
                return f"ran {argv}"

            def spawn_child() -> str:
                """Spawn a process: refused by the sandbox, allowed by the local executor."""
                done = subprocess.run(  # noqa: S603 — fixed argv, no shell
                    [sys.executable, "-c", "print('child ran')"],
                    capture_output=True,
                    text=True,
                )
                return done.stdout.strip()

            return [
                ToolSpec(name=fn.__name__, description=fn.__doc__ or "", fn=fn)
                for fn in (write_note, list_notes, run_command, spawn_child)
            ]

        if entry == "register_core_tools":

            def register_core_tools(registry, workspace):
                for spec in build(workspace):
                    registry.register(spec)
                return registry

            module.register_core_tools = register_core_tools
        else:
            module.core_tools = build

        monkeypatch.setitem(sys.modules, "grapharc.tools", module)
        return module

    return install


@pytest.fixture
def two_runs(tmp_path) -> Path:
    """A trace with three runs of one graph: `a` and `b` identical, `c` differing."""
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    doc = tmp_path / "doc.md"
    doc.write_text(DEMO_DOC, encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text(DEMO_DOC + "\n\nan extra sentence entirely.\n", encoding="utf-8")
    report = tmp_path / "report.md"
    graph = build_stage0(trace=trace)
    graph.invoke({"doc_path": str(doc), "report_path": str(report)}, run_id="a")
    graph.invoke({"doc_path": str(doc), "report_path": str(report)}, run_id="b")
    graph.invoke({"doc_path": str(other), "report_path": str(report)}, run_id="c")
    return tmp_path / "trace.jsonl"


# -- --json is on every command -----------------------------------------------


EVERY_COMMAND = [
    ["demo", "stage0"],
    ["agent", "do the thing"],
    ["serve"],
    ["models"],
    ["replay", "t.jsonl", "r1"],
    ["diff", "t.jsonl", "r1", "r2"],
    ["trace", "t.jsonl"],
    ["metrics", "t.jsonl", "r1"],
    ["viz", "t.jsonl", "r1"],
]


def test_every_command_accepts_json_after_its_arguments():
    """Requirement, not decoration: any command must be drivable from a script.

    Parsed rather than introspected, and with `--json` last, because that is
    where a shell user types it — a flag that only works before the positional
    arguments is a flag that does not work.
    """
    parser = build_parser()
    for argv in EVERY_COMMAND:
        args = parser.parse_args([*argv, "--json"])
        assert args.json is True, argv
        assert args.command == argv[0]


def test_building_the_parser_imports_no_optional_package():
    """`--help` must not depend on a package this checkout may not have.

    Run in a fresh interpreter on purpose: `sys.modules` in this one is already
    full of everything the rest of the suite imported, so the question can only
    be asked somewhere that has imported nothing yet.

    `grapharc.observe.replay` is not on the list — `grapharc.observe/__init__`
    imports it, and the CLI needs `grapharc.observe.metrics` to read a trace.
    """
    probe = (
        "import sys, grapharc.cli.main as m; m.build_parser(); "
        "print([n for n in "
        "('grapharc.tools','grapharc.server','grapharc.harness','grapharc.gateway',"
        "'fastapi','uvicorn','langchain_openai') if n in sys.modules])"
    )
    done = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


def test_json_failures_are_documents_on_stdout(tmp_path, capsys):
    code, payload, err = call_json(["metrics", str(tmp_path / "nope.jsonl"), "r1"], capsys)
    assert code == 2
    assert payload == {
        "ok": False,
        "command": "metrics",
        "error": f"no such trace file: {tmp_path / 'nope.jsonl'}",
    }
    assert err == ""


def test_text_failures_go_to_stderr(tmp_path, capsys):
    code, out, err = call(["metrics", str(tmp_path / "nope.jsonl"), "r1"], capsys)
    assert code == 2
    assert out == ""
    assert "no such trace file" in err


def test_a_read_of_a_missing_trace_creates_nothing(tmp_path, capsys):
    """`TraceRecorder` makes its parent directory; a typo in a reader must not."""
    missing = tmp_path / "never" / "trace.jsonl"
    code, _, _ = call(["trace", str(missing)], capsys)
    assert code == 2
    assert not missing.parent.exists()


# -- run / trace / metrics / viz ----------------------------------------------


def test_run_example_writes_a_trace_and_reports_it(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    code, out, _ = call(["demo", "stage0", "--trace", str(trace)], capsys)
    assert code == 0
    assert trace.exists()
    assert str(trace) in out


def test_run_example_json_carries_the_result(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    code, payload, _ = call_json(["demo", "stage0", "--trace", str(trace)], capsys)
    assert code == 0
    assert payload["ok"] is True
    assert payload["example"] == "stage0"
    assert payload["trace"] == str(trace)
    assert payload["result"]["report_written"] is True
    assert payload["result"]["counts"]


def test_run_against_a_model_emits_one_document(tmp_path, capsys, monkeypatch):
    """`--model` takes the live path. The `live` marker guards the spend, not the shape.

    `grapharc.cli.live` binds `get_model` at import, so the patch has to land on
    that name rather than on the gateway's.
    """
    from grapharc.testing import ScriptedChatModel

    monkeypatch.setattr(
        "grapharc.cli.live.get_model",
        lambda spec, **kw: ScriptedChatModel(
            responses=['{"term": "budgets"}', '{"term": "verifier"}'], on_exhausted="repeat"
        ),
    )
    code, payload, _ = call_json(
        ["demo", "stage1", "--trace", str(tmp_path / "t.jsonl"), "--model", "mock/x"], capsys
    )
    assert code == 0
    assert payload["live"] is True
    assert payload["model"] == "mock/x"
    assert payload["run_id"] == "live-stage1"
    assert payload["metrics"]["tokens"] > 0


def test_run_against_a_model_says_when_an_example_has_no_live_wiring(
    tmp_path, capsys, monkeypatch
):
    from grapharc.testing import ScriptedChatModel

    monkeypatch.setattr("grapharc.cli.live.get_model", lambda spec, **kw: ScriptedChatModel())
    code, payload, _ = call_json(
        ["demo", "stage0", "--trace", str(tmp_path / "t.jsonl"), "--model", "mock/x"], capsys
    )
    assert code == 1
    assert "no live wiring" in payload["error"]


def test_trace_and_metrics_read_the_same_record(two_runs, capsys):
    _, trace_payload, _ = call_json(["trace", str(two_runs), "--run-id", "a"], capsys)
    _, metrics_payload, _ = call_json(["metrics", str(two_runs), "a"], capsys)
    ends = [e for e in trace_payload["events"] if e["phase"] == "end"]
    assert trace_payload["count"] == len(trace_payload["events"])
    assert metrics_payload["nodes_executed"] == len(ends)
    assert metrics_payload["graph"] == trace_payload["events"][0]["graph"]


def test_metrics_for_an_unknown_run_exits_one(two_runs, capsys):
    code, payload, _ = call_json(["metrics", str(two_runs), "ghost"], capsys)
    assert code == 1
    assert payload["ok"] is False
    assert "ghost" in payload["error"]


def test_viz_renders_the_executed_path(two_runs, capsys):
    code, payload, _ = call_json(["viz", str(two_runs), "a"], capsys)
    assert code == 0
    assert payload["mermaid"].startswith("flowchart TD")
    assert "load" in payload["mermaid"]


# Every reading command, with the arguments it needs beyond the path. `metrics`
# and `viz` never reach the run id: the file refuses before any run is looked up.
READERS = [["trace"], ["metrics", "r1"], ["viz", "r1"]]


def _bad_trace(tmp_path) -> Path:
    """A trace whose second line is not an event — a process killed mid-write."""
    trace = TraceRecorder(tmp_path / "bad.jsonl")
    trace.event(run_id="r1", graph="g", node="load", phase="start", step=1)
    with trace.path.open("a", encoding="utf-8") as f:
        f.write('{"not": "a trace event"}\n')
    return trace.path


@pytest.mark.parametrize("argv", READERS, ids=lambda argv: argv[0])
def test_a_malformed_trace_is_a_report_not_a_traceback(argv, tmp_path, capsys):
    """The contract in `output.py` names "an unreadable trace" as exit 2."""
    bad = _bad_trace(tmp_path)
    code, out, err = call([argv[0], str(bad), *argv[1:]], capsys)
    assert code == 2
    assert out == ""
    assert f"error: unreadable trace file: {bad}: line 2 is not a trace event\n" == err


@pytest.mark.parametrize("argv", READERS, ids=lambda argv: argv[0])
def test_a_malformed_trace_fails_as_one_json_document(argv, tmp_path, capsys):
    bad = _bad_trace(tmp_path)
    code, payload, err = call_json([argv[0], str(bad), *argv[1:]], capsys)
    assert code == 2
    assert payload == {
        "ok": False,
        "command": argv[0],
        "error": f"unreadable trace file: {bad}: line 2 is not a trace event",
    }
    assert err == ""


# A path that exists but cannot be read is the same class of failure as a
# malformed one, and used to escape as a traceback with exit 1: `_existing_trace`
# tested `exists()` and the handlers catch only `TraceReadError`, so every other
# `OSError` went straight past both.
@pytest.mark.parametrize("argv", READERS, ids=lambda argv: argv[0])
def test_a_directory_where_a_trace_belongs_is_a_report_not_a_traceback(
    argv, tmp_path, capsys
):
    directory = tmp_path / "adir"
    directory.mkdir()
    code, out, err = call([argv[0], str(directory), *argv[1:]], capsys)
    assert code == 2
    assert out == ""
    assert err.startswith(f"error: unreadable trace file: {directory}: ")
    assert "Traceback" not in err


@pytest.mark.parametrize("argv", READERS, ids=lambda argv: argv[0])
def test_a_directory_where_a_trace_belongs_fails_as_one_json_document(
    argv, tmp_path, capsys
):
    directory = tmp_path / "adir"
    directory.mkdir()
    code, payload, err = call_json([argv[0], str(directory), *argv[1:]], capsys)
    assert code == 2
    assert payload["ok"] is False
    assert payload["command"] == argv[0]
    assert payload["error"].startswith(f"unreadable trace file: {directory}: ")
    assert err == ""


# -- models -------------------------------------------------------------------


def test_models_resolves_a_spec(capsys):
    code, payload, _ = call_json(["models", "openrouter/anthropic/claude-haiku-4.5"], capsys)
    assert code == 0
    assert payload["backend"] == "openrouter"
    assert payload["model"] == "anthropic/claude-haiku-4.5"


def test_models_quotes_no_model_count(capsys):
    """The old help line advertised '~400 models'. Nothing here counts them."""
    _, out, _ = call(["models"], capsys)
    assert "400 models" not in out
    assert "~400" not in out


def test_models_check_never_prints_the_key(monkeypatch, capsys):
    secret = "sk-or-v1-0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    code, out, _ = call(["models", "--check"], capsys)
    _, payload, _ = call_json(["models", "--check"], capsys)
    assert code == 0  # a key is configured, so at least one provider is usable
    assert secret not in out
    assert secret not in json.dumps(payload)
    openrouter = next(b for b in payload["backends"] if b["backend"] == "openrouter")
    assert openrouter["usable"] is True
    assert openrouter["credential"] == "sk-or-v…cdef"


def test_models_check_exits_one_when_nothing_is_configured(monkeypatch, capsys):
    for name in (
        "OPENROUTER_API_KEY", "OPENROUTER_KEY", "open-router-api-key",
        "OPENAI_API_KEY", "OPENAI_KEY", "openai-api-key",
        "OLLAMA_HOST", "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("grapharc.gateway.config.find_env_file", lambda start=None: None)
    # No claude binary and no ollama binary: nothing on this machine is real.
    monkeypatch.setattr("grapharc.cli.probe.shutil.which", lambda _: None)
    code, payload, _ = call_json(["models", "--check"], capsys)
    assert code == 1
    assert payload["ok"] is False
    usable = {b["backend"]: b["usable"] for b in payload["backends"]}
    assert usable == {
        "claude-cli": False,
        "openrouter": False,
        "openai": False,
        "ollama": False,
        "mock": True,
    }
    for backend in ("openrouter", "openai"):
        assert next(b for b in payload["backends"] if b["backend"] == backend)[
            "credential"
        ] == "<unset>"


def test_models_check_never_prints_the_openai_key(monkeypatch, capsys):
    secret = "sk-proj-0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    code, out, _ = call(["models", "--check"], capsys)
    _, payload, _ = call_json(["models", "--check"], capsys)
    assert code == 0  # a key is configured, so at least one provider is usable
    assert secret not in out
    assert secret not in json.dumps(payload)
    openai = next(b for b in payload["backends"] if b["backend"] == "openai")
    assert openai["usable"] is True
    assert openai["credential"] == "sk-proj…cdef"


def test_models_lists_the_ollama_address_but_no_key(monkeypatch, capsys):
    """It is an address, not a secret — printed whole, and not a claim that
    anything is listening on it."""
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    _, payload, out = call_json(["models"], capsys)
    assert payload["ollama_base_url"] == "http://gpu-box:11434/v1"
    assert "ollama_key" not in payload


def test_models_check_flags_a_backend_it_cannot_probe(monkeypatch, capsys):
    """A backend added to the gateway must not silently vanish from the report."""
    monkeypatch.setattr(
        "grapharc.gateway.registry.BACKENDS", ("claude-cli", "openrouter", "mock", "bedrock")
    )
    _, payload, _ = call_json(["models", "--check"], capsys)
    bedrock = next(b for b in payload["backends"] if b["backend"] == "bedrock")
    assert bedrock["usable"] is None
    assert "no probe" in bedrock["detail"]

    _, out, _ = call(["models", "--check"], capsys)
    assert "bedrock      unprobed" in out


def test_models_check_and_a_spec_are_different_questions(capsys):
    code, payload, _ = call_json(["models", "claude-cli/x", "--check"], capsys)
    assert code == 2
    assert "one or the other" in payload["error"]


# -- agent --------------------------------------------------------------------


def test_agent_runs_a_tool_under_a_policy(tmp_path, capsys, scripted_model, stub_tools):
    stub_tools()
    workspace = tmp_path / "ws"
    model = scripted_model(
        [
            {"tools": [("write_note", {"path": "note.txt", "content": "hello"})]},
            {"content": "wrote the note"},
        ]
    )
    code, payload, _ = call_json(
        [
            "agent",
            "write a note",
            "--model",
            "mock/x",
            "--workspace",
            str(workspace),
            "--deny",
            "run_command",
            "--run-id",
            "cli-test",
        ],
        capsys,
    )

    assert code == 0
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hello"
    assert payload["termination_reason"] == "target_met"
    assert payload["answer"] == "wrote the note"
    assert payload["tool_calls"][0]["tool"] == "write_note"
    assert payload["tool_calls"][0]["status"] == "ok"
    assert payload["tokens"] == 30  # two turns at 15 each, metered not guessed

    # What it was allowed to do, from both sides of the fence: the denied tool
    # is absent from the report *and* was never described to the model.
    assert "run_command" not in payload["tools_visible"]
    assert "run_command" not in model.bound_tools
    assert "write_note" in model.bound_tools

    assert Path(payload["trace"]).exists()
    assert payload["run_id"] == "cli-test"


def test_agent_drives_the_real_core_toolset(tmp_path, capsys, scripted_model):
    """The shipped toolset, when this checkout has one."""
    pytest.importorskip("grapharc.tools")
    workspace = tmp_path / "ws"
    model = scripted_model(
        [
            {"tools": [("write_file", {"path": "note.txt", "content": "hello"})]},
            {"content": "wrote the note"},
        ]
    )
    code, payload, _ = call_json(
        [
            "agent",
            "write a note",
            "--model",
            "mock/x",
            "--workspace",
            str(workspace),
            "--deny",
            "run_command",
        ],
        capsys,
    )
    assert code == 0, payload
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hello"
    assert "write_file" in payload["tools_visible"]
    assert "run_command" not in model.bound_tools


def test_agent_trace_is_readable_by_the_other_commands(
    tmp_path, capsys, scripted_model, stub_tools
):
    """The agent's trace is the same JSONL the readers consume — not a second format."""
    stub_tools()
    workspace = tmp_path / "ws"
    scripted_model([{"tools": [("list_notes", {})]}, {"content": "empty"}])
    _, agent_payload, _ = call_json(
        ["agent", "look around", "--model", "mock/x", "--workspace", str(workspace)], capsys
    )
    code, trace_payload, _ = call_json(
        ["trace", agent_payload["trace"], "--run-id", agent_payload["run_id"]], capsys
    )
    assert code == 0
    phases = {event["phase"] for event in trace_payload["events"]}
    assert {"model", "tool", "stop"} <= phases


def test_agent_ask_is_refused_when_there_is_nobody_to_ask(
    tmp_path, capsys, scripted_model, stub_tools
):
    stub_tools()
    workspace = tmp_path / "ws"
    scripted_model(
        [
            {"tools": [("write_note", {"path": "note.txt", "content": "hello"})]},
            {"content": "I was not allowed to write"},
        ]
    )
    code, payload, _ = call_json(
        [
            "agent",
            "write a note",
            "--model",
            "mock/x",
            "--workspace",
            str(workspace),
            "--ask",
            "write_note",
        ],
        capsys,
    )
    assert code == 0
    assert payload["denied"] == 1
    assert payload["tool_calls"][0]["refused_by"] == "policy"
    assert not (workspace / "note.txt").exists()


def test_approval_fails_closed_in_json_mode_even_on_a_tty(monkeypatch):
    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert _approval(as_json=True, stream=Tty())("write_file", {}) is False
    assert _approval(as_json=False, stream=Tty())("write_file", {}) is True


def test_agent_accepts_a_core_tools_factory(tmp_path, capsys, scripted_model, stub_tools):
    """The second supported shape of `grapharc.tools`, so it is not dead code."""
    stub_tools(entry="core_tools")
    scripted_model(
        [{"tools": [("write_note", {"path": "n.txt", "content": "x"})]}, {"content": "done"}]
    )
    code, payload, _ = call_json(
        ["agent", "note", "--model", "mock/x", "--workspace", str(tmp_path / "ws")], capsys
    )
    assert code == 0
    assert payload["tools_from"] == "grapharc.tools.core_tools"
    assert payload["tools_visible"] == ["list_notes", "run_command", "spawn_child", "write_note"]
    assert payload["tool_calls"][0]["detail"] == "wrote n.txt"
    assert (tmp_path / "ws" / "n.txt").exists()


def test_agent_refuses_a_toolset_shape_it_cannot_read(monkeypatch, tmp_path, capsys):
    stub = ModuleType("grapharc.tools")
    stub.core_tools = lambda workspace: {"read_file": "not a ToolSpec"}
    monkeypatch.setitem(sys.modules, "grapharc.tools", stub)
    code, payload, _ = call_json(
        ["agent", "x", "--model", "mock/x", "--workspace", str(tmp_path / "ws")], capsys
    )
    assert code == 2
    assert "expects a ToolRegistry" in payload["error"]


def test_agent_reports_a_missing_toolset(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.tools", None)
    code, payload, _ = call_json(
        ["agent", "x", "--model", "mock/x", "--workspace", str(tmp_path / "ws")], capsys
    )
    assert code == 2
    assert "grapharc.tools" in payload["error"]
    assert "ROADMAP" in payload["error"]


def test_agent_rejects_a_model_that_cannot_call_tools(
    monkeypatch, tmp_path, capsys, stub_tools
):
    stub_tools()

    class TextOnly:
        def bind_tools(self, schemas):
            raise NotImplementedError

    monkeypatch.setattr("grapharc.gateway.get_model", lambda spec, **kw: TextOnly())
    code, payload, _ = call_json(
        ["agent", "x", "--model", "claude-cli/y", "--workspace", str(tmp_path / "ws")], capsys
    )
    assert code == 2
    assert "bind_tools" in payload["error"]
    # The failure still records what the run was configured to do.
    assert payload["tools_visible"]
    assert payload["policy"] == {"allow": ["*"], "ask": [], "deny": []}


def test_executor_flag_picks_a_real_boundary(tmp_path, capsys, scripted_model, stub_tools):
    """`--executor local` is not a label: the sandbox refuses a spawn and local runs it."""
    stub_tools()
    argv = ["agent", "spawn", "--model", "mock/x", "--workspace", str(tmp_path / "ws")]

    scripted_model([{"tools": [("spawn_child", {})]}, {"content": "done"}])
    _, sandboxed, _ = call_json(argv, capsys)
    scripted_model([{"tools": [("spawn_child", {})]}, {"content": "done"}])
    _, local, _ = call_json([*argv, "--executor", "local"], capsys)

    assert sandboxed["executor"] == "sandbox"
    assert sandboxed["tool_calls"][0]["refused_by"] == "sandbox"
    assert sandboxed["refused"] == 1
    assert local["executor"] == "local"
    assert local["tool_calls"][0]["status"] == "ok"
    assert local["tool_calls"][0]["detail"] == "child ran"


def test_agent_wall_clock_ceiling_cuts_off_a_call_in_flight(
    monkeypatch, tmp_path, capsys, stub_tools
):
    """`--max-seconds` is enforced during a call, not only checked between turns."""
    stub_tools()

    class SlowModel:
        def bind_tools(self, schemas):
            return self

        def invoke(self, messages):
            time.sleep(30)
            return AIMessage(content="too late")

    monkeypatch.setattr("grapharc.gateway.get_model", lambda spec, **kw: SlowModel())
    started = time.monotonic()
    code, payload, _ = call_json(
        [
            "agent",
            "wait",
            "--model",
            "mock/x",
            "--workspace",
            str(tmp_path / "ws"),
            "--max-seconds",
            "0.5",
        ],
        capsys,
    )
    elapsed = time.monotonic() - started
    assert code == 1
    assert "max_seconds" in payload["error"]
    assert elapsed < 10, "the sleeping call was waited out rather than interrupted"


def test_agent_reports_an_unusable_model_spec(tmp_path, capsys, stub_tools):
    stub_tools()
    code, payload, _ = call_json(
        ["agent", "x", "--model", "notabackend/m", "--workspace", str(tmp_path / "ws")], capsys
    )
    assert code == 2
    assert "could not build model" in payload["error"]


def test_agent_stops_at_the_turn_cap_and_says_so(tmp_path, capsys, scripted_model, stub_tools):
    stub_tools()
    scripted_model(
        [
            {"tools": [("write_note", {"path": f"n{n}.txt", "content": str(n)})]}
            for n in range(6)
        ]
    )
    code, payload, _ = call_json(
        [
            "agent",
            "explore",
            "--model",
            "mock/x",
            "--workspace",
            str(tmp_path / "ws"),
            "--max-turns",
            "2",
        ],
        capsys,
    )
    assert code == 1
    assert payload["termination_reason"] == "max_iterations"
    assert payload["turns"] == 2
    assert payload["answer"] == ""  # a run that stopped short has no answer
    assert payload["partial_output"] == ""


# -- serve --------------------------------------------------------------------


def _server_stub(record: dict, *, with_runner: bool = True) -> ModuleType:
    stub = ModuleType("grapharc.server")

    def create_app(**kwargs):
        record["create_app"] = kwargs
        return "THE-APP"

    stub.create_app = create_app
    if with_runner:

        def serve(app, **kwargs):
            record["served"] = (app, kwargs)

        stub.serve = serve
    return stub


def test_serve_hands_the_app_to_the_server_packages_runner(monkeypatch, capsys):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))
    code, payload, _ = call_json(["serve", "--host", "0.0.0.0", "--port", "9111"], capsys)
    assert code == 0
    assert record["served"] == (
        "THE-APP",
        {"host": "0.0.0.0", "port": 9111, "log_level": "info"},
    )
    assert payload["url"] == "http://0.0.0.0:9111"
    assert payload["graphs"] == []


def test_serve_falls_back_to_uvicorn(monkeypatch, capsys):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record, with_runner=False))
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: record.setdefault("uvicorn", (app, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    code, _, _ = call(["serve", "--port", "9112"], capsys)
    assert code == 0
    assert record["uvicorn"][0] == "THE-APP"
    assert record["uvicorn"][1]["port"] == 9112


def test_serve_serves_the_registry_it_was_given(monkeypatch, capsys):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))

    class FakeRegistry:
        def names(self):
            return ["triage"]

    graphs = ModuleType("cli_test_graphs")
    graphs.REGISTRY = FakeRegistry()
    monkeypatch.setitem(sys.modules, "cli_test_graphs", graphs)

    code, payload, _ = call_json(["serve", "--registry", "cli_test_graphs:REGISTRY"], capsys)
    assert code == 0
    assert record["create_app"]["registry"] is graphs.REGISTRY
    assert payload["graphs"] == ["triage"]


def test_serve_rejects_a_registry_that_is_not_module_attr(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub({}))
    code, payload, _ = call_json(["serve", "--registry", "just_a_module"], capsys)
    assert code == 2
    assert "module:attr" in payload["error"]


def test_serve_reports_a_missing_server_package(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.server", None)
    code, payload, _ = call_json(["serve"], capsys)
    assert code == 2
    assert "grapharc.server" in payload["error"]
    assert "server extra" in payload["error"]


def test_serve_live_root_reaches_create_app_and_the_startup_lines(
    monkeypatch, capsys, tmp_path
):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))
    code, out, _ = call(["serve", "--live-root", str(tmp_path)], capsys)
    assert code == 0
    assert record["create_app"] == {"live_root": str(tmp_path)}
    assert "live view" in out and "/live" in out


def test_serve_live_token_is_passed_only_with_a_live_root(monkeypatch, capsys, tmp_path):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))
    code, _, _ = call(
        ["serve", "--live-root", str(tmp_path), "--live-token", "s3cret"], capsys
    )
    assert code == 0
    assert record["create_app"]["live_token"] == "s3cret"

    record.clear()
    code, _, _ = call(["serve", "--live-token", "s3cret"], capsys)
    assert code == 0
    assert record["create_app"] == {}


def test_serve_without_live_root_prints_no_live_line(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub({}))
    code, out, _ = call(["serve"], capsys)
    assert code == 0
    assert "live view" not in out  # the cookbook pins the three plain lines


def test_serve_live_root_off_loopback_warns_about_exposure(monkeypatch, capsys, tmp_path):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))
    _, loopback_out, _ = call(["serve", "--live-root", str(tmp_path)], capsys)
    assert "tunnel" not in loopback_out
    _, exposed_out, _ = call(
        ["serve", "--live-root", str(tmp_path), "--host", "0.0.0.0"], capsys
    )
    assert "tunnel" in exposed_out


def test_serve_live_root_must_be_a_directory(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub({}))
    code, payload, _ = call_json(["serve", "--live-root", str(tmp_path / "absent")], capsys)
    assert code == 2
    assert "--live-root" in payload["error"]


# -- replay / diff ------------------------------------------------------------


class FakeDiff(BaseModel):
    run_a: str
    run_b: str
    identical: bool


@pytest.fixture
def stub_engine(monkeypatch):
    """A stand-in for `grapharc.observe.replay`, recording what the CLI passed it.

    The engine is another agent's module; these tests are about the CLI's half
    of the contract — that it resolves the entry point, hands over a reader for
    the right file, renders through the engine's own formatter, and turns the
    verdict into an exit code — and they must hold with or without it.
    """
    record: dict = {}

    module = ModuleType("grapharc.observe.replay")

    def replay(source, run_id):
        record["replay"] = (source, run_id)
        if run_id == "boom":
            raise ValueError("no events for run_id 'boom'")
        return {"run_id": run_id, "path": ["load", "report"]}

    def diff_trace(source, run_a, run_b):
        record["diff"] = (source, run_a, run_b)
        return FakeDiff(run_a=run_a, run_b=run_b, identical=run_b == "b")

    module.replay = replay
    module.diff_trace = diff_trace
    module.format_replay = lambda run: f"FORMATTED-REPLAY {run['run_id']}"
    module.format_diff = lambda d: f"FORMATTED-DIFF {d.run_a}->{d.run_b}"
    monkeypatch.setitem(sys.modules, "grapharc.observe.replay", module)
    return record


def test_replay_delegates_to_the_engine(two_runs, capsys, stub_engine):
    code, payload, _ = call_json(["replay", str(two_runs), "a"], capsys)
    assert code == 0
    assert payload["via"] == "grapharc.observe.replay.replay"
    assert payload["result"] == {"run_id": "a", "path": ["load", "report"]}
    source, run_id = stub_engine["replay"]
    assert isinstance(source, TraceRecorder)
    assert source.path == two_runs
    assert run_id == "a"


def test_replay_renders_through_the_engines_formatter(two_runs, capsys, stub_engine):
    code, out, _ = call(["replay", str(two_runs), "a"], capsys)
    assert code == 0
    assert out.strip() == "FORMATTED-REPLAY a"


def test_replay_reports_an_engine_failure(two_runs, capsys, stub_engine):
    code, payload, _ = call_json(["replay", str(two_runs), "boom"], capsys)
    assert code == 1
    assert "boom" in payload["error"]


def test_replay_of_a_missing_file_exits_two(tmp_path, capsys, stub_engine):
    code, payload, _ = call_json(["replay", str(tmp_path / "nope.jsonl"), "a"], capsys)
    assert code == 2
    assert "no such trace file" in payload["error"]
    assert "replay" not in stub_engine  # the engine is not asked about a file we do not have


def test_replay_reports_a_missing_engine(monkeypatch, two_runs, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.observe.replay", None)
    code, payload, _ = call_json(["replay", str(two_runs), "a"], capsys)
    assert code == 2
    assert "grapharc.observe.replay" in payload["error"]


def test_diff_of_identical_runs_exits_zero(two_runs, capsys, stub_engine):
    code, payload, _ = call_json(["diff", str(two_runs), "a", "b"], capsys)
    assert code == 0
    assert payload["identical"] is True
    assert stub_engine["diff"][1:] == ("a", "b")


def test_diff_exits_one_when_the_runs_differ(two_runs, capsys, stub_engine):
    """`diff(1)`'s convention: a difference is reportable in `$?`, not only on stdout."""
    code, payload, _ = call_json(["diff", str(two_runs), "a", "c"], capsys)
    assert code == 1
    assert payload["identical"] is False


def test_diff_renders_through_the_engines_formatter(two_runs, capsys, stub_engine):
    _, out, _ = call(["diff", str(two_runs), "a", "b"], capsys)
    assert out.strip() == "FORMATTED-DIFF a->b"


def test_diff_reports_a_missing_engine(monkeypatch, two_runs, capsys):
    monkeypatch.setitem(sys.modules, "grapharc.observe.replay", None)
    code, payload, _ = call_json(["diff", str(two_runs), "a", "b"], capsys)
    assert code == 2
    assert "ROADMAP" in payload["error"]


# The shipped engine, when this checkout has one. Everything above holds either
# way; these two say the wiring matches the real thing rather than only the stub.


def test_replay_against_the_shipped_engine(two_runs, capsys):
    pytest.importorskip("grapharc.observe.replay")
    code, payload, _ = call_json(["replay", str(two_runs), "a"], capsys)
    assert code == 0, payload
    assert payload["result"]["run_id"] == "a"
    assert [e["node"] for e in payload["result"]["executions"]] == [
        "load",
        "split",
        "count",
        "report",
    ]


def test_diff_against_the_shipped_engine(two_runs, capsys):
    pytest.importorskip("grapharc.observe.replay")
    same, identical_payload, _ = call_json(["diff", str(two_runs), "a", "b"], capsys)
    differs, differing_payload, _ = call_json(["diff", str(two_runs), "a", "c"], capsys)
    assert (same, identical_payload["identical"]) == (0, True)
    assert (differs, differing_payload["identical"]) == (1, False)


# --------------------------------------------------------------------------
# §12.4 / §8.7 — `--memory PATH`. The durable claim store existed and no
# shipped command used one, so every `grapharc run` forgot everything the
# moment it exited. The property below is the only one that matters: a second
# run, in a second process, recalls what the first one persisted.
# --------------------------------------------------------------------------


def test_run_without_memory_stays_in_process_and_writes_no_file(tmp_path, capsys):
    """The default must not start writing files nobody asked for."""
    code, _, _ = call(["demo", "capstone", "--trace", str(tmp_path / "t.jsonl")], capsys)

    assert code == 0
    assert list(tmp_path.glob("*.sqlite")) == []


def test_run_with_memory_persists_claims_to_the_named_store(tmp_path, capsys):
    store = tmp_path / "claims.sqlite"

    code, payload, _ = call_json(
        ["demo", "capstone", "--trace", str(tmp_path / "t.jsonl"), "--memory", str(store)],
        capsys,
    )

    assert code == 0
    assert payload["result"]["persisted_claim_ids"]
    assert store.exists()


def test_a_second_run_recalls_what_the_first_one_persisted(tmp_path, capsys):
    """Durability, stated as the behaviour a reader would check."""
    store = tmp_path / "claims.sqlite"
    args = ["demo", "capstone", "--trace", str(tmp_path / "t.jsonl"), "--memory", str(store)]

    _, first, _ = call_json([*args], capsys)
    _, second, _ = call_json([*args], capsys)

    assert "No prior knowledge" in first["result"]["recalled"]
    assert "No prior knowledge" not in second["result"]["recalled"]
    assert first["result"]["persisted_claim_ids"] != second["result"]["persisted_claim_ids"]


def test_the_durable_store_survives_a_real_process_boundary(tmp_path):
    """Two interpreters, one file — an in-process dict cannot fake this."""
    store = tmp_path / "claims.sqlite"
    cmd = [
        sys.executable, "-m", "grapharc.cli.main", "demo", "capstone",
        "--json", "--trace", str(tmp_path / "t.jsonl"), "--memory", str(store),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    second = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "No prior knowledge" not in json.loads(second.stdout)["result"]["recalled"]


# --------------------------------------------------------------------------
# §12.1 — `grapharc plan`. The governed loop existed, was tested, and no
# shipped command drove it, so the crux of the architecture was reachable only
# by importing the library. These pin the surface *and* the two properties it
# exists to make visible: a refusal is recorded, and a later round re-enters
# the gate rather than inheriting round 1's answer.
# --------------------------------------------------------------------------

_DENY_DEPLOY = """version = "1"
default = "allow"

[[rule]]
id = "no-deploy"
resource = "edge"
match = "*->deploy"
effect = "deny"
"""

_PERMISSIVE = 'version = "1"\ndefault = "allow"\n'

# Issue #66's document, plus the catch-all its author's `default = "deny"`
# needs. Its `node` rules used to be compiled by nobody: the run admitted the
# denied kind and executed it, and the only hint was a rule count in the banner.
_DENY_DEPLOY_NODE = """version = "1"
default = "deny"

[[rule]]
id = "no-deploy-node"
resource = "node"
match = "deploy"
effect = "deny"
reason = "deploying from a plan is never permitted"

[[rule]]
id = "other-nodes-run"
resource = "node"
match = "*"
effect = "allow"

[[rule]]
id = "ordinary-work-flows"
resource = "edge"
match = "*->*"
effect = "allow"
"""


def test_plan_runs_the_governed_loop_and_reports_every_round(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    code, out, _ = call(
        ["plan", "look into the outage", "--scripted", "--go", "--trace",
         str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert "goal_met" in out
    assert "round 1: rejected" in out
    assert "edge_denied" in out
    assert "round 2: admitted" in out


def test_plan_traces_the_plan_and_every_node_it_executed(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    """README's "the trace holds an `admission` event per round, a `round` event
    per round, the executed nodes' own `start`/`end` pairs and one `stop` event".

    That was true of a hand-wired loop and false of the shipped registry, which is
    the one this command drives: it withheld the recorder from the `PlannerNode`
    and the `Materializer`, so the file held no `plan` event and — because the
    built subgraph inherits the materializer's recorder — no `start`/`end` pair
    for any node the loop ran. Three nodes executed and none of them appeared.
    """
    from grapharc.observe import cost, metrics
    from grapharc.observe.trace import TraceRecorder

    path = tmp_path / "t.jsonl"
    code, payload, _ = call_json([
        "plan",
        "look into the outage",
        "--scripted",
        "--go",
        "--trace",
        str(path),
    ], capsys)
    assert code == 0

    recorder = TraceRecorder(path)
    events = recorder.read_events()
    run_id = events[0].run_id
    phases = {phase: 0 for phase in ("plan", "admission", "round", "start", "end", "stop")}
    for event in events:
        if event.phase in phases:
            phases[event.phase] += 1

    assert phases["plan"] == 2, "the planner's own event is missing"
    assert phases["admission"] == 2
    assert phases["round"] == 2
    assert phases["stop"] == 1
    # The admitted round ran three nodes; each owes a start/end pair.
    assert phases["start"] == 3
    assert phases["end"] == 3

    summary = metrics.summarize(recorder, run_id)
    assert summary.nodes_executed == 3
    assert set(summary.per_node) == {"triage", "patch", "verify"}
    # And the two readers of that one file still agree with each other.
    assert cost.attribute(recorder, run_id).tokens == summary.tokens


def test_plan_json_carries_the_rounds_and_the_stop_reason(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    code, payload, _ = call_json(
        ["plan", "look into the outage", "--scripted", "--go", "--trace",
         str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["stop"] == "goal_met"
    assert payload["rejections"] == ["edge_denied"]
    assert [r["status"] for r in payload["rounds"]] == ["rejected", "admitted"]
    assert payload["rounds"][0]["executed"] is False
    assert payload["rounds"][1]["executed"] is True
    assert "deploy" in payload["kinds"], "the denied kind is registered — it is the edge that fails"


def test_a_policy_document_is_what_refuses_the_transition(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    """§12.2 end to end: the TOML file constrains the run, not Python."""
    doc = tmp_path / "policy.toml"
    doc.write_text(_DENY_DEPLOY, encoding="utf-8")

    code, payload, _ = call_json(
        ["plan", "look into the outage", "--scripted", "--go", "--policy", str(doc),
         "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert payload["rejections"] == ["edge_denied"]
    assert str(doc) in payload["policy"]


def test_a_document_that_denies_a_node_kind_stops_it_running(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    """Issue #66, end to end: a `resource = "node"` deny rule is enforced.

    The shipped script proposes `deploy` in round 1. Before the fix the whole
    node half of the document was discarded and `deploy ran` landed in the
    state; now the kind is refused with the operator's own reason and the
    planner replans around it.
    """
    doc = tmp_path / "nodepolicy.toml"
    doc.write_text(_DENY_DEPLOY_NODE, encoding="utf-8")

    code, payload, _ = call_json(
        ["plan", "fix the outage", "--scripted", "--go", "--policy", str(doc),
         "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert payload["rejections"] == ["node_denied"]
    assert payload["rounds"][0]["status"] == "rejected"
    assert payload["rounds"][0]["executed"] is False
    assert "deploy ran" not in payload["state"]["notes"]
    assert payload["state"]["notes"] == ["triage ran", "patch ran", "verify ran"]
    # The banner counts both halves of the document, so a reader can see that
    # the node rules were read rather than skimmed past.
    assert "1 edge rule(s), 2 node rule(s)" in payload["policy"]


def test_a_node_denial_is_traced_with_the_reason_the_document_gave(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # (registry.py + grapharc.toml) must not steer these runs.
    monkeypatch.chdir(tmp_path)
    from grapharc.observe.trace import TraceRecorder

    doc = tmp_path / "nodepolicy.toml"
    doc.write_text(_DENY_DEPLOY_NODE, encoding="utf-8")
    path = tmp_path / "t.jsonl"

    call([
        "plan",
        "fix the outage",
        "--scripted",
        "--policy",
        str(doc),
        "--trace",
        str(path),
    ], capsys)
    admissions = [e for e in TraceRecorder(path).read_events() if e.phase == "admission"]

    assert admissions, "the gate's decision has to be on the record"
    assert "policy/node_denied" in (admissions[0].error or "")


def test_a_permissive_document_admits_what_the_strict_one_refused(tmp_path, capsys):
    """The control for the test above: without the rule, round 1 runs."""
    doc = tmp_path / "permissive.toml"
    doc.write_text(_PERMISSIVE, encoding="utf-8")

    code, payload, _ = call_json(
        ["plan", "look into the outage", "--scripted", "--go", "--policy", str(doc),
         "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert payload["rejections"] == []
    assert payload["rounds"][0]["status"] == "admitted"
    assert payload["rounds"][0]["executed"] is True


def test_plan_stops_short_with_a_reason_and_a_failure_code(tmp_path, capsys):
    """One round is not enough to reach the goal; that is a stop, not a crash."""
    code, payload, _ = call_json(
        ["plan", "look into the outage", "--scripted", "--max-rounds", "1",
         "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stop"] == "max_rounds"


def test_plan_refuses_a_registry_that_does_not_resolve(tmp_path, capsys):
    code, _, err = call(
        [
            "plan",
            "x",
            "--scripted",
            "--registry",
            "no.such.module:thing",
            "--trace",
            str(tmp_path / "t.jsonl"),
        ],
        capsys,
    )

    assert code == 2
    assert "--registry" in err


def test_plan_refuses_a_registry_target_with_no_colon(tmp_path, capsys):
    code, _, err = call(
        [
            "plan",
            "x",
            "--scripted",
            "--registry",
            "grapharc.examples",
            "--trace",
            str(tmp_path / "t.jsonl"),
        ],
        capsys,
    )

    assert code == 2
    assert "module:attr" in err


def test_plan_refuses_a_policy_file_that_is_not_there(tmp_path, capsys):
    code, _, err = call(
        ["plan", "x", "--scripted", "--policy", str(tmp_path / "nope.toml"),
         "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 2
    assert "--policy" in err


def test_every_plan_round_is_auditable_from_the_trace_alone(tmp_path, capsys):
    trace_path = tmp_path / "t.jsonl"
    call([
        "plan",
        "look into the outage",
        "--scripted",
        "--trace",
        str(trace_path),
        "--run-id",
        "p1",
    ], capsys)

    events = TraceRecorder(trace_path).read_events("p1")
    phases = [e.phase for e in events]

    assert trace_path.exists()
    assert "admission" in phases, "the gate's decision has to be on the record"
    # The refusal is a first-class event, not an absence of one.
    assert any(e.phase == "admission" and (e.error or "") for e in events) or any(
        "rejected" in str(e.state_delta or {}) for e in events
    )


def test_the_shipped_command_is_what_finally_imports_the_policy_package(tmp_path, capsys):
    """ROADMAP §7.5 was "nothing imports this package". This is the importer."""
    import grapharc.cli.plan as plan_module

    doc = tmp_path / "policy.toml"
    doc.write_text(_DENY_DEPLOY, encoding="utf-8")

    policy, description = plan_module.resolve_policy(doc, tenant="default")

    assert policy.edge.rules, "the document's edge rules must reach the admission gate"
    assert "tenant 'default'" in description
    # This document declares no node rules, so nothing governs kinds but the
    # registry. Compiling one anyway would read "said nothing about nodes" as
    # "denied every node" and refuse the whole run.
    assert policy.node is None


def test_a_documents_node_rules_are_compiled_alongside_its_edge_rules(tmp_path, capsys):
    """The half of the compile that did not exist before issue #66."""
    import grapharc.cli.plan as plan_module

    doc = tmp_path / "policy.toml"
    doc.write_text(_DENY_DEPLOY_NODE, encoding="utf-8")

    from grapharc.harness.permissions import Decision

    policy, description = plan_module.resolve_policy(doc, tenant="default")

    assert policy.node is not None
    assert policy.node.decide("deploy") is Decision.DENY
    assert policy.node.decide("triage") is Decision.ALLOW
    assert "2 node rule(s)" in description


# --------------------------------------------------------------------------
# `grapharc run` — a topology the operator wrote, through the same gate.
# `run` used to mean "execute one of eight canned examples", which left no
# command for the common case: a graph you already know the shape of. The
# examples moved to `demo`. A hand-written file earns no exemption from
# admission — that is the property these pin.
# --------------------------------------------------------------------------

_LEGAL_GRAPH = {
    "nodes": [{"name": "triage"}, {"name": "fix", "kind": "patch"}, {"name": "verify"}],
    "edges": [
        {"source": "__start__", "target": "triage"},
        {"source": "triage", "target": "fix"},
        {"source": "fix", "target": "verify"},
        {"source": "verify", "target": "__end__"},
    ],
}
_DENIED_GRAPH = {
    "nodes": [{"name": "triage"}, {"name": "ship", "kind": "deploy"}],
    "edges": [
        {"source": "__start__", "target": "triage"},
        {"source": "triage", "target": "ship"},
        {"source": "ship", "target": "__end__"},
    ],
}


def _write_graph(tmp_path, document, name="graph.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_run_executes_a_topology_the_operator_wrote(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # must not steer these runs.
    monkeypatch.chdir(tmp_path)
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)

    code, payload, _ = call_json(
        ["run", str(graph), "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 0
    assert payload["admitted"] is True
    assert payload["checked_only"] is False
    assert payload["nodes"] == 3
    assert payload["state"]["notes"] == ["triage ran", "fix ran", "verify ran"]


def test_a_hand_written_graph_is_refused_like_any_other_proposal(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # must not steer these runs.
    monkeypatch.chdir(tmp_path)
    """The gate does not care who authored the topology."""
    graph = _write_graph(tmp_path, _DENIED_GRAPH)

    code, payload, _ = call_json(
        ["run", str(graph), "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["admitted"] is False
    assert [r["code"] for r in payload["rejections"]] == ["edge_denied"]
    assert "state" not in payload, "nothing may run when the gate refused"


def test_check_only_validates_without_executing(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # must not steer these runs.
    monkeypatch.chdir(tmp_path)
    """Admission as a linter: legal or not, and nothing runs either way."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)

    code, payload, _ = call_json(
        ["run", str(graph), "--check-only", "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 0
    assert payload["checked_only"] is True
    assert payload["fingerprint"]
    assert "state" not in payload


def test_check_only_still_fails_on_an_illegal_topology(tmp_path, capsys, monkeypatch):
    # A scratch cwd: a developer's own `grapharc init` in the checkout
    # must not steer these runs.
    monkeypatch.chdir(tmp_path)
    graph = _write_graph(tmp_path, _DENIED_GRAPH)

    code, payload, _ = call_json(
        ["run", str(graph), "--check-only", "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 1
    assert [r["code"] for r in payload["rejections"]] == ["edge_denied"]


def test_a_toml_topology_works_the_same_as_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    graph = tmp_path / "graph.toml"
    graph.write_text(
        '[[nodes]]\nname = "triage"\n\n'
        '[[nodes]]\nname = "fix"\nkind = "patch"\n\n'
        '[[edges]]\nsource = "__start__"\ntarget = "triage"\n\n'
        '[[edges]]\nsource = "triage"\ntarget = "fix"\n\n'
        '[[edges]]\nsource = "fix"\ntarget = "__end__"\n',
        encoding="utf-8",
    )

    code, payload, _ = call_json(
        ["run", str(graph), "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 0
    assert payload["admitted"] is True


def test_a_topology_file_cannot_carry_a_node_body(tmp_path, capsys):
    """The same boundary a model is held to. A file is authored by whoever can
    write to the directory, so it gets no more trust than a proposal."""
    graph = _write_graph(
        tmp_path,
        {"nodes": [{"name": "triage", "body": "os.system('id')"}], "edges": []},
    )

    code, _, err = call(["run", str(graph), "--trace", str(tmp_path / "t.jsonl")], capsys)

    assert code == 2
    assert "not a valid topology" in err


def test_run_redirects_the_old_stage_form_to_demo(tmp_path, capsys):
    """`grapharc run stage0` worked before the split; it must not read as a
    missing file."""
    code, _, err = call(["run", "stage0"], capsys)

    assert code == 2
    assert "grapharc demo stage0" in err


def test_run_says_which_file_is_missing(tmp_path, capsys):
    code, _, err = call(["run", str(tmp_path / "nope.json")], capsys)

    assert code == 2
    assert "no such graph file" in err


def test_run_reports_a_graph_file_that_is_not_utf8(tmp_path, capsys):
    """`UnicodeDecodeError` is a `ValueError`, so neither decoder caught it.

    A topology saved as UTF-16 or truncated in transit used to exit 1 with a
    traceback and an empty document.
    """
    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\xff\xfe\x00binary")

    code, payload, err = call_json(["run", str(binary)], capsys)

    assert code == 2
    assert payload["ok"] is False
    assert "utf-8" in payload["error"]
    assert err == ""


def test_a_policy_document_gates_a_hand_written_graph_too(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """§12.2 on the deterministic path: the TOML file decides here as well."""
    graph = _write_graph(tmp_path, _DENIED_GRAPH)
    permissive = tmp_path / "allow.toml"
    permissive.write_text(_PERMISSIVE, encoding="utf-8")

    code, payload, _ = call_json(
        ["run", str(graph), "--policy", str(permissive), "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 0
    assert payload["admitted"] is True, "the deny rule was the only thing refusing it"


# --------------------------------------------------------------------------
# Defects an adversarial end-to-end pass found after `run`/`demo` landed. Each
# one violated a promise the CLI makes in its own `--help`: that the reading
# commands and the runtime share one record, and that in JSON mode the failure
# *is* the document rather than a traceback with empty stdout.
# --------------------------------------------------------------------------


def test_run_records_the_execution_it_says_it_performed(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """It reported "ADMITTED and executed" and wrote only the admission event —
    `Materializer` takes a `trace=` and the call omitted it."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    trace = tmp_path / "t.jsonl"

    call(["run", str(graph), "--trace", str(trace)], capsys)

    phases = [e.phase for e in TraceRecorder(trace).read_events()]
    assert phases.count("end") == 3, phases
    assert "admission" in phases


def test_run_honours_the_run_id_it_was_given(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """`--run-id` was accepted and discarded, so `metrics <trace> <id>` found
    nothing under the id the operator chose."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    trace = tmp_path / "t.jsonl"

    call(["run", str(graph), "--trace", str(trace), "--run-id", "chosen"], capsys)
    code, payload, _ = call_json(["metrics", str(trace), "chosen"], capsys)

    assert code == 0
    assert payload["run_id"] == "chosen"
    assert payload["nodes_executed"] == 3


# -- a run id names one run ---------------------------------------------------


def test_a_reused_run_id_is_refused_before_the_second_run_writes_anything(tmp_path, capsys):
    """Two runs under one id merge, and every reader then reports the blend as
    one run: doubled tokens from `metrics`, a welded path from `viz`."""
    trace = tmp_path / "t.jsonl"
    first, _, _ = call(
        ["plan", "goal one", "--scripted", "--trace", str(trace), "--run-id", "r1"], capsys
    )
    assert first == 0
    before = TraceRecorder(trace).read_events("r1")

    code, _, err = call([
        "plan",
        "goal two",
        "--scripted",
        "--trace",
        str(trace),
        "--run-id",
        "r1",
    ], capsys)

    assert code == 2
    assert "r1" in err and str(trace) in err
    assert "--run-id" in err, "the message has to say how to proceed"
    after = TraceRecorder(trace).read_events("r1")
    assert [e.model_dump() for e in after] == [e.model_dump() for e in before]


def test_a_reused_run_id_fails_as_one_json_document(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    call(["plan", "goal one", "--scripted", "--trace", str(trace), "--run-id", "r1"], capsys)

    code, payload, err = call_json(
        ["plan", "goal two", "--scripted", "--trace", str(trace), "--run-id", "r1"], capsys
    )

    assert code == 2
    assert payload["ok"] is False
    assert payload["command"] == "plan"
    assert payload["run_id"] == "r1"
    assert payload["trace"] == str(trace)
    assert err == ""


def test_run_refuses_a_reused_run_id_before_the_admission_event(tmp_path, capsys):
    """`run` writes its verdict under the id, so `--check-only` collides too."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    trace = tmp_path / "t.jsonl"
    call(["run", str(graph), "--trace", str(trace), "--run-id", "chosen"], capsys)
    before = len(TraceRecorder(trace).read_events("chosen"))

    code, _, err = call(
        ["run", str(graph), "--check-only", "--trace", str(trace), "--run-id", "chosen"], capsys
    )

    assert code == 2
    assert "chosen" in err
    assert len(TraceRecorder(trace).read_events("chosen")) == before


def test_agent_refuses_a_reused_run_id(tmp_path, capsys):
    """Checked before the model is built, so the refusal costs no backend call."""
    trace = tmp_path / "t.jsonl"
    TraceRecorder(trace).event(
        run_id="a1", graph="cli-agent", node="agent", phase="start", step=1
    )

    code, _, err = call(
        ["agent", "do the thing", "--workspace", str(tmp_path),
         "--trace", str(trace), "--run-id", "a1"],
        capsys,
    )

    assert code == 2
    assert "a1" in err and "1 event" in err


def test_different_run_ids_in_one_trace_stay_supported(tmp_path, capsys):
    """`grapharc diff` reads two runs out of one file; that is the pattern the
    guard must not touch. The file being appendable is correct — the id being
    reused is the defect."""
    trace = tmp_path / "t.jsonl"

    assert call([
        "plan",
        "goal one",
        "--scripted",
        "--trace",
        str(trace),
        "--run-id",
        "r1",
    ], capsys)[0] == 0
    assert call([
        "plan",
        "goal two",
        "--scripted",
        "--trace",
        str(trace),
        "--run-id",
        "r2",
    ], capsys)[0] == 0

    assert TraceRecorder(trace).run_ids() == ["r1", "r2"]


def test_a_generated_run_id_is_never_guarded(tmp_path, capsys):
    """Fresh by construction, so it must not pay for a scan of the file either."""
    trace = tmp_path / "t.jsonl"

    assert call(["plan", "goal one", "--scripted", "--trace", str(trace)], capsys)[0] == 0
    assert call(["plan", "goal two", "--scripted", "--trace", str(trace)], capsys)[0] == 0

    assert len(TraceRecorder(trace).run_ids()) == 2


def test_the_guard_counts_a_line_it_can_read_and_skips_the_rest(tmp_path):
    """A torn line is the readers' business to report. The guard looks for one
    id, so it skips what it cannot parse rather than raising over it — and it
    reports nothing at all for a file that is not there."""
    from grapharc.cli.runid import count_events

    trace = tmp_path / "t.jsonl"
    assert count_events(trace, "r1") == 0
    assert count_events(tmp_path, "r1") == 0, "a directory is not a trace"

    TraceRecorder(trace).event(run_id="r1", graph="g", node="n", phase="start", step=1)
    with trace.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n\n")
    TraceRecorder(trace).event(run_id="r2", graph="g", node="n", phase="start", step=1)

    assert count_events(trace, "r1") == 1
    assert count_events(trace, "r2") == 1
    assert count_events(trace, "r3") == 0


def test_check_only_refuses_a_topology_that_passes_the_gate_but_cannot_be_built(
    tmp_path, capsys
, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """A linter that says ADMITTED and then crashes on the same file is worse
    than no linter. All three of these pass admission and fail materialisation."""
    unbuildable = {
        "no edge out of START": {"nodes": [{"name": "triage"}], "edges": []},
        "no nodes at all": {"nodes": [], "edges": []},
        "a node START cannot reach": {
            "nodes": [{"name": "triage"}, {"name": "orphan", "kind": "patch"}],
            "edges": [{"source": "__start__", "target": "triage"}],
        },
    }
    for label, document in unbuildable.items():
        graph = _write_graph(tmp_path, document, name=f"{abs(hash(label))}.json")
        code, payload, _ = call_json(
            ["run", str(graph), "--check-only", "--trace", str(tmp_path / "t.jsonl")], capsys
        )
        assert code == 1, label
        assert payload["buildable"] is False, label
        assert payload["error"], label


def test_a_topology_that_cannot_be_built_reports_rather_than_crashing(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """`MaterializationError` escaped `run_graph` as a raw traceback, leaving
    stdout empty in --json mode."""
    graph = _write_graph(tmp_path, {"nodes": [{"name": "triage"}], "edges": []})

    code, payload, _ = call_json(
        ["run", str(graph), "--trace", str(tmp_path / "t.jsonl")], capsys
    )

    assert code == 1
    assert payload["ok"] is False
    assert "has no edge out of START" in payload["error"]


def test_run_can_be_given_a_token_ceiling(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """A comment claimed the budget dimension was bounded; `Budget()` is
    unlimited on every dimension, so a 400,000-token worst case was admitted."""
    chain = {
        "nodes": [{"name": f"p{i}", "kind": "patch"} for i in range(40)],
        "edges": (
            [{"source": "__start__", "target": "p0"}]
            + [{"source": f"p{i}", "target": f"p{i + 1}"} for i in range(39)]
            + [{"source": "p39", "target": "__end__"}]
        ),
    }
    graph = _write_graph(tmp_path, chain)
    args = ["run", str(graph), "--check-only", "--trace", str(tmp_path / "t.jsonl")]

    _, unbounded, _ = call_json(args, capsys)
    code, bounded, _ = call_json([*args, "--max-tokens", "1000"], capsys)

    assert unbounded["admitted"] is True, "no ceiling given, so nothing bounds it"
    assert code == 1
    assert "over_token_budget" in [r["code"] for r in bounded["rejections"]]




def test_run_budget_flags_reach_json_payload(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    """All four Budget dimensions are settable from `grapharc run` (#5)."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    code, payload, _ = call_json(
        [
            "run",
            str(graph),
            "--check-only",
            "--trace",
            str(tmp_path / "t.jsonl"),
            "--max-tokens",
            "5000",
            "--max-iterations",
            "12",
            "--max-seconds",
            "3.5",
            "--max-concurrency",
            "2",
        ],
        capsys,
    )
    assert code == 0
    assert payload["max_tokens"] == 5000
    assert payload["max_iterations"] == 12
    assert payload["max_seconds"] == 3.5
    assert payload["max_concurrency"] == 2


def test_run_unset_budget_flags_stay_unlimited_in_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a checkout-level grapharc.toml must not steer this
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    code, payload, _ = call_json(
        ["run", str(graph), "--check-only", "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )
    assert code == 0
    assert payload["max_tokens"] is None
    assert payload["max_iterations"] is None
    assert payload["max_seconds"] is None
    assert payload["max_concurrency"] is None

def test_viz_answers_an_unknown_run_id_as_a_document(tmp_path, capsys):
    """Every other reading command did; `viz` let `ReplayError` out raw."""
    graph = _write_graph(tmp_path, _LEGAL_GRAPH)
    trace = tmp_path / "t.jsonl"
    call(["run", str(graph), "--trace", str(trace)], capsys)

    code, payload, _ = call_json(["viz", str(trace), "nosuchrun"], capsys)

    assert code == 1
    assert payload["ok"] is False
    assert "nosuchrun" in payload["error"]


def test_a_mistyped_backend_exits_two_with_a_document(capsys):
    """The top-level help lists "a model spec names no backend" under exit 2;
    this crashed with exit 1 and no output."""
    code, payload, _ = call_json(["models", "openrouterr/whatever"], capsys)

    assert code == 2
    assert payload["ok"] is False
    assert "unknown backend" in payload["error"]


def test_an_unusable_memory_path_reports_rather_than_crashing(tmp_path, capsys):
    """`--memory` takes a path from a human, so it can name a directory."""
    code, payload, _ = call_json(
        ["demo", "stage6", "--memory", str(tmp_path), "--trace", str(tmp_path / "t.jsonl")],
        capsys,
    )

    assert code == 2
    assert payload["ok"] is False
    assert "--memory" in payload["error"]


def test_config_is_only_accepted_by_commands_that_read_it(capsys):
    """It was on the shared parser, so all eleven accepted it and nine ignored
    it — including erroring on a missing file for two of them and not the rest."""
    for command in (["plan", "g", "--scripted"], ["demo", "stage0"], ["run", "x.json"]):
        code, _, err = call([*command, "--config", "/no/such/file.toml"], capsys)
        assert code == 2 and "--config" in err, command
    for command in (["models"], ["trace", "f"], ["viz", "f", "r"]):
        with pytest.raises(SystemExit):
            main([*command, "--config", "/no/such/file.toml"])


# ---------------------------------------------------------------------------
# The default trace lands under the live root, and the watch line finds it.
# ---------------------------------------------------------------------------


def test_the_default_trace_lands_under_grapharc_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    trace = Path(payload["trace"]).resolve()
    assert (tmp_path / ".grapharc" / "runs").resolve() in trace.parents
    assert trace.name == "trace.jsonl"
    assert trace.is_file()


def test_without_a_server_the_watch_line_is_an_instruction(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["plan", "look into it", "--scripted"])
    printed = capsys.readouterr().out
    assert code == 0
    watch = next(line for line in printed.splitlines() if line.startswith("watch"))
    assert "grapharc serve --live-root .grapharc/runs" in watch
    assert "http://127.0.0.1:8000/live/view?trace=" in watch


def _write_live_marker(tmp_path, *, host="127.0.0.1", port=0):
    import json as _json

    marker = tmp_path / ".grapharc" / "live-server.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        _json.dumps(
            {
                "url": f"http://{host}:{port}",
                "host": host,
                "port": port,
                "live_root": str((tmp_path / ".grapharc" / "runs").resolve()),
                "pid": 999999,
            }
        ),
        encoding="utf-8",
    )
    return marker


def test_a_reachable_server_yields_the_exact_watch_url(tmp_path, monkeypatch, capsys):
    import socket

    monkeypatch.chdir(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        _write_live_marker(tmp_path, port=port)
        code = main(["plan", "look into it", "--scripted", "--json"])
        payload = json.loads(capsys.readouterr().out)
    finally:
        listener.close()
    assert code == 0
    url = payload["watch_url"]
    assert url is not None
    assert url.startswith(f"http://127.0.0.1:{port}/live/view?trace=")
    # The trace path in the URL is relative to the marker's live root.
    from urllib.parse import unquote

    rel = unquote(url.split("trace=", 1)[1])
    assert (tmp_path / ".grapharc" / "runs" / rel).is_file()


def test_a_stale_marker_with_no_listener_falls_back_to_the_hint(
    tmp_path, monkeypatch, capsys
):
    import socket

    monkeypatch.chdir(tmp_path)
    # A port that was just released: nothing is listening on it.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    _write_live_marker(tmp_path, port=dead_port)
    code = main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["watch_url"] is None


def test_a_malformed_marker_degrades_to_the_hint_instead_of_crashing(
    tmp_path, monkeypatch
):
    """The marker shapes JSON allows but the reader does not: null port, a
    non-object document. Both used to escape `watch_url` as a TypeError
    traceback from a command that only wanted to print a courtesy URL."""
    from grapharc.cli.plan import watch_url

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / ".grapharc" / "live-server.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    trace = tmp_path / ".grapharc" / "runs" / "r1" / "trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("", encoding="utf-8")

    marker.write_text(
        json.dumps(
            {
                "url": "http://127.0.0.1:8000",
                "host": "127.0.0.1",
                "port": None,
                "live_root": str((tmp_path / ".grapharc" / "runs").resolve()),
            }
        ),
        encoding="utf-8",
    )
    assert watch_url(trace) is None

    marker.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert watch_url(trace) is None


def test_an_unresolved_marker_root_still_matches_through_a_symlink(
    tmp_path, monkeypatch
):
    """`serve` writes its root resolved; a hand-edited marker may not be. The
    comparison resolves both sides now, so a symlinked spelling of the same
    directory is the same directory rather than a lexical mismatch."""
    import socket

    from grapharc.cli.plan import watch_url

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real-runs"
    real.mkdir()
    link = tmp_path / "link-runs"
    link.symlink_to(real, target_is_directory=True)
    trace = real / "r1" / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("", encoding="utf-8")

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        marker = tmp_path / ".grapharc" / "live-server.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{port}",
                    "host": "127.0.0.1",
                    "port": port,
                    "live_root": str(link),  # deliberately unresolved
                }
            ),
            encoding="utf-8",
        )
        url = watch_url(trace)
    finally:
        listener.close()
    assert url is not None and "trace=r1%2Ftrace.jsonl" in url


def test_the_hint_quotes_the_marker_port_when_the_server_is_down(
    tmp_path, monkeypatch, capsys
):
    """A stale marker still names the port the operator actually serves on;
    an instruction quoting a different port than their own `grapharc serve`
    command is a wrong instruction."""
    import socket

    monkeypatch.chdir(tmp_path)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    _write_live_marker(tmp_path, port=dead_port)
    code = main(["plan", "look into it", "--scripted"])
    printed = capsys.readouterr().out
    assert code == 0
    watch = next(line for line in printed.splitlines() if line.startswith("watch"))
    assert "grapharc serve --live-root .grapharc/runs" in watch
    assert f"127.0.0.1:{dead_port}/live/view?trace=" in watch


def test_a_trace_outside_the_live_root_gets_no_exact_url(tmp_path, monkeypatch, capsys):
    import socket

    monkeypatch.chdir(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        _write_live_marker(tmp_path, port=listener.getsockname()[1])
        elsewhere = tmp_path / "elsewhere" / "trace.jsonl"
        code = main(["plan", "look into it", "--scripted", "--trace", str(elsewhere), "--json"])
        payload = json.loads(capsys.readouterr().out)
    finally:
        listener.close()
    assert code == 0
    assert payload["watch_url"] is None


def test_serve_writes_a_discovery_marker_and_removes_it(
    monkeypatch, capsys, tmp_path
):
    """`plan`/`go` find the live server through this marker; it must exist
    while the server runs, carry no token, and be gone afterwards."""
    record: dict = {}
    stub = ModuleType("grapharc.server")

    def create_app(**kwargs):
        record["create_app"] = kwargs
        return "THE-APP"

    def running_serve(app, **kwargs):
        marker = Path(".grapharc") / "live-server.json"
        record["marker_during"] = json.loads(marker.read_text(encoding="utf-8"))
        record["marker_bytes"] = marker.read_bytes()

    stub.create_app = create_app
    stub.serve = running_serve
    monkeypatch.setitem(sys.modules, "grapharc.server", stub)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()

    code, _, _ = call(
        ["serve", "--live-root", str(root), "--live-token", "s3cret"], capsys
    )
    assert code == 0
    marker = record["marker_during"]
    assert marker["url"] == "http://127.0.0.1:8000"
    assert marker["live_root"] == str(root.resolve())
    assert marker["pid"] == __import__("os").getpid()
    assert b"s3cret" not in record["marker_bytes"]
    # Unlinked on the way out.
    assert not (tmp_path / ".grapharc" / "live-server.json").exists()


def test_serve_without_a_live_root_writes_no_marker(monkeypatch, capsys, tmp_path):
    record: dict = {}
    monkeypatch.setitem(sys.modules, "grapharc.server", _server_stub(record))
    monkeypatch.chdir(tmp_path)
    code, _, _ = call(["serve"], capsys)
    assert code == 0
    assert not (tmp_path / ".grapharc").exists()


# -- go: plan with doing-defaults ---------------------------------------------


def test_go_requires_a_model(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["go", "get it done", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False
    assert "models --check" in payload["error"]


def test_go_defaults_to_the_stdlib_registry(tmp_path, monkeypatch, capsys):
    """With a model spec that cannot construct, the failure message proves the
    resolution order: the model is reached before any registry import."""
    monkeypatch.chdir(tmp_path)
    recorded = {}

    def fake_plan(goal, **kwargs):
        recorded.update(kwargs, goal=goal)
        return 0

    import grapharc.cli.plan as plan_module

    monkeypatch.setattr(plan_module, "plan", fake_plan)
    code = main(["go", "get it done", "--model", "ollama/fake"])
    assert code == 0
    assert recorded["command"] == "go"
    assert recorded["go_after"] is True, "go executes; plan-only is plan's job"
    assert recorded["model_spec"] == "ollama/fake"


def test_go_and_plan_share_every_planning_flag():
    parser = build_parser()
    plan_actions = {
        a.option_strings[0]
        for a in parser._subparsers._group_actions[0].choices["plan"]._actions
        if a.option_strings
    }
    go_actions = {
        a.option_strings[0]
        for a in parser._subparsers._group_actions[0].choices["go"]._actions
        if a.option_strings
    }
    # `--scripted` and `--go` are plan-only by design: go means do (no
    # scripted doing), and go needs no flag to do what its name says.
    assert plan_actions - go_actions == {"--scripted", "--go"}
    assert go_actions - plan_actions == set()


# -- init: the scaffold -------------------------------------------------------


def test_init_scaffolds_a_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["init", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (tmp_path / "registry.py").is_file()
    assert (tmp_path / "grapharc.toml").is_file()
    assert (tmp_path / ".grapharc" / "runs").is_dir()
    assert payload["registry"] == "registry.py"
    # The template compiles and the config parses.
    compile((tmp_path / "registry.py").read_text(encoding="utf-8"), "registry.py", "exec")
    from grapharc.cli.config import load

    assert load(tmp_path / "grapharc.toml").values["registry"] == "registry.py:build_registry"


def test_init_refuses_to_overwrite_and_names_the_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    capsys.readouterr()
    code = main(["init"])
    err = capsys.readouterr().err
    assert code == 2
    assert "registry.py" in err and "grapharc.toml" in err
    # The runs dir alone never blocks a scaffold.
    assert (tmp_path / ".grapharc" / "runs").is_dir()


def test_an_init_scaffold_plans_end_to_end(tmp_path, monkeypatch, capsys):
    """The money test: template + path-form loader + build_loop handoff +
    scripted replies compose into the refuse-then-admit first run."""
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    capsys.readouterr()
    code = main(["plan", "try it", "--scripted", "--go"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "round 1: rejected" in printed and "edge_denied" in printed
    assert "round 2: admitted" in printed
    assert "goal_met" in printed


def test_the_scaffold_state_merges_parallel_writers(tmp_path, monkeypatch):
    """Two kinds writing `notes` in the same superstep compose via the reducer.

    The shape any real planner eventually proposes: `gather` and `analyse`
    both fanned out of START, joining at `report`. With a plain `list[str]`
    this run died on LangGraph's InvalidUpdateError before `report` ever ran;
    the scaffold's `notes` is a reducer now, and this test is what keeps it
    one.
    """
    monkeypatch.chdir(tmp_path)
    from grapharc.cli.init_cmd import REGISTRY_TEMPLATE
    from grapharc.testing import ScriptedChatModel

    module = ModuleType("scaffold_registry")
    # The path-form loader registers the module before executing it, and
    # pydantic needs that to resolve the template's deferred annotations.
    monkeypatch.setitem(sys.modules, "scaffold_registry", module)
    exec(compile(REGISTRY_TEMPLATE, "registry.py", "exec"), module.__dict__)
    plan = json.dumps(
        {
            "nodes": [{"name": "gather"}, {"name": "analyse"}, {"name": "report"}],
            "edges": [
                {"source": "__start__", "target": "gather"},
                {"source": "__start__", "target": "analyse"},
                {"source": "gather", "target": "report"},
                {"source": "analyse", "target": "report"},
                {"source": "report", "target": "__end__"},
            ],
        }
    )
    loop = module.build_loop(ScriptedChatModel(responses=[plan]))
    result = loop.run("report on this directory, twice over", module.State())
    assert result.stop.value == "goal_met"
    assert any(note.startswith("gather:") for note in result.state.notes)
    assert any(note.startswith("analyse:") for note in result.state.notes)


def test_the_path_form_registry_shares_one_module_object(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reg.py").write_text(
        "COUNTER = []\n"
        "def build_registry(model=None):\n"
        "    from grapharc.planner import NodeRegistry\n"
        "    COUNTER.append(1)\n"
        "    return NodeRegistry([])\n",
        encoding="utf-8",
    )
    from grapharc.cli.plan import _registry_module

    first, attr = _registry_module("reg.py:build_registry")
    second, _ = _registry_module("reg.py:build_registry")
    assert first is second
    assert attr == "build_registry"


def test_a_missing_registry_file_exits_2_naming_the_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["plan", "x", "--scripted", "--registry", "./nope.py:build_registry", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "no such file" in payload["error"]
    assert "nope.py" in payload["error"]


# -- start and the bare invocation --------------------------------------------


def test_bare_invocation_orients_and_exits_zero(capsys):
    code = main([])
    out, err = capsys.readouterr().out, capsys.readouterr().err
    assert code == 0
    assert "grapharc start" in out
    assert err == ""


def test_start_prints_the_guided_path(capsys):
    code = main(["start"])
    out = capsys.readouterr().out
    assert code == 0
    for expected in ("init", "serve --live-root", "plan", "go", "approve", "watch"):
        assert expected in out, expected


def test_start_json_is_one_document(capsys):
    code = main(["start", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "start"
    assert any("init" in step["command"] for step in payload["path"])


def test_help_is_a_command_not_an_error(capsys):
    """`grapharc help` is what people type; it must be `-h`, not an argparse
    scolding."""
    code = main(["help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "usage: grapharc" in out
    assert "plan" in out and "go" in out


# -- plan plans; go goes ------------------------------------------------------


def test_plan_plans_only_and_saves_the_plan(tmp_path, monkeypatch, capsys):
    """`plan` executes nothing: the gate runs, the plan lands on disk, and
    the nodes wait for `go`."""
    monkeypatch.chdir(tmp_path)
    code = main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["stop"] == "planned"
    assert all(r["executed"] is False for r in payload["rounds"])
    plan_file = Path(payload["plan_file"])
    record = json.loads(plan_file.read_text(encoding="utf-8"))
    assert record["goal"] == "look into it"
    assert record["proposal"]["nodes"], "the admitted proposal is stored whole"


def test_go_executes_the_newest_saved_plan_and_marks_it(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["plan", "look into it", "--scripted", "--json"])
    planned = json.loads(capsys.readouterr().out)
    code = main(["go", "--json"])
    executed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert executed["executed"] is True
    assert executed["stop"] == "goal_met"
    record = json.loads(Path(planned["plan_file"]).read_text(encoding="utf-8"))
    assert record["executed_run_id"] == executed["run_id"]
    # And a second bare `go` finds nothing left to do.
    assert main(["go", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_go_takes_a_specific_run_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["plan", "look into it", "--scripted", "--json"])
    planned = json.loads(capsys.readouterr().out)
    run_dir = str(Path(planned["plan_file"]).parent)
    code = main(["go", run_dir, "--json"])
    executed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert executed["plan"].startswith(run_dir)


def test_plan_go_is_one_shot(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["plan", "look into it", "--scripted", "--go", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["stop"] == "goal_met"
    assert any(r["executed"] for r in payload["rounds"])


def test_a_registry_py_in_cwd_wins_when_nothing_is_configured(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    (tmp_path / "grapharc.toml").unlink()  # no config: detection must carry it
    capsys.readouterr()
    code = main(["plan", "look into it", "--scripted", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["registry"] == "registry.py:build_registry"


def test_default_flag_forces_the_builtin_kinds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    capsys.readouterr()
    code = main(["plan", "look into it", "--scripted", "--default", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["registry"] == "grapharc.stdlib:build_registry"
