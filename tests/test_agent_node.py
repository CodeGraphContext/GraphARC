"""§4.1 gates — the agent node.

The loop's guarantees are behavioural, so every test here drives a real
`AgentNode` over a scripted model and asserts what came back out: which tools
the model was ever shown, what the harness did with a request, what the model
saw next, and which `StopReason` the run ended on.

`grapharc.testing.ScriptedChatModel` scripts text only and `BaseChatModel`
leaves `bind_tools` unimplemented, so `ToolScriptedChatModel` below extends it
with the two pieces a tool loop needs. Everything else — cursor, call log,
usage metadata — is the shipped double.
"""

import sys
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from grapharc.harness import (
    AgentConfigError,
    AgentNode,
    AgentResult,
    Harness,
    HookAction,
    HookDecision,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolCallStatus,
    ToolRegistry,
    ToolSpec,
    tool_schema,
    tool_schemas,
)
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import END, START, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel

requires_sandbox = pytest.mark.skipif(
    not hasattr(sys, "addaudithook") or sys.platform == "win32",
    reason="requires POSIX fork + audit hooks",
)


try:  # the OpenRouter backend's converter lives in an optional extra
    from langchain_openai.chat_models.base import _convert_dict_to_message
except ImportError:  # pragma: no cover — exercised only in a bare install
    _convert_dict_to_message = None

requires_openai_wire = pytest.mark.skipif(
    _convert_dict_to_message is None,
    reason="needs the openrouter extra (langchain-openai) for the real wire converter",
)


class ToolScriptedChatModel(ScriptedChatModel):
    """ScriptedChatModel plus tool-call turns and a bind_tools that records.

    `tool_call_script[i]` is attached to scripted response `i` as the turn's
    `tool_calls`; a missing or empty entry means a plain text turn, which is
    how the agent learns the task is done. `invalid_tool_call_script[i]` is the
    same for `invalid_tool_calls` — the field langchain routes a call to when
    its JSON arguments will not parse, leaving `tool_calls` empty.
    """

    tool_call_script: list[list[dict[str, Any]]] = Field(default_factory=list)
    invalid_tool_call_script: list[list[dict[str, Any]]] = Field(default_factory=list)
    _bound_tools: list[list[Any]] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self._bound_tools.append(list(tools))
        return self  # the loop must keep talking to this object's script

    @property
    def bound_tools(self) -> list[list[Any]]:
        return self._bound_tools

    @property
    def bound_tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._bound_tools[-1]] if self._bound_tools else []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = min(self._cursor, max(len(self.responses) - 1, 0))
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        calls = self.tool_call_script[index] if index < len(self.tool_call_script) else None
        invalid = (
            self.invalid_tool_call_script[index]
            if index < len(self.invalid_tool_call_script)
            else None
        )
        if not calls and not invalid:
            return result
        base = result.generations[0].message
        message = AIMessage(
            content=base.content,
            usage_metadata=base.usage_metadata,
            tool_calls=[{"type": "tool_call", **call} for call in calls or []],
            invalid_tool_calls=[{"type": "invalid_tool_call", **c} for c in invalid or []],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _policy(rules):
    return PermissionPolicy(rules=[PermissionRule(**r) for r in rules])


def _call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id}


def _broken_call(name, arguments, id="call_broken"):
    """A tool call whose arguments never parsed, as langchain reports it."""
    return {
        "name": name,
        "args": arguments,
        "id": id,
        "error": f"Function {name} arguments:\n\n{arguments}\n\nare not valid JSON.",
    }


def _harness(specs, rules, **kwargs):
    """A harness on the in-process executor — permissions under test, not the sandbox."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    kwargs.setdefault("executor", LocalExecutor())
    return Harness(registry, _policy(rules), **kwargs)


def _sandboxed_harness(specs, rules, workspace):
    """A harness on the default SandboxedExecutor, confined to `workspace`."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return Harness(registry, _policy(rules), workspace=str(workspace))


def _ctx(budget: Budget | None = None) -> RunContext:
    return RunContext(run_id="run-test", graph="test", meter=BudgetMeter(budget or Budget()))


def _tool_messages(messages) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


# -- tool exposure ------------------------------------------------------------


def test_only_visible_tools_are_bound():
    """Policy before schema: a denied tool is never described to the model, so
    the model cannot even name it."""
    harness = _harness(
        [
            ToolSpec(name="read", description="read a file", fn=lambda path: "x"),
            ToolSpec(name="rm", description="delete a file", fn=lambda path: "gone"),
        ],
        [{"action": "allow", "pattern": "read"}, {"action": "deny", "pattern": "rm"}],
    )
    model = ToolScriptedChatModel(responses=["done"])
    AgentNode(model, harness).run("go", _ctx())

    assert model.bound_tool_names == ["read"]


def test_no_visible_tools_means_no_binding_and_a_plain_answer():
    """Everything denied is a legal configuration, not a crash: several
    providers reject an empty tool list, so binding is skipped."""
    harness = _harness(
        [ToolSpec(name="rm", description="", fn=lambda: "gone")],
        [{"action": "deny", "pattern": "*"}],
    )
    model = ToolScriptedChatModel(responses=["nothing I can do"])
    result = AgentNode(model, harness).run("go", _ctx())

    assert model.bound_tools == []
    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "nothing I can do"


def test_tool_schema_is_derived_from_the_callable_signature():
    def grep(pattern: str, path: str, ignore_case: bool = False) -> list[str]:
        return []

    schema = tool_schema(ToolSpec(name="grep", description="search", fn=grep))
    function = schema["function"]

    assert schema["type"] == "function"
    assert function["name"] == "grep"
    assert function["description"] == "search"
    assert function["parameters"]["properties"] == {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
        "ignore_case": {"type": "boolean"},
    }
    assert function["parameters"]["required"] == ["pattern", "path"]


def test_tool_schema_declares_a_kwargs_tool_open():
    def anything(**kwargs):
        return kwargs

    parameters = tool_schema(ToolSpec(name="anything", description="", fn=anything))["function"][
        "parameters"
    ]
    assert parameters["additionalProperties"] is True
    assert "required" not in parameters


def test_tool_schemas_follow_the_policy():
    harness = _harness(
        [
            ToolSpec(name="read", description="", fn=lambda path: ""),
            ToolSpec(name="send", description="", fn=lambda to: ""),
            ToolSpec(name="rm", description="", fn=lambda path: ""),
        ],
        [
            {"action": "allow", "pattern": "read"},
            {"action": "ask", "pattern": "send"},
            {"action": "deny", "pattern": "rm"},
        ],
    )
    # ask-tools stay visible: the model may request one, a human decides.
    assert [s["function"]["name"] for s in tool_schemas(harness)] == ["read", "send"]


# -- the happy path -----------------------------------------------------------


def test_successful_tool_call_result_reaches_the_model(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("budgets bound work", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    harness = _harness(
        [ToolSpec(name="read_file", description="read a file", fn=read_file)],
        [{"action": "allow", "pattern": "read_file"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "the notes say budgets bound work"],
        tool_call_script=[[_call("read_file", {"path": str(target)})]],
    )
    result = AgentNode(model, harness).run("what do the notes say?", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "the notes say budgets bound work"
    assert result.iterations == 2
    assert [c.status for c in result.tool_calls] == [ToolCallStatus.OK]
    assert result.tool_calls[0].detail == "budgets bound work"
    # The model's second turn saw the tool output, not a summary of it.
    second_turn = _tool_messages(model.calls[1])
    assert [m.content for m in second_turn] == ["budgets bound work"]
    assert second_turn[0].status == "success"


def test_non_string_tool_results_are_rendered_for_the_model():
    harness = _harness(
        [ToolSpec(name="stat", description="", fn=lambda: {"lines": 3, "ok": True})],
        [{"action": "allow", "pattern": "stat"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "three lines"], tool_call_script=[[_call("stat", {})]]
    )
    result = AgentNode(model, harness).run("count", _ctx())

    assert result.tool_calls[0].detail == '{"lines": 3, "ok": true}'
    assert result.termination_reason is StopReason.TARGET_MET


def test_long_tool_output_is_truncated_with_a_marker():
    harness = _harness(
        [ToolSpec(name="dump", description="", fn=lambda: "x" * 500)],
        [{"action": "allow", "pattern": "dump"}],
    )
    model = ToolScriptedChatModel(responses=["", "ok"], tool_call_script=[[_call("dump", {})]])
    agent = AgentNode(model, harness, max_tool_result_chars=100)
    result = agent.run("dump it", _ctx())

    detail = result.tool_calls[0].detail
    assert detail.startswith("x" * 100)
    assert "truncated 400 chars" in detail


def test_multiple_tool_calls_in_one_turn_each_get_a_result():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "done"],
        tool_call_script=[
            [_call("echo", {"text": "a"}, id="c1"), _call("echo", {"text": "b"}, id="c2")]
        ],
    )
    result = AgentNode(model, harness).run("echo both", _ctx())

    assert [c.detail for c in result.tool_calls] == ["a", "b"]
    assert [m.tool_call_id for m in _tool_messages(model.calls[1])] == ["c1", "c2"]


# -- refusals are data, not exceptions ----------------------------------------


def test_denied_tool_is_fed_back_and_the_agent_recovers():
    """A denial must not end the run: the model is told, and picks another
    approach. Ending the run here is what pressures callers to widen policy."""
    executed = []

    def send(to: str) -> str:
        executed.append(to)
        return "sent"

    harness = _harness(
        [
            ToolSpec(name="send", description="send mail", fn=send),
            ToolSpec(name="note", description="write a note", fn=lambda text: f"noted: {text}"),
        ],
        [{"action": "ask", "pattern": "send"}, {"action": "allow", "pattern": "note"}],
    )  # no approval callback -> ask fails closed
    model = ToolScriptedChatModel(
        responses=["", "", "I could not send mail, so I left a note"],
        tool_call_script=[
            [_call("send", {"to": "ops@example.com"})],
            [_call("note", {"text": "mail blocked"})],
        ],
    )
    result = AgentNode(model, harness).run("tell ops", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "I could not send mail, so I left a note"
    assert [c.status for c in result.tool_calls] == [ToolCallStatus.DENIED, ToolCallStatus.OK]
    assert len(result.denied) == 1
    assert result.denied[0].tool == "send"
    assert executed == []  # the denied tool body never ran

    denial = _tool_messages(model.calls[1])[0]
    assert denial.content.startswith("PERMISSION_DENIED:")
    assert "requires approval" in denial.content
    assert denial.status == "error"


def test_unknown_tool_request_is_denied_not_fatal():
    harness = _harness(
        [ToolSpec(name="note", description="", fn=lambda text: "noted")],
        [{"action": "allow", "pattern": "note"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "that tool does not exist"],
        tool_call_script=[[_call("hallucinated", {"x": 1})]],
    )
    result = AgentNode(model, harness).run("go", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.tool_calls[0].status is ToolCallStatus.DENIED
    assert "unknown tool" in result.tool_calls[0].detail


def test_hook_denial_also_comes_back_as_a_tool_result():
    def block_rm(name, args):
        if "rm " in str(args.get("cmd", "")):
            return HookDecision(action=HookAction.DENY, reason="rm blocked")
        return None

    harness = _harness(
        [ToolSpec(name="run", description="", fn=lambda cmd: f"ran {cmd}")],
        [{"action": "allow", "pattern": "run"}],
        pre_hooks=(block_rm,),
    )
    model = ToolScriptedChatModel(
        responses=["", "", "I will not delete anything"],
        tool_call_script=[[_call("run", {"cmd": "rm -rf /"})], [_call("run", {"cmd": "ls"})]],
    )
    result = AgentNode(model, harness).run("clean up", _ctx())

    assert [c.status for c in result.tool_calls] == [ToolCallStatus.DENIED, ToolCallStatus.OK]
    assert "rm blocked" in result.tool_calls[0].detail
    assert result.termination_reason is StopReason.TARGET_MET


def test_crashing_tool_becomes_a_tool_error():
    def explode(x: int) -> int:
        raise ValueError("nope")

    harness = _harness(
        [ToolSpec(name="explode", description="", fn=explode)],
        [{"action": "allow", "pattern": "explode"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "the tool failed"], tool_call_script=[[_call("explode", {"x": 1})]]
    )
    result = AgentNode(model, harness).run("go", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.tool_calls[0].status is ToolCallStatus.ERROR
    assert result.tool_calls[0].detail.startswith("TOOL_ERROR:")
    assert "nope" in result.tool_calls[0].detail


def test_invented_argument_names_get_the_real_schema_in_the_error():
    """A model that calls `read(filename=...)` when the tool takes `path` is
    told the actual parameter list — the bare TypeError names the wrong
    argument but not the right ones, which a weak model cannot recover from."""

    def read(path: str, limit: int = 0) -> str:
        return path

    harness = _harness(
        [ToolSpec(name="read", description="", fn=read)],
        [{"action": "allow", "pattern": "read"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "ok"], tool_call_script=[[_call("read", {"filename": "a.md"})]]
    )
    result = AgentNode(model, harness).run("go", _ctx())

    detail = result.tool_calls[0].detail
    assert result.tool_calls[0].status is ToolCallStatus.ERROR
    assert "unexpected keyword argument" in detail
    assert "takes exactly: path, limit=…" in detail


@requires_sandbox
def test_sandbox_violation_surfaces_as_a_tool_error(tmp_path):
    """The executor's boundary reaches the model as a readable result — the
    run continues, and the file outside the workspace is still unread."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "allowed.txt").write_text("inside", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("password123", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    harness = _sandboxed_harness(
        [ToolSpec(name="read_file", description="read a file", fn=read_file)],
        [{"action": "allow", "pattern": "read_file"}],
        workspace,
    )
    model = ToolScriptedChatModel(
        responses=["", "", "I can only read inside the workspace: inside"],
        tool_call_script=[
            [_call("read_file", {"path": str(secret)})],
            [_call("read_file", {"path": str(workspace / "allowed.txt")})],
        ],
    )
    result = AgentNode(model, harness).run("read the secret", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert [c.status for c in result.tool_calls] == [ToolCallStatus.ERROR, ToolCallStatus.OK]
    violation = result.tool_calls[0].detail
    assert violation.startswith("SANDBOX_VIOLATION:")
    assert "outside its workspace" in violation
    assert "password123" not in violation
    assert result.tool_calls[1].detail == "inside"


@requires_sandbox
def test_agent_edits_a_file_inside_the_sandbox_and_is_blocked_outside(tmp_path):
    """§4.1's acceptance shape: a model-chosen edit lands, and the same tool on
    a path outside the grant does not — the difference is enforced in code."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    inside = workspace / "notes.md"
    inside.write_text("draft\n", encoding="utf-8")
    outside = tmp_path / "notes.md"
    outside.write_text("original\n", encoding="utf-8")

    def append_line(path: str, line: str) -> str:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return f"appended to {path}"

    harness = _sandboxed_harness(
        [ToolSpec(name="append_line", description="append a line to a file", fn=append_line)],
        [{"action": "allow", "pattern": "append_line"}],
        workspace,
    )
    model = ToolScriptedChatModel(
        responses=["", "", "I appended inside the workspace only"],
        tool_call_script=[
            [_call("append_line", {"path": str(inside), "line": "reviewed"})],
            [_call("append_line", {"path": str(outside), "line": "reviewed"})],
        ],
    )
    result = AgentNode(model, harness).run("append 'reviewed' to the notes", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert [c.status for c in result.tool_calls] == [ToolCallStatus.OK, ToolCallStatus.ERROR]
    assert inside.read_text(encoding="utf-8") == "draft\nreviewed\n"
    assert outside.read_text(encoding="utf-8") == "original\n"
    assert result.tool_calls[1].detail.startswith("SANDBOX_VIOLATION:")


def test_a_denied_call_records_which_gate_refused_it():
    harness = _harness(
        [ToolSpec(name="send", description="", fn=lambda to: "sent")],
        [{"action": "ask", "pattern": "send"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "I could not send it"],
        tool_call_script=[[_call("send", {"to": "ops"})]],
    )
    result = AgentNode(model, harness).run("tell ops", _ctx())

    assert [c.refused_by for c in result.refused] == ["policy"]
    assert result.refused == result.denied


@requires_sandbox
def test_a_sandbox_blocked_run_audits_as_refused_not_merely_errored(tmp_path):
    """A sandbox violation reaches the model as an error so the run can
    continue, which keeps it out of `denied` — the policy list. A run the
    sandbox blocked from end to end would then audit as "nothing was refused",
    so the question has its own answer."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("password123", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    harness = _sandboxed_harness(
        [ToolSpec(name="read_file", description="read a file", fn=read_file)],
        [{"action": "allow", "pattern": "read_file"}],
        workspace,
    )
    model = ToolScriptedChatModel(
        responses=["", "I could not read outside the workspace"],
        tool_call_script=[[_call("read_file", {"path": str(secret)})]],
    )
    result = AgentNode(model, harness).run("read the secret", _ctx())

    assert result.denied == []  # unchanged: `denied` stays the policy list
    assert [c.refused_by for c in result.refused] == ["sandbox"]
    assert result.refused[0].tool == "read_file"
    assert [c.refused_by for c in result.succeeded] == []


# -- malformed tool calls -----------------------------------------------------


def test_a_malformed_tool_call_is_fed_back_so_the_model_can_correct_it():
    """Broken argument JSON arrives in `invalid_tool_calls` with `tool_calls`
    left empty. Reading only `tool_calls` takes that for "no request, so the
    model is done" — a run that reports success having done nothing, without
    ever telling the model its JSON was broken."""
    read = []

    def read_file(path: str) -> str:
        read.append(path)
        return "budgets bound work"

    harness = _harness(
        [ToolSpec(name="read_file", description="read a file", fn=read_file)],
        [{"action": "allow", "pattern": "read_file"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "the notes say budgets bound work"],
        invalid_tool_call_script=[[_broken_call("read_file", '{"path": "/tmp/notes.md')]],
    )
    result = AgentNode(model, harness).run("what do the notes say?", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "the notes say budgets bound work"
    assert result.iterations == 2  # the broken turn was a turn, not an ending
    assert read == []  # nothing ran: there were no arguments to run it with

    record = result.tool_calls[0]
    assert record.status is ToolCallStatus.ERROR
    assert record.detail.startswith("INVALID_TOOL_CALL:")
    assert "not valid JSON" in record.detail
    assert record.refused_by == ""  # no gate refused it; it never became a request

    # The model is told, under the id it asked with — providers echo a
    # malformed call back on the wire and require a result for that id.
    feedback = _tool_messages(model.calls[1])[0]
    assert feedback.tool_call_id == "call_broken"
    assert feedback.status == "error"
    assert "valid JSON" in feedback.content


def test_repeated_malformed_tool_calls_end_the_run_without_claiming_success():
    """The retry is bounded by the same limits as everything else, and the run
    never ends `target_met` on a request that never parsed."""
    harness = _harness(
        [ToolSpec(name="read_file", description="", fn=lambda path: "x")],
        [{"action": "allow", "pattern": "read_file"}],
    )
    model = ToolScriptedChatModel(
        responses=[""],
        invalid_tool_call_script=[[_broken_call("read_file", '{"path": ')]],
        on_exhausted="repeat",
    )
    result = AgentNode(model, harness, max_iterations=20).run("read it", _ctx())

    assert result.termination_reason is StopReason.NO_PROGRESS
    assert result.iterations == 2  # max_stalled_turns, not the iteration cap
    assert [c.status for c in result.tool_calls] == [ToolCallStatus.ERROR] * 2
    assert result.output == ""


@requires_openai_wire
def test_the_real_openrouter_wire_shape_of_a_malformed_call_is_handled():
    """Not an assumed shape: the message is built by the same converter the
    shipped OpenRouter backend (a `ChatOpenAI` subclass) runs every reply
    through, from a raw OpenAI-format assistant turn with broken arguments."""
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_wire",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/x'},
            }
        ],
    }
    converted = _convert_dict_to_message(raw)
    assert converted.tool_calls == []  # the empty list a loop reads as "done"
    assert [c["id"] for c in converted.invalid_tool_calls] == ["call_wire"]

    class WireChatModel(ToolScriptedChatModel):
        """Turn one is the converter's own message; later turns are scripted."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            if self.call_count > 1:
                return result
            message = _convert_dict_to_message(raw)
            message.usage_metadata = result.generations[0].message.usage_metadata
            return ChatResult(generations=[ChatGeneration(message=message)])

    harness = _harness(
        [ToolSpec(name="read_file", description="", fn=lambda path: "inside")],
        [{"action": "allow", "pattern": "read_file"}],
    )
    model = WireChatModel(responses=["", "I fixed the arguments"])
    result = AgentNode(model, harness).run("read it", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "I fixed the arguments"
    assert result.tool_calls[0].detail.startswith("INVALID_TOOL_CALL:")
    assert _tool_messages(model.calls[1])[0].tool_call_id == "call_wire"


def test_tool_arguments_that_are_not_a_mapping_become_a_tool_error():
    """`AIMessage` does not validate on assignment, so a merged streaming turn
    or a hand-built one can hand the loop args that are not a JSON object.
    `dict(["not", "a", "mapping"])` raises from inside the stdlib; the model
    has to read that as a tool error, not the caller as a `ValueError`."""

    class NonMappingArgsModel(ToolScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            if self.call_count > 1:
                return result
            message = result.generations[0].message
            message.tool_calls = [
                {"name": "echo", "args": ["not", "a", "mapping"], "id": "c1", "type": "tool_call"}
            ]
            return ChatResult(generations=[ChatGeneration(message=message)])

    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = NonMappingArgsModel(responses=["", "I sent the wrong shape"])
    result = AgentNode(model, harness).run("echo a", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    record = result.tool_calls[0]
    assert record.status is ToolCallStatus.ERROR
    assert record.detail.startswith(
        "TOOL_ERROR: tool arguments must be a JSON object, got list"
    )
    # The wrong-shape error also names the real parameters — a model that sent
    # a list needs the schema as much as one that misnamed a keyword.
    assert "takes exactly: text" in record.detail
    assert record.args == {}
    assert _tool_messages(model.calls[1])[0].content == record.detail


# -- stopping -----------------------------------------------------------------


def test_budget_exhaustion_stops_the_loop_cleanly():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=["keep going"],
        tool_call_script=[[_call("echo", {"text": "a"})]],
        on_exhausted="repeat",
    )
    # Every scripted turn charges tokens, so the meter crosses the line mid-loop.
    ctx = _ctx(Budget(max_tokens=200))
    result = AgentNode(model, harness, max_iterations=50, max_stalled_turns=99).run("go", ctx)

    assert result.termination_reason is StopReason.BUDGET_EXHAUSTED
    assert "max_tokens" in result.note
    assert 0 < model.call_count < 50
    assert ctx.meter.tokens >= 200


def test_budget_iterations_stop_the_loop_cleanly():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=["keep going"],
        tool_call_script=[[_call("echo", {"text": "a"})]],
        on_exhausted="repeat",
    )
    ctx = _ctx(Budget(max_iterations=3))
    result = AgentNode(model, harness, max_iterations=50, max_stalled_turns=99).run("go", ctx)

    assert result.termination_reason is StopReason.BUDGET_EXHAUSTED
    assert "max_iterations" in result.note
    # Each model call is one charged iteration, so the run's ceiling bounds the
    # loop inside the node, not just the number of nodes.
    assert model.call_count == 3
    assert ctx.meter.iterations == 3


def test_max_iterations_stops_the_loop():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=["keep going"],
        tool_call_script=[[_call("echo", {"text": "a"})]],
        on_exhausted="repeat",
    )
    result = AgentNode(model, harness, max_iterations=3, max_stalled_turns=99).run("go", _ctx())

    assert result.termination_reason is StopReason.MAX_ITERATIONS
    assert result.iterations == 3
    assert model.call_count == 3


def test_repeating_a_denied_call_stops_for_no_progress():
    """Arguing with a policy the model cannot change is the classic non-
    terminating loop; two fruitless turns end it."""
    harness = _harness(
        [ToolSpec(name="send", description="", fn=lambda to: "sent")],
        [{"action": "ask", "pattern": "send"}],
    )
    model = ToolScriptedChatModel(
        responses=["retrying"],
        tool_call_script=[[_call("send", {"to": "ops"})]],
        on_exhausted="repeat",
    )
    result = AgentNode(model, harness, max_iterations=20).run("send it", _ctx())

    assert result.termination_reason is StopReason.NO_PROGRESS
    assert result.iterations == 2
    assert all(c.status is ToolCallStatus.DENIED for c in result.tool_calls)


def test_repeating_an_identical_successful_call_is_not_progress():
    """A tool that keeps succeeding on the same arguments is still a loop: the
    second identical request buys nothing the model does not already have."""
    harness = _harness(
        [ToolSpec(name="clock", description="", fn=lambda: "12:00")],
        [{"action": "allow", "pattern": "clock"}],
    )
    model = ToolScriptedChatModel(
        responses=["checking"], tool_call_script=[[_call("clock", {})]], on_exhausted="repeat"
    )
    result = AgentNode(model, harness, max_iterations=20).run("what time is it?", _ctx())

    assert result.termination_reason is StopReason.NO_PROGRESS
    assert result.iterations == 3  # new call, then two repeats
    assert all(c.status is ToolCallStatus.OK for c in result.tool_calls)


def test_a_repeated_call_that_returns_something_new_is_progress():
    """The shapes §4.1 exists for — poll until ready, re-run the suite after an
    edit, `git status` between edits — all re-send byte-identical arguments and
    learn something new each time. Fingerprinting the request calls that a
    stall and kills the run two turns in; the results are what changed."""
    observations = iter(["pending", "building", "ready"])

    def deploy_status() -> str:
        return next(observations, "ready")

    harness = _harness(
        [ToolSpec(name="deploy_status", description="", fn=deploy_status)],
        [{"action": "allow", "pattern": "deploy_status"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "", "", "the deploy is ready"],
        tool_call_script=[
            [_call("deploy_status", {}, id="p1")],
            [_call("deploy_status", {}, id="p2")],
            [_call("deploy_status", {}, id="p3")],
        ],
    )
    result = AgentNode(model, harness, max_iterations=20).run("poll until ready", _ctx())

    assert result.termination_reason is StopReason.TARGET_MET
    assert result.output == "the deploy is ready"
    assert [c.detail for c in result.tool_calls] == ["pending", "building", "ready"]


def test_a_stopped_run_does_not_hand_back_a_mid_loop_utterance_as_the_answer():
    """Only `target_met` produces an answer. Anything else stopped the loop
    mid-work, and the model's last words there are commentary — a downstream
    node reading `answer` without `termination_reason` must not find something
    that reads like a result."""
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )

    def looping_model():
        return ToolScriptedChatModel(
            responses=["Let me check the logs first."],
            tool_call_script=[[_call("echo", {"text": "a"})]],
            on_exhausted="repeat",
        )

    result = AgentNode(looping_model(), harness, max_iterations=2, max_stalled_turns=99).run(
        "go", _ctx()
    )

    assert result.termination_reason is StopReason.MAX_ITERATIONS
    assert result.output == ""
    assert result.partial_output == "Let me check the logs first."  # kept, just not as the answer

    # And through the state write, which is where a downstream node reads it.
    agent = AgentNode(looping_model(), harness, max_iterations=2, max_stalled_turns=99)
    update = agent(AgentGraphState(task="go"), _ctx())
    assert update["answer"] == ""
    assert update["termination_reason"] == StopReason.MAX_ITERATIONS.value


def test_model_failure_terminates_with_error_not_a_traceback():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=[""], tool_call_script=[[_call("echo", {"text": "a"})]]
    )  # on_exhausted="raise": the second turn blows up
    result = AgentNode(model, harness).run("go", _ctx())

    assert result.termination_reason is StopReason.ERROR
    assert "exhausted" in result.note
    assert result.iterations == 2


_LOOPING = [_call("echo", {"text": "a"})]


@pytest.mark.parametrize(
    ("responses", "script", "budget", "caps", "on_exhausted", "expected"),
    [
        (["done"], [], None, (3, 2), "repeat", StopReason.TARGET_MET),
        (["x"], [_LOOPING], None, (3, 99), "repeat", StopReason.MAX_ITERATIONS),
        (["x"], [_LOOPING], None, (20, 2), "repeat", StopReason.NO_PROGRESS),
        (
            ["x"],
            [_LOOPING],
            Budget(max_tokens=1),
            (20, 99),
            "repeat",
            StopReason.BUDGET_EXHAUSTED,
        ),
        ([""], [_LOOPING], None, (3, 99), "raise", StopReason.ERROR),
    ],
    ids=["target_met", "max_iterations", "no_progress", "budget_exhausted", "error"],
)
def test_every_run_ends_with_a_machine_readable_reason(
    responses, script, budget, caps, on_exhausted, expected
):
    """Whatever happens, `termination_reason` is a StopReason — never None,
    never "the loop just stopped"."""
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=responses, tool_call_script=script, on_exhausted=on_exhausted
    )
    max_iterations, max_stalled = caps
    agent = AgentNode(model, harness, max_iterations=max_iterations, max_stalled_turns=max_stalled)
    result = agent.run("go", _ctx(budget))

    assert isinstance(result.termination_reason, StopReason)
    assert result.termination_reason is expected


def test_hard_budget_exception_is_not_downgraded_to_a_stop_reason():
    """`max_seconds` is delivered as an exception into the running node. A loop
    that caught it and returned a tidy reason would turn the run's hard ceiling
    into a soft one, and its writes would reach state anyway."""

    class DeadlineModel(ToolScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise BudgetExceeded("max_seconds reached while a node was running")

    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    with pytest.raises(BudgetExceeded, match="max_seconds"):
        AgentNode(DeadlineModel(responses=["x"]), harness).run("go", _ctx())


def test_hard_budget_exception_from_a_tool_is_not_downgraded():
    def interrupted(text: str) -> str:
        raise BudgetExceeded("max_seconds reached while a node was running")

    harness = _harness(
        [ToolSpec(name="echo", description="", fn=interrupted)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "done"], tool_call_script=[[_call("echo", {"text": "a"})]]
    )
    with pytest.raises(BudgetExceeded, match="max_seconds"):
        AgentNode(model, harness).run("go", _ctx())


def test_a_model_without_bind_tools_is_a_configuration_error():
    """The Claude-CLI backend cannot call tools at all; the error has to say so
    rather than surfacing a bare NotImplementedError from inside LangChain."""
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    agent = AgentNode(ScriptedChatModel(responses=["done"]), harness)
    with pytest.raises(AgentConfigError, match="bind_tools"):
        agent.run("go", _ctx())


def test_a_run_with_no_budget_context_still_works():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(responses=["done"])
    result = AgentNode(model, harness).run("go")  # no RunContext at all

    assert result.termination_reason is StopReason.TARGET_MET


# -- composing as a graph node ------------------------------------------------


class AgentGraphState(GraphARCState):
    task: str
    answer: str = ""
    termination_reason: str | None = None
    agent_result: AgentResult | None = None


def test_agent_node_runs_inside_a_graph(tmp_path, trace):
    target = tmp_path / "doc.md"
    target.write_text("typed state contracts", encoding="utf-8")

    def read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    harness = _harness(
        [
            ToolSpec(name="read_file", description="read a file", fn=read_file),
            ToolSpec(name="rm", description="delete a file", fn=lambda path: "gone"),
        ],
        [{"action": "allow", "pattern": "read_file"}, {"action": "deny", "pattern": "rm"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "the doc mentions typed state contracts"],
        tool_call_script=[[_call("read_file", {"path": str(target)})]],
    )
    agent = AgentNode(model, harness, name="reader", record_field="agent_result", trace=trace)

    g = GraphARC(AgentGraphState, name="agent_graph", budget=Budget(max_iterations=20), trace=trace)
    g.add_node("reader", agent, writes=agent.writes)
    g.add_edge(START, "reader")
    g.add_edge("reader", END)
    compiled = g.compile()
    out = compiled.invoke({"task": "what does the doc mention?"})

    assert out["answer"] == "the doc mentions typed state contracts"
    assert out["termination_reason"] == StopReason.TARGET_MET.value
    assert isinstance(out["agent_result"], AgentResult)
    assert [c.tool for c in out["agent_result"].tool_calls] == ["read_file"]
    assert model.bound_tool_names == ["read_file"]

    events = trace.read_events(compiled.last_run.run_id)
    phases = [(e.node, e.phase) for e in events]
    assert ("reader", "start") in phases
    assert ("reader:model", "model") in phases
    assert ("reader:read_file", "tool") in phases
    assert ("reader:stop", "stop") in phases
    # Sub-node steps come from the same run counter, so each loop step is its
    # own replay point rather than collapsing into the node's single step.
    node_step = next(e.step for e in events if e.node == "reader")
    inner = [
        e.step for e in events if e.node != "reader" and e.phase != "topology"
    ]
    assert inner == sorted(inner)
    assert len(set(inner)) == len(inner)
    assert min(inner) > node_step

    # Each turn's usage is charged once, not once by the agent and again by the
    # runtime's automatic usage callback.
    per_turn = sum(e.tokens or 0 for e in events if e.phase == "model")
    assert per_turn > 0
    assert compiled.last_run.meter.tokens == per_turn
    node_end = next(e for e in events if e.node == "reader" and e.phase == "end")
    assert node_end.tokens == per_turn


def test_writes_declares_exactly_what_the_node_returns():
    """The graph rejects an undeclared write, so `writes` has to be exact."""
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(responses=["done"])
    agent = AgentNode(model, harness, record_field="agent_result")

    assert agent.writes == {"answer", "termination_reason", "agent_result"}
    update = agent(AgentGraphState(task="go"), _ctx())
    assert set(update) == agent.writes

    plain = AgentNode(ToolScriptedChatModel(responses=["done"]), harness)
    assert plain.writes == {"answer", "termination_reason"}
    assert set(plain(AgentGraphState(task="go"), _ctx())) == plain.writes


def test_denied_tool_stays_denied_inside_a_graph_run(trace):
    """The gate is the harness, not the prompt: the same policy that hides a
    tool also refuses it when the model names it anyway."""
    deleted = []

    harness = _harness(
        [ToolSpec(name="rm", description="delete", fn=lambda path: deleted.append(path))],
        [{"action": "deny", "pattern": "rm"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "I am not allowed to delete that"],
        tool_call_script=[[_call("rm", {"path": "/etc/passwd"})]],
    )
    agent = AgentNode(model, harness, name="cleaner", record_field="agent_result", trace=trace)

    g = GraphARC(AgentGraphState, name="denied_graph", trace=trace)
    g.add_node("cleaner", agent, writes=agent.writes)
    g.add_edge(START, "cleaner")
    g.add_edge("cleaner", END)
    out = g.compile().invoke({"task": "delete /etc/passwd"})

    assert deleted == []
    assert out["termination_reason"] == StopReason.TARGET_MET.value
    record = out["agent_result"].tool_calls[0]
    assert record.status is ToolCallStatus.DENIED
    assert "denied by policy" in record.detail


def test_missing_task_field_is_a_configuration_error():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    agent = AgentNode(ToolScriptedChatModel(responses=["done"]), harness, task_field="question")
    with pytest.raises(AgentConfigError, match="question"):
        agent(AgentGraphState(task="go"), _ctx())


def test_prompt_fn_overrides_the_task_field():
    harness = _harness(
        [ToolSpec(name="echo", description="", fn=lambda text: text)],
        [{"action": "allow", "pattern": "echo"}],
    )
    model = ToolScriptedChatModel(responses=["done"])
    agent = AgentNode(model, harness, prompt_fn=lambda s: f"CUSTOM: {s.task}")
    agent(AgentGraphState(task="go"), _ctx())

    assert model.calls[0][-1].content == "CUSTOM: go"


def test_max_iterations_below_one_is_rejected():
    harness = _harness([], [])
    with pytest.raises(AgentConfigError, match="max_iterations"):
        AgentNode(ToolScriptedChatModel(responses=[]), harness, max_iterations=0)


def test_a_type_error_inside_a_correct_call_gets_no_signature_hint():
    """`len(None)` inside the tool body is the tool's bug, not the model's —
    a hint asserting the (valid) arguments were wrong steers the model into
    rewriting a call that was fine."""

    def broken_inside(path: str) -> int:
        return len(None)  # type: ignore[arg-type]

    harness = _harness(
        [ToolSpec(name="broken", description="", fn=broken_inside)],
        [{"action": "allow", "pattern": "broken"}],
    )
    model = ToolScriptedChatModel(
        responses=["", "ok"], tool_call_script=[[_call("broken", {"path": "a"})]]
    )
    result = AgentNode(model, harness).run("go", _ctx())
    detail = result.tool_calls[0].detail
    assert result.tool_calls[0].status is ToolCallStatus.ERROR
    assert "takes exactly" not in detail, detail
