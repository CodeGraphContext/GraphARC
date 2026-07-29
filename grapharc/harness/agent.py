"""The agent node: a model, a harness, and a loop whose every gate is code.

This is the unit that joins the three halves of the toolkit. The registry,
the permission policy and the sandboxed executor already existed; nothing
called them. `AgentNode` is what calls them, and it composes as an ordinary
GraphARC node so budgets, write permissions and traces apply to an agent turn
the same way they apply to a deterministic one.

    observe state -> model (visible tools bound) -> tool request
                  -> harness.call: permission -> hooks -> sandbox
                  -> result (or denial, or violation) back to the model
                  -> repeat until a recorded StopReason

Three properties the loop is built around:

- **Nothing routes around the harness.** Tool arguments come from a model, so
  the node never touches `ToolSpec.fn`; every execution goes through
  `Harness.call`, which is where deny/ask/allow and the sandbox live. The
  model is only offered `harness.visible_tools()`, so a denied tool's schema
  is never even described to it.
- **A refusal is data, not an exception.** A denied tool, a sandbox violation,
  a crashing tool and a call whose arguments would not parse all come back as
  tool results the model can read and react to. Killing the run on a denial
  would teach every caller to widen their policy until nothing is denied.
- **Every exit is named.** The loop cannot fall out of the bottom: it ends on
  a `StopReason`, recorded in the returned `AgentResult` and written to state.
  Only `target_met` fills the answer field: a run that stopped for any other
  reason has no answer to give, and its last mid-loop utterance is kept in
  `partial_output` where nothing downstream can mistake it for one.

Two things the loop deliberately does *not* treat as an ending. A tool call
whose JSON arguments are malformed arrives in `invalid_tool_calls` with
`tool_calls` left empty; read as "no request" it would end the run as
`target_met` with no answer, so the parse error goes back to the model as that
call's result instead. And progress is measured in *observations*, not
requests: re-sending identical arguments is how you poll until ready or re-run
a suite after an edit, so a repeated call that returns a new result counts as
progress and only a repeated call returning the same result stalls.

Traces: sub-node steps are recorded with their own phases — `"model"` per
model call, `"tool"` per tool call, `"stop"` for the termination — rather than
the node-level `"start"/"end"/"error"`. `observe.metrics` counts node
executions and run tokens from `"end"` events, so reusing that phase inside a
node would double-count the tokens the node wrapper already reports.
"""

from __future__ import annotations

import inspect
import json
import time
import types
import typing
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from grapharc.harness.core import Harness
from grapharc.harness.executor import SandboxViolation
from grapharc.harness.permissions import PermissionDenied
from grapharc.harness.tools import ToolSpec
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.convergence import StopReason
from grapharc.runtime.graph import RunContext

DEFAULT_SYSTEM_PROMPT = (
    "You are a tool-using agent inside a GraphARC graph.\n"
    "Call a tool when you need information or an effect you cannot produce "
    "yourself; reply without a tool call when you are done.\n"
    "Permissions and the sandbox are enforced in code, not by these "
    "instructions: a call can come back PERMISSION_DENIED or "
    "SANDBOX_VIOLATION. That answer is final. Do not retry the same call — "
    "take another approach, or say plainly what you could not do.\n"
    "An INVALID_TOOL_CALL result is different: nothing ran because your "
    "arguments were not valid JSON. Send that call again with the JSON "
    "corrected.\n"
    "Your first reply that requests no tool is the answer."
)

DEFAULT_MAX_ITERATIONS = 8
# Consecutive turns that produced no new successful tool result before the
# loop calls it: a model arguing with a policy it cannot change.
DEFAULT_MAX_STALLED_TURNS = 2
DEFAULT_MAX_TOOL_RESULT_CHARS = 4000


class AgentConfigError(Exception):
    """The node was wired to state (or a model) it cannot work with."""


class ToolCallStatus(StrEnum):
    OK = "ok"
    DENIED = "denied"
    ERROR = "error"


class ToolCallRecord(BaseModel):
    """One attempted tool call and what the harness did with it."""

    iteration: int
    tool: str
    args: dict[str, Any] = {}
    status: ToolCallStatus
    detail: str = ""  # the rendered result, the denial reason, or the error
    duration_ms: float = 0.0
    # Which gate stopped this call before its body ran: "policy" | "sandbox",
    # empty if it was never refused. Recorded as a field rather than inferred
    # from `detail`, so an audit is a lookup and not a string match.
    refused_by: str = ""


class AgentResult(BaseModel):
    """The outcome of one agent loop — always carrying a termination reason."""

    # The answer. Non-empty only when `termination_reason` is TARGET_MET: any
    # other reason means the loop stopped mid-work, and mid-work commentary is
    # not an answer.
    output: str = ""
    termination_reason: StopReason
    iterations: int = 0
    tool_calls: list[ToolCallRecord] = []
    note: str = ""  # the budget line, the error repr: why this reason, not just which
    # The last thing the model said before a non-TARGET_MET stop. Kept for
    # debugging and hand-off, out of the field a caller reads as the answer.
    partial_output: str = ""

    @property
    def denied(self) -> list[ToolCallRecord]:
        """Calls the permission policy refused. See `refused` for every gate."""
        return [c for c in self.tool_calls if c.status is ToolCallStatus.DENIED]

    @property
    def refused(self) -> list[ToolCallRecord]:
        """Every call a gate stopped before it ran — policy *and* sandbox.

        A sandbox violation is an error to the model (the run continues), so it
        is recorded with `status=ERROR` and never appears in `denied`. Auditing
        "was this run refused anything?" through `denied` alone therefore reads
        clean for a run the sandbox blocked from end to end; this is the
        question that check should ask.
        """
        return [c for c in self.tool_calls if c.refused_by]

    @property
    def errored(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.status is ToolCallStatus.ERROR]

    @property
    def succeeded(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.status is ToolCallStatus.OK]


# -- ToolSpec -> LangChain/OpenAI tool schema ---------------------------------

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON-schema type, defaulting to string.

    Unknown and unannotated parameters become strings rather than being
    dropped: a parameter the model cannot see is a parameter it cannot fill.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "string"
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is not type(None):
                return _json_type(arg)
        return "string"
    if origin is not None:
        annotation = origin
    if not isinstance(annotation, type):
        return "string"
    return _JSON_TYPES.get(annotation, "string")


def tool_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a `ToolSpec` as an OpenAI-format function schema for `bind_tools`.

    A ToolSpec carries no parameter schema of its own, so the callable's
    signature is the contract: annotations become JSON types, defaulted
    parameters become optional, and a `**kwargs` tool is declared open.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    open_ended = False
    try:
        signature: inspect.Signature | None = inspect.signature(spec.fn)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        signature = None

    if signature is None:
        open_ended = True
    else:
        try:
            hints = typing.get_type_hints(spec.fn)
        except Exception:  # noqa: BLE001 — an unresolvable hint is not a fatal error
            hints = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                open_ended = True
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue  # a JSON object cannot express *args
            properties[name] = {"type": _json_type(hints.get(name, param.annotation))}
            if param.default is inspect.Parameter.empty:
                required.append(name)

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    if open_ended:
        parameters["additionalProperties"] = True
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": parameters,
        },
    }


def tool_schemas(harness: Harness) -> list[dict[str, Any]]:
    """Schemas for exactly the tools the policy would let this harness run."""
    return [tool_schema(spec) for spec in harness.visible_tools()]


# -- the node -----------------------------------------------------------------


def _message_text(message: BaseMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = message.content
    return content if isinstance(content, str) else str(content)


def _outcome_key(record: ToolCallRecord) -> tuple[str, str, str]:
    """What a call actually produced — the unit progress is measured in.

    Progress is a changed *observation*, not an unrepeated *request*. Polling
    until a job is ready, re-running the suite after an edit and `git status`
    between edits all re-send byte-identical arguments and learn something new
    every time; fingerprinting the request alone calls that a stall and ends
    the run at `max_stalled_turns`. Including the result is the difference: a
    repeat that returns something new is progress, a repeat that returns the
    same thing is not.
    """
    return (record.tool, json.dumps(record.args, sort_keys=True, default=str), record.detail)


def _coerce_args(raw: Any) -> dict[str, Any]:
    """Model-supplied arguments as a mapping, or a TypeError naming the shape.

    `dict(["a", "b"])` raises a ValueError from inside the stdlib. Raised here,
    inside `_execute`'s try, it becomes an ordinary tool-error result the model
    can read and correct rather than an exception thrown out of `run()`.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(f"tool arguments must be a JSON object, got {type(raw).__name__}")


class AgentNode:
    """A tool-using agent loop, shaped as a GraphARC node.

    Call it directly (`node(state, ctx)`) or register it:

        agent = AgentNode(model, harness, trace=trace)
        g.add_node("agent", agent, writes=agent.writes)

    `writes` reports exactly the state fields it will write, so the graph's
    write-permission check stays declarative.
    """

    def __init__(
        self,
        model: BaseChatModel,
        harness: Harness,
        *,
        name: str = "agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_stalled_turns: int = DEFAULT_MAX_STALLED_TURNS,
        task_field: str = "task",
        output_field: str = "answer",
        reason_field: str = "termination_reason",
        record_field: str | None = None,
        prompt_fn: Callable[[Any], str] | None = None,
        trace: TraceRecorder | None = None,
        max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
    ) -> None:
        if max_iterations < 1:
            raise AgentConfigError("max_iterations must be at least 1")
        self.model = model
        self.harness = harness
        self.name = name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_stalled_turns = max_stalled_turns
        self.task_field = task_field
        self.output_field = output_field
        self.reason_field = reason_field
        self.record_field = record_field
        self.prompt_fn = prompt_fn
        self.trace = trace
        self.max_tool_result_chars = max_tool_result_chars

    @property
    def writes(self) -> set[str]:
        fields = {self.output_field, self.reason_field}
        if self.record_field:
            fields.add(self.record_field)
        return fields

    def __call__(self, state: Any, ctx: RunContext) -> dict[str, Any]:
        result = self.run(self.observe(state), ctx)
        update: dict[str, Any] = {
            self.output_field: result.output,
            self.reason_field: result.termination_reason.value,
        }
        if self.record_field:
            update[self.record_field] = result
        return update

    def observe(self, state: Any) -> str:
        """State -> prompt. Override via `prompt_fn` for anything richer."""
        if self.prompt_fn is not None:
            return self.prompt_fn(state)
        value = state.get(self.task_field) if isinstance(state, dict) else None
        if value is None:
            value = getattr(state, self.task_field, None)
        if value is None:
            raise AgentConfigError(
                f"agent {self.name!r} found no task in state field "
                f"{self.task_field!r}; set task_field or pass prompt_fn"
            )
        return str(value)

    def run(self, prompt: str, ctx: RunContext | None = None) -> AgentResult:
        """Run the loop to a termination reason. Never raises on a tool refusal.

        `ctx` supplies the budget meter and the trace identity; without one a
        fresh unbounded context is created so the node is usable (and testable)
        outside a graph.
        """
        if ctx is None:
            ctx = RunContext(
                run_id=uuid.uuid4().hex[:12], graph=self.name, meter=BudgetMeter(Budget())
            )

        model = self._bind_tools()
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        records: list[ToolCallRecord] = []
        seen_outcomes: set[tuple[str, str, str]] = set()
        iterations = 0
        stalled = 0
        output = ""
        reason: StopReason | None = None
        note = ""

        while True:
            exhausted = ctx.meter.exceeded()
            if exhausted is not None:
                reason, note = StopReason.BUDGET_EXHAUSTED, exhausted
                break
            if iterations >= self.max_iterations:
                reason = StopReason.MAX_ITERATIONS
                note = f"iteration cap reached ({iterations}/{self.max_iterations})"
                break

            # One model call is one unit of bounded work: charging it here is what
            # keeps a loop *inside* a node under the same ceiling as the graph.
            ctx.meter.charge_iteration()
            iterations += 1

            started = time.perf_counter()
            try:
                message = model.invoke(messages)
            except BudgetExceeded:
                # The run's hard ceiling — including a wall-clock interrupt
                # delivered mid-call. Downgrading it to a soft stop reason is
                # how a ceiling stops being one.
                raise
            except Exception as exc:  # noqa: BLE001 — reported as a reason, not a crash
                self._emit(
                    ctx,
                    "model",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=repr(exc),
                )
                reason, note = StopReason.ERROR, repr(exc)
                break
            duration_ms = (time.perf_counter() - started) * 1000

            tokens = self._charge_tokens(ctx, message)
            messages.append(message)
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            # A tool call whose arguments are not parseable JSON lands here with
            # `tool_calls` left empty (langchain_openai, and so every OpenAI-wire
            # backend including the shipped OpenRouter one). Reading only
            # `tool_calls` would take a broken request for "the model is done".
            invalid_calls = list(getattr(message, "invalid_tool_calls", None) or [])
            price, model_name = self._last_price(model)
            self._emit(
                ctx,
                "model",
                duration_ms=duration_ms,
                tokens=tokens,
                cost_usd=price,
                model=model_name,
                state_delta={
                    "iteration": iterations,
                    "tool_calls": len(tool_calls),
                    "invalid_tool_calls": len(invalid_calls),
                },
            )

            text = _message_text(message)
            if text:
                output = text
            if not tool_calls and not invalid_calls:
                reason = StopReason.TARGET_MET
                break

            progressed = False
            for index, call in enumerate(tool_calls):
                record = self._execute(ctx, iterations, call)
                records.append(record)
                messages.append(self._tool_message(record, call, iterations, index))
                if record.status is not ToolCallStatus.OK:
                    continue
                # A repeat that produced something new is progress; a repeat
                # that produced the same thing again is the loop that never
                # ends. Comparing results, not requests, is what tells them
                # apart — see `_outcome_key`.
                key = _outcome_key(record)
                progressed = progressed or key not in seen_outcomes
                seen_outcomes.add(key)

            # A malformed call is the model's mistake to correct, not an answer
            # and not progress: it gets the parse error back as that call's
            # result, and the turn still counts against the loop's limits.
            for index, call in enumerate(invalid_calls, start=len(tool_calls)):
                record = self._invalid_call(ctx, iterations, call)
                records.append(record)
                messages.append(self._tool_message(record, call, iterations, index))

            stalled = 0 if progressed else stalled + 1
            if stalled >= self.max_stalled_turns:
                reason = StopReason.NO_PROGRESS
                note = f"{stalled} consecutive turns produced no new tool result"
                break

        if reason is None:  # unreachable — but "always" has to mean always
            reason, note = StopReason.ERROR, "loop exited without a termination reason"
        partial = ""
        if reason is not StopReason.TARGET_MET:
            # Whatever the model last said, it said it mid-work. Leaving it in
            # `output` hands a downstream node that reads `answer` without
            # reading `termination_reason` a plausible-looking non-answer, so
            # the answer field stays empty and the utterance is kept beside it.
            partial, output = output, ""
        result = AgentResult(
            output=output,
            termination_reason=reason,
            iterations=iterations,
            tool_calls=records,
            note=note,
            partial_output=partial,
        )
        self._emit(
            ctx,
            "stop",
            state_delta={
                "termination_reason": reason.value,
                "iterations": iterations,
                "tool_calls": len(records),
                "denied": len(result.denied),
                "refused": len(result.refused),
                "note": note,
            },
        )
        return result

    # -- internals ------------------------------------------------------------

    def _bind_tools(self) -> Any:
        """Bind the policy-filtered tool set. Denied tools are never described."""
        schemas = tool_schemas(self.harness)
        if not schemas:
            # Binding an empty tool list is an error on several providers, and
            # a toolless turn is a legitimate configuration (everything denied).
            return self.model
        try:
            return self.model.bind_tools(schemas)
        except NotImplementedError as exc:
            # `claude -p` is a whole agent driven as a text endpoint; it has no
            # tool-calling wire format at all. Say that, rather than surfacing a
            # bare NotImplementedError from three frames down.
            raise AgentConfigError(
                f"model {type(self.model).__name__} does not implement bind_tools, "
                f"so it cannot drive a tool loop ({len(schemas)} tools are visible "
                "to it); use a tool-calling backend such as openrouter/*, "
                "openai/* or ollama/*"
            ) from exc

    @staticmethod
    def _last_price(model: Any) -> tuple[float | None, str | None]:
        """What the last call to `model` cost, if the backend reported it.

        Read off the backend's `last_usage` envelope — the uniform shape both
        shipped gateways publish — rather than off the message, because a
        `cost` is a property of the call the provider billed and not of the
        text it returned. A model that publishes no envelope yields `(None,
        None)`, which records the call as unpriced instead of free.

        The value is *this call's* price, not a running total: it is read
        immediately after `invoke()` returns, and both gateways overwrite the
        envelope per call.
        """
        usage = getattr(model, "last_usage", None)
        if not isinstance(usage, dict):
            return None, None
        cost = usage.get("cost_usd")
        name = usage.get("model")
        return (None if cost is None else float(cost)), (None if name is None else str(name))

    def _charge_tokens(self, ctx: RunContext, message: AIMessage) -> int:
        """Charge this turn's reported usage, and report it for the trace.

        Naming the message is what keeps the turn counted exactly once. The
        runtime's usage callback already meters model calls made inside a node,
        so an anonymous charge would be a second, indistinguishable spend;
        identifying the call lets the meter recognise a re-report and drop it.
        Outside a graph there is no scope to claim against, so the charge lands
        normally and a standalone AgentNode still meters itself.
        """
        usage = getattr(message, "usage_metadata", None) or {}
        total = int(usage.get("total_tokens") or 0)
        if total:
            ctx.meter.charge_tokens(total, source=message)
        return total

    def _execute(self, ctx: RunContext, iteration: int, call: dict[str, Any]) -> ToolCallRecord:
        """Route one model-requested call through the harness, never around it."""
        name = str(call.get("name") or "")
        args: dict[str, Any] = {}
        refused_by = ""
        started = time.perf_counter()
        try:
            # Inside the try on purpose: the arguments are model-supplied, so
            # their *shape* is as untrusted as their values.
            args = _coerce_args(call.get("args"))
            value = self.harness.call(name, args)
        except PermissionDenied as exc:
            status, detail = ToolCallStatus.DENIED, f"PERMISSION_DENIED: {exc}"
            refused_by = "policy"
        except SandboxViolation as exc:
            status, detail = ToolCallStatus.ERROR, f"SANDBOX_VIOLATION: {exc}"
            refused_by = "sandbox"
        except BudgetExceeded:
            raise  # the run's ceiling, not this tool's failure
        except Exception as exc:  # noqa: BLE001 — a broken tool must not end the run
            status, detail = ToolCallStatus.ERROR, f"TOOL_ERROR: {exc}"
        else:
            status, detail = ToolCallStatus.OK, self._render(value)
        duration_ms = (time.perf_counter() - started) * 1000

        record = ToolCallRecord(
            iteration=iteration,
            tool=name,
            args=args,
            status=status,
            detail=detail,
            duration_ms=duration_ms,
            refused_by=refused_by,
        )
        delta: dict[str, Any] = {"tool": name, "args": args, "status": status.value}
        if refused_by:
            delta["refused_by"] = refused_by
        self._emit(
            ctx,
            "tool",
            node=f"{self.name}:{name or '<unnamed>'}",
            duration_ms=duration_ms,
            state_delta=delta,
            error=None if status is ToolCallStatus.OK else detail,
        )
        return record

    def _invalid_call(
        self, ctx: RunContext, iteration: int, call: dict[str, Any]
    ) -> ToolCallRecord:
        """Hand a model back the parse error for a call it could not spell.

        Nothing runs and nothing is refused — the request never became one. The
        point is that the model *hears about it*: told what it got wrong, it can
        re-send the call, which is the whole difference between a retryable
        mistake and a run that ends `target_met` with an empty answer.
        """
        name = str(call.get("name") or "")
        error = str(call.get("error") or "").strip()
        if not error:  # streaming chunks arrive with the raw text and no error
            error = f"arguments are not valid JSON: {call.get('args')!r}"
        detail = (
            f"INVALID_TOOL_CALL: {self._render(error)}\n"
            "Nothing ran. Send this call again with valid JSON arguments."
        )
        record = ToolCallRecord(
            iteration=iteration,
            tool=name,
            status=ToolCallStatus.ERROR,
            detail=detail,
        )
        self._emit(
            ctx,
            "tool",
            node=f"{self.name}:{name or '<unnamed>'}",
            state_delta={"tool": name, "status": record.status.value, "invalid_arguments": True},
            error=detail,
        )
        return record

    def _tool_message(
        self, record: ToolCallRecord, call: dict[str, Any], iteration: int, index: int
    ) -> ToolMessage:
        """The result the model reads next, tied to the id it asked under.

        Every id in the assistant turn needs an answer — including the ids of
        calls that failed to parse, which providers still echo back on the wire.
        """
        return ToolMessage(
            content=record.detail,
            tool_call_id=str(call.get("id") or f"{self.name}-{iteration}-{index}"),
            name=record.tool or None,
            status="success" if record.status is ToolCallStatus.OK else "error",
        )

    def _render(self, result: Any) -> str:
        if isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = repr(result)
        limit = self.max_tool_result_chars
        if len(text) > limit:
            text = text[:limit] + f"…[truncated {len(text) - limit} chars]"
        return text

    def _emit(self, ctx: RunContext, phase: str, *, node: str | None = None, **fields: Any) -> None:
        if self.trace is None:
            return
        self.trace.event(
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            attempt=ctx.attempt,
            graph=ctx.graph,
            node=node or f"{self.name}:{phase}",
            phase=phase,
            step=ctx.next_step(),
            **fields,
        )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentConfigError",
    "AgentNode",
    "AgentResult",
    "ToolCallRecord",
    "ToolCallStatus",
    "tool_schema",
    "tool_schemas",
]
