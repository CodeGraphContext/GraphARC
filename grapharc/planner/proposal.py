"""Proposals: a typed description of work that has not been authorised yet.

A planner's entire output is one of these. It names nodes and the edges between
them and nothing else: a `ProposedNode` carries a registry *key*, never a body,
so nothing in this module has anything to execute even if it wanted to. That is
the invariant the architecture rests on — a node may *propose* topology and may
never *run* it. What turns a proposal into work is
`grapharc.planner.admission`, and only after every check there has passed.

Three properties the types carry rather than describe:

- **Frozen.** Fields cannot be reassigned after construction, so the object an
  admission checker approved is the object a materialiser later reads. Deep
  immutability is *not* claimed: `ProposedNode.args` is an ordinary dict whose
  contents can still be mutated in place. `Subgraph.fingerprint()` is what
  detects that, by hashing the content rather than trusting the reference.
- **Well-formed by construction, not governed by construction.** Duplicate node
  names inside one subgraph, a name colliding with the graph's START/END
  sentinels, and a name outside the identifier charset are all
  `ValidationError`s. The governance questions — is this kind registered, is
  this edge permitted, does it fit the budget — are deliberately absent here:
  they belong to admission, where a failure is a recorded reason a planner can
  read back rather than an exception that ends the run.
- **No condition strings.** An edge carries a source, a target and an audit
  note. It carries no predicate, because a predicate authored by a model is
  model prose steering an edge — the exact thing this repo's code-only routing
  rule exists to prevent.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import END, START, RunContext
from grapharc.runtime.parsing import extract_json

# Node and kind names live in trace lines, Mermaid labels and fnmatch patterns.
# The charset excludes `>` and whitespace so an "a -> b" edge rendering can
# never be ambiguous about where the name ends.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_SENTINELS = (START, END)


class PlannerConfigError(Exception):
    """The planner node was wired to state (or a model) it cannot work with."""


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


def _check_name(value: str, *, what: str) -> str:
    if not _NAME.fullmatch(value):
        raise ValueError(
            f"{what} {value!r} is not a valid name: expected "
            r"[A-Za-z_][A-Za-z0-9_.-]{0,63}"
        )
    return value


class ProposedNode(BaseModel):
    """One node a planner wants to exist.

    `name` is the instance's identity inside the proposal; `kind` is the
    registry key that says what it *is*. They are separate so a fan-out can
    propose `worker_a`, `worker_b` and `worker_c` all of kind `edit_worker`
    while the registry still only has to allow one thing. `kind` defaults to
    `name` for the common one-off case.

    Only `kind` is governed. Admission resolves every rule — registered or not,
    permitted or not, and what it may cost — against the kind, because that is
    the thing an operator authorised; `name` resolves which node an edge points
    at and labels the node in traces, and no policy rule is ever matched against
    it. Renaming an instance therefore changes nothing about what it is allowed
    to do, in either direction.

    `args` are carried for a future materialiser and are **not** inspected by
    admission — see `grapharc.planner.admission`. Nothing consumes them today.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    # A node that expands into another proposal. This is what `max_depth`
    # bounds. Nothing here recurses into it — `Subgraph.scopes()` walks it
    # iteratively so a hostile depth cannot exhaust the stack.
    subgraph: Subgraph | None = None

    @model_validator(mode="before")
    @classmethod
    def _kind_defaults_to_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("kind"):
            name = data.get("name")
            if isinstance(name, str) and name:
                return {**data, "kind": name}
        return data

    @field_validator("name", "kind")
    @classmethod
    def _valid_name(cls, value: str, info: Any) -> str:
        if value in _SENTINELS:
            raise ValueError(f"{info.field_name} {value!r} is reserved for the graph's entry/exit")
        return _check_name(value, what=info.field_name)


class ProposedEdge(BaseModel):
    """One transition a planner wants to be allowed.

    `note` is for the audit trail. It is never parsed and never evaluated: an
    edge is admitted or refused on its endpoints alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    note: str = ""

    @field_validator("source", "target")
    @classmethod
    def _valid_endpoint(cls, value: str, info: Any) -> str:
        if value in _SENTINELS:
            return value
        return _check_name(value, what=info.field_name)

    def render(self) -> str:
        return f"{self.source} -> {self.target}"


class Subgraph(BaseModel):
    """A proposal: nodes, edges between them, and why.

    An empty proposal is legal and means "no further work" — a planner needs a
    way to say that which is not an exception. Admitting one authorises
    nothing, which is the correct outcome.

    Edge endpoints may name a node in *this* subgraph or the START/END
    sentinels. They may also name a node that already exists in the live graph;
    whether that is allowed is an admission question (`known_nodes`), not a
    construction one, because this type does not know what graph it is destined
    for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[ProposedNode, ...] = ()
    edges: tuple[ProposedEdge, ...] = ()
    rationale: str = ""
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Who produced this. `PlannerNode` overwrites whatever a model put here.
    origin: str = ""

    @model_validator(mode="after")
    def _unique_node_names(self) -> Subgraph:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                duplicates.add(node.name)
            seen.add(node.name)
        if duplicates:
            raise ValueError(f"duplicate node names in one subgraph: {sorted(duplicates)}")
        return self

    def node_names(self) -> frozenset[str]:
        """Names declared in *this* subgraph. Nested scopes are not included."""
        return frozenset(n.name for n in self.nodes)

    def scopes(self) -> Iterator[tuple[str, int, Subgraph]]:
        """Yield `(path, depth, subgraph)` for this proposal and every nested one.

        Iterative on purpose: depth is attacker-controlled in the sense that a
        proposal can arrive from a model, and a recursive walk would turn a deep
        proposal into a `RecursionError` before the depth check could reject it.
        Root is `("", 0, self)`; a nested scope's path is its owning nodes'
        names joined with `/`.
        """
        stack: list[tuple[str, int, Subgraph]] = [("", 0, self)]
        while stack:
            path, depth, sub = stack.pop()
            yield path, depth, sub
            for node in reversed(sub.nodes):
                if node.subgraph is not None:
                    child = f"{path}/{node.name}" if path else node.name
                    stack.append((child, depth + 1, node.subgraph))

    def nesting_depth(self) -> int:
        """Levels of this proposal that actually contain nodes.

        0 for an empty proposal, 1 for a flat one, 2 when a node expands into a
        subgraph that has nodes of its own. An empty nested subgraph adds
        nothing, because it authorises nothing.
        """
        deepest = 0
        for _path, depth, sub in self.scopes():
            if sub.nodes:
                deepest = max(deepest, depth + 1)
        return deepest

    def node_count(self) -> int:
        return sum(len(sub.nodes) for _path, _depth, sub in self.scopes())

    def fingerprint(self) -> str:
        """A content hash of this exact proposal, for "what ran is what was admitted".

        Identity, not semantic equality: reordering `args` keys or changing the
        rationale changes the fingerprint, because the recorded decision was
        made about these bytes.
        """
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]


ProposedNode.model_rebuild()


# -- the planner node ---------------------------------------------------------

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are the planner inside a GraphARC graph.\n"
    "You do not act. You call no tools and you run nothing. Your only output is "
    "a proposal: nodes, and the edges between them.\n"
    "Every node has a `name` (unique within the proposal) and a `kind` taken "
    "from the catalog below. A kind that is not in the catalog is rejected by "
    "the admission checker and nothing runs.\n"
    "The checker judges nodes and edges by `kind`, never by the `name` you "
    "choose: renaming a node does not change what it is permitted to do, so "
    "re-proposing a refused node under a different name is a wasted turn.\n"
    f"Edge endpoints must be nodes you proposed, or the literals {START!r} "
    f"(graph entry) and {END!r} (graph exit).\n"
    "Leave `subgraph` unset on every node.\n"
    "Propose no nodes at all when there is no further work to do.\n"
    "Admission is deterministic code, not a conversation: arguing with a "
    "rejection does not change it. Re-propose something that satisfies it."
)

_NO_CATALOG = "(no catalog supplied; the admission registry decides what is allowed)"


def _catalog_text(catalog: Mapping[str, str] | Sequence[str] | None) -> str:
    if not catalog:
        return _NO_CATALOG
    if isinstance(catalog, Mapping):
        items = sorted(catalog.items())
    else:
        items = sorted((str(k), "") for k in catalog)
    return "\n".join(f"- {name}: {desc}" if desc else f"- {name}" for name, desc in items)


def _message_text(message: BaseMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = message.content
    return content if isinstance(content, str) else str(content)


def _first_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    return f"{first['msg']}" + (f" (at {where})" if where else "")


class PlanningOutcome(BaseModel):
    """What one planning turn produced — a proposal, or the reason there isn't one."""

    model_config = ConfigDict(frozen=True)

    proposal: Subgraph | None = None
    error: str = ""
    raw: str = ""
    # True when the model's own structured-output path produced the object,
    # False when it was recovered from text. Recorded because the two paths
    # have different failure modes and an audit should not have to guess.
    structured: bool = False
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.proposal is not None


class PlannerNode:
    """A model turned into a proposal, shaped as a GraphARC node.

    Call it directly (`node(state, ctx)`) or register it::

        planner = PlannerNode(model, catalog=registry.catalog())
        g.add_node("planner", planner, writes=planner.writes)

    **It cannot execute what it proposes**, and that is structural rather than
    promised: the only callables it holds are the chat model and the caller's
    own `prompt_fn` state reader. It is given no node registry, no harness and
    no graph, so a proposed node has no body here to run. Admission is a
    separate call the caller makes, and a `NodeSpec.factory` — the only place a
    node's body ever appears — lives on the far side of it.

    Structured output is used when the backend implements it, with the raw
    message kept (`include_raw=True`) so the turn's tokens are still charged and
    a parse failure can fall back to tolerant JSON extraction. Backends without
    it — the Claude CLI adapter — go straight to the text path. Neither path
    raises on a bad reply: an unusable answer comes back as a `PlanningOutcome`
    carrying the reason, which is what lets a caller feed it back for a retry.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str = "planner",
        catalog: Mapping[str, str] | Sequence[str] | None = None,
        system_prompt: str = DEFAULT_PLANNER_SYSTEM_PROMPT,
        instructions: str = "",
        task_field: str = "task",
        proposal_field: str = "proposal",
        error_field: str | None = None,
        prompt_fn: Callable[[Any], str] | None = None,
        trace: TraceRecorder | None = None,
        origin: str | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.catalog = catalog
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.task_field = task_field
        self.proposal_field = proposal_field
        self.error_field = error_field
        self.prompt_fn = prompt_fn
        self.trace = trace
        self.origin = origin or f"planner:{name}"
        self._structured_cache: Any = _UNSET

    @property
    def writes(self) -> set[str]:
        fields = {self.proposal_field}
        if self.error_field:
            fields.add(self.error_field)
        return fields

    def __call__(self, state: Any, ctx: RunContext) -> dict[str, Any]:
        outcome = self.propose(self.observe(state), ctx)
        update: dict[str, Any] = {self.proposal_field: outcome.proposal}
        if self.error_field:
            update[self.error_field] = outcome.error
        return update

    def observe(self, state: Any) -> str:
        """State -> task text. Override via `prompt_fn` for anything richer."""
        if self.prompt_fn is not None:
            return self.prompt_fn(state)
        value = state.get(self.task_field) if isinstance(state, dict) else None
        if value is None:
            value = getattr(state, self.task_field, None)
        if value is None:
            raise PlannerConfigError(
                f"planner {self.name!r} found no task in state field "
                f"{self.task_field!r}; set task_field or pass prompt_fn"
            )
        return str(value)

    def propose(
        self, task: str, ctx: RunContext | None = None, *, feedback: str = ""
    ) -> PlanningOutcome:
        """One planning turn. `feedback` is an `AdmissionResult.feedback()` to replan from.

        Never raises on a bad model reply — only a `BudgetExceeded`, which is
        the run's hard ceiling and not this turn's business, is allowed through.
        """
        if ctx is None:
            ctx = RunContext(
                run_id=uuid.uuid4().hex[:12], graph=self.name, meter=BudgetMeter(Budget())
            )
        messages = self._messages(task, feedback)
        runnable, structured = self._runnable()

        started = time.perf_counter()
        raw_message: BaseMessage | None = None
        proposal: Subgraph | None = None
        error = ""
        try:
            if structured:
                envelope = runnable.invoke(messages)
                raw_message = envelope.get("raw") if isinstance(envelope, dict) else None
                parsed = envelope.get("parsed") if isinstance(envelope, dict) else envelope
                parse_error = envelope.get("parsing_error") if isinstance(envelope, dict) else None
                if isinstance(parsed, Subgraph):
                    proposal = parsed
                elif parsed is not None:
                    proposal = Subgraph.model_validate(parsed)
                elif parse_error is not None:
                    error = f"structured output failed to parse: {parse_error!r}"
            else:
                raw_message = self.model.invoke(messages)
        except BudgetExceeded:
            raise
        except ValidationError as exc:
            error = f"proposal did not validate: {_first_validation_error(exc)}"
        except Exception as exc:  # noqa: BLE001 — a bad turn is an outcome, not a crash
            error = f"planner model call failed: {exc!r}"

        raw_text = _message_text(raw_message) if raw_message is not None else ""
        tokens = self._charge(ctx, raw_message)
        if proposal is None and raw_text:
            proposal, recovered_error = self._from_text(raw_text)
            error = error or recovered_error
        if proposal is None and not error:
            error = "the planner produced no proposal and no parseable reply"
        if proposal is not None:
            error = ""
            proposal = self._stamp(proposal)

        self._emit(
            ctx,
            duration_ms=(time.perf_counter() - started) * 1000,
            tokens=tokens,
            state_delta={
                "structured": structured,
                "nodes": proposal.node_count() if proposal else 0,
                "edges": len(proposal.edges) if proposal else 0,
                "proposal_id": proposal.proposal_id if proposal else "",
            },
            error=error or None,
        )
        return PlanningOutcome(
            proposal=proposal, error=error, raw=raw_text, structured=structured, tokens=tokens
        )

    # -- internals ------------------------------------------------------------

    def _messages(self, task: str, feedback: str) -> list[BaseMessage]:
        system = f"{self.system_prompt}\n\nAvailable node kinds:\n{_catalog_text(self.catalog)}"
        if self.instructions:
            system = f"{system}\n\n{self.instructions}"
        messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=task)]
        if feedback:
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous proposal was not admitted. The admission "
                        "checker reported:\n\n"
                        f"{feedback}\n\n"
                        "Propose again, fixing every line above."
                    )
                )
            )
        return messages

    def _runnable(self) -> tuple[Any, bool]:
        """The structured-output runnable and whether it is one, computed once."""
        if self._structured_cache is _UNSET:
            try:
                self._structured_cache = self.model.with_structured_output(
                    Subgraph, include_raw=True
                )
            except (NotImplementedError, ValueError, TypeError):
                # `with_structured_output` raises NotImplementedError on backends
                # without tool calling (the Claude CLI adapter); the other two
                # cover a backend that rejects the schema outright.
                self._structured_cache = None
        runnable = self._structured_cache
        return (runnable, True) if runnable is not None else (self.model, False)

    def _from_text(self, text: str) -> tuple[Subgraph | None, str]:
        data = extract_json(text)
        if data is None:
            return None, "no JSON object found in the planner's reply"
        try:
            return Subgraph.model_validate(data), ""
        except ValidationError as exc:
            return None, f"proposal did not validate: {_first_validation_error(exc)}"

    def _stamp(self, proposal: Subgraph) -> Subgraph:
        """Re-issue the proposal under this planner's identity.

        A model may put anything in `proposal_id` and `origin`; provenance that
        the proposer wrote about itself is not provenance. Rebuilding through
        the constructor also re-validates whatever the structured-output path
        handed back.
        """
        return Subgraph(
            nodes=proposal.nodes,
            edges=proposal.edges,
            rationale=proposal.rationale,
            origin=self.origin,
        )

    def _charge(self, ctx: RunContext, message: BaseMessage | None) -> int:
        """Charge the turn's reported usage, naming the call so it counts once.

        The runtime's usage callback already meters model calls made inside a
        node; naming the message lets the meter recognise this as a re-report
        rather than a second spend. Outside a graph nothing metered the call and
        the charge lands, so a standalone planner still meters itself.
        """
        usage = getattr(message, "usage_metadata", None) or {}
        total = int(usage.get("total_tokens") or 0)
        if total:
            ctx.meter.charge_tokens(total, source=message)
        return total

    def _emit(self, ctx: RunContext, **fields: Any) -> None:
        # Phase "plan", not "end": `observe.metrics` sums tokens and counts node
        # executions from "end" events, so reusing it inside a node would
        # double-count what the node wrapper already reported.
        if self.trace is None:
            return
        self.trace.event(
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            attempt=ctx.attempt,
            graph=ctx.graph,
            node=f"{self.name}:plan",
            phase="plan",
            step=ctx.next_step(),
            **fields,
        )


__all__ = [
    "DEFAULT_PLANNER_SYSTEM_PROMPT",
    "PlannerConfigError",
    "PlannerNode",
    "PlanningOutcome",
    "ProposedEdge",
    "ProposedNode",
    "Subgraph",
]
