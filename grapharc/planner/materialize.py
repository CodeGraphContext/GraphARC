"""Materialisation: an admitted proposal becomes a runnable graph, and nothing else does.

`admission` authorises a *shape*. This module is the only thing that turns one
into work, and it is written so that the authorisation cannot be skipped: the
entry point takes the `AdmissionResult` first and the `Subgraph` second, refuses
a result that is not `ADMITTED`, and refuses a proposal whose content hash is
not the one the result recorded. "What ran is what was admitted" is therefore a
checked equality rather than a convention — handing in a second, edited proposal
alongside a genuine approval fails the same way an outright rejection does.

**Where a node body comes from.** Only from the `NodeRegistry`. A `ProposedNode`
carries a `name` (which instance) and a `kind` (what it is), and the registry's
`NodeSpec.factory` — operator code, registered before any planning happened — is
the single source of every body; this module calls it once per proposed instance
with a `NodeBuild` describing what to build. There is no body-shaped field for a
proposal to fill: `ProposedNode` forbids extra fields, and `args`, the one
free-form field, is JSON a model wrote. A callable put there in-process cannot
even be hashed, so such a proposal never yields the `AdmissionResult` this
module demands.

**What a planner's `args` can reach.** Nothing, by default. Admission states
plainly that it does not inspect `ProposedNode.args`, so forwarding them is
opt-in here: with `forward_args=False` (the default) `NodeBuild.args` is empty
whatever the proposal said. `forward_args=True` hands the raw dict to your
factory unchecked — no gate has looked at it, and a factory that pulls a
callable out of it and runs it has re-opened the boundary this module exists to
hold.

**Built through the kernel, not around it.** The graph is assembled with
`GraphARC.add_node` / `add_edge` / `compile`, so every promise the kernel makes
applies unchanged: declared writes, typed state, the per-run budget meter,
deep-copy isolation of node inputs, checked routing and trace events. There is
no second execution path here.

**Topology, precisely.** Edges are wired as static edges, so fan-out is edge
multiplicity out of one node and a join is edge multiplicity into one, both with
LangGraph's super-step semantics. Conditional routing has no representation in a
proposal on purpose — `ProposedEdge` carries no predicate, because a predicate
authored by a model is model prose steering an edge. Dynamic routing is still
available to the *bodies*, which are operator code: a body may return
`Command(goto=...)`, and this module confines it to a destination the proposal
declared an edge to (or `END`), raising `UnadmittedTransition` otherwise. So the
admitted edge set bounds dynamic routing, not merely the admitted node set.

What is refused rather than guessed at: a node whose kind has left the registry
or carries no factory; a nested `ProposedNode.subgraph` (there is no
sub-materialisation here); an edge endpoint that is neither a proposed node nor
a sentinel — including a live node the checker allowed via `known_nodes`, since
this builds a standalone graph and has nothing to attach to; a proposal with no
edge out of `START`; a proposed node `START` cannot reach, which would otherwise
be admitted and then silently never run; and an empty proposal, which authorises
nothing and so has nothing to build.

**Whose boundary this is.** "The authorisation cannot be skipped" is a claim
about the *planner*, which emits JSON and has no channel to construct a Python
object. It is not a claim of unforgeability: an `AdmissionResult` is ordinary
Pydantic data, so a caller who hand-builds one with a matching `proposal_id` and
`fingerprint` will materialise and run anything, and no signature check here
would help — the caller already has the interpreter. The line this module holds
is between the planner and the operator, not between the operator and itself.

Four limits worth stating rather than implying. `UnadmittedTransition` covers a
`Command` a body returns; it says nothing about what that body does *inside*
itself — calling another graph, spawning a thread — which is the tool and
permission planes' business, not this one's. A `Command(goto=…)` is *additive*
to the static edges the proposal declared rather than a replacement for them, so
a confined body cannot use `goto` to skip an admitted successor: routing to
`END` alongside a declared edge still runs the declared edge. A kind's
`worst_case` from admission is an estimate the operator wrote; the meter passed
to `invoke()` is what actually stops the run. And a proposal that is *cyclic* —
which needs a checker built with `require_acyclic=False` — bounded only by a
default `Budget` will run to LangGraph's recursion limit, on the order of ten
thousand node executions, before it stops. That is bounded rather than a hang,
but it is not a bound anyone chose: give such a run an explicit budget.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict, Field

from grapharc.observe.trace import TraceRecorder
from grapharc.planner.admission import AdmissionResult, NodeRegistry
from grapharc.planner.proposal import ProposedNode, Subgraph
from grapharc.runtime.budget import Budget
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC, _is_async_callable


class MaterializationError(Exception):
    """A proposal could not be turned into a graph.

    Nothing was compiled and nothing ran: the structural refusals all happen
    before the first node is registered, and a builder left half-populated by a
    failing factory is local to the failed call and is discarded with it.
    """


class NotAdmitted(MaterializationError):
    """Materialisation was attempted without a successful, matching `AdmissionResult`.

    Raised for all three ways the authorisation can be absent: the wrong type in
    the first argument, a result whose status is not `ADMITTED`, and a result
    that authorised a *different* proposal than the one handed in.
    """


class UnadmittedTransition(Exception):
    """A node body routed somewhere the proposal did not declare an edge to.

    Raised inside the run, from the materialised node's wrapper, before the
    kernel resolves the destination — so the transition does not happen and the
    node's writes never land.
    """


class NodeBuild(BaseModel):
    """What a `NodeSpec.factory` is told about the one instance it is building.

    `args` is the planner's own dictionary and is empty unless the materialiser
    was built with `forward_args=True`. Nothing has validated it: admission
    authorises a *kind*, never that kind's arguments.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    args: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    proposal_id: str
    fingerprint: str


def _confine(name: str, body: Callable[..., Any], allowed: frozenset[str]) -> Callable[..., Any]:
    """Wrap `body` so a dynamic `goto` cannot leave the admitted topology.

    The wrapper keeps the body's arity and its sync/async nature, because the
    kernel reads both off the callable it is handed: two parameters mean "give
    me the RunContext", and a coroutine function selects the async node path.

    A non-string, non-`Send` `goto` sequence is materialised into a tuple and put
    back on the command, so a one-shot iterable is not consumed by this check and
    then delivered empty to the kernel. A `goto` this cannot iterate is passed
    through untouched for `GraphARC._check_goto` to reject with its own message.
    """

    def guard(result: Any) -> Any:
        if not isinstance(result, Command) or not result.goto:
            return result
        goto = result.goto
        single = isinstance(goto, (str, Send))
        if single:
            targets: tuple[Any, ...] = (goto,)
        else:
            try:
                targets = tuple(goto)
            except TypeError:
                return result
        for target in targets:
            destination = target.node if isinstance(target, Send) else target
            if isinstance(destination, str) and (destination == END or destination in allowed):
                continue
            raise UnadmittedTransition(
                f"node {name!r} routed to {destination!r}, which the admitted proposal "
                f"declared no edge to. Admitted destinations for {name!r}: "
                f"{', '.join([*sorted(allowed - {END}), 'END'])}. A body may route "
                "dynamically only along an edge the gate authorised"
            )
        return result if single else replace(result, goto=targets)

    wants_ctx = len(inspect.signature(body).parameters) >= 2

    if _is_async_callable(body):
        if wants_ctx:

            async def aconfined_ctx(state: Any, ctx: Any) -> Any:
                return guard(await body(state, ctx))

            return aconfined_ctx

        async def aconfined(state: Any) -> Any:
            return guard(await body(state))

        return aconfined

    if wants_ctx:

        def confined_ctx(state: Any, ctx: Any) -> Any:
            return guard(body(state, ctx))

        return confined_ctx

    def confined(state: Any) -> Any:
        return guard(body(state))

    return confined


class Materializer:
    """Turns an admitted proposal into a `CompiledGraphARC`. Nothing else here runs.

    Constructed once with the things an *operator* owns — the registry the
    bodies come from, the state schema they read and write, the write table, the
    trace recorder, the default budget — and then called per admitted proposal.
    None of those can be influenced by a proposal.
    """

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        state_schema: type[BaseModel],
        name: str = "plan",
        writes: Mapping[str, Iterable[str]] | None = None,
        trace: TraceRecorder | None = None,
        budget: Budget | None = None,
        dag: bool = False,
        forward_args: bool = False,
    ) -> None:
        self.registry = registry
        self.state_schema = state_schema
        self.name = name
        # kind -> the state fields instances of that kind may write. An entry
        # here wins over a `writes` attribute on the body; both are operator
        # declarations, and a kind nobody declared writes for may write nothing.
        self.writes: dict[str, frozenset[str]] = {
            str(kind): frozenset(fields) for kind, fields in (writes or {}).items()
        }
        self.trace = trace
        self.budget = budget
        self.dag = dag
        self.forward_args = forward_args

    def materialize(
        self,
        admitted: AdmissionResult,
        proposal: Subgraph,
        *,
        checkpointer: Any = None,
    ) -> CompiledGraphARC:
        """Build the graph `admitted` authorised. Raises rather than improvising.

        `admitted` comes first because it is the authorisation: there is no
        overload that takes a bare `Subgraph`, and a result that did not admit
        *this* proposal is refused by fingerprint.
        """
        self._require_admission(admitted, proposal)
        self._require_buildable(proposal)

        graph = GraphARC(
            self.state_schema,
            name=f"{self.name}:{proposal.proposal_id}",
            dag=self.dag,
            budget=self.budget,
            trace=self.trace,
        )
        outgoing = self._outgoing(proposal)
        # Assembly is wrapped so that *nothing* leaves this method as a foreign
        # exception type. The kernel and LangGraph below it both raise their own
        # classes — `WritePermissionError` for a declared write the schema has no
        # field for, a bare `ValueError` for a node name LangGraph reserves — and
        # a caller that catches `MaterializationError` (as `GovernedLoop` does)
        # would otherwise see those escape and turn a recorded stop into a crash.
        # The checks above catch the cases worth a specific message; this catches
        # the rest, including whatever the next LangGraph release decides to
        # reject.
        try:
            for node in proposal.nodes:
                body = self._body(node, proposal)
                graph.add_node(
                    node.name,
                    _confine(node.name, body, outgoing.get(node.name, frozenset())),
                    writes=self._writes_for(node, body),
                )
            for edge in proposal.edges:
                graph.add_edge(edge.source, edge.target)
            return graph.compile(checkpointer=checkpointer)
        except MaterializationError:
            raise
        except Exception as exc:
            raise MaterializationError(
                f"proposal {proposal.proposal_id} was admitted but could not be "
                f"assembled into a graph: {type(exc).__name__}: {exc}"
            ) from exc

    # -- the refusals ---------------------------------------------------------

    def _require_admission(self, admitted: AdmissionResult, proposal: Subgraph) -> None:
        if not isinstance(admitted, AdmissionResult):
            raise NotAdmitted(
                f"materialize() takes the AdmissionResult first, got "
                f"{type(admitted).__name__}; a proposal on its own carries no "
                "authorisation and there is no entry point that accepts one"
            )
        # Exact type, not `isinstance`. Every way of matching a proposal to its
        # authorisation ends in a method call on the proposal, and a subclass can
        # override any of them: overriding `fingerprint()` was the first attack,
        # and overriding the `model_dump_json()` that `fingerprint()` calls
        # defeats an unbound `Subgraph.fingerprint(proposal)` just as well. There
        # is no method here a subclass cannot reach, so the type is refused
        # instead of the methods being hardened one at a time. A planner's reply
        # is parsed into exactly this class, so nothing legitimate is lost.
        if type(proposal) is not Subgraph:
            raise NotAdmitted(
                f"materialize() takes a Subgraph second, got "
                f"{type(proposal).__name__}. A subclass is refused rather than "
                "trusted: the fingerprint match is computed by calling methods on "
                "this object, and a subclass can override them to report an "
                "admitted proposal's hash while carrying different nodes"
            )
        if not admitted.admitted:
            raise NotAdmitted(
                f"proposal {admitted.proposal_id} was not admitted "
                f"(status: {admitted.status.value}); nothing may be built from it. "
                f"{admitted.feedback()}"
            )
        # `Subgraph.fingerprint(proposal)`, not `proposal.fingerprint()`: the
        # bound call is virtual, and a subclass that overrides it to return the
        # hash of some *other*, admitted proposal passes this check while
        # carrying different nodes — verified by writing that subclass. The
        # unbound call hashes what this object actually contains. It does not
        # make the result unforgeable (see the trust-boundary note in the module
        # docstring); it removes the cheapest way to lie about the proposal.
        actual = Subgraph.fingerprint(proposal)
        if admitted.fingerprint != actual or admitted.proposal_id != proposal.proposal_id:
            raise NotAdmitted(
                f"this AdmissionResult authorised proposal {admitted.proposal_id} "
                f"(fingerprint {admitted.fingerprint}), not proposal "
                f"{proposal.proposal_id} (fingerprint {actual}); what runs must be "
                "what was admitted"
            )

    def _require_buildable(self, proposal: Subgraph) -> None:
        """Structural refusals, all of them before a single node is registered."""
        if not proposal.nodes:
            raise MaterializationError(
                f"proposal {proposal.proposal_id} declares no nodes; it authorises "
                "nothing and there is nothing to build. An empty proposal is a "
                "planner saying 'no further work', which is a caller's decision to "
                "act on, not a graph"
            )
        nested = [node.name for node in proposal.nodes if node.subgraph is not None]
        if nested:
            raise MaterializationError(
                f"nodes {sorted(nested)} carry a nested subgraph; this materialiser "
                "builds one flat graph and has no sub-materialisation step, so the "
                "inner proposal would be silently dropped. Propose the inner work in "
                "a later round"
            )

        names = proposal.node_names()
        for edge in proposal.edges:
            for role, endpoint in (("source", edge.source), ("target", edge.target)):
                if endpoint in (START, END) or endpoint in names:
                    continue
                raise MaterializationError(
                    f"edge {edge.render()!r} names {endpoint!r} as its {role}, which "
                    "is not a node of this proposal; materialisation builds a "
                    "standalone graph, so an edge to a node that lives in some other "
                    "graph has nothing to attach to"
                )
        if not any(edge.source == START for edge in proposal.edges):
            raise MaterializationError(
                f"proposal {proposal.proposal_id} has no edge out of START, so the "
                f"graph has no entry point and none of {sorted(names)} could run"
            )
        unreachable = sorted(names - self._reachable(proposal))
        if unreachable:
            raise MaterializationError(
                f"nodes {unreachable} cannot be reached from START; they were "
                "admitted and would never run, which is a proposal that does not "
                "mean what it says"
            )
        self._require_declarable_writes(proposal)

    def _require_declarable_writes(self, proposal: Subgraph) -> None:
        """The writes *table* is checked here; a body's own `writes` cannot be.

        An operator table entry naming a field the state schema does not have is
        a typo in operator code, and catching it at registration time would send
        the kernel's `WritePermissionError` out of `materialize()` — a different
        exception class than the one this module documents, and one
        `GovernedLoop` does not catch. Checking it here keeps the promise in
        `MaterializationError`'s docstring literally true: the refusal happens
        before the first node is registered.

        Only the table is reachable this early. A body's `writes` attribute is
        not known until its factory has run, so that case is caught by the
        wrapper in `materialize()` instead.
        """
        fields = set(self.state_schema.model_fields)
        kinds = {node.kind for node in proposal.nodes}
        for kind in sorted(kinds & set(self.writes)):
            unknown = sorted(self.writes[kind] - fields)
            if unknown:
                raise MaterializationError(
                    f"the writes table entry for kind {kind!r} names "
                    f"{unknown}, which {self.state_schema.__name__} has no field "
                    f"for; declared writes name state fields, and this is an "
                    f"operator typo rather than anything the proposal did"
                )

    @staticmethod
    def _reachable(proposal: Subgraph) -> set[str]:
        adjacency: dict[str, list[str]] = {}
        for edge in proposal.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        seen: set[str] = set()
        stack: list[str] = [START]
        while stack:
            for target in adjacency.get(stack.pop(), ()):
                if target == END or target in seen:
                    continue
                seen.add(target)
                stack.append(target)
        return seen

    @staticmethod
    def _outgoing(proposal: Subgraph) -> dict[str, frozenset[str]]:
        """node -> the destinations the proposal declared an edge to."""
        targets: dict[str, set[str]] = {}
        for edge in proposal.edges:
            targets.setdefault(edge.source, set()).add(edge.target)
        return {source: frozenset(dests) for source, dests in targets.items()}

    # -- bodies ---------------------------------------------------------------

    def _body(self, node: ProposedNode, proposal: Subgraph) -> Callable[..., Any]:
        """The callable for one proposed instance, from the registry and nowhere else."""
        spec = self.registry.get(node.kind)
        if spec is None:
            raise MaterializationError(
                f"node {node.name!r} has kind {node.kind!r}, which is not in the node "
                f"registry; admission passed, so the registry changed underneath the "
                f"decision. Allowed kinds: {', '.join(self.registry.names()) or '(none)'}"
            )
        if spec.factory is None:
            raise MaterializationError(
                f"kind {node.kind!r} is registered but carries no factory, so there is "
                "no body for it; the registry is the only place a node body may come "
                "from, and a proposal cannot supply one"
            )
        build = NodeBuild(
            name=node.name,
            kind=node.kind,
            args=dict(node.args) if self.forward_args else {},
            note=node.note,
            proposal_id=proposal.proposal_id,
            fingerprint=proposal.fingerprint(),
        )
        try:
            body = spec.factory(build)
        except Exception as exc:
            raise MaterializationError(
                f"the factory for kind {node.kind!r} raised while building node "
                f"{node.name!r}: {exc!r}"
            ) from exc
        if not callable(body):
            raise MaterializationError(
                f"the factory for kind {node.kind!r} returned {type(body).__name__}; a "
                "factory is a builder and must return the node's callable body"
            )
        return body

    def _writes_for(self, node: ProposedNode, body: Callable[..., Any]) -> frozenset[str]:
        """The state fields this instance may write — an operator's answer, never a planner's."""
        declared: Any = self.writes.get(node.kind)
        source = f"the writes table entry for kind {node.kind!r}"
        if declared is None:
            declared = getattr(body, "writes", None)
            source = f"the `writes` attribute of the body built for node {node.name!r}"
        if declared is None:
            # Nobody said, so nothing is granted: the node runs and the first
            # field it tries to write raises WritePermissionError in the kernel.
            return frozenset()
        if isinstance(declared, str) or not isinstance(declared, (set, frozenset, list, tuple)):
            raise MaterializationError(
                f"{source} is {type(declared).__name__}; declared writes must be a "
                "collection of state field names"
            )
        bad = sorted(repr(field) for field in declared if not isinstance(field, str))
        if bad:
            raise MaterializationError(f"{source} contains non-string field names: {bad}")
        return frozenset(declared)


__all__ = [
    "MaterializationError",
    "Materializer",
    "NodeBuild",
    "NotAdmitted",
    "UnadmittedTransition",
]
