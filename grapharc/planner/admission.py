"""Admission: the deterministic gate between proposing work and doing it.

No model is involved here and none may be. A proposal arrives as data
(`grapharc.planner.proposal.Subgraph`) and leaves as an `AdmissionResult` that
either authorises it or names, per failed check, exactly why not. That
asymmetry is the whole design: the planner may be as inventive as you like
because nothing it invents runs until code that cannot be argued with says yes.

Five checks, all of which run on every proposal so a planner gets the complete
list rather than the first complaint:

| Check | Question | Authority |
|---|---|---|
| REGISTRY | is every node's `kind` allowed, and does every edge endpoint exist? | `NodeRegistry` |
| POLICY | may each node's `kind` run, and each edge be taken? | `NodePolicy`, `EdgePolicy` |
| BUDGET | does the worst case fit what is *left*? | `RemainingBudget` |
| DEPTH | is the nesting within the limit? | `AdmissionLimits.max_depth` |
| ACYCLICITY | is the topology acyclic where that is required? | `AdmissionLimits` |

Both policy objects tier deny → ask → allow, and a checker given no
`NodePolicy` decides node kinds on registry membership alone, as it always did.

Four properties worth stating precisely, because each is a place where a gate
usually leaks:

- **Nothing runs during a check.** `NodeSpec.factory` is never called, no state
  is touched and no budget is charged; the meter is read, not written. An
  over-budget proposal is therefore refused before its first node exists, not
  after the run notices it overspent. The one side effect a check has is the
  trace line it writes.
- **A rejection is a returned object, not a fallback.** There is no "admit a
  reduced version", no truncation of an over-large fan-out and no downgrade of
  a denied edge. When a `TraceRecorder` is supplied, every decision — admitted
  and rejected alike — is also written to the trace as a `phase="admission"`
  event. That phase is deliberately not `"end"`: `observe.metrics` counts node
  executions from `"end"` events, and an admission decision is not one.
- **Costs come from the registry, never from the proposal.** A `ProposedNode`
  cannot declare what it will cost. The worst case is the sum of the registry's
  own `worst_case` figures for the kinds proposed, so a planner cannot buy
  admission by claiming to be cheap. Where the registry's numbers come from is
  the operator's problem; that they are the operator's numbers is the point.
- **Every decision keys on the registry `kind`, never on the instance `name`.**
  A `ProposedNode` carries both, and only one of them means anything: `kind` is
  the key of a thing an operator registered, `name` is a label the planner
  invented. So an `EdgeRule` denying `deploy` denies an edge into
  `ProposedNode(name="helper", kind="deploy")` exactly as it denies one into
  `ProposedNode(name="deploy")` — renaming an instance cannot launder it, and
  naming an instance after a permitted kind cannot borrow that kind's
  permission. Names still resolve *which* node an edge means and still appear in
  rejections so an audit can find the edge; no rule is ever matched against one.
  The corollary is that an endpoint whose kind the checker cannot determine — a
  live node the caller listed by name alone — is refused rather than assumed
  safe (`policy/unresolved_endpoint_kind`); pass `known_nodes` as a
  `{name: kind}` mapping to say what those nodes are.

**Disclosure is not enforcement.** `EdgePolicy.disclosure()` and
`NodePolicy.disclosure()` render a policy's deny rules as sentences a planner
can be *shown* before it proposes anything, which is what
`grapharc.planner.proposal.PlannerNode` puts in its system prompt. Nothing in
this module reads them back, no check consults them, and a model that ignores
them — or never saw them — is refused by byte-identical code. Telling a planner
the rule is a courtesy that saves rounds; the gate is what decides.

What this module does *not* do. It does not build a runnable graph — admission
authorises a shape, and turning one into work is `grapharc.planner.materialize`,
which takes the `AdmissionResult` this returns and refuses to build anything
else. It does not inspect `ProposedNode.args`: no rule here can constrain them,
so `Materializer(forward_args=True)`, which hands them to a factory unchecked,
has to gate them itself — admission authorises the *kind*, not its arguments. It
does not govern `name` either, in either direction: a name is not matched
against any rule, and is refused only for being unusable (the sentinels, the
orchestrator's `__`-prefixed namespace, a duplicate, or the charset). It does
not route approvals: an edge whose policy says `ask` yields
`AdmissionStatus.NEEDS_APPROVAL`, which is not admitted, and who gets asked is
someone else's job (ROADMAP §7.2). It cannot see the live graph's own edges, so
a cycle formed *between* a proposed node and an existing one via `known_nodes`
is outside what the acyclicity check can observe. And `parent_depth` is
supplied by the caller: the checker has no way to verify how deep the run
actually is, so a caller that always passes 0 has no recursion limit beyond the
nesting visible inside a single proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from fnmatch import fnmatch
from typing import Any

from pydantic import BaseModel, ConfigDict

from grapharc.harness.permissions import Decision
from grapharc.observe.trace import TraceRecorder
from grapharc.planner.proposal import ProposedEdge, Subgraph
from grapharc.runtime.budget import BudgetMeter
from grapharc.runtime.graph import END, START, RunContext

_SENTINELS = (START, END)


class Check(StrEnum):
    """The named gates. A rejection always carries one of these."""

    REGISTRY = "registry"
    POLICY = "policy"
    BUDGET = "budget"
    DEPTH = "depth"
    ACYCLICITY = "acyclicity"
    # Structural runnability: a graph with no entry, or with nodes nothing can
    # reach, is not a plan even when every kind and edge is permitted. The
    # materializer refused these anyway — but *after* admission had said yes,
    # so the loop learned "could not be built" instead of a rejection it could
    # replan against, and burned its execution-failure allowance discovering it.
    REACHABILITY = "reachability"


class AdmissionStatus(StrEnum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    # At least one edge resolved to `ask` and nothing resolved to `deny`. Not
    # admitted: a pending approval is a stop, not a slow yes.
    NEEDS_APPROVAL = "needs_approval"


class AdmissionRejected(Exception):
    """Raised by `AdmissionChecker.admit` when a proposal was not authorised."""

    def __init__(self, result: AdmissionResult) -> None:
        super().__init__(result.feedback())
        self.result = result


class Rejection(BaseModel):
    """One failed check, in a shape a planner can be handed back verbatim."""

    model_config = ConfigDict(frozen=True)

    check: Check
    # A stable machine-readable subcode. Match on this, not on `detail`.
    code: str
    # What the rejection is about: a node path, a rendered edge, a budget
    # dimension. Scoped by `/` for nodes inside a nested subgraph.
    subject: str
    detail: str
    remedy: str = ""

    def render(self) -> str:
        line = f"[{self.check.value}/{self.code}] {self.subject}: {self.detail}"
        return f"{line} {self.remedy}" if self.remedy else line


class CostEstimate(BaseModel):
    """Worst-case resource use, in the registry's terms.

    A dimension left at zero is simply unconstrained by that dimension's
    budget check. The registry declaring nothing is not evidence that the node
    costs nothing — it is evidence that nobody said.
    """

    model_config = ConfigDict(frozen=True)

    tokens: int = 0
    iterations: int = 0
    seconds: float = 0.0

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            tokens=self.tokens + other.tokens,
            iterations=self.iterations + other.iterations,
            seconds=self.seconds + other.seconds,
        )

    def __mul__(self, factor: int) -> CostEstimate:
        return CostEstimate(
            tokens=self.tokens * factor,
            iterations=self.iterations * factor,
            seconds=round(self.seconds * factor, 6),
        )

    def __rmul__(self, factor: int) -> CostEstimate:
        return self.__mul__(factor)


class RemainingBudget(BaseModel):
    """What is left, per dimension. `None` means unlimited, matching `Budget`."""

    model_config = ConfigDict(frozen=True)

    tokens: int | None = None
    iterations: int | None = None
    seconds: float | None = None

    @classmethod
    def from_meter(cls, meter: BudgetMeter) -> RemainingBudget:
        """Read a live run's headroom. Read-only: nothing is charged.

        Values go negative when a run has already overspent, which is correct —
        every non-empty proposal is then over budget.
        """
        budget = meter.budget
        return cls(
            tokens=None if budget.max_tokens is None else budget.max_tokens - meter.tokens,
            iterations=(
                None
                if budget.max_iterations is None
                else budget.max_iterations - meter.iterations
            ),
            seconds=meter.remaining_seconds(),
        )


class NodeSpec(BaseModel):
    """A node kind the operator allows, and what admitting one may cost.

    `factory` is carried so a later materialising step has somewhere to get the
    node from. **Admission never calls it** — being in the registry is a
    licence to be proposed, not an invitation to run.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str = ""
    worst_case: CostEstimate = CostEstimate(iterations=1)
    factory: Callable[..., Any] | None = None


class NodeRegistry:
    """The allowlist of node kinds. Absence is refusal; there is no wildcard.

    Mutable by default, because assembling one is ordinary start-up code. That
    mutability outlives start-up, though: a driver that says every round is
    checked "against the same registry" means the same *object*, and a node body
    — operator code, but code a long run gives many chances to execute — can
    call `register()` between rounds and widen the allowlist under the checker.
    `freeze()` closes that, and a run that plans repeatedly should use it.
    """

    def __init__(self, specs: Iterable[NodeSpec] = ()) -> None:
        self._specs: dict[str, NodeSpec] = {}
        self._frozen = False
        for spec in specs:
            self.register(spec)

    def register(self, spec: NodeSpec) -> NodeRegistry:
        if self._frozen:
            raise ValueError(
                f"node kind {spec.name!r} cannot be registered: this registry is "
                "frozen. The allowlist a decision was made against must not change "
                "underneath it"
            )
        if spec.name in self._specs:
            raise ValueError(f"node kind {spec.name!r} already registered")
        self._specs[spec.name] = spec
        return self

    def freeze(self) -> NodeRegistry:
        """Refuse further registration. Idempotent, and returns self for chaining."""
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(self, name: str) -> NodeSpec | None:
        return self._specs.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)

    def catalog(self) -> dict[str, str]:
        """kind -> description, in the shape `PlannerNode(catalog=...)` wants."""
        return {name: spec.description for name, spec in sorted(self._specs.items())}


class EdgeRule(BaseModel):
    """One deny/ask/allow rule over an edge, matched by fnmatch on both ends.

    `source` and `target` are patterns over **registry kinds** — plus the
    literal `START`/`END` sentinels, which no node may be named. They are never
    matched against a planner's instance name.

    `reason` is the operator's own words, carried from the policy document that
    compiled to this rule, exactly as `NodeRule.reason` is. A refusal quotes it
    so a planner reads *why* rather than only `edge_denied`, and
    `EdgePolicy.disclosure()` puts it in front of the model before the first
    round. A rule without one still refuses; nothing decides on this string.
    """

    model_config = ConfigDict(frozen=True)

    action: Decision
    source: str = "*"
    target: str = "*"
    reason: str = ""


class EdgePolicy(BaseModel):
    """Which transitions are permitted: deny → ask → allow, unmatched is deny.

    The tier ordering is `grapharc.harness.permissions.PermissionPolicy`'s,
    reused because it is the semantics that survived contact with reality: a
    broad deny beats a narrower allow, so there is no allowlist exception
    hiding inside a denial. This is a separate object over *edges* rather than
    tool names — it shares the `Decision` vocabulary and nothing else, so
    widening what an agent may call cannot widen what a planner may wire.

    The strings it decides over are **kinds**, resolved from the registry by
    `AdmissionChecker`, not the names a planner chose for its instances. A rule
    is a statement about what a node *is*, which is the only thing an operator
    is in a position to authorise.
    """

    model_config = ConfigDict(frozen=True)

    rules: tuple[EdgeRule, ...] = ()
    default: Decision = Decision.DENY

    def rule_for(self, source_kind: str, target_kind: str) -> EdgeRule | None:
        """The rule that decides this transition, or None when the default applies."""
        for tier in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            for rule in self.rules:
                if (
                    rule.action == tier
                    and fnmatch(source_kind, rule.source)
                    and fnmatch(target_kind, rule.target)
                ):
                    return rule
        return None

    def decide(self, source_kind: str, target_kind: str) -> Decision:
        """Decide one transition. Both arguments are kinds (or a sentinel)."""
        rule = self.rule_for(source_kind, target_kind)
        return self.default if rule is None else rule.action

    def disclosure(self) -> tuple[str, ...]:
        """The deny rules as sentences a planner can be shown before it proposes.

        **Disclosure, not enforcement.** Nothing reads this back: `decide` is
        the only thing that decides, and a planner handed these lines and
        ignoring them is refused exactly as one that never saw them. It exists
        because `edge_denied` on round three is a fact the model could have had
        on round one — the observed failure was a run that proposed an edge into
        a denied kind every round until the loop gave up, unable to infer "no
        edge may enter this, ever" from a check name.

        Deny rules only. An allow rule and the default say what is *permitted*,
        which the catalog and the structural rules already cover, and listing
        them would turn a short warning into a policy dump the model has to
        read past. `ask` is left out for a different reason: its remedy is to
        obtain approval, not to propose something else.
        """
        return _denial_lines(
            (_edge_subject(rule.source, rule.target), rule.reason)
            for rule in self.rules
            if rule.action is Decision.DENY
        )


class NodeRule(BaseModel):
    """One deny/ask/allow rule over a node kind, matched by fnmatch.

    `match` is a pattern over a **registry kind**, never over the instance name
    a planner chose — the same rule the rest of this module follows, so a denied
    kind cannot be renamed out of its denial.

    `reason` is the operator's own words, carried from the policy document that
    compiled to this rule so a refusal can quote why rather than only what.
    """

    model_config = ConfigDict(frozen=True)

    action: Decision
    match: str = "*"
    reason: str = ""


class NodePolicy(BaseModel):
    """Which node kinds may be proposed at all: deny → ask → allow, tiered.

    The counterpart of `EdgePolicy` for nodes, and deliberately a *second* gate
    rather than a replacement for `NodeRegistry`: the registry is the operator's
    allowlist of kinds that exist and what each costs, while this answers "may
    this kind run here" from a document that can be edited without touching the
    code that builds the registry. A kind has to pass both.

    Same semantics as `EdgePolicy` and `PermissionPolicy` — every `deny` is
    tried before every `ask` and every `ask` before every `allow`, first match
    within a tier wins, and an unmatched kind gets `default`. `default` is
    `deny`, so an empty `NodePolicy()` admits nothing; that is the same shape
    `EdgePolicy` has and the same reason: a policy object that governs nothing
    should have to be written down on purpose.
    """

    model_config = ConfigDict(frozen=True)

    rules: tuple[NodeRule, ...] = ()
    default: Decision = Decision.DENY

    def rule_for(self, kind: str) -> NodeRule | None:
        """The rule that decides `kind`, or None when the default applies."""
        for tier in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            for rule in self.rules:
                if rule.action == tier and fnmatch(kind, rule.match):
                    return rule
        return None

    def decide(self, kind: str) -> Decision:
        """Decide one node kind. The argument is a kind, never an instance name."""
        rule = self.rule_for(kind)
        return self.default if rule is None else rule.action

    def disclosure(self) -> tuple[str, ...]:
        """The deny rules as sentences a planner can be shown. See `EdgePolicy.disclosure`.

        A denied kind is worth stating for the same reason a denied edge is: the
        registry lists it as proposable — it is registered — and the document
        then forbids it, so the catalog alone reads as an invitation.
        """
        return _denial_lines(
            (_node_subject(rule.match), rule.reason)
            for rule in self.rules
            if rule.action is Decision.DENY
        )


class AdmissionLimits(BaseModel):
    """Structural limits, set by the operator and not by the proposal."""

    model_config = ConfigDict(frozen=True)

    # Total planning depth allowed: the `parent_depth` the caller reports plus
    # the nesting inside the proposal. 1 means flat proposals only — a planner
    # running inside an admitted subgraph reports parent_depth=1, so its
    # proposal lands at 2 and is refused. Raise it deliberately. The caller
    # owns `parent_depth`; this limit is only as honest as that number.
    max_depth: int = 1
    # Cycles are earned, here as in the kernel's `dag=True`: the default refuses
    # them and a proposal that genuinely needs one has to be admitted by a
    # checker configured to allow it.
    require_acyclic: bool = True
    # Whether every scope must have an entry from START and no unreachable
    # node — the two conditions `Materializer` enforces when it builds a
    # *standalone* graph. Off by default because admission is deliberately the
    # broader gate: a proposal may name `known_nodes` of a graph already
    # running, where the entry lives outside the proposal entirely. Any driver
    # that materializes standalone (every `GovernedLoop` does) should turn it
    # on, so a structurally unrunnable plan comes back as a rejection the
    # planner can act on rather than a build failure it cannot.
    require_entry: bool = False


class AdmissionResult(BaseModel):
    """The decision, with everything needed to audit or replan from it."""

    model_config = ConfigDict(frozen=True)

    status: AdmissionStatus
    proposal_id: str
    # Content hash of the exact proposal that was checked, so "what ran" can be
    # matched against "what was admitted" after the fact.
    fingerprint: str
    rejections: tuple[Rejection, ...] = ()
    checks_run: tuple[Check, ...] = ()
    worst_case: CostEstimate = CostEstimate()
    # False when a proposed kind was not in the registry, so its cost is
    # unknown and `worst_case` is a lower bound. Such a proposal is always
    # rejected by REGISTRY anyway; the flag exists so nobody reads the number
    # as complete.
    worst_case_complete: bool = True
    remaining: RemainingBudget = RemainingBudget()
    depth: int = 0
    node_count: int = 0

    @property
    def admitted(self) -> bool:
        return self.status is AdmissionStatus.ADMITTED

    @property
    def needs_approval(self) -> bool:
        return self.status is AdmissionStatus.NEEDS_APPROVAL

    def failed_checks(self) -> tuple[Check, ...]:
        """The distinct checks that produced a rejection, in gate order."""
        failed = {r.check for r in self.rejections}
        return tuple(c for c in Check if c in failed)

    def reasons(self, check: Check | None = None) -> tuple[Rejection, ...]:
        if check is None:
            return self.rejections
        return tuple(r for r in self.rejections if r.check is check)

    def feedback(self) -> str:
        """The rejection as text to hand straight back to a planner."""
        if self.admitted:
            return f"Proposal {self.proposal_id} was admitted."
        head = (
            f"Proposal {self.proposal_id} was not admitted "
            f"(status: {self.status.value}). Failed checks: "
            f"{', '.join(c.value for c in self.failed_checks())}."
        )
        return "\n".join([head, *(f"- {r.render()}" for r in self.rejections)])


class AdmissionChecker:
    """The gate. Deterministic, model-free, and the only thing that says yes."""

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        edge_policy: EdgePolicy,
        node_policy: NodePolicy | None = None,
        limits: AdmissionLimits | None = None,
        known_nodes: Iterable[str] | Mapping[str, str] = (),
        trace: TraceRecorder | None = None,
        name: str = "admission",
    ) -> None:
        self.registry = registry
        self.edge_policy = edge_policy
        # `None` is not "allow everything": it means no *document* governs node
        # kinds here, and the registry — an allowlist with no wildcard — remains
        # the only node gate, which is where it was before this check existed.
        # A `NodePolicy` supplied on purpose is consulted for every proposed
        # node, and an empty one denies every kind.
        self.node_policy = node_policy
        self.limits = limits or AdmissionLimits()
        # Nodes already live in the graph this proposal attaches to. Edges may
        # reference them; proposed nodes may not reuse their names.
        #
        # Pass a `{name: kind}` mapping to say what those live nodes *are*: the
        # policy decides over kinds, and the checker cannot see the live graph,
        # so a name listed without a kind is a node no rule can be evaluated
        # against. Edges touching one are refused (`unresolved_endpoint_kind`)
        # rather than waved through. These kinds come from the caller, not from
        # a proposal, and need not be registered — the registry is the list of
        # kinds a *planner* may propose, not of kinds that may already exist.
        self.known_kinds: dict[str, str] = (
            {str(name_): str(kind) for name_, kind in known_nodes.items() if kind}
            if isinstance(known_nodes, Mapping)
            else {}
        )
        self.known_nodes = frozenset(known_nodes)
        self.trace = trace
        self.name = name

    def check(
        self,
        proposal: Subgraph,
        *,
        meter: BudgetMeter | None = None,
        remaining: RemainingBudget | None = None,
        parent_depth: int = 0,
        ctx: RunContext | None = None,
    ) -> AdmissionResult:
        """Decide. Runs every check, charges nothing, executes nothing.

        `remaining` wins over `meter`; with neither, every budget dimension is
        unlimited — the same meaning `Budget()` has in the runtime.
        """
        if parent_depth < 0:
            raise ValueError(f"parent_depth must be >= 0, got {parent_depth}")
        if remaining is None:
            remaining = RemainingBudget() if meter is None else RemainingBudget.from_meter(meter)

        rejections: list[Rejection] = []
        rejections.extend(self._check_registry(proposal))
        rejections.extend(self._check_node_policy(proposal))
        rejections.extend(self._check_policy(proposal))
        worst_case, complete = self._worst_case(proposal)
        rejections.extend(self._check_budget(worst_case, complete, remaining))
        depth = parent_depth + proposal.nesting_depth()
        rejections.extend(self._check_depth(depth, parent_depth, proposal))
        checks_run = [Check.REGISTRY, Check.POLICY, Check.BUDGET, Check.DEPTH]
        if self.limits.require_entry:
            rejections.extend(self._check_reachability(proposal))
            checks_run.append(Check.REACHABILITY)
        if self.limits.require_acyclic:
            rejections.extend(self._check_acyclicity(proposal))
            checks_run.append(Check.ACYCLICITY)

        ordered = tuple(sorted(rejections, key=lambda r: (list(Check).index(r.check), r.subject)))
        result = AdmissionResult(
            status=_status_for(ordered),
            proposal_id=proposal.proposal_id,
            fingerprint=proposal.fingerprint(),
            rejections=ordered,
            checks_run=tuple(checks_run),
            worst_case=worst_case,
            worst_case_complete=complete,
            remaining=remaining,
            depth=depth,
            node_count=proposal.node_count(),
        )
        self._emit(result, proposal, ctx)
        return result

    def admit(self, proposal: Subgraph, **kwargs: Any) -> Subgraph:
        """`check`, but fail closed: return the proposal or raise `AdmissionRejected`.

        For callers whose next line would otherwise be `if not result.admitted:
        raise`. The rejection is still recorded on the trace first, and the
        result travels on the exception.
        """
        result = self.check(proposal, **kwargs)
        if not result.admitted:
            raise AdmissionRejected(result)
        return proposal

    # -- the checks -----------------------------------------------------------

    def _check_registry(self, proposal: Subgraph) -> list[Rejection]:
        allowed = self.registry.names()
        hint = f"allowed kinds: {', '.join(allowed) or '(none registered)'}"
        out: list[Rejection] = []
        for path, _depth, sub in proposal.scopes():
            names = sub.node_names()
            for node in sub.nodes:
                subject = _scoped(path, node.name)
                if node.kind not in self.registry:
                    out.append(
                        Rejection(
                            check=Check.REGISTRY,
                            code="unregistered_node",
                            subject=subject,
                            detail=f"kind {node.kind!r} is not in the node registry",
                            remedy=hint,
                        )
                    )
                if node.name in self.known_nodes:
                    out.append(
                        Rejection(
                            check=Check.REGISTRY,
                            code="name_collides_with_existing_node",
                            subject=subject,
                            detail=(
                                f"node name {node.name!r} already exists in the graph; "
                                "admitting it would rebind a node that is already running"
                            ),
                            remedy="propose a different name",
                        )
                    )
            for edge in sub.edges:
                out.extend(self._check_endpoints(path, edge, names))
        return out

    def _check_endpoints(
        self, path: str, edge: ProposedEdge, names: frozenset[str]
    ) -> list[Rejection]:
        out: list[Rejection] = []
        for role, endpoint in (("source", edge.source), ("target", edge.target)):
            if endpoint in _SENTINELS or endpoint in names or endpoint in self.known_nodes:
                continue
            out.append(
                Rejection(
                    check=Check.REGISTRY,
                    code="unknown_edge_endpoint",
                    subject=_scoped(path, edge.render()),
                    detail=(
                        f"{role} {endpoint!r} is neither a node proposed in this scope "
                        "nor a node already in the graph"
                    ),
                    remedy=f"propose {endpoint!r} as a node, or point the edge elsewhere",
                )
            )
        return out

    def _check_node_policy(self, proposal: Subgraph) -> list[Rejection]:
        """Decide every proposed node on its *kind*, when a node policy exists.

        Part of the POLICY check rather than a gate of its own: REGISTRY answers
        "does this kind exist and what does it cost", POLICY answers "is it
        permitted", and a planner replanning from feedback should read one
        answer to the second question whether the refusal was about a node or an
        edge. With no node policy configured this decides nothing — the registry
        is then the only node gate, as it was before.
        """
        if self.node_policy is None:
            return []
        out: list[Rejection] = []
        for path, _depth, sub in proposal.scopes():
            for node in sub.nodes:
                rule = self.node_policy.rule_for(node.kind)
                decision = self.node_policy.default if rule is None else rule.action
                if decision is Decision.ALLOW:
                    continue
                denied = decision is Decision.DENY
                verb = "denies" if denied else "requires approval for"
                # The operator's own words, when the rule carried any. A rule
                # without a `reason` still refuses; the detail then names the
                # kind and nothing more, which is the whole of what was decided.
                because = f": {rule.reason}" if rule is not None and rule.reason else ""
                out.append(
                    Rejection(
                        check=Check.POLICY,
                        code="node_denied" if denied else "node_needs_approval",
                        subject=_scoped(path, node.name),
                        detail=(
                            f"the node policy {verb} this kind: "
                            f"{_describe(node.name, node.kind)}{because}"
                        ),
                        remedy=(
                            "the decision is made on the registry kind, not the name you "
                            "chose: renaming the node will not change it — propose a "
                            "permitted kind"
                            if denied
                            else "obtain approval and re-submit"
                        ),
                    )
                )
        return out

    def _check_policy(self, proposal: Subgraph) -> list[Rejection]:
        """Decide every edge on the *kinds* of its endpoints.

        The planner picks an instance name; the operator authorised a kind. So
        the name is used only to work out which node an endpoint refers to, and
        the kind that node declared — which REGISTRY has independently required
        to be one the operator registered — is what a rule matches. An endpoint
        whose kind cannot be resolved is refused, because "no rule matched" must
        never be reachable by hiding what a node is.
        """
        out: list[Rejection] = []
        for path, _depth, sub in proposal.scopes():
            # A name declared here shadows a live node of the same name; that
            # collision is itself a REGISTRY rejection, so the proposal cannot
            # be admitted either way and the local declaration is the honest
            # reading of what the edge refers to.
            kinds = {node.name: node.kind for node in sub.nodes}
            for edge in sub.edges:
                subject = _scoped(path, edge.render())
                source_kind = self._kind_of(edge.source, kinds)
                target_kind = self._kind_of(edge.target, kinds)
                if source_kind is None or target_kind is None:
                    out.extend(
                        self._unresolved_endpoints(subject, edge, source_kind, target_kind)
                    )
                    continue
                rule = self.edge_policy.rule_for(source_kind, target_kind)
                decision = self.edge_policy.default if rule is None else rule.action
                if decision is Decision.ALLOW:
                    continue
                denied = decision is Decision.DENY
                transition = (
                    f"{_describe(edge.source, source_kind)} -> "
                    f"{_describe(edge.target, target_kind)}"
                )
                # The operator's own words, when the rule carried any — the same
                # courtesy `_check_node_policy` extends. A planner told only
                # `edge_denied` has to guess how wide the denial is; told "deploys
                # are the operator's decision" it can stop proposing one.
                because = f": {rule.reason}" if rule is not None and rule.reason else ""
                out.append(
                    Rejection(
                        check=Check.POLICY,
                        code="edge_denied" if denied else "edge_needs_approval",
                        subject=subject,
                        detail=(
                            f"the edge policy denies this transition: "
                            f"{transition}{because}"
                            if denied
                            else (
                                "the edge policy requires approval for this "
                                f"transition: {transition}{because}"
                            )
                        ),
                        remedy=(
                            "the decision is made on the registry kind, not the name you "
                            "chose: renaming the node will not change it — route through "
                            "a permitted kind"
                            if denied
                            else "obtain approval and re-submit"
                        ),
                    )
                )
        return out

    def _kind_of(self, endpoint: str, kinds: Mapping[str, str]) -> str | None:
        """The policy identity of an edge endpoint, or None when unknowable.

        A sentinel stands for itself — `START`/`END` are reserved words no node
        may be named, so they cannot be spoofed. Everything else is a node, and
        a node's identity is its registry kind.
        """
        if endpoint in _SENTINELS:
            return endpoint
        return kinds.get(endpoint) or self.known_kinds.get(endpoint)

    def _unresolved_endpoints(
        self,
        subject: str,
        edge: ProposedEdge,
        source_kind: str | None,
        target_kind: str | None,
    ) -> list[Rejection]:
        """Fail closed, per endpoint that could not be identified."""
        out: list[Rejection] = []
        unresolved = [
            (role, endpoint)
            for role, endpoint, kind in (
                ("source", edge.source, source_kind),
                ("target", edge.target, target_kind),
            )
            if kind is None
        ]
        for role, endpoint in unresolved:
            live = endpoint in self.known_nodes
            out.append(
                Rejection(
                    check=Check.POLICY,
                    code="unresolved_endpoint_kind",
                    subject=subject,
                    detail=(
                        f"{role} {endpoint!r} has no resolvable registry kind, so no rule "
                        "can be evaluated for it; the edge is refused rather than assumed "
                        "permitted"
                        + (
                            " (it is a live node the caller listed by name only)"
                            if live
                            else ""
                        )
                    ),
                    remedy=(
                        f"declare what it is: known_nodes={{{endpoint!r}: '<kind>'}}"
                        if live
                        else f"propose {endpoint!r} as a node in this scope"
                    ),
                )
            )
        return out

    def _worst_case(self, proposal: Subgraph) -> tuple[CostEstimate, bool]:
        total = CostEstimate()
        complete = True
        for _path, _depth, sub in proposal.scopes():
            max_step = len(sub.nodes) + 1
            step_set: dict[str, set[int]] = {node.name: set() for node in sub.nodes}
            step_set[START] = {0}
            for k in self.known_nodes:
                step_set[k] = {0}
            queue = [START, *[k for k in self.known_nodes if k in {e.source for e in sub.edges}]]
            while queue:
                curr = queue.pop(0)
                curr_steps = step_set.get(curr, set())
                for edge in sub.edges:
                    if edge.source == curr:
                        next_steps = {s + 1 for s in curr_steps if s + 1 <= max_step}
                        target = edge.target
                        if target in step_set:
                            added = next_steps - step_set[target]
                            if added:
                                step_set[target].update(added)
                                if target not in queue:
                                    queue.append(target)
            for node in sub.nodes:
                spec = self.registry.get(node.kind)
                if spec is None:
                    complete = False
                    continue
                multiplier = max(1, len(step_set.get(node.name, set())))
                total = total + (spec.worst_case * multiplier)
        return total, complete

    def _check_budget(
        self, worst_case: CostEstimate, complete: bool, remaining: RemainingBudget
    ) -> list[Rejection]:
        suffix = "" if complete else " (a lower bound: some proposed kinds are unregistered)"
        dimensions = (
            ("tokens", "over_token_budget", worst_case.tokens, remaining.tokens),
            ("iterations", "over_iteration_budget", worst_case.iterations, remaining.iterations),
            ("seconds", "over_time_budget", worst_case.seconds, remaining.seconds),
        )
        out: list[Rejection] = []
        for label, code, want, left in dimensions:
            if left is None or want <= left:
                continue
            out.append(
                Rejection(
                    check=Check.BUDGET,
                    code=code,
                    subject=label,
                    detail=(
                        f"worst case needs {want} {label} but only {left} remain{suffix}"
                    ),
                    remedy="propose fewer or cheaper nodes",
                )
            )
        return out

    def _check_depth(self, depth: int, parent_depth: int, proposal: Subgraph) -> list[Rejection]:
        if depth <= self.limits.max_depth:
            return []
        return [
            Rejection(
                check=Check.DEPTH,
                code="too_deep",
                subject=proposal.proposal_id,
                detail=(
                    f"depth {depth} exceeds max_depth {self.limits.max_depth} "
                    f"(admitted at depth {parent_depth}, proposal nests "
                    f"{proposal.nesting_depth()} level(s))"
                ),
                remedy="flatten the proposal, or propose the inner work in a later round",
            )
        ]

    def _check_reachability(self, proposal: Subgraph) -> list[Rejection]:
        """Every scope needs an entry, and every node needs a way in.

        Same two conditions `Materializer` enforces, checked here so a planner
        gets a rejection it can act on — with the remedy spelled out — instead
        of an admitted proposal that dies at build time.
        """
        out: list[Rejection] = []
        for path, _depth, sub in proposal.scopes():
            names = sub.node_names()
            if not names:
                continue
            if not any(edge.source == START for edge in sub.edges):
                out.append(
                    Rejection(
                        check=Check.REACHABILITY,
                        code="no_entry_edge",
                        subject=_scoped(path, START),
                        detail=(
                            "no edge leaves START, so the graph has no entry point "
                            f"and none of {sorted(names)} could run"
                        ),
                        remedy=f"add an edge from {START!r} to the first node to run",
                    )
                )
                continue
            unreachable = sorted(names - _reachable_from_start(sub))
            for name in unreachable:
                out.append(
                    Rejection(
                        check=Check.REACHABILITY,
                        code="unreachable_node",
                        subject=_scoped(path, name),
                        detail=(
                            "nothing leads to this node from START, so it would "
                            "never run — a proposal that does not mean what it says"
                        ),
                        remedy=(
                            f"add an edge from a node reachable from {START!r} to "
                            f"{name!r}, or drop the node"
                        ),
                    )
                )
        return out

    def _check_acyclicity(self, proposal: Subgraph) -> list[Rejection]:
        out: list[Rejection] = []
        for path, _depth, sub in proposal.scopes():
            cycle = _find_cycle(sub.node_names(), sub.edges)
            if cycle is None:
                continue
            out.append(
                Rejection(
                    check=Check.ACYCLICITY,
                    code="cycle",
                    subject=_scoped(path, " -> ".join(cycle)),
                    detail="this checker requires acyclic proposals and found a cycle",
                    remedy="break the cycle, or use a checker with require_acyclic=False",
                )
            )
        return out

    # -- recording ------------------------------------------------------------

    def _emit(
        self, result: AdmissionResult, proposal: Subgraph, ctx: RunContext | None
    ) -> None:
        """Write the decision to the trace. A rejection is an event, not a silence."""
        if self.trace is None:
            return
        self.trace.event(
            # Without a run context the decision still has to be findable, so the
            # proposal's own id stands in for the run id.
            run_id=ctx.run_id if ctx is not None else proposal.proposal_id,
            thread_id=ctx.thread_id if ctx is not None else None,
            attempt=ctx.attempt if ctx is not None else 1,
            graph=ctx.graph if ctx is not None else self.name,
            node=f"{self.name}:{proposal.proposal_id}",
            phase="admission",
            step=ctx.next_step() if ctx is not None else 0,
            state_delta={
                "status": result.status.value,
                "fingerprint": result.fingerprint,
                "origin": proposal.origin,
                "nodes": result.node_count,
                "depth": result.depth,
                "checks_run": [c.value for c in result.checks_run],
                "failed_checks": [c.value for c in result.failed_checks()],
                "worst_case": result.worst_case.model_dump(),
            },
            error=(
                None
                if result.admitted
                else "; ".join(f"{r.check.value}/{r.code}" for r in result.rejections)
            ),
        )


def _status_for(rejections: tuple[Rejection, ...]) -> AdmissionStatus:
    if not rejections:
        return AdmissionStatus.ADMITTED
    if all(r.code in ("edge_needs_approval", "node_needs_approval") for r in rejections):
        return AdmissionStatus.NEEDS_APPROVAL
    return AdmissionStatus.REJECTED


def _reachable_from_start(sub: Subgraph) -> set[str]:
    """Nodes a walk from START can arrive at, within one scope."""
    adjacency: dict[str, list[str]] = {}
    for edge in sub.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    seen: set[str] = set()
    stack = list(adjacency.get(START, ()))
    while stack:
        node = stack.pop()
        if node in seen or node == END:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return seen


def _scoped(path: str, subject: str) -> str:
    return f"{path}/{subject}" if path else subject


def _pattern_text(pattern: str) -> str:
    """A rule's pattern as prose: a bare kind is quoted, a glob is described as one."""
    return (
        f"kinds matching {pattern!r}"
        if any(char in pattern for char in "*?[")
        else repr(pattern)
    )


def _edge_subject(source: str, target: str) -> str:
    """What one edge deny rule is about, in the plural so a line reads as a warning."""
    if source == "*" and target == "*":
        return "all edges"
    if source == "*":
        return f"edges into {_pattern_text(target)}"
    if target == "*":
        return f"edges out of {_pattern_text(source)}"
    return f"edges from {_pattern_text(source)} to {_pattern_text(target)}"


def _node_subject(match: str) -> str:
    return "all node kinds" if match == "*" else f"nodes of kind {_pattern_text(match)}"


def _denial_lines(subjects: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """`(subject, reason)` pairs -> one line each, in rule order, without repeats.

    Two rules can render the same sentence — a document scoped per tenant is the
    ordinary way — and saying it twice would only cost the reader attention.
    """
    lines: list[str] = []
    for subject, reason in subjects:
        line = f"{subject} are denied by policy — do not propose them"
        if reason.strip():
            line = f"{line}: {reason.strip()}"
        if line not in lines:
            lines.append(line)
    return tuple(lines)


def _describe(endpoint: str, kind: str) -> str:
    """An endpoint as the rejection should name it: what it is, then what it is called.

    The kind leads because the kind is what the rule matched; the name follows
    so a planner (and an audit) can tell which edge is being talked about.
    """
    if endpoint in _SENTINELS:
        return endpoint
    return f"kind {kind!r} (proposed as {endpoint!r})"


def _find_cycle(names: frozenset[str], edges: tuple[ProposedEdge, ...]) -> list[str] | None:
    """The first cycle among `names`, or None. Iterative, so depth cannot crash it.

    Edges touching anything outside `names` (the START/END sentinels, a node in
    the live graph) are ignored: this can only speak about the topology the
    proposal itself declares.
    """
    adjacency: dict[str, list[str]] = {name: [] for name in names}
    for edge in edges:
        if edge.source in names and edge.target in names:
            adjacency[edge.source].append(edge.target)
    for targets in adjacency.values():
        targets.sort()

    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(names, white)
    for root in sorted(names):
        if colour[root] != white:
            continue
        colour[root] = grey
        path = [root]
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, successors = stack[-1]
            nxt = next(successors, None)
            if nxt is None:
                colour[node] = black
                stack.pop()
                path.pop()
                continue
            if colour[nxt] == grey:
                return [*path[path.index(nxt) :], nxt]
            if colour[nxt] == white:
                colour[nxt] = grey
                path.append(nxt)
                stack.append((nxt, iter(adjacency[nxt])))
    return None


__all__ = [
    "AdmissionChecker",
    "AdmissionLimits",
    "AdmissionRejected",
    "AdmissionResult",
    "AdmissionStatus",
    "Check",
    "CostEstimate",
    "EdgePolicy",
    "EdgeRule",
    "NodePolicy",
    "NodeRegistry",
    "NodeRule",
    "NodeSpec",
    "Rejection",
    "RemainingBudget",
]
