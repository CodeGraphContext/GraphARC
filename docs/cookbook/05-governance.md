# Governance

This is the part that makes GraphARC a *governed* runtime rather than a loop
with a nice trace file.

**The invariant**: a node may
**propose** nodes and edges; it may never **execute** them. Dynamism lives in
construction, governance lives in admission.

Why the split exists. A static graph cannot handle "investigate this incident" —
the shape of the work is discovered while doing it, and you cannot draw it in
advance. A fully dynamic loop handles that fine but cannot tell you afterwards
what it was *allowed* to do; the only record is what it happened to do. The
split gives you both: the planner is as inventive as you like, and nothing it
invents runs until deterministic, model-free code says yes.

Two packages implement it:

- `grapharc.planner` — four modules, one per step. `proposal` holds `Subgraph`
  (the typed thing a planner emits) and `PlannerNode`; `admission` holds
  `AdmissionChecker`, the deterministic gate; `materialize` turns an admitted
  proposal into a runnable graph and refuses everything else; `loop` drives the
  whole cycle to a recorded stop.
- `grapharc.policy` — a TOML document over tools, nodes, edges and spend, with
  tenant scoping, approval routing and a decision audit.

The recipes below build up in that order. Everything is honest about where the
edges are; the closing section lists what this does *not* give you, and none of
it is hidden until then.

---

## How do I let a planner invent a subgraph without letting it run one?

Three objects: a registry of node *kinds* an operator allows to exist, an edge
policy over transitions, and a checker that combines them. A proposal names
registry keys; it never carries a node body, so there is nothing in it to run
even if something wanted to.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)
from grapharc.runtime.graph import END, START

# What the operator allows to exist. Absence is refusal; there is no wildcard.
registry = NodeRegistry(
    [
        NodeSpec(name="fetch", description="read a URL"),
        NodeSpec(name="summarise", description="condense text"),
    ]
)

# What the operator allows to be wired. Unmatched is deny, so say something.
policy = EdgePolicy(rules=(EdgeRule(action="allow"),))

gate = AdmissionChecker(registry=registry, edge_policy=policy)

# What a planner wants. Nodes here carry a registry *key*, never a body.
proposal = Subgraph(
    nodes=(ProposedNode(name="fetch"), ProposedNode(name="summarise")),
    edges=(
        ProposedEdge(source=START, target="fetch"),
        ProposedEdge(source="fetch", target="summarise"),
        ProposedEdge(source="summarise", target=END),
    ),
    rationale="read the page, then condense it",
)

result = gate.check(proposal)
print("status:     ", result.status.value)
print("admitted:   ", result.admitted)
print("checks run: ", [c.value for c in result.checks_run])
print("worst case: ", result.worst_case)
```

```
status:      admitted
admitted:    True
checks run:  ['registry', 'policy', 'budget', 'depth', 'acyclicity']
worst case:  tokens=0 iterations=2 seconds=0.0
```

**Why it works this way.** `EdgePolicy`'s default is `deny`, so an empty policy
admits nothing — the allow-all rule above is what you write when you have not
decided yet, and it is deliberately something you have to type. All five checks
run on every proposal rather than short-circuiting on the first failure, because
a planner replanning from feedback should get the whole list, not one complaint
at a time.

An empty `Subgraph()` is legal and is admitted. That is correct: a planner needs
a way to say "no further work" that is not an exception, and admitting nothing
authorises nothing.

---

## How do I prove nothing runs during a check?

Give every registered kind a factory that raises, then admit a proposal that
uses it.

```python
from grapharc.planner import (
    AdmissionChecker,
    AdmissionRejected,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedNode,
    Subgraph,
)


def never_call(*args, **kwargs):
    raise AssertionError("admission executed a node factory")


registry = NodeRegistry([NodeSpec(name="fetch", factory=never_call)])
gate = AdmissionChecker(
    registry=registry, edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),))
)

result = gate.check(Subgraph(nodes=(ProposedNode(name="fetch"),)))
print("admitted:", result.admitted)
print("factory still uncalled:", registry.get("fetch").factory is never_call)

# `admit` is `check` that fails closed, for callers whose next line is `raise`.
try:
    gate.admit(Subgraph(nodes=(ProposedNode(name="shell"),)))
except AdmissionRejected as exc:
    print("raised:", exc.result.status.value, exc.result.rejections[0].code)
```

```
admitted: True
factory still uncalled: True
raised: rejected unregistered_node
```

`NodeSpec.factory` exists so that a later materialising step has somewhere to
get the node body from. Being in the registry is a licence to be *proposed*, not
an invitation to run. The only side effect a check has is the trace line it
writes.

---

## How do I stop a planner proposing a node I never registered?

You do not have to do anything: an unregistered `kind` is the REGISTRY check's
job, and the registry has no wildcard.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="fetch"), NodeSpec(name="summarise")]),
    edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
)

proposal = Subgraph(
    nodes=(ProposedNode(name="fetch"), ProposedNode(name="shell")),
    edges=(ProposedEdge(source="fetch", target="shell"),),
)

result = gate.check(proposal)
print("status:", result.status.value)
print("failed:", [c.value for c in result.failed_checks()])
for rejection in result.rejections:
    print(rejection.render())
```

```
status: rejected
failed: ['registry']
[registry/unregistered_node] shell: kind 'shell' is not in the node registry allowed kinds: fetch, summarise
```

Every rejection carries a `check`, a stable machine-readable `code`, a
`subject`, a `detail` and a `remedy`. Match on `code`; `detail` is prose and may
be reworded.

`ProposedNode(name="shell")` has kind `"shell"` — `kind` defaults to `name` for
the one-off case. Fill in `kind` explicitly whenever the instance name differs,
which is the whole point of a fan-out: `worker_a`, `worker_b`, `worker_c` all of
kind `edit_worker` means the registry only has to allow one thing.

---

## How do I forbid a transition?

`EdgeRule` is fnmatch over both endpoints, tiered deny → ask → allow. A broad
deny beats a narrower allow, so there is no allowlist exception hiding inside a
denial.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)
from grapharc.runtime.graph import START

policy = EdgePolicy(
    rules=(
        EdgeRule(action="deny", source="*", target="deploy"),
        EdgeRule(action="allow"),
    )
)
gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="build"), NodeSpec(name="deploy")]),
    edge_policy=policy,
)

proposal = Subgraph(
    nodes=(ProposedNode(name="build"), ProposedNode(name="deploy")),
    edges=(
        ProposedEdge(source=START, target="build"),
        ProposedEdge(source="build", target="deploy", note="ship it"),
    ),
)

result = gate.check(proposal)
print("status:", result.status.value)
for rejection in result.rejections:
    print(rejection.render())
```

```
status: rejected
[policy/edge_denied] build -> deploy: the edge policy denies this transition: kind 'build' (proposed as 'build') -> kind 'deploy' (proposed as 'deploy') the decision is made on the registry kind, not the name you chose: renaming the node will not change it — route through a permitted kind
```

**Why it works this way.** `ProposedEdge` carries a `source`, a `target` and a
`note`, and nothing else. There is deliberately no condition string: a predicate
authored by a model is model prose steering an edge, and the `note` is never
parsed or evaluated — it exists for the audit trail. An edge is admitted or
refused on its endpoints alone.

---

## How do I refuse a plan that cannot afford itself?

Register the worst case per kind and hand the checker the run's live meter. The
check reads the meter; it never charges it.

```python
from grapharc.planner import (
    AdmissionChecker,
    CostEstimate,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedNode,
    Subgraph,
)
from grapharc.runtime.budget import Budget, BudgetMeter

# The registry declares the cost. A proposal cannot.
gate = AdmissionChecker(
    registry=NodeRegistry(
        [NodeSpec(name="research", worst_case=CostEstimate(tokens=5_000, iterations=1))]
    ),
    edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
)

meter = BudgetMeter(Budget(max_tokens=20_000))
meter.charge_tokens(16_000)  # the run has already spent this much

proposal = Subgraph(
    nodes=(
        ProposedNode(name="research_a", kind="research"),
        ProposedNode(name="research_b", kind="research"),
    )
)

result = gate.check(proposal, meter=meter)
print("status:   ", result.status.value)
print("worst case", result.worst_case.tokens, "tokens")
print("remaining ", result.remaining.tokens, "tokens")
for rejection in result.rejections:
    print(rejection.render())
print("meter after the check:", meter.tokens, "tokens charged")
```

```
status:    rejected
worst case 10000 tokens
remaining  4000 tokens
[budget/over_token_budget] tokens: worst case needs 10000 tokens but only 4000 remain propose fewer or cheaper nodes
meter after the check: 16000 tokens charged
```

**Why it works this way.** Costs come from the registry, never from the
proposal — a `ProposedNode` has no field in which to claim it will be cheap. The
comparison is against what is *left*, not the total, so the same proposal that
was affordable at the start of a run is refused later in it. And the refusal
happens before the first node exists, rather than after the run notices it
overspent.

Two sharp edges. A dimension the registry leaves at zero is simply unconstrained
by that dimension's check — `worst_case` defaulting to `CostEstimate(iterations=1)`
means "nobody said what this costs in tokens", not "this is free". And if you
pass `remaining=` explicitly instead of `meter=`, that wins; with neither, every
dimension is unlimited.

---

## How do I stop planners planning planners forever?

`AdmissionLimits.max_depth` bounds nesting. It defaults to `1`, which means flat
proposals only.

```python
from grapharc.planner import (
    AdmissionChecker,
    AdmissionLimits,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedNode,
    Subgraph,
)

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="planner"), NodeSpec(name="fetch")]),
    edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
    limits=AdmissionLimits(max_depth=1),  # this is the default
)

nested = Subgraph(
    nodes=(
        ProposedNode(
            name="sub_planner",
            kind="planner",
            subgraph=Subgraph(nodes=(ProposedNode(name="inner", kind="fetch"),)),
        ),
    ),
    proposal_id="nested-demo",
)
print("nesting_depth:", nested.nesting_depth(), " node_count:", nested.node_count())

result = gate.check(nested)
print("status:", result.status.value, " depth:", result.depth)
for rejection in result.rejections:
    print(rejection.render())

# Same proposal, but arriving from a planner that is already one level in.
flat = Subgraph(nodes=(ProposedNode(name="fetch"),), proposal_id="flat-demo")
deep = gate.check(flat, parent_depth=1)
print("flat at parent_depth=1:", deep.status.value, "->", deep.rejections[0].detail)
```

```
nesting_depth: 2  node_count: 2
status: rejected  depth: 2
[depth/too_deep] nested-demo: depth 2 exceeds max_depth 1 (admitted at depth 0, proposal nests 2 level(s)) flatten the proposal, or propose the inner work in a later round
flat at parent_depth=1: rejected -> depth 2 exceeds max_depth 1 (admitted at depth 1, proposal nests 1 level(s))
```

**The sharp edge: `parent_depth` is supplied by you.** The checker has no way to
observe how deep the run actually is. A caller that always passes `0` has no
recursion limit beyond the nesting visible inside a single proposal — a planner
admitted at depth 1 can propose another flat plan forever. If you run planners
inside admitted subgraphs, threading the real depth through is your job, and it
is the one number in this section the gate cannot verify for you.

`Subgraph.scopes()` walks nesting iteratively, not recursively, so a hostile
depth exhausts the limit rather than the stack.

---

## How do I allow (or refuse) a loop?

Cycles are earned. `require_acyclic` defaults to `True`; a proposal that
genuinely needs a loop has to be admitted by a checker configured for it.

```python
from grapharc.planner import (
    AdmissionChecker,
    AdmissionLimits,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

registry = NodeRegistry([NodeSpec(name="draft"), NodeSpec(name="review")])
policy = EdgePolicy(rules=(EdgeRule(action="allow"),))

loop = Subgraph(
    nodes=(ProposedNode(name="draft"), ProposedNode(name="review")),
    edges=(
        ProposedEdge(source="draft", target="review"),
        ProposedEdge(source="review", target="draft", note="revise until good"),
    ),
)

strict = AdmissionChecker(registry=registry, edge_policy=policy)
print("default checker: ", strict.check(loop).status.value)
print(strict.check(loop).rejections[0].render())

permissive = AdmissionChecker(
    registry=registry,
    edge_policy=policy,
    limits=AdmissionLimits(require_acyclic=False),
)
result = permissive.check(loop)
print("permissive:      ", result.status.value)
print("checks run:      ", [c.value for c in result.checks_run])
```

```
default checker:  rejected
[acyclicity/cycle] draft -> review -> draft: this checker requires acyclic proposals and found a cycle break the cycle, or use a checker with require_acyclic=False
permissive:       admitted
checks run:       ['registry', 'policy', 'budget', 'depth']
```

Note the last line: with `require_acyclic=False` the ACYCLICITY check does not
appear in `checks_run` at all. `checks_run` is what actually ran, not a fixed
list.

**The limit.** The cycle search only sees the topology a proposal declares.
Edges touching the `START`/`END` sentinels or a node already live in the graph
are ignored, so a cycle formed *between* a proposed node and an existing one is
outside what this check can observe.

---

## How do I make sure renaming a node cannot evade a denial?

This is the subtle one, and it is the reason the `name` / `kind` split exists.
The planner chooses the instance `name`. The operator registered the `kind`.
Every check that grants anything resolves against the kind.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="build"), NodeSpec(name="deploy")]),
    edge_policy=EdgePolicy(
        rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
    ),
)


def attempt(label, node):
    proposal = Subgraph(
        nodes=(ProposedNode(name="build"), node),
        edges=(ProposedEdge(source="build", target=node.name),),
    )
    result = gate.check(proposal)
    print(f"{label:<38} {result.status.value}")
    for rejection in result.rejections:
        print(f"    {rejection.check.value}/{rejection.code}: {rejection.detail}")


attempt("name=deploy  kind=deploy", ProposedNode(name="deploy"))
attempt("name=helper  kind=deploy", ProposedNode(name="helper", kind="deploy"))
attempt("name=deploy  kind=build ", ProposedNode(name="deploy", kind="build"))
```

```
name=deploy  kind=deploy               rejected
    policy/edge_denied: the edge policy denies this transition: kind 'build' (proposed as 'build') -> kind 'deploy' (proposed as 'deploy')
name=helper  kind=deploy               rejected
    policy/edge_denied: the edge policy denies this transition: kind 'build' (proposed as 'build') -> kind 'deploy' (proposed as 'helper')
name=deploy  kind=build                admitted
```

Read all three lines. Renaming the denied kind to `helper` does not launder it.
And the third line is the same rule running in the other direction: a node
*named* `deploy` that is really a `build` is admitted, because the name borrows
no permission either.

**Why the distinction exists.** The name is chosen by the untrusted side. A
rule matched against it would be a rule the thing being governed gets to
control, which is not a rule. A rule matched against the kind is a statement
about what a node *is* — the only thing an operator is in a position to
authorise. Names survive as labels: they resolve which node an edge refers to,
and they appear in the rejection (`proposed as 'helper'`) so an audit can find
the specific edge. No rule is ever matched against one.

The corollary is that an endpoint whose kind cannot be determined is refused
rather than assumed safe — see the `known_nodes` recipe below. "No rule matched"
must never be reachable by hiding what a node is.

---

## What does admission *not* check?

**Arguments.** `ProposedNode.args` are carried for a future materialiser and are
never inspected. Admission authorises a *kind*, not its arguments.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedNode,
    Subgraph,
)

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="run_sql")]),
    edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
)

for query in ("SELECT count(*) FROM users", "DROP TABLE users"):
    proposal = Subgraph(
        nodes=(ProposedNode(name="q", kind="run_sql", args={"query": query}),),
        proposal_id="args-demo",
    )
    result = gate.check(proposal)
    print(f"{result.status.value:<9} fingerprint={result.fingerprint}  query={query!r}")
```

```
admitted  fingerprint=5278b17f1c582cc7  query='SELECT count(*) FROM users'
admitted  fingerprint=00d7b0935c1a641b  query='DROP TABLE users'
```

Both are admitted, and no rule in this package can change that. Registering
`run_sql` at all is a decision to trust whatever gates its arguments downstream —
the tool plane's permissions (`grapharc.harness.permissions`) are where that
belongs, not here.

`Materializer` takes the same position and makes it opt-in: with the default
`forward_args=False`, `NodeBuild.args` is empty whatever the proposal said, so a
planner's arguments reach nothing. `forward_args=True` hands the raw dict to
your factory **unchecked** — no gate has looked at it, and a factory that pulls
a callable out of it and runs it has re-opened the boundary the whole design
exists to hold.

What the fingerprints do give you: the two proposals hash differently, so an
audit can tell after the fact which one was admitted, even though the gate could
not tell them apart. `Subgraph.fingerprint()` is a content hash — identity, not
semantic equality. Reordering `args` keys changes it, because the recorded
decision was made about those bytes.

(The `proposal_id` is pinned above so the output is reproducible. Left alone it
is a fresh uuid per proposal, and the fingerprint moves with it.)

---

## How do I attach a proposal to nodes that are already running?

Pass `known_nodes`. Pass it as a `{name: kind}` mapping, not a list of names —
a name alone tells the checker nothing it can evaluate a rule against.

```python
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

registry = NodeRegistry([NodeSpec(name="patch"), NodeSpec(name="deploy")])
policy = EdgePolicy(
    rules=(EdgeRule(action="deny", source="deploy"), EdgeRule(action="allow"))
)
proposal = Subgraph(
    nodes=(ProposedNode(name="patch"),),
    edges=(ProposedEdge(source="live_step", target="patch"),),
)

# By name only: the checker cannot tell what `live_step` is, so it refuses.
by_name = AdmissionChecker(
    registry=registry, edge_policy=policy, known_nodes=["live_step"]
)
print(by_name.check(proposal).rejections[0].render())

# With its kind declared, the rule can actually be evaluated.
declared = AdmissionChecker(
    registry=registry, edge_policy=policy, known_nodes={"live_step": "deploy"}
)
print(declared.check(proposal).rejections[0].render())

benign = AdmissionChecker(
    registry=registry, edge_policy=policy, known_nodes={"live_step": "patch"}
)
print(benign.check(proposal).status.value)
```

```
[policy/unresolved_endpoint_kind] live_step -> patch: source 'live_step' has no resolvable registry kind, so no rule can be evaluated for it; the edge is refused rather than assumed permitted (it is a live node the caller listed by name only) declare what it is: known_nodes={'live_step': '<kind>'}
[policy/edge_denied] live_step -> patch: the edge policy denies this transition: kind 'deploy' (proposed as 'live_step') -> kind 'patch' (proposed as 'patch') the decision is made on the registry kind, not the name you chose: renaming the node will not change it — route through a permitted kind
admitted
```

These kinds come from you, not from a proposal, and they need not be registered:
the registry is the list of kinds a *planner* may propose, not of kinds that may
already exist. A proposed node may not reuse a live node's name — that is a
separate `registry/name_collides_with_existing_node` rejection, because
admitting it would rebind a node that is already running.

**The sharp edge is one step later.** `known_nodes` gets a proposal *admitted*;
it does not get it *built*. `Materializer` assembles a standalone graph and
refuses an edge endpoint that is not one of the proposal's own nodes or a
sentinel, live node or not — it has nothing to attach to. So `known_nodes` is
for a caller that materialises the result some other way, not for the shipped
materialiser below.

---

## How do I require a human before an edge is wired?

An edge whose policy resolves to `ask` produces `NEEDS_APPROVAL`. That is not
admitted.

```python
from grapharc.planner import (
    AdmissionChecker,
    AdmissionStatus,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="patch"), NodeSpec(name="deploy")]),
    edge_policy=EdgePolicy(
        rules=(EdgeRule(action="ask", target="deploy"), EdgeRule(action="allow"))
    ),
)

result = gate.check(
    Subgraph(
        nodes=(ProposedNode(name="patch"), ProposedNode(name="deploy")),
        edges=(ProposedEdge(source="patch", target="deploy"),),
    )
)
print("status:        ", result.status.value)
print("admitted:      ", result.admitted)
print("needs_approval:", result.needs_approval)
print(result.rejections[0].render())
print(AdmissionStatus.NEEDS_APPROVAL is not AdmissionStatus.ADMITTED)
```

```
status:         needs_approval
admitted:       False
needs_approval: True
[policy/edge_needs_approval] patch -> deploy: the edge policy requires approval for this transition: kind 'patch' (proposed as 'patch') -> kind 'deploy' (proposed as 'deploy') obtain approval and re-submit
True
```

**A pending approval is a stop, not a slow yes.** `result.admitted` is `False`.
`NEEDS_APPROVAL` is reported only when *every* rejection is an approval request —
one denial anywhere outranks the whole thing and the status is `REJECTED`.

**The limit.** `AdmissionChecker` does not route the approval. It tells you an
edge needs one; who gets asked is not wired up on this path. The policy engine
below does route approvals for *tools*, by role — but nothing today carries an
`edge_needs_approval` from the checker into that router. You write that hop.

---

## How do I get a rejection into the audit trail?

Give the checker a `TraceRecorder`. Every decision — admitted and rejected
alike — is written as a `phase="admission"` event.

```python
import json
import tempfile
from pathlib import Path

from grapharc.observe.trace import TraceRecorder
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)

trace = TraceRecorder(Path(tempfile.mkdtemp()) / "trace.jsonl")
gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="build"), NodeSpec(name="deploy")]),
    edge_policy=EdgePolicy(
        rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
    ),
    trace=trace,
)

result = gate.check(
    Subgraph(
        nodes=(ProposedNode(name="build"), ProposedNode(name="ship", kind="deploy")),
        edges=(ProposedEdge(source="build", target="ship"),),
        proposal_id="p-0001",
        origin="planner:release",
    )
)

event = json.loads(trace.path.read_text().splitlines()[0])
del event["ts"]  # wall clock; everything else is reproducible
print(json.dumps(event, indent=2))
print()
print(result.feedback())
```

```
{
  "run_id": "p-0001",
  "attempt": 1,
  "graph": "admission",
  "node": "admission:p-0001",
  "phase": "admission",
  "step": 0,
  "state_delta": {
    "status": "rejected",
    "fingerprint": "ae82f551624a29fd",
    "origin": "planner:release",
    "nodes": 2,
    "depth": 1,
    "checks_run": [
      "registry",
      "policy",
      "budget",
      "depth",
      "acyclicity"
    ],
    "failed_checks": [
      "policy"
    ],
    "worst_case": {
      "tokens": 0,
      "iterations": 2,
      "seconds": 0.0
    }
  },
  "error": "policy/edge_denied"
}

Proposal p-0001 was not admitted (status: rejected). Failed checks: policy.
- [policy/edge_denied] build -> ship: the edge policy denies this transition: kind 'build' (proposed as 'build') -> kind 'deploy' (proposed as 'ship') the decision is made on the registry kind, not the name you chose: renaming the node will not change it — route through a permitted kind
```

**Why it works this way.** The phase is `"admission"`, deliberately not `"end"`:
`grapharc.observe.metrics` counts node executions from `"end"` events, and an
admission decision is not one. Reusing the phase would inflate every metric
derived from the trace.

Passing a `RunContext` as `ctx=` stamps the real `run_id`, `thread_id` and step
number instead. Without one the proposal's own id stands in, so the decision is
still findable.

---

## How do I feed a rejection back for replanning?

`AdmissionResult.feedback()` renders the whole rejection as text a planner can
be handed verbatim, and `PlannerNode.propose(task, feedback=...)` takes it.

```python
import json

from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    PlannerNode,
)
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel

registry = NodeRegistry(
    [
        NodeSpec(name="triage", description="read the report, classify it"),
        NodeSpec(name="patch", description="write a fix on a branch"),
        NodeSpec(name="deploy", description="push to production"),
    ]
)
gate = AdmissionChecker(
    registry=registry,
    edge_policy=EdgePolicy(
        rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
    ),
)

first = {
    "rationale": "triage, then ship the fix",
    "nodes": [{"name": "triage"}, {"name": "shipit", "kind": "deploy"}],
    "edges": [
        {"source": START, "target": "triage"},
        {"source": "triage", "target": "shipit"},
        {"source": "shipit", "target": END},
    ],
}
second = {
    "rationale": "triage, then patch and stop short of production",
    "nodes": [{"name": "triage"}, {"name": "patch"}],
    "edges": [
        {"source": START, "target": "triage"},
        {"source": "triage", "target": "patch"},
        {"source": "patch", "target": END},
    ],
}
model = ScriptedChatModel(responses=[json.dumps(first), json.dumps(second)])
planner = PlannerNode(model, name="release", catalog=registry.catalog())

task = "fix the login bug and get it live"
outcome = planner.propose(task)
result = gate.check(outcome.proposal)
print("round 1:", result.status.value, [c.value for c in result.failed_checks()])

if not result.admitted:
    outcome = planner.propose(task, feedback=result.feedback())
    result = gate.check(outcome.proposal)
print("round 2:", result.status.value, [n.name for n in outcome.proposal.nodes])
print("origin: ", outcome.proposal.origin)

# The rejection really was what the second turn was told.
print("---- what the planner saw on round 2 ----")
print(model.calls[1][-1].content)
```

```
round 1: rejected ['policy']
round 2: admitted ['triage', 'patch']
origin:  planner:release
---- what the planner saw on round 2 ----
Your previous proposal was not admitted. The admission checker reported:

Proposal 2d90379fbf52 was not admitted (status: rejected). Failed checks: policy.
- [policy/edge_denied] triage -> shipit: the edge policy denies this transition: kind 'triage' (proposed as 'triage') -> kind 'deploy' (proposed as 'shipit') the decision is made on the registry kind, not the name you chose: renaming the node will not change it — route through a permitted kind

Propose again, fixing every line above.
```

(`2d90379fbf52` is a per-proposal uuid; yours will differ. Everything else is
reproducible.)

**Why it works this way.** `PlannerNode` cannot execute what it proposes, and
that is structural rather than promised: the only callables it holds are the
chat model and your own state reader. It is given no node registry, no harness
and no graph, so a proposed node has no body available to it. `NodeSpec.factory`
— the only place a node body ever appears — lives on the far side of the gate.

Two details worth knowing. The planner *re-stamps* provenance: whatever the
model puts in `proposal_id` and `origin` is overwritten (`planner:release`
above), because provenance the proposer wrote about itself is not provenance.
And a bad model reply is a `PlanningOutcome` carrying `error`, not an exception —
which is exactly what lets you retry it. Only `BudgetExceeded` is allowed
through, because that is the run's hard ceiling and not this turn's business.

The default planner system prompt already tells the model that renaming a
refused node is a wasted turn (`DEFAULT_PLANNER_SYSTEM_PROMPT`). That is a
courtesy to save a round trip. It is not the enforcement — the gate is.

### With a real model

Swap the scripted model for a real one; nothing else changes.

```python
from grapharc.gateway import get_model
from grapharc.planner import PlannerNode

model = get_model("claude-cli/claude-sonnet-5")  # needs a Claude Code subscription
planner = PlannerNode(model, catalog=registry.catalog())
```

**This snippet was not executed** — it needs a subscription or an API key, and
nothing in this section calls a paid backend. `PlannerNode` prefers the
backend's structured-output path when it has one and falls back to tolerant JSON
extraction from the text otherwise; `outcome.structured` records which path
produced the object, because the two have different failure modes and an audit
should not have to guess. The Claude CLI adapter has no tool calling, so it
takes the text path.

---

## How do I run what was admitted — and only what was admitted?

`Materializer` is the only thing that turns a proposal into a graph. It takes
the `AdmissionResult` **first**, because that is the authorisation: there is no
entry point that accepts a bare `Subgraph`.

```python
from pydantic import BaseModel

from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    Materializer,
    NodeRegistry,
    NodeSpec,
    NotAdmitted,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)
from grapharc.runtime.graph import END, START


class State(BaseModel):
    notes: list[str] = []


def step_factory(build):
    """Called once per proposed instance, with a NodeBuild describing it."""

    def body(state: State) -> dict:
        return {"notes": [*state.notes, f"{build.name} ran (kind {build.kind})"]}

    body.writes = {"notes"}
    return body


registry = NodeRegistry(
    [
        NodeSpec(name="fetch", factory=step_factory),
        NodeSpec(name="summarise", factory=step_factory),
    ]
)
gate = AdmissionChecker(
    registry=registry, edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),))
)
materializer = Materializer(registry=registry, state_schema=State)

proposal = Subgraph(
    nodes=(ProposedNode(name="fetch"), ProposedNode(name="summarise")),
    edges=(
        ProposedEdge(source=START, target="fetch"),
        ProposedEdge(source="fetch", target="summarise"),
        ProposedEdge(source="summarise", target=END),
    ),
    proposal_id="build-demo",
)

result = gate.check(proposal)
graph = materializer.materialize(result, proposal)  # the result first: it is the authority
print(graph.invoke(State()))

# A result that did not admit *this* proposal cannot build it.
edited = Subgraph(
    nodes=proposal.nodes,
    edges=proposal.edges,
    rationale="one word changed",
    proposal_id="build-demo",
)
try:
    materializer.materialize(result, edited)
except NotAdmitted as exc:
    print(exc)
```

```
{'notes': ['fetch ran (kind fetch)', 'summarise ran (kind summarise)']}
this AdmissionResult authorised proposal build-demo (fingerprint cba0d7416bec398d), not proposal build-demo (fingerprint f3083ceac3a28260); what runs must be what was admitted
```

**Why it works this way.** "What ran is what was admitted" is a checked equality,
not a convention. The second call reuses a genuine approval alongside a proposal
whose only difference is its `rationale` string, and the fingerprint catches it —
so smuggling an edited proposal past a real authorisation fails exactly as an
outright rejection does. Note both proposals even share a `proposal_id`; the
hash is over content.

**Where a body comes from.** Only `NodeSpec.factory`, which is operator code
registered before any planning happened. `ProposedNode` forbids extra fields, so
a proposal has no channel for a callable at all. The factory is called once per
proposed instance with a `NodeBuild` (`name`, `kind`, `note`, `proposal_id`,
`fingerprint`, and `args` only if you opted in) and must return the node's body.

**Built through the kernel, not around it.** The graph is assembled with the
ordinary `GraphARC` builder, so declared writes, typed state, the budget meter,
deep-copy isolation and trace events all still apply. `body.writes` above is how
this instance declares what it may write; a `writes={kind: fields}` table on the
`Materializer` overrides it, and a kind nobody declared writes for may write
nothing — the first field it touches raises `WritePermissionError`.

Five things it refuses rather than guesses at: a nested `ProposedNode.subgraph`
(there is no sub-materialisation, so the inner proposal would be silently
dropped), an edge endpoint that is not a node of this proposal, a proposal with
no edge out of `START`, a node `START` cannot reach (admitted and then never
run is a proposal that does not mean what it says), and an empty proposal.

---

## How do I stop a node body routing somewhere it was not admitted to?

Node bodies are operator code and may route dynamically with `Command(goto=...)`.
The materialiser confines each body to the destinations *the proposal declared an
edge from that node to*, plus `END`.

```python
from langgraph.types import Command
from pydantic import BaseModel

from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    Materializer,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
    UnadmittedTransition,
)
from grapharc.runtime.graph import END, START


class State(BaseModel):
    notes: list[str] = []


def router_factory(build):
    # Operator code decides where this kind routes. `router_skip` jumps ahead.
    target = "risky" if build.kind == "router_skip" else "safe"

    def body(state: State) -> Command:
        return Command(goto=target, update={"notes": [*state.notes, f"routed to {target}"]})

    body.writes = {"notes"}
    return body


def step_factory(build):
    def body(state: State) -> dict:
        return {"notes": [*state.notes, f"{build.name} ran"]}

    body.writes = {"notes"}
    return body


registry = NodeRegistry(
    [
        NodeSpec(name="router_ok", factory=router_factory),
        NodeSpec(name="router_skip", factory=router_factory),
        NodeSpec(name="safe", factory=step_factory),
        NodeSpec(name="risky", factory=step_factory),
    ]
)
gate = AdmissionChecker(
    registry=registry, edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),))
)
materializer = Materializer(registry=registry, state_schema=State)


def run(kind: str):
    proposal = Subgraph(
        nodes=(
            ProposedNode(name="plan", kind=kind),
            ProposedNode(name="safe"),
            ProposedNode(name="risky"),
        ),
        edges=(
            ProposedEdge(source=START, target="plan"),
            ProposedEdge(source="plan", target="safe"),
            ProposedEdge(source="safe", target="risky"),
            ProposedEdge(source="risky", target=END),
        ),
    )
    graph = materializer.materialize(gate.check(proposal), proposal)
    return graph.invoke(State())


print(run("router_ok"))
try:
    run("router_skip")
except UnadmittedTransition as exc:
    print(exc)
```

```
{'notes': ['routed to safe', 'safe ran', 'risky ran']}
node 'plan' routed to 'risky', which the admitted proposal declared no edge to. Admitted destinations for 'plan': safe, END. A body may route dynamically only along an edge the gate authorised
```

`risky` is a node of the graph and it does run in the first case — reached the
long way, through `safe`. What is refused is `plan` *skipping to it* along an
edge nobody admitted. So the admitted **edge** set bounds dynamic routing, not
merely the admitted node set. The exception is raised before the kernel resolves
the destination, so the transition does not happen and that node's writes never
land.

**The limit, stated rather than implied.** This covers a `Command` a body
returns. It says nothing about what a body does *inside* itself — calling
another graph, spawning a thread, shelling out. That is the tool and permission
planes' business (`grapharc.harness`), not this one's.

Conditional routing has no representation in a proposal on purpose:
`ProposedEdge` carries no predicate, because a predicate authored by a model is
model prose steering an edge.

---

## How do I drive the whole cycle to a recorded stop?

`GovernedLoop` runs plan → admit → materialise → execute → observe → replan
until something says stop, and the stop is always a reason.

```python
import json

from pydantic import BaseModel

from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    GovernedLoop,
    Materializer,
    NodeRegistry,
    NodeSpec,
    PlannerNode,
)
from grapharc.runtime.graph import END, START
from grapharc.testing import ScriptedChatModel


class State(BaseModel):
    notes: list[str] = []


def step_factory(build):
    def body(state: State) -> dict:
        return {"notes": [*state.notes, f"{build.name} ran"]}

    body.writes = {"notes"}
    return body


registry = NodeRegistry(
    [
        NodeSpec(name="triage", description="classify the incident", factory=step_factory),
        NodeSpec(name="patch", description="write a fix", factory=step_factory),
        NodeSpec(name="deploy", description="push to production", factory=step_factory),
    ]
)
gate = AdmissionChecker(
    registry=registry,
    edge_policy=EdgePolicy(
        rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
    ),
)
materializer = Materializer(registry=registry, state_schema=State)


def plan(*names):
    endpoints = [START, *names, END]
    return json.dumps(
        {
            "nodes": [{"name": n} for n in names],
            "edges": [
                {"source": a, "target": b}
                for a, b in zip(endpoints, endpoints[1:], strict=False)
            ],
        }
    )


model = ScriptedChatModel(responses=[plan("triage", "deploy"), plan("triage", "patch")])
loop = GovernedLoop(
    planner=PlannerNode(model, name="incident", catalog=registry.catalog()),
    checker=gate,
    materializer=materializer,
    goal_reached=lambda state: len(state.notes) >= 2,
)

done = loop.run("investigate the incident")
print("stop:  ", done.stop.value, "|", done.detail)
print("rounds:", len(done.rounds))
for record in done.rounds:
    status = record.admission.status.value if record.admission else "not_proposed"
    print(f"  round {record.round}: {status:<9} executed={record.executed}")
print("rejections:", [r.code for r in done.rejections()])
print("state: ", done.state)
```

```
stop:   goal_met | the goal check was satisfied
rounds: 2
  round 1: rejected  executed=False
  round 2: admitted  executed=True
rejections: ['edge_denied']
state:  notes=['triage ran', 'patch ran']
```

**The property this exists to hold.** Work discovered mid-run does not bypass
admission. Round 7's proposal goes through the same checker, the same registry
and the same edge policy as round 1's — there is no "already approved" path, no
cached authorisation, and no way for an executing node to add topology, because
a node's only channel out is the state schema and the loop's only source of
topology is `planner.propose`.

**Every stop is a reason**, and `LoopResult.stop` is always set. The vocabulary
is `goal_met`, `no_further_work`, `max_rounds`, `budget_exhausted`,
`no_progress`, `admission_refused`, `planning_failed`, `execution_failed`,
`human_stopped`. `succeeded` is true only for `goal_met` — `no_further_work`
means the planner ran out of work, which is an answer, not a success.

**Two sets of bounds, deliberately separate.** `Budget` bounds resources
(tokens, iterations, seconds) across the whole run: each round executes under a
budget derived from what the shared meter has *left*, so the last round cannot
spend what earlier rounds already did, and the planner's own tokens are charged
to the same meter. `LoopLimits` bounds the *loop* — `max_rounds`, and consecutive
counters for rejections, unusable planner replies, execution failures and stalled
rounds. Those catch a planner that is failing productively: proposing something
illegal, unparseable, or ineffective, forever, inside budget.

**`goal_reached` is your code.** The loop has no opinion about what "done" means
and will not ask a model — that is exactly the question a run must not be able
to talk itself into answering yes. With none supplied, a run ends on the planner
proposing no further work, or on a bound.

Two limits worth knowing: a `GovernedLoop` instance is not safe for concurrent
`run()` calls, and `request_halt()` is permanent for the object that received it.
A halt takes effect at a round boundary, never mid-node, so it cannot leave a
node half-applied.

With a `TraceRecorder` wired to the loop, the checker and the materialiser, one
file answers all three questions ARCHITECTURE §6 asks: what it did (node
`start`/`end` events), what it was *allowed* to do (`phase="admission"` events
carrying the fingerprint of the exact proposal), and why it stopped (one
`phase="stop"` event). They share a `run_id`; each round executes on its own
trace thread `<run_id>/r<n>` so per-thread step numbers stay unique.

---

## How do I write the policy down instead of coding it?

`grapharc.policy` loads a TOML document over four resource kinds — `tool`,
`node`, `edge`, `spend` — and answers decisions against it. Save this as
`policy.toml`:

```toml
version = "2026-07-01"
name = "cookbook"
default = "deny"                  # unmatched requests are refused
tenants = ["default", "acme"]     # a request naming anything else is denied

[[rule]]
id = "reads-are-free"
resource = "tool"
match = "read_*"
effect = "allow"

[[rule]]
id = "no-deletes"
resource = "tool"
match = "delete_*"
effect = "deny"
reason = "destructive tools are never permitted, for anyone"

[[rule]]
id = "deploys-need-sre"
resource = "tool"
match = "deploy_*"
effect = "ask"
approver_role = "sre"             # required on every `ask` rule
reason = "a deploy is a production change"

[[rule]]
id = "acme-may-write"
resource = "tool"
match = "write_*"
effect = "allow"
tenant = "acme"                   # fnmatch over the requesting tenant

[[rule]]
id = "no-shell-nodes"
resource = "node"
match = "shell_*"
effect = "deny"

[[rule]]
id = "other-nodes-run"
resource = "node"
match = "*"
effect = "allow"

[[rule]]
id = "nothing-routes-into-deploy"
resource = "edge"
match = "*->deploy"
effect = "deny"
reason = "production is entered by an operator, not by a plan"

[[rule]]
id = "other-edges-are-fine"
resource = "edge"
match = "*->*"
effect = "allow"

[[rule]]
id = "acme-period-cap"
resource = "spend"
effect = "deny"
over_usd = 10.0
basis = "cumulative"              # committed spend + this request
tenant = "acme"

[[rule]]
id = "over-a-dollar-asks-finance"
resource = "spend"
effect = "ask"
over_usd = 1.0
approver_role = "finance"

[[rule]]
id = "small-spend-is-fine"
resource = "spend"
effect = "allow"
```

```python
from grapharc.policy import PolicyEngine

engine = PolicyEngine.from_file("policy.toml")

decisions = [
    engine.check_tool("read_config"),
    engine.check_tool("delete_bucket"),
    engine.check_tool("deploy_prod"),
    engine.check_tool("write_file"),
    engine.check_tool("write_file", tenant="acme"),
    engine.check_node("shell_exec"),
    engine.check_node("summarise"),
    engine.check_edge("triage", "deploy"),
    engine.check_edge("triage", "patch"),
    engine.check_spend(0.40),
    engine.check_spend(4.20),
]
for d in decisions:
    role = f"  ask:{d.approver_role}" if d.requires_approval else ""
    print(
        f"{d.resource.value:<5} {d.subject:<16} {d.tenant:<8} "
        f"{d.effect.value:<6} rule={d.rule_id}{role}"
    )

print()
print("policy version:", engine.version, " digest:", engine.digest[:16])
print("audit records: ", len(engine.audit))
```

```
tool  read_config      default  allow  rule=reads-are-free
tool  delete_bucket    default  deny   rule=no-deletes
tool  deploy_prod      default  ask    rule=deploys-need-sre  ask:sre
tool  write_file       default  deny   rule=None
tool  write_file       acme     allow  rule=acme-may-write
node  shell_exec       default  deny   rule=no-shell-nodes
node  summarise        default  allow  rule=other-nodes-run
edge  triage->deploy   default  deny   rule=nothing-routes-into-deploy
edge  triage->patch    default  allow  rule=other-edges-are-fine
spend *                default  allow  rule=small-spend-is-fine
spend *                default  ask    rule=over-a-dollar-asks-finance  ask:finance

policy version: 2026-07-01  digest: b1d593faa1d41fcf
audit records:  11
```

**Why it works this way.**

- **Tiered, not positional.** Every `deny` is tried before every `ask`, and
  every `ask` before every `allow`. Within a tier, first match in document
  order wins, and that rule's `id` is what the audit record names. So you
  cannot write an exception *inside* a deny — `no-deletes` beats any allow
  anyone adds later, including one scoped to a single tenant.
- **`rule=None` is not a bug.** `write_file` for the default tenant matched no
  rule, so the document `default` applied. The audit record says so in words.
- **TOML, not YAML.** `tomllib` is standard library from 3.11, so loading a
  policy costs no dependency, and `[[rule]]` keeps rule order visible in the
  file — which matters because order decides which rule is reported as the
  cause. `tomllib` cannot write, so nothing round-trips: policies are authored
  by hand and versioned in git.
- **`default = "ask"` is a load error.** An unmatched request matches no rule
  and therefore names no approver, so it could only fail closed. Write an
  explicit catch-all `ask` rule with an `approver_role` instead.
- **Typos fail loud at load.** An `ask` rule with no `approver_role`, an
  `approver_role` on a non-`ask` rule, an `over_usd` on a `tool` rule, a
  duplicate rule id, an edge `match` without an arrow, or a rule scoped to a
  tenant that is not declared — each is a `PolicyError` when the document is
  loaded, not a silent misfire at decision time.

---

## How do I scope policy to a tenant and cap its spend?

`tenants` in the document is the enumeration of real tenants; a rule's `tenant`
field is an fnmatch pattern. A request naming an undeclared tenant is denied and
recorded rather than raised, because a tenant name arrives with the request —
i.e. from data.

```python
from grapharc.policy import PolicyEngine

engine = PolicyEngine.from_file("policy.toml")

# A tenant the document does not declare is refused, and the refusal is recorded.
d = engine.check_tool("read_config", tenant="stranger")
print(d.effect.value, "|", d.reason)

# Cumulative spend: the cap moves only when spend is *recorded*.
print("acme committed:", engine.committed_usd("acme"))
print("  $6 now:", engine.check_spend(6.0, tenant="acme").effect.value)
engine.record_spend(6.0, tenant="acme")
print("acme committed:", engine.committed_usd("acme"))
again = engine.check_spend(6.0, tenant="acme")
print("  $6 again:", again.effect.value, "| rule:", again.rule_id)
print("  considered:", again.request)
```

```
deny | tenant 'stranger' is not declared by policy '2026-07-01'; declared: ('default', 'acme')
acme committed: 0.0
  $6 now: ask
acme committed: 6.0
  $6 again: deny | rule: acme-period-cap
  considered: {'amount_usd': 6.0, 'committed_usd': 6.0, 'basis': 'cumulative', 'considered_usd': 12.0}
```

**The sharp edge: checking is not committing.** `check_spend` compares a
`cumulative` rule's threshold against `committed_usd(tenant) + amount`, and that
committed total only moves when you call `record_spend`. A caller that checks
and never records will pass a cumulative cap forever. The ledger is also
in-process and in-memory — it does not survive a restart, and two processes do
not share it.

---

## How do I route an approval to the right role?

An `ask` rule names `approver_role`. `approval_router` reads it off the decision
and hands the request to that role's handler.

```python
from grapharc.policy import PolicyEngine

engine = PolicyEngine.from_file("policy.toml")


def sre(request):
    print(f"    [{request.approver_role}] {request.subject}: {request.reason}")
    return request.subject != "deploy_prod"  # staging yes, production no


router = engine.approval_router({"sre": sre})

for tool in ("read_config", "delete_bucket", "deploy_staging", "deploy_prod"):
    print(f"{tool:<16} granted={router(tool, {})}")

print()
for record in engine.audit.entries(kind="approval"):
    print(f"{record.subject:<16} granted={record.granted!s:<5} {record.reason}")

# An `ask` whose role has no handler fails closed rather than falling through.
print()
orphan = engine.approval_router({})
print("no handler registered:", orphan("deploy_staging", {}))
print(engine.audit.entries(kind="approval")[-1].reason)
```

```
read_config      granted=True
delete_bucket    granted=False
    [sre] deploy_staging: a deploy is a production change
deploy_staging   granted=True
    [sre] deploy_prod: a deploy is a production change
deploy_prod      granted=False

read_config      granted=True  allowed by policy: rule 'reads-are-free' (allow) matched tool 'read_config' for tenant 'default'
delete_bucket    granted=False denied by policy: destructive tools are never permitted, for anyone
deploy_staging   granted=True  sre granted approval
deploy_prod      granted=False sre refused approval

no handler registered: False
no approval handler registered for role 'sre'
```

Every path fails closed and every path is recorded: no handler for the role,
a handler that raises (a broken approval channel is not consent), an `ask`
decision carrying no role, and a `deny` — for which no handler is called at all.

The router is callable as `(tool_name, args) -> bool`, which is the shape
`grapharc.harness.core.ApprovalCallback` expects, so it drops straight into a
`Harness`. It is bound to one tenant, because that callback signature carries
none.

### Compiling the document down to a `PermissionPolicy`

A `Harness` enforces tool permissions with `PermissionPolicy`, not with the
engine. `permission_policy()` compiles one tenant's tool rules into one:

```python
from grapharc.policy import PolicyEngine

engine = PolicyEngine.from_file("policy.toml")
compiled = engine.permission_policy(tenant="acme")

for tool in ("read_config", "write_file", "delete_bucket", "deploy_prod"):
    print(f"{tool:<14} {compiled.decide(tool).value}")

print("audit records left by the compiled object:", len(engine.audit))
```

```
read_config    allow
write_file     allow
delete_bucket  deny
deploy_prod    ask
audit records left by the compiled object: 0
```

**The last line is the honest part.** The compiled object cannot carry the
approver role, the rule id, or the audit record — it is the one deliberately
unrecorded path. A `Harness` holding it will ASK without knowing who to ask and
will decide without leaving a policy record. Pair it with `approval_router()`,
which comes back through the engine and does both.

---

## How do I answer "what was it allowed to do?" after the fact?

Give the engine an `AuditLog` with a path. Every decision lands as JSONL,
stamped with the policy version *and* digest it was decided under.

```python
import tempfile
from pathlib import Path

from grapharc.policy import AuditLog, PolicyEngine, parse_document

log = AuditLog(Path(tempfile.mkdtemp()) / "policy-audit.jsonl")
engine = PolicyEngine.from_file("policy.toml", audit=log)

ctx = {"run_id": "run-42", "node": "patch"}
engine.check_tool("write_file", tenant="acme", context=ctx)
engine.check_tool("delete_bucket", tenant="acme", context=ctx)
engine.check_edge("triage", "deploy", tenant="acme", context=ctx)

for row in log.read_jsonl():
    print(
        f"{row['resource']:<5} {row['subject']:<16} {row['effect']:<5} "
        f"rule={str(row['rule_id']):<25} v={row['policy_version']} "
        f"digest={row['policy_digest'][:8]} ctx={row['context']}"
    )

# The version string is a claim by the author; the digest is evidence.
edited = Path("policy.toml").read_text().replace('match = "delete_*"', 'match = "delete_tmp_*"')
print()
print("same version:", parse_document(edited).version == engine.version)
print("same digest: ", parse_document(edited).digest == engine.digest)
```

```
tool  write_file       allow rule=acme-may-write            v=2026-07-01 digest=b1d593fa ctx={'run_id': 'run-42', 'node': 'patch'}
tool  delete_bucket    deny  rule=no-deletes                v=2026-07-01 digest=b1d593fa ctx={'run_id': 'run-42', 'node': 'patch'}
edge  triage->deploy   deny  rule=nothing-routes-into-deploy v=2026-07-01 digest=b1d593fa ctx={'run_id': 'run-42', 'node': 'patch'}

same version: True
same digest:  False
```

**Why the digest matters.** Someone widened `no-deletes` from `delete_*` to
`delete_tmp_*` without touching `version`. The version string still says
`2026-07-01`; the digest does not. An audit record pins which bytes of policy
actually decided, so "it was allowed under policy 2026-07-01" is checkable
rather than asserted. The digest is taken over the *parsed* document, so
comments and formatting are not in it.

`context=` is yours: put the run id, node and session in it and the policy log
joins to the run trace. `AuditLog.entries(kind=..., tenant=...)` filters
in-memory; `read_jsonl()` re-reads the file. `max_entries` caps the in-memory
list only — the file keeps everything, which is why capping without a path
really does lose records and why it defaults to off.

---

## How do I make my TOML document govern admission?

It does not, by default. `AdmissionChecker` takes an `EdgePolicy` built in code;
`PolicyEngine.check_edge` answers over a document. **There is no shipped
compiler between them** — `permission_policy()` exists for tools and has no edge
equivalent. Here is the bridge, which is about fifteen lines:

```python
from grapharc.harness.permissions import Decision
from grapharc.planner import (
    AdmissionChecker,
    EdgePolicy,
    EdgeRule,
    NodeRegistry,
    NodeSpec,
    ProposedEdge,
    ProposedNode,
    Subgraph,
)
from grapharc.policy import PolicyEngine, ResourceKind
from grapharc.policy.document import EDGE_ARROW


def edge_policy_for(engine, tenant="default"):
    """Compile a document's `edge` rules for one tenant into an `EdgePolicy`."""
    document = engine.document
    if not document.declares_tenant(tenant):
        return EdgePolicy(rules=(), default=Decision.DENY)
    rules = []
    for rule in document.rules_for(ResourceKind.EDGE):
        if not rule.matches_tenant(tenant):
            continue
        source, _, target = rule.match.partition(EDGE_ARROW)
        rules.append(
            EdgeRule(action=rule.effect, source=source.strip(), target=target.strip())
        )
    return EdgePolicy(rules=tuple(rules), default=document.default)


engine = PolicyEngine.from_file("policy.toml")
compiled = edge_policy_for(engine, tenant="acme")

# The compiled object must answer exactly as the engine does.
pairs = [("triage", "deploy"), ("triage", "patch"), ("__start__", "triage"), ("patch", "deploy")]
for source, target in pairs:
    engine_says = engine.check_edge(source, target, tenant="acme").effect
    compiled_says = compiled.decide(source, target)
    edge = f"{source} -> {target}"
    print(
        f"{edge:<20} engine={engine_says.value:<6} "
        f"compiled={compiled_says.value:<6} agree={engine_says is compiled_says}"
    )

gate = AdmissionChecker(
    registry=NodeRegistry([NodeSpec(name="triage"), NodeSpec(name="deploy")]),
    edge_policy=compiled,
)
result = gate.check(
    Subgraph(
        nodes=(ProposedNode(name="triage"), ProposedNode(name="ship", kind="deploy")),
        edges=(ProposedEdge(source="triage", target="ship"),),
    )
)
print()
print(result.status.value)
print(result.rejections[0].render())
```

```
triage -> deploy     engine=deny   compiled=deny   agree=True
triage -> patch      engine=allow  compiled=allow  agree=True
__start__ -> triage  engine=allow  compiled=allow  agree=True
patch -> deploy      engine=deny   compiled=deny   agree=True

rejected
[policy/edge_denied] triage -> ship: the edge policy denies this transition: kind 'triage' (proposed as 'triage') -> kind 'deploy' (proposed as 'ship') the decision is made on the registry kind, not the name you chose: renaming the node will not change it — route through a permitted kind
```

The two agree because they share their semantics: both tier deny → ask → allow
over the same ordered rule list, both fnmatch each endpoint separately, and both
fall through to the document default. The last two lines show it holding through
the gate — the document's `nothing-routes-into-deploy` rule refuses the edge, and
renaming the instance to `ship` does not help, because admission still resolves
the kind.

Same caveat as `permission_policy()`: the compiled object leaves no audit
record. If you need the decision logged, call `engine.check_edge` alongside it.
`tests/test_cookbook_governance.py` pins the equivalence across a matrix, so it
cannot drift silently.

---

## What this section does not give you

Stated plainly, because a governance layer that overstates itself is worse than
none:

1. **Arguments are not governed.** `ProposedNode.args` are never inspected.
   Admission authorises a kind. `forward_args=True` hands them to your factory
   unchecked, and that is your gate to build.
2. **`parent_depth` is on your honour.** The checker cannot observe how deep the
   run really is.
3. **Edge approvals are not routed.** `NEEDS_APPROVAL` tells you an edge needs a
   human; nothing carries it to one. The `ApprovalRouter` handles tools.
4. **Cycles across the boundary are invisible.** The acyclicity check sees only
   the topology inside the proposal.
5. **`known_nodes` and `Materializer` do not compose.** A proposal wired to a
   live node can be admitted, but the shipped materialiser builds a standalone
   graph and has nothing to attach it to.
6. **Nested proposals are admitted, not built.** `Materializer` refuses a
   `ProposedNode.subgraph` rather than silently dropping the inner proposal;
   there is no sub-materialisation step.
7. **Confinement covers routing, not conduct.** `UnadmittedTransition` bounds
   where a body may `goto`. What a body does inside itself belongs to the tool
   and permission planes.
8. **Frozen is not deeply frozen.** `Subgraph` and `ProposedNode` block field
   reassignment, but `args` is an ordinary dict whose contents can be mutated in
   place. `fingerprint()` is what detects that, by hashing content rather than
   trusting the reference — and `Materializer` checks it for you.
9. **No shipped edge-policy compiler.** The TOML document's `edge` rules do not
   reach `AdmissionChecker` on their own; the bridge above is fifteen lines you
   write and this section's tests pin.
10. **The spend ledger is in-process.** It does not survive a restart and is not
    shared between processes.

The parts that *are* enforced, and that every snippet above demonstrates: a
proposal cannot execute itself, an unregistered kind cannot run, a denied
transition cannot be renamed into an allowed one, an over-budget plan is refused
before its first node exists, and every decision — yes and no alike — is a
recorded event carrying the reason.
