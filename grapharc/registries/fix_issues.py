"""A listener/fixer registry: find every issue, fix each one, in parallel.

The job this module ships is "fix all the issues in this repository", and its
shape cannot be pre-authored: how many fixers a round needs depends on what the
listener found. So the fan-out is decided by the planner, round over round, and
every widening re-enters the admission gate — a round may carry a continuing
`scan_issues` *and* one `fix_one` per already-found issue, which is how the
listener and the fixers overlap without a bus, a queue, or any channel outside
declared state.

The kinds are roles, not operations:

- `scan_issues` — the listener. Read-only tools; appends one issue per entry.
- `fix_one` — the fixer. File tools; **mutating**, so the default policy denies
  every edge into it. One instance per issue — a proposal names `fix_1`,
  `fix_2`, … all of this kind, all edged from the same predecessor, so they
  execute in one superstep, concurrently.
- `verify_fixes` — read-only check; reports what is still wrong.
- `report` — toolless; writes the human-facing outcome that completes the run.

Every list field on `FixState` carries an `operator.add` reducer, because two
fixers finishing together must merge rather than collide — the same lesson
`stdlib.WorkState` records.

Registered but denied is the point, here as in the incident demo: `fix_one`
exists because fixing is the job, and the default edge policy still refuses it
until an operator says otherwise — one `EdgeRule`, or one line of TOML. The
scripted rehearsal below runs against that default on purpose, so the free path
shows the refusal and the honest report, not a simulation of consent.

Given the bare scripted stand-in model (`--scripted`), every kind gets a
deterministic stand-in body, the incident demo's pattern — the rehearsal is
about the gate, not the agents. A *subclass* (a tool-calling test double) gets
the real agent phases, so tests can drive them. A real model gets `AgentNode`
phases with the allowlists fixed in `TOOLS_FOR`. With no model at all, nothing
is registered: a proposal naming these kinds then fails admission as
`unregistered_node`, which tells the truth, instead of passing the gate and
failing at materialisation.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from grapharc.harness.permissions import Decision

READ_ONLY_TOOLS = ("read_file", "list_dir", "glob", "grep")
WRITE_TOOLS = ("read_file", "list_dir", "glob", "grep", "edit_file", "write_file")

#: Tools each kind may call, fixed here by an operator. The fixer gets file
#: tools and nothing that runs commands; the listener and the verifier cannot
#: write at all, which is what makes running them in parallel with fixers safe
#: to allow by default.
TOOLS_FOR: dict[str, tuple[str, ...]] = {
    "scan_issues": READ_ONLY_TOOLS,
    "fix_one": WRITE_TOOLS,
    "verify_fixes": READ_ONLY_TOOLS,
    "report": (),
}


class FixState(BaseModel):
    """One state contract for the whole run, however wide the fan-out gets.

    Every list carries an `operator.add` reducer: K fixers finishing in one
    superstep each return only what they add, and LangGraph merges. A plain
    list here dies with `InvalidUpdateError` the first time two fixers land
    together — which is the normal case, not the edge case.
    """

    goal: str = ""
    #: What the listener found, one issue per entry.
    issues: Annotated[list[str], operator.add] = []
    #: What the fixers did, one entry per fix.
    fixes: Annotated[list[str], operator.add] = []
    #: What the verifier still objects to.
    failures: Annotated[list[str], operator.add] = []
    #: The human-facing outcome; writing here is what completes the run.
    notes: Annotated[list[str], operator.add] = []


#: What each kind may write. A kind absent here may write nothing —
#: `Materializer` enforces that, not the node.
WRITES: dict[str, set[str]] = {
    "scan_issues": {"issues"},
    "fix_one": {"fixes"},
    "verify_fixes": {"failures"},
    "report": {"notes"},
}

STATE_SCHEMA = FixState
AGENT_KINDS = ("scan_issues", "fix_one", "verify_fixes", "report")


class FixAssignment(BaseModel):
    """The one argument a `fix_one` proposal must carry: its issue, verbatim.

    Declared as the kind's `args_schema`, so admission validates every fixer's
    assignment and the fingerprint an approval binds to includes who was
    assigned what. `extra="forbid"` because an argument nobody declared is an
    argument nobody checked. The text feeds the fixer's *prompt*, never a tool
    call — each tool call is still gated per call — and the length bound is
    also a performance bound: state is deep-copied per node entry.
    """

    model_config = ConfigDict(extra="forbid")

    issue: str = Field(min_length=1, max_length=2000)

#: The one kind that can change files, and therefore the one the default
#: policy denies. Read by the policy generator through `RegistryBundle`.
MUTATING_KINDS = ("fix_one",)

#: The single field each kind appends its output to.
OUTPUT_FIELD: dict[str, str] = {
    "scan_issues": "issues",
    "fix_one": "fixes",
    "verify_fixes": "failures",
    "report": "notes",
}

_PROMPTS = {
    "scan_issues": (
        "Find concrete issues relevant to the goal using the read-only tools. "
        "Report one issue per line, each specific enough that a fixer given "
        "only that line could act on it. Do not attempt to fix anything — you "
        "have no tools that can."
    ),
    "fix_one": (
        "Fix exactly the ONE issue you were assigned, with the file tools, "
        "making the smallest change that resolves it. Quote the assigned "
        "issue and state which paths you changed. Leave every other issue "
        "alone: each has its own fixer."
    ),
    "verify_fixes": (
        "Check each reported fix against its issue using the read-only tools. "
        "Report one line per problem that remains, quoting what you read. If "
        "everything holds, say so in one line."
    ),
    "report": (
        "Summarise for a human reader: what was found, what was fixed, what "
        "remains, from the state you were given. You have no tools; do not "
        "claim to have checked anything."
    ),
}


def unfixed(state: Any) -> list[str]:
    """The issues no fix entry mentions — the diff that drives the next round.

    Matching is by containment of the issue's exact text, because that text is
    what a fixer was told to take; a fixer that cannot quote its issue did not
    fix it. Deterministic code, never a model.
    """
    issues = getattr(state, "issues", None) or []
    fixes = getattr(state, "fixes", None) or []
    return [issue for issue in issues if not any(issue in fix for fix in fixes)]


def _context(state: FixState) -> str:
    """The prompt a phase is given: the goal, plus the run's ledger so far."""
    parts = [f"Goal: {state.goal}"]
    if state.issues:
        parts.append("Issues found:")
        parts += [f"- {line}" for line in state.issues]
    if state.fixes:
        parts.append("Fixes so far:")
        parts += [f"- {line}" for line in state.fixes]
    remaining = unfixed(state)
    if remaining:
        parts.append("Still outstanding:")
        parts += [f"- {line}" for line in remaining]
    if state.failures:
        parts.append("Verification found:")
        parts += [f"- {line}" for line in state.failures]
    return "\n".join(parts)


#: The listener's canned findings for the scripted rehearsal. Three, so the
#: fan-out story has a width worth drawing.
_SCRIPTED_ISSUES = (
    "issue: the changelog names a version the package does not ship",
    "issue: a docstring promises a flag that does not exist",
    "issue: a test asserts on a message the code no longer prints",
)


def _scripted_factory(kind: str) -> Any:
    """Deterministic stand-in bodies for the spend-free rehearsal.

    Same contract as the agent phases — same writes, same fields — with canned
    content, so `--scripted` exercises admission, refusal, reducers and the
    trace without a model. The fixer's stand-in takes the first outstanding
    issue, which is all it *can* do before proposals carry an assignment; the
    real per-issue assignment arrives with admission-checked args.
    """

    def factory(spec: Any) -> Any:
        assigned = (getattr(spec, "args", None) or {}).get("issue", "")

        def body(state: FixState) -> dict:
            if kind == "scan_issues":
                from grapharc.runtime.fanout import dedupe

                return {"issues": dedupe(list(_SCRIPTED_ISSUES), key=str)}
            if kind == "fix_one":
                if assigned:
                    taken = assigned
                else:
                    remaining = unfixed(state)
                    taken = remaining[0] if remaining else "nothing left to fix"
                return {"fixes": [f"fixed: {taken}"]}
            if kind == "verify_fixes":
                return {"failures": [f"unfixed: {line}" for line in unfixed(state)]}
            remaining = unfixed(state)
            line = (
                f"{len(state.issues)} issue(s) found, {len(state.fixes)} fixed, "
                f"{len(remaining)} outstanding"
            )
            if remaining and not state.fixes:
                line += (
                    " — fixing was not admitted; enable it with one edge rule "
                    "into fix_one"
                )
            return {"notes": [line]}

        body.writes = WRITES[kind]
        return body

    return factory


def _split_issues(text: str) -> list[str]:
    """One issue per non-empty line, list markers stripped, order kept."""
    from grapharc.runtime.fanout import dedupe

    lines = [line.strip().lstrip("-•*").strip() for line in (text or "").splitlines()]
    return dedupe([line for line in lines if line], key=str.casefold)


def _agent_factory(model: Any, harness_for: Any, kind: str) -> Any:
    """Build one agent-backed role, its tool allowlist fixed in `TOOLS_FOR`.

    The listener's body parses the agent's report into one issue per entry —
    the planner fans out over *entries*, so a report that arrived as one blob
    would collapse the whole run to a single fixer.
    """
    field = OUTPUT_FIELD[kind]

    def factory(spec: Any) -> Any:
        from grapharc.harness import AgentNode

        node = AgentNode(
            model,
            harness_for(TOOLS_FOR[kind]),
            name=kind,
            system_prompt=_PROMPTS[kind],
        )
        # The admission-validated assignment, when this kind carries one. It
        # reaches the prompt and nothing else; every tool call the fixer makes
        # with it is still gated per call.
        assigned = (getattr(spec, "args", None) or {}).get("issue", "")

        def body(state: FixState, ctx: Any) -> dict:
            prompt = _context(state)
            if assigned:
                prompt += f"\nYour assigned issue — fix this one and no other:\n{assigned}"
            result = node.run(prompt, ctx)
            reason = result.termination_reason.value
            if kind == "scan_issues" and reason == "target_met":
                return {field: _split_issues(result.output)}
            line = result.output if reason == "target_met" else f"[{reason}] {result.output}"
            return {field: [line]}

        body.writes = {field}
        return body

    return factory


def _is_bare_scripted(model: Any) -> bool:
    """Exactly the scripted stand-in, not a subclass.

    The CLI's `--scripted` hands `build_registry` the very `ScriptedChatModel`
    it built from `scripted_planner_replies`, and canned planner JSON is no
    way to drive an agent loop — so that model, precisely, gets stand-in
    bodies. A subclass is a test double that chose to implement more (the
    stdlib tests' tool-calling double), and it gets the real phases.
    """
    from grapharc.testing import ScriptedChatModel

    return type(model) is ScriptedChatModel


def build_registry(
    model: Any = None, *, harness_for: Any = None, workspace: Any = None
) -> Any:
    """The listener/fixer registry. Kinds exist only when a model does.

    With no model, nothing is registered — a proposal naming `scan_issues`
    then fails admission as `unregistered_node` with the (empty) list of what
    is allowed, instead of passing the gate and failing later. With the bare
    scripted stand-in, every kind gets a deterministic body. With anything
    else, `AgentNode` phases with the `TOOLS_FOR` allowlists.

    `workspace` confines the agent kinds' tools to one directory; ignored when
    the caller supplies its own `harness_for`, which already decided that.
    """
    from grapharc.planner import CostEstimate, NodeRegistry, NodeSpec
    from grapharc.stdlib import default_harness

    if model is None:
        return NodeRegistry([])
    scripted = _is_bare_scripted(model)
    harness_for = harness_for or (lambda tools: default_harness(tools, workspace))
    described = {
        "scan_issues": "find issues; one per entry, ready to hand to a fixer",
        "fix_one": (
            "fix exactly one outstanding issue; propose one instance PER issue"
        ),
        "verify_fixes": "check the fixes and report what still fails",
        "report": (
            "write the final human-facing report; the run is complete once "
            "this has run"
        ),
    }
    specs = []
    for kind, tokens in (
        ("scan_issues", 5000),
        ("fix_one", 6000),
        ("verify_fixes", 4000),
        ("report", 1500),
    ):
        specs.append(
            NodeSpec(
                name=kind,
                description=described[kind],
                factory=(
                    _scripted_factory(kind)
                    if scripted
                    else _agent_factory(model, harness_for, kind)
                ),
                worst_case=CostEstimate(iterations=1, tokens=tokens),
                # Every fixer proposal must say which issue it takes, and
                # admission checks it — the assignment is part of what the
                # fingerprint binds, so an approval covers who fixes what.
                args_schema=FixAssignment if kind == "fix_one" else None,
            )
        )
    return NodeRegistry(specs)


def default_edge_policy() -> Any:
    """Allow every transition except one into the kind that changes files.

    Scanning and verifying are read-only and run without a decision; fixing is
    the job and still needs one. Turning it on is one rule, not a code change.
    """
    from grapharc.planner import EdgePolicy, EdgeRule

    rules = [EdgeRule(action=Decision.DENY, target=kind) for kind in MUTATING_KINDS]
    rules.append(EdgeRule(action=Decision.ALLOW))
    return EdgePolicy(rules=tuple(rules))


def goal_met(state: Any) -> bool:
    """Done when the report landed in `notes`. Deterministic, never a model."""
    return len(getattr(state, "notes", ()) or ()) >= 1


def _observe(state: Any) -> str:
    """What the planner sees between rounds: the unfixed diff, above all.

    The diff is the whole scheduling signal — the planner proposes one fixer
    per line of it, so a round that shows the full ledger but not the diff
    would make the model re-derive the one list this module can compute.
    """
    issues = getattr(state, "issues", None) or []
    fixes = getattr(state, "fixes", None) or []
    failures = getattr(state, "failures", None) or []
    remaining = unfixed(state)
    parts = [
        f"issues found: {len(issues)}, fixed: {len(fixes)}, "
        f"outstanding: {len(remaining)}"
    ]
    if remaining:
        parts.append("outstanding issues (one fix_one instance each):")
        parts += [f"- {line}" for line in remaining[:10]]
    for line in failures[-3:]:
        parts.append(f"verification: {line}")
    return "\n".join(parts)


#: Told to the planner verbatim. The completion rule is deterministic code the
#: model cannot argue with, and the refusal path is described up front so a
#: denied round costs one replan, not the run.
_PLANNER_INSTRUCTIONS = (
    "The run is judged complete by deterministic code when a report lands in "
    "`notes`, and only `report` writes there. Start with `scan_issues`. Then "
    "propose one `fix_one` node PER outstanding issue — named fix_1, fix_2, … "
    "with kind fix_one — all taking an edge from the same predecessor so they "
    'execute in parallel. Every fix_one MUST carry "args": {"issue": "<the '
    'issue text, verbatim from the outstanding list>"}; a fixer without its '
    "assignment is rejected at admission. A `scan_issues` may run in the same "
    "round as fixers for issues already found. Finish with `verify_fixes`, "
    "then `report`. Edges into `fix_one` may be denied by policy: if a round "
    "is rejected for that, replan without fixers and still finish with "
    "`verify_fixes` and a `report` that says what was refused."
)


def scripted_planner_replies() -> list[str]:
    """The rehearsal: an eager fix refused, an honest replan, a true report.

    Round 1 proposes a fixer before anything was scanned — and wires an edge
    into `fix_one`, which the default policy denies, so the round is refused
    and nothing runs. Round 2 replans to the listener alone. Round 3 verifies
    and reports, and the report says out loud that fixing was not admitted.
    Spend-free, deterministic, and the refusal is the demonstration.
    """
    import json

    from grapharc.runtime.graph import END, START

    return [
        json.dumps(
            {
                "nodes": [
                    {"name": "scan_issues"},
                    {"name": "fix_1", "kind": "fix_one"},
                ],
                "edges": [
                    {"source": START, "target": "scan_issues"},
                    {"source": "scan_issues", "target": "fix_1"},
                    {"source": "fix_1", "target": END},
                ],
                "rationale": "scan, then fix whatever turns up",
            }
        ),
        json.dumps(
            {
                "nodes": [{"name": "scan_issues"}],
                "edges": [
                    {"source": START, "target": "scan_issues"},
                    {"source": "scan_issues", "target": END},
                ],
                "rationale": "fixing was refused; scan first and replan",
            }
        ),
        json.dumps(
            {
                "nodes": [{"name": "verify_fixes"}, {"name": "report"}],
                "edges": [
                    {"source": START, "target": "verify_fixes"},
                    {"source": "verify_fixes", "target": "report"},
                    {"source": "report", "target": END},
                ],
                "rationale": "verify what stands and report honestly",
            }
        ),
    ]


def build_loop(
    model: Any,
    *,
    edge_policy: Any = None,
    node_policy: Any = None,
    trace: Any = None,
    budget: Any = None,
    limits: Any = None,
    registry: Any = None,
    state_schema: Any = None,
    writes: dict[str, set[str]] | None = None,
    approval: Any = None,
) -> Any:
    """Assemble the listener/fixer loop; same shape as stdlib's, its own goal.

    Read by `grapharc plan --registry grapharc.registries.fix_issues:build_registry`
    through `RegistryBundle.build_loop`, which is what lets this job own its
    completion rule and its observer — the unfixed diff — instead of
    inheriting another module's.
    """
    from grapharc.planner import (
        AdmissionChecker,
        AdmissionLimits,
        GovernedLoop,
        Materializer,
        PlannerNode,
    )

    registry = registry or build_registry(model)
    registry.freeze()
    # One policy object, disclosed to the planner and applied by the checker —
    # resolving the default twice would describe one object and enforce another.
    edge_policy = edge_policy or default_edge_policy()
    return GovernedLoop(
        planner=PlannerNode(
            model,
            name="fix_issues",
            catalog=registry.catalog(),
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
            instructions=_PLANNER_INSTRUCTIONS,
        ),
        checker=AdmissionChecker(
            registry=registry,
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
            limits=AdmissionLimits(require_entry=True),
        ),
        materializer=Materializer(
            registry=registry,
            state_schema=state_schema or FixState,
            writes=writes if writes is not None else WRITES,
            trace=trace,
        ),
        budget=budget,
        limits=limits,
        trace=trace,
        name="fix_issues_loop",
        goal_reached=goal_met,
        observe=_observe,
        approval=approval,
    )


__all__ = [
    "AGENT_KINDS",
    "MUTATING_KINDS",
    "OUTPUT_FIELD",
    "READ_ONLY_TOOLS",
    "STATE_SCHEMA",
    "TOOLS_FOR",
    "WRITES",
    "WRITE_TOOLS",
    "FixAssignment",
    "FixState",
    "build_loop",
    "build_registry",
    "default_edge_policy",
    "goal_met",
    "scripted_planner_replies",
    "unfixed",
]
