"""A shipped registry of general-purpose node kinds.

This exists so that zero-config runs are possible **without generating code**.
An LLM asked to set up a workflow *selects* from these kinds; it never authors a
node body. That distinction is the whole safety argument: the registry is the
only source of a body, so if a model could write registries it could write code,
and the gate would be checking a list the gated thing wrote.

**The kinds are phases, not operations.** There is no `read_file` node, because a
node cannot know *which* file — admission authorises a kind and `Materializer`
drops a proposal's `args` by default, so a node that needed parameters from the
proposal would be a node that needed the sharp edge turned on. Instead, file
access happens *inside* an agent-backed phase, where `read_file` is a **tool**
the model calls with arguments the permission layer checks per call. Coarse
nodes, fine-grained tools.

Two tiers:

- **Deterministic** — `collect_context`, `checkpoint`. No model, always
  available, and the reason a no-model run is not useless.
- **Agent-backed** — `investigate`, `apply_change`, `verify`, `summarize`. Each
  is an `AgentNode` with a *fixed tool allowlist* baked in here, by an operator,
  not chosen by the model at run time. `apply_change` is the only one that can
  write, which is what makes it the one worth denying by default.

All but one of the agent kinds want a **tool-calling** backend, because that is
the only way GraphARC can run the loop itself and gate each call. Given the
Claude CLI — which has no tool-calling wire format — `AgentNode` delegates the
whole loop to Claude Code instead, warning at construction and marking the
trace: the fixed allowlists described above do not apply to a delegated run,
because the tools are Claude Code's rather than this registry's. `summarize` is
the exception either way — it is toolless by design, so it binds nothing and
runs anywhere.

Registered but denied is the interesting state: **given a model**, `apply_change`
is in the registry because changing files is a real capability, and the default
policy denies every edge into it. So a refusal is a decision about a transition
rather than a hole in the allowlist, and turning it on is one line in a TOML file
rather than a code change. With **no** model it is not registered at all — see
`build_registry` — so it is refused twice over, as `unregistered_node` *and*
`edge_denied`. Two paragraphs here used to disagree about which.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel

from grapharc.harness.permissions import Decision

#: Tools each agent-backed kind may call. Authored here, on purpose: a phase's
#: reach is an operator decision, and a model that picked its own tools would
#: have no permissions at all.
READ_ONLY_TOOLS = ("read_file", "list_dir", "glob", "grep")
WRITE_TOOLS = ("read_file", "list_dir", "glob", "grep", "edit_file", "write_file")

#: Exactly which tools each phase gets. An explicit map rather than inferred from
#: "is it a mutating kind", because a phase whose prompt says it has no tools and
#: whose harness gives it four is a phase that lies to its own model.
TOOLS_FOR: dict[str, tuple[str, ...]] = {
    "investigate": READ_ONLY_TOOLS,
    "verify": READ_ONLY_TOOLS,
    "apply_change": WRITE_TOOLS,
    # Toolless on purpose. `AgentNode` skips `bind_tools` when nothing is
    # visible, so this is the one agent-backed kind that runs on a backend with
    # no tool-calling wire format — the Claude CLI subscription included.
    "summarize": (),
}


class WorkState(BaseModel):
    """The state contract for stdlib graphs.

    Deliberately small and append-only. Every list field carries an `operator.add`
    reducer, which is what actually makes concurrency safe: without one,
    LangGraph refuses two writes to the same key in one superstep
    (`InvalidUpdateError`), so a fan-out of parallel phases — the shape a
    planner proposes for any decomposable job — died at execution instead of
    merging. Phases therefore return only what they *add*; the reducer appends
    it, and two phases finishing together cannot clobber each other.
    """

    goal: str = ""
    #: What was learned. Read by later phases as their context.
    findings: Annotated[list[str], operator.add] = []
    #: Narrative for a human reading the result.
    notes: Annotated[list[str], operator.add] = []


#: The single list field each agent-backed phase appends its report to.
#: One field per phase, deliberately: an `AgentNode` returns prose, and two
#: phases writing the same scalar would silently overwrite each other. Appending
#: to a list makes the accumulation the state contract rather than a convention.
OUTPUT_FIELD: dict[str, str] = {
    "investigate": "findings",
    "verify": "findings",
    "apply_change": "notes",
    "summarize": "notes",
}

#: What each kind may write. A kind absent from this map may write nothing —
#: `Materializer` enforces that, and it is why adding a kind here is a decision.
WRITES: dict[str, set[str]] = {
    "collect_context": {"findings"},
    "checkpoint": set(),
    "investigate": {"findings"},
    "apply_change": {"notes"},
    "verify": {"findings"},
    "summarize": {"notes"},
}

STATE_SCHEMA = WorkState

#: Kinds that need no model. Named so callers can say what a no-model run can do.
DETERMINISTIC_KINDS = ("collect_context", "checkpoint")
AGENT_KINDS = ("investigate", "apply_change", "verify", "summarize")

#: The one kind that can change anything. Everything else is read-only, which is
#: what lets the default policy be "allow, except into here".
MUTATING_KINDS = ("apply_change",)

_PROMPTS = {
    "investigate": (
        "Investigate the goal using the read-only tools available. Report what "
        "you found as concise factual statements. Do not attempt to change "
        "anything — you have no tools that can."
    ),
    "apply_change": (
        "Make the smallest change that addresses the goal, using the file tools. "
        "State which paths you changed. Do not run commands; you cannot."
    ),
    "verify": (
        "Check whether the goal has been met, using the read-only tools. Report "
        "evidence for your conclusion, quoting what you read. If it has not been "
        "met, say precisely what is still wrong."
    ),
    "summarize": (
        "Summarise what happened for a human reader, from the state you were "
        "given. You have no tools; do not claim to have checked anything."
    ),
}


def _collect_context_for(workspace: Any = None) -> Any:
    """A factory listing *the run's* workspace, so a run has somewhere to
    start with no model. Parameterised rather than hard-coded to `Path.cwd()`:
    a `--workspace` that moved the tools but not this listing would report a
    directory the agents cannot reach — the one lie a context phase must not
    tell."""

    def factory(spec: Any) -> Any:
        def body(state: WorkState) -> dict:
            from pathlib import Path

            root = Path(workspace) if workspace is not None else Path.cwd()
            names = sorted(
                str(p.relative_to(root))
                for p in root.iterdir()
                if not p.name.startswith(".")
            )[:50]
            found = f"workspace contains {len(names)} visible entries: {', '.join(names)}"
            # Only the new item: `findings` carries an `operator.add` reducer,
            # so returning the accumulated list would append it to itself.
            return {"findings": [found]}

        body.writes = {"findings"}
        return body

    return factory


#: The unparameterised form, kept for direct importers.
_collect_context = _collect_context_for()


def _checkpoint(spec: Any) -> Any:
    """A node that does nothing, on purpose.

    Useful as a join point, and as the one kind that is always safe to admit —
    a topology can be exercised end to end without any phase doing work.
    """

    def body(state: WorkState) -> None:
        return None

    body.writes = set()
    return body


def _context(state: WorkState) -> str:
    """The prompt a phase is given: the goal, plus what earlier phases learned.

    Built here rather than left to `AgentNode.observe`, because the default maps
    one state field to the prompt and a later phase that could not see earlier
    findings would be re-doing their work every time.
    """
    parts = [f"Goal: {state.goal}"]
    if state.findings:
        parts.append("What earlier phases found:")
        parts += [f"- {line}" for line in state.findings]
    if state.notes:
        parts.append("Notes so far:")
        parts += [f"- {line}" for line in state.notes]
    return "\n".join(parts)


def _agent_factory(model: Any, harness_for: Any, kind: str) -> Any:
    """Build one agent-backed phase, with its tool allowlist fixed here.

    The `AgentNode` is driven through `run()` rather than used as the node body
    directly: its own state mapping writes `answer` and `termination_reason`,
    and this schema accumulates into lists instead. Wrapping it keeps the write
    set to the one field `WRITES` declares.
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

        def body(state: WorkState, ctx: Any) -> dict:
            result = node.run(_context(state), ctx)
            # The stop reason is carried with the text, so a phase that gave up
            # is not indistinguishable from one that finished.
            reason = result.termination_reason.value
            line = result.output if reason == "target_met" else f"[{reason}] {result.output}"
            # The reducer appends; returning the accumulated list would double it.
            return {field: [line]}

        body.writes = {field}
        return body

    return factory


def default_harness(tools: tuple[str, ...], workspace: Any = None) -> Any:
    """A `Harness` whose registry holds exactly `tools`, everything else denied.

    Two independent controls, deliberately: a tool that is not **registered**
    cannot be called even if a rule allowed it, and a registered tool still has
    to pass `deny -> ask -> allow`. Absence is the primary one, because it needs
    no rule to be right.

    `workspace` defaults to the working directory, and every core tool confines
    its own path arguments to it — the confinement is in the tool, not only in
    the executor, because `LocalExecutor` confines nothing.
    """
    from pathlib import Path

    from grapharc.harness import (
        Harness,
        LocalExecutor,
        PermissionPolicy,
        PermissionRule,
        ToolRegistry,
    )
    from grapharc.tools import core_tools

    registry = ToolRegistry()
    for spec in core_tools(Path(workspace or Path.cwd()), include=tools):
        registry.register(spec)
    # `literal`, not a bare pattern: these names come from a registry, not from
    # an operator writing globs, and an ALLOW rule is the one tier where a name
    # read as a pattern could grant more than was asked for.
    policy = PermissionPolicy(
        rules=[PermissionRule.literal(Decision.ALLOW, name) for name in tools],
        default=Decision.DENY,
    )
    return Harness(registry=registry, policy=policy, executor=LocalExecutor())


def build_registry(
    model: Any = None, *, harness_for: Any = None, workspace: Any = None
) -> Any:
    """The shipped registry. Deterministic kinds always; agent kinds with a model.

    With no model the agent-backed kinds are **left out entirely** rather than
    registered without a factory. A proposal naming one then fails admission with
    `unknown kind` and the list of what is allowed — which tells the truth. A
    registered-but-bodyless kind would instead pass the gate and fail at
    materialisation, which is a worse error at a later moment.

    `workspace` confines the agent kinds' tools — and the `collect_context`
    listing — to one directory instead of the process cwd. Ignored when the
    caller supplies its own `harness_for`, which already decided that.
    """
    from grapharc.planner import CostEstimate, NodeRegistry, NodeSpec

    harness_for = harness_for or (lambda tools: default_harness(tools, workspace))
    specs = [
        NodeSpec(
            name="collect_context",
            description="list the workspace so later phases have somewhere to start",
            factory=_collect_context_for(workspace),
            worst_case=CostEstimate(iterations=1),
        ),
        NodeSpec(
            name="checkpoint",
            description="a no-op join point; writes nothing",
            factory=_checkpoint,
            worst_case=CostEstimate(iterations=1),
        ),
    ]
    if model is not None:
        # The catalog is what a planner actually chooses from, so `summarize`
        # states the completion rule outright rather than leaving the model to
        # infer it from failed rounds.
        described = {
            "summarize": (
                "write the final human-facing report; the run is complete "
                "once this has run"
            ),
        }
        for kind, tokens in (
            ("investigate", 4000),
            ("apply_change", 6000),
            ("verify", 3000),
            ("summarize", 1500),
        ):
            specs.append(
                NodeSpec(
                    name=kind,
                    description=described.get(kind, _PROMPTS[kind].split(".")[0]),
                    factory=_agent_factory(model, harness_for, kind),
                    worst_case=CostEstimate(iterations=1, tokens=tokens),
                )
            )
    return NodeRegistry(specs)


def default_edge_policy() -> Any:
    """Allow every transition except one into a kind that can change files.

    The honest default for a zero-config run: investigating costs money and
    reveals things, changing files costs money and breaks things. So read-only
    work needs no decision and mutation needs one.
    """
    from grapharc.planner import EdgePolicy, EdgeRule

    rules = [EdgeRule(action=Decision.DENY, target=kind) for kind in MUTATING_KINDS]
    rules.append(EdgeRule(action=Decision.ALLOW))
    return EdgePolicy(rules=tuple(rules))


def catalog_for_prompt(model: Any = None) -> dict[str, str]:
    """Kind -> description, for handing to a model that must choose among them."""
    return build_registry(model).catalog()


#: Told to the planner verbatim. The completion rule below is deterministic
#: code the model cannot argue with; a model that was never told it burned its
#: rounds on investigate-only plans that could never finish, then stopped on
#: `no_progress` with the work done and the run reported failed.
_PLANNER_INSTRUCTIONS = (
    "The run is judged complete by deterministic code when a report lands in "
    "`notes`, and only `apply_change` and `summarize` write there. "
    "Investigation alone never finishes the run: end every plan with a "
    "`summarize` node that takes an edge from your other phases and writes "
    "the final report. A goal with several independent parts requires one "
    "`investigate` node PER part — named after its part, e.g. "
    "investigate_repo, investigate_docs — all taking an edge from the same "
    "predecessor so they execute in parallel, with `summarize` joining "
    "them. A single combined investigate node for a multi-part goal is "
    "wrong: it serializes work the goal asked to run simultaneously."
)


def goal_met(state: Any) -> bool:
    """Done when a report landed in `notes`.

    `apply_change` and `summarize` are the kinds that write `notes`, so this
    reads as "the run produced its human-facing outcome". Deterministic code,
    never a model — and defensive about the schema, so a custom state supplied
    through `--registry` cannot turn "am I done" into an AttributeError.
    """
    return len(getattr(state, "notes", ()) or ()) >= 1


def _observe(state: Any) -> str:
    """What the planner is shown between rounds: the goal's progress so far."""
    parts = []
    findings = getattr(state, "findings", None) or []
    notes = getattr(state, "notes", None) or []
    parts.append(f"findings so far: {len(findings)}")
    for line in findings[-5:]:
        parts.append(f"- {line}")
    parts.append(f"notes so far: {len(notes)}")
    for line in notes[-3:]:
        parts.append(f"- {line}")
    return "\n".join(parts)


def scripted_planner_replies() -> list[str]:
    """One model-free round: the two deterministic kinds, chained.

    With no model only `collect_context` and `checkpoint` are registered, so
    this is the only proposal the scripted path can make that admission will
    take — and it is what makes `grapharc plan --registry grapharc.stdlib:...`
    runnable spend-free as a smoke test.
    """
    import json

    from grapharc.runtime.graph import END, START

    return [
        json.dumps(
            {
                "nodes": [{"name": "collect_context"}, {"name": "checkpoint"}],
                "edges": [
                    {"source": START, "target": "collect_context"},
                    {"source": "collect_context", "target": "checkpoint"},
                    {"source": "checkpoint", "target": END},
                ],
                "rationale": "list the workspace, then join",
            }
        ),
        # Then nothing more: the deterministic kinds cannot write `notes`, so
        # the honest second answer is "no further work" — the loop stops
        # cleanly instead of burning rounds on an exhausted script.
        json.dumps({"nodes": [], "edges": [], "rationale": "no further work"}),
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
    """Assemble the stdlib loop: same shape as the incident demo's, its own goal.

    Read by `grapharc plan --registry grapharc.stdlib:build_registry` through
    `RegistryBundle.build_loop` — which is what lets this registry own its goal
    check and its observer instead of inheriting the demo's `len(notes) >= 3`.
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
    # One object, disclosed to the planner and applied by the checker. Resolving
    # the default twice would build two, and a prompt describing a different
    # object from the one the gate applies is worse than no disclosure at all.
    edge_policy = edge_policy or default_edge_policy()
    return GovernedLoop(
        planner=PlannerNode(
            model,
            name="stdlib",
            catalog=registry.catalog(),
            # The deny rules, in front of the model before round 1. Disclosure
            # only — the checker below is what refuses, whether or not the model
            # read this.
            edge_policy=edge_policy,
            node_policy=node_policy,
            trace=trace,
            instructions=_PLANNER_INSTRUCTIONS,
        ),
        checker=AdmissionChecker(
            registry=registry,
            edge_policy=edge_policy,
            # None unless a policy document declared node rules; the registry is
            # otherwise the only thing deciding which kinds may run.
            node_policy=node_policy,
            trace=trace,
            # Rounds are materialized standalone, so "can this actually run"
            # is part of admission here: a plan with no entry, or with nodes
            # nothing reaches, comes back as a rejection carrying a remedy
            # instead of a build failure the planner cannot act on.
            limits=AdmissionLimits(require_entry=True),
        ),
        materializer=Materializer(
            registry=registry,
            state_schema=state_schema or WorkState,
            writes=writes if writes is not None else WRITES,
            trace=trace,
        ),
        budget=budget,
        limits=limits,
        trace=trace,
        name="stdlib_loop",
        goal_reached=goal_met,
        # The planner should see what earlier phases learned, or every round
        # re-plans blind against the same goal text.
        observe=_observe,
        approval=approval,
    )


__all__ = [
    "AGENT_KINDS",
    "DETERMINISTIC_KINDS",
    "MUTATING_KINDS",
    "OUTPUT_FIELD",
    "READ_ONLY_TOOLS",
    "STATE_SCHEMA",
    "TOOLS_FOR",
    "WRITES",
    "WRITE_TOOLS",
    "WorkState",
    "build_loop",
    "build_registry",
    "catalog_for_prompt",
    "default_edge_policy",
    "default_harness",
    "goal_met",
    "scripted_planner_replies",
]
