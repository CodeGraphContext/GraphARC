"""Executable gate for `docs/cookbook/05-governance.md`.

The cookbook's only real promise is that every snippet in it runs exactly as
printed. This file enforces that mechanically rather than by review: each
```python block in the governance section is extracted, executed, and diffed
against the plain block that follows it. A snippet that stops working — or an
output block someone tidied by hand — fails here with a diff.

Two things are deliberately not left to the differ:

- The one snippet the doc says it did *not* run (the real-model swap, which
  needs a subscription) is asserted to be the *only* unpaired one, so a future
  edit cannot quietly add a second unexecuted snippet.
- The governance claims the section is built on — a proposal cannot execute
  itself, a denial keys on the registry kind and not the planner's instance
  name, arguments are not inspected — are pinned again as direct assertions.
  The differ proves the printed output is real; these prove it still means what
  the prose says it means.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from pydantic import BaseModel

from grapharc.harness.permissions import Decision
from grapharc.planner import (
    AdmissionChecker,
    AdmissionStatus,
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
from grapharc.policy import PolicyEngine
from grapharc.runtime.graph import START


class _EmptyState(BaseModel):
    """The narrowest state a materialised graph can be built over."""

DOC = Path(__file__).resolve().parents[1] / "docs" / "cookbook" / "05-governance.md"
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)

# The proposal id is a fresh uuid per proposal, and the doc says so inline.
_PROPOSAL_ID = re.compile(r"\bProposal [0-9a-f]{12}\b")


def _blocks() -> list[tuple[str, str, int]]:
    """(language, body, 1-based line number) for every fenced block, in order."""
    text = DOC.read_text(encoding="utf-8")
    return [
        (m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1)
        for m in FENCE.finditer(text)
    ]


def _snippets() -> tuple[list[dict], list[dict]]:
    """Split python blocks into (runnable, unpaired).

    A python block immediately followed by an unlabelled block is a snippet
    plus its expected output. One with anything else after it is a fragment the
    doc does not claim to have run.
    """
    blocks = _blocks()
    runnable: list[dict] = []
    unpaired: list[dict] = []
    for index, (lang, body, line) in enumerate(blocks):
        if lang != "python":
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if following is not None and following[0] == "":
            runnable.append({"code": body, "expected": following[1], "line": line})
        else:
            unpaired.append({"code": body, "line": line})
    return runnable, unpaired


def _policy_toml() -> str:
    toml = [body for lang, body, _line in _blocks() if lang == "toml"]
    assert len(toml) == 1, "the section documents exactly one policy document"
    return toml[0]


RUNNABLE, UNPAIRED = _snippets()


def _normalise(text: str) -> str:
    return _PROPOSAL_ID.sub("Proposal <id>", text).rstrip("\n")


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A scratch cwd holding the `policy.toml` the snippets load by name."""
    (tmp_path / "policy.toml").write_text(_policy_toml(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_section_exists_and_has_snippets():
    assert DOC.is_file()
    assert len(RUNNABLE) >= 15


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "snippet", RUNNABLE, ids=[f"line{s['line']}" for s in RUNNABLE]
)
def test_every_snippet_prints_exactly_what_the_doc_says(snippet, workdir):
    buffer = io.StringIO()
    namespace: dict = {"__name__": "__cookbook__"}
    with redirect_stdout(buffer):
        exec(compile(snippet["code"], f"{DOC}:{snippet['line']}", "exec"), namespace)
    assert _normalise(buffer.getvalue()) == _normalise(snippet["expected"])


def test_the_only_snippet_the_doc_did_not_run_is_the_real_model_one():
    """Exactly one fragment, and the doc says in words that it was not executed."""
    assert len(UNPAIRED) == 1
    fragment = UNPAIRED[0]
    compile(fragment["code"], "<fragment>", "exec")  # still has to be valid Python
    assert "get_model(" in fragment["code"]
    assert "**This snippet was not executed**" in DOC.read_text(encoding="utf-8")


def test_no_snippet_reaches_a_live_backend():
    """The scripted double is the only model in the section."""
    for snippet in RUNNABLE:
        assert "get_model(" not in snippet["code"]
        assert "ClaudeCodeCLIChatModel" not in snippet["code"]
        assert "OpenRouter" not in snippet["code"]


# -- the claims the prose makes, pinned independently of the printed output ----


def test_admission_never_calls_a_node_factory():
    def explode(*args, **kwargs):  # pragma: no cover - calling it is the failure
        raise AssertionError("admission executed a node factory")

    registry = NodeRegistry([NodeSpec(name="fetch", factory=explode)])
    gate = AdmissionChecker(
        registry=registry, edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),))
    )
    assert gate.check(Subgraph(nodes=(ProposedNode(name="fetch"),))).admitted


def _deny_deploy_gate() -> AdmissionChecker:
    return AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="build"), NodeSpec(name="deploy")]),
        edge_policy=EdgePolicy(
            rules=(EdgeRule(action="deny", target="deploy"), EdgeRule(action="allow"))
        ),
    )


@pytest.mark.parametrize("instance_name", ["deploy", "helper", "totally_fine", "build2"])
def test_renaming_the_instance_never_evades_the_denial(instance_name):
    result = _deny_deploy_gate().check(
        Subgraph(
            nodes=(
                ProposedNode(name="build"),
                ProposedNode(name=instance_name, kind="deploy"),
            ),
            edges=(ProposedEdge(source="build", target=instance_name),),
        )
    )
    assert result.status is AdmissionStatus.REJECTED
    assert [r.code for r in result.rejections] == ["edge_denied"]
    # The rejection names the kind that was matched, and the name so it is findable.
    assert "kind 'deploy'" in result.rejections[0].detail
    assert instance_name in result.rejections[0].detail


def test_naming_an_instance_after_a_denied_kind_borrows_no_denial():
    """The rule runs in the other direction too: the name grants nothing either."""
    result = _deny_deploy_gate().check(
        Subgraph(
            nodes=(ProposedNode(name="build"), ProposedNode(name="deploy", kind="build")),
            edges=(ProposedEdge(source="build", target="deploy"),),
        )
    )
    assert result.admitted


def test_admission_does_not_inspect_arguments():
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="run_sql")]),
        edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
    )
    harmless = Subgraph(
        nodes=(ProposedNode(name="q", kind="run_sql", args={"query": "SELECT 1"}),),
        proposal_id="fixed",
    )
    hostile = Subgraph(
        nodes=(ProposedNode(name="q", kind="run_sql", args={"query": "DROP TABLE t"}),),
        proposal_id="fixed",
    )
    assert gate.check(harmless).admitted
    assert gate.check(hostile).admitted
    # The gate could not tell them apart; the fingerprint still can.
    assert harmless.fingerprint() != hostile.fingerprint()


def test_forwarding_a_planners_args_is_opt_in():
    """The doc says `NodeBuild.args` is empty by default. Both branches pinned."""
    seen: list[dict] = []

    def factory(build):
        seen.append(dict(build.args))
        return lambda state: {}

    proposal = Subgraph(
        nodes=(ProposedNode(name="q", kind="run_sql", args={"query": "DROP TABLE t"}),),
        edges=(ProposedEdge(source=START, target="q"),),
    )
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="run_sql", factory=factory)]),
        edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
    )
    result = gate.check(proposal)
    for forward in (False, True):
        Materializer(
            registry=NodeRegistry([NodeSpec(name="run_sql", factory=factory)]),
            state_schema=_EmptyState,
            forward_args=forward,
        ).materialize(result, proposal)
    assert seen == [{}, {"query": "DROP TABLE t"}]


def test_what_runs_is_checked_against_what_was_admitted():
    """A genuine approval cannot be reused for an edited proposal."""
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="fetch", factory=lambda b: lambda s: {})]),
        edge_policy=EdgePolicy(rules=(EdgeRule(action="allow"),)),
    )
    proposal = Subgraph(
        nodes=(ProposedNode(name="fetch"),),
        edges=(ProposedEdge(source=START, target="fetch"),),
        proposal_id="pinned",
    )
    result = gate.check(proposal)
    edited = Subgraph(
        nodes=proposal.nodes,
        edges=proposal.edges,
        rationale="and delete the backups",
        proposal_id="pinned",  # the id alone is not enough
    )
    materializer = Materializer(
        registry=NodeRegistry([NodeSpec(name="fetch", factory=lambda b: lambda s: {})]),
        state_schema=_EmptyState,
    )
    materializer.materialize(result, proposal)  # the one it authorised builds
    with pytest.raises(NotAdmitted):
        materializer.materialize(result, edited)


def test_a_rejected_proposal_cannot_be_materialised_at_all():
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="fetch", factory=lambda b: lambda s: {})]),
        edge_policy=EdgePolicy(rules=(EdgeRule(action="deny"),)),
    )
    proposal = Subgraph(
        nodes=(ProposedNode(name="fetch"),),
        edges=(ProposedEdge(source=START, target="fetch"),),
    )
    result = gate.check(proposal)
    assert not result.admitted
    with pytest.raises(NotAdmitted):
        Materializer(
            registry=NodeRegistry(
                [NodeSpec(name="fetch", factory=lambda b: lambda s: {})]
            ),
            state_schema=_EmptyState,
        ).materialize(result, proposal)


# -- the bridge the section documents, held to the engine's own answers --------


def _bridge_namespace(workdir) -> dict:
    """Execute the doc's own `edge_policy_for` block and hand back its globals."""
    block = next(s for s in RUNNABLE if "def edge_policy_for" in s["code"])
    namespace: dict = {"__name__": "__cookbook__"}
    with redirect_stdout(io.StringIO()):
        exec(compile(block["code"], "<bridge>", "exec"), namespace)
    return namespace


EDGE_MATRIX = [
    ("triage", "deploy"),
    ("deploy", "triage"),
    ("triage", "patch"),
    ("patch", "patch"),
    ("__start__", "triage"),
    ("patch", "__end__"),
    ("deploy", "deploy"),
    ("anything", "deployment"),
    ("shell_exec", "deploy"),
]


@pytest.mark.parametrize("tenant", ["default", "acme", "stranger"])
def test_the_compiled_edge_policy_answers_as_the_engine_does(tenant, workdir):
    """The bridge in the doc must not diverge from `PolicyEngine.check_edge`."""
    namespace = _bridge_namespace(workdir)
    engine = PolicyEngine.from_file("policy.toml")
    compiled = namespace["edge_policy_for"](engine, tenant=tenant)
    for source, target in EDGE_MATRIX:
        assert compiled.decide(source, target) is engine.check_edge(
            source, target, tenant=tenant
        ).effect, f"{source}->{target} for tenant {tenant!r}"


NODE_MATRIX = ["shell_exec", "shell_", "summarise", "deploy", "unheard_of"]


@pytest.mark.parametrize("tenant", ["default", "acme", "stranger"])
def test_the_shipped_node_compiler_answers_as_the_engine_does(tenant, workdir):
    """The section's node recipe uses `node_policy()`; it must not diverge."""
    engine = PolicyEngine.from_file("policy.toml")
    compiled = engine.node_policy(tenant=tenant)
    for kind in NODE_MATRIX:
        assert compiled.decide(kind) is engine.check_node(kind, tenant=tenant).effect, (
            f"{kind!r} for tenant {tenant!r}"
        )


def test_the_documents_node_rule_is_what_refuses_the_kind(workdir):
    """The claim the node recipe makes, pinned apart from its printed output."""
    engine = PolicyEngine.from_file("policy.toml")
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="shell_exec"), NodeSpec(name="summarise")]),
        edge_policy=engine.edge_policy(),
        node_policy=engine.node_policy(),
    )

    refused = gate.check(Subgraph(nodes=(ProposedNode(name="helper", kind="shell_exec"),)))
    admitted = gate.check(Subgraph(nodes=(ProposedNode(name="summarise"),)))

    assert [r.code for r in refused.rejections] == ["node_denied"]
    # The rule's own reason travels into the rejection, not just its effect.
    assert "a shell node is an unbounded tool" in refused.rejections[0].detail
    assert admitted.admitted


def test_an_undeclared_tenant_compiles_to_a_policy_that_denies_everything(workdir):
    namespace = _bridge_namespace(workdir)
    engine = PolicyEngine.from_file("policy.toml")
    compiled = namespace["edge_policy_for"](engine, tenant="stranger")
    assert compiled.rules == ()
    assert compiled.default is Decision.DENY
