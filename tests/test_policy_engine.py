"""ROADMAP §7 gates — the declarative policy layer.

What each group pins:

(a) **Loading** — a document that says something impossible (an `ask` naming no
    approver, an edge match with no arrow, a rule scoped to a tenant that does
    not exist) fails at load, loudly, rather than becoming a rule that quietly
    never fires.
(b) **Deny beats allow** — tiered, not positional, and a tenant-scoped allow is
    not an exception to a global deny.
(c) **Tenant scoping isolates** — one tenant's grant is not another's, and a
    tenant the document never declared is refused *and recorded*.
(d) **Approval routing** — an `ask` names a role, the router asks that role and
    no other, and every way the channel can fail resolves to "not approved".
(e) **Spend** — thresholds fire per call and cumulatively against a per-tenant
    ledger, and checking is not committing.
(f) **Audit and versioning** — every decision names the rule that caused it and
    the policy digest in force, so "what was it allowed to do" is answerable
    from the record after the document has moved on.
(g) **Composition** — the document compiles to a `PermissionPolicy` that gives
    the same answers, so a `Harness` enforces the document without knowing it
    exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import grapharc.policy
from grapharc.harness import (
    Decision,
    Harness,
    LocalExecutor,
    PermissionDenied,
    ToolRegistry,
    ToolSpec,
)
from grapharc.policy import (
    ApprovalRequest,
    AuditLog,
    PolicyEngine,
    PolicyError,
    ResourceKind,
    load_document,
    parse_document,
)

EXAMPLE_DOCUMENT = Path(grapharc.policy.__file__).parent / "example.toml"

POLICY_TOML = """
version = "1.4.0"
name = "test-policy"
default = "deny"
tenants = ["default", "acme", "globex"]

[[rule]]
id = "reads"
resource = "tool"
match = "read_*"
effect = "allow"

[[rule]]
id = "reads-again"
resource = "tool"
match = "read_file"
effect = "allow"

# Deliberately written *above* the deny it tries to escape: tiering, not
# document order, is what makes the deny win.
[[rule]]
id = "acme-delete-exception"
resource = "tool"
match = "delete_tmp"
effect = "allow"
tenant = "acme"

[[rule]]
id = "no-delete"
resource = "tool"
match = "delete_*"
effect = "deny"
reason = "destructive tools are never permitted"

[[rule]]
id = "deploy-asks-sre"
resource = "tool"
match = "deploy_*"
effect = "ask"
approver_role = "sre"
reason = "a deploy is a production change"

[[rule]]
id = "acme-writes"
resource = "tool"
match = "write_*"
effect = "allow"
tenant = "acme"

[[rule]]
id = "globex-pays"
resource = "tool"
match = "pay_*"
effect = "ask"
approver_role = "finance"
tenant = "globex"

[[rule]]
id = "no-shell"
resource = "node"
match = "shell_*"
effect = "deny"

[[rule]]
id = "nodes-ok"
resource = "node"
match = "*"
effect = "allow"

[[rule]]
id = "no-into-prod"
resource = "edge"
match = "*->deploy_prod"
effect = "deny"
reason = "the production deploy node is entered by an operator"

[[rule]]
id = "plan-fanout"
resource = "edge"
match = "plan*->run"
effect = "allow"

[[rule]]
id = "hard-cap"
resource = "spend"
effect = "deny"
over_usd = 50.0

[[rule]]
id = "model-cap"
resource = "spend"
match = "model"
effect = "deny"
over_usd = 2.0

[[rule]]
id = "globex-period-cap"
resource = "spend"
effect = "deny"
over_usd = 10.0
basis = "cumulative"
tenant = "globex"
reason = "globex is capped at $10 for the period"

[[rule]]
id = "over-a-dollar"
resource = "spend"
effect = "ask"
over_usd = 1.0
approver_role = "finance"

[[rule]]
id = "petty-cash"
resource = "spend"
effect = "allow"
"""


@pytest.fixture
def engine():
    return PolicyEngine.from_toml(POLICY_TOML)


def _echo(**kwargs):
    return kwargs


# ---- (a) loading -------------------------------------------------------------


def test_rules_load_with_their_fields_and_order():
    doc = parse_document(POLICY_TOML)
    assert doc.version == "1.4.0"
    assert doc.default is Decision.DENY
    assert [r.id for r in doc.rules_for(ResourceKind.TOOL)][:4] == [
        "reads",
        "reads-again",
        "acme-delete-exception",
        "no-delete",
    ]
    ask = next(r for r in doc.rules if r.id == "deploy-asks-sre")
    assert ask.effect is Decision.ASK
    assert ask.approver_role == "sre"
    assert ask.resource is ResourceKind.TOOL
    spend = next(r for r in doc.rules if r.id == "globex-period-cap")
    assert spend.over_usd == 10.0
    assert spend.spend_basis.value == "cumulative"
    assert spend.tenant == "globex"


def test_shipped_example_document_loads_and_decides():
    engine = PolicyEngine.from_file(EXAMPLE_DOCUMENT)
    assert engine.version == "2026-07-01"
    assert engine.check_tool("read_config", tenant="acme").allowed
    assert engine.check_tool("delete_everything", tenant="acme").denied
    assert engine.check_tool("deploy_prod", tenant="acme").approver_role == "sre"
    assert engine.check_edge("planner", "deploy_prod", tenant="acme").denied
    assert engine.check_node("shell_worker", tenant="acme").denied


def test_the_shipped_examples_node_rules_reach_the_admission_gate():
    """`no-shell-nodes` is the document's canonical "what may run" rule.

    It used to be inert — the compiler dropped every `node` rule (issue #66), so
    an operator who copied the shipped example got a policy that denied nothing.
    """
    from grapharc.planner import AdmissionChecker, NodeRegistry, NodeSpec, Subgraph
    from grapharc.planner.proposal import ProposedNode

    engine = PolicyEngine.from_file(EXAMPLE_DOCUMENT)
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="shell_exec"), NodeSpec(name="summarise")]),
        edge_policy=engine.edge_policy(tenant="acme"),
        node_policy=engine.node_policy(tenant="acme"),
    )

    refused = gate.check(Subgraph(nodes=(ProposedNode(name="helper", kind="shell_exec"),)))
    admitted = gate.check(Subgraph(nodes=(ProposedNode(name="summarise"),)))

    assert [r.code for r in refused.rejections] == ["node_denied"]
    assert "a shell node is an unbounded tool" in refused.rejections[0].detail
    assert admitted.admitted


def test_a_loaded_document_cannot_be_edited_underneath_a_decision():
    """The digest describes the document; the document must not move after it."""
    doc = parse_document(POLICY_TOML)
    with pytest.raises(ValidationError):
        doc.default = Decision.ALLOW
    with pytest.raises(AttributeError):
        doc.rules.append(doc.rules[0])  # a tuple, not a list
    with pytest.raises(ValidationError):
        doc.rules[0].effect = Decision.ALLOW


def test_load_document_reads_a_path(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(POLICY_TOML, encoding="utf-8")
    assert load_document(path).version == "1.4.0"


def test_missing_file_is_a_policy_error(tmp_path):
    with pytest.raises(PolicyError, match="cannot read policy document"):
        load_document(tmp_path / "absent.toml")


def test_malformed_toml_is_a_policy_error():
    with pytest.raises(PolicyError, match="not valid TOML"):
        parse_document('version = "1"\nthis is not toml')


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        # A version is the anchor of the audit record; there is no default.
        ('name = "x"\n', "version"),
        # extra="forbid" at both levels: a typo must not be silently ignored.
        ('version = "1"\nmispelled = true\n', "mispelled"),
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "allow"\n'
            'matchh = "*"\n',
            "matchh",
        ),
        # An ask that names nobody could only ever fail closed.
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "ask"\n',
            "must name an approver_role",
        ),
        # approver_role on a non-ask rule is a typo for effect = "ask".
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "allow"\n'
            'approver_role = "sre"\n',
            "only meaningful on an 'ask' rule",
        ),
        # An edge match has two sides.
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "edge"\neffect = "allow"\nmatch = "*"\n',
            "must be '<source>-><target>'",
        ),
        # Spend fields on a non-spend rule would silently never apply.
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "allow"\n'
            "over_usd = 5.0\n",
            "apply to a 'spend' rule",
        ),
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "spend"\neffect = "allow"\n'
            "over_usd = -1.0\n",
            "must not be negative",
        ),
        # Two rules with one id: the audit record could not name a single cause.
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "allow"\n'
            '[[rule]]\nid = "a"\nresource = "node"\neffect = "allow"\n',
            "duplicate rule id",
        ),
        # An unmatched request matches no rule, so a default of ask names no approver.
        ('version = "1"\ndefault = "ask"\n', "default = 'ask' is not allowed"),
        # A rule scoped to a tenant the document does not declare can never fire.
        (
            'version = "1"\ntenants = ["acme"]\n[[rule]]\nid = "a"\nresource = "tool"\n'
            'effect = "allow"\ntenant = "globex"\n',
            "could never fire",
        ),
        ('version = "1"\ntenants = ["acme", "acme"]\n', "duplicate entry in `tenants`"),
        ('version = "   "\n', "non-empty version"),
        ('version = "1"\ndefault = "maybe"\n', "default"),
        (
            'version = "1"\n[[rule]]\nid = "a"\nresource = "planet"\neffect = "allow"\n',
            "resource",
        ),
    ],
)
def test_a_document_that_cannot_mean_what_it_says_fails_to_load(body, needle):
    with pytest.raises(PolicyError) as excinfo:
        parse_document(body)
    assert needle in str(excinfo.value)


# ---- (b) deny beats allow ----------------------------------------------------


@pytest.mark.parametrize("deny_first", [True, False])
def test_deny_beats_allow_whatever_the_document_order(deny_first):
    """Tiered, not positional: moving the allow above the deny changes nothing."""
    deny = '[[rule]]\nid = "d"\nresource = "tool"\nmatch = "danger_*"\neffect = "deny"\n'
    allow = '[[rule]]\nid = "a"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
    order = (deny + allow) if deny_first else (allow + deny)
    engine = PolicyEngine.from_toml('version = "1"\n' + order)
    assert engine.check_tool("safe_read").effect is Decision.ALLOW
    decision = engine.check_tool("danger_rm")
    assert decision.effect is Decision.DENY
    assert decision.rule_id == "d"


def test_deny_beats_a_tenant_scoped_allow(engine):
    """`acme-delete-exception` allows `delete_tmp` for acme, and it still loses.

    There is no exception inside a deny — the same rule the harness follows.
    """
    decision = engine.check_tool("delete_tmp", tenant="acme")
    assert decision.effect is Decision.DENY
    assert decision.rule_id == "no-delete"
    assert decision.reason == "destructive tools are never permitted"


def test_ask_beats_allow(engine):
    """`deploy_*` asks; nothing allows it, but the tier order is what decides."""
    engine2 = PolicyEngine.from_toml(
        'version = "1"\n'
        '[[rule]]\nid = "a"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
        '[[rule]]\nid = "k"\nresource = "tool"\nmatch = "deploy_*"\neffect = "ask"\n'
        'approver_role = "sre"\n'
    )
    assert engine2.check_tool("read_x").effect is Decision.ALLOW
    ask = engine2.check_tool("deploy_prod")
    assert ask.effect is Decision.ASK
    assert ask.rule_id == "k"


def test_first_rule_of_the_winning_tier_is_the_recorded_cause(engine):
    """Two allow rules match `read_file`; the earlier one is named as the cause."""
    assert engine.check_tool("read_file").rule_id == "reads"


def test_unmatched_falls_to_the_document_default(engine):
    decision = engine.check_tool("something_nobody_thought_about")
    assert decision.effect is Decision.DENY
    assert decision.rule_id is None
    assert "document default is 'deny'" in decision.reason


def test_a_permissive_default_is_honoured():
    engine = PolicyEngine.from_toml('version = "1"\ndefault = "allow"\n')
    decision = engine.check_tool("anything")
    assert decision.effect is Decision.ALLOW
    assert decision.rule_id is None


def test_a_resource_kind_with_no_rules_denies(engine):
    """Tool rules say nothing about spend; spend is not thereby free."""
    tools_only = PolicyEngine.from_toml(
        'version = "1"\n[[rule]]\nid = "a"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
    )
    assert tools_only.check_tool("x").allowed
    assert tools_only.check_spend(0.01).denied
    assert tools_only.check_edge("a", "b").denied
    assert tools_only.check_node("n").denied


# ---- (c) tenant scoping ------------------------------------------------------


def test_tenant_scoping_isolates(engine):
    """`acme-writes` is acme's grant. globex gets the default, not the grant."""
    assert engine.check_tool("write_doc", tenant="acme").effect is Decision.ALLOW
    other = engine.check_tool("write_doc", tenant="globex")
    assert other.effect is Decision.DENY
    assert other.rule_id is None  # no rule matched, not "a rule denied it"


def test_a_tenant_scoped_ask_does_not_leak(engine):
    globex = engine.check_tool("pay_invoice", tenant="globex")
    assert globex.effect is Decision.ASK
    assert globex.approver_role == "finance"
    assert engine.check_tool("pay_invoice", tenant="acme").effect is Decision.DENY


def test_a_global_rule_applies_to_every_tenant(engine):
    for tenant in ("default", "acme", "globex"):
        assert engine.check_tool("read_file", tenant=tenant).effect is Decision.ALLOW


def test_an_undeclared_tenant_is_denied_and_recorded(engine):
    """The tenant arrives with the request, so a bad one is refused, not raised —
    and the refusal is in the audit log like any other decision."""
    decision = engine.check_tool("read_file", tenant="not-a-customer")
    assert decision.effect is Decision.DENY
    assert decision.rule_id is None
    assert "not declared" in decision.reason
    assert engine.audit.last() is decision


def test_tenant_patterns_scope_a_family():
    engine = PolicyEngine.from_toml(
        'version = "1"\n'
        '[[rule]]\nid = "eu"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
        'tenant = "eu-*"\n'
    )
    assert engine.check_tool("x", tenant="eu-berlin").allowed
    assert engine.check_tool("x", tenant="us-west").denied


# ---- nodes and edges ---------------------------------------------------------


def test_node_rules_gate_by_name(engine):
    assert engine.check_node("shell_worker").denied
    assert engine.check_node("summarise").allowed
    assert engine.check_node("shell_worker").rule_id == "no-shell"


def test_edge_decision_names_both_ends(engine):
    decision = engine.check_edge("plan", "deploy_prod")
    assert decision.subject == "plan->deploy_prod"
    assert decision.effect is Decision.DENY
    assert decision.rule_id == "no-into-prod"


def test_edge_sides_are_matched_separately():
    """A wildcard on the source must not swallow the arrow and eat the target.

    `plan*->run` against `plan->x->run`: matched as one string the `*` consumes
    `->x` and the pattern hits. Matched per side it does not, because the target
    is `x->run` and the target pattern is `run`.
    """
    engine = PolicyEngine.from_toml(
        'version = "1"\n[[rule]]\nid = "e"\nresource = "edge"\nmatch = "plan*->run"\n'
        'effect = "allow"\n'
    )
    assert engine.check_edge("plan_a", "run").allowed
    assert engine.check_edge("plan", "x->run").denied


# ---- (e) spend ---------------------------------------------------------------


def test_spend_thresholds_fire_by_tier(engine):
    small = engine.check_spend(0.50)
    assert (small.effect, small.rule_id) == (Decision.ALLOW, "petty-cash")
    medium = engine.check_spend(5.00)
    assert (medium.effect, medium.rule_id) == (Decision.ASK, "over-a-dollar")
    assert medium.approver_role == "finance"
    large = engine.check_spend(500.00)
    assert (large.effect, large.rule_id) == (Decision.DENY, "hard-cap")


def test_a_spend_threshold_is_exclusive(engine):
    """`over_usd = 1.0` means *over* a dollar: exactly a dollar is not over it."""
    assert engine.check_spend(1.00).effect is Decision.ALLOW
    assert engine.check_spend(1.01).effect is Decision.ASK


def test_spend_rules_can_be_scoped_to_a_category(engine):
    assert engine.check_spend(3.00, subject="model").rule_id == "model-cap"
    assert engine.check_spend(3.00, subject="tool").rule_id == "over-a-dollar"


def test_cumulative_spend_rule_reads_the_tenant_ledger(engine):
    """globex is capped at $10 for the period; the cap only bites once the
    committed total plus this request crosses it."""
    first = engine.check_spend(5.00, tenant="globex")
    assert first.effect is Decision.ASK  # 0 + 5 is under the cap
    engine.record_spend(8.00, tenant="globex")
    assert engine.committed_usd("globex") == 8.00
    blocked = engine.check_spend(5.00, tenant="globex")
    assert blocked.effect is Decision.DENY
    assert blocked.rule_id == "globex-period-cap"
    assert blocked.request["considered_usd"] == pytest.approx(13.00)
    assert blocked.request["committed_usd"] == pytest.approx(8.00)


def test_checking_spend_does_not_commit_it(engine):
    for _ in range(20):
        assert engine.check_spend(5.00, tenant="globex").effect is Decision.ASK
    assert engine.committed_usd("globex") == 0.0


def test_the_spend_ledger_is_per_tenant(engine):
    engine.record_spend(40.00, tenant="acme")
    assert engine.committed_usd("acme") == 40.00
    assert engine.committed_usd("globex") == 0.0
    # acme's spending must not push globex through globex's cumulative cap.
    assert engine.check_spend(5.00, tenant="globex").effect is Decision.ASK


def test_a_cumulative_cap_does_not_apply_to_another_tenant(engine):
    engine.record_spend(50.00, tenant="acme")
    assert engine.check_spend(5.00, tenant="acme").effect is Decision.ASK


def test_negative_amounts_are_rejected(engine):
    with pytest.raises(ValueError, match="must not be negative"):
        engine.check_spend(-1.0)
    with pytest.raises(ValueError, match="must not be negative"):
        engine.record_spend(-1.0)


# ---- (d) approval routing ----------------------------------------------------


def test_an_approval_required_tool_returns_ask_and_names_the_role(engine):
    decision = engine.check_tool("deploy_prod", tenant="acme")
    assert decision.effect is Decision.ASK
    assert decision.requires_approval
    assert decision.approver_role == "sre"
    assert decision.rule_id == "deploy-asks-sre"


def test_the_router_asks_the_named_role_and_no_other(engine):
    asked: list[ApprovalRequest] = []
    other: list[ApprovalRequest] = []
    router = engine.approval_router(
        {"sre": lambda req: asked.append(req) or True, "finance": other.append},
        tenant="acme",
    )
    assert router("deploy_prod", {"target": "eu-1"}) is True
    assert other == []
    assert len(asked) == 1
    assert asked[0].approver_role == "sre"
    assert asked[0].subject == "deploy_prod"
    assert asked[0].tenant == "acme"
    assert asked[0].rule_id == "deploy-asks-sre"
    assert asked[0].args == {"target": "eu-1"}
    assert asked[0].policy_version == "1.4.0"


def test_the_router_refuses_when_the_approver_says_no(engine):
    router = engine.approval_router({"sre": lambda req: False}, tenant="acme")
    assert router("deploy_prod", {}) is False


def test_the_router_fails_closed_with_no_handler_for_the_role(engine):
    router = engine.approval_router({"finance": lambda req: True}, tenant="acme")
    assert router("deploy_prod", {}) is False
    record = engine.audit.entries(kind="approval")[-1]
    assert record.granted is False
    assert "no approval handler registered for role 'sre'" in record.reason


def test_the_router_fails_closed_when_the_handler_raises(engine):
    def broken(req):
        raise RuntimeError("approval service down")

    router = engine.approval_router({"sre": broken}, tenant="acme")
    assert router("deploy_prod", {}) is False
    assert "approval service down" in engine.audit.entries(kind="approval")[-1].reason


def test_a_denied_tool_is_never_routed_to_an_approver(engine):
    called: list[ApprovalRequest] = []
    router = engine.approval_router({"sre": lambda req: called.append(req) or True}, tenant="acme")
    assert router("delete_tmp", {}) is False
    assert called == []
    record = engine.audit.entries(kind="approval")[-1]
    assert record.granted is False
    assert "denied by policy" in record.reason


def test_the_router_is_bound_to_its_tenant(engine):
    """`write_*` is acme's grant, so the same router bound to globex refuses."""
    acme = engine.approval_router({}, tenant="acme")
    globex = engine.approval_router({}, tenant="globex")
    assert acme("write_doc", {}) is True
    assert globex("write_doc", {}) is False


def test_every_approval_is_audited_with_its_role_and_rule(engine):
    router = engine.approval_router({"sre": lambda req: True}, tenant="acme")
    router("deploy_prod", {})
    record = engine.audit.entries(kind="approval")[-1]
    assert record.approver_role == "sre"
    assert record.rule_id == "deploy-asks-sre"
    assert record.granted is True
    assert record.tenant == "acme"
    assert record.policy_version == "1.4.0"
    assert record.policy_digest == engine.digest


# ---- (f) audit and versioning ------------------------------------------------


def test_every_decision_is_auditable_with_the_rule_that_caused_it(engine):
    engine.check_tool("read_file", tenant="acme")
    engine.check_tool("delete_tmp", tenant="acme")
    engine.check_edge("plan", "deploy_prod", tenant="acme")
    engine.check_spend(500.0, tenant="acme")
    engine.check_node("shell_worker", tenant="acme")

    trail = [
        (e.resource.value, e.subject, e.effect.value, e.rule_id)
        for e in engine.audit.entries(kind="decision")
    ]
    assert trail == [
        ("tool", "read_file", "allow", "reads"),
        ("tool", "delete_tmp", "deny", "no-delete"),
        ("edge", "plan->deploy_prod", "deny", "no-into-prod"),
        ("spend", "*", "deny", "hard-cap"),
        ("node", "shell_worker", "deny", "no-shell"),
    ]
    assert all(e.policy_version == "1.4.0" for e in engine.audit.entries())
    assert all(e.policy_digest == engine.digest for e in engine.audit.entries())


def test_caller_context_rides_along_into_the_record(engine):
    engine.check_tool("read_file", context={"run_id": "r-7", "node": "investigate"})
    assert engine.audit.last().context == {"run_id": "r-7", "node": "investigate"}


def test_the_audit_log_is_filterable(engine):
    engine.check_tool("read_file", tenant="acme")
    engine.check_tool("read_file", tenant="globex")
    engine.approval_router({"sre": lambda req: True}, tenant="acme")("deploy_prod", {})
    assert len(engine.audit.entries(kind="approval")) == 1
    assert len(engine.audit.entries(kind="decision")) == 3
    assert len(engine.audit.entries(tenant="globex")) == 1


def test_the_audit_log_is_written_as_jsonl(tmp_path):
    log = AuditLog(tmp_path / "audit" / "policy.jsonl")
    engine = PolicyEngine.from_toml(POLICY_TOML, audit=log)
    engine.check_tool("delete_tmp", tenant="acme")
    engine.approval_router({"sre": lambda req: True}, tenant="acme")("deploy_prod", {})

    rows = log.read_jsonl()
    # The router re-asks the engine, so routing writes a decision and then its outcome.
    assert [row["kind"] for row in rows] == ["decision", "decision", "approval"]
    denial = rows[0]
    assert denial["subject"] == "delete_tmp"
    assert denial["effect"] == "deny"
    assert denial["rule_id"] == "no-delete"
    assert denial["tenant"] == "acme"
    assert denial["policy_version"] == "1.4.0"
    assert denial["policy_digest"] == engine.digest
    assert denial["ts"]
    assert rows[-1]["granted"] is True


def test_capping_memory_does_not_cap_the_file(tmp_path):
    log = AuditLog(tmp_path / "policy.jsonl", max_entries=2)
    engine = PolicyEngine.from_toml(POLICY_TOML, audit=log)
    for _ in range(5):
        engine.check_tool("read_file")
    assert len(log) == 2
    assert len(log.read_jsonl()) == 5


def test_a_zero_cap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        AuditLog(tmp_path / "a.jsonl", max_entries=0)


def test_the_digest_changes_when_a_rule_changes_under_the_same_version():
    """A version string is a claim by the author; the digest is evidence."""
    base = 'version = "1.0.0"\n[[rule]]\nid = "a"\nresource = "tool"\nmatch = "read_*"\n'
    tight = PolicyEngine.from_toml(base + 'effect = "deny"\n')
    loose = PolicyEngine.from_toml(base + 'effect = "allow"\n')
    assert tight.version == loose.version == "1.0.0"
    assert tight.digest != loose.digest
    assert tight.check_tool("read_x").denied and loose.check_tool("read_x").allowed


def test_the_digest_ignores_comments_and_formatting():
    plain = 'version = "1.0.0"\n[[rule]]\nid = "a"\nresource = "tool"\neffect = "allow"\n'
    decorated = (
        "# a comment nobody enforces\n"
        '\n\nversion   =   "1.0.0"\n\n'
        '[[rule]]\n  id = "a"\n  resource = "tool"\n  effect = "allow"\n'
    )
    assert PolicyEngine.from_toml(plain).digest == PolicyEngine.from_toml(decorated).digest


def test_rule_order_is_part_of_the_digest():
    """Order decides which rule of a tier is named as the cause, so it is content."""
    one = '[[rule]]\nid = "a"\nresource = "tool"\nmatch = "read_*"\neffect = "allow"\n'
    two = '[[rule]]\nid = "b"\nresource = "tool"\nmatch = "read_file"\neffect = "allow"\n'
    first = PolicyEngine.from_toml('version = "1"\n' + one + two)
    second = PolicyEngine.from_toml('version = "1"\n' + two + one)
    assert first.digest != second.digest
    assert first.check_tool("read_file").rule_id == "a"
    assert second.check_tool("read_file").rule_id == "b"


# ---- (g) composition with the harness ----------------------------------------

_TOOL_MATRIX = ["read_file", "delete_tmp", "deploy_prod", "write_doc", "pay_bill", "unheard_of"]


@pytest.mark.parametrize("tenant", ["default", "acme", "globex", "not-a-customer"])
def test_the_compiled_permission_policy_agrees_with_the_engine(engine, tenant):
    """The two must not drift: same tenant, same answer, for every tool."""
    policy = engine.permission_policy(tenant=tenant)
    for name in _TOOL_MATRIX:
        assert policy.decide(name) is engine.check_tool(name, tenant=tenant).effect, name


def test_the_compiled_permission_policy_carries_a_permissive_default():
    """The compiled object has to reproduce the document's default too, not
    just its rules — a `deny`-defaulting `PermissionPolicy` under an
    `allow`-defaulting document would silently be a stricter policy."""
    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "allow"\n'
        '[[rule]]\nid = "d"\nresource = "tool"\nmatch = "rm_*"\neffect = "deny"\n'
    )
    policy = engine.permission_policy()
    for name in ("anything", "rm_rf"):
        assert policy.decide(name) is engine.check_tool(name).effect, name


def test_the_compiled_policy_carries_only_that_tenants_rules(engine):
    patterns = {rule.pattern for rule in engine.permission_policy(tenant="globex").rules}
    assert "write_*" not in patterns  # acme's grant
    assert "pay_*" in patterns
    assert engine.permission_policy(tenant="not-a-customer").rules == []


def test_a_harness_built_from_the_document_enforces_it(engine):
    """End to end: the document decides what an existing `Harness` will run."""
    registry = ToolRegistry()
    for name in ("read_file", "delete_tmp", "deploy_prod", "write_doc"):
        registry.register(ToolSpec(name=name, description="", fn=_echo))

    approvals: list[str] = []
    harness = Harness(
        registry,
        engine.permission_policy(tenant="acme"),
        executor=LocalExecutor(),
        approval=engine.approval_router(
            {"sre": lambda req: approvals.append(req.subject) or True}, tenant="acme"
        ),
    )

    assert harness.call("read_file", {"path": "a"}) == {"path": "a"}
    assert harness.call("write_doc", {"path": "b"}) == {"path": "b"}  # acme's grant
    assert harness.call("deploy_prod", {"env": "eu"}) == {"env": "eu"}
    assert approvals == ["deploy_prod"]  # ran only after the named role approved

    with pytest.raises(PermissionDenied, match="denied by policy"):
        harness.call("delete_tmp", {})

    # Policy before schema: a denied tool's description never reaches the model.
    assert [spec.name for spec in harness.visible_tools()] == [
        "deploy_prod",
        "read_file",
        "write_doc",
    ]


def test_a_harness_for_another_tenant_gets_another_answer(engine):
    registry = ToolRegistry()
    registry.register(ToolSpec(name="write_doc", description="", fn=_echo))
    harness = Harness(
        registry, engine.permission_policy(tenant="globex"), executor=LocalExecutor()
    )
    with pytest.raises(PermissionDenied):
        harness.call("write_doc", {})
    assert harness.visible_tools() == []


def test_refusing_approval_stops_the_tool_running(engine):
    registry = ToolRegistry()
    ran: list[str] = []
    registry.register(
        ToolSpec(name="deploy_prod", description="", fn=lambda **kw: ran.append("ran"))
    )
    harness = Harness(
        registry,
        engine.permission_policy(tenant="acme"),
        executor=LocalExecutor(),
        approval=engine.approval_router({"sre": lambda req: False}, tenant="acme"),
    )
    with pytest.raises(PermissionDenied, match="requires approval"):
        harness.call("deploy_prod", {})
    assert ran == []
    assert engine.audit.entries(kind="approval")[-1].granted is False


# --------------------------------------------------------------------------
# §7.5 / §12.2 — `edge_policy()`. The document already understood `edge` rules
# and `AdmissionChecker` already consulted an `EdgePolicy`; nothing joined
# them, so a TOML file could answer questions about a transition but could not
# stop one. The tests mirror the `permission_policy` ones deliberately: the
# compiled object has to agree with the engine, or the two answers drift.
# --------------------------------------------------------------------------

_EDGE_MATRIX = [
    ("plan", "deploy_prod"),
    ("plan", "read_file"),
    ("__start__", "plan"),
    ("deploy_prod", "__end__"),
    ("shell_worker", "deploy_prod"),
    ("unheard_of", "also_unheard_of"),
]


@pytest.mark.parametrize("tenant", ["default", "acme", "globex", "not-a-customer"])
def test_the_compiled_edge_policy_agrees_with_the_engine(engine, tenant):
    """Same tenant, same answer, for every edge in the matrix."""
    policy = engine.edge_policy(tenant=tenant)
    for source, target in _EDGE_MATRIX:
        assert policy.decide(source, target) is engine.check_edge(
            source, target, tenant=tenant
        ).effect, f"{source}->{target} for {tenant!r}"


def test_an_undeclared_tenant_compiles_to_a_policy_that_permits_nothing(engine):
    policy = engine.edge_policy(tenant="not-a-customer")

    assert policy.rules == ()
    assert policy.default is Decision.DENY
    assert policy.decide("anything", "anywhere") is Decision.DENY


def test_the_compiled_edge_policy_carries_the_documents_default():
    """An `allow`-defaulting document must not compile to a denying policy."""
    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "allow"\n'
        '[[rule]]\nid = "e"\nresource = "edge"\nmatch = "*->deploy"\neffect = "deny"\n'
    )
    policy = engine.edge_policy()

    assert policy.decide("plan", "deploy") is Decision.DENY
    assert policy.decide("plan", "patch") is Decision.ALLOW


def test_each_side_of_an_edge_rule_is_matched_separately():
    """A `*` on one side must not swallow the arrow and match the other."""
    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "allow"\n'
        '[[rule]]\nid = "e"\nresource = "edge"\nmatch = "triage->*"\neffect = "deny"\n'
    )
    policy = engine.edge_policy()

    assert policy.decide("triage", "anything") is Decision.DENY
    assert policy.decide("other", "triage") is Decision.ALLOW


def test_the_compiled_edge_policy_carries_the_documents_reason(engine):
    """Issue #45: the words an operator wrote are what a refusal quotes and what
    the planner is told up front. Dropped at compile time, both could only say
    `edge_denied`."""
    rule = engine.edge_policy().rule_for("plan", "deploy_prod")

    assert rule is not None
    assert rule.action is Decision.DENY
    assert rule.reason == "the production deploy node is entered by an operator"
    assert rule.reason in engine.edge_policy().disclosure()[0]


def test_tool_rules_do_not_leak_into_the_edge_policy():
    """Widening what an agent may call must not widen what a planner may wire."""
    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "deny"\n'
        '[[rule]]\nid = "t"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
    )
    policy = engine.edge_policy()

    assert policy.rules == ()
    assert policy.decide("a", "b") is Decision.DENY


# --------------------------------------------------------------------------
# Issue #66 — `node_policy()`. `check_node` was correct and had no runtime
# caller: a document's `node` rules were compiled by nobody, so `no-shell-nodes`
# in a policy file governed nothing and a plan proposing a denied kind ran it.
# Same shape as the `edge_policy()` tests above: the compiled object has to
# agree with the engine, and the document has to be what refuses the node.
# --------------------------------------------------------------------------

_NODE_MATRIX = ["shell_worker", "shell_", "summarise", "deploy_prod", "unheard_of"]


@pytest.mark.parametrize("tenant", ["default", "acme", "globex", "not-a-customer"])
def test_the_compiled_node_policy_agrees_with_the_engine(engine, tenant):
    """Same tenant, same answer, for every kind in the matrix."""
    policy = engine.node_policy(tenant=tenant)
    for kind in _NODE_MATRIX:
        assert policy.decide(kind) is engine.check_node(kind, tenant=tenant).effect, (
            f"{kind!r} for {tenant!r}"
        )


def test_an_undeclared_tenant_compiles_to_a_node_policy_that_permits_nothing(engine):
    policy = engine.node_policy(tenant="not-a-customer")

    assert policy.rules == ()
    assert policy.default is Decision.DENY
    assert policy.decide("anything") is Decision.DENY


def test_the_compiled_node_policy_carries_the_documents_reason(engine):
    """The words an operator wrote have to survive the compile, or a refusal
    can only say what happened and never why."""
    rule = engine.node_policy().rule_for("shell_worker")

    assert rule is not None
    assert rule.action is Decision.DENY
    assert engine.check_node("shell_worker").rule_id == "no-shell"


def test_a_document_with_no_node_rules_compiles_to_its_own_default():
    """Faithful to `check_node`, which is the whole point of the compile."""
    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "deny"\n'
        '[[rule]]\nid = "e"\nresource = "edge"\nmatch = "*->*"\neffect = "allow"\n'
    )
    policy = engine.node_policy()

    assert policy.rules == ()
    assert policy.decide("triage") is engine.check_node("triage").effect is Decision.DENY


def test_the_document_stops_a_planner_running_a_denied_node_kind():
    """End to end, and the exact shape of issue #66: the TOML file refuses it."""
    from grapharc.planner import AdmissionChecker, NodeRegistry, NodeSpec, Subgraph
    from grapharc.planner.proposal import ProposedNode

    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "deny"\n'
        '[[rule]]\nid = "no-deploy-node"\nresource = "node"\nmatch = "deploy"\n'
        'effect = "deny"\nreason = "deploying from a plan is never permitted"\n'
        '[[rule]]\nid = "others"\nresource = "node"\nmatch = "*"\neffect = "allow"\n'
        '[[rule]]\nid = "edges"\nresource = "edge"\nmatch = "*->*"\neffect = "allow"\n'
    )
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="triage"), NodeSpec(name="deploy")]),
        edge_policy=engine.edge_policy(),
        node_policy=engine.node_policy(),
    )

    # Renamed instance, denied kind — the rule is about what a node *is*.
    refused = gate.check(Subgraph(nodes=(ProposedNode(name="ship", kind="deploy"),)))
    admitted = gate.check(Subgraph(nodes=(ProposedNode(name="triage"),)))

    assert not refused.admitted
    assert [r.code for r in refused.rejections] == ["node_denied"]
    assert "deploying from a plan is never permitted" in refused.rejections[0].detail
    assert admitted.admitted


def test_the_document_stops_a_planner_wiring_a_denied_transition():
    """End to end: the TOML file, not Python, is what refuses the edge."""
    from grapharc.planner import (
        AdmissionChecker,
        NodeRegistry,
        NodeSpec,
        ProposedEdge,
        ProposedNode,
        Subgraph,
    )

    engine = PolicyEngine.from_toml(
        'version = "1"\ndefault = "allow"\n'
        '[[rule]]\nid = "no-deploy"\nresource = "edge"\n'
        'match = "*->deploy"\neffect = "deny"\n'
    )
    gate = AdmissionChecker(
        registry=NodeRegistry([NodeSpec(name="triage"), NodeSpec(name="deploy")]),
        edge_policy=engine.edge_policy(),
    )

    refused = gate.check(
        Subgraph(
            # Renamed instance, denied kind — the rule is about what a node *is*.
            nodes=(ProposedNode(name="triage"), ProposedNode(name="ship", kind="deploy")),
            edges=(ProposedEdge(source="triage", target="ship"),),
        )
    )
    admitted = gate.check(
        Subgraph(
            nodes=(ProposedNode(name="triage"),),
            edges=(ProposedEdge(source="__start__", target="triage"),),
        )
    )

    assert not refused.admitted
    assert any(r.code == "edge_denied" for r in refused.rejections)
    assert admitted.admitted
