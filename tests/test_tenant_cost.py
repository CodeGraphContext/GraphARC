"""Explicit run tenants must survive every producer and trace-only cost reader."""

from __future__ import annotations

import json

import pytest

from grapharc.cli.main import main
from grapharc.examples import plan_incident, plan_research
from grapharc.harness import AgentNode, Harness, LocalExecutor, PermissionPolicy, ToolRegistry
from grapharc.observe import RateCard, ReplayError, TraceEvent, TraceRecorder, attribute
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import END, START, GraphARC, RunContext
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


class State(GraphARCState):
    answer: str = ""


def priced_graph(trace, *, fail=False):
    model = ScriptedChatModel(responses=["done"], cost_usd=0.002, model_name="acme/fast")
    graph = GraphARC(State, name="priced", trace=trace)

    def body(state):
        answer = model.invoke("work").content
        if fail:
            raise ValueError("failed after spending")
        return {"answer": answer}

    graph.add_node("think", body, writes={"answer"})
    graph.add_edge(START, "think")
    graph.add_edge("think", END)
    return graph.compile()


def tenant_cost(source, tenant, **kwargs):
    # Lazy import keeps pre-feature collection working: the new tests can
    # demonstrate the missing behavior rather than aborting collection.
    from grapharc.observe import attribute_tenant

    return attribute_tenant(source, tenant, **kwargs)


def event(trace, run, tenant, phase="end", step=1, **kwargs):
    trace.event(
        run_id=run, tenant=tenant, graph="g", node="worker", phase=phase, step=step, **kwargs
    )


def test_legacy_trace_bytes_and_parsing_are_unchanged(tmp_path):
    trace = TraceRecorder(tmp_path / "old.jsonl")
    old = {
        "ts": "2026-01-01",
        "run_id": "old",
        "attempt": 1,
        "graph": "g",
        "node": "n",
        "phase": "end",
        "step": 1,
        "tokens": 7,
    }
    trace.record(TraceEvent(**old))
    assert trace.path.read_text() == json.dumps(old) + "\n"
    parsed = trace.read_events()[0]
    assert parsed.tenant is None
    assert "tenant" not in parsed.model_dump(exclude_none=True)
    assert attribute(trace, "old").tokens == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["invoke", "stream", "ainvoke", "astream", "astream_events"])
async def test_every_execution_entry_carries_the_tenant(tmp_path, entry):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    compiled = priced_graph(trace)
    if entry == "invoke":
        compiled.invoke({}, tenant="alpha", run_id="run")
    elif entry == "stream":
        list(compiled.stream({}, tenant="alpha", run_id="run"))
    elif entry == "ainvoke":
        await compiled.ainvoke({}, tenant="alpha", run_id="run")
    else:
        _ = [item async for item in getattr(compiled, entry)({}, tenant="alpha", run_id="run")]
    events = trace.read_events()
    assert {e.phase for e in events} >= {"topology", "start", "end"}
    assert {e.tenant for e in events} == {"alpha"}
    assert compiled.last_run.tenant == "alpha"
    cost = tenant_cost(trace, "alpha")
    assert cost.recorded_cost_usd == pytest.approx(0.002)
    assert cost.tokens == attribute(trace, "run").tokens > 0


def test_error_events_keep_the_tenant_and_spend(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="failed after spending"):
        priced_graph(trace, fail=True).invoke({}, tenant="alpha", run_id="failed")
    assert {e.tenant for e in trace.read_events()} == {"alpha"}
    cost = tenant_cost(trace, "alpha")
    assert cost.runs[0].errors == 1
    assert cost.recorded_cost_usd == attribute(trace, "failed").recorded_cost_usd
    assert cost.tokens > 0
    assert cost.unpriced_tokens == cost.tokens


def test_tenants_partition_runs_without_double_counting_substeps(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    event(trace, "a1", "alpha", "start")
    event(trace, "b1", "beta", tokens=200, cost_usd=0.02)
    event(trace, "a1", "alpha", "model", step=2, tokens=100, cost_usd=0.01)
    event(trace, "a1", "alpha", tokens=100, cost_usd=0.01)
    event(trace, "a2", "alpha", "start")
    event(trace, "a2", "alpha", "model", step=2, tokens=50, model="acme/fast")
    event(trace, "a2", "alpha", tokens=50)
    # Legacy spend cannot silently enter a named tenant's bill.
    event(trace, "old", None, tokens=9, cost_usd=1.0)
    rates = RateCard(per_model={"acme/": 2.0})
    alpha = tenant_cost(trace.path, "alpha", rates=rates)
    beta = tenant_cost(trace, "beta", rates=rates)
    assert [r.run_id for r in alpha.runs] == ["a1", "a2"]
    assert alpha.tokens == 150
    assert beta.tokens == 200
    assert alpha.recorded_cost_usd == pytest.approx(0.01)
    assert alpha.estimated_cost_usd == pytest.approx(0.1)
    assert alpha.cost_usd == pytest.approx(0.11)
    assert alpha.unpriced_tokens == 0
    assert alpha.tokens + beta.tokens == sum(attribute(trace, r).tokens for r in ("a1", "a2", "b1"))
    assert alpha.recorded_cost_usd + beta.recorded_cost_usd == pytest.approx(0.03)
    assert tenant_cost(trace, "alpha").unpriced_tokens == 50
    assert tenant_cost(trace, "alpha").estimated_cost_usd is None


@pytest.mark.parametrize("labels", [("alpha", "beta"), ("alpha", None)])
def test_a_run_with_conflicting_labels_cannot_be_billed(tmp_path, labels):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    for step, label in enumerate(labels, start=1):
        event(trace, "mixed", label, step=step, tokens=10)
    with pytest.raises(ReplayError, match="inconsistent tenant"):
        tenant_cost(trace, "alpha")


def test_an_unknown_tenant_is_distinct_from_an_empty_bill(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    event(trace, "old", None, tokens=9)
    with pytest.raises(ReplayError, match="no events for tenant"):
        tenant_cost(trace, "alpha")


@pytest.mark.parametrize("label", ["研究:team/one", "a" * 2500], ids=["unicode", "long"])
def test_tenant_identifiers_are_preserved_exactly(tmp_path, label):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    event(trace, "r", label, tokens=5)
    assert trace.read_events()[0].tenant == label
    assert tenant_cost(trace, label).tokens == 5


@pytest.mark.parametrize("label", [None, "alpha", "a" * 2500], ids=["unlabelled", "named", "long"])
def test_broadcast_recorder_matches_the_file_contract(tmp_path, label):
    pytest.importorskip("fastapi")
    from grapharc.server.runtime import BroadcastRecorder

    seen = []
    trace = BroadcastRecorder(tmp_path / "broadcast.jsonl", seen.append)
    priced_graph(trace).invoke({}, run_id="run", tenant=label)
    on_disk = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert seen == on_disk
    assert {e.tenant for e in trace.read_events()} == {label}
    if label is None:
        assert all("tenant" not in item for item in seen)
    else:
        assert tenant_cost(trace, label).recorded_cost_usd == pytest.approx(0.002)


def test_tenant_attribution_reads_one_snapshot(tmp_path):
    class CountingRecorder(TraceRecorder):
        reads = 0

        def read_events(self, run_id=None):
            self.reads += 1
            return super().read_events(run_id)

    trace = CountingRecorder(tmp_path / "trace.jsonl")
    event(trace, "a", "alpha", tokens=10)
    event(trace, "b", "alpha", tokens=20)
    assert tenant_cost(trace, "alpha").tokens == 30
    assert trace.reads == 1


class ToolModel(ScriptedChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_an_agent_inherits_its_run_context_tenant(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ToolModel(responses=["done"], cost_usd=0.004)
    harness = Harness(ToolRegistry(), PermissionPolicy(), executor=LocalExecutor())
    node = AgentNode(model, harness, trace=trace)
    ctx = RunContext(run_id="agent", graph="g", meter=BudgetMeter(Budget()), tenant="alpha")
    node.run("finish", ctx)
    assert {e.phase for e in trace.read_events()} == {"model", "stop"}
    assert {e.tenant for e in trace.read_events()} == {"alpha"}
    assert tenant_cost(trace, "alpha").recorded_cost_usd == pytest.approx(0.004)


@pytest.mark.parametrize("explicit", [False, True])
def test_policy_governed_agent_cli_records_only_an_explicit_tenant(
    tmp_path, monkeypatch, capsys, explicit
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRAPHARC_TENANT", raising=False)
    monkeypatch.delenv("GRAPHARC_CONFIG", raising=False)
    policy = tmp_path / "policy.toml"
    policy.write_text(
        'version = "1"\ndefault = "deny"\ntenants = ["default", "acme"]\n'
        '[[rule]]\nid = "tools"\nresource = "tool"\nmatch = "*"\neffect = "allow"\n'
    )
    model = ToolModel(responses=["done"], cost_usd=0.004)
    monkeypatch.setattr("grapharc.gateway.get_model", lambda *a, **kw: model)
    path = tmp_path / "trace.jsonl"
    flags = ["--tenant", "acme"] if explicit else []
    assert (
        main(
            [
                "agent",
                "finish",
                "--model",
                "test-double",
                "--policy",
                str(policy),
                "--workspace",
                str(tmp_path / "work"),
                "--trace",
                str(path),
                *flags,
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    events = TraceRecorder(path).read_events()
    assert events
    assert {e.tenant for e in events} == ({"acme"} if explicit else {None})
    if explicit:
        assert tenant_cost(path, "acme").recorded_cost_usd == pytest.approx(0.004)
    else:
        assert attribute(path, payload["run_id"]).tenant is None
        assert all("tenant" not in e.model_dump(exclude_none=True) for e in events)


@pytest.mark.parametrize("failed", [False, True])
def test_delegated_agent_events_keep_the_tenant(tmp_path, monkeypatch, failed):
    from grapharc.cli.delegate import DelegatedRun, DelegationError

    def delegate(*args, **kwargs):
        if failed:
            raise DelegationError("failed", reason="test_failure")
        return DelegatedRun(True, "done", "done", 1, 15, 0.02, None, [], [])

    monkeypatch.setattr("grapharc.cli.delegate.delegate_task", delegate)
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    harness = Harness(
        ToolRegistry(), PermissionPolicy(), executor=LocalExecutor(), workspace=str(tmp_path)
    )
    node = AgentNode(ToolModel(responses=[]), harness, trace=trace)
    ctx = RunContext(run_id="delegated", graph="g", meter=BudgetMeter(Budget()), tenant="alpha")
    node._run_delegated("finish", ctx)
    assert {e.tenant for e in trace.read_events()} == {"alpha"}
    cost = tenant_cost(trace, "alpha")
    assert cost.tokens == (0 if failed else 15)
    assert cost.recorded_cost_usd == (None if failed else pytest.approx(0.02))


@pytest.mark.parametrize("label", ["alpha", None])
def test_saved_plan_execution_uses_the_current_explicit_tenant(
    tmp_path, monkeypatch, capsys, label
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRAPHARC_TENANT", raising=False)
    monkeypatch.delenv("GRAPHARC_CONFIG", raising=False)
    path = tmp_path / "trace.jsonl"
    assert (
        main(
            [
                "plan",
                "fix incident",
                "--scripted",
                "--trace",
                str(path),
                "--tenant",
                "planning",
                "--json",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    old_runs = {e.run_id for e in TraceRecorder(path).read_events()}
    flags = ["--tenant", label] if label is not None else []
    assert main(["go", planned["plan_file"], *flags, "--json"]) == 0
    _ = capsys.readouterr()
    events = TraceRecorder(path).read_events()
    executed = [e for e in events if e.run_id not in old_runs]
    assert any(e.phase == "end" for e in executed)
    assert {e.tenant for e in executed} == {label}
    assert {e.tenant for e in events if e.run_id in old_runs} == {"planning"}


def test_governed_planning_and_rounds_share_one_tenant_bill(tmp_path):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = ScriptedChatModel(responses=plan_research.scripted_planner_replies(), cost_usd=0.003)
    loop = plan_research.build_loop(model, trace=trace)
    result = loop.run("explain elevated latency", run_id="loop", tenant="research")
    events = trace.read_events()
    assert {e.phase for e in events} >= {
        "plan",
        "admission",
        "topology",
        "start",
        "end",
        "round",
        "stop",
    }
    assert len({e.thread_id for e in events}) > 1
    assert {e.tenant for e in events} == {"research"}
    assert any(r.executed for r in result.rounds)
    cost = tenant_cost(trace, "research", rates=RateCard(default=2.0))
    assert [r.run_id for r in cost.runs] == ["loop"]
    # Planner events currently carry tokens, not a provider price. That
    # missing price must remain an estimate, even when the model has one.
    assert cost.recorded_cost_usd is None
    assert cost.tokens == attribute(trace, "loop").tokens > 0
    assert cost.estimated_cost_usd == pytest.approx(cost.tokens * 2.0 / 1000)
    unpriced = tenant_cost(trace, "research")
    assert unpriced.unpriced_tokens == unpriced.tokens
    assert unpriced.complete is False


def test_unlabelled_planner_spend_is_also_priced_or_marked_incomplete(tmp_path):
    trace = TraceRecorder(tmp_path / "legacy.jsonl")
    trace.event(run_id="old", graph="g", node="planner", phase="plan", step=1, tokens=50)
    unpriced = attribute(trace, "old")
    assert unpriced.tokens == 50
    assert unpriced.unpriced_tokens == 50
    assert unpriced.complete is False
    estimated = attribute(trace, "old", rates=RateCard(default=2.0))
    assert estimated.estimated_cost_usd == pytest.approx(0.1)
    assert estimated.recorded_cost_usd is None
    assert estimated.complete is True


@pytest.mark.parametrize("command", ["run", "plan", "go"])
@pytest.mark.parametrize("source", ["flag", "env", "config", "default"])
def test_cli_propagates_only_explicit_tenants(tmp_path, monkeypatch, capsys, command, source):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRAPHARC_TENANT", raising=False)
    monkeypatch.delenv("GRAPHARC_CONFIG", raising=False)
    trace = tmp_path / "trace.jsonl"
    extra = []
    if source == "flag":
        extra = ["--tenant", "alpha"]
    elif source == "env":
        monkeypatch.setenv("GRAPHARC_TENANT", "alpha")
    elif source == "config":
        (tmp_path / "grapharc.toml").write_text('tenant = "alpha"\n')
    if command == "run":
        graph = tmp_path / "graph.json"
        graph.write_text(
            json.dumps(
                {
                    "nodes": [{"name": "triage"}],
                    "edges": [
                        {"source": START, "target": "triage"},
                        {"source": "triage", "target": END},
                    ],
                }
            )
        )
        args = ["run", str(graph)]
    else:
        model = ScriptedChatModel(responses=plan_incident.scripted_planner_replies())
        monkeypatch.setattr("grapharc.gateway.get_model", lambda *a, **kw: model)
        model_flags = ["--scripted"] if command == "plan" else ["--model", "test-double"]
        args = [
            command,
            "fix incident",
            "--registry",
            "grapharc.examples.plan_incident:build_registry",
            *model_flags,
        ]
    main([*args, *extra, "--trace", str(trace), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    events = TraceRecorder(trace).read_events()
    assert events
    expected = None if source == "default" else "alpha"
    assert {e.tenant for e in events} == {expected}
    if source == "default":
        assert all("tenant" not in e.model_dump(exclude_none=True) for e in events)


@pytest.mark.parametrize("scope", ["tenant", "run-id"])
@pytest.mark.parametrize("as_json", [True, False])
def test_cli_cost_scopes_text_and_json(tmp_path, capsys, scope, as_json):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    event(trace, "a", "alpha", tokens=10, cost_usd=0.01)
    event(trace, "b", "beta", tokens=50, cost_usd=0.05)
    value = "alpha" if scope == "tenant" else "a"
    code = main(["cost", str(trace.path), f"--{scope}", value, *(["--json"] if as_json else [])])
    out = capsys.readouterr().out
    assert code == 0
    if as_json:
        payload = json.loads(out)
        assert payload["tokens"] == 10
        assert payload["recorded_cost_usd"] == pytest.approx(0.01)
        assert payload["estimated_cost_usd"] is None
        assert payload["unpriced_tokens"] == 0
        assert payload["tenant"] == "alpha"
    else:
        assert "alpha" in out
        assert "0.010000" in out
        assert "beta" not in out


@pytest.mark.parametrize(
    "problem,expected",
    [("missing", 2), ("malformed", 2), ("utf8", 2), ("unknown", 1), ("mixed", 1)],
)
def test_cli_cost_failures_are_structured(tmp_path, capsys, problem, expected):
    path = tmp_path / "trace.jsonl"
    if problem == "malformed":
        path.write_text("not json\n")
    elif problem == "utf8":
        path.write_bytes(b"\xff\n")
    elif problem in ("unknown", "mixed"):
        trace = TraceRecorder(path)
        event(trace, "r", "beta", tokens=5)
        if problem == "mixed":
            event(trace, "r", "alpha", step=2, tokens=10)
    code = main(["cost", str(path), "--tenant", "alpha", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == expected
    assert payload["ok"] is False
    assert payload["command"] == "cost"
    assert payload["error"]
