"""A reported dollar price is independent of optional token metadata."""

import pytest

from grapharc.observe import TraceRecorder, attribute, attribute_tenant
from grapharc.observe.cost import attribute_thread
from grapharc.observe.replay import replay
from grapharc.runtime.graph import END, START, GraphARC
from grapharc.runtime.state import GraphARCState
from grapharc.testing import ScriptedChatModel


@pytest.mark.parametrize("terminal", ["end", "error"])
@pytest.mark.parametrize("price", [None, 0.0, 0.003])
def test_recorded_prices_do_not_require_token_counts(tmp_path, terminal, price):
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    common = dict(run_id="run", thread_id="thread", tenant="acme", graph="g")
    trace.event(**common, node="worker", phase="start", step=1)
    # A per-call breakdown cannot be added to its enclosing terminal again.
    trace.event(**common, node="worker:model", phase="model", step=2, cost_usd=price)
    trace.event(**common, node="worker", phase=terminal, step=1,
                error="failed" if terminal == "error" else None, cost_usd=price)
    costs = [attribute(trace, "run"), attribute_thread(trace, "thread"),
             attribute_tenant(trace, "acme")]
    for cost in costs:
        assert cost.tokens == 0
        assert cost.recorded_cost_usd == price
    assert replay(trace, "run").recorded_cost_usd == price
    assert attribute(trace, "run").per_node[0].recorded_cost_usd == price


def test_a_failed_priced_model_without_usage_metadata_keeps_its_bill(tmp_path):
    class PriceWithoutTokens(ScriptedChatModel):
        def _generate(self, *args, **kwargs):
            result = super()._generate(*args, **kwargs)
            result.generations[0].message.usage_metadata = None
            return result

    trace = TraceRecorder(tmp_path / "trace.jsonl")
    model = PriceWithoutTokens(responses=["paid"], cost_usd=0.002)
    graph = GraphARC(GraphARCState, name="g", trace=trace)

    def fail(state):
        model.invoke("work")
        raise ValueError("failed after spending")

    graph.add_node("worker", fail, writes=set())
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    with pytest.raises(ValueError, match="failed after spending"):
        graph.compile().invoke({}, run_id="run", thread_id="thread", tenant="acme")
    terminal, = [e for e in trace.read_events() if e.phase == "error"]
    assert terminal.tokens == 0
    assert terminal.cost_usd == pytest.approx(0.002)
    for cost in [attribute(trace, "run"), attribute_thread(trace, "thread"),
                 attribute_tenant(trace, "acme")]:
        assert cost.recorded_cost_usd == pytest.approx(0.002)
