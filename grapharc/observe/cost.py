"""Cost attribution, derived from the trace and nothing else.

Two sources of money, kept apart on purpose:

- **Recorded** — a `cost_usd` a producer wrote onto the trace event. This is
  the provider's own number and is used as-is. The kernel's node wrapper
  stamps it on a node's terminal event when the meter carries one, and
  `AgentNode` stamps its model calls — so a gateway-priced run carries
  recorded cost per node. A trace from a producer that never priced its calls
  still reads as `recorded_cost_usd = None`, never as an estimate.
- **Estimated** — tokens × a `RateCard` the caller supplies. Priced off the
  *total* token count, because that is what the trace records: there is no
  input/output split on a trace event, so a card carries one blended rate per
  model and the result is an estimate. It is reported in its own field and
  never added into `recorded_cost_usd`, so nobody can mistake an estimate for
  an invoice.

Tokens are counted from the `end` events of node executions, which is exactly
the rule `metrics.summarize` uses. That is not a coincidence and not a
duplicate implementation: `RunCost.tokens` is asserted equal to
`RunMetrics.tokens` by the test suite, because a cost report and an audit trail
that disagree are worse than either alone.

Two limits, stated rather than smoothed over:

- A node that raises never emits an `end` event, and the kernel's `error` event
  carries no token count — so tokens spent inside a node that then failed are
  invisible at node level. Where the node was an `AgentNode`, its per-call
  `"model"` sub-events still hold them, and they are reported as
  `tokens_before_error` rather than folded into the total that must match
  `metrics`.
- There is no tenant on a trace event, so tenant attribution is not offered
  here. Run, thread (session) and node are what the format supports today.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from grapharc.observe.replay import (
    NodeExecution,
    ReplayedRun,
    ReplayError,
    replay,
    replay_thread,
)
from grapharc.observe.trace import TraceRecorder


class RateCard(BaseModel):
    """USD per 1,000 tokens, blended across input and output.

    Blended because a trace event records one `tokens` total and no split.
    Pricing an unknown mix at a single rate is an estimate; calling it anything
    else would be the kind of confident prose this repo has been burned by.
    """

    per_model: dict[str, float] = Field(default_factory=dict)
    default: float | None = None

    def rate_for(self, model: str | None) -> float | None:
        """Exact model id, then the longest matching prefix, then the default.

        Prefix matching is what makes `{"openrouter/anthropic/": 9.0}` usable
        against the fully-qualified ids the gateway reports.
        """
        if model:
            exact = self.per_model.get(model)
            if exact is not None:
                return exact
            prefixes = [k for k in self.per_model if model.startswith(k)]
            if prefixes:
                return self.per_model[max(prefixes, key=len)]
        return self.default

    def price(self, tokens: int, model: str | None) -> float | None:
        rate = self.rate_for(model)
        return None if rate is None else tokens / 1000 * rate


class NodeCost(BaseModel):
    """One graph node's share of a run."""

    node: str
    executions: int = 0
    errors: int = 0
    tokens: int = 0
    duration_ms: float = 0.0
    recorded_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    # Tokens with neither a recorded cost nor a rate to price them at. Non-zero
    # here means `cost_usd` is an under-count, and by how much.
    unpriced_tokens: int = 0
    models: list[str] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float | None:
        """Recorded plus estimated. None when neither is known for this node."""
        return _combine(self.recorded_cost_usd, self.estimated_cost_usd)


class ModelCallCost(BaseModel):
    """One model call, as whatever made it reported it.

    With a `parent_node`, this is a breakdown *within* that `NodeCost`, never
    an addition to it: the runtime charges the enclosing node's meter for calls
    made inside it, so the same tokens appear in both, and summing the two
    would double the bill.

    With `parent_node` None, the call happened outside any node span — the
    shape `grapharc agent` produces, driving an `AgentNode` against a bare
    `RunContext`. Nothing else counts those tokens, so they *are* added to the
    run total, and no `NodeCost` reports them.
    """

    node: str
    parent_node: str | None = None
    step: int
    tokens: int = 0
    duration_ms: float | None = None
    model: str | None = None
    recorded_cost_usd: float | None = None
    estimated_cost_usd: float | None = None


class RunCost(BaseModel):
    """Per-run, per-node token and cost attribution."""

    run_id: str
    graph: str
    thread_id: str | None = None
    executions: int = 0
    errors: int = 0
    tokens: int = 0
    duration_ms: float = 0.0
    recorded_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    unpriced_tokens: int = 0
    per_node: list[NodeCost] = Field(default_factory=list)
    model_calls: list[ModelCallCost] = Field(default_factory=list)
    # Tokens reported by sub-steps of nodes that ended in `error`. Held apart
    # from `tokens` because the node-level total must keep matching
    # `metrics.summarize`, and because "spend that bought nothing" is the
    # number an incident review actually wants.
    tokens_before_error: int = 0

    @property
    def cost_usd(self) -> float | None:
        return _combine(self.recorded_cost_usd, self.estimated_cost_usd)

    @property
    def complete(self) -> bool:
        """True when no counted token was left unpriced.

        A run that spent no tokens is trivially complete — and its `cost_usd`
        is still None, because nothing was priced rather than priced at zero.
        """
        return self.unpriced_tokens == 0

    def node(self, name: str) -> NodeCost | None:
        return next((n for n in self.per_node if n.node == name), None)

    def format(self) -> str:
        lines = [f"run {self.run_id} · graph {self.graph}"]
        for entry in self.per_node:
            cost = "—" if entry.cost_usd is None else f"${entry.cost_usd:.6f}"
            lines.append(
                f"  {entry.node:<24} {entry.executions:>3}x "
                f"{entry.tokens:>8} tok  {entry.duration_ms:>9.1f}ms  {cost}"
            )
        total = "—" if self.cost_usd is None else f"${self.cost_usd:.6f}"
        lines.append(f"  {'TOTAL':<24} {self.executions:>3}x {self.tokens:>8} tok  {total}")
        if self.unpriced_tokens:
            lines.append(
                f"  {self.unpriced_tokens} tokens had no recorded cost and no rate: "
                "the total above is a lower bound"
            )
        if self.tokens_before_error:
            lines.append(
                f"  {self.tokens_before_error} tokens were spent inside nodes that then "
                "failed and are not in the total"
            )
        return "\n".join(lines)


class ThreadCost(BaseModel):
    """Every run recorded against one thread — a session's bill."""

    thread_id: str
    graph: str
    runs: list[RunCost] = Field(default_factory=list)
    per_node: list[NodeCost] = Field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.runs)

    @property
    def duration_ms(self) -> float:
        return round(sum(r.duration_ms for r in self.runs), 2)

    @property
    def unpriced_tokens(self) -> int:
        return sum(r.unpriced_tokens for r in self.runs)

    @property
    def recorded_cost_usd(self) -> float | None:
        return _sum_optional(r.recorded_cost_usd for r in self.runs)

    @property
    def estimated_cost_usd(self) -> float | None:
        return _sum_optional(r.estimated_cost_usd for r in self.runs)

    @property
    def cost_usd(self) -> float | None:
        return _combine(self.recorded_cost_usd, self.estimated_cost_usd)


def _combine(*values: float | None) -> float | None:
    """Sum what is known; None only when nothing is."""
    known = [v for v in values if v is not None]
    return round(sum(known), 10) if known else None


def _sum_optional(values: Iterable[float | None]) -> float | None:
    return _combine(*values)


def _estimate(execution: NodeExecution, card: RateCard) -> tuple[float | None, int, list[str]]:
    """Estimate one execution's cost: (estimate, unpriced tokens, models seen).

    A node that called two different models is priced call by call rather than
    at whichever model happened to run first — but only when its model
    sub-steps account for the node's whole token count. If they do not, the
    node holds spend the sub-steps cannot explain, and blending it at one rate
    is the best the trace supports.
    """
    model_events = [s for s in execution.sub_events if s.phase == "model"]
    models = list(dict.fromkeys(s.model for s in model_events if s.model))
    if len(models) > 1 and sum(s.tokens or 0 for s in model_events) == execution.tokens:
        priced: list[float] = []
        unpriced = 0
        for sub in model_events:
            amount = card.price(sub.tokens or 0, sub.model)
            if amount is None:
                unpriced += sub.tokens or 0
            else:
                priced.append(amount)
        return _combine(*priced), unpriced, models
    amount = card.price(execution.tokens, models[0] if models else None)
    return amount, 0 if amount is not None else execution.tokens, models


def _price_run(run: ReplayedRun, rates: RateCard | None) -> RunCost:
    card = rates or RateCard()
    nodes: dict[str, NodeCost] = {}
    order: list[str] = []
    recorded_total: list[float] = []
    estimated_total: list[float] = []
    unpriced = 0
    calls: list[ModelCallCost] = []
    tokens_before_error = 0

    for execution in run.executions:
        entry = nodes.get(execution.node)
        if entry is None:
            entry = nodes[execution.node] = NodeCost(node=execution.node)
            order.append(execution.node)

        for sub in execution.sub_events:
            if sub.phase != "model":
                continue
            sub_estimate = (
                None if sub.cost_usd is not None else card.price(sub.tokens or 0, sub.model)
            )
            calls.append(
                ModelCallCost(
                    node=sub.node,
                    parent_node=execution.node,
                    step=sub.step,
                    tokens=sub.tokens or 0,
                    duration_ms=sub.duration_ms,
                    model=sub.model,
                    recorded_cost_usd=sub.cost_usd,
                    estimated_cost_usd=sub_estimate,
                )
            )

        if not execution.ok:
            # No `end` event was written, so the node-level token count for this
            # execution does not exist; whatever sub-steps it emitted before
            # failing are the only record of what it spent.
            entry.errors += 1
            tokens_before_error += execution.sub_tokens
            continue

        entry.executions += 1
        entry.tokens += execution.tokens
        # Accumulated unrounded and rounded once at the end, so the run total
        # is bit-for-bit what `metrics.summarize` computes from the same events.
        entry.duration_ms += execution.duration_ms or 0.0

        estimate, unpriced_here, models = _estimate(execution, card)
        entry.models = list(dict.fromkeys([*entry.models, *models]))

        if execution.cost_usd is not None:
            entry.recorded_cost_usd = _combine(entry.recorded_cost_usd, execution.cost_usd)
            recorded_total.append(execution.cost_usd)
            continue
        if estimate is not None:
            entry.estimated_cost_usd = _combine(entry.estimated_cost_usd, estimate)
            estimated_total.append(estimate)
        entry.unpriced_tokens += unpriced_here
        unpriced += unpriced_here

    # Spend that happened outside any node span. `grapharc agent` drives an
    # `AgentNode` with no enclosing graph, so every event it writes is an
    # orphan; charging only `per_node` billed such a run at zero. Orphans are
    # disjoint from every execution's `sub_events`, so this adds and never
    # double-counts. They get no `per_node` entry and raise no execution count:
    # no kernel node ran, and `RunCost` must keep agreeing with `RunMetrics`
    # on `executions` and `per_node`.
    orphan_tokens = 0
    orphan_ms = 0.0
    for sub in run.orphan_sub_events:
        orphan_tokens += sub.tokens or 0
        orphan_ms += sub.duration_ms or 0.0
        if sub.phase != "model":
            continue
        sub_estimate = (
            None if sub.cost_usd is not None else card.price(sub.tokens or 0, sub.model)
        )
        calls.append(
            ModelCallCost(
                node=sub.node,
                parent_node=None,
                step=sub.step,
                tokens=sub.tokens or 0,
                duration_ms=sub.duration_ms,
                model=sub.model,
                recorded_cost_usd=sub.cost_usd,
                estimated_cost_usd=sub_estimate,
            )
        )
        if sub.cost_usd is not None:
            recorded_total.append(sub.cost_usd)
        elif sub_estimate is not None:
            estimated_total.append(sub_estimate)
        else:
            unpriced += sub.tokens or 0

    per_node = [nodes[name] for name in order]
    total_ms = sum(n.duration_ms for n in per_node) + orphan_ms
    for entry in per_node:
        entry.duration_ms = round(entry.duration_ms, 2)
    return RunCost(
        run_id=run.run_id,
        graph=run.graph,
        thread_id=run.thread_id,
        executions=sum(n.executions for n in per_node),
        errors=sum(n.errors for n in per_node),
        tokens=sum(n.tokens for n in per_node) + orphan_tokens,
        duration_ms=round(total_ms, 2),
        recorded_cost_usd=_combine(*recorded_total),
        estimated_cost_usd=_combine(*estimated_total),
        unpriced_tokens=unpriced,
        per_node=per_node,
        model_calls=calls,
        tokens_before_error=tokens_before_error,
    )


def attribute(
    source: TraceRecorder | str | Path | ReplayedRun,
    run_id: str | None = None,
    *,
    rates: RateCard | None = None,
) -> RunCost:
    """Attribute one run's tokens and cost, per node.

    Pass a recorder or path plus a `run_id`, or an already-reconstructed
    `ReplayedRun`. Raises `ReplayError` when the run is not in the trace.
    """
    if isinstance(source, ReplayedRun):
        run = source
    else:
        if run_id is None:
            raise TypeError("attribute() needs a run_id unless given a ReplayedRun")
        run = replay(source, run_id)
    return _price_run(run, rates)


def attribute_thread(
    source: TraceRecorder | str | Path,
    thread_id: str,
    *,
    rates: RateCard | None = None,
) -> ThreadCost:
    """Attribute every run on one thread — a session bill across resumes.

    Raises `ReplayError` when the thread has no events.
    """
    runs = replay_thread(source, thread_id)
    if not runs:
        raise ReplayError(f"no events for thread_id {thread_id!r} in the trace")
    costs = [_price_run(run, rates) for run in runs]
    merged: dict[str, NodeCost] = {}
    order: list[str] = []
    for cost in costs:
        for entry in cost.per_node:
            existing = merged.get(entry.node)
            if existing is None:
                merged[entry.node] = entry.model_copy(deep=True)
                order.append(entry.node)
                continue
            existing.executions += entry.executions
            existing.errors += entry.errors
            existing.tokens += entry.tokens
            existing.duration_ms = round(existing.duration_ms + entry.duration_ms, 4)
            existing.unpriced_tokens += entry.unpriced_tokens
            existing.recorded_cost_usd = _combine(
                existing.recorded_cost_usd, entry.recorded_cost_usd
            )
            existing.estimated_cost_usd = _combine(
                existing.estimated_cost_usd, entry.estimated_cost_usd
            )
            existing.models = list(dict.fromkeys([*existing.models, *entry.models]))
    return ThreadCost(
        thread_id=thread_id,
        graph=runs[0].graph,
        runs=costs,
        per_node=[merged[name] for name in order],
    )


def by_node(source: TraceRecorder | str | Path, *, rates: RateCard | None = None) -> list[NodeCost]:
    """Every node's cost across every run in a trace file, most expensive first.

    Cost is not a total order when some nodes are priced and some are not, so
    the sort key is (cost if known else 0, tokens) — an unpriced node sinks to
    where its token count puts it rather than vanishing.
    """
    recorder = source if isinstance(source, TraceRecorder) else TraceRecorder(source)
    merged: dict[str, NodeCost] = {}
    for run_id in recorder.run_ids():
        for entry in attribute(recorder, run_id, rates=rates).per_node:
            existing = merged.get(entry.node)
            if existing is None:
                merged[entry.node] = entry.model_copy(deep=True)
                continue
            existing.executions += entry.executions
            existing.errors += entry.errors
            existing.tokens += entry.tokens
            existing.duration_ms = round(existing.duration_ms + entry.duration_ms, 4)
            existing.unpriced_tokens += entry.unpriced_tokens
            existing.recorded_cost_usd = _combine(
                existing.recorded_cost_usd, entry.recorded_cost_usd
            )
            existing.estimated_cost_usd = _combine(
                existing.estimated_cost_usd, entry.estimated_cost_usd
            )
    return sorted(merged.values(), key=lambda n: (n.cost_usd or 0.0, n.tokens), reverse=True)


def tokens_by_model(source: TraceRecorder | str | Path) -> dict[str, int]:
    """Tokens per model id, over every `model`-phase event that names one.

    Empty on a trace whose producer never recorded a model id — which is every
    trace today. See the module docstring.
    """
    totals: dict[str, int] = defaultdict(int)
    recorder = source if isinstance(source, TraceRecorder) else TraceRecorder(source)
    for event in recorder.read_events():
        if event.model and event.tokens:
            totals[event.model] += event.tokens
    return dict(totals)


__all__ = [
    "ModelCallCost",
    "NodeCost",
    "RateCard",
    "RunCost",
    "ThreadCost",
    "attribute",
    "attribute_thread",
    "by_node",
    "tokens_by_model",
]
