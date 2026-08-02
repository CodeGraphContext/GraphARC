"""OpenTelemetry export — optional by construction.

GraphARC does not depend on `opentelemetry-api` or `opentelemetry-sdk`, and
importing this module does not either. The dependency is confined to
`OTelSpanExporter`, which imports it lazily and raises `OTelUnavailable` with an
install line if it is not there. Everything above that line — turning a trace
into a span tree — is ordinary code with no optional import in it, so it is
testable and tested in a tree without the package.

The seam is the `SpanExporter` Protocol. The default is `NullSpanExporter`,
which does nothing; a caller who wants spans supplies something that satisfies
the Protocol, whether that is the OTel adapter here, a collector of their own,
or a list in a test.

Span construction, precisely:

- One root span per run. One child span per node execution, from its `start`
  event, ended at `start + duration_ms`.
- Sub-step events (`AgentNode`'s `"model"` / `"tool"` / `"stop"`) become
  children of the node execution `replay` attributed them to. That attribution
  is inference — the trace records no parent pointer — so a sub-step whose
  parent could not be identified is parented to the run span rather than to a
  guess.
- Timestamps come from the event's `ts`, which the recorder writes at
  millisecond resolution. Span boundaries are therefore accurate to about a
  millisecond, and a node's end is *derived* from its recorded duration rather
  than observed. Two spans can legitimately show the same start nanosecond.

Honest status: `to_spans` and the Protocol are covered by tests.
`OTelSpanExporter` is exercised against a stub that implements the OTel tracer
API's shape; no real `opentelemetry` package was installed in the tree it was
written in, so treat it as unverified against the real SDK until someone runs
it with the dependency present.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from grapharc.observe.replay import ReplayedRun, replay
from grapharc.observe.trace import TraceEvent, TraceRecorder

TRACER_NAME = "grapharc"


class OTelUnavailable(ImportError):
    """`opentelemetry` is not installed, and this exporter needs it."""


class Span(BaseModel):
    """A span, in the subset of the OTel data model a trace event can fill."""

    name: str
    span_id: str
    parent_id: str | None = None
    start_unix_nano: int
    end_unix_nano: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "unset"  # "unset" | "ok" | "error"
    error: str | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_unix_nano - self.start_unix_nano


@runtime_checkable
class SpanExporter(Protocol):
    """Anything that can receive spans. The whole optional-dependency seam."""

    def export(self, spans: Sequence[Span]) -> None: ...


class NullSpanExporter:
    """The default: accepts spans and does nothing with them.

    Present so that "no exporter configured" is a working exporter rather than
    a None check at every call site, and so nothing breaks without the
    optional dependency.
    """

    def export(self, spans: Sequence[Span]) -> None:
        return None


class ListSpanExporter:
    """Keeps spans in memory. For tests, and for reading a run in a REPL."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, spans: Sequence[Span]) -> None:
        self.spans.extend(spans)


def _to_nanos(ts: str) -> int | None:
    try:
        stamp = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _delta_attributes(delta: dict[str, Any]) -> dict[str, Any]:
    """A state delta as `grapharc.delta.*` attributes.

    Non-primitive values are stringified: OTel attributes are primitives or
    sequences of them, and a nested dict an exporter silently drops is worse
    than a `repr` that survives.
    """
    return {
        f"grapharc.delta.{key}": (
            value if isinstance(value, (str, int, float, bool)) else repr(value)
        )
        for key, value in delta.items()
    }


def _attributes(event: TraceEvent) -> dict[str, Any]:
    """Every field of one trace event, as span attributes."""
    attributes: dict[str, Any] = {
        "grapharc.run_id": event.run_id,
        "grapharc.graph": event.graph,
        "grapharc.node": event.node,
        "grapharc.phase": event.phase,
        "grapharc.step": event.step,
        "grapharc.attempt": event.attempt,
    }
    if event.thread_id is not None:
        attributes["grapharc.thread_id"] = event.thread_id
    if event.tokens is not None:
        attributes["grapharc.tokens"] = event.tokens
    if event.cost_usd is not None:
        attributes["grapharc.cost_usd"] = event.cost_usd
    if event.model is not None:
        attributes["grapharc.model"] = event.model
    attributes.update(_delta_attributes(event.state_delta or {}))
    return attributes


def to_spans(
    source: TraceRecorder | str | Path | ReplayedRun, run_id: str | None = None
) -> list[Span]:
    """Build the span tree for one run: root, node spans, sub-step spans.

    Returned parent-before-child, which is the order an exporter that builds
    real OTel spans has to emit them in.
    """
    if isinstance(source, ReplayedRun):
        run = source
    else:
        if run_id is None:
            raise TypeError("to_spans() needs a run_id unless given a ReplayedRun")
        run = replay(source, run_id)

    stamps = [n for n in (_to_nanos(e.ts) for e in run.events) if n is not None]
    if not stamps:
        return []
    root_start = min(stamps)
    root_id = f"{run.run_id}:run"

    spans: list[Span] = []
    root_end = root_start
    for execution in run.executions:
        start = _to_nanos(execution.started_at)
        if start is None:
            continue
        span_id = f"{run.run_id}:{execution.step}:{execution.attempt}"
        end = start + int((execution.duration_ms or 0.0) * 1_000_000)
        root_end = max(root_end, end)
        node_span = Span(
            name=execution.node,
            span_id=span_id,
            parent_id=root_id,
            start_unix_nano=start,
            end_unix_nano=end,
            attributes={
                "grapharc.run_id": run.run_id,
                "grapharc.graph": run.graph,
                "grapharc.node": execution.node,
                "grapharc.step": execution.step,
                "grapharc.attempt": execution.attempt,
                "grapharc.tokens": execution.tokens,
                **_delta_attributes(execution.state_delta),
            },
            status="error" if execution.error else ("ok" if execution.ok else "unset"),
            error=execution.error,
        )
        if execution.cost_usd is not None:
            node_span.attributes["grapharc.cost_usd"] = execution.cost_usd
        spans.append(node_span)
        for sub in execution.sub_events:
            sub_span = _sub_span(run, sub, parent_id=span_id)
            if sub_span is not None:
                root_end = max(root_end, sub_span.end_unix_nano)
                spans.append(sub_span)

    for orphan in run.orphan_sub_events:
        if orphan.phase == "topology":
            # The graph's declared shape, not work that happened: it has no
            # duration and no parent, and a span for it would report the act of
            # stating the topology as if it were an executed step.
            continue
        # Parented to the run rather than to a guessed node: see `replay._attach`.
        sub_span = _sub_span(run, orphan, parent_id=root_id)
        if sub_span is not None:
            root_end = max(root_end, sub_span.end_unix_nano)
            spans.append(sub_span)

    root = Span(
        name=run.graph,
        span_id=root_id,
        parent_id=None,
        start_unix_nano=root_start,
        end_unix_nano=max(root_end, max(stamps)),
        attributes={
            "grapharc.run_id": run.run_id,
            "grapharc.graph": run.graph,
            "grapharc.thread_id": run.thread_id or run.run_id,
            "grapharc.nodes": len(run.path),
            "grapharc.tokens": run.tokens,
            "grapharc.termination_reason": run.termination_reason or "",
        },
        status="error" if run.errors else "ok",
    )
    return [root, *spans]


def _sub_span(run: ReplayedRun, event: TraceEvent, *, parent_id: str) -> Span | None:
    start = _to_nanos(event.ts)
    if start is None:
        return None
    return Span(
        name=event.node,
        span_id=f"{run.run_id}:{event.step}:{event.attempt}:{event.phase}",
        parent_id=parent_id,
        start_unix_nano=start,
        end_unix_nano=start + int((event.duration_ms or 0.0) * 1_000_000),
        attributes=_attributes(event),
        status="error" if event.error else "ok",
        error=event.error,
    )


def export_run(
    source: TraceRecorder | str | Path | ReplayedRun,
    run_id: str | None = None,
    *,
    exporter: SpanExporter | None = None,
) -> list[Span]:
    """Build a run's spans and hand them to `exporter` (default: no-op).

    Returns the spans either way, so a caller with no exporter still gets
    something to assert on.
    """
    spans = to_spans(source, run_id)
    (exporter or NullSpanExporter()).export(spans)
    return spans


def _require_otel() -> Any:
    try:
        from opentelemetry import trace as otel_trace
    except ImportError as exc:  # pragma: no cover — covered by a stubbed-import test
        raise OTelUnavailable(
            "OpenTelemetry export needs the opentelemetry API, which GraphARC does "
            "not depend on: pip install opentelemetry-sdk. Without it, spans still "
            "build (observe.otel.to_spans) and NullSpanExporter is the default."
        ) from exc
    return otel_trace


class OTelSpanExporter:
    """Emits GraphARC spans through an installed OpenTelemetry tracer.

    Parentage is rebuilt with `set_span_in_context`, so a node span nests under
    its run span in whatever backend the caller configured. Spans are created
    and ended with explicit `start_time` / `end_time` in nanoseconds — the
    times the run actually happened, not the times it was exported, which is
    the whole point of exporting from a trace file after the fact.

    Not verified against the real SDK: see the module docstring.
    """

    def __init__(self, tracer: Any = None, *, tracer_provider: Any = None) -> None:
        self._otel = _require_otel()
        if tracer is None:
            provider = tracer_provider or self._otel.get_tracer_provider()
            tracer = provider.get_tracer(TRACER_NAME)
        self._tracer = tracer

    def export(self, spans: Sequence[Span]) -> None:
        contexts: dict[str, Any] = {}
        for span in spans:
            parent = contexts.get(span.parent_id) if span.parent_id else None
            live = self._tracer.start_span(
                span.name,
                context=parent,
                start_time=span.start_unix_nano,
                attributes=dict(span.attributes),
            )
            if span.status == "error":
                live.set_status(
                    self._otel.Status(self._otel.StatusCode.ERROR, span.error or "error")
                )
            elif span.status == "ok":
                live.set_status(self._otel.Status(self._otel.StatusCode.OK))
            # Recorded before `end`: a child span created after its parent ended
            # is still correctly parented, and spans arrive parent-first.
            contexts[span.span_id] = self._otel.set_span_in_context(live)
            live.end(end_time=span.end_unix_nano)


__all__ = [
    "TRACER_NAME",
    "ListSpanExporter",
    "NullSpanExporter",
    "OTelSpanExporter",
    "OTelUnavailable",
    "Span",
    "SpanExporter",
    "export_run",
    "to_spans",
]
