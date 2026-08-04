"""Observability: the trace file, and everything derived from it.

`trace` is the only writer. `metrics`, `replay`, `cost` and `otel` all read
that one file and nothing else, which is what keeps a dashboard from
contradicting an audit trail.
"""

from grapharc.observe.cost import (
    ModelCallCost,
    NodeCost,
    RateCard,
    RunCost,
    ThreadCost,
    attribute,
    attribute_thread,
    by_node,
    tokens_by_model,
)
from grapharc.observe.metrics import RunMetrics, summarize, to_mermaid
from grapharc.observe.otel import (
    ListSpanExporter,
    NullSpanExporter,
    OTelSpanExporter,
    OTelUnavailable,
    Span,
    SpanExporter,
    export_run,
    to_spans,
)
from grapharc.observe.replay import (
    NodeDiff,
    NodeExecution,
    ReplayedRun,
    ReplayError,
    RunDiff,
    diff_runs,
    diff_trace,
    format_diff,
    format_replay,
    replay,
    replay_thread,
)
from grapharc.observe.status import NodeState, NodeStatus, node_states
from grapharc.observe.trace import TraceEvent, TraceReadError, TraceRecorder, load_events

__all__ = [
    "ListSpanExporter",
    "ModelCallCost",
    "NodeCost",
    "NodeDiff",
    "NodeExecution",
    "NodeState",
    "NodeStatus",
    "NullSpanExporter",
    "OTelSpanExporter",
    "OTelUnavailable",
    "RateCard",
    "ReplayError",
    "ReplayedRun",
    "RunCost",
    "RunDiff",
    "RunMetrics",
    "Span",
    "SpanExporter",
    "ThreadCost",
    "TraceEvent",
    "TraceReadError",
    "TraceRecorder",
    "attribute",
    "attribute_thread",
    "by_node",
    "diff_runs",
    "diff_trace",
    "export_run",
    "format_diff",
    "format_replay",
    "load_events",
    "node_states",
    "replay",
    "replay_thread",
    "summarize",
    "to_mermaid",
    "to_spans",
    "tokens_by_model",
]
