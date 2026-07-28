"""Which graphs this server is allowed to run.

The server never builds a graph from a request body — a name maps to a builder
the operator registered in Python. A request can pick from that set and supply
input and a budget; it cannot describe a topology.

A builder takes the session's `TraceRecorder` and returns a *fresh*
`CompiledGraphARC`. One compiled graph per session, not one shared instance:
each session needs its own trace file, and `CompiledGraphARC.last_run` is a
single mutable slot that concurrent sessions would otherwise overwrite for each
other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import CompiledGraphARC
from grapharc.server.errors import UnknownGraphError

GraphBuilder = Callable[[TraceRecorder], CompiledGraphARC]


class GraphRegistry:
    """A name → builder map, with unknown names failing loudly."""

    def __init__(self, builders: Mapping[str, GraphBuilder] | None = None) -> None:
        self._builders: dict[str, GraphBuilder] = dict(builders or {})

    def register(self, name: str, builder: GraphBuilder) -> GraphRegistry:
        if not name:
            raise ValueError("graph name must be a non-empty string")
        if name in self._builders:
            raise ValueError(f"graph {name!r} is already registered")
        self._builders[name] = builder
        return self

    def names(self) -> list[str]:
        return sorted(self._builders)

    def build(self, name: str, trace: TraceRecorder) -> CompiledGraphARC:
        try:
            builder = self._builders[name]
        except KeyError:
            raise UnknownGraphError(name, self._builders) from None
        return builder(trace)

    def __contains__(self, name: object) -> bool:
        return name in self._builders

    def __len__(self) -> int:
        return len(self._builders)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


__all__ = ["GraphBuilder", "GraphRegistry"]
