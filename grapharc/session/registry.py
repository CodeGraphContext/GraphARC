"""Graphs by name, so a session id is enough to rebuild the thing it was running.

A session record can only store a *name*. The code that builds the graph cannot
be pickled into SQLite and should not be: rehydrating a session must give you
the graph as it exists in the process doing the rehydrating, not a frozen copy
of one from last week. So a process declares what it can run, and a resume
fails loudly when the name is not among them.

Binding is also where the session contract is checked. Two things are verified
before a session record exists rather than discovered mid-run:

- the graph's state schema derives from `SessionState`, so the fields the
  runtime writes are declared, and
- every node named as needing approval is actually a node in the graph.

A typo in an approval-node name is otherwise the worst failure available: the
gate silently never fires and the node runs unheld.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from grapharc.runtime.graph import CompiledGraphARC
from grapharc.session.errors import SessionContractError, UnknownGraphError
from grapharc.session.state import SESSION_FIELDS, SessionState

#: Builds a compiled graph bound to the session layer's checkpointer.
GraphFactory = Callable[[Any], CompiledGraphARC]

#: Renders "what is this node about to do" for an approval request, from the
#: node name and the checkpointed state it would have run against.
DescribeAction = Callable[[str, Mapping[str, Any]], str]


@dataclass(frozen=True, eq=False)
class GraphSpec:
    """A named graph a session can be bound to."""

    name: str
    factory: GraphFactory
    approval_nodes: tuple[str, ...] = ()
    describe: DescribeAction | None = None

    def build(self, checkpointer: Any) -> CompiledGraphARC:
        """Compile the graph and check it can actually be driven by a session."""
        compiled = self.factory(checkpointer)
        schema = compiled.arc.state_schema
        if not (isinstance(schema, type) and issubclass(schema, SessionState)):
            raise SessionContractError(
                f"graph {self.name!r} uses state schema {schema.__name__}, which is not a "
                f"SessionState subclass; a session writes {list(SESSION_FIELDS)} and those "
                "fields must be declared"
            )
        # `inner.nodes` is LangGraph's own node table; GraphARC does not expose
        # one (ROADMAP §1.3), and guessing is not good enough for this check.
        known = set(compiled.inner.nodes)
        unknown = [node for node in self.approval_nodes if node not in known]
        if unknown:
            raise SessionContractError(
                f"graph {self.name!r} declares approval nodes {unknown} that it does not "
                f"contain; known nodes: {sorted(known)}"
            )
        return compiled

    def describe_action(self, node: str, values: Mapping[str, Any]) -> str:
        if self.describe is None:
            return f"run node {node!r}"
        return self.describe(node, values)


class GraphRegistry:
    """Name -> `GraphSpec`. One per process by default; instantiate for isolation."""

    def __init__(self) -> None:
        self._specs: dict[str, GraphSpec] = {}
        self._lock = threading.Lock()

    def register(self, spec: GraphSpec, *, replace: bool = False) -> GraphSpec:
        with self._lock:
            if spec.name in self._specs and not replace:
                raise SessionContractError(
                    f"graph {spec.name!r} is already registered; pass replace=True to "
                    "override it deliberately"
                )
            self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> GraphSpec:
        with self._lock:
            spec = self._specs.get(name)
        if spec is None:
            raise UnknownGraphError(
                f"graph {name!r} is not registered in this process; registered: "
                f"{sorted(self._specs)}"
            )
        return spec

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._specs))


#: Process-wide default. Modules register into it at import time.
REGISTRY = GraphRegistry()


def register_graph(
    name: str,
    factory: GraphFactory,
    *,
    approval_nodes: tuple[str, ...] = (),
    describe: DescribeAction | None = None,
    registry: GraphRegistry | None = None,
    replace: bool = False,
) -> GraphSpec:
    """Declare that this process can build `name`."""
    spec = GraphSpec(
        name=name, factory=factory, approval_nodes=tuple(approval_nodes), describe=describe
    )
    return (registry or REGISTRY).register(spec, replace=replace)


def get_graph_spec(name: str, *, registry: GraphRegistry | None = None) -> GraphSpec:
    return (registry or REGISTRY).get(name)


__all__ = [
    "REGISTRY",
    "DescribeAction",
    "GraphFactory",
    "GraphRegistry",
    "GraphSpec",
    "get_graph_spec",
    "register_graph",
]
