"""Typed shared state.

The state object is the graph's contract: every node reads from it and writes
to it, so a sloppy write in node two becomes a confident input for node five.
GraphARC states are Pydantic models with unknown fields forbidden — a typo'd
update key fails loudly at the edge instead of silently polluting downstream
nodes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GraphARCState(BaseModel):
    """Base class for graph state schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
