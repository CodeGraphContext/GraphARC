"""GraphARC — a graph engineering toolkit.

Disciplined multi-agent graphs on top of LangGraph: typed state contracts,
per-node write permissions, budgets, convergence guards, verification,
replay, and (eventually) durable graph memory.
"""

from grapharc.runtime.budget import Budget, BudgetExceeded, BudgetMeter
from grapharc.runtime.graph import ArcGraph, WritePermissionError
from grapharc.runtime.state import ArcState

__version__ = "0.1.0a0"

__all__ = [
    "ArcGraph",
    "ArcState",
    "Budget",
    "BudgetExceeded",
    "BudgetMeter",
    "WritePermissionError",
    "__version__",
]
