"""The model plane: one interface over several ways of reaching a model.

Backends are selected by spec string via `get_model`, so which provider serves
a node is configuration rather than code:

    get_model("claude-cli/claude-sonnet-5")             # subscription, no API key
    get_model("openrouter/anthropic/claude-sonnet-4.5") # many providers, one key
    get_model("mock/x", responses=[...])                # deterministic tests

Two cross-backend policies live here rather than in either adapter, so a graph
gets the same behaviour whichever one serves it:

    get_model(spec, retry_policy=RetryPolicy(max_attempts=5))  # transient only
    get_model(spec, cost_ceiling_usd=0.25)                     # raises when passed
    get_model(spec, spend=shared_meter)                        # one ceiling, many models

`OpenRouterChatModel` is imported lazily — it needs `langchain-openai`, which
is an optional extra.
"""

from grapharc.gateway.claude_cli import ClaudeCodeCLIChatModel
from grapharc.gateway.config import openrouter_api_key, redact
from grapharc.gateway.errors import (
    CostCeilingExceeded,
    GatewayError,
    TransientGatewayError,
)
from grapharc.gateway.registry import (
    UnknownBackendError,
    describe,
    different_providers,
    get_model,
    split_spec,
)
from grapharc.gateway.resilience import (
    DEFAULT_RETRY_POLICY,
    NO_RETRY,
    RetryPolicy,
    call_with_retry,
    is_transient,
)
from grapharc.gateway.spend import SpendMeter

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "NO_RETRY",
    "ClaudeCodeCLIChatModel",
    "CostCeilingExceeded",
    "GatewayError",
    "RetryPolicy",
    "SpendMeter",
    "TransientGatewayError",
    "UnknownBackendError",
    "call_with_retry",
    "describe",
    "different_providers",
    "get_model",
    "is_transient",
    "openrouter_api_key",
    "redact",
    "split_spec",
]


def __getattr__(name: str):
    """Lazily expose the OpenRouter symbols so the extra stays optional."""
    if name in ("OpenRouterChatModel", "OpenRouterError"):
        from grapharc.gateway import openrouter

        return getattr(openrouter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
