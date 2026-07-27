"""The model plane: one interface over several ways of reaching a model.

Backends are selected by spec string via `get_model`, so which provider serves
a node is configuration rather than code:

    get_model("claude-cli/claude-sonnet-5")             # subscription, no API key
    get_model("openrouter/anthropic/claude-sonnet-4.5") # ~400 models, one key
    get_model("mock/x", responses=[...])                # deterministic tests

`OpenRouterChatModel` is imported lazily — it needs `langchain-openai`, which
is an optional extra.
"""

from grapharc.gateway.claude_cli import ClaudeCodeCLIChatModel, GatewayError
from grapharc.gateway.config import openrouter_api_key, redact
from grapharc.gateway.registry import (
    UnknownBackendError,
    describe,
    different_providers,
    get_model,
    split_spec,
)

__all__ = [
    "ClaudeCodeCLIChatModel",
    "GatewayError",
    "UnknownBackendError",
    "describe",
    "different_providers",
    "get_model",
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
