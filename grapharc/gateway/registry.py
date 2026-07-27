"""Resolve a model spec string to a backend.

A graph should name the capability it wants, not the vendor it came from — so
nodes take a spec string and the registry decides which adapter serves it:

    claude-cli/claude-sonnet-5              -> Claude Code CLI (subscription, no key)
    openrouter/anthropic/claude-sonnet-4.5  -> OpenRouter
    openrouter/openai/gpt-4o:floor          -> OpenRouter, cheapest provider
    mock/anything                           -> scripted test double

This is the seam that makes the backend a config change rather than a rewrite,
and it is what lets a reviewer run on a genuinely different *provider* from the
author instead of merely a different object.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_BACKEND = "claude-cli"


class UnknownBackendError(Exception):
    """The spec named a backend that is not registered."""


BACKENDS = ("claude-cli", "openrouter", "mock")

# Authors that appear in OpenRouter model ids. A spec starting with one of
# these is a model name, not a mistyped backend — `anthropic/claude-haiku-4.5`
# is not someone failing to spell `openrouter`.
KNOWN_AUTHORS = (
    "anthropic", "openai", "google", "meta-llama", "mistralai", "deepseek",
    "qwen", "x-ai", "cohere", "nvidia", "perplexity", "amazon", "microsoft",
    "z-ai", "moonshotai", "inclusionai", "nousresearch", "openrouter",
)


def split_spec(spec: str) -> tuple[str, str]:
    """Split `backend/model` — bare names get the default backend.

    OpenRouter ids are themselves `author/slug`, so only the first segment is
    treated as a backend, and the remainder passes through intact.

    A slash-prefixed spec whose head is neither a known backend nor a known
    model author is rejected rather than silently folded into a model name —
    otherwise a typo like `opnerouter/anthropic/x` becomes a Claude-CLI call
    with a nonsense model and fails much later with a confusing error.
    """
    head, sep, rest = spec.partition("/")
    if sep and head in BACKENDS:
        return head, rest
    if sep and head not in KNOWN_AUTHORS:
        raise UnknownBackendError(
            f"unknown backend {head!r} in spec {spec!r}; expected one of: "
            f"{', '.join(BACKENDS)} — or a bare model name for the "
            f"{DEFAULT_BACKEND} default"
        )
    return DEFAULT_BACKEND, spec


def get_model(spec: str, **kwargs: Any) -> BaseChatModel:
    """Build a chat model from a spec string.

    Imports are local so a missing optional dependency only fails for the
    backend that needs it — asking for OpenRouter should not require the
    Claude CLI, and vice versa.
    """
    backend, model = split_spec(spec)

    if backend == "claude-cli":
        from grapharc.gateway.claude_cli import ClaudeCodeCLIChatModel

        return ClaudeCodeCLIChatModel(model=model, **kwargs)

    if backend == "openrouter":
        try:
            from grapharc.gateway.openrouter import OpenRouterChatModel
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise UnknownBackendError(
                "The OpenRouter backend needs langchain-openai. "
                "Install it with: uv sync --extra openrouter"
            ) from exc

        return OpenRouterChatModel(model, **kwargs)

    if backend == "mock":
        from grapharc.testing import ScriptedChatModel

        return ScriptedChatModel(**kwargs)

    raise UnknownBackendError(  # pragma: no cover - split_spec guards this
        f"unknown backend {backend!r} in spec {spec!r}; "
        f"expected one of: {', '.join(BACKENDS)}"
    )


def describe(spec: str) -> dict[str, str]:
    """What a spec resolves to, for logs and CLI output. Never touches secrets."""
    backend, model = split_spec(spec)
    return {"spec": spec, "backend": backend, "model": model}


def different_providers(a: str, b: str) -> bool:
    """True when two specs come from genuinely different providers.

    The correlated-agreement guard wants real independence. Two OpenRouter
    specs from the same author (`anthropic/…` vs `anthropic/…`) share a model
    family and are weaker evidence than a cross-vendor pair.
    """
    backend_a, model_a = split_spec(a)
    backend_b, model_b = split_spec(b)
    if backend_a != backend_b:
        return True
    author_a = model_a.split("/")[0] if "/" in model_a else backend_a
    author_b = model_b.split("/")[0] if "/" in model_b else backend_b
    return author_a != author_b
