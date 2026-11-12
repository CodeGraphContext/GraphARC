"""UberLLM backend — one key for models from many vendors, priced per call.

`uberllm/deepseek/deepseek-v4.1-flash` reaches UberLLM's own OpenAI-compatible
endpoint (`https://api.uberllm.dev/v1`), not api.openai.com, so this builds on
`OpenAICompatChatModel` the same way `openrouter.py` and `novita.py` do rather
than on `openai.py`: the endpoint is fixed, not an override of OpenAI's own.

Model ids on UberLLM are themselves `vendor/model` — `deepseek/deepseek-v4.1-flash`,
`anthropic/claude-sonnet-4.6` — the same shape OpenRouter uses, so `vendor()` in
`registry.py` already reads the right author off an UberLLM spec with no
backend-specific handling: `BACKEND_VENDOR` stays absent for `uberllm`.

**UberLLM prices its own calls.** The chat-completions usage block carries
`cost_usd`, what the call was billed, so like OpenRouter a `cost_ceiling_usd`
on this backend is an enforcement rather than a counter — except on streamed
turns, where the usage block arrives in the final SSE chunk and does not survive
chunk conversion.
"""

from __future__ import annotations

from typing import Any

from grapharc.gateway.config import uberllm_api_key
from grapharc.gateway.openai_compat import OpenAICompatChatModel

UBERLLM_BASE_URL = "https://api.uberllm.dev/v1"


class UberLLMError(Exception):
    """The UberLLM backend could not be constructed or used."""


class UberLLMChatModel(OpenAICompatChatModel):
    """A LangChain chat model over UberLLM's OpenAI-compatible endpoint."""

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None) or uberllm_api_key()
        if not api_key:
            raise UberLLMError(
                "No UberLLM API key found. Set UBERLLM_API_KEY in the environment, "
                "or add one of UBERLLM_API_KEY / uberllm-api-key to a .env file."
            )
        # One retry layer, not two — same reasoning as the OpenRouter backend.
        kwargs.setdefault("max_retries", 0)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None) or UBERLLM_BASE_URL,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "grapharc-uberllm"

    def _provider_cost(self, usage: dict[str, Any]) -> float | None:
        """UberLLM reports what the call was billed in its usage block."""
        return usage.get("cost_usd")


__all__ = ["UBERLLM_BASE_URL", "UberLLMChatModel", "UberLLMError"]
