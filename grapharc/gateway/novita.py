"""Novita backend — a GPU cloud hosting open-weight models, one key.

`novita/moonshotai/kimi-k3` reaches Novita's own OpenAI-compatible endpoint
(`https://api.novita.ai/openai`), not api.openai.com, so this builds on
`OpenAICompatChatModel` the same way `openrouter.py` and `ollama.py` do rather
than on `openai.py`: the endpoint is fixed, not an override of OpenAI's own.

Model ids on Novita are themselves `author/slug` — `moonshotai/kimi-k3`,
`zai-org/glm-5.2` — the same shape OpenRouter uses, so `vendor()` in
`registry.py` already reads the right author off a Novita spec with no
backend-specific handling: `BACKEND_VENDOR` stays absent for `novita`, exactly
as it is absent for `openrouter`.

**Novita reports no per-call cost.** Unlike OpenRouter, the chat-completions
response carries token counts and nothing else, so this backend is the OpenAI
one in that respect: `_provider_cost` is the base class's `None`, and
`cost_ceiling_usd` counts calls in `SpendMeter.unpriced_calls` unless a caller
supplies `price_per_million=`.
"""

from __future__ import annotations

from typing import Any

from grapharc.gateway.config import novita_api_key
from grapharc.gateway.openai_compat import OpenAICompatChatModel

NOVITA_BASE_URL = "https://api.novita.ai/openai"


class NovitaError(Exception):
    """The Novita backend could not be constructed or used."""


class NovitaChatModel(OpenAICompatChatModel):
    """A LangChain chat model over Novita's OpenAI-compatible endpoint."""

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None) or novita_api_key()
        if not api_key:
            raise NovitaError(
                "No Novita API key found. Set NOVITA_API_KEY in the environment, "
                "or add one of NOVITA_API_KEY / novita-api-key to a .env file."
            )
        # One retry layer, not two — same reasoning as the OpenRouter backend.
        kwargs.setdefault("max_retries", 0)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None) or NOVITA_BASE_URL,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "grapharc-novita"


__all__ = ["NOVITA_BASE_URL", "NovitaChatModel", "NovitaError"]
