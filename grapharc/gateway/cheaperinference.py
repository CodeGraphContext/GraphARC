"""Cheaper Inference backend — an OpenAI-compatible gateway, one key.

`cheaperinference/gpt-5.4-mini` reaches Cheaper Inference's own
OpenAI-compatible endpoint (`https://api.cheaperinference.com/v1`), not
api.openai.com, so this builds on `OpenAICompatChatModel` the same way
`novita.py` and `openrouter.py` do rather than on `openai.py`: the endpoint is
fixed, not an override of OpenAI's own.

**Model ids here are bare** — `gpt-5.4-mini`, `claude-sonnet-5`,
`deepseek-v4-flash` — with no `author/` prefix, unlike OpenRouter and Novita.
So `vendor()` in `registry.py` has no author to read, and every spec on this
backend reads as the vendor `cheaperinference`. `BACKEND_VENDOR` deliberately
does not map it to one lab, because the gateway serves several. The
consequence for `different_providers` is the re-seller limit its docstring
already names: two `cheaperinference/…` specs always read as correlated, even
across labs, and `cheaperinference/gpt-5.4-mini` against `openai/gpt-5.4-mini`
reads as independent although both name the same OpenAI model.

**This backend reads no per-call cost.** `_provider_cost` is the base class's
`None`, so it is the OpenAI one in that respect: `cost_ceiling_usd` counts
calls in `SpendMeter.unpriced_calls` unless a caller supplies
`price_per_million=`.
"""

from __future__ import annotations

from typing import Any

from grapharc.gateway.config import cheaper_inference_api_key
from grapharc.gateway.openai_compat import OpenAICompatChatModel

CHEAPER_INFERENCE_BASE_URL = "https://api.cheaperinference.com/v1"


class CheaperInferenceError(Exception):
    """The Cheaper Inference backend could not be constructed or used."""


class CheaperInferenceChatModel(OpenAICompatChatModel):
    """A LangChain chat model over Cheaper Inference's OpenAI-compatible endpoint."""

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None) or cheaper_inference_api_key()
        if not api_key:
            raise CheaperInferenceError(
                "No Cheaper Inference API key found. Set CHEAPER_INFERENCE_API_KEY "
                "in the environment, or add one of CHEAPER_INFERENCE_API_KEY / "
                "cheaper-inference-api-key to a .env file."
            )
        # One retry layer, not two — same reasoning as the OpenRouter backend.
        kwargs.setdefault("max_retries", 0)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None) or CHEAPER_INFERENCE_BASE_URL,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "grapharc-cheaperinference"


__all__ = [
    "CHEAPER_INFERENCE_BASE_URL",
    "CheaperInferenceChatModel",
    "CheaperInferenceError",
]
