"""OpenAI backend — the API directly, with a key of your own.

`openai/gpt-4o-mini` goes straight to api.openai.com; nothing is routed through
a broker. Use it when you already have an OpenAI account, when a contract says
which company may see the prompt, or when you want a vendor that is genuinely
not Anthropic behind a reviewer node.

Everything a node needs from a chat model — `bind_tools`,
`with_structured_output`, streaming, async, the retry policy and the usage
envelope — comes from `OpenAICompatChatModel`. This adds credential
resolution, an endpoint override, and one honest limitation:

**The OpenAI API does not tell you what a call cost.** The response carries
token counts and no price, so `cost_usd` is None unless you supply a rate card,
and `cost_ceiling_usd` on this backend counts calls in
`SpendMeter.unpriced_calls` instead of enforcing a dollar limit. Two ways to
get a real number, both explicit because a price table baked into this repo
would go stale silently:

    # per-call cost, charged against the ceiling
    get_model("openai/gpt-4o-mini", price_per_million={"input": 0.15,
                                                       "cached_input": 0.075,
                                                       "output": 0.60})

    # or price the whole trace afterwards, with observe.cost.RateCard

Token counts are always real, so a token budget (`runtime.budget`) bites on
this backend whether or not you priced anything.

`OPENAI_BASE_URL` is honoured, which makes this the backend to point at any
OpenAI-compatible endpoint that is not Ollama — a gateway, a proxy, a
self-hosted vLLM. Note that such an endpoint is not api.openai.com and the
`_llm_type` will still say `grapharc-openai`; the spec string is what a trace
records, so name your model accordingly.
"""

from __future__ import annotations

from typing import Any

from grapharc.gateway.config import openai_api_key, openai_base_url
from grapharc.gateway.openai_compat import OpenAICompatChatModel


class OpenAIError(Exception):
    """The OpenAI backend could not be constructed or used."""


class OpenAIChatModel(OpenAICompatChatModel):
    """A LangChain chat model over the OpenAI API, metered the GraphARC way."""

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None) or openai_api_key()
        if not api_key:
            raise OpenAIError(
                "No OpenAI API key found. Set OPENAI_API_KEY in the environment, "
                "or add one of OPENAI_API_KEY / openai-api-key to a .env file."
            )
        # Only sent when something asked for it: the SDK knows its own default
        # base URL, and passing a copy of it from here would be one more string
        # to get wrong when it changes.
        base_url = kwargs.pop("base_url", None) or openai_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        # One retry layer, not two — same reasoning as the OpenRouter backend.
        kwargs.setdefault("max_retries", 0)
        # No `max_tokens` default: unlike OpenRouter, OpenAI reserves no credit
        # against it, so capping it here would only truncate replies for a
        # problem this backend does not have.
        super().__init__(model=model, api_key=api_key, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "grapharc-openai"


__all__ = ["OpenAIChatModel", "OpenAIError"]
