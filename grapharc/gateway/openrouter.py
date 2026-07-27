"""OpenRouter backend — one API over ~400 models from ~60 providers.

OpenRouter speaks the OpenAI wire format, so this subclasses `ChatOpenAI`
rather than hand-rolling a client. That is a deliberate trade: it brings
`bind_tools`, `with_structured_output`, streaming, and async for free, which
the Claude-CLI backend cannot offer and without which no agent node can call a
tool.

What this adds on top of `ChatOpenAI` is OpenRouter's routing semantics:

- **Two independent fallback layers.** `fallback_models` is model-level (try
  another model if this one fails); `provider_order` / `allow_provider_fallbacks`
  is provider-level (try another host serving the same model). Failed requests
  are not billed, so failover is safe to leave on.
- **Routing postures** — `sort="price"` or `"throughput"` or `"latency"`, plus
  the `:floor` / `:nitro` slug suffixes.
- **A uniform usage envelope** matching `ClaudeCodeCLIChatModel`, so a budget
  meter reads the same fields whichever backend produced the turn. Cached input
  counts as input; under-counting it is how budgets silently miss by an order
  of magnitude.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

from grapharc.gateway.config import openrouter_api_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter reserves credit against `max_tokens` before generating, so a model's
# full ceiling (often 64k+) is refused outright on a small balance with a 402 —
# even for a ten-token reply. Default to something a node actually needs and let
# callers raise it deliberately.
DEFAULT_MAX_TOKENS = 4096

# Sent for attribution on openrouter.ai; harmless if the site does not exist.
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/CodeGraphContext/GraphARC",
    "X-Title": "GraphARC",
}


class OpenRouterError(Exception):
    """The OpenRouter backend could not be constructed or used."""


def _strip_prefix(model: str) -> str:
    """`openrouter/anthropic/claude-sonnet-4.5` -> `anthropic/claude-sonnet-4.5`."""
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


class OpenRouterChatModel(ChatOpenAI):
    """A LangChain chat model over OpenRouter, with routing and usage accounting."""

    # Model-level fallback chain: tried in order if the primary fails.
    fallback_models: list[str] = Field(default_factory=list)
    # Provider-level routing within a single model.
    provider_order: list[str] = Field(default_factory=list)
    allow_provider_fallbacks: bool = True
    sort: str | None = None  # "price" | "throughput" | "latency"
    max_price_per_million: float | None = None
    require_parameters: bool = False

    _last_usage: dict[str, Any] | None = PrivateAttr(default=None)

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None) or openrouter_api_key()
        if not api_key:
            raise OpenRouterError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY in the "
                "environment, or add one of OPENROUTER_API_KEY / "
                "open-router-api-key to a .env file."
            )
        headers = {**DEFAULT_HEADERS, **(kwargs.pop("default_headers", None) or {})}
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        super().__init__(
            model=_strip_prefix(model),
            api_key=api_key,
            base_url=kwargs.pop("base_url", None) or OPENROUTER_BASE_URL,
            default_headers=headers,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "grapharc-openrouter"

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """Uniform usage envelope — same shape the CLI backend reports."""
        return self._last_usage

    def _routing_body(self) -> dict[str, Any]:
        """OpenRouter-specific request fields, omitted entirely when unused."""
        body: dict[str, Any] = {"usage": {"include": True}}
        if self.fallback_models:
            body["models"] = [self.model_name, *map(_strip_prefix, self.fallback_models)]
        provider: dict[str, Any] = {}
        if self.provider_order:
            provider["order"] = self.provider_order
        if not self.allow_provider_fallbacks:
            provider["allow_fallbacks"] = False
        if self.sort:
            provider["sort"] = self.sort
        if self.max_price_per_million is not None:
            provider["max_price"] = {"prompt": self.max_price_per_million}
        if self.require_parameters:
            provider["require_parameters"] = True
        if provider:
            body["provider"] = provider
        return body

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # OpenRouter's routing fields are not OpenAI parameters, so the SDK
        # rejects them at the top level. `extra_body` is merged into the JSON
        # body verbatim, which is where they belong.
        extra_body = {**(payload.get("extra_body") or {}), **self._routing_body()}
        payload["extra_body"] = extra_body
        return payload

    def _record_usage(self, result: ChatResult) -> None:
        """Fold cached input into the total and normalize into one envelope."""
        info = result.llm_output or {}
        usage = info.get("token_usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cache_read = int(details.get("cached_tokens") or 0)
        # OpenRouter reports prompt_tokens inclusive of cached reads.
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_creation": 0, "cache_read": cache_read},
            "uncached_input_tokens": max(0, input_tokens - cache_read),
            "cost_usd": usage.get("cost"),
            "model": info.get("model_name") or self.model_name,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._record_usage(result)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        self._record_usage(result)
        return result
