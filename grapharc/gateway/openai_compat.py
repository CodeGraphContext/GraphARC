"""What the OpenAI-wire-format backends share.

OpenRouter, OpenAI and Ollama all speak the same protocol, so all three
subclass `ChatOpenAI` — which brings `bind_tools`, `with_structured_output`,
streaming and async for free, none of which the Claude-CLI backend can offer.
What `ChatOpenAI` does *not* bring is the two policies GraphARC applies across
every backend, and they live here rather than being written three times:

- **A uniform usage envelope.** Same field names the Claude-CLI backend
  reports, so a budget meter reads the same keys whichever backend produced the
  turn. Cached input is folded into the input total; under-counting it is how
  budgets silently miss by an order of magnitude.
- **An explicit retry policy and an enforced cost ceiling.** The SDK's own
  `max_retries` is defaulted to 0 by each subclass so the two do not compose
  into a silent `max_attempts * sdk_retries` fan-out. Retries happen *inside*
  `_generate`, below the callback boundary, so a retried call fires one
  `on_llm_end` and is metered once.

**Where the money comes from differs, and that is the one thing a subclass must
decide.** OpenRouter returns a per-call cost in the response body. The OpenAI
API returns none, and Ollama has none to return. So:

- `_provider_cost` is the hook for a cost the provider itself reported;
- `price_per_million` is an optional caller-supplied rate card, used when the
  provider reported nothing — opt-in, because a price table baked into this
  repo would be stale the first time a vendor changed one and nobody would
  notice;
- a call with neither lands in `SpendMeter.unpriced_calls`. A ceiling on a
  backend that reports no cost is then a counter, not an enforcement, and the
  meter says so rather than implying it saw the whole bill.

Streaming is a hole in the accounting for every one of these backends, and it
is a hole rather than a gap in the docs. LangChain routes a streamed call
through `_stream`, never `_generate`, so a streamed turn is checked against the
ceiling *before* it starts but its cost is not recorded afterwards: the usage
block arrives in the final SSE chunk and `langchain-openai` does not surface
it. Those calls are counted as unpriced. They are not retried either — tokens
already handed to the caller cannot be un-handed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

from grapharc.gateway.resilience import (
    DEFAULT_RETRY_POLICY,
    NO_RETRY,
    RetryPolicy,
    acall_with_retry,
    call_with_retry,
)
from grapharc.gateway.spend import SpendMeter


class OpenAICompatChatModel(ChatOpenAI):
    """A `ChatOpenAI` with GraphARC's retry policy, usage envelope and ceiling."""

    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY
    # Seeds `spend.ceiling_usd` at construction. Pass `spend=` instead to share
    # one ceiling across several model objects.
    cost_ceiling_usd: float | None = None
    spend: SpendMeter = Field(default_factory=SpendMeter, exclude=True)
    #: USD per million tokens: {"input": …, "output": …, "cached_input": …}.
    #: Used only when the provider reported no cost of its own. `cached_input`
    #: defaults to the `input` rate, which over-states a discount rather than
    #: inventing one.
    price_per_million: dict[str, float] | None = None

    _last_usage: dict[str, Any] | None = PrivateAttr(default=None)
    _retries: int = PrivateAttr(default=0)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.cost_ceiling_usd is not None and self.spend.ceiling_usd is None:
            self.spend.ceiling_usd = self.cost_ceiling_usd

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """Uniform usage envelope — same shape the CLI backend reports."""
        return self._last_usage

    # ------------------------------------------------------------- resilience

    def _effective_policy(self) -> RetryPolicy:
        """No retries while streaming.

        `ChatOpenAI._generate` consumes `_stream` itself when `streaming=True`,
        so by the time a mid-stream failure surfaces the caller has already been
        handed tokens through `on_llm_new_token`. Re-issuing would replay them.
        """
        return NO_RETRY if self.streaming else self.retry_policy

    def _count_retry(self, attempt: int, delay: float, exc: BaseException) -> None:
        self._retries = attempt

    # ------------------------------------------------------------------ money

    def _provider_cost(self, usage: dict[str, Any]) -> float | None:
        """The cost this provider reported, or None if it reports none.

        The base answer is None: the OpenAI wire format has no price field, and
        a backend that fills one in (OpenRouter) is the exception.
        """
        return None

    def _cost_usd(
        self,
        usage: dict[str, Any],
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> float | None:
        """Provider's own figure first, then a caller's rate card, then None."""
        reported = self._provider_cost(usage)
        if reported is not None:
            return float(reported)
        card = self.price_per_million
        if not card:
            return None
        cached_rate = card.get("cached_input", card.get("input", 0.0))
        uncached = max(0, input_tokens - cached_tokens)
        return (
            uncached * card.get("input", 0.0)
            + cached_tokens * cached_rate
            + output_tokens * card.get("output", 0.0)
        ) / 1_000_000

    def _record_usage(self, result: ChatResult) -> None:
        """Fold cached input into the total and normalize into one envelope.

        Written before the ceiling is enforced, so a caller that catches
        `CostCeilingExceeded` can still read what the last call cost.
        """
        info = result.llm_output or {}
        usage = info.get("token_usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cache_read = int(details.get("cached_tokens") or 0)
        # Every one of these providers reports prompt_tokens inclusive of
        # cached reads.
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        cost_usd = self._cost_usd(
            usage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cache_read,
        )
        model = info.get("model_name") or self.model_name
        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_creation": 0, "cache_read": cache_read},
            "uncached_input_tokens": max(0, input_tokens - cache_read),
            "cost_usd": cost_usd,
            "model": model,
            "retries": self._retries,
            "cumulative_cost_usd": self.spend.spent_usd + float(cost_usd or 0.0),
        }

    def _settle(self, result: ChatResult) -> ChatResult:
        self._record_usage(result)
        usage = self._last_usage or {}
        self.spend.charge(usage.get("cost_usd"), model=str(usage.get("model")))
        return result

    # ------------------------------------------------------------ generation

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.spend.ensure_headroom(model=self.model_name)
        self._retries = 0
        result = call_with_retry(
            lambda: super(OpenAICompatChatModel, self)._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ),
            policy=self._effective_policy(),
            on_retry=self._count_retry,
        )
        return self._settle(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.spend.ensure_headroom(model=self.model_name)
        self._retries = 0
        result = await acall_with_retry(
            lambda: super(OpenAICompatChatModel, self)._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ),
            policy=self._effective_policy(),
            on_retry=self._count_retry,
        )
        return self._settle(result)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Streamed turns: ceiling checked up front, cost recorded as unpriced.

        LangChain dispatches here instead of `_generate` for any streamed call,
        so this is the only place a ceiling can bite on that path. What it
        cannot do is charge: the usage block arrives in the final SSE chunk and
        does not survive `langchain-openai`'s chunk conversion. The envelope is
        cleared rather than left holding a previous call's numbers.
        """
        self.spend.ensure_headroom(model=self.model_name)
        self._last_usage = None
        yield from super()._stream(*args, **kwargs)
        self.spend.charge(None, model=self.model_name)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        self.spend.ensure_headroom(model=self.model_name)
        self._last_usage = None
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
        self.spend.charge(None, model=self.model_name)


__all__ = ["OpenAICompatChatModel"]
