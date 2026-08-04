"""Ollama backend — models running on your own machine, no key and no bill.

`ollama/llama3.1` talks to the Ollama daemon's OpenAI-compatible endpoint
(`/v1` on `OLLAMA_HOST`, localhost by default), so it arrives with the same
`bind_tools` / `with_structured_output` / streaming / async surface as the
other OpenAI-wire backends and the same usage envelope. That makes it the
cheapest way to exercise a graph that needs tool-calling — the Claude-CLI
backend cannot drive an agent node at all, and OpenRouter charges for every
attempt.

Three things are different enough to state plainly:

- **Cost is zero, and that is a fact rather than a missing number.** No
  provider invoices you for a local process, so calls are charged 0.0 and do
  *not* land in `unpriced_calls`. Electricity and a hot GPU are real costs and
  are not provider charges; if you want them attributed anyway, pass
  `price_per_million=` and that card is used instead. A `cost_ceiling_usd` on
  an unpriced Ollama model will therefore never fire, which is correct.
- **There is no key by default.** The OpenAI client still requires a non-empty
  one, so a placeholder is sent and the daemon ignores it. Set `OLLAMA_API_KEY`
  when the address points at an authenticating proxy rather than the daemon.
- **Tool-calling depends on the model you pulled, not on this adapter.** Ollama
  accepts a `tools` array for every model and silently returns prose for one
  that was not trained to emit tool calls. `bind_tools` therefore cannot raise
  the way the Claude-CLI backend's does; the failure shows up as an agent loop
  that never calls a tool. Pull a model whose card says it supports tools.

Nothing here checks that the server is running. A stopped daemon surfaces as a
connection error on the first call, and `grapharc models --check` reports
configuration only — it sends no request.
"""

from __future__ import annotations

from typing import Any, ClassVar

from grapharc.gateway.config import ollama_api_key, ollama_base_url
from grapharc.gateway.openai_compat import OpenAICompatChatModel

#: Ollama ignores the Authorization header; the OpenAI client refuses to send
#: an empty one. This placeholder is not a credential and is never redacted
#: because there is nothing in it to protect.
PLACEHOLDER_API_KEY = "ollama"


class OllamaError(Exception):
    """The Ollama backend could not be constructed or used."""


class OllamaChatModel(OpenAICompatChatModel):
    """A LangChain chat model over a local Ollama server."""

    #: Read by `PlannerNode` (duck-typed, like `disclosure()`): Ollama compiles
    #: a strict `json_schema` into its grammar-constrained decoder, and
    #: `Subgraph`'s schema — recursive, every field required, docstring-laden —
    #: reliably breaks small models there; `json_mode` suppresses the reasoning
    #: phase models like qwen3 plan with. The text path with the slim proposal
    #: shape is the one that works, so this backend asks for it.
    reliable_structured_output: ClassVar[bool] = False

    def __init__(self, model: str, /, **kwargs: Any) -> None:
        if not model:
            raise OllamaError(
                "No Ollama model named. Use a spec like `ollama/llama3.1` — the "
                "model half must be a tag you have pulled (`ollama list`)."
            )
        kwargs.setdefault("base_url", ollama_base_url())
        kwargs.setdefault("api_key", ollama_api_key() or PLACEHOLDER_API_KEY)
        # One retry layer, not two — same reasoning as the OpenRouter backend.
        kwargs.setdefault("max_retries", 0)
        super().__init__(model=model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "grapharc-ollama"

    def _provider_cost(self, usage: dict[str, Any]) -> float | None:
        """Free, unless the caller asked for their own tokens to be priced.

        Returning 0.0 rather than None is the load-bearing part: None would
        count the call as unpriced, which reads as "the meter missed this one"
        when in fact the meter knows exactly what it cost.
        """
        return None if self.price_per_million else 0.0


__all__ = ["PLACEHOLDER_API_KEY", "OllamaChatModel", "OllamaError"]
