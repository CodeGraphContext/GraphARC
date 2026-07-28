"""Test doubles for model-calling nodes.

Gate tests run against scripted models by default (`--live` runs real ones):
the gates verify *orchestration mechanics* — routing, budgets, traces, replay —
which scripted responses exercise deterministically.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a fixed script of responses.

    Each call returns the next scripted response, with usage_metadata attached
    so token metering paths are exercised. Calls beyond the script raise by
    default (`on_exhausted="raise"`) or repeat the last response
    (`on_exhausted="repeat"`).
    """

    responses: list[str] = Field(default_factory=list)
    on_exhausted: str = "raise"
    # Set `cost_usd` to make this double report a per-call price the way a real
    # backend does, through the same `llm_output` envelope OpenRouter uses. Left
    # None, the model reports no cost at all — which is also a real case, and the
    # one that makes `observe.cost` fall back to a rate-card estimate.
    cost_usd: float | None = None
    model_name: str = "scripted"
    last_usage: dict[str, Any] | None = Field(default=None, exclude=True)
    _cursor: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "grapharc-scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        if self._cursor >= len(self.responses):
            if self.on_exhausted == "repeat" and self.responses:
                content = self.responses[-1]
            else:
                raise RuntimeError(
                    f"ScriptedChatModel exhausted after {len(self.responses)} responses"
                )
        else:
            content = self.responses[self._cursor]
            self._cursor += 1

        input_tokens = sum(_estimate_tokens(str(m.content)) for m in messages)
        output_tokens = _estimate_tokens(content)
        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        # The same usage envelope both real gateways publish, so a caller that
        # reads a price off this double reads it the way it reads a real one.
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": self.cost_usd,
            "model": self.model_name,
        }
        if self.cost_usd is None:
            return ChatResult(generations=[ChatGeneration(message=message)])
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": {"cost": self.cost_usd},
                "model_name": self.model_name,
            },
        )


def charge_usage(ctx: Any, message: AIMessage) -> None:
    """Charge an AIMessage's usage_metadata against a RunContext's budget meter.

    Redundant now that the runtime's usage callback meters every model call
    automatically, and kept only so hand-metered nodes keep working. Naming the
    message is what lets the meter recognise this as a re-report of a call it
    already counted rather than a second, separate spend.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total:
        ctx.meter.charge_tokens(total, source=message)
