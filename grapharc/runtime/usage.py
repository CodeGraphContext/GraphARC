"""Automatic token accounting for model calls made inside a node.

`Budget.max_tokens` is only a ceiling if something charges the meter. While that
was the node author's job it was advisory: a live capstone run over OpenRouter
reported `spent: 0 tokens across 7 nodes` while costing real money, because the
one node that called the reviewer never charged.

The mechanism is a LangChain callback installed for the duration of every node.
langchain-core keeps a registry of context variables (`register_configure_hook`,
the machinery behind `get_usage_metadata_callback`) whose handlers are added to
every callback manager built inside that context — so any chat model invoked
anywhere under the node reports its usage to the run's meter, including calls
buried in library code the node merely calls.

The callback is also where `max_tokens` is *enforced*. `on_llm_end` runs while
the node is still executing, which is strictly earlier than the node boundary:
the run stops at the call that crosses the line instead of after every remaining
call has been paid for. This is the same shape as `max_seconds`, which is
enforced by an interrupt delivered into the running node rather than by a
counter read between nodes.

Scope, precisely:

- **Covered:** any `BaseChatModel` invocation on the node's own thread, however
  deeply nested in the call stack.
- **Not covered:** invocations on a thread the node starts itself.
  `threading.Thread` does not inherit context variables, so the handler is
  invisible there; a helper that spawns threads must run its target through
  `contextvars.copy_context().run(...)` for usage to be seen.
- **Not covered:** spend a provider never reports. No `usage_metadata` on the
  returned `AIMessage` means there is nothing to charge, and nothing to enforce
  against until the node charges it by hand.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook

from grapharc.runtime.budget import BudgetMeter


class MeterCallbackHandler(BaseCallbackHandler):
    """Charges every finished model call's `usage_metadata` to a `BudgetMeter`.

    Deliberately not langchain-core's `UsageMetadataCallbackHandler`: that one
    keys usage by `response_metadata["model_name"]` and discards the record
    entirely when a backend reports no model name, which would silently
    under-charge exactly the backends whose spend is hardest to see.

    Each generated message is charged with itself as `source`, which is what
    lets a node hand-report the same call (`charge_usage(ctx, message)`) without
    paying for it twice: the meter recognises the *call*, not the number.
    """

    # langchain's `handle_event` logs and swallows callback exceptions unless
    # the handler asks for them to propagate. A budget ceiling that a callback
    # swallowed would not be a ceiling. It still logs the exception on the way
    # out, so a stopped run leaves one "Error in ... callback" warning behind.
    raise_error = True

    def __init__(self, meter: BudgetMeter) -> None:
        super().__init__()
        self.meter = meter
        self.calls = 0
        # The provider's own price for the calls seen in this scope, and the
        # models that charged it. `None` until some call reports one, because a
        # backend that reports no price must not be recorded as having cost
        # zero — that is the difference between a measurement and a guess, and
        # `observe.cost` keeps recorded and estimated figures apart.
        self.cost_usd: float | None = None
        self.models: list[str] = []
        # This node execution's own token spend, stamped by `charging` when the
        # scope closes. Read from the meter's node scope rather than differenced
        # off the run total, which credited a fan-out worker with whatever its
        # siblings spent while it was running.
        self.tokens = 0

    def _record_price(self, response: LLMResult) -> None:
        """Accumulate the price a backend reported through `llm_output`.

        Both shipped gateways publish the same envelope: `token_usage.cost` and
        `model_name`. Reading the envelope rather than the model object keeps
        this working for a call made deep inside library code the node merely
        called, which is the whole reason this handler exists.
        """
        info = response.llm_output or {}
        cost = (info.get("token_usage") or {}).get("cost")
        if cost is None:
            cost = info.get("cost_usd")
        if cost is not None:
            self.cost_usd = (self.cost_usd or 0.0) + float(cost)
        name = info.get("model_name")
        if name and str(name) not in self.models:
            self.models.append(str(name))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        charged = 0
        for generations in response.generations:
            for generation in generations:
                if not isinstance(generation, ChatGeneration):
                    continue
                usage = getattr(generation.message, "usage_metadata", None) or {}
                total = usage.get("total_tokens") or 0
                if total:
                    self.meter.charge_tokens(total, automatic=True, source=generation.message)
                    charged += total
        if charged:
            self.calls += 1
        self._record_price(response)
        # Enforced here, not at the node boundary: this is the last moment
        # before the node is free to make another call. Overspend is bounded by
        # the call that crossed the line.
        self.meter.check_tokens()


_ACTIVE_HANDLER: ContextVar[MeterCallbackHandler | None] = ContextVar(
    "grapharc_usage_meter", default=None
)
# Registered once at import. Registering per call — as get_usage_metadata_callback
# does — would grow langchain's global hook list on every node execution.
register_configure_hook(_ACTIVE_HANDLER, True)


@contextmanager
def charging(meter: BudgetMeter) -> Iterator[MeterCallbackHandler]:
    """Charge every model call made inside this block to `meter`.

    Also opens the meter's automatic scope, so a call metered here can be
    re-reported by hand inside this block — and only inside it.

    On the way out it stamps the scope's token tally onto the handler. Callers
    read `handler.tokens` after the block has closed — the node wrapper reports
    it on the node's `end` event — and by then the scope itself is gone.
    """
    handler = MeterCallbackHandler(meter)
    token = _ACTIVE_HANDLER.set(handler)
    try:
        with meter.automatic_scope():
            try:
                yield handler
            finally:
                handler.tokens = meter.scope_tokens() or 0
    finally:
        _ACTIVE_HANDLER.reset(token)


__all__ = ["MeterCallbackHandler", "charging"]
