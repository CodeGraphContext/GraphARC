"""The Claude Code CLI as a chat model — text completion only, nothing else.

This is GraphARC's primary model backend: it drives `claude -p`, which runs on
a Claude subscription with no API key. That CLI is a *full agent* with its own
tools, settings, and CLAUDE.md pickup — none of which GraphARC's permission
engine could see or veto. So the adapter's job is to invoke it as a pure
inference endpoint:

- every built-in tool is explicitly disallowed (`--disallowedTools`),
- no settings sources are loaded (`--setting-sources ""` — the user's
  allowlists, hooks, and MCP servers cannot leak into graph model calls;
  auth credentials are stored separately and still work),
- the working directory is an empty scratch dir (no CLAUDE.md pickup),
- no session persistence,
- the prompt travels via stdin and flags via an argv array — never through a
  shell, never interpolated.

Operational caveats (by design, documented in the plan): no prompt caching on
this path — keep node contexts lean; subscription quota burn — budget caps are
load-bearing; the backend stays swappable (LangChain `init_chat_model` /
OpenRouter can replace it via config when a key exists).

Failures are classified before they are retried (see `resilience`). The CLI
reports everything through an exit code and free text, so the classifier reads
that text for evidence of transience and defaults to *not* retrying — a
mis-classified 401 would burn three logins' worth of nothing. Note that a
timeout counts as transient, so worst-case latency is `max_attempts *
timeout_seconds`; lower one of the two for a latency-sensitive node.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import ExitStack
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from grapharc.gateway.errors import GatewayError, TransientGatewayError
from grapharc.gateway.resilience import DEFAULT_RETRY_POLICY, RetryPolicy, call_with_retry
from grapharc.gateway.spend import SpendMeter

__all__ = [
    "ClaudeCodeCLIChatModel",
    "GatewayError",
    "TransientGatewayError",
]

# Known built-in tools, denied by name, plus a wildcard for good measure.
# The CLI filters denied tools before the model ever sees their schemas.
_ALL_TOOLS = [
    "*",
    "Task",
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "SlashCommand",
    "Skill",
    "ExitPlanMode",
]


# Evidence that the CLI hit something temporary. Matched case-insensitively
# against stderr / the error result.
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate-limit",
    "429",
    "overloaded",
    "500 internal",
    "502",
    "503",
    "504",
    "529",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "connection error",
    "econnreset",
    "etimedout",
    "try again",
)

# Checked first, and they win: an auth failure whose text ends "please try
# again" is still an auth failure, and retrying it three times fixes nothing.
_DETERMINISTIC_MARKERS = (
    "not logged in",
    "invalid api key",
    "authentication",
    "unauthorized",
    "forbidden",
    "permission denied",
    "invalid model",
    "unknown model",
    "credit balance",
    "usage limit reached",
)


def _canonical_model(model: str) -> str:
    """Accept OpenClaw-style refs (`anthropic/claude-sonnet-5`) and bare IDs."""
    return model.split("/", 1)[1] if model.startswith("anthropic/") else model


def _payload_error(stdout: str) -> str:
    """The CLI's own explanation of a failure, which it writes to *stdout*.

    `claude -p` reports a refused model, an expired login and their siblings
    with a non-zero exit code, an **empty stderr**, and the whole reason in the
    JSON envelope on stdout::

        {"is_error": true, "result": "There's an issue with the selected model
         (claude-cli). It may not exist or you may not have access to it."}

    Reading only stderr therefore produced ``claude -p exited 1: `` — a
    sentence that stops at the colon — and discarded the one string that said
    what was wrong. It matters for classification too: `_cli_failure` decides
    transient-vs-deterministic from this text, and it was deciding from "".

    Returns "" when stdout is not the documented envelope, so the caller falls
    back to stderr rather than inventing a reason.
    """
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError):  # JSONDecodeError is a ValueError
        return ""
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result")
    return str(result).strip() if result else ""


def _cli_failure(message: str, evidence: str) -> GatewayError:
    """Build the right error class for a CLI failure from its own text.

    Closed by default: without positive evidence of transience the failure is
    deterministic and is raised on the first attempt.
    """
    text = evidence.lower()
    if any(marker in text for marker in _DETERMINISTIC_MARKERS):
        return GatewayError(message)
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return TransientGatewayError(message)
    return GatewayError(message)


class ClaudeCodeCLIChatModel(BaseChatModel):
    """LangChain chat model over `claude -p`, locked to text completion."""

    model: str = "claude-sonnet-5"
    claude_path: str = "claude"
    timeout_seconds: float = 600.0
    workdir: str | None = None  # empty scratch dir by default

    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY
    # Seeds `spend.ceiling_usd` at construction. Pass `spend=` instead to share
    # one ceiling across several model objects.
    cost_ceiling_usd: float | None = None
    spend: SpendMeter = Field(default_factory=SpendMeter, exclude=True)

    last_usage: dict[str, Any] | None = Field(default=None, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.cost_ceiling_usd is not None and self.spend.ceiling_usd is None:
            self.spend.ceiling_usd = self.cost_ceiling_usd

    @property
    def _llm_type(self) -> str:
        return "grapharc-claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def _build_argv(self, system: str | None) -> list[str]:
        argv = [
            self.claude_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            _canonical_model(self.model),
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--disallowedTools",
            *_ALL_TOOLS,
        ]
        if system:
            argv += ["--system-prompt", system]
        return argv

    @staticmethod
    def _render_prompt(messages: list[BaseMessage]) -> tuple[str | None, str]:
        """Flatten a message list into (system, prompt-text) for print mode."""
        system_parts, prompt_parts = [], []
        for m in messages:
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            if isinstance(m, SystemMessage):
                system_parts.append(content)
            else:
                role = getattr(m, "type", "user")
                prefix = "" if role == "human" else f"[{role}] "
                prompt_parts.append(f"{prefix}{content}")
        system = "\n\n".join(system_parts) or None
        return system, "\n\n".join(prompt_parts)

    def _invoke_cli(self, argv: list[str], prompt: str, cwd: str) -> dict[str, Any]:
        """One attempt: run the CLI and parse its JSON, or raise a classified error."""
        try:
            proc = subprocess.run(  # noqa: S603 — argv array, no shell
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise GatewayError(
                f"claude CLI not found at {self.claude_path!r} — install Claude Code "
                "and log in, or configure a different backend"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TransientGatewayError(
                f"claude -p timed out after {self.timeout_seconds}s"
            ) from exc

        if proc.returncode != 0:
            # stdout first: the CLI puts its reason in the JSON envelope there
            # and leaves stderr empty. See `_payload_error`.
            detail = _payload_error(proc.stdout) or proc.stderr.strip()
            raise _cli_failure(f"claude -p exited {proc.returncode}: {detail[:500]}", detail)

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            # Deterministic on purpose: truncated or non-JSON output is far more
            # often a wrong flag or a wrapper script than a blip, and a retry
            # cannot tell the difference.
            raise GatewayError(
                f"claude -p returned non-JSON output: {proc.stdout[:200]!r}"
            ) from exc

        if payload.get("is_error"):
            result = str(payload.get("result"))
            raise _cli_failure(f"claude -p reported an error: {result!r}", result)
        return payload

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, prompt = self._render_prompt(messages)
        argv = self._build_argv(system)
        self.spend.ensure_headroom(model=self.model)

        retries = 0

        def _count_retry(attempt: int, delay: float, exc: BaseException) -> None:
            nonlocal retries
            retries = attempt

        with ExitStack() as stack:
            # The scratch dir must outlive every attempt and die with the call:
            # `mkdtemp` here leaked one directory per invocation. Cleanup errors
            # are swallowed rather than raised, so an undeletable directory
            # cannot turn a completed (and paid-for) call into a failure.
            cwd = self.workdir or stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="grapharc-gateway-", ignore_cleanup_errors=True
                )
            )
            payload = call_with_retry(
                lambda: self._invoke_cli(argv, prompt, cwd),
                policy=self.retry_policy,
                on_retry=_count_retry,
            )

        text = str(payload.get("result", ""))
        usage = payload.get("usage") or {}
        # Cached input is still input. Counting only `input_tokens` would let a
        # budget under-count by an order of magnitude on real calls, since most
        # of a turn's prompt arrives as cache creation or cache reads.
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        uncached = int(usage.get("input_tokens", 0))
        input_tokens = uncached + cache_creation + cache_read
        output_tokens = int(usage.get("output_tokens", 0))
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "cache_creation": cache_creation,
                "cache_read": cache_read,
            },
        }
        cost_usd = payload.get("total_cost_usd")
        # Uniform usage envelope (OpenRouter discipline): native counts + cost.
        # Recorded before the ceiling is enforced, so a caller that catches
        # `CostCeilingExceeded` can still read what the last call cost.
        self.last_usage = {
            **usage_metadata,
            "uncached_input_tokens": uncached,
            "cost_usd": cost_usd,
            "model": self.model,
            "retries": retries,
            "cumulative_cost_usd": self.spend.spent_usd + float(cost_usd or 0.0),
        }
        self.spend.charge(cost_usd, model=self.model)
        message = AIMessage(content=text, usage_metadata=usage_metadata)
        # `llm_output` carries the price in the same shape OpenRouter uses, so
        # the runtime's usage callback can record what a node's calls cost
        # without knowing which backend served them. Without it, a node calling
        # this backend would trace `cost_usd: None` while the CLI reported a
        # figure it had already charged to the spend meter.
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"token_usage": {"cost": cost_usd}, "model_name": self.model},
        )
