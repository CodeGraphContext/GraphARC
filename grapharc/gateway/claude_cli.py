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
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

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


class GatewayError(Exception):
    """The CLI backend failed or returned something unusable."""


def _canonical_model(model: str) -> str:
    """Accept OpenClaw-style refs (`anthropic/claude-sonnet-5`) and bare IDs."""
    return model.split("/", 1)[1] if model.startswith("anthropic/") else model


class ClaudeCodeCLIChatModel(BaseChatModel):
    """LangChain chat model over `claude -p`, locked to text completion."""

    model: str = "claude-sonnet-5"
    claude_path: str = "claude"
    timeout_seconds: float = 600.0
    workdir: str | None = None  # empty scratch dir by default

    last_usage: dict[str, Any] | None = Field(default=None, exclude=True)

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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, prompt = self._render_prompt(messages)
        argv = self._build_argv(system)
        cwd = self.workdir or tempfile.mkdtemp(prefix="grapharc-gateway-")
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
            raise GatewayError(f"claude -p timed out after {self.timeout_seconds}s") from exc

        if proc.returncode != 0:
            raise GatewayError(
                f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise GatewayError(
                f"claude -p returned non-JSON output: {proc.stdout[:200]!r}"
            ) from exc

        if payload.get("is_error"):
            raise GatewayError(f"claude -p reported an error: {payload.get('result')!r}")

        text = str(payload.get("result", ""))
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        # Uniform usage envelope (OpenRouter discipline): native counts + cost.
        self.last_usage = {
            **usage_metadata,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cost_usd": payload.get("total_cost_usd"),
            "model": self.model,
        }
        message = AIMessage(content=text, usage_metadata=usage_metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])
