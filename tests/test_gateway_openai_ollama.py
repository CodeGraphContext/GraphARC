"""The OpenAI and Ollama backends.

Everything here is offline. The Ollama tests never contact a daemon and the
OpenAI tests never contact api.openai.com: constructing a `ChatOpenAI` opens no
socket, so the wiring — credentials, endpoint, cost accounting — is checkable
without spending anything. The `live` tests at the bottom are opt-in and are
the only ones that talk to a server.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from grapharc.gateway import config, describe, get_model, split_spec, vendor
from grapharc.gateway.registry import UnknownBackendError

pytest.importorskip("langchain_openai", reason="both backends need langchain-openai")

from grapharc.gateway.ollama import (  # noqa: E402
    PLACEHOLDER_API_KEY,
    OllamaChatModel,
    OllamaError,
)
from grapharc.gateway.openai import OpenAIChatModel, OpenAIError  # noqa: E402

OPENAI_ENV = (*config.OPENAI_KEYS, *config.OPENAI_BASE_URL_KEYS)
OLLAMA_ENV = (*config.OLLAMA_KEYS, *config.OLLAMA_BASE_URL_KEYS)


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No key in the environment, and a working directory holding no .env."""
    for name in (*OPENAI_ENV, *OLLAMA_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _result(**token_usage) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="x"))],
        llm_output={"model_name": "test-model", "token_usage": token_usage},
    )


# ---------------------------------------------------------------- credentials


def test_openai_key_is_read_from_a_dotenv_file(tmp_path, monkeypatch):
    """`openai-api-key` cannot be a shell variable, so the file is parsed."""
    for name in config.OPENAI_KEYS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text('openai-api-key="sk-fromfile"\n', encoding="utf-8")
    assert config.openai_api_key(env_file=env) == "sk-fromfile"


def test_openai_process_env_beats_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("openai-api-key=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert config.openai_api_key(env_file=env) == "from-env"


def test_constructing_openai_without_a_key_explains_how_to_fix_it(no_credentials):
    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        OpenAIChatModel("gpt-4o-mini")


def test_openai_key_never_appears_in_a_description_or_a_redaction(monkeypatch):
    secret = "sk-proj-0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    assert secret not in str(describe("openai/gpt-4o-mini"))
    assert secret not in config.redact(secret)


def test_ollama_needs_no_credential_at_all(no_credentials):
    """The backend that must construct on a machine with nothing configured."""
    model = OllamaChatModel("llama3.1")
    assert model.openai_api_key.get_secret_value() == PLACEHOLDER_API_KEY
    assert str(model.openai_api_base) == config.DEFAULT_OLLAMA_BASE_URL


def test_ollama_api_key_is_used_when_the_endpoint_is_behind_auth(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "proxy-token")
    assert OllamaChatModel("llama3.1").openai_api_key.get_secret_value() == "proxy-token"


def test_an_empty_ollama_model_is_refused_rather_than_sent(no_credentials):
    with pytest.raises(OllamaError, match="ollama/llama3.1"):
        OllamaChatModel("")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
        ("http://gpu-box:11434/v1", "http://gpu-box:11434/v1"),
        # A scheme means it is a URL: 443, not a guessed 11434 nothing serves.
        ("https://ollama.internal/v1/", "https://ollama.internal/v1"),
        # OLLAMA_HOST is routinely a bare hostname; the port is Ollama's own.
        ("gpu-box", "http://gpu-box:11434/v1"),
        ("gpu-box/proxy", "http://gpu-box:11434/proxy/v1"),
    ],
)
def test_ollama_host_shorthands_become_usable_urls(configured, expected, monkeypatch):
    """OLLAMA_HOST is an ollama-CLI variable, not a URL, and people set it as one."""
    monkeypatch.setenv("OLLAMA_HOST", configured)
    assert config.ollama_base_url() == expected


def test_ollama_base_url_defaults_to_localhost_when_nothing_is_set(no_credentials):
    assert config.ollama_base_url() == "http://localhost:11434/v1"


def test_explicit_ollama_base_url_beats_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere:1234/v1")
    assert config.ollama_base_url() == "http://elsewhere:1234/v1"


def test_openai_base_url_override_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.internal/v1")
    assert str(OpenAIChatModel("gpt-4o-mini").openai_api_base) == "https://gateway.internal/v1"


def test_openai_without_an_override_sends_no_base_url_of_its_own(monkeypatch):
    """The SDK knows where api.openai.com is; a copy here would be one more
    string to get wrong when it moves."""
    for name in config.OPENAI_BASE_URL_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIChatModel("gpt-4o-mini").openai_api_base is None


# ------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("spec", "backend", "model"),
    [
        ("openai/gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("ollama/llama3.1", "ollama", "llama3.1"),
        ("ollama/qwen2.5-coder:7b", "ollama", "qwen2.5-coder:7b"),
        # Only the first segment is a backend, so the broker route is intact.
        ("openrouter/openai/gpt-4o-mini", "openrouter", "openai/gpt-4o-mini"),
    ],
)
def test_the_new_backends_split_the_way_the_docs_say(spec, backend, model):
    assert split_spec(spec) == (backend, model)
    assert describe(spec) == {"spec": spec, "backend": backend, "model": model}


def test_openai_as_a_backend_beats_openai_as_an_author_slug():
    """`openai` is both. Someone typing `openai/gpt-4o` means the OpenAI API,
    not a bare model name handed to the Claude CLI — which is what this
    resolved to before the backend existed."""
    assert split_spec("openai/gpt-4o")[0] == "openai"
    assert split_spec("anthropic/claude-haiku-4.5")[0] == "claude-cli"


def test_the_registry_builds_both_backends(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_model("openai/gpt-4o-mini")._llm_type == "grapharc-openai"
    assert get_model("ollama/llama3.1")._llm_type == "grapharc-ollama"


def test_a_missing_openai_key_surfaces_through_the_registry(no_credentials):
    with pytest.raises(OpenAIError):
        get_model("openai/gpt-4o-mini")


def test_an_unknown_backend_still_names_every_real_one():
    with pytest.raises(UnknownBackendError, match="openai, ollama"):
        get_model("opnai/gpt-4o-mini")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openai/gpt-4o-mini", "openai"),
        ("openrouter/openai/gpt-4o-mini", "openai"),
        ("ollama/llama3.1", "ollama"),
        ("claude-cli/claude-sonnet-5", "anthropic"),
        ("mock/x", "mock"),
    ],
)
def test_vendor_ignores_how_a_model_is_reached(spec, expected):
    assert vendor(spec) == expected


# ------------------------------------------------- capabilities and accounting


def test_both_backends_can_bind_tools_and_structure_output(monkeypatch):
    """The capability the Claude-CLI backend cannot offer, and the reason an
    agent node can run against a local model at all."""
    from langchain_core.tools import tool
    from pydantic import BaseModel

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    class Verdict(BaseModel):
        supported: bool

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for model in (OpenAIChatModel("gpt-4o-mini"), OllamaChatModel("llama3.1")):
        model.bind_tools([get_weather])
        model.with_structured_output(Verdict)


def test_the_usage_envelope_matches_every_other_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = OpenAIChatModel("gpt-4o-mini")
    model._record_usage(
        _result(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details={"cached_tokens": 800},
        )
    )
    usage = model.last_usage
    assert usage["input_tokens"] == 1000  # cached input still counts
    assert usage["total_tokens"] == 1050
    assert usage["input_token_details"]["cache_read"] == 800
    assert usage["uncached_input_tokens"] == 200


def test_openai_reports_no_cost_and_says_so_rather_than_guessing(monkeypatch):
    """The API returns tokens and no price. An invented number would be worse
    than an admitted gap, so the call is counted as unpriced."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = OpenAIChatModel("gpt-4o-mini")
    model._settle(_result(prompt_tokens=1000, completion_tokens=50))
    assert model.last_usage["cost_usd"] is None
    assert model.spend.unpriced_calls == 1
    assert model.spend.spent_usd == 0.0


def test_a_rate_card_prices_a_backend_the_provider_does_not(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = OpenAIChatModel(
        "gpt-4o-mini",
        price_per_million={"input": 0.15, "cached_input": 0.075, "output": 0.60},
    )
    model._settle(
        _result(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            prompt_tokens_details={"cached_tokens": 400_000},
        )
    )
    # 600k uncached @ 0.15 + 400k cached @ 0.075 + 1M output @ 0.60
    assert model.last_usage["cost_usd"] == pytest.approx(0.09 + 0.03 + 0.60)
    assert model.spend.unpriced_calls == 0


def test_a_cached_rate_defaults_to_the_uncached_one(monkeypatch):
    """Over-stating the price of a cache hit beats inventing a discount."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = OpenAIChatModel("gpt-4o-mini", price_per_million={"input": 1.0, "output": 0.0})
    model._settle(
        _result(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            prompt_tokens_details={"cached_tokens": 1_000_000},
        )
    )
    assert model.last_usage["cost_usd"] == pytest.approx(1.0)


def test_local_inference_costs_zero_and_that_is_not_a_missing_number(no_credentials):
    """`unpriced_calls` means the meter missed a bill. Ollama has none to miss."""
    model = OllamaChatModel("llama3.1")
    model._settle(_result(prompt_tokens=1000, completion_tokens=50))
    assert model.last_usage["cost_usd"] == 0.0
    assert model.spend.unpriced_calls == 0
    assert model.spend.spent_usd == 0.0


def test_a_rate_card_can_attribute_local_gpu_time_anyway(no_credentials):
    model = OllamaChatModel("llama3.1", price_per_million={"input": 1.0, "output": 2.0})
    model._settle(_result(prompt_tokens=1_000_000, completion_tokens=1_000_000))
    assert model.last_usage["cost_usd"] == pytest.approx(3.0)


def test_a_ceiling_still_stops_a_priced_local_run(no_credentials):
    from grapharc.gateway import CostCeilingExceeded

    model = OllamaChatModel(
        "llama3.1", price_per_million={"input": 1.0, "output": 0.0}, cost_ceiling_usd=0.5
    )
    with pytest.raises(CostCeilingExceeded):
        model._settle(_result(prompt_tokens=1_000_000, completion_tokens=0))
    # The charge lands before the raise: a resumed run knows what it spent.
    assert model.spend.spent_usd == pytest.approx(1.0)


def test_the_sdk_retry_layer_is_off_on_both_backends(monkeypatch):
    """Two retry layers would compose into max_attempts * sdk_retries requests
    against a provider that just said 429."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIChatModel("gpt-4o-mini").max_retries == 0
    assert OllamaChatModel("llama3.1").max_retries == 0
    assert OpenAIChatModel("gpt-4o-mini", max_retries=3).max_retries == 3


def test_the_retry_policy_covers_the_new_backends_too(monkeypatch):
    """The policy is inherited rather than reimplemented, so this guards against
    a future subclass quietly overriding `_generate` and losing it."""
    from langchain_openai import ChatOpenAI

    from grapharc.gateway.resilience import RetryPolicy

    class FakeHTTPError(Exception):
        status_code = 429

    calls: list[int] = []

    def fake(self, messages, stop=None, run_manager=None, **kwargs):
        calls.append(1)
        if len(calls) < 2:
            raise FakeHTTPError()
        return _result(prompt_tokens=10, completion_tokens=2)

    monkeypatch.setattr(ChatOpenAI, "_generate", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0, jitter=0.0)
    model = OpenAIChatModel("gpt-4o-mini", retry_policy=policy)
    model.invoke("hi")
    assert len(calls) == 2
    assert model.last_usage["retries"] == 1


def test_openai_sets_no_max_tokens_ceiling_of_its_own(monkeypatch):
    """OpenRouter defaults it to dodge a credit-reservation 402. OpenAI reserves
    nothing, so a default here would only truncate replies."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIChatModel("gpt-4o-mini").max_tokens is None
    assert OllamaChatModel("llama3.1").max_tokens is None


# ------------------------------------------------------------------ live ----

live_ollama = pytest.mark.skipif(
    not __import__("shutil").which("ollama"), reason="no ollama on PATH"
)


@pytest.mark.live
@live_ollama
def test_live_local_completion():
    """Needs a pulled model; `OLLAMA_TEST_MODEL` names which one."""
    import os

    from langchain_core.messages import HumanMessage

    model = get_model(f"ollama/{os.environ.get('OLLAMA_TEST_MODEL', 'llama3.1')}")
    reply = model.invoke([HumanMessage(content="Reply with exactly one word: pong")])
    assert reply.content.strip()
    assert model.last_usage["total_tokens"] > 0
    assert model.last_usage["cost_usd"] == 0.0


@pytest.mark.live
@pytest.mark.skipif(not config.openai_api_key(), reason="no OpenAI API key configured")
def test_live_openai_tool_calling():
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    model = get_model("openai/gpt-4o-mini", temperature=0, max_tokens=512)
    reply = model.bind_tools([get_weather]).invoke(
        [HumanMessage(content="What is the weather in Paris? Use the tool.")]
    )
    assert reply.tool_calls
    assert reply.tool_calls[0]["name"] == "get_weather"
