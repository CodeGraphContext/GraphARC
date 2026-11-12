"""The UberLLM backend.

Everything here is offline except the test marked `live`: constructing a
`ChatOpenAI` opens no socket, so the wiring — credentials, endpoint, cost
accounting — is checkable without spending anything.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from grapharc.gateway import config, describe, get_model, split_spec, vendor
from grapharc.gateway.registry import UnknownBackendError

pytest.importorskip("langchain_openai", reason="UberLLM needs the uberllm extra")

from grapharc.gateway.uberllm import (  # noqa: E402
    UBERLLM_BASE_URL,
    UberLLMChatModel,
    UberLLMError,
)


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No key in the environment, and a working directory holding no .env."""
    for name in config.UBERLLM_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _result(**token_usage) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="x"))],
        llm_output={"model_name": "test-model", "token_usage": token_usage},
    )


# ---------------------------------------------------------------- credentials


def test_uberllm_key_is_read_from_a_dotenv_file(tmp_path, monkeypatch):
    """`uberllm-api-key` cannot be a shell variable, so the file is parsed."""
    for name in config.UBERLLM_KEYS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text('uberllm-api-key="ull-fromfile"\n', encoding="utf-8")
    assert config.uberllm_api_key(env_file=env) == "ull-fromfile"


def test_uberllm_process_env_beats_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("uberllm-api-key=from-file\n", encoding="utf-8")
    monkeypatch.setenv("UBERLLM_API_KEY", "from-env")
    assert config.uberllm_api_key(env_file=env) == "from-env"


def test_constructing_uberllm_without_a_key_explains_how_to_fix_it(no_credentials):
    with pytest.raises(UberLLMError, match="UBERLLM_API_KEY"):
        UberLLMChatModel("deepseek/deepseek-v4.1-flash")


def test_uberllm_key_never_appears_in_a_description_or_a_redaction(monkeypatch):
    secret = "ull_0123456789abcdef0123456789abcdef0123456"
    monkeypatch.setenv("UBERLLM_API_KEY", secret)
    assert secret not in str(describe("uberllm/deepseek/deepseek-v4.1-flash"))
    assert secret not in config.redact(secret)


def test_uberllm_always_points_at_its_own_endpoint(monkeypatch):
    """Unlike OpenAI, there is no override — the endpoint is fixed."""
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel("deepseek/deepseek-v4.1-flash")
    assert str(model.openai_api_base) == UBERLLM_BASE_URL
    assert UBERLLM_BASE_URL == "https://api.uberllm.dev/v1"


# ------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("spec", "backend", "model"),
    [
        ("uberllm/deepseek/deepseek-v4.1-flash", "uberllm", "deepseek/deepseek-v4.1-flash"),
        ("uberllm/anthropic/claude-sonnet-4.6", "uberllm", "anthropic/claude-sonnet-4.6"),
    ],
)
def test_uberllm_specs_split_the_way_the_docs_say(spec, backend, model):
    assert split_spec(spec) == (backend, model)
    assert describe(spec) == {"spec": spec, "backend": backend, "model": model}


def test_the_registry_builds_the_uberllm_backend(monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    assert get_model("uberllm/deepseek/deepseek-v4.1-flash")._llm_type == "grapharc-uberllm"


def test_a_missing_uberllm_key_surfaces_through_the_registry(no_credentials):
    with pytest.raises(UberLLMError):
        get_model("uberllm/deepseek/deepseek-v4.1-flash")


def test_an_unknown_backend_still_names_uberllm():
    with pytest.raises(UnknownBackendError, match="uberllm"):
        get_model("uberlm/deepseek/deepseek-v4.1-flash")


def test_vendor_reads_the_model_authors_uberllm_ids_carry():
    """UberLLM ids are themselves `vendor/model`, the same shape OpenRouter uses,
    so `vendor()` needs no UberLLM-specific entry in `BACKEND_VENDOR`."""
    assert vendor("uberllm/deepseek/deepseek-v4.1-flash") == "deepseek"
    assert vendor("uberllm/anthropic/claude-sonnet-4.6") == "anthropic"


# ------------------------------------------------- capabilities and accounting


def test_uberllm_can_bind_tools_and_structure_output(monkeypatch):
    """The capability the Claude-CLI backend cannot offer."""
    from langchain_core.tools import tool
    from pydantic import BaseModel

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    class Verdict(BaseModel):
        supported: bool

    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel("deepseek/deepseek-v4.1-flash")
    model.bind_tools([get_weather])
    model.with_structured_output(Verdict)


def test_the_usage_envelope_matches_every_other_backend(monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel("deepseek/deepseek-v4.1-flash")
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


def test_uberllm_charges_the_cost_it_reports(monkeypatch):
    """UberLLM puts what the call was billed in `usage.cost_usd`, so the meter
    is charged that figure rather than counting the call as unpriced."""
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel("deepseek/deepseek-v4.1-flash")
    model._settle(_result(prompt_tokens=35, completion_tokens=10, cost_usd=0.00042))
    assert model.last_usage["cost_usd"] == 0.00042
    assert model.spend.spent_usd == pytest.approx(0.00042)
    assert model.spend.unpriced_calls == 0


def test_the_reported_cost_wins_over_a_rate_card(monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel(
        "deepseek/deepseek-v4.1-flash",
        price_per_million={"input": 100.0, "output": 100.0},
    )
    model._settle(_result(prompt_tokens=1000, completion_tokens=50, cost_usd=0.00042))
    assert model.last_usage["cost_usd"] == 0.00042


def test_without_a_reported_cost_the_call_is_counted_as_unpriced(monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    model = UberLLMChatModel("deepseek/deepseek-v4.1-flash")
    model._settle(_result(prompt_tokens=1000, completion_tokens=50))
    assert model.last_usage["cost_usd"] is None
    assert model.spend.unpriced_calls == 1


def test_uberllm_sets_no_max_tokens_ceiling_of_its_own(monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ull-test")
    assert UberLLMChatModel("deepseek/deepseek-v4.1-flash").max_tokens is None


# ------------------------------------------------------------------ live ----


@pytest.mark.live
@pytest.mark.skipif(not config.uberllm_api_key(), reason="no UberLLM API key configured")
def test_live_uberllm_completion_and_cost_accounting():
    model = get_model("uberllm/deepseek/deepseek-v4.1-flash", temperature=0, max_tokens=512)
    reply = model.invoke("Reply with exactly: ok")
    assert reply.content
    assert model.last_usage["cost_usd"] is not None
    assert model.spend.unpriced_calls == 0
