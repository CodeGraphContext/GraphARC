"""The Novita backend.

Everything here is offline: constructing a `ChatOpenAI` opens no socket, so the
wiring — credentials, endpoint, cost accounting — is checkable without
spending anything.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from grapharc.gateway import config, describe, get_model, split_spec, vendor
from grapharc.gateway.registry import UnknownBackendError

pytest.importorskip("langchain_openai", reason="Novita needs the novita extra")

from grapharc.gateway.novita import NOVITA_BASE_URL, NovitaChatModel, NovitaError  # noqa: E402


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No key in the environment, and a working directory holding no .env."""
    for name in config.NOVITA_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _result(**token_usage) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="x"))],
        llm_output={"model_name": "test-model", "token_usage": token_usage},
    )


# ---------------------------------------------------------------- credentials


def test_novita_key_is_read_from_a_dotenv_file(tmp_path, monkeypatch):
    """`novita-api-key` cannot be a shell variable, so the file is parsed."""
    for name in config.NOVITA_KEYS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text('novita-api-key="sk-fromfile"\n', encoding="utf-8")
    assert config.novita_api_key(env_file=env) == "sk-fromfile"


def test_novita_process_env_beats_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("novita-api-key=from-file\n", encoding="utf-8")
    monkeypatch.setenv("NOVITA_API_KEY", "from-env")
    assert config.novita_api_key(env_file=env) == "from-env"


def test_constructing_novita_without_a_key_explains_how_to_fix_it(no_credentials):
    with pytest.raises(NovitaError, match="NOVITA_API_KEY"):
        NovitaChatModel("moonshotai/kimi-k3")


def test_novita_key_never_appears_in_a_description_or_a_redaction(monkeypatch):
    secret = "sk-novita-0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("NOVITA_API_KEY", secret)
    assert secret not in str(describe("novita/moonshotai/kimi-k3"))
    assert secret not in config.redact(secret)


def test_novita_always_points_at_its_own_endpoint(monkeypatch):
    """Unlike OpenAI, there is no override — the endpoint is fixed."""
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    model = NovitaChatModel("moonshotai/kimi-k3")
    assert str(model.openai_api_base) == NOVITA_BASE_URL
    assert NOVITA_BASE_URL == "https://api.novita.ai/openai"


# ------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("spec", "backend", "model"),
    [
        ("novita/moonshotai/kimi-k3", "novita", "moonshotai/kimi-k3"),
        ("novita/zai-org/glm-5.2", "novita", "zai-org/glm-5.2"),
    ],
)
def test_novita_specs_split_the_way_the_docs_say(spec, backend, model):
    assert split_spec(spec) == (backend, model)
    assert describe(spec) == {"spec": spec, "backend": backend, "model": model}


def test_the_registry_builds_the_novita_backend(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    assert get_model("novita/moonshotai/kimi-k3")._llm_type == "grapharc-novita"


def test_a_missing_novita_key_surfaces_through_the_registry(no_credentials):
    with pytest.raises(NovitaError):
        get_model("novita/moonshotai/kimi-k3")


def test_an_unknown_backend_still_names_novita():
    with pytest.raises(UnknownBackendError, match="novita"):
        get_model("nvita/moonshotai/kimi-k3")


def test_vendor_reads_the_model_authors_novita_ids_carry():
    """Novita ids are themselves `author/slug`, the same shape OpenRouter uses,
    so `vendor()` needs no Novita-specific entry in `BACKEND_VENDOR`."""
    assert vendor("novita/moonshotai/kimi-k3") == "moonshotai"
    assert vendor("novita/zai-org/glm-5.2") == "zai-org"


# ------------------------------------------------- capabilities and accounting


def test_novita_can_bind_tools_and_structure_output(monkeypatch):
    """The capability the Claude-CLI backend cannot offer."""
    from langchain_core.tools import tool
    from pydantic import BaseModel

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    class Verdict(BaseModel):
        supported: bool

    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    model = NovitaChatModel("moonshotai/kimi-k3")
    model.bind_tools([get_weather])
    model.with_structured_output(Verdict)


def test_the_usage_envelope_matches_every_other_backend(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    model = NovitaChatModel("moonshotai/kimi-k3")
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


def test_novita_reports_no_cost_and_says_so_rather_than_guessing(monkeypatch):
    """Like the OpenAI API, Novita's response carries tokens and no price. An
    invented number would be worse than an admitted gap, so the call is
    counted as unpriced."""
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    model = NovitaChatModel("moonshotai/kimi-k3")
    model._settle(_result(prompt_tokens=1000, completion_tokens=50))
    assert model.last_usage["cost_usd"] is None
    assert model.spend.unpriced_calls == 1
    assert model.spend.spent_usd == 0.0


def test_a_rate_card_prices_novita_the_way_it_prices_openai(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    model = NovitaChatModel(
        "moonshotai/kimi-k3",
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


def test_novita_sets_no_max_tokens_ceiling_of_its_own(monkeypatch):
    """OpenRouter defaults it to dodge a credit-reservation 402. Novita has no
    such reservation, so a default here would only truncate replies."""
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    assert NovitaChatModel("moonshotai/kimi-k3").max_tokens is None


# ------------------------------------------------------------------ live ----


@pytest.mark.live
@pytest.mark.skipif(not config.novita_api_key(), reason="no Novita API key configured")
def test_live_novita_tool_calling():
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    model = get_model("novita/moonshotai/kimi-k3", temperature=0, max_tokens=512)
    reply = model.bind_tools([get_weather]).invoke(
        [HumanMessage(content="What is the weather in Paris? Use the tool.")]
    )
    assert reply.tool_calls
    assert reply.tool_calls[0]["name"] == "get_weather"
