"""OpenRouter backend and the model registry.

Everything here is offline except the tests marked `live`, which cost money and
are deselected by default. Run them with `pytest -m live`.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from grapharc.gateway import config, describe, different_providers, get_model, split_spec
from grapharc.gateway.registry import UnknownBackendError

pytest.importorskip("langchain_openai", reason="OpenRouter needs the openrouter extra")

from grapharc.gateway.openrouter import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    OpenRouterChatModel,
    OpenRouterError,
    _strip_prefix,
)

LIVE_MODEL = "openrouter/anthropic/claude-haiku-4.5"


# ---------------------------------------------------------------- credentials


def test_env_file_is_parsed_with_either_key_spelling(tmp_path, monkeypatch):
    """A key named `open-router-api-key` cannot be a shell variable, so the
    file has to be parsed rather than sourced."""
    for name in config.OPENROUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        'open-router-api-key="sk-or-v1-fromfile"\n'
        "UNRELATED=x\n",
        encoding="utf-8",
    )
    assert config.openrouter_api_key(env_file=env) == "sk-or-v1-fromfile"


def test_process_env_beats_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("open-router-api-key=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert config.openrouter_api_key(env_file=env) == "from-env"


def test_a_parent_directory_dotenv_is_never_read(tmp_path, monkeypatch):
    """The rule `grapharc.toml` follows, on the file that spends money.

    The loader used to walk to `/`, so a run started three directories below a
    `.env` — a scratch subdirectory under a client checkout, a `$HOME` one on a
    shared box — silently billed against a key the operator never put in scope.
    Nothing prints which file paid, so there was no way to notice. The start
    directory is now the only directory consulted.
    """
    for name in config.OPENROUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-parent\n", encoding="utf-8")
    deep = tmp_path / "deeply" / "nested" / "project"
    deep.mkdir(parents=True)

    assert config.find_env_file(deep) is None
    monkeypatch.chdir(deep)
    assert config.find_env_file() is None
    assert config.openrouter_api_key() is None

    # ... and the one directory that *is* consulted still is.
    (deep / ".env").write_text("OPENROUTER_API_KEY=sk-or-here\n", encoding="utf-8")
    assert config.find_env_file() == (deep / ".env").resolve()
    assert config.openrouter_api_key() == "sk-or-here"


def test_an_explicit_env_file_is_read_wherever_it_lives(tmp_path, monkeypatch):
    """The escape hatch for a file outside the working directory: name it."""
    for name in config.OPENROUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    elsewhere = tmp_path / "secrets"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("OPENROUTER_API_KEY=sk-or-named\n", encoding="utf-8")
    run_from = tmp_path / "project"
    run_from.mkdir()
    monkeypatch.chdir(run_from)

    assert config.openrouter_api_key(env_file=elsewhere / ".env") == "sk-or-named"


def test_a_process_variable_beats_a_dotenv_in_the_working_directory(tmp_path, monkeypatch):
    """Narrowing discovery did not reorder precedence: the environment wins."""
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")

    assert config.find_env_file() == (tmp_path / ".env").resolve()
    assert config.openrouter_api_key() == "sk-or-from-env"


def test_missing_key_returns_none_not_a_crash(tmp_path, monkeypatch):
    for name in config.OPENROUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    assert config.openrouter_api_key(env_file=tmp_path / "nope.env") is None


@pytest.mark.parametrize("secret", [None, "", "short", "sk-or-v1-abcdefghijklmnop"])
def test_redact_never_leaks_a_usable_key(secret):
    """The property that matters: the output must not be usable as a key."""
    out = config.redact(secret)
    assert secret not in ("", None) or out == "<unset>"
    if secret:
        assert secret not in out
        assert len(out) < len(secret) + 8


def test_constructing_without_a_key_explains_how_to_fix_it(tmp_path, monkeypatch):
    for name in config.OPENROUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # an empty directory, so no .env to find
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterChatModel("openai/gpt-4o-mini")


# ------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("spec", "backend", "model"),
    [
        ("openrouter/anthropic/claude-haiku-4.5", "openrouter", "anthropic/claude-haiku-4.5"),
        ("openrouter/openai/gpt-4o-mini:floor", "openrouter", "openai/gpt-4o-mini:floor"),
        ("claude-cli/claude-sonnet-5", "claude-cli", "claude-sonnet-5"),
        ("mock/whatever", "mock", "whatever"),
        ("claude-sonnet-5", "claude-cli", "claude-sonnet-5"),  # bare -> default
    ],
)
def test_spec_splitting_keeps_openrouter_author_slugs_intact(spec, backend, model):
    assert split_spec(spec) == (backend, model)
    assert describe(spec) == {"spec": spec, "backend": backend, "model": model}


def test_a_typoed_backend_is_caught_early_not_folded_into_a_model_name():
    """`opnerouter/...` must fail here, not become a Claude-CLI call with a
    nonsense model that errors confusingly much later."""
    with pytest.raises(UnknownBackendError, match="claude-cli, openrouter, openai, ollama, mock"):
        split_spec("opnerouter/anthropic/claude-haiku-4.5")
    with pytest.raises(UnknownBackendError):
        get_model("nope/some-model")


def test_a_bare_author_slug_still_resolves_to_the_default_backend():
    assert split_spec("anthropic/claude-haiku-4.5") == (
        "claude-cli",
        "anthropic/claude-haiku-4.5",
    )


def test_registry_builds_a_mock_without_credentials():
    model = get_model("mock/x", responses=["hi"])
    assert model.invoke("q").content == "hi"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("openrouter/anthropic/claude-haiku-4.5", "openrouter/openai/gpt-4o-mini", True),
        ("openrouter/anthropic/claude-opus-4.5", "openrouter/anthropic/claude-haiku-4.5", False),
        ("claude-cli/claude-sonnet-5", "openrouter/openai/gpt-4o-mini", True),
        # The same vendor reached two ways is not two vendors. Both of these
        # read as independent while the check compared backends.
        ("claude-cli/claude-sonnet-5", "openrouter/anthropic/claude-haiku-4.5", False),
        ("openai/gpt-4o-mini", "openrouter/openai/gpt-4o-mini", False),
        ("openai/gpt-4o-mini", "ollama/llama3.1", True),
    ],
)
def test_different_providers_distinguishes_vendor_from_model(a, b, expected):
    """Two models from the same author share a family — weaker independence
    evidence than a genuine cross-vendor pair."""
    assert different_providers(a, b) is expected


# ------------------------------------------------------- routing request body


def _model(**kwargs) -> OpenRouterChatModel:
    return OpenRouterChatModel("openai/gpt-4o-mini", api_key="sk-or-test", **kwargs)


def test_routing_fields_ride_in_extra_body_not_the_top_level():
    """The OpenAI SDK rejects unknown top-level kwargs, so OpenRouter's routing
    fields must be merged into extra_body."""
    payload = _model()._get_request_payload([HumanMessage(content="hi")])
    assert "provider" not in payload and "usage" not in payload
    assert payload["extra_body"]["usage"] == {"include": True}


def test_fallback_chain_puts_the_primary_first():
    model = _model(
        fallback_models=["openrouter/anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"]
    )
    body = model._get_request_payload([HumanMessage(content="hi")])["extra_body"]
    assert body["models"] == [
        "openai/gpt-4o-mini",
        "anthropic/claude-haiku-4.5",   # the openrouter/ prefix is stripped
        "google/gemini-2.5-flash",
    ]


def test_provider_preferences_are_omitted_when_unset():
    body = _model()._get_request_payload([HumanMessage(content="hi")])["extra_body"]
    assert "provider" not in body


def test_provider_preferences_are_forwarded_when_set():
    model = _model(
        provider_order=["anthropic", "google-vertex"],
        allow_provider_fallbacks=False,
        sort="price",
        max_price_per_million=2.5,
        require_parameters=True,
    )
    provider = model._get_request_payload([HumanMessage(content="hi")])["extra_body"]["provider"]
    assert provider["order"] == ["anthropic", "google-vertex"]
    assert provider["allow_fallbacks"] is False
    assert provider["sort"] == "price"
    assert provider["max_price"] == {"prompt": 2.5}
    assert provider["require_parameters"] is True


def test_default_max_tokens_avoids_the_credit_reservation_402():
    """OpenRouter reserves credit against max_tokens, so a model's full ceiling
    is refused outright on a small balance — even for a ten-token reply."""
    assert _model().max_tokens == DEFAULT_MAX_TOKENS
    assert _model(max_tokens=99).max_tokens == 99


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("openrouter/anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
    ],
)
def test_prefix_stripping(given, expected):
    assert _strip_prefix(given) == expected


# ------------------------------------------------------------ usage envelope


def test_usage_envelope_matches_the_cli_backend_shape():
    """A budget meter must read the same fields whichever backend ran the turn."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    model = _model()
    model._record_usage(
        ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="x"))],
            llm_output={
                "model_name": "openai/gpt-4o-mini",
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "cost": 0.00042,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        )
    )
    usage = model.last_usage
    assert usage["input_tokens"] == 1000            # cached input still counts
    assert usage["total_tokens"] == 1050
    assert usage["input_token_details"]["cache_read"] == 800
    assert usage["uncached_input_tokens"] == 200
    assert usage["cost_usd"] == 0.00042
    assert set(usage) >= {
        "input_tokens", "output_tokens", "total_tokens",
        "input_token_details", "cost_usd", "model",
    }


def test_missing_usage_block_does_not_crash():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    model = _model()
    model._record_usage(
        ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))], llm_output=None)
    )
    assert model.last_usage["total_tokens"] == 0


# ------------------------------------------------------------------ live ----

live = pytest.mark.skipif(
    not config.openrouter_api_key(), reason="no OpenRouter API key configured"
)


@pytest.mark.live
@live
def test_live_completion_and_cost_accounting():
    model = get_model(LIVE_MODEL, temperature=0, max_tokens=32)
    reply = model.invoke([HumanMessage(content="Reply with exactly one word: pong")])
    assert "pong" in reply.content.lower()
    usage = model.last_usage
    assert usage["total_tokens"] > 0
    assert usage["cost_usd"] is not None


@pytest.mark.live
@live
def test_live_tool_calling():
    """The capability that unblocks agent nodes — the CLI backend cannot do this."""
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    model = get_model(LIVE_MODEL, temperature=0, max_tokens=512)
    reply = model.bind_tools([get_weather]).invoke(
        [HumanMessage(content="What is the weather in Paris? Use the tool.")]
    )
    assert reply.tool_calls
    assert reply.tool_calls[0]["name"] == "get_weather"
    assert reply.tool_calls[0]["args"]["city"].lower() == "paris"


@pytest.mark.live
@live
def test_live_structured_output():
    class Verdict(BaseModel):
        supported: bool = Field(description="does the evidence support the claim")
        reason: str

    model = get_model(LIVE_MODEL, temperature=0, max_tokens=512)
    verdict = model.with_structured_output(Verdict).invoke(
        [HumanMessage(content="Claim: the sky is green. Evidence: the sky is blue. Supported?")]
    )
    assert isinstance(verdict, Verdict)
    assert verdict.supported is False


@pytest.mark.live
@live
def test_live_cross_provider_pair_is_genuinely_independent():
    """Two different vendors, so a reviewer is not grading its own family."""
    author_spec = "openrouter/anthropic/claude-haiku-4.5"
    reviewer_spec = "openrouter/openai/gpt-4o-mini"
    assert different_providers(author_spec, reviewer_spec)
    for spec in (author_spec, reviewer_spec):
        reply = get_model(spec, temperature=0, max_tokens=32).invoke(
            [HumanMessage(content="Reply with exactly one word: ok")]
        )
        assert reply.content.strip()


@pytest.mark.live
@live
def test_live_async():
    import asyncio

    model = get_model(LIVE_MODEL, temperature=0, max_tokens=32)
    reply = asyncio.run(model.ainvoke([HumanMessage(content="say: async-ok")]))
    assert "async" in reply.content.lower()


@pytest.mark.live
@live
def test_live_streaming_yields_multiple_chunks():
    model = get_model(LIVE_MODEL, temperature=0, max_tokens=64)
    prompt = [HumanMessage(content="Count 1 to 5, one per line.")]
    chunks = [c.content for c in model.stream(prompt)]
    assert len(chunks) > 1
    assert "5" in "".join(chunks)


@pytest.mark.live
@live
def test_live_key_is_never_echoed_in_a_description():
    assert config.openrouter_api_key() not in str(describe(LIVE_MODEL))
    assert os.environ.get("OPENROUTER_API_KEY", "@@") not in config.redact(
        config.openrouter_api_key()
    )
