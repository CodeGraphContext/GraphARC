"""Gateway resilience: retries, backoff, failure classification, cost ceilings.

Everything here runs against mocked transports — no process is spawned and no
socket is opened. The properties under test are the ones that cost money when
they are wrong:

- a 429 or a timeout is retried; a 400, a 401 and a refusal are not;
- backoff grows on every draw, not merely on average;
- a cost ceiling raises, and the call that crossed it is still accounted for;
- the CLI backend's scratch directory is removed when the call ends.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile

import pytest

from grapharc.gateway import claude_cli, resilience
from grapharc.gateway.claude_cli import ClaudeCodeCLIChatModel
from grapharc.gateway.errors import (
    CostCeilingExceeded,
    GatewayError,
    TransientGatewayError,
)
from grapharc.gateway.resilience import (
    NO_RETRY,
    RetryPolicy,
    acall_with_retry,
    call_with_retry,
    is_transient,
    retry_after_seconds,
)
from grapharc.gateway.spend import SpendMeter

# --------------------------------------------------------------- fake errors


class FakeHTTPError(Exception):
    """A provider error shaped like httpx/openai: a status code and headers."""

    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = _FakeResponse(status_code, retry_after)


class _FakeResponse:
    def __init__(self, status_code: int, retry_after: str | None) -> None:
        self.status_code = status_code
        self.headers = {} if retry_after is None else {"retry-after": retry_after}


class Recorder:
    """Stands in for `time.sleep`, remembering what it was asked to wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)

    async def acall(self, seconds: float) -> None:
        self.delays.append(seconds)


# ------------------------------------------------------------------- backoff


def test_backoff_grows_exponentially_until_the_cap():
    policy = RetryPolicy(initial_backoff_seconds=1.0, multiplier=2.0, max_backoff_seconds=6.0)
    assert [policy.base_delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 6.0, 6.0]


def test_every_jittered_draw_is_larger_than_the_previous_attempt():
    """Growth must be a property of each draw, not of the average — otherwise a
    burst of unlucky short waits hammers a provider that just said 429."""
    policy = RetryPolicy(initial_backoff_seconds=0.5, multiplier=2.0, jitter=0.25)
    for _ in range(200):
        delays = [policy.delay_for(n) for n in range(1, 5)]
        assert delays == sorted(delays)
        assert delays[0] < delays[1] < delays[2] < delays[3]
        for n, delay in enumerate(delays, start=1):
            base = policy.base_delay(n)
            assert base * 0.75 <= delay <= base


def test_jitter_actually_varies_the_delay():
    """A constant delay would satisfy the growth test above; this rules it out."""
    policy = RetryPolicy(initial_backoff_seconds=1.0, jitter=0.5)
    draws = {policy.delay_for(3) for _ in range(50)}
    assert len(draws) > 1
    assert min(draws) >= policy.base_delay(3) * 0.5


def test_zero_jitter_is_exactly_the_base_delay():
    policy = RetryPolicy(initial_backoff_seconds=0.25, jitter=0.0)
    assert policy.delay_for(2) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"jitter": 1.5},
        {"jitter": -0.1},
        {"multiplier": 0.5},
        {"initial_backoff_seconds": -1},
    ],
)
def test_a_nonsensical_policy_is_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_retry_after_lengthens_the_wait_but_stays_bounded():
    policy = RetryPolicy(initial_backoff_seconds=0.5, max_backoff_seconds=10.0, jitter=0.0)
    assert policy.delay_for(1, retry_after=7.0) == pytest.approx(7.0)
    # A provider asking for five minutes gets the cap, not a parked process.
    assert policy.delay_for(1, retry_after=300.0) == pytest.approx(10.0)
    # Retry-After never shortens a backoff that has already grown past it.
    assert policy.delay_for(4, retry_after=0.1) == pytest.approx(4.0)


def test_retry_after_is_read_from_the_response_headers():
    assert retry_after_seconds(FakeHTTPError(429, retry_after="12")) == 12.0
    assert retry_after_seconds(FakeHTTPError(429)) is None
    # HTTP-date form is ignored rather than misparsed.
    http_date = FakeHTTPError(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
    assert retry_after_seconds(http_date) is None
    assert retry_after_seconds(ValueError("no response")) is None


# ------------------------------------------------------------ classification


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (FakeHTTPError(429), True),  # rate limited
        (FakeHTTPError(408), True),  # request timeout
        (FakeHTTPError(409), True),
        (FakeHTTPError(500), True),
        (FakeHTTPError(502), True),
        (FakeHTTPError(503), True),
        (FakeHTTPError(529), True),  # provider overloaded
        (FakeHTTPError(400), False),  # malformed request
        (FakeHTTPError(401), False),  # auth
        (FakeHTTPError(402), False),  # out of credit
        (FakeHTTPError(403), False),
        (FakeHTTPError(404), False),
        (FakeHTTPError(422), False),  # schema rejection
        (TimeoutError("read timed out"), True),
        (ConnectionError("reset by peer"), True),
        (TransientGatewayError("overloaded"), True),
        (GatewayError("not logged in"), False),
        (ValueError("I cannot help with that"), False),  # a refusal is a verdict
        (KeyError("bug"), False),
    ],
)
def test_transient_and_deterministic_failures_are_told_apart(exc, expected):
    assert is_transient(exc) is expected


def test_a_cost_ceiling_error_is_never_transient():
    """Spending more cannot fix having spent too much."""
    exc = CostCeilingExceeded("over", spent_usd=1.0, ceiling_usd=0.5)
    assert is_transient(exc) is False


def test_real_openai_exceptions_are_classified_correctly():
    """The rule has to hold for the actual objects the SDK raises, not only for
    look-alikes: two of them carry no status code at all."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    rate_limited = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=request, headers={"retry-after": "3"}),
        body=None,
    )
    bad_request = openai.BadRequestError(
        "bad", response=httpx.Response(400, request=request), body=None
    )
    unauthorized = openai.AuthenticationError(
        "no key", response=httpx.Response(401, request=request), body=None
    )
    server_error = openai.InternalServerError(
        "boom", response=httpx.Response(503, request=request), body=None
    )

    assert is_transient(rate_limited) is True
    assert retry_after_seconds(rate_limited) == 3.0
    assert is_transient(server_error) is True
    assert is_transient(bad_request) is False
    assert is_transient(unauthorized) is False
    # No status code on these: the request never reached a verdict.
    assert is_transient(openai.APITimeoutError(request=request)) is True
    assert is_transient(openai.APIConnectionError(message="dns", request=request)) is True
    assert is_transient(httpx.ConnectTimeout("timed out", request=request)) is True


# -------------------------------------------------------------- retry driver


def test_a_transient_failure_is_retried_until_it_succeeds():
    calls = {"n": 0}
    sleeper = Recorder()

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeHTTPError(429)
        return "ok"

    assert call_with_retry(flaky, sleep=sleeper) == "ok"
    assert calls["n"] == 3
    assert len(sleeper.delays) == 2
    assert sleeper.delays[0] < sleeper.delays[1]


def test_a_deterministic_failure_is_raised_on_the_first_attempt():
    calls = {"n": 0}
    sleeper = Recorder()

    def refused():
        calls["n"] += 1
        raise FakeHTTPError(400)

    with pytest.raises(FakeHTTPError):
        call_with_retry(refused, sleep=sleeper)
    assert calls["n"] == 1
    assert sleeper.delays == []


def test_retries_stop_at_max_attempts_and_the_last_error_propagates():
    calls = {"n": 0}
    sleeper = Recorder()

    def always_429():
        calls["n"] += 1
        raise FakeHTTPError(429)

    policy = RetryPolicy(max_attempts=4, initial_backoff_seconds=0.01)
    with pytest.raises(FakeHTTPError, match="429"):
        call_with_retry(always_429, policy=policy, sleep=sleeper)
    assert calls["n"] == 4
    assert len(sleeper.delays) == 3


def test_no_retry_policy_makes_exactly_one_attempt():
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise FakeHTTPError(429)

    with pytest.raises(FakeHTTPError):
        call_with_retry(always_429, policy=NO_RETRY, sleep=Recorder())
    assert calls["n"] == 1


def test_on_retry_sees_every_attempt_with_its_delay():
    seen: list[tuple[int, float, str]] = []

    def always_503():
        raise FakeHTTPError(503)

    policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01)
    with pytest.raises(FakeHTTPError):
        call_with_retry(
            always_503,
            policy=policy,
            sleep=Recorder(),
            on_retry=lambda attempt, delay, exc: seen.append((attempt, delay, str(exc))),
        )
    assert [attempt for attempt, _, _ in seen] == [1, 2]
    assert all(delay > 0 for _, delay, _ in seen)


def test_the_provider_retry_after_is_what_we_actually_sleep():
    sleeper = Recorder()
    policy = RetryPolicy(max_attempts=2, initial_backoff_seconds=0.01, max_backoff_seconds=30.0)

    def rate_limited():
        raise FakeHTTPError(429, retry_after="9")

    with pytest.raises(FakeHTTPError):
        call_with_retry(rate_limited, policy=policy, sleep=sleeper)
    assert sleeper.delays == [pytest.approx(9.0)]


def test_async_retry_follows_the_same_policy():
    calls = {"n": 0}
    sleeper = Recorder()

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise FakeHTTPError(500)
        return "ok"

    result = asyncio.run(
        acall_with_retry(
            flaky,
            policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01),
            sleep=sleeper.acall,
        )
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleeper.delays) == 1


def test_async_retry_does_not_retry_a_deterministic_failure():
    calls = {"n": 0}

    async def refused():
        calls["n"] += 1
        raise FakeHTTPError(401)

    with pytest.raises(FakeHTTPError):
        asyncio.run(acall_with_retry(refused, sleep=Recorder().acall))
    assert calls["n"] == 1


# ------------------------------------------------------------- spend metering


def test_the_meter_raises_on_the_call_that_crosses_the_ceiling():
    meter = SpendMeter(ceiling_usd=0.10)
    assert meter.charge(0.04) == pytest.approx(0.04)
    assert meter.charge(0.05) == pytest.approx(0.09)
    with pytest.raises(CostCeilingExceeded) as excinfo:
        meter.charge(0.03, model="m")
    # The crossing call is charged before the raise: a resumed run knows the bill.
    assert meter.spent_usd == pytest.approx(0.12)
    assert excinfo.value.spent_usd == pytest.approx(0.12)
    assert excinfo.value.ceiling_usd == 0.10
    assert meter.calls == 3


def test_the_meter_refuses_before_a_call_once_the_ceiling_is_reached():
    meter = SpendMeter(ceiling_usd=0.05)
    with pytest.raises(CostCeilingExceeded):
        meter.charge(0.06)
    with pytest.raises(CostCeilingExceeded, match="before this call"):
        meter.ensure_headroom()


def test_no_ceiling_means_accounting_without_enforcement():
    meter = SpendMeter()
    for _ in range(5):
        meter.charge(1000.0)
    meter.ensure_headroom()
    assert meter.spent_usd == pytest.approx(5000.0)


def test_an_unpriced_call_is_counted_rather_than_guessed_at():
    """A backend that reports no cost cannot be enforced against; the count is
    the signal that the ceiling saw less than the whole bill."""
    meter = SpendMeter(ceiling_usd=0.01)
    meter.charge(None)
    meter.charge(None)
    assert meter.spent_usd == 0.0
    assert meter.unpriced_calls == 2
    assert meter.calls == 2
    meter.ensure_headroom()  # nothing recorded, so nothing to refuse


def test_spend_is_attributed_per_model():
    meter = SpendMeter()
    meter.charge(0.01, model="a")
    meter.charge(0.02, model="b")
    meter.charge(0.03, model="a")
    assert meter.per_model_usd == {"a": pytest.approx(0.04), "b": pytest.approx(0.02)}
    assert meter.snapshot()["calls"] == 3


# --------------------------------------------------- CLI backend: retry paths


class FakeCompleted:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _ok_payload(text: str = "hello", cost: float | None = 0.0123) -> str:
    return json.dumps(
        {
            "type": "result",
            "result": text,
            "is_error": False,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": cost,
        }
    )


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace the gateway's sleep so retry tests cost no wall-clock time."""
    sleeper = Recorder()
    monkeypatch.setattr(resilience, "_sleep", sleeper)
    return sleeper


@pytest.fixture
def cli_runs(monkeypatch):
    """Script `subprocess.run`: each element is a CompletedProcess or an exception."""

    def install(*results):
        queued = list(results)
        seen: list[dict] = []

        def fake_run(argv, **kwargs):
            seen.append(kwargs)
            item = queued.pop(0) if len(queued) > 1 else queued[0]
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr(subprocess, "run", fake_run)
        return seen

    return install


def test_a_rate_limited_cli_call_is_retried_and_then_succeeds(cli_runs, no_sleep):
    seen = cli_runs(
        FakeCompleted(returncode=1, stderr="API Error: 429 rate limit exceeded"),
        FakeCompleted(_ok_payload()),
    )
    model = ClaudeCodeCLIChatModel()
    message = model.invoke("hi")

    assert message.content == "hello"
    assert len(seen) == 2
    assert len(no_sleep.delays) == 1
    assert model.last_usage["retries"] == 1


def test_a_cli_timeout_is_retried(cli_runs, no_sleep):
    seen = cli_runs(
        subprocess.TimeoutExpired(cmd="claude", timeout=1.0),
        FakeCompleted(_ok_payload()),
    )
    model = ClaudeCodeCLIChatModel()
    assert model.invoke("hi").content == "hello"
    assert len(seen) == 2


def test_an_exhausted_cli_retry_budget_raises_a_transient_error(cli_runs, no_sleep):
    seen = cli_runs(FakeCompleted(returncode=1, stderr="503 service unavailable"))
    model = ClaudeCodeCLIChatModel(retry_policy=RetryPolicy(max_attempts=3))
    with pytest.raises(TransientGatewayError, match="503"):
        model.invoke("hi")
    assert len(seen) == 3
    assert len(no_sleep.delays) == 2
    assert no_sleep.delays[0] < no_sleep.delays[1]


@pytest.mark.parametrize(
    ("stderr", "attempts"),
    [
        ("API Error: 429 rate limit exceeded", 3),
        ("Error: overloaded_error", 3),
        ("Invalid API key · Please run /login", 1),
        ("Not logged in. Please try again after running /login", 1),  # veto beats "try again"
        ("Error: 400 invalid_request_error", 1),
        ("something nobody has a rule for", 1),  # closed by default
    ],
)
def test_the_cli_retries_transient_text_and_not_verdicts(cli_runs, no_sleep, stderr, attempts):
    seen = cli_runs(FakeCompleted(returncode=1, stderr=stderr))
    model = ClaudeCodeCLIChatModel(retry_policy=RetryPolicy(max_attempts=3))
    with pytest.raises(GatewayError):
        model.invoke("hi")
    assert len(seen) == attempts


def test_a_missing_cli_binary_is_never_retried(no_sleep, monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, **kwargs):
        calls["n"] += 1
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GatewayError, match="not found"):
        ClaudeCodeCLIChatModel().invoke("hi")
    assert calls["n"] == 1


def test_non_json_output_is_not_retried(cli_runs, no_sleep):
    seen = cli_runs(FakeCompleted("<html>proxy error</html>"))
    with pytest.raises(GatewayError, match="non-JSON"):
        ClaudeCodeCLIChatModel().invoke("hi")
    assert len(seen) == 1


def test_an_error_payload_is_classified_from_its_own_text(cli_runs, no_sleep):
    seen = cli_runs(
        FakeCompleted(json.dumps({"is_error": True, "result": "Overloaded"})),
        FakeCompleted(_ok_payload()),
    )
    assert ClaudeCodeCLIChatModel().invoke("hi").content == "hello"
    assert len(seen) == 2


# ------------------------------------------------- CLI backend: the temp dir


def test_the_scratch_dir_is_gone_when_the_call_returns(cli_runs, no_sleep):
    seen = cli_runs(FakeCompleted(_ok_payload()))
    ClaudeCodeCLIChatModel().invoke("hi")
    cwd = seen[0]["cwd"]
    assert "grapharc-gateway-" in cwd
    assert not os.path.exists(cwd)


def test_the_scratch_dir_exists_while_the_cli_runs(monkeypatch, no_sleep):
    """It must be a real, empty directory during the call — that is what stops
    CLAUDE.md pickup — and only then be removed."""
    state = {}

    def fake_run(argv, **kwargs):
        cwd = kwargs["cwd"]
        state["cwd"] = cwd
        state["was_dir"] = os.path.isdir(cwd)
        state["was_empty"] = os.listdir(cwd) == []
        return FakeCompleted(_ok_payload())

    monkeypatch.setattr(subprocess, "run", fake_run)
    ClaudeCodeCLIChatModel().invoke("hi")
    assert state["was_dir"] and state["was_empty"]
    assert not os.path.exists(state["cwd"])


def test_repeated_calls_leave_no_orphaned_directories(monkeypatch, tmp_path, cli_runs, no_sleep):
    """The leak this replaces: one `mkdtemp` per call, never removed."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cli_runs(FakeCompleted(_ok_payload()))
    model = ClaudeCodeCLIChatModel()
    for _ in range(3):
        model.invoke("hi")
    assert list(tmp_path.iterdir()) == []


def test_a_failing_call_still_cleans_up_its_scratch_dir(monkeypatch, tmp_path, cli_runs, no_sleep):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cli_runs(FakeCompleted(returncode=1, stderr="not logged in"))
    with pytest.raises(GatewayError):
        ClaudeCodeCLIChatModel().invoke("hi")
    assert list(tmp_path.iterdir()) == []


def test_retried_attempts_share_one_scratch_dir_and_clean_it_up(
    monkeypatch, tmp_path, cli_runs, no_sleep
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = cli_runs(
        FakeCompleted(returncode=1, stderr="429 rate limit"),
        FakeCompleted(_ok_payload()),
    )
    ClaudeCodeCLIChatModel().invoke("hi")
    assert len(seen) == 2
    assert seen[0]["cwd"] == seen[1]["cwd"]
    assert list(tmp_path.iterdir()) == []


def test_an_explicit_workdir_is_used_and_never_deleted(cli_runs, no_sleep, tmp_path):
    workdir = tmp_path / "keep"
    workdir.mkdir()
    seen = cli_runs(FakeCompleted(_ok_payload()))
    ClaudeCodeCLIChatModel(workdir=str(workdir)).invoke("hi")
    assert seen[0]["cwd"] == str(workdir)
    assert workdir.is_dir()


# ------------------------------------------------ CLI backend: cost ceilings


def test_the_cli_backend_enforces_a_cost_ceiling(cli_runs, no_sleep):
    seen = cli_runs(FakeCompleted(_ok_payload(cost=0.0123)))
    model = ClaudeCodeCLIChatModel(cost_ceiling_usd=0.02)

    model.invoke("first")  # 0.0123 total — under
    assert model.last_usage["cumulative_cost_usd"] == pytest.approx(0.0123)

    with pytest.raises(CostCeilingExceeded, match="ceiling exceeded"):
        model.invoke("second")  # 0.0246 total — over
    assert len(seen) == 2
    # The call that crossed the line still reports what it cost.
    assert model.last_usage["cost_usd"] == pytest.approx(0.0123)
    assert model.last_usage["cumulative_cost_usd"] == pytest.approx(0.0246)

    with pytest.raises(CostCeilingExceeded, match="before this call"):
        model.invoke("third")
    assert len(seen) == 2, "an exhausted ceiling must not reach the provider"


def test_the_usage_envelope_carries_cumulative_spend(cli_runs, no_sleep):
    cli_runs(FakeCompleted(_ok_payload(cost=0.01)))
    model = ClaudeCodeCLIChatModel()
    model.invoke("a")
    model.invoke("b")
    model.invoke("c")
    assert model.last_usage["cost_usd"] == pytest.approx(0.01)
    assert model.last_usage["cumulative_cost_usd"] == pytest.approx(0.03)
    assert model.spend.calls == 3


def test_one_meter_shared_by_two_models_shares_one_ceiling(cli_runs, no_sleep):
    cli_runs(FakeCompleted(_ok_payload(cost=0.01)))
    meter = SpendMeter(ceiling_usd=0.015)
    a = ClaudeCodeCLIChatModel(model="claude-sonnet-5", spend=meter)
    b = ClaudeCodeCLIChatModel(model="claude-haiku-4-5", spend=meter)
    a.invoke("hi")
    with pytest.raises(CostCeilingExceeded):
        b.invoke("hi")
    assert meter.spent_usd == pytest.approx(0.02)
    assert set(meter.per_model_usd) == {"claude-sonnet-5", "claude-haiku-4-5"}


def test_a_cost_ceiling_failure_is_not_retried(cli_runs, no_sleep):
    seen = cli_runs(FakeCompleted(_ok_payload(cost=1.0)))
    model = ClaudeCodeCLIChatModel(cost_ceiling_usd=0.5)
    with pytest.raises(CostCeilingExceeded):
        model.invoke("hi")
    assert len(seen) == 1
    assert no_sleep.delays == []


# ------------------------------------------------------- OpenRouter backend

pytest.importorskip("langchain_openai", reason="OpenRouter needs the openrouter extra")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import (  # noqa: E402
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_openai import ChatOpenAI  # noqa: E402

from grapharc.gateway.openrouter import OpenRouterChatModel  # noqa: E402


def _or_result(cost: float | None = 0.001) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="pong"))],
        llm_output={
            "model_name": "openai/gpt-4o-mini",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": cost},
        },
    )


def _or_model(**kwargs) -> OpenRouterChatModel:
    return OpenRouterChatModel("openai/gpt-4o-mini", api_key="sk-or-test", **kwargs)


@pytest.fixture
def or_generate(monkeypatch):
    """Script `ChatOpenAI._generate`, the layer the OpenRouter class wraps."""

    def install(*results, attr: str = "_generate"):
        queued = list(results)
        calls: list[int] = []

        def fake(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(1)
            item = queued.pop(0) if len(queued) > 1 else queued[0]
            if isinstance(item, BaseException):
                raise item
            return item

        async def afake(self, messages, stop=None, run_manager=None, **kwargs):
            return fake(self, messages, stop=stop, run_manager=run_manager, **kwargs)

        monkeypatch.setattr(ChatOpenAI, attr, afake if attr == "_agenerate" else fake)
        return calls

    return install


def test_openrouter_retries_a_429_and_then_succeeds(or_generate, no_sleep):
    calls = or_generate(FakeHTTPError(429), _or_result())
    model = _or_model()
    assert model.invoke("hi").content == "pong"
    assert len(calls) == 2
    assert len(no_sleep.delays) == 1
    assert model.last_usage["retries"] == 1


def test_openrouter_does_not_retry_a_400(or_generate, no_sleep):
    calls = or_generate(FakeHTTPError(400))
    with pytest.raises(FakeHTTPError):
        _or_model().invoke("hi")
    assert len(calls) == 1
    assert no_sleep.delays == []


def test_openrouter_backoff_grows_across_attempts(or_generate, no_sleep):
    calls = or_generate(FakeHTTPError(503))
    model = _or_model(retry_policy=RetryPolicy(max_attempts=4, initial_backoff_seconds=0.5))
    with pytest.raises(FakeHTTPError):
        model.invoke("hi")
    assert len(calls) == 4
    assert len(no_sleep.delays) == 3
    assert no_sleep.delays[0] < no_sleep.delays[1] < no_sleep.delays[2]


def test_openrouter_leaves_the_sdk_retry_layer_off_by_default():
    """Two retry layers would compose into max_attempts * sdk_retries calls."""
    assert _or_model().max_retries == 0
    assert _or_model(max_retries=4).max_retries == 4


def test_openrouter_does_not_retry_while_streaming(or_generate, no_sleep):
    """Tokens already delivered to the caller cannot be un-delivered.

    Driven through `_generate` directly: `ChatOpenAI._generate` consumes the
    stream internally when `streaming=True`, which is the case the guard covers.
    """
    calls = or_generate(FakeHTTPError(429))
    model = _or_model(streaming=True)
    with pytest.raises(FakeHTTPError):
        model._generate([HumanMessage(content="hi")])
    assert len(calls) == 1

    # Same failure, same model, streaming off -> retried. Without the guard the
    # two cases would be indistinguishable.
    calls_streaming_off = or_generate(FakeHTTPError(429))
    with pytest.raises(FakeHTTPError):
        _or_model()._generate([HumanMessage(content="hi")])
    assert len(calls_streaming_off) == 3


def test_a_streamed_call_is_checked_against_the_ceiling_before_it_starts(monkeypatch, no_sleep):
    """LangChain routes streamed calls past `_generate`, so `_stream` is the
    only place a ceiling can bite on that path."""
    chunks = [
        ChatGenerationChunk(message=AIMessageChunk(content="po")),
        ChatGenerationChunk(message=AIMessageChunk(content="ng")),
    ]
    monkeypatch.setattr(ChatOpenAI, "_stream", lambda self, *a, **k: iter(chunks))

    meter = SpendMeter(ceiling_usd=0.001)
    model = _or_model(spend=meter, streaming=True)

    assert "".join(c.content for c in model.stream("hi")) == "pong"
    # Cost is not recoverable from the chunks, so the meter records that it
    # missed one rather than implying it saw the whole bill.
    assert meter.unpriced_calls == 1
    assert meter.calls == 1
    assert model.last_usage is None

    meter.spent_usd = 0.005  # a prior non-streamed call took it over
    with pytest.raises(CostCeilingExceeded, match="before this call"):
        list(model.stream("hi"))
    assert meter.calls == 1, "the refused stream never reached the provider"


def test_openrouter_async_retries_the_same_way(or_generate, no_sleep):
    calls = or_generate(FakeHTTPError(429), _or_result(), attr="_agenerate")
    model = _or_model()
    reply = asyncio.run(model.ainvoke("hi"))
    assert reply.content == "pong"
    assert len(calls) == 2


def test_openrouter_enforces_a_cost_ceiling(or_generate, no_sleep):
    calls = or_generate(_or_result(cost=0.004))
    model = _or_model(cost_ceiling_usd=0.005)
    model.invoke("first")
    with pytest.raises(CostCeilingExceeded):
        model.invoke("second")
    assert model.last_usage["cumulative_cost_usd"] == pytest.approx(0.008)
    with pytest.raises(CostCeilingExceeded, match="before this call"):
        model.invoke("third")
    assert len(calls) == 2, "an exhausted ceiling must not reach the provider"


def test_openrouter_reports_cumulative_spend_in_the_envelope(or_generate, no_sleep):
    or_generate(_or_result(cost=0.002))
    model = _or_model()
    model.invoke("a")
    model.invoke("b")
    assert model.last_usage["cumulative_cost_usd"] == pytest.approx(0.004)
    assert model.spend.spent_usd == pytest.approx(0.004)


def test_both_backends_still_report_the_same_envelope_keys(or_generate, cli_runs, no_sleep):
    """A budget meter reads one shape whichever backend ran the turn."""
    or_generate(_or_result())
    cli_runs(FakeCompleted(_ok_payload()))
    openrouter_model = _or_model()
    openrouter_model.invoke("hi")
    cli_model = ClaudeCodeCLIChatModel()
    cli_model.invoke("hi")
    assert set(openrouter_model.last_usage) == set(cli_model.last_usage)
    assert {"cost_usd", "cumulative_cost_usd", "retries"} <= set(cli_model.last_usage)


# ------------------------------------------------------------- doc hygiene


def test_no_module_claims_a_hardcoded_model_count():
    """The number changes weekly and three files disagreed about it."""
    from grapharc import gateway
    from grapharc.gateway import openrouter as openrouter_module

    pattern = re.compile(r"~?\d{2,4}\s+models")
    for module in (gateway, openrouter_module, claude_cli, resilience):
        assert not pattern.search(module.__doc__ or ""), module.__name__
