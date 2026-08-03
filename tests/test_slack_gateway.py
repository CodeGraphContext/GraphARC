"""The Slack gate, runner and formatter — everything except Slack itself.

The layering under test: `command.py` decides what Slack text may become an
argv, `runner.py` runs it against this interpreter's grapharc, `format.py`
turns the result into one message. `bot.py` is glue; its one behaviour worth a
test here (the missing-extra error) is tested by faking the import failure,
so none of this file needs a Slack token or a network.
"""

from __future__ import annotations

import sys

import pytest

from grapharc.slack.command import SlackCommandError, parse_command, usage_text
from grapharc.slack.config import SlackBotConfig, SlackConfigError
from grapharc.slack.format import MAX_FENCE_CHARS, format_result, mermaid_live_url
from grapharc.slack.runner import CommandResult, run_command

# ---------------------------------------------------------------------------
# The gate: what text is allowed to become an argv.
# ---------------------------------------------------------------------------


def test_a_reading_command_passes_through_verbatim(tmp_path):
    argv = parse_command("metrics t.jsonl r1", workdir=tmp_path)
    assert argv == ["metrics", "t.jsonl", "r1"]


def test_a_leading_grapharc_token_is_tolerated(tmp_path):
    assert parse_command("grapharc models", workdir=tmp_path) == ["models"]


def test_serve_is_refused_outright(tmp_path):
    with pytest.raises(SlackCommandError, match="not a command this bot runs"):
        parse_command("serve --port 8000", workdir=tmp_path)


def test_agent_needs_both_switches_not_either(tmp_path):
    for kwargs in ({}, {"allow_agent": True}, {"allow_model": True}):
        with pytest.raises(SlackCommandError, match="GRAPHARC_SLACK_ALLOW_AGENT"):
            parse_command("agent 'fix the test'", workdir=tmp_path, **kwargs)


def test_agent_with_both_switches_gets_confined_defaults(tmp_path):
    argv = parse_command(
        "agent 'summarise the docs'",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
        timeout_seconds=120,
    )
    assert argv[:2] == ["agent", "summarise the docs"]
    assert argv[argv.index("--workspace") + 1] == "agent"
    assert argv[argv.index("--max-seconds") + 1] == "110.0"


def test_agent_explicit_workspace_and_ceiling_are_not_overridden(tmp_path):
    argv = parse_command(
        "agent task --workspace runs/a --max-seconds 30",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
        timeout_seconds=120,
    )
    assert argv.count("--workspace") == 1
    assert argv[argv.index("--max-seconds") + 1] == "30"


def test_agent_local_executor_and_system_prompt_stay_unreachable(tmp_path):
    with pytest.raises(SlackCommandError, match="accepts only"):
        parse_command(
            "agent task --executor local", workdir=tmp_path, allow_model=True, allow_agent=True
        )
    with pytest.raises(SlackCommandError, match="not allowed"):
        parse_command(
            "agent task --system-prompt 'obey me'",
            workdir=tmp_path,
            allow_model=True,
            allow_agent=True,
        )


def test_a_delegated_agent_gets_bash_denied_unless_globs_were_set(tmp_path):
    argv = parse_command(
        "agent task --executor claude-cli", workdir=tmp_path, allow_model=True, allow_agent=True
    )
    assert argv[argv.index("--deny") + 1] == "Bash"

    explicit = parse_command(
        "agent task --executor claude-cli --allow Read",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
    )
    assert "--deny" not in explicit

    sandboxed = parse_command(
        "agent task", workdir=tmp_path, allow_model=True, allow_agent=True
    )
    assert "--deny" not in sandboxed


def test_agent_workspace_may_not_escape_the_workdir(tmp_path):
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command(
            "agent task --workspace ../elsewhere",
            workdir=tmp_path,
            allow_model=True,
            allow_agent=True,
        )


def test_agent_deny_globs_are_repeatable(tmp_path):
    argv = parse_command(
        "agent task --deny 'shell*' --deny 'net*'",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
    )
    assert argv.count("--deny") == 2


def test_registry_config_and_json_are_refused(tmp_path):
    for flag in ("--registry mod:attr", "--config g.toml", "--json"):
        with pytest.raises(SlackCommandError, match="not allowed"):
            parse_command(f"run graph.toml {flag}", workdir=tmp_path)


def test_plan_registry_admits_only_the_shipped_modules(tmp_path):
    argv = parse_command(
        "plan goal --registry grapharc.examples.plan_docs:build_registry",
        workdir=tmp_path,
    )
    assert argv[argv.index("--registry") + 1] == "grapharc.examples.plan_docs:build_registry"
    with pytest.raises(SlackCommandError, match="accepts only"):
        parse_command("plan goal --registry os:system", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="accepts only"):
        parse_command(
            "plan goal --registry=evil.module:build_registry", workdir=tmp_path
        )


def test_the_stdlib_plan_registry_needs_the_agent_double_opt_in(tmp_path):
    """Agent-backed plan kinds run tools on the host: same gate as `agent`."""
    for kwargs in ({}, {"allow_agent": True}, {"allow_model": True}):
        with pytest.raises(SlackCommandError, match="GRAPHARC_SLACK_ALLOW_AGENT"):
            parse_command(
                "plan goal --registry grapharc.stdlib:build_registry",
                workdir=tmp_path,
                **kwargs,
            )


def test_a_slack_stdlib_plan_is_always_parked_on_the_approval_gate(tmp_path):
    argv = parse_command(
        "plan goal --registry grapharc.stdlib:build_registry",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
        timeout_seconds=600,
    )
    assert "--approve" in argv
    assert argv[argv.index("--approval-timeout") + 1] == "300.0"

    # A requester-set gate config wins over the injection.
    explicit = parse_command(
        "plan goal --registry grapharc.stdlib:build_registry --approval-timeout 90",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
        timeout_seconds=600,
    )
    assert explicit.count("--approval-timeout") == 1
    assert explicit[explicit.index("--approval-timeout") + 1] == "90"


def test_the_demo_plan_registries_stay_reachable_without_opt_ins(tmp_path):
    argv = parse_command(
        "plan goal --registry grapharc.examples.plan_docs:build_registry",
        workdir=tmp_path,
    )
    assert "--approve" not in argv  # only the host-acting registry is parked


def test_a_repeated_registry_cannot_walk_the_agent_opt_in(tmp_path):
    """The gate read the first `--registry`; argparse would have run the last.

    A benign registry in front of the stdlib one was admitted against the
    benign value and executed against the agent one, with the forced
    `--approve` skipped in the same step. Both orders, so neither "the gate
    reads the last" nor "the gate reads the first" can pass this again.
    """
    kwargs = dict(workdir=tmp_path, allow_model=True, allow_agent=False)
    benign = "grapharc.examples.plan_docs:build_registry"
    stdlib = "grapharc.stdlib:build_registry"
    for first, second in ((benign, stdlib), (stdlib, benign)):
        with pytest.raises(SlackCommandError, match="twice"):
            parse_command(
                f"plan 'ship it' --model openrouter/x/y "
                f"--registry {first} --registry {second}",
                **kwargs,
            )
    # The single-flag refusal is untouched.
    with pytest.raises(SlackCommandError, match="GRAPHARC_SLACK_ALLOW_AGENT"):
        parse_command(f"plan 'ship it' --model openrouter/x/y --registry {stdlib}", **kwargs)
    # And so is the single-flag admission of a benign registry.
    argv = parse_command(f"plan 'ship it' --model openrouter/x/y --registry {benign}", **kwargs)
    assert argv.count("--registry") == 1


def test_a_repeated_model_cannot_smuggle_a_backend_past_the_spend_gate(tmp_path):
    with pytest.raises(SlackCommandError, match="paid backend"):
        parse_command("plan goal --model mock/x --model openrouter/a/b", workdir=tmp_path)
    # Opted in, a second `--model` is still refused: the gate must never have
    # to choose which of two values the CLI is going to use.
    with pytest.raises(SlackCommandError, match="twice"):
        parse_command(
            "plan goal --model mock/x --model openrouter/a/b",
            workdir=tmp_path,
            allow_model=True,
        )
    # The `=` form is the same flag, whichever way each occurrence is spelled.
    with pytest.raises(SlackCommandError, match="twice"):
        parse_command(
            "plan goal --model mock/x --model=openrouter/a/b",
            workdir=tmp_path,
            allow_model=True,
        )


def test_every_gated_flag_is_refused_in_its_duplicated_form(tmp_path):
    """Exhaustive over the allowlist, so a future gate cannot reopen the gap.

    Every admitted flag except the ones the CLI accumulates (`action="append"`)
    must be refused when it appears twice — the gate reads one occurrence, and
    a flag whose two occurrences could differ is a flag the gate cannot judge.
    """
    from grapharc.slack.command import ALLOWED_COMMANDS

    checked = 0
    for name, spec in ALLOWED_COMMANDS.items():
        flags = set(spec.bool_flags) | set(spec.model_flags) | set(spec.value_flags)
        flags |= set(spec.choice_flags)
        for flag in sorted(flags - set(spec.repeatable_flags)):
            if flag in spec.bool_flags:
                text = f"{name} {flag} {flag}"
            else:
                if flag in spec.choice_flags:
                    value = sorted(spec.choice_flags[flag])[0]
                elif spec.value_flags.get(flag, False):
                    value = "inside.jsonl"
                else:
                    value = "1"
                text = f"{name} {flag} {value} {flag} {value}"
            with pytest.raises(SlackCommandError, match="twice"):
                parse_command(text, workdir=tmp_path, allow_model=True, allow_agent=True)
            checked += 1
    assert checked > 20, "the allowlist shrank; this sweep should still be broad"


def test_the_flags_the_cli_accumulates_stay_repeatable(tmp_path):
    """`--allow`/`--deny` are argparse `append`: every occurrence reaches the run."""
    argv = parse_command(
        "agent task --deny 'shell*' --deny 'net*' --allow 'read*' --allow 'list*'",
        workdir=tmp_path,
        allow_model=True,
        allow_agent=True,
    )
    assert argv.count("--deny") == 2
    assert argv.count("--allow") == 2


def test_a_command_pasted_with_code_backticks_still_parses(tmp_path):
    """Copying from a code-formatted Slack message brings the backticks along."""
    argv = parse_command(
        "`approve slack-runs/x/trace.jsonl`", workdir=tmp_path
    )
    assert argv == ["approve", "slack-runs/x/trace.jsonl"]
    # A single stray trailing backtick — half a copy — is tolerated too.
    argv = parse_command("approve slack-runs/x/trace.jsonl`", workdir=tmp_path)
    assert argv == ["approve", "slack-runs/x/trace.jsonl"]


def test_approve_is_admitted_with_path_confinement(tmp_path):
    argv = parse_command("approve slack-runs/x/trace.jsonl --deny", workdir=tmp_path)
    assert argv == ["approve", "slack-runs/x/trace.jsonl", "--deny"]
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("approve ../elsewhere/trace.jsonl", workdir=tmp_path)


def test_model_is_refused_by_default_and_admitted_on_opt_in(tmp_path):
    with pytest.raises(SlackCommandError, match="paid backend"):
        parse_command("plan 'a goal' --model mock/x", workdir=tmp_path)
    argv = parse_command("plan 'a goal' --model mock/x", workdir=tmp_path, allow_model=True)
    assert argv[:4] == ["plan", "a goal", "--model", "mock/x"]
    assert "--trace" in argv  # tracing commands get a trace the bot can find


def test_a_path_positional_may_not_escape_the_workdir(tmp_path):
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("trace ../outside.jsonl", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("trace /etc/passwd", workdir=tmp_path)


def test_a_path_flag_value_may_not_escape_the_workdir_either_form(tmp_path):
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("plan goal --trace ../t.jsonl", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="escapes"):
        parse_command("plan goal --trace=../t.jsonl", workdir=tmp_path)


def test_a_nul_byte_is_a_refusal_not_an_exception(tmp_path):
    """`Path.resolve()` raises `ValueError` on a NUL; the bot must still reply.

    `handle_text_live` catches `SlackCommandError` and nothing else, so a
    `ValueError` out of the gate escaped the bolt listener and the requester
    saw no reply at all — indistinguishable from the bot being down.
    """
    from grapharc.slack.bot import handle_text

    for text in ("trace a\x00b", "plan goal --trace a\x00b", "plan a\x00b --model mock/x"):
        with pytest.raises(SlackCommandError, match="NUL byte"):
            parse_command(text, workdir=tmp_path, allow_model=True)

    config = SlackBotConfig(bot_token="xoxb-x", app_token="xapp-x", workdir=tmp_path)
    reply = handle_text("trace a\x00b", config)
    assert "NUL byte" in reply

    # `_confined` keeps the guarantee for any other caller of its own.
    from grapharc.slack.command import _confined

    with pytest.raises(SlackCommandError, match="NUL byte"):
        _confined("a\x00b", tmp_path)


def test_a_single_dash_token_is_refused_as_a_flag_not_taken_as_a_path(tmp_path):
    """The flag allowlist is meant to be exhaustive; `--` let short options by."""
    with pytest.raises(SlackCommandError, match="not allowed"):
        parse_command("trace -h", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="not allowed"):
        parse_command("run graph.toml -x", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="not allowed"):
        parse_command("metrics -", workdir=tmp_path)


def test_a_path_inside_the_workdir_is_admitted_even_absolute(tmp_path):
    inside = tmp_path / "runs" / "t.jsonl"
    argv = parse_command(f"trace {inside}", workdir=tmp_path)
    assert argv == ["trace", str(inside)]


def test_a_quoted_goal_survives_as_one_argument(tmp_path):
    argv = parse_command('plan "investigate the checkout outage"', workdir=tmp_path)
    assert argv[:2] == ["plan", "investigate the checkout outage"]


def test_a_tracing_command_gets_a_unique_injected_trace(tmp_path):
    first = parse_command("run graph.toml", workdir=tmp_path)
    second = parse_command("run graph.toml", workdir=tmp_path)
    first_trace = first[first.index("--trace") + 1]
    second_trace = second[second.index("--trace") + 1]
    assert first_trace.startswith("slack-runs/")
    assert first_trace.endswith("trace.jsonl")
    assert first_trace != second_trace, "a reused path would replay another run"


def test_a_requester_named_trace_wins_over_injection(tmp_path):
    argv = parse_command("run graph.toml --trace runs/mine.jsonl", workdir=tmp_path)
    assert argv.count("--trace") == 1
    assert argv[argv.index("--trace") + 1] == "runs/mine.jsonl"


def test_agent_also_gets_an_injected_trace(tmp_path):
    argv = parse_command(
        "agent task", workdir=tmp_path, allow_model=True, allow_agent=True
    )
    assert "--trace" in argv


def test_readers_get_no_trace_injection(tmp_path):
    assert "--trace" not in parse_command("metrics t.jsonl r1", workdir=tmp_path)
    assert "--trace" not in parse_command("models", workdir=tmp_path)


def test_trace_path_resolves_the_injected_and_named_forms(tmp_path):
    from grapharc.slack.command import trace_path

    argv = parse_command("run graph.toml", workdir=tmp_path)
    resolved = trace_path(argv, tmp_path)
    assert resolved is not None and resolved.is_relative_to(tmp_path)

    inline = trace_path(["run", "g.toml", "--trace=runs/t.jsonl"], tmp_path)
    assert inline == tmp_path / "runs" / "t.jsonl"

    assert trace_path(["metrics", "t.jsonl", "r1"], tmp_path) is None


def test_empty_text_answers_with_usage_not_a_traceback(tmp_path):
    with pytest.raises(SlackCommandError) as excinfo:
        parse_command("", workdir=tmp_path)
    assert "Allowed here" in str(excinfo.value)
    assert "agent" in usage_text()


def test_a_value_flag_missing_its_value_is_refused(tmp_path):
    with pytest.raises(SlackCommandError, match="needs a value"):
        parse_command("trace t.jsonl --run-id", workdir=tmp_path)


# ---------------------------------------------------------------------------
# Runner and formatter, end to end against the real CLI.
# ---------------------------------------------------------------------------


def test_models_runs_and_formats_as_success(tmp_path):
    result = run_command(["models"], workdir=tmp_path, timeout_seconds=60)
    assert result.exit_code == 0
    message = format_result(result)
    assert "did its job" in message
    assert "```" in message
    assert "\x1b" not in message, "an escape reached a Slack message"


def test_a_missing_graph_formats_as_could_not_run(tmp_path):
    result = run_command(
        ["run", str(tmp_path / "missing.toml"), "--trace", str(tmp_path / "t.jsonl")],
        workdir=tmp_path,
        timeout_seconds=60,
    )
    assert result.exit_code == 2
    message = format_result(result)
    assert "could not run" in message
    assert "stderr:" in message


def test_the_timeout_kills_the_process_and_says_so(tmp_path):
    # Interpreter startup alone exceeds this, so the timeout always fires.
    result = run_command(["models"], workdir=tmp_path, timeout_seconds=0.05)
    assert result.exit_code is None
    assert "was stopped" in format_result(result)


def test_truncation_is_announced_never_silent():
    result = CommandResult(
        argv=["trace", "t.jsonl"],
        exit_code=0,
        stdout="x" * (MAX_FENCE_CHARS + 500),
        stderr="",
        duration_seconds=0.1,
        timeout_seconds=60,
    )
    message = format_result(result)
    assert "500 earlier characters not shown" in message


def test_a_fence_in_the_output_cannot_break_out():
    result = CommandResult(
        argv=["trace", "t.jsonl"],
        exit_code=0,
        stdout="before\n```\nafter",
        stderr="",
        duration_seconds=0.1,
        timeout_seconds=60,
    )
    body = format_result(result).split("```", 1)[1]
    assert "\n```\n" not in body.rsplit("```", 1)[0]


def test_a_successful_viz_gets_a_render_link_and_the_url_round_trips():
    import base64
    import json
    import zlib

    mermaid = 'flowchart TD\n  start((start)) --> n0["triage"]'
    result = CommandResult(
        argv=["viz", "t.jsonl", "r1"],
        exit_code=0,
        stdout=mermaid + "\n",
        stderr="",
        duration_seconds=0.1,
        timeout_seconds=60,
    )
    message = format_result(result)
    assert "mermaid.live/view#pako:" in message

    packed = mermaid_live_url(mermaid).split("#pako:", 1)[1]
    decoded = json.loads(zlib.decompress(base64.urlsafe_b64decode(packed)))
    assert decoded["code"] == mermaid


def test_a_failed_or_non_viz_command_gets_no_render_link():
    for argv, code in ((["viz", "t.jsonl", "r1"], 1), (["trace", "t.jsonl"], 0)):
        result = CommandResult(
            argv=argv,
            exit_code=code,
            stdout="flowchart TD",
            stderr="",
            duration_seconds=0.1,
            timeout_seconds=60,
        )
        assert "mermaid.live" not in format_result(result)


# ---------------------------------------------------------------------------
# Config and the bot's import posture.
# ---------------------------------------------------------------------------


def test_missing_tokens_name_every_missing_variable():
    with pytest.raises(SlackConfigError, match="SLACK_BOT_TOKEN and SLACK_APP_TOKEN"):
        SlackBotConfig.from_env({})


def test_a_non_numeric_timeout_is_a_named_error_not_a_traceback(tmp_path):
    env = {
        "SLACK_BOT_TOKEN": "xoxb-x",
        "SLACK_APP_TOKEN": "xapp-x",
        "GRAPHARC_SLACK_TIMEOUT": "forever",
    }
    with pytest.raises(SlackConfigError, match="GRAPHARC_SLACK_TIMEOUT"):
        SlackBotConfig.from_env(env)


def test_config_reads_workdir_timeout_and_model_opt_in(tmp_path):
    config = SlackBotConfig.from_env(
        {
            "SLACK_BOT_TOKEN": "xoxb-x",
            "SLACK_APP_TOKEN": "xapp-x",
            "GRAPHARC_SLACK_WORKDIR": str(tmp_path),
            "GRAPHARC_SLACK_TIMEOUT": "5",
            "GRAPHARC_SLACK_ALLOW_MODEL": "1",
            "GRAPHARC_SLACK_ALLOW_AGENT": "1",
        }
    )
    assert config.workdir == tmp_path
    assert config.timeout_seconds == 5.0
    assert config.allow_model
    assert config.allow_agent


def test_live_config_defaults_on_and_reads_the_switches(tmp_path):
    base = {"SLACK_BOT_TOKEN": "xoxb-x", "SLACK_APP_TOKEN": "xapp-x"}
    assert SlackBotConfig.from_env(dict(base)).live is True
    assert SlackBotConfig.from_env({**base, "GRAPHARC_SLACK_LIVE": "0"}).live is False
    config = SlackBotConfig.from_env({**base, "GRAPHARC_SLACK_LIVE_INTERVAL": "5"})
    assert config.live_interval_seconds == 5.0


def test_a_bad_live_interval_is_a_named_error(tmp_path):
    base = {"SLACK_BOT_TOKEN": "xoxb-x", "SLACK_APP_TOKEN": "xapp-x"}
    with pytest.raises(SlackConfigError, match="GRAPHARC_SLACK_LIVE_INTERVAL"):
        SlackBotConfig.from_env({**base, "GRAPHARC_SLACK_LIVE_INTERVAL": "soon"})
    with pytest.raises(SlackConfigError, match="positive"):
        SlackBotConfig.from_env({**base, "GRAPHARC_SLACK_LIVE_INTERVAL": "0"})


def test_live_url_base_is_validated_and_stripped(tmp_path):
    base = {"SLACK_BOT_TOKEN": "xoxb-x", "SLACK_APP_TOKEN": "xapp-x"}
    assert SlackBotConfig.from_env(dict(base)).live_url_base is None
    config = SlackBotConfig.from_env(
        {**base, "GRAPHARC_SLACK_LIVE_URL": "https://laptop.tailnet.ts.net/"}
    )
    assert config.live_url_base == "https://laptop.tailnet.ts.net"
    with pytest.raises(SlackConfigError, match="GRAPHARC_SLACK_LIVE_URL"):
        SlackBotConfig.from_env({**base, "GRAPHARC_SLACK_LIVE_URL": "laptop:8000"})


def test_live_view_url_is_composed_only_for_tracing_argvs(tmp_path):
    from grapharc.slack.format import live_view_url

    argv = parse_command("run graph.toml", workdir=tmp_path)
    url = live_view_url(argv, base="https://laptop.example", workdir=tmp_path)
    assert url is not None
    assert url.startswith("https://laptop.example/live/view?trace=slack-runs%2F")

    assert live_view_url(argv, base=None, workdir=tmp_path) is None
    reader = parse_command("metrics t.jsonl r1", workdir=tmp_path)
    assert live_view_url(reader, base="https://laptop.example", workdir=tmp_path) is None


def test_live_view_url_carries_the_run_id_when_named(tmp_path):
    from grapharc.slack.format import live_view_url

    argv = parse_command("run graph.toml --run-id r7", workdir=tmp_path)
    url = live_view_url(argv, base="https://laptop.example", workdir=tmp_path)
    assert url is not None and url.endswith("&run=r7")


def test_handle_text_turns_a_refusal_into_a_message_not_an_exception(tmp_path):
    from grapharc.slack.bot import handle_text

    config = SlackBotConfig(bot_token="xoxb-x", app_token="xapp-x", workdir=tmp_path)
    reply = handle_text("<@U012345> agent rm -rf /", config)
    assert "GRAPHARC_SLACK_ALLOW_AGENT" in reply
    assert "not a command this bot runs" in handle_text("<@U012345> serve", config)


def test_a_missing_slack_extra_is_an_install_hint_not_an_import_error(monkeypatch, tmp_path):
    from grapharc.slack import bot

    monkeypatch.setitem(sys.modules, "slack_bolt", None)
    config = SlackBotConfig(bot_token="xoxb-x", app_token="xapp-x", workdir=tmp_path)
    with pytest.raises(SlackCommandError, match="slack"):
        bot.build_app(config)
