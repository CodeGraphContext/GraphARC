"""The `grapharc` CLI: run agents, serve the API, and read what a run recorded.

Every command speaks two languages. The human form is the default; `--json`
prints the same payload as one document on stdout, so a run can be driven from
a script without parsing prose. Errors follow the same rule — in JSON mode the
failure is the document, not a line on stderr.

Exit codes are part of the interface: `0` the command did its job, `1` it ran
and the answer was negative (an agent stopped short, a run id had no events, no
backend was usable, two runs differed), `2` it could not run at all (a component
is missing, a file does not exist, a model spec names no backend).

Reading commands (`trace`, `metrics`, `viz`, `diff`) all read the same JSONL the
runtime writes, so the metrics and the audit trail cannot disagree: there is
only one record.

Human output is coloured only when stdout is an interactive terminal, and the
colour is the *only* difference: same lines, same words, same column widths. So
`grapharc … | grep`, a captured log and the transcripts printed in `README.md`
all keep the bytes they had before `grapharc.cli.style` existed. `--no-color`,
`NO_COLOR` and `--json` each turn it off.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from grapharc import __version__

# `grapharc.cli.agent` is safe to import here: it pulls in nothing but the
# stdlib and this package. Every component it *drives* — the toolset, the
# gateway, the harness — it imports inside the command.
from grapharc.cli import style
from grapharc.cli.agent import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    run_agent,
)
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.replay import ReplayError
from grapharc.observe.trace import TraceReadError, TraceRecorder

EXAMPLES = (
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "capstone",
)


def _memory_store(path: Path | None, backend: str = "sqlite"):
    """The claim store the shipped graphs get.

    In-process by default, so `grapharc run` stays hermetic and repeatable and
    writes nothing a caller did not ask for. Given `--memory PATH`, a durable
    backend instead — the same stores the memory tests prove survive a process
    restart. This is the only difference between a demo that forgets and one
    whose claims are still there on the next run.

    `--memory-backend` picks which durable one. SQLite is the default because
    it needs no extra and tolerates concurrent processes; LadybugDB stores the
    same claims as a property graph you can query in Cypher, at the cost of an
    exclusive lock on the database (see `grapharc.memory.ladybug_store`).
    """
    if path is None:
        from grapharc.memory import MemoryStore

        return MemoryStore()
    if backend == "ladybug":
        from grapharc.memory import LadybugMemoryStore

        return LadybugMemoryStore(path)
    from grapharc.memory import SQLiteMemoryStore

    return SQLiteMemoryStore(path)


def _run_example(
    name: str,
    trace_path: Path,
    memory_path: Path | None = None,
    memory_backend: str = "sqlite",
) -> dict:
    trace = TraceRecorder(trace_path)
    workdir = Path(tempfile.mkdtemp(prefix=f"grapharc-{name}-"))
    if name == "stage0":
        from grapharc.examples.stage0_dag import DEMO_DOC, build_stage0

        doc = workdir / "doc.md"
        doc.write_text(DEMO_DOC, encoding="utf-8")
        report = workdir / "report.md"
        result = build_stage0(trace=trace).invoke(
            {"doc_path": str(doc), "report_path": str(report)}
        )
        result["report"] = report.read_text(encoding="utf-8")
        return result
    if name == "stage1":
        from grapharc.examples.stage1_loop import DEMO_CHUNKS, build_stage1
        from grapharc.testing import ScriptedChatModel

        model = ScriptedChatModel(responses=['{"term": "budgets"}', '{"term": "verifier"}'])
        return build_stage1(model, trace=trace).invoke(
            {"chunks": DEMO_CHUNKS, "targets": ["budgets", "verifier"]}
        )
    if name == "stage2":
        from grapharc.examples.stage2_claims import DEMO_SOURCE, build_stage2
        from grapharc.testing import ScriptedChatModel

        source = workdir / "source.md"
        source.write_text(DEMO_SOURCE, encoding="utf-8")
        model = ScriptedChatModel(
            responses=[
                json.dumps(
                    {
                        "claims": [
                            {
                                "text": "GraphARC uses typed state contracts",
                                "citation": "typed state contracts",
                            }
                        ]
                    }
                ),
                "GraphARC enforces typed state contracts.",
            ]
        )
        return build_stage2(model, trace=trace).invoke({"source_path": str(source)})
    if name == "stage3":
        from grapharc.examples.stage3_fanout import DEMO_CHUNKS, build_stage3
        from grapharc.testing import ScriptedChatModel

        model = ScriptedChatModel(responses=["Budgets cap iterations, tokens, and time."])
        return build_stage3(model, trace=trace).invoke(
            {"question": "How do budgets bound work?", "chunks": DEMO_CHUNKS}
        )
    if name == "stage4":
        from grapharc.examples.stage4_investigation import DEMO_CORPUS, build_stage4
        from grapharc.testing import ScriptedChatModel

        model = ScriptedChatModel(
            responses=['{"query": "budget meters"}', '{"query": "trace lines"}'],
            on_exhausted="repeat",
        )
        return build_stage4(model, trace=trace).invoke(
            {"corpus": DEMO_CORPUS, "goal": "map the runtime", "target_findings": 2}
        )
    if name == "stage5":
        from grapharc.examples.stage5_verifier import DEMO_SOURCE, build_stage5
        from grapharc.testing import ScriptedChatModel

        author = ScriptedChatModel(
            responses=[
                json.dumps(
                    {
                        "claims": [
                            {
                                "text": "GraphARC enforces typed state contracts",
                                "citation": "typed state contracts",
                            },
                            {
                                "text": "GraphARC is 10x faster",
                                "citation": "benchmarked at 10x",
                            },
                        ]
                    }
                )
            ]
        )
        reviewer = ScriptedChatModel(
            responses=[json.dumps({"supported": True, "reason": "evidence matches"})]
        )
        return build_stage5(author, reviewer, trace=trace).invoke(
            {"source_text": DEMO_SOURCE}
        )
    if name == "stage6":
        from grapharc.examples.stage6_memory import build_stage6
        from grapharc.testing import ScriptedChatModel

        store = _memory_store(memory_path, memory_backend)
        model = ScriptedChatModel(
            responses=[
                json.dumps(
                    {
                        "claims": [
                            {
                                "subject": "GraphARC",
                                "predicate": "orchestration runtime",
                                "object": "LangGraph",
                            }
                        ]
                    }
                ),
                "GraphARC runs on LangGraph [plan.md].",
            ]
        )
        return build_stage6(model, store, trace=trace).invoke(
            {
                "entities": ["GraphARC"],
                "source_name": "plan.md",
                "source_text": "GraphARC's orchestration runtime is LangGraph.",
            }
        )
    if name == "capstone":
        from grapharc.examples.capstone import DEMO_CORPUS, build_capstone
        from grapharc.testing import ScriptedChatModel

        worker = ScriptedChatModel(
            responses=["Budgets cap iterations and tokens [doc 0]."], on_exhausted="repeat"
        )
        reviewer = ScriptedChatModel(
            responses=['{"supported": true, "reason": "quote supports it"}'],
            on_exhausted="repeat",
        )
        return build_capstone(
            worker, reviewer, _memory_store(memory_path, memory_backend), trace=trace
        ).invoke(
            {
                "question": "How does GraphARC bound work with budgets?",
                "corpus": DEMO_CORPUS,
                "entities": ["GraphARC"],
            }
        )
    raise SystemExit(f"unknown example: {name!r}")


def _existing_trace(path: Path, *, command: str, as_json: bool) -> TraceRecorder | int:
    """A recorder for an existing file, or the exit code for saying it is not there.

    Checked before constructing the recorder because `TraceRecorder.__init__`
    creates the parent directory: a typo in a read-only command should not leave
    a directory behind.

    Existence is not enough. A directory, a file whose permissions forbid the
    read, or any other `OSError` used to escape as a traceback with exit 1,
    because the handlers below catch only `TraceReadError` — so the file is
    opened here, where the failure is still reportable as the exit-2 document
    the contract promises.
    """
    if not path.exists():
        return fail(f"no such trace file: {path}", as_json=as_json, command=command)
    try:
        path.open("rb").close()
    except OSError as exc:
        return fail(
            f"unreadable trace file: {path}: {exc.strerror or exc}",
            as_json=as_json,
            command=command,
        )
    return TraceRecorder(path)


# -- command handlers ---------------------------------------------------------


def _cmd_demo(args: argparse.Namespace) -> int:
    from grapharc.cli.config import ConfigError
    from grapharc.cli.config import load as load_settings

    trace_path = args.trace or Path(tempfile.mkdtemp(prefix="grapharc-")) / "trace.jsonl"
    # `model`, `reviewer_model` and `memory` are declared in `config.KEYS`, and
    # for a while nothing read the last two — the same file made `plan` fail on
    # a bad model spec and left `demo` running scripted. Resolving them here is
    # what makes those keys mean something.
    try:
        settings = load_settings(args.config)
        args.model = settings.resolve("model", args.model)
        args.reviewer_model = settings.resolve("reviewer_model", args.reviewer_model)
        memory = settings.resolve_path("memory", args.memory)
        args.memory = memory
    except ConfigError as exc:
        return fail(str(exc), as_json=args.json, command="demo", example=args.example)
    if args.model:
        from grapharc.cli.live import run_live

        return run_live(
            args.example,
            trace_path,
            model_spec=args.model,
            reviewer_spec=args.reviewer_model,
            as_json=args.json,
            memory_path=args.memory,
            memory_backend=args.memory_backend,
        )
    try:
        result = _run_example(
            args.example,
            trace_path,
            memory_path=args.memory,
            memory_backend=args.memory_backend,
        )
    except (OSError, sqlite3.Error) as exc:
        # `--memory` takes a path from a human, so it can name a directory, a
        # parent that does not exist, or somewhere unwritable. That is a report,
        # not a traceback with empty stdout in `--json` mode.
        return fail(
            f"--memory {str(args.memory)!r}: {exc}",
            as_json=args.json,
            command="demo",
            example=args.example,
        )
    payload = {
        "ok": True,
        "command": "demo",
        "example": args.example,
        "trace": str(trace_path),
        "result": result,
    }
    lines = style.kv_block((str(key), str(value)) for key, value in result.items())
    lines += ["", style.kv("trace", str(trace_path), tint=style.accent)]
    emit(payload, lines, as_json=args.json)
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    from grapharc.cli.graphrun import run_graph

    return run_graph(
        args.graph,
        registry_target=args.registry,
        policy_path=args.policy,
        tenant=args.tenant,
        trace_path=args.trace,
        run_id=args.run_id,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
        max_concurrency=args.max_concurrency,
        check_only=args.check_only,
        config_path=args.config,
        as_json=args.json,
    )


def _cmd_plan(args: argparse.Namespace) -> int:
    from grapharc.cli.plan import plan

    return plan(
        args.goal,
        model_spec=args.model,
        registry_target=args.registry,
        policy_path=args.policy,
        tenant=args.tenant,
        trace_path=args.trace,
        run_id=args.run_id,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        max_planning_failures=args.max_planning_failures,
        workspace=args.workspace,
        model_arg_pairs=args.model_arg,
        config_path=args.config,
        approve=args.approve,
        approval_timeout=args.approval_timeout,
        as_json=args.json,
        scripted=args.scripted,
        go_after=args.go_after,
        use_default_registry=args.default_registry,
    )


def _cmd_init(args: argparse.Namespace) -> int:
    from grapharc.cli.init_cmd import init

    return init(as_json=args.json)


def _cmd_start(args: argparse.Namespace) -> int:
    from grapharc.cli.start import start

    return start(as_json=args.json)


def _cmd_go(args: argparse.Namespace) -> int:
    from grapharc.cli.plan import PLAN_FILENAME, execute_plan, plan

    # Two natures, one word. Bare `go` — or `go <run-dir>` — executes a plan
    # `grapharc plan` saved; `go "some goal"` is plan-and-execute in one run.
    target = args.goal
    if target is None:
        return execute_plan(
            None,
            model_spec=args.model,
            model_arg_pairs=args.model_arg,
            workspace=args.workspace,
            policy_path=args.policy,
            tenant=args.tenant,
            run_id=args.run_id,
            max_tokens=args.max_tokens,
            config_path=args.config,
            as_json=args.json,
        )
    candidate = Path(target)
    if candidate.name == PLAN_FILENAME and candidate.is_file() or (
        candidate.is_dir() and (candidate / PLAN_FILENAME).is_file()
    ):
        return execute_plan(
            target,
            model_spec=args.model,
            model_arg_pairs=args.model_arg,
            workspace=args.workspace,
            policy_path=args.policy,
            tenant=args.tenant,
            run_id=args.run_id,
            max_tokens=args.max_tokens,
            config_path=args.config,
            as_json=args.json,
        )
    return plan(
        target,
        model_spec=args.model,
        registry_target=args.registry,
        policy_path=args.policy,
        tenant=args.tenant,
        trace_path=args.trace,
        run_id=args.run_id,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        max_planning_failures=args.max_planning_failures,
        workspace=args.workspace,
        model_arg_pairs=args.model_arg,
        config_path=args.config,
        approve=args.approve,
        approval_timeout=args.approval_timeout,
        as_json=args.json,
        command="go",
        go_after=True,
        use_default_registry=args.default_registry,
    )


def _cmd_approve(args: argparse.Namespace) -> int:
    from grapharc.cli.approve import approve

    return approve(args.path, deny=args.deny, as_json=args.json)


def _cmd_models(args: argparse.Namespace) -> int:
    from grapharc.gateway import (
        describe,
        ollama_base_url,
        openai_api_key,
        openrouter_api_key,
        redact,
    )
    from grapharc.gateway.registry import BACKENDS

    if args.check:
        from grapharc.cli.probe import any_provider_usable, probe_backends, render

        results = probe_backends()
        usable = any_provider_usable(results)
        emit(
            {"ok": usable, "command": "models", "check": True, "backends": results},
            # Styled by `probe.render`, which owns the three-column layout.
            render(results),
            as_json=args.json,
        )
        return EXIT_OK if usable else EXIT_FAILED

    if args.spec:
        # Imported here, not at module scope: `test_building_the_parser_imports_
        # no_optional_package` pins that building the parser pulls in no optional
        # dependency, so `--help` stays fast and works without extras installed.
        from grapharc.gateway.registry import UnknownBackendError

        try:
            resolved = describe(args.spec)
        except UnknownBackendError as exc:
            # The top-level help lists "a model spec names no backend" under
            # exit 2; this used to escape as a traceback with exit 1.
            return fail(str(exc), as_json=args.json, command="models", spec=args.spec)
        emit(
            {"ok": True, "command": "models", **resolved},
            style.kv_block((str(key), str(value)) for key, value in resolved.items()),
            as_json=args.json,
        )
        return EXIT_OK

    # This line used to advertise "~400 models". No count is quoted now: the
    # CLI cannot count what OpenRouter serves today, a number nobody re-checks
    # rots, and the gateway settled on the same wording ("many providers, one
    # key"). `--check` reports what this machine can actually reach.
    examples = {
        "claude-cli/claude-sonnet-5": "subscription, no API key",
        "openrouter/anthropic/claude-haiku-4.5": "many providers, one key",
        "openrouter/openai/gpt-4o-mini:floor": "cheapest provider for that model",
        "openai/gpt-4o-mini": "the OpenAI API directly, your key",
        "ollama/llama3.1": "a local server, no key and no bill",
    }
    payload = {
        "ok": True,
        "command": "models",
        "backends": list(BACKENDS),
        "openrouter_key": redact(openrouter_api_key()),
        "openai_key": redact(openai_api_key()),
        # An address, not a secret: it is printed whole, and it is where a
        # request would go rather than proof that anything is listening.
        "ollama_base_url": ollama_base_url(),
        "examples": examples,
    }
    # A redacted key is left the default colour on purpose: highlighting the one
    # field on the page that is a fingerprint of a secret invites reading it out.
    lines = [
        style.kv("backends", ", ".join(BACKENDS)),
        style.kv("openrouter key", redact(openrouter_api_key())),
        style.kv("openai key", redact(openai_api_key())),
        style.kv("ollama url", ollama_base_url(), tint=style.accent),
        "",
        style.heading("examples:"),
        *[
            f"  {style.cell(spec, 38, tint=style.accent)}  {style.dim(note)}"
            for spec, note in examples.items()
        ],
        "",
        style.accent("grapharc models --check")
        + style.dim("  probes which of these this machine can use"),
    ]
    emit(payload, lines, as_json=args.json)
    return EXIT_OK


def _cmd_agent(args: argparse.Namespace) -> int:
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="grapharc-agent-"))
    return run_agent(
        args.task,
        model_spec=args.model,
        workspace=Path(workspace),
        trace_path=args.trace,
        allow=args.allow,
        deny=args.deny,
        ask=args.ask,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
        executor=args.executor,
        system_prompt=args.system_prompt,
        run_id=args.run_id,
        as_json=args.json,
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    from grapharc.cli.serve import serve

    return serve(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        registry_target=args.registry,
        live_root=args.live_root,
        live_token=args.live_token,
        as_json=args.json,
    )


def _cmd_replay(args: argparse.Namespace) -> int:
    from grapharc.cli.replay import replay

    return replay(args.path, args.run_id, as_json=args.json)


def _cmd_diff(args: argparse.Namespace) -> int:
    from grapharc.cli.replay import diff

    return diff(args.path, args.run_a, args.run_b, as_json=args.json)


def _cmd_trace(args: argparse.Namespace) -> int:
    recorder = _existing_trace(args.path, command="trace", as_json=args.json)
    if isinstance(recorder, int):
        return recorder
    try:
        events = recorder.read_events(args.run_id)
    except TraceReadError as exc:
        # A bad line — a truncated write, a hand edit — is the "unreadable
        # trace" the exit-code contract names, not a traceback. Same for
        # `metrics` and `viz` below: all three read this file.
        return fail(str(exc), as_json=args.json, command="trace")
    payload: dict[str, Any] = {
        "ok": True,
        "command": "trace",
        "path": str(args.path),
        "run_id": args.run_id,
        "count": len(events),
        "events": [e.model_dump(exclude_none=True) for e in events],
    }
    # `[  3] act                  end    Δ{'candidate': 1}` — the step counter and
    # the phase are scaffolding, the node name and the state delta are the reason
    # anyone reads a trace, and `!` is the only thing that ever needs finding in a
    # thousand lines. Columns are unchanged; the colour is the index.
    lines = []
    for event in events:
        delta = f" {style.accent('Δ')}{event.state_delta}" if event.state_delta else ""
        failed = f" {style.err(f'!{event.error}')}" if event.error else ""
        lines.append(
            f"{style.dim(f'[{event.step:>3}]')} "
            f"{style.cell(event.node, 20, tint=style.accent)} "
            f"{style.cell(event.phase, 6, tint=style.dim)}{delta}{failed}"
        )
    emit(payload, lines, as_json=args.json)
    return EXIT_OK


def _cmd_metrics(args: argparse.Namespace) -> int:
    recorder = _existing_trace(args.path, command="metrics", as_json=args.json)
    if isinstance(recorder, int):
        return recorder
    try:
        metrics = summarize(recorder, args.run_id)
    except TraceReadError as exc:
        return fail(str(exc), as_json=args.json, command="metrics")
    if metrics is None:
        return fail(
            f"no events for run {args.run_id!r} in {args.path}",
            as_json=args.json,
            command="metrics",
            code=EXIT_FAILED,
        )
    data = metrics.model_dump()
    emit(
        {"ok": True, "command": "metrics", **data},
        style.kv_block((str(key), str(value)) for key, value in data.items()),
        as_json=args.json,
    )
    return EXIT_OK


def _cmd_viz(args: argparse.Namespace) -> int:
    recorder = _existing_trace(args.path, command="viz", as_json=args.json)
    if isinstance(recorder, int):
        return recorder
    try:
        mermaid = to_mermaid(recorder, args.run_id)
    except TraceReadError as exc:
        return fail(str(exc), as_json=args.json, command="viz")
    except ReplayError as exc:
        # Every other reading command answers this as a document; `viz` used to
        # let it out as a traceback with empty stdout, which breaks the CLI's own
        # promise that in JSON mode the failure *is* the document.
        return fail(str(exc), as_json=args.json, command="viz", code=EXIT_FAILED)
    emit(
        {
            "ok": True,
            "command": "viz",
            "path": str(args.path),
            "run_id": args.run_id,
            "mermaid": mermaid,
        },
        # Deliberately unstyled, on a terminal too. This output exists to be
        # pasted into a Mermaid renderer; a box around it or an escape sequence
        # inside it would make the one thing the command is for stop working.
        [mermaid],
        as_json=args.json,
    )
    return EXIT_OK


# -- parser -------------------------------------------------------------------


def _add_planning_flags(parser: argparse.ArgumentParser) -> None:
    """The flags `plan` and `go` share, added identically to both.

    One list, two parsers: a flag that exists on one and not the other is a
    doc bug waiting to be filed, and the handlers thread every one of these
    into the same `plan()` call.
    """
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        metavar="PATH",
        help="TOML policy document whose edge rules become the admission gate's EdgePolicy",
    )
    parser.add_argument(
        "--tenant", default=None, metavar="NAME", help="tenant to compile --policy for"
    )
    parser.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")
    parser.add_argument(
        "--run-id", default=None, help="name this run; refused if --trace already holds it"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help="planning rounds the loop may take (default: 8)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="run token ceiling across every round (default: 100000)",
    )
    parser.add_argument(
        "--max-planning-failures", type=int, default=None, metavar="N",
        help="consecutive unusable planner replies before giving up (default: 3)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "confine the registry's tool-using kinds (and its workspace listing) "
            "to DIR; refused when the registry cannot take one"
        ),
    )
    parser.add_argument(
        "--model-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="constructor argument for --model (repeatable), e.g. --model-arg temperature=0",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="pause each admitted round until `grapharc approve` answers next to the trace",
    )
    parser.add_argument(
        "--approval-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="how long --approve waits before the round counts as unapproved (default: 300)",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        dest="default_registry",
        help=(
            "use the built-in general-purpose kinds (read/edit files, write a "
            "report) instead of a registry.py in this directory"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grapharc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"grapharc {__version__}")
    # Not required: bare `grapharc` orients a newcomer (exit 0, like `-h`)
    # instead of erroring — `main()` handles the no-command branch.
    sub = parser.add_subparsers(dest="command", required=False)

    # `--json` is attached to every subcommand rather than to the top level, so
    # `grapharc run stage0 --json` works — the position a shell user reaches for.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="print one JSON document instead of text"
    )
    # On `common` rather than the top level, for the same reason `--json` is: the
    # position a shell user reaches for is after the arguments. Declaring it in
    # both places would not work anyway — the subparser's default would overwrite
    # a value given before the subcommand, in the one namespace they share.
    #
    # Colour is off whenever stdout is not a terminal, so this is only ever needed
    # to *keep* it off somewhere `isatty()` says yes and the reader disagrees: a
    # tty being recorded, a pager that shows escapes, a screen reader. `NO_COLOR`
    # in the environment does the same thing for every run at once.
    common.add_argument(
        "--no-color",
        action="store_true",
        help="never colour the output, even when stdout is a terminal (see also NO_COLOR)",
    )
    # `--config` belongs only to the commands that resolve settings. It used to
    # live on `common`, so every command accepted it and nine silently ignored
    # it — including rejecting a missing file on two commands and not the rest.
    configurable = argparse.ArgumentParser(add_help=False)
    configurable.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "read defaults from PATH instead of ./grapharc.toml. Parent "
            "directories are never searched: a run must not be governed by a "
            "file the operator did not know about"
        ),
    )

    demo = sub.add_parser(
        "demo",
        parents=[common, configurable],
        help="run one of the built-in example graphs",
    )
    demo.add_argument("example", choices=EXAMPLES)
    demo.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")
    demo.add_argument(
        "--model",
        default=None,
        metavar="SPEC",
        help=(
            "run against a real model instead of scripted responses, e.g. "
            "openrouter/anthropic/claude-haiku-4.5 or claude-cli/claude-sonnet-5"
        ),
    )
    demo.add_argument(
        "--reviewer-model",
        default=None,
        metavar="SPEC",
        help="model for verifier nodes; should be a different provider from --model",
    )
    demo.add_argument(
        "--memory",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "persist claims to a durable store at PATH instead of the "
            "in-process one, so stage6 and capstone remember across runs"
        ),
    )
    demo.add_argument(
        "--memory-backend",
        choices=("sqlite", "ladybug"),
        default="sqlite",
        help=(
            "durable backend for --memory: sqlite (default, no extra needed, "
            "safe for concurrent processes) or ladybug (a LadybugDB property "
            "graph you can query in Cypher; needs the ladybug extra, and only "
            "one process may hold it at a time)"
        ),
    )
    demo.set_defaults(handler=_cmd_demo)

    run = sub.add_parser(
        "run",
        parents=[common, configurable],
        help="admit and execute a topology file you wrote",
    )
    run.add_argument("graph", help="path to a JSON or TOML topology file")
    run.add_argument(
        "--registry",
        default=None,
        metavar="MODULE:ATTR",
        help="the node kinds this graph may name",
    )
    run.add_argument(
        "--policy",
        type=Path,
        default=None,
        metavar="PATH",
        help="TOML policy document whose edge rules gate this graph",
    )
    run.add_argument(
        "--tenant", default=None, metavar="NAME", help="tenant to compile --policy for"
    )
    run.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")
    run.add_argument(
        "--run-id", default=None, help="name this run; refused if --trace already holds it"
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "refuse a topology whose worst case exceeds this many tokens. "
            "Without it this budget dimension is unlimited"
        ),
    )
    run.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="stop the run after this many node iterations (unlimited if omitted)",
    )
    run.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="wall-clock run ceiling; interrupts a running node (unlimited if omitted)",
    )
    run.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="max concurrent node executions (unlimited if omitted)",
    )
    run.add_argument(
        "--check-only",
        action="store_true",
        help="validate the topology against the policy and stop; execute nothing",
    )
    run.set_defaults(handler=_cmd_run)

    from grapharc.cli.plan import DEFAULT_REGISTRY

    plan = sub.add_parser(
        "plan",
        parents=[common, configurable],
        help="rehearse the governed planning loop against a goal (see also: go)",
    )
    plan.add_argument("goal", help="what the planner should plan for")
    plan.add_argument(
        "--model",
        default=None,
        metavar="SPEC",
        help="plan with a real model instead of the scripted demo planner",
    )
    plan.add_argument(
        "--registry",
        default=None,
        metavar="MODULE:ATTR",
        help=f"the node kinds a planner may propose (default: {DEFAULT_REGISTRY})",
    )
    plan.add_argument(
        "--go",
        action="store_true",
        dest="go_after",
        help="plan and immediately execute it, in one run",
    )
    plan.add_argument(
        "--scripted",
        action="store_true",
        help=(
            "rehearse with the registry's built-in stand-in planner instead of "
            "a model — free, deterministic, no AI involved; contradicts --model"
        ),
    )
    _add_planning_flags(plan)
    plan.set_defaults(handler=_cmd_plan)

    # `go` is `plan` with doing-defaults: the stdlib registry (real tool-using
    # kinds) and a required model — go means do, and the scripted planner does
    # nothing worth doing. Same handler underneath; only the defaults differ.
    go = sub.add_parser(
        "go",
        parents=[common, configurable],
        help="plan AND do: a model proposes the graph, stdlib kinds do real work",
    )
    go.add_argument(
        "goal",
        nargs="?",
        default=None,
        help=(
            "what to get done — or a run directory holding a saved plan.json; "
            "with nothing, the newest unexecuted plan runs"
        ),
    )
    go.add_argument(
        "--model",
        default=None,
        metavar="SPEC",
        help="required: the model that plans and works (see `grapharc models --check`)",
    )
    go.add_argument(
        "--registry",
        default=None,
        metavar="MODULE:ATTR",
        help="the node kinds a planner may propose (default: grapharc.stdlib:build_registry)",
    )
    _add_planning_flags(go)
    go.set_defaults(handler=_cmd_go)

    ini = sub.add_parser(
        "init",
        parents=[common],
        help="scaffold a registry, a config and a runs directory in this directory",
    )
    ini.set_defaults(handler=_cmd_init)

    st = sub.add_parser(
        "start",
        parents=[common],
        help="the guided tour: concept, first run, live view",
    )
    st.set_defaults(handler=_cmd_start)

    ap = sub.add_parser(
        "approve",
        parents=[common],
        help="answer a plan run waiting on its approval gate",
    )
    ap.add_argument("path", type=Path, help="the paused run's trace file (or its directory)")
    ap.add_argument("--deny", action="store_true", help="refuse the plan instead of approving it")
    ap.set_defaults(handler=_cmd_approve)

    agent = sub.add_parser(
        "agent", parents=[common], help="run an agent node against a task with the core tools"
    )
    agent.add_argument("task", help="what the agent should do")
    agent.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="SPEC",
        help="model spec; needs a tool-calling backend (default: %(default)s)",
    )
    agent.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory the tools work in (default: a fresh temp dir)",
    )
    agent.add_argument("--trace", type=Path, default=None, help="default: <workspace>/trace.jsonl")
    agent.add_argument(
        "--run-id",
        default=None,
        help="name this run, refused if --trace already holds it (default: agent-<random>)",
    )
    agent.add_argument(
        "--allow",
        action="append",
        default=None,
        metavar="PATTERN",
        help="tool-name glob the agent may call, repeatable (default: *)",
    )
    agent.add_argument(
        "--deny",
        action="append",
        default=None,
        metavar="PATTERN",
        help="tool-name glob to refuse; evaluated first, so it beats --allow",
    )
    agent.add_argument(
        "--ask",
        action="append",
        default=None,
        metavar="PATTERN",
        help="tool-name glob to confirm interactively; refused when stdin is not a tty",
    )
    agent.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="model calls the loop may make (default: %(default)s)",
    )
    agent.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="run token ceiling, enforced on the call that crosses it (default: %(default)s)",
    )
    agent.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="wall clock, interrupted into a call already running (default: %(default)s)",
    )
    agent.add_argument(
        "--executor",
        choices=("sandbox", "local", "claude-cli"),
        default="sandbox",
        help=(
            "local runs tools in this process with no confinement; claude-cli "
            "delegates the whole loop to Claude Code's headless agent on your "
            "subscription (default: sandbox)"
        ),
    )
    agent.add_argument("--system-prompt", default=None)
    agent.set_defaults(handler=_cmd_agent)

    serve = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.add_argument(
        "--registry",
        default=None,
        metavar="MODULE:ATTR",
        help="graph registry to serve; without one the app starts with no graphs",
    )
    serve.add_argument(
        "--live-root",
        default=None,
        metavar="PATH",
        help="also serve a read-only live view of the trace files under PATH at /live",
    )
    serve.add_argument(
        "--live-token",
        default=os.environ.get("GRAPHARC_LIVE_TOKEN") or None,
        metavar="TOKEN",
        help="require this token on every /live request (default: GRAPHARC_LIVE_TOKEN)",
    )
    serve.set_defaults(handler=_cmd_serve)

    md = sub.add_parser("models", parents=[common], help="what a model spec resolves to")
    md.add_argument("spec", nargs="?", default=None)
    md.add_argument(
        "--check",
        action="store_true",
        help="probe which backends this machine can use (local check; no provider is called)",
    )
    md.set_defaults(handler=_cmd_models)

    rp = sub.add_parser("replay", parents=[common], help="re-execute a recorded run")
    rp.add_argument("path", type=Path)
    rp.add_argument("run_id")
    rp.set_defaults(handler=_cmd_replay)

    df = sub.add_parser("diff", parents=[common], help="compare two runs in one trace")
    df.add_argument("path", type=Path)
    df.add_argument("run_a")
    df.add_argument("run_b")
    df.set_defaults(handler=_cmd_diff)

    tr = sub.add_parser("trace", parents=[common], help="pretty-print a trace JSONL file")
    tr.add_argument("path", type=Path)
    tr.add_argument("--run-id", default=None, help="filter to one run")
    tr.set_defaults(handler=_cmd_trace)

    mx = sub.add_parser("metrics", parents=[common], help="summarize a run from its trace")
    mx.add_argument("path", type=Path)
    mx.add_argument("run_id")
    mx.set_defaults(handler=_cmd_metrics)

    vz = sub.add_parser("viz", parents=[common], help="render a run's executed path as Mermaid")
    vz.add_argument("path", type=Path)
    vz.add_argument("run_id")
    vz.set_defaults(handler=_cmd_viz)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    # `grapharc help` is what people type; making it an argparse error taught
    # nothing. It is `-h`, and `help <command>` is `<command> -h`.
    if raw[:1] == ["help"]:
        parser = build_parser()
        rest = raw[1:]
        if rest:
            return main([*rest, "-h"])
        parser.print_help()
        return EXIT_OK
    args = build_parser().parse_args(argv)
    # `--json` disables colour too, belt and braces: a JSON run has no human
    # reader, and `tests/test_cli.py` requires that exactly one document reaches
    # stdout and nothing at all reaches stderr. Turning styling off at the source
    # means no future call site can leak an escape sequence into a payload.
    style.configure(
        no_color=getattr(args, "no_color", False) or getattr(args, "json", False)
    )
    if args.command is None:
        from grapharc.cli.start import orientation

        for line in orientation():
            print(line)
        return EXIT_OK
    if args.command == "models" and args.check and args.spec:
        return fail(
            "`models --check` probes the configured backends and `models <spec>` "
            "resolves one spec; ask for one or the other",
            as_json=args.json,
            command="models",
        )
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
