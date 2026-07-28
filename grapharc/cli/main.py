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
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from grapharc import __version__

# `grapharc.cli.agent` is safe to import here: it pulls in nothing but the
# stdlib and this package. Every component it *drives* — the toolset, the
# gateway, the harness — it imports inside the command.
from grapharc.cli.agent import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    run_agent,
)
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit, fail
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.trace import TraceRecorder

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


def _memory_store(path: Path | None):
    """The claim store the shipped graphs get.

    In-process by default, so `grapharc run` stays hermetic and repeatable and
    writes nothing a caller did not ask for. Given `--memory PATH`, the durable
    SQLite backend instead — the same store the memory tests prove survives a
    process restart. This is the only difference between a demo that forgets
    and one whose claims are still there on the next run.
    """
    if path is None:
        from grapharc.memory import MemoryStore

        return MemoryStore()
    from grapharc.memory import SQLiteMemoryStore

    return SQLiteMemoryStore(path)


def _run_example(name: str, trace_path: Path, memory_path: Path | None = None) -> dict:
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

        store = _memory_store(memory_path)
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
        return build_capstone(worker, reviewer, _memory_store(memory_path), trace=trace).invoke(
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
    """
    if not path.exists():
        return fail(f"no such trace file: {path}", as_json=as_json, command=command)
    return TraceRecorder(path)


# -- command handlers ---------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    trace_path = args.trace or Path(tempfile.mkdtemp(prefix="grapharc-")) / "trace.jsonl"
    if args.model:
        from grapharc.cli.live import run_live

        return run_live(
            args.example,
            trace_path,
            model_spec=args.model,
            reviewer_spec=args.reviewer_model,
            as_json=args.json,
            memory_path=args.memory,
        )
    result = _run_example(args.example, trace_path, memory_path=args.memory)
    payload = {
        "ok": True,
        "command": "run",
        "example": args.example,
        "trace": str(trace_path),
        "result": result,
    }
    lines = [f"{key}: {value}" for key, value in result.items()]
    lines += ["", f"trace: {trace_path}"]
    emit(payload, lines, as_json=args.json)
    return EXIT_OK


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
        as_json=args.json,
    )


def _cmd_models(args: argparse.Namespace) -> int:
    from grapharc.gateway import describe, openrouter_api_key, redact
    from grapharc.gateway.registry import BACKENDS

    if args.check:
        from grapharc.cli.probe import any_provider_usable, probe_backends, render

        results = probe_backends()
        usable = any_provider_usable(results)
        emit(
            {"ok": usable, "command": "models", "check": True, "backends": results},
            render(results),
            as_json=args.json,
        )
        return EXIT_OK if usable else EXIT_FAILED

    if args.spec:
        resolved = describe(args.spec)
        emit(
            {"ok": True, "command": "models", **resolved},
            [f"{key}: {value}" for key, value in resolved.items()],
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
    }
    payload = {
        "ok": True,
        "command": "models",
        "backends": list(BACKENDS),
        "openrouter_key": redact(openrouter_api_key()),
        "examples": examples,
    }
    lines = [
        f"backends: {', '.join(BACKENDS)}",
        f"openrouter key: {redact(openrouter_api_key())}",
        "",
        "examples:",
        *[f"  {spec:<38}  {note}" for spec, note in examples.items()],
        "",
        "grapharc models --check  probes which of these this machine can use",
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
    events = recorder.read_events(args.run_id)
    payload: dict[str, Any] = {
        "ok": True,
        "command": "trace",
        "path": str(args.path),
        "run_id": args.run_id,
        "count": len(events),
        "events": [e.model_dump(exclude_none=True) for e in events],
    }
    lines = []
    for event in events:
        delta = f" Δ{event.state_delta}" if event.state_delta else ""
        err = f" !{event.error}" if event.error else ""
        lines.append(f"[{event.step:>3}] {event.node:<20} {event.phase:<6}{delta}{err}")
    emit(payload, lines, as_json=args.json)
    return EXIT_OK


def _cmd_metrics(args: argparse.Namespace) -> int:
    recorder = _existing_trace(args.path, command="metrics", as_json=args.json)
    if isinstance(recorder, int):
        return recorder
    metrics = summarize(recorder, args.run_id)
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
        [f"{key}: {value}" for key, value in data.items()],
        as_json=args.json,
    )
    return EXIT_OK


def _cmd_viz(args: argparse.Namespace) -> int:
    recorder = _existing_trace(args.path, command="viz", as_json=args.json)
    if isinstance(recorder, int):
        return recorder
    mermaid = to_mermaid(recorder, args.run_id)
    emit(
        {
            "ok": True,
            "command": "viz",
            "path": str(args.path),
            "run_id": args.run_id,
            "mermaid": mermaid,
        },
        [mermaid],
        as_json=args.json,
    )
    return EXIT_OK


# -- parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grapharc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"grapharc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # `--json` is attached to every subcommand rather than to the top level, so
    # `grapharc run stage0 --json` works — the position a shell user reaches for.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="print one JSON document instead of text"
    )

    run = sub.add_parser("run", parents=[common], help="run a built-in example graph")
    run.add_argument("example", choices=EXAMPLES)
    run.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")
    run.add_argument(
        "--model",
        default=None,
        metavar="SPEC",
        help=(
            "run against a real model instead of scripted responses, e.g. "
            "openrouter/anthropic/claude-haiku-4.5 or claude-cli/claude-sonnet-5"
        ),
    )
    run.add_argument(
        "--reviewer-model",
        default=None,
        metavar="SPEC",
        help="model for verifier nodes; should be a different provider from --model",
    )
    run.add_argument(
        "--memory",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "persist claims to a durable SQLite store at PATH instead of the "
            "in-process one, so stage6 and capstone remember across runs"
        ),
    )
    run.set_defaults(handler=_cmd_run)

    from grapharc.cli.plan import DEFAULT_REGISTRY

    plan = sub.add_parser(
        "plan", parents=[common], help="drive the governed planning loop against a goal"
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
        default=DEFAULT_REGISTRY,
        metavar="MODULE:ATTR",
        help="the node kinds a planner may propose (default: %(default)s)",
    )
    plan.add_argument(
        "--policy",
        type=Path,
        default=None,
        metavar="PATH",
        help="TOML policy document whose edge rules become the admission gate's EdgePolicy",
    )
    plan.add_argument(
        "--tenant",
        default="default",
        metavar="NAME",
        help="tenant to compile --policy for (default: %(default)s)",
    )
    plan.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")
    plan.add_argument("--run-id", default=None, help="name this run")
    plan.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="planning rounds the loop may take (default: %(default)s)",
    )
    plan.add_argument(
        "--max-tokens",
        type=int,
        default=100_000,
        help="run token ceiling across every round (default: %(default)s)",
    )
    plan.set_defaults(handler=_cmd_plan)

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
    agent.add_argument("--run-id", default=None, help="name this run (default: agent-<random>)")
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
        choices=("sandbox", "local"),
        default="sandbox",
        help="local runs tools in this process with no confinement (default: sandbox)",
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
    args = build_parser().parse_args(argv)
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
