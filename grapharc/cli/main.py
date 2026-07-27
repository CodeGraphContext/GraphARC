"""The `grapharc` CLI: run the stage examples, read traces, and inspect runs.

Every command reads the same JSONL trace the runtime writes — the metrics and
the audit trail cannot disagree because there is only one record.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from grapharc import __version__
from grapharc.observe.metrics import summarize, to_mermaid
from grapharc.observe.trace import TraceRecorder


def _run_example(name: str, trace_path: Path) -> dict:
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
        from grapharc.memory import MemoryStore
        from grapharc.testing import ScriptedChatModel

        store = MemoryStore()
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
        from grapharc.memory import MemoryStore
        from grapharc.testing import ScriptedChatModel

        worker = ScriptedChatModel(
            responses=["Budgets cap iterations and tokens [doc 0]."], on_exhausted="repeat"
        )
        reviewer = ScriptedChatModel(
            responses=['{"supported": true, "reason": "quote supports it"}'],
            on_exhausted="repeat",
        )
        return build_capstone(worker, reviewer, MemoryStore(), trace=trace).invoke(
            {
                "question": "How does GraphARC bound work with budgets?",
                "corpus": DEMO_CORPUS,
                "entities": ["GraphARC"],
            }
        )
    raise SystemExit(f"unknown example: {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grapharc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"grapharc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    examples = [
        "stage0",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "stage5",
        "stage6",
        "capstone",
    ]
    run = sub.add_parser("run", help="run a built-in example graph (scripted models)")
    run.add_argument("example", choices=examples)
    run.add_argument("--trace", type=Path, default=None, help="trace JSONL output path")

    tr = sub.add_parser("trace", help="pretty-print a trace JSONL file")
    tr.add_argument("path", type=Path)
    tr.add_argument("--run-id", default=None, help="filter to one run")

    mx = sub.add_parser("metrics", help="summarize a run from its trace")
    mx.add_argument("path", type=Path)
    mx.add_argument("run_id")

    vz = sub.add_parser("viz", help="render a run's executed path as Mermaid")
    vz.add_argument("path", type=Path)
    vz.add_argument("run_id")

    args = parser.parse_args(argv)

    if args.command == "run":
        trace_path = args.trace or Path(tempfile.mkdtemp(prefix="grapharc-")) / "trace.jsonl"
        result = _run_example(args.example, trace_path)
        for key, value in result.items():
            print(f"{key}: {value}")
        print(f"\ntrace: {trace_path}")
        return 0

    if args.command == "trace":
        for event in TraceRecorder(args.path).read_events(args.run_id):
            delta = f" Δ{event.state_delta}" if event.state_delta else ""
            err = f" !{event.error}" if event.error else ""
            print(f"[{event.step:>3}] {event.node:<20} {event.phase:<6}{delta}{err}")
        return 0

    if args.command == "metrics":
        metrics = summarize(TraceRecorder(args.path), args.run_id)
        if metrics is None:
            print(f"no events for run {args.run_id!r} in {args.path}", file=sys.stderr)
            return 1
        for key, value in metrics.model_dump().items():
            print(f"{key}: {value}")
        return 0

    if args.command == "viz":
        print(to_mermaid(TraceRecorder(args.path), args.run_id))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
