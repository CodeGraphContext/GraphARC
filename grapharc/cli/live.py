"""Run the example graphs against real models.

Until now every `grapharc run` used scripted responses, so nothing was known
about how routers, convergence guards, or the verifier behave on real model
output — which is exactly the regime this library claims to have engineered
for. This wires the example graphs to the gateway so they can be run for real.

Verifier graphs take a second, deliberately different provider. If both sides
come from the same vendor the run still proceeds, but it says so — a reviewer
grading its own model family is weaker evidence, and silently pretending
otherwise is the failure mode the verifier exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

from grapharc.cli import style
from grapharc.cli.output import EXIT_FAILED, EXIT_OK, emit
from grapharc.gateway import different_providers, get_model
from grapharc.observe.metrics import summarize
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget

DEFAULT_REVIEWER = "openrouter/openai/gpt-4o-mini"

#: The header labels line up at nine characters, which is what `model`,
#: `reviewer` and `budget` have always printed at.
LABEL_WIDTH = 9

# Real models are open-ended, so a live run always carries a ceiling.
LIVE_BUDGET = Budget(max_iterations=40, max_tokens=200_000, max_seconds=600)


def _needs_reviewer(example: str) -> bool:
    return example in ("stage5", "capstone")


def run_live(
    example: str,
    trace_path: Path,
    *,
    model_spec: str,
    reviewer_spec: str | None = None,
    as_json: bool = False,
    memory_path: Path | None = None,
    memory_backend: str = "sqlite",
) -> int:
    """Run one example against real models. Returns the process exit code.

    The header is printed as it is decided rather than with the result, because
    a live run can take minutes and an operator watching it needs to see which
    models were picked before the bill starts. JSON output has no such reader,
    so it is emitted once, at the end, as a single document.
    """
    trace = TraceRecorder(trace_path)
    run_id = f"live-{example}"

    def say(line: str) -> None:
        if not as_json:
            print(line)

    say(style.kv("model", model_spec, width=LABEL_WIDTH, tint=style.accent))
    model = get_model(model_spec, temperature=0)

    reviewer = None
    correlated = None
    if _needs_reviewer(example):
        reviewer_spec = reviewer_spec or DEFAULT_REVIEWER
        say(style.kv("reviewer", reviewer_spec, width=LABEL_WIDTH, tint=style.accent))
        correlated = not different_providers(model_spec, reviewer_spec)
        if correlated:
            # Amber, not red: the run is still valid, the evidence is just weaker.
            say(
                style.warn(
                    "  warning: author and reviewer share a provider — correlated "
                    "agreement makes this weaker evidence than a cross-vendor pair"
                )
            )
        reviewer = get_model(reviewer_spec, temperature=0)
    say(
        style.kv(
            "budget",
            f"{LIVE_BUDGET.max_tokens:,}{style.dim(' tokens / ')}"
            f"{LIVE_BUDGET.max_seconds:.0f}{style.dim('s')}",
            width=LABEL_WIDTH,
        )
    )
    say("")

    header = {
        "command": "run",
        "example": example,
        "live": True,
        "model": model_spec,
        "reviewer_model": reviewer_spec,
        "correlated_reviewer": correlated,
        "run_id": run_id,
        "trace": str(trace_path),
    }

    result = _build_and_run(
        example, model, reviewer, trace, run_id, memory_path, memory_backend
    )
    if result is None:
        emit(
            {"ok": False, **header, "error": f"'{example}' has no live wiring yet"},
            [style.err(f"'{example}' has no live wiring yet")],
            as_json=as_json,
        )
        return EXIT_FAILED

    metrics = summarize(trace, run_id)
    lines = []
    for key, value in result.items():
        rendered = str(value)
        clipped = f"{rendered[:400]}{'…' if len(rendered) > 400 else ''}"
        lines.append(style.kv(str(key), clipped))
    if metrics:
        lines.append("")
        lines.append(
            style.kv(
                "spent",
                f"{metrics.tokens:,}{style.dim(' tokens across ')}{metrics.nodes_executed}"
                f"{style.dim(' nodes in ')}{metrics.duration_ms / 1000:.1f}{style.dim('s')}",
            )
        )
    lines.append(style.kv("trace", str(trace_path), tint=style.accent))

    emit(
        {"ok": True, **header, "result": result, "metrics": metrics},
        lines,
        as_json=as_json,
    )
    return EXIT_OK


def _build_and_run(
    example, model, reviewer, trace, run_id, memory_path=None, memory_backend="sqlite"
):
    from grapharc.cli.main import _memory_store

    common = {"trace": trace, "budget": LIVE_BUDGET}

    if example == "stage1":
        from grapharc.examples.stage1_loop import DEMO_CHUNKS, build_stage1

        return build_stage1(model, **common).invoke(
            {"chunks": DEMO_CHUNKS, "targets": ["budgets", "verifier"]}, run_id=run_id
        )

    if example == "stage2":
        from grapharc.examples.stage2_claims import DEMO_SOURCE, build_stage2

        source = _write(trace, "source.md", DEMO_SOURCE)
        return build_stage2(model, **common).invoke(
            {"source_path": str(source)}, run_id=run_id
        )

    if example == "stage3":
        from grapharc.examples.stage3_fanout import DEMO_CHUNKS, build_stage3

        return build_stage3(model, **common).invoke(
            {"question": "How do budgets bound work?", "chunks": DEMO_CHUNKS},
            run_id=run_id,
        )

    if example == "stage4":
        from grapharc.examples.stage4_investigation import DEMO_CORPUS, build_stage4

        return build_stage4(model, **common).invoke(
            {"corpus": DEMO_CORPUS, "goal": "map the runtime", "target_findings": 2},
            run_id=run_id,
        )

    if example == "stage5":
        from grapharc.examples.stage5_verifier import DEMO_SOURCE, build_stage5

        return build_stage5(model, reviewer, **common).invoke(
            {"source_text": DEMO_SOURCE}, run_id=run_id
        )

    if example == "stage6":
        from grapharc.examples.stage6_memory import build_stage6

        return build_stage6(model, _memory_store(memory_path, memory_backend), **common).invoke(
            {
                "entities": ["GraphARC"],
                "source_name": "plan.md",
                "source_text": "GraphARC's orchestration runtime is LangGraph.",
            },
            run_id=run_id,
        )

    if example == "capstone":
        from grapharc.examples.capstone import DEMO_CORPUS, build_capstone

        store = _memory_store(memory_path, memory_backend)
        return build_capstone(model, reviewer, store, **common).invoke(
            {
                "question": "How does GraphARC bound work with budgets?",
                "corpus": DEMO_CORPUS,
                "entities": ["GraphARC"],
            },
            run_id=run_id,
        )

    return None  # stage0 is deterministic; there is no model to swap in


def _write(trace: TraceRecorder, name: str, content: str) -> Path:
    path = trace.path.parent / name
    path.write_text(content, encoding="utf-8")
    return path
