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

from grapharc.gateway import different_providers, get_model
from grapharc.memory import MemoryStore
from grapharc.observe.metrics import summarize
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget

DEFAULT_REVIEWER = "openrouter/openai/gpt-4o-mini"

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
) -> int:
    trace = TraceRecorder(trace_path)
    run_id = f"live-{example}"

    print(f"model    : {model_spec}")
    model = get_model(model_spec, temperature=0)

    reviewer = None
    if _needs_reviewer(example):
        reviewer_spec = reviewer_spec or DEFAULT_REVIEWER
        print(f"reviewer : {reviewer_spec}")
        if not different_providers(model_spec, reviewer_spec):
            print(
                "  warning: author and reviewer share a provider — correlated "
                "agreement makes this weaker evidence than a cross-vendor pair"
            )
        reviewer = get_model(reviewer_spec, temperature=0)
    print(f"budget   : {LIVE_BUDGET.max_tokens:,} tokens / {LIVE_BUDGET.max_seconds:.0f}s")
    print()

    result = _build_and_run(example, model, reviewer, trace, run_id)
    if result is None:
        print(f"'{example}' has no live wiring yet")
        return 1

    for key, value in result.items():
        rendered = str(value)
        print(f"{key}: {rendered[:400]}{'…' if len(rendered) > 400 else ''}")

    metrics = summarize(trace, run_id)
    if metrics:
        print(
            f"\nspent: {metrics.tokens:,} tokens across {metrics.nodes_executed} nodes "
            f"in {metrics.duration_ms / 1000:.1f}s"
        )
    print(f"trace: {trace_path}")
    return 0


def _build_and_run(example, model, reviewer, trace, run_id):
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

        return build_stage6(model, MemoryStore(), **common).invoke(
            {
                "entities": ["GraphARC"],
                "source_name": "plan.md",
                "source_text": "GraphARC's orchestration runtime is LangGraph.",
            },
            run_id=run_id,
        )

    if example == "capstone":
        from grapharc.examples.capstone import DEMO_CORPUS, build_capstone

        return build_capstone(model, reviewer, MemoryStore(), **common).invoke(
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
