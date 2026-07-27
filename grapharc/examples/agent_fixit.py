"""A real agent: read a failing project, fix it, and prove the fix by testing.

This is ROADMAP §4.1's gate, and the first example in GraphARC where a model
actually *does* something rather than describing something. It composes the
three subsystems that were built separately and, until now, never met: the
gateway calls a model with tools bound, the harness decides which of those
tools may run and executes them, and the runtime meters and traces the whole
thing.

What makes it a gate rather than a demo is what is enforced in code:

  * `delete_file` is registered and **denied by policy**, so it never appears
    in the schemas the model is given. The model cannot ask for what it was
    never offered — that is policy-before-schema, not a prompt instruction.
  * The loop stops for a machine-readable reason, always.
  * Tokens are metered automatically, so the budget is real.

Run it with a real model:

    python -m grapharc.examples.agent_fixit --model openrouter/anthropic/claude-haiku-4.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from grapharc.gateway import get_model
from grapharc.harness import (
    AgentNode,
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
    ToolSpec,
)
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import RunContext

# `add` subtracts. The test suite catches it; the agent has to find and fix it.
BROKEN_SOURCE = "def add(a, b):\n    return a - b\n"
TEST_SOURCE = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"

SYSTEM_PROMPT = (
    "You are fixing a bug in a small Python project. Read calc.py to find the "
    "bug, use write_file to correct it, then use run_tests to confirm the suite "
    "passes. Stop once the tests pass."
)
TASK = (
    "The test suite is failing. Read calc.py, fix the bug, and run the tests "
    "until they pass."
)


def make_workspace(root: Path | None = None) -> Path:
    workspace = root or Path(tempfile.mkdtemp(prefix="grapharc-fixit-"))
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "calc.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    (workspace / "test_calc.py").write_text(TEST_SOURCE, encoding="utf-8")
    return workspace


def run_pytest(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )


def build_harness(workspace: Path) -> Harness:
    """Registry + policy for the fix-it task.

    `run_tests` shells out to pytest, which the sandboxed executor refuses by
    design — a child process runs without the audit hook, so it cannot be
    confined. This example therefore uses `LocalExecutor` and leans on the
    permission layer, which still applies. A task needing both real isolation
    and a subprocess wants the container executor (ROADMAP §3.2).
    """

    def read_file(path: str) -> str:
        """Read a file from the project."""
        return (workspace / Path(path).name).read_text(encoding="utf-8")

    def write_file(path: str, content: str) -> str:
        """Replace a file in the project with new content."""
        (workspace / Path(path).name).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {path}"

    def run_tests() -> str:
        """Run the project's test suite and return its output."""
        proc = run_pytest(workspace)
        return (proc.stdout + proc.stderr)[-1500:]

    def delete_file(path: str) -> str:
        """Delete a file from the project."""
        (workspace / Path(path).name).unlink()
        return f"deleted {path}"

    registry = ToolRegistry()
    for fn in (read_file, write_file, delete_file):
        registry.register(ToolSpec(name=fn.__name__, description=fn.__doc__ or "", fn=fn))
    registry.register(
        ToolSpec(
            name="run_tests", description=run_tests.__doc__ or "", fn=run_tests,
            timeout_seconds=120,
        )
    )

    policy = PermissionPolicy(
        rules=[
            # Deny is evaluated first, so this beats the wildcard below.
            PermissionRule(action=Decision.DENY, pattern="delete_file"),
            PermissionRule(action=Decision.ALLOW, pattern="*"),
        ]
    )
    return Harness(registry, policy, executor=LocalExecutor(), workspace=str(workspace))


def run_fixit(model_spec: str, workspace: Path | None = None) -> dict:
    """Run the agent over a freshly broken project. Returns the outcome."""
    workspace = make_workspace(workspace)
    harness = build_harness(workspace)
    trace = TraceRecorder(workspace / "trace.jsonl")
    ctx = RunContext(
        run_id="fixit",
        graph="agent_fixit",
        meter=BudgetMeter(Budget(max_iterations=60, max_tokens=120_000, max_seconds=300)),
    )
    agent = AgentNode(
        model=get_model(model_spec, temperature=0, max_tokens=2000),
        harness=harness,
        max_iterations=12,
        trace=trace,
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.run(TASK, ctx)
    verification = run_pytest(workspace)
    return {
        "workspace": workspace,
        "visible_tools": [t.name for t in harness.visible_tools()],
        "result": result,
        "tests_pass": verification.returncode == 0,
        "source": (workspace / "calc.py").read_text(encoding="utf-8").strip(),
        "tokens": ctx.meter.tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openrouter/anthropic/claude-haiku-4.5")
    args = parser.parse_args(argv)

    outcome = run_fixit(args.model)
    result = outcome["result"]

    print(f"workspace : {outcome['workspace']}")
    print(f"tools     : {outcome['visible_tools']}")
    print(f"denied    : delete_file — offered to the model: "
          f"{'delete_file' in outcome['visible_tools']}")
    print(f"\nstopped   : {result.termination_reason}")
    print(f"turns     : {result.iterations}   tool calls: {len(result.tool_calls)}")
    for call in result.tool_calls:
        print(f"   {call.status.value:<18} {call.tool}")
    print(f"\ncalc.py now:\n  {outcome['source']}")
    print(f"\nindependent test run: {'PASS' if outcome['tests_pass'] else 'FAIL'}")
    print(f"tokens metered: {outcome['tokens']}")
    return 0 if outcome["tests_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
