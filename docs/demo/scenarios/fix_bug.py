"""Demo 2 — GraphARC fixing a real bug in GraphARC, under its own gate.

The bug is genuine and was on this project's backlog: `mcp.driver.graph_status`
read a running trace with the strict reader, so the supervised agent's own
status tool raised `TraceReadError` on a half-written line — at exactly the
moment it is meant to be useful, while the run it is reporting on is still
writing. `tests/test_torn_trace_read.py` is the bug report, in the form that
can be checked.

What the recording shows, in order: the red test; this repository's standing
rule that an agent does not edit source unattended; a plan made under that rule
(which routes around `apply_change` and says so, and is `mutating: false`); a
human amending the rule for this one piece of work; the plan made *after* that,
which now contains `apply_change`; the execution; the green test; the diff.

The agent is Claude Code, delegated to by every stdlib agent kind because the
Claude CLI has no tool-calling wire format. It is spending a real subscription
and editing real files — in a **copy** of the repository, so a demo cannot
damage the tree it is a demo of. The fix it produces is inspected afterwards
and landed deliberately, by a person, in a separate commit.
"""

from __future__ import annotations

import os
import subprocess

WORKDIR = os.environ["GRAPHARC_DEMO_WORKDIR"]  # a copy of the repo; see README
TITLE = "grapharc — fixing a bug in grapharc"
SUBTITLE = "the gate decides what the agent may even propose"

TEST = "tests/test_torn_trace_read.py"
GOAL = (
    f"{TEST} fails: graph_status in grapharc/mcp/driver.py raises TraceReadError "
    "on a half-written trace line. Fix the source so the test passes."
)
#: `--trace` is not decoration here: it puts `plan.json` in `runs/`, which is
#: what makes `grapharc go runs` resolve to *this* plan. Without it the plan
#: lands under `.grapharc/runs/<stamp>/` and `go runs` reads "runs" as a
#: goal string — planning something unrelated and executing that instead.
COMMON = (
    "--model claude-cli --registry grapharc.stdlib:build_registry "
    "--trace runs/plan.jsonl"
)
#: The planner's own words for why it proposed what it proposed. Worth showing
#: rather than paraphrasing: under the deny it says out loud that it cannot
#: reach `apply_change`, which is the demo's whole point.
#: The delegation warning is long, correct, and printed once per agent node —
#: it belongs in a log, not across three frames of a video. Dropped from the
#: recording only; nothing suppresses it in a real run.
QUIET = "grep -viE 'warning|node = '"
RATIONALE = (
    "python -c \"import json;d=json.load(open('runs/plan.json'));"
    "print('kinds     :', [n['kind'] for n in d['proposal']['nodes']]);"
    "print('mutating  :', d['mutating']);"
    "print('rationale :', d['proposal']['rationale'])\""
)

STEPS = [
    {
        "run": f"python -m pytest {TEST} -q --no-header 2>&1 | tail -6",
        "caption": "the bug, as a test: graph_status crashes on a trace being written",
        "hold": 5.0,
    },
    {
        "run": "cat policy.toml",
        "caption": "this repository's standing rule for automated planners",
        "hold": 6.0,
    },
    {
        "run": (
            f"grapharc plan '{GOAL}' {COMMON} --policy policy.toml "
            f"--max-rounds 2 --run-id under-deny 2>&1 | {QUIET}"
        ),
        "caption": "planned under the deny — admitted, but look at how many nodes",
        "timeout": 600,
        "hold": 6.0,
    },
    {
        "run": RATIONALE,
        "caption": "the planner says it: it cannot reach apply_change, so it only investigates",
        "hold": 7.0,
    },
    {
        "run": "cat policy-approved.toml",
        "caption": "a human reads the plan and amends the rule — for this work, in writing",
        "hold": 5.5,
    },
    {
        "run": (
            f"grapharc plan '{GOAL}' {COMMON} --policy policy-approved.toml "
            f"--max-rounds 2 --run-id after-approval 2>&1 | {QUIET}"
        ),
        "caption": "same goal, same model, different rule: five nodes now",
        "timeout": 600,
        "hold": 5.0,
    },
    {
        "run": RATIONALE,
        "caption": "apply_change is in the graph, and the plan is marked mutating",
        "hold": 6.0,
    },
    {
        "run": (
            f"grapharc go runs {COMMON} --policy policy-approved.toml "
            f"--max-rounds 2 2>&1 | {QUIET} | tail -18"
        ),
        "caption": "now it runs — Claude Code investigates, edits, verifies, reports",
        "timeout": 1500,
        "hold": 7.0,
    },
    {
        "run": "git --no-pager diff --stat && git --no-pager diff",
        "caption": "the change it made: the strict reader swapped for the tolerant one",
        "hold": 8.0,
    },
    {
        "run": f"python -m pytest {TEST} -q --no-header 2>&1 | tail -4",
        "caption": "green — the same test that opened this recording",
        "hold": 6.0,
    },
]


def setup(workdir) -> None:
    """Back to the committed baseline, so a re-record starts where this one did."""
    import shutil

    subprocess.run(["git", "-C", str(workdir), "checkout", "--", "."], check=False)
    for leftover in ("runs", ".grapharc"):
        shutil.rmtree(workdir / leftover, ignore_errors=True)
