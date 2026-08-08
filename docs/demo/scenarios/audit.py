"""Demo 3 — reading a finished run back out of its trace.

The claim this checks is the one about the file: the dashboard cannot disagree
with the audit trail because they are the same file. So every command here is
pointed at one `trace.jsonl` and nothing else — the diagram, the bill, the
reconstruction of state at a chosen step, and the comparison of two runs all
come out of it.

The run being audited is produced first, free and scripted, so this recording
needs no key and reproduces byte for byte.
"""

from __future__ import annotations

import os

WORKDIR = os.environ.get("GRAPHARC_DEMO_WORKDIR", "/tmp/grapharc-demo-audit")
TITLE = "grapharc — the run, read back"
SUBTITLE = "one JSONL file answers every question about what happened"

TRACE = "runs/audit.jsonl"

STEPS = [
    {
        "run": (
            "grapharc plan 'investigate the checkout outage' --scripted --go "
            f"--trace {TRACE} --run-id run-a"
        ),
        "caption": "one governed run, written to one file",
        "hold": 4.0,
    },
    {
        # Same goal, tighter ceiling. The worst case is priced during admission,
        # so this is refused *before* a node runs rather than dying partway —
        # which is the difference `diff` shows at the end.
        "run": (
            "grapharc plan 'investigate the checkout outage' --scripted --go "
            f"--trace {TRACE} --run-id run-b --max-tokens 1700"
        ),
        "caption": "the same goal under a tighter budget: priced, refused, nothing ran",
        "expect_exit": 1,
        "hold": 5.0,
    },
    {
        "run": f"grapharc trace {TRACE}",
        "caption": "what is in this file: every run, every phase",
        "hold": 4.5,
    },
    {
        "run": f"grapharc replay {TRACE} run-a",
        "caption": "the run reconstructed — node by node, from the events alone",
        "hold": 5.0,
    },
    {
        "run": f"grapharc diff {TRACE} run-a run-b",
        "caption": "two runs compared — three nodes against none, and why",
        # Exit 1 means "they differ", as `diff(1)` has always meant.
        "expect_exit": 1,
        "hold": 6.0,
    },
]


def setup(workdir) -> None:
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
