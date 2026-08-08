"""Demo 1 — the admission gate refusing, then admitting.

The project's central claim, on the cheapest possible path: no model, no key,
no spend. The scripted planner's first proposal names a kind the policy denies,
so round 1 is refused *before anything runs*; the refusal is structured
feedback, the planner replans against it, and round 2 goes through the same
checker and executes.

Everything here is deterministic, so this recording is reproducible byte for
byte by anyone who checks out the repo.
"""

from __future__ import annotations

import os

WORKDIR = os.environ.get("GRAPHARC_DEMO_WORKDIR", "/tmp/grapharc-demo-gate")
TITLE = "grapharc — the admission gate"
SUBTITLE = "a model proposes; a deterministic checker decides"

#: Named rather than generated, so `viz` and `metrics` below can be written
#: down. A reused id in one trace is refused by the CLI, which is why `setup`
#: clears the directory.
RUN = "gate-demo"
TRACE = "runs/gate.jsonl"

STEPS = [
    {
        "run": (
            "grapharc plan 'investigate the checkout outage' "
            f"--scripted --go --trace {TRACE} --run-id {RUN}"
        ),
        "caption": "round 1 wanted to deploy and never ran; round 2 replanned and executed",
        "hold": 6.0,
    },
    {
        "run": f"grapharc viz {TRACE} {RUN}",
        "caption": "the graph that actually ran, drawn from the trace",
        "hold": 4.5,
    },
    {
        "run": f"grapharc metrics {TRACE} {RUN}",
        "caption": "the bill, per node — read out of the same file as the diagram",
        "hold": 5.0,
    },
]


def setup(workdir) -> None:
    """A clean directory: the refusal must not be a leftover from last time."""
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
