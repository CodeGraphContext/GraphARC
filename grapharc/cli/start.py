"""`grapharc start` — the guided tour, and the bare-invocation orientation.

The `-h` text is a contract document (flags, exit codes, colour policy); this
is the onboarding one, written for someone who installed five minutes ago and
knows none of the project's words. Every term is defined where it is used, in
plain language — a tour that explains jargon with more jargon teaches nothing,
which was the first complaint a real user filed against the first version of
this text.

Text follows the style contract — same bytes on a terminal and in a pipe,
colour the only difference — so it joins the style tests like any command.
"""

from __future__ import annotations

from grapharc.cli import style
from grapharc.cli.output import EXIT_OK, emit


def orientation() -> list[str]:
    """The short block bare `grapharc` prints: what this is, where to begin."""
    return [
        "grapharc — you give a goal in English; a language model turns it into",
        "a small workflow; grapharc checks that workflow against rules you set",
        "and only then runs it, recording every step to a file you can watch",
        "live in a browser and replay afterwards.",
        "",
        f"  {style.accent('grapharc start')}     the guided tour (ten minutes, free)",
        f"  {style.accent('grapharc init')}      create the starter files in this directory",
        f"  {style.accent('grapharc help')}      every command and flag",
    ]


_WORDS = [
    (
        "node",
        "one box of work: read some files, write a report, apply a change",
    ),
    (
        "edge",
        "one arrow between boxes: which box is allowed to run after which",
    ),
    (
        "graph",
        "the whole workflow — boxes plus arrows — sharing one typed state",
    ),
    (
        "registry",
        "a plain Python file listing the KINDS of box a model may use. The "
        "model only picks from this menu; it can never write code of its "
        "own. You do not write this file from scratch: `grapharc init` "
        "creates a working, heavily-commented one (registry.py) to edit",
    ),
    (
        "plan",
        "PLANS ONLY: a model proposes the workflow, admission checks it, and "
        "the plan is saved — nothing executes. Needs --model, or --scripted "
        "for a free rehearsal with canned replies instead of AI",
    ),
    (
        "go",
        "EXECUTES: bare `go` runs the newest saved plan (`go <run-dir>` for a "
        "specific one); `go \"a goal\"` plans and executes in one shot. So "
        "does `plan --go`",
    ),
    (
        "admission",
        "the checker: before ANY box runs, deterministic code tests the "
        "proposed workflow against your registry, rules and budget, and a "
        "refusal comes back as a written reason the model must fix",
    ),
    (
        "approval",
        "plan → look at it → go IS the approval. For one-shot runs, "
        "--approve pauses mid-run until `grapharc approve <run-dir>` answers",
    ),
    (
        "live view",
        "a browser page drawing the run as it happens: violet boxes waiting "
        "for your approval, amber while running, green when done — with "
        "your goal in the header and each box's token bill on the box",
    ),
]

_PATH = [
    (
        "1",
        "grapharc init",
        "creates three things here: registry.py (the menu of box-kinds — "
        "open it, every section is explained), grapharc.toml (saved "
        "settings), and .grapharc/runs/ (where each run's record lands)",
    ),
    (
        "2",
        "grapharc serve --live-root .grapharc/runs",
        "in a second terminal — this is the browser page's server "
        "(pip install 'grapharc[server]' if it says the extra is missing)",
    ),
    (
        "3",
        'grapharc plan "review what is in this directory" --scripted',
        "plans and STOPS: the proposed boxes are saved and drawn in the "
        "browser, nothing has run. --scripted says it out loud: no AI here — "
        "the planner replies are canned text from your registry.py. It "
        "prints  watch : http://127.0.0.1:8000/…  — open that link",
    ),
    (
        "4",
        "grapharc go",
        "executes the plan you just looked at; the page goes live as the "
        "boxes run, and when it finishes, drag the slider to replay any "
        "moment",
    ),
]

_REAL = [
    (
        "check",
        "grapharc models --check   — which model backends this machine can use",
    ),
    (
        "exact tag",
        "for local models the name must be EXACTLY what `ollama list` shows: "
        "ollama/qwen3:8b works; ollama/qwen3:8 is a different, missing model "
        "and the run stops immediately saying so",
    ),
    (
        "go",
        'grapharc go "summarize how this repo is laid out" '
        "--model ollama/qwen3:8b --model-arg temperature=0",
    ),
    (
        "author",
        "open registry.py and make it yours: rename the kinds, rewrite the "
        "bodies, grow the State — the file explains each part where it sits",
    ),
    (
        "policy",
        "the first --model run writes .grapharc/generated-policy.*.toml — "
        "the rules it guessed from your goal. Read it; edit it; it is yours",
    ),
]


def start(*, as_json: bool = False) -> int:
    """Print the guided tour. The payload mirrors the prose for `--json`."""
    lines: list[str] = [
        "grapharc, in one paragraph",
        style.dim(
            "  You give a goal in English. A language model turns it into a small"
        ),
        style.dim(
            "  workflow of boxes and arrows. grapharc runs that workflow only after"
        ),
        style.dim(
            "  deterministic checks pass — and, if you ask, only after you say yes —"
        ),
        style.dim(
            "  recording every step to a file you can watch live and replay."
        ),
        "",
        "the words you will meet",
    ]
    for label, text in _WORDS:
        lines.append(f"  {style.accent(f'{label:<10}')}{style.dim(text)}")
    lines += ["", "try it in ten minutes (free — nothing here calls a paid model)"]
    for number, command, note in _PATH:
        lines.append(f"  {number}  {style.accent(command)}")
        lines.append(f"       {style.dim(note)}")
    lines += ["", "do it for real (a real model plans; real tools do the work)"]
    for label, text in _REAL:
        lines.append(f"  {style.accent(f'{label:<10}')}{style.dim(text)}")
    lines += ["", style.dim("the long version: README.md · docs/cookbook/")]

    payload = {
        "ok": True,
        "command": "start",
        "words": dict(_WORDS),
        "path": [{"step": n, "command": c, "note": note} for n, c, note in _PATH],
        "going_real": dict(_REAL),
        "docs": ["README.md", "docs/cookbook/"],
    }
    emit(payload, lines, as_json=as_json)
    return EXIT_OK


__all__ = ["orientation", "start"]
