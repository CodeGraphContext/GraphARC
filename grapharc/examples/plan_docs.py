"""A planning registry whose kinds can actually read — ROADMAP §12.1's sequel.

`plan_incident` proves the governance; its node bodies are stubs, and a goal
like "summarise the docs in this workspace" gets either an honest negative or
a hollow success, depending on how vague the goal is. This registry closes
that gap for one bounded job: **reading documentation under the current
working directory and reporting what is there.** Four kinds:

    survey     list the documentation files under cwd
    read       read them and record an excerpt of each
    summarise  one note distilling title + first paragraph per file
    propose    the model authors a recommendation from what was read

The first three are deterministic operator code. The model still does the
*planning* — which kinds, in what order, replanning on refusal — and those
bodies never call it, so a scripted run produces real file content and spends
nothing. `propose` is the one deliberate exception: its body hands the notes
the deterministic kinds collected to the model and records what comes back,
labelled as the model's. With no real model behind the run it says so in its
note instead of pretending — a scripted planner cannot author a proposal, and
the note is honest about which kind of run this was.

The confinement matters more than the capability: bodies read, never write,
and only under `Path.cwd()` — which, launched from the Slack bot, is the
bot's working directory. A proposal cannot steer them elsewhere because a
proposal names kinds and carries no arguments; there is no path field to
inject (the gap issue #10 tracks does not open here because these bodies
take no arguments at all).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from grapharc.harness.permissions import Decision
from grapharc.planner import CostEstimate, EdgePolicy, EdgeRule, NodeRegistry, NodeSpec

#: Read at most this many files, this much of each. Documentation, not a dump.
MAX_FILES = 20
MAX_CHARS = 40_000
_SUFFIXES = (".md", ".txt", ".rst")
_SKIP_DIRS = {".git", ".grapharc", ".venv", "__pycache__", "node_modules"}


class DocsState(BaseModel):
    """`notes` is deliberately the whole record: the loop's goal check reads it."""

    goal: str = ""
    notes: list[str] = []


def _docs_files(root: Path) -> list[Path]:
    """Every documentation file under `root`, and nothing outside it."""
    found = []
    for path in sorted(root.rglob("*")):
        if len(found) >= MAX_FILES:
            break
        if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        # Belt and braces: rglob cannot leave root, but a symlink can point
        # anywhere. Resolve and check before a single byte is read.
        if not path.resolve().is_relative_to(root):
            continue
        found.append(path)
    return found


def _excerpt(path: Path, root: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")[:MAX_CHARS]
    lines = [line.strip() for line in text.splitlines()]
    title = next((line.lstrip("# ") for line in lines if line), path.name)
    body = next((line for line in lines if line and not line.startswith("#")), "")
    return f"{path.relative_to(root)}: {title}" + (f" — {body[:160]}" if body else "")


def _survey_body(state: DocsState) -> dict:
    root = Path.cwd().resolve()
    files = _docs_files(root)
    listing = ", ".join(str(f.relative_to(root)) for f in files) or "none found"
    return {"notes": [*state.notes, f"survey: {len(files)} documentation file(s): {listing}"]}


def _read_body(state: DocsState) -> dict:
    root = Path.cwd().resolve()
    notes = [f"read {f.relative_to(root)}: {_excerpt(f, root)}" for f in _docs_files(root)]
    return {"notes": [*state.notes, *(notes or ["read: nothing to read"])]}


def _summarise_body(state: DocsState) -> dict:
    root = Path.cwd().resolve()
    files = _docs_files(root)
    if not files:
        summary = "summary: no documentation files under the working directory"
    else:
        parts = "; ".join(_excerpt(f, root) for f in files[:10])
        summary = f"summary of {len(files)} file(s): {parts}"
    return {"notes": [*state.notes, summary]}


def _propose_body_for(model: Any) -> Any:
    """The one body with a model in it, built per-run so it closes over the
    run's own backend rather than importing one.

    `model` is whatever drove the planner. A scripted planner's model cannot
    author anything (invoking it would consume the planner's own scripted
    replies), so anything without a usable `invoke` — or a call that fails —
    becomes an honest note instead of an exception: the deterministic notes
    already in state are worth keeping even when the proposal step cannot run.
    """

    def body(state: DocsState) -> dict:
        evidence = "\n".join(state.notes)
        real = model is not None and not isinstance(model, _scripted_type())
        if not real:
            note = (
                "propose: needs a real model (--model); no proposal was "
                "authored — the notes above are deterministic reads only"
            )
            return {"notes": [*state.notes, note]}
        try:
            reply = model.invoke(
                "You are the propose step of a documentation-review plan. "
                f"The goal: {state.goal!r}. The evidence collected so far:\n"
                f"{evidence}\n\n"
                "Write one concise, concrete recommendation that satisfies "
                "the goal, grounded only in the evidence above."
            )
            content = getattr(reply, "content", reply)
            note = f"proposal (model-authored): {str(content).strip()[:2000]}"
        except Exception as exc:  # noqa: BLE001 — a failed proposal is a note, not a crash
            note = f"propose: model call failed ({exc}); deterministic notes stand"
        return {"notes": [*state.notes, note]}

    body.writes = {"notes"}
    return body


def _scripted_type() -> type:
    from grapharc.testing import ScriptedChatModel

    return ScriptedChatModel


_BODIES = {"survey": _survey_body, "read": _read_body, "summarise": _summarise_body}


def _factory_for(model: Any):
    def factory(build: Any) -> Any:
        # `build` is the materialiser's NodeBuild: `name` is the instance the
        # planner chose ("readme_survey"), `kind` is what the registry
        # licensed. Behaviour keys on the kind; the name is the planner's.
        if build.kind == "propose":
            return _propose_body_for(model)
        body = _BODIES[build.kind]
        body.writes = {"notes"}
        return body

    return factory


WRITES: dict[str, set[str]] = {kind: {"notes"} for kind in (*_BODIES, "propose")}


def build_registry(model: Any = None) -> NodeRegistry:
    """Three read-only kinds and one that writes what the model recommends.

    Accepting `model` is the CLI's contract: a registry factory with a
    positional parameter is handed the run's backend (`resolve_registry` in
    `grapharc.cli.plan`), which is how `propose` speaks with the same model
    that planned — and stays honest when that model is scripted.
    """
    factory = _factory_for(model)
    return NodeRegistry(
        [
            NodeSpec(
                name="survey",
                description="list the documentation files under the working directory",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=300),
            ),
            NodeSpec(
                name="read",
                description="read each documentation file and record an excerpt",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=1500),
            ),
            NodeSpec(
                name="summarise",
                description="distil what was read into one summary note",
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=800),
            ),
            NodeSpec(
                name="propose",
                description=(
                    "author a recommendation from the collected notes; needs a real model"
                ),
                factory=factory,
                worst_case=CostEstimate(iterations=1, tokens=2500),
            ),
        ]
    )


def default_edge_policy() -> EdgePolicy:
    """Allow every transition: nothing here mutates, so nothing needs denying."""
    return EdgePolicy(rules=(EdgeRule(action=Decision.ALLOW),))


def scripted_planner_replies() -> list[str]:
    """One reply: survey → read → summarise → propose. Read by `grapharc plan`
    when no `--model` is given, so the free path exercises the same registry
    the paid one does — including the `propose` note honestly declining to
    author without a real model. In an empty directory the chain still yields
    enough notes for the loop's goal check either way."""
    import json

    from grapharc.runtime.graph import END, START

    chain = ["survey", "read", "summarise", "propose"]
    endpoints = [START, *chain, END]
    return [
        json.dumps(
            {
                "nodes": [{"name": kind} for kind in chain],
                "edges": [
                    {"source": a, "target": b}
                    for a, b in zip(endpoints, endpoints[1:], strict=False)
                ],
            }
        )
    ]


STATE_SCHEMA = DocsState

#: Nothing in this registry writes outside run state, so the policy generator
#: has nothing to deny — and saying so explicitly beats being defaulted.
MUTATING_KINDS: tuple[str, ...] = ()

__all__ = [
    "MUTATING_KINDS",
    "STATE_SCHEMA",
    "WRITES",
    "DocsState",
    "build_registry",
    "default_edge_policy",
    "scripted_planner_replies",
]
