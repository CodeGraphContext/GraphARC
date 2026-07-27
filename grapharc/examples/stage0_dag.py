"""Stage 0 — a deterministic DAG. No LLMs.

load a document -> split it -> count terms -> write a report.

The point is to earn the mechanics before adding probabilistic nodes:
typed state, declared writes, trace recording, checkpointing — and the gate:
kill the process mid-write, restart, and the report exists exactly once
(atomic tmp-file + rename, resume from the last checkpoint).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import END, START, CompiledGraphARC, GraphARC
from grapharc.runtime.state import GraphARCState

# Test injection point: when True, the report node crashes once after writing
# the tmp file but before the atomic rename, then clears itself.
_CRASH_BEFORE_RENAME = False


def crash_once_before_rename() -> None:
    global _CRASH_BEFORE_RENAME
    _CRASH_BEFORE_RENAME = True


class Stage0State(GraphARCState):
    doc_path: str
    report_path: str
    text: str = ""
    chunks: list[str] = []
    counts: dict[str, int] = {}
    report_written: bool = False


def load(state: Stage0State) -> dict:
    return {"text": Path(state.doc_path).read_text(encoding="utf-8")}


def split(state: Stage0State) -> dict:
    chunks = [c.strip() for c in state.text.split("\n\n") if c.strip()]
    return {"chunks": chunks}


def count(state: Stage0State) -> dict:
    words = re.findall(r"[a-zA-Z']+", " ".join(state.chunks).lower())
    top = Counter(words).most_common(10)
    return {"counts": dict(top)}


def report(state: Stage0State) -> dict:
    global _CRASH_BEFORE_RENAME
    target = Path(state.report_path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    lines = [f"# Term report for {state.doc_path}", ""]
    lines += [f"- {term}: {n}" for term, n in state.counts.items()]
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if _CRASH_BEFORE_RENAME:
        _CRASH_BEFORE_RENAME = False
        raise RuntimeError("injected crash: killed after tmp write, before atomic rename")
    os.replace(tmp, target)
    return {"report_written": True}


def build_stage0(
    *, trace: TraceRecorder | None = None, checkpointer=None
) -> CompiledGraphARC:
    g = GraphARC(Stage0State, name="stage0_dag", dag=True, trace=trace)
    g.add_node("load", load, writes={"text"})
    g.add_node("split", split, writes={"chunks"})
    g.add_node("count", count, writes={"counts"})
    g.add_node("report", report, writes={"report_written"})
    g.add_edge(START, "load")
    g.add_edge("load", "split")
    g.add_edge("split", "count")
    g.add_edge("count", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


DEMO_DOC = """GraphARC is a graph engineering toolkit.

Graphs coordinate loops. Loops need budgets, budgets need meters.

Every edge means something, every path can be explained.
"""
