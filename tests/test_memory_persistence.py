"""Durability proved across process boundaries, not across objects.

The Stage 6 gate used to pass with an in-process dict because its three "runs"
shared one Python object. Nothing in this file shares an object: every claim is
written by one interpreter and read back by a different one, identified by PID.
That is the test that would have caught the original overclaim.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from grapharc.memory import Claim, LadybugMemoryStore, MemoryStore, SQLiteMemoryStore
from grapharc.memory.store import ClaimStore

REPO_ROOT = Path(__file__).resolve().parents[1]

INTERFACE = ("add", "get", "supersede", "current", "history", "dead_ends", "all_claims")

_PRELUDE = """\
import json, os, sys, time
from grapharc.memory import Claim, MemoryStore, SQLiteMemoryStore

DB = sys.argv[1]
ARGS = sys.argv[2:]


def emit(**payload):
    payload["pid"] = os.getpid()
    sys.stdout.write("RESULT " + json.dumps(payload) + "\\n")

"""


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / f"{name}.py"
    script.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    return script


def _spawn(script: Path, db: Path, *args: str) -> subprocess.Popen:
    # PYTHONPATH rather than an installed package, so the child imports the
    # same working tree the test does even without an editable install.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(
        [sys.executable, str(script), str(db), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )


def _result(proc: subprocess.Popen) -> dict:
    out, err = proc.communicate(timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"child failed ({proc.returncode}):\n{err}")
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"child emitted no result:\nstdout={out}\nstderr={err}")


def run_child(tmp_path: Path, name: str, body: str, db: Path, *args: str) -> dict:
    """Run `body` in a separate interpreter and return its emitted payload."""
    payload = _result(_spawn(_write_script(tmp_path, name, body), db, *args))
    assert payload["pid"] != os.getpid(), "child ran in the test's own process"
    return payload


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "memory.sqlite"


def test_claims_written_by_one_process_are_read_by_another(tmp_path, db):
    writer = run_child(
        tmp_path,
        "writer",
        """
        store = SQLiteMemoryStore(DB)
        store.add(Claim(subject="GraphARC", predicate="orchestration runtime",
                        object="LangGraph", source="final-plan.md", run_id="run12"))
        store.close()
        emit(ok=True)
        """,
        db,
    )
    reader = run_child(
        tmp_path,
        "reader",
        """
        store = SQLiteMemoryStore(DB)
        facts = [(c.object, c.source, c.run_id, c.observed_at)
                 for c in store.current("grapharc")]
        emit(facts=facts, total=len(store.all_claims()))
        """,
        db,
    )

    assert reader["pid"] != writer["pid"]
    assert reader["total"] == 1
    (obj, source, run_id, observed_at), = reader["facts"]
    assert obj == "LangGraph"
    assert (source, run_id) == ("final-plan.md", "run12")  # provenance survived
    assert observed_at  # and so did the observation time


def test_the_in_memory_store_does_not_survive_the_process(tmp_path, db):
    """The control for the experiment: same script, undurable backend.

    MemoryStore is not broken — it is in-process by design. This pins down what
    the SQLite backend is actually adding, so nobody re-inherits the belief
    that a dict "persists across runs".
    """
    first = run_child(
        tmp_path,
        "mem_writer",
        """
        store = MemoryStore()
        store.add(Claim(subject="GraphARC", predicate="p", object="v", source="s"))
        emit(seen=len(store.current("GraphARC")))
        """,
        db,
    )
    second = run_child(
        tmp_path,
        "mem_reader",
        """
        store = MemoryStore()
        emit(seen=len(store.current("GraphARC")))
        """,
        db,
    )
    assert first["seen"] == 1
    assert second["seen"] == 0


def test_supersession_and_provenance_survive_the_process_boundary(tmp_path, db):
    writer = run_child(
        tmp_path,
        "correcting_writer",
        """
        store = SQLiteMemoryStore(DB)
        old = store.add(Claim(subject="GraphARC", predicate="memory backend",
                              object="a vector store", source="early-sketch.md",
                              run_id="run12"))
        new = store.supersede(old.id, Claim(
            subject="GraphARC", predicate="memory backend",
            object="a provenance graph", source="final-plan.md", run_id="run37"))
        store.close()
        emit(old_id=old.id, new_id=new.id, old_observed_at=old.observed_at)
        """,
        db,
    )

    reader = run_child(
        tmp_path,
        "correcting_reader",
        """
        store = SQLiteMemoryStore(DB)
        old = store.get(ARGS[0])
        emit(
            current=[c.object for c in store.current("GraphARC", "memory backend")],
            history=[c.object for c in store.history("GraphARC", "memory backend")],
            dead=[c.object for c in store.dead_ends("GraphARC")],
            old={"object": old.object, "source": old.source, "run_id": old.run_id,
                 "observed_at": old.observed_at, "superseded_by": old.superseded_by,
                 "superseded_at": old.superseded_at},
        )
        """,
        db,
        writer["old_id"],
    )

    assert reader["current"] == ["a provenance graph"]
    assert reader["history"] == ["a vector store", "a provenance graph"]
    assert reader["dead"] == ["a vector store"]
    old = reader["old"]
    assert old["superseded_by"] == writer["new_id"]
    assert old["superseded_at"] is not None
    # The corrected claim keeps its own provenance — it was not overwritten.
    assert old["run_id"] == "run12"
    assert old["source"] == "early-sketch.md"
    assert old["observed_at"] == writer["old_observed_at"]


def test_double_supersede_is_refused_in_a_later_process(tmp_path, db):
    """The refusal is a property of the stored data, not of in-memory state."""
    writer = run_child(
        tmp_path,
        "first_correction",
        """
        store = SQLiteMemoryStore(DB)
        old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
        store.supersede(old.id, Claim(subject="a", predicate="p", object="2", source="s"))
        store.close()
        emit(old_id=old.id)
        """,
        db,
    )
    second = run_child(
        tmp_path,
        "second_correction",
        """
        store = SQLiteMemoryStore(DB)
        loser = Claim(subject="a", predicate="p", object="3", source="s")
        try:
            store.supersede(ARGS[0], loser)
            emit(refused=False, error="", orphan=store.get(loser.id) is not None)
        except ValueError as exc:
            emit(refused=True, error=str(exc), orphan=store.get(loser.id) is not None)
        """,
        db,
        writer["old_id"],
    )
    assert second["refused"], second["error"]
    assert "already superseded" in second["error"]
    assert not second["orphan"]  # the rejected claim was not left behind

    store = SQLiteMemoryStore(db)
    assert [c.object for c in store.history("a", "p")] == ["1", "2"]
    store.close()


def test_unicode_entities_survive_the_process_boundary(tmp_path, db):
    """Normalization happens in Python; SQLite's own folding is ASCII-only, so
    a backend that matched entities in SQL would merge these on the way back."""
    run_child(
        tmp_path,
        "unicode_writer",
        """
        store = SQLiteMemoryStore(DB)
        store.add(Claim(subject="東京", predicate="is in", object="Japan", source="s"))
        store.add(Claim(subject="北京", predicate="is in", object="China", source="s"))
        store.add(Claim(subject="Müller", predicate="wrote", object="paper A", source="s"))
        store.add(Claim(subject="M ller", predicate="wrote", object="paper B", source="s"))
        store.add(Claim(subject="Neo4j", predicate="is a", object="graph db", source="s"))
        store.close()
        emit(ok=True)
        """,
        db,
    )
    reader = run_child(
        tmp_path,
        "unicode_reader",
        """
        store = SQLiteMemoryStore(DB)
        emit(
            tokyo=[c.object for c in store.current("東京")],
            beijing=[c.object for c in store.current("北京")],
            muller=[c.object for c in store.current("Müller")],
            stripped=[c.object for c in store.current("M ller")],
            shouty=[c.object for c in store.current("NEO4J")],
        )
        """,
        db,
    )
    assert reader["tokyo"] == ["Japan"]
    assert reader["beijing"] == ["China"]
    assert reader["muller"] == ["paper A"]
    assert reader["stripped"] == ["paper B"]
    assert reader["shouty"] == ["graph db"]


@pytest.mark.timeout(180)
def test_stage6_gate_holds_when_the_three_runs_are_three_processes(tmp_path, db):
    """The Stage 6 gate with nothing shared but a file path.

    Same claim as tests/test_stage6_gate.py — run #51 uses a fact from run #12
    and avoids what run #37 superseded — except each run is its own
    interpreter, so no Python object can be carrying the facts forward.
    """
    graph_prelude = """
        import json
        from grapharc.examples.stage6_memory import build_stage6
        from grapharc.testing import ScriptedChatModel
        store = SQLiteMemoryStore(DB)
    """

    run12 = run_child(
        tmp_path,
        "run12",
        graph_prelude
        + """
        model = ScriptedChatModel(responses=[
            json.dumps({"claims": [{"subject": "GraphARC",
                                    "predicate": "orchestration runtime",
                                    "object": "LangGraph"}]}),
            "GraphARC runs on LangGraph [final-plan.md].",
        ])
        result = build_stage6(model, store).invoke(
            {"entities": ["GraphARC"], "source_name": "final-plan.md",
             "source_text": "GraphARC's orchestration runtime is LangGraph."},
            run_id="run12")
        store.close()
        emit(new_claim_ids=result["new_claim_ids"], recalled=result["recalled"])
        """,
        db,
    )
    assert len(run12["new_claim_ids"]) == 1
    learned_id = run12["new_claim_ids"][0]
    assert "No prior knowledge" in run12["recalled"]  # nothing existed before it

    run37 = run_child(
        tmp_path,
        "run37",
        graph_prelude
        + """
        wrong = store.add(Claim(subject="GraphARC", predicate="memory backend",
                                object="a vector store", source="early-sketch.md",
                                run_id="run12"))
        store.supersede(wrong.id, Claim(
            subject="GraphARC", predicate="memory backend",
            object="a provenance graph", source="final-plan.md", run_id="run37"))
        store.close()
        emit(ok=True)
        """,
        db,
    )

    run51 = run_child(
        tmp_path,
        "run51",
        graph_prelude
        + """
        model = ScriptedChatModel(responses=[
            json.dumps({"claims": []}),
            "GraphARC runs on LangGraph and stores memory in a provenance graph.",
        ])
        result = build_stage6(model, store).invoke(
            {"entities": ["GraphARC"], "source_name": "notes.md",
             "source_text": "(nothing new)"},
            run_id="run51")
        emit(recalled=result["recalled"],
             avoided=result["avoided_dead_ends"],
             learned_run_id=store.get(ARGS[0]).run_id)
        """,
        db,
        learned_id,
    )

    assert len({run12["pid"], run37["pid"], run51["pid"], os.getpid()}) == 4

    # (a) run #51 uses a fact first learned by run #12, in another process
    assert "LangGraph" in run51["recalled"]
    assert run51["learned_run_id"] == "run12"
    # (b) it sees what run #37 superseded, and (c) is told not to re-derive it
    assert any("vector store" in d for d in run51["avoided"])
    assert "Superseded" in run51["recalled"]
    known_facts = run51["recalled"].split("Superseded")[0]
    assert "vector store" not in known_facts
    assert "provenance graph" in known_facts


def test_concurrent_writers_in_separate_processes_all_persist(tmp_path, db):
    """Processes writing at once must block on the write lock, not fail.

    They also open the database at the same instant, on a file that does not
    exist yet — the case that crashed the store on `PRAGMA journal_mode = WAL`
    before `_enable_wal` retried it.
    """
    run_ids = ["runA", "runB", "runC", "runD"]
    script = _write_script(
        tmp_path,
        "concurrent_writer",
        """
        run_id, start = ARGS[0], float(ARGS[1])
        while time.time() < start:
            time.sleep(0.001)
        store = SQLiteMemoryStore(DB)
        for i in range(25):
            store.add(Claim(subject="X", predicate=run_id + str(i), object=str(i),
                            source="s", run_id=run_id))
        store.close()
        emit(ok=True)
        """,
    )
    start = str(time.time() + 1.5)
    procs = [_spawn(script, db, run_id, start) for run_id in run_ids]
    for proc in procs:
        _result(proc)

    store = SQLiteMemoryStore(db)
    claims = store.all_claims()
    store.close()
    assert len(claims) == 25 * len(run_ids)
    assert sorted({c.run_id for c in claims}) == run_ids


def test_only_one_of_two_racing_processes_can_supersede(tmp_path, db):
    """Supersede reads then writes; without one write-locked transaction both
    racers would see an un-superseded claim and both would win."""
    store = SQLiteMemoryStore(db)
    old = store.add(Claim(subject="a", predicate="p", object="1", source="s"))
    store.close()

    script = _write_script(
        tmp_path,
        "racer",
        """
        old_id, start = ARGS[0], float(ARGS[1])
        store = SQLiteMemoryStore(DB)
        replacement = Claim(subject="a", predicate="p", object=ARGS[2], source="s")
        while time.time() < start:
            time.sleep(0.001)
        try:
            store.supersede(old_id, replacement)
            emit(won=True, error="", object=replacement.object)
        except ValueError as exc:
            emit(won=False, error=str(exc), object=replacement.object)
        """,
    )
    start = str(time.time() + 1.0)
    procs = [_spawn(script, db, old.id, start, obj) for obj in ("2", "3")]
    results = [_result(proc) for proc in procs]

    assert sum(r["won"] for r in results) == 1, results
    loser = next(r for r in results if not r["won"])
    winner = next(r for r in results if r["won"])
    assert "already superseded" in loser["error"]

    store = SQLiteMemoryStore(db)
    history = [c.object for c in store.history("a", "p")]
    store.close()
    # The loser's replacement was never written, not even as a loose claim.
    assert history == ["1", winner["object"]]


def test_concurrent_threads_share_one_store_safely(tmp_path, db):
    """One connection is shared across threads, so the lock has to hold."""
    store = SQLiteMemoryStore(db)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(25):
                store.add(
                    Claim(subject="X", predicate=f"t{n}-{i}", object=str(i), source="s")
                )
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store.all_claims()) == 200
    store.close()


def test_every_backend_satisfies_the_claim_store_protocol(db, tmp_path):
    assert isinstance(MemoryStore(), ClaimStore)
    store = SQLiteMemoryStore(db)
    assert isinstance(store, ClaimStore)
    store.close()
    pytest.importorskip("real_ladybug", reason="needs the ladybug extra")
    ladybug = LadybugMemoryStore(tmp_path / "memory.lbdb")
    assert isinstance(ladybug, ClaimStore)
    ladybug.close()


def test_the_three_backends_agree_on_every_interface_signature():
    """isinstance against a runtime protocol only checks names, so check the
    signatures too — a backend that renamed a parameter would still pass.

    `LadybugMemoryStore` is checked without its driver installed, which is the
    point of loading that driver in the constructor rather than at import: the
    signature contract is verifiable in an environment that cannot open a
    LadybugDB database at all."""
    for name in INTERFACE:
        protocol = inspect.signature(getattr(ClaimStore, name))
        assert inspect.signature(getattr(MemoryStore, name)) == protocol
        assert inspect.signature(getattr(SQLiteMemoryStore, name)) == protocol
        assert inspect.signature(getattr(LadybugMemoryStore, name)) == protocol
