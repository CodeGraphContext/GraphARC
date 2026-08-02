"""Executable gate for `docs/cookbook/06-serving-and-ops.md`.

The cookbook's only real promise is that every snippet in it runs exactly as
printed. This file enforces that mechanically rather than by review: each
```python block is extracted, executed, and diffed against the plain block that
follows it. A snippet that stops working — or an output block someone tidied by
hand — fails here with a diff.

The ```console transcripts are held to the same standard, marked the way
`tests/test_cookbook_models.py` marks its page's:

    <!-- verified: cli -->        every command re-run through a real shell and
                                  byte-compared (run ids mapped, wall clock masked)
    <!-- verified: cli varies -->  every command re-run and required to succeed;
                                  the output describes one machine, not compared
    <!-- needs-credentials -->    never executed — needs a live model

An unmarked transcript fails the suite, so a future edit cannot quietly put an
unverified command on the page. The transcripts once rotted exactly that way:
`/healthz` showed version 0.1.0 against an 0.1.1 tree, and `grapharc metrics`
had grown two fields the page did not show.

Three things are deliberately not left to the differ:

- **Non-deterministic fields are normalised, not ignored.** Approval request
  ids, child pids and node durations change on every run and the doc says so
  inline. `_NORMALISERS` lists exactly which spans are allowed to vary; every
  other character has to match.
- **Snippets are exec'd into a real module object registered in `sys.modules`.**
  A `Send` payload that is a Pydantic model is stored in the checkpoint by
  class path, so a class defined in a namespace no importer can find comes back
  from the checkpoint unusable and the fan-out snippet silently sends nothing.
  Running it as a module is what a reader gets when they save the snippet to a
  file, and it is what the test has to reproduce.
- **The two unpaired snippets are pinned by name.** One is the server registry
  module the shell transcript imports (checked by building it), the other is
  the real-model swap the doc says in words it did not run. A future edit
  cannot quietly add a third unexecuted snippet.

The section's own claims — an async graph refuses the sync entry point before
running anything, an unknown `goto` raises, a rejected node's body never runs,
a hold is per task and not per recipient, a sync-only checkpointer fails the
server — are pinned again below as direct assertions. The differ proves the
printed output is real; these prove it still means what the prose says.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc import __version__
from grapharc.observe.trace import TraceRecorder
from grapharc.runtime.graph import (
    END,
    START,
    AsyncNodeError,
    Command,
    GraphARC,
    GraphRoutingError,
)
from grapharc.runtime.state import GraphARCState
from grapharc.server import GraphRegistry as ServerRegistry
from grapharc.server import create_app
from grapharc.session import REGISTRY, SessionManager, SessionStatus
from grapharc.session.demo import GRAPH_NAME

DOC = Path(__file__).resolve().parents[1] / "docs" / "cookbook" / "06-serving-and-ops.md"
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)

#: Spans the doc says out loud will differ per run. Anything not listed here
#: has to match character for character.
_NORMALISERS = (
    # `... awaiting approval 'dc4259cee88f'; call decide() ...`
    (re.compile(r"awaiting approval '[0-9a-f]{12}'"), "awaiting approval '<id>'"),
    # `process A 479013 awaiting_approval ...` — child pids
    (re.compile(r"^(process [AB]) \d+", re.MULTILINE), r"\1 <pid>"),
    # `1 ok  draft (0.4ms, 7 tok)` — measured wall clock
    (re.compile(r"\(\d+\.\d+ms"), "(<t>ms"),
)


def _blocks() -> list[tuple[str, str, int]]:
    """(language, body, 1-based line number) for every fenced block, in order."""
    text = DOC.read_text(encoding="utf-8")
    return [
        (m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1)
        for m in FENCE.finditer(text)
    ]


def _snippets() -> tuple[list[dict], list[dict]]:
    """Split python blocks into (runnable, unpaired).

    A python block immediately followed by an unlabelled block is a snippet
    plus its expected stdout. One with anything else after it — prose, a
    ```console transcript, a table — is a fragment the doc does not claim to
    have run, and every one of those is accounted for by name below.
    """
    blocks = _blocks()
    runnable: list[dict] = []
    unpaired: list[dict] = []
    for index, (lang, body, line) in enumerate(blocks):
        if lang != "python":
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if following is not None and following[0] == "":
            runnable.append({"code": body, "expected": following[1], "line": line})
        else:
            unpaired.append({"code": body, "line": line})
    return runnable, unpaired


RUNNABLE, UNPAIRED = _snippets()


def _normalise(text: str) -> str:
    for pattern, replacement in _NORMALISERS:
        text = pattern.sub(replacement, text)
    return text.rstrip("\n")


def _exec_snippet(code: str, name: str) -> str:
    """Run one snippet as a module and return its stdout."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, f"{DOC}:{name}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolated_graph_registry():
    """Snippets register graphs by name; keep that out of the rest of the suite."""
    before = dict(REGISTRY._specs)
    yield
    REGISTRY._specs.clear()
    REGISTRY._specs.update(before)


# -- the differ ---------------------------------------------------------------


def test_the_section_exists_and_has_snippets():
    assert DOC.is_file()
    assert len(RUNNABLE) >= 15


@pytest.mark.parametrize("snippet", RUNNABLE, ids=[f"line{s['line']}" for s in RUNNABLE])
def test_every_snippet_prints_exactly_what_the_doc_says(snippet):
    printed = _exec_snippet(snippet["code"], f"cookbook_serving_L{snippet['line']}")
    assert _normalise(printed) == _normalise(snippet["expected"])


def test_the_unrun_snippets_are_the_two_the_doc_accounts_for():
    assert len(UNPAIRED) == 2
    registry_module, real_model = UNPAIRED
    assert "mygraphs.py" in registry_module["code"]
    assert "get_model(" in real_model["code"]
    compile(real_model["code"], "<fragment>", "exec")  # still has to be valid Python
    assert "**This snippet was not executed**" in DOC.read_text(encoding="utf-8")


def test_the_registry_module_the_shell_transcript_imports_actually_builds():
    """The one unpaired snippet that *can* be run is run, just not for its stdout."""
    code = UNPAIRED[0]["code"]
    module = types.ModuleType("cookbook_mygraphs")
    sys.modules["cookbook_mygraphs"] = module
    try:
        exec(compile(code, "<mygraphs.py>", "exec"), module.__dict__)
        registry = module.REGISTRY
        assert registry.names() == ["qa"]
        with TestClient(create_app(registry=registry)) as client:
            assert client.get("/healthz").json()["graphs"] == ["qa"]
    finally:
        sys.modules.pop("cookbook_mygraphs", None)


def test_no_snippet_reaches_a_live_backend():
    """The scripted double is the only model the executed snippets touch."""
    for snippet in RUNNABLE:
        assert "get_model(" not in snippet["code"]
        assert "ClaudeCodeCLIChatModel" not in snippet["code"]
        assert "OpenRouter" not in snippet["code"]


# -- the console transcripts --------------------------------------------------

_MARKER = re.compile(r"^<!--\s*(.*?)\s*-->$")
_CLI_MARKERS = {"verified: cli", "verified: cli varies", "needs-credentials"}
#: A run id as the CLI prints it — 12 hex digits, fresh per run.
_RUN_ID = re.compile(r"\b[0-9a-f]{12}\b")


def _console_blocks() -> list[dict]:
    """Console transcripts in order, each carrying the marker comment above it."""
    lines = DOC.read_text(encoding="utf-8").splitlines()
    transcripts: list[dict] = []
    for lang, body, line in _blocks():
        if lang != "console":
            continue
        marker = None
        for above in reversed(lines[: line - 1]):
            if above.strip():
                found = _MARKER.match(above.strip())
                marker = found.group(1) if found else None
                break
        transcripts.append({"body": body, "line": line, "marker": marker})
    return transcripts


def _steps(body: str) -> list[tuple[str, str]]:
    """(command, expected output) pairs; a trailing backslash continues a command."""
    raw: list[tuple[list[str], list[str]]] = []
    continuing = False
    for line in body.splitlines():
        if continuing:
            raw[-1][0].append(line)
            continuing = line.endswith("\\")
        elif line.startswith("$ "):
            raw.append(([line[2:]], []))
            continuing = line.endswith("\\")
        elif raw:
            raw[-1][1].append(line)
    return [("\n".join(cmd), "\n".join(out).strip("\n")) for cmd, out in raw]


CONSOLE = _console_blocks()
_CLI_EXACT = [b for b in CONSOLE if b["marker"] == "verified: cli"]
_CLI_VARIES = [
    b for b in CONSOLE
    if b["marker"] == "verified: cli varies" and "grapharc serve" not in b["body"]
]
_CLI_SERVE = [
    b for b in CONSOLE
    if b["marker"] == "verified: cli varies" and "grapharc serve" in b["body"]
]
_CLI_UNRUN = [b for b in CONSOLE if b["marker"] == "needs-credentials"]

#: Run ids the exact transcripts quote, in order of first appearance. They are
#: random per run, so before anything is compared each is remapped to the id
#: the re-run actually produced — same order, because the trace file appends.
_DOC_RUN_IDS = [
    match.group(0)
    for block in _CLI_EXACT
    for match in _RUN_ID.finditer(block["body"])
]
_DOC_RUN_IDS = list(dict.fromkeys(_DOC_RUN_IDS))

_CONSOLE_NORMALISERS = (
    # `duration_ms: 0.68` / `"duration_ms": 0.75,` — measured wall clock
    (re.compile(r'(duration_ms"?: )\d+\.\d+'), r"\1<t>"),
    # `8 ok  finish_target_met (0.0ms)` — replay's measured wall clock
    (re.compile(r"\(\d+\.\d+ms"), "(<t>ms"),
)


def _normalise_console(text: str) -> str:
    """Wall-clock spans masked; line ends rstripped, because `grapharc trace`
    pads its columns with trailing spaces and markdown files carry none."""
    for pattern, replacement in _CONSOLE_NORMALISERS:
        text = pattern.sub(replacement, text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def _substituted(text: str, mapping: dict[str, str]) -> str:
    for placeholder, real in mapping.items():
        text = text.replace(placeholder, real)
    return text


def _grapharc_on_path(bindir: Path) -> dict[str, str]:
    """An env whose `grapharc` is this interpreter's, wherever pytest runs from."""
    bindir.mkdir(parents=True, exist_ok=True)
    shim = bindir / "grapharc"
    shim.write_text(
        f'#!/usr/bin/env bash\nexec "{sys.executable}" -m grapharc.cli.main "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_console(command: str, workdir: Path, env: dict[str, str]):
    """Run one transcript command through a real shell, the way a reader would.

    Unlike the python snippets these need a shell: the transcripts use pipes
    (`| jq`, `| head`), an env-var prefix and `$?`. The script is the page's
    own text, not anything a caller supplies.
    """
    return subprocess.run(  # noqa: S603 — fixed argv; the script is the doc's own text
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=180,
    )


def _actual_run_ids(workdir: Path) -> list[str]:
    """Run ids in the session's trace file, in order of first appearance."""
    path = workdir / "trace.jsonl"  # the name every transcript command uses
    ids: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            run_id = json.loads(line)["run_id"]
            if run_id not in ids:
                ids.append(run_id)
    return ids


_EXACT_STEPS = [
    (block["line"], index)
    for block in _CLI_EXACT
    for index in range(len(_steps(block["body"])))
]


@pytest.fixture(scope="module")
def console_session(tmp_path_factory):
    """Every exact transcript's commands, run once, in order, in one directory.

    The transcripts are one shell session: the `--json` block reads the trace
    file the tour wrote, and the diff command names a run id that only exists
    because an earlier command was redirected to `/dev/null`. So the commands
    run in document order and share a directory, and the run-id map is refreshed
    from the trace file before and after each one.
    """
    if shutil.which("jq") is None:
        pytest.skip("the transcript pipes through jq, which is not on PATH")
    workdir = tmp_path_factory.mktemp("console-session")
    env = _grapharc_on_path(workdir / "bin")
    results: dict[tuple[int, int], dict] = {}
    for block in _CLI_EXACT:
        for index, (command, expected) in enumerate(_steps(block["body"])):
            mapping = dict(zip(_DOC_RUN_IDS, _actual_run_ids(workdir), strict=False))
            proc = _run_console(_substituted(command, mapping), workdir, env)
            mapping = dict(zip(_DOC_RUN_IDS, _actual_run_ids(workdir), strict=False))
            results[(block["line"], index)] = {
                "command": command,
                "expected": _substituted(expected, mapping),
                "proc": proc,
            }
    return results


def test_every_console_transcript_is_marked():
    """An unmarked transcript would be an unverified claim on the page."""
    unmarked = [b["line"] for b in CONSOLE if b["marker"] not in _CLI_MARKERS]
    assert not unmarked, f"add a marker comment above the ```console block(s) at {unmarked}"


def test_the_console_gate_selects_what_the_page_promises():
    """Guards against a parser change that silently selects nothing."""
    assert len(_CLI_EXACT) == 3  # the CLI tour, `--json`, `models <spec>`
    assert len(_CLI_SERVE) == 1
    assert len(_CLI_VARIES) == 1  # `models --check`
    assert len(_CLI_UNRUN) == 1  # `grapharc agent`
    assert len(_EXACT_STEPS) >= 10


def test_console_transcripts_only_launch_the_grapharc_cli_and_curl():
    """Pipes to jq/head/tail only shape output; nothing else gets executed."""
    for block in CONSOLE:
        for command, _ in _steps(block["body"]):
            first = next(
                token
                for token in shlex.split(command.replace("\\\n", " "), comments=True)
                if "=" not in token  # skip env-var prefixes like PYTHONPATH=.
            )
            assert first in {"grapharc", "curl"}, command


def test_the_unrun_transcript_is_the_agent_command_and_shows_no_output():
    """A command nobody ran must not display output it never printed."""
    (block,) = _CLI_UNRUN
    ((command, output),) = _steps(block["body"])
    assert command.startswith("grapharc agent ")
    assert output == ""


def test_the_health_version_the_serve_transcript_shows_is_the_real_one():
    """This line once rotted — 0.1.0 on the page against an 0.1.1 tree — and it
    sits in a varies-marked transcript the differ does not reach, so it is
    pinned to the package version by name."""
    expected = f'{{"status":"ok","version":"{__version__}","graphs":["qa"]}}'
    assert expected in DOC.read_text(encoding="utf-8")


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("line", "index"), _EXACT_STEPS, ids=[f"line{li}-step{i}" for li, i in _EXACT_STEPS]
)
def test_console_command_prints_exactly_what_the_page_shows(console_session, line, index):
    step = console_session[(line, index)]
    proc = step["proc"]
    assert proc.returncode == 0, f"$ {step['command']}\n{proc.stdout}{proc.stderr}"
    printed = _normalise_console(proc.stdout)
    expected = _normalise_console(step["expected"])
    if expected.startswith("...\n"):
        # The page elides the head of this output; the tail is still exact.
        assert printed.endswith(expected[4:]), f"$ {step['command']}\n{proc.stdout}"
    else:
        assert printed == expected, f"$ {step['command']}\n{proc.stdout}"


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("line", "index"),
    [(b["line"], i) for b in _CLI_VARIES for i in range(len(_steps(b["body"])))],
)
def test_console_command_with_machine_dependent_output_still_runs(line, index, tmp_path):
    """`models --check` reports this machine — and exits 1 when no real backend
    is usable here — so only its shape is asserted, as in 02-models.md."""
    block = next(b for b in _CLI_VARIES if b["line"] == line)
    command, _ = _steps(block["body"])[index]
    proc = _run_console(command, tmp_path, _grapharc_on_path(tmp_path / "bin"))
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.stdout.strip()


@pytest.mark.timeout(180)
def test_the_serve_transcript_runs_against_a_real_server(tmp_path):
    """The `serve` transcript, replayed end to end.

    The registry module is the page's own `mygraphs.py`, the server is started
    with the page's own command (on a free port — 8124 is nobody's to claim in
    a test suite), and every curl in the block must succeed against it. The
    outputs are machine-dependent — ids, timestamps, durations — so they are
    not byte-compared; the banner and the `/healthz` body are deterministic and
    are.
    """
    if shutil.which("curl") is None:
        pytest.skip("the transcript uses curl, which is not on PATH")
    (block,) = _CLI_SERVE
    serve_step, health_step, post_step, get_step, trace_step = _steps(block["body"])
    assert serve_step[0].startswith("PYTHONPATH=. grapharc serve ")

    (tmp_path / "mygraphs.py").write_text(UNPAIRED[0]["code"], encoding="utf-8")
    env = _grapharc_on_path(tmp_path / "bin")
    env["PYTHONUNBUFFERED"] = "1"  # the banner has to cross the pipe before the block

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        mapping = {"8124": str(probe.getsockname()[1])}

    server = subprocess.Popen(  # noqa: S603 — fixed argv; the script is the doc's own text
        ["bash", "-c", _substituted(serve_step[0], mapping)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # uvicorn's own request log
        stdin=subprocess.DEVNULL,
    )
    try:
        banner = [server.stdout.readline().rstrip("\n") for _ in range(3)]
        assert banner == _substituted(serve_step[1], mapping).splitlines()

        # The banner is printed *before* the server binds — the page says so —
        # so retry the first curl until the socket opens.
        deadline = time.monotonic() + 30
        while True:
            health = _run_console(_substituted(health_step[0], mapping), tmp_path, env)
            if health.returncode == 0 and health.stdout:
                break
            assert time.monotonic() < deadline, "the server never came up"
            time.sleep(0.05)
        assert health.stdout == health_step[1]

        created = _run_console(_substituted(post_step[0], mapping), tmp_path, env)
        assert created.returncode == 0, created.stderr
        body = json.loads(created.stdout)
        assert (body["graph"], body["status"]) == ("qa", "queued")
        # The page's session id came from *its* run; map it to this run's.
        mapping[json.loads(post_step[1].removesuffix(", ...}") + "}")["id"]] = body["id"]

        # Watch GET /sessions/{id} until the run lands, as the page's reader did.
        deadline = time.monotonic() + 30
        while True:
            view = _run_console(_substituted(get_step[0], mapping), tmp_path, env)
            assert view.returncode == 0, view.stderr
            if json.loads(view.stdout)["status"] in ("succeeded", "failed", "interrupted"):
                break
            assert time.monotonic() < deadline, "the session never reached a terminal status"
            time.sleep(0.05)
        assert json.loads(view.stdout)["status"] == "succeeded"

        trace = _run_console(_substituted(trace_step[0], mapping), tmp_path, env)
        assert trace.returncode == 0, trace.stderr
        phases = [json.loads(line)["phase"] for line in trace.stdout.splitlines()]
        assert phases == ["topology", "start", "end"], trace.stdout
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


# -- the claims, pinned independently of the printed output -------------------


class _State(GraphARCState):
    log: list[str] = []


def _mixed_graph():
    ran: list[str] = []

    def cheap(state: _State) -> dict:
        ran.append("cheap")
        return {"log": [*state.log, "cheap"]}

    async def costly(state: _State) -> dict:
        await asyncio.sleep(0)
        return {"log": [*state.log, "costly"]}

    g = GraphARC(_State, name="mixed")
    g.add_node("cheap", cheap, writes={"log"})
    g.add_node("costly", costly, writes={"log"})
    g.add_edge(START, "cheap")
    g.add_edge("cheap", "costly")
    g.add_edge("costly", END)
    return g.compile(), ran


def test_a_sync_entry_point_refuses_before_any_node_runs():
    """The doc's claim that `cheap ran` prints once, not twice."""
    graph, ran = _mixed_graph()
    with pytest.raises(AsyncNodeError):
        graph.invoke({})
    assert ran == [], "a refused invoke() executed a node"

    assert asyncio.run(graph.ainvoke({}))["log"] == ["cheap", "costly"]
    assert ran == ["cheap"]


def test_stream_refuses_async_nodes_and_names_the_async_twin():
    graph, _ = _mixed_graph()
    with pytest.raises(AsyncNodeError) as caught:
        list(graph.stream({}))
    assert "use astream()" in str(caught.value)


@pytest.mark.parametrize("destination", ["esclate", "", "END "])
def test_an_unknown_goto_raises_rather_than_being_dropped(destination):
    def route(state: _State) -> Command:
        return Command(goto=destination)

    g = GraphARC(_State, name="triage")
    g.add_node("route", route, writes={"log"})
    g.add_node("escalate", lambda state: {"log": ["escalated"]}, writes={"log"})
    g.add_edge(START, "route")
    g.add_edge("escalate", END)
    graph = g.compile()

    with pytest.raises(GraphRoutingError) as caught:
        graph.invoke({})
    assert repr(destination) in str(caught.value)


def test_a_goto_that_names_a_real_node_still_works():
    def route(state: _State) -> Command:
        return Command(goto="escalate")

    g = GraphARC(_State, name="triage")
    g.add_node("route", route, writes={"log"})
    g.add_node("escalate", lambda state: {"log": ["escalated"]}, writes={"log"})
    g.add_edge(START, "route")
    g.add_edge("escalate", END)
    assert g.compile().invoke({})["log"] == ["escalated"]


def test_interrupt_before_is_a_stream_keyword_not_an_invoke_one():
    """The doc calls this out as a sharp edge; pin it so the edge stays where it is."""
    g = GraphARC(_State, name="one")
    g.add_node("only", lambda state: {"log": ["ran"]}, writes={"log"})
    g.add_edge(START, "only")
    g.add_edge("only", END)
    graph = g.compile(checkpointer=InMemorySaver())

    with pytest.raises(TypeError, match="interrupt_before"):
        graph.invoke({}, thread_id="t", interrupt_before=["only"])


def test_a_rejected_gated_node_never_executes_its_body(tmp_path):
    with SessionManager(tmp_path / "sessions") as manager:
        session = manager.create(GRAPH_NAME, session_id="s")
        first = session.run({})
        assert first.status is SessionStatus.AWAITING_APPROVAL
        assert "apply" not in first.state["log"]

        session.decide(approved=False, decided_by="ops", reason="no")
        final = session.run()

    assert final.status is SessionStatus.IDLE
    assert final.skipped == ("apply",)
    assert final.nodes == ("report",)
    # Append-only log: the gated node's body left no mark at all.
    assert final.state["log"] == ["ingest", "plan", "report"]
    assert final.state["outcome"].startswith("refused by ops")


def test_a_fanout_hold_is_per_task_and_the_requests_are_indistinguishable():
    """Approving k of n runs k; the doc says which k is not yours to pick."""
    code = next(s["code"] for s in RUNNABLE if "add_fanout_edge" in s["code"])
    printed = _exec_snippet(code, "cookbook_serving_fanout_claim")

    holds = [line for line in printed.splitlines() if line.startswith("  hold ")]
    assert len(holds) == 3
    assert len(set(holds)) == 1, "the doc claims the three holds are indistinguishable"

    sent = printed.splitlines()[-1]
    assert sent.startswith("sent   : [")
    # Two of three recipients, and exactly one task walked past.
    assert sent.count("@x") == 2
    assert "skipped: ('send',)" in printed


class _AsyncState(GraphARCState):
    question: str
    answer: str = ""


def _async_graph(checkpointer, trace: TraceRecorder):
    async def answer(state: _AsyncState) -> dict:
        await asyncio.sleep(0)
        return {"answer": "42"}

    g = GraphARC(_AsyncState, name="qa", trace=trace)
    g.add_node("answer", answer, writes={"answer"})
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)


def _drive(client, graph: str) -> dict:
    created = client.post("/sessions", json={"graph": graph, "input": {"question": "q"}})
    session_id = created.json()["id"]
    for _ in range(2000):
        view = client.get(f"/sessions/{session_id}").json()
        if view["status"] in ("succeeded", "failed", "interrupted"):
            return view
    raise AssertionError("session never reached a terminal status")


@pytest.mark.parametrize(
    ("saver", "expected"),
    [("sqlite", "failed"), ("memory", "succeeded")],
)
def test_the_server_needs_an_async_checkpointer_for_async_nodes(saver, expected, tmp_path):
    conn = sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn) if saver == "sqlite" else InMemorySaver()
    registry = ServerRegistry({"g": lambda trace: _async_graph(checkpointer, trace)})
    try:
        with TestClient(create_app(registry=registry)) as client:
            view = _drive(client, "g")
    finally:
        conn.close()

    assert view["status"] == expected
    if expected == "failed":
        # The message the doc prints, and both of the ways out it names.
        assert "CheckpointerNotAsyncError" in view["error"]
        assert "AsyncSqliteSaver" in view["error"]
        assert "InMemorySaver" in view["error"]
    else:
        assert view["result"]["answer"] == "42"


def test_a_sync_graph_on_a_sync_saver_is_served_by_the_fallback_driver(tmp_path):
    """The row of the doc's table that would otherwise only be asserted in prose."""

    class State(GraphARCState):
        question: str
        answer: str = ""

    conn = sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)

    def build(trace: TraceRecorder):
        g = GraphARC(State, name="sync", trace=trace)
        g.add_node("answer", lambda state: {"answer": "42"}, writes={"answer"})
        g.add_edge(START, "answer")
        g.add_edge("answer", END)
        return g.compile(checkpointer=SqliteSaver(conn))

    try:
        with TestClient(create_app(registry=ServerRegistry({"g": build}))) as client:
            view = _drive(client, "g")
    finally:
        conn.close()

    assert view["status"] == "succeeded"
    assert view["result"]["answer"] == "42"


def test_posting_a_message_is_accepted_but_not_applied():
    """`202` does not mean the run was steered — the doc leads with this."""

    class State(GraphARCState):
        question: str
        answer: str = ""

    def build(trace: TraceRecorder):
        g = GraphARC(State, name="qa", trace=trace)
        g.add_node("answer", lambda state: {"answer": "42"}, writes={"answer"})
        g.add_edge(START, "answer")
        g.add_edge("answer", END)
        return g.compile()

    with TestClient(create_app(registry=ServerRegistry({"g": build}))) as client:
        view = _drive(client, "g")
        ack = client.post(
            f"/sessions/{view['id']}/events", json={"type": "message", "data": {}}
        )

    assert ack.status_code == 202
    assert ack.json()["event"]["applied"] is False
    assert "does not deliver" in ack.json()["event"]["detail"]


def test_a_session_directory_is_all_a_second_manager_needs(tmp_path):
    """The restart snippet's mechanism, without the subprocess spawn.

    The subprocess proof is in the doc (and in `tests/test_session.py`); this
    pins the part the doc's prose rests on — that a *fresh* manager over the
    same directory picks the thread up where it stopped, with no node re-run.
    """
    root = tmp_path / "sessions"
    with SessionManager(root) as first:
        session = first.create(GRAPH_NAME, session_id="incident-42")
        assert first.list()[0].id == "incident-42"
        assert session.run({}).status is SessionStatus.AWAITING_APPROVAL

    with SessionManager(root) as second:
        resumed = second.resume("incident-42")
        resumed.decide(approved=True, decided_by="ops")
        turn = resumed.run()

    assert turn.nodes == ("apply", "report")
    assert turn.state["log"] == ["ingest", "plan", "apply", "report"]


def test_the_doc_names_a_real_temp_root_pattern():
    """Every session snippet writes under a fresh temp dir, never the repo."""
    for snippet in RUNNABLE:
        if "SessionManager(" not in snippet["code"]:
            continue
        assert "tempfile.mkdtemp" in snippet["code"]


def test_tempdirs_the_snippets_create_are_outside_the_repo():
    assert not Path(tempfile.gettempdir()).is_relative_to(DOC.parents[2])
