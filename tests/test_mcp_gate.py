"""The MCP supervision surface — `grapharc.mcp`.

The property under test is the one the module refuses to compromise: a
supervised agent may request and may check, and can never decide. Everything
else is the thin-shim contract — the server drives the tested CLI, confines
what a client may name, honours the plan's own mutating verdict fail-closed,
and keeps stdout for the protocol.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from grapharc.cli.main import main
from grapharc.mcp import FORBIDDEN_TOOL_WORDS, build_server
from grapharc.mcp.driver import (
    DriverError,
    confine_run_dir,
    plan_is_mutating,
    read_plan_record,
)
from grapharc.planner.approval_file import REQUEST_FILENAME


def _unwrap(raw) -> dict:
    """FastMCP's call_tool result as the tool's own dict, across SDK shapes."""
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        structured = raw[1]
        # Structured output arrives either as the dict itself or under "result".
        return structured.get("result", structured)
    blocks = raw[0] if isinstance(raw, tuple) else raw
    text = "".join(getattr(block, "text", "") for block in blocks)
    return json.loads(text)


@pytest.mark.asyncio
async def test_the_surface_is_three_tools_and_no_approval_verb(tmp_path):
    """A client that could call approve() would be approving its own
    proposal, which is not approval. The refusal is a property, not prose."""
    server = build_server(tmp_path)
    tools = await server.list_tools()

    names = {tool.name for tool in tools}
    assert names == {"plan", "show_graph", "execute"}
    for name in names:
        for word in FORBIDDEN_TOOL_WORDS:
            assert word not in name.lower()


@pytest.mark.asyncio
async def test_the_plan_tool_offers_no_registry_policy_or_model_parameter(tmp_path):
    """Those resolve from the operator's grapharc.toml in the server's root;
    the requester's call must not be able to widen what the operator set."""
    server = build_server(tmp_path)
    plan_tool = next(t for t in await server.list_tools() if t.name == "plan")

    parameters = set((plan_tool.inputSchema or {}).get("properties", {}))
    assert parameters <= {"goal", "scripted", "max_rounds"}


@pytest.mark.asyncio
async def test_plan_show_execute_scripted_end_to_end(tmp_path, capsys):
    """The whole supervised flow, spend-free: propose, read the shape back,
    execute the read-only plan on the spot, and read the record of the run.
    The server itself writes nothing to stdout — the protocol owns it."""
    server = build_server(tmp_path)

    planned = _unwrap(await server.call_tool("plan", {"goal": "investigate", "scripted": True}))
    assert planned["stop"] == "planned"
    assert planned["proposal"]["nodes"], "the admitted shape travels as data"
    assert planned["fingerprint"]
    assert planned["mutating"] is False  # the incident replan admits no deploy
    run_dir = planned["run_dir"]

    shown = _unwrap(await server.call_tool("show_graph", {"run_dir": run_dir}))
    assert shown["status"] == "planned"
    assert shown["awaiting_approval"] is False

    done = _unwrap(await server.call_tool("execute", {"run_dir": run_dir}))
    assert done["executed"] is True  # read-only: no park, the host prompt sufficed

    after = _unwrap(await server.call_tool("show_graph", {"run_dir": run_dir}))
    assert after["status"] == "done"
    assert after["executed_run_id"]
    assert "mermaid" in after and after["metrics"]["nodes_executed"]

    assert capsys.readouterr().out == ""


def _mark_mutating(root: Path, run_dir: str) -> Path:
    """Flip the record's verdict, resolving the CLI's root-relative run_dir."""
    resolved = Path(run_dir) if Path(run_dir).is_absolute() else root / run_dir
    plan_file = resolved / "plan.json"
    record = json.loads(plan_file.read_text())
    record["mutating"] = True
    plan_file.write_text(json.dumps(record, indent=2) + "\n")
    return resolved


@pytest.mark.asyncio
async def test_a_mutating_plan_parks_and_a_timeout_leaves_it_unexecuted(tmp_path):
    server = build_server(tmp_path)
    planned = _unwrap(await server.call_tool("plan", {"goal": "fix", "scripted": True}))
    run_dir = _mark_mutating(tmp_path, planned["run_dir"])

    outcome = _unwrap(
        await server.call_tool("execute", {"run_dir": str(run_dir), "approval_timeout": 0.8})
    )

    assert outcome["executed"] is False
    assert outcome["stop"] == "approval_timeout"
    assert "grapharc approve" in outcome["approve_command"]
    assert "executed_run_id" not in json.loads((run_dir / "plan.json").read_text())


@pytest.mark.asyncio
async def test_a_mutating_plan_executes_after_an_out_of_band_approval(tmp_path):
    """The park is the request and the human's CLI is the decision — the MCP
    connection never carries a yes."""
    server = build_server(tmp_path)
    planned = _unwrap(await server.call_tool("plan", {"goal": "fix", "scripted": True}))
    run_dir = _mark_mutating(tmp_path, planned["run_dir"])

    def answer():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (run_dir / REQUEST_FILENAME).exists():
                assert main(["approve", str(run_dir), "--json"]) == 0
                return
            time.sleep(0.05)

    thread = threading.Thread(target=answer)
    thread.start()
    outcome = _unwrap(
        await server.call_tool("execute", {"run_dir": str(run_dir), "approval_timeout": 20})
    )
    thread.join()

    assert outcome["executed"] is True


def test_a_plan_record_without_the_verdict_reads_as_mutating(tmp_path):
    """Absent is never safe: an old plan.json predates the field, and the
    driver must park it rather than assume it read-only."""
    assert plan_is_mutating({}) is True
    assert plan_is_mutating({"mutating": "false"}) is True  # a string is not a verdict
    assert plan_is_mutating({"mutating": False}) is False


def test_a_run_dir_outside_the_root_is_refused(tmp_path):
    (tmp_path / "inside").mkdir()
    assert confine_run_dir(tmp_path, "inside") == (tmp_path / "inside").resolve()

    with pytest.raises(DriverError) as refusal:
        confine_run_dir(tmp_path, "/etc")
    assert "outside" in str(refusal.value)

    with pytest.raises(DriverError):
        confine_run_dir(tmp_path, "../elsewhere")


def test_a_directory_without_a_plan_names_the_missing_step(tmp_path):
    with pytest.raises(DriverError) as refusal:
        read_plan_record(tmp_path)
    assert "call plan first" in str(refusal.value)
