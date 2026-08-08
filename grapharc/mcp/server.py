"""`grapharc mcp` — the supervision surface an external agent plugs into.

Three tools, and deliberately not a fourth:

- `plan(goal, ...)` proposes a graph for the goal through the governed loop
  and returns the admitted shape as data — nodes, edges, rationale,
  fingerprint, and whether executing it can change anything.
- `show_graph(run_dir)` is the read-only view: the proposal, whether a human
  is being asked right now, and — after execution — the metrics and the
  Mermaid rendering of what actually ran.
- `execute(run_dir, ...)` re-admits and runs the saved plan. A plan
  containing a mutating kind parks on the file handshake until a human
  answers `grapharc approve <run_dir>` out of band; an all-read-only plan
  runs on the host's own prompt.

**There is no approve, deny, or decide tool, and there never will be.** The
gate is between the agent and the operator; a client that could call
`approve()` would be approving its own proposal, which is not approval. The
supervised agent may request (execute parks) and may check (show_graph says
`awaiting_approval`); the decision belongs to a human with a terminal or the
live view, outside this connection.

The server drives the `grapharc` CLI as subprocesses and never prints to
stdout — the stdio transport owns it. Diagnostics go to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from grapharc.mcp import driver
from grapharc.mcp.driver import DEFAULT_APPROVAL_TIMEOUT, DriverError

#: Verbs this surface refuses to grow. Checked by a gate test against the
#: registered tool names, so the refusal is a property, not a comment.
FORBIDDEN_TOOL_WORDS = ("approve", "deny", "decide")


def build_server(root: Path | None = None) -> FastMCP:
    """The FastMCP server, rooted where it was started.

    `root` confines every `run_dir` a client names and is the working
    directory of every spawned CLI — which is what makes the operator's
    `grapharc.toml` there, and nothing the client says, decide the registry,
    the policy and the model.
    """
    base = (root or Path.cwd()).resolve()
    server = FastMCP(
        "grapharc",
        instructions=(
            "GraphARC supervision: plan proposes a governed graph for a goal "
            "and returns the admitted shape; show_graph reads a run "
            "directory back; execute runs a saved plan, parking for "
            "out-of-band human approval when the plan can change files. "
            "There is no approve tool: the decision belongs to the human, "
            "via `grapharc approve <run_dir>` or the live view."
        ),
    )

    @server.tool()
    async def plan(
        goal: str, scripted: bool = False, max_rounds: int | None = None
    ) -> dict[str, Any]:
        """Propose a governed graph for `goal` and return the admitted shape.

        The registry, policy and model come from the operator's grapharc.toml
        in the server's directory — they are not parameters, on purpose.
        `scripted=True` rehearses with the registry's canned planner: free,
        deterministic, no model call.
        """
        code, out, err = await driver.run_cli(
            driver.build_plan_argv(goal, scripted=scripted, max_rounds=max_rounds),
            cwd=base,
        )
        document = driver.parse_document(out, command="plan")
        if code != 0 and not document.get("ok", False):
            # The CLI's failure document is the answer — setup problems arrive
            # with the CLI's own instructive text rather than a bare error.
            return document
        return {
            "run_dir": document.get("run_dir"),
            "goal": document.get("goal"),
            "stop": document.get("stop"),
            "proposal": document.get("proposal"),
            "fingerprint": document.get("fingerprint"),
            "mutating": document.get("mutating", True),
            "kinds": document.get("kinds", []),
            "policy": document.get("policy"),
            "rejections": document.get("rejections", []),
            "rounds": document.get("rounds", []),
            "watch_url": document.get("watch_url"),
            "plan_file": document.get("plan_file"),
            "approve_command": (
                f"grapharc approve {document.get('run_dir')}"
                if document.get("mutating", True) and document.get("run_dir")
                else None
            ),
        }

    @server.tool()
    async def show_graph(run_dir: str) -> dict[str, Any]:
        """Read one run directory back: the admitted proposal, whether a human
        is being asked right now, and the record of what ran. Read-only."""
        resolved = driver.confine_run_dir(base, run_dir)
        return driver.graph_status(resolved)

    @server.tool()
    async def execute(
        run_dir: str, approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT
    ) -> dict[str, Any]:
        """Execute the plan saved in `run_dir`, re-admitted through the gate.

        A plan with a mutating kind parks until a human answers
        `grapharc approve <run_dir>` out of band; tell the user, do not
        answer it yourself. A timeout leaves the plan unexecuted and this
        call safe to reissue.
        """
        resolved = driver.confine_run_dir(base, run_dir)
        record = driver.read_plan_record(resolved)
        mutating = driver.plan_is_mutating(record)
        code, out, err = await driver.run_cli(
            driver.build_execute_argv(
                resolved, mutating=mutating, approval_timeout=approval_timeout
            ),
            cwd=base,
            # The subprocess bounds its own park via --approval-timeout; this
            # outer bound only catches a wedged process, generously.
            timeout=approval_timeout + 120.0 if mutating else None,
        )
        document = driver.parse_document(out, command="go")
        document["mutating"] = mutating
        if mutating and not document.get("executed", False):
            document["approve_command"] = f"grapharc approve {resolved}"
        return document

    return server


def serve_stdio(root: Path | None = None) -> int:
    """Run the server on stdio until the client hangs up. The CLI entry."""
    try:
        build_server(root).run(transport="stdio")
    except KeyboardInterrupt:  # a Ctrl-C is a clean goodbye, not a stack trace
        print("grapharc mcp: stopped", file=sys.stderr)
    return 0


__all__ = ["FORBIDDEN_TOOL_WORDS", "DriverError", "build_server", "serve_stdio"]
