"""`grapharc init --claude-code` — adopt GraphARC as a supervision layer.

Writes the two files that turn a Claude Code checkout into a supervised one:
`.mcp.json`, registering `grapharc mcp` as a project MCP server, and
`.claude/skills/grapharc/SKILL.md`, the contract that routes multi-step and
state-changing work through plan → show → execute instead of a stream of
unsupervised edits.

Templates are string constants, the `init` convention: nothing here reads
packaged data files, and what a test asserts about the skill is asserted
against the exact bytes a user gets. Neither file is ever overwritten —
an existing one is the operator's, and the refusal names it.

The skill's load-bearing clause is the last one: the agent must never answer
the approval itself. The MCP surface enforces that on its own connection (no
approve tool exists), but the host agent also holds Write and Bash — the
skill states the boundary for the hands the server cannot see, and the trust
boundary remains the working directory, as it is for the Slack workspace.
"""

from __future__ import annotations

from pathlib import Path

from grapharc.cli import style
from grapharc.cli.output import EXIT_OK, EXIT_UNAVAILABLE, emit

MCP_CONFIG_FILENAME = ".mcp.json"
SKILL_PATH = Path(".claude/skills/grapharc/SKILL.md")

MCP_CONFIG_TEMPLATE = """\
{
  "mcpServers": {
    "grapharc": {
      "command": "grapharc",
      "args": ["mcp"]
    }
  }
}
"""

SKILL_TEMPLATE = """\
---
name: grapharc
description: >-
  Route multi-step or state-changing repository work through GraphARC
  supervision: propose the work as a governed graph, show the user the graph,
  execute only what the gate admits — and what a human approved, when the plan
  can change files. Use for repo-wide fixes, refactors, migrations, or any
  task that would otherwise be a long stream of unsupervised edits.
---

# GraphARC supervision

This project routes substantial work through GraphARC's admission gate. The
`grapharc` MCP server exposes three tools — `plan`, `show_graph`, `execute` —
and deliberately nothing that decides an approval.

## The flow

1. **Propose, don't act.** For multi-step or state-changing work, call
   `plan` with the user's goal instead of editing files directly. The
   registry, policy and model come from this project's `grapharc.toml` — if
   `plan` reports no registry or model, tell the user to run `grapharc init`
   and set `model` in `grapharc.toml`; do not work around it.
2. **Show the user the graph.** Render the returned `proposal` — its nodes,
   edges and rationale — in your reply, and give them `watch_url` when it is
   set: the live page is where the graph draws itself while it runs.
3. **Execute under the gate.** Call `execute` with the returned `run_dir`.
   A plan that can change files **parks**: tell the user it is waiting and
   quote `approve_command` — they answer with `grapharc approve <run_dir>`
   (or `--deny`) in a terminal, or from the live view. A read-only plan runs
   immediately.
4. **A timeout means ask, not retry.** If `execute` comes back
   `approval_timeout`, the plan is unexecuted and the call is safe to
   reissue — after the user says so. Never loop on `execute` waiting for a
   yes that has not been given.
5. **Report from the record.** After execution, call `show_graph` for the
   metrics and the Mermaid rendering of what actually ran, and cite those
   rather than your recollection.

## What you must never do

- Never run `grapharc approve`, in any form, for any reason.
- Never create, edit or delete `approval-request.json` or
  `approval-decision.json` — those files are the human's channel, not yours.
- Never edit `plan.json` to change what was admitted or whether it counts as
  mutating; an edited plan is a new proposal, and the gate treats it as one.

The decision belongs to the user. Your job is to make the question easy to
answer: show the graph, quote the command, and wait.
"""


def adopt_claude_code(*, as_json: bool = False) -> int:
    """Write `.mcp.json` and the skill, or refuse naming what already exists."""
    mcp_config = Path(MCP_CONFIG_FILENAME)
    skill = SKILL_PATH

    existing = [str(p) for p in (mcp_config, skill) if p.exists()]
    if existing:
        message = (
            f"refusing to overwrite: {', '.join(existing)} — these are yours "
            "once written; move one aside if you want it regenerated"
        )
        if as_json:
            emit({"ok": False, "command": "init", "error": message}, [], as_json=True)
        else:
            import sys

            print(f"error: {message}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    skill.parent.mkdir(parents=True, exist_ok=True)
    mcp_config.write_text(MCP_CONFIG_TEMPLATE, encoding="utf-8")
    skill.write_text(SKILL_TEMPLATE, encoding="utf-8")

    payload = {
        "ok": True,
        "command": "init",
        "mcp_config": str(mcp_config),
        "skill": str(skill),
    }
    width = style.LABEL_WIDTH
    lines = [
        style.kv("wrote", str(mcp_config), width=width, tint=style.accent),
        style.kv("wrote", str(skill), width=width, tint=style.accent),
        "",
        style.kv(
            "next",
            "open Claude Code here; multi-step work now routes through "
            "plan -> show -> execute, and approvals stay yours",
            width=width,
        ),
    ]
    emit(payload, lines, as_json=as_json)
    return EXIT_OK


__all__ = [
    "MCP_CONFIG_FILENAME",
    "MCP_CONFIG_TEMPLATE",
    "SKILL_PATH",
    "SKILL_TEMPLATE",
    "adopt_claude_code",
]
