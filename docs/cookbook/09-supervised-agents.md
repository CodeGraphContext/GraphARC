# 09 — Supervised agents: GraphARC as an MCP server

The agents people already use — Claude Code first among them — act alone: one
loop, its own judgment, edits landing as fast as it can type them. This
chapter turns that around without replacing the agent. The agent keeps its
intelligence; GraphARC supplies the gate. Work is proposed as a graph, the
human sees the graph, and execution happens under budgets, onto the trace,
with the mutating case parked until a human answers out of band.

## The shape of it

`grapharc mcp` is a stdio MCP server exposing exactly three tools:

| tool | what it does | what it cannot do |
|---|---|---|
| `plan(goal, scripted?, max_rounds?)` | run the governed planning loop for the goal; return the admitted shape — nodes, edges, rationale, fingerprint — plus a `mutating` verdict | choose a registry, policy or model: those resolve from the operator's `grapharc.toml` in the server's root, never from the call |
| `show_graph(run_dir)` | read a run directory back: the proposal, whether a human is being asked right now, and — after execution — metrics and the Mermaid of what ran | see raw state: like the live view, it serves rendered summaries, never `state_delta` |
| `execute(run_dir, approval_timeout?)` | re-admit the saved plan through the gate and run it | approve itself: a mutating plan parks on the file handshake until a human answers `grapharc approve <run_dir>` |

There is no `approve` tool, and there never will be. The gate is between the
agent and the operator; a client that could call `approve()` would be
approving its own proposal, which is not approval. The supervised agent may
*request* (execute parks) and may *check* (`show_graph` says
`awaiting_approval`); the decision belongs to a human with a terminal or the
live view, outside the connection entirely.

The tiering is deliberate: a plan whose admitted kinds are all read-only
executes on the host agent's own permission prompt — the human already said
yes to the tool call — while any plan containing a mutating kind waits for
the out-of-band answer. The verdict is computed at plan time against the
registry module's own `MUTATING_KINDS`, is stored in `plan.json` beside the
fingerprint, and fails closed twice over: a registry that declared nothing
reads as mutating, and so does a plan file without the field.

## Adopting it in a Claude Code project

```bash
cd your-project
grapharc init                 # once, if there is no registry.py/grapharc.toml yet
grapharc init --claude-code   # writes .mcp.json and .claude/skills/grapharc/SKILL.md
```

`.mcp.json` registers `grapharc mcp` as a project server, so Claude Code
starts it on demand. The skill is the behavioural half: it routes multi-step
and state-changing work through plan → show → execute, tells the agent to
render the proposal and the `watch_url` to the user, and states the boundary
the server cannot enforce on the host's *other* hands — never run
`grapharc approve`, never touch the request or decision files, and a timeout
means ask, not retry. Neither file is ever overwritten; an existing one is
yours, and the command refuses by name.

Watch a supervised run the same way as any other: `grapharc serve
--live-root .grapharc/runs`, and the parked proposal is drawn on the live
page with the approve command beside it.

## The trust boundary, stated plainly

The MCP gate binds the MCP surface, not the machine. A host agent holds its
own Write and Bash, and a process in the working directory can forge
`approval-decision.json` — the same posture as the Slack gate's workspace:
the trust boundary is the directory, and the skill's never-clauses are the
contract for hands the server cannot see. The park also lives inside one MCP
call: a host that times the tool out kills the wait, the plan stays
unexecuted, and the call is safe to reissue. And approval records the
decision, never the decider — the trace has no actor field.
