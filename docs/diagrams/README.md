# Architecture diagrams

Rendered from [`architecture.py`](architecture.py) with
[`diagrams`](https://diagrams.mingrammer.com) over Graphviz. Four views, matching
how `ARCHITECTURE.md` decomposes the system.

```bash
uv pip install diagrams          # not a project dependency; `uv sync` will drop it
sudo apt install graphviz        # provides `dot`
.venv/bin/python docs/diagrams/architecture.py
```

| | view | what it answers |
|---|---|---|
| 1 | [`01-lifecycle.png`](01-lifecycle.png) | trigger → outcome, and the loop back through the gate (§1, §2) |
| 2 | [`02-planes.png`](02-planes.png) | what every node sits on, and what constrains it (§3) |
| 3 | [`03-agent-node.png`](03-agent-node.png) | inside an agent node, gate by gate (§4) |
| 4 | [`04-subsystems.png`](04-subsystems.png) | the twelve packages, and which ones actually import which |

**View 1** is the thesis: the two coloured curves back to *③ PLAN* are what make
this a governed loop rather than a pipeline. Rejections return as traced reason
codes, and work discovered mid-run re-enters admission — there is no
already-approved path.

**View 4 is drawn from the import graph, not from intent.** It used to show
`planner/` in its own box with arrows leaving and none arriving —
`ARCHITECTURE.md` §7 gap 1, rendered rather than described. `grapharc plan`
closed it, so the box is now reachable and the arrow into it is the point.
`server/` carries the one remaining gap marker. Re-derive before trusting the
picture — these are a snapshot and rot the same way prose does:

```bash
grep -rn "grapharc.planner" grapharc/ --include=*.py | grep -v "^grapharc/planner/"
```

Line counts in view 4 come from `find grapharc/<pkg> -name '*.py' | xargs cat | wc -l`.
