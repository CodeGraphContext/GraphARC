# Architecture diagrams

Views of the same system, from the trust boundary outwards, from two sources.

`architecture.png` — the canonical view, the image at the top of the README — is
laid out by hand in [`grapharc-architecture.drawio`](grapharc-architecture.drawio),
whose other two pages cover the trust boundary and the import graph. Edit it at
[app.diagrams.net](https://app.diagrams.net) (File → Open From → Device) and
re-export page 0 as `architecture.png`.

Everything numbered is generated instead:

```bash
uv pip install diagrams          # not a project dependency; `uv sync` will drop it
sudo apt install graphviz        # provides `dot`
.venv/bin/python docs/diagrams/architecture.py
```

| | view | what it answers |
|---|---|---|
| **0** | [**`architecture.png`**](architecture.png) | **the whole system in one frame — start here** |
| | [`00-architecture.png`](00-architecture.png) | the same view, rendered by `architecture.py` through Graphviz |
| 1 | [`01-lifecycle.png`](01-lifecycle.png) | trigger → outcome, and the loop back through the gate (§1, §2) |
| 2 | [`02-planes.png`](02-planes.png) | what every node sits on, and what constrains it (§3) |
| 3 | [`03-agent-node.png`](03-agent-node.png) | inside an agent node, gate by gate (§4) |
| 4 | [`04-subsystems.png`](04-subsystems.png) | the twelve packages, and which ones actually import which |
| 5 | [`05-trust-boundary.png`](05-trust-boundary.png) | **who supplies what** — the operator declares, the model proposes, the checker decides |

**View 0 is the canonical one** — the whole runtime on one spine, and the image at the top of the README. The five that follow each answer a single question in more depth. The two renderings of view 0 are the same diagram: `architecture.png` is the hand-laid one the README embeds, `00-architecture.png` what `architecture.py` produces.

**View 5 is the one to read after it.** The other four show structure; this one
shows the *boundary*, which is the only question that decides whether the rest is
a safety argument or decoration. An operator authors the registry, the policy, the
schema and the budgets — all before a model runs. The model contributes exactly
one thing: JSON naming kinds and edges. It cannot supply a node body (`extra=
"forbid"`), cannot smuggle a callable (not JSON-serialisable, so it cannot be
fingerprinted and never reaches the gate), cannot grant itself budget, and cannot
add a kind (the registry is frozen).

**View 1** is the same claim as a lifecycle: the two coloured curves back to
*③ PLAN* are what make this a governed loop rather than a pipeline. Rejections
return as traced reason codes, and work discovered mid-run re-enters admission —
there is no already-approved path.

**View 4 is drawn from the import graph, not from intent.** It used to show
`planner/` in its own box with arrows leaving and none arriving —
the largest gap this project had, rendered rather than described. `grapharc plan`
closed it, so the box is now reachable and the arrow into it is the point.
`server/` carries the one remaining gap marker. Re-derive before trusting the
picture — these are a snapshot and rot the same way prose does:

```bash
grep -rn "grapharc.planner" grapharc/ --include=*.py | grep -v "^grapharc/planner/"
```

Line counts in view 4 come from `find grapharc/<pkg> -name '*.py' | xargs cat | wc -l`.
