<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/grapharc-logo-ondark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/brand/grapharc-logo-light.svg">
    <img alt="GraphARC" src="docs/brand/grapharc-logo-light.svg" width="440">
  </picture>
</p>

# GraphARC

[![PyPI](https://img.shields.io/pypi/v/grapharc.svg)](https://pypi.org/project/grapharc/)
[![Downloads](https://static.pepy.tech/badge/grapharc)](https://pepy.tech/project/grapharc)
[![Python](https://img.shields.io/pypi/pyversions/grapharc.svg)](https://pypi.org/project/grapharc/)
[![CI](https://github.com/CodeGraphContext/GraphARC/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeGraphContext/GraphARC/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The admission gate for agent graphs**, built on [LangGraph](https://github.com/langchain-ai/langgraph).

A model proposes a graph of work; a deterministic checker admits it or refuses it with reasons; only admitted graphs execute, under budgets, onto one replayable JSONL trace. No step runs unadmitted, the worst case is priced before a run and billed per node after, and the dashboard cannot disagree with the audit trail because they are the same file. The edges are documented in [Limits](#limits).

![A nine-node incident-investigation graph running live in the browser: triage fans out into four parallel evidence pulls, they join at correlate, then hypothesize, verify and report — each node amber while it runs and green with its own token bill when done.](docs/media/grapharc-incident-demo.gif)

*One question in, a governed graph out, live in the browser. ([mp4](docs/media/grapharc-incident-demo.mp4))*

## Install

```bash
pip install grapharc     # Python >= 3.12
grapharc demo stage0     # costs nothing, needs no key
```

Backends are extras: `grapharc[openrouter]`, `[openai]`, `[ollama]`, `[server]`, `[all]`. The default backend drives the `claude` CLI on your PATH — a Claude subscription, no API key.

## Quick start

```bash
grapharc start                    # guided tour
grapharc init                     # scaffold registry.py + grapharc.toml
grapharc plan "look into the outage" --model ollama/qwen3:8b   # propose -> admit -> save
grapharc go                       # execute the saved plan
grapharc plan "..." --scripted    # free rehearsal, no AI involved
grapharc serve --live-root .grapharc/runs   # live browser view of every run
grapharc replay <trace> <run-id>  # reconstruct a run from its trace
```

Building a graph directly:

```python
from grapharc import GraphARC, GraphARCState, Budget
from grapharc.runtime.graph import START, END

class State(GraphARCState):
    question: str
    answer: str = ""

def answer(state: State) -> dict:
    return {"answer": f"42 (asked: {state.question})"}

g = GraphARC(State, name="demo", budget=Budget(max_iterations=10))
g.add_node("answer", answer, writes={"answer"})   # undeclared writes raise
g.add_edge(START, "answer")
g.add_edge("answer", END)
print(g.compile().invoke({"question": "meaning of life"}))
# {'question': 'meaning of life', 'answer': '42 (asked: meaning of life)'}
```

Typed state, declared writes, a budget — none optional. The rest of the surface (agents, tools, memory, sessions, policy documents, cost attribution, OTel) is in the [deep dive](docs/deep-dive.md) and the [cookbook](docs/cookbook/).

## The admission gate

You cannot pre-author a graph for "investigate this incident" — the shape is discovered while working. So the graph is proposed at runtime, and a deterministic checker stands between proposing and running: registry, policy, remaining budget, depth, acyclicity, all on every proposal. A rejection is structured feedback the planner replans against; work discovered mid-run re-enters the same gate. Watch it refuse (free, scripted, the first proposal names a policy-denied kind):

```bash
grapharc plan "investigate the checkout outage" --scripted --go
```

```
goal      : investigate the checkout outage
model     : scripted stand-in (--scripted)
registry  : grapharc.examples.plan_incident:build_registry
kinds     : deploy, patch, triage, verify
policy    : grapharc.examples.plan_incident:build_registry default (deny -> deploy, otherwise allow)  [registry-default]
config    : no grapharc.toml (flags and defaults only)

stopped   : goal_met  (the goal check was satisfied)
rounds    : 2 of max 8
   round 1: rejected  nodes=2 executed=False  rejected: edge_denied
   round 2: admitted  nodes=3 executed=True

state     : goal='investigate the checkout outage' notes=['triage ran', 'patch ran', 'verify ran']
```

Round 1 wanted to deploy and **never executed**. Round 2 went through the same checker and ran. The full assembly, runnable as written:

```python
from pydantic import BaseModel

from grapharc.harness.permissions import Decision
from grapharc.planner import (
    AdmissionChecker, CostEstimate, EdgePolicy, EdgeRule, GovernedLoop,
    LoopLimits, Materializer, NodeRegistry, NodeSpec, PlannerNode,
)
from grapharc.runtime.budget import Budget
from grapharc.testing import ScriptedChatModel


class State(BaseModel):
    found: str = ""
    fixed: str = ""


def factory(spec):                              # bodies come from HERE, never a proposal
    def body(state):
        return {"found": "cause"} if spec.name == "search" else {"fixed": "patch"}
    return body


registry = NodeRegistry([                        # the kinds a planner may propose
    NodeSpec(name="search", factory=factory, worst_case=CostEstimate(tokens=500)),
    NodeSpec(name="edit",   factory=factory, worst_case=CostEstimate(tokens=2000)),
    NodeSpec(name="deploy", factory=factory),
]).freeze()
policy = EdgePolicy(rules=(                      # deny -> ask -> allow, unmatched is deny
    EdgeRule(action=Decision.DENY, target="deploy"),
    EdgeRule(action=Decision.ALLOW),
))

plan = '{"nodes": [{"name": "%s"}], "edges": [{"source": "__start__", "target": "%s"}]}'
loop = GovernedLoop(
    planner=PlannerNode(
        ScriptedChatModel(responses=[plan % ("deploy", "deploy"), plan % ("edit", "edit")]),
        catalog=registry.catalog(),
    ),
    checker=AdmissionChecker(registry=registry, edge_policy=policy),
    materializer=Materializer(
        registry=registry, state_schema=State,
        writes={"search": {"found"}, "edit": {"fixed"}, "deploy": set()},
    ),
    budget=Budget(max_tokens=100_000),
    limits=LoopLimits(max_rounds=8),
    goal_reached=lambda s: bool(s.fixed),
)
result = loop.run("find and fix the bug", State())

print(result.stop.value)
for record in result.rounds:
    print(record.round, record.admission.status.value, record.executed)
print([r.code for r in result.rejections()])
```

Output:

```
goal_met
1 rejected False
2 admitted True
['edge_denied']
```

Both blocks are executed by `tests/test_readme.py` against every commit, so this page cannot drift from the code.

## Where it sits

| | GraphARC | Claude Code | OpenClaw | raw LangGraph |
|---|---|---|---|---|
| Shape | Governed multi-node graph runtime | Interactive single-agent loop | Personal AI assistant gateway | Graph mechanism library |
| Who authorizes work | Deterministic gate, pre-execution, with reasons | A human, live, per action | Configuration and allowlists | Nobody — convention |
| Cost control | Worst-case admission + per-node bill, fail-closed | Usage visibility | Spend settings | None built in |
| Audit | One replayable JSONL trace | Session transcripts | Logs | Checkpoints (state, not *why*) |

Different jobs, not competitors — GraphARC's default backend drives the Claude CLI. See [benchmarks](bench/) for measured comparisons against third-party agents on the same tasks: success, cost, wall time, and policy violations, with raw logs committed.

## Limits

The edges are documented, not denied — the full list with mechanisms is in the [deep dive](docs/deep-dive.md#limits).

- Admission authorises a node's *kind*, never its arguments.
- The in-process sandbox is defense in depth; `ContainerExecutor` is the real boundary. `run_command` children are unconfined.
- The HTTP API does not yet use the durable session layer.
- On the Claude CLI backend an agent node is *delegated*, not governed.
- Policy documents govern planning; the tool plane still reads CLI flags.

Version `0.1.5` · [changelog](CHANGELOG.md) · [roadmap](ROADMAP.md) · [website](https://codegraphcontext.github.io/GraphARC/) · MIT
