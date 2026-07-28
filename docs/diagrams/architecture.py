"""Render GraphARC's architecture as PNGs with `diagrams` (mingrammer).

Four views, matching how `ARCHITECTURE.md` decomposes the system:

  1. lifecycle   — trigger to outcome, including the replan loop (§1, §2)
  2. planes      — what every node sits on, and what constrains it (§3)
  3. agent-node  — the reusable unit, gate by gate (§4)
  4. subsystems  — the twelve packages, and which ones actually import which

View 4 is drawn from the import graph rather than from intent. It used to show
`planner/` sitting unwired — nothing in the tree imported it — which was
ARCHITECTURE.md §7 gap 1; `grapharc plan` closed it, and `policy/` now reaches
the admission gate through `PolicyEngine.edge_policy()`. Verify before trusting
the picture, because a diagram rots exactly the way prose does:

    grep -rn "grapharc.planner" grapharc/ --include=*.py | grep -v "^grapharc/planner/"

Requires the `diagrams` package and a Graphviz `dot` binary:

    uv pip install diagrams && sudo apt install graphviz
    .venv/bin/python docs/diagrams/architecture.py
"""

from __future__ import annotations

import pathlib

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.flowchart import (
    Action,
    Database,
    Decision,
    Document,
    InputOutput,
    Inspection,
    ManualInput,
    PredefinedProcess,
    StartEnd,
    StoredData,
)

OUT = pathlib.Path(__file__).parent

# Palette. Deliberately consistent with ARCHITECTURE.md's mermaid classDefs so
# the two renderings of the same system do not look like two systems.
INK = "#1b1b1f"
TRIGGER = "#4a90d9"
SESSION = "#5cb85c"
PLAN = "#d9a441"
GATE = "#d95c5c"
REJECT = "#ff6b6b"
EXEC = "#7aa5d2"
OUTCOME = "#a57ad2"
MODEL = "#4a90d9"
TOOL = "#5cb85c"
MEMORY = "#d9a441"

GRAPH_ATTR = {
    "fontname": "Helvetica",
    "fontsize": "20",
    "labelloc": "t",
    "bgcolor": "white",
    "pad": "0.6",
    "nodesep": "0.45",
    "ranksep": "0.75",
    "splines": "spline",
}
NODE_ATTR = {"fontname": "Helvetica", "fontsize": "11", "color": INK}
EDGE_ATTR = {"fontname": "Helvetica", "fontsize": "10", "color": "#55555a"}


def cluster(label: str, colour: str) -> Cluster:
    return Cluster(
        label,
        graph_attr={
            "bgcolor": colour + "18",
            "color": colour,
            "penwidth": "1.6",
            "fontname": "Helvetica-Bold",
            "fontsize": "13",
            "fontcolor": colour,
            "style": "rounded",
            "margin": "14",
        },
    )


def lifecycle() -> None:
    """§1 + §2 — trigger to outcome, and the loop that re-enters the gate.

    Laid out left-to-right so the pipeline reads in order. The two feedback
    edges carry `constraint=false`: without it Graphviz ranks the replan target
    *after* execution and the whole diagram reorders around the back-edge.

    Admission is one node, not five, because `AdmissionChecker.check()` is one
    call returning every rejection at once — a planner replanning from feedback
    gets the whole list, not the first complaint.
    """
    with Diagram(
        "GraphARC — governed lifecycle: propose → admit → materialise → execute → replan",
        filename=str(OUT / "01-lifecycle"),
        show=False,
        direction="LR",
        outformat="png",
        graph_attr={**GRAPH_ATTR, "ranksep": "1.1", "nodesep": "0.6"},
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with cluster("① TRIGGERS", TRIGGER):
            cli = ManualInput("grapharc run / agent")
            api = ManualInput("HTTP API + SSE")
            resume = ManualInput("session resume")
            triggers = [cli, api, resume]

        with cluster("② SESSION RUNTIME  ·  grapharc/session", SESSION):
            sess = Action("create / resume\ncross-process, SQLite")
            steer = Action("interrupt & steering\nread at the next boundary")
            appr = Decision("human approval\nsuspends the graph")
            sess >> Edge(color=SESSION) >> steer >> Edge(color=SESSION) >> appr

        with cluster("③ PLAN  ·  planner/proposal.py", PLAN):
            planner = PredefinedProcess(
                "PlannerNode.propose()\n"
                "emits a typed Subgraph\n"
                "a node names a KIND —\n"
                "there is no field a body could arrive in"
            )

        with cluster("④ ADMISSION — the governance gate  ·  planner/admission.py", GATE):
            check = Decision(
                "AdmissionChecker.check()\n"
                "deterministic · model-free\n"
                "① every kind in the registry?\n"
                "② every edge policy-permitted?\n"
                "③ worst case within remaining budget?\n"
                "④ depth within max_depth?\n"
                "⑤ acyclic?"
            )
            rej = Document("REJECTED\nevery rejection at once,\neach a traced reason code")
            check >> Edge(label="  no", color=REJECT, style="dashed") >> rej

        with cluster("⑤ MATERIALISE  ·  planner/materialize.py", EXEC):
            mat = PredefinedProcess(
                "Materializer.materialize(\n    admitted, proposal)\n"
                "the authorisation is argument one;\n"
                "matched to the proposal by fingerprint"
            )
            bodies = StoredData("NodeRegistry factories\nthe only source\nof a node body")
            bodies >> Edge(color=EXEC, style="dotted", label="builds") >> mat

        with cluster("⑥ GOVERNED EXECUTION  ·  grapharc/runtime", EXEC):
            kernel = Action(
                "kernel\ntyped state · declared writes\nbudget meter · validated routers"
            )
            agent = Action("agent nodes\nobserve → act → verify")
            ver = Inspection("verifier nodes\nanchored to evidence")
            memn = Database("memory nodes\nrecall / persist")
            kernel >> Edge(color=EXEC) >> agent >> Edge(color=EXEC) >> ver
            ver >> Edge(color=EXEC) >> memn

        with cluster("⑦ OUTCOME", OUTCOME):
            res = InputOutput("answer + artifacts")
            dur = Database("durable memory\nclaims with provenance")
            aud = Document("replayable audit trail\nJSONL — one record")

        for t in triggers:
            t >> Edge(color=TRIGGER) >> sess
        appr >> Edge(color=SESSION) >> planner
        planner >> Edge(color=PLAN, label="proposal") >> check
        check >> Edge(label="ADMITTED", color=GATE, penwidth="2.4") >> mat
        mat >> Edge(color=EXEC, penwidth="2.4", label="compiled graph") >> kernel

        memn >> Edge(color=OUTCOME) >> res
        memn >> Edge(color=OUTCOME) >> dur
        memn >> Edge(color=OUTCOME) >> aud

        # The two edges that make this a loop rather than a pipeline. Both are
        # unconstrained so they do not drag the ranks they point back into.
        rej >> Edge(
            color=REJECT,
            style="dashed",
            constraint="false",
            label="traced reason\nfeeds replanning",
        ) >> planner
        memn >> Edge(
            color=PLAN,
            penwidth="2.4",
            constraint="false",
            label="work discovered mid-run RE-ENTERS the gate\n"
            "no already-approved path, no cached authorisation",
        ) >> planner


def planes() -> None:
    """§3 — execution is the thin layer; the planes are where promises bind."""
    with Diagram(
        "GraphARC — planes: what every node sits on, and what constrains it",
        filename=str(OUT / "02-planes"),
        show=False,
        direction="TB",
        outformat="png",
        graph_attr=GRAPH_ATTR,
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with cluster("POLICY & GOVERNANCE  ·  grapharc/policy", GATE):
            rules = Document("declarative TOML rules over\nnodes · edges · tools · spend")
            approute = Action("approval routing")
            tenant = Action("multi-tenant scoping")

        with cluster("EXECUTION  ·  grapharc/runtime", EXEC):
            n1 = Action("node")
            n2 = Action("node")
            n3 = Action("node")

        with cluster("MODEL PLANE  ·  grapharc/gateway", MODEL):
            gw = Action("Claude CLI\ntext completion only\nno bind_tools\nby design")
            orr = Action("OpenRouter\ntool calling · structured output")
            fo = Action("retries, failover,\nspend ceilings")

        with cluster("TOOL PLANE  ·  grapharc/harness + grapharc/tools", TOOL):
            regt = StoredData("registry\nseven core tools")
            perm = Decision("permissions\ndeny → ask → allow")
            sbx = Action(
                "executors\naudit-hook sandbox (in-process)\n"
                "ContainerExecutor (real boundary)"
            )
            regt >> Edge(color=TOOL) >> perm >> Edge(color=TOOL) >> sbx

        with cluster("MEMORY PLANE  ·  grapharc/memory", MEMORY):
            claim = Database("claims with provenance\nsuperseded, never overwritten")
            retr = Action("traversal + retrieval,\ntoken-budgeted")
            contra = Inspection("contradiction detection")
            claim >> Edge(color=MEMORY) >> retr >> Edge(color=MEMORY) >> contra

        with cluster("OBSERVABILITY  ·  grapharc/observe", OUTCOME):
            tr = Document("JSONL traces\nas replay points")
            met = Action("metrics + cost\nattribution")
            rp = Action("replay · diff · OTel")
            tr >> Edge(color=OUTCOME) >> met >> Edge(color=OUTCOME) >> rp

        n2 >> Edge(color=MODEL, label="model calls") >> gw
        gw >> Edge(color=MODEL) >> orr >> Edge(color=MODEL) >> fo
        n2 >> Edge(color=TOOL, label="tool calls") >> regt
        n2 >> Edge(color=MEMORY, label="recall / persist") >> claim
        n2 >> Edge(color=OUTCOME, style="dotted", label="every step") >> tr

        for source, target in ((rules, n1), (approute, n2), (tenant, n3)):
            source >> Edge(color=GATE, style="dashed", label="constrains") >> target


def agent_node() -> None:
    """§4 — the reusable unit. Every gate here is code, not prompt text."""
    with Diagram(
        "GraphARC — inside an agent node: every gate is code, not prompt text",
        filename=str(OUT / "03-agent-node"),
        show=False,
        direction="LR",
        outformat="png",
        graph_attr={**GRAPH_ATTR, "ranksep": "0.9"},
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        start = StartEnd("task + context")
        recall = Database("recall memory")
        model = Action("model call\nvia the gateway")
        wants = Decision("wants a tool?")
        perm = Decision("permitted?\ndeny → ask → allow")
        ask = ManualInput("human approval")
        sbx = Action("sandboxed execution\nworkspace-confined")
        budget = Decision("budget remains?")
        verify = Decision("verified against\nevidence?")
        out = StartEnd("result + provenance")
        stop = StartEnd("stop with a\nrecorded reason")

        start >> Edge(color=EXEC) >> recall >> Edge(color=EXEC) >> model
        model >> Edge(color=EXEC) >> wants
        wants >> Edge(label="no", color=EXEC) >> verify
        wants >> Edge(label="yes", color=TOOL) >> perm
        perm >> Edge(label="deny — never reaches the model", color=REJECT, style="dashed") >> model
        perm >> Edge(label="ask", color=PLAN) >> ask >> Edge(color=TOOL) >> sbx
        perm >> Edge(label="allow", color=TOOL) >> sbx
        sbx >> Edge(color=EXEC) >> budget
        budget >> Edge(label="yes", color=EXEC) >> model
        budget >> Edge(label="no", color=REJECT) >> stop
        verify >> Edge(label="no", color=REJECT, style="dashed") >> model
        verify >> Edge(label="yes", color=OUTCOME) >> out


def subsystems() -> None:
    """The twelve packages, drawn from the import graph rather than intent."""
    with Diagram(
        "GraphARC — twelve subsystems, drawn from what actually imports what",
        filename=str(OUT / "04-subsystems"),
        show=False,
        direction="TB",
        outformat="png",
        graph_attr=GRAPH_ATTR,
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with cluster("ENTRY POINTS", TRIGGER):
            cli = Action("cli/  1849\nten commands\n--json on each")
            server = Action("server/  1320\nFastAPI + SSE\nown sessions (gap)")
            examples = Action("examples/  1445\nstages 0-6 · capstone\nagent_fixit\nplan_incident")

        with cluster("CORE", EXEC):
            runtime = Action(
                "runtime/  1960\ntyped state, declared writes,\nbudgets, async, traces"
            )
            session = Action("session/  2084\nresume, interrupt, approval")

        with cluster("PLANES", TOOL):
            harness = Action("harness/  2189\ntools, permissions,\nsandbox, AgentNode")
            gateway = Action("gateway/  1223\nClaude CLI\n+ OpenRouter")
            memory = Action("memory/  2165\nclaims, SQLite, traversal")
            tools = Action("tools/  1054\nseven core tools")
            policy = Action("policy/  908\nTOML rules\n→ admission gate")
            observe = Action("observe/  1801\ntraces, replay, metrics, OTel")

        with cluster("THE GOVERNED LOOP  ·  now reachable", PLAN):
            planner = Action(
                "planner/  2696  ← the largest subsystem\n"
                "proposal · admission · materialize · loop"
            )
            surface = Action(
                "grapharc plan <goal>\n"
                "cli/plan.py + examples/plan_incident.py\n"
                "--registry swaps the kinds\n"
                "--policy compiles the TOML into the gate"
            )
            surface >> Edge(color=PLAN, penwidth="2.0", label="drives") >> planner

        for entry in (cli, server, examples):
            entry >> Edge(color=TRIGGER) >> runtime
        cli >> Edge(color=TRIGGER) >> harness
        server >> Edge(color=TRIGGER) >> session

        session >> Edge(color=EXEC) >> runtime
        runtime >> Edge(color=EXEC) >> observe
        runtime >> Edge(color=EXEC) >> memory
        harness >> Edge(color=TOOL) >> gateway
        harness >> Edge(color=TOOL) >> tools
        harness >> Edge(color=TOOL) >> policy

        # The arrows that used to be missing: something now points *into* planner.
        cli >> Edge(color=PLAN, penwidth="2.0", constraint="false", label="plan") >> surface
        planner >> Edge(color=PLAN, style="dashed", constraint="false") >> runtime
        planner >> Edge(color=PLAN, style="dashed", constraint="false") >> observe
        planner >> Edge(color=PLAN, style="dashed", constraint="false") >> policy


if __name__ == "__main__":
    lifecycle()
    planes()
    agent_node()
    subsystems()
    for png in sorted(OUT.glob("*.png")):
        print(f"wrote {png.relative_to(pathlib.Path.cwd())}  ({png.stat().st_size:,} bytes)")
