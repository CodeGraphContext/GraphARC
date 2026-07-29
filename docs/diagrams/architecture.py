"""Render GraphARC's architecture as PNGs with `diagrams` (mingrammer).

Five views of the same system, from the trust boundary outwards:

  1. lifecycle   — trigger to outcome, including the replan loop (§1, §2)
  2. planes      — what every node sits on, and what constrains it (§3)
  3. agent-node  — the reusable unit, gate by gate (§4)
  4. subsystems  — the twelve packages, and which ones actually import which

View 4 is drawn from the import graph rather than from intent. It used to show
`planner/` sitting unwired — nothing in the tree imported it — the largest gap
this project had; `grapharc plan` closed it, and `policy/` now reaches
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
from diagrams.generic.storage import Storage
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.monitoring import Grafana
from diagrams.onprem.security import Vault
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
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

OUT = pathlib.Path(__file__).parent

# Palette. One colour per plane, reused across all five views so the same idea
# is the same colour wherever it appears.
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
            cli = ManualInput("grapharc plan / run / demo / agent")
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
            cli = Action("cli/  2711\neleven commands\n--json on each")
            server = Action("server/  1320\nFastAPI + SSE\nown sessions (gap)")
            examples = Action("examples/  1450\nstages 0-6 · capstone\nagent_fixit · plan_incident")

        with cluster("CORE", EXEC):
            runtime = Action(
                "runtime/  1960\ntyped state, declared writes,\nbudgets, async, traces"
            )
            session = Action("session/  2084\nresume, interrupt, approval")

        with cluster("PLANES", TOOL):
            harness = Action("harness/  2190\ntools, permissions,\nsandbox, AgentNode")
            gateway = Action("gateway/  1679\nClaude CLI \u00b7 OpenRouter\nOpenAI \u00b7 Ollama")
            memory = Action("memory/  2530\nclaims, SQLite, traversal")
            tools = Action("tools/  1054\nseven core tools")
            policy = Action("policy/  908\nTOML rules\n→ admission gate")
            observe = Action("observe/  1801\ntraces, replay, metrics, OTel")

        with cluster("THE GOVERNED LOOP  ·  now reachable", PLAN):
            planner = Action(
                "planner/  2696\n"
                "proposal · admission · materialize · loop"
            )
            surface = Action(
                "grapharc plan <goal>   a model proposes\n"
                "grapharc run graph.json   you wrote it\n"
                "        --check-only   admission as a linter\n\n"
                "--registry swaps the kinds (stdlib ships some)\n"
                "--policy compiles a TOML into the gate\n"
                "grapharc.toml supplies either as a default"
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


def trust_boundary() -> None:
    """The architecture at its most honest: who supplies what, and the gate.

    The other four views show structure. This one shows the *boundary* — the
    only question that decides whether the rest is a safety argument or
    decoration. Everything an operator authors happens before a model runs. The
    model contributes exactly one thing: JSON naming kinds and edges.

    Kept deliberately sparse. An earlier version routed every operator input to
    the gate and collided its own labels; the secondary inputs are stated inside
    the cluster instead of drawn, because a diagram that has to be decoded is
    not doing its job.
    """
    with Diagram(
        "GraphARC — the trust boundary: an operator declares, a model proposes, a checker decides",
        filename=str(OUT / "05-trust-boundary"),
        show=False,
        direction="LR",
        outformat="png",
        graph_attr={**GRAPH_ATTR, "ranksep": "1.4", "nodesep": "0.5"},
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with cluster("① THE OPERATOR DECLARES  ·  before any model runs", TOOL):
            registry = StoredData(
                "registry.py  ·  CODE\n"
                "a name -> a real function\n"
                "'patch' = an agent with edit_file,\n"
                "a token cap, one writable field\n"
                "absence is refusal, no wildcard"
            )
            policy = Document(
                "policy.toml  ·  DATA\n"
                "which transitions are permitted\n"
                "deny > ask > allow; unmatched denies\n\n"
                "named by a flag, by grapharc.toml,\n"
                "or — with none of those — GENERATED\n"
                "once, written to .grapharc/, and\n"
                "disclosed as policy_source"
            )
            Document(
                "also the operator's, not drawn:\n"
                "the state schema, Budget, LoopLimits,\n"
                "and grapharc.toml — which supplies any\n"
                "of these as a default.\n"
                "flag > env > grapharc.toml > built-in;\n"
                "parent directories are never searched"
            )

        with cluster("② THE MODEL PROPOSES  ·  nothing else", PLAN):
            proposal = InputOutput(
                "a Subgraph, as JSON\n"
                '{"nodes": [{"name": "triage"}],\n'
                ' "edges": [...]}\n'
                "names and edges — the whole surface"
            )
            Document(
                "CANNOT SUPPLY\n"
                "a node body (extra='forbid')\n"
                "a callable (not JSON-serialisable,\n"
                "so it never reaches the gate)\n"
                "its own budget or approval\n"
                "a new kind (the registry is frozen)"
            )

        with cluster("③ THE CHECKER DECIDES  ·  deterministic, model-free", GATE):
            gate = Decision(
                "AdmissionChecker.check()\n"
                "all five, every round:\n"
                "kind registered? edge permitted?\n"
                "worst case within REMAINING budget?\n"
                "depth? acyclic?"
            )
            refused = Document(
                "REFUSED\n"
                "every objection at once,\n"
                "each a typed reason code.\n"
                "No factory ran. Nothing was built."
            )

        with cluster("④ ONLY THEN IS ANYTHING BUILT", EXEC):
            mat = PredefinedProcess(
                "Materializer\n"
                "authorisation is argument ONE,\n"
                "hash-matched to the proposal.\n"
                "Bodies looked up in the registry —\n"
                "no code is ever generated"
            )
            kernel = Action(
                "the kernel runs it\n"
                "declared writes · deep-copy isolation\n"
                "budget metered without the node's help"
            )

        with cluster("⑤ ONE RECORD", OUTCOME):
            trace = Document(
                "trace.jsonl\n"
                "what it did\n"
                "what it was\n"
                "ALLOWED to do\n"
                "why it stopped"
            )

        registry >> Edge(
            color=TOOL, penwidth="2.0", label="the only source\nof a node body"
        ) >> gate
        policy >> Edge(color=TOOL, label="decides edges") >> gate
        proposal >> Edge(color=PLAN, penwidth="2.2", label="names only") >> gate

        gate >> Edge(label="  no", color=REJECT, style="dashed") >> refused
        gate >> Edge(taillabel="  ADMITTED  ", color=GATE, penwidth="2.4") >> mat
        mat >> Edge(color=EXEC, penwidth="2.2") >> kernel
        kernel >> Edge(color=OUTCOME, penwidth="2.0") >> trace

        # The loop. Unconstrained so the back-edges do not reorder the ranks.
        refused >> Edge(color=REJECT, style="dashed", constraint="false",
                        label="reason codes feed\nthe next proposal") >> proposal
        kernel >> Edge(color=PLAN, penwidth="2.4", constraint="false",
                       label="work discovered mid-run RE-ENTERS the gate\n"
                             "no already-approved path") >> proposal




def architecture() -> None:
    """The canonical view: the whole system in one frame.

    The other five views each answer one question. This is the one for the top of
    a README.

    Four decisions about how it is drawn, each learned by drawing it badly first:

    - **Restraint over completeness.** The first attempt had twenty nodes and
      twenty-five edges, and the back-edges swept across the whole page. Bands of
      three or four, and exactly two feedback edges.
    - **Labels are two to four words.** The prose belongs in the README.
    - **Icons are semantic.** Docker *is* the real sandbox boundary; Vault *is*
      the policy document; a diamond *is* a decision. An earlier draft used the
      Postgres elephant for the claim store, which implies a dependency this
      project does not have — memory is SQLite. A wrong icon is a wrong claim.
    - **The gate is on the critical path, drawn heaviest.** Everything above it
      proposes; nothing below it runs without passing through.
    """
    with Diagram(
        "GraphARC — a governed agent runtime",
        filename=str(OUT / "00-architecture"),
        show=False,
        # LR, not TB. A banded top-to-bottom layout is a 2D grid, and Graphviz
        # is a hierarchical engine — asking for bands made the operator's inputs
        # sweep around the outside and cross their own cluster titles. Left to
        # right gives the spine one rank per stage and lets what the operator
        # supplies sit above it, and the planes below it, without crossings.
        direction="LR",
        outformat="png",
        graph_attr={**GRAPH_ATTR, "fontsize": "26", "ranksep": "1.3", "nodesep": "0.7"},
        node_attr={**NODE_ATTR, "fontsize": "12"},
        edge_attr=EDGE_ATTR,
    ):
        with cluster("WHAT STARTS A RUN", TRIGGER):
            cli = Python("grapharc CLI")
            api = Fastapi("HTTP API")

        with cluster("THE OPERATOR DECLARES  ·  before any model runs", TOOL):
            registry = Storage("registry\nname -> function")
            policy = Vault("policy.toml\npermitted edges")

        with cluster("THE CONTROL PLANE  ·  propose, admit, build", GATE):
            planner = PredefinedProcess("planner\nemits typed JSON")
            gate = Decision("ADMISSION\nregistered? permitted?\nin budget? acyclic?")
            refused = Document("REFUSED\nevery reason at once")
            builder = PredefinedProcess("materialise\nbound by hash")
            planner >> Edge(color=PLAN, penwidth="2.6", label="proposal") >> gate
            gate >> Edge(label=" no", color=REJECT, style="dashed") >> refused
            gate >> Edge(label="ADMITTED", color=GATE, penwidth="3.0") >> builder

        with cluster("THE GRAPH RUNS  ·  typed state, declared writes, metered", EXEC):
            kernel = Server("kernel")
            agent = Action("agent nodes")
            verifier = Inspection("verifiers")
            kernel >> Edge(color=EXEC) >> agent >> Edge(color=EXEC) >> verifier

        with cluster("THE PLANES EVERY NODE SITS ON", MODEL):
            models = Server("model plane\nClaude CLI · OpenRouter")
            tools = Docker("tool plane\ndeny > ask > allow")
            memory = Storage("memory plane\nSQLite · provenance")

        record_title = (
            "ONE RECORD  ·  what it did · what it was allowed to do · why it stopped"
        )
        with cluster(record_title, OUTCOME):
            trace = Document("trace.jsonl")
            readers = Grafana("metrics · replay\ndiff · viz")
            trace >> Edge(color=OUTCOME) >> readers

        cli >> Edge(color=TRIGGER, penwidth="2.4") >> planner
        api >> Edge(color=TRIGGER, penwidth="2.4") >> planner
        registry >> Edge(
            color=TOOL, penwidth="2.2", label="the only source\nof a node body"
        ) >> gate
        policy >> Edge(color=TOOL, penwidth="2.2") >> gate
        builder >> Edge(color=EXEC, penwidth="3.0") >> kernel
        agent >> Edge(color=MODEL, penwidth="2.0") >> models
        agent >> Edge(color=TOOL, penwidth="2.0") >> tools
        verifier >> Edge(color=MEMORY, penwidth="2.0") >> memory
        kernel >> Edge(color=OUTCOME, style="dotted") >> trace

        # The two edges that make this a runtime and not a pipeline. Both
        # unconstrained, or Graphviz reorders every rank above them.
        refused >> Edge(
            color=REJECT, style="dashed", constraint="false",
            label="reasons feed\nthe next proposal",
        ) >> planner
        verifier >> Edge(
            color=PLAN, penwidth="2.8", constraint="false",
            label="work found mid-run\nRE-ENTERS the gate",
        ) >> planner


if __name__ == "__main__":
    architecture()
    lifecycle()
    planes()
    agent_node()
    subsystems()
    trust_boundary()
    for png in sorted(OUT.glob("*.png")):
        print(f"wrote {png.relative_to(pathlib.Path.cwd())}  ({png.stat().st_size:,} bytes)")
