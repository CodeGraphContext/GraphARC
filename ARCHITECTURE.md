# GraphARC — the golden stage

The target architecture. This is what GraphARC is *trying to become*, not what
it is today ([ROADMAP.md](ROADMAP.md) tracks the distance; [ASSESSMENT.md](ASSESSMENT.md)
is honest about the gap). Read [VISION.md](VISION.md) first for the thesis.

**The one-sentence version:** a request enters, a planner *proposes* a subgraph,
an admission gate *authorizes* it, and only then does anything execute — so
every transition was permitted, every loop was bounded, and afterwards you can
prove exactly what happened and why it stopped.

---

## 1. End-to-end lifecycle

```mermaid
flowchart TB
    subgraph T["① TRIGGERS"]
        direction LR
        CLI["CLI"]:::e
        API["HTTP API"]:::e
        CRON["Schedule"]:::e
        HOOK["Webhook"]:::e
        CHAT["Chat channel"]:::e
    end

    subgraph S["② SESSION RUNTIME"]
        direction LR
        SESS["Create / resume<br/>long-lived session"]:::s
        EVQ["Event queue<br/>multi-turn input"]:::s
        STEER["Interrupt<br/>& steering"]:::s
    end

    subgraph P["③ PLAN"]
        PLANNER["Planner node<br/><b>proposes</b> a subgraph<br/>— never executes one"]:::p
    end

    subgraph A["④ ADMISSION — the governance gate"]
        direction TB
        REG{"every node<br/>registered?"}:::g
        POL{"every edge<br/>policy-permitted?"}:::g
        BUD{"worst case<br/>within budget?"}:::g
        DEP{"depth within<br/>limit?"}:::g
        REJ["REJECTED<br/>recorded as a first-class<br/>traced event"]:::r
    end

    subgraph X["⑤ GOVERNED EXECUTION GRAPH"]
        direction LR
        AGENT["Agent nodes<br/>observe → act → verify"]:::x
        TOOLN["Tool nodes"]:::x
        VER["Verifier nodes<br/>anchored to evidence"]:::x
        HUM["Human approval<br/>suspends the graph"]:::x
        RTR["Deterministic<br/>routers"]:::x
        MEMN["Memory<br/>recall / persist"]:::x
    end

    subgraph O["⑥ OUTCOME"]
        direction LR
        RES["Answer + artifacts"]:::o
        DUR["Durable memory<br/>with provenance"]:::o
        AUD["Replayable<br/>audit trail"]:::o
    end

    T --> S --> P --> REG
    REG -- no --> REJ
    REG -- yes --> POL
    POL -- no --> REJ
    POL -- yes --> BUD
    BUD -- no --> REJ
    BUD -- yes --> DEP
    DEP -- no --> REJ
    DEP -- yes --> X
    REJ -.->|"traced reason<br/>feeds replanning"| PLANNER
    X -->|"new work discovered"| PLANNER
    X --> O

    classDef e fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef s fill:#2d4a3e,stroke:#5cb85c,color:#fff
    classDef p fill:#4a3f2d,stroke:#d9a441,color:#fff
    classDef g fill:#5c2d2d,stroke:#d95c5c,color:#fff
    classDef r fill:#7a1f1f,stroke:#ff6b6b,color:#fff
    classDef x fill:#2d3a4a,stroke:#7aa5d2,color:#fff
    classDef o fill:#3a2d4a,stroke:#a57ad2,color:#fff
```

The loop from ⑤ back to ③ is the part that makes this general-purpose. Work
discovered mid-run doesn't bypass the gate — it re-enters through it.

---

## 2. The admission cycle — the crux

This is the component with no prior art to copy, and where most of the risk
lives. It is what lets the topology be dynamic without the governance being
advisory.

```mermaid
flowchart LR
    GOAL(["Goal or<br/>discovered work"]):::n
    PROP["Planner proposes:<br/>nodes + edges + budget"]:::p
    CHK{"Admission<br/>checker"}:::g
    RUN["Execute the<br/>admitted subgraph"]:::x
    OBS["Observe outcome<br/>+ remaining budget"]:::n
    DONE(["Terminate with a<br/>recorded reason"]):::o

    GOAL --> PROP --> CHK
    CHK -->|admitted| RUN --> OBS
    CHK -->|"rejected + reason"| PROP
    OBS -->|"more work,<br/>budget remains"| PROP
    OBS -->|"goal met / budget spent /<br/>no progress / human halt"| DONE

    classDef n fill:#2b2b2b,stroke:#888,color:#fff
    classDef p fill:#4a3f2d,stroke:#d9a441,color:#fff
    classDef g fill:#5c2d2d,stroke:#d95c5c,color:#fff
    classDef x fill:#2d3a4a,stroke:#7aa5d2,color:#fff
    classDef o fill:#3a2d4a,stroke:#a57ad2,color:#fff
```

**The invariant:** a node may *propose* nodes and edges; it may never *execute*
them directly. Dynamism lives in construction, governance lives in admission.

**Why this matters.** Static graphs can't handle "investigate this incident" —
the shape is discovered while working. Fully dynamic loops can't tell you what
they were allowed to do. This splits the difference: the planner is as free as
you like, and nothing it invents runs until a deterministic checker says yes.

---

## 3. Planes — what every node sits on

Execution is the thin layer. The planes below are cross-cutting, and each is a
place where a promise is *enforced* rather than requested.

```mermaid
flowchart TB
    subgraph EX["EXECUTION"]
        N1["node"]:::x
        N2["node"]:::x
        N3["node"]:::x
    end

    subgraph MP["MODEL PLANE"]
        direction LR
        GW["Gateway:<br/>tool-calling · structured output · streaming"]:::m
        RT["Routing:<br/>cost / latency / capability"]:::m
        FO["Failover +<br/>cost ceilings"]:::m
    end

    subgraph TP["TOOL PLANE"]
        direction LR
        REGT["Registry +<br/>progressive disclosure"]:::t
        PERM["Permissions:<br/>deny → ask → allow"]:::t
        SBX["Sandboxed executor<br/>fs · shell · browser · net · MCP"]:::t
    end

    subgraph MEM["MEMORY PLANE"]
        direction LR
        CLAIM["Claims with provenance<br/>superseded, never overwritten"]:::mm
        ART["Artifact store"]:::mm
        RETR["Hybrid retrieval,<br/>token-budgeted"]:::mm
    end

    subgraph GOV["POLICY & GOVERNANCE"]
        direction LR
        RULES["Declarative rules over<br/>nodes · edges · tools · spend"]:::g
        APPR["Approval routing"]:::g
        TEN["Multi-tenant scoping"]:::g
    end

    subgraph OBS2["OBSERVABILITY"]
        direction LR
        TR["Traces as replay points"]:::o
        MET["Metrics + cost attribution"]:::o
        RP["Replay · rollback · versioning"]:::o
    end

    EX --> MP
    EX --> TP
    EX --> MEM
    GOV -.->|constrains| EX
    GOV -.->|constrains| TP
    GOV -.->|constrains| MP
    EX --> OBS2

    classDef x fill:#2d3a4a,stroke:#7aa5d2,color:#fff
    classDef m fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef t fill:#2d4a3e,stroke:#5cb85c,color:#fff
    classDef mm fill:#4a3f2d,stroke:#d9a441,color:#fff
    classDef g fill:#5c2d2d,stroke:#d95c5c,color:#fff
    classDef o fill:#3a2d4a,stroke:#a57ad2,color:#fff
```

---

## 4. Inside an agent node

The reusable unit. Every gate here is code, not prompt text.

```mermaid
flowchart LR
    IN(["task + context"]):::n
    RECALL["Recall<br/>memory"]:::mm
    MODEL["Model call<br/>via gateway"]:::m
    WANT{"wants a<br/>tool?"}:::g
    PERM{"permitted?"}:::g
    ASK["Human<br/>approval"]:::r
    SBX["Sandboxed<br/>execution"]:::t
    VER{"verified against<br/>evidence?"}:::g
    BUD{"budget<br/>remains?"}:::g
    OUT(["result + provenance"]):::o
    STOP(["stop with<br/>recorded reason"]):::o

    IN --> RECALL --> MODEL --> WANT
    WANT -- no --> VER
    WANT -- yes --> PERM
    PERM -- deny --> MODEL
    PERM -- ask --> ASK --> SBX
    PERM -- allow --> SBX
    SBX --> BUD
    BUD -- yes --> MODEL
    BUD -- no --> STOP
    VER -- no --> MODEL
    VER -- yes --> OUT

    classDef n fill:#2b2b2b,stroke:#888,color:#fff
    classDef m fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef t fill:#2d4a3e,stroke:#5cb85c,color:#fff
    classDef mm fill:#4a3f2d,stroke:#d9a441,color:#fff
    classDef g fill:#5c2d2d,stroke:#d95c5c,color:#fff
    classDef r fill:#7a1f1f,stroke:#ff6b6b,color:#fff
    classDef o fill:#3a2d4a,stroke:#a57ad2,color:#fff
```

---

## 5. A worked example: "refactor this repo and run tests"

What the golden stage does with a real task.

```mermaid
flowchart TB
    REQ(["refactor this repo<br/>and run tests"]):::n
    P1["Planner: inspect repo,<br/>propose a search fan-out"]:::p
    A1{"admit"}:::g
    INSP["Inspect + index"]:::x
    P2["Planner: propose one<br/>edit-worker per module"]:::p
    A2{"admit<br/>· workers ≤ cap<br/>· budget covers worst case<br/>· edit tool permitted"}:::g
    W1["worker: module A"]:::x
    W2["worker: module B"]:::x
    W3["worker: module C"]:::x
    APPLY["Apply edits<br/>in sandbox"]:::t
    TEST["Run tests"]:::t
    PASS{"green?"}:::g
    DIAG["Diagnose failures"]:::x
    P3["Planner: propose<br/>a repair subgraph"]:::p
    A3{"admit<br/>+ retry budget"}:::g
    HUM["Human approval<br/>to touch main"]:::r
    VERD["Verify diff<br/>against the goal"]:::x
    REPORT(["Report + durable<br/>provenance"]):::o
    HALT(["Halt: budget spent<br/>or no progress"]):::o

    REQ --> P1 --> A1 -->|yes| INSP --> P2 --> A2
    A2 -->|yes| W1 & W2 & W3 --> APPLY --> TEST --> PASS
    PASS -- yes --> VERD --> HUM --> REPORT
    PASS -- no --> DIAG --> P3 --> A3
    A3 -->|yes| APPLY
    A3 -->|"no — retries exhausted"| HALT

    classDef n fill:#2b2b2b,stroke:#888,color:#fff
    classDef p fill:#4a3f2d,stroke:#d9a441,color:#fff
    classDef g fill:#5c2d2d,stroke:#d95c5c,color:#fff
    classDef x fill:#2d3a4a,stroke:#7aa5d2,color:#fff
    classDef t fill:#2d4a3e,stroke:#5cb85c,color:#fff
    classDef r fill:#7a1f1f,stroke:#ff6b6b,color:#fff
    classDef o fill:#3a2d4a,stroke:#a57ad2,color:#fff
```

Note what is *not* in this diagram: nowhere does the model decide it may write
to disk, spawn twenty workers, or touch `main`. Those are admission decisions.

---

## 6. The five invariants

If the golden stage means anything, it means these hold — and each is testable.

| # | Invariant | Enforced by |
|---|---|---|
| 1 | No transition executes that the graph did not permit | Admission checker (§2) |
| 2 | No work happens that the budget did not authorize | Budget meters, checked *and* enforced |
| 3 | A node may propose, never directly execute, new topology | Planner / admission split |
| 4 | Every rejection, stop, and refusal is recorded with a reason | Traces as first-class events |
| 5 | Any run can be replayed and attributed after the fact | Checkpoints + trace + versioned config |

**The test of the whole design:** after an incident you must be able to answer
three questions from the record alone — *what did it do*, *what was it allowed
to do*, and *why did it stop*. Every mainstream harness today answers the first
one well, the second one vaguely, and the third one not at all.

---

## 7. Where we are against this

| Stage | Status |
|---|---|
| ① Triggers | demo CLI only |
| ② Session runtime | not started |
| ③ Planner | not started |
| ④ **Admission** | **not started — the crux** |
| ⑤ Execution graph | kernel ~70%; agent nodes 0% |
| ⑥ Outcome | traces yes; durable memory no |
| Model plane | ~15% — text-only, no tool-calling |
| Tool plane | ~30% — built, wired to nothing |
| Memory plane | ~20% — dies with the process |
| Policy | not started |
| Observability | ~20% |

The shortest path onto this diagram is §4 — the agent node — because it is the
first box that requires the model plane, tool plane, and kernel to work
together at all. See [ROADMAP.md](ROADMAP.md).
