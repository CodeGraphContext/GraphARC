# Verification and memory

Two subsystems, one idea: a model's confidence is not evidence. `grapharc.runtime.verify`
puts a deterministic check in front of the reviewing model so a hallucinated citation costs
nothing, and `grapharc.memory` records every fact with where it came from so a correction
adds history instead of erasing it.

Every snippet below was run with the repo's `.venv/bin/python` and the output pasted back
unedited. Where a value is genuinely random — a claim id, a timestamp, a pid — it is called
out; everything else reproduces. One snippet, clearly marked, calls a real model and was
*not* run.

---

## How do I reject a fabricated citation without paying for a model call?

`verify_claim` checks that the citation literally exists in the source *before* it prompts
the reviewer. A quote the model invented never reaches the reviewer at all.

```python
from grapharc.runtime.verify import verify_claim
from grapharc.testing import ScriptedChatModel

SOURCE = """GraphARC wraps LangGraph with typed state contracts.
Budgets put hard ceilings on iterations, tokens, and wall-clock time.
"""

# A reviewer that would approve anything — the point is that it is never asked.
reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "looks right"}'])

verdict = verify_claim(
    reviewer,
    text="GraphARC was benchmarked at 10x faster than raw LangGraph",
    citation="benchmarked at 10x faster",
    source_text=SOURCE,
)

print("accepted:      ", verdict.accepted)
print("anchor_ok:     ", verdict.anchor_ok)
print("model_accepted:", verdict.model_accepted)
print("reason:        ", verdict.reason)
print("reviewer calls:", reviewer.call_count)
```

```text
accepted:       False
anchor_ok:      False
model_accepted: None
reason:         citation does not exist in the source (deterministic anchor)
reviewer calls: 0
```

`reviewer.call_count == 0` is the whole recipe. The reviewer was configured to rubber-stamp
anything and it did not get the chance.

**Why it works this way.** `model_accepted` is `None`, not `False`, and the distinction is
load-bearing: `None` means *never consulted*, `False` means *consulted and unconvinced*.
When you are auditing a batch of rejections, that field tells you which ones were free.

The anchor also refuses citations under `MIN_CITATION_CHARS` (12) before doing anything
else — `"the"` appears in every document, so it anchors nothing.

---

## How do I keep a line-wrapped source from causing a false reject?

Real sources have newlines in the middle of sentences. A model that quotes one perfectly
still writes a space where the file has a `\n`, and exact matching would reject correct
work. The anchor collapses whitespace — and nothing else.

```python
from grapharc.runtime.verify import verify_claim
from grapharc.testing import ScriptedChatModel

# A line-wrapped source, as any real document is.
SOURCE = (
    "GraphARC is a toolkit built on LangGraph. Every node declares\n"
    "which state fields it may write, and an undeclared write raises."
)

reviewer = ScriptedChatModel(
    responses=['{"supported": true, "reason": "the quote states it"}'],
    on_exhausted="repeat",
)

quotes = [
    "Every node declares which state fields it may write",   # newline -> space
    "Every node  declares   which state fields",              # extra spaces
    "Each node declares which state fields it may write",     # paraphrase
    "Every node declares which state field it may write",     # one letter off
]

for quote in quotes:
    v = verify_claim(
        reviewer, text="Nodes declare their writes", citation=quote, source_text=SOURCE
    )
    print(f"{v.anchor_ok!s:<5}  {quote!r}")

print("verbatim in source:", "Every node declares which state fields it may write" in SOURCE)
print("reviewer calls:    ", reviewer.call_count)
```

```text
True   'Every node declares which state fields it may write'
True   'Every node  declares   which state fields'
False  'Each node declares which state fields it may write'
False  'Every node declares which state field it may write'
verbatim in source: False
reviewer calls:     2
```

The first quote is not in the source as a Python substring, and the anchor accepts it. The
third and fourth differ by one word and one letter, and the anchor refuses both — without a
model call, which is why the count is 2 and not 4.

**Why it works this way.** The citation is split on whitespace and rejoined as a regex with
`\s+` between the tokens (`_find_span`). Every non-space character still has to match
exactly, so the latitude is bounded to the one artifact that actually causes false rejects.
A paraphrase is not "close enough" — it is a different string, and the anchor's job is
existence, not similarity.

---

## How do I catch a quote lifted out of a negated sentence?

The anchor proves a quote exists, not that it means what the claim says. `"benchmarked at
10x faster"` really does appear in `"was never benchmarked at 10x faster"`. So the reviewer
is shown a mechanically extracted window of surrounding source.

```python
from grapharc.runtime.verify import verify_claim
from grapharc.testing import ScriptedChatModel

SOURCE = (
    "GraphARC was never benchmarked at 10x faster than raw LangGraph; "
    "no such measurement exists."
)

reviewer = ScriptedChatModel(
    responses=['{"supported": false, "reason": "the sentence negates the quote"}']
)

verdict = verify_claim(
    reviewer,
    text="GraphARC was benchmarked at 10x faster than raw LangGraph",
    citation="benchmarked at 10x faster than raw LangGraph",
    source_text=SOURCE,
)

print("anchor_ok:", verdict.anchor_ok, "| accepted:", verdict.accepted)
print("reason:   ", verdict.reason)
print("--- the one message the reviewer received ---")
(call,) = reviewer.calls
print(call[0].content)
```

```text
anchor_ok: True | accepted: False
reason:    the sentence negates the quote
--- the one message the reviewer received ---
You are verifying a single claim against its cited evidence. Judge ONLY whether the evidence supports the claim. The surrounding source context is included: if it negates, reverses, or otherwise contradicts what the quote alone implies, the claim is NOT supported.
Claim: GraphARC was benchmarked at 10x faster than raw LangGraph
Evidence (verbatim from source): benchmarked at 10x faster than raw LangGraph
Surrounding source context: GraphARC was never benchmarked at 10x faster than raw LangGraph; no such measurement exists.
Reply with JSON: {"supported": true|false, "reason": "..."}
```

That is the reviewer's entire context: one `HumanMessage`, no system prompt, no history.

**Why it works this way.** The window is `CONTEXT_WINDOW_CHARS` (240) either side of the
match, sliced straight out of `source_text`. Two properties fall out of that. It is
*mechanical*, so no part of the author's conversation can leak in — the reviewer cannot be
persuaded by the reasoning that produced the claim. And it is *bounded*, so verifying a
claim against a 400 KB document costs a fixed-size prompt.

The sharp edge is the same 240 characters: a negation that lands 300 characters away from
the quote is outside the window and the reviewer will not see it. This catches quote-mining
within a sentence or a paragraph, not a document-scale reversal.

---

## What happens when the reviewer replies with something that isn't a verdict?

Every ambiguity resolves toward rejection. This matters more than it sounds: `bool("false")`
is `True` in Python, so a model that quotes its boolean would otherwise be an accept.

```python
from grapharc.runtime.verify import verify_claim
from grapharc.testing import ScriptedChatModel

SOURCE = "Budgets put hard ceilings on iterations, tokens, and wall-clock time."
CITATION = "Budgets put hard ceilings on iterations"

for reply in ['sounds right to me!', '{"supported": "false"}', '{"supported": "true"}',
              '{"supported": 1}', '{"verdict": true}', '{"supported": true}']:
    reviewer = ScriptedChatModel(responses=[reply])
    v = verify_claim(reviewer, text="Budgets bound work",
                     citation=CITATION, source_text=SOURCE)
    print(f"{v.accepted!s:<5} {reply!r:<26} {v.reason or '(reviewer gave no reason)'}")
```

```text
False 'sounds right to me!'      reviewer reply unparseable — failing closed
False '{"supported": "false"}'   reviewer 'supported' was not a JSON boolean — failing closed
False '{"supported": "true"}'    reviewer 'supported' was not a JSON boolean — failing closed
False '{"supported": 1}'         reviewer 'supported' was not a JSON boolean — failing closed
False '{"verdict": true}'        reviewer reply unparseable — failing closed
True  '{"supported": true}'      (reviewer gave no reason)
```

Only a real JSON `true` accepts. `1` does not, `"true"` does not.

**Why it works this way.** `isinstance(raw_supported, bool)` rather than a truthiness test.
The cost is real — a model that reliably answers `"supported": "yes"` will have every claim
rejected — and that is the trade the module takes deliberately: a systematic false-reject is
visible in the stats below, a systematic false-accept is not.

---

## How do I tell whether my verifier is actually any good?

A verifier that rejects everything has zero false accepts. It also has zero value.
`evaluate_verdicts` scores against known ground truth and keeps the two error types apart.

```python
from grapharc.runtime.verify import evaluate_verdicts, verify_claim
from grapharc.testing import ScriptedChatModel

SOURCE = """GraphARC wraps LangGraph with typed state contracts.
Budgets put hard ceilings on iterations, tokens, and wall-clock time.
"""

CLAIMS = [
    ("GraphARC enforces typed state contracts", "typed state contracts"),
    ("GraphARC was benchmarked at 10x faster", "benchmarked at 10x faster"),
    ("Budgets make graphs run faster",
     "Budgets put hard ceilings on iterations, tokens, and wall-clock time."),
]
TRUTH = {
    "GraphARC enforces typed state contracts": True,
    "GraphARC was benchmarked at 10x faster": False,
    "Budgets make graphs run faster": False,
}


def run(reviewer):
    return evaluate_verdicts(
        [verify_claim(reviewer, text=t, citation=c, source_text=SOURCE) for t, c in CLAIMS],
        TRUTH,
    )


good = ScriptedChatModel(responses=[
    '{"supported": true,  "reason": "the source says exactly this"}',
    '{"supported": false, "reason": "ceilings bound work, they do not speed it up"}',
])
paranoid = ScriptedChatModel(responses=['{"supported": false, "reason": "no"}'],
                             on_exhausted="repeat")

print("good verifier: ", run(good).model_dump())
print("rejects-all:   ", run(paranoid).model_dump())
```

```text
good verifier:  {'true_accepts': 1, 'true_rejects': 2, 'false_accepts': 0, 'false_rejects': 0}
rejects-all:    {'true_accepts': 0, 'true_rejects': 2, 'false_accepts': 0, 'false_rejects': 1}
```

Both verifiers have `false_accepts == 0`. Only one of them ever accepts a true claim. Track
a single "accuracy" number and these two look similar; track the four cells and the
rejects-all verifier is obviously broken — `true_accepts == 0` with a non-empty positive
class is the tell.

**Why it works this way.** `VerifierStats` is four counters, not a score, because the two
error types have different costs and different owners. A false accept ships a wrong claim; a
false reject burns a retry loop and eventually makes people turn verification off.

Note the second verifier is asked three times and the first only twice — the fabricated
claim never reaches either, so `good`'s two-response script is exactly enough. Hence
`on_exhausted="repeat"` on the paranoid one.

Sharp edge: `evaluate_verdicts` indexes `ground_truth` by `claim_text` and raises `KeyError`
for a claim it does not know about. That is deliberate — silently scoring an unknown claim
as anything would corrupt the counts — but it means the ground-truth dict has to cover every
claim you pass, and two claims with identical text collapse into one key.

---

## How do I make sure the reviewer is genuinely a different model?

Two enforcement points, and they are not equally strong. Knowing which is which is the
recipe.

```python
from grapharc.examples.stage5_verifier import build_stage5
from grapharc.gateway import different_providers
from grapharc.testing import ScriptedChatModel

model = ScriptedChatModel(responses=["x"])
try:
    build_stage5(model, model)
except ValueError as exc:
    print("refused:", exc)

# Two separate instances of the same model pass that check — it is identity only.
twin_a = ScriptedChatModel(responses=["x"])
twin_b = ScriptedChatModel(responses=["x"])
build_stage5(twin_a, twin_b)
print("two instances of the same model: accepted by build_stage5")

# The stronger check works on gateway specs, and is what the CLI warns on.
pairs = [
    ("openrouter/anthropic/claude-haiku-4.5", "openrouter/anthropic/claude-sonnet-4.5"),
    ("openrouter/anthropic/claude-haiku-4.5", "openrouter/openai/gpt-4o-mini"),
    ("claude-cli/claude-sonnet-5", "openrouter/openai/gpt-4o-mini"),
]
for a, b in pairs:
    print(f"different_providers({a!r}, {b!r}) -> {different_providers(a, b)}")
```

```text
refused: author and reviewer must be different model instances: a model grading its own homework is the correlated-agreement failure mode
two instances of the same model: accepted by build_stage5
different_providers('openrouter/anthropic/claude-haiku-4.5', 'openrouter/anthropic/claude-sonnet-4.5') -> False
different_providers('openrouter/anthropic/claude-haiku-4.5', 'openrouter/openai/gpt-4o-mini') -> True
different_providers('claude-cli/claude-sonnet-5', 'openrouter/openai/gpt-4o-mini') -> True
```

`build_stage5` catches `author is reviewer` and nothing more. Line 2 of the output is the
gap stated plainly: two instances of the same weights sail through. If you want real
independence you have to check the specs yourself, or let the CLI do it.

Against real backends — **this snippet was not executed; it needs an OpenRouter key or a
Claude subscription and would spend money or quota:**

```python
# Requires a real model (API key or Claude subscription). Not executed by the docs test.
from grapharc.examples.stage5_verifier import DEMO_SOURCE, build_stage5
from grapharc.gateway import different_providers, get_model

author_spec = "openrouter/anthropic/claude-haiku-4.5"
reviewer_spec = "openrouter/openai/gpt-4o-mini"
if not different_providers(author_spec, reviewer_spec):
    raise SystemExit("author and reviewer share a vendor — correlated agreement")

compiled = build_stage5(get_model(author_spec), get_model(reviewer_spec))
print(compiled.invoke({"source_text": DEMO_SOURCE})["accepted"])
```

The CLI equivalent, which warns on a shared vendor rather than making you write the check:

```bash
grapharc demo stage5 --model openrouter/anthropic/claude-haiku-4.5 \
                    --reviewer-model openrouter/openai/gpt-4o-mini
```

**Why it works this way.** `different_providers` compares the *vendor* each spec reaches —
the model author when the id names one (`anthropic/…` vs `openai/…`), the backend's own
vendor otherwise — so two Anthropic models on OpenRouter come back `False` even though they
are different models, and so does a Claude-CLI author paired with an Anthropic model over
OpenRouter. Correlated agreement follows training lineage, not model names, and not which
API you happened to reach the model through.

---

## How do I wire verification into a graph?

`build_stage5` is a working three-node graph — draft, verify, report — and reading its
source (`grapharc/examples/stage5_verifier.py`) is the fastest way to see how `verify_claim`
sits inside a node.

```python
import json

from grapharc.examples.stage5_verifier import DEMO_SOURCE, build_stage5
from grapharc.testing import ScriptedChatModel

author = ScriptedChatModel(responses=[json.dumps({"claims": [
    {"text": "GraphARC enforces typed state contracts", "citation": "typed state contracts"},
    {"text": "GraphARC is 10x faster", "citation": "benchmarked at 10x faster"},
    {"text": "Budgets make graphs run faster",
     "citation": "Budgets put hard ceilings on iterations, tokens, and wall-clock time."},
]})])
reviewer = ScriptedChatModel(responses=[
    '{"supported": true,  "reason": "the source says exactly this"}',
    '{"supported": false, "reason": "ceilings bound work, they do not speed it up"}',
])

result = build_stage5(author, reviewer).invoke({"source_text": DEMO_SOURCE})

print("accepted:", result["accepted"])
print("rejected:", result["rejected"])
print("author calls:", author.call_count, "| reviewer calls:", reviewer.call_count)
for v in result["verdicts"]:
    print(f"  anchor_ok={v.anchor_ok!s:<5} model_accepted={v.model_accepted!s:<5} {v.claim_text}")
```

```text
accepted: ['GraphARC enforces typed state contracts']
rejected: ['GraphARC is 10x faster', 'Budgets make graphs run faster']
author calls: 1 | reviewer calls: 2
  anchor_ok=True  model_accepted=True  GraphARC enforces typed state contracts
  anchor_ok=False model_accepted=None  GraphARC is 10x faster
  anchor_ok=True  model_accepted=False Budgets make graphs run faster
```

Three claims, two reviewer calls. The `Verdict` list is kept in state, so a downstream node
can route on *why* something was rejected rather than only on the boolean.

---

## How do I record a fact so I can audit where it came from?

A `Claim` is a subject-predicate-object triple plus provenance. `source` is required; the
rest of the provenance is optional and worth filling in.

```python
from grapharc.memory import Claim, MemoryStore

store = MemoryStore()
claim = store.add(
    Claim(
        subject="GraphARC",
        predicate="depends on",
        object="LangGraph",
        source="README.md#L3",       # required: where it came from
        run_id="run-12",             # optional: which run learned it
        confidence=0.9,
    )
)

print("id:         ", claim.id)
print("observed_at:", claim.observed_at)
print("is_current: ", claim.is_current)
print("key:        ", claim.key)

for lookup in ("GraphARC", "  grapharc  ", "GRAPHARC!", "Graph-ARC"):
    print(f"current({lookup!r:<14}) -> {[c.object for c in store.current(lookup)]}")
```

```text
id:          bf12abda3985
observed_at: 2026-07-28T04:27:25.893671+00:00
is_current:  True
key:         ('grapharc', 'depends on')
current('GraphARC'    ) -> ['LangGraph']
current('  grapharc  ') -> ['LangGraph']
current('GRAPHARC!'   ) -> ['LangGraph']
current('Graph-ARC'   ) -> []
```

The id and timestamp are generated, so yours will differ. Everything else reproduces.

**Why it works this way.** Lookups go through `_normalize`: NFKC, casefold, then every
non-word character collapsed to a single space. So case and trailing punctuation do not
split an entity — but punctuation *inside* a name does, which is the last line. `"GraphARC"`
normalizes to `grapharc` and `"Graph-ARC"` to `graph arc`, and those are different entities.
Normalization is Unicode-aware rather than ASCII-only precisely so that `東京` and `北京` do
not both collapse to the empty key.

`observed_at` comes from a strictly increasing clock at microsecond resolution, so no two
claims a process writes can tie. That is not cosmetic: ties in a `reverse=True` sort are
*not* reversed by Python's stable sort, which once made the two backends return dead ends in
opposite orders from identical input.

---

## How do I correct a fact without losing what we used to believe?

`supersede`. It never deletes: the old claim stays, marked, pointing at its replacement.
This is the "run #51 sees that run #37 replaced run #12's fact" story, end to end.

```python
from grapharc.memory import Claim, MemoryStore

store = MemoryStore()


def fact(claim_id, value, source, run):
    # ids are random by default; fixed here so the output below is stable.
    return Claim(id=claim_id, subject="auth-service", predicate="times out after",
                 object=value, source=source, run_id=run)


store.add(fact("v1", "30s", "config/prod.yaml@2026-05", "run-12"))
store.supersede("v1", fact("v2", "10s", "config/prod.yaml@2026-06", "run-37"))
store.supersede("v2", fact("v3", "15s", "incident-4412 postmortem", "run-51"))

print("current: ", [(c.id, c.object) for c in store.current("auth-service")])
print("dead ends:", [(c.id, c.object, "->", c.superseded_by)
                     for c in store.dead_ends("auth-service")])
print("history:")
for c in store.history("auth-service", "times out after"):
    print(f"  {c.id} {c.object:<4} source={c.source:<26} run={c.run_id} "
          f"superseded_by={c.superseded_by}")

for label, old_id, new in [
    ("already superseded", "v1", fact("v4", "20s", "guess", "run-52")),
    ("self-supersede", "v3", fact("v3", "20s", "guess", "run-52")),
    ("unknown claim", "nope", fact("v5", "20s", "guess", "run-52")),
]:
    try:
        store.supersede(old_id, new)
    except (ValueError, KeyError) as exc:
        print(f"{label}: {type(exc).__name__}: {exc}")
```

```text
current:  [('v3', '15s')]
dead ends: [('v1', '30s', '->', 'v2'), ('v2', '10s', '->', 'v3')]
history:
  v1 30s  source=config/prod.yaml@2026-05   run=run-12 superseded_by=v2
  v2 10s  source=config/prod.yaml@2026-06   run=run-37 superseded_by=v3
  v3 15s  source=incident-4412 postmortem   run=run-51 superseded_by=None
already superseded: ValueError: claim 'v1' was already superseded by 'v2'
self-supersede: ValueError: claim 'v3' cannot supersede itself — a correction is a new claim with its own id, so the old one survives to be pointed at
unknown claim: KeyError: "unknown claim 'nope'"
```

`current` returns one claim, `history` returns the whole chain oldest-first with each link's
own source and run, and `dead_ends` returns exactly what was retracted and by what.

**Why it works this way.** Three preconditions live in one shared `validate_supersede` that
both backends call. The self-supersede case is why: when each backend implemented it
separately, `MemoryStore` and `SQLiteMemoryStore` disagreed about what `supersede(id, claim)`
with `claim.id == id` should leave behind, and both ended up with a claim whose
`superseded_by` pointed at itself — invisible to `current`, self-referential in `dead_ends`.
There is no coherent reading of that, so it raises.

Note `history` takes a predicate and `dead_ends` does not. `history` answers "what did we
believe about this one attribute over time"; `dead_ends` answers "what has this subject been
wrong about", which is the question a node about to re-derive something needs.

---

## How do I keep memory across processes?

`MemoryStore` is a dict and dies with the interpreter. `SQLiteMemoryStore` implements the
same `ClaimStore` protocol against a file. Swapping one for the other is a one-line change,
and the difference only shows up across a process boundary — so that is how to test it.

```python
import subprocess
import sys
import tempfile
from pathlib import Path

from grapharc.memory import ClaimStore, MemoryStore, SQLiteMemoryStore

WRITER = """
import os, sys
from grapharc.memory import Claim, SQLiteMemoryStore

with SQLiteMemoryStore(sys.argv[1]) as store:
    store.add(Claim(id="v1", subject="auth-service", predicate="times out after",
                    object="30s", source="config/prod.yaml", run_id="run-12"))
    print(f"writer pid={os.getpid()}: wrote v1")
"""

READER = """
import os, sys
from grapharc.memory import Claim, SQLiteMemoryStore

with SQLiteMemoryStore(sys.argv[1]) as store:
    (believed,) = store.current("auth-service")
    print(f"reader pid={os.getpid()}: found {believed.id} = {believed.object!r} "
          f"from {believed.source} (run {believed.run_id})")
    store.supersede(believed.id, Claim(id="v2", subject="auth-service",
                                       predicate="times out after", object="10s",
                                       source="incident-4412", run_id="run-37"))
    print(f"reader pid={os.getpid()}: superseded {believed.id} -> v2")
"""

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "memory.sqlite"
    for name, script in (("writer", WRITER), ("reader", READER)):
        path = Path(tmp) / f"{name}.py"
        path.write_text(script)
        print(subprocess.run([sys.executable, str(path), str(db)],
                             capture_output=True, text=True, check=True).stdout, end="")

    # A third process — this one — reads what the other two left behind.
    with SQLiteMemoryStore(db) as store:
        print("parent:", [(c.id, c.object) for c in store.current("auth-service")])
        print("parent dead ends:", [(c.id, c.object) for c in store.dead_ends("auth-service")])
        print("both satisfy ClaimStore:",
              isinstance(MemoryStore(), ClaimStore), isinstance(store, ClaimStore))
```

```text
writer pid=478023: wrote v1
reader pid=478024: found v1 = '30s' from config/prod.yaml (run run-12)
reader pid=478024: superseded v1 -> v2
parent: [('v2', '10s')]
parent dead ends: [('v1', '30s')]
both satisfy ClaimStore: True True
```

Three different pids (yours will differ), one file, and the correction chain survives every
one of them. Nothing here shares a Python object — which is exactly the test that would have
caught the original version of this claim, where three "runs" shared one dict.

**Why it works this way.** The store opens in autocommit and wraps `supersede` in
`BEGIN IMMEDIATE`, so its read-then-write cannot interleave with another process's — a
deferred transaction takes the write lock at the first write, which lets two processes both
read a claim as live and both supersede it. It runs `synchronous = FULL` under WAL, because
losing the last few commits on a host failure would defeat the one thing this backend is
for.

Two caveats worth knowing before you rely on the protocol. `isinstance(x, ClaimStore)` is a
`runtime_checkable` `Protocol`, so it checks *method names only* — passing it is not evidence
the invariants hold; the shared conformance suite in `tests/` is what enforces those.
And `SQLiteMemoryStore` holds an open connection: use it as a context manager or call
`close()`, or you leak a file handle per store.

---

## Can I query the claim graph in Cypher?

Yes, with the third backend. `LadybugMemoryStore` implements the same `ClaimStore` protocol
against [LadybugDB](https://ladybugdb.com/) — an embedded property-graph database with
Cypher, forked from Kuzu after Apple acquired and closed it. Embedded means the same deal
SQLite offers: a path on disk, no server to run.

The difference is what gets stored. The other two backends keep claims as rows and rebuild
the graph in Python — `ClaimIndex` scans the whole corpus to build the subject/object
adjacency, every time you construct one. This backend stores the edges: `superseded_by` is a
`SUPERSEDED_BY` edge rather than a column, and the subject and object of every claim are
`Entity` nodes. So a correction chain is a path you can walk, and the escape hatch from the
seven-method protocol is a query rather than a scan.

```python
# Not executed by the docs test — needs the `ladybug` extra, which CI does not install.
from grapharc.memory import Claim, LadybugMemoryStore

with LadybugMemoryStore("memory.lbdb") as store:
    old = store.add(Claim(subject="auth-service", predicate="times out after",
                          object="30s", source="config/prod.yaml", run_id="run-12"))
    store.supersede(old.id, Claim(subject="auth-service", predicate="times out after",
                                  object="5s", source="incident-114", run_id="run-37"))

    # What did an earlier run believe that a later one corrected, and on whose authority?
    for was, now, source in store.cypher(
        "MATCH (old:Claim)-[:SUPERSEDED_BY]->(new:Claim) "
        "RETURN old.object, new.object, new.source"
    ):
        print(f"{was} -> {now} ({source})")

    # A question about A reaching facts about B, as a hop in the database.
    store.cypher(
        "MATCH (:Claim {subject: $s})-[:MENTIONS]->(e:Entity)<-[:ABOUT]-(next:Claim) "
        "RETURN next.subject, next.predicate, next.object",
        {"s": "auth-service"},
    )
```

```text
30s -> 5s (incident-114)
```

Install it with `pip install 'grapharc[ladybug]'`. **The distribution is `real-ladybug`, not
`ladybug`** — that name on PyPI belongs to Ladybug Tools, an unrelated building-science
package, and installing it will not give you a database. The store checks what it imported
and says so rather than failing later with an `AttributeError`.

**The cost, which decides whether you can use it at all.** LadybugDB takes an *exclusive lock
on the database*. One process may open it for writing, or several may open it read-only, and
those two groups cannot overlap — while a writer holds it, a second process opening the same
path fails outright, even with `read_only=True`, rather than waiting. `SQLiteMemoryStore` in
WAL mode allows concurrent readers alongside a writer and makes competing writers queue on
`busy_timeout`. So sequential hand-off works (run #12 writes and exits, run #37 opens the
same path and reads it), and concurrent multi-process writing does not. If two agent
processes must write the same memory at once, use SQLite. That limitation is not just
documented — `tests/test_ladybug_store.py` spawns a second process against a held database
and asserts it fails, so if a later release relaxes the lock, the claim above gets corrected
instead of quietly going stale.

Pass values as parameters, never by formatting them into the query string: a claim's object
is arbitrary text, and string interpolation is the graph-database spelling of SQL injection.

---

## How do I find the claims relevant to a question?

`search` ranks by relevance and returns *why* each claim was returned. Two ways in: name
entities you already know are relevant, or pass free text.

```python
from grapharc.memory import Claim, MemoryStore, search

store = MemoryStore()
FACTS = [
    ("a1", "auth-service", "depends on", "redis", "arch.md"),
    ("a2", "auth-service", "owned by", "platform team", "CODEOWNERS"),
    ("r1", "redis", "runs version", "7.2.4", "infra/redis.tf"),
    ("r2", "redis", "has known issue", "connection leak under TLS", "incident-4412"),
    ("b1", "billing-service", "depends on", "auth-service", "arch.md"),
    ("p1", "platform team", "on call rotation", "PagerDuty schedule P7", "runbook.md"),
]
for cid, s, p, o, src in FACTS:
    store.add(Claim(id=cid, subject=s, predicate=p, object=o, source=src))

print("--- entities=['auth-service'] ---")
for hit in search(store, entities=["auth-service"]):
    c = hit.claim
    print(f"{hit.score:.3f} hop={hit.hops} via={hit.via!s:<5} {hit.channel:<8} "
          f"{c.id} {c.subject} {c.predicate} {c.object}")

print("--- query='connection leak' (no entity named) ---")
for hit in search(store, query="connection leak"):
    c = hit.claim
    print(f"{hit.score:.3f} hop={hit.hops} via={hit.via!s:<5} {hit.channel:<8} "
          f"{c.id} {c.subject} {c.predicate} {c.object}")
```

```text
--- entities=['auth-service'] ---
1.000 hop=0 via=None  entity   a2 auth-service owned by platform team
1.000 hop=0 via=None  entity   a1 auth-service depends on redis
0.450 hop=1 via=a2    graph    p1 platform team on call rotation PagerDuty schedule P7
0.450 hop=1 via=a2    graph    b1 billing-service depends on auth-service
0.450 hop=1 via=a1    graph    r2 redis has known issue connection leak under TLS
0.450 hop=1 via=a1    graph    r1 redis runs version 7.2.4
--- query='connection leak' (no entity named) ---
0.900 hop=0 via=None  lexical  r2 redis has known issue connection leak under TLS
0.405 hop=1 via=r2    graph    a1 auth-service depends on redis
```

This is the graph traversal doing work. Nobody asked about redis, but a question about
`auth-service` surfaced *redis has known issue connection leak under TLS* — reached through
`a1`, whose object is `redis`. Incoming edges are followed too, which is how
`billing-service depends on auth-service` shows up: that is the "who calls me" answer.

**Why it works this way.** Scores land in two bands that cannot overlap. Naming an entity is
worth a flat `1.0`, because the caller *knows* the subject matters; the text channels
contribute at most `0.9` on top. So a named subject always outranks a claim found only by
its words, and relevance orders the claims inside each band. A hop multiplies the parent's
score by `0.45`, so a neighbour lands below the claim it came from and reads as supporting
evidence rather than as the answer. `hops`, `via` and `channel` are on every `ScoredClaim`
so a node can show its work rather than presenting a second-hand fact as a direct hit.

Defaults: one hop, 20 claims. Two hops on a dense graph is already a lot of drift.

---

## What if the query is misspelled, or there is no query at all?

Three behaviours worth knowing before you debug an empty result list.

```python
from grapharc.memory import Claim, ClaimIndex, HashingEmbedder, MemoryStore, known_entities, search

store = MemoryStore()
for cid, s, p, o in [
    ("a1", "auth-service", "depends on", "redis"),
    ("r2", "redis", "has known issue", "connection leak under TLS"),
    ("b1", "billing-service", "depends on", "auth-service"),
]:
    store.add(Claim(id=cid, subject=s, predicate=p, object=o, source="arch.md"))

# Build the index once, query it many times. Otherwise every search rebuilds it.
index = ClaimIndex.from_store(store)
for query in ("connection leak", "who depends on auth-service"):
    print(f"{query!r:<32} -> {[h.claim.id for h in search(store, query=query, index=index)]}")

# No query and no entity is not "everything" — it is nothing.
print("search(store) with neither     ->", search(store))

# 'authservice' shares no whole token with anything, so BM25 scores zero.
print("'authservice' lexical only     ->",
      [h.claim.id for h in search(store, query="authservice")])
print("'authservice' + HashingEmbedder->",
      [h.claim.id for h in search(store, query="authservice", embedder=HashingEmbedder())])
print("'redsi' + HashingEmbedder      ->",
      [h.claim.id for h in search(store, query="redsi", embedder=HashingEmbedder())])

print("known entities:", sorted(known_entities(store)))
```

```text
'connection leak'                -> ['r2', 'a1']
'who depends on auth-service'    -> ['a1', 'b1', 'r2']
search(store) with neither     -> []
'authservice' lexical only     -> []
'authservice' + HashingEmbedder-> ['a1', 'b1', 'r2']
'redsi' + HashingEmbedder      -> []
known entities: ['auth service', 'billing service', 'redis']
```

**Why it works this way.**

*No query and no entity returns nothing*, not the whole store. There is nothing to be
relevant to, and handing a node an arbitrary slice of memory labelled "context" is worse
than handing it none.

*`HashingEmbedder` is a spelling channel, not a semantic one.* It is hashed word tokens plus
character n-grams — so `authservice` finds `auth-service` where BM25 scores a flat zero, and
`redsi` still finds nothing, because two transposed characters in a five-letter word leave
too little n-gram overlap to clear `MIN_SIMILARITY` (0.15). It will never relate "car" to
"automobile". For meaning, inject a real embedder through the `Embedder` protocol and lower
`lexical_weight` from its 0.6 default.

*`known_entities` returns normalized keys*, which is why `auth-service` comes back as
`auth service`. Use it to decide what to look up, not to render.

Cost: `search` builds a `ClaimIndex` over the whole store — O(claims) per call. Pass
`index=` to pay once, as above.

---

## Why did my reused index stop finding new claims?

Because an index is a snapshot, and it does not notice writes.

```python
from grapharc.memory import Claim, ClaimIndex, MemoryStore, search

store = MemoryStore()
store.add(Claim(id="a", subject="redis", predicate="runs version", object="7.2", source="tf"))
index = ClaimIndex.from_store(store)

store.add(Claim(id="b", subject="redis", predicate="has issue", object="leak", source="i-1"))

print("stale index:", [h.claim.id for h in search(store, entities=["redis"], index=index)])
print("rebuilt:    ", [h.claim.id for h in
                       search(store, entities=["redis"], index=ClaimIndex.from_store(store))])
```

```text
stale index: ['a']
rebuilt:     ['b', 'a']
```

This fails silently — no exception, just a fact quietly missing from a node's context. There
is deliberately no `ClaimIndex.add`: term statistics change with every write, and an index
that is a few claims stale does not error, it ranks wrongly. Rebuild after writing, or drop
`index=` and eat the O(claims) rebuild.

---

## How do I fit memory into a prompt without blowing the context window?

`render_context` produces the brief a node actually puts in its prompt: what is believed,
what is disputed, what was already retracted — with provenance on every line.

```python
from grapharc.memory import Claim, MemoryStore, render_context

store = MemoryStore()


def c(cid, subject, predicate, obj, source):
    return Claim(id=cid, subject=subject, predicate=predicate, object=obj, source=source)


store.add(c("a1", "auth-service", "depends on", "redis", "arch.md"))
store.add(c("r1", "redis", "runs version", "7.2.4", "infra/redis.tf"))
# Two live sources disagree about the timeout — nobody has decided yet.
store.add(c("t1", "auth-service", "times out after", "30s", "config/prod.yaml"))
store.add(c("t2", "auth-service", "times out after", "10s", "helm/values.yaml"))
# Three corrections, oldest first.
store.add(c("p1", "auth-service", "listens on port", "8080", "old-README"))
store.supersede("p1", c("p2", "auth-service", "listens on port", "8443", "ingress.yaml"))
store.add(c("m1", "auth-service", "deployed to", "eu-west-1", "terraform@2025"))
store.supersede("m1", c("m2", "auth-service", "deployed to", "eu-central-1", "terraform@2026"))

print(render_context(store, entities=["auth-service"]))
print()
print("=== same brief, max_tokens=90 ===")
print(render_context(store, entities=["auth-service"], max_tokens=90))
```

```text
Known facts (with provenance):
- auth-service deployed to eu-central-1 [source: terraform@2026, observed: 2026-07-28T04:29:22.317681+00:00]
- auth-service listens on port 8443 [source: ingress.yaml, observed: 2026-07-28T04:29:22.317665+00:00]
- auth-service times out after 10s [source: helm/values.yaml, observed: 2026-07-28T04:29:22.317658+00:00]
- auth-service times out after 30s [source: config/prod.yaml, observed: 2026-07-28T04:29:22.317655+00:00]
- auth-service depends on redis [source: arch.md, observed: 2026-07-28T04:29:22.317631+00:00]
- redis runs version 7.2.4 [source: infra/redis.tf, observed: 2026-07-28T04:29:22.317650+00:00] (related via auth-service)

Contradictions — same subject and predicate, different values:
- auth-service times out after: '30s' (t1, config/prod.yaml) | '10s' (t2, helm/values.yaml)

Superseded — do not re-derive these:
- auth-service deployed to eu-west-1 (superseded by m2)
- auth-service listens on port 8080 (superseded by p2)

=== same brief, max_tokens=90 ===
Known facts (with provenance):
- auth-service deployed to eu-central-1 [source: terraform@2026, observed: 2026-07-28T04:29:22.317681+00:00]
- auth-service listens on port 8443 [source: ingress.yaml, observed: 2026-07-28T04:29:22.317665+00:00]
(+7 lines omitted to fit a 90-token budget)
```

(Timestamps are generated at run time, so yours will differ.)

Three sections in priority order, and the truncated version keeps the most load-bearing one.
The `(related via auth-service)` tag marks the hop-1 fact so the node can tell a direct
answer from a neighbour.

**Why it works this way.** `_fit` drops whole entries from the tail and never truncates
mid-line — half a claim with half its provenance is worse than no claim — and a section
whose header fits but whose first entry does not is dropped entirely, so a header never
appears over nothing. The omission note is the one line that is never dropped: a brief that
silently omits facts is worse than a short one, because the reader cannot tell.

Two sharp edges, both stated in the source and worth repeating where you hit them. First,
`max_tokens` bounds an *estimate* — four characters per token, no tokenizer ships with this
library — and ids, hyphenated values and ISO timestamps all split into more tokens than
their length implies, so a brief that fits the estimate can still overrun a real tokenizer.
Pass `count_tokens=` when the budget has to be exact. Second, a `max_tokens` below the
24-token note reserve is overshot by the note alone; that is the only case where the output
exceeds the budget, and it is the right trade.

---

## How do I keep the "already tried that" section from growing forever?

Every correction adds a dead end about the same subject, so on a long-lived project this
section grows without bound. It is capped by count, and it says what it dropped.

```python
from grapharc.memory import Claim, MemoryStore, render_context, retrieve_dead_ends

store = MemoryStore()


def c(cid, obj, source):
    return Claim(id=cid, subject="auth-service", predicate="times out after",
                 object=obj, source=source)


store.add(c("v1", "30s", "guess-2024"))
for i, (obj, source) in enumerate(
    [("25s", "guess-2025"), ("20s", "tuning-run"), ("15s", "load-test"), ("10s", "incident-4412")],
    start=2,
):
    store.supersede(f"v{i - 1}", c(f"v{i}", obj, source))

kept, omitted = retrieve_dead_ends(store, entities=["auth-service"], max_dead_ends=2)
print("shown:", [(k.id, k.object) for k in kept], "omitted:", omitted)
print()
print(render_context(store, entities=["auth-service"], max_dead_ends=2))
```

```text
shown: [('v4', '15s'), ('v3', '20s')] omitted: 2

Known facts (with provenance):
- auth-service times out after 10s [source: incident-4412, observed: 2026-07-28T04:29:33.999234+00:00]

Superseded — do not re-derive these:
- auth-service times out after 15s (superseded by v5)
- auth-service times out after 20s (superseded by v4)
- (+2 older corrections omitted)
```

**Why it works this way.** Newest correction first, because the oldest dead ends are the
least likely to be the one a node is about to walk into — and since the cap discards the
tail, that ordering decides *which* corrections the node is shown at all. The sort key is
three fields (correction time, then observation time, then id) so it is a total order and
both backends render the same section from the same claims. Ties used to be left to
`list.sort`'s stability, which preserves rather than reverses input order, and `reverse=True`
quietly handed back tied groups oldest-first.

`retrieve_dead_ends` also takes `query=`, which re-ranks by relevance first and correction
time second — scored against the dead ends alone, not the whole corpus, because the question
is "which of these retracted facts matter to me".

Note the default is 10 (`DEFAULT_MAX_DEAD_ENDS`), deliberately smaller than the 20 for
facts: a dead end is a hint, not the answer.

---

## How do I notice that memory contradicts itself?

`supersede` needs you to already know the id of the claim being replaced. An agent has a
fact in hand and no idea what the store already believes. `add_and_detect` closes that gap.

```python
from grapharc.memory import (
    Claim,
    MemoryStore,
    add_and_detect,
    supersede_conflicting,
    unresolved_contradictions,
)

store = MemoryStore()
store.add(Claim(id="t1", subject="auth-service", predicate="times out after",
                object="30s", source="config/prod.yaml"))

incoming = Claim(id="t2", subject="auth-service", predicate="times out after",
                 object="10s", source="incident-4412")
stored, conflicts = add_and_detect(store, incoming)
print("stored anyway:", stored.id)
for conflict in conflicts:
    print("conflict:", conflict.describe())

# Resolution is a separate, deliberate call.
supersede_conflicting(store, conflicts[0])
print("after resolution, current:", [(c.id, c.object) for c in store.current("auth-service")])
print("dead ends:", [(c.id, c.object) for c in store.dead_ends("auth-service")])

# The limit: a legitimately multi-valued predicate reads as disagreement.
store.add(Claim(id="d1", subject="billing", predicate="depends on",
                object="postgres", source="arch.md"))
store.add(Claim(id="d2", subject="billing", predicate="depends on",
                object="redis", source="arch.md"))
for group in unresolved_contradictions(store):
    print("flagged:", group.subject, group.predicate,
          [claim.object for claim in group.claims])
```

```text
stored anyway: t2
conflict: auth-service times out after: '10s' (from incident-4412) contradicts '30s' (from config/prod.yaml, claim t1)
after resolution, current: [('t2', '10s')]
dead ends: [('t1', '30s')]
flagged: billing depends on ['postgres', 'redis']
```

The conflicting claim is stored regardless, and detection reports rather than resolves.

**Why it works this way.** Detection and resolution are separate calls on purpose, and the
last two lines of output are the reason. The test is structural — same normalized
`(subject, predicate)`, different normalized object — so a genuinely multi-valued predicate
like `depends on` reads as three-way disagreement. Auto-superseding on detection would
delete the second half of a multi-valued fact, inside the one subsystem whose promise is
that facts are never destroyed. So it reports, and a caller who has a basis for a decision
calls `supersede_conflicting`.

The other limits, since they are easy to assume away: it does not relate "is fast" to "is
slow" (different predicates), and it does not see `"LangGraph"` and `"the LangGraph library"`
as the same object (`_normalize` compares strings, not meanings).

`unresolved_contradictions(store)` with no `entities` scans the whole corpus — a maintenance
report, not something to call inside a loop. `render_context` runs a cheaper version scoped
to the facts it is already showing.

---

## How do I record the files an agent produced?

Claims answer "what is true"; artifacts answer "what was produced". Same rules: append-only,
provenance mandatory.

```python
from grapharc.memory import MemoryArtifactStore, render_artifacts

store = MemoryArtifactStore()

v1 = store.put("report.md", "# Findings\nauth-service times out after 30s\n",
               source="node:summarize", run_id="run-12", node="summarize")
v2 = store.put("report.md", "# Findings\nauth-service times out after 10s\n",
               source="node:summarize", run_id="run-37", node="summarize",
               parents=[v1.id], claim_ids=["t2"])

print("versions:", [(a.id, a.run_id) for a in store.versions("report.md")])
print("latest:  ", store.latest("report.md").id, "==", v2.id)
print("v1 still readable:", store.read_text(v1.id).splitlines()[1])
print("media_type:", v2.media_type, "| sha256:", v2.sha256[:16], "| parents:", v2.parents)
print("Report.md is a different name:", store.versions("Report.md"))

try:
    store.put("notes.txt", "x", source="")
except ValueError as exc:
    print("no provenance:", exc)

traversal = store.put("../../etc/passwd", b"not a path", source="node:evil")
print("name is metadata:", traversal.name, "| stored under:", traversal.blob_key)

print()
print(render_artifacts(store))
```

```text
versions: [('07ecc720d4b1', 'run-12'), ('decc2c843fc3', 'run-37')]
latest:   decc2c843fc3 == decc2c843fc3
v1 still readable: auth-service times out after 30s
media_type: text/plain; charset=utf-8 | sha256: 00f97cfbcd1181fe | parents: ('07ecc720d4b1',)
Report.md is a different name: []
no provenance: artifact 'notes.txt' needs a source — an artifact with no provenance cannot be attributed after the fact
name is metadata: ../../etc/passwd | stored under: ('2c', '2c2cc34cd0084471e6f45f217a6d1716ae9e7178722ed56585f95a6bfc51e1e6')

Artifacts (id · name · provenance):
- fb86dfa2e567 ../../etc/passwd (application/octet-stream, 10 bytes) [source: node:evil]
- decc2c843fc3 report.md (text/plain; charset=utf-8, 44 bytes) [source: node:summarize, run: run-37, node: summarize]
```

(Artifact ids are random; the sha256 values are content-addressed and reproduce.)

Writing `report.md` twice records a second version rather than replacing the first, and v1's
bytes are still readable. `render_artifacts` lists metadata only — never content — because a
run that produced four hundred files must not be able to spend a context window listing them.

**Why it works this way.** `name` is metadata and is never used to build a path: the
traversal-looking name is stored under the hash of its content like everything else. Names
are matched *exactly*, unlike a claim's subject — `Report.md` is not `report.md`, because a
filename is not an entity and entity resolution on filenames would merge two real files.
Content addressing means two artifacts with identical bytes share one blob while keeping
their own rows and their own provenance: the same report produced by two runs is two events.

---

## How do I put claims and artifacts in one file?

Point `SQLiteArtifactStore` at the same path as `SQLiteMemoryStore`. Two connections to one
SQLite database is normal; both take the write lock and wait out contention.

```python
import tempfile
from pathlib import Path

from grapharc.memory import Claim, SQLiteArtifactStore, SQLiteMemoryStore

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "memory.sqlite"
    with SQLiteMemoryStore(db) as claims, SQLiteArtifactStore(db) as artifacts:
        claim = claims.add(Claim(subject="auth-service", predicate="times out after",
                                 object="10s", source="incident-4412", run_id="run-37"))
        art = artifacts.put("report.md", "auth-service times out after 10s\n",
                            source="node:summarize", run_id="run-37",
                            claim_ids=[claim.id])
        print("claim:", claim.id, "| artifact:", art.id, "| links to:", art.claim_ids)
        print("blob:", artifacts.blob_path(art).relative_to(tmp))
        print("journal_mode:", claims.journal_mode)

    # A later process opens the same file and finds both halves.
    with SQLiteMemoryStore(db) as claims, SQLiteArtifactStore(db) as artifacts:
        print("reopened claims:   ", [c.object for c in claims.current("auth-service")])
        print("reopened artifacts:", [a.name for a in artifacts.by_run("run-37")])
        print("content:", artifacts.read_text(art.id).strip())

    print("files:", sorted(p.name for p in Path(tmp).iterdir()))
```

```text
claim: c25bee060492 | artifact: 2a8b76fc4ad0 | links to: ('c25bee060492',)
blob: memory.sqlite.blobs/52/52bb15f4df1a5a838214381ba412ef6988ba5ccbbd52afd5b9bdb8db552131a3
journal_mode: wal
reopened claims:    ['10s']
reopened artifacts: ['report.md']
content: auth-service times out after 10s
files: ['memory.sqlite', 'memory.sqlite.blobs']
```

(The claim and artifact ids are random; the blob's sha256 is content-addressed.)

`claim_ids` on the artifact is the edge between the two halves: this report was produced
from that claim, by id, so nothing can drift out of sync.

**Why it works this way.** Metadata is a SQLite row and content is a file under
`<db>.blobs/<first two hex>/<sha256>` — so the database stays small enough to query while a
200 MB core dump is still storable. The blob is written and fsynced *before* the row is
inserted: crash between the two and you have an unreferenced blob (garbage), never a row
pointing at content that does not exist (a lie).

The consequence to plan for: the database file and its `.blobs` directory are one unit.
Copy the `.sqlite` alone and `read()` raises `FileNotFoundError` naming the missing path —
loudly, which is the point, but it is still a footgun when moving a store between machines.

---

## Putting it together

The pattern the two subsystems are meant to be used in:

1. **Recall before working** — `render_context(store, entities=[...])` into the node's
   prompt, so the run starts from what is known and its dead ends.
2. **Verify before writing** — `verify_claim` each extracted fact against its source; the
   anchor makes fabricated citations free to reject.
3. **Write with provenance** — `Claim(..., source=..., run_id=ctx.run_id)`, and
   `add_and_detect` if something may already be believed.
4. **Correct, do not overwrite** — `supersede`, so the next run can see the correction
   instead of re-deriving the old answer.

`grapharc/examples/stage5_verifier.py` and `grapharc/examples/stage6_memory.py` are the
smallest complete graphs doing exactly this; `grapharc demo stage5` and `grapharc demo stage6`
execute them on scripted models for free.
