# Agents and tools

Recipes for the half of GraphARC that *does* things: the core toolset, the harness
that decides what a model is allowed to run, the loop that drives it, and the
executors that bound what a tool can touch once it has started.

Every snippet on this page was executed against this repo's `.venv` before it was
written down, and every block labelled *Output* is pasted from that run. Two
snippets are marked as **not run** — they need an API key or a paid subscription,
and nothing here claims a result for them. Where output would depend on a temp
directory, a clock or a machine path, the snippet blanks that part itself and says
so in a comment, so what you see is what you get.

Verified against `grapharc 0.1.0`, Python 3.14.6, `langgraph 1.2.9`,
`langchain-core 1.5.1`, `pydantic 2.13.4`, Docker 29.4.1.

Each snippet is a complete file. Save it and run it; nothing carries over between
recipes except one shared helper, which says where to save it.

---

## How do I give an agent a set of file tools?

`grapharc.tools` ships seven: `read_file`, `write_file`, `edit_file`, `list_dir`,
`glob`, `grep`, `run_command`. `register_core_tools` puts them in a registry bound
to one workspace directory, which must already exist.

<!-- cookbook name=01-core-toolset.py -->
```python
import tempfile
from pathlib import Path

from grapharc.harness import (
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.tools import CORE_TOOL_NAMES, register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

print(CORE_TOOL_NAMES)

registry = ToolRegistry()
register_core_tools(registry, workspace, exclude=["run_command"])

policy = PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")])
harness = Harness(registry, policy, executor=LocalExecutor(), workspace=str(workspace))

print([spec.name for spec in harness.visible_tools()])
print(harness.call("read_file", {"path": "calc.py"}))
```

Output:

```
('read_file', 'write_file', 'edit_file', 'list_dir', 'glob', 'grep', 'run_command')
['edit_file', 'glob', 'grep', 'list_dir', 'read_file', 'write_file']
def add(a, b):
    return a - b
```

**Why it works this way.** Registering a tool and *authorising* it are two separate
decisions, and they are made in two different objects. `include`/`exclude` is the
coarse one — a tool never registered can never be called, however the policy is
edited later — and the policy is the fine one. Excluding `run_command` from a
toolset that has no business shelling out is a stronger statement than denying it,
because there is nothing left to deny.

A typo in `exclude` raises rather than being ignored: silently accepting
`exclude=["run_comand"]` would leave you believing the shell tool is off while it
is registered and callable. And `exclude="run_command"` — a bare string, which
iterates as characters — is refused with a message telling you to wrap it in a
list.

`visible_tools()` comes back in sorted order while `CORE_TOOL_NAMES` is in build
order. `core_tools()` always returns `CORE_TOOL_NAMES` order regardless of what you
asked for, so two callers requesting the same set get byte-identical schema lists —
this list goes into a model prompt, where ordering changes cache keys.

---

## How do I stop a tool reading outside its workspace?

You do not have to. Every core tool routes every path argument through
`Workspace.resolve` before it does anything else.

<!-- cookbook name=02-confinement.py -->
```python
import os
import tempfile
from pathlib import Path

from grapharc.tools import WorkspaceEscape, core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
os.symlink("/etc/passwd", workspace / "innocent.txt")

read_file = {spec.name: spec.fn for spec in core_tools(workspace)}["read_file"]


def attempt(path: str) -> None:
    try:
        print(f"{path!r:<20} -> {read_file(path)!r}")
    except WorkspaceEscape as exc:
        # The workspace root is a fresh temp dir; blank it so the output is stable.
        print(f"{path!r:<20} -> {str(exc).replace(str(workspace), '<ws>')}")


attempt("calc.py")
attempt("subdir/../calc.py")  # `..` is legal — only where it lands is judged
attempt("../../etc/passwd")
attempt("/etc/passwd")
attempt("innocent.txt")  # a symlink pointing out of the workspace
```

Output:

```
'calc.py'            -> 'def add(a, b):\n    return a - b\n'
'subdir/../calc.py'  -> 'def add(a, b):\n    return a - b\n'
'../../etc/passwd'   -> path '../../etc/passwd' resolves to /etc/passwd, which is outside the workspace <ws>. Core tools only reach paths inside the workspace; use a path relative to its root.
'/etc/passwd'        -> path '/etc/passwd' resolves to /etc/passwd, which is outside the workspace <ws>. Core tools only reach paths inside the workspace; use a path relative to its root.
'innocent.txt'       -> path 'innocent.txt' resolves to /etc/passwd, which is outside the workspace <ws>. Core tools only reach paths inside the workspace; use a path relative to its root.
```

**Why it works this way.** Absolute paths, `..` segments and symlinks are not three
special cases with three rules. The path is resolved first and then judged by one
rule: the thing the OS would actually open has to be inside the root. That is why
`subdir/../calc.py` is fine — it is an ordinary in-workspace path, and a blocklist
that refuses it on spelling while a symlink walks straight past is the shape of
every path-traversal CVE. Containment is compared component-wise, so a sibling
directory named `<workspace>-evil` is outside, not inside.

Three sharp edges, in the order you will hit them:

* **Resolution and use are not atomic.** `resolve` follows the symlinks and checks
  where the path lands; the tool then opens that resolved path. A symlink swapped
  in between the two is a race this does not close. Reads and writes pass
  `O_NOFOLLOW` on the final component, which narrows the window to the parent
  directories and no further.
* **The refusal names the resolved path.** A model that asks for `../../etc/passwd`
  is told the absolute path it would have reached, and can infer where its
  workspace sits on the host. That is deliberate — a refusal the caller cannot act
  on just gets retried — but it is a disclosure. Successful results are rendered
  workspace-relative for the same reason.
* **`run_command` is not confined by any of this.** Its `cwd` is workspace-resolved
  and its environment is an allowlist; past that it is an ordinary child process
  with your privileges. See the `run_command` recipe below.

---

## How do I add a tool of my own?

A `ToolSpec` is a name, a description, a callable, and two flags. The schema the
model sees is derived from the callable's signature, so annotate it.

<!-- cookbook name=03-custom-tool.py -->
```python
import json

from grapharc.harness import ToolSpec, tool_schema


def find_owner(service: str, region: str = "us-east-1", page: int = 1) -> str:
    """Look up the on-call owner for a service."""
    return f"{service}@{region} p{page}"


spec = ToolSpec(
    name="find_owner",
    description=find_owner.__doc__ or "",
    fn=find_owner,
    needs_network=True,      # the sandbox blocks sockets unless a tool declares this
    timeout_seconds=10.0,    # a hang past this kills the tool's whole process group
)
print(json.dumps(tool_schema(spec), indent=2))
```

Output:

```
{
  "type": "function",
  "function": {
    "name": "find_owner",
    "description": "Look up the on-call owner for a service.",
    "parameters": {
      "type": "object",
      "properties": {
        "service": {
          "type": "string"
        },
        "region": {
          "type": "string"
        },
        "page": {
          "type": "integer"
        }
      },
      "required": [
        "service"
      ]
    }
  }
}
```

**Why it works this way.** A `ToolSpec` carries no parameter schema of its own, so
the signature is the contract: annotations become JSON types, defaulted parameters
drop out of `required`, and a `**kwargs` tool is declared open with
`additionalProperties`. Anything unannotated or unrecognised becomes `"string"`
rather than being dropped — a parameter the model cannot see is a parameter it
cannot fill.

The description is the only thing telling the model *when* to reach for this tool,
and it is not decoration. Compare the core `run_command` description, which spends
several sentences saying that `argv` is a list, that there is no shell, and that
`grep` is cheaper for reading files.

`needs_network=True` grants the tool the *whole* network under both executors —
there is no per-host filtering. `timeout_seconds` is wall clock; under
`SandboxedExecutor` a tool that blows through it has its whole process group
SIGKILLed, and under `ContainerExecutor` the clock starts at process launch, so
container startup is charged against it.

---

## How do I run the agent loop without calling a real model?

`grapharc.testing.ScriptedChatModel` scripts **text only** and `BaseChatModel`
leaves `bind_tools` unimplemented, so it cannot drive a tool loop on its own. The
two missing pieces are about fifteen lines. Save this as `cookbook_model.py`
alongside the snippets that follow.

<!-- cookbook name=cookbook_model.py compare=off -->
```python
"""A scripted model that can request tools — the shipped double scripts text only."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from grapharc.testing import ScriptedChatModel


class ToolScriptedModel(ScriptedChatModel):
    """`ScriptedChatModel` plus the two pieces an agent loop needs.

    `tool_call_script[i]` is attached to scripted response `i`; an empty entry
    means a plain text turn, which is how the loop learns the task is done.
    """

    tool_call_script: list[list[dict[str, Any]]] = Field(default_factory=list)
    bound: list[list[dict[str, Any]]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound.append(list(tools))
        return self  # the loop must keep talking to this object's script

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = min(self._cursor, max(len(self.responses) - 1, 0))
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        calls = self.tool_call_script[index] if index < len(self.tool_call_script) else None
        if not calls:
            return result
        base = result.generations[0].message
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=base.content,
            usage_metadata=base.usage_metadata,
            tool_calls=[{"type": "tool_call", **call} for call in calls],
        ))])
```

Now an `AgentNode`: a model, a harness, and a loop.

<!-- cookbook name=04-agent-loop.py -->
```python
import tempfile
from pathlib import Path

from cookbook_model import ToolScriptedModel

from grapharc.harness import (
    AgentNode,
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

registry = ToolRegistry()
register_core_tools(registry, workspace, include=["read_file", "edit_file"])
policy = PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")])
harness = Harness(registry, policy, executor=LocalExecutor(), workspace=str(workspace))

model = ToolScriptedModel(
    responses=["let me look", "fixing it", "add() now adds."],
    tool_call_script=[
        [{"name": "read_file", "args": {"path": "calc.py"}, "id": "1"}],
        [{"name": "edit_file",
          "args": {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"},
          "id": "2"}],
        [],  # no tool call: the loop reads this turn as the answer
    ],
)

agent = AgentNode(model, harness, max_iterations=6)
result = agent.run("calc.add subtracts. Fix it.")

print("stopped   :", result.termination_reason)
print("answer    :", result.output)
print("turns     :", result.iterations)
for call in result.tool_calls:
    print(f"  {call.status.value:<8} {call.tool:<10} {call.detail.splitlines()[0][:40]}")
print("calc.py   :", (workspace / "calc.py").read_text().strip())
```

Output:

```
stopped   : target_met
answer    : add() now adds.
turns     : 3
  ok       read_file  def add(a, b):
  ok       edit_file  edited calc.py: replaced 1 occurrence
calc.py   : def add(a, b):
    return a + b
```

**Why it works this way.** The loop is:

```
observe state -> model (visible tools bound) -> tool request
              -> harness.call: permission -> approval -> hooks -> executor
              -> result (or denial, or violation) back to the model
              -> repeat until a recorded StopReason
```

Two properties are worth knowing before you build on it.

**Nothing routes around the harness.** The node never touches `ToolSpec.fn`; every
execution goes through `Harness.call`, which is where deny/ask/allow, the approval
gate, the hooks and the executor live. Tool arguments come from a model, so their
*shape* is as untrusted as their values: a call arriving with `"args": ["a", "b"]`
comes back as `TOOL_ERROR: tool arguments must be a JSON object, got list` — a
result the model can read and correct, not an exception thrown out of `run()`.

**A refusal is data, not an exception.** A denied tool, a sandbox violation and a
crashing tool all come back as tool results. `run()` does not raise on any of them.
The one exception is `BudgetExceeded`, which is the run's hard ceiling and is
re-raised.

One member of that family is worth knowing about because the failure it prevents is
silent. When a model emits a tool call whose JSON arguments will not parse,
langchain puts it in `invalid_tool_calls` and leaves `tool_calls` **empty** — and a
loop reading only `tool_calls` sees "no request" and ends the run `target_met` with
no answer. `AgentNode` reads both and sends the parse error back as that call's
result, telling the model nothing ran and to re-send it with corrected JSON. The
turn still counts against the loop's limits.

`AgentNode.run()` can be called with no `RunContext` at all, as here; it builds a
fresh unbounded one so the node is usable outside a graph. Pass a real `ctx` to get
budgets and traces (next recipe).

---

## How do I find out why the loop stopped?

`AgentResult.termination_reason` is always set — the loop cannot fall out of the
bottom — and `note` says why *this* reason and not just which.

<!-- cookbook name=05-termination.py -->
```python
import tempfile
from pathlib import Path

from cookbook_model import ToolScriptedModel

from grapharc.harness import (
    AgentNode,
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.runtime.budget import Budget, BudgetMeter
from grapharc.runtime.graph import RunContext
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")


def build(**agent_kwargs):
    registry = ToolRegistry()
    register_core_tools(registry, workspace, include=["read_file"])
    policy = PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")])
    harness = Harness(registry, policy, executor=LocalExecutor(), workspace=str(workspace))
    read = [{"name": "read_file", "args": {"path": "calc.py"}, "id": "r"}]
    model = ToolScriptedModel(
        responses=["still working"], on_exhausted="repeat", tool_call_script=[read]
    )
    return AgentNode(model, harness, **agent_kwargs)


# 1. The model keeps re-reading the same file and learning nothing new.
stalled = build(max_iterations=10).run("read it forever")
print(stalled.termination_reason, "|", stalled.note)
print("  answer:", repr(stalled.output), " partial:", repr(stalled.partial_output))

# 2. The turn cap, with the stall guard raised out of the way.
capped = build(max_iterations=3, max_stalled_turns=99).run("read it forever")
print(capped.termination_reason, "|", capped.note)

# 3. The meter's ceiling, checked before every turn.
ctx = RunContext(run_id="demo", graph="cookbook", meter=BudgetMeter(Budget(max_tokens=200)))
broke = build(max_iterations=99, max_stalled_turns=99).run("read it forever", ctx)
print(broke.termination_reason, "|", broke.note)
```

Output:

```
no_progress | 2 consecutive turns produced no new tool result
  answer: ''  partial: 'still working'
max_iterations | iteration cap reached (3/3)
budget_exhausted | max_tokens reached (335/200)
```

**Why it works this way.** The reasons are `StopReason` members: `target_met`,
`max_iterations`, `no_progress`, `budget_exhausted`, `human_stopped`, `error`.

**Only `target_met` fills `output`.** Look at the first case: the model's last words
were `'still working'`, and they are in `partial_output`, not `answer`. A run that
stopped mid-work has no answer to give, and a downstream node reading `answer`
without reading `termination_reason` must not be handed a plausible-looking
non-answer.

**Progress is measured in observations, not requests.** Re-sending byte-identical
arguments is how you poll until a job is ready, or re-run a suite after an edit —
so a repeated call that returns something *new* counts as progress, and only a
repeat that returns the same result again counts as a stall. That is why case 1
ends at `no_progress`: the file never changed.

`max_stalled_turns` defaults to 2 and `max_iterations` to 8. Note case 3: the loop's
own turn cap and the meter's ceiling are different mechanisms, and setting both to
similar values makes an ordinary turn-limited stop report as `budget_exhausted`.
The `grapharc agent` CLI deliberately leaves iterations to the loop and gives the
meter only tokens and wall clock.

---

## How do I decide which tool an agent may call?

A `PermissionPolicy` is a list of rules over `fnmatch` patterns. They are evaluated
**by tier** — every `deny` rule, then every `ask`, then every `allow` — not in the
order you wrote them.

<!-- cookbook name=06-permissions.py -->
```python
from grapharc.harness import Decision, PermissionPolicy, PermissionRule


def show(label, policy, names):
    print(label)
    for name in names:
        print(f"   {name:<16} {policy.decide(name).value}")


names = ("read_file", "write_file", "run_command", "deploy_to_prod")

show("wildcard allow, narrower ask and deny:", PermissionPolicy(rules=[
    PermissionRule(action=Decision.ALLOW, pattern="*"),           # written first...
    PermissionRule(action=Decision.ASK, pattern="write_file"),
    PermissionRule(action=Decision.DENY, pattern="run_command"),  # ...matched first
]), names)

show("\nno wildcard — everything unnamed falls to the default:", PermissionPolicy(rules=[
    PermissionRule(action=Decision.ALLOW, pattern="read_file"),
    PermissionRule(action=Decision.ASK, pattern="write_file"),
]), names)

show("\na broad deny beats a narrow allow — there is no exception list:",
     PermissionPolicy(rules=[
         PermissionRule(action=Decision.DENY, pattern="*_file"),
         PermissionRule(action=Decision.ALLOW, pattern="read_file"),
     ]), names)

print("\ndefault when nothing matches:", PermissionPolicy().default.value)
```

Output:

```
wildcard allow, narrower ask and deny:
   read_file        allow
   write_file       ask
   run_command      deny
   deploy_to_prod   allow

no wildcard — everything unnamed falls to the default:
   read_file        allow
   write_file       ask
   run_command      deny
   deploy_to_prod   deny

a broad deny beats a narrow allow — there is no exception list:
   read_file        deny
   write_file       deny
   run_command      deny
   deploy_to_prod   deny

default when nothing matches: deny
```

**Why it works this way.** Three things follow from tier-before-order, and the third
is the one that bites.

1. **Rule order in the list is irrelevant.** You can append a deny to an existing
   policy and it takes effect over everything already there.
2. **The default is `deny`.** A tool nobody thought about is a tool that does not
   run. The third policy in the snippet reaches `deny` for `deploy_to_prod` through
   the default, not through the `*_file` rule.
3. **You cannot punch a hole in a deny.** `DENY "*_file"` plus `ALLOW "read_file"`
   does not mean "reads are fine, other file tools are not" — it means everything
   matching `*_file` is denied, `read_file` included. If you want an exception, do
   not write the broad deny; enumerate what you allow.

Patterns match the **tool name** only, never its arguments. `DENY "run_command"`
stops the shell tool entirely; it cannot express "deny `rm` but allow `ls`". That
distinction belongs in a pre-hook, two recipes down.

**A name that is also a pattern.** `fnmatch` reads `[`…`]` as a character class,
so a tool called `exfil[all]` — or an MCP-style `mcp__srv__do[all]` — is not the
same string as the pattern that spells it. A `deny` or `ask` rule therefore also
fires on an **exact literal match**, so pasting a tool's name into a rule refuses
it whatever characters it holds. That fallback is deliberately not extended to
`allow`: equality can only ever add a refusal, never a grant. To *allow* one tool
whose name carries `*`, `?` or `[`, build the rule with
`PermissionRule.literal(Decision.ALLOW, name)`, which escapes the name instead of
widening the match.

---

## How do I make sure a denied tool is never even offered to the model?

You do not have to do anything: `Harness.visible_tools()` filters by policy, and
`AgentNode` binds exactly that set. A denied tool's schema is never described to
the model — and if it asks anyway, the call is still refused.

<!-- cookbook name=07-policy-before-schema.py -->
```python
import tempfile
from pathlib import Path

from cookbook_model import ToolScriptedModel

from grapharc.harness import (
    AgentNode,
    Decision,
    Harness,
    LocalExecutor,
    PermissionDenied,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
    tool_schemas,
)
from grapharc.tools import CORE_TOOL_NAMES, register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

registry = ToolRegistry()
register_core_tools(registry, workspace)  # all seven are registered
policy = PermissionPolicy(rules=[
    PermissionRule(action=Decision.DENY, pattern="run_command"),
    PermissionRule(action=Decision.DENY, pattern="write_file"),
    PermissionRule(action=Decision.ALLOW, pattern="*"),
])
harness = Harness(registry, policy, executor=LocalExecutor(), workspace=str(workspace))

print("registered:", list(CORE_TOOL_NAMES))
print("offered   :", [s["function"]["name"] for s in tool_schemas(harness)])

# The model never sees the schema — and calling it anyway is still refused.
try:
    harness.call("write_file", {"path": "calc.py", "content": "whatever"})
except PermissionDenied as exc:
    print("direct    : PermissionDenied:", exc)

# Inside the loop the refusal is a tool result, not a crash: the run goes on.
model = ToolScriptedModel(
    responses=["deleting the tests", "ok, reading instead", "add() subtracts."],
    tool_call_script=[
        [{"name": "write_file", "args": {"path": "calc.py", "content": ""}, "id": "1"}],
        [{"name": "read_file", "args": {"path": "calc.py"}, "id": "2"}],
        [],
    ],
)
result = AgentNode(model, harness).run("break the build")
print("\nstopped   :", result.termination_reason)
for call in result.tool_calls:
    first_line = call.detail.splitlines()[0]
    print(f"  {call.status.value:<8} refused_by={call.refused_by or '-':<8} {first_line}")
print("calc.py unchanged:", (workspace / "calc.py").read_text().strip().endswith("a - b"))
```

Output:

```
registered: ['read_file', 'write_file', 'edit_file', 'list_dir', 'glob', 'grep', 'run_command']
offered   : ['edit_file', 'glob', 'grep', 'list_dir', 'read_file']
direct    : PermissionDenied: tool 'write_file' denied by policy

stopped   : target_met
  denied   refused_by=policy   PERMISSION_DENIED: tool 'write_file' denied by policy
  ok       refused_by=-        def add(a, b):
calc.py unchanged: True
```

**Why it works this way.** Two enforcement points, and you need both. Filtering the
schemas is the cheap one — the model cannot ask for what it was never offered, and
it costs no tokens describing tools it may not use. Re-checking inside
`Harness.call` is the one that actually holds: schemas are advisory, a model can
hallucinate a tool name, and a second caller may reach the harness without going
through `visible_tools()` at all.

`refused_by` is a recorded field, not something you infer from the message text, so
an audit is a lookup and not a string match. It is `"policy"` for a permission
denial and `"sandbox"` for a sandbox violation. Note the asymmetry that follows:
a sandbox violation is reported to the model as an *error* so the run can continue,
so it has `status=ERROR` and never appears in `result.denied`. Auditing "was this
run refused anything?" through `denied` alone reads clean for a run the sandbox
blocked from end to end — use `result.refused`, which covers both gates.

Finally, note that the run still ended `target_met`. A denial is not a failure of
the run; it is information the model has to work around. Killing the run on a
denial would teach every caller to widen their policy until nothing is denied.

---

## How do I put a human in front of a tool?

Mark it `ASK` and give the harness an `approval` callback. It receives the tool
name and the arguments and returns a bool.

<!-- cookbook name=08-approval.py -->
```python
import tempfile
from pathlib import Path
from typing import Any

from grapharc.harness import (
    Decision,
    Harness,
    LocalExecutor,
    PermissionDenied,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

policy = PermissionPolicy(rules=[
    PermissionRule(action=Decision.ASK, pattern="write_file"),
    PermissionRule(action=Decision.ALLOW, pattern="read_file"),
])

asked: list[tuple[str, dict[str, Any]]] = []


def approve_small_writes(tool: str, args: dict[str, Any]) -> bool:
    """The gate. Return True to let the call through, False to refuse it."""
    asked.append((tool, args))
    return len(args.get("content", "")) < 100


def harness_with(approval):
    registry = ToolRegistry()
    register_core_tools(registry, workspace, include=["read_file", "write_file"])
    return Harness(registry, policy, executor=LocalExecutor(),
                   workspace=str(workspace), approval=approval)


gated = harness_with(approve_small_writes)
print("approved  :", gated.call("write_file", {"path": "calc.py", "content": "ok\n"}))
try:
    gated.call("write_file", {"path": "big.py", "content": "x" * 200})
except PermissionDenied as exc:
    print("refused   :", exc)
print("asked     :", [(t, sorted(a)) for t, a in asked])

# No approval channel at all: ASK fails closed. It is not "allow if nobody objects".
try:
    harness_with(None).call("write_file", {"path": "calc.py", "content": "hi\n"})
except PermissionDenied as exc:
    print("no channel:", exc)

# ALLOW never reaches the gate.
print("allowed   :", repr(gated.call("read_file", {"path": "calc.py"})))
print("asked     :", len(asked), "times total")
```

Output:

```
approved  : overwrote calc.py (3 bytes)
refused   : tool 'write_file' requires approval and none was granted
asked     : [('write_file', ['content', 'path']), ('write_file', ['content', 'path'])]
no channel: tool 'write_file' requires approval and none was granted
allowed   : 'ok\n'
asked     : 2 times total
```

**Why it works this way.** `ASK` with no `approval=` is `DENY`. There is no
"proceed unless someone objects" mode — an absent channel means no approval, and
the count at the end proves the callback was never even reached in that case.

Unlike the policy, the callback **does** see the arguments, so this is where
argument-level rules go: a size cap, a path allowlist, a check that the diff does
not touch `main`. The refusal is a `PermissionDenied`, which `AgentNode` turns into
an ordinary `PERMISSION_DENIED` tool result — the model can revise and try again,
which is usually what you want from a human gate.

Two limits worth stating. This gate is **synchronous**: it blocks inside
`Harness.call` while it waits, so it suits a terminal prompt or an in-process
policy function, not a Slack round trip that takes ten minutes. For an approval
that outlives the process, GraphARC has a different mechanism — the session runtime
holds the *node* before it executes and records an `ApprovalRequest` in the session
store (`grapharc.session.approval`), so nothing the node would have done has
happened yet and a rejection costs nothing to honour.

The CLI's own gate (`grapharc agent --ask 'write_*'`) shows the other failure to
plan for: it refuses when `--json` is set or stdin is not a tty, because a prompt
written into a pipe either hangs the run or reads the next line of piped data as
consent.

---

## How do I block or rewrite a call deterministically?

Pre-hooks run after the permission check and before the executor. They can deny a
call or rewrite its arguments. Post-hooks transform the result.

<!-- cookbook name=09-hooks.py -->
```python
import tempfile
from pathlib import Path
from typing import Any

from grapharc.harness import (
    Decision,
    Harness,
    HookAction,
    HookDecision,
    LocalExecutor,
    PermissionDenied,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
(workspace / ".env").write_text("OPENROUTER_API_KEY=sk-real-key\n")


def no_secrets(tool: str, args: dict[str, Any]) -> HookDecision | None:
    """Deny any read of a dotfile. Returning None means 'no opinion'."""
    if tool == "read_file" and Path(args.get("path", "")).name.startswith("."):
        return HookDecision(action=HookAction.DENY, reason="dotfiles are off limits")
    return None


def bound_reads(tool: str, args: dict[str, Any]) -> HookDecision | None:
    """Rewrite, rather than refuse: cap every read at 50 lines."""
    if tool == "read_file" and not args.get("limit"):
        return HookDecision(action=HookAction.REWRITE, args={**args, "limit": 50})
    return None


def redact(tool: str, args: dict[str, Any], result: Any) -> Any:
    """A post-hook sees the result and may transform it."""
    return result.replace("sk-real-key", "sk-***") if isinstance(result, str) else result


registry = ToolRegistry()
register_core_tools(registry, workspace, include=["read_file", "grep"])
harness = Harness(
    registry,
    PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")]),
    executor=LocalExecutor(),
    workspace=str(workspace),
    pre_hooks=(no_secrets, bound_reads),
    post_hooks=(redact,),
)

try:
    harness.call("read_file", {"path": ".env"})
except PermissionDenied as exc:
    print("denied  :", exc)

print("rewritten:", repr(harness.call("read_file", {"path": "calc.py"})))
print("redacted :", repr(harness.call("grep", {"pattern": "sk-real-key"})))
```

Output:

```
denied  : tool 'read_file' blocked by hook: dotfiles are off limits
rewritten: 'def add(a, b):\n    return a - b\n\n[lines 1-2 of 2]'
redacted : '1 match(es) in . (2 file(s) searched)\n.env:1: OPENROUTER_API_KEY=sk-***'
```

**Why it works this way.** Hooks are code. They fire every time, whatever the model
was told in a system prompt, which is the difference between a rule and a request.
A pre-hook is where argument-level policy belongs, since the permission layer only
ever sees a tool name.

Two mechanics that will surprise you:

* **The first non-`None` pre-hook decision wins and stops the chain.** Not just for
  `DENY` — a `REWRITE` also ends the loop, so `bound_reads` never runs on a call
  `no_secrets` has already rewritten. Order your hooks with that in mind, or make
  one hook do all the work for a given tool. (A `REWRITE` with `args=None` is a
  no-op that still stops the chain.)
* **Post-hooks are different: all of them run, in order, each fed the previous
  one's output.**

And look hard at the last line of that output. `no_secrets` denies `read_file` on a
dotfile — and `grep` walked straight into `.env` and printed the key, because it is
a different tool and the hook never looked at it. The post-hook caught it here, but
only because the secret was a literal it knew to search for. A hook is exactly as
broad as the predicate you wrote; if the rule is "this file is off limits", it has
to be enforced against every tool that can read a file, or enforced by not
registering those tools.

---

## What does the default executor actually confine?

If you do not pass `executor=`, `Harness` builds a `SandboxedExecutor` over
`workspace=`. It forks a child per call, scrubs its environment, and installs a
CPython audit hook.

<!-- cookbook name=10-default-executor.py -->
```python
import tempfile
from pathlib import Path

from grapharc.harness import (
    Decision,
    Harness,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

registry = ToolRegistry()
register_core_tools(registry, workspace, exclude=["run_command"])
harness = Harness(
    registry,
    PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")]),
    workspace=str(workspace),  # no executor= -> SandboxedExecutor over this workspace
)
print(type(harness.executor).__name__, "|", harness.executor.workspace == str(workspace))
print(harness.call("edit_file", {"path": "calc.py", "old_string": "-", "new_string": "+"}))
print(harness.call("read_file", {"path": "calc.py"}))
```

Output:

```
SandboxedExecutor | True
edited calc.py: replaced 1 occurrence
def add(a, b):
    return a + b
```

Here is what the hook refuses. Each tool below is an ordinary Python function; none
of them get near the workspace's own guards.

<!-- cookbook name=11-sandbox-refusals.py -->
```python
import site
import tempfile
from pathlib import Path

from grapharc.harness import SandboxedExecutor, SandboxViolation, ToolSpec

SITE_PACKAGES = site.getsitepackages()[0]
executor = SandboxedExecutor(tempfile.mkdtemp(prefix="cookbook-"))


def attempt(name: str, fn) -> None:
    spec = ToolSpec(name=name, description="", fn=fn)
    try:
        print(f"{name:<18} ok       {executor.run(spec, {})!r}")
    except SandboxViolation as exc:
        # Everything after the ';' is advice, and the site-packages path is this
        # machine's; both are trimmed so the output below is stable.
        print(f"{name:<18} REFUSED  "
              f"{str(exc).split(';')[0].replace(SITE_PACKAGES, '<site-packages>')}")


def write_here():
    Path("notes.txt").write_text("scratch")   # cwd *is* the workspace
    return Path("notes.txt").read_text()


def read_stdlib():
    import json                                # imports have to keep working
    return json.__name__


def read_etc():
    return open("/etc/passwd").readline()


def write_sitepackages():
    return open(Path(SITE_PACKAGES) / "evil.pth", "w").write("x")


def shell_out():
    import subprocess
    return subprocess.run(["echo", "hi"], capture_output=True).stdout


def phone_home():
    import socket
    return socket.getaddrinfo("example.com", 80)[0][4]


def go_native():
    import ctypes
    return ctypes.CDLL(None).getpid()


for name, fn in [
    ("write_here", write_here),
    ("read_stdlib", read_stdlib),
    ("read_etc", read_etc),
    ("write_sitepackages", write_sitepackages),
    ("shell_out", shell_out),
    ("phone_home", phone_home),
    ("go_native", go_native),
]:
    attempt(name, fn)
```

Output:

```
write_here         ok       'scratch'
read_stdlib        ok       'json'
read_etc           REFUSED  tool 'read_etc' touched '/etc/passwd' outside its workspace (open)
write_sitepackages REFUSED  tool 'write_sitepackages' tried to modify '<site-packages>/evil.pth' outside its workspace (open)
shell_out          REFUSED  tool 'shell_out' tried to spawn a process (subprocess.Popen)
phone_home         REFUSED  tool 'phone_home' attempted network access (socket.getaddrinfo) without declaring needs_network
go_native          REFUSED  tool 'go_native' tried to reach native code through ctypes (ctypes.dlsym)
```

**Why it works this way.** There are **two grants, deliberately different sizes**.
Reads reach the workspace *and* the interpreter's runtime paths — stdlib,
site-packages, `/usr/lib` — because a tool has to be able to import, which is why
`read_stdlib` passes. Mutations reach the workspace and nothing else. That split is
the whole point: while the runtime paths were writable, a tool could drop a `.pth`
file into site-packages, and a `.pth` file executes arbitrary Python on every later
start of that interpreter — an escape from this hook, in another process, outliving
the run.

The rest follows from one rule: **anything that leaves this interpreter leaves the
hook.** A subprocess runs without it, so spawning is refused (including
`_posixsubprocess.fork_exec`, the C entry point behind `subprocess`, not just
`Popen`). Machine code runs without it, so `ctypes`, sqlite extension loading, and
importing a compiled extension from outside the runtime paths are refused. Network
is off unless the `ToolSpec` declared `needs_network`, and then it is the whole
network — no per-host filtering.

The cwd is the workspace and `TMPDIR`/`TEMP`/`TMP` point at it, so `tempfile` works
without tripping the path check. `open` is classified per call from the mode in its
own audit arguments; `sqlite3.connect` is treated as a mutation because it opens
read/write in C and the audit event carries no flag saying otherwise, and a
`file:`-prefixed sqlite name is refused outright because sqlite re-decodes that
name itself, so `realpath` would be vetting a fiction.

---

## What does the sandbox *not* confine?

This matters more than the previous recipe. `SandboxedExecutor` is an in-process
audit-hook confinement. **It is not a kernel boundary**, and the holes are known,
documented and still open.

<!-- cookbook name=12-sandbox-holes.py -->
```python
import os
import tempfile
from pathlib import Path

from grapharc.harness import SandboxedExecutor, SandboxViolation, ToolSpec

executor = SandboxedExecutor(tempfile.mkdtemp(prefix="cookbook-"))

# A file outside the workspace, a descriptor onto it, a module global and an
# environment variable — all established in the parent, before the fork.
outside = Path(tempfile.mkdtemp()) / "outside.txt"
outside.write_text("a secret on disk\n")
leaked_fd = os.open(outside, os.O_RDONLY)
API_KEY = "sk-a-secret-this-module-already-holds"
os.environ["MY_API_KEY"] = "sk-a-secret-in-the-environment"


def run(name, fn):
    spec = ToolSpec(name=name, description="", fn=fn)
    try:
        print(f"{name:<14} {executor.run(spec, {})!r}")
    except SandboxViolation as exc:
        print(f"{name:<14} REFUSED  {str(exc).replace(str(outside.parent), '<tmp>')}")


run("open()", lambda: outside.read_text())        # confined: content reads are checked
run("os.stat", lambda: os.stat(outside).st_size)  # not confined: no audit event exists
run("raw fd", lambda: os.read(leaked_fd, 64))     # not confined: an fd carries no path
run("global", lambda: API_KEY)                    # not confined: fork copies the heap
run("environ", lambda: os.environ.get("MY_API_KEY"))  # confined: the env is scrubbed
```

Output:

```
open()         REFUSED  tool 'open()' touched '<tmp>/outside.txt' outside its workspace (open)
os.stat        17
raw fd         b'a secret on disk\n'
global         'sk-a-secret-this-module-already-holds'
environ        None
```

**Why it works this way.** Read those five lines as the actual threat model.

* **Metadata reads.** CPython raises no audit event for `os.stat`, so size, mtime
  and existence of files anywhere on the host stay readable. Content reads and
  mutations do not.
* **Raw file descriptors.** `os.read`, `os.write`, `os.dup` and `mmap` on an
  already-open descriptor carry no path to check, and `fork` hands the child every
  descriptor the parent had open. The descriptors cannot simply be closed: the child
  cannot tell `multiprocessing`'s own sentinel pipe from an inherited one, and
  closing that breaks the timeout-kill path. `os.readlink` *is* guarded, so
  `/proc/self/fd/N` cannot be used to enumerate them — which costs an attacker the
  enumeration step, not the guess.
* **The forked heap.** `fork` copies the parent's memory. `os.environ` is scrubbed
  to an allowlist (hence `environ -> None`), but a secret a parent module already
  holds in a global stays reachable through `sys.modules`. Only a non-`fork` start
  method would fix this, and that would require every tool function to be
  picklable.
* **Monkeypatched guards.** The `os.readlink` guard is a function wrapper, not an
  audit hook, so unlike everything else here it is in principle removable from
  inside — the original is reachable through the wrapper's closure. An audit hook
  cannot be unregistered; this can.
* **Trusted runtime extensions.** Compiled modules shipped with the interpreter and
  its site-packages are trusted by necessity — importing any of them executes native
  code the hook never sees.

The practical reading: `SandboxedExecutor` is a strong guard against a *confused*
tool and a decent one against a careless model. Against a tool you would not run on
your laptop, it is not a boundary. Use a container.

---

## Why do my tools say "outside its workspace" when the path is right?

Because `Harness(registry, policy)` without `workspace=` builds a `SandboxedExecutor`
over a **fresh temp directory**, which is not the directory your tools were built
for. Two confinements, two roots, and the sandbox wins.

<!-- cookbook name=13-workspace-mismatch.py -->
```python
import tempfile
from pathlib import Path

from grapharc.harness import (
    Decision,
    Harness,
    PermissionPolicy,
    PermissionRule,
    SandboxViolation,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

registry = ToolRegistry()
register_core_tools(registry, workspace, exclude=["run_command"])
harness = Harness(  # note the missing workspace= argument
    registry,
    PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")]),
)
print("same directory?  :", harness.executor.workspace == str(workspace))
try:
    harness.call("read_file", {"path": "calc.py"})
except SandboxViolation as exc:
    print("result           :", str(exc).replace(str(workspace), "<tools-ws>"))
```

Output:

```
same directory?  : False
result           : tool 'read_file' touched '<tools-ws>/calc.py' outside its workspace (open)
```

**Why it works this way.** The tools confine paths and the executor confines paths,
independently and on purpose — `LocalExecutor` confines nothing, and `run_command`
leaves the interpreter the audit hook lives in, so neither layer can be the only
one. The cost is that you have to point both at the same directory. Pass the same
path to `register_core_tools` and to `Harness(..., workspace=...)`, or pass an
executor you built yourself.

---

## Why does `run_command` fail under the default executor?

Because the audit hook refuses to spawn a process, and `run_command` spawns a
process. This is a design decision, not a bug, and there is no flag to turn it off.

<!-- cookbook name=14-run-command.py -->
```python
import tempfile
from pathlib import Path

from grapharc.harness import (
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    SandboxViolation,
    ToolRegistry,
)
from grapharc.tools import register_core_tools

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n")
policy = PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")])


def harness_with(executor):
    registry = ToolRegistry()
    register_core_tools(registry, workspace, include=["run_command"])
    return Harness(registry, policy, executor=executor, workspace=str(workspace))


try:  # the default executor — SandboxedExecutor — refuses every subprocess
    harness_with(None).call("run_command", {"argv": ["echo", "hi"]})
except SandboxViolation as exc:
    print("sandbox:", exc)

print("local  :", harness_with(LocalExecutor()).call("run_command", {"argv": ["ls"]}))
```

Output:

```
sandbox: tool 'run_command' tried to modify '/dev/null' outside its workspace (open); the runtime paths are readable but never writable — code left in them runs in later interpreters, where no hook is watching
local  : exit code 0 after 0.00s
stdout:
calc.py

stderr:
(empty)
```

**Why it works this way.** Note *which* refusal you get: not the spawn, but the
`/dev/null` open that `stdin=DEVNULL` performs a moment earlier. Same outcome, one
audit event sooner — worth knowing so you recognise it.

So `run_command` needs `LocalExecutor` (no confinement at all) or
`ContainerExecutor` (a real one). Under `LocalExecutor` the child is an ordinary
process with your privileges and can reach the whole filesystem: what limits it is
the permission policy deciding whether it may run at all. Registering `run_command`
is a decision of a different size from registering `read_file`.

What `run_command` *does* give you is discipline around the call. `argv` is a list
and `shell=False`, always — a single string is refused rather than split, because
splitting reintroduces quoting rules you never asked for and `shell=True` makes
every model-supplied substring an injection site. A pipeline is spelled
`["bash", "-lc", "..."]`: explicit, visible in the tool-call record, and your
decision to permit. The deadline kills the whole process group rather than only the
child it started, and the environment is an allowlist so your provider key is not
handed to whatever was run.

---

## How do I get a real boundary?

`ContainerExecutor` runs each call in a fresh throwaway container, so the
confinement is the kernel's — namespaces, cgroups, dropped capabilities — rather
than CPython's. It has the same `run(spec, args)` interface, so callers never
branch on which executor they hold.

This snippet needs a working `docker` or `podman` and the `python:3.12-slim` image.
It was run on Docker 29.4.1; the two Python version strings are this machine's and
yours will differ.

<!-- cookbook name=15-container.py requires=docker compare=off -->
```python
import os
import platform
import tempfile
from pathlib import Path

from grapharc.harness import SandboxViolation, ToolSpec
from grapharc.harness.container import ContainerExecutor, runtime_available

print("runtime reachable:", runtime_available())

workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n")
executor = ContainerExecutor(str(workspace), image="python:3.12-slim")

version = ToolSpec(name="version", description="", fn=platform.python_version)
listing = ToolSpec(name="listing", description="", fn=os.listdir)
mkdir = ToolSpec(name="mkdir", description="", fn=os.mkdir)

print("entrypoints      :", executor.entrypoint_for(version), executor.entrypoint_for(listing))
print("python here      :", platform.python_version())
print("python in image  :", executor.run(version, {}))
print("workspace inside :", executor.run(listing, {"path": "/workspace"}))
print("host -> container:", executor.container_path(workspace / "calc.py"))

executor.run(mkdir, {"path": "/workspace/made-inside"})
print("host sees        :", sorted(p.name for p in workspace.iterdir()))


# The constraint: the container resolves a tool by importing `module:qualname`.
def my_tool() -> str:
    return "never runs"


try:
    executor.run(ToolSpec(name="my_tool", description="", fn=my_tool), {})
except SandboxViolation as exc:
    print("\nlocal function   :", str(exc).split(";")[0])
```

Output:

```
runtime reachable: True
entrypoints      : platform:python_version posix:listdir
python here      : 3.14.6
python in image  : 3.12.13
workspace inside : ['calc.py']
host -> container: /workspace/calc.py
host sees        : ['calc.py', 'made-inside']

local function   : tool 'my_tool' is defined in '__main__', which names a different file inside the container than it does here
```

**Why it works this way — and the constraint you will hit in the first five
minutes.** A container is a different filesystem and a different interpreter;
`spec.fn` is a live object in *this* one and cannot cross. What crosses is an
import path — `module:qualname`, derived from `spec.fn` — and the tool is resolved
by importing that module **inside the container**. So the tool's code must already
be installed in the image. That is why the snippet's tools are stdlib functions:
`python:3.12-slim` contains no GraphARC and none of your code.

Which means the honest summary is: `ContainerExecutor` is not a drop-in replacement
for `SandboxedExecutor` on the core toolset. Running your own tools in it means
building an image that has them, and then either the derived path resolves there or
you name it yourself with `entrypoints={"my_tool": "mypkg.tools:my_tool"}`.

Refused before a container even starts: lambdas and nested functions (`<lambda>`,
`<locals>` in the qualname), `functools.partial` and callable objects (no
`__qualname__` to derive), bound methods (the derived path names the underlying
function and `self` cannot cross), and anything in `__main__`, as above. Where the
derived path *can* be checked without importing anything — the named module is
already in this process's `sys.modules` — it is checked here; where it cannot, the
question is deliberately left to the container, because importing a module in the
host to validate a name would execute its top-level code unsandboxed, which is the
failure this class exists to prevent.

Also crossing the boundary as JSON only: arguments go in as a JSON object on stdin
and become keyword arguments, and the result comes back as JSON on stdout. Objects,
file handles and generators are refused rather than mangled; a tuple comes back as
a list and non-string dict keys arrive as strings. Paths in `args` must be
*container* paths — `container_path()` does that translation. Anything the tool
prints is discarded: fd 1 is pointed at stderr before the tool gets control, so
only the return value crosses and a print cannot forge a result.

What the container gets: one bind mount (the workspace, read-write, at
`/workspace`, which is also the cwd), `--network none` unless the tool declared
`needs_network`, all capabilities dropped, `no-new-privileges`, a non-root uid, a
read-only rootfs, and memory and pid limits. Nothing else from the host — but the
*image's* filesystem is there, which is why the image is part of the security
decision and is yours to choose.

Limits stated rather than hidden: the container runtime is trusted (Docker's daemon
runs as root; a container escape defeats this), the image's contents are not
verified, the read-only rootfs does not cover the read-write workspace and no disk
quota is imposed, and `--user` means different things under docker and rootless
podman — under rootless podman pass `extra_run_args=("--userns=keep-id",)` or set
`user=` to match. If no runtime is found, `run()` refuses rather than falling back
to running the tool locally: a silent downgrade from "contained" to "not contained"
is the bug this class exists to prevent.

---

## How do I put an agent inside a graph?

`AgentNode` is callable as a node, and `agent.writes` reports exactly the state
fields it writes, so the graph's write-permission check stays declarative.

<!-- cookbook name=16-agent-in-graph.py -->
```python
import tempfile
from pathlib import Path

from cookbook_model import ToolScriptedModel

from grapharc import GraphARC, GraphARCState
from grapharc.harness import (
    AgentNode,
    Decision,
    Harness,
    LocalExecutor,
    PermissionPolicy,
    PermissionRule,
    ToolRegistry,
)
from grapharc.runtime.graph import END, START
from grapharc.tools import register_core_tools


class State(GraphARCState):
    task: str = ""
    answer: str = ""
    termination_reason: str = ""
    verdict: str = ""


workspace = Path(tempfile.mkdtemp(prefix="cookbook-"))
(workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")

registry = ToolRegistry()
register_core_tools(registry, workspace, include=["read_file"])
harness = Harness(
    registry,
    PermissionPolicy(rules=[PermissionRule(action=Decision.ALLOW, pattern="*")]),
    executor=LocalExecutor(),
    workspace=str(workspace),
)
model = ToolScriptedModel(
    responses=["", "add() subtracts where it should add."],
    tool_call_script=[[{"name": "read_file", "args": {"path": "calc.py"}, "id": "1"}], []],
)
agent = AgentNode(model, harness)  # reads state.task; writes answer + termination_reason
print("agent.writes:", sorted(agent.writes))


def judge(state: State) -> dict:
    """A downstream node reads the reason, not just the answer."""
    return {"verdict": "reviewed" if state.termination_reason == "target_met" else "incomplete"}


g = GraphARC(State, name="review")
g.add_node("agent", agent, writes=agent.writes)
g.add_node("judge", judge, writes={"verdict"})
g.add_edge(START, "agent")
g.add_edge("agent", "judge")
g.add_edge("judge", END)

out = g.compile().invoke({"task": "what is wrong with calc.py?"})
print("answer      :", out["answer"])
print("reason      :", out["termination_reason"])
print("verdict     :", out["verdict"])
```

Output:

```
agent.writes: ['answer', 'termination_reason']
answer      : add() subtracts where it should add.
reason      : target_met
verdict     : reviewed
```

**Why it works this way.** The state field names are configurable
(`task_field`, `output_field`, `reason_field`, `record_field`), and `writes` is
derived from whichever you chose, so the two can never drift apart. For a prompt
richer than one state field, pass `prompt_fn=lambda state: ...`; `observe()` raises
`AgentConfigError` if it finds neither.

An agent turn inside a graph is metered like any other node: `AgentNode` charges one
iteration per model call against the `RunContext`'s meter, and names the message
when it charges tokens so the runtime's usage callback recognises a re-report rather
than double-counting. Set `record_field` to also land the whole `AgentResult` — every
tool call, every refusal — in state.

Traces get sub-node phases of their own: `"model"` per model call, `"tool"` per tool
call, `"stop"` for the termination. Not `"start"`/`"end"`, because `observe.metrics`
counts node executions and run tokens from `"end"` events and reusing that phase
inside a node would double-count what the node wrapper already reported.

---

## How do I see all of this working end to end?

`grapharc/examples/agent_fixit.py` is the worked example: a project whose `add()`
subtracts, a test suite that catches it, and an agent that has to find the bug, fix
it, and prove the fix by running the tests. `delete_file` is registered and denied,
so the shortest path to a green suite is closed off in code.

The example ships a `--model` flag for a real model. Here it is driven by the
scripted model instead, so it runs with no key — the harness, the policy and the
verification are the example's own.

<!-- cookbook name=17-fixit.py -->
```python
from cookbook_model import ToolScriptedModel

from grapharc.examples.agent_fixit import (
    SYSTEM_PROMPT,
    TASK,
    build_harness,
    make_workspace,
    run_pytest,
)
from grapharc.harness import AgentNode

workspace = make_workspace()          # calc.py with `a - b`, and a test that wants `a + b`
harness = build_harness(workspace)    # read/write/run_tests, and a denied delete_file

print("registered:", sorted(harness.registry.get(n).name
                            for n in ("read_file", "write_file", "delete_file", "run_tests")))
print("offered   :", [spec.name for spec in harness.visible_tools()])
print("before    :", (workspace / "calc.py").read_text().strip())

model = ToolScriptedModel(
    responses=["", "", "", "add() adds now; the suite is green."],
    tool_call_script=[
        [{"name": "delete_file", "args": {"path": "test_calc.py"}, "id": "1"}],
        [{"name": "write_file",
          "args": {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
          "id": "2"}],
        [{"name": "run_tests", "args": {}, "id": "3"}],
        [],
    ],
)
result = AgentNode(model, harness, system_prompt=SYSTEM_PROMPT, max_iterations=12).run(TASK)

print("\nstopped   :", result.termination_reason, "| turns:", result.iterations)
for call in result.tool_calls:
    print(f"  {call.status.value:<8} {call.tool:<12} refused_by={call.refused_by or '-'}")
print("after     :", (workspace / "calc.py").read_text().strip())
print("tests exist:", (workspace / "test_calc.py").exists())
print("independent test run:", "PASS" if run_pytest(workspace).returncode == 0 else "FAIL")
```

Output:

```
registered: ['delete_file', 'read_file', 'run_tests', 'write_file']
offered   : ['read_file', 'run_tests', 'write_file']
before    : def add(a, b):
    return a - b

stopped   : target_met | turns: 4
  denied   delete_file  refused_by=policy
  ok       write_file   refused_by=-
  ok       run_tests    refused_by=-
after     : def add(a, b):
    return a + b
tests exist: True
independent test run: PASS
```

**Why it works this way.** The last line is the point of the whole example. The
agent said the suite was green; `run_pytest` is run *afterwards, outside the loop*
and says so independently. An agent's claim about its own work is not evidence.

Note also that `build_harness` uses `LocalExecutor` and says why in its docstring:
`run_tests` shells out to pytest, which `SandboxedExecutor` refuses by design. The
example leans on the permission layer instead, which still applies — hence
`delete_file` denied, never offered, and refused when asked for anyway. A task
needing both real isolation and a subprocess wants `ContainerExecutor`.

To run the example against a real model, which costs money and is **not run here**:

<!-- cookbook norun="needs an OpenRouter API key" -->
```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python -m grapharc.examples.agent_fixit --model openrouter/anthropic/claude-haiku-4.5
```

---

## How do I run an agent from the shell?

`grapharc agent` takes the task from you, the tools from `grapharc.tools`, and the
permissions from flags. `--allow`, `--deny` and `--ask` are repeatable globs;
`--executor` picks `sandbox` (default) or `local`; `--max-turns`, `--max-tokens` and
`--max-seconds` bound the run; `--json` prints one document instead of prose.

The `mock/` backend resolves to `ScriptedChatModel`, which — as the loop recipe
above explained — has no `bind_tools`. That makes it a way to see the wiring without
spending anything: the tools are loaded, the policy is applied, and the command then
refuses to pretend a text-only model can drive a tool loop.

<!-- cookbook shell name=18-cli.sh exit=2 -->
```bash
grapharc agent --model mock/none --workspace /tmp/grapharc-cookbook-ws --run-id demo \
  --deny 'run_command' --deny 'write_file' --ask 'edit_file' --json "list the files"
```

Output:

```
{
  "ok": false,
  "command": "agent",
  "error": "model ScriptedChatModel does not implement bind_tools, so it cannot drive a tool loop (5 tools are visible to it); use a tool-calling backend such as openrouter/*, openai/* or ollama/*",
  "task": "list the files",
  "model": "mock/none",
  "run_id": "demo",
  "workspace": "/tmp/grapharc-cookbook-ws",
  "trace": "/tmp/grapharc-cookbook-ws/trace.jsonl",
  "executor": "sandbox",
  "tools_from": "grapharc.tools.register_core_tools",
  "policy": {
    "allow": [
      "*"
    ],
    "ask": [
      "edit_file"
    ],
    "deny": [
      "run_command",
      "write_file"
    ]
  },
  "tools_visible": [
    "edit_file",
    "glob",
    "grep",
    "list_dir",
    "read_file"
  ]
}
```

**Why it works this way.** The payload answers the three questions a run is graded
on: what it was allowed to do (`policy`, `tools_visible`), what it did (`tool_calls`,
absent here because none ran), and why it stopped (`termination_reason`, `note` — or
`error`, as here). `tools_visible` is the same policy-before-schema filter from
earlier, visible from the shell: `write_file` and `run_command` were registered and
are not in the list.

Exit codes are part of the interface. This command exits **2** — it could not run at
all. Only `termination_reason == "target_met"` exits 0; every other termination (the
turn cap, a stall, an exhausted budget) exits 1, because a script that ran an agent
needs to know the task was not finished without parsing prose first.

With a real key it is the same command with a tool-calling model. **Not run here** —
it costs money:

<!-- cookbook norun="needs an OpenRouter API key" -->
```bash
export OPENROUTER_API_KEY=sk-or-...
grapharc agent --workspace ./scratch --deny 'run_command' --ask 'write_file' \
  --model openrouter/anthropic/claude-haiku-4.5 \
  "read calc.py and tell me what is wrong with add()"
```

A trace lands at `<workspace>/trace.jsonl` either way; `grapharc trace`,
`grapharc metrics` and `grapharc viz` read it.
