# Models — the gateway

A GraphARC node takes a LangChain chat model object. Which provider is behind that object
is a string, resolved at run time by `grapharc.gateway.get_model`, so swapping a subscription
for an API, or an author for a reviewer, is a config change rather than an edit to the graph.

Every Python snippet below is executed by `tests/test_cookbook_models.py`, and the output
blocks are that test's captured output — stdout and stderr merged, unbuffered. Two snippets
need a real credential and are marked as such; those were **not** run, and no output is
claimed for them.

---

## How do I get a model?

`get_model(spec)` is the only entry point you need. A spec is `backend/model`; `describe()`
tells you how one splits without building anything.

<!-- verified -->
```python
from grapharc.gateway import describe, get_model

for spec in (
    "claude-cli/claude-sonnet-5",
    "openrouter/anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "ollama/llama3.1",
    "mock/anything",
):
    print(describe(spec))

# The mock backend needs no credential and never leaves the process.
model = get_model("mock/anything", responses=["hello from a scripted model"])
print(model.invoke("say hi").content)
```

```text
{'spec': 'claude-cli/claude-sonnet-5', 'backend': 'claude-cli', 'model': 'claude-sonnet-5'}
{'spec': 'openrouter/anthropic/claude-haiku-4.5', 'backend': 'openrouter', 'model': 'anthropic/claude-haiku-4.5'}
{'spec': 'openai/gpt-4o-mini', 'backend': 'openai', 'model': 'gpt-4o-mini'}
{'spec': 'ollama/llama3.1', 'backend': 'ollama', 'model': 'llama3.1'}
{'spec': 'mock/anything', 'backend': 'mock', 'model': 'anything'}
hello from a scripted model
```

There are five backends:

| backend | credential | what it is for |
| --- | --- | --- |
| `claude-cli` | a Claude subscription, no API key | text completion on quota you already pay for |
| `openrouter` | `OPENROUTER_API_KEY` | one key, most vendors, per-call cost in the response |
| `openai` | `OPENAI_API_KEY` | the OpenAI API directly, or any endpoint via `OPENAI_BASE_URL` |
| `ollama` | none — a local server | models on your own machine, free and offline |
| `mock` | none | a scripted test double |

Extra keyword arguments go straight to the backend's constructor — `temperature=0`,
`max_tokens=512`, and the gateway-wide `retry_policy=` / `cost_ceiling_usd=` /
`price_per_million=` / `spend=` all arrive this way. The model half of a `mock/` spec is
decoration; only `responses=` matters.

**Why it works this way.** The registry imports each adapter lazily, inside the branch that
needs it. Asking for `claude-cli` therefore does not require `langchain-openai`, and asking
for `openrouter` does not require the Claude CLI to be installed. A missing optional
dependency fails for the backend that wanted it and nothing else.

`openrouter`, `openai` and `ollama` all speak the OpenAI wire format and share one base
class, so they behave identically on everything except money and routing: same
`bind_tools`, same `with_structured_output`, same streaming and async, same retry policy,
same usage envelope.

---

## How do I write a spec, and what happens if I typo one?

<!-- verified -->
```python
from grapharc.gateway import split_spec
from grapharc.gateway.registry import UnknownBackendError

print(split_spec("claude-sonnet-5"))                    # bare -> default backend
print(split_spec("openrouter/openai/gpt-4o-mini:floor"))  # author/slug survives
print(split_spec("anthropic/claude-haiku-4.5"))         # a known author, NOT openrouter
print(split_spec("openai/gpt-4o-mini"))                 # a backend that is also an author

try:
    split_spec("opnerouter/openai/gpt-4o-mini")
except UnknownBackendError as exc:
    print(f"UnknownBackendError: {exc}")
```

```text
('claude-cli', 'claude-sonnet-5')
('openrouter', 'openai/gpt-4o-mini:floor')
('claude-cli', 'anthropic/claude-haiku-4.5')
('openai', 'gpt-4o-mini')
UnknownBackendError: unknown backend 'opnerouter' in spec 'opnerouter/openai/gpt-4o-mini'; expected one of: claude-cli, openrouter, openai, ollama, mock — or a bare model name for the claude-cli default
```

Only the first segment is a backend, because OpenRouter model ids are themselves
`author/slug` and must survive intact. `:floor` and `:nitro` suffixes are part of the model
id and pass through untouched.

**The sharp edge is line three.** `anthropic/claude-haiku-4.5` looks like an OpenRouter spec
and is not one — `anthropic` is a recognised *model author*, so the spec is treated as a bare
model name and gets the default backend, `claude-cli`. You will only find out when the Claude
CLI is asked for a model it does not know. If you mean OpenRouter, write `openrouter/` in
front. Unrecognised heads (`opnerouter`) are rejected immediately rather than folded into a
model name, which is the case this rule exists to catch.

**Line four is the same collision resolved the other way.** `openai` is both a backend name
and a model author, and the backend wins: `openai/gpt-4o-mini` is the OpenAI API, which is
what someone typing it means. (Before the backend existed, that string resolved to a bare
model name on `claude-cli` — a spec that could only ever have failed.) Reaching the same
model through the broker is still `openrouter/openai/gpt-4o-mini`.

---

## How do I see what my machine can actually reach?

`grapharc models` prints the backends and a few example specs. It contacts nothing.

<!-- verified: cli -->
```console
$ grapharc models
backends: claude-cli, openrouter, openai, ollama, mock
openrouter key: <unset>
openai key: <unset>
ollama url: http://localhost:11434/v1

examples:
  claude-cli/claude-sonnet-5              subscription, no API key
  openrouter/anthropic/claude-haiku-4.5   many providers, one key
  openrouter/openai/gpt-4o-mini:floor     cheapest provider for that model
  openai/gpt-4o-mini                      the OpenAI API directly, your key
  ollama/llama3.1                         a local server, no key and no bill

grapharc models --check  probes which of these this machine can use
```

(`<unset>` is what you see with no key configured; with one, that line shows a redacted
fingerprint — never the key. The Ollama line is an address rather than a credential, so it
is printed whole; it is where a request *would* go, not evidence that anything is
listening.)

Give it a spec and it resolves that one:

<!-- verified: cli -->
```console
$ grapharc models openrouter/openai/gpt-4o-mini:floor
spec: openrouter/openai/gpt-4o-mini:floor
backend: openrouter
model: openai/gpt-4o-mini:floor
```

`--check` probes credentials, optional dependencies and `PATH`. Output depends on your
machine; this is one real run, on a box with the Claude CLI and Ollama installed and no API
keys at all:

<!-- verified: cli varies -->
```console
$ grapharc models --check
claude-cli   usable    'claude' on PATH at /home/shashank/.local/bin/claude
                       credential: claude subscription login (no API key)
openrouter   unusable  no API key (set OPENROUTER_API_KEY, or add one to .env)
                       credential: <unset>
openai       unusable  no API key (set OPENAI_API_KEY, or add one to .env)
                       credential: <unset>
ollama       usable    local server at http://localhost:11434/v1
                       credential: none needed (local server)
mock         usable    scripted test double; never reaches a provider

local probe only — no provider was contacted, so a configured key
is not a validated one.
```

Read the last two lines literally. `--check` looks for a credential, a package and a binary.
It cannot tell you the key is valid, in credit, or entitled to the model you named — finding
that out costs a request, and this command deliberately does not make one. It exits non-zero
when no *real* provider is usable; `mock` being always-available does not count.

**`ollama usable` is the weakest line in that report, and knowingly so.** There is no
credential to check, so what stands in for one is the `ollama` binary on `PATH` or an
`OLLAMA_HOST` someone set deliberately. Neither says the daemon is running, and neither says
you have pulled the model you are about to name. A stopped server shows up as a connection
error on the first call.

---

## How do I run on a Claude subscription with no API key?

Use the `claude-cli` backend. It drives `claude -p`, so it authenticates with your existing
Claude Code login and there is no API key anywhere.

**This snippet spends subscription quota, so it is not run by the test suite and no output is
claimed for it.** It needs Claude Code installed and logged in.

<!-- needs-credentials -->
```python
from grapharc.gateway import get_model

model = get_model("claude-cli/claude-sonnet-5", timeout_seconds=120)
print(model.invoke("Reply with exactly one word: pong").content)
print(model.last_usage)
```

The tradeoff is real and you should know it before you build on it:

- **No tool-calling.** `bind_tools` raises. No agent node, no ReAct loop, no MCP.
- **No structured output.** `with_structured_output` raises. You parse text yourself.
- **No real streaming, and async only by thread.** Only `_generate` is implemented, so
  `.stream()` yields exactly one chunk — the whole finished message — and `.ainvoke()` runs
  the blocking subprocess call in an executor thread. Neither is an error; neither is what you
  wanted either.
- **No caching you control.** Every call is a fresh `claude -p` process with
  `--no-session-persistence` and an empty working directory, so nothing is carried between
  node calls. The adapter does fold whatever `cache_creation` / `cache_read` counts the CLI
  reports into the usage envelope, but you cannot arrange a cache hit from GraphARC. Keep
  node contexts lean.
- **It spends your subscription quota**, and quota is not dollars. The `total_cost_usd` the
  CLI reports is what the gateway charges the spend meter with; treat a `cost_ceiling_usd` on
  this backend as a proxy for "how much work" rather than a bill you will receive.

The first two are enforced, not documented-and-hoped:

<!-- verified -->
```python
from langchain_core.tools import tool
from pydantic import BaseModel

from grapharc.gateway import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


class Verdict(BaseModel):
    supported: bool


model = get_model("claude-cli/claude-sonnet-5")   # constructs; calls nothing
print(type(model).__name__, "->", model._llm_type)

for label, call in (
    ("bind_tools", lambda: model.bind_tools([get_weather])),
    ("with_structured_output", lambda: model.with_structured_output(Verdict)),
):
    try:
        call()
    except NotImplementedError as exc:
        print(f"{label}: NotImplementedError({str(exc)!r})")
```

```text
ClaudeCodeCLIChatModel -> grapharc-claude-cli
bind_tools: NotImplementedError('')
with_structured_output: NotImplementedError('with_structured_output is not implemented for this model.')
```

**Why it works this way.** `claude -p` is a full agent with its own tools, settings, hooks and
`CLAUDE.md` pickup — none of which GraphARC's permission engine can see or veto. So the
adapter invokes it as a pure inference endpoint and gives up the agent features on purpose.
Here is the exact argv it builds (`_build_argv` is internal; this is shown because the claim
is a security one and you should be able to check it):

<!-- verified -->
```python
from grapharc.gateway import get_model

model = get_model("claude-cli/claude-sonnet-5")
print(model._build_argv(system="be terse"))
```

```text
['claude', '-p', '--output-format', 'json', '--model', 'claude-sonnet-5', '--setting-sources', '', '--no-session-persistence', '--disallowedTools', '*', 'Task', 'Bash', 'BashOutput', 'KillShell', 'Read', 'Write', 'Edit', 'MultiEdit', 'NotebookEdit', 'Glob', 'Grep', 'WebFetch', 'WebSearch', 'TodoWrite', 'SlashCommand', 'Skill', 'ExitPlanMode', '--system-prompt', 'be terse']
```

Every tool denied by name plus a wildcard, no settings sources, no session. The prompt is not
in that list: it travels over stdin, and the flags are an argv array handed to
`subprocess.run` with no shell. A prompt that says "run this command" has no tool to run it
with and no shell to be interpolated into.

---

## How do I get tool-calling, structured output, streaming and async?

Use OpenRouter. One key reaches models from most vendors, and because the backend subclasses
`ChatOpenAI` you get the whole LangChain chat-model surface.

Install the extra and set a key first:

```bash
uv sync --extra openrouter
export OPENROUTER_API_KEY=sk-or-...      # or put it in a .env file
```

<!-- verified -->
```python
from langchain_core.tools import tool
from pydantic import BaseModel

from grapharc.gateway import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


class Verdict(BaseModel):
    supported: bool


# The dummy key is only so this snippet runs offline; nothing below opens a
# socket. Drop `api_key=` and the backend reads OPENROUTER_API_KEY from the
# environment or the nearest .env.
model = get_model("openrouter/openai/gpt-4o-mini", api_key="sk-or-not-a-real-key")

print(model._llm_type, "|", model.model_name)
print("max_tokens:", model.max_tokens, "| sdk max_retries:", model.max_retries)

# Neither of these raises here, and both raise NotImplementedError on claude-cli.
model.bind_tools([get_weather])
model.with_structured_output(Verdict)
print("bind_tools + with_structured_output: available")
print("ainvoke:", callable(model.ainvoke), "| stream:", callable(model.stream))
```

```text
grapharc-openrouter | openai/gpt-4o-mini
max_tokens: 4096 | sdk max_retries: 0
bind_tools + with_structured_output: available
ainvoke: True | stream: True
```

With a real key those bindings do the obvious thing. **The next snippet costs money and was
not run; no output is claimed for it.**

<!-- needs-credentials -->
```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from grapharc.gateway import get_model


class Verdict(BaseModel):
    supported: bool = Field(description="does the evidence support the claim")
    reason: str


model = get_model("openrouter/anthropic/claude-haiku-4.5", temperature=0, max_tokens=512)
verdict = model.with_structured_output(Verdict).invoke(
    [HumanMessage(content="Claim: the sky is green. Evidence: the sky is blue. Supported?")]
)
print(verdict)
```

Two constructor defaults printed by the verified snippet above are worth knowing about.

`max_tokens` defaults to **4096** rather than the model's ceiling. OpenRouter reserves credit
against `max_tokens` before it generates anything, so asking for a 64k ceiling on a small
balance is refused outright with a 402 — even for a ten-token reply. Raise it deliberately
when a node needs the room.

`max_retries` (the OpenAI SDK's own retry layer) defaults to **0**, because GraphARC has its
own. Two layers would compose into `max_attempts * sdk_retries` requests against a provider
that just said 429. Pass `max_retries=` explicitly if you want the SDK's back.

---

## How do I use my own OpenAI key?

`openai/…` goes straight to api.openai.com — no broker in between, which is what a contract
that names who may see the prompt tends to require.

```bash
uv sync --extra openai
export OPENAI_API_KEY=sk-...              # or put it in a .env file
```

The same alternate spellings and `.env` support as every other key: `OPENAI_API_KEY`,
`OPENAI_KEY`, or `openai-api-key` in a file. `langchain-openai` would read the environment
variable by itself; going through the gateway is what adds the file, the spellings, and an
error that names the variable instead of surfacing an SDK exception four frames down.

<!-- verified -->
```python
from grapharc.gateway import get_model

# The dummy key is only so this snippet runs offline; nothing below opens a socket.
model = get_model("openai/gpt-4o-mini", api_key="sk-not-a-real-key")

print(model._llm_type, "|", model.model_name)
print("max_tokens:", model.max_tokens, "| sdk max_retries:", model.max_retries)
print("bind_tools:", callable(model.bind_tools), "| stream:", callable(model.stream))
```

```text
grapharc-openai | gpt-4o-mini
max_tokens: None | sdk max_retries: 0
bind_tools: True | stream: True
```

`max_tokens` is `None` here where OpenRouter defaults it to 4096: that default exists to dodge
OpenRouter's credit reservation, and OpenAI reserves nothing, so a cap here would only truncate
replies for a problem this backend does not have.

**The one thing to know before budgeting against it: the OpenAI API does not tell you what a
call cost.** The response carries token counts and no price. So `cost_usd` is `None`, the call
lands in `SpendMeter.unpriced_calls`, and a `cost_ceiling_usd` on this backend counts calls
instead of enforcing dollars. Two ways to get a real number, both explicit — a price table
baked into this repo would go stale the first time a vendor changed one, and nobody would
notice:

<!-- verified -->
```python
from grapharc.gateway import get_model

priced = get_model(
    "openai/gpt-4o-mini",
    api_key="sk-not-a-real-key",
    price_per_million={"input": 0.15, "cached_input": 0.075, "output": 0.60},
)
unpriced = get_model("openai/gpt-4o-mini", api_key="sk-not-a-real-key")

# `_settle` is what a real call runs after the provider replies; the canned usage
# block below is the shape OpenAI returns, so no request is made here.
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

reply = ChatResult(
    generations=[ChatGeneration(message=AIMessage(content="ok"))],
    llm_output={
        "model_name": "gpt-4o-mini",
        "token_usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 100_000,
            "prompt_tokens_details": {"cached_tokens": 400_000},
        },
    },
)

for label, model in (("with a rate card", priced), ("without one", unpriced)):
    model._settle(reply)
    print(f"{label:<17} cost_usd={model.last_usage['cost_usd']} "
          f"unpriced_calls={model.spend.unpriced_calls}")
```

```text
with a rate card  cost_usd=0.18 unpriced_calls=0
without one       cost_usd=None unpriced_calls=1
```

The other route is to price the whole trace afterwards with `observe.cost.RateCard`, which
keeps the rates in one place instead of on every model object. Token counts are always real
either way, so a *token* budget (`runtime.budget`) bites on this backend whether or not you
priced anything.

`OPENAI_BASE_URL` (or the older `OPENAI_API_BASE`) is honoured, which makes this the backend
for any OpenAI-compatible endpoint that is not Ollama — a corporate gateway, a proxy, a
self-hosted vLLM. Be aware that the trace will still say `grapharc-openai` and your spec
string is the only record of where the request actually went.

---

## How do I run models on my own machine?

`ollama/…` talks to a local [Ollama](https://ollama.com) server over its OpenAI-compatible
endpoint. No key, no bill, no network egress — and, unlike `claude-cli`, full tool-calling,
so it is the cheapest way to exercise an agent node.

```bash
uv sync --extra ollama
ollama pull llama3.1                      # the model half of the spec is a tag you pulled
```

<!-- verified -->
```python
from grapharc.gateway import get_model

model = get_model("ollama/llama3.1")

print(model._llm_type, "|", model.model_name)
print("base_url:", model.openai_api_base)
print("api key sent:", model.openai_api_key.get_secret_value())
```

```text
grapharc-ollama | llama3.1
base_url: http://localhost:11434/v1
api key sent: ollama
```

That "key" is a placeholder. Ollama ignores the `Authorization` header and the OpenAI client
refuses to send an empty one, so a constant is sent and there is nothing in it to protect. Set
`OLLAMA_API_KEY` when the address points at an authenticating proxy rather than the daemon.

`OLLAMA_HOST` — the variable the `ollama` CLI itself reads, so pointing the CLI at a remote box
points GraphARC there too — is accepted in the shorthand forms people actually write it in:

<!-- verified -->
```python
from grapharc.gateway.config import normalize_ollama_base_url

for raw in ("127.0.0.1:11434", "gpu-box", "http://gpu-box:11434", "https://ollama.internal/v1/"):
    print(f"{raw:<28} -> {normalize_ollama_base_url(raw)}")
```

```text
127.0.0.1:11434              -> http://127.0.0.1:11434/v1
gpu-box                      -> http://gpu-box:11434/v1
http://gpu-box:11434         -> http://gpu-box:11434/v1
https://ollama.internal/v1/  -> https://ollama.internal/v1
```

The port is filled in only for the bare form. A value that already has a scheme is a URL and
is left to URL rules, so `https://ollama.internal` stays on 443 rather than being rewritten to
a port nothing is listening on.

Three things to know before you rely on it:

- **Cost is zero, and that is a fact rather than a missing number.** Nobody invoices you for a
  local process, so calls are charged `0.0` and do *not* land in `unpriced_calls` — which
  means "the meter missed a bill", and here there is none to miss. Electricity and an occupied
  GPU are real costs and are not provider charges; pass `price_per_million=` if you want them
  attributed anyway, and that card is used instead.
- **Tool-calling depends on the model you pulled, not on this adapter.** Ollama accepts a
  `tools` array for every model and quietly returns prose for one that was not trained to emit
  tool calls. So `bind_tools` cannot raise the way `claude-cli`'s does; the failure shows up as
  an agent loop that never calls a tool. Pull a model whose card says it supports tools.
- **Nothing here checks that the server is running.** A stopped daemon is a connection error on
  the first call, and `grapharc models --check` reports configuration only.

---

## How do I give a reviewer a genuinely different provider?

A verifier that grades its author's own model family is correlated evidence. `different_providers()`
answers the question, comparing the *vendor* rather than the object.

<!-- verified -->
```python
from grapharc.gateway import different_providers

pairs = [
    ("openrouter/anthropic/claude-haiku-4.5", "openrouter/openai/gpt-4o-mini"),
    ("openrouter/anthropic/claude-opus-4.5", "openrouter/anthropic/claude-haiku-4.5"),
    ("claude-cli/claude-sonnet-5", "openrouter/openai/gpt-4o-mini"),
    ("claude-cli/claude-sonnet-5", "claude-cli/claude-haiku-4.5"),
    ("claude-cli/claude-sonnet-5", "openrouter/anthropic/claude-haiku-4.5"),
    ("openai/gpt-4o-mini", "openrouter/openai/gpt-4o-mini"),
    ("ollama/llama3.1", "openai/gpt-4o-mini"),
]
for author, reviewer in pairs:
    print(f"{different_providers(author, reviewer)!s:<5} {author}  vs  {reviewer}")
```

```text
True  openrouter/anthropic/claude-haiku-4.5  vs  openrouter/openai/gpt-4o-mini
False openrouter/anthropic/claude-opus-4.5  vs  openrouter/anthropic/claude-haiku-4.5
True  claude-cli/claude-sonnet-5  vs  openrouter/openai/gpt-4o-mini
False claude-cli/claude-sonnet-5  vs  claude-cli/claude-haiku-4.5
False claude-cli/claude-sonnet-5  vs  openrouter/anthropic/claude-haiku-4.5
False openai/gpt-4o-mini  vs  openrouter/openai/gpt-4o-mini
True  ollama/llama3.1  vs  openai/gpt-4o-mini
```

Two specs from the same author are `False` even though they are different models — that is
the point. Rows five and six are the same vendor reached two different ways, and they are
`False` too: the comparison is on *vendor*, not on backend. Both used to read as `True`,
because the check short-circuited whenever the two backends differed, and adding a direct
`openai` backend made that failure trivial to hit — `openai/gpt-4o-mini` reviewing
`openrouter/openai/gpt-4o-mini` is the same model twice.

**Two blind spots remain, and neither is fixable by comparing strings.** A re-seller that
fronts someone else's model under its own slug is invisible. And the last row is arguably
wrong in the other direction: `ollama/llama3.1` and a Llama served over OpenRouter are the
same family of weights on two machines, but `ollama` is treated as its own vendor because
what it serves is whatever you pulled. Read the result as "am I obviously grading my own
family", not as a proof of independence.

The CLI wires this into live runs. `grapharc demo stage5 --model … --reviewer-model …` warns
when the pair is correlated and proceeds anyway, because a stated weakness beats a silent one.

---

## How do I add fallbacks and steer provider routing?

OpenRouter has two independent failover layers, and GraphARC exposes both as constructor
arguments. `fallback_models` is model-level — try a different model. `provider_order` and
friends are provider-level — try a different host serving the *same* model.

<!-- verified -->
```python
import json

from langchain_core.messages import HumanMessage

from grapharc.gateway import get_model

model = get_model(
    "openrouter/openai/gpt-4o-mini",
    api_key="sk-or-not-a-real-key",
    fallback_models=["openrouter/anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"],
    provider_order=["openai", "azure"],
    allow_provider_fallbacks=False,
    sort="price",
    max_price_per_million=2.5,
    require_parameters=True,
)

body = model._get_request_payload([HumanMessage(content="hi")])["extra_body"]
print(json.dumps(body, indent=2))
```

```text
{
  "usage": {
    "include": true
  },
  "models": [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash"
  ],
  "provider": {
    "order": [
      "openai",
      "azure"
    ],
    "allow_fallbacks": false,
    "sort": "price",
    "max_price": {
      "prompt": 2.5
    },
    "require_parameters": true
  }
}
```

The primary model is always first in `models`, and an `openrouter/` prefix on a fallback is
stripped. `sort` takes `"price"`, `"throughput"` or `"latency"`; `max_price_per_million` caps
the prompt price; `require_parameters=True` filters out providers that would silently drop a
parameter you sent. Set none of them and the whole `provider` block is omitted rather than
sent empty.

**Why it works this way.** None of these are OpenAI parameters, so the SDK rejects them at the
top level of the request. They ride in `extra_body`, which `langchain-openai` merges into the
JSON body verbatim. `_get_request_payload` is internal — it is used here only to show you the
bytes; in normal use you set the constructor arguments and forget about it.

Failed requests are not billed by OpenRouter, which is why `allow_provider_fallbacks` defaults
to `True`: leaving failover on costs nothing when it fires.

---

## How do I know what a call cost?

Every backend fills in the same `last_usage` envelope after every non-streamed call. This
snippet stubs `subprocess.run` with a canned `claude -p` reply, so it shows the real envelope
and spends nothing.

<!-- verified -->
```python
import json
import subprocess
from unittest.mock import patch

from grapharc.gateway import get_model

CANNED = json.dumps(
    {
        "type": "result",
        "result": "pong",
        "is_error": False,
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 1500,
            "output_tokens": 5,
        },
        "total_cost_usd": 0.0123,
    }
)


class Completed:
    returncode, stdout, stderr = 0, CANNED, ""


model = get_model("claude-cli/claude-sonnet-5")
with patch.object(subprocess, "run", lambda *a, **k: Completed()):
    print(model.invoke("ping").content)
for key, value in model.last_usage.items():
    print(f"  {key}: {value}")
```

```text
pong
  input_tokens: 1812
  output_tokens: 5
  total_tokens: 1817
  input_token_details: {'cache_creation': 300, 'cache_read': 1500}
  uncached_input_tokens: 12
  cost_usd: 0.0123
  model: claude-sonnet-5
  retries: 0
  cumulative_cost_usd: 0.0123
```

`input_tokens` is **1812**, not 12. Cached input is still input: the envelope folds
`cache_creation` and `cache_read` into the total and keeps the breakdown in
`input_token_details`, with the raw uncached figure preserved as `uncached_input_tokens`.
Counting only the provider's `input_tokens` field is how a budget under-counts a real run by
an order of magnitude, because most of a turn's prompt arrives as cache traffic.

The other backends produce the same keys, so a meter reads one shape whichever ran the turn.
`cost_usd` is `None` when nobody could price the call — see the streaming section below, and
the OpenAI section above for the backend where that is the normal case rather than the
exception.

---

## How do I stop a graph spending more than a fixed amount?

A `SpendMeter` accumulates `cost_usd` and refuses to go past a ceiling. It enforces at two
points: after a call, so overspend is bounded by the single call that crossed the line, and
before the next one, so an exhausted budget never reaches the provider at all.

<!-- verified -->
```python
from grapharc.gateway import CostCeilingExceeded, SpendMeter

meter = SpendMeter(ceiling_usd=0.10)
meter.charge(0.04, model="claude-sonnet-5")
meter.charge(0.05, model="claude-sonnet-5")
try:
    meter.charge(0.03, model="gpt-4o-mini")
except CostCeilingExceeded as exc:
    print(exc)
    print("spent after the raise:", exc.spent_usd)

# The next call is refused before it reaches a provider.
try:
    meter.ensure_headroom(model="gpt-4o-mini")
except CostCeilingExceeded as exc:
    print(exc)

print(meter.snapshot())
```

```text
cost ceiling exceeded: $0.120000 spent of $0.100000 after 3 call(s); this call cost $0.030000 — model 'gpt-4o-mini'
spent after the raise: 0.12
cost ceiling reached before this call: $0.120000 spent of $0.100000 over 3 call(s) — model 'gpt-4o-mini'
{'spent_usd': 0.12, 'ceiling_usd': 0.1, 'calls': 3, 'unpriced_calls': 0, 'per_model_usd': {'claude-sonnet-5': 0.09, 'gpt-4o-mini': 0.03}}
```

Note the two different messages — "exceeded" is the call that crossed, "reached before this
call" is every call after it. The crossing call is charged *before* the raise, so a run that
catches `CostCeilingExceeded` still knows what it actually spent.

You rarely build the meter by hand. `cost_ceiling_usd=` seeds one per model; `spend=` shares
one across several, which is what you want for a whole run:

<!-- verified -->
```python
from grapharc.gateway import SpendMeter, get_model

run_budget = SpendMeter(ceiling_usd=0.50)
author = get_model("claude-cli/claude-sonnet-5", spend=run_budget)
reviewer = get_model(
    "openrouter/openai/gpt-4o-mini", api_key="sk-or-not-a-real-key", spend=run_budget
)
print(author.spend is reviewer.spend, run_budget.ceiling_usd)
```

```text
True 0.5
```

End to end, against the same stubbed CLI as before:

<!-- verified -->
```python
import json
import subprocess
from unittest.mock import patch

from grapharc.gateway import CostCeilingExceeded, get_model

CANNED = json.dumps(
    {"result": "pong", "is_error": False, "usage": {}, "total_cost_usd": 0.0123}
)


class Completed:
    returncode, stdout, stderr = 0, CANNED, ""


model = get_model("claude-cli/claude-sonnet-5", cost_ceiling_usd=0.02)
with patch.object(subprocess, "run", lambda *a, **k: Completed()):
    model.invoke("first")                       # 0.0123 — under
    try:
        model.invoke("second")                  # 0.0246 — over
    except CostCeilingExceeded as exc:
        print(exc)
    try:
        model.invoke("third")                   # refused before the call
    except CostCeilingExceeded as exc:
        print(exc)
print("snapshot:", model.spend.snapshot())
```

```text
cost ceiling exceeded: $0.024600 spent of $0.020000 after 2 call(s); this call cost $0.012300 — model 'claude-sonnet-5'
cost ceiling reached before this call: $0.024600 spent of $0.020000 over 2 call(s) — model 'claude-sonnet-5'
snapshot: {'spent_usd': 0.0246, 'ceiling_usd': 0.02, 'calls': 2, 'unpriced_calls': 0, 'per_model_usd': {'claude-sonnet-5': 0.0246}}
```

`calls` stayed at 2: the third invocation never spawned a process.

**The limit, stated plainly.** A ceiling can only enforce against costs the provider reports.
A call that reports none is counted in `unpriced_calls` rather than guessed at, so
`unpriced_calls > 0` is your signal that the ceiling saw less than the whole bill. And the
meter is unsynchronised: sharing one across concurrent nodes means a check can interleave with
a charge, so the bound is approximate by at most the number of calls in flight.

This is separate from `Budget(max_tokens=…)`, which the *runtime* meters per run. The spend
meter is per model object (or per shared group) and counts dollars; the budget is per run and
counts tokens, iterations and seconds.

---

## What happens to cost accounting when I stream?

This is the one place where the accounting is knowingly incomplete, so it gets its own recipe
rather than a footnote.

<!-- verified -->
```python
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from grapharc.gateway import CostCeilingExceeded, SpendMeter, get_model

CHUNKS = [ChatGenerationChunk(message=AIMessageChunk(content=t)) for t in ("po", "ng")]

meter = SpendMeter(ceiling_usd=0.01)
model = get_model(
    "openrouter/openai/gpt-4o-mini",
    api_key="sk-or-not-a-real-key",
    spend=meter,
    streaming=True,
)

with patch.object(ChatOpenAI, "_stream", lambda self, *a, **k: iter(CHUNKS)):
    print("".join(c.content for c in model.stream("ping")))
    print("last_usage:", model.last_usage)
    print("meter:", meter.snapshot())

    meter.spent_usd = 0.05          # some earlier, priced call took it over
    try:
        list(model.stream("ping"))
    except CostCeilingExceeded as exc:
        print(exc)
    print("calls after the refusal:", meter.calls)
```

```text
pong
last_usage: None
meter: {'spent_usd': 0.0, 'ceiling_usd': 0.01, 'calls': 1, 'unpriced_calls': 1, 'per_model_usd': {}}
cost ceiling reached before this call: $0.050000 spent of $0.010000 over 1 call(s) — model 'openai/gpt-4o-mini'
calls after the refusal: 1
```

A streamed call is checked against the ceiling **before** it starts and lands in
`unpriced_calls` afterwards. LangChain routes streamed calls through `_stream`, never
`_generate`, and OpenRouter reports the per-call cost in the final SSE chunk, which
`langchain-openai` does not surface. So `last_usage` is cleared to `None` rather than left
holding the previous call's numbers, and the meter records that it missed one instead of
implying it saw the whole bill.

Streamed calls are also never retried: tokens already handed to the caller cannot be
un-handed. If cost enforcement matters more than time-to-first-token for a node, do not stream
that node.

---

## What gets retried, and what does not?

A model call fails in two ways and treating them alike is expensive in both directions.
`is_transient` is the whole policy, and it is closed by default: an exception has to present
evidence of transience to be retried.

<!-- verified -->
```python
from grapharc.gateway import GatewayError, TransientGatewayError, is_transient


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


cases = [
    HTTPError(429),                      # rate limited
    HTTPError(503),                      # provider down
    HTTPError(529),                      # anthropic "overloaded"
    HTTPError(400),                      # malformed request
    HTTPError(401),                      # bad credential
    HTTPError(402),                      # out of credit
    TimeoutError("read timed out"),
    ConnectionError("reset by peer"),
    TransientGatewayError("overloaded"),
    GatewayError("not logged in"),
    ValueError("I cannot help with that"),   # a refusal is a verdict
]
for exc in cases:
    print(f"{is_transient(exc)!s:<5} {type(exc).__name__}: {exc}")
```

```text
True  HTTPError: HTTP 429
True  HTTPError: HTTP 503
True  HTTPError: HTTP 529
False HTTPError: HTTP 400
False HTTPError: HTTP 401
False HTTPError: HTTP 402
True  TimeoutError: read timed out
True  ConnectionError: reset by peer
True  TransientGatewayError: overloaded
False GatewayError: not logged in
False ValueError: I cannot help with that
```

Retried: 408, 409, 425, 429, any 5xx (529 included), connection and timeout errors that never
reached a verdict, and anything raised as `TransientGatewayError`. Not retried: 400, 401, 402,
403, 404, 422, a content refusal, `CostCeilingExceeded` (spending more cannot fix having spent
too much), and — importantly — **anything unrecognised**. A missed retry costs one failed
call; a wrong retry multiplies a deterministic failure by `max_attempts`.

---

## How long does it wait between attempts?

Delay before attempt *n+1* is `initial * multiplier**(n-1)`, capped at `max_backoff_seconds`,
then multiplied by a random factor in `[1 - jitter, 1]`. The defaults are 3 attempts, 0.5s
initial, 20s cap, ×2, 25% jitter.

<!-- verified: varies -->
```python
from grapharc.gateway import RetryPolicy, call_with_retry


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


policy = RetryPolicy(max_attempts=4, initial_backoff_seconds=0.5, jitter=0.25)
print("un-jittered backoff:", [policy.base_delay(n) for n in range(1, 5)])

attempts, waited = [], []


def flaky():
    attempts.append(1)
    if len(attempts) < 3:
        raise HTTPError(429)
    return "ok"


# `sleep=` is the seam that keeps this fast: nothing actually waits.
print(call_with_retry(flaky, policy=policy, sleep=waited.append))
print("attempts:", len(attempts), "waits:", [round(w, 3) for w in waited])


def refused():
    attempts.append(1)
    raise HTTPError(400)


attempts.clear()
waited.clear()
try:
    call_with_retry(refused, policy=policy, sleep=waited.append)
except HTTPError as exc:
    print(f"{exc} -> attempts: {len(attempts)} waits: {waited}")
```

```text
un-jittered backoff: [0.5, 1.0, 2.0, 4.0]
ok
attempts: 3 waits: [0.486, 0.976]
HTTP 400 -> attempts: 1 waits: []
```

**The `waits:` figures differ on every run** — that is the jitter, and it is the one snippet on
this page whose output the test runs but does not compare byte-for-byte. The shape holds
though: jitter *shrinks* the delay rather than centring it, so every draw is strictly larger
than the previous attempt's, not merely larger on average. A burst of unlucky short waits
hammering a provider that just said 429 is exactly what that buys you.

A provider's `Retry-After` header raises the wait but never lowers it, and is itself capped by
`max_backoff_seconds` — a provider asking for five minutes gets a bounded wait and then an
error, not a silently parked process.

Wire a policy into a model with `retry_policy=`. `NO_RETRY` is the one-attempt policy:

<!-- verified -->
```python
import json
import subprocess
from unittest.mock import patch

from grapharc.gateway import GatewayError, RetryPolicy, get_model

OK = json.dumps({"result": "pong", "is_error": False, "usage": {}})


class Reply:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def scripted(*replies):
    queue, seen = list(replies), []

    def run(argv, **kwargs):
        seen.append(argv)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return run, seen


policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01)

run, seen = scripted(
    Reply(returncode=1, stderr="API Error: 429 rate limit exceeded"),
    Reply(returncode=1, stderr="Error: overloaded_error"),
    Reply(stdout=OK),
)
model = get_model("claude-cli/claude-sonnet-5", retry_policy=policy)
with patch.object(subprocess, "run", run):
    print(model.invoke("ping").content, "after", len(seen), "attempts")
print("retries recorded in the envelope:", model.last_usage["retries"])

run, seen = scripted(Reply(returncode=1, stderr="Invalid API key · Please run /login"))
model = get_model("claude-cli/claude-sonnet-5", retry_policy=policy)
with patch.object(subprocess, "run", run):
    try:
        model.invoke("ping")
    except GatewayError as exc:
        print(f"{exc} -> {len(seen)} attempt(s)")
```

```text
pong after 3 attempts
retries recorded in the envelope: 2
claude -p exited 1: Invalid API key · Please run /login -> 1 attempt(s)
```

The CLI has no status codes, so its failures are classified from their own text — and the
deterministic markers are checked first and win. `"Not logged in. Please try again"` contains
"try again" and is still not retried, because retrying a login failure three times fixes
nothing.

One sharp edge on this backend: a timeout counts as transient, and `timeout_seconds` defaults
to 600. Worst case latency is `max_attempts * timeout_seconds` — half an hour on the defaults.
Lower one of the two for a latency-sensitive node.

---

## How do I test all of this without spending anything?

`ScriptedChatModel` replays a fixed list of responses and records what it was asked. This is
how the entire GraphARC test suite runs — the gate tests verify orchestration mechanics
(routing, budgets, traces, replay), and scripted responses exercise those deterministically.

<!-- verified -->
```python
from grapharc.testing import ScriptedChatModel

model = ScriptedChatModel(responses=["first answer", "second answer"])
print(model.invoke("q1").content)
reply = model.invoke("q2")
print(reply.content, reply.usage_metadata)
print("calls:", model.call_count, "| first prompt:", model.calls[0][0].content)

try:
    model.invoke("q3")
except RuntimeError as exc:
    print("exhausted:", exc)

repeating = ScriptedChatModel(responses=["same"], on_exhausted="repeat")
print([repeating.invoke(str(n)).content for n in range(3)])
```

```text
first answer
second answer {'input_tokens': 1, 'output_tokens': 3, 'total_tokens': 4}
calls: 2 | first prompt: q1
exhausted: ScriptedChatModel exhausted after 2 responses
['same', 'same', 'same']
```

Running off the end of the script raises by default. That is deliberate: a graph that made
more model calls than you scripted has changed behaviour, and silently repeating the last
answer would hide it. `on_exhausted="repeat"` is there for loops whose iteration count is not
the thing under test.

Token counts are estimated (`len(text) // 4`), not tokenised — good enough to exercise
metering paths, useless as a token count.

Dropped into a graph, it is the same object the real backends are:

<!-- verified -->
```python
from grapharc import Budget, BudgetExceeded, GraphARC, GraphARCState
from grapharc.gateway import get_model
from grapharc.runtime.graph import END, START


class State(GraphARCState):
    question: str
    answer: str = ""


def build(model):
    def ask(state: State) -> dict:
        return {"answer": model.invoke(state.question).content}

    g = GraphARC(State, name="ask", dag=True, budget=Budget(max_tokens=50))
    g.add_node("ask", ask, writes={"answer"})
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile()


# Swapping the backend is a one-line change: get_model("claude-cli/claude-sonnet-5")
# or get_model("openrouter/anthropic/claude-haiku-4.5") builds the same graph.
scripted = get_model("mock/x", responses=["42"])
print(build(scripted).invoke({"question": "meaning of life?"}))

# The runtime meters the model call itself — the node never touches the budget.
greedy = get_model("mock/x", responses=["x" * 4000])
try:
    build(greedy).invoke({"question": "meaning of life?"})
except BudgetExceeded as exc:
    print(f"BudgetExceeded: {exc}")
```

```text
{'question': 'meaning of life?', 'answer': '42'}
Error in MeterCallbackHandler.on_llm_end callback: BudgetExceeded('max_tokens reached (1004/50)')
BudgetExceeded: max_tokens reached (1004/50)
```

That middle line is on stderr, not stdout, and it is expected. LangChain logs callback
exceptions on the way out, and GraphARC's token meter *is* a callback — it sets
`raise_error = True` so a budget ceiling cannot be swallowed by the callback machinery. The
cost of keeping the ceiling load-bearing is one log line per stopped run.

The node never charged anything by hand. A LangChain callback is installed for the duration of
every node, so any chat model invoked on that thread reports usage to the run's meter,
including calls buried in library code the node merely called. Note `1004/50`: the ceiling is
enforced *inside* `on_llm_end`, at the call that crossed the line, rather than at the node
boundary after everything else has already been paid for.

Two things it does not see: a model invoked on a thread the node started itself
(`threading.Thread` does not inherit context variables — run the target through
`contextvars.copy_context().run(...)` if you need this), and spend a provider never reports.

Finally, the same graphs run against real models from the CLI when you want to check
behaviour rather than mechanics — that costs money and quota, so it is opt-in:

```bash
grapharc demo stage1 --model openrouter/anthropic/claude-haiku-4.5
grapharc demo stage5 --model claude-cli/claude-sonnet-5 \
                    --reviewer-model openrouter/openai/gpt-4o-mini
```

The test suite's own live tests are marked `live` and deselected by default; `pytest -m live`
is the only way to run them.
