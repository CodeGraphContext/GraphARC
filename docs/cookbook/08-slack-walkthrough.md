# A Slack session, command by command

[07-slack.md](07-slack.md) covers setting the bot up; this page is what using
it actually looks like. Every exchange below is from a real session — the
first hour of driving `grapharc` from Slack — including the refusals and the
mistakes, because those teach the model of the tool faster than the successes.

One rule carries the whole page: **one message is one command.** The bot
treats the entire message as a single command line; pasting three commands in
one message glues them into one argv and the CLI rejects it (exit 2). Send
commands one at a time, and wait for a long one to answer before reading its
trace.

## First contact

```
/grapharc models
```

Instant, offline, spend-free — the right first ping. You get the backend
table and which specs need a key. `/grapharc` with nothing after it returns
the allowed-command list.

If the slash command says *not a valid command*, the app needs a reinstall
after the command was registered; mentions (`@grapharc models`) work as soon
as the bot is invited to the channel.

## The run → read loop

The shape of everything else: run something with a **named trace and run id**,
then read that trace back.

```
@grapharc demo stage2 --trace demo.jsonl --run-id first-demo
@grapharc trace demo.jsonl
@grapharc metrics demo.jsonl first-demo
@grapharc viz demo.jsonl first-demo
```

Name both, always. An unnamed run mints a random id you then have to fish out
of the trace, and two runs pointed at the same `--trace` file interleave their
events in one JSONL — legal, but confusing to read. Fresh file, explicit id,
every run.

`trace` is the audit log — every event, verbatim. `metrics` is the summary:
tokens, durations, per-node counts, termination reason. `viz` prints the
executed path as Mermaid, and the reply ends with a *render this diagram*
link that draws it in your browser (the diagram travels compressed inside the
URL fragment, which a browser sends to no server).

## Planning with a real model

The bot's default is spend-free: `--model` is refused until the operator
restarts the bot with `GRAPHARC_SLACK_ALLOW_MODEL=1` — a decision made in the
shell that starts the bot, deliberately not from Slack. Once it's on:

```
@grapharc plan "investigate improvements for the readme file" --model claude-cli/claude-sonnet-5 --trace plan.jsonl --run-id readme-1
```

A scripted `plan` answers in under a second; a real one takes tens of
seconds, and the mention reply arrives only when it finishes. Reading the
trace afterwards is where the governance becomes visible: rounds where the
model's reply had no parseable proposal, a structurally valid proposal
rejected by the edge policy (`policy/edge_denied`), and finally an admitted
round that executed. Every model failure contained, every decision recorded.

## What a refusal looks like, and why

Two refusals from the same session, both correct:

- `@grapharc agent "do something"` → *not a command this bot runs*. `agent`
  executes tools on the host on behalf of anyone in the workspace, so it is
  excluded at the gate, not hidden.
- `plan "make a new file for the docs" --model …` → **ran; the answer was
  negative**. The planner may only propose node kinds from its registry —
  the incident-response demo set — and none of them can create a file. The
  model found no admissible work and said so, which is exit 1 doing its job:
  a weaker loop would have executed three irrelevant nodes and called it
  success. File-creation belongs to `grapharc agent`, on the host.

## The paths rule

Every path in a command must resolve inside the bot's working directory —
`trace ../somewhere-else.jsonl` is refused before a process spawns. The
corollary: tools that default their output *outside* the workdir (a bare
`plan` writes its trace to a temp directory) leave you with a file Slack
cannot read back. Passing `--trace <name>.jsonl` keeps the whole
run-then-read loop inside the bot's world, which is the point of it.
