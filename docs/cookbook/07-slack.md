# Running the CLI from Slack

`grapharc` on a laptop, driven from the Slack app on a phone. The bot in
`grapharc.slack` holds one *outbound* Socket Mode connection to Slack, so it
needs no public URL, no open port and no reverse proxy — home Wi‑Fi behind NAT
is enough. A workspace member types `/grapharc metrics t.jsonl r1` (or
mentions the bot in a channel); the bot runs the command on the host and posts
the output back in the thread.

Setup lives here; what a real session looks like, exchange by exchange, is
[08-slack-walkthrough.md](08-slack-walkthrough.md).

Nothing in this page is byte-compared by the test suite — Slack is on the
other end of every interesting command. What *is* tested, in
`tests/test_slack_gateway.py`, is everything short of Slack itself: the gate
that decides what text may become an argv, the runner, and the formatter.

## What the bot will and will not run

Anyone in the workspace can talk to the bot, so admission is the design, not
an afterthought. The defaults:

| Reachable from Slack | Refused from Slack |
|---|---|
| `demo`, `run`, `plan`, `models`, `replay`, `diff`, `trace`, `metrics`, `viz` | `serve` |
| Paths that resolve inside the bot's working directory | Any path that escapes it (`trace ../../.env` is refused before a process spawns) |
| The budget, policy and trace flags each command already has | `--registry` (imports an arbitrary module), `--config`, `--json`, `--no-color` |
| `plan --registry`, for exactly the two registries the package ships | any other `--registry` value |
| `agent`, only behind the double opt-in below | `--model` / `--reviewer-model`, unless the operator opts in |
| Each admitted flag, once; `agent --allow`/`--deny` accumulate as the CLI does | The same flag twice (`--registry <demo> --registry <stdlib>`), because the gate would judge one occurrence and the CLI would run the other |

With `--model` off, every reachable command runs the scripted, spend-free
path. The default answer to "can someone in Slack cost me money?" is **no**;
`GRAPHARC_SLACK_ALLOW_MODEL=1` changes that answer deliberately, in the shell
that starts the bot, not from Slack.

Output is the CLI's piped-mode bytes in a code fence. stdout in the bot is a
pipe, so by the CLI's own contract there is no colour to strip and the bytes
match what `grapharc … | cat` prints on the host.

A successful `viz` reply carries one extra line: a *render this diagram* link.
The whole diagram is zlib-compressed into the URL fragment, which a browser
never sends to any server — mermaid.live's JavaScript renders it locally, so
following the link ships the diagram to no one.

## Slack app setup (once, ~5 minutes)

1. <https://api.slack.com/apps> → **Create New App** → *From a manifest*, pick
   the workspace, and paste:

   ```yaml
   display_information:
     name: grapharc
   features:
     bot_user:
       display_name: grapharc
     slash_commands:
       - command: /grapharc
         description: run a grapharc command on the host
         usage_hint: "metrics t.jsonl r1"
   oauth_config:
     scopes:
       bot:
         - commands
         - app_mentions:read
         - chat:write
   settings:
     event_subscriptions:
       bot_events:
         - app_mention
     socket_mode_enabled: true
     interactivity:
       is_enabled: true
   ```

2. **Basic Information → App-Level Tokens** → generate one with the
   `connections:write` scope. That is `SLACK_APP_TOKEN` (`xapp-…`).
3. **Install App** to the workspace. The bot token on the OAuth page is
   `SLACK_BOT_TOKEN` (`xoxb-…`).
4. Invite the bot to a channel: `/invite @grapharc`.

## Running it

```bash
uv sync --extra slack        # or: pip install 'grapharc[slack]'

export SLACK_BOT_TOKEN=xoxb-…
export SLACK_APP_TOKEN=xapp-…
mkdir -p ~/grapharc-slack && cd ~/grapharc-slack   # the bot's whole world
python -m grapharc.slack
```

The startup line states the resolved working directory, the timeout and
whether model flags are on — the three decisions that matter — then blocks
until interrupted. From Slack:

```
/grapharc plan "investigate the checkout outage" --scripted
/grapharc trace t.jsonl
@grapharc metrics t.jsonl <run-id>
```

Configuration is environment-only, read once at startup:

| Variable | Default | Meaning |
|---|---|---|
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | — (required) | the two tokens from the app page |
| `GRAPHARC_SLACK_WORKDIR` | the bot's cwd | the directory every path must resolve inside |
| `GRAPHARC_SLACK_TIMEOUT` | `120` | seconds one command may run before it is killed |
| `GRAPHARC_SLACK_ALLOW_MODEL` | off | `1` admits `--model`/`--reviewer-model` |
| `GRAPHARC_SLACK_ALLOW_AGENT` | off | `1` admits `agent` — only together with `ALLOW_MODEL` |
| `GRAPHARC_SLACK_COMMAND` | `/grapharc` | the slash command to answer to |
| `GRAPHARC_SLACK_LIVE` | on | `0` turns off the live-edited status message |
| `GRAPHARC_SLACK_LIVE_INTERVAL` | `2.5` | seconds between two edits of the status message |
| `GRAPHARC_SLACK_LIVE_URL` | unset | base URL of a `grapharc serve --live-root` the requester can reach; posts a "watch live" link |

The bot reads tokens from the process environment only. The model gateway's
`.env` loader is deliberately not used here — even though it now reads the
working directory alone rather than searching upward: a bot that a whole
workspace can drive must not discover credentials in a file the operator did
not point it at, and its working directory is somewhere other things write.

## Live progress

A command that traces (`demo`, `run`, `plan`, `agent`) is narrated while it
runs. The gate gives every such command a trace path the bot knows — a unique
`slack-runs/<stamp>/trace.jsonl` under the working directory, unless the
request named its own `--trace` — and the bot tails that file from a side
thread while the subprocess runs. What you see in Slack is one status message,
edited in place every couple of seconds:

```
`grapharc run pipeline.toml --trace slack-runs/…/trace.jsonl` — running (14s)
✓ ingest  312ms
✓ extract  1.8s  1543 tok
✗ verify  err: citation not found
▸ report  running…
6 events · 2/4 nodes done · 1543 tok
<open live view>
```

The `open live view` link points at your own live server's run page
(`GRAPHARC_SLACK_LIVE_URL` + the trace path), where the graph redraws in real
time. With no live URL configured, the message carries a `current diagram`
link instead — the same mermaid.live fragment URL `viz` gets, the diagram
compressed into the URL itself and shipped to no one, refreshed on every edit
so mid-run it renders the path *so far*. When the command finishes, the status
message is edited one last time into the same final result the bot has always
posted, keeping the run-page link (or, without one, a final-diagram link).

Everything about this path is best-effort by construction. If the bot cannot
post the status message (it is not in the channel, the API errored), the whole
live layer steps aside and you get today's single blocking reply; if a mid-run
edit fails, the narration goes quiet; and if the *final* edit fails, the result
is posted as an ordinary reply instead. A broken live view can cost you the
narration, never the answer. One visibility note: for a slash command the
status message is posted to the channel (a `respond()`-style reply would allow
only five updates), so a live run is visible to everyone in it — the mention
path threads it under your message as before.

Because the trace now lands inside the working directory, the run is also
inspectable afterwards from Slack itself: `/grapharc metrics
slack-runs/<stamp>/trace.jsonl <run-id>`, `viz` for the finished diagram,
`replay` for the reconstruction. The `slack-runs/` directories are the audit
trail and are never cleaned up automatically; prune them like any other logs.

## Watching it live in a browser

The status message is text. For the actual diagram redrawing itself as nodes
run, pair the bot with the live view server on the same machine:

```bash
grapharc serve --live-root "$GRAPHARC_SLACK_WORKDIR" --port 8300
export GRAPHARC_SLACK_LIVE_URL=https://laptop.tailnet.ts.net:8300
python -m grapharc.slack
```

With the URL configured, the bot's first status message includes
`watch live: <url>/live/view?trace=slack-runs/…` — a page that renders the
Mermaid diagram and the run's numbers and updates itself over SSE as the trace
file grows. `/live` lists every trace under the root. The server is read-only,
confines every requested path inside the root, and never serves `state_delta`
contents — what the page shows is what `viz` and `metrics` already show.

Reachability is deliberately your problem, not the bot's: the bot never opens
a port (that is the whole point of Socket Mode), and `serve` still binds
loopback by default. Put a tailnet or tunnel (Tailscale, cloudflared) in front
for the person on the phone, and add `--live-token` if the URL is guessable —
the person then signs in once on the page rather than carrying the token in
the link, which is what keeps it out of access logs and browser history.
Details in [06-serving-and-ops.md](06-serving-and-ops.md).

## A `plan` that reads

The default planning registry is the incident-response demo: its node bodies
are stubs, so a goal like "summarise the docs here" gets an honest negative
(or, if phrased vaguely enough, a hollow success). The shipped alternative
has bodies that really read — `survey` / `read` / `summarise`, read-only,
confined to the bot's working directory — plus one kind, `propose`, whose
body hands what was read to the model and records the recommendation that
comes back, labelled `proposal (model-authored):`:

```
@grapharc plan "summarise the docs and propose how to merge them" --registry grapharc.examples.plan_docs:build_registry --model claude-cli/claude-sonnet-5 --trace docs.jsonl --run-id docs-1
```

The scripted planner (no `--model`) runs the same chain free: the reading
notes are identical — file names, titles, excerpts are operator code, not
model output — and the `propose` note says plainly that authoring needs a
real model rather than pretending. `--registry` from Slack accepts exactly
these two shipped modules and nothing else — the flag's general form imports
arbitrary code, which stays refused.

## The `agent` opt-in

`agent` is the command with file tools, which is exactly why it is off by
default: it acts on the host on behalf of anyone in the workspace. Turning it
on takes **two** switches — `GRAPHARC_SLACK_ALLOW_AGENT=1` because it acts,
and `GRAPHARC_SLACK_ALLOW_MODEL=1` because it cannot run spend-free. The
startup line reports `agent on` only when both hold. Then:

```
@grapharc agent "read every markdown file and list the broken links" --max-turns 6
```

What the gate does to every Slack-launched agent, non-negotiably:

- the executor stays `sandbox` — `--executor` is not admitted, so `local`
  (no confinement) is unreachable;
- `--system-prompt` is not admitted;
- the workspace defaults to `<workdir>/agent` rather than the CLI's fresh
  temp dir, so the run's `trace.jsonl` and outputs stay where `trace` /
  `metrics` / `viz` can read them back; `--workspace` may pick another
  directory, confined to the workdir like every other path;
- unless `--max-seconds` is given, it defaults to ten seconds under the
  bot's timeout, so the run ends with the CLI's graceful interrupt-and-report
  rather than the bot's kill.

`--allow` / `--deny` tool globs pass through and are repeatable; deny beats
allow, as in the CLI.

### The delegated executor

`--executor claude-cli` hands the whole task to Claude Code's own headless
agent on the operator's subscription — no API key, no `bind_tools`, because
the loop and the tools are Claude Code's, not grapharc's:

```
@grapharc agent "summarise the markdown here" --executor claude-cli --workspace .
```

Honest trade, stated plainly: governance is coarser (Claude Code's permission
model, not grapharc's per-call gate), and the token figure is what the
sub-agent reports rather than what a meter charged inline. The frame stays
grapharc's — workspace confined, wall clock enforced from outside, the run
recorded to the trace. From Slack, tool names are Claude Code's (`Read`,
`Grep`, `Edit`, `Bash`, …), and a delegated run with no explicit
`--allow`/`--deny` gets `--deny Bash` injected: an unsandboxed shell on the
host is not something a bare Slack message should carry. `--executor local`
stays unreachable from Slack.

## The honest caveats

- **The bot is alive while the process is.** Laptop lid closed means commands
  from a phone go unanswered — Slack shows the slash command timing out, and
  nothing queues. The same script runs unchanged on any always-on box.
- **Slack's three-second ack.** The bot acks immediately ("running …"), then
  narrates a tracing command through the live status message and lands the
  result there when it finishes; the timeout bounds how long that can be.
- **The workspace is the trust boundary.** The gate stops path escapes,
  module imports and spend, but anyone in the workspace can run every allowed
  command against every file in the working directory. Give the bot a
  directory that contains nothing you would not show the whole channel.
