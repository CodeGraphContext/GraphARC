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
| `agent`, only behind the double opt-in below | `--model` / `--reviewer-model`, unless the operator opts in |

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
/grapharc plan "investigate the checkout outage"
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

The bot reads tokens from the process environment only. The `.env`
upward-directory search that the model gateway performs is deliberately not
used here: a bot that a whole workspace can drive must not discover
credentials in a file the operator did not point it at.

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

## The honest caveats

- **The bot is alive while the process is.** Laptop lid closed means commands
  from a phone go unanswered — Slack shows the slash command timing out, and
  nothing queues. The same script runs unchanged on any always-on box.
- **Slack's three-second ack.** The bot acks immediately ("running …") and
  posts the result when the command finishes; the timeout bounds how long
  that can be.
- **The workspace is the trust boundary.** The gate stops path escapes,
  module imports and spend, but anyone in the workspace can run every allowed
  command against every file in the working directory. Give the bot a
  directory that contains nothing you would not show the whole channel.
