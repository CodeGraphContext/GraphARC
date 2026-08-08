# The demos, and how to re-make them

Four recordings, two pipelines. Everything with behaviour runs for real; what
the renderers add is a typeface, a background and a readable pace. Each section
below says exactly which is which, because a demo you cannot check is a claim,
not evidence.

| recording | what it shows | needs a model? |
|---|---|---|
| `grapharc-cli-gate` | a proposal refused, replanned, admitted, executed | no — `--scripted` |
| `grapharc-cli-fix-bug` | GraphARC fixing a real bug in GraphARC, under its own policy | yes — Claude subscription |
| `grapharc-cli-audit` | one trace file answering every question about a run | no — `--scripted` |
| `grapharc-slack-supervised` | one Slack message, the graph, the buttons, the run | yes — Claude subscription |

## The CLI recordings

`capture_cli.py` runs a scenario's commands in a **pseudo-terminal**, so the
CLI takes its tty branch and emits the colour a person actually sees. Piping
would give the byte-stable colourless form instead — also real, also tested,
but not what a demo is about. What lands in the recording is each command's
bytes, its exit code and its wall-clock duration.

```bash
python docs/demo/capture_cli.py docs/demo/scenarios/gate.py  --out /tmp/gate.json
python docs/demo/render_cli.py  /tmp/gate.json --out docs/media/grapharc-cli-gate.mp4 \
                                               --gif docs/media/grapharc-cli-gate.gif
```

`render_cli.py` parses SGR colour — including the `38;5;N` form the CLI emits
under a 256-colour terminal — and draws a terminal grid. Any other escape is
dropped rather than half-interpreted; the CLI emits none, so guessing at one
would be inventing behaviour to render.

**Real:** the commands, in that order, with those exit codes and those
durations. Every character on screen is a byte the CLI wrote.
**Invented:** the pacing. A command is typed out, output is revealed a couple
of lines at a time, and each step holds long enough to read. The real duration
is not thrown away — it is on the frame, bottom right, so a 43-second agent
phase says so while still being watchable.

### Demo 2 is the one worth checking closely

`scenarios/fix_bug.py` fixes a bug that was really on this project's backlog:
`mcp.driver.graph_status` read a running trace with the strict reader, so the
supervised agent's own status tool raised `TraceReadError` on a half-written
line — at exactly the moment it is meant to be useful.
`tests/test_torn_trace_read.py` is that bug report in checkable form.

The recording runs against a **copy** of the repository, made with `git init`
and one baseline commit, so a demo cannot damage the tree it is a demo of:

```bash
SRC=$(pwd); DEMO=/tmp/grapharc-demo-repo
rm -rf "$DEMO" && mkdir -p "$DEMO"
tar --exclude=.git --exclude=.venv --exclude=__pycache__ -cf - -C "$SRC" . | tar xf - -C "$DEMO"
cd "$DEMO" && git init -q && git add -A && git commit -qm baseline && cd "$SRC"

GRAPHARC_DEMO_WORKDIR=$DEMO python docs/demo/capture_cli.py \
    docs/demo/scenarios/fix_bug.py --out /tmp/fix.json
python docs/demo/render_cli.py /tmp/fix.json \
    --out docs/media/grapharc-cli-fix-bug.mp4 --gif docs/media/grapharc-cli-fix-bug.gif
```

The two policy documents in that copy (`policy.toml`, `policy-approved.toml`)
are written by hand, and the demo turns on the difference between them. That is
deliberate rather than incidental: left to itself, `plan` **generates** a policy
from the goal, and a goal containing the word "fix" gets a generated policy that
permits `apply_change` — which is the designed behaviour ("investigate the
outage" and "fix the outage" deserve different answers) but makes for a demo
whose rule changes under you. An explicit `--policy` is also the production
path, so the recording shows the mechanism a real operator would use.

What the recording establishes, and none of it is staged:

- under the deny, the planner proposes three read-only nodes and its own
  rationale says why — *"Since apply_change cannot be reached by an edge, this
  round investigates … for a human to act on"* — and the plan is `mutating:
  false`;
- after the amendment, the same goal and the same model produce a five-node
  graph containing `apply_change`, marked `mutating: true`;
- the execution is Claude Code, delegated to because the Claude CLI has no
  tool-calling wire format, editing real files in the copy;
- the fix it produced was `TraceRecorder` → `TailRecorder`, which is the
  answer the rest of the codebase already uses for reading a file mid-write;
- the test that opened the recording red closes it green.

That fix was then read by a person and landed here deliberately, with the
regression test — the demo is where it was found working, not how it shipped.

## The Slack recording

`capture_supervised_slack.py` drives `grapharc.slack.bot`'s own
`handle_text_live` — the function the bolt listeners call — against a sink that
keeps every edit instead of posting it, and clicks Approve through
`handle_approval_action` exactly as a real click does.

```bash
# free, no model, no key
python docs/demo/capture_supervised_slack.py --workdir /tmp/ws --scripted --out /tmp/s.json
# the real thing, on a Claude subscription
python docs/demo/capture_supervised_slack.py --workdir /tmp/ws \
    --model claude-cli --registry grapharc.stdlib:build_registry \
    --goal "explain what flaky.py does" --out /tmp/s.json
python docs/demo/render_demo.py /tmp/s.json --out docs/media/grapharc-slack-supervised.mp4 \
                                            --gif docs/media/grapharc-slack-supervised.gif
```

**Real:** the admission gate, its rule that `--go` implies `--approve`, the
planner, the proposed graph, the file handshake, the fingerprint check on the
click, the execution, the trace. Every character of message text is what
`handle_text_live` produced, byte for byte.
**Not real:** the Slack transport. No workspace, no token, no socket — a
recording sink stands in for `chat.postMessage`/`chat.update`, exactly as the
test suite's sink does. The chrome (channel name, avatars, colours) is drawn by
`render_demo.py` and is not Slack's.
**Selected, not fabricated:** a code fence longer than 20 lines is cut, and the
frame says how many lines it hid.

The closing frame is computed from the trace file rather than from any message:

```
phase order    approval_request → approval_response → start
```

`start` is the first node beginning work, and it comes last. The same property
is asserted against a real CLI subprocess in
`tests/test_slack_supervision.py::test_a_slack_go_parks_shows_the_graph_is_approved_by_button_and_then_runs`,
so it does not depend on anyone re-recording a video.

## Requirements

The capture scripts need nothing but grapharc itself. The renderers need
Pillow, `ffmpeg` on PATH, and the DejaVu fonts (`fonts-dejavu-core` on
Debian/Ubuntu). None of it is a dependency of the package — these are
documentation tools.

Two glyphs the Slack path emits (`⏸`, `⬜`) have no outline in DejaVu and are
substituted (`‖`, `▫`) so they do not render as .notdef boxes; the table is at
the bottom of `render_demo.py`.
