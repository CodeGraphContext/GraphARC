"""Drive the real Slack path end to end and record every message it produced.

This is the capture half of the demo video. It runs `grapharc.slack.bot`'s own
`handle_text_live` — the same function the bolt listeners call — against a sink
that keeps every edit instead of posting it, clicks the Approve button through
`handle_approval_action` the way a real click would, and writes the result to
a JSON file for `render_demo.py` to turn into frames.

What is real here: the admission gate, the planner (a real model with
`--model`), the proposed graph, the approval file handshake, the fingerprint
check, the execution, and the trace. Every line of message text in the output
is the text the bot would have sent to Slack, byte for byte.

What is not real: the Slack transport. There is no workspace, no token and no
socket — the sink stands in for `chat.postMessage`/`chat.update`, exactly as
the test suite's sink does. That substitution is the point of the module
layout: everything with behaviour is testable, and demonstrable, without Slack.

    python docs/demo/capture_supervised_slack.py --workdir /tmp/ws --scripted
    python docs/demo/capture_supervised_slack.py --workdir /tmp/ws \
        --model claude-cli --registry grapharc.stdlib:build_registry \
        --goal "summarise what flaky.py does"
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from grapharc.slack.bot import handle_approval_action, handle_text_live
from grapharc.slack.config import SlackBotConfig


class RecordingSink:
    """A `LiveSink` that timestamps every message instead of sending it."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def _record(self, kind: str, text: str, blocks=None) -> None:
        elements = [
            element
            for block in (blocks or [])
            if block.get("type") == "actions"
            for element in block["elements"]
        ]
        with self._lock:
            self.frames.append(
                {
                    "at": round(time.monotonic() - self.started, 2),
                    "kind": kind,
                    "text": text,
                    # Labels for the renderer, values for the click. The value
                    # is what Slack would hand back on a real press, so the
                    # click below is the same call the bolt listener makes.
                    "buttons": [element["text"]["text"] for element in elements],
                    "values": [element["value"] for element in elements],
                }
            )

    def post(self, text: str):
        self._record("post", text)
        return ("C_DEMO", "1.0")

    def update(self, handle, text: str, blocks=None) -> bool:
        self._record("update", text, blocks)
        return True

    def wait_for_buttons(self, timeout: float):
        """Block until an edit carries buttons; return its value, or None."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for frame in self.frames:
                    if frame["buttons"]:
                        return frame
            time.sleep(0.1)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/demo/session.json"))
    parser.add_argument("--goal", default="investigate the checkout outage")
    parser.add_argument("--model", default=None, help="omit to use --scripted")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--deny", action="store_true", help="click Deny instead")
    parser.add_argument("--approve-after", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    request = f'plan "{args.goal}" --go'
    if args.scripted:
        request += " --scripted"
    if args.model:
        request += f" --model {args.model}"
    if args.registry:
        request += f" --registry {args.registry}"

    config = SlackBotConfig(
        bot_token="xoxb-demo",
        app_token="xapp-demo",
        workdir=workdir,
        timeout_seconds=120.0,
        work_timeout_seconds=args.timeout,
        live_interval_seconds=1.0,
        allow_model=True,
        allow_agent=True,
    )

    sink = RecordingSink()
    decision: dict = {}

    def click() -> None:
        frame = sink.wait_for_buttons(timeout=args.timeout)
        if frame is None:
            decision["error"] = "the run never offered a button"
            return
        # A human reads the plan before answering; the pause is the demo's way
        # of showing that the run is genuinely waiting rather than racing.
        time.sleep(args.approve_after)
        value = frame["values"][1 if args.deny else 0]
        decision["at"] = round(time.monotonic() - sink.started, 2)
        decision["answer"] = handle_approval_action(
            value, config, deny=args.deny, actor="U_DEMO"
        )

    clicker = threading.Thread(target=click, daemon=True)
    clicker.start()
    reply = handle_text_live(request, config, sink)
    clicker.join(timeout=30.0)

    session = {
        "request": request,
        "workdir": str(workdir),
        "frames": sink.frames,
        "decision": decision,
        "trailing_reply": reply,
        "audit": _audit(workdir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(session, indent=2), encoding="utf-8")
    print(f"{len(sink.frames)} frames -> {args.out}")
    print(f"decision: {decision.get('answer') or decision.get('error')}")
    return 0


def _audit(workdir: Path) -> dict:
    """What the trace says happened, read back the way any auditor would.

    This is the claim the whole design makes, checked against the file rather
    than against the messages: the plan was asked about, a human answered, and
    only then did a node start. The order of those three phases is the proof;
    a run that executed before its `approval_response` would show it here.
    """
    from grapharc.observe.replay import replay
    from grapharc.observe.trace import TailRecorder

    traces = sorted(workdir.glob("slack-runs/*/trace.jsonl"))
    if not traces:
        return {}
    recorder = TailRecorder(traces[-1])
    run_id = recorder.run_ids()[-1]
    events = list(recorder.read_events(run_id))
    run = replay(recorder, run_id)
    order = [
        event.phase
        for event in events
        if event.phase in ("approval_request", "approval_response", "start")
    ]
    return {
        "trace": str(traces[-1].relative_to(workdir)),
        "run_id": run_id,
        "events": len(events),
        "phase_order": list(dict.fromkeys(order)),
        "executed": [execution.node for execution in run.executions if execution.completed],
        "tokens": run.tokens,
        "cost_usd": run.recorded_cost_usd,
        "approved_before_first_node": (
            "approval_response" in order
            and "start" in order
            and order.index("approval_response") < order.index("start")
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
