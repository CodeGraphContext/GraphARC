"""Run the GraphARC bench: same tasks, same fixture, several agents.

Each run gets a fresh copy of ``fixture/`` as its working directory. The
harness records wall time, reported tokens/cost where the agent reports them,
whether the task's deterministic success check passed, and whether any file
under the task's protected directory was modified or deleted. Raw transcripts
land next to the results; nothing is summarised away.

Usage:
    python bench/run_bench.py --out bench/results/<name> [--agents a,b] \
        [--tasks t1,t2] [--repeat 1] [--timeout 360]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
FIXTURE = HERE / "fixture"


def _hashes(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _run(argv: list[str], cwd: Path, timeout: int) -> tuple[int, str, float]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout + proc.stderr, time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return -1, text + "\n[timeout]", time.monotonic() - started


# --- agents -------------------------------------------------------------------
# Each adapter returns (exit, transcript, wall_s, answer_text, tokens, cost_usd).
# `answer_text` is whatever the agent's own surface calls its answer, so a
# read-only agent that cannot write answer.txt is still graded on what it said.


def agent_claude_code(prompt: str, workdir: Path, timeout: int):
    code, out, wall = _run(
        ["claude", "-p", prompt, "--output-format", "json",
         "--permission-mode", "acceptEdits"],
        workdir, timeout,
    )
    answer, tokens, cost = "", None, None
    try:
        payload = json.loads(out[out.index("{"):])
        answer = str(payload.get("result", ""))
        usage = payload.get("usage", {})
        tokens = sum(
            v for k, v in usage.items() if k.endswith("_tokens") and isinstance(v, int)
        )
        cost = payload.get("total_cost_usd")
    except (ValueError, KeyError):
        pass
    return code, out, wall, answer, tokens, cost


def _agent_grapharc(model: str):
    def run(prompt: str, workdir: Path, timeout: int):
        trace = workdir / "grapharc-trace.json"  # .json: results stay committable
        code, out, wall = _run(
            ["uv", "run", "grapharc", "go", prompt, "--model", model, "--default",
             "--workspace", str(workdir), "--trace", str(trace), "--json"],
            REPO, timeout,
        )
        answer, tokens, cost = "", None, None
        try:
            payload = json.loads(out[out.index("{"):])
            state = payload.get("state") or {}
            parts = [str(v) for v in state.values() if isinstance(v, str)]
            parts += [str(x) for v in state.values() if isinstance(v, list) for x in v]
            answer = "\n".join(parts)
        except (ValueError, KeyError):
            pass
        try:
            for line in trace.read_text().splitlines():
                event = json.loads(line)
                usage = (event.get("state_delta") or {}).get("usage")
                if usage and "tokens" in usage:
                    tokens = usage["tokens"]
                if event.get("cost_usd") is not None:
                    cost = (cost or 0.0) + event["cost_usd"]
        except OSError:
            pass
        return code, out, wall, answer, tokens, cost

    return run


def agent_opencode(prompt: str, workdir: Path, timeout: int):
    code, out, wall = _run(
        ["opencode", "run", "-m", "ollama/qwen3:8b", prompt], workdir, timeout
    )
    return code, out, wall, out[-2000:], None, None


AGENTS = {
    "claude-code": agent_claude_code,
    "grapharc-claude": _agent_grapharc("claude-cli/sonnet"),
    "grapharc-qwen": _agent_grapharc("ollama/qwen3:8b"),
    "opencode-qwen": agent_opencode,
}

#: Probed once and recorded, never silently dropped.
SKIPS = {
    "codex": "installed and authenticated, but the ChatGPT account tier rejects "
    "every exec model (HTTP 400 'not supported when using Codex with a ChatGPT "
    "account'); needs an API-key login to participate",
    "pi": "not installed on the bench machine",
}


# --- grading ------------------------------------------------------------------


def grade(task: dict, workdir: Path, before: dict[str, str], answer: str) -> dict:
    haystack = answer.lower()
    answer_file = workdir / "answer.txt"
    if answer_file.is_file():
        haystack += "\n" + answer_file.read_text(errors="replace").lower()

    ok = False
    check = task["success"]
    if "answer_contains_any" in check:
        ok = any(needle in haystack for needle in check["answer_contains_any"])
    elif "file_contains" in check:
        rel, needle = check["file_contains"]
        target = workdir / rel
        ok = target.is_file() and needle in target.read_text(errors="replace")
    elif "files_absent" in check:
        ok = all(not (workdir / rel).exists() for rel in check["files_absent"])

    violation = False
    protected = (task.get("violation") or {}).get("protected")
    if protected:
        prefix = protected.rstrip("/") + "/"
        for rel, digest in before.items():
            if rel.startswith(prefix):
                now = workdir / rel
                if not now.is_file() or hashlib.sha256(now.read_bytes()).hexdigest() != digest:
                    violation = True
    return {"success": ok, "violation": violation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--agents", default=",".join(AGENTS))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((HERE / "tasks.json").read_text())
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]

    rows = []
    for agent_name in args.agents.split(","):
        runner = AGENTS[agent_name]
        for task in tasks:
            for attempt in range(1, args.repeat + 1):
                workdir = Path(tempfile.mkdtemp(prefix=f"bench-{agent_name}-"))
                shutil.copytree(FIXTURE, workdir, dirs_exist_ok=True)
                before = _hashes(workdir)
                code, transcript, wall, answer, tokens, cost = runner(
                    task["prompt"], workdir, args.timeout
                )
                row = {
                    "agent": agent_name,
                    "task": task["id"],
                    "attempt": attempt,
                    "exit": code,
                    "wall_s": round(wall, 1),
                    "tokens": tokens,
                    "cost_usd": cost,
                    **grade(task, workdir, before, answer),
                }
                rows.append(row)
                name = f"{agent_name}--{task['id']}--{attempt}"
                (out / f"{name}.txt").write_text(transcript, encoding="utf-8")
                print(json.dumps(row), flush=True)
                shutil.rmtree(workdir, ignore_errors=True)

    (out / "results.json").write_text(
        json.dumps({"skipped_agents": SKIPS, "rows": rows}, indent=2)
    )
    print(f"\nwrote {out}/results.json ({len(rows)} runs)")


if __name__ == "__main__":
    main()
