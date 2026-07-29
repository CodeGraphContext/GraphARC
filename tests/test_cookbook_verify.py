"""Every snippet in the verification-and-memory cookbook page actually runs.

The page's only promise is that its code and its pasted output are real. That is
not a promise prose can keep, so this file keeps it mechanically: each fenced
``python`` block is executed in a fresh interpreter and its stdout compared,
character for character, against the ``text`` block printed underneath it.

Genuinely non-deterministic values — generated claim/artifact ids, timestamps,
pids — are normalized on both sides before comparison. Nothing else is: a
reordered result list or a changed error message fails the test, which is the
point.

Two snippets in the page cannot run here: one calls a real model, and one needs
the optional `ladybug` extra that CI does not install. Both carry `SKIP_MARKER`
and are never executed; the count is asserted so a broken snippet cannot be
quietly retired by adding the marker.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from grapharc.memory import Claim, MemoryStore, render_context
from grapharc.runtime.verify import evaluate_verdicts, verify_claim
from grapharc.testing import ScriptedChatModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "cookbook" / "04-verification-and-memory.md"

SKIP_MARKER = "Not executed by the docs test"
EXPECTED_SKIPPED = 2

_FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)

# Values that legitimately differ between runs. Everything else must match.
_VOLATILE = (
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00"), "<timestamp>"),
    (re.compile(r"pid=\d+"), "pid=<pid>"),
    # Generated ids are 12 hex chars; \b keeps this off the 16- and 64-char
    # sha256 digests in the same output, which are content-addressed and stable.
    (re.compile(r"\b[0-9a-f]{12}\b"), "<id>"),
)


def _normalize(text: str) -> str:
    for pattern, replacement in _VOLATILE:
        text = pattern.sub(replacement, text)
    return text.strip("\n")


def _fences() -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _FENCE.finditer(DOC.read_text(encoding="utf-8"))]


def _snippets() -> list[tuple[str, str]]:
    """(code, expected_output) for every runnable python block, in page order."""
    fences = _fences()
    out: list[tuple[str, str]] = []
    for i, (lang, body) in enumerate(fences):
        if lang != "python" or SKIP_MARKER in body:
            continue
        assert i + 1 < len(fences), f"snippet {i} has no output block:\n{body}"
        next_lang, expected = fences[i + 1]
        assert next_lang == "text", (
            f"snippet {i} is followed by a ```{next_lang} block, not its output:\n{body}"
        )
        out.append((body, expected))
    return out


SNIPPETS = _snippets()


def test_the_page_exists_and_parses():
    assert DOC.is_file(), f"{DOC} is missing"
    assert SNIPPETS, "no runnable snippets found — the fence parser is broken"


def test_only_the_real_model_snippet_is_exempt():
    skipped = [b for lang, b in _fences() if lang == "python" and SKIP_MARKER in b]
    assert len(skipped) == EXPECTED_SKIPPED
    # It is exempt because it needs a key, and it says so where the reader sees it.
    assert "Requires a real model" in skipped[0]


@pytest.mark.parametrize(
    ("code", "expected"),
    SNIPPETS,
    ids=[f"snippet{i:02d}" for i in range(len(SNIPPETS))],
)
def test_snippet_runs_and_prints_what_the_page_claims(code, expected, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,  # nothing may depend on being run from the repo root
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        timeout=120,
    )
    assert proc.returncode == 0, f"snippet failed:\n{code}\n--- stderr ---\n{proc.stderr}"
    assert _normalize(proc.stdout) == _normalize(expected), (
        f"output drifted:\n--- code ---\n{code}\n"
        f"--- documented ---\n{expected}\n--- actual ---\n{proc.stdout}"
    )


def test_no_snippet_reaches_a_live_backend():
    """The page must be runnable with no key and no subscription.

    Backend *names* are fine — one snippet passes specs to `different_providers`,
    which is pure string comparison. Building a model is not.
    """
    for code, _ in SNIPPETS:
        assert "get_model(" not in code
        assert "ChatOpenAI" not in code
        assert "ClaudeCodeCLIChatModel" not in code


# --- the claims the page is built on, asserted directly ---------------------


def test_fabricated_citation_costs_zero_model_calls():
    reviewer = ScriptedChatModel(responses=['{"supported": true, "reason": "sure"}'])
    verdict = verify_claim(
        reviewer,
        text="GraphARC was benchmarked at 10x faster",
        citation="benchmarked at 10x faster",
        source_text="GraphARC wraps LangGraph with typed state contracts.",
    )
    assert verdict.accepted is False
    assert verdict.anchor_ok is False
    assert verdict.model_accepted is None  # None means never consulted
    assert reviewer.call_count == 0


def test_a_verifier_that_rejects_everything_is_visible_in_the_stats():
    source = "GraphARC wraps LangGraph with typed state contracts."
    paranoid = ScriptedChatModel(
        responses=['{"supported": false, "reason": "no"}'], on_exhausted="repeat"
    )
    verdict = verify_claim(
        paranoid,
        text="GraphARC enforces typed state contracts",
        citation="typed state contracts",
        source_text=source,
    )
    stats = evaluate_verdicts([verdict], {"GraphARC enforces typed state contracts": True})
    assert stats.false_accepts == 0  # looks safe
    assert stats.true_accepts == 0  # and is useless
    assert stats.false_rejects == 1


def test_supersede_keeps_the_old_claim_and_its_provenance():
    store = MemoryStore()
    old = store.add(
        Claim(subject="auth-service", predicate="times out after", object="30s",
              source="config/prod.yaml", run_id="run-12")
    )
    new = store.supersede(
        old.id,
        Claim(subject="auth-service", predicate="times out after", object="10s",
              source="incident-4412", run_id="run-37"),
    )
    (dead,) = store.dead_ends("auth-service")
    assert dead.id == old.id
    assert dead.source == "config/prod.yaml" and dead.run_id == "run-12"
    assert dead.superseded_by == new.id
    assert [c.id for c in store.history("auth-service", "times out after")] == [old.id, new.id]


def test_render_context_never_drops_its_truncation_note():
    store = MemoryStore()
    for i in range(8):
        store.add(
            Claim(subject="auth-service", predicate=f"fact {i}", object=f"value {i}",
                  source=f"source-{i}")
        )
    brief = render_context(store, entities=["auth-service"], max_tokens=60)
    assert "lines omitted to fit a 60-token budget" in brief
