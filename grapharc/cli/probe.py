"""Which model backends this machine could use — checked locally, costing nothing.

`grapharc models --check` answers the question that otherwise costs a failed run
to discover: is a credential configured, is the optional dependency installed,
is the CLI on PATH.

What this is and is not: a *local* probe. No request is sent to any provider, so
it can report that a key is configured and never that the key is valid, in
credit, or entitled to a given model. Each result says which of the two it is.

No secret is printed. API keys are rendered through `gateway.redact`, which is
the only form of them that leaves this module.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any

from grapharc.cli import style

#: The three columns every row prints, in characters. Constants, not measurements
#: of the terminal: `tests/test_cli.py` asserts on `"bedrock      unprobed"`, and
#: the page in `docs/cookbook/02-models.md` shows this table verbatim.
BACKEND_COLUMN = 12
MARK_COLUMN = 9

# A backend that is a test double rather than a provider. `--check` exits
# non-zero when nothing real is usable, and `mock` being always-usable must not
# make an unconfigured machine look ready.
KIND_PROVIDER = "provider"
KIND_DOUBLE = "test-double"


def _probe_claude_cli(claude_path: str) -> dict[str, Any]:
    found = shutil.which(claude_path)
    return {
        "backend": "claude-cli",
        "kind": KIND_PROVIDER,
        "usable": found is not None,
        "credential": "claude subscription login (no API key)",
        "detail": (
            f"{claude_path!r} on PATH at {found}"
            if found
            else f"{claude_path!r} not found on PATH — install Claude Code and log in"
        ),
        "checked": "executable on PATH only; no `claude -p` call was made",
    }


def _probe_openrouter() -> dict[str, Any]:
    from grapharc.gateway import openrouter_api_key, redact

    key = openrouter_api_key()
    has_dependency = importlib.util.find_spec("langchain_openai") is not None
    missing = []
    if not key:
        missing.append("no API key (set OPENROUTER_API_KEY, or add one to .env)")
    if not has_dependency:
        missing.append("langchain-openai not installed (uv sync --extra openrouter)")
    return {
        "backend": "openrouter",
        "kind": KIND_PROVIDER,
        "usable": bool(key) and has_dependency,
        "credential": redact(key),
        "detail": "; ".join(missing) or "api key configured and langchain-openai installed",
        "checked": "credential presence only; no request was sent to openrouter.ai",
    }


def _probe_openai() -> dict[str, Any]:
    from grapharc.gateway import openai_api_key, openai_base_url, redact

    key = openai_api_key()
    endpoint = openai_base_url()
    has_dependency = importlib.util.find_spec("langchain_openai") is not None
    missing = []
    if not key:
        missing.append("no API key (set OPENAI_API_KEY, or add one to .env)")
    if not has_dependency:
        missing.append("langchain-openai not installed (uv sync --extra openai)")
    ready = "api key configured and langchain-openai installed"
    if endpoint:
        # Worth printing: a key plus a base-url override means requests are not
        # going where the backend name implies.
        ready += f"; endpoint override {endpoint}"
    return {
        "backend": "openai",
        "kind": KIND_PROVIDER,
        "usable": bool(key) and has_dependency,
        "credential": redact(key),
        "detail": "; ".join(missing) or ready,
        "checked": "credential presence only; no request was sent to the endpoint",
    }


def _probe_ollama() -> dict[str, Any]:
    from grapharc.gateway import ollama_base_url

    base_url = ollama_base_url()
    has_dependency = importlib.util.find_spec("langchain_openai") is not None
    binary = shutil.which("ollama")
    # No credential to check, so what stands in for one is evidence that a
    # server exists: the CLI on PATH, or an address someone set on purpose.
    # Neither proves the daemon is running — only a request would, and this
    # command makes none.
    configured = binary is not None or _ollama_host_is_explicit()
    missing = []
    if not has_dependency:
        missing.append("langchain-openai not installed (uv sync --extra ollama)")
    if not configured:
        missing.append("no 'ollama' on PATH and no OLLAMA_HOST set")
    return {
        "backend": "ollama",
        "kind": KIND_PROVIDER,
        "usable": has_dependency and configured,
        "credential": "none needed (local server)",
        "detail": "; ".join(missing) or f"local server at {base_url}",
        "checked": "PATH and configuration only; the server was not contacted",
    }


def _ollama_host_is_explicit() -> bool:
    from grapharc.gateway.config import OLLAMA_BASE_URL_KEYS, get_secret

    return get_secret(OLLAMA_BASE_URL_KEYS) is not None


def _probe_mock() -> dict[str, Any]:
    return {
        "backend": "mock",
        "kind": KIND_DOUBLE,
        "usable": True,
        "credential": None,
        "detail": "scripted test double; never reaches a provider",
        "checked": "nothing to check",
    }


def probe_backends(*, claude_path: str = "claude") -> list[dict[str, Any]]:
    """Probe every backend the gateway registry declares.

    Driven by `BACKENDS` rather than a list kept here, so a backend added to the
    gateway shows up as `usable: null` — "nobody wrote a probe for this" — which
    is visibly different from "checked and unusable". A hard-coded list would
    have omitted it silently.
    """
    from grapharc.gateway.registry import BACKENDS

    probes = {
        "claude-cli": lambda: _probe_claude_cli(claude_path),
        "openrouter": _probe_openrouter,
        "openai": _probe_openai,
        "ollama": _probe_ollama,
        "mock": _probe_mock,
    }
    results = []
    for backend in BACKENDS:
        probe = probes.get(backend)
        if probe is None:
            results.append(
                {
                    "backend": backend,
                    "kind": KIND_PROVIDER,
                    "usable": None,
                    "credential": None,
                    "detail": "registered in the gateway, but this CLI has no probe for it",
                    "checked": "nothing",
                }
            )
        else:
            results.append(probe())
    return results


def any_provider_usable(results: list[dict[str, Any]]) -> bool:
    """True when at least one real provider is configured. Test doubles do not count."""
    return any(r["usable"] is True and r["kind"] == KIND_PROVIDER for r in results)


def render(results: list[dict[str, Any]]) -> list[str]:
    """The table `grapharc models --check` prints.

    On a terminal the mark column carries the verdict — green usable, red
    unusable, amber for a backend nobody wrote a probe for — which is the one
    thing a reader is scanning this table for. The columns are the same width
    either way, and the credential line stays indented under its own row.
    """
    lines = []
    for r in results:
        mark = {True: "usable", False: "unusable", None: "unprobed"}[r["usable"]]
        lines.append(
            f"{style.cell(r['backend'], BACKEND_COLUMN, tint=style.accent)} "
            f"{style.cell(mark, MARK_COLUMN, tint=style.tint_for(r['usable']))} "
            f"{r['detail']}"
        )
        if r["credential"]:
            indent = " " * (BACKEND_COLUMN + 1 + MARK_COLUMN + 1)
            lines.append(f"{indent}{style.dim('credential:')} {r['credential']}")
    lines.append("")
    # The caveat, not the answer: dim, so the table above it reads first.
    lines.append(style.dim("local probe only — no provider was contacted, so a configured key"))
    lines.append(style.dim("is not a validated one."))
    return lines


__all__ = ["any_provider_usable", "probe_backends", "render"]
