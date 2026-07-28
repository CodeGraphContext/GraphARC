"""Which model backends this machine could use — checked locally, costing nothing.

`grapharc models --check` answers the question that otherwise costs a failed run
to discover: is a credential configured, is the optional dependency installed,
is the CLI on PATH.

What this is and is not: a *local* probe. No request is sent to any provider, so
it can report that a key is configured and never that the key is valid, in
credit, or entitled to a given model. Each result says which of the two it is.

No secret is printed. The OpenRouter key is rendered through `gateway.redact`,
which is the only form of it that leaves this module.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any

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
    lines = []
    for r in results:
        mark = {True: "usable", False: "unusable", None: "unprobed"}[r["usable"]]
        lines.append(f"{r['backend']:<12} {mark:<9} {r['detail']}")
        if r["credential"]:
            lines.append(f"{'':<12} {'':<9} credential: {r['credential']}")
    lines.append("")
    lines.append("local probe only — no provider was contacted, so a configured key")
    lines.append("is not a validated one.")
    return lines


__all__ = ["any_provider_usable", "probe_backends", "render"]
