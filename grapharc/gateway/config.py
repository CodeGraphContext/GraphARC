"""Credential resolution for gateway backends.

Reads a `.env` file and the process environment, accepting several spellings
of each key because real projects are inconsistent about them — a key named
`open-router-api-key` cannot be a shell variable at all, so the file has to be
parsed rather than sourced.

Secrets are returned, never logged. Anything that renders a config for humans
goes through `redact`.
"""

from __future__ import annotations

import os
from pathlib import Path

# First spelling that resolves wins. The conventional name is listed first so
# a real environment variable always beats a checked-out file.
OPENROUTER_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "open-router-api-key",
    "openrouter_api_key",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal `.env` parser — `KEY=value`, `#` comments, optional quotes."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Nearest `.env` walking up from `start` (default: cwd)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def get_secret(names: tuple[str, ...], *, env_file: Path | None = None) -> str | None:
    """First value found under any of `names`, process env before file."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    path = env_file or find_env_file()
    if path is None:
        return None
    values = _parse_env_file(path)
    for name in names:
        if values.get(name):
            return values[name].strip()
    return None


def openrouter_api_key(*, env_file: Path | None = None) -> str | None:
    return get_secret(OPENROUTER_KEYS, env_file=env_file)


def redact(secret: str | None) -> str:
    """A form safe to print: enough to identify the key, not enough to use it."""
    if not secret:
        return "<unset>"
    if len(secret) <= 12:
        return "…" * 3
    return f"{secret[:7]}…{secret[-4:]}"
