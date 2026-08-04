"""Credential resolution for gateway backends.

Reads a `.env` file and the process environment, accepting several spellings
of each key because real projects are inconsistent about them — a key named
`open-router-api-key` cannot be a shell variable at all, so the file has to be
parsed rather than sourced.

**The `.env` is read from one directory and no other** — the working directory,
or whatever `start` names — which is the same rule `grapharc.toml` follows, and
for a stronger reason. The config layer refuses an upward search because a run
must never be silently governed by a policy file in a directory the operator did
not know about; this file *spends money*, so a key discovered three directories
up is the worse version of that failure. It used to walk to `/`, which meant a
`.env` in `$HOME` billed every experiment on the box to that key, and `redact`
being the only thing that ever prints a key meant nothing revealed which file
paid. The escape hatches are explicit and unchanged: a real environment variable
still wins over any file, and a caller can still name a file anywhere with
`env_file=`.

Secrets are returned, never logged. Anything that renders a config for humans
goes through `redact`.

Three key-holding backends, plus one that usually holds none:

- **OpenRouter** — `OPENROUTER_API_KEY`.
- **OpenAI** — `OPENAI_API_KEY`, optionally with `OPENAI_BASE_URL` for an
  Azure-style or proxied endpoint. `langchain-openai` reads `OPENAI_API_KEY`
  from the process environment on its own; going through here as well is what
  adds `.env` support, the alternate spellings, and a failure that names the
  variable instead of surfacing an SDK error.
- **Ollama** — no credential by default: it is a server on your own machine.
  What it needs is an address, so `ollama_base_url()` always returns one
  (`OLLAMA_HOST` / `OLLAMA_BASE_URL`, else localhost). `OLLAMA_API_KEY` exists
  for the case where that address is an authenticating proxy rather than the
  daemon itself, and is None the rest of the time.
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

OPENAI_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "openai-api-key",
    "openai_api_key",
)

# `OPENAI_API_BASE` is the older spelling and is still what a lot of tooling
# sets; both are accepted, the current one first.
OPENAI_BASE_URL_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "openai-base-url",
)

OLLAMA_KEYS = (
    "OLLAMA_API_KEY",
    "ollama-api-key",
)

# `OLLAMA_HOST` is what the ollama CLI itself reads, so a machine that already
# points its CLI at a remote daemon points GraphARC there too, with no second
# variable to set.
OLLAMA_BASE_URL_KEYS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_HOST",
    "ollama-base-url",
)

OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_HOST = f"http://localhost:{OLLAMA_DEFAULT_PORT}"
#: Ollama's OpenAI-compatible surface lives under `/v1`; the native API does not.
OLLAMA_OPENAI_PATH = "/v1"
DEFAULT_OLLAMA_BASE_URL = OLLAMA_DEFAULT_HOST + OLLAMA_OPENAI_PATH


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
    """The `.env` in `start` itself (default: cwd), or None. Parents are not read."""
    candidate = (start or Path.cwd()).resolve() / ".env"
    return candidate if candidate.is_file() else None


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


def openai_api_key(*, env_file: Path | None = None) -> str | None:
    return get_secret(OPENAI_KEYS, env_file=env_file)


def openai_base_url(*, env_file: Path | None = None) -> str | None:
    """An endpoint override, or None for api.openai.com.

    None rather than a default, because the OpenAI SDK already knows its own
    base URL and hard-coding a copy of it here would be one more string to get
    wrong when it changes.
    """
    return get_secret(OPENAI_BASE_URL_KEYS, env_file=env_file)


def ollama_api_key(*, env_file: Path | None = None) -> str | None:
    """Usually None. Set only when the Ollama endpoint is behind auth."""
    return get_secret(OLLAMA_KEYS, env_file=env_file)


def normalize_ollama_base_url(raw: str) -> str:
    """Turn what people actually set into a URL the OpenAI client can use.

    `OLLAMA_HOST` is an ollama-CLI variable, not a URL variable, and it is
    routinely set to bare forms like `127.0.0.1:11434` or `my-box`. The OpenAI
    client needs a full URL ending in the compatibility path, and gets a
    confusing `Invalid URL` much later if it does not, so the shapes are
    reconciled here:

        127.0.0.1:11434  ->  http://127.0.0.1:11434/v1
        my-box           ->  http://my-box:11434/v1
        https://ai/v1/   ->  https://ai/v1

    **The port is only filled in for the bare form.** A value with a scheme is
    a URL and is left to URL rules, so `https://ollama.internal` stays on 443
    rather than being rewritten to a port nothing is listening on — a proxied
    Ollama is exactly the case where that guess would be wrong. An IPv6
    literal is bracketed and already contains colons, so it is treated as
    carrying its own port; write `http://[::1]:11434` if it does not.
    """
    url = raw.strip().rstrip("/")
    if "://" not in url:
        host, slash, path = url.partition("/")
        if ":" not in host:
            host = f"{host}:{OLLAMA_DEFAULT_PORT}"
        url = f"http://{host}{slash}{path}"
    if not url.endswith(OLLAMA_OPENAI_PATH):
        url += OLLAMA_OPENAI_PATH
    return url


def ollama_base_url(*, env_file: Path | None = None) -> str:
    """Where the Ollama server is. Always a URL — localhost when nothing is set.

    Unlike the API keys this never returns None: an unconfigured Ollama is not
    an unusable one, it is one running where Ollama installs itself.
    """
    configured = get_secret(OLLAMA_BASE_URL_KEYS, env_file=env_file)
    return normalize_ollama_base_url(configured) if configured else DEFAULT_OLLAMA_BASE_URL


def redact(secret: str | None) -> str:
    """A form safe to print: enough to identify the key, not enough to use it."""
    if not secret:
        return "<unset>"
    if len(secret) <= 12:
        return "…" * 3
    return f"{secret[:7]}…{secret[-4:]}"
