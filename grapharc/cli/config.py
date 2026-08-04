"""`grapharc.toml` — defaults for the flags nobody wants to retype.

Four layers, highest wins:

    flag  >  environment  >  grapharc.toml  >  built-in default

Every resolved value carries where it came from, and commands print that. "Which
policy was I subject to" is the question an incident review opens with, and a
config file makes the answer *less* visible on the command line, not more — so
the provenance is part of the payload rather than something a reader has to
reconstruct.

**It does not walk up parent directories, and that is deliberate.** git, npm and
cargo all search upward, and for a build tool that is right. For a tool whose
job is to prove what a run was permitted to do, it is wrong: a run must never be
silently governed by a policy file in a directory the operator did not know
about. The file is read from the current directory, or from an explicit
`--config PATH`, and from nowhere else. A missing file is not an error — every
key has a default.

TOML rather than JSON because the rest of the operator-authored surface is TOML,
and because this is a file where a comment explaining *why* a rule exists is
worth more than the rule.

    # grapharc.toml
    [grapharc]
    model      = "openrouter/anthropic/claude-haiku-4.5"
    registry   = "myco.incident:build_registry"
    policy     = "policy.toml"
    tenant     = "acme"
    max_rounds = 6
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "grapharc.toml"
TABLE = "grapharc"
ENV_PREFIX = "GRAPHARC_"

#: Keys the file may set, and the type each is coerced to. An unknown key is a
#: load error rather than a silent no-op: a typo in a governance file that is
#: quietly ignored is the same failure as a policy rule nobody reads.
KEYS: dict[str, type] = {
    "model": str,
    "reviewer_model": str,
    "registry": str,
    "policy": str,
    "tenant": str,
    "memory": str,
    "max_rounds": int,
    "max_tokens": int,
    "max_planning_failures": int,
    "max_iterations": int,
    "max_seconds": float,
    "max_concurrency": int,
}


class ConfigError(Exception):
    """Raised before anything runs, so a bad config never half-executes."""


@dataclass
class Settings:
    """Resolved defaults, and where each one came from."""

    path: Path | None = None
    values: dict[str, Any] = field(default_factory=dict)
    #: key -> "flag" | "env" | "config" | "default", filled in by `resolve`.
    sources: dict[str, str] = field(default_factory=dict)

    def resolve(self, key: str, flag_value: Any, default: Any = None) -> Any:
        """Pick a value for `key` and record which layer supplied it."""
        if flag_value is not None:
            self.sources[key] = "flag"
            return flag_value
        env = os.environ.get(f"{ENV_PREFIX}{key.upper()}")
        if env:
            self.sources[key] = "env"
            want = KEYS.get(key, str)
            # Same named error the file layer raises for a mistyped value. An
            # exported variable is invisible on the command line, which is
            # exactly why the message must say which one it was.
            try:
                return want(env)
            except ValueError as exc:
                raise ConfigError(
                    f"{ENV_PREFIX}{key.upper()} must be {want.__name__}, got {env!r}"
                ) from exc
        if key in self.values:
            self.sources[key] = "config"
            return self.values[key]
        self.sources[key] = "default"
        return default

    def resolve_path(self, key: str, flag_value: Path | None) -> Path | None:
        """Same, for a key whose value is a filesystem path.

        A relative path that came *from the config file* is resolved against
        that file's directory, not the working directory. Anchoring it to cwd
        instead would make a config usable only from the directory it lives in,
        which defeats `--config` — and would mean the same file named two
        different policies depending on where you happened to be standing.

        A path from a flag or an environment variable is left alone: someone
        typing a path means the one relative to where they typed it.
        """
        value = self.resolve(key, flag_value)
        if value is None:
            return None
        path = Path(value)
        if self.sources.get(key) == "config" and not path.is_absolute() and self.path is not None:
            return self.path.parent / path
        return path

    def provenance(self, *, policy_source: str | None = None) -> dict[str, Any]:
        """The block commands put in their payload, so the trail is on the record.

        `policy_source` overrides the `policy` entry. Without it the two
        disagreed: a policy read from `.grapharc/` reported `policy_source:
        generated-cached` while `sources.policy` still said `default`, because no
        flag or config named it. A reviewer reading `sources` alone would
        conclude no file was involved. The layer that actually supplied the
        policy wins.
        """
        sources = dict(self.sources)
        if policy_source is not None and sources.get("policy") in (None, "default"):
            sources["policy"] = policy_source
        return {
            "config_file": str(self.path) if self.path else None,
            "sources": sources,
        }

    def describe(self) -> str:
        """One line for the human view."""
        if self.path is None:
            return "no grapharc.toml (flags and defaults only)"
        named = ", ".join(sorted(self.values)) or "nothing"
        return f"{self.path} sets {named}"


def load(explicit: Path | None = None, *, cwd: Path | None = None) -> Settings:
    """Load `grapharc.toml` from `cwd`, or the file `--config` named.

    An explicit path that does not exist is an error — the operator asked for
    that file by name. An absent default file is not, because most runs have no
    config at all and inventing one would be worse than having none.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigError(f"--config: no such file: {path}")
    else:
        path = Path(cwd or Path.cwd()) / CONFIG_NAME
        if not path.is_file():
            return Settings()

    try:
        # `UnicodeDecodeError` is a `ValueError`, so it belongs in this tuple
        # explicitly: without it a stray binary `grapharc.toml` in the working
        # directory tracebacks out of every configurable command, because this
        # file is picked up implicitly rather than named by the operator.
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    table = document.get(TABLE, document)
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [{TABLE}] must be a table")

    # A key beside the table, not inside it, used to be dropped in silence: a
    # stray top-level `policy = "strict.toml"` was ignored and the run used the
    # permissive default. Validated here so the promise below holds for the
    # whole document rather than one table of it.
    if TABLE in document:
        stray = sorted(k for k in document if k != TABLE)
        if stray:
            raise ConfigError(
                f"{path}: {', '.join(stray)} sits beside the [{TABLE}] table "
                f"instead of inside it, so nothing reads it. Move it under "
                f"[{TABLE}]."
            )

    unknown = sorted(set(table) - set(KEYS))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {', '.join(unknown)} — "
            f"known keys are {', '.join(sorted(KEYS))}"
        )

    values: dict[str, Any] = {}
    for key, want in KEYS.items():
        if key not in table:
            continue
        given = table[key]
        # bool is an int subclass; a `true` where a count belongs is a mistake
        # worth naming rather than coercing to 1.
        if isinstance(given, bool) or not isinstance(given, want):
            raise ConfigError(
                f"{path}: {key} must be {want.__name__}, got {type(given).__name__}"
            )
        values[key] = given
    return Settings(path=path, values=values)


__all__ = ["CONFIG_NAME", "ENV_PREFIX", "KEYS", "ConfigError", "Settings", "load"]
