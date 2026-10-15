"""Resolve bare Slack readers to an explicitly named recent run.

Channel history is process-local. After restart, or in a new channel, readers
fall back to the newest trace under the bot's own `slack-runs/` directory and
say so. Explicit reader arguments never pass through this resolver.
"""

from __future__ import annotations

import threading
from pathlib import Path

from grapharc.observe.trace import TraceReadError, TraceRecorder
from grapharc.slack.command import SlackCommandError, confine_path

READERS = frozenset({"metrics", "viz", "replay", "trace"})


class RecentRuns:
    """Latest completed trace per channel, shared by a bot's worker threads."""

    def __init__(self) -> None:
        self._channels: dict[tuple[Path, str], Path] = {}
        self._lock = threading.Lock()

    def remember(self, workdir: Path, channel: str, path: Path | None) -> None:
        if not channel or path is None:
            return
        try:
            workdir = workdir.resolve()
            resolved = path.resolve()
            confine_path(str(resolved), workdir)
            if not resolved.is_relative_to((workdir / "slack-runs").resolve()):
                return
            if not resolved.is_file():
                return
        except (SlackCommandError, OSError):
            return
        with self._lock:
            self._channels[(workdir.resolve(), channel)] = resolved

    def resolve(self, argv: list[str], workdir: Path, channel: str = "") -> tuple[list[str], str]:
        """Complete a bare reader or refuse an empty/ambiguous history."""
        if len(argv) != 1 or argv[0] not in READERS:
            return argv, ""
        workdir = workdir.resolve()
        name = argv[0]
        with self._lock:
            selected = self._channels.get((workdir.resolve(), channel)) if channel else None
        fallback = selected is None
        if selected is not None:
            confine_path(str(selected), workdir)
            if not selected.is_file():
                selected = None
                fallback = True
        if selected is None:
            root = workdir / "slack-runs"
            confine_path(str(root), workdir)
            candidates: list[tuple[int, str, Path]] = []
            try:
                for path in root.glob("*/trace.jsonl"):
                    # Recheck before stat/read: generated paths and symlinks
                    # must obey the same boundary as explicitly typed paths.
                    try:
                        confine_path(str(path), workdir)
                        if path.is_file():
                            candidates.append((path.stat().st_mtime_ns, str(path), path))
                    except (SlackCommandError, OSError):
                        continue
            except OSError as exc:
                raise SlackCommandError(f"could not read recent Slack runs: {exc}") from None
            if not candidates:
                raise SlackCommandError(
                    f"No prior Slack run for `{name}`. Run `demo stage0` first, "
                    f"or supply `{name} path/to/trace.jsonl"
                    + (" run_id`." if name != "trace" else "`.")
                )
            selected = max(candidates)[2]
        confine_path(str(selected), workdir)
        relative = str(selected.resolve().relative_to(workdir.resolve()))
        try:
            runs = TraceRecorder(selected).run_ids()
        except (OSError, TraceReadError) as exc:
            raise SlackCommandError(
                f"cannot read the latest Slack trace `{relative}`: {exc}"
            ) from None
        if len(runs) != 1:
            raise SlackCommandError(
                f"Latest Slack trace `{relative}` contains {len(runs)} runs. "
                f"Supply an explicit path and run id for `{name}`."
            )
        run_id = runs[0]
        resolved = [name, relative]
        resolved.extend(["--run-id", run_id] if name == "trace" else [run_id])
        source = "workdir fallback" if fallback else "this channel"
        return resolved, f"{name} for `{relative}` · run `{run_id}` ({source})\n"
