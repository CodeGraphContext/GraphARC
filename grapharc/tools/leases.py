"""Write leases: the first writer holds a path, the second is refused by name.

Two fixers landing in one superstep genuinely run at the same time, and two
concurrent writers to one file is corruption however well each behaves. The
lease makes the conflict *data* instead: the first `write_file`/`edit_file`
to touch a path claims it for that node, the loser's call is denied with the
holder named, and — because an agent's tool refusals are observations it
reads — the losing fixer learns who has the file rather than crashing the
batch. Nothing merges divergent edits; this prevents the silent version of
the problem, not the disagreement itself.

Enforcement sits where the harness puts enforcement: a pre-hook, consulted
per call after permissions, before the executor. The tools themselves are
untouched — a harness built without the hook behaves exactly as before.

Scope, stated plainly: a lease is advisory within one process and covers the
core write tools only. `run_command` children and delegated agents mutate
un-leased, and a second *process* is outside this object entirely — it is a
coordination device for one governed run, not a cross-process file lock.
"""

from __future__ import annotations

import os
import threading

from grapharc.harness.hooks import HookAction, HookDecision, PreHook
from grapharc.tools.workspace import ToolError, Workspace

#: The tools a lease gates. Everything else — reads, searches, the shell —
#: passes untouched; gating reads would serialize the listeners, which is the
#: parallelism the lease exists to keep safe.
LEASED_TOOLS = ("write_file", "edit_file")


class PathLeases:
    """Per-run lease table. One instance per governed loop, shared by its nodes.

    Reentrant for the holder: a fixer that writes, reads, and writes again
    holds its path throughout. Released whole per holder — a node's lease
    lives exactly as long as its execution, which is what lets a later round
    edit a file an earlier round's fixer wrote.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holders: dict[str, str] = {}

    def acquire(self, key: str, holder: str) -> str | None:
        """Claim `key` for `holder`. None on success; the current holder's name
        when the claim is lost. First writer wins, atomically."""
        with self._lock:
            current = self._holders.get(key)
            if current is None:
                self._holders[key] = holder
                return None
            return None if current == holder else current

    def holder_of(self, key: str) -> str | None:
        with self._lock:
            return self._holders.get(key)

    def release_all(self, holder: str) -> None:
        """Release every path `holder` held. Idempotent — releasing a holder
        that holds nothing is not an error, so a body's `finally` never is."""
        with self._lock:
            for key in [k for k, h in self._holders.items() if h == holder]:
                del self._holders[key]

    def hook(self, holder: str, workspace: str | os.PathLike[str]) -> PreHook:
        """The pre-hook enforcing this table for one node instance.

        Paths are resolved against the workspace exactly as the tools resolve
        them, so `a.txt` and `./sub/../a.txt` contend for one lease. A path
        the workspace refuses is left for the tool to refuse — the tool's own
        message names the escape; a lease denial here would misname the
        problem.
        """
        root = Workspace(workspace)

        def lease_gate(tool_name: str, args: dict) -> HookDecision | None:
            if tool_name not in LEASED_TOOLS:
                return None
            path = args.get("path")
            if not isinstance(path, str):
                return None  # the tool refuses malformed input with its own message
            try:
                key = str(root.resolve(path))
            except ToolError:
                return None
            other = self.acquire(key, holder)
            if other is None:
                return None
            return HookDecision(
                action=HookAction.DENY,
                reason=(
                    f"{root.display(root.resolve(path))} is being changed by "
                    f"{other!r} right now. Leave it to {other!r} and work on "
                    "something else; if your change depends on that file, say "
                    "so in your report instead of editing it."
                ),
            )

        return lease_gate


__all__ = ["LEASED_TOOLS", "PathLeases"]
