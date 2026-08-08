"""Write leases — `grapharc.tools.leases`.

Two fixers in one superstep genuinely run at the same time, and two concurrent
writers to one file is corruption however well each behaves. The tests pin the
property that makes parallel fixers safe to admit: exactly one writer lands,
the loser is refused *by name*, the refusal is data an agent reads rather than
a crash that sinks the batch — and the lease dies with its node, so a later
round may edit what an earlier round wrote.
"""

from __future__ import annotations

import pytest

from grapharc.harness.permissions import PermissionDenied
from grapharc.runtime.fanout import run_guarded
from grapharc.stdlib import WRITE_TOOLS, default_harness
from grapharc.tools.leases import PathLeases


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "shared.txt").write_text("original\n")
    return tmp_path


def _pair(workspace):
    leases = PathLeases()
    one = default_harness(WRITE_TOOLS, workspace, leases=leases, lease_holder="fix_1")
    two = default_harness(WRITE_TOOLS, workspace, leases=leases, lease_holder="fix_2")
    return leases, one, two


def test_the_second_writer_is_refused_and_the_refusal_names_the_holder(workspace):
    _, one, two = _pair(workspace)
    one.call("write_file", {"path": "a.txt", "content": "first"})

    with pytest.raises(PermissionDenied) as refusal:
        two.call("write_file", {"path": "a.txt", "content": "second"})

    assert "fix_1" in str(refusal.value)
    assert (workspace / "a.txt").read_text() == "first"


def test_edit_contends_for_the_same_lease_as_write(workspace):
    _, one, two = _pair(workspace)
    one.call("edit_file", {"path": "shared.txt", "old_string": "original", "new_string": "one"})

    with pytest.raises(PermissionDenied):
        two.call(
            "edit_file", {"path": "shared.txt", "old_string": "one", "new_string": "two"}
        )


def test_the_lease_is_reentrant_for_its_holder(workspace):
    """A fixer that writes, reads, and writes again holds its path throughout."""
    _, one, _ = _pair(workspace)
    one.call("write_file", {"path": "a.txt", "content": "first"})
    one.call("write_file", {"path": "a.txt", "content": "second"})
    assert (workspace / "a.txt").read_text() == "second"


def test_different_paths_do_not_contend(workspace):
    _, one, two = _pair(workspace)
    one.call("write_file", {"path": "a.txt", "content": "one"})
    two.call("write_file", {"path": "b.txt", "content": "two"})
    assert (workspace / "b.txt").read_text() == "two"


def test_reads_are_never_gated(workspace):
    """Gating reads would serialize the listeners — the parallelism the lease
    exists to keep safe."""
    _, one, two = _pair(workspace)
    one.call("write_file", {"path": "shared.txt", "content": "held"})
    assert two.call("read_file", {"path": "shared.txt"}) == "held"


def test_a_dressed_up_path_contends_with_its_plain_spelling(workspace):
    """Leases key on the resolved path, exactly as the tools resolve it."""
    (workspace / "sub").mkdir()
    _, one, two = _pair(workspace)
    one.call("write_file", {"path": "a.txt", "content": "one"})

    with pytest.raises(PermissionDenied):
        two.call("write_file", {"path": "sub/../a.txt", "content": "two"})


def test_release_ends_the_lease_so_a_later_round_can_edit(workspace):
    leases, one, two = _pair(workspace)
    one.call("write_file", {"path": "a.txt", "content": "round one"})

    leases.release_all("fix_1")
    two.call("write_file", {"path": "a.txt", "content": "round two"})
    assert (workspace / "a.txt").read_text() == "round two"


def test_racing_writers_produce_one_file_and_one_named_refusal(workspace):
    """The concurrent case the lease exists for: exactly one write lands, the
    loser's failure is data carrying the holder's name, the batch completes."""
    _, one, two = _pair(workspace)

    def writer(harness, name):
        def work():
            harness.call("write_file", {"path": "raced.txt", "content": name})
            return [{"worker": name}]

        return work

    results = [
        run_guarded(writer(h, n), worker=n, timeout_seconds=10)
        for h, n in ((one, "fix_1"), (two, "fix_2"))
    ]

    winners = [r for r in results if r.ok]
    losers = [r for r in results if not r.ok]
    assert len(winners) == 1 and len(losers) == 1
    assert winners[0].worker in ("fix_1", "fix_2")
    assert winners[0].worker in (workspace / "raced.txt").read_text()
    assert winners[0].worker in losers[0].error  # the refusal names the holder


def test_a_workspace_escape_is_the_tools_refusal_not_a_lease(workspace):
    """A path the workspace refuses is left for the tool to refuse — a lease
    denial would misname the problem."""
    from grapharc.tools.workspace import ToolError

    leases, one, _ = _pair(workspace)
    with pytest.raises(ToolError):
        one.call("write_file", {"path": "../outside.txt", "content": "x"})
    assert leases.holder_of(str(workspace.parent / "outside.txt")) is None


def test_a_fix_one_body_releases_its_leases_when_it_finishes(workspace):
    """The lease lives exactly as long as the node's execution: the factory's
    `finally` releases the instance's holdings even on the happy path."""
    from pydantic import PrivateAttr

    from grapharc.planner import NodeBuild
    from grapharc.registries import fix_issues
    from grapharc.runtime.budget import Budget, BudgetMeter
    from grapharc.runtime.graph import RunContext
    from grapharc.testing import ScriptedChatModel

    class ToolCallingModel(ScriptedChatModel):
        _bound: list = PrivateAttr(default_factory=list)

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            self._bound.append(tools)
            return self

    leases = PathLeases()
    registry = fix_issues.build_registry(
        ToolCallingModel(responses=["took the issue; changed nothing"]),
        workspace=workspace,
        leases=leases,
    )
    factory = registry.get("fix_one").factory
    body = factory(
        NodeBuild(
            name="fix_1",
            kind="fix_one",
            args={"issue": "issue: a"},
            proposal_id="p-1",
            fingerprint="f-1",
        )
    )

    leases.acquire(str(workspace / "held.txt"), "fix_1")
    ctx = RunContext(run_id="r-1", graph="fix", meter=BudgetMeter(Budget()))
    update = body(fix_issues.FixState(goal="fix", issues=["issue: a"]), ctx)

    assert update["fixes"]
    assert leases.holder_of(str(workspace / "held.txt")) is None
