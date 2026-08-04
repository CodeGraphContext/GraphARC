"""The server-side layout: deterministic, downward, and cycle-safe.

The layout runs inside the live server's snapshot thread, so the properties
under test are load-bearing: identical input must give identical geometry
(the page patches in place on that promise), and a cyclic topology must
terminate rather than hang the thread.
"""

from __future__ import annotations

from grapharc.observe.layout import layout_graph
from grapharc.observe.viewmodel import ClusterView, EdgeView, GraphSnapshot, NodeView


def _diamond(status: str = "pending") -> GraphSnapshot:
    return GraphSnapshot(
        kind="topology",
        nodes=[
            NodeView(id="a", label="a", status=status),
            NodeView(id="b", label="b"),
            NodeView(id="c", label="c"),
            NodeView(id="d", label="d"),
        ],
        edges=[
            EdgeView(source="a", target="b"),
            EdgeView(source="a", target="c"),
            EdgeView(source="b", target="d"),
            EdgeView(source="c", target="d"),
        ],
    )


def _geometry(snapshot: GraphSnapshot) -> list[tuple]:
    return [(n.id, n.x, n.y, n.w, n.h) for n in snapshot.nodes] + [
        (e.source, e.target, tuple(map(tuple, e.points))) for e in snapshot.edges
    ]


def test_identical_input_gives_identical_geometry():
    assert _geometry(layout_graph(_diamond())) == _geometry(layout_graph(_diamond()))


def test_geometry_is_stable_when_only_statuses_change():
    # The flicker-free contract: a status flip must not move a single node.
    pending = layout_graph(_diamond())
    running = layout_graph(_diamond(status="running"))
    assert [(n.id, n.x, n.y) for n in pending.nodes] == [
        (n.id, n.x, n.y) for n in running.nodes
    ]


def test_forward_edges_rank_downward():
    snapshot = layout_graph(_diamond())
    y = {n.id: n.y for n in snapshot.nodes}
    assert y["a"] < y["b"] == y["c"] < y["d"]


def test_no_overlap_within_a_rank():
    snapshot = layout_graph(_diamond())
    b, c = (next(n for n in snapshot.nodes if n.id == i) for i in ("b", "c"))
    left, right = (b, c) if b.x < c.x else (c, b)
    assert left.x + left.w <= right.x


def test_cyclic_topology_terminates_and_marks_the_back_edge():
    snapshot = GraphSnapshot(
        kind="topology",
        nodes=[NodeView(id=i, label=i) for i in ("extract", "verify", "retry")],
        edges=[
            EdgeView(source="extract", target="verify"),
            EdgeView(source="verify", target="retry"),
            EdgeView(source="retry", target="extract"),  # the loop back
        ],
    )
    laid = layout_graph(snapshot)  # must return, not hang
    y = {n.id: n.y for n in laid.nodes}
    assert y["extract"] < y["verify"] < y["retry"]
    # The back-edge routed as a side arc: strictly right of the block's nodes.
    back = next(e for e in laid.edges if (e.source, e.target) == ("retry", "extract"))
    rightmost = max(n.x + n.w for n in laid.nodes)
    assert max(x for x, _ in back.points) > rightmost


def test_self_loop_terminates_and_bulges_off_the_node():
    snapshot = GraphSnapshot(
        kind="topology",
        nodes=[NodeView(id="loop", label="loop")],
        edges=[EdgeView(source="loop", target="loop")],
    )
    laid = layout_graph(snapshot)
    node = laid.nodes[0]
    assert max(x for x, _ in laid.edges[0].points) > node.x + node.w


def test_clusters_stack_vertically_and_contain_their_nodes():
    snapshot = GraphSnapshot(
        kind="topology",
        clusters=[
            ClusterView(id="g0", label="round 1", round=1),
            ClusterView(id="g1", label="round 2", round=2),
        ],
        nodes=[
            NodeView(id="g0.work", label="work", cluster="g0"),
            NodeView(id="g1.work", label="work", cluster="g1"),
        ],
        edges=[EdgeView(source="g0", target="g1", kind="state")],
    )
    laid = layout_graph(snapshot)
    first, second = laid.clusters
    assert first.y + first.h <= second.y
    for node in laid.nodes:
        frame = first if node.cluster == "g0" else second
        assert frame.x <= node.x and node.x + node.w <= frame.x + frame.w
        assert frame.y <= node.y and node.y + node.h <= frame.y + frame.h
    # The state edge runs frame-to-frame.
    state = laid.edges[0]
    assert state.points[0][1] == first.y + first.h
    assert state.points[-1][1] == second.y


def test_canvas_covers_every_node():
    laid = layout_graph(_diamond())
    assert laid.width >= max(n.x + n.w for n in laid.nodes)
    assert laid.height >= max(n.y + n.h for n in laid.nodes)


def test_empty_snapshot_is_zero_sized():
    laid = layout_graph(GraphSnapshot(kind="empty"))
    assert (laid.width, laid.height) == (0.0, 0.0)
