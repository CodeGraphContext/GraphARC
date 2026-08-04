"""Positioned geometry for a `GraphSnapshot`, computed server-side.

The live page stays dumb by design (`server.live` docstring) — so the layout
runs here, in pure Python with no dependencies, and the browser receives
coordinates it merely draws. That buys three things at once: the layout is
ordinary tested code rather than the only untested JavaScript in the repo;
it is deterministic, so two snapshots of the same topology carry identical
geometry and the page can patch statuses in place without anything moving;
and it is importable without the `server` extra, so a static SVG export can
reuse it later.

The algorithm is a small Sugiyama: longest-path ranks over the forward edges,
a few barycenter sweeps for in-rank order, then top-down coordinates. Cycles
are a fact of these graphs — a LangGraph router looping back to an earlier
node is normal — so back-edges are found by DFS first and excluded from
ranking. That step is a hard requirement, not an optimization: the snapshot
builder runs inside the live server's poll thread, and a hand-written cyclic
trace must terminate, never hang it.

Text width is estimated from character count rather than measured — at the
graph sizes this serves (tens of nodes), a clamped estimate and an ellipsis
in the renderer beat shipping a font metrics table.
"""

from __future__ import annotations

from grapharc.observe.viewmodel import EdgeView, GraphSnapshot, NodeView

NODE_H = 52.0
NODE_W_MIN = 150.0
NODE_W_MAX = 230.0
TERMINAL_SIZE = 28.0
GAP_X = 36.0
GAP_Y = 56.0
MARGIN = 24.0
CLUSTER_PAD = 20.0
CLUSTER_GAP = 40.0
#: Barycenter passes: down, up, down. More sweeps buy nothing at these sizes.
ORDER_SWEEPS = 3


def _node_width(node: NodeView) -> float:
    if node.role != "node":
        return TERMINAL_SIZE
    return min(NODE_W_MAX, max(NODE_W_MIN, 24 + 7.5 * len(node.label)))


def _back_edges(order: list[str], adjacency: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Edges that point at an ancestor of the DFS path — iterative, cycle-safe."""
    back: set[tuple[str, str]] = set()
    visited: set[str] = set()
    on_path: set[str] = set()
    for root in order:
        if root in visited:
            continue
        # (node, iterator-index) pairs; an explicit stack instead of recursion
        # so a long chain cannot hit the interpreter's recursion limit.
        stack: list[tuple[str, int]] = [(root, 0)]
        on_path.add(root)
        visited.add(root)
        while stack:
            node, index = stack[-1]
            targets = adjacency.get(node, [])
            if index >= len(targets):
                stack.pop()
                on_path.discard(node)
                continue
            stack[-1] = (node, index + 1)
            target = targets[index]
            if target in on_path:
                back.add((node, target))
            elif target not in visited:
                visited.add(target)
                on_path.add(target)
                stack.append((target, 0))
    return back


def _ranks(
    ids: list[str], edges: list[tuple[str, str]]
) -> tuple[dict[str, int], set[tuple[str, str]]]:
    """Longest-path rank per node over the forward edges; also the back set."""
    adjacency: dict[str, list[str]] = {i: [] for i in ids}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
    back = _back_edges(ids, adjacency)
    forward = [(s, t) for s, t in edges if (s, t) not in back and s in adjacency and t in adjacency]

    indegree = dict.fromkeys(ids, 0)
    for _, target in forward:
        indegree[target] += 1
    rank = dict.fromkeys(ids, 0)
    queue = [i for i in ids if indegree[i] == 0]
    while queue:
        node = queue.pop(0)
        for source, target in forward:
            if source != node:
                continue
            rank[target] = max(rank[target], rank[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return rank, back


def _ordered_rows(
    ids: list[str], rank: dict[str, int], edges: list[tuple[str, str]]
) -> list[list[str]]:
    """Nodes per rank, ordered by a few barycenter sweeps; ties keep declaration order."""
    rows: list[list[str]] = [[] for _ in range(max(rank.values(), default=0) + 1)]
    for i in ids:
        rows[rank[i]].append(i)

    neighbors_down: dict[str, list[str]] = {}
    neighbors_up: dict[str, list[str]] = {}
    for source, target in edges:
        neighbors_down.setdefault(source, []).append(target)
        neighbors_up.setdefault(target, []).append(source)

    def sweep(rows: list[list[str]], neighbors: dict[str, list[str]], indices: range) -> None:
        for row_index in indices:
            adjacent = rows[row_index - 1] if neighbors is neighbors_up else rows[row_index + 1]
            position = {name: i for i, name in enumerate(adjacent)}
            keyed: dict[str, float] = {}
            for i, name in enumerate(rows[row_index]):
                anchors = [position[n] for n in neighbors.get(name, []) if n in position]
                # A node with no neighbor in the adjacent rank keeps its place.
                keyed[name] = sum(anchors) / len(anchors) if anchors else float(i)
            rows[row_index].sort(key=lambda name: keyed[name])  # stable: ties hold order

    for pass_index in range(ORDER_SWEEPS):
        if len(rows) < 2:
            break
        if pass_index % 2 == 0:
            sweep(rows, neighbors_up, range(1, len(rows)))
        else:
            sweep(rows, neighbors_down, range(len(rows) - 2, -1, -1))
    return rows


def _place_scope(
    nodes: list[NodeView], edges: list[EdgeView]
) -> tuple[float, float]:
    """Lay out one scope (the whole graph, or one cluster) at origin.

    Returns (width, height) of the laid-out block. Mutates node geometry and
    the points of edges whose endpoints are both in this scope.
    """
    by_id = {n.id: n for n in nodes}
    pairs = [(e.source, e.target) for e in edges if e.source in by_id and e.target in by_id]
    rank, back = _ranks(list(by_id), pairs)
    rows = _ordered_rows(list(by_id), rank, [p for p in pairs if p not in back])

    for node in nodes:
        node.w = _node_width(node)
        node.h = TERMINAL_SIZE if node.role != "node" else NODE_H

    row_widths = [
        sum(by_id[i].w for i in row) + GAP_X * max(0, len(row) - 1) for row in rows
    ]
    block_width = max(row_widths, default=0.0)
    y = 0.0
    for row, row_width in zip(rows, row_widths, strict=True):
        x = (block_width - row_width) / 2
        row_height = max((by_id[i].h for i in row), default=NODE_H)
        for i in row:
            node = by_id[i]
            node.x = x
            # Terminals are shorter than nodes; center them on the row.
            node.y = y + (row_height - node.h) / 2
            x += node.w + GAP_X
        y += row_height + GAP_Y
    block_height = max(0.0, y - GAP_Y)

    for edge in edges:
        source, target = by_id.get(edge.source), by_id.get(edge.target)
        if source is None or target is None:
            continue
        if edge.source == edge.target:
            # Self-loop: a small bulge off the node's right edge.
            rx, cy = source.x + source.w, source.y + source.h / 2
            edge.points = [
                [rx, cy - 8],
                [rx + 40, cy - 22],
                [rx + 40, cy + 22],
                [rx, cy + 8],
            ]
        elif (edge.source, edge.target) in back:
            # Back-edge: arc around the right of the block, bottom to top.
            sx, sy = source.x + source.w, source.y + source.h / 2
            tx, ty = target.x + target.w, target.y + target.h / 2
            arc_x = block_width + 48
            edge.points = [[sx, sy], [arc_x, sy], [arc_x, ty], [tx, ty]]
        else:
            sx, sy = source.x + source.w / 2, source.y + source.h
            tx, ty = target.x + target.w / 2, target.y
            bend = min(GAP_Y / 2, max(12.0, (ty - sy) / 2))
            edge.points = [[sx, sy], [sx, sy + bend], [tx, ty - bend], [tx, ty]]
    return block_width, block_height


def layout_graph(snapshot: GraphSnapshot) -> GraphSnapshot:
    """Fill geometry in place and return the snapshot.

    Clusters are laid out independently and stacked vertically — each round
    of a planner run really is its own graph — with the dotted `state` edge
    drawn frame-to-frame between them.
    """
    if not snapshot.nodes:
        snapshot.width = snapshot.height = 0.0
        return snapshot

    if snapshot.clusters:
        y_offset = MARGIN
        max_width = 0.0
        for cluster in snapshot.clusters:
            members = [n for n in snapshot.nodes if n.cluster == cluster.id]
            member_ids = {n.id for n in members}
            scoped_edges = [
                e
                for e in snapshot.edges
                if e.kind != "state" and e.source in member_ids and e.target in member_ids
            ]
            width, height = _place_scope(members, scoped_edges)
            for node in members:
                node.x += MARGIN + CLUSTER_PAD
                node.y += y_offset + CLUSTER_PAD
            for edge in scoped_edges:
                edge.points = [
                    [x + MARGIN + CLUSTER_PAD, y + y_offset + CLUSTER_PAD]
                    for x, y in edge.points
                ]
            cluster.x = MARGIN
            cluster.y = y_offset
            cluster.w = width + 2 * CLUSTER_PAD
            cluster.h = height + 2 * CLUSTER_PAD
            max_width = max(max_width, cluster.w)
            y_offset += cluster.h + CLUSTER_GAP
        frames = {c.id: c for c in snapshot.clusters}
        for edge in snapshot.edges:
            if edge.kind != "state":
                continue
            source, target = frames.get(edge.source), frames.get(edge.target)
            if source is None or target is None:
                continue
            sx, sy = source.x + source.w / 2, source.y + source.h
            tx, ty = target.x + target.w / 2, target.y
            edge.points = [[sx, sy], [sx, sy + 16], [tx, ty - 16], [tx, ty]]
        snapshot.width = max_width + 2 * MARGIN + 56  # room for back-edge arcs
        snapshot.height = y_offset - CLUSTER_GAP + MARGIN
    else:
        width, height = _place_scope(snapshot.nodes, snapshot.edges)
        for node in snapshot.nodes:
            node.x += MARGIN
            node.y += MARGIN
        for edge in snapshot.edges:
            edge.points = [[x + MARGIN, y + MARGIN] for x, y in edge.points]
        snapshot.width = width + 2 * MARGIN + 56
        snapshot.height = height + 2 * MARGIN
    return snapshot


__all__ = ["layout_graph"]
