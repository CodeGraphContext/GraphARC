"""The CLI's topology comparison is stable without weakening proposal identity."""

from __future__ import annotations

import copy

import pytest

from grapharc.cli.graphrun import build_proposal, topology_fingerprint


def _document():
    return {
        "proposal_id": "outer-a",
        "origin": "operator-a",
        "nodes": [{
            "name": "worker",
            "args": {"path": "src", "origin": "argument-origin", "proposal_id": "argument-id"},
            "subgraph": {
                "proposal_id": "inner-a",
                "origin": "planner-a",
                "nodes": [{"name": "fetch"}],
                "edges": [
                    {"source": "__start__", "target": "fetch"},
                    {"source": "fetch", "target": "__end__"},
                ],
            },
        }],
        "edges": [
            {"source": "__start__", "target": "worker"},
            {"source": "worker", "target": "__end__"},
        ],
    }


def test_topology_digest_ignores_provenance_in_every_scope():
    document = _document()
    first = build_proposal(document)
    document["proposal_id"] = "outer-b"
    document["origin"] = "operator-b"
    nested = document["nodes"][0]["subgraph"]
    nested["proposal_id"] = "inner-b"
    nested["origin"] = "planner-b"
    second = build_proposal(document)

    assert topology_fingerprint(first) == topology_fingerprint(second)
    assert first.fingerprint() != second.fingerprint()


def test_topology_digest_normalises_mapping_order_and_validated_defaults():
    first = _document()
    second = dict(reversed(list(first.items())))
    second = copy.deepcopy(second)
    second["nodes"][0]["kind"] = "worker"  # the implicit kind in the first file
    second["nodes"][0]["args"] = dict(reversed(list(second["nodes"][0]["args"].items())))

    assert topology_fingerprint(build_proposal(first)) == topology_fingerprint(
        build_proposal(second)
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("nodes", 0, "kind"), "other_worker"),
        (("nodes", 0, "args", "path"), "other-src"),
        (("nodes", 0, "args", "origin"), "changed-argument-origin"),
        (("nodes", 0, "args", "proposal_id"), "changed-argument-id"),
        (("nodes", 0, "subgraph", "nodes", 0, "kind"), "other_fetch"),
        (("edges", 0, "target"), "other_worker"),
        (("rationale",), "different reason"),
    ],
)
def test_topology_digest_detects_content_changes(path, value):
    first = _document()
    second = copy.deepcopy(first)
    parent = second
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value

    assert topology_fingerprint(build_proposal(first)) != topology_fingerprint(
        build_proposal(second)
    )
