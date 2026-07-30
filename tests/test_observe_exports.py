"""Package-level observe re-exports that must stay public."""

from grapharc.observe import by_node, tokens_by_model


def test_by_node_and_tokens_by_model_are_exported_from_package_root():
    """Callers should not need to reach into grapharc.observe.cost (#1)."""
    assert callable(by_node)
    assert callable(tokens_by_model)
