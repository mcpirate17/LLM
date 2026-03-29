from __future__ import annotations
from aria_designer.api.app.component_identity import (
    canonicalize_component_id,
    canonicalize_workflow,
    canonicalize_workflow_ids,
    collect_unresolved_component_ids,
    discover_concepts,
)


def test_canonicalize_component_id():
    # Leaf names → canonical
    assert canonicalize_component_id("relu") == "math/relu"
    assert canonicalize_component_id("linear_proj") == "linear_algebra/linear_proj"
    assert canonicalize_component_id("softmax_attention") == "mixing/softmax_attention"
    assert (
        canonicalize_component_id("token_difficulty_proj")
        == "routing/token_difficulty_proj"
    )
    assert canonicalize_component_id("rmsnorm") == "linear_algebra/rmsnorm"
    assert canonicalize_component_id("layernorm") == "normalization/layernorm"

    # Already canonical
    assert canonicalize_component_id("io/input") == "io/input"
    assert canonicalize_component_id("math/relu") == "math/relu"

    # Unknown
    assert canonicalize_component_id("unknown/id") == "unknown/id"
    assert canonicalize_component_id("completely_new_op") == "completely_new_op"


def test_canonicalize_workflow():
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "input"},
            {"id": "n2", "component_type": "token_difficulty_proj"},
            {"id": "n3", "component_type": "linear_proj"},
            {"id": "n4", "component_type": "output_head"},
        ]
    }
    canonicalize_workflow(wf)

    assert wf["nodes"][0]["component_type"] == "io/input"
    assert wf["nodes"][1]["component_type"] == "routing/token_difficulty_proj"
    assert wf["nodes"][2]["component_type"] == "linear_algebra/linear_proj"
    assert wf["nodes"][3]["component_type"] == "io/output_head"


def test_discover_concepts():
    message = "I want a model with softmax_attention and a token_difficulty_proj"
    found = discover_concepts(message)

    types = {f["component_type"] for f in found}
    assert "mixing/softmax_attention" in types
    assert "routing/token_difficulty_proj" in types


def test_canonicalize_idempotent():
    """Canonicalizing an already-canonical workflow is a no-op."""
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "io/input"},
            {"id": "n2", "component_type": "linear_algebra/linear_proj"},
            {"id": "n3", "component_type": "mixing/softmax_attention"},
            {"id": "n4", "component_type": "io/output_head"},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
            {"id": "e3", "source": "n3", "target": "n4"},
        ],
    }
    import copy

    original = copy.deepcopy(wf)
    canonicalize_workflow(wf)
    assert wf["nodes"] == original["nodes"]
    assert wf["edges"] == original["edges"]


def test_canonicalize_mixed_ids():
    """Workflows with mixed bare-leaf and category/id forms all normalize."""
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "input"},
            {"id": "n2", "component_type": "rmsnorm"},
            {"id": "n3", "component_type": "linear_algebra/linear_proj"},
            {"id": "n4", "component_type": "softmax_attention"},
            {"id": "n5", "component_type": "output"},
        ]
    }
    canonicalize_workflow(wf)
    for node in wf["nodes"]:
        assert "/" in node["component_type"], (
            f"Bare leaf ID not resolved: {node['component_type']}"
        )


def test_collect_unresolved():
    """Unresolved IDs are detected after canonicalization."""
    registry = {"io/input", "linear_algebra/linear_proj", "io/output_head"}
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "io/input"},
            {"id": "n2", "component_type": "linear_algebra/linear_proj"},
            {"id": "n3", "component_type": "completely_fake_op"},
            {"id": "n4", "component_type": "io/output_head"},
        ]
    }
    unresolved = collect_unresolved_component_ids(wf, registry)
    assert "completely_fake_op" in unresolved
    assert len(unresolved) == 1


def test_canonicalize_workflow_ids_preserves_raw():
    """preserve_raw_ids stores original IDs in metadata."""
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "input"},
            {"id": "n2", "component_type": "relu"},
        ],
        "metadata": {},
    }
    canonicalize_workflow_ids(wf, preserve_raw_ids=True)
    originals = wf["metadata"].get("original_component_types", {})
    assert "n1" in originals
    assert originals["n1"] == "input"


def test_round_trip_bare_to_canonical():
    """Bare IDs → canonical → re-canonicalize is stable."""
    bare_ids = [
        "relu",
        "softmax_attention",
        "linear_proj",
        "token_difficulty_proj",
        "concat",
    ]
    for bare in bare_ids:
        first = canonicalize_component_id(bare)
        second = canonicalize_component_id(first)
        assert first == second, f"Not idempotent: {bare} → {first} → {second}"
        assert "/" in first, f"Not canonical form: {bare} → {first}"


def test_empty_and_none():
    """Edge cases: empty and falsy inputs."""
    assert canonicalize_component_id("") == ""
    assert canonicalize_component_id("  ") == ""
