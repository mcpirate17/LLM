"""Tests for subgraph composition (extract, expand, builtin blocks)."""

import pytest

from aria_designer.runtime.subgraph import (
    create_block,
    extract_block,
    expand_block,
    list_builtin_blocks,
    BUILTIN_BLOCKS,
)


# ── Fixtures ─────────────────────────────────────────────────────────

SIMPLE_WORKFLOW = {
    "workflow_id": "test_wf",
    "schema_version": "workflow_graph.v1",
    "name": "Test Workflow",
    "nodes": [
        {
            "id": "in",
            "component_type": "graph_input",
            "params": {},
            "ui_meta": {"x": 100, "y": 100},
        },
        {
            "id": "l1",
            "component_type": "linear_proj",
            "params": {"out_dim": 256},
            "ui_meta": {"x": 200, "y": 100},
        },
        {
            "id": "act",
            "component_type": "gelu",
            "params": {},
            "ui_meta": {"x": 300, "y": 100},
        },
        {
            "id": "l2",
            "component_type": "linear_proj",
            "params": {"out_dim": 256},
            "ui_meta": {"x": 400, "y": 100},
        },
        {
            "id": "out",
            "component_type": "graph_output",
            "params": {},
            "ui_meta": {"x": 500, "y": 100},
        },
    ],
    "edges": [
        {"id": "e0", "source": "in", "target": "l1"},
        {"id": "e1", "source": "l1", "target": "act"},
        {"id": "e2", "source": "act", "target": "l2"},
        {"id": "e3", "source": "l2", "target": "out"},
    ],
}


# ── create_block ─────────────────────────────────────────────────────


def test_create_block_schema():
    block = create_block(
        name="Test Block",
        nodes=[{"id": "n1", "component_type": "gelu", "params": {}}],
        edges=[],
        input_ports=[{"name": "in_0", "target_node": "n1", "target_port": "in"}],
        output_ports=[{"name": "out_0", "source_node": "n1", "source_port": "out"}],
    )
    assert block["schema_version"] == "block.v1"
    assert block["name"] == "Test Block"
    assert block["block_id"].startswith("block_")
    assert len(block["nodes"]) == 1
    assert len(block["input_ports"]) == 1
    assert len(block["output_ports"]) == 1


def test_create_block_with_params_and_metadata():
    block = create_block(
        name="Param Block",
        nodes=[],
        edges=[],
        input_ports=[],
        output_ports=[],
        params={"dim": 512},
        metadata={"author": "test"},
    )
    assert block["params"]["dim"] == 512
    assert block["metadata"]["author"] == "test"


# ── extract_block ────────────────────────────────────────────────────


def test_extract_block_basic():
    """Extract l1+act as a block from simple workflow."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "FFN Part")

    # Block should contain the extracted nodes
    assert block["name"] == "FFN Part"
    block_node_ids = {n["id"] for n in block["nodes"]}
    assert block_node_ids == {"l1", "act"}

    # Block should have 1 internal edge (l1→act)
    assert len(block["edges"]) == 1
    assert block["edges"][0]["source"] == "l1"
    assert block["edges"][0]["target"] == "act"

    # Block should have 1 input port (from in→l1) and 1 output port (act→l2)
    assert len(block["input_ports"]) == 1
    assert len(block["output_ports"]) == 1


def test_extract_block_modified_workflow():
    """Modified workflow should have a block node replacing the extracted nodes."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "FFN Part")

    modified_node_ids = {n["id"] for n in modified_wf["nodes"]}
    # Original had {in, l1, act, l2, out} → now {in, l2, out, blk_XXXX}
    assert "l1" not in modified_node_ids
    assert "act" not in modified_node_ids
    assert "in" in modified_node_ids
    assert "l2" in modified_node_ids
    assert "out" in modified_node_ids

    # Should have one block node
    block_nodes = [n for n in modified_wf["nodes"] if n.get("block_ref")]
    assert len(block_nodes) == 1
    assert block_nodes[0]["block_ref"] == block["block_id"]


def test_extract_block_edge_count():
    """Modified workflow should have correct number of edges."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "Test")

    # Original: in→l1, l1→act, act→l2, l2→out (4 edges)
    # After: in→blk, blk→l2, l2→out (3 edges)
    assert len(modified_wf["edges"]) == 3


def test_extract_single_node():
    """Extract a single node as a block."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"act"}, "Single Op")
    assert len(block["nodes"]) == 1
    assert len(block["edges"]) == 0
    assert len(block["input_ports"]) == 1
    assert len(block["output_ports"]) == 1


# ── expand_block ─────────────────────────────────────────────────────


def test_expand_block_roundtrip():
    """extract → expand should restore the graph structure."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "FFN Part")

    # Find the block node
    block_node = [n for n in modified_wf["nodes"] if n.get("block_ref")][0]

    # Expand it back
    restored = expand_block(modified_wf, block_node["id"], block)

    # Should have same number of nodes as original (with prefixed IDs)
    assert len(restored["nodes"]) == len(SIMPLE_WORKFLOW["nodes"])

    # All component types should be preserved
    original_types = sorted(n["component_type"] for n in SIMPLE_WORKFLOW["nodes"])
    restored_types = sorted(n["component_type"] for n in restored["nodes"])
    assert original_types == restored_types


def test_expand_block_prefixes_ids():
    """Expanded nodes should have prefixed IDs to avoid collisions."""
    block, modified_wf = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "FFN Part")
    block_node = [n for n in modified_wf["nodes"] if n.get("block_ref")][0]

    restored = expand_block(modified_wf, block_node["id"], block)

    # The expanded nodes should have block_node_id prefix
    expanded_ids = {n["id"] for n in restored["nodes"]}
    prefix = f"{block_node['id']}_"
    prefixed_nodes = [nid for nid in expanded_ids if nid.startswith(prefix)]
    assert len(prefixed_nodes) == 2  # l1 and act were extracted


def test_expand_nonexistent_block_raises():
    """Expanding a nonexistent block node should raise ValueError."""
    block = create_block("Empty", [], [], [], [])
    with pytest.raises(ValueError, match="not found"):
        expand_block(SIMPLE_WORKFLOW, "nonexistent_id", block)


# ── Built-in blocks ──────────────────────────────────────────────────


def test_builtin_block_registry():
    """All expected builtin blocks should be registered."""
    expected = {"ffn", "attention", "transformer_layer", "ssm", "hybrid_attn_ssm"}
    assert set(BUILTIN_BLOCKS.keys()) == expected


def test_list_builtin_blocks():
    """list_builtin_blocks should return all blocks with correct structure."""
    blocks = list_builtin_blocks(model_dim=128)
    assert len(blocks) == len(BUILTIN_BLOCKS)
    for item in blocks:
        assert "key" in item
        assert "block" in item
        block = item["block"]
        assert block["schema_version"] == "block.v1"
        assert len(block["nodes"]) > 0
        assert len(block["input_ports"]) > 0
        assert len(block["output_ports"]) > 0


def test_ffn_block():
    block = BUILTIN_BLOCKS["ffn"](model_dim=128, expansion=4)
    assert len(block["nodes"]) == 3  # up, act, down
    assert len(block["edges"]) == 2
    # Check dimensions
    up_node = [n for n in block["nodes"] if n["id"] == "up"][0]
    assert up_node["params"]["out_dim"] == 512  # 128 * 4
    down_node = [n for n in block["nodes"] if n["id"] == "down"][0]
    assert down_node["params"]["out_dim"] == 128


def test_attention_block():
    block = BUILTIN_BLOCKS["attention"](model_dim=256)
    assert len(block["nodes"]) == 7  # q, k, v, qk, sm, av, proj
    assert len(block["edges"]) == 6
    assert len(block["input_ports"]) == 3  # q, k, v inputs
    assert len(block["output_ports"]) == 1


def test_transformer_layer_block():
    block = BUILTIN_BLOCKS["transformer_layer"](model_dim=256)
    assert len(block["nodes"]) == 14  # LN + attn + res + LN + FFN + res
    # Has both main and skip input ports
    port_names = {p["name"] for p in block["input_ports"]}
    assert "in_0" in port_names
    assert "in_skip" in port_names


def test_ssm_block():
    block = BUILTIN_BLOCKS["ssm"](model_dim=256)
    assert len(block["nodes"]) == 3  # proj_in, scan, proj_out
    node_types = {n["component_type"] for n in block["nodes"]}
    assert "selective_scan" in node_types


def test_hybrid_layer_block():
    block = BUILTIN_BLOCKS["hybrid_attn_ssm"](model_dim=256)
    node_types = {n["component_type"] for n in block["nodes"]}
    # Should have both attention and SSM components
    assert "matmul" in node_types  # attention path
    assert "selective_scan" in node_types  # SSM path
    assert "add" in node_types  # merge + residual


def test_builtin_blocks_different_dims():
    """Builtin blocks should respect model_dim parameter."""
    small = BUILTIN_BLOCKS["ffn"](model_dim=64)
    large = BUILTIN_BLOCKS["ffn"](model_dim=512)
    small_up = [n for n in small["nodes"] if n["id"] == "up"][0]
    large_up = [n for n in large["nodes"] if n["id"] == "up"][0]
    assert small_up["params"]["out_dim"] == 256  # 64 * 4
    assert large_up["params"]["out_dim"] == 2048  # 512 * 4


# ── Edge cases ────────────────────────────────────────────────────────


def test_extract_all_inner_nodes():
    """Extracting all non-IO nodes should work."""
    block, modified_wf = extract_block(
        SIMPLE_WORKFLOW, {"l1", "act", "l2"}, "All Inner"
    )
    assert len(block["nodes"]) == 3
    # Modified workflow should have: in, out, block_node
    assert len(modified_wf["nodes"]) == 3


def test_double_extract_expand():
    """Extract then expand twice should be stable."""
    block1, wf1 = extract_block(SIMPLE_WORKFLOW, {"l1", "act"}, "Block1")
    blk_node1 = [n for n in wf1["nodes"] if n.get("block_ref")][0]
    restored1 = expand_block(wf1, blk_node1["id"], block1)

    # Do it again on the restored workflow
    restored_inner_ids = set()
    for n in restored1["nodes"]:
        if n["component_type"] in ("linear_proj", "gelu"):
            if "act" in n["id"] or (
                "l1" in n["id"] and n["component_type"] == "linear_proj"
            ):
                restored_inner_ids.add(n["id"])

    if len(restored_inner_ids) >= 2:
        block2, wf2 = extract_block(restored1, restored_inner_ids, "Block2")
        assert len(block2["nodes"]) == len(restored_inner_ids)
