"""
Subgraph Composition — package groups of nodes as reusable blocks.

A "block" is a subgraph with defined input/output ports that can be
instantiated as a single node in a parent workflow. This enables:
  - Reusable attention heads, FFN blocks, SSM cells
  - Hierarchical architecture design
  - Template-based composition

Usage:
    from aria_designer.runtime.subgraph import extract_block, expand_block, BUILTIN_BLOCKS
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# ── Block Schema ─────────────────────────────────────────────────────


def create_block(
    name: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    input_ports: List[Dict[str, str]],
    output_ports: List[Dict[str, str]],
    params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a block definition from a set of nodes and edges.

    Args:
        name: human-readable block name (e.g., "Attention Head")
        nodes: list of node dicts (from workflow_graph.v1)
        edges: list of edge dicts (internal edges only)
        input_ports: list of {"name": str, "target_node": str, "target_port": str}
        output_ports: list of {"name": str, "source_node": str, "source_port": str}
        params: configurable parameters that get forwarded to inner nodes
        metadata: extra metadata

    Returns:
        Block definition dict
    """
    block_id = f"block_{uuid4().hex[:8]}"
    return {
        "schema_version": "block.v1",
        "block_id": block_id,
        "name": name,
        "nodes": nodes,
        "edges": edges,
        "input_ports": input_ports,
        "output_ports": output_ports,
        "params": params or {},
        "metadata": metadata or {},
    }


def extract_block(
    workflow: Dict[str, Any],
    node_ids: Set[str],
    block_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract a subgraph from a workflow as a reusable block.

    Args:
        workflow: full workflow JSON
        node_ids: set of node IDs to extract
        block_name: name for the new block

    Returns:
        (block_definition, modified_workflow) — the block and the workflow
        with those nodes replaced by a single block node.
    """
    all_nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    inner_nodes = [all_nodes[nid] for nid in node_ids if nid in all_nodes]
    inner_edges, incoming_edges, outgoing_edges, outer_edges = _partition_block_edges(
        workflow.get("edges", []),
        node_ids,
    )
    input_ports = _input_ports_for_edges(incoming_edges)
    output_ports = _output_ports_for_edges(outgoing_edges)
    block = create_block(
        name=block_name,
        nodes=inner_nodes,
        edges=inner_edges,
        input_ports=input_ports,
        output_ports=output_ports,
    )

    block_node = _replacement_block_node(block, inner_nodes)
    modified_wf = _workflow_with_extracted_block(
        workflow,
        node_ids,
        block_node,
        _rewired_block_edges(
            block_node["id"],
            outer_edges,
            incoming_edges,
            outgoing_edges,
        ),
    )
    return block, modified_wf


def _partition_block_edges(edges: List[Dict[str, Any]], node_ids: Set[str]):
    inner_edges = []
    incoming_edges = []
    outgoing_edges = []
    outer_edges = []
    for edge in edges:
        src_in = edge["source"] in node_ids
        tgt_in = edge["target"] in node_ids
        if src_in and tgt_in:
            inner_edges.append(edge)
        elif not src_in and tgt_in:
            incoming_edges.append(edge)
        elif src_in and not tgt_in:
            outgoing_edges.append(edge)
        else:
            outer_edges.append(edge)
    return inner_edges, incoming_edges, outgoing_edges, outer_edges


def _input_ports_for_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "name": f"in_{idx}",
            "target_node": edge["target"],
            "target_port": edge.get("target_port", "in"),
        }
        for idx, edge in enumerate(edges)
    ]


def _output_ports_for_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "name": f"out_{idx}",
            "source_node": edge["source"],
            "source_port": edge.get("source_port", "out"),
        }
        for idx, edge in enumerate(edges)
    ]


def _replacement_block_node(
    block: Dict[str, Any],
    inner_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    block_node_id = f"blk_{uuid4().hex[:6]}"
    return {
        "id": block_node_id,
        "component_type": f"block/{block['block_id']}",
        "params": {},
        "ui_meta": _centroid(inner_nodes),
        "block_ref": block["block_id"],
    }


def _rewired_block_edges(
    block_node_id: str,
    outer_edges: List[Dict[str, Any]],
    incoming_edges: List[Dict[str, Any]],
    outgoing_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rewired_edges = list(outer_edges)
    rewired_edges.extend(
        {
            "id": edge["id"],
            "source": edge["source"],
            "source_port": edge.get("source_port", "out"),
            "target": block_node_id,
            "target_port": f"in_{idx}",
        }
        for idx, edge in enumerate(incoming_edges)
    )
    rewired_edges.extend(
        {
            "id": edge["id"],
            "source": block_node_id,
            "source_port": f"out_{idx}",
            "target": edge["target"],
            "target_port": edge.get("target_port", "in"),
        }
        for idx, edge in enumerate(outgoing_edges)
    )
    return rewired_edges


def _workflow_with_extracted_block(
    workflow: Dict[str, Any],
    node_ids: Set[str],
    block_node: Dict[str, Any],
    rewired_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    remaining_nodes = [node for node in workflow["nodes"] if node["id"] not in node_ids]
    remaining_nodes.append(block_node)
    modified_wf = copy.deepcopy(workflow)
    modified_wf["nodes"] = remaining_nodes
    modified_wf["edges"] = rewired_edges
    return modified_wf


def expand_block(
    workflow: Dict[str, Any],
    block_node_id: str,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    """Expand a block node back into its constituent nodes.

    This is the inverse of extract_block — takes a block reference node
    and replaces it with the full subgraph.

    Args:
        workflow: workflow containing the block node
        block_node_id: ID of the block node to expand
        block: the block definition

    Returns:
        Modified workflow with the block node replaced by its inner nodes.
    """
    wf = copy.deepcopy(workflow)
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}

    block_node = nodes_by_id.get(block_node_id)
    if block_node is None:
        raise ValueError(f"Block node '{block_node_id}' not found")

    # Generate unique prefixed IDs for inner nodes to avoid collisions
    prefix = f"{block_node_id}_"
    id_map = {}
    expanded_nodes = []
    for inner_node in block["nodes"]:
        new_id = f"{prefix}{inner_node['id']}"
        id_map[inner_node["id"]] = new_id
        expanded_node = copy.deepcopy(inner_node)
        expanded_node["id"] = new_id
        expanded_nodes.append(expanded_node)

    # Remap inner edges
    expanded_edges = []
    for inner_edge in block["edges"]:
        new_edge = copy.deepcopy(inner_edge)
        new_edge["id"] = f"{prefix}{inner_edge['id']}"
        new_edge["source"] = id_map.get(inner_edge["source"], inner_edge["source"])
        new_edge["target"] = id_map.get(inner_edge["target"], inner_edge["target"])
        expanded_edges.append(new_edge)

    # Rewire external edges
    new_workflow_edges = []
    for e in wf["edges"]:
        if e["target"] == block_node_id:
            # Find matching input port
            port_name = e.get("target_port", "in_0")
            for ip in block["input_ports"]:
                if ip["name"] == port_name:
                    new_edge = copy.deepcopy(e)
                    new_edge["target"] = id_map.get(
                        ip["target_node"], ip["target_node"]
                    )
                    new_edge["target_port"] = ip["target_port"]
                    new_workflow_edges.append(new_edge)
                    break
            else:
                new_workflow_edges.append(e)  # fallback: keep as-is
        elif e["source"] == block_node_id:
            port_name = e.get("source_port", "out_0")
            for op in block["output_ports"]:
                if op["name"] == port_name:
                    new_edge = copy.deepcopy(e)
                    new_edge["source"] = id_map.get(
                        op["source_node"], op["source_node"]
                    )
                    new_edge["source_port"] = op["source_port"]
                    new_workflow_edges.append(new_edge)
                    break
            else:
                new_workflow_edges.append(e)
        else:
            new_workflow_edges.append(e)

    # Replace block node with expanded nodes
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] != block_node_id] + expanded_nodes
    wf["edges"] = new_workflow_edges + expanded_edges

    return wf


def _centroid(nodes: List[Dict]) -> Dict[str, Any]:
    """Compute average UI position for a set of nodes."""
    xs = [n.get("ui_meta", {}).get("x", 300) for n in nodes]
    ys = [n.get("ui_meta", {}).get("y", 300) for n in nodes]
    if not xs:
        return {"x": 300, "y": 300}
    return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}


# ── Built-in Block Templates ────────────────────────────────────────


def _make_ffn_block(model_dim: int = 256, expansion: int = 4) -> Dict[str, Any]:
    """Standard FFN: Linear(D→4D) → GELU → Linear(4D→D)."""
    inner_dim = model_dim * expansion
    return create_block(
        name=f"FFN Block ({model_dim}→{inner_dim}→{model_dim})",
        nodes=[
            {
                "id": "up",
                "component_type": "linear_proj_up",
                "params": {"out_dim": inner_dim},
            },
            {"id": "act", "component_type": "gelu", "params": {}},
            {
                "id": "down",
                "component_type": "linear_proj_down",
                "params": {"out_dim": model_dim},
            },
        ],
        edges=[
            {"id": "e0", "source": "up", "target": "act"},
            {"id": "e1", "source": "act", "target": "down"},
        ],
        input_ports=[{"name": "in_0", "target_node": "up", "target_port": "in"}],
        output_ports=[{"name": "out_0", "source_node": "down", "source_port": "out"}],
        params={"model_dim": model_dim, "expansion": expansion},
        metadata={
            "category": "blocks",
            "description": "Standard Feed-Forward Network block",
        },
    )


def _make_attention_block(model_dim: int = 256) -> Dict[str, Any]:
    """Self-Attention: Q/K/V projections → matmul → softmax → matmul → out proj."""
    return create_block(
        name=f"Self-Attention ({model_dim}d)",
        nodes=[
            {
                "id": "q",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "k",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "v",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "qk", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "av", "component_type": "matmul", "params": {}},
            {
                "id": "proj",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
        ],
        edges=[
            {"id": "e_qk0", "source": "q", "target": "qk"},
            {"id": "e_qk1", "source": "k", "target": "qk"},
            {"id": "e_sm", "source": "qk", "target": "sm"},
            {"id": "e_av0", "source": "sm", "target": "av"},
            {"id": "e_av1", "source": "v", "target": "av"},
            {"id": "e_proj", "source": "av", "target": "proj"},
        ],
        input_ports=[
            {"name": "in_0", "target_node": "q", "target_port": "in"},
            {"name": "in_1", "target_node": "k", "target_port": "in"},
            {"name": "in_2", "target_node": "v", "target_port": "in"},
        ],
        output_ports=[{"name": "out_0", "source_node": "proj", "source_port": "out"}],
        params={"model_dim": model_dim},
        metadata={"category": "blocks", "description": "Standard self-attention block"},
    )


def _make_transformer_layer(model_dim: int = 256) -> Dict[str, Any]:
    """Full transformer layer: LN → Attn → Residual → LN → FFN → Residual."""
    return create_block(
        name=f"Transformer Layer ({model_dim}d)",
        nodes=[
            {"id": "ln1", "component_type": "rmsnorm", "params": {}},
            {
                "id": "q",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "k",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "v",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "qk", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "av", "component_type": "matmul", "params": {}},
            {
                "id": "attn_proj",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "res1", "component_type": "add", "params": {}},
            {"id": "ln2", "component_type": "rmsnorm", "params": {}},
            {
                "id": "ffn_up",
                "component_type": "linear_proj_up",
                "params": {"out_dim": model_dim * 4},
            },
            {"id": "ffn_act", "component_type": "gelu", "params": {}},
            {
                "id": "ffn_down",
                "component_type": "linear_proj_down",
                "params": {"out_dim": model_dim},
            },
            {"id": "res2", "component_type": "add", "params": {}},
        ],
        edges=[
            {"id": "e_ln1_q", "source": "ln1", "target": "q"},
            {"id": "e_ln1_k", "source": "ln1", "target": "k"},
            {"id": "e_ln1_v", "source": "ln1", "target": "v"},
            {"id": "e_qk0", "source": "q", "target": "qk"},
            {"id": "e_qk1", "source": "k", "target": "qk"},
            {"id": "e_sm", "source": "qk", "target": "sm"},
            {"id": "e_av0", "source": "sm", "target": "av"},
            {"id": "e_av1", "source": "v", "target": "av"},
            {"id": "e_proj", "source": "av", "target": "attn_proj"},
            # res1 gets attn_proj + original input (in_0 wired externally)
            {"id": "e_res1_attn", "source": "attn_proj", "target": "res1"},
            {"id": "e_ln2", "source": "res1", "target": "ln2"},
            {"id": "e_ffn_up", "source": "ln2", "target": "ffn_up"},
            {"id": "e_ffn_act", "source": "ffn_up", "target": "ffn_act"},
            {"id": "e_ffn_down", "source": "ffn_act", "target": "ffn_down"},
            # res2 gets ffn_down + res1
            {"id": "e_res2_ffn", "source": "ffn_down", "target": "res2"},
            {"id": "e_res2_skip", "source": "res1", "target": "res2"},
        ],
        input_ports=[
            {"name": "in_0", "target_node": "ln1", "target_port": "in"},
            {"name": "in_skip", "target_node": "res1", "target_port": "in_1"},
        ],
        output_ports=[{"name": "out_0", "source_node": "res2", "source_port": "out"}],
        params={"model_dim": model_dim},
        metadata={
            "category": "blocks",
            "description": "Full pre-norm transformer layer",
        },
    )


def _make_ssm_block(model_dim: int = 256) -> Dict[str, Any]:
    """SSM-style block: Linear → Selective Scan → Linear."""
    return create_block(
        name=f"SSM Block ({model_dim}d)",
        nodes=[
            {
                "id": "proj_in",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "scan", "component_type": "selective_scan", "params": {}},
            {
                "id": "proj_out",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
        ],
        edges=[
            {"id": "e0", "source": "proj_in", "target": "scan"},
            {"id": "e1", "source": "scan", "target": "proj_out"},
        ],
        input_ports=[{"name": "in_0", "target_node": "proj_in", "target_port": "in"}],
        output_ports=[
            {"name": "out_0", "source_node": "proj_out", "source_port": "out"}
        ],
        params={"model_dim": model_dim},
        metadata={
            "category": "blocks",
            "description": "Selective State Space Model block (Mamba-style)",
        },
    )


def _make_hybrid_layer(model_dim: int = 256) -> Dict[str, Any]:
    """Hybrid Attention+SSM layer: parallel attention and SSM paths merged."""
    return create_block(
        name=f"Hybrid Attn+SSM Layer ({model_dim}d)",
        nodes=[
            {"id": "ln", "component_type": "rmsnorm", "params": {}},
            # Attention path
            {
                "id": "q",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "k",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {
                "id": "v",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "qk", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "av", "component_type": "matmul", "params": {}},
            {
                "id": "attn_proj",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            # SSM path
            {
                "id": "ssm_in",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            {"id": "ssm", "component_type": "selective_scan", "params": {}},
            {
                "id": "ssm_out",
                "component_type": "linear_proj",
                "params": {"out_dim": model_dim},
            },
            # Merge
            {"id": "merge", "component_type": "add", "params": {}},
            # Residual
            {"id": "res", "component_type": "add", "params": {}},
        ],
        edges=[
            {"id": "e_ln_q", "source": "ln", "target": "q"},
            {"id": "e_ln_k", "source": "ln", "target": "k"},
            {"id": "e_ln_v", "source": "ln", "target": "v"},
            {"id": "e_ln_ssm", "source": "ln", "target": "ssm_in"},
            {"id": "e_qk0", "source": "q", "target": "qk"},
            {"id": "e_qk1", "source": "k", "target": "qk"},
            {"id": "e_sm", "source": "qk", "target": "sm"},
            {"id": "e_av0", "source": "sm", "target": "av"},
            {"id": "e_av1", "source": "v", "target": "av"},
            {"id": "e_ap", "source": "av", "target": "attn_proj"},
            {"id": "e_ssm0", "source": "ssm_in", "target": "ssm"},
            {"id": "e_ssm1", "source": "ssm", "target": "ssm_out"},
            {"id": "e_merge_attn", "source": "attn_proj", "target": "merge"},
            {"id": "e_merge_ssm", "source": "ssm_out", "target": "merge"},
            {"id": "e_res_merge", "source": "merge", "target": "res"},
        ],
        input_ports=[
            {"name": "in_0", "target_node": "ln", "target_port": "in"},
            {"name": "in_skip", "target_node": "res", "target_port": "in_1"},
        ],
        output_ports=[{"name": "out_0", "source_node": "res", "source_port": "out"}],
        params={"model_dim": model_dim},
        metadata={
            "category": "blocks",
            "description": "Hybrid layer: parallel attention + SSM paths (targets GPT+Mamba fusion)",
        },
    )


# Registry of built-in blocks
BUILTIN_BLOCKS = {
    "ffn": _make_ffn_block,
    "attention": _make_attention_block,
    "transformer_layer": _make_transformer_layer,
    "ssm": _make_ssm_block,
    "hybrid_attn_ssm": _make_hybrid_layer,
}


def list_builtin_blocks(model_dim: int = 256) -> List[Dict[str, Any]]:
    """List all built-in block templates."""
    return [
        {
            "key": key,
            "block": factory(model_dim),
        }
        for key, factory in BUILTIN_BLOCKS.items()
    ]
