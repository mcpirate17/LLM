"""Shared workflow-graph builders for aria_designer tests.

test_bridge / test_profiler / test_perf_regression / test_patcher previously
carried near-identical copies of these dicts, differing only in activation,
out_dim, and whether edges carry explicit port names.
"""

from typing import Any


def chain_edges(node_ids: list[str], *, ports: bool = True) -> list[dict[str, Any]]:
    """Edges for a linear chain node_ids[0] → node_ids[1] → …"""
    edges: list[dict[str, Any]] = []
    for i, (src, dst) in enumerate(zip(node_ids, node_ids[1:])):
        edge: dict[str, Any] = {"id": f"e{i}", "source": src, "target": dst}
        if ports:
            edge["source_port"] = "out"
            edge["target_port"] = "in"
        edges.append(edge)
    return edges


def make_mlp_workflow(
    *, activation: str = "relu", out_dim: int = 256, ports: bool = True
) -> dict[str, Any]:
    """input → linear → activation → linear → output."""
    return {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {
                "id": "n1",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {"id": "n2", "component_type": activation, "params": {}},
            {
                "id": "n3",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {"id": "n4", "component_type": "graph_output", "params": {}},
        ],
        "edges": chain_edges(["n0", "n1", "n2", "n3", "n4"], ports=ports),
    }


_ATTENTION_EDGES = [
    ("in", "q", "out", "in"),
    ("in", "k", "out", "in"),
    ("in", "v", "out", "in"),
    ("q", "attn", "out", "a"),
    ("k", "attn", "out", "b"),
    ("attn", "sm", "out", "in"),
    ("sm", "av", "out", "a"),
    ("v", "av", "out", "b"),
    ("av", "proj", "out", "in"),
    ("proj", "out", "out", "in"),
]


def make_attention_workflow(
    *, out_dim: int = 256, ports: bool = True
) -> dict[str, Any]:
    """input → Q/K/V projections → matmul → softmax → matmul → proj → output."""
    edges: list[dict[str, Any]] = []
    for i, (src, dst, sp, tp) in enumerate(_ATTENTION_EDGES):
        edge: dict[str, Any] = {"id": f"e{i}", "source": src, "target": dst}
        if ports:
            edge["source_port"] = sp
            edge["target_port"] = tp
        edges.append(edge)
    return {
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}},
            {
                "id": "q",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {
                "id": "k",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {
                "id": "v",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {"id": "attn", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "av", "component_type": "matmul", "params": {}},
            {
                "id": "proj",
                "component_type": "linear_proj",
                "params": {"out_dim": out_dim},
            },
            {"id": "out", "component_type": "graph_output", "params": {}},
        ],
        "edges": edges,
    }
