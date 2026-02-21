"""
Survivor Importer: research/ lab notebook → aria-designer workflows.

Reads top-performing architectures (stage1 survivors) from the research
LabNotebook and converts them into editable aria-designer workflow JSON.

This is the reverse of bridge.py (which goes workflow → ComputationGraph).

Usage:
    from runtime.importer import import_survivors, graph_to_workflow

    workflows = import_survivors(n=10, sort_by="loss_ratio")
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

_RESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research"))
if _RESEARCH_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_RESEARCH_ROOT))

from research.synthesis.graph import ComputationGraph, OpNode
from research.synthesis.primitives import PRIMITIVE_REGISTRY

# ── Category mapping (primitive category → aria-designer category) ───

_CATEGORY_MAP = {
    "elementwise_unary": "math",
    "elementwise_binary": "math",
    "reduction": "math",
    "linear_algebra": "linear_algebra",
    "structural": "structural",
    "parameterized": "normalization",  # default; overridden per-op below
    "sequence": "sequence",
    "frequency": "frequency",
    "functional": "functional",
}

_OP_CATEGORY_OVERRIDES = {
    "linear_proj": "mixing",
    "linear_proj_down": "mixing",
    "linear_proj_up": "mixing",
    "fused_linear_gelu": "mixing",
    "rmsnorm": "normalization",
    "learnable_scale": "normalization",
    "learnable_bias": "normalization",
    "selective_scan": "sequence",
    "conv1d_seq": "mixing",
    "topk_gate": "routing",
    "nm_sparse_linear": "mixing",
    "block_sparse_linear": "mixing",
    "semi_structured_2_4_linear": "mixing",
    "multi_head_mix": "mixing",
}


def _get_component_type(op_name: str) -> str:
    """Map a research primitive name to an aria-designer component_type.

    Returns ``category/op_name`` (e.g. ``math/add``, ``mixing/softmax_attention``).
    """
    if op_name in _OP_CATEGORY_OVERRIDES:
        cat = _OP_CATEGORY_OVERRIDES[op_name]
    elif op_name in PRIMITIVE_REGISTRY:
        prim = PRIMITIVE_REGISTRY[op_name]
        cat_val = prim.category.value if hasattr(prim.category, "value") else str(prim.category)
        cat = _CATEGORY_MAP.get(cat_val, "math")
    else:
        cat = "math"
    return f"{cat}/{op_name}"


def _get_input_port_names(op_name: str, n_inputs: int) -> List[str]:
    """Return the port names matching aria-designer component manifests.

    Convention:
      - 1 input:  ["x"]
      - 2 inputs: ["a", "b"]
      - 3+ inputs: ["a", "b", "c", ...]  (rare)
    """
    if n_inputs <= 0:
        return []
    if n_inputs == 1:
        return ["x"]
    # Binary and beyond: a, b, c, ...
    return [chr(ord("a") + i) for i in range(n_inputs)]


# ── Graph → Workflow conversion ──────────────────────────────────────

def graph_to_workflow(
    graph: ComputationGraph,
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a research ComputationGraph to an aria-designer workflow JSON.

    Args:
        graph: ComputationGraph from the research pipeline
        workflow_id: optional ID (generated if not provided)
        name: optional human-readable name
        metadata: optional extra metadata

    Returns:
        workflow_graph.v1 JSON dict
    """
    if workflow_id is None:
        workflow_id = f"imported_{uuid4().hex[:8]}"

    nodes = []
    edges = []

    topo = graph.topological_order()
    layout = _compute_dag_layout(graph, topo)

    # Map computation graph node IDs to aria node IDs
    cg_to_aria: Dict[int, str] = {}
    # Track output port index for multi-output nodes (split2, split3)
    _source_port_counter: Dict[int, int] = {}

    for cg_id in topo:
        node = graph.nodes[cg_id]
        pos = layout.get(cg_id, {"position": {"x": 300, "y": len(nodes) * 120}})

        if node.is_input:
            aria_id = f"input_{cg_id}"
            cg_to_aria[cg_id] = aria_id
            nodes.append({
                "id": aria_id,
                "component_type": "io/input",
                "params": {},
                "ui_meta": pos,
            })
            continue

        aria_id = f"op_{cg_id}_{node.op_name}"
        cg_to_aria[cg_id] = aria_id

        # Build params from config
        params = dict(node.config) if node.config else {}

        nodes.append({
            "id": aria_id,
            "component_type": _get_component_type(node.op_name),
            "params": params,
            "ui_meta": pos,
        })

        # Create edges from input_ids using correct port names
        n_inputs = len(node.input_ids)
        port_names = _get_input_port_names(node.op_name, n_inputs)
        for port_idx, input_cg_id in enumerate(node.input_ids):
            if input_cg_id in cg_to_aria:
                edge_id = f"e_{cg_to_aria[input_cg_id]}_{aria_id}_{port_idx}"
                target_port = port_names[port_idx] if port_idx < len(port_names) else f"x_{port_idx}"

                # Determine source_port — multi-output nodes (split2, split3)
                # use y0, y1, y2; single-output nodes use y.
                src_node = graph.nodes[input_cg_id]
                src_op = src_node.op_name if not src_node.is_input else ""
                if src_op.startswith("split"):
                    out_idx = _source_port_counter.get(input_cg_id, 0)
                    source_port = f"y{out_idx}"
                    _source_port_counter[input_cg_id] = out_idx + 1
                else:
                    source_port = "y"

                edges.append({
                    "id": edge_id,
                    "source": cg_to_aria[input_cg_id],
                    "source_port": source_port,
                    "target": aria_id,
                    "target_port": target_port,
                })

    # Add output node — position one row below the deepest node
    if graph._output_node_id is not None and graph._output_node_id in cg_to_aria:
        output_aria_id = "output_0"
        max_y = max((p["position"]["y"] for p in layout.values()), default=200)
        out_x = layout.get(graph._output_node_id, {"position": {"x": 150}})["position"]["x"]
        nodes.append({
            "id": output_aria_id,
            "component_type": "io/output_head",
            "params": {},
            "ui_meta": {"position": {"x": out_x, "y": max_y + 100}},
        })
        edges.append({
            "id": f"e_{cg_to_aria[graph._output_node_id]}_{output_aria_id}",
            "source": cg_to_aria[graph._output_node_id],
            "source_port": "y",
            "target": output_aria_id,
            "target_port": "x",
        })

    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": workflow_id,
        "name": name or f"Imported Architecture ({graph.n_ops()} ops)",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "model_dim": graph.model_dim,
            "source": "research_import",
            "graph_fingerprint": graph.fingerprint(),
            "n_params_estimate": int(graph.n_params_estimate()),
            **(metadata or {}),
        },
    }


def _compute_dag_layout(
    graph: "ComputationGraph",
    topo: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Compute top-to-bottom DAG layout positions based on topological depth.

    Returns a mapping of cg_id -> {"position": {"x": ..., "y": ...}}.
    """
    X_SPACING = 200
    Y_SPACING = 120
    X_OFFSET = 60
    Y_OFFSET = 40

    # Compute depth: longest path from any input to this node
    depth: Dict[int, int] = {}
    for cg_id in topo:
        node = graph.nodes[cg_id]
        if node.is_input or not node.input_ids:
            depth[cg_id] = 0
        else:
            depth[cg_id] = max(depth.get(iid, 0) for iid in node.input_ids) + 1

    # Group nodes by depth (row)
    by_depth: Dict[int, List[int]] = {}
    for cg_id in topo:
        d = depth[cg_id]
        by_depth.setdefault(d, []).append(cg_id)

    # Assign positions: y by depth (rows), x staggered within row
    positions: Dict[int, Dict[str, Any]] = {}
    max_row_width = max(len(ids) for ids in by_depth.values()) if by_depth else 1
    center_x = X_OFFSET + (max_row_width - 1) * X_SPACING // 2

    for d, ids in by_depth.items():
        row_width = len(ids) * X_SPACING
        x_start = center_x - row_width // 2 + X_SPACING // 2
        for i, cg_id in enumerate(ids):
            positions[cg_id] = {
                "position": {
                    "x": x_start + i * X_SPACING,
                    "y": Y_OFFSET + d * Y_SPACING,
                }
            }

    return positions


# ── Notebook access ──────────────────────────────────────────────────

def _get_notebook():
    """Get a LabNotebook connection to the research database."""
    from research.scientist.notebook import LabNotebook
    db_path = os.path.join(_RESEARCH_ROOT, "lab_notebook.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Lab notebook not found at {db_path}")
    return LabNotebook(db_path)


def import_survivors(
    n: int = 10,
    sort_by: str = "loss_ratio",
    min_novelty: float = 0.0,
) -> List[Dict[str, Any]]:
    """Import top survivors from the research lab notebook as aria-designer workflows.

    Args:
        n: number of survivors to import
        sort_by: ranking metric ("loss_ratio", "novelty_score", "structural_novelty", "behavioral_novelty")
        min_novelty: minimum novelty_score threshold

    Returns:
        List of workflow dicts in workflow_graph.v1 format, enriched with
        research metadata (scores, fingerprint, etc.)
    """
    nb = _get_notebook()

    survivors = nb.get_top_programs(n=n * 2, sort_by=sort_by)  # fetch extra for filtering

    workflows = []
    seen_fingerprints = set()

    for prog in survivors:
        if len(workflows) >= n:
            break

        # Filter by novelty
        nov = prog.get("novelty_score") or 0
        if nov < min_novelty:
            continue

        # Deduplicate by graph fingerprint
        fp = prog.get("graph_fingerprint", "")
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        # Parse graph
        graph_json_str = prog.get("graph_json")
        if not graph_json_str:
            continue

        try:
            from research.synthesis.serializer import graph_from_json
            graph = graph_from_json(graph_json_str)
        except Exception:
            continue

        # Check if all ops are in PRIMITIVE_REGISTRY
        unknown_ops = []
        for node in graph.nodes.values():
            if not node.is_input and node.op_name not in PRIMITIVE_REGISTRY:
                unknown_ops.append(node.op_name)

        # Convert to workflow
        result_id = prog.get("result_id", uuid4().hex[:8])
        wf = graph_to_workflow(
            graph,
            workflow_id=f"survivor_{result_id}",
            name=f"Survivor {result_id} (novelty={nov:.2f})",
            metadata={
                "result_id": result_id,
                "experiment_id": prog.get("experiment_id", ""),
                "novelty_score": nov,
                "structural_novelty": prog.get("structural_novelty", 0),
                "behavioral_novelty": prog.get("behavioral_novelty", 0),
                "loss_ratio": prog.get("loss_ratio", 0),
                "param_count": prog.get("param_count", 0),
                "final_loss": prog.get("final_loss", 0),
                "compatible": len(unknown_ops) == 0,
                "unknown_ops": unknown_ops,
            },
        )
        workflows.append(wf)

    return workflows


def import_single(result_id: str) -> Dict[str, Any]:
    """Import a single program by its result_id.

    Returns:
        Workflow dict in workflow_graph.v1 format.

    Raises:
        ValueError: if the result_id is not found or graph is invalid.
    """
    nb = _get_notebook()
    detail = nb.get_program_detail(result_id)
    if detail is None:
        raise ValueError(f"Program result '{result_id}' not found")

    graph_json_str = detail.get("graph_json")
    if not graph_json_str:
        raise ValueError(f"Program '{result_id}' has no graph_json")

    from research.synthesis.serializer import graph_from_json
    graph = graph_from_json(graph_json_str)

    return graph_to_workflow(
        graph,
        workflow_id=f"imported_{result_id}",
        name=f"Imported {result_id}",
        metadata={
            "result_id": result_id,
            "experiment_id": detail.get("experiment_id", ""),
            "novelty_score": detail.get("novelty_score", 0),
            "loss_ratio": detail.get("loss_ratio", 0),
        },
    )
