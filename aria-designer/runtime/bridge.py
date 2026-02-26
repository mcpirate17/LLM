"""
Runtime Bridge: aria-designer workflow → research/ eval pipeline.

Converts aria-designer WorkflowGraphModel JSON into research ComputationGraph,
then drives compilation, sandbox evaluation, fingerprinting, and novelty scoring.

Usage:
    from runtime.bridge import evaluate_workflow, workflow_to_graph

    result = evaluate_workflow(workflow_json, model_dim=256, device="cuda")
"""

from __future__ import annotations

import copy
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Ensure research/ is importable
_RESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research"))
if _RESEARCH_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_RESEARCH_ROOT))

from research.synthesis.graph import ComputationGraph, ShapeInfo
from research.synthesis.primitives import PRIMITIVE_REGISTRY, get_primitive
from research.mathspaces.registry import register_all_mathspaces

# Ensure mathspace primitives are available for resolution.
register_all_mathspaces()


# ── Component ID → Primitive Name mapping ────────────────────────────

# aria-designer component_type can be "relu", "math/relu", etc.
# We strip the category prefix and map to PRIMITIVE_REGISTRY keys.
# Most IDs match 1:1; a few need aliasing.

_DEFAULT_COMPONENT_ALIASES = {
    # aria-designer ID → research primitive name
    "relu_op": "relu",
    "gelu_op": "gelu",
    "silu_op": "silu",
    "linear": "linear_proj",
    "linear_down": "linear_proj_down",
    "linear_up": "linear_proj_up",
    # SSM / normalization components used in examples
    "state_space": "selective_scan",
    "rmsnorm_pre": "rmsnorm",
    "layernorm_pre": "layernorm",
    # Transformer components
    "softmax_attention": "softmax_attention",
    "swiglu_mlp": "swiglu_mlp",
}

# IO components that don't map to primitives
_IO_COMPONENTS = {"graph_input", "graph_output", "input", "output", "output_head"}

_MAPPING_FILE = Path(__file__).resolve().parent / "component_mapping.yaml"
_MAPPING_CONFIG: Dict[str, Any] = {}

if _MAPPING_FILE.exists():
    try:
        _MAPPING_CONFIG = yaml.safe_load(_MAPPING_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        _MAPPING_CONFIG = {}

_COMPONENT_ALIASES = {
    **_DEFAULT_COMPONENT_ALIASES,
    **(_MAPPING_CONFIG.get("aliases") or {}),
}
_CATEGORY_EXECUTION_CLASS = _MAPPING_CONFIG.get("category_execution_class") or {}
_COMPONENT_EXECUTION_CLASS = _MAPPING_CONFIG.get("component_execution_class") or {}
_PASSTHROUGH_COMPONENTS = set(_MAPPING_CONFIG.get("passthrough_components") or [])
_CONTROL_PASSTHROUGH_COMPONENTS = {"loop"}
_SOURCE_COMPONENTS = set(_MAPPING_CONFIG.get("source_components") or [])
_TEMPLATE_LOWERED_COMPONENTS = set(_MAPPING_CONFIG.get("template_lowered_components") or [])
_APPROXIMATE_ALIAS_NOTES = _MAPPING_CONFIG.get("approximate_alias_notes") or {}

_DATA_SOURCE_COMPONENTS = {
    "csv_reader",
    "file_loader",
    "binary_file_reader",
    "hf_dataset_loader",
    "random_data_source",
    "synthetic_data_source",
}

_DATA_TRANSFORM_COMPONENTS = {
    "filter",
    "dataset_filter",
    "dataset_map",
    "select_columns",
    "split_train_val_test",
}

_DATA_SINK_COMPONENTS = {
    "file_writer",
}


def _execution_class(component_leaf: str, category: Optional[str]) -> str:
    if component_leaf in _COMPONENT_EXECUTION_CLASS:
        return str(_COMPONENT_EXECUTION_CLASS[component_leaf])
    if category and category in _CATEGORY_EXECUTION_CLASS:
        return str(_CATEGORY_EXECUTION_CLASS[category])
    return "primitive"


def _is_passthrough_component(component_leaf: str) -> bool:
    return component_leaf in _PASSTHROUGH_COMPONENTS or component_leaf in _CONTROL_PASSTHROUGH_COMPONENTS


def _is_source_component(component_leaf: str) -> bool:
    return component_leaf in _SOURCE_COMPONENTS


def _is_data_source_component(component_leaf: str) -> bool:
    return component_leaf in _DATA_SOURCE_COMPONENTS


def _is_data_transform_component(component_leaf: str) -> bool:
    return component_leaf in _DATA_TRANSFORM_COMPONENTS


def _is_data_sink_component(component_leaf: str) -> bool:
    return component_leaf in _DATA_SINK_COMPONENTS


def _is_template_lowered_component(component_leaf: str) -> bool:
    return component_leaf in _TEMPLATE_LOWERED_COMPONENTS


def _alias_semantic_info(component_leaf: str, primitive_name: str) -> Dict[str, Any]:
    """Return semantic fidelity metadata for alias mappings."""
    if component_leaf == primitive_name:
        return {"semantic_fidelity": "exact", "warnings": []}
    note = _APPROXIMATE_ALIAS_NOTES.get(component_leaf)
    if note:
        return {"semantic_fidelity": "approximate", "warnings": [str(note)]}
    return {"semantic_fidelity": "exact", "warnings": []}


def _lower_template_component(
    graph: ComputationGraph,
    component_leaf: str,
    input_cg_ids: List[int],
    model_dim: int,
) -> int:
    """Lower high-level block node IDs into explicit primitive subgraphs."""
    if not input_cg_ids:
        raise ValueError(f"Template-lowered component '{component_leaf}' requires at least one input")
    x = input_cg_ids[0]
    d = int(model_dim)

    if component_leaf in {"u_net", "hourglass"}:
        down = graph.add_op("linear_proj_down", [x], {"out_dim": d})
        mid = graph.add_op("gelu", [down], {})
        up = graph.add_op("linear_proj_up", [mid], {"out_dim": d})
        return up

    if component_leaf == "dense_net":
        proj = graph.add_op("linear_proj", [x], {"out_dim": d})
        act = graph.add_op("relu", [proj], {})
        return graph.add_op("add", [x, act], {})

    if component_leaf == "fractal":
        a = graph.add_op("relu", [x], {})
        b = graph.add_op("gelu", [x], {})
        return graph.add_op("add", [a, b], {})

    if component_leaf == "parallel_streams":
        a = graph.add_op("relu", [x], {})
        b = graph.add_op("silu", [x], {})
        return graph.add_op("add", [a, b], {})

    if component_leaf == "feedback_loop":
        proj = graph.add_op("linear_proj", [x], {"out_dim": d})
        gate = graph.add_op("tanh", [proj], {})
        return graph.add_op("add", [x, gate], {})

    if component_leaf == "mixture_of_paths":
        p1 = graph.add_op("relu", [x], {})
        p2 = graph.add_op("gelu", [x], {})
        mix = graph.add_op("add", [p1, p2], {})
        return graph.add_op("linear_proj", [mix], {"out_dim": d})

    raise ValueError(f"No template-lowering implementation for '{component_leaf}'")


def _resolve_primitive(component_type: str) -> Optional[str]:
    """Resolve an aria-designer component_type to a research primitive name.

    Returns None for IO nodes (input/output).
    Raises ValueError for unknown components.
    """
    # Strip category prefix: "math/relu" → "relu"
    parts = component_type.split("/")
    cid = parts[-1]
    category = parts[0] if len(parts) > 1 else None

    if cid in _IO_COMPONENTS:
        return None

    # Direct match
    if cid in PRIMITIVE_REGISTRY:
        return cid

    # Alias lookup
    if cid in _COMPONENT_ALIASES:
        return _COMPONENT_ALIASES[cid]

    if _is_source_component(cid):
        raise ValueError(
            f"Component '{component_type}' is configured as source-lowered "
            "and should be handled before primitive resolution."
        )

    if _is_passthrough_component(cid):
        raise ValueError(
            f"Component '{component_type}' is configured as passthrough-lowered "
            "and should be handled before primitive resolution."
        )

    if _is_template_lowered_component(cid):
        raise ValueError(
            f"Component '{component_type}' is configured as template-lowered "
            "and should be handled before primitive resolution."
        )

    exec_class = _execution_class(cid, category)
    if exec_class != "primitive":
        raise ValueError(
            f"Unsupported component '{component_type}' for research primitive bridge: "
            f"execution_class={exec_class}. Add lowering/expansion path before evaluation."
        )

    raise ValueError(
        f"Unknown component '{component_type}': not in PRIMITIVE_REGISTRY "
        f"and no alias defined. Available primitives: {sorted(PRIMITIVE_REGISTRY.keys())[:10]}..."
    )


def get_component_execution_capability(component_type: str) -> Dict[str, Any]:
    """Return bridge execution capability metadata for a component type."""
    parts = component_type.split("/")
    cid = parts[-1]
    category = parts[0] if len(parts) > 1 else None

    if cid in _IO_COMPONENTS:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "io",
            "execution_class": "io",
            "semantic_fidelity": "exact",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": [],
            "reason": "IO passthrough node.",
        }

    if _is_data_source_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "source",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Data-source lowered to deterministic graph input in bridge mode."],
            "reason": "Supported via data-source lowering in bridge (deterministic eval fallback).",
        }

    if _is_data_transform_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "passthrough",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Data-transform is currently shape-safe passthrough in bridge mode."],
            "reason": "Supported via shape-safe data-transform lowering in bridge.",
        }

    if _is_data_sink_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "passthrough",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Data sink is non-materialized passthrough during bridge evaluation."],
            "reason": "Supported as terminal/observability sink with passthrough gradient path.",
        }

    if _is_template_lowered_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "template",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Template component expands to an approximate primitive subgraph."],
            "reason": "Supported via template lowering (expanded primitive subgraph).",
        }

    if cid in PRIMITIVE_REGISTRY:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "direct",
            "execution_class": "primitive",
            "semantic_fidelity": "exact",
            "bridge_supported": True,
            "primitive_name": cid,
            "warnings": [],
            "reason": "Direct primitive mapping.",
        }

    if cid in _COMPONENT_ALIASES:
        alias_target = _COMPONENT_ALIASES[cid]
        semantic = _alias_semantic_info(cid, alias_target)
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "alias",
            "execution_class": "primitive",
            "semantic_fidelity": semantic["semantic_fidelity"],
            "bridge_supported": True,
            "primitive_name": alias_target,
            "warnings": semantic["warnings"],
            "reason": "Mapped via component alias.",
        }

    if _is_source_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "source",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Source component lowered to graph input in bridge mode."],
            "reason": "Supported via source lowering (graph input seed).",
        }

    if _is_passthrough_component(cid):
        reason = (
            "Supported via control-flow passthrough lowering in bridge."
            if cid in _CONTROL_PASSTHROUGH_COMPONENTS
            else "Supported via passthrough lowering (wire-through)."
        )
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "passthrough",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Component is currently wire-through passthrough in bridge mode."],
            "reason": reason,
        }

    exec_class = _execution_class(cid, category)
    return {
        "component_type": component_type,
        "component_leaf": cid,
        "mapping_kind": "unsupported",
        "execution_class": exec_class,
        "semantic_fidelity": "unsupported",
        "bridge_supported": False,
        "primitive_name": None,
        "warnings": [],
        "reason": (
            "No primitive mapping. Requires lowering/expansion path."
            if exec_class in {"composite", "data_control", "control"}
            else "No primitive mapping registered."
        ),
    }


# ── Workflow → ComputationGraph conversion ───────────────────────────

def workflow_to_graph(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
    return_id_map: bool = False,
) -> ComputationGraph:
    """Convert an aria-designer workflow JSON to a research ComputationGraph.

    The workflow_json should follow the workflow_graph.v1 schema:
    {
        "nodes": [{"id": "n1", "component_type": "relu", "params": {...}}, ...],
        "edges": [{"source": "n1", "target": "n2", "source_port": "out", "target_port": "in"}, ...],
        ...
    }

    Returns a ComputationGraph ready for compilation and evaluation.
    """
    graph = ComputationGraph(model_dim=model_dim)

    nodes = workflow_json.get("nodes", [])
    edges = workflow_json.get("edges", [])

    if not nodes:
        raise ValueError("Workflow has no nodes")

    # Build adjacency info
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}

    # Find nodes with incoming edges (targets)
    targets_set = {e["target"] for e in edges}
    sources_set = {e["source"] for e in edges}

    # Identify input nodes: either explicitly typed as input/graph_input,
    # or nodes with no incoming edges
    input_node_ids = []
    output_node_ids = []
    for n in nodes:
        cid = n["component_type"].split("/")[-1]
        if cid in ("graph_input", "input"):
            input_node_ids.append(n["id"])
        elif cid in ("graph_output", "output", "output_head"):
            output_node_ids.append(n["id"])

    # If no explicit input nodes, infer from topology (nodes with no incoming edges)
    if not input_node_ids:
        input_node_ids = [n["id"] for n in nodes if n["id"] not in targets_set]

    # If no explicit output nodes, infer (nodes with no outgoing edges)
    if not output_node_ids:
        output_node_ids = [n["id"] for n in nodes if n["id"] not in sources_set]

    if not input_node_ids:
        raise ValueError("Workflow has no input nodes (no source nodes found)")
    if not output_node_ids:
        raise ValueError("Workflow has no output nodes (no sink nodes found)")

    # Build edge map: target_id → [(source_id, source_port, target_port)]
    incoming: Dict[str, List[Tuple[str, str, str]]] = {n["id"]: [] for n in nodes}
    for e in edges:
        incoming[e["target"]].append((e["source"], e.get("source_port", "out"), e.get("target_port", "in")))

    # Topological sort via Kahn's algorithm
    in_degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        in_degree[e["target"]] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        nid = queue.pop(0)
        topo_order.append(nid)
        for e in edges:
            if e["source"] == nid:
                in_degree[e["target"]] -= 1
                if in_degree[e["target"]] == 0:
                    queue.append(e["target"])

    if len(topo_order) != len(nodes):
        raise ValueError("Workflow contains a cycle")

    # Map aria node IDs → ComputationGraph node IDs
    aria_to_cg: Dict[str, int] = {}

    for aria_id in topo_order:
        node_cfg = node_by_id[aria_id]
        cid = node_cfg["component_type"].split("/")[-1]
        params = node_cfg.get("params", {})

        if cid in ("graph_input", "input") or (aria_id in input_node_ids and cid in _IO_COMPONENTS):
            # Add as graph input
            cg_id = graph.add_input()
            aria_to_cg[aria_id] = cg_id
            continue

        if _is_source_component(cid):
            # Data-source nodes become explicit graph inputs in eval bridge mode.
            cg_id = graph.add_input()
            aria_to_cg[aria_id] = cg_id
            continue

        if cid in ("graph_output", "output", "output_head"):
            # Output node: just wire through from its input
            inc = incoming.get(aria_id, [])
            if inc:
                src_aria_id = inc[0][0]
                if src_aria_id in aria_to_cg:
                    aria_to_cg[aria_id] = aria_to_cg[src_aria_id]
            continue

        if _is_data_source_component(cid):
            cg_id = graph.add_input()
            aria_to_cg[aria_id] = cg_id
            continue

        # Gather input node IDs from edges
        inc = incoming.get(aria_id, [])
        if _is_template_lowered_component(cid):
            input_cg_ids = [aria_to_cg[src] for src, _sp, _tp in inc if src in aria_to_cg]
            if not input_cg_ids and input_node_ids and input_node_ids[0] in aria_to_cg:
                input_cg_ids = [aria_to_cg[input_node_ids[0]]]
            try:
                aria_to_cg[aria_id] = _lower_template_component(
                    graph, cid, input_cg_ids, model_dim
                )
            except ValueError as e:
                raise ValueError(
                    f"Template lowering error at node '{aria_id}' ({cid}): {e}"
                ) from e
            continue

        if _is_data_transform_component(cid) or _is_data_sink_component(cid):
            if inc:
                src_aria_id = inc[0][0]
                if src_aria_id not in aria_to_cg:
                    raise ValueError(
                        f"Data-plane lowered component '{node_cfg['component_type']}' "
                        f"references unresolved source node '{src_aria_id}'."
                    )
                aria_to_cg[aria_id] = aria_to_cg[src_aria_id]
            else:
                if input_node_ids and input_node_ids[0] in aria_to_cg:
                    aria_to_cg[aria_id] = aria_to_cg[input_node_ids[0]]
                else:
                    cg_id = graph.add_input()
                    aria_to_cg[aria_id] = cg_id
            continue

        if _is_passthrough_component(cid):
            if not inc:
                raise ValueError(
                    f"Passthrough-lowered component '{node_cfg['component_type']}' "
                    "requires an incoming edge."
                )
            src_aria_id = inc[0][0]
            if src_aria_id not in aria_to_cg:
                raise ValueError(
                    f"Passthrough-lowered component '{node_cfg['component_type']}' "
                    f"references unresolved source node '{src_aria_id}'."
                )
            aria_to_cg[aria_id] = aria_to_cg[src_aria_id]
            continue

        prim_name = _resolve_primitive(node_cfg["component_type"])
        if prim_name is None:
            continue

        prim = get_primitive(prim_name)

        input_cg_ids = []
        for src_aria_id, _sp, _tp in inc:
            if src_aria_id in aria_to_cg:
                input_cg_ids.append(aria_to_cg[src_aria_id])

        # If node has no connected inputs but needs them, use last input node
        if not input_cg_ids and prim.n_inputs >= 1:
            if input_node_ids and input_node_ids[0] in aria_to_cg:
                input_cg_ids = [aria_to_cg[input_node_ids[0]]]

        # For binary ops that only have one input connected, duplicate it
        if prim.n_inputs == 2 and len(input_cg_ids) == 1:
            input_cg_ids = [input_cg_ids[0], input_cg_ids[0]]

        # Build config from params
        config = {}
        for key in prim.config_keys:
            if key in params:
                config[key] = params[key]

        # Auto-set out_dim for linear ops if not specified
        if "out_dim" in prim.config_keys and "out_dim" not in config:
            config["out_dim"] = model_dim

        try:
            cg_id = graph.add_op(prim_name, input_cg_ids, config)
            aria_to_cg[aria_id] = cg_id
        except ValueError as e:
            raise ValueError(
                f"Shape error at node '{aria_id}' ({prim_name}): {e}"
            ) from e

    # Set output
    for out_id in output_node_ids:
        if out_id in aria_to_cg:
            try:
                graph.set_output(aria_to_cg[out_id])
                break
            except ValueError:
                continue
    else:
        # Try last node in topo order
        for aria_id in reversed(topo_order):
            if aria_id in aria_to_cg:
                try:
                    graph.set_output(aria_to_cg[aria_id])
                    break
                except ValueError:
                    continue
        else:
            raise ValueError(
                "Could not set graph output: no node produces (B, S, model_dim) output"
            )

    if return_id_map:
        return graph, aria_to_cg
    return graph


# ── Compression / Efficiency Analysis ─────────────────────────────────

# Sparse ops and their effective density factors (fraction of weights actually used)
_SPARSE_OP_DENSITY = {
    "nm_sparse_linear": 0.50,        # 2:4 sparsity → 50% density
    "block_sparse_linear": 0.25,     # default block_density
    "semi_structured_2_4_linear": 0.50,
}


@dataclass
class CompressionResult:
    """Result of compression & efficiency analysis."""
    # Pruning curve: list of {sparsity, loss, loss_ratio}
    pruning_curve: List[Dict[str, float]] = field(default_factory=list)
    baseline_loss: float = 0.0
    pruning_tolerance: float = 0.0  # 0-1, how gracefully it degrades

    # Param compression
    dense_params: int = 0
    effective_params: int = 0
    compression_ratio: float = 1.0  # dense / effective

    # Sparse op coverage
    sparse_ops: int = 0
    total_ops: int = 0
    sparse_op_coverage: float = 0.0
    sparse_op_names: List[str] = field(default_factory=list)

    # Theoretical sizes
    theoretical_size_fp16_mb: float = 0.0
    theoretical_size_int8_mb: float = 0.0
    theoretical_size_int4_mb: float = 0.0

    # Composite scores
    memory_efficiency_score: float = 0.0  # 0-1, smaller → higher
    efficiency_score: float = 0.0         # composite 0-1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        return d


def analyze_compression(
    model,
    graph,
    vocab_size: int = 32000,
    device: str = "cpu",
    batch_size: int = 2,
    seq_len: int = 64,
) -> CompressionResult:
    """Analyze compression characteristics: pruning tolerance, sparse coverage, sizes."""
    import torch

    result = CompressionResult()

    # --- 1. Walk graph → sparse op coverage & effective params ---
    sparse_names = []
    total_op_count = 0
    effective_param_count = 0

    for node in graph.nodes.values():
        if node.is_input:
            continue
        total_op_count += 1
        op_name = node.op_name
        if op_name in _SPARSE_OP_DENSITY:
            sparse_names.append(op_name)
            density = node.config.get("block_density", _SPARSE_OP_DENSITY[op_name])
        else:
            density = 1.0

        # Estimate params for this op
        if op_name in PRIMITIVE_REGISTRY:
            prim = PRIMITIVE_REGISTRY[op_name]
            if prim.has_params:
                D = graph.model_dim
                out_dim = node.config.get("out_dim", D)
                op_params = D * out_dim  # rough estimate
                effective_param_count += int(op_params * density)

    # Get dense param count from model
    dense_params = sum(p.numel() for p in model.parameters())
    if effective_param_count == 0:
        effective_param_count = dense_params

    result.dense_params = dense_params
    result.effective_params = effective_param_count
    result.compression_ratio = dense_params / max(effective_param_count, 1)
    result.sparse_ops = len(sparse_names)
    result.total_ops = total_op_count
    result.sparse_op_coverage = len(sparse_names) / max(total_op_count, 1)
    result.sparse_op_names = sorted(set(sparse_names))

    # --- 2. Theoretical sizes ---
    result.theoretical_size_fp16_mb = (dense_params * 2) / (1024 * 1024)
    result.theoretical_size_int8_mb = (dense_params * 1) / (1024 * 1024)
    result.theoretical_size_int4_mb = (dense_params * 0.5) / (1024 * 1024)

    # --- 3. Pruning curve (one-shot at 4 sparsity levels) ---
    try:
        from research.eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss

        dev = torch.device(device)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

        # Baseline loss
        baseline = estimate_lm_ce_loss(model, [input_ids], dev)
        if baseline is not None and baseline > 0:
            result.baseline_loss = baseline
            sparsity_levels = [0.25, 0.50, 0.75, 0.90]
            curve = []

            for sp in sparsity_levels:
                t0 = time.monotonic()
                try:
                    pruned = copy.deepcopy(model)
                    pruned.to(dev)
                    apply_one_shot_pruning(pruned, target_sparsity=sp)
                    loss = estimate_lm_ce_loss(pruned, [input_ids], dev)
                    del pruned
                    if loss is not None:
                        ratio = loss / max(baseline, 1e-8)
                        curve.append({"sparsity": sp, "loss": round(loss, 4), "loss_ratio": round(ratio, 4)})
                except Exception:
                    pass
                # Time-box: skip remaining levels if this one exceeded 5s
                if (time.monotonic() - t0) > 5.0:
                    break

            result.pruning_curve = curve

            # Pruning tolerance: inverse of average loss ratio degradation
            if curve:
                avg_ratio = sum(p["loss_ratio"] for p in curve) / len(curve)
                # tolerance=1 means no degradation, tolerance=0 means 2x+ degradation
                result.pruning_tolerance = max(0.0, min(1.0, 2.0 - avg_ratio))
    except Exception:
        pass

    # --- 4. Memory efficiency score (smaller models → higher) ---
    # Scale: 0 at 100M+ params, 1 at <1M params (log scale)
    import math
    if dense_params > 0:
        log_p = math.log10(max(dense_params, 1))
        result.memory_efficiency_score = max(0.0, min(1.0, (8.0 - log_p) / 2.0))

    # --- 5. Composite efficiency score ---
    result.efficiency_score = (
        0.35 * result.pruning_tolerance
        + 0.25 * min(result.compression_ratio / 4.0, 1.0)
        + 0.20 * result.sparse_op_coverage
        + 0.20 * result.memory_efficiency_score
    )

    return result


# ── Evaluation results ───────────────────────────────────────────────

@dataclass
class BridgeResult:
    """Complete evaluation result from the research pipeline."""
    status: str  # "success", "error", "failed_sandbox"
    error: Optional[str] = None
    error_stage: Optional[str] = None

    # Graph info
    graph_fingerprint: Optional[str] = None
    n_ops: int = 0
    depth: int = 0
    n_params_estimate: int = 0
    has_gradient_path: bool = False

    # Sandbox results (Stage 0 / 0.5)
    sandbox_passed: bool = False
    compile_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    param_count: int = 0
    peak_memory_mb: float = 0.0
    grad_norm: float = 0.0
    stability_score: float = 0.0

    # Fingerprint (Stage 1)
    cka_vs_transformer: float = 0.0
    cka_vs_ssm: float = 0.0
    cka_vs_conv: float = 0.0
    interaction_locality: float = 0.0
    interaction_sparsity: float = 0.0
    intrinsic_dim: float = 0.0
    isotropy: float = 0.0

    # Novelty
    structural_novelty: float = 0.0
    behavioral_novelty: float = 0.0
    overall_novelty: float = 0.0
    most_similar_to: str = ""

    # Compression / Efficiency
    compression_ratio: float = 1.0
    pruning_tolerance: float = 0.0
    sparse_op_coverage: float = 0.0
    efficiency_score: float = 0.0

    # Timing
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy bools/ints to native Python types for JSON serialization
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        return d


# ── Main evaluation pipeline ─────────────────────────────────────────

def evaluate_workflow(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
    vocab_size: int = 32000,
    device: str = "cpu",
    run_fingerprint: bool = True,
    run_novelty: bool = True,
    batch_size: int = 2,
    seq_len: int = 128,
) -> BridgeResult:
    """Full pipeline: workflow → graph → compile → eval → fingerprint → novelty.

    Args:
        workflow_json: aria-designer workflow in workflow_graph.v1 format
        model_dim: feature dimension (D in (B, S, D))
        vocab_size: vocabulary size for the synthesized model
        device: "cpu" or "cuda"
        run_fingerprint: whether to compute behavioral fingerprint
        run_novelty: whether to compute novelty scores
        batch_size: batch size for evaluation
        seq_len: sequence length for evaluation

    Returns:
        BridgeResult with all evaluation metrics
    """
    result = BridgeResult(status="error")
    t0 = time.monotonic()

    # Step 1: Convert workflow to ComputationGraph
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
    except ValueError as e:
        result.error = str(e)
        result.error_stage = "conversion"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 2: Graph-level analysis (no GPU needed)
    result.graph_fingerprint = graph.fingerprint()
    result.n_ops = graph.n_ops()
    result.depth = graph.depth()
    result.n_params_estimate = graph.n_params_estimate()
    result.has_gradient_path = bool(graph.has_gradient_path())

    if not result.has_gradient_path:
        result.status = "error"
        result.error = "No differentiable path from input to output"
        result.error_stage = "analysis"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 3: Compile to PyTorch module
    try:
        from research.synthesis.compiler import compile_model
        model = compile_model([graph], vocab_size=vocab_size)
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.error_stage = "compilation"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 4: Sandbox evaluation (Stage 0 / 0.5)
    try:
        from research.eval.sandbox import safe_eval
        sandbox = safe_eval(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
        )
        result.sandbox_passed = sandbox.passed
        result.compile_time_ms = sandbox.compile_time_ms
        result.forward_time_ms = sandbox.forward_time_ms
        result.backward_time_ms = sandbox.backward_time_ms
        result.param_count = sandbox.param_count
        result.peak_memory_mb = sandbox.peak_memory_mb
        result.grad_norm = sandbox.grad_norm
        result.stability_score = getattr(sandbox, "stability_score", 0.0)

        if not sandbox.passed:
            result.status = "failed_sandbox"
            result.error = sandbox.error
            result.error_stage = sandbox.stage
            result.total_time_ms = (time.monotonic() - t0) * 1000
            return result
    except Exception as e:
        result.status = "error"
        result.error = f"Sandbox error: {e}"
        result.error_stage = "sandbox"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 5: Behavioral fingerprint (Stage 1)
    if run_fingerprint:
        try:
            from research.eval.fingerprint import compute_fingerprint
            fp = compute_fingerprint(
                model,
                seq_len=min(seq_len, 64),
                model_dim=model_dim,
                vocab_size=vocab_size,
                device=device,
            )
            result.cka_vs_transformer = getattr(fp, "cka_vs_transformer", 0.0)
            result.cka_vs_ssm = getattr(fp, "cka_vs_ssm", 0.0)
            result.cka_vs_conv = getattr(fp, "cka_vs_conv", 0.0)
            result.interaction_locality = getattr(fp, "interaction_locality", 0.0)
            result.interaction_sparsity = getattr(fp, "interaction_sparsity", 0.0)
            result.intrinsic_dim = getattr(fp, "intrinsic_dim", 0.0)
            result.isotropy = getattr(fp, "isotropy", 0.0)
        except Exception:
            pass  # Fingerprint is optional; don't fail the whole eval

    # Step 6: Novelty scoring
    if run_novelty:
        try:
            from research.eval.metrics import novelty_score
            fp_obj = None
            if run_fingerprint:
                try:
                    from research.eval.fingerprint import compute_fingerprint
                    fp_obj = compute_fingerprint(
                        model, seq_len=min(seq_len, 64),
                        model_dim=model_dim, vocab_size=vocab_size,
                        device=device,
                    )
                except Exception:
                    pass
            metrics = novelty_score(graph, fingerprint=fp_obj)
            result.structural_novelty = metrics.structural_novelty
            result.behavioral_novelty = metrics.behavioral_novelty
            result.overall_novelty = metrics.overall_novelty
            result.most_similar_to = getattr(metrics, "most_similar_to", "")
        except Exception:
            pass  # Novelty is optional

    result.status = "success"
    result.total_time_ms = (time.monotonic() - t0) * 1000
    return result


# ── Utility functions for the API layer ──────────────────────────────

def list_available_primitives() -> List[Dict[str, Any]]:
    """List all primitives available for use in aria-designer workflows."""
    result = []
    for name, op in sorted(PRIMITIVE_REGISTRY.items()):
        result.append({
            "name": op.name,
            "category": op.category.value if hasattr(op.category, "value") else str(op.category),
            "n_inputs": op.n_inputs,
            "shape_rule": op.shape_rule,
            "has_params": op.has_params,
            "param_formula": op.param_formula,
            "config_keys": list(op.config_keys) if op.config_keys else [],
        })
    return result


def validate_workflow_graph(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
) -> Dict[str, Any]:
    """Validate that a workflow can be converted to a valid ComputationGraph.

    Returns {"valid": True, "graph_info": {...}} or {"valid": False, "error": "..."}.
    """
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
        return {
            "valid": True,
            "graph_info": {
                "fingerprint": graph.fingerprint(),
                "n_ops": graph.n_ops(),
                "depth": graph.depth(),
                "n_params_estimate": int(graph.n_params_estimate()),
                "has_gradient_path": bool(graph.has_gradient_path()),
                "model_dim": model_dim,
            },
        }
    except ValueError as e:
        return {"valid": False, "error": str(e)}


def estimate_performance(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
) -> Dict[str, Any]:
    """Quick performance estimate without running the model.

    Returns param count, estimated FLOPs, op breakdown.
    """
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    # Op histogram
    op_counts: Dict[str, int] = {}
    for node in graph.nodes.values():
        if not node.is_input:
            op_counts[node.op_name] = op_counts.get(node.op_name, 0) + 1

    # Category histogram
    cat_counts: Dict[str, int] = {}
    for node in graph.nodes.values():
        if not node.is_input and node.op_name in PRIMITIVE_REGISTRY:
            cat = PRIMITIVE_REGISTRY[node.op_name].category
            cat_name = cat.value if hasattr(cat, "value") else str(cat)
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

    # Rough FLOP estimate (per token)
    D = model_dim
    flops = 0
    for node in graph.nodes.values():
        if node.is_input:
            continue
        if node.op_name in PRIMITIVE_REGISTRY:
            op = PRIMITIVE_REGISTRY[node.op_name]
            if op.shape_rule == "linear":
                out_dim = node.config.get("out_dim", D)
                flops += 2 * D * out_dim  # matmul
            elif op.shape_rule == "matmul":
                flops += 2 * D * D
            elif op.shape_rule == "identity":
                flops += D  # elementwise
            elif op.shape_rule == "binary_broadcast":
                flops += D
            else:
                flops += D  # conservative estimate

    return {
        "valid": True,
        "n_params_estimate": graph.n_params_estimate(),
        "flops_per_token_estimate": flops,
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "op_counts": op_counts,
        "category_counts": cat_counts,
        "has_gradient_path": graph.has_gradient_path(),
    }
