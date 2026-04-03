"""
Workflow Converter: Unified conversion between frontend JSON and backend ComputationGraph.
Single source of truth for graph transformation logic.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
from .graph import ComputationGraph
from .component_registry import registry, fe_type_to_op_name
from .primitives import PRIMITIVE_REGISTRY

logger = logging.getLogger(__name__)


def _index_edges(
    edges: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    incoming_by_target: Dict[str, List[Dict[str, Any]]] = {}
    outgoing_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        incoming_by_target.setdefault(edge["target"], []).append(edge)
        outgoing_by_source.setdefault(edge["source"], []).append(edge)
    return incoming_by_target, outgoing_by_source


def _lower_template_component(
    graph: ComputationGraph,
    component_leaf: str,
    input_cg_ids: List[int],
    model_dim: int,
) -> int:
    """Lower high-level block node IDs into explicit primitive subgraphs."""
    if not input_cg_ids:
        raise ValueError(
            f"Template-lowered component '{component_leaf}' requires at least one input"
        )
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

    if component_leaf == "uniform":
        # Lower to identity but tagged for routing telemetry
        return graph.add_op("identity", [x], {"routing_kind": "uniform"})

    if component_leaf == "layerdrop":
        # Lower to topk_gate with k=1 and deterministic dropout-like behavior
        # For simplicity in Phase 0/1, we'll use a gate that can skip
        return graph.add_op("topk_gate", [x], {"k": 1, "routing_kind": "layerdrop"})

    raise ValueError(f"No template-lowering implementation for '{component_leaf}'")


def workflow_to_computation_graph(
    workflow_json: Dict[str, Any],
    default_model_dim: int = 256,
    return_id_map: bool = False,
) -> ComputationGraph | Tuple[ComputationGraph, Dict[str, int]]:
    """
    Convert frontend workflow JSON to a research ComputationGraph.
    """
    nodes = workflow_json.get("nodes", [])
    edges = workflow_json.get("edges", [])
    metadata = workflow_json.get("metadata", {})
    incoming_by_target, outgoing_by_source = _index_edges(edges)

    model_dim = metadata.get("model_dim", default_model_dim)
    graph = ComputationGraph(model_dim)

    # Z13: Unify routing/compression schema validation
    rc = metadata.get("routing_compression")
    if rc:
        from ..schemas.validator import validate_routing_compression

        try:
            validate_routing_compression(rc)
            graph.metadata["routing_compression"] = rc
        except ValueError as e:
            logger.warning(f"Invalid routing_compression metadata: {e}")

    # Map frontend string IDs to backend integer IDs
    fe_to_be: Dict[str, int] = {}

    # 1. Identify input nodes
    fe_inputs = [
        n
        for n in nodes
        if fe_type_to_op_name(n["component_type"]) in ("input", "graph_input")
    ]
    if not fe_inputs:
        # If no explicit input node, look for nodes with no incoming edges
        fe_inputs = [n for n in nodes if n["id"] not in incoming_by_target]

    if not fe_inputs:
        raise ValueError("Graph has no detectable input nodes.")

    # For now, we only support a single input node in ComputationGraph
    main_input = fe_inputs[0]
    be_input_id = graph.add_input()
    fe_to_be[main_input["id"]] = be_input_id

    # 2. Add other nodes in topological order
    fe_outputs = [
        n
        for n in nodes
        if fe_type_to_op_name(n["component_type"])
        in ("output", "output_head", "graph_output")
    ]
    output_fe_id: Optional[str] = None
    if fe_outputs:
        output_fe_id = fe_outputs[0]["id"]

    pending = [n for n in nodes if n["id"] not in fe_to_be]
    added_any = True
    first_error: Optional[str] = None

    while pending and added_any:
        added_any = False
        next_pending = []
        for node in pending:
            comp_type = node["component_type"]
            leaf_id = comp_type.split("/")[-1]
            op_name = registry.get_primitive_name(comp_type)

            # Strict routing: routing components must lower to a real primitive or template.
            if comp_type.startswith("routing/"):
                if (
                    not registry.is_passthrough(comp_type)
                    and leaf_id not in registry.template_lowered_components
                    and op_name not in PRIMITIVE_REGISTRY
                ):
                    raise ValueError(
                        f"Routing component '{comp_type}' is not supported in the research bridge. "
                        "Implement lowering or add a real primitive mapping before use."
                    )

            # Skip explicit output nodes during construction
            if node["id"] == output_fe_id:
                next_pending.append(node)
                continue

            # Passthrough handling
            if registry.is_passthrough(comp_type):
                incoming = incoming_by_target.get(node["id"], [])
                if incoming and incoming[0]["source"] in fe_to_be:
                    fe_to_be[node["id"]] = fe_to_be[incoming[0]["source"]]
                    added_any = True
                    continue
                elif not incoming:
                    fe_to_be[node["id"]] = be_input_id
                    added_any = True
                    continue

            # Find incoming edges
            incoming = incoming_by_target.get(node["id"], [])
            source_fe_ids = [e["source"] for e in incoming]

            if all(sid in fe_to_be for sid in source_fe_ids):
                be_input_ids = [fe_to_be[sid] for sid in source_fe_ids]

                if leaf_id in registry.template_lowered_components:
                    try:
                        be_id = _lower_template_component(
                            graph, leaf_id, be_input_ids, model_dim
                        )
                        fe_to_be[node["id"]] = be_id
                        added_any = True
                        continue
                    except Exception as e:
                        err_msg = str(e)
                        if not first_error:
                            first_error = err_msg
                        logger.error(
                            f"Template lowering failed for {node['id']} ({leaf_id}): {err_msg}"
                        )

                if not be_input_ids and op_name != "input":
                    logger.warning(
                        f"Node {node['id']} ({op_name}) has no inputs, skipping."
                    )
                    continue

                try:
                    be_id = graph.add_op(
                        op_name,
                        be_input_ids,
                        node.get("params", node.get("paramValues", {})),
                    )
                    fe_to_be[node["id"]] = be_id
                    added_any = True
                except Exception as e:
                    err_msg = str(e)
                    if not first_error:
                        first_error = err_msg
                    logger.error(
                        f"Failed to add node {node['id']} ({op_name}): {err_msg}"
                    )
                    next_pending.append(node)
            else:
                next_pending.append(node)
        pending = next_pending

    if pending:
        # Check if only the output node remains
        remaining_non_output = [n for n in pending if n["id"] != output_fe_id]
        if remaining_non_output:
            msg = f"Graph has cycles or disconnected components. Remaining nodes: {[n['id'] for n in remaining_non_output]}"
            if first_error:
                msg += f" (First error: {first_error})"
            raise ValueError(msg)

    # 3. Set output node
    if output_fe_id:
        incoming_to_output = incoming_by_target.get(output_fe_id, [])
        if incoming_to_output:
            last_source_fe_id = incoming_to_output[0]["source"]
            if last_source_fe_id in fe_to_be:
                graph.set_output(fe_to_be[last_source_fe_id])
            else:
                raise ValueError(
                    f"Output node '{output_fe_id}' depends on unmapped source "
                    f"'{last_source_fe_id}'"
                )
        else:
            raise ValueError(f"Output node '{output_fe_id}' has no incoming edge")
    else:
        _set_fallback_output(graph, fe_to_be, nodes, outgoing_by_source)

    if return_id_map:
        return graph, fe_to_be
    return graph


def _set_fallback_output(
    graph: ComputationGraph,
    fe_to_be: Dict[str, int],
    nodes: List[Dict],
    outgoing_by_source: Dict[str, List[Dict[str, Any]]],
):
    """Find a suitable sink node to use as graph output."""
    fe_sinks = [
        n for n in nodes if n["id"] not in outgoing_by_source and n["id"] in fe_to_be
    ]
    if fe_sinks:
        for sink in reversed(fe_sinks):
            try:
                graph.set_output(fe_to_be[sink["id"]])
                return
            except ValueError:
                continue
        raise ValueError("No sink node could be used as graph output")

    raise ValueError("Graph has no detectable sink node for output")


def graph_to_workflow(
    graph: ComputationGraph,
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a ComputationGraph to frontend workflow JSON.
    """
    from .primitives import PRIMITIVE_REGISTRY

    nodes = []
    edges = []

    # Calculate depth for layout
    depths = {}
    topo = graph.topological_order()

    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            depths[nid] = 0
        else:
            depths[nid] = 1 + max([depths[iid] for iid in node.input_ids] or [0])

    # Group by depth for horizontal spreading
    by_depth = {}
    for nid, d in depths.items():
        by_depth.setdefault(d, []).append(nid)

    for nid in topo:
        node = graph.nodes[nid]
        op_name = node.op_name

        # Determine component_type
        if node.is_input:
            comp_type = "io/input"
        else:
            # Reconstruct from registry if possible
            prim = PRIMITIVE_REGISTRY.get(op_name)
            cat = "math"
            if prim:
                from .primitives import OpCategory

                CAT_MAP = {
                    OpCategory.ELEMENTWISE_UNARY: "math",
                    OpCategory.ELEMENTWISE_BINARY: "math",
                    OpCategory.REDUCTION: "reduction",
                    OpCategory.LINEAR_ALGEBRA: "linear_algebra",
                    OpCategory.STRUCTURAL: "structural",
                    OpCategory.PARAMETERIZED: "linear_algebra",
                    OpCategory.SEQUENCE: "sequence",
                    OpCategory.FREQUENCY: "frequency",
                    OpCategory.MATH_SPACE: "math_space",
                    OpCategory.FUNCTIONAL: "functional",
                }
                cat = CAT_MAP.get(prim.category, "math")
            comp_type = f"{cat}/{op_name}"

        depth = depths.get(nid, 0)
        idx = by_depth[depth].index(nid)

        fe_id = f"node_{nid}"
        nodes.append(
            {
                "id": fe_id,
                "component_type": comp_type,
                "params": node.config,
                "ui_meta": {"position": {"x": 100 + depth * 250, "y": 100 + idx * 120}},
            }
        )

        # Add edges
        for i, iid in enumerate(node.input_ids):
            target_port = "x" if len(node.input_ids) == 1 else ("a" if i == 0 else "b")
            edges.append(
                {
                    "id": f"e_{iid}_{nid}",
                    "source": f"node_{iid}",
                    "source_port": "y",
                    "target": fe_id,
                    "target_port": target_port,
                }
            )

    # Add explicit output node
    if graph._output_node_id is not None:
        max_depth = max(depths.values()) if depths else 0
        nodes.append(
            {
                "id": "node_out",
                "component_type": "io/output_head",
                "params": {},
                "ui_meta": {"position": {"x": 100 + (max_depth + 1) * 250, "y": 100}},
            }
        )
        edges.append(
            {
                "id": "e_out",
                "source": f"node_{graph._output_node_id}",
                "source_port": "y",
                "target": "node_out",
                "target_port": "x",
            }
        )

    return {
        "workflow_id": workflow_id or f"wf_{id(graph)}",
        "name": name or "Imported Graph",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            **metadata,
            "model_dim": graph.model_dim,
            "source": "research_import",
            "graph_fingerprint": graph.fingerprint(),
        }
        if metadata
        else {
            "model_dim": graph.model_dim,
            "source": "research_import",
            "graph_fingerprint": graph.fingerprint(),
        },
        "schema_version": "workflow_graph.v1",
    }
