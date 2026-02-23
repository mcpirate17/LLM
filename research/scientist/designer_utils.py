"""
Utilities for the Aria Designer API integration.
Handles conversion between frontend workflow JSON and backend ComputationGraph.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set
import sys
import os

# Ensure project root and aria-designer are in sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DESIGNER_ROOT = os.path.join(_PROJECT_ROOT, "aria-designer")

for p in [_PROJECT_ROOT, _DESIGNER_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from research.synthesis.graph import ComputationGraph, ShapeInfo
from research.synthesis.compiler import compile_graph
from research.synthesis.primitives import PRIMITIVE_REGISTRY

try:
    from runtime.profiler import profile_workflow
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
except Exception as e:
    logger.error(f"Unexpected error importing profiler: {e}")
    HAS_PROFILER = False

logger = logging.getLogger(__name__)

def fe_type_to_op_name(fe_type: str) -> str:
    """Map frontend component type to backend op name."""
    if not fe_type:
        return "identity"
    # "io/input" -> "input"
    # "math/relu" -> "relu"
    return fe_type.split("/")[-1]

def workflow_to_computation_graph(workflow_json: Dict[str, Any], default_model_dim: int = 256) -> ComputationGraph:
    """
    Convert frontend workflow JSON to a ComputationGraph.
    
    Expected format:
    {
        "nodes": [{"id": "n1", "component_type": "io/input", "params": {}}, ...],
        "edges": [{"source": "n1", "target": "n2", ...}, ...]
    }
    """
    nodes = workflow_json.get("nodes", [])
    edges = workflow_json.get("edges", [])
    metadata = workflow_json.get("metadata", {})
    
    model_dim = metadata.get("model_dim", default_model_dim)
    graph = ComputationGraph(model_dim)
    
    # Map frontend string IDs to backend integer IDs
    fe_to_be: Dict[str, int] = {}
    
    # 1. Identify input nodes
    fe_inputs = [n for n in nodes if fe_type_to_op_name(n["component_type"]) == "input"]
    if not fe_inputs:
        # If no explicit input node, look for nodes with no incoming edges
        target_ids = {e["target"] for e in edges}
        fe_inputs = [n for n in nodes if n["id"] not in target_ids]
        
    if not fe_inputs:
        raise ValueError("Graph has no detectable input nodes.")
        
    # For now, we only support a single input node in ComputationGraph
    main_input = fe_inputs[0]
    be_input_id = graph.add_input()
    fe_to_be[main_input["id"]] = be_input_id
    
    # 2. Add other nodes in topological order
    # We use a simple iterative approach to add nodes whose inputs are already resolved
    pending = [n for n in nodes if n["id"] not in fe_to_be]
    
    # Also handle the case where "io/output" is used as a sink
    fe_outputs = [n for n in nodes if fe_type_to_op_name(n["component_type"]) == "output"]
    output_fe_id: Optional[str] = None
    if fe_outputs:
        output_fe_id = fe_outputs[0]["id"]

    added_any = True
    while pending and added_any:
        added_any = False
        next_pending = []
        for node in pending:
            op_name = fe_type_to_op_name(node["component_type"])
            
            # Skip the explicit output node for now, we'll use its input as the graph output
            if op_name == "output":
                next_pending.append(node)
                continue
                
            # Find incoming edges
            incoming = [e for e in edges if e["target"] == node["id"]]
            source_fe_ids = [e["source"] for e in incoming]
            
            if all(sid in fe_to_be for sid in source_fe_ids):
                # All inputs resolved, add this op
                be_input_ids = [fe_to_be[sid] for sid in source_fe_ids]
                
                # Validation: if no inputs and not an input node, it's a disconnected op (unless it's a constant, but we don't have those yet)
                if not be_input_ids and op_name != "input":
                    # Treat as identity if it has no inputs but is expected? 
                    # Actually, Synthesis usually starts with input.
                    # For now, just skip or raise error.
                    logger.warning(f"Node {node['id']} ({op_name}) has no inputs, skipping.")
                    continue

                try:
                    be_id = graph.add_op(op_name, be_input_ids, node.get("params", {}))
                    fe_to_be[node["id"]] = be_id
                    added_any = True
                except Exception as e:
                    raise ValueError(f"Failed to add node {node['id']} ({op_name}): {e}")
            else:
                next_pending.append(node)
        pending = next_pending
        
    if pending:
        # Check if only the output node remains
        remaining_non_output = [n for n in pending if fe_type_to_op_name(n["component_type"]) != "output"]
        if remaining_non_output:
            raise ValueError(f"Graph has cycles or disconnected components. Remaining nodes: {[n['id'] for n in remaining_non_output]}")

    # 3. Set output node
    if output_fe_id:
        # Find what's connected to the output node
        incoming_to_output = [e for e in edges if e["target"] == output_fe_id]
        if incoming_to_output:
            last_source_fe_id = incoming_to_output[0]["source"]
            if last_source_fe_id in fe_to_be:
                graph.set_output(fe_to_be[last_source_fe_id])
            else:
                raise ValueError(f"Output node connected to unresolved node {last_source_fe_id}")
        else:
            # If output node exists but nothing connected to it, find a sink
            _set_fallback_output(graph, fe_to_be, nodes, edges)
    else:
        _set_fallback_output(graph, fe_to_be, nodes, edges)
        
    return graph

def _set_fallback_output(graph: ComputationGraph, fe_to_be: Dict[str, int], nodes: List[Dict], edges: List[Dict]):
    """Find a suitable sink node to use as graph output."""
    source_fe_ids = {e["source"] for e in edges}
    # Sinks are nodes that are NOT sources for any edge
    fe_sinks = [n for n in nodes if n["id"] not in source_fe_ids and n["id"] in fe_to_be]
    if fe_sinks:
        # Prefer standard sequence output if possible (dim == model_dim)
        for sink in reversed(fe_sinks):
            try:
                graph.set_output(fe_to_be[sink["id"]])
                return
            except Exception:
                continue
        # If no standard output found, try the last added node
        try:
            graph.set_output(fe_to_be[fe_sinks[-1]["id"]])
        except Exception as e:
            raise ValueError(f"Could not find a valid output node: {e}")
    else:
        # Fallback to the last node added to the graph
        topo = graph.topological_order()
        if topo:
            graph.set_output(topo[-1])
        else:
            raise ValueError("Empty graph, cannot set output.")

def validate_designer_graph(workflow_json: Dict[str, Any], default_model_dim: int = 256) -> Dict[str, Any]:
    """
    Validate a workflow graph and return per-node status.
    """
    nodes = workflow_json.get("nodes", [])
    edges = workflow_json.get("edges", [])
    metadata = workflow_json.get("metadata", {})
    model_dim = metadata.get("model_dim", default_model_dim)
    
    node_statuses: Dict[str, Dict[str, Any]] = {n["id"]: {"valid": True, "errors": []} for n in nodes}
    global_errors = []
    
    try:
        # 1. Check for input node
        fe_inputs = [n for n in nodes if fe_type_to_op_name(n["component_type"]) == "input"]
        if not fe_inputs:
            target_ids = {e["target"] for e in edges}
            fe_inputs = [n for n in nodes if n["id"] not in target_ids]
        
        if not fe_inputs:
            global_errors.append("Graph has no detectable input nodes.")
        
        # 2. Check each node for required inputs
        for node in nodes:
            op_name = fe_type_to_op_name(node["component_type"])
            if op_name == "input":
                continue
                
            prim = PRIMITIVE_REGISTRY.get(op_name)
                
            incoming = [e for e in edges if e["target"] == node["id"]]
            if prim:
                if len(incoming) < prim.n_inputs:
                    msg = f"Node {node['id']} ({op_name}) requires {prim.n_inputs} inputs, but only has {len(incoming)}."
                    node_statuses[node["id"]]["valid"] = False
                    node_statuses[node["id"]]["errors"].append(msg)
            elif op_name != "output":
                node_statuses[node["id"]]["valid"] = False
                node_statuses[node["id"]]["errors"].append(f"Unknown component type: {node['component_type']}")

        # 3. Check for cycles and disconnected components
        # We can reuse the topological sort logic but track which nodes fail
        fe_to_be = {}
        if fe_inputs:
            fe_to_be[fe_inputs[0]["id"]] = 0 # Dummy ID
            
        pending = [n for n in nodes if n["id"] not in fe_to_be]
        added_any = True
        while pending and added_any:
            added_any = False
            next_pending = []
            for node in pending:
                op_name = fe_type_to_op_name(node["component_type"])
                if op_name == "output":
                    next_pending.append(node)
                    continue
                incoming = [e for e in edges if e["target"] == node["id"]]
                if all(e["source"] in fe_to_be for e in incoming):
                    fe_to_be[node["id"]] = len(fe_to_be)
                    added_any = True
                else:
                    next_pending.append(node)
            pending = next_pending
            
        if pending:
            remaining_non_output = [n for n in pending if fe_type_to_op_name(n["component_type"]) != "output"]
            for node in remaining_non_output:
                node_statuses[node["id"]]["valid"] = False
                node_statuses[node["id"]]["errors"].append("Cycle detected or node is disconnected from input.")

        # 4. Shape compatibility (optional/best-effort without full graph build)
        # For a full check, we need to build the ComputationGraph
        if not global_errors and all(s["valid"] for s in node_statuses.values()):
            try:
                workflow_to_computation_graph(workflow_json, model_dim)
            except Exception as e:
                global_errors.append(f"Graph construction failed: {e}")

    except Exception as e:
        global_errors.append(f"Validation failed with internal error: {e}")

    return {
        "success": len(global_errors) == 0 and all(s["valid"] for s in node_statuses.values()),
        "node_statuses": node_statuses,
        "global_errors": global_errors
    }

def compile_designer_graph(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile a workflow and return metadata.
    """
    try:
        graph = workflow_to_computation_graph(workflow_json)
        
        # Compile to module
        # Note: we use use_ir=False for now to get a standard nn.Module if needed,
        # but the profiler might want IR.
        module = compile_graph(graph, use_ir=False)
        
        param_count = sum(p.numel() for p in module.parameters())
        
        # Simple shape analysis
        # We can describe the graph
        description = graph.describe()
        
        return {
            "success": True,
            "workflow_id": workflow_json.get("workflow_id"),
            "param_count": param_count,
            "description": description,
            "fingerprint": graph.fingerprint(),
            "n_ops": graph.n_ops(),
            "depth": graph.depth(),
        }
    except Exception as e:
        import traceback
        logger.error(f"Compilation error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "workflow_id": workflow_json.get("workflow_id"),
        }

def run_designer_graph(workflow_json: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """
    Execute a forward pass of the workflow and return metrics.
    """
    # ... (already implemented)
    if not HAS_PROFILER:
        return {
            "success": False,
            "error": "Profiler/Runtime not available in this environment.",
            "workflow_id": workflow_json.get("workflow_id"),
        }

    try:
        metadata = workflow_json.get("metadata", {})
        model_dim = metadata.get("model_dim", 256)
        
        # Run profiling (which includes execution)
        report = profile_workflow(
            workflow_json, 
            model_dim=model_dim, 
            device=device,
            runtime=True,
            batch_size=1,
            seq_len=128
        )
        
        report_dict = report.to_dict()
        
        return {
            "success": True,
            "workflow_id": workflow_json.get("workflow_id"),
            "metrics": {
                "param_count": report_dict["total_params"],
                "flops_per_token": report_dict["total_flops_per_token"],
                "forward_ms": report_dict["forward_time_ms"],
                "peak_memory_mb": report_dict["peak_memory_mb"],
                "throughput": report_dict["throughput_tokens_per_sec"],
            },
            "bottlenecks": report_dict["bottleneck_ops"],
            "native_coverage": report_dict["native_coverage"],
        }
    except Exception as e:
        import traceback
        logger.error(f"Run error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "workflow_id": workflow_json.get("workflow_id"),
        }

def get_designer_components() -> List[Dict[str, Any]]:
    """
    Return all available primitives formatted for the designer component sidebar.
    """
    from research.synthesis.primitives import PRIMITIVE_REGISTRY, OpCategory
    
    # 1. Start with standard IO components
    components = [
        {
            "id": "io/input",
            "name": "Input",
            "category": "io",
            "description": "Standard graph input (B, S, D).",
            "inputs": [],
            "outputs": [{"name": "y", "dtype": "tensor"}],
            "params_schema": {},
            "icon": "plug"
        },
        {
            "id": "io/output",
            "name": "Output Head",
            "category": "io",
            "description": "Standard graph output (B, S, D).",
            "inputs": [{"name": "x", "dtype": "tensor"}],
            "outputs": [],
            "params_schema": {},
            "icon": "arrow-up-circle"
        }
    ]
    
    # Category mapping
    CAT_MAP = {
        OpCategory.ELEMENTWISE_UNARY: "math",
        OpCategory.ELEMENTWISE_BINARY: "math",
        OpCategory.REDUCTION: "reduction",
        OpCategory.LINEAR_ALGEBRA: "linear_algebra",
        OpCategory.STRUCTURAL: "structural",
        OpCategory.PARAMETERIZED: "linear_algebra", # Most are linear-like
        OpCategory.SEQUENCE: "sequence",
        OpCategory.FREQUENCY: "frequency",
        OpCategory.MATH_SPACE: "math_space",
        OpCategory.FUNCTIONAL: "functional",
    }
    
    # Icon mapping (basic)
    ICON_MAP = {
        "math": "calculator",
        "reduction": "bar-chart",
        "linear_algebra": "grid",
        "structural": "layers",
        "sequence": "activity",
        "frequency": "wind",
        "math_space": "box",
        "functional": "framer",
    }

    for name, op in PRIMITIVE_REGISTRY.items():
        if name == "input":
            continue
            
        cat = CAT_MAP.get(op.category, "other")
        
        # Build ports
        inputs = []
        if op.n_inputs == 1:
            inputs.append({"name": "x", "dtype": "tensor"})
        elif op.n_inputs == 2:
            inputs.append({"name": "a", "dtype": "tensor"})
            inputs.append({"name": "b", "dtype": "tensor"})
            
        outputs = [{"name": "y", "dtype": "tensor"}]
        
        # Build params
        params = {}
        for key in op.config_keys:
            param_type = "integer"
            if "prob" in key or "scale" in key or "damping" in key:
                param_type = "float"
            elif "operator" in key:
                param_type = "enum"
                
            params[key] = {
                "type": param_type,
                "default": None,
                "description": f"Parameter {key}"
            }
            if param_type == "enum":
                params[key]["options"] = [">", "<", ">=", "<=", "==", "!="]
        
        # Special case for out_dim
        if "out_dim" in op.config_keys:
            params["out_dim"]["description"] = "Output dimension. Defaults to model dimension."
            params["out_dim"]["default"] = 256

        components.append({
            "id": f"{cat}/{name}",
            "name": name.replace("_", " ").title(),
            "category": cat,
            "description": op.description,
            "inputs": inputs,
            "outputs": outputs,
            "params_schema": params,
            "icon": ICON_MAP.get(cat, "circle"),
            "performance": {
                "has_params": op.has_params,
                "param_formula": op.param_formula,
                "preserves_gradient": op.preserves_gradient,
                "numerically_risky": op.numerically_risky
            }
        })
        
    return components

def generate_python_module(workflow_json: Dict[str, Any]) -> str:
    """
    Generate standalone PyTorch module code for a workflow.
    """
    graph = workflow_to_computation_graph(workflow_json)
    topo = graph.topological_order()
    
    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "class StandaloneModule(nn.Module):",
        "    def __init__(self, model_dim=256):",
        "        super().__init__()",
        f"        self.model_dim = {graph.model_dim}"
    ]
    
    # Initialize layers
    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            continue
        
        op = node.op
        if op.has_params:
            # We need to map op names to init code
            # Simplified for now
            if "linear" in node.op_name:
                out_dim = node.config.get("out_dim", graph.model_dim)
                # We don't know in_dim easily from just the node, need its input node's out_dim
                # For now, assume model_dim if not found
                in_dim = graph.model_dim
                if node.input_ids:
                    in_node = graph.nodes[node.input_ids[0]]
                    in_dim = in_node.output_shape.dim
                
                lines.append(f"        self.layer_{nid} = nn.Linear({in_dim}, {out_dim})")
            elif "rmsnorm" in node.op_name:
                lines.append(f"        self.layer_{nid} = nn.Parameter(torch.ones({graph.model_dim}))")
            else:
                lines.append(f"        # No custom init code for {node.op_name}")
                
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        outputs = {}")
    
    # Forward pass
    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            lines.append("        outputs[0] = x")
            continue
            
        inputs_str = ", ".join([f"outputs[{iid}]" for iid in node.input_ids])
        
        # Map op to forward code
        # Simplified for now
        if node.op_name == "relu":
            lines.append(f"        outputs[{nid}] = F.relu({inputs_str})")
        elif "linear" in node.op_name:
            lines.append(f"        outputs[{nid}] = self.layer_{nid}({inputs_str})")
        elif "add" in node.op_name:
            lines.append(f"        outputs[{nid}] = outputs[{node.input_ids[0]}] + outputs[{node.input_ids[1]}]")
        else:
            lines.append(f"        # TODO: Implement forward for {node.op_name}")
            lines.append(f"        outputs[{nid}] = {inputs_str}")
            
    output_id = graph._output_node_id
    lines.append(f"        return outputs[{output_id}]")
    
    return "\n".join(lines)

def import_research_program(graph_json_str: str) -> Dict[str, Any]:
    """
    Convert a backend ComputationGraph JSON to designer workflow JSON.
    """
    import json
    data = json.loads(graph_json_str)
    
    # ComputationGraph format usually has:
    # nodes: { "0": {"id": 0, "op_name": "input", "input_ids": [], ...}, ... }
    
    nodes = []
    edges = []
    
    be_nodes = data.get("nodes", {})
    model_dim = data.get("model_dim", 256)

    # Calculate depth-based layout
    depths = {}
    def get_depth(nid):
        if nid in depths: return depths[nid]
        node = be_nodes.get(str(nid))
        if not node or not node.get("input_ids"):
            depths[nid] = 0
            return 0
        d = 1 + max(get_depth(iid) for iid in node["input_ids"])
        depths[nid] = d
        return d

    for nid_str in be_nodes:
        get_depth(int(nid_str))

    # Group by depth for horizontal spreading
    by_depth = {}
    for nid, d in depths.items():
        by_depth.setdefault(d, []).append(nid)

    for nid_str, be_node in be_nodes.items():
        nid = int(nid_str)
        op_name = be_node["op_name"]
        
        # Map op_name back to component_type
        category = "other"
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
        if op_name in PRIMITIVE_REGISTRY:
            category = PRIMITIVE_REGISTRY[op_name].category.value
            if "unary" in category or "binary" in category: category = "math"
            elif "param" in category: category = "linear_algebra"
        
        fe_id = f"node_{nid}"
        comp_type = f"{category}/{op_name}"
        if op_name == "input": comp_type = "io/input"

        # Calculate position based on depth
        depth = depths.get(nid, 0)
        nodes_at_depth = by_depth.get(depth, [])
        idx_at_depth = nodes_at_depth.index(nid) if nid in nodes_at_depth else 0
        
        pos_x = 50 + depth * 250
        pos_y = 50 + idx_at_depth * 120

        nodes.append({
            "id": fe_id,
            "component_type": comp_type,
            "params": be_node.get("config", {}),
            "ui_meta": {"position": {"x": pos_x, "y": pos_y}}
        })
        
        # Add edges from input_ids
        for iid in be_node.get("input_ids", []):
            edges.append({
                "id": f"edge_{iid}_{nid}",
                "source": f"node_{iid}",
                "source_port": "y",
                "target": fe_id,
                "target_port": "x" if len(be_node["input_ids"]) == 1 else ("a" if iid == be_node["input_ids"][0] else "b")
            })
            
    # Add explicit output node connected to the graph's output
    output_be_id = data.get("output_node_id")
    if output_be_id is not None:
        max_depth = max(depths.values()) if depths else 0
        nodes.append({
            "id": "node_out",
            "component_type": "io/output",
            "params": {},
            "ui_meta": {"position": {"x": 50 + (max_depth + 1) * 250, "y": 50}}
        })
        edges.append({
            "id": "edge_to_out",
            "source": f"node_{output_be_id}",
            "source_port": "y",
            "target": "node_out",
            "target_port": "x"
        })

    return {
        "workflow_id": f"imported_{hash(graph_json_str) % 10000}",
        "name": f"Imported Architecture",
        "nodes": nodes,
        "edges": edges,
        "metadata": {"model_dim": model_dim}
    }
