"""
Utilities for the Aria Designer API integration.
Handles conversion between frontend workflow JSON and backend ComputationGraph.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set
import sys
import os

# Ensure project root and aria_designer are in sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DESIGNER_ROOT = os.path.join(_PROJECT_ROOT, "aria_designer")

for p in [_PROJECT_ROOT, _DESIGNER_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_graph
from research.synthesis.primitives import PRIMITIVE_REGISTRY, OpCategory
from research.synthesis.workflow_converter import workflow_to_computation_graph as _w2cg, graph_to_workflow as _g2w

logger = logging.getLogger(__name__)

try:
    from runtime.profiler import profile_workflow
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
except Exception as e:
    logger.error(f"Unexpected error importing profiler: {e}")
    HAS_PROFILER = False

def workflow_to_computation_graph(workflow_json: Dict[str, Any], default_model_dim: int = 256) -> ComputationGraph:
    """Canonical conversion from frontend JSON to backend ComputationGraph."""
    return _w2cg(workflow_json, default_model_dim)

def validate_designer_graph(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a workflow graph and return structural diagnostics."""
    try:
        graph = workflow_to_computation_graph(workflow_json)
        return {
            "success": True,
            "valid": True,
            "workflow_id": workflow_json.get("workflow_id"),
            "n_ops": graph.n_ops(),
            "depth": graph.depth(),
            "fingerprint": graph.fingerprint(),
        }
    except Exception as e:
        return {
            "success": True,
            "valid": False,
            "workflow_id": workflow_json.get("workflow_id"),
            "error": str(e),
        }

def import_research_program(graph_json_str: str) -> Dict[str, Any]:
    """Convert a backend ComputationGraph JSON to designer workflow JSON."""
    from research.synthesis.serializer import graph_from_json as _gfj
    graph = _gfj(graph_json_str)
    return _g2w(graph)

def compile_designer_graph(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a workflow and return metadata."""
    try:
        graph = workflow_to_computation_graph(workflow_json)
        module = compile_graph(graph, use_ir=False)
        return {
            "success": True,
            "workflow_id": workflow_json.get("workflow_id"),
            "param_count": sum(p.numel() for p in module.parameters()),
            "description": graph.describe(),
            "fingerprint": graph.fingerprint(),
            "n_ops": graph.n_ops(),
            "depth": graph.depth(),
        }
    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return {"success": False, "error": str(e), "workflow_id": workflow_json.get("workflow_id")}

def run_designer_graph(workflow_json: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """Execute a forward pass of the workflow and return metrics."""
    if not HAS_PROFILER:
        return {"success": False, "error": "Profiler not available.", "workflow_id": workflow_json.get("workflow_id")}

    try:
        model_dim = workflow_json.get("metadata", {}).get("model_dim", 256)
        report = profile_workflow(workflow_json, model_dim=model_dim, device=device, runtime=True, batch_size=1, seq_len=128)
        rd = report.to_dict()
        return {
            "success": True,
            "workflow_id": workflow_json.get("workflow_id"),
            "metrics": {
                "param_count": rd["total_params"],
                "flops_per_token": rd["total_flops_per_token"],
                "forward_ms": rd["forward_time_ms"],
                "peak_memory_mb": rd["peak_memory_mb"],
                "throughput": rd["throughput_tokens_per_sec"],
            },
            "bottlenecks": rd["bottleneck_ops"],
            "native_coverage": rd["native_coverage"],
        }
    except Exception as e:
        logger.error(f"Run error: {e}")
        return {"success": False, "error": str(e), "workflow_id": workflow_json.get("workflow_id")}

def get_designer_components() -> List[Dict[str, Any]]:
    """Return all available primitives formatted for the designer component sidebar."""
    # This logic is mostly UI-specific formatting, so keeping it here but utilizing PRIMITIVE_REGISTRY
    components = [
        {"id": "io/input", "name": "Input", "category": "io", "description": "Graph input (B, S, D).", 
         "inputs": [], "outputs": [{"name": "y", "dtype": "tensor"}], "params_schema": {}, "icon": "plug"},
        {"id": "io/output", "name": "Output Head", "category": "io", "description": "Graph output (B, S, D).", 
         "inputs": [{"name": "x", "dtype": "tensor"}], "outputs": [], "params_schema": {}, "icon": "arrow-up-circle"}
    ]
    
    CAT_MAP = {
        OpCategory.ELEMENTWISE_UNARY: "math", OpCategory.ELEMENTWISE_BINARY: "math",
        OpCategory.REDUCTION: "reduction", OpCategory.LINEAR_ALGEBRA: "linear_algebra",
        OpCategory.STRUCTURAL: "structural", OpCategory.PARAMETERIZED: "linear_algebra",
        OpCategory.SEQUENCE: "sequence", OpCategory.FREQUENCY: "frequency",
        OpCategory.MATH_SPACE: "math_space", OpCategory.FUNCTIONAL: "functional",
    }
    
    ICON_MAP = {"math": "calculator", "reduction": "bar-chart", "linear_algebra": "grid", 
                "structural": "layers", "sequence": "activity", "frequency": "wind", 
                "math_space": "box", "functional": "framer"}

    for name, op in PRIMITIVE_REGISTRY.items():
        if name == "input": continue
        cat = CAT_MAP.get(op.category, "other")
        inputs = [{"name": "x", "dtype": "tensor"}] if op.n_inputs == 1 else \
                 ([{"name": "a", "dtype": "tensor"}, {"name": "b", "dtype": "tensor"}] if op.n_inputs == 2 else [])
        
        params = {}
        for key in op.config_keys:
            p_type = "float" if any(k in key for k in ["prob", "scale", "damping"]) else \
                     ("enum" if "operator" in key else "integer")
            params[key] = {"type": p_type, "default": None, "description": f"Parameter {key}"}
            if p_type == "enum": params[key]["options"] = [">", "<", ">=", "<=", "==", "!="]
        
        if "out_dim" in op.config_keys:
            params["out_dim"]["default"] = 256

        components.append({
            "id": f"{cat}/{name}", "name": name.replace("_", " ").title(), "category": cat,
            "description": op.description, "inputs": inputs, "outputs": [{"name": "y", "dtype": "tensor"}],
            "params_schema": params, "icon": ICON_MAP.get(cat, "circle"),
            "performance": {"has_params": op.has_params, "param_formula": op.param_formula,
                            "preserves_gradient": op.preserves_gradient, "numerically_risky": op.numerically_risky}
        })
    return components

def generate_python_module(workflow_json: Dict[str, Any]) -> str:
    """Generate standalone PyTorch module code for a workflow."""
    from research.synthesis.serializer import graph_to_python
    graph = workflow_to_computation_graph(workflow_json)
    return graph_to_python(graph)
