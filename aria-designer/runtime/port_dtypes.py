from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def find_unsupported_edge_dtype_pairings(
    workflow: Dict[str, Any],
    get_component_manifest: Callable[[str], Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return manifest-port dtype mismatches for workflow edges.

    Rule: edge source and target port dtypes must be identical.
    """
    nodes_by_id = {
        str(node.get("id")): node
        for node in workflow.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }

    issues: List[Dict[str, Any]] = []
    for edge in workflow.get("edges", []):
        if not isinstance(edge, dict):
            continue

        source_id = str(edge.get("source", ""))
        target_id = str(edge.get("target", ""))
        source_port_name = str(edge.get("source_port", ""))
        target_port_name = str(edge.get("target_port", ""))

        src_node = nodes_by_id.get(source_id)
        tgt_node = nodes_by_id.get(target_id)
        if src_node is None or tgt_node is None:
            continue

        src_comp = get_component_manifest(str(src_node.get("component_type", "")))
        tgt_comp = get_component_manifest(str(tgt_node.get("component_type", "")))
        if not isinstance(src_comp, dict) or not isinstance(tgt_comp, dict):
            continue

        src_port = next(
            (p for p in src_comp.get("outputs", []) if isinstance(p, dict) and p.get("name") == source_port_name),
            None,
        )
        tgt_port = next(
            (p for p in tgt_comp.get("inputs", []) if isinstance(p, dict) and p.get("name") == target_port_name),
            None,
        )
        if src_port is None or tgt_port is None:
            continue

        src_dtype = src_port.get("dtype")
        tgt_dtype = tgt_port.get("dtype")
        if src_dtype == tgt_dtype:
            continue

        # Implicitly compatible pairings (warning, not error)
        _IMPLICIT_COMPAT = {
            ("complex_tensor", "tensor"),   # runtime takes .real
            ("tensor", "complex_tensor"),   # runtime zero-fills imaginary
        }
        if (src_dtype, tgt_dtype) in _IMPLICIT_COMPAT:
            continue

        # P3.13: Add explicit errors for unsupported edge type pairings.
        specific_error = None
        if src_dtype == "dataset" and tgt_dtype == "tensor":
            specific_error = "Dataset cannot be connected directly to a tensor port; use a data_transform component first."
        elif src_dtype == "list" and tgt_dtype == "scalar":
            specific_error = "List cannot be connected to a scalar port; use a reduction or indexing op."

        message = (
            f"Unsupported edge dtype pairing on edge {edge.get('id', '')}: "
            f"{source_id}.{source_port_name} ({src_dtype}) -> "
            f"{target_id}.{target_port_name} ({tgt_dtype})."
        )
        if specific_error:
            message = f"{message} {specific_error}"
        else:
            message = f"{message} Supported pairings currently require matching source/target dtypes."

        issues.append(
            {
                "edge_id": str(edge.get("id", "")),
                "source": source_id,
                "target": target_id,
                "source_port": source_port_name,
                "target_port": target_port_name,
                "source_dtype": src_dtype,
                "target_dtype": tgt_dtype,
                "message": message,
            }
        )

    return issues
