"""
Constraint visualization — identify incompatible components for a given graph.

Given a current workflow, computes which palette components would be
incompatible if added, and why. This powers the UI's grayed-out palette
items and tooltip explanations.
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_manifest(component_id: str, components_dir: str) -> Optional[Dict]:
    """Load a component's manifest.yaml."""
    for category in os.listdir(components_dir):
        candidate = os.path.join(components_dir, category, component_id, "manifest.yaml")
        if os.path.isfile(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
    return None


def _collect_tags(manifest: Dict) -> Set[str]:
    """Extract constraint tags from a manifest."""
    tags = set()
    for t in manifest.get("tags", []):
        tags.add(t)
    category = manifest.get("category", "")
    if category:
        tags.add(f"cat:{category}")
    return tags


def check_compatibility(
    workflow: Dict[str, Any],
    candidate_id: str,
    components_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Check if adding a candidate component to a workflow would violate constraints.

    Returns:
        {
            "compatible": bool,
            "reasons": [str],  # human-readable reasons if incompatible
            "severity": "ok" | "warning" | "error"
        }
    """
    if components_dir is None:
        components_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "components")
        )

    # Load candidate manifest
    candidate_manifest = _load_manifest(candidate_id, components_dir)
    if candidate_manifest is None:
        return {"compatible": True, "reasons": [], "severity": "ok"}

    candidate_incompat = set(candidate_manifest.get("constraints", {}).get("incompatible_with", []))
    candidate_requires = set(candidate_manifest.get("constraints", {}).get("requires", []))
    candidate_tags = _collect_tags(candidate_manifest)

    # Collect all existing component types and their tags
    existing_types = set()
    existing_tags = set()
    for node in workflow.get("nodes", []):
        ct = node.get("component_type", "")
        existing_types.add(ct)
        if ct in ("graph_input", "graph_output"):
            continue
        manifest = _load_manifest(ct, components_dir)
        if manifest:
            existing_tags.update(_collect_tags(manifest))

    reasons = []

    # Check: candidate declares incompatibility with existing tags
    for tag in candidate_incompat:
        if tag in existing_tags or tag in existing_types:
            reasons.append(f"'{candidate_id}' is incompatible with '{tag}' (already in graph)")

    # Check: existing components declare incompatibility with candidate tags
    for node in workflow.get("nodes", []):
        ct = node.get("component_type", "")
        if ct in ("graph_input", "graph_output"):
            continue
        manifest = _load_manifest(ct, components_dir)
        if not manifest:
            continue
        node_incompat = set(manifest.get("constraints", {}).get("incompatible_with", []))
        for tag in node_incompat:
            if tag in candidate_tags or tag == candidate_id:
                reasons.append(f"Existing '{ct}' is incompatible with '{candidate_id}'")

    # Check: candidate requires certain components that are missing
    for req in candidate_requires:
        if req not in existing_types and req not in existing_tags:
            reasons.append(f"'{candidate_id}' requires '{req}' (not in graph)")

    # Structural constraints (hard-coded common knowledge)
    reasons.extend(_check_structural_constraints(existing_types, candidate_id, candidate_manifest))

    if reasons:
        # Deduplicate
        reasons = list(dict.fromkeys(reasons))
        return {"compatible": False, "reasons": reasons, "severity": "error"}

    return {"compatible": True, "reasons": [], "severity": "ok"}


def _check_structural_constraints(
    existing_types: Set[str],
    candidate_id: str,
    candidate_manifest: Dict,
) -> List[str]:
    """Hard-coded structural constraints for common architecture patterns."""
    reasons = []

    # Duplicate IO nodes
    if candidate_id == "graph_input" and "graph_input" in existing_types:
        reasons.append("Only one graph_input node is allowed")
    if candidate_id == "graph_output" and "graph_output" in existing_types:
        reasons.append("Only one graph_output node is allowed")

    return reasons


def compute_palette_constraints(
    workflow: Dict[str, Any],
    component_ids: List[str],
    components_dir: Optional[str] = None,
    selected_node_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute compatibility status for every component in the palette."""
    result = {}
    
    # 1. Base compatibility for all
    for cid in component_ids:
        result[cid] = check_compatibility(workflow, cid, components_dir)
        result[cid]["suggested"] = False

    # 2. Context-aware suggestions if a node is selected
    if selected_node_id:
        selected_node = next((n for n in workflow.get("nodes", []) if n["id"] == selected_node_id), None)
        if selected_node:
            ctype = selected_node.get("component_type", "").split("/")[-1]
            
            # Heuristic suggestions for "likely next" nodes
            suggestions = []
            if "linear" in ctype or "proj" in ctype:
                suggestions = ["relu", "gelu", "silu", "rmsnorm"]
            elif "matmul" in ctype:
                suggestions = ["softmax_last", "softmax_seq", "add"]
            elif "input" in ctype:
                suggestions = ["linear_proj", "rmsnorm", "rope"]
            elif "attention" in ctype:
                suggestions = ["linear_proj", "add"]
            elif "norm" in ctype:
                suggestions = ["linear_proj", "softmax_attention", "state_space"]
            
            for s in suggestions:
                # Find the full component ID in the palette that matches this short name
                for cid in result:
                    if cid.endswith(f"/{s}") or cid == s:
                        result[cid]["suggested"] = True

    return result
