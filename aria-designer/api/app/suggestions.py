from typing import List, Dict, Any
from .database import list_components

def suggest_components(workflow: Dict[str, Any], prompt: str | None = None) -> List[Dict[str, Any]]:
    """
    Suggest next components based on the current graph state.
    Returns a list of component manifests to suggest.
    """
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    
    # Simple heuristic: look at leaf nodes (nodes with no outgoing edges)
    # and suggest compatible components.
    
    node_ids = {n["id"] for n in nodes}
    source_ids = {e["source"] for e in edges}
    leaf_nodes = [n for n in nodes if n["id"] not in source_ids]
    
    suggestions = []
    
    # Get all approved components
    all_components = list_components(status="approved")
    # Map by category for easy access
    by_category = {}
    for c in all_components:
        by_category.setdefault(c["category"], []).append(c)
        
    if not leaf_nodes:
        # Empty graph? Suggest Input
        input_comps = by_category.get("io", [])
        for c in input_comps:
            if "input" in c["name"].lower():
                suggestions.append(_make_suggestion(c, "Start with an input node."))
        prompt_boosted = _suggest_from_prompt(by_category, all_components, prompt)
        if prompt_boosted:
            suggestions.extend(prompt_boosted)
        return _dedupe_suggestions(suggestions)[:5]

    prompt_boosted = _suggest_from_prompt(by_category, all_components, prompt)
    if prompt_boosted:
        suggestions.extend(prompt_boosted)

    for node in leaf_nodes:
        comp_type = node["component_type"] # e.g. "math/relu" or just "relu"
        
        # Heuristics
        if "input" in comp_type:
            # Suggest Linear, Conv, or Math
            suggestions.extend(_suggest_category(by_category, "linear_algebra", "Add a linear layer."))
            suggestions.extend(_suggest_category(by_category, "math", "Apply an elementwise operation."))
            
        elif "linear" in comp_type:
            # Suggest Activation
            suggestions.extend(_suggest_category(by_category, "math", "Add an activation function."))
            # Suggest Norm
            suggestions.extend(_suggest_category(by_category, "normalization", "Normalize the output."))
            
        elif "relu" in comp_type or "gelu" in comp_type or "silu" in comp_type:
            # Suggest Linear (MLP)
            suggestions.extend(_suggest_category(by_category, "linear_algebra", "Project to a new dimension."))
            # Suggest Block
            suggestions.extend(_suggest_category(by_category, "blocks", "Add a transformer block."))
            
        elif "norm" in comp_type:
            # Suggest Output or Attention
            suggestions.extend(_suggest_category(by_category, "mixing", "Add attention."))
            
    return _dedupe_suggestions(suggestions)[:5]


def _dedupe_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for s in suggestions:
        unique[s["component"]["id"]] = s
    return list(unique.values())


def _suggest_from_prompt(by_cat: Dict[str, List[Dict[str, Any]]], all_components: List[Dict[str, Any]], prompt: str | None) -> List[Dict[str, Any]]:
    if not prompt:
        return []

    lower = prompt.lower()
    out: List[Dict[str, Any]] = []

    has_dataflow_focus = any(k in lower for k in [
        "data/control",
        "data flow",
        "join",
        "filter",
        "schema",
        "columns",
        "dataset",
        "hygiene",
    ])

    if has_dataflow_focus:
        out.extend(_suggest_component_ids(
            all_components,
            ["join", "dataset_filter", "select_columns"],
            "Optimize data/control flow by tightening joins, filters, and schema-column hygiene.",
        ))
        out.extend(_suggest_category(
            by_cat,
            "control_flow",
            "Add explicit control-flow guards to keep data/control routing predictable.",
        ))
        out.extend(_suggest_category(
            by_cat,
            "data_transform",
            "Refine dataset transforms for cleaner schema handling and deterministic filtering.",
        ))

    if "join" in lower and not has_dataflow_focus:
        out.extend(_suggest_component_ids(
            all_components,
            ["join"],
            "Add/adjust join nodes for explicit key-based dataset merging.",
        ))
    if "filter" in lower and not has_dataflow_focus:
        out.extend(_suggest_component_ids(
            all_components,
            ["dataset_filter"],
            "Add filter nodes to enforce row-level quality gates early.",
        ))
    if "schema" in lower or "column" in lower:
        out.extend(_suggest_component_ids(
            all_components,
            ["select_columns"],
            "Use schema-aware column selection to avoid downstream mismatches.",
        ))

    return _dedupe_suggestions(out)


def _suggest_component_ids(all_components: List[Dict[str, Any]], component_ids: List[str], reason: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    targets = {cid.lower() for cid in component_ids}
    for comp in all_components:
        cid = str(comp.get("id", "")).lower()
        if cid in targets:
            out.append(_make_suggestion(comp, reason))
    return out

def _suggest_category(by_cat, category, reason):
    res = []
    for c in by_cat.get(category, []):
        res.append(_make_suggestion(c, reason))
    return res

def _make_suggestion(component, reason):
    return {
        "component": component,
        "reason": reason,
        "action": "add_node"
    }
