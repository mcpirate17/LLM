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

    # Direct component name matching from prompt
    if prompt:
        name_matched = _suggest_by_name(all_components, prompt)
        if name_matched:
            suggestions.extend(name_matched)

    # Collect all component types for graph-level analysis
    all_types = {str(n.get("component_type", "")).split("/")[-1].lower() for n in nodes}

    # Look at leaf nodes; if the only leaf is graph_output, use its predecessors
    analysis_nodes = leaf_nodes
    if all("output" in str(n.get("component_type", "")) for n in leaf_nodes) and edges:
        target_ids = {e["target"] for e in edges if e["target"] in {n["id"] for n in leaf_nodes}}
        pred_ids = {e["source"] for e in edges if e["target"] in target_ids}
        analysis_nodes = [n for n in nodes if n["id"] in pred_ids] or leaf_nodes

    for node in analysis_nodes:
        comp_type = str(node.get("component_type", "")).split("/")[-1].lower()

        if "input" in comp_type:
            suggestions.extend(_suggest_category(by_category, "linear_algebra", "Add a linear layer."))
            suggestions.extend(_suggest_category(by_category, "math", "Apply an elementwise operation."))

        elif "linear" in comp_type:
            suggestions.extend(_suggest_category(by_category, "math", "Add an activation function."))
            suggestions.extend(_suggest_category(by_category, "normalization", "Normalize the output."))

        elif "relu" in comp_type or "gelu" in comp_type or "silu" in comp_type:
            suggestions.extend(_suggest_category(by_category, "linear_algebra", "Project to a new dimension."))
            suggestions.extend(_suggest_category(by_category, "blocks", "Add a transformer block."))

        elif "norm" in comp_type:
            suggestions.extend(_suggest_category(by_category, "mixing", "Add attention."))

    # Graph-level gap analysis: suggest what's missing for a strong architecture
    has_norm = any(t in all_types for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre"))
    has_attn = any(t in all_types for t in ("softmax_attention", "linear_attention", "graph_attention"))

    # Stability/brittleness analysis: always prioritize normalization if missing
    is_stability_prompt = prompt and any(
        kw in prompt.lower() for kw in ("brittle", "gradient", "unstable", "nan", "explod", "stabil", "zero grad")
    )
    if is_stability_prompt and not has_norm:
        suggestions.insert(0, _make_suggestion(
            next((c for c in all_components if c.get("id") == "rmsnorm"), None) or
            next((c for c in by_category.get("normalization", [])), {}),
            "Critical: add normalization to prevent exploding logits and zero gradients. "
            "Without normalization before the output head, magnitudes grow unchecked, "
            "causing softmax saturation and training failure."
        ))
    elif is_stability_prompt and has_norm:
        # Has norm but still brittle — suggest residual connections
        if "add" not in all_types:
            suggestions.insert(0, _make_suggestion(
                next((c for c in all_components if c.get("id") == "add"), None) or {},
                "Add residual (skip) connections to stabilize gradient flow. "
                "Without skip connections, deep graphs suffer from vanishing gradients."
            ))

    if not suggestions:
        if not has_norm:
            suggestions.extend(_suggest_category(by_category, "normalization", "Add normalization for training stability."))
        if not has_attn and len(nodes) > 3:
            suggestions.extend(_suggest_category(by_category, "mixing", "Add a mixing/attention layer for richer representations."))

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


def _suggest_by_name(all_components: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    """Match component IDs mentioned in the prompt."""
    lower = prompt.lower()
    # Tokenize prompt into words
    words = set(lower.replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split())
    out: List[Dict[str, Any]] = []
    # Skip common English words that clash with op names
    skip_words = {
        "add", "sub", "exp", "log", "cos", "sin", "abs", "neg", "sign", "sort", "mul", "div",
        "input", "output", "split", "join", "loop", "none", "filter", "from", "with", "that",
        "this", "then", "have", "will", "been", "into", "over", "each", "more", "like", "make",
        "graph", "block", "dense", "scale", "speed", "model", "layer", "point", "batch", "step",
        "gate", "head", "base", "loss", "task", "test", "down", "last", "first",
        "quality", "stability", "benchmark", "target", "gaps", "novelty", "preserving",
        "propose", "patch", "closes", "downstream", "while",
    }
    for c in all_components:
        cid = str(c.get("id", "")).lower()
        if cid in skip_words:
            continue
        # Only match against component ID, not full name (names contain English sentences)
        for w in words:
            if len(w) < 5:
                continue
            if w in skip_words:
                continue
            if cid.startswith(w) or w == cid:
                out.append(_make_suggestion(c, f"Matched '{w}' → {cid}"))
                break
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
