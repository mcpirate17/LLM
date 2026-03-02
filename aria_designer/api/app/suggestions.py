from typing import List, Dict, Any, Optional, Set
from .database import list_components

_OP_ALIAS = {
    "token_merging": "token_merge",
    "token_merge": "token_merge",
    "rmsnorm": "rmsnorm_pre",
    "layernorm": "layernorm_pre",
}


def _canon_op_name(name: str) -> str:
    token = str(name or "").strip().lower()
    return _OP_ALIAS.get(token, token)


def suggest_components(
    workflow: Dict[str, Any],
    prompt: str | None = None,
    research_signals: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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

    node_types = [str(n.get("component_type", "")) for n in nodes]
    categories_present = {t.split("/", 1)[0] for t in node_types if "/" in t}
    graph_size = len(nodes)
    prompt_lower = (prompt or "").lower()
    used_component_ids = {
        str(n.get("component_type", "")).split("/")[-1].lower()
        for n in nodes
        if str(n.get("component_type", "")).strip()
    }
    ctx: Dict[str, Any] = {
        "graph_size": graph_size,
        "categories_present": categories_present,
        "prompt_lower": prompt_lower,
        "used_component_ids": used_component_ids,
        "all_types": set(),
        "is_stability_prompt": False,
        "missing_norm": False,
        "missing_attn": False,
        "research_signals": research_signals if isinstance(research_signals, dict) else {},
    }
        
    if not leaf_nodes:
        # Empty graph? Suggest Input
        input_comps = by_category.get("io", [])
        for c in input_comps:
            if "input" in c["name"].lower():
                suggestions.append(_make_suggestion(
                    c,
                    "Start with an input node.",
                    score=0.98,
                    evidence=["Canvas is empty", "No graph input nodes found"],
                    context=ctx,
                ))
        prompt_boosted = _suggest_from_prompt(by_category, all_components, prompt, context=ctx)
        if prompt_boosted:
            suggestions.extend(prompt_boosted)
        return _dedupe_suggestions(suggestions)[:5]

    prompt_boosted = _suggest_from_prompt(by_category, all_components, prompt, context=ctx)
    if prompt_boosted:
        suggestions.extend(prompt_boosted)

    # Direct component name matching from prompt
    if prompt:
        name_matched = _suggest_by_name(all_components, prompt, context=ctx)
        if name_matched:
            suggestions.extend(name_matched)

    # Collect all component types for graph-level analysis
    all_types = {str(n.get("component_type", "")).split("/")[-1].lower() for n in nodes}
    ctx["all_types"] = set(all_types)

    # Look at leaf nodes; if the only leaf is graph_output, use its predecessors
    analysis_nodes = leaf_nodes
    if all("output" in str(n.get("component_type", "")) for n in leaf_nodes) and edges:
        target_ids = {e["target"] for e in edges if e["target"] in {n["id"] for n in leaf_nodes}}
        pred_ids = {e["source"] for e in edges if e["target"] in target_ids}
        analysis_nodes = [n for n in nodes if n["id"] in pred_ids] or leaf_nodes

    for node in analysis_nodes:
        comp_type = str(node.get("component_type", "")).split("/")[-1].lower()

        if "input" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "linear_algebra", "Add a linear layer.",
                score=0.72, evidence=[f"Leaf node '{comp_type}' usually feeds projection"], context=ctx
            ))
            suggestions.extend(_suggest_category(
                by_category, "math", "Apply an elementwise operation.",
                score=0.66, evidence=[f"Leaf node '{comp_type}' benefits from nonlinearity"], context=ctx
            ))

        elif "linear" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "math", "Add an activation function.",
                score=0.78, evidence=["Linear layer at leaf", "Activation not yet applied downstream"], context=ctx
            ))
            suggestions.extend(_suggest_category(
                by_category, "normalization", "Normalize the output.",
                score=0.74, evidence=["Linear output can drift in scale"], context=ctx, exclude_ids={"no_norm", "none"}
            ))

        elif "relu" in comp_type or "gelu" in comp_type or "silu" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "linear_algebra", "Project to a new dimension.",
                score=0.70, evidence=["Activation leaf found", "Projection expands modeling capacity"], context=ctx
            ))
            suggestions.extend(_suggest_category(
                by_category, "blocks", "Add a transformer block.",
                score=0.57, evidence=["Activation stage can be wrapped in reusable block"], context=ctx
            ))

        elif "norm" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "mixing", "Add attention.",
                score=0.69, evidence=["Normalized features are ready for mixing"], context=ctx
            ))

    # Graph-level gap analysis: suggest what's missing for a strong architecture
    has_norm = any(t in all_types for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre"))
    has_attn = any(t in all_types for t in ("softmax_attention", "linear_attention", "graph_attention"))

    # Stability/brittleness analysis: always prioritize normalization if missing
    is_stability_prompt = prompt and any(
        kw in prompt.lower() for kw in ("brittle", "gradient", "unstable", "nan", "explod", "stabil", "zero grad")
    )
    ctx["is_stability_prompt"] = bool(is_stability_prompt)
    ctx["missing_norm"] = not has_norm
    ctx["missing_attn"] = not has_attn
    if is_stability_prompt and not has_norm:
        suggestions.insert(0, _make_suggestion(
            next((c for c in all_components if c.get("id") == "rmsnorm"), None) or
            next((c for c in by_category.get("normalization", [])), {}),
            "Critical: add normalization to prevent exploding logits and zero gradients. "
            "Without normalization before the output head, magnitudes grow unchecked, "
            "causing softmax saturation and training failure.",
            score=0.99,
            evidence=["Prompt indicates stability concern", "No normalization operators detected"],
            context=ctx,
        ))
    elif is_stability_prompt and has_norm:
        # Has norm but still brittle — suggest residual connections
        if "add" not in all_types:
            suggestions.insert(0, _make_suggestion(
                next((c for c in all_components if c.get("id") == "add"), None) or {},
                "Add residual (skip) connections to stabilize gradient flow. "
                "Without skip connections, deep graphs suffer from vanishing gradients.",
                score=0.94,
                evidence=["Prompt indicates gradient instability", "No residual add operator found"],
                context=ctx,
            ))

    if not suggestions:
        if not has_norm:
            suggestions.extend(_suggest_category(
                by_category, "normalization", "Add normalization for training stability.",
                score=0.76, evidence=["No normalization category detected in graph"], context=ctx, exclude_ids={"no_norm", "none"}
            ))
        if not has_attn and len(nodes) > 3:
            suggestions.extend(_suggest_category(
                by_category, "mixing", "Add a mixing/attention layer for richer representations.",
                score=0.72, evidence=["Graph depth suggests representation bottleneck", "No attention-like mixing found"], context=ctx
            ))

    if "io" not in categories_present and graph_size > 0:
        suggestions.extend(_suggest_component_ids(
            all_components,
            ["output_head"],
            "Consider adding an explicit output head for clearer endpoint semantics.",
            score=0.65,
            evidence=["Graph has no explicit io/output head node"],
            context=ctx,
        ))

    ranked = sorted(_dedupe_suggestions(suggestions), key=lambda s: s.get("score", 0), reverse=True)
    return ranked[:5]


def _dedupe_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for s in suggestions:
        cid = s["component"]["id"]
        if cid not in unique or s.get("score", 0) > unique[cid].get("score", 0):
            unique[cid] = s
    return list(unique.values())


def _suggest_from_prompt(
    by_cat: Dict[str, List[Dict[str, Any]]],
    all_components: List[Dict[str, Any]],
    prompt: str | None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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
            score=0.92,
            evidence=["Prompt requests data/control optimization"],
            context=context,
        ))
        out.extend(_suggest_category(
            by_cat,
            "control_flow",
            "Add explicit control-flow guards to keep data/control routing predictable.",
            score=0.79,
            evidence=["Prompt includes control/data-flow intent"],
            context=context,
        ))
        out.extend(_suggest_category(
            by_cat,
            "data_transform",
            "Refine dataset transforms for cleaner schema handling and deterministic filtering.",
            score=0.81,
            evidence=["Prompt includes schema/filter language"],
            context=context,
        ))

    if "join" in lower and not has_dataflow_focus:
        out.extend(_suggest_component_ids(
            all_components,
            ["join"],
            "Add/adjust join nodes for explicit key-based dataset merging.",
            score=0.84,
            evidence=["Prompt contains join intent"],
            context=context,
        ))
    if "filter" in lower and not has_dataflow_focus:
        out.extend(_suggest_component_ids(
            all_components,
            ["dataset_filter"],
            "Add filter nodes to enforce row-level quality gates early.",
            score=0.83,
            evidence=["Prompt contains filter intent"],
            context=context,
        ))
    if "schema" in lower or "column" in lower:
        out.extend(_suggest_component_ids(
            all_components,
            ["select_columns"],
            "Use schema-aware column selection to avoid downstream mismatches.",
            score=0.87,
            evidence=["Prompt references schema/columns"],
            context=context,
        ))

    return _dedupe_suggestions(out)


def _suggest_by_name(
    all_components: List[Dict[str, Any]],
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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
                out.append(_make_suggestion(
                    c,
                    f"Matched '{w}' → {cid}",
                    score=0.88,
                    evidence=[f"Prompt token '{w}' matched component id"],
                    context=context,
                ))
                break
    return _dedupe_suggestions(out)


def _suggest_component_ids(
    all_components: List[Dict[str, Any]],
    component_ids: List[str],
    reason: str,
    score: float = 0.65,
    evidence: List[str] | None = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    targets = {cid.lower() for cid in component_ids}
    for comp in all_components:
        cid = str(comp.get("id", "")).lower()
        if cid in targets:
            out.append(_make_suggestion(comp, reason, score=score, evidence=evidence, context=context))
    return out

def _suggest_category(
    by_cat,
    category,
    reason,
    score: float = 0.6,
    evidence: List[str] | None = None,
    context: Optional[Dict[str, Any]] = None,
    exclude_ids: Optional[Set[str]] = None,
):
    res: List[Dict[str, Any]] = []
    excluded = {str(x).lower() for x in (exclude_ids or set())}
    for c in by_cat.get(category, []):
        cid = str(c.get("id", "")).lower()
        if cid in excluded:
            continue
        res.append(_make_suggestion(c, reason, score=score, evidence=evidence, context=context))
    return sorted(res, key=lambda s: s.get("score", 0), reverse=True)

def _score_adjustment(
    component: Dict[str, Any],
    reason: str,
    evidence: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> float:
    cid = str(component.get("id", "")).lower()
    category = str(component.get("category", "")).lower()
    ctx = context or {}
    prompt_lower = str(ctx.get("prompt_lower") or "")
    used_ids = set(ctx.get("used_component_ids") or set())
    all_types = set(ctx.get("all_types") or set())
    is_stability_prompt = bool(ctx.get("is_stability_prompt"))
    missing_norm = bool(ctx.get("missing_norm"))
    missing_attn = bool(ctx.get("missing_attn"))
    research = ctx.get("research_signals") if isinstance(ctx.get("research_signals"), dict) else {}

    delta = 0.0
    delta += min(len(evidence), 3) * 0.01

    if cid in used_ids:
        delta -= 0.04

    if category == "normalization":
        if "rmsnorm" in cid:
            delta += 0.05
        elif "layernorm" in cid:
            delta += 0.04
        elif "group_norm" in cid or "dynamic_norm" in cid:
            delta += 0.03
        elif cid in {"no_norm", "none"} or "no_norm" in cid:
            delta -= 0.25
    if is_stability_prompt and category == "normalization":
        delta += 0.03

    if missing_norm and category == "normalization":
        delta += 0.02
    if missing_attn and category == "mixing":
        delta += 0.03

    if any(tok in prompt_lower for tok in ("join", "filter", "schema", "column")):
        if cid in {"join", "dataset_filter", "select_columns"}:
            delta += 0.04
    if any(tok in prompt_lower for tok in ("stabil", "nan", "gradient", "brittle", "explod")):
        if category == "normalization" or cid == "add":
            delta += 0.04
    if any(tok in prompt_lower for tok in ("compress", "flop", "latency", "efficien")):
        if cid in {"low_rank", "bottleneck_proj", "token_merging", "progressive_compression_gate"}:
            delta += 0.04

    if cid in {"softmax_attention", "linear_attention", "graph_attention"} and all_types:
        if cid in all_types:
            delta -= 0.02

    if "normalization" in reason.lower() and (cid in {"no_norm", "none"} or "no_norm" in cid):
        delta -= 0.3

    # Research-driven priors (cross-system signals from research analytics).
    canon_cid = _canon_op_name(cid)
    op_priors = research.get("op_priors") if isinstance(research, dict) else None
    if isinstance(op_priors, list):
        op_rate = None
        for row in op_priors:
            op_name = _canon_op_name(str((row or {}).get("op_name") or ""))
            if op_name == canon_cid:
                try:
                    op_rate = float((row or {}).get("s1_rate"))
                except Exception:
                    op_rate = None
                break
        if op_rate is not None:
            # Reward high-S1 operators and penalize low-S1 operators.
            delta += (op_rate - 0.5) * 0.2

    toxic_ops = research.get("toxic_ops") if isinstance(research, dict) else None
    if isinstance(toxic_ops, list):
        toxic_set = {_canon_op_name(str(x)) for x in toxic_ops}
        if canon_cid in toxic_set:
            delta -= 0.12

    comp_techniques = research.get("compression_techniques") if isinstance(research, dict) else None
    if isinstance(comp_techniques, list) and any(tok in prompt_lower for tok in ("compress", "flop", "latency", "efficien")):
        low_rank_like = {"low_rank", "bottleneck", "grouped_linear", "structured_sparse", "shared_basis"}
        if canon_cid in low_rank_like and any(canon_cid in str(t).lower() for t in comp_techniques):
            delta += 0.06

    insights = research.get("insights") if isinstance(research, dict) else None
    if isinstance(insights, list):
        stability_keywords = ("stability", "exploding", "nan", "gradient")
        if any(tok in prompt_lower for tok in stability_keywords):
            for ins in insights[:40]:
                category = str((ins or {}).get("category") or "").lower()
                content = str((ins or {}).get("content") or "").lower()
                if category in {"failure_mode", "success_factor"} and any(k in content for k in stability_keywords):
                    if category == "success_factor" and str(component.get("category", "")).lower() == "normalization":
                        delta += 0.02
                    break

    return delta


def _make_suggestion(
    component,
    reason,
    score: float = 0.6,
    evidence: List[str] | None = None,
    context: Optional[Dict[str, Any]] = None,
):
    evidence_list = list(evidence or [])
    adjusted_score = float(score) + _score_adjustment(component, reason, evidence_list, context=context)
    safe_score = max(0.0, min(1.0, adjusted_score))
    return {
        "component": component,
        "reason": reason,
        "action": "add_node",
        "score": safe_score,
        "evidence": evidence_list,
    }
