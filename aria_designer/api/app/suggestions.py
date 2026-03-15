from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from .component_identity import canonicalize_component_id, component_leaf
from .database import list_components
from .intent_parser import compute_insertion_point
from .research_signals import fetch_leaderboard_top_entries

logger = logging.getLogger(__name__)


def _canon_leaf(name: str) -> str:
    """Resolve aliases to canonical leaf name for matching."""
    canonical = canonicalize_component_id(name)
    return component_leaf(canonical) if canonical else component_leaf(name)


def _build_component_indexes(
    all_components: List[Dict[str, Any]],
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    input_components: List[Dict[str, Any]] = []
    for component in all_components:
        category = str(component.get("category", ""))
        component_id = str(component.get("id", "")).lower()
        by_category.setdefault(category, []).append(component)
        if component_id:
            by_id[component_id] = component
        if "input" in str(component.get("name", "")).lower():
            input_components.append(component)
    return by_category, by_id, input_components


def suggest_components(
    workflow: Dict[str, Any],
    prompt: str | None = None,
    research_signals: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Suggest next components based on the current graph state."""
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    source_ids = {e["source"] for e in edges}
    leaf_nodes = [n for n in nodes if n["id"] not in source_ids]

    all_components = list_components(status="approved")
    by_category, by_id, input_components = _build_component_indexes(all_components)

    node_types = [str(n.get("component_type", "")) for n in nodes]
    categories_present = {t.split("/", 1)[0] for t in node_types if "/" in t}
    ctx: Dict[str, Any] = {
        "graph_size": len(nodes),
        "categories_present": categories_present,
        "prompt_lower": (prompt or "").lower(),
        "used_component_ids": {
            str(n.get("component_type", "")).split("/")[-1].lower()
            for n in nodes if str(n.get("component_type", "")).strip()
        },
        "all_types": set(),
        "is_stability_prompt": False,
        "missing_norm": False,
        "missing_attn": False,
        "research_signals": research_signals if isinstance(research_signals, dict) else {},
        "workflow_nodes": nodes,
        "workflow_edges": edges,
        "leaderboard_entries": None,
        "leaderboard_component_sets": None,
        "leaderboard_total_entries": 0,
        "components_by_id": by_id,
        "insertion_hints_by_type": {},
    }

    suggestions: List[Dict[str, Any]] = []

    if not leaf_nodes:
        for component in input_components:
            suggestions.append(_make_suggestion(
                component, "Start with an input node.",
                score=0.98, evidence=["Canvas is empty", "No graph input nodes found"], context=ctx,
            ))
        suggestions.extend(_suggest_from_prompt(by_category, by_id, prompt, context=ctx))
        return _dedupe_suggestions(suggestions)[:5]

    suggestions.extend(_suggest_from_prompt(by_category, by_id, prompt, context=ctx))
    if prompt:
        suggestions.extend(_suggest_by_name(by_id, prompt, context=ctx))

    all_types = {str(n.get("component_type", "")).split("/")[-1].lower() for n in nodes}
    ctx["all_types"] = set(all_types)

    suggestions.extend(_suggest_for_leaf_nodes(leaf_nodes, nodes, edges, by_category, ctx))
    suggestions.extend(_suggest_graph_gaps(all_types, nodes, by_category, by_id, prompt, ctx))

    if "io" not in categories_present and len(nodes) > 0:
        suggestions.extend(_suggest_component_ids(
            by_id, ["output_head"],
            "Consider adding an explicit output head for clearer endpoint semantics.",
            score=0.65, evidence=["Graph has no explicit io/output head node"], context=ctx,
        ))

    ranked = sorted(_dedupe_suggestions(suggestions), key=lambda s: s.get("score", 0), reverse=True)
    return ranked[:5]


def _suggest_for_leaf_nodes(
    leaf_nodes: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    by_category: Dict[str, List[Dict[str, Any]]],
    ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Suggest components based on leaf node types."""
    suggestions: List[Dict[str, Any]] = []

    # If the only leaf is graph_output, use its predecessors
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
                score=0.72, evidence=[f"Leaf node '{comp_type}' usually feeds projection"], context=ctx,
            ))
            suggestions.extend(_suggest_category(
                by_category, "math", "Apply an elementwise operation.",
                score=0.66, evidence=[f"Leaf node '{comp_type}' benefits from nonlinearity"], context=ctx,
            ))
        elif "linear" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "math", "Add an activation function.",
                score=0.78, evidence=["Linear layer at leaf", "Activation not yet applied downstream"], context=ctx,
            ))
            suggestions.extend(_suggest_category(
                by_category, "normalization", "Normalize the output.",
                score=0.74, evidence=["Linear output can drift in scale"], context=ctx, exclude_ids={"no_norm", "none"},
            ))
        elif "relu" in comp_type or "gelu" in comp_type or "silu" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "linear_algebra", "Project to a new dimension.",
                score=0.70, evidence=["Activation leaf found", "Projection expands modeling capacity"], context=ctx,
            ))
            suggestions.extend(_suggest_category(
                by_category, "blocks", "Add a transformer block.",
                score=0.57, evidence=["Activation stage can be wrapped in reusable block"], context=ctx,
            ))
        elif "norm" in comp_type:
            suggestions.extend(_suggest_category(
                by_category, "mixing", "Add attention.",
                score=0.69, evidence=["Normalized features are ready for mixing"], context=ctx,
            ))
    return suggestions


def _suggest_graph_gaps(
    all_types: Set[str],
    nodes: List[Dict[str, Any]],
    by_category: Dict[str, List[Dict[str, Any]]],
    by_id: Dict[str, Dict[str, Any]],
    prompt: str | None,
    ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Suggest missing architectural components (normalization, attention, stability)."""
    suggestions: List[Dict[str, Any]] = []
    has_norm = any(t in all_types for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre"))
    has_attn = any(t in all_types for t in ("softmax_attention", "linear_attention", "graph_attention"))

    is_stability_prompt = prompt and any(
        kw in prompt.lower() for kw in ("brittle", "gradient", "unstable", "nan", "explod", "stabil", "zero grad")
    )
    ctx["is_stability_prompt"] = bool(is_stability_prompt)
    ctx["missing_norm"] = not has_norm
    ctx["missing_attn"] = not has_attn

    if is_stability_prompt and not has_norm:
        suggestions.insert(0, _make_suggestion(
            by_id.get("rmsnorm") or next((c for c in by_category.get("normalization", [])), {}),
            "Critical: add normalization to prevent exploding logits and zero gradients. "
            "Without normalization before the output head, magnitudes grow unchecked, "
            "causing softmax saturation and training failure.",
            score=0.99,
            evidence=["Prompt indicates stability concern", "No normalization operators detected"],
            context=ctx,
        ))
    elif is_stability_prompt and has_norm and "add" not in all_types:
        suggestions.insert(0, _make_suggestion(
            by_id.get("add") or {},
            "Add residual (skip) connections to stabilize gradient flow. "
            "Without skip connections, deep graphs suffer from vanishing gradients.",
            score=0.94,
            evidence=["Prompt indicates gradient instability", "No residual add operator found"],
            context=ctx,
        ))

    if not has_norm:
        suggestions.extend(_suggest_category(
            by_category, "normalization", "Add normalization for training stability.",
            score=0.76, evidence=["No normalization category detected in graph"], context=ctx, exclude_ids={"no_norm", "none"},
        ))
    if not has_attn and len(nodes) > 3:
        suggestions.extend(_suggest_category(
            by_category, "mixing", "Add a mixing/attention layer for richer representations.",
            score=0.72, evidence=["Graph depth suggests representation bottleneck", "No attention-like mixing found"], context=ctx,
        ))
    return suggestions


def _dedupe_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for s in suggestions:
        cid = s["component"]["id"]
        if cid not in unique or s.get("score", 0) > unique[cid].get("score", 0):
            unique[cid] = s
    return list(unique.values())


def _suggest_from_prompt(
    by_cat: Dict[str, List[Dict[str, Any]]],
    by_id: Dict[str, Dict[str, Any]],
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
            by_id,
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
            by_id,
            ["join"],
            "Add/adjust join nodes for explicit key-based dataset merging.",
            score=0.84,
            evidence=["Prompt contains join intent"],
            context=context,
        ))
    if "filter" in lower and not has_dataflow_focus:
        out.extend(_suggest_component_ids(
            by_id,
            ["dataset_filter"],
            "Add filter nodes to enforce row-level quality gates early.",
            score=0.83,
            evidence=["Prompt contains filter intent"],
            context=context,
        ))
    if "schema" in lower or "column" in lower:
        out.extend(_suggest_component_ids(
            by_id,
            ["select_columns"],
            "Use schema-aware column selection to avoid downstream mismatches.",
            score=0.87,
            evidence=["Prompt references schema/columns"],
            context=context,
        ))

    return _dedupe_suggestions(out)


def _suggest_by_name(
    by_id: Dict[str, Dict[str, Any]],
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
    candidate_words = [word for word in words if len(word) >= 5 and word not in skip_words]
    for word in candidate_words:
        component = by_id.get(word)
        if component is not None:
            out.append(_make_suggestion(
                component,
                f"Matched '{word}' → {word}",
                score=0.88,
                evidence=[f"Prompt token '{word}' matched component id"],
                context=context,
            ))
            continue
        for component_id, component in by_id.items():
            if component_id in skip_words:
                continue
            if component_id.startswith(word):
                out.append(_make_suggestion(
                    component,
                    f"Matched '{word}' → {component_id}",
                    score=0.88,
                    evidence=[f"Prompt token '{word}' matched component id"],
                    context=context,
                ))
                break
    return _dedupe_suggestions(out)


def _suggest_component_ids(
    by_id: Dict[str, Dict[str, Any]],
    component_ids: List[str],
    reason: str,
    score: float = 0.65,
    evidence: List[str] | None = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for component_id in component_ids:
        component = by_id.get(component_id.lower())
        if component is not None:
            out.append(_make_suggestion(component, reason, score=score, evidence=evidence, context=context))
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

    delta += _research_score_delta(component, cid, prompt_lower, all_types, research)

    leaderboard_component_sets, leaderboard_total_entries = _get_leaderboard_component_sets(ctx)
    if leaderboard_component_sets:
        lb_delta, _ = _leaderboard_boost(component, leaderboard_component_sets, leaderboard_total_entries)
        delta += lb_delta

    return delta


def _research_score_delta(
    component: Dict[str, Any],
    cid: str,
    prompt_lower: str,
    all_types: Set[str],
    research: Dict[str, Any],
) -> float:
    """Score adjustment from research analytics signals (op priors, pairs, failures)."""
    if not isinstance(research, dict) or not research:
        return 0.0

    delta = 0.0
    canon_cid = _canon_leaf(cid)

    # Op success rate priors
    op_priors = research.get("op_priors")
    if isinstance(op_priors, list):
        for row in op_priors:
            if _canon_leaf(str((row or {}).get("op_name") or "")) == canon_cid:
                try:
                    delta += (float((row or {}).get("s1_rate")) - 0.5) * 0.2
                except Exception:
                    pass
                break

    # Toxic ops penalty
    toxic_ops = research.get("toxic_ops")
    if isinstance(toxic_ops, list):
        if canon_cid in {_canon_leaf(str(x)) for x in toxic_ops}:
            delta -= 0.12

    # Compression technique boost
    comp_techniques = research.get("compression_techniques")
    if isinstance(comp_techniques, list) and any(tok in prompt_lower for tok in ("compress", "flop", "latency", "efficien")):
        low_rank_like = {"low_rank", "bottleneck", "grouped_linear", "structured_sparse", "shared_basis"}
        if canon_cid in low_rank_like and any(canon_cid in str(t).lower() for t in comp_techniques):
            delta += 0.06

    # Insight-driven stability boost
    insights = research.get("insights")
    if isinstance(insights, list):
        stability_keywords = ("stability", "exploding", "nan", "gradient")
        if any(tok in prompt_lower for tok in stability_keywords):
            for ins in insights[:40]:
                cat = str((ins or {}).get("category") or "").lower()
                content = str((ins or {}).get("content") or "").lower()
                if cat in {"failure_mode", "success_factor"} and any(k in content for k in stability_keywords):
                    if cat == "success_factor" and str(component.get("category", "")).lower() == "normalization":
                        delta += 0.02
                    break

    # Op-pair priors: boost components that pair well with existing graph ops
    op_pair_priors = research.get("op_pair_priors")
    if isinstance(op_pair_priors, list) and all_types:
        pair_boost = 0.0
        pair_count = 0
        for pair in op_pair_priors:
            if not isinstance(pair, dict):
                continue
            op_a = _canon_leaf(str(pair.get("op_a") or ""))
            op_b = _canon_leaf(str(pair.get("op_b") or ""))
            try:
                rate = float(pair.get("success_rate"))
            except (TypeError, ValueError):
                continue
            if (op_a == canon_cid and op_b in all_types) or (op_b == canon_cid and op_a in all_types):
                pair_boost += (rate - 0.5) * 0.15
                pair_count += 1
        if pair_count > 0:
            delta += min(0.08, pair_boost / pair_count)

    # Failure risk signatures
    failure_sigs = research.get("failure_risk_signatures")
    if isinstance(failure_sigs, list):
        for sig in failure_sigs:
            if not isinstance(sig, dict):
                continue
            if canon_cid in str(sig.get("pattern") or "").lower():
                try:
                    delta -= min(0.1, float(sig.get("penalty") or 0.0) * 0.1)
                except (TypeError, ValueError):
                    pass
                break

    return delta


def _leaderboard_boost(
    component: Dict[str, Any],
    component_sets: List[Set[str]],
    total_entries: int,
) -> tuple[float, str | None]:
    cid = _canon_leaf(str((component or {}).get("id") or ""))
    if not cid or not component_sets or total_entries <= 0:
        return 0.0, None
    count = sum(1 for component_ids in component_sets if cid in component_ids)
    if count == 0:
        return 0.0, None
    delta = min(0.06, count * 0.01)
    evidence_line = f"Used in {count} of top {total_entries} architectures"
    return delta, evidence_line


def _get_leaderboard_component_sets(context: Optional[Dict[str, Any]]) -> tuple[List[Set[str]], int]:
    ctx = context if isinstance(context, dict) else {}
    component_sets = ctx.get("leaderboard_component_sets")
    total_entries = int(ctx.get("leaderboard_total_entries") or 0)
    if isinstance(component_sets, list) and total_entries > 0:
        return component_sets, total_entries

    entries = ctx.get("leaderboard_entries")
    if not isinstance(entries, list):
        entries = fetch_leaderboard_top_entries() or []
        ctx["leaderboard_entries"] = entries

    parsed_sets = [_extract_component_ids_from_entry(entry) for entry in entries if isinstance(entry, dict)]
    ctx["leaderboard_component_sets"] = parsed_sets
    ctx["leaderboard_total_entries"] = len(entries)
    return parsed_sets, len(entries)


def _extract_component_ids_from_entry(entry: Dict[str, Any]) -> Set[str]:
    precomputed = entry.get("_component_ids")
    if isinstance(precomputed, list):
        return {_canon_leaf(str(token)) for token in precomputed if str(token).strip()}

    graph_json = entry.get("graph_json") or entry.get("_graph_json")
    if isinstance(graph_json, str) and graph_json:
        try:
            graph = json.loads(graph_json)
        except Exception:
            logger.debug("Failed to parse graph_json in leaderboard entry", exc_info=True)
            graph = None
        if isinstance(graph, dict):
            nodes = graph.get("nodes")
            if isinstance(nodes, list):
                return {
                    _canon_leaf(str(node.get("component_type", "")).split("/")[-1])
                    for node in nodes
                    if isinstance(node, dict) and str(node.get("component_type", "")).strip()
                }

    text = str(entry.get("program_text") or entry.get("architecture_desc") or "").lower()
    if not text:
        return set()
    token_counts = Counter(_canon_leaf(token) for token in text.replace("/", " ").replace(",", " ").split())
    return {token for token, count in token_counts.items() if token and count > 0}


def _make_suggestion(
    component: Dict[str, Any],
    reason: str,
    score: float = 0.6,
    evidence: List[str] | None = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_list = list(evidence or [])
    adjusted_score = float(score) + _score_adjustment(component, reason, evidence_list, context=context)
    safe_score = max(0.0, min(1.0, adjusted_score))
    # Enrich evidence with research signal summaries
    ctx = context or {}
    research = ctx.get("research_signals") if isinstance(ctx.get("research_signals"), dict) else {}
    if isinstance(research, dict) and research:
        cid = _canon_leaf(str((component or {}).get("id") or ""))
        op_priors = research.get("op_priors")
        if isinstance(op_priors, list):
            for row in op_priors:
                if _canon_leaf(str((row or {}).get("op_name") or "")) == cid:
                    rate = row.get("s1_rate")
                    n = row.get("n_used")
                    if rate is not None and n is not None:
                        evidence_list.append(f"Research: {rate:.0%} success rate over {n} trials")
                    break

    leaderboard_component_sets, leaderboard_total_entries = _get_leaderboard_component_sets(ctx)
    if leaderboard_component_sets:
        _, lb_evidence = _leaderboard_boost(component, leaderboard_component_sets, leaderboard_total_entries)
        if lb_evidence:
            evidence_list.append(lb_evidence)

    # Compute insertion hint
    insertion_hint: Dict[str, str | None] | None = None
    nodes = ctx.get("workflow_nodes")
    edges = ctx.get("workflow_edges")
    comp_type = str((component or {}).get("category", "")) + "/" + str((component or {}).get("id", ""))
    if isinstance(nodes, list) and isinstance(edges, list):
        hint_cache = ctx.get("insertion_hints_by_type")
        if not isinstance(hint_cache, dict):
            hint_cache = {}
            ctx["insertion_hints_by_type"] = hint_cache
        insertion_hint = hint_cache.get(comp_type)
        if insertion_hint is None:
            insertion_hint = compute_insertion_point(nodes, edges, comp_type)
            hint_cache[comp_type] = insertion_hint

    result: Dict[str, Any] = {
        "component": component,
        "reason": reason,
        "action": "add_node",
        "score": safe_score,
        "evidence": evidence_list,
    }
    if insertion_hint is not None:
        result["insertion_hint"] = insertion_hint
    return result
