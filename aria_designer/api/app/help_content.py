"""Component help content and compatibility matrix.

Derives "works well with" / "avoid with" from intent_parser category
adjacency rules and research failure_risk_signatures. Results are cached
aggressively since the underlying data changes infrequently.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .intent_parser import _COMPONENT_GROUPS, _LEAF_GROUPS, component_groups

__all__ = ["get_component_tips", "get_patterns_summary"]

# ── Category adjacency rules ─────────────────────────────────────────
# Maps a category to (good_neighbors, bad_neighbors).  Derived from
# common architecture patterns: norm→mixing, projection→activation, etc.

_ADJACENCY: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "activation": (("projection", "normalization", "mixing"), ("io",)),
    "normalization": (("mixing", "projection", "activation"), ("io",)),
    "projection": (("activation", "normalization", "mixing"), ("io",)),
    "mixing": (("normalization", "projection", "structural"), ("io",)),
    "routing": (("mixing", "projection", "normalization"), ("io",)),
    "structural": (("mixing", "projection", "routing"), ("io",)),
    "compression": (("projection", "normalization"), ("io", "routing")),
    "io": ((), ()),
}

# ── Reverse index: leaf → component_id ────────────────────────────────
# Built once at import from _LEAF_GROUPS.

_LEAF_TO_CATEGORIES: Dict[str, Tuple[str, ...]] = dict(_LEAF_GROUPS)

# ── Compatibility cache ───────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_COMPAT_CACHE: Dict[str, Dict[str, Any]] = {}
_COMPAT_CACHE_AT: float = 0.0
_CACHE_TTL_S: float = 300.0  # 5 minutes


def _categories_for(component_id: str) -> Tuple[str, ...]:
    """Resolve categories for a component ID using the shared intent_parser logic."""
    return component_groups(component_id)


def _good_neighbors(categories: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: list[str] = []
    for cat in categories:
        for neighbor in _ADJACENCY.get(cat, ((), ()))[0]:
            if neighbor not in seen:
                seen.append(neighbor)
    return tuple(seen)


def _bad_neighbors(categories: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: list[str] = []
    for cat in categories:
        for neighbor in _ADJACENCY.get(cat, ((), ()))[1]:
            if neighbor not in seen:
                seen.append(neighbor)
    return tuple(seen)


def _leaf_ids_in_categories(categories: Tuple[str, ...]) -> List[str]:
    """Return leaf component IDs that belong to any of the given categories."""
    cat_set = set(categories)
    result: list[str] = []
    for leaf_id, leaf_cats in _LEAF_TO_CATEGORIES.items():
        if cat_set & set(leaf_cats):
            result.append(leaf_id)
    return result


def _build_tips(component_id: str) -> Dict[str, Any]:
    cats = _categories_for(component_id)
    good_cats = _good_neighbors(cats)
    bad_cats = _bad_neighbors(cats)

    works_well = _leaf_ids_in_categories(good_cats)
    avoid_with = _leaf_ids_in_categories(bad_cats)

    # Build human-readable patterns
    patterns: list[str] = []
    if "normalization" in cats:
        patterns.append("Place before mixing/attention layers for stable activations")
        patterns.append("RMSNorm is cheaper than LayerNorm — prefer unless batch stats needed")
    if "activation" in cats:
        patterns.append("Place after linear projections to introduce nonlinearity")
        patterns.append("SiLU/GELU outperform ReLU in modern architectures")
    if "projection" in cats:
        patterns.append("Use after normalization, before activation")
        patterns.append("Low-rank variants reduce FLOPs at slight accuracy cost")
    if "mixing" in cats:
        patterns.append("Place after normalization for stable attention scores")
        patterns.append("Linear attention trades quality for O(n) sequence scaling")
    if "routing" in cats:
        patterns.append("MoE routing works best with 2-8 experts and top-2 gating")
        patterns.append("Place after projection layers for richer expert inputs")
    if "structural" in cats:
        patterns.append("Split/concat enable parallel computation paths")
        patterns.append("Residual (add) connections critical for deep graphs")
    if "compression" in cats:
        patterns.append("Low-rank projections trade capacity for efficiency")
        patterns.append("Combine with normalization to prevent activation drift")

    return {
        "component_id": component_id,
        "categories": list(cats),
        "works_well_with": works_well[:10],
        "avoid_with": avoid_with[:5],
        "patterns": patterns,
    }


def get_component_tips(
    component_id: str,
    research_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get compatibility tips for a component, with optional research enrichment."""
    global _COMPAT_CACHE_AT

    now = time.monotonic()
    cache_key = str(component_id or "").strip().lower()

    with _CACHE_LOCK:
        if (now - _COMPAT_CACHE_AT) <= _CACHE_TTL_S and cache_key in _COMPAT_CACHE:
            tips = _COMPAT_CACHE[cache_key]
        else:
            tips = None

    if tips is None:
        tips = _build_tips(cache_key)
        with _CACHE_LOCK:
            _COMPAT_CACHE[cache_key] = tips
            _COMPAT_CACHE_AT = now

    # Enrich with research failure risk signatures if available
    if isinstance(research_signals, dict):
        failure_sigs = research_signals.get("failure_risk_signatures")
        if isinstance(failure_sigs, list):
            warnings: list[str] = []
            for sig in failure_sigs:
                if not isinstance(sig, dict):
                    continue
                pattern = str(sig.get("pattern") or "").lower()
                if cache_key in pattern:
                    desc = str(sig.get("description") or sig.get("pattern") or "")
                    if desc:
                        warnings.append(desc)
            if warnings:
                tips = {**tips, "research_warnings": warnings[:3]}

        # Enrich with leaderboard usage data
        top_entries = research_signals.get("top_entries")
        if isinstance(top_entries, list):
            usage_count = sum(
                1 for entry in top_entries
                if isinstance(entry, dict)
                and cache_key in str(entry.get("program_text") or "").lower()
            )
            if usage_count > 0:
                tips = {
                    **tips,
                    "leaderboard_usage": f"Used in {usage_count} of top {len(top_entries)} architectures",
                }

    return tips


def get_patterns_summary() -> Dict[str, Any]:
    """Return a summary of common architecture patterns for the help panel."""
    return {
        "patterns": [
            {
                "name": "Transformer Block",
                "description": "LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual",
                "components": ["layernorm", "softmax_attention", "add", "linear_proj", "gelu"],
            },
            {
                "name": "SSM Block (Mamba-style)",
                "description": "Linear -> Selective Scan -> Linear with gating",
                "components": ["linear_proj", "selective_scan", "silu"],
            },
            {
                "name": "MoE Layer",
                "description": "Router selects top-K experts from parallel FFN paths",
                "components": ["moe_topk", "linear_proj", "gelu", "linear_proj_down"],
            },
            {
                "name": "Hybrid Attention + SSM",
                "description": "Parallel attention and SSM paths merged for best of both",
                "components": ["softmax_attention", "selective_scan", "concat", "linear_proj"],
            },
            {
                "name": "Compressed FFN",
                "description": "Low-rank projection reduces FFN parameters while preserving capacity",
                "components": ["low_rank_proj", "gelu", "linear_proj_down", "rmsnorm"],
            },
        ],
        "tips": [
            "Always include normalization before attention/mixing layers",
            "Residual connections are critical for graphs deeper than 3 layers",
            "RMSNorm is ~30% faster than LayerNorm with similar quality",
            "Start simple (input -> linear -> activation -> output) and iterate",
        ],
    }
