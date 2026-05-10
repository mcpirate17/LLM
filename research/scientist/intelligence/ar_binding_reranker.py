"""Sibling AR/binding overlay reranker for GBM-screening survivors.

This module deliberately does not alter ``GBMPredictor.predict_rank_composite``.
It consumes the shared meta-analysis overlay after the production composite
rank head has already ordered survivors, then optionally reorders candidates
with holdout-cleared AR/binding evidence.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from research.meta_analysis.ar_binding_overlay import overlay_for_graph
from research.scientist.shared_utils import coerce_finite_float as _finite


OverlayFn = Callable[[dict[str, Any]], dict[str, Any]]


def score_ar_binding_overlay(overlay: dict[str, Any]) -> float | None:
    """Return ordering score for holdout-cleared overlays."""
    if not overlay or overlay.get("holdout_required"):
        return None
    ar_gain = _finite(overlay.get("expected_ar_gain"))
    binding_gain = _finite(overlay.get("expected_binding_gain"))
    retention_risk = _finite(overlay.get("retention_risk"))
    collapse_risk = _finite(overlay.get("collapse_risk"))
    if ar_gain is None and binding_gain is None:
        return None

    score = 0.0
    if ar_gain is not None:
        score += ar_gain
    if binding_gain is not None:
        score += binding_gain
    if retention_risk is not None:
        score -= retention_risk
    if collapse_risk is not None:
        score -= collapse_risk
    return score if math.isfinite(score) else None


def rerank_graphs_by_ar_binding(
    graphs: list[Any],
    *,
    overlay_fn: OverlayFn = overlay_for_graph,
) -> tuple[list[Any], dict[str, Any]]:
    """Annotate and rerank graphs by advisory AR/binding overlay evidence."""
    scored: list[tuple[int, Any, float | None]] = []
    holdout_required = 0
    usable_scores: list[float] = []

    for idx, graph in enumerate(graphs):
        overlay = overlay_fn(_graph_dict(graph))
        score = score_ar_binding_overlay(overlay)
        if overlay.get("holdout_required"):
            holdout_required += 1
        if score is not None:
            usable_scores.append(score)
        _tag_graph(graph, overlay, score)
        scored.append((idx, graph, score))

    used = bool(usable_scores)
    if used:
        scored.sort(
            key=lambda item: (
                item[2] is None,
                -(item[2] if item[2] is not None else 0.0),
                item[0],
            )
        )

    stats = {
        "used": used,
        "scored": len(graphs),
        "usable": len(usable_scores),
        "holdout_required": holdout_required,
        "score_min": min(usable_scores) if usable_scores else None,
        "score_max": max(usable_scores) if usable_scores else None,
    }
    return [graph for _, graph, _ in scored], stats


def _graph_dict(graph: Any) -> dict[str, Any]:
    if isinstance(graph, dict):
        return graph
    if hasattr(graph, "to_dict"):
        maybe = graph.to_dict()
        if isinstance(maybe, dict):
            return maybe
    return {}


def _tag_graph(graph: Any, overlay: dict[str, Any], score: float | None) -> None:
    metadata = getattr(graph, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata["ar_binding_overlay"] = overlay
    if score is not None:
        metadata["ar_binding_rerank_score"] = score
