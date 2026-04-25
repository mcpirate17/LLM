"""Adaptive threshold calibration helpers for phase-7 auto-escalation."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

from ..notebook import LabNotebook

RewardFn = Callable[[Dict[str, Any]], float | None]


def _record_threshold_fallback(
    nb: LabNotebook,
    *,
    context: str,
    tier_clause: str,
    floor: float,
    percentile: float,
    selected_threshold: float,
    sample_size: int,
    labeled_size: int = 0,
    positive_count: int | None = None,
    negative_count: int | None = None,
    reason: str,
) -> None:
    payload: Dict[str, Any] = {
        "context": context,
        "tier_clause": tier_clause,
        "floor": floor,
        "percentile": percentile,
        "selected_threshold": selected_threshold,
        "fallback_threshold": selected_threshold,
        "sample_size": sample_size,
        "labeled_size": labeled_size,
        "metrics": {
            "mode": "floor_fallback" if reason == "insufficient_rows" else "fallback"
        },
        "metadata": {"reason": reason},
    }
    if positive_count is not None:
        payload["positive_count"] = positive_count
    if negative_count is not None:
        payload["negative_count"] = negative_count
    nb.record_threshold_calibration(**payload)


def _labeled_threshold_rewards(
    rows: List[Dict[str, Any]],
    reward_fn: RewardFn,
) -> tuple[List[Tuple[float, float]], int, int]:
    labeled: List[Tuple[float, float]] = []
    for row in rows:
        reward = reward_fn(row)
        if reward is not None:
            labeled.append((float(row["composite_score"]), float(reward)))
    positive_count = sum(1 for _, reward in labeled if reward >= 0.55)
    negative_count = sum(1 for _, reward in labeled if reward <= 0.45)
    return labeled, positive_count, negative_count


def _best_threshold_metrics(np, labeled: List[Tuple[float, float]], fallback: float):
    candidate_thresholds = np.unique(
        np.quantile(
            np.array([score for score, _ in labeled], dtype=np.float64),
            np.linspace(0.10, 0.95, 32),
        )
    )
    best_threshold = fallback
    best_objective = float("-inf")
    best_metrics: Dict[str, Any] = {
        "mode": "fallback",
        "promoted_count": 0,
        "held_count": len(labeled),
    }
    for threshold in candidate_thresholds:
        promoted = [reward for score, reward in labeled if score >= threshold]
        held = [reward for score, reward in labeled if score < threshold]
        if len(promoted) < 5:
            continue
        tp = sum(1 for reward in promoted if reward >= 0.55)
        fp = sum(1 for reward in promoted if reward <= 0.45)
        fn = sum(1 for reward in held if reward >= 0.55)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        avg_reward = sum(promoted) / max(len(promoted), 1)
        rejection_quality = (
            sum(1 for reward in held if reward <= 0.45) / max(len(held), 1)
            if held
            else 0.0
        )
        objective = 0.55 * f1 + 0.30 * avg_reward + 0.15 * rejection_quality
        if objective <= best_objective:
            continue
        best_objective = objective
        best_threshold = float(threshold)
        best_metrics = {
            "mode": "adaptive",
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "avg_reward": round(avg_reward, 6),
            "rejection_quality": round(rejection_quality, 6),
            "promoted_count": len(promoted),
            "held_count": len(held),
            "selected_quantile": round(
                float(
                    np.mean(
                        np.array(
                            [score <= threshold for score, _ in labeled],
                            dtype=np.float64,
                        )
                    )
                ),
                6,
            ),
        }
    return best_threshold, best_objective, best_metrics, candidate_thresholds


def calibrated_promotion_threshold(
    nb: LabNotebook,
    *,
    rows: List[Dict[str, Any]],
    tier_clause: str,
    floor: float,
    percentile: float,
    context: str,
    reward_fn: RewardFn | None,
) -> float:
    """Pick a stage threshold from recent realized downstream outcomes."""
    import numpy as np

    sample_size = len(rows)
    if len(rows) < 20:
        _record_threshold_fallback(
            nb,
            context=context,
            tier_clause=tier_clause,
            floor=floor,
            percentile=percentile,
            selected_threshold=floor,
            sample_size=sample_size,
            reason="insufficient_rows",
        )
        return floor

    scores = np.array([float(row["composite_score"]) for row in rows], dtype=np.float64)
    fallback_threshold = max(float(np.percentile(scores, percentile)), floor)
    if reward_fn is None:
        _record_threshold_fallback(
            nb,
            context=context,
            tier_clause=tier_clause,
            floor=floor,
            percentile=percentile,
            selected_threshold=fallback_threshold,
            sample_size=sample_size,
            reason="unsupported_context",
        )
        return fallback_threshold

    labeled, positive_count, negative_count = _labeled_threshold_rewards(
        rows, reward_fn
    )
    if len(labeled) < 24:
        _record_threshold_fallback(
            nb,
            context=context,
            tier_clause=tier_clause,
            floor=floor,
            percentile=percentile,
            selected_threshold=fallback_threshold,
            sample_size=sample_size,
            labeled_size=len(labeled),
            positive_count=positive_count,
            negative_count=negative_count,
            reason="insufficient_labeled_rows",
        )
        return fallback_threshold

    best_threshold, best_objective, best_metrics, candidate_thresholds = (
        _best_threshold_metrics(np, labeled, fallback_threshold)
    )
    selected_threshold = max(float(best_threshold), float(floor))
    nb.record_threshold_calibration(
        context=context,
        tier_clause=tier_clause,
        floor=floor,
        percentile=percentile,
        selected_threshold=selected_threshold,
        fallback_threshold=fallback_threshold,
        sample_size=sample_size,
        labeled_size=len(labeled),
        positive_count=positive_count,
        negative_count=negative_count,
        objective=best_objective if math.isfinite(best_objective) else None,
        metrics=best_metrics,
        metadata={"candidate_threshold_count": len(candidate_thresholds)},
    )
    return selected_threshold
