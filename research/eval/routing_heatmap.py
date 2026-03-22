"""Routing Heatmap & Route Collapse Detection.

Extracts routing telemetry from compiled models to detect route collapse
(all tokens taking the same expert/lane) and measure routing diversity.

Provides:
- Per-module expert utilization balance (Gini coefficient)
- Routing entropy analysis
- Route collapse detection
- Aggregate routing_collapse_score for leaderboard
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def _gini_coefficient(counts: np.ndarray) -> float:
    """Compute Gini coefficient of expert utilization. 0=perfectly balanced, 1=fully collapsed."""
    if len(counts) < 2 or counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    return float(
        (2.0 * np.sum((np.arange(1, n + 1) * sorted_counts)) / (n * cumsum[-1]))
        - (n + 1) / n
    )


def _max_entropy(n_experts: int) -> float:
    """Maximum possible entropy for n_experts (uniform distribution)."""
    if n_experts <= 1:
        return 0.0
    return math.log(n_experts)


def evaluate_routing_heatmap(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Extract routing telemetry and compute route collapse metrics.

    Runs a forward pass with heatmap capture enabled, then analyzes
    the routing_telemetry dicts attached to model modules by the compiler.

    Args:
        model: Compiled model with routing ops.
        input_batches: List of input_ids tensors for evaluation.
        device: Device for evaluation.

    Returns:
        Dict with routing_collapse_score (0-1, higher = healthier routing),
        per-module routing stats, and collapse flags.
    """
    if not input_batches:
        return {"routing_collapse_score": None, "has_routing": False}

    model.eval()

    # Enable heatmap capture on all modules
    for m in model.modules():
        m._capture_heatmap = True

    # Clear any stale routing telemetry from prior runs
    for m in model.modules():
        if hasattr(m, "routing_telemetry"):
            delattr(m, "routing_telemetry")

    # Run forward passes to collect routing telemetry
    try:
        with torch.no_grad():
            for batch in input_batches:
                input_ids = batch.to(device)
                model(input_ids)
    except Exception as e:
        logger.debug("routing_heatmap forward failed: %s", e)
        return {"routing_collapse_score": None, "has_routing": False, "error": str(e)}
    finally:
        # Disable heatmap capture
        for m in model.modules():
            m._capture_heatmap = False

    # Collect routing telemetry from all modules
    routing_modules: List[Dict[str, Any]] = []
    for name, m in model.named_modules():
        rt = getattr(m, "routing_telemetry", None)
        if rt is None:
            continue

        expert_counts = rt.get("expert_counts", None)
        if expert_counts is not None:
            if isinstance(expert_counts, torch.Tensor):
                counts_np = expert_counts.cpu().numpy()
            else:
                counts_np = np.array(expert_counts, dtype=np.float64)
        else:
            counts_np = None

        n_experts = len(counts_np) if counts_np is not None else 0
        tokens_total = rt.get("tokens_total", 0)
        entropy_sum = rt.get("entropy_sum", 0.0)
        count = rt.get("count", 0)

        # Compute per-module metrics
        gini = (
            _gini_coefficient(counts_np)
            if counts_np is not None and n_experts > 1
            else 0.0
        )
        avg_entropy = (entropy_sum / count) if count > 0 else 0.0
        max_ent = _max_entropy(n_experts)
        normalized_entropy = (avg_entropy / max_ent) if max_ent > 0 else 0.0

        # Collapse detection: Gini > 0.8 or normalized entropy < 0.2
        is_collapsed = (gini > 0.8) or (normalized_entropy < 0.2 and n_experts > 1)

        # Dominant expert fraction: what % of tokens went to the most-used expert
        dominant_frac = 0.0
        if counts_np is not None and counts_np.sum() > 0:
            dominant_frac = float(counts_np.max() / counts_np.sum())

        module_info = {
            "module": name,
            "n_experts": n_experts,
            "tokens_total": tokens_total,
            "gini": round(gini, 4),
            "avg_entropy": round(avg_entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "dominant_expert_fraction": round(dominant_frac, 4),
            "is_collapsed": bool(is_collapsed),
            "heatmap": rt.get("heatmap"),
        }

        # Include merge-specific stats if present
        if "merge_kept" in rt:
            module_info["merge_kept"] = rt["merge_kept"]
            module_info["merge_dropped"] = rt.get("merge_dropped", 0)

        routing_modules.append(module_info)

    if not routing_modules:
        return {"routing_collapse_score": None, "has_routing": False}

    # Aggregate score: average of per-module health scores
    # Health = (1 - gini) * normalized_entropy for expert-routing modules
    # For merge modules (no expert counts), health = 1.0 (merge is structural, not collapseble)
    health_scores = []
    for rm in routing_modules:
        if rm["n_experts"] > 1:
            health = (1.0 - rm["gini"]) * max(rm["normalized_entropy"], 0.0)
            health_scores.append(health)

    if health_scores:
        collapse_score = float(sum(health_scores) / len(health_scores))
    else:
        # Only merge modules found — no expert routing to evaluate
        collapse_score = 1.0

    n_collapsed = sum(1 for rm in routing_modules if rm["is_collapsed"])

    return {
        "routing_collapse_score": round(collapse_score, 4),
        "has_routing": True,
        "n_routing_modules": len(routing_modules),
        "n_collapsed_modules": n_collapsed,
        "modules": routing_modules,
    }
