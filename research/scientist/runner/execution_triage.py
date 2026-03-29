"""Post-S1 Triage: cheap evals to fill composite score dimensions.

Runs on every S1 passer immediately after screening, before
record_program_result. Cost: ~0ms for graph-derived metrics,
model is already in memory from screening.

Fills leaderboard columns that are currently NULL for 99% of entries:
- scaling_param_efficiency (from param count + published scaling curves)
- n_routing_ops, n_sparse_ops, n_moe_ops (from graph structure)
- routing_expert_count, routing_confidence_mean, routing_drop_rate
- routing_savings_ratio (from routing telemetry)
- compression_ratio (from weight entropy)
- activation_sparsity_score (from model weights)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Op sets for routing/sparse/MoE census ───────────────────────────

_ROUTING_OPS = frozenset(
    {
        "confidence_token_gate",
        "learned_token_gate",
        "route_lanes",
        "route_recursion",
        "depth_weighted_proj",
        "difficulty_blend_3way",
        "sparse_bottleneck_moe",
        "signal_conditioned_compression",
        "adjacent_token_merge",
        "cheap_verify_blend",
        "score_depth_blend",
    }
)

_SPARSE_OPS = frozenset(
    {
        "nm_sparse_linear",
        "semi_structured_2_4_linear",
        "block_sparse_linear",
        "route_topk",
        "sparse_threshold",
        "ternary_projection",
    }
)

_MOE_OPS = frozenset(
    {
        "moe_topk",
        "moe_2expert",
        "tropical_moe",
        "relu_gated_moe",
        "sparse_bottleneck_moe",
        "dual_compression_blend",
        "topk_gate",
    }
)


def run_triage(
    model: nn.Module,
    graph: Any,
    result: Dict[str, Any],
    model_dim: int,
) -> Dict[str, Any]:
    """Run cheap post-S1 triage evals. Returns dict of leaderboard column values.

    Args:
        model: The trained model (still in memory from screening).
        graph: ComputationGraph object (or None if unavailable).
        result: The screening result dict (contains loss_ratio, etc).
        model_dim: Model dimension from config.

    Returns:
        Dict mapping leaderboard column names to values.
    """
    triage: Dict[str, Any] = {}

    try:
        _estimate_param_efficiency(model, result, model_dim, triage)
    except Exception as e:
        logger.debug("Triage param efficiency failed: %s", e)

    try:
        _census_routing_ops(graph, triage)
    except Exception as e:
        logger.debug("Triage routing census failed: %s", e)

    try:
        _extract_routing_telemetry(model, triage)
    except Exception as e:
        logger.debug("Triage routing telemetry failed: %s", e)

    try:
        _estimate_compression(model, triage)
    except Exception as e:
        logger.debug("Triage compression failed: %s", e)

    try:
        _estimate_activation_sparsity(model, triage)
    except Exception as e:
        logger.debug("Triage activation sparsity failed: %s", e)

    return triage


# ── Param efficiency estimate ────────────────────────────────────────


def _estimate_param_efficiency(
    model: nn.Module,
    result: Dict[str, Any],
    model_dim: int,
    out: Dict[str, Any],
) -> None:
    """Estimate param efficiency as loss-quality-per-parameter.

    True scaling comparison requires retraining at multiple scales (d=256,
    d=512) and is done in the validation stage. Triage instead computes a
    cheap quality-per-param score that's useful for ranking candidates:

        quality = (1 - loss_ratio)^1.5  (penalizes weak learners)
        efficiency = quality / (params_scaled / 1M)

    Higher = better architecture (more learning per parameter).
    Marked scaling_confidence="triage_qpp" to distinguish from the real
    scaling comparison done in validation.
    """
    loss_ratio = result.get("loss_ratio")
    if loss_ratio is None or loss_ratio >= 0.95:
        return

    candidate_params = sum(p.numel() for p in model.parameters())
    if candidate_params <= 0 or model_dim <= 0:
        return

    # Normalize param count to d=256 equivalent for cross-dim comparison
    scale_factor = (256.0 / model_dim) ** 2
    scaled_params_m = (candidate_params * scale_factor) / 1_000_000

    if scaled_params_m <= 0:
        return

    # Quality: how much of the loss gap was closed, with superlinear reward
    quality = max(0.0, 1.0 - loss_ratio) ** 1.5

    efficiency = quality / scaled_params_m
    if efficiency > 0:
        out["param_efficiency"] = round(efficiency, 4)
        out["scaling_confidence"] = "triage_qpp"


# ── Routing op census ────────────────────────────────────────────────


def _census_routing_ops(graph: Any, out: Dict[str, Any]) -> None:
    """Count routing/sparse/MoE ops in the graph structure."""
    if graph is None:
        return

    op_names = set()
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        return

    for node in nodes.values():
        if hasattr(node, "op_name") and not getattr(node, "is_input", False):
            op_names.add(node.op_name)

    n_routing = len(op_names & _ROUTING_OPS)
    n_sparse = len(op_names & _SPARSE_OPS)
    n_moe = len(op_names & _MOE_OPS)

    out["n_routing_ops"] = n_routing
    out["n_sparse_ops"] = n_sparse
    out["n_moe_ops"] = n_moe


# ── Routing telemetry extraction ─────────────────────────────────────


def _extract_routing_telemetry(model: nn.Module, out: Dict[str, Any]) -> None:
    """Extract routing statistics from model's telemetry attrs (set during forward)."""
    total_tokens = 0
    total_fast = 0
    expert_counts_all = []
    confidence_sum = 0.0
    confidence_count = 0
    drop_total = 0
    drop_count = 0

    for module in model.modules():
        telemetry = getattr(module, "routing_telemetry", None)
        if telemetry is None:
            continue

        tokens = telemetry.get("tokens_total", 0)
        total_tokens += tokens

        # Expert count from expert_counts tensor
        ec = telemetry.get("expert_counts")
        if ec is not None and isinstance(ec, torch.Tensor) and ec.numel() > 0:
            n_experts = int(ec.numel())
            expert_counts_all.append(n_experts)
            # Drop rate: fraction of experts with 0 tokens
            n_active = int((ec > 0).sum().item())
            if n_experts > 0:
                drop_total += n_experts - n_active
                drop_count += n_experts

        # Confidence from telemetry
        cs = telemetry.get("confidence_sum", 0.0)
        cc = telemetry.get("confidence_count", 0)
        if cc > 0:
            confidence_sum += cs
            confidence_count += cc

        # Fast fraction from merge/exit telemetry
        merge_dropped = telemetry.get("merge_dropped", 0)
        if merge_dropped > 0 and tokens > 0:
            total_fast += merge_dropped

    if expert_counts_all:
        out["routing_expert_count"] = max(expert_counts_all)

    if confidence_count > 0:
        out["routing_confidence_mean"] = round(confidence_sum / confidence_count, 4)

    if drop_count > 0:
        out["routing_drop_rate"] = round(drop_total / drop_count, 4)

    # Routing savings: fraction of compute saved via routing
    # Approximated by merge drop rate + early exit attenuation
    if total_tokens > 0 and total_fast > 0:
        out["routing_savings_ratio"] = round(min(1.0, total_fast / total_tokens), 4)


# ── Compression ratio estimate ───────────────────────────────────────


def _estimate_compression(model: nn.Module, out: Dict[str, Any]) -> None:
    """Estimate compression ratio from weight entropy.

    Higher entropy = less compressible. Ratio represents estimated
    compressed size / original size. Low ratio = highly compressible.
    """
    all_params = []
    total_numel = 0
    for p in model.parameters():
        if p.numel() > 0:
            all_params.append(p.detach().flatten())
            total_numel += p.numel()

    if total_numel == 0:
        return

    all_weights = torch.cat(all_params)

    # Estimate entropy via histogram binning
    # Use 256 bins (8-bit quantization equivalent)
    n_bins = 256
    w_min = all_weights.min().item()
    w_max = all_weights.max().item()
    if w_max - w_min < 1e-10:
        out["compression_ratio"] = 0.0  # All weights identical
        return

    counts = torch.histc(all_weights, bins=n_bins, min=w_min, max=w_max)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zero bins
    entropy = -torch.sum(probs * torch.log2(probs)).item()

    # Compression ratio: entropy (bits) / 32 (float32 bits)
    # Lower = more compressible
    out["compression_ratio"] = round(entropy / 32.0, 4)


# ── Activation sparsity estimate ─────────────────────────────────────


def _estimate_activation_sparsity(model: nn.Module, out: Dict[str, Any]) -> None:
    """Estimate activation sparsity from weight structure.

    Models with sparse/structured weights tend to produce sparse activations.
    Uses weight sparsity (fraction of near-zero weights) as a proxy.
    """
    total_elements = 0
    near_zero = 0
    threshold = 1e-4

    for p in model.parameters():
        if p.numel() > 0:
            total_elements += p.numel()
            near_zero += int((p.detach().abs() < threshold).sum().item())

    if total_elements > 0:
        sparsity = near_zero / total_elements
        out["activation_sparsity_score"] = round(sparsity, 4)
