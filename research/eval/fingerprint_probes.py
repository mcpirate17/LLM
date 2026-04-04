"""Representation, interaction, routing, and geometry probes."""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE

from .fingerprint_native import (
    geometry_metrics,
    interaction_metrics,
    mean_abs_linear_delta,
)
from .hierarchy_probe import hierarchy_fitness
from ..synthesis.grammar import _ROUTING_COMPRESSION_MOE_OPS

logger = logging.getLogger(__name__)


def _resolve_representation_hook_module(model: nn.Module) -> nn.Module | None:
    cached = model.__dict__.get("_fingerprint_capture_module", None)
    if cached is False:
        return None
    if cached is not None:
        return cached

    last_candidate: nn.Module | None = None
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.Linear)):
            last_candidate = mod

    model.__dict__["_fingerprint_capture_module"] = (
        last_candidate if last_candidate is not None else False
    )
    return last_candidate


def get_representations(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run the model and capture the last hidden-like activation for CKA fallback."""
    try:
        direct_impl = getattr(model, "_fingerprint_representations", None)
        if callable(direct_impl):
            logits, reps = direct_impl(input_ids)
            if isinstance(logits, torch.Tensor) and isinstance(reps, torch.Tensor):
                logits._cka_intermediate_reps = reps.detach()
            return logits

        captured: dict[str, torch.Tensor] = {}
        hooks = []
        mod = _resolve_representation_hook_module(model)

        if mod is not None:

            def _hook(_module, _inp, out):
                if isinstance(out, torch.Tensor) and out.dim() >= 2:
                    captured["last"] = out.detach()

            hooks.append(mod.register_forward_hook(_hook))

        logits = model(input_ids)

        for hook in hooks:
            hook.remove()

        if captured:
            logits._cka_intermediate_reps = captured["last"]
        return logits
    except Exception as exc:
        logger.warning("Failed to get representations: %s", exc)
        return None


def interaction_influence_matrix(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    ids = input_ids[:1]
    n_positions = int(positions.numel())
    pre_logits_from_embed = getattr(model, "_fingerprint_pre_logits_from_embed", None)
    if (
        callable(pre_logits_from_embed)
        and hasattr(model, "embed")
        and hasattr(model, "lm_head")
    ):
        base_embed = model.embed(ids)
        base_pre = pre_logits_from_embed(base_embed)
        perturbed_embed = base_embed.expand(n_positions, -1, -1).clone()
        row_idx = torch.arange(n_positions, device=positions.device)
        replacement_ids = (ids[0, positions] + 1) % vocab_size
        perturbed_embed[row_idx, positions] = model.embed(replacement_ids)
        delta = pre_logits_from_embed(perturbed_embed) - base_pre
        native_metric = mean_abs_linear_delta(delta, model.lm_head.weight)
        if native_metric is not None:
            return native_metric
        return F.linear(delta, model.lm_head.weight).abs().mean(dim=-1)

    logits_from_embed = getattr(model, "_fingerprint_logits_from_embed", None)
    if callable(logits_from_embed) and hasattr(model, "embed"):
        base_embed = model.embed(ids)
        base_out = logits_from_embed(base_embed)
        perturbed_embed = base_embed.expand(n_positions, -1, -1).clone()
        row_idx = torch.arange(n_positions, device=positions.device)
        replacement_ids = (ids[0, positions] + 1) % vocab_size
        perturbed_embed[row_idx, positions] = model.embed(replacement_ids)
        return (logits_from_embed(perturbed_embed) - base_out).abs().mean(dim=-1)

    base_out = model(ids)
    perturbed_batch = ids.expand(n_positions, -1).clone()
    row_idx = torch.arange(n_positions, device=positions.device)
    perturbed_batch[row_idx, positions] = (
        perturbed_batch[row_idx, positions] + 1
    ) % vocab_size
    return (model(perturbed_batch) - base_out).abs().mean(dim=-1)


def analyze_interactions(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
) -> Dict[str, float]:
    result = {
        "locality": 0.5,
        "sparsity": 0.5,
        "symmetry": 0.5,
        "hierarchy": 0.5,
        "_succeeded": False,
    }
    try:
        n_positions = min(8, seq_len)
        positions = torch.linspace(0, seq_len - 1, n_positions, device=device).long()
        influence = interaction_influence_matrix(
            model,
            input_ids[:1],
            positions,
            vocab_size=vocab_size,
        )
        result.update(interaction_metrics(influence, positions))
        result["_succeeded"] = True
    except Exception as exc:
        logger.warning("Interaction analysis failed: %s", exc)
    return result


def analyze_routing(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    result = {
        "selectivity": 0.0,
        "compute_ratio": 0.0,
        "lane_correlation": 0.0,
        "_has_routing": False,
    }
    has_routing = False
    if hasattr(model, "graph") and model.graph is not None:
        for node in model.graph.nodes.values():
            if not node.is_input and node.op_name in _ROUTING_COMPRESSION_MOE_OPS:
                has_routing = True
                break
    if not has_routing:
        return result

    result["_has_routing"] = True
    try:
        with torch.no_grad():
            model(input_ids)

        if hasattr(model, "last_routing_scores"):
            scores = model.last_routing_scores
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                result["selectivity"] = float(scores.std().item())

        if hasattr(model, "get_routing_compute_stats"):
            stats = model.get_routing_compute_stats()
            slow = stats.get("slow_flops", 0)
            fast = stats.get("fast_flops", 1)
            result["compute_ratio"] = float(slow / max(fast, 1e-6))
        elif hasattr(model, "routing_compute_ratio"):
            result["compute_ratio"] = float(model.routing_compute_ratio)

        if hasattr(model, "last_routing_decisions"):
            decisions = model.last_routing_decisions
            if isinstance(decisions, torch.Tensor) and decisions.dim() >= 2:
                batch_size, seq_len = decisions.shape[:2]
                positions = (
                    torch.arange(seq_len, device=device)
                    .float()
                    .expand(batch_size, seq_len)
                )
                result["lane_correlation"] = float(
                    _pearson_corr(decisions.float(), positions).item()
                )
    except Exception as exc:
        logger.debug("Routing analysis failed: %s", exc)
    return result


def analyze_geometry(reps: torch.Tensor) -> Dict[str, float]:
    result = {
        "intrinsic_dim": 0.0,
        "isotropy": 0.0,
        "rank_ratio": 0.0,
        "_succeeded": False,
    }
    try:
        flat = reps.reshape(-1, reps.shape[-1]).float()
        num_rows, width = flat.shape
        if num_rows < 2 or width < 2:
            return result

        native = geometry_metrics(reps)
        if native is not None:
            result.update(native)
            result["_succeeded"] = True
            return result

        flat = flat - flat.mean(dim=0, keepdim=True)
        subset = flat[
            torch.randperm(num_rows, device=flat.device)[: min(num_rows, 500)]
        ]
        try:
            if subset.shape[0] >= subset.shape[1]:
                gram = subset.transpose(0, 1) @ subset
            else:
                gram = subset @ subset.transpose(0, 1)
            singular_values = torch.linalg.eigvalsh(gram).clamp_min(1e-20).sqrt()
        except Exception as exc:
            logger.debug("SVD failed in geometry analysis: %s", exc)
            return result

        singular_values = singular_values.clamp(min=1e-10)
        normalized = singular_values / singular_values.sum()
        result["intrinsic_dim"] = float((1.0 / (normalized**2).sum()).item())
        result["isotropy"] = float(
            (singular_values.min() / singular_values.max()).item()
        )
        entropy = float((-(normalized * torch.log(normalized))).sum().item())
        result["rank_ratio"] = math.exp(entropy) / len(singular_values)
        result["_succeeded"] = True
    except Exception as exc:
        logger.warning("Geometry analysis failed: %s", exc)
    return result


def analyze_hierarchy(reps: torch.Tensor) -> Dict[str, float]:
    return hierarchy_fitness(reps, max_tokens=100)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return (x_centered * y_centered).sum() / (
        torch.norm(x_centered) * torch.norm(y_centered) + 1e-8
    )
