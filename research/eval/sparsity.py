"""
Activation Sparsity and Dead Neuron Detection.

Hooks into model execution to capture activation statistics across all layers.
Identifies neurons that are never activated (dead neurons) or highly sparse.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import torch
import torch.nn as nn
import numpy as np

from ._eval_native import load_eval_native

_SPIKING_OPS = {
    "lif_neuron",
    "spike_rate_code",
    "stdp_attention",
    "sparse_threshold",
}


@dataclass(slots=True)
class SparsityResult:
    """Statistics for a single layer's activation sparsity."""

    layer_id: str
    layer_type: str
    n_neurons: int
    dead_neurons: int
    mean_sparsity: float  # Fraction of zero activations
    p90_sparsity: float
    max_sparsity: float
    is_collapsed: bool  # True if >95% neurons are dead


@dataclass(slots=True)
class ModelSparsityReport:
    """Comprehensive sparsity report for a whole model."""

    layers: List[SparsityResult]
    total_neurons: int
    total_dead_neurons: int
    overall_sparsity: float
    max_layer_collapse: float
    dead_neuron_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [asdict(l) for l in self.layers],
            "total_neurons": self.total_neurons,
            "total_dead_neurons": self.total_dead_neurons,
            "overall_sparsity": self.overall_sparsity,
            "max_layer_collapse": self.max_layer_collapse,
            "dead_neuron_ratio": self.dead_neuron_ratio,
        }


def _make_activation_hook(
    stats: Dict[str, Dict[str, Any]], name: str, module, threshold: float
):
    def hook(mod, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        flat = out.detach().reshape(-1, out.shape[-1])
        if name not in stats:
            stats[name] = {
                "zero_counts": torch.zeros(flat.shape[-1], device=flat.device),
                "total_counts": 0,
                "type": mod.__class__.__name__,
                "op_name": getattr(mod, "op_name", ""),
            }

        try:
            zero_counts = load_eval_native().zero_count_last_dim(flat, float(threshold))
            zero_counts = zero_counts.to(device=flat.device, dtype=torch.float32)
        except Exception:
            zero_counts = (flat.abs() < threshold).sum(dim=0, dtype=torch.float32)
        stats[name]["zero_counts"] += zero_counts
        stats[name]["total_counts"] += flat.shape[0]

    return hook


def _register_sparsity_hooks(
    model: nn.Module,
    stats: Dict[str, Dict[str, Any]],
    threshold: float,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)) or hasattr(
            module, "op_name"
        ):
            hooks.append(
                module.register_forward_hook(
                    _make_activation_hook(stats, name, module, threshold)
                )
            )
    return hooks


def _run_sparsity_batches(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    n_batches: int,
) -> None:
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            input_ids = (
                batch[0].to(device)
                if isinstance(batch, (list, tuple))
                else batch.to(device)
            )
            model(input_ids)


def _layer_sparsity_result(name: str, data: Dict[str, Any]) -> SparsityResult:
    zero_frac = (data["zero_counts"] / max(1, data["total_counts"])).cpu().numpy()
    dead = int((zero_frac > 0.999).sum())
    op_name = data.get("op_name", "")
    collapse_threshold = 0.99 if op_name in _SPIKING_OPS else 0.95
    return SparsityResult(
        layer_id=name,
        layer_type=data["type"],
        n_neurons=len(zero_frac),
        dead_neurons=dead,
        mean_sparsity=float(zero_frac.mean()),
        p90_sparsity=float(np.percentile(zero_frac, 90)),
        max_sparsity=float(zero_frac.max()),
        is_collapsed=dead > collapse_threshold * len(zero_frac),
    )


def _build_sparsity_report(stats: Dict[str, Dict[str, Any]]) -> ModelSparsityReport:
    layer_results = [_layer_sparsity_result(name, data) for name, data in stats.items()]
    total_neurons = sum(result.n_neurons for result in layer_results)
    total_dead = sum(result.dead_neurons for result in layer_results)
    overall_sparsity = (
        sum(result.mean_sparsity for result in layer_results) / len(layer_results)
        if layer_results
        else 0.0
    )
    max_layer_collapse = (
        max(result.dead_neurons / result.n_neurons for result in layer_results)
        if layer_results
        else 0.0
    )
    return ModelSparsityReport(
        layers=layer_results,
        total_neurons=total_neurons,
        total_dead_neurons=total_dead,
        overall_sparsity=overall_sparsity,
        max_layer_collapse=max_layer_collapse,
        dead_neuron_ratio=total_dead / max(1, total_neurons),
    )


def check_activation_sparsity(
    model: nn.Module, dataloader: Any, n_batches: int = 1, threshold: float = 1e-6
) -> ModelSparsityReport:
    """
    Run the model on sample data and track activation statistics.
    Uses forward hooks to capture intermediate tensor values.
    """
    model.eval()
    device = next(model.parameters()).device

    stats: Dict[str, Dict[str, Any]] = {}
    hooks = []

    try:
        hooks = _register_sparsity_hooks(model, stats, threshold)
        _run_sparsity_batches(model, dataloader, device, n_batches)
    finally:
        for hook in hooks:
            hook.remove()

    return _build_sparsity_report(stats)


def evaluate_activation_sparsity(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    """Runner-compatible wrapper for activation sparsity analysis.

    Args:
        model: Trained model to evaluate.
        input_batches: List of input_ids tensors.
        device: Device for evaluation.
        threshold: Absolute value below which an activation is "zero".

    Returns:
        Dict with activation_sparsity_score (0-1, higher = healthier),
        dead_neuron_ratio, per-layer details, and collapsed_layers list.
    """
    if not input_batches:
        return {"activation_sparsity_score": 0.0, "dead_neuron_ratio": 0.0}

    report = check_activation_sparsity(
        model, input_batches, n_batches=len(input_batches), threshold=threshold
    )

    # Score: 1.0 = perfectly healthy (no dead neurons), 0.0 = fully collapsed
    # Penalize dead neurons and layer collapses
    if report.total_neurons == 0:
        score = 0.0
    else:
        alive_ratio = 1.0 - report.dead_neuron_ratio
        collapse_penalty = sum(1 for l in report.layers if l.is_collapsed) / max(
            len(report.layers), 1
        )
        score = alive_ratio * (1.0 - collapse_penalty)

    collapsed = [l.layer_id for l in report.layers if l.is_collapsed]

    return {
        "activation_sparsity_score": round(float(score), 4),
        "dead_neuron_ratio": round(float(report.dead_neuron_ratio), 4),
        "overall_sparsity": round(float(report.overall_sparsity), 4),
        "total_neurons": report.total_neurons,
        "total_dead_neurons": report.total_dead_neurons,
        "collapsed_layers": collapsed,
        "n_layers_analyzed": len(report.layers),
    }
