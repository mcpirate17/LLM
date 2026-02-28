"""
Activation Sparsity and Dead Neuron Detection.

Hooks into model execution to capture activation statistics across all layers.
Identifies neurons that are never activated (dead neurons) or highly sparse.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

@dataclass
class SparsityResult:
    """Statistics for a single layer's activation sparsity."""
    layer_id: str
    layer_type: str
    n_neurons: int
    dead_neurons: int
    mean_sparsity: float  # Fraction of zero activations
    p90_sparsity: float
    max_sparsity: float
    is_collapsed: bool    # True if >95% neurons are dead

@dataclass
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

def check_activation_sparsity(model: nn.Module, 
                             dataloader: Any, 
                             n_batches: int = 1,
                             threshold: float = 1e-6) -> ModelSparsityReport:
    """
    Run the model on sample data and track activation statistics.
    Uses forward hooks to capture intermediate tensor values.
    """
    model.eval()
    device = next(model.parameters()).device
    
    stats = {}
    hooks = []

    def get_hook(name, module):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                # Flatten batch and seq dimensions: (B, S, D) -> (B*S, D)
                flat = out.detach().reshape(-1, out.shape[-1])
                # Count zeros per neuron across all tokens in the batch
                is_zero = (flat.abs() < threshold).float()
                
                if name not in stats:
                    stats[name] = {
                        "zero_counts": torch.zeros(flat.shape[-1], device=flat.device),
                        "total_counts": 0,
                        "type": mod.__class__.__name__,
                        "op_name": getattr(mod, "op_name", ""),
                    }
                
                stats[name]["zero_counts"] += is_zero.sum(dim=0)
                stats[name]["total_counts"] += flat.shape[0]
        return hook

    # Register hooks for all interesting layers
    # We focus on linear layers, normalized outputs, and MoE experts
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)):
            hooks.append(module.register_forward_hook(get_hook(name, module)))
        # Also catch custom CompiledOp if it has an execute_fn that returns a tensor
        elif hasattr(module, 'op_name'):
            hooks.append(module.register_forward_hook(get_hook(name, module)))

    try:
        # Run inference
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                else:
                    input_ids = batch.to(device)
                
                model(input_ids)
    finally:
        # Always remove hooks
        for h in hooks:
            h.remove()

    # Process results
    layer_results = []
    total_neurons = 0
    total_dead = 0
    
    # Spiking ops produce intentionally sparse binary outputs — use relaxed threshold
    _SPIKING_OPS = {"lif_neuron", "spike_rate_code", "stdp_attention", "sparse_threshold"}

    for name, data in stats.items():
        zero_frac = (data["zero_counts"] / max(1, data["total_counts"])).cpu().numpy()
        dead = int((zero_frac > 0.999).sum())

        # Relaxed collapse threshold for spiking ops (binary outputs are naturally sparse)
        op_name = data.get("op_name", "")
        collapse_threshold = 0.99 if op_name in _SPIKING_OPS else 0.95

        res = SparsityResult(
            layer_id=name,
            layer_type=data["type"],
            n_neurons=len(zero_frac),
            dead_neurons=dead,
            mean_sparsity=float(zero_frac.mean()),
            p90_sparsity=float(np.percentile(zero_frac, 90)),
            max_sparsity=float(zero_frac.max()),
            is_collapsed=dead > collapse_threshold * len(zero_frac)
        )
        layer_results.append(res)
        total_neurons += res.n_neurons
        total_dead += res.dead_neurons

    overall_sparsity = 0.0
    if layer_results:
        overall_sparsity = sum(r.mean_sparsity for r in layer_results) / len(layer_results)

    return ModelSparsityReport(
        layers=layer_results,
        total_neurons=total_neurons,
        total_dead_neurons=total_dead,
        overall_sparsity=overall_sparsity,
        max_layer_collapse=max([r.dead_neurons/r.n_neurons for r in layer_results]) if layer_results else 0.0,
        dead_neuron_ratio=total_dead / max(1, total_neurons)
    )


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
        collapse_penalty = sum(1 for l in report.layers if l.is_collapsed) / max(len(report.layers), 1)
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
