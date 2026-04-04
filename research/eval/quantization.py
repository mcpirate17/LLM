"""
Fake-quantization utilities for sparse+quant co-design evaluation.

Provides simulated low-bit quantization (INT8/INT4) to evaluate how well
architectures retain quality under combined sparsity + quantization.
No actual kernel compilation — uses fake-quant (quantize-then-dequantize)
to measure quality degradation in full-precision arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .utils import iter_eligible_params


@dataclass(slots=True)
class FakeQuantResult:
    bits: int
    target_sparsity: float
    actual_sparsity: float
    n_params_total: int
    n_params_quantized: int
    bytes_per_param_original: float
    bytes_per_param_effective: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "bits": self.bits,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self.actual_sparsity,
            "n_params_total": self.n_params_total,
            "n_params_quantized": self.n_params_quantized,
            "bytes_per_param_original": self.bytes_per_param_original,
            "bytes_per_param_effective": self.bytes_per_param_effective,
        }


def fake_quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Simulate low-bit quantization via quantize-then-dequantize.

    Per-tensor symmetric quantization: maps [-max_abs, max_abs] to
    [-2^(bits-1), 2^(bits-1)-1], then dequantizes back to float.
    """
    if bits >= 16:
        return tensor.clone()

    max_abs = tensor.abs().max()
    if max_abs == 0:
        return tensor.clone()

    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax

    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
    return quantized * scale


def apply_fake_quantization(
    model: nn.Module,
    bits: int = 8,
) -> FakeQuantResult:
    """In-place fake-quantization of eligible weight matrices.

    Replaces each weight tensor with its fake-quantized version.
    This simulates the numerical effects of low-bit storage without
    requiring actual kernel support.
    """
    bits = max(2, min(16, bits))
    total_params = 0
    total_quantized = 0
    n_sparse = 0

    for _name, param in iter_eligible_params(model):
        numel = int(param.numel())
        total_params += numel
        total_quantized += numel

        # Count existing zeros (from prior pruning/sparsity)
        n_sparse += numel - int(torch.count_nonzero(param.data).item())

        with torch.no_grad():
            param.copy_(fake_quantize_tensor(param.data, bits=bits))

    actual_sparsity = n_sparse / total_params if total_params > 0 else 0.0
    density = 1.0 - actual_sparsity
    bytes_original = 4.0  # float32
    bytes_effective = density * (bits / 8.0)

    return FakeQuantResult(
        bits=bits,
        target_sparsity=actual_sparsity,  # sparsity is pre-existing
        actual_sparsity=actual_sparsity,
        n_params_total=total_params,
        n_params_quantized=total_quantized,
        bytes_per_param_original=bytes_original,
        bytes_per_param_effective=round(bytes_effective, 4),
    )


def evaluate_sparse_quant_quality(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    target_sparsity: float = 0.5,
    bits: int = 8,
    pruning_method: str = "wanda",
) -> Optional[Dict]:
    """Evaluate quality retention under combined sparsity + quantization.

    Pipeline:
    1. Measure dense baseline loss
    2. Apply one-shot pruning at target_sparsity
    3. Measure sparse-only loss
    4. Apply fake quantization at target bits
    5. Measure sparse+quant loss
    6. Compute quality-retention-per-byte metrics

    Returns dict with all metrics, or None if evaluation fails.
    """
    if not input_batches:
        return None

    # Import pruning here to avoid circular imports
    from .pruning import apply_one_shot_pruning, estimate_lm_ce_loss

    # 1. Dense baseline
    dense_loss = estimate_lm_ce_loss(model, input_batches, device)
    if dense_loss is None:
        return None

    # 2. Apply sparsity
    prune_result = apply_one_shot_pruning(
        model, target_sparsity=target_sparsity, method=pruning_method
    )

    # 3. Sparse-only loss
    sparse_loss = estimate_lm_ce_loss(model, input_batches, device)

    # 4. Apply fake quantization on top
    quant_result = apply_fake_quantization(model, bits=bits)

    # 5. Sparse+quant loss
    sparse_quant_loss = estimate_lm_ce_loss(model, input_batches, device)

    if sparse_loss is None or sparse_quant_loss is None:
        return None

    # 6. Compute quality-retention-per-byte
    bytes_original = 4.0  # float32
    bytes_effective = quant_result.bytes_per_param_effective
    compression_ratio = bytes_original / max(bytes_effective, 0.01)

    sparse_retention = dense_loss / max(sparse_loss, 1e-8)
    full_retention = dense_loss / max(sparse_quant_loss, 1e-8)
    quant_degradation = sparse_quant_loss - sparse_loss

    # Quality-retention-per-byte: higher is better
    # A model that retains quality well while using fewer bytes scores high
    quality_per_byte = full_retention * compression_ratio

    return {
        "dense_loss": round(dense_loss, 6),
        "sparse_loss": round(sparse_loss, 6),
        "sparse_quant_loss": round(sparse_quant_loss, 6),
        "sparse_retention": round(sparse_retention, 4),
        "full_retention": round(full_retention, 4),
        "quant_degradation": round(quant_degradation, 6),
        "compression_ratio": round(compression_ratio, 2),
        "quality_per_byte": round(quality_per_byte, 4),
        "bits": bits,
        "target_sparsity": target_sparsity,
        "actual_sparsity": prune_result.actual_sparsity,
        "pruning_method": pruning_method,
        "bytes_per_param_effective": bytes_effective,
        "n_params_total": prune_result.n_params_total,
    }
