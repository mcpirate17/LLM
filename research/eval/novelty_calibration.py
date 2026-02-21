"""Novelty calibration utilities.

Estimates novelty-score noise floor from repeated baseline-transformer probes,
so novelty measurements are comparable across time/reference versions.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import torch

from .baseline import _BaselineTransformer
from .fingerprint import build_novelty_reference_version, compute_fingerprint


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    idx = max(0.0, min(1.0, q)) * (len(data) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return data[lo]
    frac = idx - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def calibrate_baseline_transformer_novelty(
    n_runs: int = 8,
    seq_len: int = 32,
    model_dim: int = 64,
    vocab_size: int = 1024,
    device: str = "cpu",
    seed: int = 1234,
) -> Dict[str, Any]:
    """Run repeated baseline probes and estimate novelty noise/confidence bands."""
    runs = max(1, int(n_runs))
    novelty_values: List[float] = []
    jac_spec_values: List[float] = []
    jac_rank_values: List[float] = []
    cka_t_values: List[float] = []
    cka_s_values: List[float] = []
    cka_c_values: List[float] = []

    cka_source = "none"
    cka_artifact_version = None
    probe_hash = None

    for i in range(runs):
        torch.manual_seed(seed + i)
        model = _BaselineTransformer(vocab_size=vocab_size, d_model=model_dim, n_layers=2)
        fp = compute_fingerprint(
            model,
            seq_len=seq_len,
            model_dim=model_dim,
            vocab_size=vocab_size,
            device=device,
            n_probes=16,
        )
        novelty_values.append(float(fp.novelty_score))
        jac_spec_values.append(float(fp.jacobian_spectral_norm))
        jac_rank_values.append(float(fp.jacobian_effective_rank))
        cka_t_values.append(float(fp.cka_vs_transformer))
        cka_s_values.append(float(fp.cka_vs_ssm))
        cka_c_values.append(float(fp.cka_vs_conv))

        cka_source = fp.cka_source or cka_source
        cka_artifact_version = fp.cka_artifact_version or cka_artifact_version
        probe_hash = fp.cka_probe_protocol_hash or probe_hash

    ref_version = build_novelty_reference_version(
        cka_source=cka_source,
        cka_artifact_version=cka_artifact_version,
        cka_probe_protocol_hash=probe_hash,
    )

    noise_mean = float(mean(novelty_values)) if novelty_values else 0.0
    noise_std = float(pstdev(novelty_values)) if len(novelty_values) > 1 else 0.0

    return {
        "reference_version": ref_version,
        "cka_source": cka_source,
        "cka_artifact_version": cka_artifact_version,
        "probe_protocol_hash": probe_hash,
        "n_runs": runs,
        "noise_floor_mean": noise_mean,
        "noise_floor_std": noise_std,
        "confidence_low": _quantile(novelty_values, 0.05),
        "confidence_high": _quantile(novelty_values, 0.95),
        "distribution": {
            "novelty_score": novelty_values,
            "jacobian_spectral_norm": jac_spec_values,
            "jacobian_effective_rank": jac_rank_values,
            "cka_vs_transformer": cka_t_values,
            "cka_vs_ssm": cka_s_values,
            "cka_vs_conv": cka_c_values,
        },
        "metadata": {
            "model": "baseline_transformer",
            "seq_len": seq_len,
            "model_dim": model_dim,
            "vocab_size": vocab_size,
            "device": device,
            "seed": seed,
        },
    }


def novelty_stability_under_small_perturbations(
    base_model: torch.nn.Module,
    seq_len: int = 32,
    model_dim: int = 64,
    vocab_size: int = 1024,
    device: str = "cpu",
    perturbation_std: float = 1e-4,
    n_trials: int = 4,
    seed: int = 4321,
) -> Dict[str, Any]:
    """Estimate novelty-score drift under tiny parameter perturbations."""
    torch.manual_seed(seed)
    base_fp = compute_fingerprint(
        base_model,
        seq_len=seq_len,
        model_dim=model_dim,
        vocab_size=vocab_size,
        device=device,
        n_probes=16,
    )
    base = float(base_fp.novelty_score)
    drifts: List[float] = []

    for i in range(max(1, int(n_trials))):
        model = _BaselineTransformer(vocab_size=vocab_size, d_model=model_dim, n_layers=2)
        model.load_state_dict(base_model.state_dict())
        with torch.no_grad():
            torch.manual_seed(seed + i + 1)
            for p in model.parameters():
                p.add_(torch.randn_like(p) * float(perturbation_std))

        fp = compute_fingerprint(
            model,
            seq_len=seq_len,
            model_dim=model_dim,
            vocab_size=vocab_size,
            device=device,
            n_probes=16,
        )
        drifts.append(abs(float(fp.novelty_score) - base))

    return {
        "base_novelty": base,
        "mean_abs_drift": float(mean(drifts)) if drifts else 0.0,
        "max_abs_drift": max(drifts) if drifts else 0.0,
        "drifts": drifts,
    }
