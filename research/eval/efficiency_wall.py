"""Memory/FLOP Efficiency Wall Detection.

Profiles a model at increasing sequence lengths to detect where memory
usage becomes non-linear (quadratic attention) or exceeds hardware budgets.
Measures peak memory, forward time, and detects the "efficiency wall" —
the sequence length at which cost growth becomes unsustainable.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_SEQ_LENS = (64, 128, 256, 512, 1024)
_MEMORY_BUDGET_MB = 2048  # 2GB default budget


def _measure_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    n_warmup: int = 1,
    n_measure: int = 3,
) -> Dict[str, float]:
    """Measure peak memory and forward time for a single input."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                model(input_ids)
            except Exception:
                return {"error": "forward_failed", "peak_mb": 0.0, "time_ms": 0.0}

    # Measure
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            try:
                model(input_ids)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                return {
                    "error": "oom",
                    "peak_mb": float("inf"),
                    "time_ms": float("inf"),
                }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        # CPU: estimate from tensor sizes in the model
        peak_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (
            1024 * 1024
        )
        # Add rough activation estimate: batch_size * seq_len * dim * 4 bytes * n_layers
        sum(p.numel() for p in model.parameters())
        peak_mb += input_ids.numel() * 4 * 8 / (1024 * 1024)  # rough

    avg_time = sum(times) / len(times) if times else 0.0

    return {
        "peak_mb": round(peak_mb, 2),
        "time_ms": round(avg_time, 2),
    }


def _detect_scaling_regime(
    measurements: List[Dict[str, Any]],
) -> str:
    """Detect whether memory scaling is linear, quadratic, or worse."""
    valid = [
        (m["seq_len"], m["peak_mb"])
        for m in measurements
        if m.get("peak_mb", 0) > 0 and m.get("error") is None
    ]

    if len(valid) < 3:
        return "insufficient_data"

    # Compute growth ratios between consecutive points
    ratios = []
    for i in range(1, len(valid)):
        len_ratio = valid[i][0] / valid[i - 1][0]
        mem_ratio = valid[i][1] / max(valid[i - 1][1], 1e-6)
        if len_ratio > 1:
            # If memory grows proportionally to seq_len^k, then
            # log(mem_ratio) / log(len_ratio) ≈ k
            import math

            k = math.log(max(mem_ratio, 1e-6)) / math.log(len_ratio)
            ratios.append(k)

    if not ratios:
        return "unknown"

    avg_k = sum(ratios) / len(ratios)

    if avg_k < 1.3:
        return "linear"
    elif avg_k < 2.3:
        return "quadratic"
    else:
        return "super_quadratic"


def evaluate_efficiency_wall(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    seq_lens: tuple[int, ...] = _DEFAULT_SEQ_LENS,
    batch_size: int = 2,
    memory_budget_mb: float = _MEMORY_BUDGET_MB,
) -> Dict[str, Any]:
    """Profile model at increasing sequence lengths to find the efficiency wall.

    Args:
        model: Compiled model to profile.
        vocab_size: Vocabulary size for generating random inputs.
        device: Device for profiling.
        seq_lens: Sequence lengths to test (ascending).
        batch_size: Batch size for profiling.
        memory_budget_mb: Memory budget in MB for wall detection.

    Returns:
        Dict with efficiency_wall_score (0-1), wall_seq_len, scaling_regime,
        and per-length measurements.
    """
    t0 = time.perf_counter()
    model.eval()

    measurements = []
    wall_seq_len = None

    for seq_len in sorted(seq_lens):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        result = _measure_forward(model, input_ids, device)
        result["seq_len"] = seq_len

        del input_ids

        if result.get("error") == "oom":
            wall_seq_len = seq_len
            measurements.append(result)
            break

        if result["peak_mb"] > memory_budget_mb and wall_seq_len is None:
            wall_seq_len = seq_len

        measurements.append(result)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Detect scaling regime
    scaling_regime = _detect_scaling_regime(measurements)

    # Find max viable seq_len (within budget, no errors)
    max_viable = 0
    for m in measurements:
        if m.get("error") is None and m["peak_mb"] <= memory_budget_mb:
            max_viable = m["seq_len"]

    # Efficiency wall score: fraction of tested seq_lens that are viable
    viable_count = sum(
        1
        for m in measurements
        if m.get("error") is None and m["peak_mb"] <= memory_budget_mb
    )
    total_tested = len(measurements) if measurements else 1

    # Bonus for linear scaling
    regime_bonus = {"linear": 0.2, "quadratic": 0.0, "super_quadratic": -0.2}
    base_score = viable_count / total_tested
    efficiency_wall_score = max(
        0.0, min(1.0, base_score + regime_bonus.get(scaling_regime, 0.0))
    )

    # Compute time scaling factor (ratio of last to first forward time)
    time_scaling = None
    valid_times = [
        m for m in measurements if m.get("error") is None and m["time_ms"] > 0
    ]
    if len(valid_times) >= 2:
        time_scaling = round(valid_times[-1]["time_ms"] / valid_times[0]["time_ms"], 2)

    return {
        "efficiency_wall_score": round(efficiency_wall_score, 4),
        "wall_seq_len": wall_seq_len,
        "max_viable_seq_len": max_viable,
        "scaling_regime": scaling_regime,
        "time_scaling_factor": time_scaling,
        "measurements": measurements,
        "memory_budget_mb": memory_budget_mb,
        "elapsed_ms": round(elapsed_ms, 1),
    }
