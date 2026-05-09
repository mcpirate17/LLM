"""Pure-arithmetic helpers shared across the scoring package.

Leaf module — no imports from siblings. Stdlib + ``..thresholds`` only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..thresholds import GPT2_REF


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_metric(
    metrics: Dict[str, Any],
    errors: Dict[str, list[str]],
    name: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    positive: bool = False,
) -> Optional[float]:
    value = metrics.get(name)
    if value is None:
        errors["missing"].append(name)
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        errors["corrupt"].append(name)
        return None
    if not math.isfinite(f):
        errors["corrupt"].append(name)
        return None
    if positive and f <= 0.0:
        errors["corrupt"].append(name)
        return None
    if min_value is not None and f < min_value:
        errors["corrupt"].append(name)
        return None
    if max_value is not None and f > max_value:
        errors["corrupt"].append(name)
        return None
    return f


def _first_present(metrics: Dict[str, Any], *names: str) -> tuple[str, Any]:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return name, value
    return names[0], None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _falsey_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", "off"}
    return value is False


def _safe_float01(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    try:
        return _clamp01(float(value))
    except (TypeError, ValueError):
        return None


def _scurve(ratio: float, k: float = 4.0) -> float:
    """Sigmoid S-curve centered at ratio=1.0.

    Returns 0-1:
      ratio=1.0 → 0.5 (frontier parity)
      ratio>1.0 → approaches 1.0 (better than frontier)
      ratio<1.0 → approaches 0.0 (worse than frontier)

    k controls steepness. k=4 gives:
      ratio=0.5 → 0.12,  ratio=1.5 → 0.88,  ratio=2.0 → 0.98
    """
    return 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))


def _scurve_lower_better(value: Optional[float], anchor: float) -> float:
    """S-curve where MORE NEGATIVE is better (id_collapse, erf_decay, icld).

    Negates the value (so negative inputs become positive ratios) then
    runs the standard _scurve. Anchor is the absolute value of the
    frontier-equivalent (e.g. 0.01 for id_collapse means a frontier rate
    of -0.01). Returns 0.0 for missing/positive values (positive =
    collapsing, which is the bad direction).
    """
    if value is None:
        return 0.0
    flipped = -float(value)
    if flipped <= 0.0:
        return 0.0
    return _scurve(flipped / max(anchor, 1e-9))


def _scurve_higher_better(value: Optional[float], anchor: float) -> float:
    """S-curve where higher is better (erf_density, logit_margin, etc.)."""
    if value is None or value <= 0.0:
        return 0.0
    return _scurve(value / max(anchor, 1e-9))


def _cv_penalty_multiplier(
    cv: Optional[float],
    lam: float,
    floor: float,
) -> float:
    """1.0 if cv is None or non-positive; else max(floor, 1 - lambda * cv)."""
    if cv is None:
        return 1.0
    try:
        cv_f = float(cv)
    except (TypeError, ValueError):
        return 1.0
    if cv_f <= 0.0:
        return 1.0
    return max(floor, 1.0 - lam * cv_f)


def compute_efficiency_multiple(
    loss_ratio: Optional[float] = None,
    param_count: Optional[float] = None,
    flops_forward: Optional[float] = None,
    throughput_tok_s: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
    forward_time_ms: Optional[float] = None,
    is_moe: bool = False,
) -> Optional[Dict[str, float]]:
    """Geometric mean of per-dimension ratios vs GPT-2.

    All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
    dimensions to return a result (graceful with missing data).

    For MoE models (is_moe=True), total param count is excluded from
    the geomean since MoE activates only a fraction of params per token.
    Returns dict with per-dimension ratios and ``geomean``, or None.
    """
    ref = GPT2_REF
    ratios: Dict[str, float] = {}

    if loss_ratio is not None and loss_ratio > 0:
        ratios["x_quality"] = ref["loss_ratio"] / loss_ratio
    # MoE: skip param count penalty — total params != active params
    if param_count is not None and param_count > 0 and not is_moe:
        ratios["x_params"] = ref["param_count"] / param_count
    if flops_forward is not None and flops_forward > 0:
        ratios["x_flops"] = ref["flops_forward"] / flops_forward
    if throughput_tok_s is not None and throughput_tok_s > 0:
        ratios["x_throughput"] = throughput_tok_s / ref["throughput_tok_s"]
    if peak_memory_mb is not None and peak_memory_mb > 0:
        ratios["x_memory"] = ref["peak_memory_mb"] / peak_memory_mb
    if forward_time_ms is not None and forward_time_ms > 0:
        ratios["x_latency"] = ref["forward_time_ms"] / forward_time_ms

    if len(ratios) < 3:
        return None

    geomean = 1.0
    for v in ratios.values():
        geomean *= v
    geomean = geomean ** (1.0 / len(ratios))
    ratios["geomean"] = geomean
    ratios["n_dimensions"] = float(len(ratios) - 1)
    return ratios
