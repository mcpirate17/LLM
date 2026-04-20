from __future__ import annotations

import torch


def clamp_progress(progress: float | None) -> float:
    if progress is None:
        return 1.0
    return float(max(0.0, min(1.0, progress)))


def get_routing_progress(module) -> float:
    return clamp_progress(getattr(module, "_routing_progress", 1.0))


def curriculum_enabled(config: dict) -> bool:
    return bool(config.get("curriculum_enabled", False))


def stage_name(progress: float, warmup_frac: float, mid_frac: float) -> str:
    if progress < warmup_frac:
        return "early"
    if progress < mid_frac:
        return "mid"
    return "late"


def piecewise_schedule(
    progress: float,
    *,
    start: float,
    mid: float,
    end: float,
    warmup_frac: float,
    mid_frac: float,
) -> float:
    progress = clamp_progress(progress)
    warmup_frac = max(0.0, min(1.0, warmup_frac))
    mid_frac = max(warmup_frac + 1e-6, min(1.0, mid_frac))
    if progress <= warmup_frac:
        alpha = 1.0 if warmup_frac <= 1e-6 else progress / warmup_frac
        return start + (mid - start) * alpha
    alpha = (progress - warmup_frac) / max(mid_frac - warmup_frac, 1e-6)
    alpha = max(0.0, min(1.0, alpha))
    return mid + (end - mid) * alpha


def scheduled_scalar(
    module,
    config: dict,
    *,
    key: str,
    default: float,
) -> float:
    if not curriculum_enabled(config):
        return float(config.get(key, default))
    progress = get_routing_progress(module)
    warmup_frac = float(config.get("curriculum_warmup_frac", 0.25))
    mid_frac = float(config.get("curriculum_mid_frac", 0.65))
    start = float(config.get(f"{key}_start", config.get(key, default)))
    mid = float(config.get(f"{key}_mid", config.get(key, default)))
    end = float(config.get(f"{key}_end", config.get(key, default)))
    return piecewise_schedule(
        progress,
        start=start,
        mid=mid,
        end=end,
        warmup_frac=warmup_frac,
        mid_frac=mid_frac,
    )


def scheduled_int(
    module,
    config: dict,
    *,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = scheduled_scalar(module, config, key=key, default=float(default))
    return max(minimum, min(maximum, int(round(value))))


def branch_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1, keepdim=True).add_(1e-6).sqrt_().to(x.dtype)
