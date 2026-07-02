"""Mix-speed metric for a single forward op.

How fast does a perturbation at one input position propagate to other
positions in the output? `mix_half_life` is the smallest distance from
the injection point at which the response decays below half of its peak.

- Local op (conv k=3): half-life ~1.
- Norm-only op (rmsnorm): half-life = inf (doesn't mix positions).
- Global mixer (softmax_attn): half-life = 0 (instant global mix).

Intrinsic op metric — no training, no DB lookup, no registry coupling.
Runs in seconds on random inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class MixSpeedScorecard:
    mix_half_life: float
    peak_response_at_offset: int
    peak_response_magnitude: float
    response_decay: tuple[float, ...]
    mixes_globally: bool
    is_pure_local: bool
    n_trials: int


def _sample_trial_responses(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_trials: int,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    inject_at: int,
    delta_scale: float,
    generator: torch.Generator,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    x = torch.randn(
        n_trials,
        batch_size,
        seq_len,
        feature_dim,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    delta = (
        torch.randn(
            n_trials,
            batch_size,
            feature_dim,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        * delta_scale
    )
    x_perturbed = x.clone()
    x_perturbed[:, :, inject_at, :] = x_perturbed[:, :, inject_at, :] + delta
    x_flat = x.reshape(n_trials * batch_size, seq_len, feature_dim)
    xp_flat = x_perturbed.reshape(n_trials * batch_size, seq_len, feature_dim)

    with torch.no_grad():
        y = forward_fn(x_flat)
        y_perturbed = forward_fn(xp_flat)

    if y.shape != x_flat.shape or y_perturbed.shape != x_flat.shape:
        raise ValueError(
            f"forward_fn must preserve shape; got input {tuple(x_flat.shape)}, "
            f"output {tuple(y.shape)}"
        )

    diff = (
        (y_perturbed - y)
        .reshape(n_trials, batch_size, seq_len, feature_dim)
        .pow(2)
        .sum(dim=-1)
        .sqrt()
    )
    return diff.mean(dim=1)


def _fold_response_by_offset(
    response: torch.Tensor, inject_at: int
) -> tuple[float, ...]:
    seq_len = int(response.shape[0])
    bucket: dict[int, list[float]] = {}
    for j in range(seq_len):
        offset = abs(j - inject_at)
        bucket.setdefault(offset, []).append(float(response[j].item()))
    max_offset = max(bucket)
    decay: list[float] = []
    for offset in range(max_offset + 1):
        vals = bucket.get(offset, [0.0])
        decay.append(sum(vals) / len(vals))
    return tuple(decay)


def _classify_decay(
    decay: tuple[float, ...],
    *,
    half_life_threshold: float,
    global_mixing_threshold: float,
    local_only_eps: float,
) -> tuple[float, int, float, bool, bool]:
    peak = max(decay) if decay else 0.0
    if peak <= 0.0:
        return float("inf"), 0, 0.0, False, True

    peak_offset = decay.index(peak)
    threshold = half_life_threshold * peak
    half_life = float("inf")
    for offset in range(peak_offset + 1, len(decay)):
        if decay[offset] <= threshold:
            half_life = float(offset - peak_offset)
            break

    mid_offset = len(decay) // 2
    mixes_globally = decay[mid_offset] > global_mixing_threshold * peak
    far_field = decay[min(4, len(decay)) :]
    is_pure_local = bool(far_field) and all(d <= local_only_eps for d in far_field)
    return half_life, peak_offset, peak, mixes_globally, is_pure_local


def measure_mix_speed(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    seq_len: int = 64,
    feature_dim: int = 32,
    batch_size: int = 2,
    delta_scale: float = 1e-2,
    n_trials: int = 8,
    inject_at: int = 0,
    half_life_threshold: float = 0.5,
    global_mixing_threshold: float = 0.1,
    local_only_eps: float = 1e-6,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> MixSpeedScorecard:
    """Probe ``forward_fn`` for information-mixing speed.

    The fn is invoked with ``[B, L, D]`` tensors; the returned tensor must
    have the same shape. We perturb one position in the input by a small
    delta, run the fn on perturbed and unperturbed inputs, and look at how
    the squared-difference response distributes across output positions.
    """
    if inject_at < 0 or inject_at >= seq_len:
        raise ValueError(f"inject_at={inject_at} out of range [0, {seq_len})")

    generator = torch.Generator(device=device).manual_seed(seed)
    responses = _sample_trial_responses(
        forward_fn,
        n_trials=n_trials,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_dim=feature_dim,
        inject_at=inject_at,
        delta_scale=delta_scale,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    response = responses.mean(dim=0).cpu()
    decay = _fold_response_by_offset(response, inject_at)
    half_life, peak_offset, peak, mixes_globally, is_pure_local = _classify_decay(
        decay,
        half_life_threshold=half_life_threshold,
        global_mixing_threshold=global_mixing_threshold,
        local_only_eps=local_only_eps,
    )

    return MixSpeedScorecard(
        mix_half_life=half_life,
        peak_response_at_offset=peak_offset,
        peak_response_magnitude=peak,
        response_decay=decay,
        mixes_globally=mixes_globally,
        is_pure_local=is_pure_local,
        n_trials=n_trials,
    )


def influence_matrix(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    seq_len: int = 24,
    feature_dim: int = 32,
    batch_size: int = 1,
    delta_scale: float = 1e-2,
    n_trials: int = 4,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """L×L cross-position influence map via finite differences.

    Perturb input position ``i`` by a small delta, measure the response at every
    output position ``j``, average over ``n_trials`` random inputs/deltas. Row i =
    injection position, col j = response position. For a causal lane the matrix
    is lower-triangular (injecting at ``i`` can only move outputs at ``j >= i``):
    the main diagonal is self-response, the strictly-below-diagonal entries
    (``j > i``) are genuine cross-token mixing, and the above-diagonal entries
    should be ~0 (an acausal leak — caught separately by the S0.5 causality gate).

    Intrinsic op metric — random inputs, no training, no DB, no registry coupling.
    Shared by ``viz/introspect.py`` (the UI plot) and ``metrics/mixing_quality.py``
    (the breadth score) so both come from ONE measurement instead of two copies
    of the same finite-diff loop.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    accum = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    positions = torch.arange(seq_len, device=device)
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(
                batch_size,
                seq_len,
                feature_dim,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            delta = (
                torch.randn(
                    batch_size,
                    feature_dim,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                * delta_scale
            )
            y = forward_fn(x)
            if y.shape != x.shape:
                raise ValueError(
                    f"forward_fn must preserve shape; got input {tuple(x.shape)}, "
                    f"output {tuple(y.shape)}"
                )
            xp = x.unsqueeze(0).expand(seq_len, -1, -1, -1).clone()
            xp[positions, :, positions, :] = xp[positions, :, positions, :] + delta
            yp = forward_fn(xp.reshape(seq_len * batch_size, seq_len, feature_dim))
            if yp.shape != (seq_len * batch_size, seq_len, feature_dim):
                raise ValueError(
                    "forward_fn must preserve shape; got batched influence output "
                    f"{tuple(yp.shape)}"
                )
            yp = yp.reshape(seq_len, batch_size, seq_len, feature_dim)
            # mean L2 response over batch at each output position -> [L, L]
            resp = (yp - y.unsqueeze(0)).pow(2).sum(dim=-1).sqrt().mean(dim=1)
            accum += resp
    return accum / n_trials
