"""Smoke + behavior tests for component_fab.metrics.mix_speed."""

from __future__ import annotations

import torch

from component_fab.metrics.mix_speed import measure_mix_speed


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _rmsnorm_like(x: torch.Tensor) -> torch.Tensor:
    norm = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
    return x / norm


def _local_boxcar(x: torch.Tensor) -> torch.Tensor:
    pad = torch.nn.functional.pad(x, (0, 0, 1, 1))
    return (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0


def _pure_global_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1, keepdim=True).expand_as(x).contiguous()


def test_identity_does_not_mix() -> None:
    card = measure_mix_speed(_identity, seq_len=32, feature_dim=8, n_trials=2)
    assert card.is_pure_local
    assert card.peak_response_at_offset == 0
    assert card.response_decay[0] > 0.0
    assert all(d == 0.0 for d in card.response_decay[1:])


def test_rmsnorm_like_does_not_mix() -> None:
    card = measure_mix_speed(_rmsnorm_like, seq_len=32, feature_dim=8, n_trials=2)
    assert card.is_pure_local
    assert not card.mixes_globally


def test_local_boxcar_mixes_locally_only() -> None:
    card = measure_mix_speed(_local_boxcar, seq_len=64, feature_dim=8, n_trials=4)
    assert not card.mixes_globally
    assert card.peak_response_at_offset <= 1
    assert card.response_decay[0] > 0.0


def test_global_mean_mixes_globally() -> None:
    card = measure_mix_speed(
        _pure_global_mean, seq_len=32, feature_dim=8, n_trials=2, inject_at=5
    )
    assert card.mixes_globally
    assert not card.is_pure_local


def test_inject_at_out_of_range_raises() -> None:
    try:
        measure_mix_speed(_identity, seq_len=16, inject_at=20)
    except ValueError:
        return
    raise AssertionError("inject_at out of range should raise ValueError")


def test_shape_violation_raises() -> None:
    def shrink(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : x.shape[-1] // 2]

    try:
        measure_mix_speed(shrink, seq_len=16, feature_dim=8, n_trials=1)
    except ValueError:
        return
    raise AssertionError("shape-violating fn should raise ValueError")


def test_influence_matrix_matches_per_position_loop() -> None:
    """The batched all-positions perturbation must equal the per-position loop.

    Pins the L->1-forward batching of ``influence_matrix``: same RNG draws per
    trial, so the only allowed difference is batching float noise.
    """
    from component_fab.metrics.mix_speed import influence_matrix

    seq_len, feature_dim, batch_size, n_trials, delta_scale, seed = 12, 8, 2, 2, 1e-2, 5
    batched = influence_matrix(
        _local_boxcar,
        seq_len=seq_len,
        feature_dim=feature_dim,
        batch_size=batch_size,
        n_trials=n_trials,
        delta_scale=delta_scale,
        seed=seed,
    )

    generator = torch.Generator().manual_seed(seed)
    reference = torch.zeros(seq_len, seq_len)
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(batch_size, seq_len, feature_dim, generator=generator)
            delta = (
                torch.randn(batch_size, feature_dim, generator=generator) * delta_scale
            )
            y = _local_boxcar(x)
            for i in range(seq_len):
                xp = x.clone()
                xp[:, i, :] = xp[:, i, :] + delta
                resp = (_local_boxcar(xp) - y).pow(2).sum(dim=-1).sqrt().mean(dim=0)
                reference[i] += resp
    reference = reference / n_trials

    assert torch.allclose(batched, reference, atol=1e-6)


def test_influence_matrix_shape_violation_raises() -> None:
    from component_fab.metrics.mix_speed import influence_matrix

    def shrink(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : x.shape[-1] // 2]

    try:
        influence_matrix(shrink, seq_len=8, feature_dim=8, n_trials=1)
    except ValueError:
        return
    raise AssertionError("shape-violating fn should raise ValueError")
