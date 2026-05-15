"""Smoke + behavior tests for component_fab.metrics.compression_quality."""

from __future__ import annotations

from torch import nn

from component_fab.metrics.compression_quality import measure_compression_quality


def test_low_rank_linear_pair_reconstructs_partially() -> None:
    input_dim = 16
    latent = 4
    compress = nn.Linear(input_dim, latent)
    restore = nn.Linear(latent, input_dim)
    card = measure_compression_quality(
        compress,
        restore,
        input_dim=input_dim,
        latent_dim_declared=latent,
        seq_len=32,
        batch_size=4,
        n_trials=2,
    )
    assert card.input_dim == input_dim
    assert card.latent_dim_declared == latent
    assert card.n_compress_params > 0
    assert card.n_restore_params > 0
    assert 0.0 < card.effective_rank <= latent
    assert 0.0 < card.effective_rank_ratio <= 1.0
    assert card.reconstruction_mse > 0.0
    assert 0.0 <= card.flops_per_token_reduction < 1.0


def test_identity_compress_restore_reconstructs_perfectly() -> None:
    input_dim = 8
    compress = nn.Identity()
    restore = nn.Identity()
    card = measure_compression_quality(
        compress,
        restore,
        input_dim=input_dim,
        latent_dim_declared=input_dim,
        n_compress_params=0,
        n_restore_params=0,
        seq_len=16,
        batch_size=2,
        n_trials=2,
    )
    assert card.reconstruction_mse < 1e-6


def test_shape_violation_raises() -> None:
    input_dim = 8
    compress = nn.Linear(input_dim, 4)
    restore = nn.Linear(4, input_dim + 1)
    try:
        measure_compression_quality(
            compress,
            restore,
            input_dim=input_dim,
            latent_dim_declared=4,
            seq_len=8,
            batch_size=2,
            n_trials=1,
        )
    except ValueError:
        return
    raise AssertionError("shape-violating restore should raise ValueError")
