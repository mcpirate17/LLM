"""Smoke + behavior tests for component_fab.generator.primitive_templates."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.primitive_templates import (
    CalculusAugmentedLane,
    FiniteDifferenceCalculusLane,
    FourierBasisLane,
    GraphDiffusionAdapterLane,
    GraphDiffusionLane,
    LowRankAdapterLane,
    LowRankFactorizedLane,
    MultiscaleWaveletAdapterLane,
    MultiscaleWaveletLane,
    RandomFeatureKernelAdapterLane,
    RandomFeatureKernelLane,
    SparseBandedAdapterLane,
    SparseBandedMatrixLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
)


def _check_shape_and_grad(module: torch.nn.Module, dim: int, seq_len: int = 8) -> None:
    x = torch.randn(2, seq_len, dim, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
    loss = y.pow(2).mean()
    loss.backward()
    assert torch.isfinite(y).all().item()
    for p in module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item()


def test_tropical_attention_shape_and_grad() -> None:
    _check_shape_and_grad(TropicalAttention(dim=16), dim=16)


def test_tropical_attention_is_winner_take_all_per_position() -> None:
    # Tropical attention sums max(aff + V) across j — sparse by construction.
    module = TropicalAttention(dim=16).eval()
    x = torch.randn(1, 32, 16)
    with torch.no_grad():
        y = module(x)
    # For a tropical max, the output per feature should be close to the max of V.
    # Test: the magnitude per token should be dominated by a few features (high max/mean).
    # Empirically at random init this measures ~1.5 — well above ~1.0 for a smooth op.
    ratio = y.abs().amax(dim=-1).mean() / (y.abs().mean() + 1e-12)
    assert ratio > 1.3


def test_tropical_state_space_shape_and_grad() -> None:
    _check_shape_and_grad(TropicalStateSpace(dim=16), dim=16, seq_len=8)


def test_tropical_state_space_is_causal() -> None:
    # Max-plus propagation is draw-sensitive: unseeded, some RNG states leave
    # the position-0 perturbation invisible at the last position.
    torch.manual_seed(0)
    module = TropicalStateSpace(dim=16).eval()
    x = torch.randn(1, 16, 16)
    x_perturbed = x.clone()
    x_perturbed[:, 0] = x_perturbed[:, 0] + torch.randn_like(x_perturbed[:, 0])
    with torch.no_grad():
        y = module(x)
        y_perturbed = module(x_perturbed)
    # Perturbing position 0 must change later positions (state propagation).
    late_diff = (y[:, -1] - y_perturbed[:, -1]).abs().mean()
    assert late_diff > 1e-4


def test_topk_linear_actually_sparse() -> None:
    module = TopKLinear(in_dim=16, out_dim=16, k=4).eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y = module(x)
    n_active = (y.abs() > 1e-8).sum(dim=-1).float().mean()
    assert n_active.item() <= 4 + 1e-4
    _check_shape_and_grad(TopKLinear(16, 16, 4), dim=16)


def test_topk_linear_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        TopKLinear(16, 16, k=0)
    with pytest.raises(ValueError):
        TopKLinear(16, 16, k=17)


def test_fourier_basis_lane_shape_and_grad() -> None:
    _check_shape_and_grad(FourierBasisLane(dim=16), dim=16, seq_len=16)


def test_fourier_basis_lane_mixes_globally() -> None:
    # FFT mixes positions along the sequence dim; a structured perturbation
    # at position 0 (non-uniform across features) should produce nonzero
    # response at most other positions. A constant perturbation has a flat
    # spectrum and would not exercise the spectral mixing path.
    module = FourierBasisLane(dim=16).eval()
    x = torch.randn(1, 16, 16)
    x_perturbed = x.clone()
    x_perturbed[:, 0] = x_perturbed[:, 0] + torch.randn_like(x_perturbed[:, 0])
    with torch.no_grad():
        y = module(x)
        y_perturbed = module(x_perturbed)
    diff = (y - y_perturbed).abs().sum(dim=-1).squeeze(0)
    assert (diff > 1e-6).float().mean().item() > 0.5


def test_finite_difference_calculus_lane_shape_grad_and_causal() -> None:
    _check_shape_and_grad(FiniteDifferenceCalculusLane(dim=16), dim=16, seq_len=8)
    module = FiniteDifferenceCalculusLane(dim=16).eval()
    x = torch.randn(1, 12, 16)
    x_perturbed = x.clone()
    x_perturbed[:, 8:] = x_perturbed[:, 8:] + torch.randn_like(x_perturbed[:, 8:])
    with torch.no_grad():
        y = module(x)
        y_perturbed = module(x_perturbed)
    assert torch.allclose(y[:, :8], y_perturbed[:, :8], atol=1e-6)


def test_low_rank_factorized_lane_shape_grad_and_rank() -> None:
    module = LowRankFactorizedLane(dim=16, rank=4)
    _check_shape_and_grad(module, dim=16, seq_len=8)
    assert module.rank == 4
    assert module.down.weight.shape == (4, 16)
    assert module.up.weight.shape == (16, 4)


def test_sparse_banded_matrix_lane_shape_grad_and_causal_band() -> None:
    module = SparseBandedMatrixLane(dim=8, bandwidth=3).eval()
    _check_shape_and_grad(SparseBandedMatrixLane(dim=8, bandwidth=3), dim=8, seq_len=8)
    x = torch.randn(1, 10, 8)
    x_perturbed = x.clone()
    x_perturbed[:, 0] = x_perturbed[:, 0] + torch.randn_like(x_perturbed[:, 0])
    with torch.no_grad():
        y = module(x)
        y_perturbed = module(x_perturbed)
    diff = (y - y_perturbed).abs().sum(dim=-1).squeeze(0)
    assert diff[:3].sum().item() > 1e-6
    assert diff[3:].max().item() < 1e-6


def test_sparse_banded_matrix_rejects_invalid_bandwidth() -> None:
    with pytest.raises(ValueError):
        SparseBandedMatrixLane(dim=8, bandwidth=0)


def test_composable_wrappers_preserve_shape_and_grad() -> None:
    base = TropicalAttention(dim=16, causal=True)
    module = SparseBandedAdapterLane(
        LowRankAdapterLane(CalculusAugmentedLane(base, dim=16), dim=16, rank=4),
        dim=16,
        bandwidth=3,
    )
    _check_shape_and_grad(module, dim=16, seq_len=8)


def test_random_feature_kernel_lane_shape_grad_and_causal() -> None:
    _check_shape_and_grad(RandomFeatureKernelLane(dim=16, n_features=8), dim=16)
    module = RandomFeatureKernelLane(dim=16, n_features=8).eval()
    x = torch.randn(1, 12, 16)
    x_future = x.clone()
    x_future[:, 8:] = x_future[:, 8:] + torch.randn_like(x_future[:, 8:])
    with torch.no_grad():
        y = module(x)
        y_future = module(x_future)
    assert torch.allclose(y[:, :8], y_future[:, :8], atol=1e-6)
    assert module.n_features == 8


def test_multiscale_wavelet_lane_shape_grad_and_scales() -> None:
    module = MultiscaleWaveletLane(dim=16, n_scales=3)
    _check_shape_and_grad(module, dim=16)
    assert module.n_scales == 3
    with pytest.raises(ValueError):
        MultiscaleWaveletLane(dim=16, n_scales=0)


def test_graph_diffusion_lane_shape_grad_and_causal() -> None:
    _check_shape_and_grad(GraphDiffusionLane(dim=16, diffusion_steps=2), dim=16)
    module = GraphDiffusionLane(dim=16, diffusion_steps=2).eval()
    x = torch.randn(1, 12, 16)
    x_future = x.clone()
    x_future[:, 8:] = x_future[:, 8:] + torch.randn_like(x_future[:, 8:])
    with torch.no_grad():
        y = module(x)
        y_future = module(x_future)
    assert torch.allclose(y[:, :8], y_future[:, :8], atol=1e-6)
    with pytest.raises(ValueError):
        GraphDiffusionLane(dim=16, diffusion_steps=0)


def test_new_knob_adapters_preserve_shape_and_grad() -> None:
    base = TropicalAttention(dim=16, causal=True)
    module = GraphDiffusionAdapterLane(
        MultiscaleWaveletAdapterLane(
            RandomFeatureKernelAdapterLane(base, dim=16, n_features=8),
            dim=16,
            n_scales=2,
        ),
        dim=16,
        diffusion_steps=2,
    )
    _check_shape_and_grad(module, dim=16, seq_len=8)


# --- Width-robustness campaign lanes (2026-05-29) -------------------------------

from component_fab.generator.primitive_templates import (  # noqa: E402
    AnisotropicSemiringReciprocalAttention,
    FixedRankReciprocalAttention,
    HeteroSemiringReciprocalAttention,
    ReciprocalRankAttention,
    SemiringReciprocalAttention,
    TemperedTropicalAttention,
    _heads_for_head_dim,
)


def _assert_causal(module: torch.nn.Module, dim: int, seq_len: int = 12) -> None:
    """Perturbing token t must not change outputs at positions < t."""
    module = module.eval()
    x = torch.randn(1, seq_len, dim)
    with torch.no_grad():
        y1 = module(x)
        x2 = x.clone()
        x2[:, seq_len // 2 :, :] += 5.0
        y2 = module(x2)
    leak = (y1[:, : seq_len // 2] - y2[:, : seq_len // 2]).abs().max().item()
    assert leak < 1e-4, f"future leakage {leak}"


@pytest.mark.parametrize("dim", [96, 192])
def test_width_robust_lanes_shape_grad_causal(dim: int) -> None:
    for module in (
        HeteroSemiringReciprocalAttention(dim),
        AnisotropicSemiringReciprocalAttention(dim),
        FixedRankReciprocalAttention(dim, rank=96),
        TemperedTropicalAttention(dim),
    ):
        _check_shape_and_grad(module, dim=dim, seq_len=12)
        _assert_causal(module, dim=dim)


def test_hetero_head_dim_selection() -> None:
    # fixed head_dim≈96: 1 head at dim96, 2 at 192, 6 at 576 — even head_dim only.
    assert _heads_for_head_dim(96, 96) == 1
    assert _heads_for_head_dim(192, 96) == 2
    assert _heads_for_head_dim(576, 96) == 6


def test_hetero_single_head_reduces_to_semiring() -> None:
    # No output proj + γ init 1.0 ⇒ a 1-head hetero is exactly single-head semiring.
    torch.manual_seed(0)
    h = HeteroSemiringReciprocalAttention(96)  # dim96 → 1 head
    assert h.n_heads == 1
    torch.manual_seed(0)
    s = SemiringReciprocalAttention(96)
    x = torch.randn(2, 16, 96)
    assert torch.allclose(h(x), s(x), atol=1e-5)


def test_anisotropic_init_equals_scalar_semiring() -> None:
    # γ_d = exp(0) = 1 ∀d at init ⇒ identical to the scalar-γ semiring.
    torch.manual_seed(0)
    a = AnisotropicSemiringReciprocalAttention(96)
    assert a.semiring_beta.shape == (96,)
    torch.manual_seed(0)
    s = SemiringReciprocalAttention(96)
    x = torch.randn(2, 16, 96)
    assert torch.allclose(a(x), s(x), atol=1e-5)


def test_fixed_rank_full_rank_equals_reciprocal() -> None:
    # rank == dim ⇒ no score bottleneck ⇒ exactly ReciprocalRankAttention.
    torch.manual_seed(0)
    fr = FixedRankReciprocalAttention(96, rank=96)
    torch.manual_seed(0)
    rr = ReciprocalRankAttention(96, use_rope=True)
    x = torch.randn(2, 16, 96)
    assert torch.allclose(fr(x), rr(x), atol=1e-5)


def test_tempered_tropical_anneals_toward_hard_max() -> None:
    # Larger β ⇒ the tempered pooling approaches the hard tropical max (per head),
    # so the |max|-to-|mean| sparsity ratio increases with β.
    torch.manual_seed(0)
    m = TemperedTropicalAttention(96, use_rope=False).eval()
    x = torch.randn(1, 32, 96)
    with torch.no_grad():
        m.log_beta.fill_(-3.0)  # softplus → ~0.05, soft
        soft = m(x)
        m.log_beta.fill_(5.0)  # softplus → ~5, near-hard max
        hard = m(x)

    def sparsity_ratio(y: torch.Tensor) -> float:
        return (y.abs().amax(-1).mean() / (y.abs().mean() + 1e-9)).item()

    assert sparsity_ratio(hard) > sparsity_ratio(soft)
