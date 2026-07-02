"""Tests for NM-F6 scale-equivariant wavelet stack.

Pins the mechanism: one mother filter is shared across dyadic à trous dilations,
the response is exactly dilation-equivariant on upsampled signals, the stack is
causal, identity-at-init, parameter sharing beats per-scale filters, gradients
reach the shared filter, and the operator is NM-10 measurable.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.scale_equivariant_wavelet import (
    ScaleEquivariantWaveletStack,
    causal_atrous_conv1d,
    causal_atrous_kernel,
    scale_wavelet_param_count,
)


def _activated(dim: int = 16, kernel_size: int = 4, n_scales: int = 4) -> ScaleEquivariantWaveletStack:
    mix = ScaleEquivariantWaveletStack(
        dim=dim,
        kernel_size=kernel_size,
        n_scales=n_scales,
    )
    with torch.no_grad():
        mix.residual_scale.fill_(1.0)
    return mix


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = _activated()
    x = torch.randn(2, 17, 16)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 8, 16, 33])
def test_identity_at_init(d: int) -> None:
    mix = ScaleEquivariantWaveletStack(dim=d, kernel_size=4, n_scales=3)
    x = torch.randn(3, 9, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d}"


def test_expanded_kernels_reuse_the_same_mother_filter() -> None:
    mix = ScaleEquivariantWaveletStack(dim=8, kernel_size=5, n_scales=4)
    for scale in range(mix.n_scales):
        dilation = mix.dilation_for_scale(scale)
        dense = mix.expanded_kernel(scale)
        recovered = dense.flip(0)[::dilation][: mix.kernel_size]
        assert torch.allclose(recovered, mix.mother_filter, atol=0.0)
        nonzero = dense.abs() > 0
        assert int(nonzero.sum()) == mix.kernel_size


def test_causal_atrous_kernel_lag_convention() -> None:
    taps = torch.tensor([2.0, 3.0, 5.0])
    kernel = causal_atrous_kernel(taps, dilation=2)
    assert torch.equal(kernel, torch.tensor([5.0, 0.0, 3.0, 0.0, 2.0]))
    x = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)
    y = causal_atrous_conv1d(x, kernel)
    # t=4: 2*x4 + 3*x2 + 5*x0 with zero-indexed x values 5,3,1.
    assert float(y[0, 4, 0]) == 2 * 5 + 3 * 3 + 5 * 1


def test_exact_dilation_equivariance_on_upsampled_signal() -> None:
    """Scale 1 on a zero-interleaved signal equals scale 0 on the coarse signal."""
    torch.manual_seed(0)
    mix = ScaleEquivariantWaveletStack(dim=3, kernel_size=4, n_scales=3)
    coarse = torch.randn(2, 11, 3)
    up = coarse.new_zeros(2, 22, 3)
    up[:, ::2] = coarse
    coarse_scale0 = mix.wavelet_features(coarse)[:, :, 0]
    up_scale1_even = mix.wavelet_features(up)[:, ::2, 1]
    torch.testing.assert_close(up_scale1_even, coarse_scale0, atol=1e-6, rtol=1e-6)


def test_causality_future_tokens_do_not_leak() -> None:
    mix = _activated()
    x = torch.randn(1, 18, 16)
    y = mix(x)
    perturbed = x.clone()
    perturbed[:, 10:] = torch.randn_like(perturbed[:, 10:])
    y_perturbed = mix(perturbed)
    assert torch.allclose(y[:, :10], y_perturbed[:, :10], atol=1e-6)
    assert not torch.allclose(y[:, 10:], y_perturbed[:, 10:], atol=1e-4)


def test_receptive_field_grows_exponentially_with_scales() -> None:
    mix = ScaleEquivariantWaveletStack(dim=4, kernel_size=3, n_scales=6)
    assert mix.max_receptive_field == 1 + 2 * (2**5)
    x = torch.zeros(1, mix.max_receptive_field + 2, 4)
    x[:, 0] = 1.0
    features = mix.wavelet_features(x)
    last_touched = (mix.kernel_size - 1) * mix.dilation_for_scale(mix.n_scales - 1)
    assert features[0, last_touched, -1].abs().sum() > 0
    assert features[0, last_touched + 1 :, -1].abs().sum() == 0


def test_param_count_uses_one_filter_not_per_scale_filters() -> None:
    d, kernel_size, n_scales = 32, 8, 6
    mix = ScaleEquivariantWaveletStack(
        dim=d,
        kernel_size=kernel_size,
        n_scales=n_scales,
    )
    expected = kernel_size + n_scales + d * d + 1
    untied_filter_cost = kernel_size * n_scales + n_scales + d * d + 1
    assert scale_wavelet_param_count(d, kernel_size, n_scales) == expected
    assert mix.num_parameters == sum(p.numel() for p in mix.parameters())
    assert mix.num_parameters == expected
    assert mix.num_parameters < untied_filter_cost


def test_zero_mean_mother_filter_rejects_constant_signal() -> None:
    mix = ScaleEquivariantWaveletStack(dim=5, kernel_size=8, n_scales=4)
    assert abs(float(mix.mother_filter.sum().detach())) < 1e-6
    x = torch.ones(2, mix.max_receptive_field + 8, 5)
    # Ignore the causal boundary where zero-padding creates a transient.
    steady = mix.wavelet_features(x)[:, mix.max_receptive_field :, :, :]
    assert steady.abs().max() < 1e-5


def test_backward_flows_to_shared_filter_and_scale_mixer() -> None:
    mix = _activated()
    x = torch.randn(2, 13, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("mother_filter", "scale_mix", "out_lift.weight", "residual_scale"):
        param = dict(mix.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        ScaleEquivariantWaveletStack(dim=0)
    with pytest.raises(ValueError):
        ScaleEquivariantWaveletStack(dim=8, kernel_size=1)
    with pytest.raises(ValueError):
        ScaleEquivariantWaveletStack(dim=8, n_scales=0)
    mix = ScaleEquivariantWaveletStack(dim=8, n_scales=2)
    with pytest.raises(ValueError):
        mix.dilation_for_scale(2)


def test_measurable_by_physics_descriptor_probe() -> None:
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = _activated()
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
