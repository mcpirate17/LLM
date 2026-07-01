"""Tests for NM-F4 internal-model integral controller.

Pins the spec: identity-at-init, zero steady-state error while the gate is open
(the internal model principle at work), EXACT retention once the gate closes —
flat where decay/EMA recurrences forget geometrically by parameterization (the
in-test EMA baseline makes the contrast quantitative) — anti-windup clamp that is
provably load-bearing, strictly positive integral gains, an O(D) control core,
and a finite NM-10 physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.integral_control_gate import (
    IntegralControlMixer,
    control_param_count,
    integral_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = IntegralControlMixer(dim=16)
    x = torch.randn(2, 10, 16)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 8, 16, 33])
def test_identity_at_init(d: int) -> None:
    """Zero-init output lift ⟹ the mixer is an exact no-op drop-in at init."""
    mix = IntegralControlMixer(dim=d)
    x = torch.randn(3, 6, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_zero_steady_state_error_gate_open() -> None:
    """Internal model principle: with the gate open and a constant reference, the
    integral action drives the readout error to zero GEOMETRICALLY — the readout
    d̂⊙s locks onto the input with no residual offset."""
    d = 8
    mix = IntegralControlMixer(dim=d)
    with torch.no_grad():
        mix.gate_bias.fill_(20.0)  # sigmoid(20) ≈ 1: gate open
    x = torch.randn(1, 1, d).repeat(1, 200, 1)  # constant reference
    states, errors = mix.integrate(x)
    e0 = errors[0, 0].abs().max()
    e_last = errors[0, -1].abs().max()
    assert e_last < 1e-3 * e0, "integral action failed to null the error"
    # Readout tracks the (identity-lifted) input exactly at steady state.
    assert torch.allclose(mix.d_hat * states[0, -1], x[0, -1], atol=1e-3)


def test_retention_is_flat_where_ema_decays() -> None:
    """The headline structural claim: once the gate closes, s_t = s_{t-1} exactly
    — 512 distractors cost nothing. The same trajectory under an EMA (decay by
    parameterization, the SSM failure mode) loses >99% of the bound value."""
    d, write_steps, distract_steps = 8, 60, 512
    mix = IntegralControlMixer(dim=d)
    with torch.no_grad():
        # Gate reads feature 0: x[0]=1 -> logit +20 (open); x[0]=0 -> -20 (closed).
        mix.gate_weight.zero_()
        mix.gate_weight[0] = 40.0
        mix.gate_bias.fill_(-20.0)
    write_token = torch.zeros(d)
    write_token[0] = 1.0
    write_token[1] = 2.0  # the fact to hold lives in channel 1
    gen = torch.Generator().manual_seed(0)
    distractors = torch.zeros(distract_steps, d)
    distractors[:, 2:] = torch.randn(distract_steps, d - 2, generator=gen)
    x = torch.cat([write_token.expand(write_steps, d), distractors], dim=0).unsqueeze(0)
    states, _ = mix.integrate(x)
    bound = states[0, write_steps - 1, 1]
    held = states[0, -1, 1]
    assert bound > 1.9  # the write phase actually converged near the target 2.0
    assert held > 0.9999 * bound, "integrator leaked state through a closed gate"
    # EMA baseline over the same distractor stream: geometric forgetting.
    ema = bound.clone()
    for _ in range(distract_steps):
        ema = 0.99 * ema  # distractor channel-1 input is 0
    assert ema < 0.01 * bound  # 0.99^512 ≈ 0.006: the decay mixer forgot the fact


def test_anti_windup_clamp_is_load_bearing() -> None:
    """Push the loop outside the stable band (k_i·d̂ > 2): with anti-windup the
    state stays inside ±s_max; the ablation without it blows past the clamp —
    proof the clamp does real work rather than decorating the recurrence."""
    d = 4
    # 12 steps: the divergent branch grows ~4^12 (finite), the clamp holds anyway.
    x = torch.full((1, 12, d), 3.0)

    def unstable(anti_windup: bool) -> torch.Tensor:
        mix = IntegralControlMixer(dim=d, anti_windup=anti_windup, s_max=10.0)
        with torch.no_grad():
            mix.raw_ki.fill_(5.0)  # softplus(5) ≈ 5 ⟹ |1 − k_i| = 4: divergent
            mix.gate_bias.fill_(20.0)
        states, _ = mix.integrate(x)
        return states.abs().max()

    assert unstable(anti_windup=True) <= 10.0
    assert unstable(anti_windup=False) > 10.0


def test_integral_gain_strictly_positive() -> None:
    mix = IntegralControlMixer(dim=16)
    with torch.no_grad():
        mix.raw_ki.normal_(0.0, 3.0)
    assert (mix.integral_gain() > 0).all()


def test_backward_flows_through_recurrence() -> None:
    mix = IntegralControlMixer(dim=8)
    with torch.no_grad():  # move off identity so gradients are non-trivial
        mix.out_lift.weight.add_(0.3 * torch.randn_like(mix.out_lift.weight))
    x = torch.randn(2, 12, 8, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mix.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert mix.raw_ki.grad.abs().sum() > 0
    assert mix.d_hat.grad.abs().sum() > 0


def test_param_counts_and_od_control_core() -> None:
    d = 32
    mix = IntegralControlMixer(dim=d)
    assert control_param_count(d) == 4 * d + 1
    assert integral_param_count(d) == 4 * d + 1 + 2 * d * d
    assert mix.control_parameters == 4 * d + 1
    assert mix.num_parameters == sum(p.numel() for p in mix.parameters())


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        IntegralControlMixer(dim=0)
    with pytest.raises(ValueError):
        IntegralControlMixer(dim=8, s_max=0.0)


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint so the mixer is scorable on the
    geometric-novelty axis alongside the other synthesis operators."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = IntegralControlMixer(dim=16)
    with torch.no_grad():  # nudge off identity for a non-trivial fingerprint
        mix.out_lift.weight.add_(0.4 * torch.randn_like(mix.out_lift.weight))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
