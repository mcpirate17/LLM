"""Tests for NM-C16 — p-adic low-precision gated mixer.

Pins the spec:
- The gate is the VALIDATED NON-softmax per-channel padic highway
  ``g = sigmoid(Wg·x + Wv·v_p(x) + b)`` (collapse-proof, mirrors
  ``_op_padic_gated_mixer``) — per-token, NOT a softmax over tokens (NM-11 twin).
- Identity-at-init via ReZero (``residual_scale=0`` ⟹ ``out=x``).
- **The NM-C16 novelty — ``padic_truncate`` keeps ``n_digits`` significant base-``p``
  digits of the projection**, and its truncation error is bounded by
  ``p^(1-n_digits)/2`` in RELATIVE magnitude INDEPENDENT of scale (the p-adic
  "low bit-width is lossless for the scales the mixer discards anyway" property).
  ``n_digits`` is the precision knob: coarser ⟹ more error / fewer distinct
  levels. ``round`` is straight-through (STE) so the low-prec forward is trainable.
- Self-contained ``[B,L,D]→[B,L,D]`` mixer ⟹ NM-10-measurable (finite physics
  fingerprint) and NM-11 softmax-twin-detectable.
- Distinct from NM-C15 (ternary WEIGHT bits): NM-C16 quantizes forward ACTIVATIONS,
  principled by p-adic valuation.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.padic_lowprec_mix import (
    PadicLowPrecMixer,
    padic_lowprec_param_count,
    padic_truncate,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = PadicLowPrecMixer(dim=32, n_digits=4)
    with torch.no_grad():
        mix.residual_scale.fill_(1.0)  # open the low-prec path
    x = torch.randn(4, 10, 32)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d", [8, 16, 32, 64])
def test_identity_at_init(d: int) -> None:
    """ReZero ``residual_scale=0`` ⟹ ``out == x`` exactly at init (safe drop-in)."""
    mix = PadicLowPrecMixer(dim=d)
    x = torch.randn(3, 7, d)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-7)


@pytest.mark.parametrize("d", [8, 16, 32])
def test_param_count_matches_helper_and_numel(d: int) -> None:
    mix = PadicLowPrecMixer(dim=d)
    assert mix.num_parameters == padic_lowprec_param_count(d)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters
    # 3 DxD matrices (gate_x, gate_v, proj) + D gate bias + 1 ReZero scale
    assert mix.num_parameters == 3 * d * d + d + 1


@pytest.mark.parametrize("p,n_digits", [(2, 4), (2, 8), (3, 4)])
def test_mantissa_bits_is_n_digits_log2_p(p: int, n_digits: int) -> None:
    mix = PadicLowPrecMixer(dim=16, n_digits=n_digits, p=p)
    assert math.isclose(mix.mantissa_bits, n_digits * math.log2(p), rel_tol=1e-9)


@pytest.mark.parametrize("n_digits", [2, 4, 8])
def test_padic_truncate_relative_precision_bound_scale_independent(
    n_digits: int,
) -> None:
    """THE NM-C16 claim: truncating to ``n_digits`` p-adic digits bounds the error
    at ``p^(1-n_digits)/2`` RELATIVE magnitude, INDEPENDENT of ``z``'s scale. A
    value of ~1e-4 and one of ~1e4 each keep ``n_digits`` significant digits, so
    the discarded (high-valuation) tail never moves more than half a kept-digit —
    "low bit-width is lossless for the scales the mixer discards anyway"."""
    p = 2
    # magnitudes spanning 8 decades, off grid so rounding actually bites
    z = torch.tensor(
        [10.0**k * (0.37 + 0.21 * i) for k in range(-4, 5) for i in range(3)],
        dtype=torch.float64,
    )
    z = torch.cat([z, -z])
    q = padic_truncate(z, p, n_digits)
    rel = (q - z).abs() / z.abs()
    bound = p ** (1 - n_digits) / 2.0
    assert float(rel.max()) <= bound * 1.5, (
        f"n_digits={n_digits}: max_rel_err={rel.max():.3e} > bound*1.5={bound * 1.5:.3e}"
    )
    # the bound is non-trivial: under full precision (large n_digits) the error -> 0
    assert bound < 1.0


def test_padic_truncate_is_valuation_adaptive() -> None:
    """Large- and small-magnitude values each keep ``n_digits`` digits: very
    different ABSOLUTE quantization steps but the SAME bounded RELATIVE precision
    (the p-adic / floating-point property a uniform fixed-step quantizer lacks)."""
    p, n_digits = 2, 4
    big = torch.tensor([1e3, 5e3, -2e3], dtype=torch.float64)
    small = torch.tensor([1e-3, 5e-3, -2e-3], dtype=torch.float64)
    q_big = padic_truncate(big, p, n_digits)
    q_small = padic_truncate(small, p, n_digits)
    rel_big = (q_big - big).abs() / big.abs()
    rel_small = (q_small - small).abs() / small.abs()
    # absolute steps differ by ~6 decades (the magnitudes do)
    assert (q_big.abs().mean() / q_small.abs().mean()) > 1e4
    # but relative precision is the SAME order (both ≤ the n_digits bound)
    bound = p ** (1 - n_digits) / 2.0
    assert float(rel_big.max()) <= bound * 1.5
    assert float(rel_small.max()) <= bound * 1.5


def test_lower_n_digits_is_coarser() -> None:
    """``n_digits`` is the precision knob: fewer digits ⟹ (1) larger relative
    error and (2) fewer distinct quantized levels over a fixed range."""
    p = 2
    z = torch.linspace(1.1, 9.9, 60, dtype=torch.float64)
    q2 = padic_truncate(z, p, 2)
    q8 = padic_truncate(z, p, 8)
    err2 = ((q2 - z).abs() / z.abs()).mean()
    err8 = ((q8 - z).abs() / z.abs()).mean()
    assert err2 > err8  # coarser ⟹ more error
    # and far fewer distinct output levels (low bit-width ⇒ collapsed codebook)
    assert (
        torch.unique(q2.round(decimals=6)).numel()
        < torch.unique(q8.round(decimals=6)).numel()
    )


def test_padic_truncate_preserves_sign() -> None:
    """Truncation rounds magnitude; it does not flip sign (a low-prec quantizer
    must not introduce sign errors into the projection path)."""
    z = torch.tensor([3.3, -7.7, 0.21, -0.91, 12.4, -0.05], dtype=torch.float64)
    q = padic_truncate(z, p=2, n_digits=3)
    assert torch.equal(torch.sign(q), torch.sign(z))


def test_ste_grad_passes_through_round_as_identity() -> None:
    """``round`` is straight-through: the forward truncates, the backward passes
    the upstream gradient through untouched (grad ≈ 1) so the low-prec path is
    trainable end-to-end."""
    z = torch.tensor([1.5, 2.5, 3.5, 4.5], requires_grad=True)
    q = padic_truncate(z, p=2, n_digits=3)
    # forward IS truncated (q lands on the grid, q != z in general)
    assert not torch.allclose(q, z)
    q.sum().backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.ones_like(z.grad))  # STE: identity through z


def test_gate_is_per_token_sigmoid_not_softmax() -> None:
    """The gate is a per-CHANNEL sigmoid in [0,1]^D informed by p-adic valuation —
    NOT a softmax over tokens (per-token sums ≠ 1). This is the validated
    collapse-proof padic gate, structurally distinct from ``softmax(QK^T)``."""
    mix = PadicLowPrecMixer(dim=16)
    x = torch.randn(3, 5, 16)
    with torch.no_grad():  # inspect gate values only
        g = mix.gate(x)
    assert g.shape == x.shape
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0
    # softmax over D would sum to 1.0 per token; a sigmoid sums to ~D/2
    sums = g.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums))
    assert float(sums.mean()) > 2.0  # well above the softmax-1 regime


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the mixer is pointwise (each token refined from
    its own features ⟹ ``cross_token_mixing ≈ 0``) and gated by a per-token
    sigmoid, not exponentiated dot-product attention. Confirmed not a softmax
    twin — the structural guarantee the softmax-router collapse cannot recur."""
    mix = PadicLowPrecMixer(dim=32, n_digits=4)
    with torch.no_grad():  # open the path + perturb weights (active, not identity)
        mix.residual_scale.fill_(1.0)
        mix.proj_w.add_(0.3 * torch.randn_like(mix.proj_w))
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mix)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f})"
    )
    assert props.cross_token_mixing < 0.1  # pointwise, not attention


def test_backward_reaches_all_weights_and_input() -> None:
    """With the path open, gradient reaches every learned weight and the input
    (STE through ``padic_truncate`` keeps the projection path differentiable)."""
    mix = PadicLowPrecMixer(dim=16, n_digits=4)
    with torch.no_grad():
        mix.residual_scale.fill_(1.0)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = mix(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    for name in ("gate_x", "gate_v", "proj_w", "gate_bias", "residual_scale"):
        p = getattr(mix, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_truncation_is_load_bearing_vs_full_precision() -> None:
    """NM-C16's distinctness from the plain validated padic mixer: the
    ``padic_truncate`` actually changes the forward. At ``n_digits`` large the
    mixer ≈ the full-precision padic highway; at small ``n_digits`` it carries
    the low-precision forward (precision efficiency, the compaction angle)."""
    x = torch.randn(2, 5, 24)
    fine = PadicLowPrecMixer(dim=24, n_digits=24)  # ≈ full precision
    coarse = PadicLowPrecMixer(dim=24, n_digits=1)  # ~1-bit mantissa
    with torch.no_grad():
        for mm in (fine, coarse):
            mm.residual_scale.fill_(1.0)
        # identical weights so the ONLY difference is the truncation width
        coarse.gate_x.copy_(fine.gate_x)
        coarse.gate_v.copy_(fine.gate_v)
        coarse.proj_w.copy_(fine.proj_w)
        coarse.gate_bias.copy_(fine.gate_bias)
        y_fine = fine(x)
        y_coarse = coarse(x)
    assert not torch.allclose(y_fine, y_coarse, atol=1e-5)
    assert float((y_fine - y_coarse).abs().max()) > 0.0


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the mixer exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside Monarch/Butterfly/Ternary."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = PadicLowPrecMixer(dim=16, n_digits=4)
    with torch.no_grad():  # open the path for a non-trivial fingerprint
        mix.residual_scale.fill_(1.0)
        mix.proj_w.add_(0.3 * torch.randn_like(mix.proj_w))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"


def test_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        PadicLowPrecMixer(dim=0)
    with pytest.raises(ValueError):
        PadicLowPrecMixer(dim=8, n_digits=0)
    with pytest.raises(ValueError):
        PadicLowPrecMixer(dim=8, p=1)
    with pytest.raises(ValueError):
        padic_truncate(torch.randn(4), n_digits=0)
