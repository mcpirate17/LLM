"""Tests for the fno_spectral_mix novel Fourier neural-operator sequence mixer (NM-6).

Covers: registry wiring, fwd/bwd finiteness, and the three structural claims that make this NOT a
softmax twin and genuinely a spectral mixer — (1) low-pass truncation: high-frequency content is
removed by the spectral branch while low-frequency content passes (a frequency-selective pass/stop
property softmax/local-conv mixers provably lack), (2) global receptive field: perturbing one input
token changes the output at a distant token (the Fourier basis couples all positions, unlike a local
conv), and (3) near-identity at init (small mode weights + zero bias -> residual-dominated).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.synthesis.compiled_op_params import CompiledOpParamInitMixin
from research.synthesis.compiler_ops_routing import OP_IMPLS
from research.synthesis.primitives import get_primitive


class _Host(nn.Module, CompiledOpParamInitMixin):
    """Minimal param host exercising the real init method + forward together."""

    model_dim = 16


def _build(dim: int = 16) -> _Host:
    host = _Host()
    host._init_fno_spectral_mix(dim)
    return host


def test_primitive_registered():
    op = get_primitive("fno_spectral_mix")
    assert op.name == "fno_spectral_mix"
    assert op.has_params
    assert op.binding_range_class == "full"
    assert "fno_spectral_mix" in OP_IMPLS


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16)
    out = OP_IMPLS["fno_spectral_mix"](host, [x], {})
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_grads_finite():
    torch.manual_seed(1)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = OP_IMPLS["fno_spectral_mix"](host, [x], {})
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name in (
        "fno_in_proj",
        "fno_out_proj",
        "fno_modes_real",
        "fno_modes_imag",
        "fno_bias",
    ):
        p = getattr(host, name)
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_low_pass_truncation():
    """The spectral branch is a low-pass filter: passes retained low frequencies, removes high ones.

    Configure the branch as a clean pass-through on the low modes (in/out = I, low modes = I, no
    bias). A pure LOW-frequency input (mode 1, within the 4 retained modes) is reproduced by the
    branch -> output approx 2x (branch + residual). A pure HIGH-frequency input (mode 8, beyond the
    4 retained modes) is zeroed by the truncation -> branch approx 0 -> output approx x (residual
    only). This frequency-selective pass/stop is the structural property that makes the mixer a
    spectral operator, not a softmax twin.
    """
    torch.manual_seed(2)
    D = 16
    S = 32  # rfft gives 17 modes; modes 0..3 retained, mode 8 truncated
    host = _build(dim=D)
    with torch.no_grad():
        eye = torch.eye(D)
        host.fno_in_proj.copy_(eye)
        host.fno_out_proj.copy_(eye)
        host.fno_modes_real.copy_(
            eye.unsqueeze(0).repeat(4, 1, 1)
        )  # pass low modes through
        host.fno_modes_imag.zero_()
        host.fno_bias.zero_()
    s = torch.arange(S, dtype=torch.float32)
    two_pi = 2.0 * 3.141592653589793
    # LOW frequency: cos(2 pi * 1 * s / S) -> energy at rfft mode 1 (retained).
    x_low = torch.zeros(1, S, D)
    x_low[0, :, 0] = torch.cos(two_pi * 1.0 * s / S)
    # HIGH frequency: cos(2 pi * 8 * s / S) -> energy at rfft mode 8 (truncated, >= k=4).
    x_high = torch.zeros(1, S, D)
    x_high[0, :, 0] = torch.cos(two_pi * 8.0 * s / S)
    out_low = OP_IMPLS["fno_spectral_mix"](host, [x_low], {})
    out_high = OP_IMPLS["fno_spectral_mix"](host, [x_high], {})
    # Low mode passes through the identity spectral filter -> out approx x_low + x_low = 2 x_low.
    assert torch.allclose(out_low[0, :, 0], 2.0 * x_low[0, :, 0], atol=1e-4)
    # High mode is truncated (zeroed) -> branch approx 0 -> out approx x_high (residual only).
    assert torch.allclose(out_high[0, :, 0], x_high[0, :, 0], atol=1e-4)


def test_global_receptive_field():
    """The FFT basis is global: perturbing input token 0 must change the output at a distant token.

    A purely LOCAL mixer (windowed/banded conv) would leave position S-1 unchanged when only position
    0 is perturbed. The FNO spectral branch couples every position to every other via the Fourier
    basis, so the distant output must move. (The direct residual only adds x[S-1], so any change at
    out[S-1] comes from the global spectral branch.)
    """
    torch.manual_seed(3)
    D = 16
    S = 16
    host = _build(dim=D)
    x = torch.randn(1, S, D)
    out_ref = OP_IMPLS["fno_spectral_mix"](host, [x], {})
    x_pert = x.clone()
    x_pert[0, 0, 0] += 1.0  # perturb ONLY token 0
    out_pert = OP_IMPLS["fno_spectral_mix"](host, [x_pert], {})
    delta_far = (out_pert - out_ref)[0, S - 1, :]  # distant output position
    assert delta_far.abs().max().item() > 1e-6, delta_far.abs().max().item()


def test_near_identity_at_init():
    """Small mode weights (std 0.02) + zero bias -> spectral branch approx 0 -> out approx x at init."""
    torch.manual_seed(4)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16)
    out = OP_IMPLS["fno_spectral_mix"](host, [x], {})
    rel = (out - x).abs().mean().item() / x.abs().mean().item()
    assert rel < 0.2, rel  # residual-dominated regime at init
