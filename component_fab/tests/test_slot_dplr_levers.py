"""Run-1 compositional levers on MultiHeadSlotTableMemoryLane:
content-aware per-slot forget + DPLR low-rank value + learnable slots.

Guards: all-off is byte-identical to the prior lane, DPLR is a no-op at init
(zero-init lr_out), every lever stays strictly causal, and content_forget
fails fast without the delta path.
"""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

_BASE = dict(
    memory_dim=32,
    n_slots=8,
    n_heads=4,
    use_delta_update=True,
    route_from_input=True,
    normalize_slot_values=True,
)


def _lane(**kw):
    torch.manual_seed(1)
    return MultiHeadSlotTableMemoryLane(64, **_BASE, **kw)


def test_all_levers_off_is_byte_identical():
    x = torch.randn(2, 16, 64)
    off = _lane()
    explicit = _lane(content_forget=False, dplr_value_rank=0, learnable_slots=False)
    with torch.no_grad():
        assert (off(x) - explicit(x)).abs().max().item() == 0.0
    assert sum(p.numel() for p in off.parameters()) == sum(
        p.numel() for p in explicit.parameters()
    )


def test_dplr_value_is_noop_at_init():
    # lr_out is zero-init, so the low-rank value correction starts as identity:
    # a fresh DPLR-value lane must match the base lane exactly before training.
    x = torch.randn(2, 16, 64)
    base = _lane()
    dplr = _lane(dplr_value_rank=16)
    with torch.no_grad():
        assert (base(x) - dplr(x)).abs().max().item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        dict(content_forget=True),
        dict(dplr_value_rank=16),
        dict(learnable_slots=True),
        dict(content_forget=True, dplr_value_rank=16, learnable_slots=True),
    ],
)
def test_levers_build_run_and_stay_causal(kw):
    lane = _lane(**kw)
    x = torch.randn(2, 24, 64)
    y = lane(x)
    assert y.shape == (2, 24, 64)
    assert torch.isfinite(y).all()
    # Perturbing the future must not change the past (strict causality).
    x2 = x.clone()
    x2[:, 12:] = torch.randn_like(x2[:, 12:])
    with torch.no_grad():
        a, b = lane(x)[:, :12], lane(x2)[:, :12]
    assert torch.allclose(a, b, atol=1e-5)


def test_content_forget_requires_delta():
    with pytest.raises(ValueError, match="content_forget requires use_delta_update"):
        MultiHeadSlotTableMemoryLane(
            64, memory_dim=32, use_delta_update=False, content_forget=True
        )


def test_learnable_and_dplr_add_params_forget_decoupled_from_route():
    base_n = sum(p.numel() for p in _lane().parameters())
    assert sum(p.numel() for p in _lane(dplr_value_rank=16).parameters()) > base_n
    assert sum(p.numel() for p in _lane(learnable_slots=True).parameters()) > base_n
    # content_forget owns its own gate (forget_route), independent of write_route.
    lane = _lane(content_forget=True)
    assert hasattr(lane, "forget_route")
    assert hasattr(lane, "forget_diag")
