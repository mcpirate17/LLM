"""Tests for invention-track mechanisms and CLI dry-run."""

from __future__ import annotations

import json

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    HierarchicalResidualCompressorLane,
    PadicSurpriseMemoryLane,
    SemiringSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)
from component_fab.generator.primitive_templates import SymplecticResidualMixerLane
from component_fab.inventor.mechanism_catalog import (
    enumerate_invention_specs,
    invention_gate_reasons,
)
from component_fab.improver.axis_variants import (
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    AnchorAxes,
    spec_for_variant,
)


def test_invention_specs_are_unanchored_and_have_contracts() -> None:
    specs = enumerate_invention_specs()
    assert len(specs) >= 4
    for spec in specs:
        assert spec.anchor_witness_op == ""
        assert spec.anchor_witnesses_all == ()
        assert spec.math_axes["op_search_track"] == "invention"
        assert spec.math_axes["op_invention_mechanism"]
        assert not invention_gate_reasons(spec)


def test_invention_gate_rejects_rehab_axis_variant() -> None:
    anchor = AnchorAxes(
        op_name="toy_anchor",
        axes={
            "op_algebraic_space": "tropical",
            "op_spectral_preferred_basis": "content",
            "op_dynamical_memory_length_class": "O(1)",
            "op_dynamical_has_state": 0,
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "local",
        },
        eval_count=1,
        pass_rate=0.0,
    )
    spec = spec_for_variant(anchor, DEFAULT_AXIS_VARIANT_TEMPLATES[0])
    reasons = invention_gate_reasons(spec)
    assert "missing invention search track" in reasons
    assert "anchored rehab/cross-anchor spec" in reasons


def test_semiring_surprise_memory_generalizes_the_family_read() -> None:
    """The semiring lane is the ONLY family member that overrides ``_read`` —
    with a learnable tempered semiring. It must stay causal/finite, and its read
    must *interpolate* the family's algebra: high beta tracks the proven max-plus
    read, low beta tracks the arithmetic mean, monotonically in beta. So it
    strictly generalizes ``TropicalSurpriseMemoryLane`` (the family's fixed
    max-plus read) via a single learnable parameter."""
    assert "_read" in SemiringSurpriseMemoryLane.__dict__  # the novelty lives here
    torch.manual_seed(0)
    lane = SemiringSurpriseMemoryLane(32)
    x = torch.randn(2, 24, 32, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # strict causality: scrambling the second half cannot move the first half
    torch.manual_seed(1)
    lane2 = SemiringSurpriseMemoryLane(16)
    x_a = torch.randn(1, 12, 16)
    x_b = x_a.clone()
    x_b[:, 6:] += torch.randn(1, 6, 16)
    assert torch.allclose(lane2(x_a)[:, :6], lane2(x_b)[:, :6], atol=1e-5)
    # Compare _read directly on a fixed memory (isolates the retrieval algebra;
    # the full scan would compound tiny per-step read deltas through the
    # recurrence). beta-large -> max-plus (tropical); beta-small -> mean.
    torch.manual_seed(3)
    semi = SemiringSurpriseMemoryLane(24)
    m = semi.memory_dim
    mem = torch.randn(2, m, m)
    addr = torch.randn(2, m)
    scores = mem + addr.unsqueeze(-1)
    max_read = scores.amax(dim=1)
    mean_read = scores.mean(dim=1)

    def read_at(beta_raw):
        with torch.no_grad():
            semi.semiring_temp.fill_(beta_raw)
            return semi._read(mem, addr)

    sharp = read_at(50.0)  # softplus -> clamp 30
    soft = read_at(-50.0)  # softplus -> 1e-2
    mid = read_at(0.0)  # softplus -> ~0.69
    # beta->0 limit is the arithmetic mean (the -log m normalizer guarantees it)
    assert (soft - mean_read).abs().max() < 1e-2
    # genuine interpolation: sharp sits toward max, soft toward mean
    assert (sharp - max_read).abs().mean() < (sharp - mean_read).abs().mean()
    assert (soft - mean_read).abs().mean() < (soft - max_read).abs().mean()
    # monotone in beta: larger beta -> strictly closer to the max-plus read
    d_soft = (soft - max_read).abs().mean()
    d_mid = (mid - max_read).abs().mean()
    d_sharp = (sharp - max_read).abs().mean()
    assert d_sharp < d_mid < d_soft


def test_invention_codegen_dispatches_mechanisms() -> None:
    specs = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    expected = {
        "causal_fast_weight_memory": CausalFastWeightMemoryLane,
        "causal_slot_router_memory": CausalSlotRouterMemoryLane,
        "hierarchical_residual_compressor": HierarchicalResidualCompressorLane,
        "symplectic_residual_mixer": SymplecticResidualMixerLane,
        "tropical_surprise_memory": TropicalSurpriseMemoryLane,
        "semiring_surprise_memory": SemiringSurpriseMemoryLane,
        "padic_surprise_memory": PadicSurpriseMemoryLane,
    }
    x = torch.randn(2, 8, 16)
    for mechanism, cls in expected.items():
        module = generate_module_from_spec(specs[mechanism], dim=16)
        assert isinstance(module, cls)
        assert module(x).shape == x.shape


def test_surprise_memory_lanes_are_causal_finite_and_share_read() -> None:
    """The Titans/TTT delta-rule lanes must be strictly causal (left-to-right
    scan), produce finite forward+backward, and share the family's max-plus
    retrieval (neither subclass overrides ``_read``)."""
    assert "_read" not in TropicalSurpriseMemoryLane.__dict__
    assert "_read" not in PadicSurpriseMemoryLane.__dict__
    for cls in (TropicalSurpriseMemoryLane, PadicSurpriseMemoryLane):
        torch.manual_seed(0)
        lane = cls(32)
        x = torch.randn(2, 24, 32, requires_grad=True)
        y = lane(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        y.pow(2).mean().backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in lane.parameters()
        )
        # Causality: scrambling the second half must not move the first half.
        with torch.no_grad():
            base = lane(x.detach())
            scrambled = x.detach().clone()
            scrambled[:, 12:] = torch.randn(2, 12, 32)
            moved = lane(scrambled)
            drift = (base[:, :12] - moved[:, :12]).abs().max().item()
        assert drift < 1e-5, f"{cls.__name__} leaked future info: drift={drift}"


def test_run_invention_dry_run_outputs_active_specs(capsys) -> None:
    from component_fab.tools.run_invention import main

    assert main(["--dry-run", "--max-specs", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["active"]) == 2
    assert payload["blocked"] == []
    assert payload["active"][0]["math_axes"]["op_search_track"] == "invention"
