"""Tests for invention-track mechanisms and CLI dry-run."""

from __future__ import annotations

import json

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.primitive_templates import (
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    HierarchicalResidualCompressorLane,
    SymplecticResidualMixerLane,
)
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


def test_invention_codegen_dispatches_mechanisms() -> None:
    specs = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    expected = {
        "causal_fast_weight_memory": CausalFastWeightMemoryLane,
        "causal_slot_router_memory": CausalSlotRouterMemoryLane,
        "hierarchical_residual_compressor": HierarchicalResidualCompressorLane,
        "symplectic_residual_mixer": SymplecticResidualMixerLane,
    }
    x = torch.randn(2, 8, 16)
    for mechanism, cls in expected.items():
        module = generate_module_from_spec(specs[mechanism], dim=16)
        assert isinstance(module, cls)
        assert module(x).shape == x.shape


def test_run_invention_dry_run_outputs_active_specs(capsys) -> None:
    from component_fab.tools.run_invention import main

    assert main(["--dry-run", "--max-specs", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["active"]) == 2
    assert payload["blocked"] == []
    assert payload["active"][0]["math_axes"]["op_search_track"] == "invention"
