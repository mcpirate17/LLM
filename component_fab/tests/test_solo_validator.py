"""Tests for component_fab.validator.solo."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.primitive_templates import TropicalAttention
from component_fab.validator.solo import (
    SoloScorecard,
    append_scorecard,
    validate_solo,
)
from component_fab.tests.conftest import make_candidate_spec


def test_validate_solo_tropical_attention_promotes() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
        }
    )
    module = TropicalAttention(dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    assert isinstance(card, SoloScorecard)
    assert card.smoke["forward_passed"]
    assert card.smoke["backward_passed"]
    assert card.smoke["output_finite"]
    assert card.metrics
    assert card.property_cross_check.get("tropical_max_to_mean_ratio") is not None
    assert card.promoted is True


def test_validate_solo_broken_module_does_not_promote() -> None:
    class _Broken(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, : x.shape[-1] // 2]

    spec = make_candidate_spec({"op_algebraic_space": "euclidean"})
    card = validate_solo(spec, _Broken(), dim=16, seq_len=16)
    assert card.smoke["forward_passed"] is False
    assert "error" in card.smoke
    assert card.promoted is False


def test_validate_solo_from_spec_round_trip() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
        }
    )
    module = generate_module_from_spec(spec, dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    assert card.smoke["forward_passed"]
    assert card.property_cross_check.get("state_consistent") is True
    assert card.property_cross_check.get("tropical_consistent") is not None


def test_validate_solo_cross_checks_composed_math_knobs() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_math_knobs": (
                "calculus_finite_difference",
                "linear_algebra_low_rank",
                "sparse_matrix_banded",
            ),
        }
    )
    module = generate_module_from_spec(spec, dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    cross = card.property_cross_check
    assert cross["calculus_consistent"] is True
    assert cross["low_rank_consistent"] is True
    assert cross["sparse_banded_consistent"] is True
    assert cross["declared_math_knobs"] == [
        "calculus_finite_difference",
        "linear_algebra_low_rank",
        "sparse_matrix_banded",
    ]


def test_validate_solo_cross_checks_new_math_knobs() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_math_knobs": (
                "kernel_random_features",
                "multiscale_wavelet",
                "graph_laplacian_diffusion",
            ),
        }
    )
    module = generate_module_from_spec(spec, dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    cross = card.property_cross_check
    assert cross["kernel_random_features_consistent"] is True
    assert cross["multiscale_wavelet_consistent"] is True
    assert cross["graph_diffusion_consistent"] is True
    assert cross["graph_diffusion_future_drift"] < 1e-5


def test_validate_solo_cross_checks_lambda_math_knob() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_math_knobs": ("lambda_functional_blend",),
            "op_lambda_gate": "content",
            "op_lambda_basis": "phase",
        }
    )
    module = generate_module_from_spec(spec, dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    cross = card.property_cross_check
    assert cross["declared_math_knobs"] == ["lambda_functional_blend"]
    assert cross["lambda_functional_consistent"] is True
    assert 0.0 <= cross["lambda_functional_gate_mean"] < 0.1


def test_validate_solo_cross_checks_exotic_math_knob_stack() -> None:
    spec = make_candidate_spec(
        {
            "op_algebraic_space": "hyperbolic",
            "op_dynamical_has_state": 0,
            "op_math_knobs": ("tropical_knob", "padic_knob"),
        }
    )
    module = generate_module_from_spec(spec, dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    cross = card.property_cross_check
    assert cross["declared_math_knobs"] == ["tropical_knob", "padic_knob"]
    assert cross["tropical_knob_consistent"] is True
    assert cross["padic_knob_consistent"] is True


def test_append_scorecard_writes_jsonl(tmp_path: Path) -> None:
    spec = make_candidate_spec({"op_algebraic_space": "tropical"})
    module = TropicalAttention(dim=16)
    card = validate_solo(spec, module, dim=16, seq_len=16)
    out = tmp_path / "proposals.jsonl"
    append_scorecard(card, out)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    blob = json.loads(lines[0])
    assert blob["proposal_id"] == spec.proposal_id
    assert blob["smoke"]["forward_passed"] is True
