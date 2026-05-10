"""Tests for the mined-chain compile-validator."""

from __future__ import annotations

from research.meta_analysis.template_validator import (
    annotate_candidates_with_validation,
    filter_to_passing,
    validate_chain,
)


def test_simple_linear_chain_validates_and_compiles():
    result = validate_chain(["linear_proj", "rmsnorm"], model_dim=64)
    assert result["validate_passed"]
    assert result["compile_passed"]
    assert result["error"] is None


def test_unknown_op_fails_at_build():
    result = validate_chain(["this_op_does_not_exist"], model_dim=64)
    assert not result["validate_passed"]
    assert not result["compile_passed"]
    assert result["failure_mode"] == "build"


def test_multi_input_op_fails_at_build():
    """Mined chains are flat sequences; binary ops must be flagged."""
    result = validate_chain(["add"], model_dim=64)
    assert not result["validate_passed"]
    assert result["failure_mode"] == "build"
    assert "inputs" in (result["error"] or "")


def test_annotate_candidates_attaches_validation_block():
    cand_a = {"proposed_template_name": "good", "chain": ["linear_proj", "rmsnorm"]}
    cand_b = {"proposed_template_name": "bad", "chain": ["nope_unknown_op"]}
    out = annotate_candidates_with_validation([cand_a, cand_b], model_dim=64)
    assert "validation" in out[0] and "validation" in out[1]
    assert cand_a["validation"]["validate_passed"]
    assert not cand_b["validation"]["validate_passed"]


def test_filter_to_passing_drops_failures():
    cand_a = {"chain": ["linear_proj", "rmsnorm"]}
    cand_b = {"chain": ["nope_unknown_op"]}
    annotate_candidates_with_validation([cand_a, cand_b], model_dim=64)
    passing = filter_to_passing([cand_a, cand_b])
    assert passing == [cand_a]


def test_filter_validate_only_keeps_compile_failures():
    cand = {"chain": ["linear_proj", "rmsnorm"]}
    annotate_candidates_with_validation([cand], model_dim=64)
    # When require_compile=False, we keep candidates that validate even if
    # compile would fail. This case validates+compiles, so it stays in.
    assert filter_to_passing([cand], require_compile=False) == [cand]


def test_phase2_forward_backward_smoke_passes_on_simple_chain():
    """Healthy chain should pass forward + backward smoke test."""
    result = validate_chain(["linear_proj", "rmsnorm"], model_dim=64)
    assert result["forward_passed"]
    assert result["backward_passed"]
    assert not result["output_has_nan"]
    assert not result["output_has_inf"]
    assert result["param_grad_finite"]


def test_phase2_run_smoke_false_skips_runtime_check():
    """When run_smoke=False, only Phase 1 (build+validate+compile) runs."""
    result = validate_chain(["linear_proj", "rmsnorm"], model_dim=64, run_smoke=False)
    assert result["compile_passed"]
    assert not result["forward_passed"]  # never attempted
    assert not result["backward_passed"]


def test_filter_to_passing_with_backward_gate():
    """require_backward filters out compile-passing but smoke-failing chains."""
    cand_good = {"chain": ["linear_proj", "rmsnorm"]}
    annotate_candidates_with_validation([cand_good], model_dim=64)
    # The good chain passes backward, so it should survive the strictest gate.
    passing = filter_to_passing(
        [cand_good], require_forward=True, require_backward=True
    )
    assert passing == [cand_good]
