"""Tests for the canonical program_result_kwargs_from_s1 builder.

Covers the contract that the regular pipeline AND the ablation pipeline
both depend on: any metric present in a `_micro_train` s1 dict lands in
the program_results row.
"""

from __future__ import annotations

import json

import pytest

from research.scientist.runner._helpers import (
    S1_REQUIRED_POST_METRIC_COLUMNS,
    program_result_kwargs_from_s1,
    s1_post_metric_completeness,
)


def _full_s1() -> dict:
    """A complete `_micro_train` s1 dict with every probe populated."""
    return {
        "passed": True,
        "final_loss": 5.0,
        "loss_ratio": 0.42,
        "initial_loss": 7.5,
        "min_loss": 5.0,
        "loss_improvement_rate": 0.3,
        "avg_step_time_ms": 12.0,
        "total_train_time_ms": 12000.0,
        "max_grad_norm": 2.5,
        "mean_grad_norm": 1.1,
        "grad_norm_std": 0.4,
        "n_train_steps": 1000,
        "final_lr": 1e-4,
        "validation_loss": 5.4,
        "validation_loss_ratio": 0.45,
        "generalization_gap": 0.03,
        "discovery_loss": 5.1,
        "discovery_loss_ratio": 0.43,
        "train_budget_steps": 1000,
        "param_count": 1234567,
        "throughput_tok_s": 8000.0,
        "wikitext_perplexity": 150.0,
        "wikitext_score": 0.55,
        "wikitext_pre_perplexity": 800.0,
        "hellaswag_acc": 0.31,
        "hellaswag_status": "ran",
        "blimp_overall_accuracy": 0.55,
        "blimp_status": "ran",
        "induction_auc": 0.21,
        "binding_auc": 0.18,
        "binding_composite": 0.12,
        "ar_auc": 0.06,
        "fp_jacobian_erf_density": 0.55,
        "fp_jacobian_erf_variance": 0.08,
        "fp_icld_velocity": -0.02,
        "fp_icld_delta_loss": -0.4,
        "fp_logit_margin_velocity": 0.1,
        "fp_logit_margin_delta": 0.3,
        "perf_report": {"flops_ms": 7.5},
        "kernel_timings_ms": {"matmul": 2.1},
        "starvation_report": {"steps": 0},
        "_behavioral_fingerprint": {
            "interaction_locality": 0.4,
            "interaction_sparsity": 0.7,
            "isotropy": 0.6,
            "rank_ratio": 0.5,
            "jacobian_spectral_norm": 1.2,
            "jacobian_effective_rank": 32.0,
            "sensitivity_uniformity": 0.55,
            "cka_vs_transformer": 0.6,
            "cka_vs_ssm": 0.4,
            "cka_vs_conv": 0.2,
            "hierarchy_fitness": 0.7,
            "gromov_delta": 0.05,
            "intrinsic_dim": 16.0,
            "interaction_symmetry": 0.5,
            "interaction_hierarchy": 0.6,
        },
        "pruning_ratio": 0.05,
    }


def test_canonical_kwargs_includes_every_required_post_s1_metric():
    s1 = _full_s1()
    kwargs = program_result_kwargs_from_s1(s1, model_source="ablation")
    for col in S1_REQUIRED_POST_METRIC_COLUMNS:
        assert col in kwargs, f"required post-S1 column missing: {col}"
        assert kwargs[col] is not None, f"required post-S1 column is None: {col}"
    assert kwargs["model_source"] == "ablation"
    assert kwargs["loss_ratio"] == pytest.approx(0.42)
    assert kwargs["final_loss"] == pytest.approx(5.0)


def test_canonical_kwargs_reconstructs_behavioral_fingerprint():
    kwargs = program_result_kwargs_from_s1(_full_s1(), model_source="ablation")
    assert "fingerprint_json" in kwargs
    fp = json.loads(kwargs["fingerprint_json"])
    assert isinstance(fp, dict) and fp
    assert kwargs["fp_interaction_locality"] == pytest.approx(0.4)
    assert kwargs["fp_jacobian_spectral_norm"] == pytest.approx(1.2)
    assert kwargs["fp_cka_vs_transformer"] == pytest.approx(0.6)


def test_canonical_kwargs_serializes_perf_kernel_starvation_reports():
    kwargs = program_result_kwargs_from_s1(_full_s1(), model_source="screening")
    assert json.loads(kwargs["perf_report_json"]) == {"flops_ms": 7.5}
    assert json.loads(kwargs["kernel_timings_json"]) == {"matmul": 2.1}
    assert json.loads(kwargs["starvation_report_json"]) == {"steps": 0}


def test_canonical_kwargs_propagates_pruning_prefix_fields():
    s1 = _full_s1()
    s1["pruning_target_ratio"] = 0.1
    s1["pruning_actual_ratio"] = 0.08
    kwargs = program_result_kwargs_from_s1(s1, model_source="ablation")
    assert kwargs["pruning_ratio"] == pytest.approx(0.05)
    assert kwargs["pruning_target_ratio"] == pytest.approx(0.1)
    assert kwargs["pruning_actual_ratio"] == pytest.approx(0.08)


def test_canonical_kwargs_skips_none_metric_values():
    s1 = _full_s1()
    s1["hellaswag_acc"] = None
    s1["wikitext_perplexity"] = None
    kwargs = program_result_kwargs_from_s1(s1, model_source="ablation")
    assert "hellaswag_acc" not in kwargs
    assert "wikitext_perplexity" not in kwargs
    audit = s1_post_metric_completeness(kwargs)
    assert "hellaswag_acc" in audit["missing"]
    assert "wikitext_perplexity" in audit["missing"]
    assert audit["is_complete"] is False


def test_canonical_kwargs_extra_overlay_wins_for_provenance():
    kwargs = program_result_kwargs_from_s1(
        _full_s1(),
        model_source="ablation",
        extra={
            "trust_label": "ablation_metric_backfill_replay",
            "comparability_label": "reconstructed_init_variant",
            "evaluation_protocol_version": "ablation_metric_backfill_v1",
        },
    )
    assert kwargs["trust_label"] == "ablation_metric_backfill_replay"
    assert kwargs["comparability_label"] == "reconstructed_init_variant"
    assert kwargs["evaluation_protocol_version"] == "ablation_metric_backfill_v1"


def test_completeness_audit_full_row_is_complete():
    full = {col: 0.5 for col in S1_REQUIRED_POST_METRIC_COLUMNS}
    audit = s1_post_metric_completeness(full)
    assert audit["is_complete"] is True
    assert audit["coverage"] == pytest.approx(1.0)
    assert not audit["missing"]


def test_completeness_audit_partial_row():
    partial = {col: 0.5 for col in S1_REQUIRED_POST_METRIC_COLUMNS[:5]}
    audit = s1_post_metric_completeness(partial)
    assert audit["is_complete"] is False
    assert len(audit["missing"]) == len(S1_REQUIRED_POST_METRIC_COLUMNS) - 5
    assert audit["coverage"] == pytest.approx(5 / len(S1_REQUIRED_POST_METRIC_COLUMNS))


def test_canonical_kwargs_passes_error_fields_when_failed():
    failed = {
        "passed": False,
        "error_type": "ShapeMismatch",
        "error": "matmul wants [B,T,D]",
    }
    kwargs = program_result_kwargs_from_s1(failed, model_source="ablation")
    assert kwargs["error_type"] == "ShapeMismatch"
    assert kwargs["error_message"] == "matmul wants [B,T,D]"
    assert kwargs["model_source"] == "ablation"
