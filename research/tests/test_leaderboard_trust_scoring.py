import pytest

from research.scientist.leaderboard_scoring import (
    compute_champion_tiny_model_score_v1,
    compute_composite_v11,
    compute_composite_v12,
)


def _high_side_channel_candidate(**overrides):
    kw = dict(
        tier="validation",
        ppl_screening=622.37,
        ppl_investigation=622.37,
        ppl_validation=622.37,
        param_count=230_400,
        ppl_at_100=161_220.88,
        ppl_at_500=546.58,
        ppl_at_1000=622.37,
        screening_lr=0.4987,
        screening_nov=0.8449,
        novelty_confidence=0.9,
        blimp_accuracy=0.53004,
        hellaswag_acc_screening=0.2233,
        hellaswag_acc_investigation=0.2233,
        hellaswag_acc_validation=0.2233,
        tinystories_score=0.5488,
        cross_task_score=0.29376,
        diagnostic_score=0.01175,
        hierarchy_fitness=0.8193,
        ar_legacy_auc=0.0041,
        induction_screening_auc=0.0376,
        binding_screening_auc=0.00766,
        induction_intermediate_inv_auc=0.755,
        binding_intermediate_inv_auc=0.0946,
        fp_jacobian_erf_density=1.0,
        fp_id_collapse_rate=-0.01,
        fp_jacobian_erf_decay_slope=-0.05,
        fp_logit_margin_velocity=0.003,
        fp_jacobian_erf_variance=1_000.0,
        fp_icld_velocity=-0.017,
        screening_wikitext_metric_version="bpe_eval_v1",
    )
    kw.update(overrides)
    return kw


def test_v11_caps_candidate_without_reproduced_trust_signal():
    result = compute_composite_v11(
        decompose=True,
        **_high_side_channel_candidate(),
    )

    assert result["composite_score"] == pytest.approx(360.0)
    assert result["breakdown"]["_v11_trust_ceiling"] == 360.0
    assert result["breakdown"]["_v11_trust_low_loss"] is False
    assert result["breakdown"]["_v11_trust_understanding"] is False
    assert result["breakdown"]["_v11_trust_nonlocal_binding"] is False


def test_v11_allows_reproduced_low_loss_candidate_above_ceiling():
    result = compute_composite_v11(
        decompose=True,
        **_high_side_channel_candidate(
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
        ),
    )

    assert result["composite_score"] > 360.0
    assert "_v11_trust_ceiling" not in result["breakdown"]


def test_v11_allows_nonlocal_binding_candidate_above_ceiling():
    result = compute_composite_v11(
        decompose=True,
        **_high_side_channel_candidate(
            induction_intermediate_inv_auc=0.20,
            binding_intermediate_inv_auc=0.25,
        ),
    )

    assert result["composite_score"] > 360.0
    assert "_v11_trust_ceiling" not in result["breakdown"]


def test_v12_reduces_loss_budget():
    v11 = compute_composite_v11(
        decompose=True,
        **_high_side_channel_candidate(
            induction_intermediate_inv_auc=0.20,
            binding_intermediate_inv_auc=0.25,
        ),
    )
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            induction_intermediate_inv_auc=0.20,
            binding_intermediate_inv_auc=0.25,
        ),
    )

    bd = result["breakdown"]
    assert bd["_v12_loss_budget_after"] < bd["_v12_loss_budget_before"]
    assert bd["_v12_loss_budget_max"] == pytest.approx(175.0)
    assert bd["_v10_base_v8style_total"] == pytest.approx(
        v11["breakdown"]["_v10_base_v8style_total"]
        - (bd["_v12_loss_budget_before"] - bd["_v12_loss_budget_after"])
    )


def test_v12_caps_no_induction_loss_only_candidate_below_champion_range():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            blimp_accuracy=0.0,
            hellaswag_acc_validation=0.0,
            tinystories_score=0.0,
            cross_task_score=0.0,
            diagnostic_score=0.0,
            hierarchy_fitness=0.0,
        ),
    )

    assert result["composite_score"] <= 360.0


def test_v12_allows_induction_qualified_candidate_above_champion_range():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.08,
            binding_intermediate_inv_auc=0.25,
        ),
    )

    assert result["composite_score"] > 360.0
    assert "_v12_champion_eligibility_ceiling" not in result["breakdown"]
    assert result["breakdown"]["_v12_champion_induction_qualified"] is True


def _good_champion_tiny_model_protocol_kwargs(**overrides):
    kw = dict(
        champion_tiny_model_protocol_version="champion_tiny_model_score_v1",
        champion_checkpoint_available=True,
        champion_steps_to_floor=4_000,
        champion_baseline_steps_to_floor=10_000,
        champion_floor_ppl=148.41,
        champion_baseline_floor_ppl=221.41,
        champion_floor_loss_std=0.015,
        champion_baseline_floor_loss_std=0.030,
        induction_validation_auc=0.94,
        induction_validation_gap_accuracy_cv=0.10,
        induction_validation_protocol_version="induction_validation_full_counterfactual_2k",
        binding_intermediate_auc=0.90,
        robustness_long_ctx_combined_score=0.80,
        champion_baseline_long_ctx_combined_score=0.80,
        ar_validation_held_pair_acc=0.82,
        ar_validation_held_class_acc=0.76,
        ar_validation_steps_to_floor=5_000,
        champion_baseline_ar_validation_steps_to_floor=10_000,
        final_loss=5.0,
    )
    kw.update(overrides)
    return kw


def _ce5_high_side_channel_overrides():
    return dict(
        ppl_screening=148.41,
        ppl_investigation=148.41,
        ppl_validation=148.41,
        ppl_at_100=180.0,
        ppl_at_500=150.0,
        ppl_at_1000=148.41,
        induction_intermediate_inv_auc=0.0,
        binding_intermediate_inv_auc=0.0,
        blimp_accuracy=0.0,
        hellaswag_acc_screening=0.0,
        hellaswag_acc_investigation=0.0,
        hellaswag_acc_validation=0.0,
        tinystories_score=1.0,
        cross_task_score=1.0,
        diagnostic_score=1.0,
        hierarchy_fitness=1.0,
        fp_jacobian_erf_density=10.0,
        fp_id_collapse_rate=-1.0,
        fp_jacobian_erf_decay_slope=-1.0,
        fp_logit_margin_velocity=1.0,
        fp_jacobian_erf_variance=10_000.0,
        fp_icld_velocity=-1.0,
    )


def test_champion_tiny_model_score_v1_accepts_ce_around_five_with_good_evidence():
    result = compute_champion_tiny_model_score_v1(
        **_good_champion_tiny_model_protocol_kwargs(final_loss=5.04)
    )

    assert result["hard_failure_reason"] is None
    assert result["total"] > 30.0
    assert result["floor_quality"] > 0.0
    assert result["induction_validation"] > 9.0


def test_v12_champion_tiny_model_protocol_replaces_final_loss_ceiling():
    old_gate = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(**_ce5_high_side_channel_overrides()),
    )
    redesigned = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            **_ce5_high_side_channel_overrides(),
            **_good_champion_tiny_model_protocol_kwargs(),
        ),
    )

    assert old_gate["composite_score"] == pytest.approx(360.0)
    assert redesigned["composite_score"] > 360.0
    assert redesigned["breakdown"]["champion_hard_failure_reason"] is None
    assert redesigned["breakdown"]["champion_tiny_model_score"] > 30.0
    assert "_v12_champion_eligibility_ceiling" not in redesigned["breakdown"]


def test_v12_champion_tiny_model_protocol_blocks_missing_required_metrics():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            **_ce5_high_side_channel_overrides(),
            **_good_champion_tiny_model_protocol_kwargs(
                ar_validation_held_pair_acc=None,
            ),
        ),
    )

    assert result["composite_score"] <= 360.0
    assert result["breakdown"]["champion_hard_failure_reason"].startswith(
        "missing_required_champion_metrics:"
    )
    assert result["breakdown"]["_champion_tiny_model_hard_failure_gate"] is True


def test_champion_tiny_model_score_v1_allows_missing_ar_validation_speed_as_zero():
    result = compute_champion_tiny_model_score_v1(
        **_good_champion_tiny_model_protocol_kwargs(
            ar_validation_steps_to_floor=None,
        )
    )

    assert result["hard_failure_reason"] is None
    assert result["ar_validation"] == pytest.approx(
        6.0 * 0.82 + 2.0 * 0.76,
    )


def test_champion_tiny_model_score_v1_blocks_corrupt_required_metric():
    result = compute_champion_tiny_model_score_v1(
        **_good_champion_tiny_model_protocol_kwargs(
            induction_validation_auc=float("nan")
        )
    )

    assert result["total"] == 0.0
    assert result["hard_failure_reason"].startswith(
        "corrupt_required_champion_metrics:"
    )


def test_champion_tiny_model_score_v1_blocks_legacy_induction_validation_protocol():
    result = compute_champion_tiny_model_score_v1(
        **_good_champion_tiny_model_protocol_kwargs(
            induction_validation_protocol_version="induction_validation_5k"
        )
    )

    assert result["total"] == 0.0
    assert (
        result["hard_failure_reason"]
        == "corrupt_required_champion_metrics:induction_validation_protocol_version"
    )


def test_v12_mamba_exception_requires_bpe_loss_and_two_non_loss_sequence_signals():
    rejected = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
        ),
    )
    allowed = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
            long_ctx_multi_hop_score=0.35,
        ),
    )

    assert rejected["composite_score"] <= 360.0
    assert rejected["breakdown"]["_v12_champion_exception_allowed"] is False
    assert allowed["breakdown"]["_v12_champion_exception_allowed"] is True


def test_v12_caps_current_leader_like_non_inducer():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            ppl_screening=65.91,
            ppl_investigation=65.91,
            ppl_validation=65.91,
            ppl_at_100=70.0,
            ppl_at_500=66.0,
            ppl_at_1000=65.91,
            screening_lr=0.593741962632699,
            induction_screening_auc=0.002,
            induction_intermediate_inv_auc=0.001,
            binding_screening_auc=0.3449,
            binding_intermediate_inv_auc=0.4406,
            ar_legacy_auc=0.003,
            long_ctx_score=0.0004,
            blimp_accuracy=0.5248,
            hellaswag_acc_screening=0.225,
            hellaswag_acc_investigation=0.225,
            hellaswag_acc_validation=0.225,
            tinystories_score=0.6524,
            cross_task_score=0.5393,
            diagnostic_score=0.005178614450192737,
        ),
    )

    assert result["composite_score"] == pytest.approx(360.0)
    assert result["breakdown"]["_v12_champion_eligibility_ceiling"] == 360.0
    assert result["breakdown"]["_v12_champion_induction_qualified"] is False
    assert result["breakdown"]["_v12_champion_binding_qualified"] is True
    assert result["breakdown"]["_v12_champion_exception_allowed"] is False


def test_v12_ssm_exception_rejects_ar_alone():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            ar_legacy_auc=0.95,
            long_ctx_score=1.0,
        ),
    )

    assert result["composite_score"] <= 360.0
    assert result["breakdown"]["_v12_champion_exception_allowed"] is False
    assert result["breakdown"]["_v12_champion_sequence_signal_count"] == 0


def test_v12_ssm_exception_rejects_non_bpe_loss():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
            long_ctx_multi_hop_score=0.35,
            blimp_accuracy=0.7,
            hellaswag_acc_validation=0.4,
            screening_wikitext_metric_version="screening_wikitext_v1",
        ),
    )

    assert result["composite_score"] <= 360.0
    assert result["breakdown"].get("_v12_champion_exception_allowed") is not True


@pytest.mark.parametrize(
    "metadata",
    [
        {"architecture_family": "mamba_ssm"},
        {"graph_ops": ["rmsnorm", "selective_scan", "linear_proj"]},
        {"op_names": "rmsnorm selective_scan linear_proj"},
        {"non_attention_model": True},
    ],
)
def test_v12_ssm_exception_detects_non_attention_metadata(metadata):
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            **metadata,
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
            long_ctx_multi_hop_score=0.35,
        ),
    )

    assert result["breakdown"]["_v12_champion_exception_allowed"] is True


def test_v12_ssm_exception_accepts_downstream_language_signals():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_intermediate_inv_auc=0.0,
            binding_intermediate_inv_auc=0.0,
            blimp_accuracy=0.58,
            hellaswag_acc_validation=0.28,
        ),
    )

    assert result["breakdown"]["_v12_champion_exception_allowed"] is True
    assert result["breakdown"]["_v12_champion_sequence_signal_count"] >= 2


def test_hard_probe_floors_score_full_ar_hellaswag_and_blimp():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            ar_legacy_auc=0.95,
            ar_gate_score=None,
            hellaswag_acc_investigation=0.50,
            hellaswag_acc_validation=0.50,
            blimp_accuracy=0.90,
        ),
    )

    bd = result["breakdown"]
    assert bd["cap_legacy_ar"] > 9.0
    assert bd["cap_ar"] == 0.0
    assert bd["hellaswag"] > 9.0
    assert bd["blimp"] > 9.0


def _validation_ar_candidate(**overrides):
    kw = _high_side_channel_candidate(
        tier="validation",
        ppl_screening=90.0,
        ppl_investigation=90.0,
        ppl_validation=90.0,
        ppl_at_100=110.0,
        ppl_at_500=92.0,
        ppl_at_1000=90.0,
        induction_intermediate_inv_auc=0.0,
        binding_intermediate_inv_auc=0.0,
        blimp_accuracy=0.0,
        hellaswag_acc_screening=0.0,
        hellaswag_acc_investigation=0.0,
        hellaswag_acc_validation=0.0,
        tinystories_score=0.45,
        cross_task_score=0.25,
        diagnostic_score=0.003,
        hierarchy_fitness=0.75,
        ar_gate_score=0.80,
        ar_validation_metric_version="ar_validation_v2_easy25",
        ar_validation_rank_score=5.5,
        ar_validation_held_pair_acc=0.62,
        ar_validation_held_class_acc=0.54,
    )
    kw.update(overrides)
    return kw


def test_ar_gate_saturation_does_not_dominate_validation_rank_ordering():
    saturated_nano_weak_small = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(
            ar_gate_score=1.0,
            ar_validation_rank_score=2.5,
            ar_validation_held_pair_acc=0.25,
            ar_validation_held_class_acc=0.25,
        ),
    )
    less_saturated_nano_strong_small = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(
            ar_gate_score=0.75,
            ar_validation_rank_score=8.0,
            ar_validation_held_pair_acc=0.82,
            ar_validation_held_class_acc=0.74,
        ),
    )

    assert (
        less_saturated_nano_strong_small["composite_score"]
        > saturated_nano_weak_small["composite_score"]
    )
    assert (
        less_saturated_nano_strong_small["breakdown"]["cap_ar_validation_validation"]
        > saturated_nano_weak_small["breakdown"]["cap_ar_validation_validation"]
    )
    assert (
        saturated_nano_weak_small["breakdown"]["cap_ar"]
        - less_saturated_nano_strong_small["breakdown"]["cap_ar"]
    ) < 10.0


def test_ar_validation_separates_otherwise_similar_validation_candidates():
    weak = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(ar_validation_rank_score=2.0),
    )
    strong = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(ar_validation_rank_score=8.0),
    )

    assert strong["composite_score"] > weak["composite_score"]
    assert (
        strong["breakdown"]["cap_ar_validation_validation"]
        > weak["breakdown"]["cap_ar_validation_validation"]
    )


def test_strong_loss_weak_ar_validation_does_not_automatically_win():
    strong_loss_weak_small = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            ar_validation_rank_score=1.1,
            ar_validation_held_pair_acc=0.10,
            ar_validation_held_class_acc=0.10,
        ),
    )
    weaker_loss_strong_small = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(
            ppl_screening=135.0,
            ppl_investigation=135.0,
            ppl_validation=135.0,
            ar_validation_rank_score=9.0,
            ar_validation_held_pair_acc=0.90,
            ar_validation_held_class_acc=0.82,
        ),
    )

    assert (
        weaker_loss_strong_small["composite_score"]
        > strong_loss_weak_small["composite_score"]
    )


def test_missing_ar_validation_falls_back_without_penalty_or_error():
    result = compute_composite_v12(
        decompose=True,
        **_validation_ar_candidate(
            ar_validation_rank_score=None,
            champion_ar_validation_score=None,
            ar_validation_held_pair_acc=None,
            ar_validation_held_class_acc=None,
            ar_validation_learning_speed_score=None,
        ),
    )

    assert result["composite_score"] > 0.0
    assert result["breakdown"]["cap_ar_validation_validation"] == pytest.approx(0.0)
    assert result["breakdown"]["_ar_validation_validation_signal"] == pytest.approx(0.0)


def test_existing_scoring_api_accepts_rows_without_ar_validation_kwargs():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            tier="validation",
            ar_gate_score=0.75,
            induction_intermediate_inv_auc=0.08,
        ),
    )

    assert result["composite_score"] > 0.0
    assert "cap_ar_validation_validation" in result["breakdown"]
