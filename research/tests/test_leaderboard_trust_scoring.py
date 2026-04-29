import pytest

from research.scientist.leaderboard_scoring import compute_composite_v11, compute_composite_v12


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
        ar_auc=0.0041,
        induction_auc=0.0376,
        binding_auc=0.00766,
        induction_v2_inv_auc=0.755,
        binding_v2_inv_auc=0.0946,
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
            induction_v2_inv_auc=0.20,
            binding_v2_inv_auc=0.25,
        ),
    )

    assert result["composite_score"] > 360.0
    assert "_v11_trust_ceiling" not in result["breakdown"]


def test_v12_reduces_loss_budget():
    v11 = compute_composite_v11(
        decompose=True,
        **_high_side_channel_candidate(
            induction_v2_inv_auc=0.20,
            binding_v2_inv_auc=0.25,
        ),
    )
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            induction_v2_inv_auc=0.20,
            binding_v2_inv_auc=0.25,
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
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
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
            induction_v2_inv_auc=0.08,
            binding_v2_inv_auc=0.25,
        ),
    )

    assert result["composite_score"] > 360.0
    assert "_v12_champion_eligibility_ceiling" not in result["breakdown"]
    assert result["breakdown"]["_v12_champion_induction_qualified"] is True


def test_v12_mamba_exception_requires_bpe_loss_and_two_non_loss_sequence_signals():
    rejected = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
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
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
            long_ctx_multi_hop_score=0.35,
        ),
    )

    assert rejected["composite_score"] <= 360.0
    assert allowed["composite_score"] > 360.0
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
            induction_auc=0.002,
            induction_v2_inv_auc=0.001,
            binding_auc=0.3449,
            binding_v2_inv_auc=0.4406,
            ar_auc=0.003,
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
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
            ar_auc=0.95,
            long_ctx_score=1.0,
        ),
    )

    assert result["composite_score"] == pytest.approx(360.0)
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
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
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
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
            long_ctx_score=1.0,
            long_ctx_passkey_score=0.4,
            long_ctx_multi_hop_score=0.35,
        ),
    )

    assert result["composite_score"] > 360.0
    assert result["breakdown"]["_v12_champion_exception_allowed"] is True


def test_v12_ssm_exception_accepts_downstream_language_signals():
    result = compute_composite_v12(
        decompose=True,
        **_high_side_channel_candidate(
            architecture_family="mamba_ssm",
            ppl_screening=35.0,
            ppl_investigation=35.0,
            ppl_validation=35.0,
            induction_v2_inv_auc=0.0,
            binding_v2_inv_auc=0.0,
            blimp_accuracy=0.58,
            hellaswag_acc_validation=0.28,
        ),
    )

    assert result["composite_score"] > 360.0
    assert result["breakdown"]["_v12_champion_exception_allowed"] is True
    assert result["breakdown"]["_v12_champion_sequence_signal_count"] >= 2
