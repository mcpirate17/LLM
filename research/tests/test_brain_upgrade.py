import pytest
import json
from research.scientist.notebook import LabNotebook
from research.scientist.analytics import ExperimentAnalytics

pytestmark = pytest.mark.unit


def test_pareto_logic(tmp_path):
    db_path = str(tmp_path / "test_brain.db")
    nb = LabNotebook(db_path)

    # exp
    exp_id = nb.start_experiment("test", {}, "test")

    # 1. High Acc, High Params (dominated)
    # 2. High Acc, Low Params (Pareto)
    # 3. Low Acc, Low Params (Pareto)

    # Accurate but huge
    r1 = nb.record_program_result(
        exp_id,
        "fp1",
        "{}",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.2,
        param_count=1000,
        trust_label="test_fixture",
    )
    # Accurate and small
    r2 = nb.record_program_result(
        exp_id,
        "fp2",
        "{}",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.2,
        param_count=100,
        trust_label="test_fixture",
    )
    # Bad but tiny
    r3 = nb.record_program_result(
        exp_id,
        "fp3",
        "{}",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.8,
        param_count=10,
        trust_label="test_fixture",
    )

    nb.flush_writes()

    analytics = ExperimentAnalytics(nb)
    pareto_ids = analytics.pareto_optimal_programs()

    # r1 is dominated by r2 (same acc, more params)
    assert r1 not in pareto_ids
    assert r2 in pareto_ids
    assert r3 in pareto_ids


def test_instability_attribution(tmp_path):
    db_path = str(tmp_path / "test_instability.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("test", {}, "test")

    # Record results with high spectral norm for architectures containing 'softmax_attention'
    # softmax_attention is in 'mixing' category
    for i in range(10):
        # We need a proper graph JSON that _extract_ops_fast can parse
        g = {"nodes": [{"op_name": "softmax_attention", "input_ids": []}]}
        nb.record_program_result(
            exp_id,
            f"fp_{i}",
            json.dumps(g),
            stage0_passed=1,
            stage1_passed=1,
            fp_jacobian_spectral_norm=100.0,
            trust_label="test_fixture",
        )

    nb.flush_writes()

    analytics = ExperimentAnalytics(nb)
    multipliers = analytics.instability_attribution()

    # 'mixing' should have a penalty (multiplier < 1.0)
    assert "mixing" in multipliers
    assert multipliers.get("mixing", 1.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
