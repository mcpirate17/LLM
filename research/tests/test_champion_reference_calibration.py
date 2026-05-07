from research.scientist.runner import ExperimentRunner
from research.scientist.runner.execution_champion_confirmation import (
    ChampionConfirmationEvaluator,
)
from research.tools.champion_reference_calibration import (
    calibration_fingerprint,
    calibration_floor_checkpoint_milestones,
    calibration_milestones,
)


def test_calibration_milestones_keep_requested_steps_and_final():
    assert calibration_milestones(25_000, [10_000, 20_000, 40_000]) == [
        10_000,
        20_000,
        25_000,
    ]


def test_floor_checkpoint_milestones_keep_regular_floor_artifacts():
    assert calibration_floor_checkpoint_milestones(3_500, 1_000) == [
        1_000,
        2_000,
        3_000,
    ]
    assert calibration_floor_checkpoint_milestones(3_500, 0) == []


def test_calibration_fingerprint_varies_by_layer_and_steps():
    base = calibration_fingerprint(
        "layer-fp",
        arch="gpt2",
        layers=4,
        steps=40_000,
        model_dim=256,
        seq_len=512,
        batch_size=8,
    )
    deeper = calibration_fingerprint(
        "layer-fp",
        arch="gpt2",
        layers=6,
        steps=40_000,
        model_dim=256,
        seq_len=512,
        batch_size=8,
    )
    longer = calibration_fingerprint(
        "layer-fp",
        arch="gpt2",
        layers=4,
        steps=80_000,
        model_dim=256,
        seq_len=512,
        batch_size=8,
    )

    assert base.startswith("gpt2_control_")
    assert base != deeper
    assert base != longer


def test_champion_snapshot_copies_investigation_v2_fields():
    runner = ExperimentRunner.__new__(ExperimentRunner)
    evaluator = ChampionConfirmationEvaluator(runner)
    metrics = {}
    evaluator._scale_up_apply_champion_snapshot(
        metrics,
        {
            "induction_v2_investigation_auc": 0.42,
            "induction_v2_investigation_max_gap_acc": 0.61,
            "induction_v2_investigation_gap_accuracies_json": '{"4": 0.7}',
            "induction_v2_investigation_steps_trained": 500,
            "induction_v2_investigation_status": "ok",
            "induction_v2_investigation_elapsed_ms": 123.4,
            "induction_v2_investigation_protocol_version": "induction_v2_test",
            "binding_v2_investigation_auc": 0.19,
            "binding_v2_investigation_max_distance_acc": 0.33,
            "binding_v2_investigation_distance_accuracies_json": '{"8": 0.4}',
            "binding_v2_investigation_train_steps": 2400,
            "binding_v2_investigation_status": "ok",
            "binding_v2_investigation_elapsed_ms": 456.7,
            "binding_v2_investigation_protocol_version": "binding_v2_test",
        },
    )

    assert metrics["induction_v2_investigation_auc"] == 0.42
    assert metrics["induction_v2_investigation_gap_accuracies_json"] == '{"4": 0.7}'
    assert metrics["binding_v2_investigation_auc"] == 0.19
    assert metrics["binding_v2_investigation_distance_accuracies_json"] == '{"8": 0.4}'
