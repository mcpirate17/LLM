from research.scientist.runner import ExperimentRunner
from research.scientist.notebook import LabNotebook
from research.scientist.runner.execution_champion_confirmation import (
    ChampionConfirmationEvaluator,
)
from research.tools.champion_reference_calibration import (
    calibration_fingerprint,
    calibration_floor_checkpoint_milestones,
    calibration_milestones,
    resolve_reference_parent_fingerprint,
)
from research.tools.backfill_champion_reference_tests import (
    _existing_champion_probe_fields,
    _run_champion_probes_if_requested,
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


def test_reference_parent_fingerprint_prefers_registered_parent(tmp_path):
    nb = LabNotebook(tmp_path / "lab.db")
    exp = nb.start_experiment("reference", {"arch": "mamba"}, "seed reference")
    nb.record_program_result(
        result_id="ref-mamba",
        experiment_id=exp,
        graph_fingerprint="canonical-mamba-fp",
        graph_json='{"nodes": []}',
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.5,
        model_source="reference",
        trust_label="test_fixture",
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id="ref-mamba",
        model_source="reference",
        architecture_desc="Mamba reference",
        is_reference=True,
        reference_name="Mamba",
        tags="reference,mamba,selective_state_space",
    )

    assert (
        resolve_reference_parent_fingerprint(
            nb,
            arch="mamba",
            reference_name="Mamba",
            layer_fingerprint="fresh-build-fp",
        )
        == "canonical-mamba-fp"
    )
    nb.close()


def test_champion_snapshot_copies_investigation_v2_fields():
    runner = ExperimentRunner.__new__(ExperimentRunner)
    evaluator = ChampionConfirmationEvaluator(runner)
    metrics = {}
    evaluator._scale_up_apply_champion_snapshot(
        metrics,
        {
            "induction_intermediate_auc": 0.42,
            "induction_intermediate_max_gap_acc": 0.61,
            "induction_intermediate_gap_accuracies_json": '{"4": 0.7}',
            "induction_intermediate_steps_trained": 500,
            "induction_intermediate_status": "ok",
            "induction_intermediate_elapsed_ms": 123.4,
            "induction_intermediate_protocol_version": "induction_intermediate_test",
            "binding_intermediate_auc": 0.19,
            "binding_intermediate_max_distance_acc": 0.33,
            "binding_intermediate_distance_accuracies_json": '{"8": 0.4}',
            "binding_intermediate_train_steps": 2400,
            "binding_intermediate_status": "ok",
            "binding_intermediate_elapsed_ms": 456.7,
            "binding_intermediate_protocol_version": "binding_intermediate_test",
        },
    )

    assert metrics["induction_intermediate_auc"] == 0.42
    assert metrics["induction_intermediate_gap_accuracies_json"] == '{"4": 0.7}'
    assert metrics["binding_intermediate_auc"] == 0.19
    assert metrics["binding_intermediate_distance_accuracies_json"] == '{"8": 0.4}'


def test_champion_reference_backfill_does_not_promote_v2_as_v3():
    fields = _existing_champion_probe_fields(
        {
            "induction_intermediate_auc": 0.94,
            "induction_intermediate_steps_trained": 2000,
            "induction_intermediate_gap_accuracies_json": '{"4": 0.9}',
        }
    )

    assert fields["induction_validation_status"] == "missing_not_run"
    assert "induction_validation_auc" not in fields


def test_champion_reference_backfill_rejects_legacy_v3_protocol():
    fields = _existing_champion_probe_fields(
        {
            "induction_validation_auc": 0.94,
            "induction_validation_protocol_version": "induction_validation_5k",
        }
    )

    assert fields["induction_validation_auc"] is None
    assert (
        fields["induction_validation_status"]
        == "invalid_protocol:induction_validation_5k"
    )


def test_champion_reference_backfill_skips_cpu_probe_without_escape_hatch(tmp_path):
    metrics, artifact = _run_champion_probes_if_requested(
        {
            "result_id": "gpt2cal",
            "experiment_id": "exp",
            "graph_json": "{}",
        },
        target={"layers": 4},
        checkpoint_root=tmp_path,
        device="cpu",
        induction_steps=2000,
        force=True,
        run_probe=True,
        allow_cpu=False,
    )

    assert artifact is None
    assert metrics["induction_validation_status"] == "missing_accelerator"
    assert metrics["binding_intermediate_status"] == "missing_accelerator"
