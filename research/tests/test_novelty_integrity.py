import os
import tempfile

from research.eval.baseline import _BaselineTransformer
from research.eval.fingerprint import build_novelty_reference_version
from research.eval.novelty_calibration import (
    calibrate_baseline_transformer_novelty,
    novelty_stability_under_small_perturbations,
)
from research.scientist.notebook import LabNotebook
from research.tools.novelty_integrity_check import run_integrity_check


def test_reference_version_scheme_is_stable():
    v1 = build_novelty_reference_version("artifact", "v3", "abc123")
    v2 = build_novelty_reference_version("artifact", "v3", "abc123")
    assert v1 == v2
    assert v1.startswith("nv1:artifact:v3:abc123")


def test_novelty_calibration_table_round_trip():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "novelty_cal.db")
    nb = LabNotebook(db_path)
    try:
        calibration = calibrate_baseline_transformer_novelty(
            n_runs=2,
            seq_len=8,
            model_dim=16,
            vocab_size=128,
            device="cpu",
            seed=11,
        )
        cal_id = nb.record_novelty_calibration(
            reference_version=calibration["reference_version"],
            cka_source=calibration["cka_source"],
            cka_artifact_version=calibration["cka_artifact_version"],
            probe_protocol_hash=calibration["probe_protocol_hash"],
            n_runs=calibration["n_runs"],
            noise_floor_mean=calibration["noise_floor_mean"],
            noise_floor_std=calibration["noise_floor_std"],
            confidence_low=calibration["confidence_low"],
            confidence_high=calibration["confidence_high"],
            distribution=calibration["distribution"],
            metadata=calibration["metadata"],
        )
        assert cal_id
        stored = nb.get_latest_novelty_calibration(calibration["reference_version"])
        assert stored is not None
        assert stored["reference_version"] == calibration["reference_version"]
        assert stored["distribution_json"]["novelty_score"]
    finally:
        nb.close()


def test_novelty_stability_under_small_perturbations():
    model = _BaselineTransformer(vocab_size=128, d_model=16, n_layers=2)
    stats = novelty_stability_under_small_perturbations(
        model,
        seq_len=8,
        model_dim=16,
        vocab_size=128,
        device="cpu",
        perturbation_std=1e-5,
        n_trials=2,
        seed=7,
    )
    assert stats["max_abs_drift"] < 0.25


def test_integrity_check_rejects_unjustified_heuristic_promotion():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "novelty_integrity.db")
    nb = LabNotebook(db_path)
    try:
        exp = nb.start_experiment("synthesis", {"n_programs": 1}, "novelty integrity")
        rid = nb.record_program_result(
            experiment_id=exp,
            graph_fingerprint="fp_a",
            graph_json='{"nodes": {"0": {"id": 0, "op_name": "input", "input_ids": []}}}',
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            novelty_score=0.7,
            novelty_confidence=0.8,
            cka_source="heuristic_fallback",
            novelty_reference_version="nv1:heuristic_fallback:none:none",
            novelty_valid_for_promotion=1,
            novelty_validity_reason="heuristic_fallback_reference",
            novelty_requires_justification=0,
        )
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            screening_passed=True,
            tier="validation",
            novelty_confidence=0.8,
        )
        nb.flush_writes()
        report = run_integrity_check(nb)
        assert report["ok"] is False
        assert any("heuristic novelty" in f for f in report["failures"])
    finally:
        nb.close()
