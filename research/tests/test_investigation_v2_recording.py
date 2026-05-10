import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.runner._helpers_benchmark import (
    _record_investigation_result,
    _run_investigation_v2_probes,
)


def _graph_json() -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"op_name": "input"},
                "1": {"op_name": "attention"},
            },
            "metadata": {"templates_used": ["attention_block"]},
        }
    )


def test_investigation_v2_probe_helper_maps_probe_outputs(monkeypatch):
    induction_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2, 8: 0.3},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction-test",
    )
    binding_result = SimpleNamespace(
        auc=0.56,
        max_distance_acc=0.78,
        distance_accuracies={4: 0.7, 8: 0.8},
        train_steps=2400,
        status="ok",
        elapsed_ms=456.0,
        protocol_version="binding-test",
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )

    updates = _run_investigation_v2_probes(object(), "cpu")

    assert updates["induction_intermediate_auc"] == pytest.approx(0.12)
    assert updates["binding_intermediate_auc"] == pytest.approx(0.56)
    assert json.loads(updates["induction_intermediate_gap_accuracies_json"]) == {
        "4": 0.2,
        "8": 0.3,
    }
    assert json.loads(updates["binding_intermediate_distance_accuracies_json"]) == {
        "4": 0.7,
        "8": 0.8,
    }


def test_investigation_v2_probe_helper_runs_intermediate_probes_only_when_flagged(
    monkeypatch,
):
    induction_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction-test",
    )
    binding_result = SimpleNamespace(
        auc=0.56,
        max_distance_acc=0.78,
        distance_accuracies={4: 0.7},
        train_steps=2400,
        status="ok",
        elapsed_ms=456.0,
        protocol_version="binding-test",
    )
    ar_result = SimpleNamespace(
        to_dict=lambda: {
            "ar_intermediate_metric_version": "ar-int-test",
            "ar_intermediate_diagnostic_score": 0.42,
            "ar_intermediate_held_pair_acc": 0.08,
            "ar_intermediate_held_pair_lift": 0.06,
            "ar_intermediate_auc_lift": 0.05,
            "ar_intermediate_status": "ok",
            "ar_intermediate_elapsed_ms": 10.0,
            "ar_intermediate_error": None,
        }
    )
    multislot_result = SimpleNamespace(
        to_dict=lambda: {
            "binding_multislot_metric_version": "multislot-test",
            "binding_multislot_diagnostic_score": 1.2,
            "binding_multislot_held_entity_slot_acc": 0.1,
            "binding_multislot_two_plus_slots_acc": 0.05,
            "binding_multislot_auc_lift": 0.07,
            "binding_multislot_status": "ok",
            "binding_multislot_elapsed_ms": 20.0,
            "binding_multislot_error": None,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.ar_intermediate_probe",
        SimpleNamespace(ar_intermediate_probe=lambda model, device: ar_result),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_multislot_probe",
        SimpleNamespace(binding_multislot_probe=lambda model, device: multislot_result),
    )

    default_updates = _run_investigation_v2_probes(object(), "cpu")
    flagged_updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        run_ar_intermediate=True,
        run_binding_multislot=True,
    )

    assert "ar_intermediate_diagnostic_score" not in default_updates
    assert "binding_multislot_diagnostic_score" not in default_updates
    assert flagged_updates["ar_intermediate_diagnostic_score"] == pytest.approx(0.42)
    assert flagged_updates["ar_intermediate_held_pair_acc"] == pytest.approx(0.08)
    assert flagged_updates["binding_multislot_diagnostic_score"] == pytest.approx(1.2)
    assert flagged_updates["binding_multislot_two_plus_slots_acc"] == pytest.approx(
        0.05
    )


def test_investigation_probe_helper_maps_ar_gate_when_enabled(monkeypatch):
    induction_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction-test",
    )
    binding_result = SimpleNamespace(
        auc=0.56,
        max_distance_acc=0.78,
        distance_accuracies={},
        train_steps=2400,
        status="ok",
        elapsed_ms=456.0,
        protocol_version="binding-test",
    )
    nano_result = SimpleNamespace(
        metric_version="ar_gate_test",
        in_dist_pair_acc=0.8,
        in_dist_class_acc=0.9,
        held_pair_acc=0.2,
        held_class_acc=0.5,
        status="ok",
        elapsed_ms=789.0,
        finetune_steps_done=400,
    )
    captured = {}

    def _ar_gate(**kwargs):
        captured.update(kwargs)
        return nano_result

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.ar_gate",
        SimpleNamespace(
            ARGateConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            ar_gate=_ar_gate,
        ),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        graph_json_str=_graph_json(),
        run_ar_gate=True,
    )

    assert captured["graph_json"] == _graph_json()
    assert captured["cfg"].from_s1 is False
    assert captured["cfg"].wikitext_warmup_steps == 2500
    assert updates["ar_gate_metric_version"] == "ar_gate_test"
    assert updates["ar_gate_score"] == pytest.approx(0.68)
    assert updates["ar_gate_in_dist_pair_acc"] == pytest.approx(0.8)
    assert updates["ar_gate_held_class_acc"] == pytest.approx(0.5)
    assert updates["ar_gate_status"] == "ok"
    assert updates["ar_gate_train_steps_done"] == 400


def test_investigation_probe_helper_keeps_failed_ar_gate_score_empty(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: SimpleNamespace(
                auc=0.12,
                max_gap_acc=0.34,
                gap_accuracies={},
                steps_trained=500,
                status="ok",
                elapsed_ms=123.0,
                protocol_version="induction-test",
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(
            run_binding_intermediate=lambda model, device: SimpleNamespace(
                auc=0.56,
                max_distance_acc=0.78,
                distance_accuracies={},
                train_steps=2400,
                status="ok",
                elapsed_ms=456.0,
                protocol_version="binding-test",
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.ar_gate",
        SimpleNamespace(
            ARGateConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            ar_gate=lambda **kwargs: SimpleNamespace(
                metric_version="ar_gate_test",
                in_dist_pair_acc=0.8,
                in_dist_class_acc=0.9,
                held_pair_acc=0.2,
                held_class_acc=0.5,
                status="compile_failed",
                elapsed_ms=789.0,
                finetune_steps_done=0,
            ),
        ),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        graph_json_str=_graph_json(),
        run_ar_gate=True,
    )

    assert updates["ar_gate_score"] is None
    assert updates["ar_gate_in_dist_pair_acc"] is None
    assert updates["ar_gate_held_class_acc"] is None
    assert updates["ar_gate_status"] == "compile_failed"
    assert updates["ar_gate_train_steps_done"] == 0


def test_investigation_v2_probe_failures_keep_status_without_zero_metric(monkeypatch):
    induction_result = SimpleNamespace(
        auc=0.0,
        max_gap_acc=0.0,
        gap_accuracies={},
        steps_trained=0,
        status="train_failed: optimizer device mismatch",
        elapsed_ms=12.0,
        protocol_version="induction-test",
    )
    binding_result = SimpleNamespace(
        auc=0.0,
        max_distance_acc=0.0,
        distance_accuracies={},
        train_steps=0,
        status="train_failed: optimizer device mismatch",
        elapsed_ms=34.0,
        protocol_version="binding-test",
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )

    updates = _run_investigation_v2_probes(object(), "cpu")

    assert updates["induction_intermediate_auc"] is None
    assert updates["induction_intermediate_max_gap_acc"] is None
    assert updates["binding_intermediate_auc"] is None
    assert updates["binding_intermediate_max_distance_acc"] is None
    assert updates["induction_intermediate_status"].startswith("train_failed")
    assert updates["binding_intermediate_status"].startswith("train_failed")


def test_record_investigation_result_persists_v2_to_source_and_rerun_row(tmp_path):
    db_path = str(tmp_path / "investigation_v2.db")
    nb = LabNotebook(db_path, use_native=False)
    source_exp = nb.start_experiment("synthesis", {}, "source")
    source_result_id = nb.record_program_result(
        experiment_id=source_exp,
        graph_fingerprint="fp_inv_v2",
        graph_json=_graph_json(),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.3,
        novelty_score=0.2,
        model_source="graph_synthesis",
        trust_label="test_fixture",
    )
    inv_exp = nb.start_experiment("investigation", {}, "investigation")

    benchmark_result = {
        "inv_wikitext_ppl": 4.2,
        "induction_intermediate_auc": 0.021,
        "induction_intermediate_max_gap_acc": 0.12,
        "induction_intermediate_gap_accuracies_json": json.dumps({"4": 0.1}),
        "induction_intermediate_steps_trained": 500,
        "induction_intermediate_status": "ok",
        "induction_intermediate_elapsed_ms": 1000.0,
        "induction_intermediate_protocol_version": "induction-test",
        "binding_intermediate_auc": 0.077,
        "binding_intermediate_max_distance_acc": 0.25,
        "binding_intermediate_distance_accuracies_json": json.dumps({"4": 0.2}),
        "binding_intermediate_train_steps": 2400,
        "binding_intermediate_status": "ok",
        "binding_intermediate_elapsed_ms": 2000.0,
        "binding_intermediate_protocol_version": "binding-test",
        "ar_gate_metric_version": "ar_gate_test",
        "ar_gate_in_dist_pair_acc": 0.8,
        "ar_gate_in_dist_class_acc": 0.9,
        "ar_gate_held_pair_acc": 0.2,
        "ar_gate_held_class_acc": 0.5,
        "ar_gate_score": 0.68,
        "ar_gate_status": "ok",
        "ar_gate_elapsed_ms": 3000.0,
        "ar_gate_train_steps_done": 400,
        "ar_intermediate_metric_version": "ar-int-test",
        "ar_intermediate_diagnostic_score": 0.42,
        "ar_intermediate_held_pair_acc": 0.08,
        "ar_intermediate_held_pair_lift": 0.06,
        "ar_intermediate_auc_lift": 0.05,
        "ar_intermediate_status": "ok",
        "ar_intermediate_elapsed_ms": 10.0,
        "binding_multislot_metric_version": "multislot-test",
        "binding_multislot_diagnostic_score": 1.2,
        "binding_multislot_held_entity_slot_acc": 0.1,
        "binding_multislot_two_plus_slots_acc": 0.05,
        "binding_multislot_auc_lift": 0.07,
        "binding_multislot_status": "ok",
        "binding_multislot_elapsed_ms": 20.0,
    }
    source = {
        "graph_fingerprint": "fp_inv_v2",
        "loss_ratio": 0.3,
        "novelty_score": 0.2,
    }

    _record_investigation_result(
        nb=nb,
        exp_id=inv_exp,
        source_result_id=source_result_id,
        source=source,
        model_source="graph_synthesis",
        graph_json_str=_graph_json(),
        arch_spec_json_str=None,
        n_passed=1,
        n_programs_tested=3,
        best_lr=0.22,
        best_tp_json=json.dumps({"recipe": "test"}),
        robustness=1.0 / 3.0,
        investigation_passed=True,
        benchmark_result=benchmark_result,
    )
    nb.flush_writes()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    source_row = conn.execute(
        """SELECT induction_intermediate_auc, binding_intermediate_auc,
                  induction_intermediate_status, binding_intermediate_status,
                  ar_gate_score, ar_gate_in_dist_pair_acc,
                  ar_gate_held_class_acc, ar_gate_status,
                  ar_gate_train_steps_done,
                  ar_intermediate_diagnostic_score,
                  ar_intermediate_held_pair_acc,
                  ar_intermediate_auc_lift,
                  ar_intermediate_status,
                  binding_multislot_diagnostic_score,
                  binding_multislot_held_entity_slot_acc,
                  binding_multislot_two_plus_slots_acc,
                  binding_multislot_auc_lift,
                  binding_multislot_status
           FROM program_results_compat WHERE result_id = ?""",
        (source_result_id,),
    ).fetchone()
    rerun_row = conn.execute(
        """SELECT result_id, induction_intermediate_auc, binding_intermediate_auc,
                  induction_intermediate_status, binding_intermediate_status,
                  ar_gate_score, ar_gate_in_dist_pair_acc,
                  ar_gate_held_class_acc, ar_gate_status,
                  ar_gate_train_steps_done,
                  ar_intermediate_diagnostic_score,
                  ar_intermediate_held_pair_acc,
                  ar_intermediate_auc_lift,
                  ar_intermediate_status,
                  binding_multislot_diagnostic_score,
                  binding_multislot_held_entity_slot_acc,
                  binding_multislot_two_plus_slots_acc,
                  binding_multislot_auc_lift,
                  binding_multislot_status
           FROM program_results_compat
           WHERE experiment_id = ? AND result_id != ?
           ORDER BY timestamp DESC LIMIT 1""",
        (inv_exp, source_result_id),
    ).fetchone()
    conn.close()
    nb.close()

    assert source_row["induction_intermediate_auc"] == pytest.approx(0.021)
    assert source_row["binding_intermediate_auc"] == pytest.approx(0.077)
    assert source_row["induction_intermediate_status"] == "ok"
    assert source_row["binding_intermediate_status"] == "ok"
    assert source_row["ar_gate_score"] == pytest.approx(0.68)
    assert source_row["ar_gate_in_dist_pair_acc"] == pytest.approx(0.8)
    assert source_row["ar_gate_held_class_acc"] == pytest.approx(0.5)
    assert source_row["ar_gate_status"] == "ok"
    assert source_row["ar_gate_train_steps_done"] == 400
    assert source_row["ar_intermediate_diagnostic_score"] == pytest.approx(0.42)
    assert source_row["ar_intermediate_held_pair_acc"] == pytest.approx(0.08)
    assert source_row["ar_intermediate_auc_lift"] == pytest.approx(0.05)
    assert source_row["ar_intermediate_status"] == "ok"
    assert source_row["binding_multislot_diagnostic_score"] == pytest.approx(1.2)
    assert source_row["binding_multislot_held_entity_slot_acc"] == pytest.approx(0.1)
    assert source_row["binding_multislot_two_plus_slots_acc"] == pytest.approx(0.05)
    assert source_row["binding_multislot_auc_lift"] == pytest.approx(0.07)
    assert source_row["binding_multislot_status"] == "ok"
    assert rerun_row is not None
    assert rerun_row["induction_intermediate_auc"] == pytest.approx(0.021)
    assert rerun_row["binding_intermediate_auc"] == pytest.approx(0.077)
    assert rerun_row["ar_gate_score"] == pytest.approx(0.68)
    assert rerun_row["ar_gate_in_dist_pair_acc"] == pytest.approx(0.8)
    assert rerun_row["ar_gate_held_class_acc"] == pytest.approx(0.5)
    assert rerun_row["ar_gate_status"] == "ok"
    assert rerun_row["ar_gate_train_steps_done"] == 400
    assert rerun_row["ar_intermediate_diagnostic_score"] == pytest.approx(0.42)
    assert rerun_row["binding_multislot_diagnostic_score"] == pytest.approx(1.2)
    assert rerun_row["binding_multislot_two_plus_slots_acc"] == pytest.approx(0.05)


def test_record_investigation_result_persists_v9_trajectory_to_source_row(tmp_path):
    """Investigation tier overwrites earlier-phase v9 trajectory fields.

    Regression for the bug where ``_record_investigation_result`` dropped
    Gemini trajectory metrics on the floor — ``fp_metric_phase`` stayed
    ``init`` or ``screening_750`` even when investigation produced a fresh
    measurement. The fix routes them through ``v9_trajectory_fields()``.
    """
    db_path = str(tmp_path / "investigation_v9.db")
    nb = LabNotebook(db_path, use_native=False)
    source_exp = nb.start_experiment("synthesis", {}, "source-v9")
    source_result_id = nb.record_program_result(
        experiment_id=source_exp,
        graph_fingerprint="fp_inv_v9",
        graph_json=_graph_json(),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.3,
        novelty_score=0.2,
        model_source="graph_synthesis",
        trust_label="test_fixture",
        fp_metric_phase="init",
        fp_jacobian_erf_density=0.1,
        fp_jacobian_erf_variance=10.0,
        fp_logit_margin_velocity=0.001,
    )
    inv_exp = nb.start_experiment("investigation", {}, "investigation-v9")

    benchmark_result = {
        "inv_wikitext_ppl": 4.2,
        "fp_metric_phase": "investigation_full",
        "fp_jacobian_erf_density": 0.55,
        "fp_jacobian_erf_variance": 50000.0,
        "fp_jacobian_erf_status": "ok",
        "fp_jacobian_spectral_norm": 1.25,
        "fp_spec_norm_status": "ok",
        "fp_icld_velocity": -0.05,
        "fp_icld_status": "ok",
        "fp_logit_margin_velocity": 0.012,
        "fp_logit_margin_status": "ok",
    }
    source = {
        "graph_fingerprint": "fp_inv_v9",
        "loss_ratio": 0.3,
        "novelty_score": 0.2,
    }

    _record_investigation_result(
        nb=nb,
        exp_id=inv_exp,
        source_result_id=source_result_id,
        source=source,
        model_source="graph_synthesis",
        graph_json_str=_graph_json(),
        arch_spec_json_str=None,
        n_passed=1,
        n_programs_tested=3,
        best_lr=0.22,
        best_tp_json=json.dumps({"recipe": "test"}),
        robustness=1.0 / 3.0,
        investigation_passed=True,
        benchmark_result=benchmark_result,
    )
    nb.flush_writes()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    source_row = conn.execute(
        """SELECT fp_metric_phase, fp_jacobian_erf_density, fp_jacobian_erf_variance,
                  fp_jacobian_spectral_norm, fp_icld_velocity, fp_logit_margin_velocity
           FROM program_results_compat WHERE result_id = ?""",
        (source_result_id,),
    ).fetchone()
    conn.close()
    nb.close()

    assert source_row["fp_metric_phase"] == "investigation_full"
    assert source_row["fp_jacobian_erf_density"] == pytest.approx(0.55)
    assert source_row["fp_jacobian_erf_variance"] == pytest.approx(50000.0)
    assert source_row["fp_jacobian_spectral_norm"] == pytest.approx(1.25)
    assert source_row["fp_icld_velocity"] == pytest.approx(-0.05)
    assert source_row["fp_logit_margin_velocity"] == pytest.approx(0.012)
