import json
import os
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.runner._helpers_benchmark import (
    _record_investigation_result,
    _run_investigation_v2_probes,
)


def _tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "investigation_v2.db")


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
        "research.eval.induction_probe_v2_investigation",
        SimpleNamespace(
            run_induction_v2_investigation=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_probe_v2_investigation",
        SimpleNamespace(
            run_binding_v2_investigation=lambda model, device: binding_result
        ),
    )

    updates = _run_investigation_v2_probes(object(), "cpu")

    assert updates["induction_v2_investigation_auc"] == pytest.approx(0.12)
    assert updates["binding_v2_investigation_auc"] == pytest.approx(0.56)
    assert json.loads(updates["induction_v2_investigation_gap_accuracies_json"]) == {
        "4": 0.2,
        "8": 0.3,
    }
    assert json.loads(updates["binding_v2_investigation_distance_accuracies_json"]) == {
        "4": 0.7,
        "8": 0.8,
    }


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
        "research.eval.induction_probe_v2_investigation",
        SimpleNamespace(
            run_induction_v2_investigation=lambda model, device: induction_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_probe_v2_investigation",
        SimpleNamespace(
            run_binding_v2_investigation=lambda model, device: binding_result
        ),
    )

    updates = _run_investigation_v2_probes(object(), "cpu")

    assert updates["induction_v2_investigation_auc"] is None
    assert updates["induction_v2_investigation_max_gap_acc"] is None
    assert updates["binding_v2_investigation_auc"] is None
    assert updates["binding_v2_investigation_max_distance_acc"] is None
    assert updates["induction_v2_investigation_status"].startswith("train_failed")
    assert updates["binding_v2_investigation_status"].startswith("train_failed")


def test_record_investigation_result_persists_v2_to_source_and_rerun_row():
    db_path = _tmp_db()
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
        "induction_v2_investigation_auc": 0.021,
        "induction_v2_investigation_max_gap_acc": 0.12,
        "induction_v2_investigation_gap_accuracies_json": json.dumps({"4": 0.1}),
        "induction_v2_investigation_steps_trained": 500,
        "induction_v2_investigation_status": "ok",
        "induction_v2_investigation_elapsed_ms": 1000.0,
        "induction_v2_investigation_protocol_version": "induction-test",
        "binding_v2_investigation_auc": 0.077,
        "binding_v2_investigation_max_distance_acc": 0.25,
        "binding_v2_investigation_distance_accuracies_json": json.dumps({"4": 0.2}),
        "binding_v2_investigation_train_steps": 2400,
        "binding_v2_investigation_status": "ok",
        "binding_v2_investigation_elapsed_ms": 2000.0,
        "binding_v2_investigation_protocol_version": "binding-test",
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
        """SELECT induction_v2_investigation_auc, binding_v2_investigation_auc,
                  induction_v2_investigation_status, binding_v2_investigation_status
           FROM program_results WHERE result_id = ?""",
        (source_result_id,),
    ).fetchone()
    rerun_row = conn.execute(
        """SELECT result_id, induction_v2_investigation_auc, binding_v2_investigation_auc,
                  induction_v2_investigation_status, binding_v2_investigation_status
           FROM program_results
           WHERE experiment_id = ? AND result_id != ?
           ORDER BY timestamp DESC LIMIT 1""",
        (inv_exp, source_result_id),
    ).fetchone()
    conn.close()
    nb.close()

    assert source_row["induction_v2_investigation_auc"] == pytest.approx(0.021)
    assert source_row["binding_v2_investigation_auc"] == pytest.approx(0.077)
    assert source_row["induction_v2_investigation_status"] == "ok"
    assert source_row["binding_v2_investigation_status"] == "ok"
    assert rerun_row is not None
    assert rerun_row["induction_v2_investigation_auc"] == pytest.approx(0.021)
    assert rerun_row["binding_v2_investigation_auc"] == pytest.approx(0.077)


def test_record_investigation_result_persists_v9_trajectory_to_source_row():
    """Investigation tier overwrites earlier-phase v9 trajectory fields.

    Regression for the bug where ``_record_investigation_result`` dropped
    Gemini trajectory metrics on the floor — ``fp_metric_phase`` stayed
    ``init`` or ``screening_750`` even when investigation produced a fresh
    measurement. The fix routes them through ``v9_trajectory_fields()``.
    """
    db_path = _tmp_db()
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
           FROM program_results WHERE result_id = ?""",
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
