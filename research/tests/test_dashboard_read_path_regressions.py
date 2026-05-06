from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time

import pytest

from research.scientist.api import create_app
from research.scientist.api_routes._strategy_preflight import (
    build_start_mode_eligibility,
)
from research.scientist.notebook import LabNotebook

pytestmark = [pytest.mark.api]


_S1_METRICS = {
    "wikitext_perplexity": 570.0,
    "hellaswag_acc": 0.2,
    "blimp_overall_accuracy": 0.51,
    "induction_auc": 0.01,
    "binding_auc": 0.0,
    "binding_composite": 0.1,
    "ar_auc": 0.01,
}


def _seed_minimal_dashboard_db(tmp_path):
    db_path = tmp_path / "dashboard_regressions.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="screening",
        config={"source": "test"},
        hypothesis="dashboard read-path regression seed",
        require_preregistration=False,
    )
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-minimal",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        novelty_score=0.0,
    )
    nb.complete_experiment(
        exp_id,
        {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 0},
        "seeded for dashboard regression coverage",
        "curious",
    )
    nb.close()
    return str(db_path)


def test_read_only_flush_writes_is_effectively_free(tmp_path):
    db_path = _seed_minimal_dashboard_db(tmp_path)
    nb = LabNotebook(db_path, read_only=True, use_native=False)
    t0 = time.perf_counter()
    for _ in range(100):
        nb.flush_writes()
    elapsed = time.perf_counter() - t0
    nb.close()
    assert elapsed < 0.25, f"read-only flush_writes took too long: {elapsed:.3f}s"


def test_dashboard_read_path_does_not_backfill_graph_features(tmp_path, monkeypatch):
    db_path = _seed_minimal_dashboard_db(tmp_path)

    def _forbid_backfill(self, *args, **kwargs):
        raise AssertionError("dashboard GET should not backfill graph features")

    monkeypatch.setattr(
        LabNotebook,
        "_ensure_graph_features",
        _forbid_backfill,
        raising=True,
    )
    app = create_app(notebook_path=db_path)
    client = app.test_client()
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["is_running"] is False


def test_program_detail_shows_requested_same_fingerprint_row(tmp_path):
    db_path = tmp_path / "program_detail_same_fingerprint.db"
    nb = LabNotebook(str(db_path))
    parent_exp = nb.start_experiment(
        experiment_type="synthesis",
        config={"source": "test"},
        hypothesis="parent fingerprint row",
        require_preregistration=False,
    )
    parent_id = nb.record_program_result(
        experiment_id=parent_exp,
        graph_fingerprint="fp-detail-parent",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.44,
        novelty_score=0.7,
        **_S1_METRICS,
    )
    nb.upsert_leaderboard(
        result_id=parent_id,
        model_source="graph_synthesis",
        architecture_desc="fp-detail-parent",
        screening_loss_ratio=0.44,
        screening_novelty=0.7,
        screening_passed=True,
        tier="screening",
        result_cohort="search",
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
    )
    child_exp = nb.start_experiment(
        experiment_type="investigation",
        config={"source": "test"},
        hypothesis="same fingerprint child",
        require_preregistration=False,
    )
    child_id = nb.record_program_result(
        experiment_id=child_exp,
        graph_fingerprint="fp-detail-parent",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.33,
        novelty_score=0.7,
        intentional_rerun_reason="exact_graph_replay",
        induction_v2_investigation_auc=0.547,
        induction_v2_investigation_status="ok",
        binding_v2_investigation_auc=0.1224,
        binding_v2_investigation_status="ok",
        **_S1_METRICS,
    )
    nb.flush_writes()
    nb.close()

    client = create_app(notebook_path=str(db_path)).test_client()
    response = client.get(f"/api/programs/{parent_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_id"] == parent_id
    assert payload["display_result_id"] == parent_id
    assert payload["fingerprint_parent_result_id"] == parent_id
    assert payload["canonical_result_id"] == parent_id
    assert payload["display_result_cohort"] == "search"
    assert payload["display_trust_label"] == "candidate_grade"
    assert payload["display_experiment_type"] == "investigation"
    assert payload["fingerprint_metric_source_result_id"] == child_id
    assert payload["induction_v2_investigation_auc"] == pytest.approx(0.547)
    assert payload["binding_v2_investigation_auc"] == pytest.approx(0.1224)
    assert not payload["superseded_requested_result"]

    child_response = client.get(f"/api/programs/{child_id}")
    assert child_response.status_code == 200
    child_payload = child_response.get_json()
    assert child_payload["result_id"] == child_id
    assert child_payload["display_result_id"] == child_id
    assert child_payload["fingerprint_parent_result_id"] == parent_id


def test_dashboard_recent_experiments_uses_program_result_metrics_when_summary_missing(
    tmp_path,
):
    db_path = tmp_path / "dashboard_recent_metrics.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="exact_graph_replay",
        config={"source": "test"},
        hypothesis="exact replay summary regression",
        require_preregistration=False,
    )
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-replay-summary",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.559,
        novelty_score=0.217,
        wikitext_perplexity=590.0,
        hellaswag_acc=0.21,
        blimp_overall_accuracy=0.53,
        induction_auc=0.04,
        binding_auc=0.03,
        binding_composite=0.18,
        ar_auc=0.02,
        model_source="exact_graph_replay",
    )
    nb.complete_experiment(
        exp_id,
        {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 1},
        "seeded exact replay",
        "focused",
    )
    nb.close()

    app = create_app(notebook_path=str(db_path))
    response = app.test_client().get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    row = next(
        exp for exp in payload["recent_experiments"] if exp["experiment_id"] == exp_id
    )
    assert row["best_loss_ratio"] == pytest.approx(0.559)
    assert row["best_novelty_score"] == pytest.approx(0.217)
    assert row["n_program_results"] == 1


def test_program_detail_marks_backfill_with_queued_candidate_confirmation(tmp_path):
    db_path = tmp_path / "dashboard_confirmation_status.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="backfill",
        config={"source": "test"},
        hypothesis="backfill confirmation status regression",
        require_preregistration=False,
    )
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-confirm-pending",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.553,
        novelty_score=0.112,
        **_S1_METRICS,
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
    )
    nb.enqueue_followup_task(
        stage="replay",
        result_ids=[result_id],
        hypothesis="confirm candidate",
        config={"stage1_steps": 750},
        evidence_pack={"stage": "screening", "n_steps": 750},
        source_context="program_detail_rerun",
        bypass_dedup=True,
    )
    nb.close()

    app = create_app(notebook_path=str(db_path))
    response = app.test_client().get(f"/api/programs/{result_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["candidate_confirmation_status"]["status"] == "queued"
    assert payload["display_result_cohort"] == "confirmation_queued"
    assert payload["display_trust_label"] == "candidate confirmation queued"


def test_validation_eligibility_rejects_unconfirmed_backfill(tmp_path):
    db_path = tmp_path / "dashboard_validation_backfill_guard.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="backfill",
        config={"source": "test"},
        hypothesis="validation backfill guard",
        require_preregistration=False,
    )
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-validation-backfill",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.553,
        novelty_score=0.112,
        **_S1_METRICS,
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
    )
    nb.upsert_leaderboard(
        result_id=result_id,
        model_source="graph_synthesis",
        screening_loss_ratio=0.553,
        investigation_loss_ratio=0.50,
        investigation_passed=True,
        tier="validation",
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
        graph_fingerprint="fp-validation-backfill",
    )

    payload = build_start_mode_eligibility(nb, "validation", [result_id])
    nb.close()

    assert payload["eligible_result_ids"] == []
    assert payload["ineligible"][0]["reason"] == "candidate_confirmation_required"


def test_validation_eligibility_resolves_backfill_to_confirmed_candidate(tmp_path):
    db_path = tmp_path / "dashboard_validation_backfill_resolution.db"
    nb = LabNotebook(str(db_path))
    backfill_exp = nb.start_experiment(
        experiment_type="backfill",
        config={"source": "test"},
        hypothesis="validation backfill source",
        require_preregistration=False,
    )
    backfill_id = nb.record_program_result(
        experiment_id=backfill_exp,
        graph_fingerprint="fp-validation-confirmed",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.553,
        novelty_score=0.112,
        **_S1_METRICS,
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
    )
    replay_exp = nb.start_experiment(
        experiment_type="exact_graph_replay",
        config={"source": "test", "candidate_confirmation": True},
        hypothesis="candidate confirmation",
        require_preregistration=False,
    )
    confirmed_id = nb.record_program_result(
        experiment_id=replay_exp,
        graph_fingerprint="fp-validation-confirmed",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.50,
        novelty_score=0.2,
        **_S1_METRICS,
        model_source="exact_graph_replay",
        source_result_id=backfill_id,
        intentional_rerun_reason="exact_graph_replay_independent_sample",
        result_cohort="search",
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
        evaluation_protocol_version="candidate_grade_v1",
    )
    nb.conn.execute(
        """
        UPDATE leaderboard
        SET result_id = ?,
            model_source = 'exact_graph_replay',
            screening_loss_ratio = 0.50,
            investigation_loss_ratio = 0.45,
            investigation_passed = 1,
            tier = 'investigation',
            result_cohort = 'search',
            trust_label = 'candidate_grade',
            comparability_label = 'candidate_comparable',
            evaluation_protocol_version = 'candidate_grade_v1',
            graph_fingerprint = 'fp-validation-confirmed'
        WHERE graph_fingerprint = 'fp-validation-confirmed'
        """,
        (confirmed_id,),
    )

    payload = build_start_mode_eligibility(nb, "validation", [backfill_id])
    nb.close()

    assert payload["eligible_result_ids"] == [confirmed_id]
    assert payload["resolved_result_ids"] == {backfill_id: confirmed_id}


def test_validation_rerun_rejects_backfill_without_confirmation(tmp_path):
    db_path = tmp_path / "dashboard_validation_rerun_backfill_guard.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="backfill",
        config={"source": "test"},
        hypothesis="validation rerun backfill guard",
        require_preregistration=False,
    )
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-validation-rerun-backfill",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.553,
        novelty_score=0.112,
        **_S1_METRICS,
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
    )
    nb.close()

    app = create_app(notebook_path=str(db_path))
    response = app.test_client().post(
        f"/api/programs/{result_id}/queue-validation-rerun",
        json={"stage": "validation"},
    )

    assert response.status_code == 409
    assert response.get_json()["code"] == "candidate_confirmation_required"


def test_backfill_rerun_pending_and_drain_follow_requested_parent(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "dashboard_rerun_requested_parent.db"
    nb = LabNotebook(str(db_path))
    backfill_exp = nb.start_experiment(
        experiment_type="backfill",
        config={"source": "test"},
        hypothesis="backfill source",
        require_preregistration=False,
    )
    backfill_id = nb.record_program_result(
        experiment_id=backfill_exp,
        graph_fingerprint="fp-rerun-parent-resolution",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.553,
        novelty_score=0.112,
        **_S1_METRICS,
        result_cohort="backfill",
        trust_label="backfill_observation",
        comparability_label="reconstructed_init_variant",
    )
    replay_exp = nb.start_experiment(
        experiment_type="exact_graph_replay",
        config={"candidate_confirmation": True},
        hypothesis="candidate confirmation",
        require_preregistration=False,
    )
    confirmed_id = nb.record_program_result(
        experiment_id=replay_exp,
        graph_fingerprint="fp-rerun-parent-resolution",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.50,
        novelty_score=0.2,
        **_S1_METRICS,
        model_source="exact_graph_replay",
        source_result_id=backfill_id,
        intentional_rerun_reason="exact_graph_replay_independent_sample",
        result_cohort="search",
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
        evaluation_protocol_version="candidate_grade_v1",
    )
    nb.close()

    app = create_app(notebook_path=str(db_path))
    client = app.test_client()
    queue_resp = client.post(
        f"/api/programs/{backfill_id}/queue-validation-rerun",
        json={"stage": "investigation", "n_steps": 2500},
    )
    assert queue_resp.status_code == 200
    queued = queue_resp.get_json()
    assert queued["requested_result_id"] == backfill_id
    assert queued["result_id"] == confirmed_id
    task_id = queued["task_ids"][0]

    parent_pending = client.get(f"/api/programs/{backfill_id}/pending-reruns")
    assert parent_pending.status_code == 200
    parent_tasks = parent_pending.get_json()["tasks"]
    assert [task["task_id"] for task in parent_tasks] == [task_id]
    assert parent_tasks[0]["requested_result_id"] == backfill_id
    assert parent_tasks[0]["result_ids"] == [confirmed_id]

    class FakeRunner:
        is_running = False
        current_experiment_id = "fake-investigation"

        def _run_pending_replay(self, task_id=None):
            raise AssertionError("drain should select investigation")

        def _run_pending_validation(self, task_id=None):
            raise AssertionError("drain should select investigation")

        def _run_pending_investigation(self, task_id=None):
            assert task_id == queued["task_ids"][0]
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    UPDATE followup_tasks
                    SET status = 'running',
                        started_timestamp = ?
                    WHERE task_id = ?
                    """,
                    (time.time(), task_id),
                )
                conn.commit()
            finally:
                conn.close()

    from research.scientist.api_routes.programs_routes import validation_rerun

    monkeypatch.setattr(
        validation_rerun,
        "get_runner",
        lambda notebook_path: FakeRunner(),
    )
    drain_resp = client.post(
        "/api/runner/drain-pending-validation-rerun",
        json={"result_id": backfill_id},
    )
    assert drain_resp.status_code == 200
    drained = drain_resp.get_json()
    assert drained["status"] == "launched"
    assert drained["stage"] == "investigation"
    assert drained["task_ids"] == [task_id]


def test_experiment_analytics_import_does_not_load_torch():
    script = """
import sys
before = set(sys.modules)
from research.scientist.analytics import ExperimentAnalytics
after = set(sys.modules)
heavy = sorted(
    name for name in (after - before)
    if name == 'torch' or name.startswith('torch.')
)
print(ExperimentAnalytics.__name__)
print(len(heavy))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/home/tim/Projects/LLM",
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip().splitlines()
    assert stdout[0] == "ExperimentAnalytics"
    assert stdout[1] == "0", result.stdout
