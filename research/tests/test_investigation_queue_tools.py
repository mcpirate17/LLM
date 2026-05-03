import json
import os
import tempfile

from research.scientist.notebook import LabNotebook
from research.tools import queue_investigation_followups as qif
from research.tools import repair_investigation_v2_targets as repair_v2


def _tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "queue_tools.db")


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


def _record_result(nb: LabNotebook, result_id_hint: str | None = None) -> str:
    exp_id = nb.start_experiment("synthesis", {}, "queue tool test")
    return nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=result_id_hint or "fp_queue_tool",
        graph_json=_graph_json(),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.2,
        trust_label="test_fixture",
    )


def test_queue_followups_is_dry_run_by_default(monkeypatch):
    db_path = _tmp_db()
    nb = LabNotebook(db_path, use_native=False)
    result_id = _record_result(nb)
    nb.close()

    monkeypatch.setattr(
        qif,
        "build_queue",
        lambda *args, **kwargs: (
            [
                {
                    "rank": 1,
                    "rank_score": 0.7,
                    "result_id": result_id,
                    "missing_signals": ["induction_v2_investigation_auc"],
                }
            ],
            {"generated_at": "test"},
        ),
    )

    report = qif.queue_followups(db_path, limit=1, batch_size=1)
    nb = LabNotebook(db_path, use_native=False)
    tasks = nb.get_followup_tasks(stage="investigation", limit=10)
    nb.close()

    assert report["apply"] is False
    assert report["dry_run_task_count"] == 1
    assert report["queued_task_count"] == 0
    assert tasks == []


def test_queue_followups_apply_suppresses_active_duplicate_ids(monkeypatch):
    db_path = _tmp_db()
    nb = LabNotebook(db_path, use_native=False)
    first_id = _record_result(nb, "fp_queue_first")
    second_id = _record_result(nb, "fp_queue_second")
    nb.enqueue_followup_task(
        stage="investigation",
        result_ids=[first_id],
        hypothesis="existing",
        config={},
        priority_score=0.1,
    )
    nb.close()

    monkeypatch.setattr(
        qif,
        "build_queue",
        lambda *args, **kwargs: (
            [
                {
                    "rank": 1,
                    "rank_score": 0.9,
                    "result_id": first_id,
                    "missing_signals": ["induction_v2_investigation_auc"],
                },
                {
                    "rank": 2,
                    "rank_score": 0.8,
                    "result_id": second_id,
                    "missing_signals": ["binding_v2_investigation_auc"],
                },
            ],
            {"generated_at": "test"},
        ),
    )

    report = qif.queue_followups(db_path, limit=2, batch_size=1, apply=True)
    nb = LabNotebook(db_path, use_native=False)
    tasks = nb.get_followup_tasks(stage="investigation", limit=10)
    nb.close()

    queued_result_sets = [set(task["result_ids_json"]) for task in tasks]
    assert report["queued_task_count"] == 1
    assert report["suppressed_active"] == 1
    assert {second_id} in queued_result_sets


def test_v2_repair_targets_include_failed_probe_status_with_numeric_zero():
    db_path = _tmp_db()
    nb = LabNotebook(db_path, use_native=False)
    exp_id = nb.start_experiment("investigation", {}, "repair target test")
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp_failed_v2_status",
        graph_json=_graph_json(),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.2,
        induction_v2_investigation_auc=0.0,
        induction_v2_investigation_status="train_failed: optimizer device mismatch",
        binding_v2_investigation_auc=0.1,
        binding_v2_investigation_status="ok",
        trust_label="test_fixture",
    )
    nb.close()

    targets, summary = repair_v2.find_repair_targets(db_path, limit=10)

    assert summary["n_targets"] == 1
    assert targets[0]["result_id"] == result_id
    assert "induction_v2_investigation_status" in targets[0]["missing_signals"]
