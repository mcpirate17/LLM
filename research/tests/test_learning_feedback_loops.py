import json
import os
import tempfile
import time
from types import SimpleNamespace

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.synthesis._template_helpers import (
    _filter_slot_candidates,
    _select_from_candidates,
)

pytestmark = pytest.mark.unit


def _graph_json(op_name: str = "attention") -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": op_name, "input_ids": [0]},
            }
        }
    )


def _tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "learning_feedback.db")


def test_selection_family_trials_resolve_from_realized_outcomes():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "family learning")
    rid = nb.record_program_result(
        exp_id,
        graph_fingerprint="fp_family",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.40,
        baseline_loss_ratio=0.72,
        novelty_score=0.25,
        throughput_tok_s=180.0,
        flops_per_token=1.0,
        peak_memory_mb=220.0,
        stability_score=0.92,
    )
    nb.conn.execute(
        """INSERT INTO leaderboard
           (entry_id, result_id, timestamp, model_source, screening_loss_ratio,
            screening_novelty, screening_passed, composite_score, tier,
            trust_label, comparability_label)
           VALUES (?, ?, ?, 'graph_synthesis', ?, ?, 1, ?, 'screening', ?, ?)""",
        (
            "lb_family",
            rid,
            time.time(),
            0.40,
            0.25,
            84.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    nb.conn.commit()

    decision_id = nb.record_selection_decision(
        context="auto_investigate_screening",
        candidate_pool_summary={"candidate_count": 1},
        score_breakdown=[],
        policy={"name": "ucb"},
        reason="family trial",
        chosen_experiments=[{"result_id": rid}],
        experiment_id=exp_id,
        trigger=None,
    )
    nb.record_selection_family_trial(
        decision_id=decision_id,
        context="auto_investigate_screening",
        family="Attention",
        chosen_result_ids=[rid],
        source_experiment_id=exp_id,
    )
    nb.conn.execute(
        """UPDATE leaderboard
           SET investigation_passed = 1,
               investigation_loss_ratio = 0.22,
               investigation_robustness = 0.80
           WHERE result_id = ?""",
        (rid,),
    )
    nb.conn.commit()

    runner._resolve_pending_selection_family_trials(nb)

    stats = nb.get_selection_family_stats()
    assert "Attention" in stats
    assert stats["Attention"]["n_trials"] == 1
    assert stats["Attention"]["mean_reward"] == pytest.approx(0.894, abs=1e-3)
    nb.close()


def test_dynamic_slot_observability_priors_affect_runtime_motif_selection():
    class _FakeGraph:
        def __init__(self, metadata):
            self.metadata = metadata

    graph = _FakeGraph(
        {
            "_active_template": "demo",
            "_active_template_slot_counter": 0,
            "_slot_motif_denylist": {"demo.slot0": ("bad_motif",)},
            "_slot_motif_weight_multipliers": {"demo.slot0": {"good_motif": 2.5}},
        }
    )
    candidates = [
        SimpleNamespace(name="bad_motif", lift=1.0),
        SimpleNamespace(name="good_motif", lift=1.0),
    ]

    filtered = _filter_slot_candidates(graph, candidates)
    assert [item.name for item in filtered] == ["good_motif"]

    graph_no_deny = _FakeGraph(
        {
            "_active_template": "demo",
            "_active_template_slot_counter": 0,
            "_slot_motif_weight_multipliers": {"demo.slot0": {"good_motif": 2.5}},
        }
    )

    class _CaptureRng:
        def __init__(self):
            self.weights = None

        def choices(self, candidates, weights, k=1):
            self.weights = list(weights)
            return [candidates[self.weights.index(max(self.weights))]]

    rng = _CaptureRng()
    picked = _select_from_candidates(
        graph_no_deny,
        candidates,
        rng,
        {"bad_motif": 1.0, "good_motif": 1.0},
    )
    assert picked.name == "good_motif"
    assert rng.weights == pytest.approx([1.0, 2.5], abs=1e-6)


def test_adaptive_synthesis_regime_uses_realized_rewards_instead_of_mod8():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    for idx in range(3):
        exp_id = nb.start_experiment(
            "synthesis",
            {"adaptive_synthesis_regime": "efficiency", "n_programs": 20},
            f"efficiency-{idx}",
        )
        nb.complete_experiment(
            exp_id,
            {
                "total": 20,
                "stage0_passed": 20,
                "stage05_passed": 18,
                "stage1_passed": 6,
                "best_loss_ratio": 0.24,
                "best_novelty_score": 0.45,
            },
        )

    for idx in range(3):
        exp_id = nb.start_experiment(
            "synthesis",
            {"adaptive_synthesis_regime": "exotic", "n_programs": 20},
            f"exotic-{idx}",
        )
        nb.complete_experiment(
            exp_id,
            {
                "total": 20,
                "stage0_passed": 20,
                "stage05_passed": 12,
                "stage1_passed": 1,
                "best_loss_ratio": 0.82,
                "best_novelty_score": 0.18,
            },
        )

    regime, decision = runner._select_synthesis_regime(
        RunConfig(),
        n_experiments=9,
        nb=nb,
    )

    assert regime == "efficiency"
    assert decision["policy"] == "ucb_realized_reward"
    assert (
        decision["score_breakdown"]["efficiency"]["score"]
        > decision["score_breakdown"]["exotic"]["score"]
    )
    nb.close()


def test_adaptive_screening_threshold_refreshes_from_recent_labeled_outcomes():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    for idx in range(15):
        nb.conn.execute(
            """INSERT INTO leaderboard
               (entry_id, result_id, timestamp, model_source, composite_score, tier,
                investigation_passed, investigation_loss_ratio, investigation_robustness,
                trust_label, comparability_label)
               VALUES (?, NULL, ?, 'graph_synthesis', ?, 'screening', 1, 0.24, 0.78, ?, ?)""",
            (
                f"pos_{idx}",
                time.time() + idx,
                82.0 + idx * 0.4,
                "candidate_grade",
                "candidate_comparable",
            ),
        )
    for idx in range(15):
        nb.conn.execute(
            """INSERT INTO leaderboard
               (entry_id, result_id, timestamp, model_source, composite_score, tier,
                investigation_passed, investigation_loss_ratio, investigation_robustness,
                trust_label, comparability_label)
               VALUES (?, NULL, ?, 'graph_synthesis', ?, 'screening', 0, 0.92, 0.10, ?, ?)""",
            (
                f"neg_{idx}",
                time.time() + 100 + idx,
                56.0 + idx * 0.5,
                "candidate_grade",
                "candidate_comparable",
            ),
        )
    nb.conn.commit()

    threshold = runner._adaptive_screening_threshold(
        nb,
        RunConfig(
            adaptive_thresholds_enabled=True, screening_promotion_percentile=90.0
        ),
        floor=50.0,
    )

    assert threshold > 63.0
    assert threshold < 90.0
    nb.close()


def test_followup_tasks_persist_and_runner_claims_highest_priority():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    low_id = nb.enqueue_followup_task(
        stage="investigation",
        result_ids=["rid_low"],
        hypothesis="low priority",
        config=RunConfig(n_programs=3, stage1_steps=25).to_dict(),
        priority_score=0.15,
        priority_reasons={"policy": "expected_information_gain", "score": 0.15},
    )
    high_id = nb.enqueue_followup_task(
        stage="investigation",
        result_ids=["rid_high_a", "rid_high_b"],
        hypothesis="high priority",
        config=RunConfig(n_programs=7, stage1_steps=55).to_dict(),
        priority_score=0.91,
        priority_reasons={"policy": "expected_information_gain", "score": 0.91},
    )
    dup_high_id = nb.enqueue_followup_task(
        stage="investigation",
        result_ids=["rid_high_b", "rid_high_a"],
        hypothesis="high priority refreshed",
        config=RunConfig(n_programs=9, stage1_steps=75).to_dict(),
        priority_score=0.95,
        priority_reasons={"policy": "expected_information_gain", "score": 0.95},
    )

    assert dup_high_id == high_id

    captured = {}

    def _fake_start_investigation(result_ids, config, hypothesis):
        captured["result_ids"] = list(result_ids)
        captured["config"] = config
        captured["hypothesis"] = hypothesis

    runner.start_investigation = _fake_start_investigation
    runner._run_pending_investigation()

    assert captured["result_ids"] == ["rid_high_a", "rid_high_b"]
    assert captured["config"].n_programs == 9
    assert captured["config"].stage1_steps == 75
    assert captured["hypothesis"] == "high priority refreshed"

    nb.close()
    nb = LabNotebook(db_path)
    tasks = nb.get_followup_tasks(stage="investigation", limit=5)
    by_id = {row["task_id"]: row for row in tasks}

    assert by_id[high_id]["status"] == "completed"
    assert by_id[high_id]["outcome"] == "launched"
    assert by_id[low_id]["status"] == "queued"
    nb.close()


def test_followup_tasks_canonicalize_duplicate_fingerprint_result_ids():
    db_path = _tmp_db()
    nb = LabNotebook(db_path)

    exp_a = nb.start_experiment("synthesis", {"n_programs": 1}, "canonical-a")
    rid_a = nb.record_program_result(
        exp_a,
        graph_fingerprint="fp_shared_followup",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.48,
    )

    exp_b = nb.start_experiment("exact_graph_replay", {"n_programs": 1}, "canonical-b")
    rid_b = nb.record_program_result(
        exp_b,
        graph_fingerprint="fp_shared_followup",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.44,
        intentional_rerun_reason="exact_graph_replay",
        model_source="exact_graph_replay",
    )

    task_id = nb.enqueue_followup_task(
        stage="investigation",
        result_ids=[rid_a, rid_b],
        hypothesis="canonicalize duplicate siblings",
        config=RunConfig().to_dict(),
        priority_score=0.5,
    )

    task = nb.get_followup_tasks(stage="investigation", limit=5)[0]
    canonical = nb.resolve_canonical_result_id(rid_a)

    assert task["task_id"] == task_id
    assert task["result_ids_json"] == [canonical]
    assert sorted(task["metadata"]["requested_result_ids"]) == sorted([rid_a, rid_b])
    nb.close()


def test_threshold_calibration_snapshots_persist_selected_operating_point():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    for idx in range(18):
        nb.conn.execute(
            """INSERT INTO leaderboard
               (entry_id, result_id, timestamp, model_source, composite_score, tier,
                investigation_passed, investigation_loss_ratio, investigation_robustness,
                trust_label, comparability_label)
               VALUES (?, NULL, ?, 'graph_synthesis', ?, 'screening', 1, 0.22, 0.80, ?, ?)""",
            (
                f"pos_hist_{idx}",
                time.time() + idx,
                80.0 + idx * 0.5,
                "candidate_grade",
                "candidate_comparable",
            ),
        )
    for idx in range(18):
        nb.conn.execute(
            """INSERT INTO leaderboard
               (entry_id, result_id, timestamp, model_source, composite_score, tier,
                investigation_passed, investigation_loss_ratio, investigation_robustness,
                trust_label, comparability_label)
               VALUES (?, NULL, ?, 'graph_synthesis', ?, 'screening', 0, 0.95, 0.10, ?, ?)""",
            (
                f"neg_hist_{idx}",
                time.time() + 100 + idx,
                58.0 + idx * 0.6,
                "candidate_grade",
                "candidate_comparable",
            ),
        )
    nb.conn.commit()

    cfg = RunConfig(
        adaptive_thresholds_enabled=True,
        screening_promotion_percentile=90.0,
    )
    threshold_1 = runner._adaptive_screening_threshold(nb, cfg, floor=50.0)

    for idx in range(10):
        nb.conn.execute(
            """INSERT INTO leaderboard
               (entry_id, result_id, timestamp, model_source, composite_score, tier,
                investigation_passed, investigation_loss_ratio, investigation_robustness,
                trust_label, comparability_label)
               VALUES (?, NULL, ?, 'graph_synthesis', ?, 'screening', 0, 0.97, 0.08, ?, ?)""",
            (
                f"neg_shift_{idx}",
                time.time() + 200 + idx,
                84.0 + idx * 0.7,
                "candidate_grade",
                "candidate_comparable",
            ),
        )
    nb.conn.commit()

    threshold_2 = runner._adaptive_screening_threshold(nb, cfg, floor=50.0)

    snapshots = nb.get_threshold_calibrations(
        context="auto_investigate_screening",
        limit=2,
    )

    assert len(snapshots) == 2
    assert snapshots[0]["selected_threshold"] == pytest.approx(threshold_2)
    assert snapshots[1]["selected_threshold"] == pytest.approx(threshold_1)
    assert snapshots[0]["threshold_delta"] == pytest.approx(threshold_2 - threshold_1)
    assert snapshots[0]["metrics"]["mode"] in {"adaptive", "fallback", "floor_fallback"}
    nb.close()


def test_pending_replay_uses_exact_replay_pipeline_and_completes_task(monkeypatch):
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    task_id = nb.enqueue_followup_task(
        stage="replay",
        result_ids=["rid_replay"],
        hypothesis="replay uncertain frontier",
        config={"device": "cpu", "repeat_per_source": 2, "fast": True},
        priority_score=0.83,
        priority_reasons={"policy": "expected_information_gain", "score": 0.83},
    )
    nb.close()

    captured = {}

    def _fake_run_exact_replay(
        *,
        db_path,
        result_ids,
        repeat_per_source,
        device,
        hypothesis,
        fast,
        verbose,
    ):
        captured["db_path"] = str(db_path)
        captured["result_ids"] = list(result_ids)
        captured["repeat_per_source"] = repeat_per_source
        captured["device"] = device
        captured["hypothesis"] = hypothesis
        captured["fast"] = fast
        captured["verbose"] = verbose
        return "exp_replay_123"

    monkeypatch.setattr(
        "research.tools.exact_graph_replay.run_exact_replay",
        _fake_run_exact_replay,
    )

    assert runner._run_pending_replay() is True

    assert captured["result_ids"] == ["rid_replay"]
    assert captured["repeat_per_source"] == 2
    assert captured["device"] == "cpu"
    assert captured["fast"] is True
    assert captured["verbose"] is False

    nb = LabNotebook(db_path)
    task = next(
        row
        for row in nb.get_followup_tasks(stage="replay", limit=5)
        if row["task_id"] == task_id
    )
    assert task["status"] == "completed"
    assert task["outcome"] == "completed"
    assert task["metadata"]["replay_experiment_id"] == "exp_replay_123"
    nb.close()


def test_active_learning_replay_queue_skips_recent_and_duplicate_targets():
    db_path = _tmp_db()
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    exp_a = nb.start_experiment("synthesis", {"n_programs": 1}, "replay-suppress-a")
    rid_a = nb.record_program_result(
        exp_a,
        graph_fingerprint="fp_replay_suppress",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.47,
    )
    exp_b = nb.start_experiment(
        "exact_graph_replay", {"n_programs": 1}, "replay-suppress-b"
    )
    rid_b = nb.record_program_result(
        exp_b,
        graph_fingerprint="fp_replay_suppress",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.45,
        intentional_rerun_reason="exact_graph_replay",
        model_source="exact_graph_replay",
    )
    exp_c = nb.start_experiment("synthesis", {"n_programs": 1}, "replay-allow")
    rid_c = nb.record_program_result(
        exp_c,
        graph_fingerprint="fp_replay_allow",
        graph_json=_graph_json("attention"),
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        loss_ratio=0.49,
    )

    blocked_task_id = nb.enqueue_followup_task(
        stage="replay",
        result_ids=[rid_a],
        hypothesis="already queued replay",
        config={"device": "cpu", "repeat_per_source": 2, "fast": True},
        priority_score=0.8,
    )
    nb.complete_followup_task(blocked_task_id, outcome="completed")

    rows = [
        {
            "result_id": rid_a,
            "_active_learning": {
                "info_gain": 0.91,
                "threshold_proximity": 0.95,
                "ambiguity": 0.62,
                "instability": 0.21,
                "n_runs": 2,
            },
        },
        {
            "result_id": rid_b,
            "_active_learning": {
                "info_gain": 0.89,
                "threshold_proximity": 0.94,
                "ambiguity": 0.61,
                "instability": 0.22,
                "n_runs": 2,
            },
        },
        {
            "result_id": rid_c,
            "_active_learning": {
                "info_gain": 0.88,
                "threshold_proximity": 0.92,
                "ambiguity": 0.55,
                "instability": 0.18,
                "n_runs": 2,
            },
        },
    ]

    task_id = runner._queue_active_learning_replays(
        nb=nb,
        config=RunConfig(auto_investigate_top_n=3),
        rows=rows,
        source_context="test_replay_suppression",
    )

    assert task_id is not None
    tasks = nb.get_followup_tasks(stage="replay", limit=10)
    queued = next(row for row in tasks if row["task_id"] == task_id)

    assert queued["result_ids_json"] == [rid_c]
    assert rid_a not in queued["result_ids_json"]
    assert rid_b not in queued["result_ids_json"]
    nb.close()
