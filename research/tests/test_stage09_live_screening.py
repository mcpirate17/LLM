from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from research.orchestrator.executor import JobResult
from research.scientist.api_routes._strategy_preflight import (
    apply_live_screening_bias,
    normalize_start_mode,
)
from research.scientist.runner import ExperimentRunner, RunConfig
from research.tools.profile_component_scaffolds import build_gpt2_attn_scaffold


def _make_runner():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = str(Path(tmpdir.name) / "runner.db")
    runner = ExperimentRunner(db_path)
    return tmpdir, runner


def test_live_screening_bias_enables_stage09_gate():
    cfg = RunConfig()

    assert normalize_start_mode("live_screening") == "live_screening"

    changes = apply_live_screening_bias(cfg)

    assert cfg.enable_stage09_cheap_train_gate is True
    assert changes["enable_stage09_cheap_train_gate"]["from"] is False
    assert changes["enable_stage09_cheap_train_gate"]["to"] is True


def test_stage09_survivor_promotes_to_full_stage1():
    tmpdir, runner = _make_runner()
    try:
        runner._merge_s1_telemetry = MagicMock()
        runner._run_full_stage1_after_stage09 = MagicMock(
            return_value=(
                7.5,
                {
                    "passed": False,
                    "loss_ratio": 0.91,
                    "final_loss": 4.25,
                    "n_train_steps": 1,
                    "total_train_time_ms": 2.0,
                    "avg_step_time_ms": 2.0,
                },
            )
        )
        nb = MagicMock()
        nb.record_program_result.return_value = "rid_stage1"

        graph = build_gpt2_attn_scaffold("softmax_attention", model_dim=16)
        cfg = RunConfig(
            device="cpu",
            model_dim=16,
            n_layers=1,
            vocab_size=64,
            max_seq_len=16,
            stage1_steps=1,
            stage1_batch_size=1,
            collect_training_curve=False,
            enable_stage09_cheap_train_gate=True,
        )
        results = {
            "stage1_passed": 0,
            "stage09_passed": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "novel_count": 0,
            "survivors": [],
            "funnel_counts": {},
        }
        jr = JobResult(
            index=0,
            s1_result={
                "passed": True,
                "loss_ratio": 0.8,
                "final_loss": 3.8,
                "n_train_steps": 1,
                "total_train_time_ms": 1.0,
                "avg_step_time_ms": 1.0,
            },
            payload={
                "metrics": {},
                "graph": graph,
                "screening_stage": "stage09",
                "screening_seed": 101,
            },
        )

        runner._record_orchestrator_result(jr, nb, "exp_stage09", results, cfg)

        assert results["stage09_passed"] == 1
        assert results["funnel_counts"]["stage09_completed"] == 1
        assert results["funnel_counts"]["stage09_survived"] == 1
        assert results["funnel_counts"]["stage1_completed"] == 1
        assert results["stage1_passed"] == 0
        assert jr.payload["metrics"]["stage09_promoted_to_s1"] == 1
        assert jr.payload["metrics"]["compile_time_ms"] == 7.5
        runner._run_full_stage1_after_stage09.assert_called_once()
        nb.record_program_result.assert_called_once()
        assert nb.record_program_result.call_args.kwargs["stage1_passed"] is False
    finally:
        tmpdir.cleanup()


def test_stage09_failure_does_not_count_stage1_completion():
    tmpdir, runner = _make_runner()
    try:
        runner._merge_s1_telemetry = MagicMock()
        runner._run_full_stage1_after_stage09 = MagicMock()
        nb = MagicMock()
        nb.record_program_result.return_value = "rid_stage09_fail"

        graph = build_gpt2_attn_scaffold("softmax_attention", model_dim=16)
        cfg = RunConfig(enable_stage09_cheap_train_gate=True)
        results = {
            "stage1_passed": 0,
            "stage09_passed": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "novel_count": 0,
            "survivors": [],
            "funnel_counts": {},
        }
        jr = JobResult(
            index=0,
            s1_result={"passed": False, "error_type": "failed_stage09_gate"},
            payload={
                "metrics": {},
                "graph": graph,
                "screening_stage": "stage09",
                "screening_seed": 101,
            },
        )

        runner._record_orchestrator_result(jr, nb, "exp_stage09_fail", results, cfg)

        assert results["stage09_passed"] == 0
        assert results["funnel_counts"]["stage09_completed"] == 1
        assert results["funnel_counts"].get("stage09_survived", 0) == 0
        assert "stage1_completed" not in results["funnel_counts"]
        runner._run_full_stage1_after_stage09.assert_not_called()
        nb.record_program_result.assert_called_once()
        assert nb.record_program_result.call_args.kwargs["stage1_passed"] is False
    finally:
        tmpdir.cleanup()


def test_perf_report_uses_preprocessing_time_for_compile_metric():
    tmpdir, runner = _make_runner()
    try:
        report = runner._build_experiment_perf_report(
            results={
                "elapsed_seconds": 1.25,
                "total": 3,
                "stage1_passed": 1,
                "_compile_times_ms": [12.5, 17.5],
            },
            queue_telemetry={"preprocessing_avg_ms": 12.5, "submit_wait_avg_ms": 1.0},
        )

        assert report["trace_avg_ms"]["compile"] == 15.0
        assert report["perf_contract"]["metrics"]["compile_time_ms"] == 15.0
    finally:
        tmpdir.cleanup()
