import pytest
from research.eval.perf_budget import evaluate_perf_budget_gate

pytestmark = pytest.mark.unit


def test_perf_budget_gate_passes_under_limits():
    perf_report = {
        "trace_avg_ms": {
            "compile": 100.0,
            "forward_pass": 20.0,
            "backward_pass": 30.0,
            "optimizer_step": 10.0,
        },
        "queue_telemetry": {
            "scheduling_wait_avg_ms": 5.0,
            "submit_wait_avg_ms": 1.0,
        },
        "gpu_starvation": {
            "max_stall_ms": 3.0,
            "total_stall_ms": 12.0,
        },
    }
    verdict = evaluate_perf_budget_gate(perf_report)
    assert verdict["passed"] is True
    assert verdict["n_failed"] == 0


def test_perf_budget_gate_fails_when_thresholds_exceeded():
    perf_report = {
        "trace_avg_ms": {
            "compile": 500.0,
            "forward_pass": 200.0,
            "backward_pass": 300.0,
            "optimizer_step": 100.0,
        },
        "queue_telemetry": {
            "scheduling_wait_avg_ms": 100.0,
            "submit_wait_avg_ms": 50.0,
        },
        "gpu_starvation": {
            "max_stall_ms": 80.0,
            "total_stall_ms": 900.0,
        },
    }
    verdict = evaluate_perf_budget_gate(perf_report)
    assert verdict["passed"] is False
    assert verdict["n_failed"] > 0


def test_perf_budget_gate_fails_for_missing_metrics():
    verdict = evaluate_perf_budget_gate({})
    assert verdict["passed"] is False
    assert verdict["n_failed"] > 0


def test_designer_budget_profile_uses_nested_metrics_and_duplicate_work():
    perf_report = {
        "metrics": {
            "total_time_ms": 400.0,
            "compile_time_ms": 120.0,
            "native_coverage": 0.75,
        },
        "duplicate_work": {
            "detected_count": 0,
        },
    }
    verdict = evaluate_perf_budget_gate(perf_report, budget_profile="designer_interactive")
    assert verdict["passed"] is True


def test_native_coverage_budget_is_minimum_not_maximum():
    perf_report = {
        "metrics": {
            "total_time_ms": 400.0,
            "compile_time_ms": 120.0,
            "native_coverage": 0.1,
        },
        "duplicate_work": {
            "detected_count": 0,
        },
    }
    verdict = evaluate_perf_budget_gate(perf_report, budget_profile="designer_interactive")
    failed = {check["metric"] for check in verdict["checks"] if not check["passed"]}
    assert "metrics.native_coverage" in failed
