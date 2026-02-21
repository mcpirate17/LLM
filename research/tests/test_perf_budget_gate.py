from research.eval.perf_budget import evaluate_perf_budget_gate


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
