from __future__ import annotations

from research.tools.profile_screening_hotpaths import build_screening_hotpath_report


def test_screening_hotpath_report_quick_shape():
    report = build_screening_hotpath_report(fixture="quick")

    assert "stage1" in report
    assert "orchestrator" in report
    assert "metrics" in report
    assert report["fixture"] == "quick"

    metrics = report["metrics"]
    assert metrics["stage1_base_ms"] > 0.0
    assert metrics["stage1_lean_ms"] > 0.0
    assert metrics["stage1_speedup"] > 0.0
    assert metrics["stage1_base_compile_ms"] > 0.0
    assert metrics["stage1_lean_compile_ms"] > 0.0
    assert metrics["orchestrator_jobs"] >= 1
    assert "queue_prep_wait_ms" in metrics
    assert "queue_scheduling_wait_ms" in metrics
    assert report["stage1_base_median_ms"] == metrics["stage1_base_ms"]
    assert report["stage09_gate_median_ms"] == metrics["stage1_lean_ms"]
    assert report["stage09_speedup_x"] == metrics["stage1_speedup"]
    assert report["orchestrator_total_ms"] == metrics["orchestrator_total_ms"]
    assert report["queue_telemetry"] == report["orchestrator"]["queue_telemetry"]

    telemetry = report["orchestrator"]["queue_telemetry"]
    assert "prep_queue_wait_avg_ms" in telemetry
    assert "scheduling_wait_avg_ms" in telemetry
    assert report["stage1"]["config"]["repeats"] >= 1
