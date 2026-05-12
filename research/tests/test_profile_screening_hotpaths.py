from __future__ import annotations

from research.tools.profile_screening_hotpaths import (
    _build_config,
    _variant_config,
    build_screening_hotpath_report,
)


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
    assert report["stage1"]["config"]["warmup_compile_ms"] > 0.0
    warmup = report["stage1"]["config"]["warmup"]
    assert warmup["base_compile_ms"] > 0.0
    assert warmup["base_elapsed_ms"] > 0.0
    assert warmup["lean_compile_ms"] > 0.0
    assert warmup["lean_elapsed_ms"] > 0.0
    assert "perf_summary_ms" in warmup["base_diagnostics"]
    assert "probe_timings_ms" in warmup["lean_diagnostics"]

    base_diag = report["stage1"]["base"]["diagnostics"]
    lean_diag = report["stage1"]["lean"]["diagnostics"]
    assert "perf_summary_ms" in base_diag
    assert "top_timing_sources" in base_diag
    assert "probe_timings_ms" in lean_diag
    assert report["stage1"]["base"]["diagnostics_runs"]
    assert report["stage1"]["lean"]["diagnostics_runs"]
    assert "job_diagnostics" in report["orchestrator"]


def test_screening_gating_variant_flags():
    config = _build_config(fixture="standard")

    language = _variant_config(config, "language_only")
    assert language.skip_binding_probes is True
    assert language.skip_ar_probe is True
    assert language.skip_ar_gate is True
    assert language.skip_screening_blimp is False

    binding = _variant_config(config, "binding_plus_ar_gate")
    assert binding.skip_screening_wikitext is True
    assert binding.skip_screening_hellaswag is True
    assert binding.skip_screening_blimp is True
    assert binding.skip_binding_probes is False
    assert binding.skip_ar_probe is True
    assert binding.skip_ar_gate is False
