"""Dashboard mixin: experiment-level perf report + contract emission."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ._perf_aggregate import (
    aggregate_gpu_starvation,
    aggregate_kernel_hotspots,
    aggregate_throughput,
    aggregate_trace_avg_ms,
    aggregate_training_program_scheduling,
    emit_research_perf_contract,
)
from ...perf_contract import build_duplicate_work_report


class _DashboardPerfMixin:
    """Aggregate per-program perf traces into one experiment-level report."""

    def _build_experiment_perf_report(
        self,
        results: Dict[str, Any],
        queue_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-program perf traces into one experiment-level JSON report."""
        perf_traces = results.get("_perf_traces", []) or []
        queue_telemetry = queue_telemetry or {}
        queue_compile_ms = float(
            queue_telemetry.get("preprocessing_avg_ms", 0.0) or 0.0
        )
        trace_avg_ms = aggregate_trace_avg_ms(
            perf_traces, results.get("_compile_times_ms", []) or [], queue_compile_ms
        )
        gpu_starvation = aggregate_gpu_starvation(results.get("_gpu_starvation") or [])
        kernel_hotspots = aggregate_kernel_hotspots(results.get("_kernel_timing") or [])
        tp_scheduling = aggregate_training_program_scheduling(
            results.get("training_program_scheduling")
        )
        duplicate_work = build_duplicate_work_report(
            repeated_keys={
                "graph_fingerprint_dedup": int(results.get("skipped_dedup", 0) or 0)
            },
            hints=[
                "Large dedup counts indicate search-space waste rather than runtime overhead."
            ],
        )
        report = {
            "generated_at": time.time(),
            "programs_profiled": len(perf_traces),
            "trace_avg_ms": trace_avg_ms,
            "avg_throughput_tok_s": round(aggregate_throughput(perf_traces), 2),
            "gpu_starvation": gpu_starvation,
            "kernel_hotspots": kernel_hotspots,
            "queue_telemetry": queue_telemetry,
            "training_program_scheduling": tp_scheduling,
            "duplicate_work": duplicate_work,
        }

        # Only emit an artifact when at least one budget-gate input carries
        # real data. Runs without any profiling produce artifacts where every
        # check reports `missing_metric` — pure noise. Partial data (e.g. just
        # compile_time) still emits so early/light runs stay visible.
        has_profiling_signal = (
            bool(trace_avg_ms)
            or bool(queue_telemetry)
            or gpu_starvation["max_stall_ms"] > 0.0
            or len(perf_traces) > 0
        )
        if has_profiling_signal:
            emit_research_perf_contract(
                report, results, queue_telemetry, duplicate_work
            )
        return report
