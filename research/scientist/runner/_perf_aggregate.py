"""Pure-function aggregators for experiment-level perf reports.

Consumes the ad-hoc dict shapes emitted by the runner (``_perf_traces``,
``_gpu_starvation``, ``_kernel_timing``, ``training_program_scheduling``)
and produces aggregate stats consumed by ``_build_experiment_perf_report``.

Kept separate from dashboard.py so the reductions are independently
testable and the main method stays under the 100-line god-function limit.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from ...perf_contract import build_perf_contract_with_gate, emit_perf_artifact


def aggregate_trace_avg_ms(
    perf_traces: Iterable[Dict[str, Any]],
    compile_times_ms: Iterable[Any],
    queue_compile_ms: float,
) -> Dict[str, float]:
    """Average the ``summary_ms`` sections of each perf trace, then overlay
    compile-time data from whichever source has non-zero values."""
    trace_totals: Dict[str, float] = defaultdict(float)
    trace_counts: Dict[str, int] = defaultdict(int)
    for trace_report in perf_traces:
        summary = (trace_report or {}).get("summary_ms", {})
        if not isinstance(summary, dict):
            continue
        for name, value in summary.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            trace_totals[name] += val
            trace_counts[name] += 1

    trace_avg_ms = {
        name: round(trace_totals[name] / max(1, trace_counts[name]), 4)
        for name in sorted(trace_totals.keys())
    }

    compile_samples = [float(v) for v in (compile_times_ms or []) if v is not None]
    compile_avg_ms = (
        sum(compile_samples) / len(compile_samples) if compile_samples else 0.0
    )
    if compile_avg_ms > 0.0:
        trace_avg_ms["compile"] = round(compile_avg_ms, 4)
    elif queue_compile_ms > 0.0 and trace_avg_ms.get("compile", 0.0) <= 0.0:
        trace_avg_ms["compile"] = round(queue_compile_ms, 4)
    return trace_avg_ms


def aggregate_throughput(perf_traces: Iterable[Dict[str, Any]]) -> float:
    vals = [
        float(t.get("avg_throughput_tok_s", 0.0) or 0.0)
        for t in perf_traces
        if t.get("avg_throughput_tok_s") is not None
    ]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_gpu_starvation(
    starvation: Iterable[Dict[str, Any]],
) -> Dict[str, float]:
    count = 0
    total_ms = 0.0
    max_ms = 0.0
    for item in starvation:
        if not isinstance(item, dict):
            continue
        count += int(item.get("count", 0) or 0)
        total_ms += float(item.get("total_stall_ms", 0.0) or 0.0)
        max_ms = max(max_ms, float(item.get("max_stall_ms", 0.0) or 0.0))
    return {
        "event_count": count,
        "total_stall_ms": round(total_ms, 4),
        "max_stall_ms": round(max_ms, 4),
    }


def aggregate_kernel_hotspots(
    kernel_samples: Iterable[Dict[str, Any]],
    *,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Aggregate per-op timings across kernel samples. Accepts both the legacy
    ``top_ops`` list format and the newer ``{op_name: ms}`` map format."""
    op_totals: Dict[str, Dict[str, float]] = {}
    for sample in kernel_samples:
        if not isinstance(sample, dict):
            continue
        if "top_ops" not in sample:
            for op_name, ms in sample.items():
                if not isinstance(ms, (int, float)):
                    continue
                slot = op_totals.setdefault(
                    op_name,
                    {"cpu_ms": 0.0, "cuda_ms": 0.0, "calls": 0.0, "samples": 0.0},
                )
                slot["cuda_ms"] += float(ms)
                slot["samples"] += 1.0
            continue
        for op in sample.get("top_ops", []) or []:
            op_name = str(op.get("op", "unknown"))
            slot = op_totals.setdefault(
                op_name,
                {"cpu_ms": 0.0, "cuda_ms": 0.0, "calls": 0.0, "samples": 0.0},
            )
            slot["cpu_ms"] += float(op.get("cpu_ms", 0.0) or 0.0)
            slot["cuda_ms"] += float(op.get("cuda_ms", 0.0) or 0.0)
            slot["calls"] += float(op.get("calls", 0.0) or 0.0)
            slot["samples"] += 1.0

    hotspots = [
        {
            "op": op_name,
            "avg_cpu_ms": round(agg["cpu_ms"] / max(1.0, agg["samples"]), 4),
            "avg_cuda_ms": round(agg["cuda_ms"] / max(1.0, agg["samples"]), 4),
            "avg_calls": round(agg["calls"] / max(1.0, agg["samples"]), 2),
        }
        for op_name, agg in op_totals.items()
    ]
    hotspots.sort(
        key=lambda row: max(row["avg_cuda_ms"], row["avg_cpu_ms"]), reverse=True
    )
    return hotspots[:top_n]


def aggregate_training_program_scheduling(
    rows: Optional[List[Dict[str, Any]]],
) -> Dict[str, float]:
    rows = rows or []
    avg_ms = [float(r.get("scheduling_avg_ms", 0.0) or 0.0) for r in rows]
    max_ms = [float(r.get("scheduling_max_ms", 0.0) or 0.0) for r in rows]
    return {
        "n_sources": len(rows),
        "avg_schedule_ms": round(sum(avg_ms) / len(avg_ms), 4) if avg_ms else 0.0,
        "max_schedule_ms": round(max(max_ms), 4) if max_ms else 0.0,
    }


def _research_contract_metrics(
    report: Dict[str, Any],
    results: Dict[str, Any],
    queue_telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    trace_avg_ms = report["trace_avg_ms"]
    return {
        "total_time_ms": round(
            float(results.get("elapsed_seconds", 0.0) or 0.0) * 1000.0, 4
        ),
        "avg_throughput_tok_s": report["avg_throughput_tok_s"],
        "programs_profiled": report["programs_profiled"],
        "compile_time_ms": trace_avg_ms.get("compile", 0.0),
        "forward_pass_ms": trace_avg_ms.get("forward_pass", 0.0),
        "backward_pass_ms": trace_avg_ms.get("backward_pass", 0.0),
        "optimizer_step_ms": trace_avg_ms.get("optimizer_step", 0.0),
        "queue_submit_wait_ms": float(
            queue_telemetry.get("submit_wait_avg_ms", 0.0) or 0.0
        ),
        "queue_prep_wait_ms": float(
            queue_telemetry.get("prep_queue_wait_avg_ms", 0.0) or 0.0
        ),
        "queue_scheduling_wait_ms": float(
            queue_telemetry.get("scheduling_wait_avg_ms", 0.0) or 0.0
        ),
        "gpu_starvation_max_ms": report["gpu_starvation"]["max_stall_ms"],
    }


def emit_research_perf_contract(
    report: Dict[str, Any],
    results: Dict[str, Any],
    queue_telemetry: Dict[str, Any],
    duplicate_work: Dict[str, Any],
) -> None:
    """Run the `research_default` budget gate, build the contract, emit the
    artifact, and attach contract/verdict/path onto ``report`` in place.

    The gate evaluates the richer ``report`` (so nested keys like
    ``trace_avg_ms.forward_pass`` resolve) while the contract itself carries a
    flat, named-metric payload for downstream consumers."""
    contract, budget_verdict = build_perf_contract_with_gate(
        component="research",
        workload="experiment_screening",
        metrics=_research_contract_metrics(report, results, queue_telemetry),
        budget_profile="research_default",
        identity={
            "experiment_id": results.get("experiment_id"),
            "total_programs": results.get("total", 0),
            "stage1_passed": results.get("stage1_passed", 0),
        },
        duplicate_work=duplicate_work,
        # research_default gates on nested keys (trace_avg_ms.*, gpu_starvation.*)
        # so the gate consumes the full aggregated report, not the flat metrics.
        gate_payload=report,
    )
    artifact_slug = str(
        results.get("experiment_id") or f"research_perf_{int(time.time())}"
    )
    artifact_path = emit_perf_artifact(contract, slug=artifact_slug)
    contract["artifact_path"] = artifact_path
    report["perf_contract"] = contract
    report["perf_budget_gate"] = budget_verdict
    report["perf_artifact_path"] = artifact_path
