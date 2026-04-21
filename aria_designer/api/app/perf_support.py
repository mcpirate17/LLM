from __future__ import annotations

from typing import Any, Dict, Optional

from research.perf_contract import build_perf_contract_with_gate, emit_perf_artifact

from .config import settings
from .type_utils import dig, safe_float


def _stage_elapsed_metrics(stages: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for stage_name, stage in (stages or {}).items():
        if not isinstance(stage, dict):
            continue
        elapsed_ms = stage.get("elapsed_ms")
        if elapsed_ms is not None:
            try:
                metrics[f"{stage_name}_time_ms"] = float(elapsed_ms)
            except (TypeError, ValueError):
                continue
    return metrics


def designer_metrics_from_stages(
    stages: Dict[str, Any],
    total_time_ms: Optional[float],
    status: str,
) -> Dict[str, Any]:
    profiling = dig(stages, "profiling", "metrics", default={})
    compilation = dig(stages, "compilation", "metrics", default={})
    sandbox = dig(stages, "sandbox", "metrics", default={})
    metrics: Dict[str, Any] = {
        "total_time_ms": safe_float(total_time_ms),
        "compile_time_ms": safe_float(compilation.get("compile_time_ms")),
        "forward_time_ms": safe_float(sandbox.get("forward_ms")),
        "backward_time_ms": safe_float(sandbox.get("backward_ms")),
        "peak_memory_mb": safe_float(sandbox.get("peak_memory_mb")),
        "native_coverage": safe_float(profiling.get("native_coverage")),
        "total_flops_per_token": safe_float(profiling.get("total_flops_per_token")),
        "total_params": safe_float(profiling.get("total_params")),
        "status_code": 1.0 if status == "success" else 0.0,
    }
    metrics.update(_stage_elapsed_metrics(stages))
    return metrics


def _build_designer_perf_bundle(
    *,
    run_id: Optional[str],
    workflow_id: Optional[str],
    metrics: Dict[str, Any],
    status: str = "unknown",
    workload: str = "workflow_evaluation",
    duplicate_work: Optional[Dict[str, Any]] = None,
    slug: Optional[str] = None,
) -> Dict[str, Any]:
    contract, budget_verdict = build_perf_contract_with_gate(
        component="aria_designer",
        workload=workload,
        metrics=metrics,
        budget_profile="designer_interactive",
        identity={"run_id": run_id, "workflow_id": workflow_id, "status": status},
        duplicate_work=duplicate_work,
    )
    artifact_path = emit_perf_artifact(
        contract, slug=(slug or run_id or workflow_id or "designer_eval")
    )
    contract["artifact_path"] = artifact_path
    return {
        "perf_contract": contract,
        "perf_artifact_path": artifact_path,
        "perf_budget_gate": budget_verdict,
    }


def _discovery_url_for_fingerprint(fingerprint: str | None) -> str | None:
    token = str(fingerprint or "").strip()
    if not token:
        return None
    return f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/?search={token}"


def _compute_eval_composite_score(stage_metrics: Dict[str, Any]) -> float:
    benchmark_score = safe_float(dig(stage_metrics, "benchmarking", "summary", "score"))
    novelty = safe_float(dig(stage_metrics, "novelty", "overall_novelty"))
    efficiency = safe_float(dig(stage_metrics, "compression", "efficiency_score"))
    stability = safe_float(dig(stage_metrics, "sandbox", "stability_score"))
    raw = (
        (benchmark_score * 100.0)
        + (novelty * 40.0)
        + (efficiency * 30.0)
        + (stability * 20.0)
    )
    return round(max(0.0, raw), 3)
