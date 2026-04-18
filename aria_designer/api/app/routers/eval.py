from __future__ import annotations

import asyncio
import json
import logging
import time as _time_mod
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import settings
from ..benchmark_targets import build_benchmark_analysis
from ..models import (
    RunWorkflowRequest,
    utc_now_iso as _utc_now,
)
from ..shared_api import (
    HAS_BRIDGE,
    HAS_PROFILER,
    _build_designer_perf_bundle,
    _collect_workflow_semantic_warnings,
    _compute_eval_composite_score,
    _discovery_url_for_fingerprint,
    _get_run,
    _list_runs,
    _require_run,
    _store_run,
    _sync_lineage_to_research,
    _update_run,
    _RESEARCH_ROOT,
    bridge_analyze_compression,
    bridge_analyze_routing,
    bridge_evaluate,
    designer_metrics_from_stages,
)
from research.defaults import MODEL_DIM, VOCAB_SIZE
from research.perf_contract import build_duplicate_work_report
from ..type_utils import dig, safe_float

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["eval"])


# ── SSE Helpers ───────────────────────────────────────────────────────


def _sse_json(obj: Any) -> str:
    """Serialize *obj* to JSON, handling numpy scalars and non-serializable types."""
    return json.dumps(obj, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def _sse_stage(
    stage: str,
    status: str,
    elapsed_ms: float | None = None,
    metrics: dict | None = None,
    error: str | None = None,
    error_details: dict | None = None,
) -> str:
    """Format a single SSE ``event: stage`` frame."""
    payload: Dict[str, Any] = {"stage": stage, "status": status}
    if elapsed_ms is not None:
        payload["elapsed_ms"] = round(elapsed_ms, 1)
    if metrics is not None:
        payload["metrics"] = metrics
    if error is not None:
        payload["error"] = error
    if error_details is not None:
        payload["error_details"] = error_details
    return f"event: stage\ndata: {_sse_json(payload)}\n\n"


def _build_error_details(
    *,
    stage: str | None,
    error: Any,
    error_type: str | None = None,
    semantic_warnings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Normalize an exception/string into a structured error payload."""
    message = str(error) if error is not None else ""
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return {
        "stage": stage,
        "error_type": error_type or getattr(error, "__class__", type(error)).__name__,
        "error_message": message,
        "root_cause_code": stage or error_type or "unknown",
        "traceback_excerpt": "\n".join(lines[-6:]) if lines else None,
        "semantic_warnings": semantic_warnings or [],
    }


# ── Stage work functions (called inside _run_stage) ──────────────────


def _try_fetch_original_graph(metadata: dict, model_dim: int):
    """Try to fetch the original ComputationGraph from research notebook.

    When a workflow was imported from research (has result_id in metadata),
    deserialize the original graph_json directly, bypassing the lossy
    workflow_to_graph() round-trip.

    Returns (graph, cg_to_aria_map) on success, None on failure.
    """
    result_id = metadata.get("result_id")
    if not result_id:
        return None

    db_path = _RESEARCH_ROOT / "lab_notebook.db"
    if not db_path.exists():
        return None

    try:
        from research.scientist.notebook import LabNotebook
        from research.synthesis.serializer import graph_from_json

        nb = LabNotebook(str(db_path))
        detail = nb.get_program_detail(str(result_id))
        if detail is None:
            return None

        graph_json_str = detail.get("graph_json")
        if not graph_json_str:
            return None

        graph = graph_from_json(graph_json_str)
        graph.model_dim = model_dim

        expected_fp = metadata.get("graph_fingerprint")
        if expected_fp and graph.fingerprint() != expected_fp:
            logger.info(
                "Workflow has been modified (fingerprint changed from %s to %s). "
                "Bypassing original graph load.",
                graph.fingerprint(),
                expected_fp,
            )
            return None

        cg_to_aria = {}
        for cg_id in graph.topological_order():
            node = graph.nodes[cg_id]
            if node.is_input:
                cg_to_aria[cg_id] = f"input_{cg_id}"
            else:
                cg_to_aria[cg_id] = f"op_{cg_id}_{node.op_name}"

        return (graph, cg_to_aria)

    except Exception:
        logger.debug(
            "Failed to fetch original graph for result_id=%s", result_id, exc_info=True
        )
        return None


async def _stage_conversion(wf: dict, model_dim: int) -> dict:
    """Stage 1 -- convert workflow JSON to ComputationGraph."""
    metadata = wf.get("metadata", {})
    original = await asyncio.to_thread(_try_fetch_original_graph, metadata, model_dim)

    if original is not None:
        graph, cg_to_aria = original
        used_original = True
    else:
        from aria_designer.runtime.bridge import workflow_to_graph as _w2g

        graph, id_map = await asyncio.to_thread(_w2g, wf, model_dim, return_id_map=True)
        cg_to_aria = {v: k for k, v in id_map.items()}
        used_original = False

    metrics = {
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "params_estimate": int(graph.n_params_estimate()),
        "has_gradient_path": bool(graph.has_gradient_path()),
        "graph_fingerprint": graph.fingerprint(),
        "used_original_graph": used_original,
    }
    return {"metrics": metrics, "graph": graph, "cg_to_aria": cg_to_aria}


async def _stage_profiling(
    graph: Any,
    cg_to_aria: dict,
    model_dim: int,
    batch_size: int,
    seq_len: int,
    duplicate_work_avoided: dict,
) -> dict:
    """Stage 2 -- static FLOPs / memory profiling."""
    if not HAS_PROFILER:
        return {
            "metrics": {"skipped": True, "reason": "profiler not available"},
            "op_profiles_for_nodes": [],
        }

    from aria_designer.runtime.profiler import profile_static_graph

    duplicate_work_avoided["workflow_to_graph"] += 1
    report = await asyncio.to_thread(
        profile_static_graph,
        graph,
        model_dim=model_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    report_dict = report.to_dict()
    mapped_profiles = [
        {**op, "aria_node_id": cg_to_aria.get(op.get("node_id"))}
        for op in report_dict.get("op_profiles", [])
    ]
    metrics = {
        "total_flops_per_token": report_dict.get("total_flops_per_token", 0),
        "total_params": report_dict.get("total_params", 0),
        "total_memory_bytes": report_dict.get("total_memory_bytes", 0),
        "flops_by_category": report_dict.get("flops_by_category", {}),
        "bottleneck_ops": report_dict.get("bottleneck_ops", []),
        "native_coverage": report_dict.get("native_coverage", 0),
        "op_profiles": mapped_profiles,
    }
    return {"metrics": metrics, "op_profiles_for_nodes": mapped_profiles}


async def _stage_compilation(graph: Any, vocab_size: int) -> dict:
    """Stage 3 -- compile ComputationGraph into a torch.nn.Module."""
    from research.synthesis.compiler import compile_model

    model = await asyncio.to_thread(compile_model, [graph], vocab_size=vocab_size)
    return {"metrics": {}, "model": model}


async def _stage_sandbox(
    model: Any, batch_size: int, seq_len: int, vocab_size: int, device: str
) -> dict:
    """Stage 4 -- safe_eval sandbox (forward + backward probe)."""
    from research.eval.sandbox import safe_eval

    sandbox = await asyncio.to_thread(
        safe_eval,
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
    )
    metrics = {
        "passed": bool(sandbox.passed),
        "forward_ms": float(getattr(sandbox, "forward_time_ms", 0)),
        "backward_ms": float(getattr(sandbox, "backward_time_ms", 0)),
        "param_count": int(getattr(sandbox, "param_count", 0)),
        "peak_memory_mb": float(getattr(sandbox, "peak_memory_mb", 0)),
        "grad_norm": float(getattr(sandbox, "grad_norm", 0)),
        "stability_score": float(getattr(sandbox, "stability_score", 0)),
        "native_abi_probe": getattr(sandbox, "native_abi_probe", None),
        "routing_report": getattr(sandbox, "routing_report", None),
    }
    return {"metrics": metrics, "sandbox": sandbox}


async def _stage_routing(model: Any, graph: Any, cg_to_aria: dict) -> dict:
    """Stage 4.5 -- routing analysis (Phase 5.1 live-sync)."""
    rt_results = await asyncio.to_thread(bridge_analyze_routing, model, graph)
    mapped_rt = [
        {**entry, "aria_node_id": cg_to_aria.get(entry.get("node_id"))}
        for entry in rt_results
        if cg_to_aria.get(entry.get("node_id"))
    ]
    return {"metrics": {"op_routing": mapped_rt}}


async def _stage_compression(
    model: Any, graph: Any, vocab_size: int, device: str, batch_size: int, seq_len: int
) -> dict:
    """Stage 5 -- compression / efficiency analysis."""
    comp_result = await asyncio.to_thread(
        bridge_analyze_compression,
        model,
        graph,
        vocab_size=vocab_size,
        device=device,
        batch_size=batch_size,
        seq_len=min(seq_len, 64),
    )
    return {"metrics": comp_result.to_dict()}


async def _stage_fingerprint(
    model: Any,
    run_fingerprint: bool,
    seq_len: int,
    model_dim: int,
    vocab_size: int,
    device: str,
) -> dict:
    """Stage 6 -- behavioural fingerprint (CKA, locality, sparsity)."""
    if not run_fingerprint:
        return {"metrics": {"skipped": True}, "fp_obj": None}

    from research.eval.fingerprint import compute_fingerprint

    fp_obj = await asyncio.to_thread(
        compute_fingerprint,
        model,
        seq_len=min(seq_len, 64),
        model_dim=model_dim,
        vocab_size=vocab_size,
        device=device,
    )
    metrics = {
        "cka_vs_transformer": float(getattr(fp_obj, "cka_vs_transformer", 0)),
        "cka_vs_ssm": float(getattr(fp_obj, "cka_vs_ssm", 0)),
        "cka_vs_conv": float(getattr(fp_obj, "cka_vs_conv", 0)),
        "locality": float(getattr(fp_obj, "interaction_locality", 0)),
        "sparsity": float(getattr(fp_obj, "interaction_sparsity", 0)),
        "intrinsic_dim": float(getattr(fp_obj, "intrinsic_dim", 0)),
        "isotropy": float(getattr(fp_obj, "isotropy", 0)),
    }
    return {"metrics": metrics, "fp_obj": fp_obj}


async def _stage_novelty(graph: Any, fp_obj: Any, run_novelty: bool) -> dict:
    """Stage 7 -- novelty scoring."""
    if not run_novelty:
        return {"metrics": {"skipped": True}}

    from research.eval.metrics import novelty_score

    ns = await asyncio.to_thread(novelty_score, graph, fingerprint=fp_obj)
    metrics = {
        "structural_novelty": float(ns.structural_novelty),
        "behavioral_novelty": float(ns.behavioral_novelty),
        "overall_novelty": float(ns.overall_novelty),
        "most_similar_to": getattr(ns, "most_similar_to", ""),
    }
    return {"metrics": metrics}


# ── Stage orchestration helpers ───────────────────────────────────────


async def _run_nonfatal_stage(
    stage_name: str,
    coro,
    ctx: "_EvalRunContext",
):
    """Run a non-fatal pipeline stage: yield SSE events, swallow errors.

    *coro* must be an awaitable that returns ``{"metrics": dict, ...}``.
    Returns the full result dict on success, or *None* on failure.
    """
    yield _sse_stage(stage_name, "running")
    t0 = _time_mod.monotonic()
    try:
        result = await coro
        elapsed = (_time_mod.monotonic() - t0) * 1000
        ctx.stage_done_payload(stage_name, elapsed, result["metrics"])
        yield _sse_stage(stage_name, "done", elapsed, result["metrics"])
    except Exception as e:
        elapsed = (_time_mod.monotonic() - t0) * 1000
        yield _sse_stage(
            stage_name,
            "error",
            elapsed,
            error=str(e),
            error_details=_build_error_details(stage=stage_name, error=e),
        )
        result = None
    ctx._last_stage_result = result


def _build_lineage_payload(
    run_id: str,
    wf: dict,
    acc: dict,
    status: str,
    error: str | None,
    error_stage: str | None,
    total_ms: float | None,
) -> dict:
    """Construct the lineage sync payload for a completed eval run."""
    return {
        "run_id": run_id,
        "workflow_id": wf.get("workflow_id"),
        "workflow_version": wf.get("version") or dig(wf, "metadata", "version"),
        "graph_fingerprint": dig(acc, "conversion", "graph_fingerprint"),
        "status": status,
        "source": "aria_designer",
        "total_time_ms": total_ms,
        "metrics": {
            "sandbox_passed": dig(acc, "sandbox", "passed"),
            "overall_novelty": dig(acc, "novelty", "overall_novelty"),
            "efficiency_score": dig(acc, "compression", "efficiency_score"),
            "benchmark_target_score": dig(acc, "benchmarking", "summary", "score"),
            "composite_score": acc.get("composite_score"),
        },
        "payload": {
            "error": error,
            "error_stage": error_stage,
            "result": acc,
        },
        "created_at": _time_mod.time(),
    }


class _EvalRunContext:
    """Encapsulates run-level state and persistence for ``event_stream``."""

    __slots__ = (
        "run_id",
        "wf",
        "budget",
        "accumulated",
        "total_t0",
        "duplicate_work_avoided",
        "_lineage_synced",
        "_last_stage_result",
    )

    def __init__(self, run_id: str, wf: dict, budget: dict) -> None:
        self.run_id = run_id
        self.wf = wf
        self.budget = budget
        self.accumulated: Dict[str, Any] = {}
        self.total_t0 = _time_mod.monotonic()
        self.duplicate_work_avoided = {"workflow_to_graph": 0}
        self._lineage_synced = False
        self._last_stage_result: dict | None = None

    # -- persistence helpers --------------------------------------------------

    def persist_stage(self, stage_name: str, stage_data: dict) -> None:
        """Write stage result to the run store for REST observability."""
        _update_run(
            self.run_id,
            {
                "stages": {
                    **(_get_run(self.run_id) or {}).get("stages", {}),
                    stage_name: stage_data,
                },
            },
        )

    def persist_done(
        self,
        status: str,
        error: str | None = None,
        error_stage: str | None = None,
        total_ms: float | None = None,
    ) -> None:
        stages = (_get_run(self.run_id) or {}).get("stages", {})
        perf_bundle = _build_designer_perf_bundle(
            run_id=self.run_id,
            workflow_id=self.wf.get("workflow_id"),
            metrics=designer_metrics_from_stages(stages, total_ms, status),
            status=status,
            duplicate_work=build_duplicate_work_report(
                avoided_keys=self.duplicate_work_avoided,
            ),
        )
        acc = self.accumulated
        acc["perf_contract"] = perf_bundle["perf_contract"]
        acc["perf_budget_gate"] = perf_bundle["perf_budget_gate"]
        acc["perf_artifact_path"] = perf_bundle["perf_artifact_path"]
        _update_run(
            self.run_id,
            {
                "status": status,
                "error": error,
                "error_stage": error_stage,
                "error_details": (
                    _build_error_details(
                        stage=error_stage,
                        error=error,
                        error_type=status,
                        semantic_warnings=self.accumulated.get("semantic_warnings"),
                    )
                    if error
                    else None
                ),
                "total_time_ms": total_ms,
                "result": acc,
                "completed_at": _utc_now(),
                "perf_contract": perf_bundle["perf_contract"],
                "perf_artifact_path": perf_bundle["perf_artifact_path"],
                "perf_budget_gate": perf_bundle["perf_budget_gate"],
            },
        )
        if not self._lineage_synced and settings.LINEAGE_SYNC_ENABLED:
            self._lineage_synced = _sync_lineage_to_research(
                self._build_lineage(status, error, error_stage, total_ms),
            )

    def attach_benchmarking(self) -> None:
        self.accumulated["benchmarking"] = build_benchmark_analysis(
            self.accumulated,
            external_observed=self.budget.get("benchmark_observed"),
        )

    def fatal_done(self, error: str, stage: str) -> str:
        """Build the ``event: done`` SSE frame for a fatal error."""
        total_ms = round((_time_mod.monotonic() - self.total_t0) * 1000, 1)
        self.attach_benchmarking()
        self.persist_done("error", error=error, error_stage=stage, total_ms=total_ms)
        return f"event: done\ndata: {_sse_json({'status': 'error', 'error': error, 'error_stage': stage, 'error_details': _build_error_details(stage=stage, error=error, error_type='error', semantic_warnings=self.accumulated.get('semantic_warnings')), 'total_time_ms': total_ms, 'benchmarking': self.accumulated.get('benchmarking')})}\n\n"

    def stage_done_payload(
        self, stage_name: str, elapsed: float, metrics: dict
    ) -> None:
        """Persist stage and accumulate metrics (used by the orchestrator)."""
        self.accumulated[stage_name] = metrics
        self.persist_stage(
            stage_name,
            {
                "status": "done",
                "elapsed_ms": round(elapsed, 1),
                "metrics": metrics,
            },
        )

    # -- private helpers ------------------------------------------------------

    def _build_lineage(
        self,
        status: str,
        error: str | None,
        error_stage: str | None,
        total_ms: float | None,
    ) -> dict:
        return _build_lineage_payload(
            self.run_id,
            self.wf,
            self.accumulated,
            status,
            error,
            error_stage,
            total_ms,
        )


def _direct_eval_metrics(result_dict: dict) -> Dict[str, Any]:
    """Flatten bridge_evaluate() output into `designer_interactive` metrics."""
    return {
        "total_time_ms": safe_float(result_dict.get("total_time_ms")),
        "compile_time_ms": safe_float(result_dict.get("compile_time_ms")),
        "forward_time_ms": safe_float(result_dict.get("forward_ms")),
        "backward_time_ms": safe_float(result_dict.get("backward_ms")),
        "peak_memory_mb": safe_float(result_dict.get("peak_memory_mb")),
        "native_coverage": safe_float(result_dict.get("native_coverage")),
        "total_flops_per_token": safe_float(result_dict.get("total_flops_per_token")),
        "total_params": safe_float(result_dict.get("param_count")),
    }


# ── Evaluate Endpoints ────────────────────────────────────────────────


@router.post("/workflows/evaluate")
def evaluate_workflow_via_bridge(req: RunWorkflowRequest) -> Dict[str, Any]:
    """Evaluate a workflow through the research pipeline (sandbox + fingerprint + novelty).

    Returns full result dict including a `run_id` for later retrieval via
    GET /api/v1/eval/runs/{run_id}.
    """
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    wf = req.workflow.model_dump()
    semantic_warnings = _collect_workflow_semantic_warnings(wf)
    budget = req.budget
    result = bridge_evaluate(
        wf,
        model_dim=budget.get("model_dim", MODEL_DIM),
        vocab_size=budget.get("vocab_size", VOCAB_SIZE),
        device=budget.get("device", "cpu"),
        run_fingerprint=budget.get("run_fingerprint", True),
        run_novelty=budget.get("run_novelty", True),
        batch_size=budget.get("batch_size", 2),
        seq_len=budget.get("seq_len", 128),
    )
    result_dict = result.to_dict()
    result_dict.setdefault("result_cohort", "designer")
    result_dict.setdefault("trust_label", "exploratory")
    result_dict.setdefault("comparability_label", "partial")
    result_dict.setdefault("evaluation_protocol_version", "designer_bridge_v1")
    result_dict["benchmarking"] = build_benchmark_analysis(
        result_dict,
        external_observed=budget.get("benchmark_observed"),
    )
    run_id = f"eval_{uuid4().hex[:12]}"
    result_dict["run_id"] = run_id
    result_dict["semantic_warnings"] = semantic_warnings
    result_dict["semantic_warning_count"] = len(semantic_warnings)
    if result_dict.get("status") != "success":
        result_dict["error_details"] = _build_error_details(
            stage=result_dict.get("error_stage"),
            error=result_dict.get("error"),
            error_type=result_dict.get("status"),
            semantic_warnings=semantic_warnings,
        )
    perf = _build_designer_perf_bundle(
        run_id=run_id,
        workflow_id=wf.get("workflow_id"),
        metrics=_direct_eval_metrics(result_dict),
        status=result_dict.get("status", "unknown"),
        workload="workflow_evaluation_direct",
        slug=run_id,
    )
    result_dict["perf_contract"] = perf["perf_contract"]
    result_dict["perf_artifact_path"] = perf["perf_artifact_path"]
    result_dict["perf_budget_gate"] = perf["perf_budget_gate"]

    created_at = _utc_now()

    _store_run(
        run_id,
        {
            "run_id": run_id,
            "workflow_id": wf.get("workflow_id"),
            "status": result_dict.get("status", "unknown"),
            "created_at": created_at,
            "total_time_ms": result_dict.get("total_time_ms"),
            "budget": budget,
            "stages": {},
            "result": result_dict,
            "semantic_warnings": semantic_warnings,
            "error_details": result_dict.get("error_details"),
            "perf_contract": perf["perf_contract"],
            "perf_artifact_path": perf["perf_artifact_path"],
            "perf_budget_gate": perf["perf_budget_gate"],
        },
    )

    lineage_payload = {
        "run_id": run_id,
        "workflow_id": wf.get("workflow_id"),
        "workflow_version": wf.get("version") or dig(wf, "metadata", "version"),
        "graph_fingerprint": result_dict.get("graph_fingerprint"),
        "status": result_dict.get("status", "unknown"),
        "source": "aria_designer",
        "total_time_ms": result_dict.get("total_time_ms"),
        "metrics": {
            "sandbox_passed": result_dict.get("sandbox_passed"),
            "overall_novelty": result_dict.get("overall_novelty"),
            "efficiency_score": result_dict.get("efficiency_score"),
            "trust_label": result_dict.get("trust_label"),
            "comparability_label": result_dict.get("comparability_label"),
            "evaluation_protocol_version": result_dict.get(
                "evaluation_protocol_version"
            ),
            "benchmark_target_score": dig(
                result_dict, "benchmarking", "summary", "score"
            ),
        },
        "payload": result_dict,
        "created_at": _time_mod.time(),
    }
    result_dict["lineage_sync"] = {
        "attempted": settings.LINEAGE_SYNC_ENABLED,
        "synced": _sync_lineage_to_research(lineage_payload)
        if settings.LINEAGE_SYNC_ENABLED
        else False,
    }
    return result_dict


def _parse_eval_budget(budget: Dict[str, Any]) -> Dict[str, Any]:
    """Extract eval budget params with defaults."""
    return {
        "model_dim": budget.get("model_dim", MODEL_DIM),
        "vocab_size": budget.get("vocab_size", VOCAB_SIZE),
        "device": budget.get("device", "cpu"),
        "batch_size": budget.get("batch_size", 2),
        "seq_len": budget.get("seq_len", 128),
        "run_fingerprint": budget.get("run_fingerprint", True),
        "run_novelty": budget.get("run_novelty", True),
    }


@router.post("/workflows/evaluate/stream")
async def evaluate_workflow_stream(req: RunWorkflowRequest):
    """Stream evaluation results via SSE as each pipeline stage completes."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")

    wf = req.workflow.model_dump()
    semantic_warnings = _collect_workflow_semantic_warnings(wf)
    budget = req.budget
    bp = _parse_eval_budget(budget)
    run_id = f"eval_{uuid4().hex[:12]}"
    ctx = _EvalRunContext(run_id, wf, budget)

    async def event_stream():
        acc = ctx.accumulated
        _store_run(
            run_id,
            {
                "run_id": run_id,
                "workflow_id": wf.get("workflow_id"),
                "status": "running",
                "created_at": _utc_now(),
                "total_time_ms": None,
                "budget": budget,
                "stages": {},
                "semantic_warnings": semantic_warnings,
                "result": None,
            },
        )
        yield f"event: run_id\ndata: {_sse_json({'run_id': run_id})}\n\n"
        if semantic_warnings:
            acc["semantic_warnings"] = semantic_warnings
            yield f"event: semantic_warnings\ndata: {_sse_json({'count': len(semantic_warnings), 'warnings': semantic_warnings})}\n\n"

        # --- Stage 1: conversion (fatal) ---
        yield _sse_stage("conversion", "running")
        t0 = _time_mod.monotonic()
        try:
            result = await _stage_conversion(wf, bp["model_dim"])
            graph, cg_to_aria = result["graph"], result["cg_to_aria"]
            ctx.stage_done_payload(
                "conversion", (_time_mod.monotonic() - t0) * 1000, result["metrics"]
            )
            yield _sse_stage(
                "conversion",
                "done",
                (_time_mod.monotonic() - t0) * 1000,
                result["metrics"],
            )
        except Exception as e:
            yield _sse_stage(
                "conversion",
                "error",
                (_time_mod.monotonic() - t0) * 1000,
                error=str(e),
                error_details=_build_error_details(stage="conversion", error=e),
            )
            yield ctx.fatal_done(str(e), "conversion")
            return

        # --- Stage 2: profiling (non-fatal) ---
        async for ev in _run_nonfatal_stage(
            "profiling",
            _stage_profiling(
                graph,
                cg_to_aria,
                bp["model_dim"],
                bp["batch_size"],
                bp["seq_len"],
                ctx.duplicate_work_avoided,
            ),
            ctx,
        ):
            yield ev

        # --- Stage 3: compilation (fatal) ---
        yield _sse_stage("compilation", "running")
        t0 = _time_mod.monotonic()
        try:
            result = await _stage_compilation(graph, bp["vocab_size"])
            model = result["model"]
            elapsed = (_time_mod.monotonic() - t0) * 1000
            metrics = {"compile_time_ms": round(elapsed, 1)}
            ctx.stage_done_payload("compilation", elapsed, metrics)
            yield _sse_stage("compilation", "done", elapsed, metrics)
        except Exception as e:
            yield _sse_stage(
                "compilation",
                "error",
                (_time_mod.monotonic() - t0) * 1000,
                error=str(e),
                error_details=_build_error_details(stage="compilation", error=e),
            )
            yield ctx.fatal_done(str(e), "compilation")
            return

        # --- Stage 4: sandbox (fatal on exception, early-exit on fail) ---
        yield _sse_stage("sandbox", "running")
        t0 = _time_mod.monotonic()
        try:
            result = await _stage_sandbox(
                model, bp["batch_size"], bp["seq_len"], bp["vocab_size"], bp["device"]
            )
            sandbox = result["sandbox"]
            ctx.stage_done_payload(
                "sandbox", (_time_mod.monotonic() - t0) * 1000, result["metrics"]
            )
            yield _sse_stage(
                "sandbox",
                "done",
                (_time_mod.monotonic() - t0) * 1000,
                result["metrics"],
            )
        except Exception as e:
            yield _sse_stage(
                "sandbox",
                "error",
                (_time_mod.monotonic() - t0) * 1000,
                error=str(e),
                error_details=_build_error_details(stage="sandbox", error=e),
            )
            yield ctx.fatal_done(str(e), "sandbox")
            return

        # --- Stage 4.5: routing (non-fatal, special "skipped" status) ---
        if bridge_analyze_routing:
            yield _sse_stage("routing", "running")
            t0 = _time_mod.monotonic()
            try:
                rt_result = await _stage_routing(model, graph, cg_to_aria)
                ctx.stage_done_payload(
                    "routing", (_time_mod.monotonic() - t0) * 1000, rt_result["metrics"]
                )
                yield _sse_stage(
                    "routing",
                    "done",
                    (_time_mod.monotonic() - t0) * 1000,
                    rt_result["metrics"],
                )
            except Exception as e:
                logger.error(f"Routing analysis failed: {e}")
                yield _sse_stage(
                    "routing",
                    "skipped",
                    (_time_mod.monotonic() - t0) * 1000,
                    error=str(e),
                    error_details=_build_error_details(stage="routing", error=e),
                )

        # Early exit if sandbox ran but model failed validation
        if not sandbox.passed:
            total_ms = round((_time_mod.monotonic() - ctx.total_t0) * 1000, 1)
            ctx.attach_benchmarking()
            sb_err = getattr(sandbox, "error", "sandbox failed")
            ctx.persist_done(
                "failed_sandbox", error=sb_err, error_stage="sandbox", total_ms=total_ms
            )
            yield f"event: done\ndata: {_sse_json({'status': 'failed_sandbox', 'error': sb_err, 'error_details': _build_error_details(stage='sandbox', error=sb_err, error_type='failed_sandbox', semantic_warnings=acc.get('semantic_warnings')), 'total_time_ms': total_ms, 'result': acc, 'benchmarking': acc.get('benchmarking')})}\n\n"
            return

        # --- Stages 5-7: compression, fingerprint, novelty (non-fatal) ---
        async for ev in _run_nonfatal_stage(
            "compression",
            _stage_compression(
                model,
                graph,
                bp["vocab_size"],
                bp["device"],
                bp["batch_size"],
                bp["seq_len"],
            ),
            ctx,
        ):
            yield ev
        async for ev in _run_nonfatal_stage(
            "fingerprint",
            _stage_fingerprint(
                model,
                bp["run_fingerprint"],
                bp["seq_len"],
                bp["model_dim"],
                bp["vocab_size"],
                bp["device"],
            ),
            ctx,
        ):
            yield ev
        fp_obj = dig(ctx._last_stage_result, "fp_obj")
        async for ev in _run_nonfatal_stage(
            "novelty",
            _stage_novelty(
                graph,
                fp_obj,
                bp["run_novelty"],
            ),
            ctx,
        ):
            yield ev

        # --- Done ---
        total_ms = round((_time_mod.monotonic() - ctx.total_t0) * 1000, 1)
        ctx.attach_benchmarking()
        acc["graph_fingerprint"] = dig(acc, "conversion", "graph_fingerprint")
        acc["discovery_url"] = _discovery_url_for_fingerprint(
            acc.get("graph_fingerprint")
        )
        acc["composite_score"] = _compute_eval_composite_score(acc)
        ctx.persist_done("success", total_ms=total_ms)
        yield f"event: done\ndata: {_sse_json({'status': 'success', 'total_time_ms': total_ms, 'result': acc, 'benchmarking': acc.get('benchmarking'), 'perf_budget_gate': acc.get('perf_budget_gate'), 'perf_artifact_path': acc.get('perf_artifact_path'), 'graph_fingerprint': acc.get('graph_fingerprint'), 'discovery_url': acc.get('discovery_url'), 'composite_score': acc.get('composite_score')})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Eval Observability Endpoints ──────────────────────────────────────


@router.get("/eval/runs")
def list_eval_runs(
    status: Optional[str] = Query(
        None, description="Filter by status: running, success, error, failed_sandbox"
    ),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """List recent evaluation runs.

    Returns summary metadata (run_id, workflow_id, status, timing) for
    each run. Use GET /api/v1/eval/runs/{run_id} for full details.
    """
    runs = _list_runs()
    if status:
        runs = [r for r in runs if r.get("status") == status]
    return runs[:limit]


@router.get("/eval/runs/{run_id}")
def get_eval_run(run_id: str) -> Dict[str, Any]:
    """Get full evaluation results for a run.

    Includes all stage metrics, per-op profiles, fingerprint, novelty,
    and the original budget parameters. Available during and after the run.
    """
    run = _require_run(run_id)
    return {k: v for k, v in run.items() if not k.startswith("_")}


@router.get("/eval/runs/{run_id}/stages")
def get_eval_run_stages(run_id: str) -> Dict[str, Any]:
    """Get stage-by-stage breakdown for a run.

    Each stage (conversion, profiling, compilation, sandbox, compression,
    fingerprint, novelty) includes status and metrics. Stages not yet reached are absent.
    """
    run = _require_run(run_id)
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "stages": run.get("stages", {}),
    }


def _get_run_stage_metrics(run_id: str, stage_name: str) -> Dict[str, Any]:
    """Extract validated stage metrics from a completed run stage."""
    run = _require_run(run_id)
    stage = run.get("stages", {}).get(stage_name, {})
    if not stage or stage.get("status") != "done":
        raise HTTPException(
            status_code=404,
            detail=f"{stage_name.capitalize()} data not available for this run",
        )
    return {"run_id": run_id, **stage.get("metrics", {})}


@router.get("/eval/runs/{run_id}/profile")
def get_eval_run_profile(run_id: str) -> Dict[str, Any]:
    """Get per-op profiling data (FLOPs, params, memory, bottlenecks, kernel coverage)."""
    return _get_run_stage_metrics(run_id, "profiling")


@router.get("/eval/runs/{run_id}/fingerprint")
def get_eval_run_fingerprint(run_id: str) -> Dict[str, Any]:
    """Get behavioral fingerprint (CKA similarity, locality, sparsity, isotropy)."""
    return _get_run_stage_metrics(run_id, "fingerprint")


@router.get("/eval/runs/{run_id}/novelty")
def get_eval_run_novelty(run_id: str) -> Dict[str, Any]:
    """Get novelty scores (structural, behavioral, overall + nearest known arch)."""
    return _get_run_stage_metrics(run_id, "novelty")


@router.get("/eval/runs/{run_id}/compression")
def get_eval_run_compression(run_id: str) -> Dict[str, Any]:
    """Get compression & efficiency analysis (pruning, sparsity, ratios)."""
    return _get_run_stage_metrics(run_id, "compression")


@router.get("/eval/runs/{run_id}/sandbox")
def get_eval_run_sandbox(run_id: str) -> Dict[str, Any]:
    """Get sandbox evaluation results (timing, params, memory, stability)."""
    return _get_run_stage_metrics(run_id, "sandbox")


@router.get("/eval/runs/{run_id}/benchmarking")
def get_eval_run_benchmarking(run_id: str) -> Dict[str, Any]:
    """Get benchmark target comparison for a run.

    Includes target table, on/off-target summary, and scaling projection.
    """
    run = _require_run(run_id)
    if run.get("status") == "running":
        raise HTTPException(
            status_code=409, detail="Benchmarking not final while run is still running"
        )
    result_payload = run.get("result") or {}
    benchmarking = result_payload.get("benchmarking")
    if not benchmarking:
        stage_metrics = {
            name: (stage.get("metrics") if isinstance(stage, dict) else None)
            for name, stage in (run.get("stages") or {}).items()
            if isinstance(stage, dict)
        }
        benchmarking = build_benchmark_analysis(stage_metrics)
    return {
        "run_id": run_id,
        **benchmarking,
    }


@router.get("/eval/runs/{run_id}/perf")
def get_eval_run_perf(run_id: str) -> Dict[str, Any]:
    """Get normalized performance contract, artifact path, and budget verdict for a run."""
    run = _require_run(run_id)
    perf_contract = run.get("perf_contract")
    if not perf_contract:
        raise HTTPException(
            status_code=404, detail="Performance contract not available for this run"
        )
    return {
        "run_id": run_id,
        "perf_contract": perf_contract,
        "perf_artifact_path": run.get("perf_artifact_path"),
        "perf_budget_gate": run.get("perf_budget_gate"),
    }
