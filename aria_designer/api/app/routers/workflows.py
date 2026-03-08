from __future__ import annotations

import json
import logging
import os
import threading
import time as _time_mod
try:
    import requests
except ImportError:
    requests = None

import asyncio
from collections import deque
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from .. import database as db
from ..config import settings
from ..benchmark_targets import benchmark_target_catalog, build_benchmark_analysis
from ..models import (
    CompileWorkflowRequest,
    RunWorkflowRequest,
    ValidateWorkflowRequest,
    ValidateWorkflowResponse,
    ValidationIssue,
    WorkflowGraphModel,
    PatchOpModel,
    utc_now_iso as _utc_now
)

from ..shared_api import (
    _optional_import,
    _store_run,
    _update_run,
    _get_run,
    _list_runs,
    _require_workflow,
    _require_run,
    _sync_lineage_to_research,
    _auto_promote_workflow_to_research,
)

(KernelDispatcher, runtime_compile,
 find_unsupported_edge_dtype_pairings) = _optional_import(
    "runtime.dispatch", ["KernelDispatcher"]) + _optional_import(
    "runtime.compiler", ["compile_workflow"]) + _optional_import(
    "runtime.port_dtypes", ["find_unsupported_edge_dtype_pairings"])

(bridge_evaluate, bridge_validate, bridge_estimate, bridge_list_primitives,
 bridge_analyze_compression, bridge_analyze_routing, bridge_component_capability) = _optional_import(
    "runtime.bridge", [
        "evaluate_workflow", "validate_workflow_graph", "estimate_performance",
        "list_available_primitives", "analyze_compression",
        "bridge_analyze_routing", "get_component_execution_capability",
    ])
HAS_BRIDGE = bridge_evaluate is not None

(bridge_profile,) = _optional_import("runtime.profiler", ["profile_workflow"])
HAS_PROFILER = bridge_profile is not None

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RESEARCH_ROOT = _PROJECT_ROOT / "research"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.get("/")
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        from research.synthesis.serializer import graph_to_json
        from research.scientist.notebook import LabNotebook
    except ImportError:
        return None

    notebook_path = _PROJECT_ROOT / "research" / "lab_notebook.db"
    if not notebook_path.exists():
        return None

    try:
        model_dim = int((workflow.get("metadata") or {}).get("model_dim") or 256)
        graph, _ = _w2g(workflow, model_dim=model_dim, return_id_map=True)
        fingerprint = graph.fingerprint()
        graph_json = graph_to_json(graph)
        
        meta = workflow.get("metadata") or {}
        loss_ratio = float(meta.get("loss_ratio") or 1.0)
        novelty_score = float(meta.get("novelty_score") or 0.0) if meta.get("novelty_score") is not None else None

        param_count = None
        try:
            from research.synthesis.compiler import compile_model
            model = compile_model(graph)
            param_count = sum(p.numel() for p in model.parameters())
        except Exception:
            pass

        nb = LabNotebook(str(notebook_path))
        exp_id = "designer_edits"
        exp_exists = nb.conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        if not exp_exists:
            now = _time_mod.time()
            nb.conn.execute(
                """INSERT INTO experiments
                   (experiment_id, timestamp, experiment_type, status, config_json, started_at, completed_at)
                   VALUES (?, ?, 'designer', 'completed', '{}', ?, ?)""",
                (exp_id, now, now, now),
            )
            nb.conn.commit()
        # Record result logic...
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fingerprint,
            graph_json=graph_json,
            model_source="designer_edit",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=loss_ratio < 1.0,
            loss_ratio=loss_ratio,
            novelty_score=novelty_score,
            param_count=param_count,
        )
        nb.close()
        return {"success": True, "result_id": result_id, "fingerprint": fingerprint} if result_id else None
    except Exception as exc:
        logger.warning("Local auto-promotion failed: %s", exc)
        return None

def _collect_workflow_semantic_warnings(workflow_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not (HAS_BRIDGE and bridge_component_capability):
        return []
    warnings: List[Dict[str, Any]] = []
    seen = set()
    for node in workflow_json.get("nodes", []):
        node_id = str(node.get("id") or "")
        component_type = str(node.get("component_type") or "")
        if not component_type: continue
        try:
            cap = bridge_component_capability(component_type)
        except Exception: continue
        if not cap.get("bridge_supported") or str(cap.get("semantic_fidelity")) != "approximate":
            continue
        primitive_name = cap.get("primitive_name")
        for msg in cap.get("warnings") or [cap.get("reason")]:
            key = (node_id, component_type, str(msg))
            if key not in seen:
                seen.add(key)
                warnings.append({
                    "node_id": node_id, "component_type": component_type,
                    "mapping_kind": cap.get("mapping_kind"),
                    "primitive_name": primitive_name, "message": str(msg),
                })
    return warnings

def _validate_fallback_cycles(node_ids, workflow, issues):
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in workflow.edges:
        if edge.source in adj:
            adj[edge.source].append(edge.target)
    visited, in_stack = set(), set()
    def has_cycle(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in in_stack: return True
            if neighbor not in visited and has_cycle(neighbor): return True
        in_stack.discard(node)
        return False
    for nid in node_ids:
        if nid not in visited and has_cycle(nid):
            issues.append(ValidationIssue(
                severity="error", code="cycle_detected",
                message="Workflow contains a cycle.",
            ))
            break

# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("")
def list_workflows() -> List[Dict[str, Any]]:
    return db.list_workflows()

@router.get("/{workflow_id}")
def get_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = _require_workflow(workflow_id)
    wf["graph"] = json.loads(wf.pop("graph_json"))
    return wf

@router.post("/validate", response_model=ValidateWorkflowResponse)
def validate_workflow(req: ValidateWorkflowRequest) -> ValidateWorkflowResponse:
    workflow = req.workflow
    issues: List[ValidationIssue] = []
    node_ids = {node.id for node in workflow.nodes}
    
    if len(node_ids) != len(workflow.nodes):
        issues.append(ValidationIssue(severity="error", code="duplicate_node_id", message="Duplicate node ids."))

    for edge in workflow.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            issues.append(ValidationIssue(severity="error", code="dangling_edge", message=f"Edge {edge.id} references missing node."))

    comp_cache = {}
    for node in workflow.nodes:
        comp = db.get_component(node.component_type)
        if comp is None:
            issues.append(ValidationIssue(severity="warning", code="unknown_component", message=f"Node {node.id}: unknown component '{node.component_type}'."))
        else:
            comp_cache[node.id] = comp

    output_types = {"output", "output_head", "graph_output", "io/output"}
    output_node_ids = {
        node.id
        for node in workflow.nodes
        if (node.component_type or "").split("/")[-1] in output_types
    }
    if output_node_ids:
        reverse_adj = {node.id: [] for node in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in node_ids and edge.target in node_ids:
                reverse_adj[edge.target].append(edge.source)

        reachable = set()
        queue = __import__("collections").deque(output_node_ids)
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            for source in reverse_adj.get(current, []):
                if source not in reachable:
                    queue.append(source)

        for node in workflow.nodes:
            if node.id not in reachable:
                issues.append(ValidationIssue(
                    node_id=node.id,
                    severity="error",
                    code="dead_branch",
                    message="Dead branch detected. Node does not connect to the final output."
                ))

    if find_unsupported_edge_dtype_pairings:
        dtype_issues = find_unsupported_edge_dtype_pairings(workflow.model_dump(), lambda ct: db.get_component(ct))
        for issue in dtype_issues:
            issues.append(ValidationIssue(severity="error", code="unsupported_edge_dtype_pairing", message=issue["message"], edge_id=issue.get("edge_id")))

    if KernelDispatcher:
        try:
            dispatcher = KernelDispatcher()
            node_ids_list = [node.id for node in workflow.nodes]
            node_to_idx = {nid: i for i, nid in enumerate(node_ids_list)}
            c_edges = [(node_to_idx[e.source], node_to_idx[e.target], 0, 0) for e in workflow.edges if e.source in node_to_idx and e.target in node_to_idx]
            res = dispatcher.validate_graph(node_ids_list, c_edges)
            if not res['valid']:
                issues.append(ValidationIssue(severity="error", code="cycle_detected", message=f"Native validation failed: {res['error']}"))
        except Exception:
            _validate_fallback_cycles(node_ids, workflow, issues)
    else:
        _validate_fallback_cycles(node_ids, workflow, issues)

    return ValidateWorkflowResponse(valid=not any(i.severity == "error" for i in issues), issues=issues)

@router.post("/compile")
def compile_workflow_endpoint(req: CompileWorkflowRequest) -> Dict[str, Any]:
    semantic_warnings = _collect_workflow_semantic_warnings(req.workflow.model_dump())
    if runtime_compile is None:
        return {"compiled": False, "error": "Compiler not available", "workflow_id": req.workflow.workflow_id, "semantic_warnings": semantic_warnings, "semantic_warning_count": len(semantic_warnings)}
    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'components'))
        model = runtime_compile(req.workflow.model_dump(), components_dir)
        return {"compiled": True, "target": req.target, "workflow_id": req.workflow.workflow_id, "node_count": len(req.workflow.nodes), "submodule_count": len(model.submodules), "semantic_warnings": semantic_warnings, "semantic_warning_count": len(semantic_warnings)}
    except Exception as e:
        return {"compiled": False, "error": str(e), "workflow_id": req.workflow.workflow_id, "semantic_warnings": semantic_warnings, "semantic_warning_count": len(semantic_warnings)}

@router.put("/{workflow_id}")
def save_workflow(workflow_id: str, workflow: WorkflowGraphModel) -> Dict[str, Any]:
    now = _utc_now()
    wf_dict = workflow.model_dump()
    existing = None
    old_fingerprint = None
    try:
        existing = db.get_workflow(workflow_id)
        if existing and existing.get("graph_json"):
            existing_graph = json.loads(existing["graph_json"])
            old_fingerprint = (existing_graph.get("metadata") or {}).get("graph_fingerprint")
    except Exception:
        existing = None
        old_fingerprint = None

    fingerprint = None
    if HAS_BRIDGE:
        try:
            from runtime.bridge import workflow_to_graph as _w2g
            model_dim = wf_dict.get("metadata", {}).get("model_dim", 256)
            graph, _ = _w2g(wf_dict, model_dim)
            fingerprint = graph.fingerprint()
            meta = wf_dict.setdefault("metadata", {})
            meta["graph_fingerprint"] = fingerprint
            if old_fingerprint and fingerprint and old_fingerprint != fingerprint:
                meta["parent_fingerprint"] = old_fingerprint
        except Exception:
            pass

    version = db.save_workflow(
        workflow_id=workflow_id,
        name=workflow.name,
        graph_json=json.dumps(wf_dict),
        author="user",
        parent_id=(f"{workflow_id}@v{existing.get('version', 0)}" if existing else None),
        created_at=now,
        updated_at=now,
    )

    if settings.LINEAGE_SYNC_ENABLED:
        try:
            lineage_payload = {
                "run_id": f"save_{uuid4().hex[:10]}",
                "workflow_id": workflow_id,
                "workflow_version": version,
                "graph_fingerprint": fingerprint,
                "status": "saved",
                "source": "aria_designer",
                "total_time_ms": 0,
                "metrics": {
                    "node_count": len(wf_dict.get("nodes", [])),
                    "edge_count": len(wf_dict.get("edges", [])),
                },
                "payload": wf_dict,
                "created_at": _time_mod.time(),
            }
            _sync_lineage_to_research(lineage_payload)
        except Exception:
            logger.debug("Failed to sync workflow lineage", exc_info=True)

    fingerprint_changed = bool(fingerprint and old_fingerprint and fingerprint != old_fingerprint)
    if fingerprint:
        _auto_promote_workflow_to_research(wf_dict)
    return {
        "workflow_id": workflow_id,
        "version": version,
        "saved_at": now,
        "fingerprint": fingerprint,
        "fingerprint_changed": fingerprint_changed,
        "parent_fingerprint": old_fingerprint if fingerprint_changed else None,
    }

def _try_fetch_original_graph(metadata: dict, model_dim: int):
    """Try to fetch the original ComputationGraph from research notebook."""
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
        res = nb.get_result(result_id)
        nb.close()

        if not res or not res.get("graph_json"):
            return None

        graph = graph_from_json(res["graph_json"])
        
        cg_to_aria = {}
        for cg_id in graph.topological_order():
            node = graph.nodes[cg_id]
            if node.is_input:
                cg_to_aria[cg_id] = f"input_{cg_id}"
            else:
                cg_to_aria[cg_id] = f"op_{cg_id}_{node.op_name}"

        return (graph, cg_to_aria)
    except Exception:
        logger.debug("Failed to fetch original graph for result_id=%s", result_id, exc_info=True)
        return None

@router.post("/evaluate/stream")
async def evaluate_workflow_stream(req: RunWorkflowRequest):
    """Stream evaluation results via SSE as each pipeline stage completes."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")

    wf = req.workflow.model_dump()
    semantic_warnings = _collect_workflow_semantic_warnings(wf)
    budget = req.budget
    model_dim = budget.get("model_dim", 256)
    vocab_size = budget.get("vocab_size", 32000)
    device = budget.get("device", "cpu")
    batch_size = budget.get("batch_size", 2)
    seq_len = budget.get("seq_len", 128)
    run_fingerprint = budget.get("run_fingerprint", True)
    run_novelty = budget.get("run_novelty", True)

    run_id = f"eval_{uuid4().hex[:12]}"

    def _json(obj):
        return json.dumps(obj, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    async def event_stream():
        total_t0 = _time_mod.monotonic()
        accumulated = {}
        lineage_synced = False

        created_at = _utc_now()
        _store_run(run_id, {
            "run_id": run_id,
            "workflow_id": wf.get("workflow_id"),
            "status": "running",
            "created_at": created_at,
            "total_time_ms": None,
            "budget": budget,
            "stages": {},
            "result": None,
        })

        def _persist_stage(stage_name, stage_data):
            _update_run(run_id, {
                "stages": {**(_get_run(run_id) or {}).get("stages", {}), stage_name: stage_data},
            })

        def _persist_done(status, error=None, error_stage=None, total_ms=None):
            nonlocal lineage_synced
            _update_run(run_id, {
                "status": status,
                "error": error,
                "error_stage": error_stage,
                "total_time_ms": total_ms,
                "result": accumulated,
                "completed_at": _utc_now(),
            })
            if not lineage_synced and settings.LINEAGE_SYNC_ENABLED:
                lineage_payload = {
                    "run_id": run_id,
                    "workflow_id": wf.get("workflow_id"),
                    "workflow_version": wf.get("version") or (wf.get("metadata") or {}).get("version"),
                    "graph_fingerprint": (accumulated.get("conversion") or {}).get("graph_fingerprint"),
                    "status": status,
                    "source": "aria_designer",
                    "total_time_ms": total_ms,
                    "metrics": {
                        "sandbox_passed": (accumulated.get("sandbox") or {}).get("passed"),
                        "overall_novelty": (accumulated.get("novelty") or {}).get("overall_novelty"),
                        "efficiency_score": (accumulated.get("compression") or {}).get("efficiency_score"),
                        "benchmark_target_score": (accumulated.get("benchmarking") or {}).get("summary", {}).get("score"),
                    },
                    "payload": {"error": error, "error_stage": error_stage, "result": accumulated},
                    "created_at": _time_mod.time(),
                }
                lineage_synced = _sync_lineage_to_research(lineage_payload)

        def _attach_benchmarking():
            accumulated["benchmarking"] = build_benchmark_analysis(
                accumulated,
                external_observed=budget.get("benchmark_observed"),
            )

        yield f"event: run_id\ndata: {_json({'run_id': run_id})}\n\n"
        if semantic_warnings:
            accumulated["semantic_warnings"] = semantic_warnings
            yield f"event: semantic_warnings\ndata: {_json({'count': len(semantic_warnings), 'warnings': semantic_warnings})}\n\n"

        # Stage 1: conversion
        yield f"event: stage\ndata: {_json({'stage': 'conversion', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            metadata = wf.get("metadata", {})
            original = await asyncio.to_thread(_try_fetch_original_graph, metadata, model_dim)
            if original is not None:
                graph, cg_to_aria = original
                used_original = True
            else:
                from runtime.bridge import workflow_to_graph as _w2g
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
            accumulated["conversion"] = metrics
            _persist_stage("conversion", {"status": "done", "metrics": metrics})
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'conversion', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'conversion', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"
            total_ms = (_time_mod.monotonic() - total_t0) * 1000
            _attach_benchmarking()
            _persist_done("error", error=str(e), error_stage="conversion", total_ms=round(total_ms, 1))
            yield f"event: done\ndata: {_json({'status': 'error', 'error': str(e), 'error_stage': 'conversion', 'total_time_ms': round(total_ms, 1), 'benchmarking': accumulated.get('benchmarking')})}\n\n"
            return

        # Stage 2: profiling
        yield f"event: stage\ndata: {_json({'stage': 'profiling', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            if HAS_PROFILER:
                report = await asyncio.to_thread(
                    bridge_profile, wf, model_dim=model_dim, device=device,
                    runtime=False, vocab_size=vocab_size, batch_size=batch_size, seq_len=seq_len,
                )
                report_dict = report.to_dict()
                mapped_profiles = []
                for op in report_dict.get("op_profiles", []):
                    aria_id = cg_to_aria.get(op.get("node_id"))
                    mapped_profiles.append({**op, "aria_node_id": aria_id})
                metrics = {
                    "total_flops_per_token": report_dict.get("total_flops_per_token", 0),
                    "total_params": report_dict.get("total_params", 0),
                    "total_memory_bytes": report_dict.get("total_memory_bytes", 0),
                    "flops_by_category": report_dict.get("flops_by_category", {}),
                    "bottleneck_ops": report_dict.get("bottleneck_ops", []),
                    "native_coverage": report_dict.get("native_coverage", 0),
                    "op_profiles": mapped_profiles,
                }
            else:
                metrics = {"skipped": True, "reason": "profiler not available"}
            accumulated["profiling"] = metrics
            _persist_stage("profiling", {"status": "done", "metrics": metrics})
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'profiling', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'profiling', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"

        # Stage 3: compilation
        yield f"event: stage\ndata: {_json({'stage': 'compilation', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            from research.synthesis.compiler import compile_model
            model = await asyncio.to_thread(compile_model, [graph], vocab_size=vocab_size)
            elapsed = (_time_mod.monotonic() - t0) * 1000
            metrics = {"compile_time_ms": round(elapsed, 1)}
            accumulated["compilation"] = metrics
            _persist_stage("compilation", {"status": "done", "metrics": metrics})
            yield f"event: stage\ndata: {_json({'stage': 'compilation', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'compilation', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"
            total_ms = (_time_mod.monotonic() - total_t0) * 1000
            _attach_benchmarking()
            _persist_done("error", error=str(e), error_stage="compilation", total_ms=round(total_ms, 1))
            yield f"event: done\ndata: {_json({'status': 'error', 'error': str(e), 'error_stage': 'compilation', 'total_time_ms': round(total_ms, 1), 'benchmarking': accumulated.get('benchmarking')})}\n\n"
            return

        # Stage 4: sandbox
        yield f"event: stage\ndata: {_json({'stage': 'sandbox', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            from research.eval.sandbox import safe_eval
            sandbox = await asyncio.to_thread(
                safe_eval, model, batch_size=batch_size, seq_len=seq_len,
                vocab_size=vocab_size, device=device,
            )
            elapsed = (_time_mod.monotonic() - t0) * 1000
            metrics = {
                "passed": bool(sandbox.passed),
                "forward_ms": float(getattr(sandbox, "forward_time_ms", 0)),
                "backward_ms": float(getattr(sandbox, "backward_time_ms", 0)),
                "param_count": int(getattr(sandbox, "param_count", 0)),
                "peak_memory_mb": float(getattr(sandbox, "peak_memory_mb", 0)),
                "grad_norm": float(getattr(sandbox, "grad_norm", 0)),
                "stability_score": float(getattr(sandbox, "stability_score", 0)),
                "native_abi_probe": getattr(sandbox, "native_abi_probe", None),
            }
            accumulated["sandbox"] = metrics
            _persist_stage("sandbox", {"status": "done", "metrics": metrics})
            yield f"event: stage\ndata: {_json({'stage': 'sandbox', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
            
            if not sandbox.passed:
                total_ms = (_time_mod.monotonic() - total_t0) * 1000
                _attach_benchmarking()
                _persist_done("failed_sandbox", error=getattr(sandbox, 'error', 'sandbox failed'), error_stage="sandbox", total_ms=round(total_ms, 1))
                yield f"event: done\ndata: {_json({'status': 'failed_sandbox', 'error': getattr(sandbox, 'error', 'sandbox failed'), 'total_time_ms': round(total_ms, 1), 'result': accumulated, 'benchmarking': accumulated.get('benchmarking')})}\n\n"
                return
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'sandbox', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"
            total_ms = (_time_mod.monotonic() - total_t0) * 1000
            _attach_benchmarking()
            _persist_done("error", error=str(e), error_stage="sandbox", total_ms=round(total_ms, 1))
            yield f"event: done\ndata: {_json({'status': 'error', 'error': str(e), 'error_stage': 'sandbox', 'total_time_ms': round(total_ms, 1), 'benchmarking': accumulated.get('benchmarking')})}\n\n"
            return

        # Stage 5: compression
        yield f"event: stage\ndata: {_json({'stage': 'compression', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            comp_result = await asyncio.to_thread(
                bridge_analyze_compression, model, graph,
                vocab_size=vocab_size, device=device,
                batch_size=batch_size, seq_len=min(seq_len, 64),
            )
            metrics = comp_result.to_dict()
            accumulated["compression"] = metrics
            _persist_stage("compression", {"status": "done", "metrics": metrics})
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'compression', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'compression', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"

        # Stage 6: fingerprint
        yield f"event: stage\ndata: {_json({'stage': 'fingerprint', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        fp_obj = None
        try:
            if run_fingerprint:
                from research.eval.fingerprint import compute_fingerprint
                fp_obj = await asyncio.to_thread(
                    compute_fingerprint, model, seq_len=min(seq_len, 64),
                    model_dim=model_dim, vocab_size=vocab_size, device=device,
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
            else:
                metrics = {"skipped": True}
            accumulated["fingerprint"] = metrics
            _persist_stage("fingerprint", {"status": "done", "metrics": metrics})
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'fingerprint', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'fingerprint', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"

        # Stage 7: novelty
        yield f"event: stage\ndata: {_json({'stage': 'novelty', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        try:
            if run_novelty:
                from research.eval.metrics import novelty_score
                ns = await asyncio.to_thread(novelty_score, graph, fingerprint=fp_obj)
                metrics = {
                    "structural_novelty": float(ns.structural_novelty),
                    "behavioral_novelty": float(ns.behavioral_novelty),
                    "overall_novelty": float(ns.overall_novelty),
                    "most_similar_to": getattr(ns, "most_similar_to", ""),
                }
            else:
                metrics = {"skipped": True}
            accumulated["novelty"] = metrics
            _persist_stage("novelty", {"status": "done", "metrics": metrics})
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'novelty', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
        except Exception as e:
            elapsed = (_time_mod.monotonic() - t0) * 1000
            yield f"event: stage\ndata: {_json({'stage': 'novelty', 'status': 'error', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"

        total_ms = (_time_mod.monotonic() - total_t0) * 1000
        _attach_benchmarking()
        _persist_done("success", total_ms=round(total_ms, 1))
        yield f"event: done\ndata: {_json({'status': 'success', 'total_time_ms': round(total_ms, 1), 'result': accumulated, 'benchmarking': accumulated.get('benchmarking')})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/eval/runs")
def list_eval_runs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    runs = _list_runs()
    if status:
        runs = [r for r in runs if r.get("status") == status]
    return runs[:limit]

@router.get("/eval/runs/{run_id}")
def get_eval_run(run_id: str) -> Dict[str, Any]:
    run = _require_run(run_id)
    return {k: v for k, v in run.items() if not k.startswith("_")}

@router.get("/eval/runs/{run_id}/stages")
def get_eval_run_stages(run_id: str) -> Dict[str, Any]:
    run = _require_run(run_id)
    return {"run_id": run_id, "status": run.get("status"), "stages": run.get("stages", {})}

@router.get("/eval/runs/{run_id}/profile")
def get_eval_run_profile(run_id: str) -> Dict[str, Any]:
    run = _require_run(run_id)
    profiling = run.get("stages", {}).get("profiling", {})
    if not profiling or profiling.get("status") != "done":
        raise HTTPException(status_code=404, detail="Profiling data not available")
    return {"run_id": run_id, **profiling.get("metrics", {})}

@router.get("/eval/runs/{run_id}/benchmarking")
def get_eval_run_benchmarking(run_id: str) -> Dict[str, Any]:
    run = _require_run(run_id)
    if run.get("status") == "running":
        raise HTTPException(status_code=409, detail="Benchmarking not final")
    result_payload = run.get("result") or {}
    benchmarking = result_payload.get("benchmarking")
    if not benchmarking:
        stage_metrics = {n: s.get("metrics") for n, s in (run.get("stages") or {}).items() if isinstance(s, dict)}
        benchmarking = build_benchmark_analysis(stage_metrics)
    return {"run_id": run_id, **benchmarking}

@router.get("/benchmarks/targets")
def get_benchmark_targets(run_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    payload: Dict[str, Any] = benchmark_target_catalog()
    if run_id:
        run = _require_run(run_id)
        result_payload = run.get("result") or {}
        benchmarking = result_payload.get("benchmarking")
        if not benchmarking:
            stage_metrics = {n: s.get("metrics") for n, s in (run.get("stages") or {}).items() if isinstance(s, dict)}
            benchmarking = build_benchmark_analysis(stage_metrics)
        payload["run_id"] = run_id
        payload["analysis"] = benchmarking
    return payload

@router.post("/profile")
def profile_workflow_endpoint(req: RunWorkflowRequest) -> Dict[str, Any]:
    if not HAS_PROFILER:
        raise HTTPException(status_code=501, detail="Profiler not available")
    wf = req.workflow.model_dump()
    budget = req.budget
    report = bridge_profile(wf, model_dim=budget.get("model_dim", 256), device=budget.get("device", "cpu"))
    return report.to_dict()

@router.post("/validate-graph")
def validate_workflow_graph_endpoint(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    wf = req.workflow.model_dump()
    return bridge_validate(wf, model_dim=req.workflow.metadata.get("model_dim", 256))

@router.post("/estimate")
def estimate_workflow_endpoint(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    if HAS_BRIDGE:
        wf = req.workflow.model_dump()
        model_dim = req.workflow.metadata.get("model_dim", 256)
        result = bridge_estimate(wf, model_dim=model_dim)
        result["workflow_id"] = req.workflow.workflow_id
        return result
    raise HTTPException(status_code=501, detail="Estimation requires research bridge")
