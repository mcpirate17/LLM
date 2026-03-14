from __future__ import annotations

import logging
import os
import sys
import threading
import time as _time_mod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException

from . import database as db
from .config import settings
from .models import utc_now_iso as _utc_now
from .research_signals import fetch_research_recommendation_signals

logger = logging.getLogger(__name__)

# ── Path setup (mirrors main.py) ─────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARIA_DESIGNER_ROOT = _PROJECT_ROOT / "aria_designer"
_ARIA_CORE_ROOT = _PROJECT_ROOT / "aria_core"
for _p in (_ARIA_DESIGNER_ROOT, _PROJECT_ROOT, _ARIA_CORE_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

_RESEARCH_ROOT = _PROJECT_ROOT / "research"

from research.eval.perf_budget import evaluate_perf_budget_gate
from research.perf_contract import (
    build_duplicate_work_report,
    build_perf_contract,
    emit_perf_artifact,
    list_recent_perf_artifacts,
    summarize_perf_artifacts,
)

# ── Optional Import Helper ────────────────────────────────────────────


def _optional_import(
    module: str, names: list[str], *, aliases: dict[str, str] | None = None,
) -> tuple[Any, ...]:
    """Import *names* from *module*, returning ``None`` for each on ImportError.

    Tries both the bare module path and an ``aria_designer.`` prefix so the
    same call works whether the package is installed or run from a checkout.
    """
    aliases = aliases or {}
    for candidate in (module, f"aria_designer.{module}" if module.startswith("runtime.") else None):
        if not candidate:
            continue
        try:
            mod = __import__(candidate, fromlist=names)
            return tuple(getattr(mod, n) for n in names)
        except ImportError:
            continue
    return tuple(None for _ in names)


# ── Optional runtime imports ──────────────────────────────────────────

(KernelDispatcher, runtime_compile,
 find_unsupported_edge_dtype_pairings) = _optional_import(
    "runtime.dispatch", ["KernelDispatcher"]) + _optional_import(
    "runtime.compiler", ["compile_workflow"]) + _optional_import(
    "runtime.port_dtypes", ["find_unsupported_edge_dtype_pairings"])

(export_onnx,) = _optional_import("runtime.export", ["export_onnx"])

(bridge_evaluate, bridge_validate, bridge_estimate, bridge_list_primitives,
 bridge_analyze_compression, bridge_analyze_routing, bridge_component_capability) = _optional_import(
    "runtime.bridge", [
        "evaluate_workflow", "validate_workflow_graph", "estimate_performance",
        "list_available_primitives", "analyze_compression",
        "bridge_analyze_routing", "get_component_execution_capability",
    ])
HAS_BRIDGE: bool = bridge_evaluate is not None

(bridge_profile,) = _optional_import("runtime.profiler", ["profile_workflow"])
HAS_PROFILER: bool = bridge_profile is not None

(import_survivors, import_single, graph_to_workflow) = _optional_import(
    "runtime.importer", ["import_survivors", "import_single", "graph_to_workflow"])
HAS_IMPORTER: bool = import_survivors is not None

(check_compatibility, compute_palette_constraints) = _optional_import(
    "runtime.constraints", ["check_compatibility", "compute_palette_constraints"])
HAS_CONSTRAINTS: bool = check_compatibility is not None

(extract_block, expand_block, list_builtin_blocks, BUILTIN_BLOCKS) = _optional_import(
    "runtime.subgraph", ["extract_block", "expand_block", "list_builtin_blocks", "BUILTIN_BLOCKS"])
HAS_SUBGRAPH: bool = extract_block is not None


# ── Eval Run Store ────────────────────────────────────────────────────
# In-memory store for evaluation run results, keyed by run_id.
# Each entry stores full stage-by-stage metrics so external apps (ARIA,
# dashboards, CI pipelines) can query results via REST after completion.

_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
_EVAL_RUNS_LOCK = threading.Lock()
_EVAL_RUNS_MAX = 200          # max runs kept in memory
_EVAL_RUNS_TTL_S = 3600       # evict runs older than 1 hour


def _evict_old_runs() -> None:
    """Remove expired runs.  Called under lock."""
    cutoff = _time_mod.time() - _EVAL_RUNS_TTL_S
    expired = [k for k, v in _EVAL_RUNS.items() if v.get("_created_ts", 0) < cutoff]
    for k in expired:
        del _EVAL_RUNS[k]
    # If still over capacity, drop oldest
    if len(_EVAL_RUNS) > _EVAL_RUNS_MAX:
        by_ts = sorted(_EVAL_RUNS.items(), key=lambda kv: kv[1].get("_created_ts", 0))
        for k, _ in by_ts[: len(_EVAL_RUNS) - _EVAL_RUNS_MAX]:
            del _EVAL_RUNS[k]


def _store_run(run_id: str, data: Dict[str, Any]) -> None:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        data["_created_ts"] = _time_mod.time()
        _EVAL_RUNS[run_id] = data


def _update_run(run_id: str, updates: Dict[str, Any]) -> None:
    with _EVAL_RUNS_LOCK:
        if run_id in _EVAL_RUNS:
            _EVAL_RUNS[run_id].update(updates)
            _EVAL_RUNS[run_id]["_updated_ts"] = _time_mod.time()


def _get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _EVAL_RUNS_LOCK:
        return _EVAL_RUNS.get(run_id)


def _list_runs() -> List[Dict[str, Any]]:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        out: List[Dict[str, Any]] = []
        for run_id, data in sorted(
            _EVAL_RUNS.items(),
            key=lambda kv: kv[1].get("_created_ts", 0),
            reverse=True,
        ):
            out.append({
                "run_id": run_id,
                "workflow_id": data.get("workflow_id"),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at"),
                "total_time_ms": data.get("total_time_ms"),
                "stages_completed": len(data.get("stages", {})),
            })
        return out


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


# ── Perf Bundle Builder ──────────────────────────────────────────────


def _build_designer_perf_bundle(
    *,
    run_id: Optional[str],
    workflow_id: Optional[str],
    stages: Dict[str, Any],
    total_time_ms: Optional[float],
    status: str,
    duplicate_work: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profiling = ((stages or {}).get("profiling") or {}).get("metrics") or {}
    compilation = ((stages or {}).get("compilation") or {}).get("metrics") or {}
    sandbox = ((stages or {}).get("sandbox") or {}).get("metrics") or {}
    metrics: Dict[str, Any] = {
        "total_time_ms": float(total_time_ms or 0.0),
        "compile_time_ms": float(compilation.get("compile_time_ms", 0.0) or 0.0),
        "forward_time_ms": float(sandbox.get("forward_ms", 0.0) or 0.0),
        "backward_time_ms": float(sandbox.get("backward_ms", 0.0) or 0.0),
        "peak_memory_mb": float(sandbox.get("peak_memory_mb", 0.0) or 0.0),
        "native_coverage": float(profiling.get("native_coverage", 0.0) or 0.0),
        "total_flops_per_token": float(profiling.get("total_flops_per_token", 0.0) or 0.0),
        "total_params": float(profiling.get("total_params", 0.0) or 0.0),
        "status_code": 1.0 if status == "success" else 0.0,
    }
    metrics.update(_stage_elapsed_metrics(stages))
    dup = duplicate_work or build_duplicate_work_report()
    report = {
        "metrics": metrics,
        "duplicate_work": dup,
    }
    budget_verdict = evaluate_perf_budget_gate(report, budget_profile="designer_interactive")
    contract = build_perf_contract(
        component="aria_designer",
        workload="workflow_evaluation",
        identity={"run_id": run_id, "workflow_id": workflow_id, "status": status},
        metrics=metrics,
        budget_profile="designer_interactive",
        budget_verdict=budget_verdict,
        duplicate_work=dup,
    )
    artifact_path = emit_perf_artifact(contract, slug=(run_id or workflow_id or "designer_eval"))
    contract["artifact_path"] = artifact_path
    return {
        "perf_contract": contract,
        "perf_artifact_path": artifact_path,
        "perf_budget_gate": budget_verdict,
    }


# ── Small Helper Functions ───────────────────────────────────────────


def _fetch_research_recommendation_signals(force: bool = False) -> Optional[Dict[str, Any]]:
    return fetch_research_recommendation_signals(force=force)


def _discovery_url_for_fingerprint(fingerprint: str | None) -> str | None:
    token = str(fingerprint or "").strip()
    if not token:
        return None
    return f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/?search={token}"


def _compute_eval_composite_score(stage_metrics: Dict[str, Any]) -> float:
    benchmark_score = float((((stage_metrics.get("benchmarking") or {}).get("summary") or {}).get("score")) or 0.0)
    novelty = float(((stage_metrics.get("novelty") or {}).get("overall_novelty")) or 0.0)
    efficiency = float(((stage_metrics.get("compression") or {}).get("efficiency_score")) or 0.0)
    stability = float(((stage_metrics.get("sandbox") or {}).get("stability_score")) or 0.0)
    raw = (benchmark_score * 100.0) + (novelty * 40.0) + (efficiency * 30.0) + (stability * 20.0)
    return round(max(0.0, raw), 3)


# ── Lookup Helpers (DRY: eliminates repeated get->None->404 pattern) ──


def _require_component(component_id: str) -> Dict[str, Any]:
    comp = db.get_component(component_id)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
    return comp


def _require_proposal(proposal_id: str) -> Dict[str, Any]:
    proposal = db.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


def _require_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = db.get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return wf


def _require_run(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Evaluation run {run_id} not found")
    return run


# ── Research Integration ──────────────────────────────────────────────


def _sync_lineage_to_research(payload: Dict[str, Any]) -> bool:
    """Best-effort sync of designer run lineage into research notebook API."""
    if not settings.LINEAGE_SYNC_ENABLED:
        return False
    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/designer/lineage/sync"
    try:
        resp = requests.post(url, json=payload, timeout=settings.LINEAGE_SYNC_TIMEOUT)
        if resp.status_code >= 400:
            logger.warning("Lineage sync failed (%s): %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        logger.warning("Lineage sync unavailable: %s", exc)
        return False


def _auto_promote_workflow_to_research(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort: promote a saved workflow into research discoveries as screening tier."""
    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/designer/commit"
    try:
        resp = requests.post(
            url,
            json={"workflow": workflow},
            timeout=max(settings.LINEAGE_SYNC_TIMEOUT, 6.0),
        )
        if resp.status_code >= 400:
            logger.warning("Auto-promotion failed (%s): %s", resp.status_code, resp.text[:300])
            return _auto_promote_workflow_locally(workflow)
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict) or not data.get("success"):
            logger.warning("Auto-promotion returned unexpected payload: %s", data)
            return _auto_promote_workflow_locally(workflow)
        if not data.get("result_id"):
            logger.warning("Auto-promotion returned no result_id: %s", data)
            return _auto_promote_workflow_locally(workflow)
        return data
    except Exception as exc:
        logger.warning("Auto-promotion unavailable: %s", exc)
    return _auto_promote_workflow_locally(workflow)


def _convert_workflow_to_graph(
    workflow: Dict[str, Any],
) -> Optional[tuple[Any, str, str, float, Optional[float], Optional[int]]]:
    """Convert a workflow dict to a research graph with metrics.

    Returns ``(graph, fingerprint, graph_json, loss_ratio, novelty_score,
    param_count)`` on success, or ``None`` on failure.
    """
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        from research.synthesis.serializer import graph_to_json
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    try:
        model_dim = int((workflow.get("metadata") or {}).get("model_dim") or 256)
    except Exception:
        model_dim = 256

    try:
        graph, _ = _w2g(workflow, model_dim=model_dim, return_id_map=True)
        fingerprint = graph.fingerprint()
        g_json = graph_to_json(graph)
    except Exception as exc:
        logger.warning("Local auto-promotion graph conversion failed: %s", exc)
        return None

    # Extract metrics from workflow metadata
    meta = workflow.get("metadata") if isinstance(workflow.get("metadata"), dict) else {}
    loss_ratio = meta.get("loss_ratio")
    try:
        loss_ratio = float(loss_ratio) if loss_ratio is not None else 1.0
    except Exception:
        loss_ratio = 1.0
    novelty_score = meta.get("novelty_score")
    try:
        novelty_score = float(novelty_score) if novelty_score is not None else None
    except Exception:
        novelty_score = None

    # Compute param_count from the graph
    param_count: int | None = None
    try:
        from research.synthesis.compiler import compile_model
        model = compile_model(graph)
        param_count = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logger.debug("Could not compute param_count for designer graph: %s", exc)

    return (graph, fingerprint, g_json, loss_ratio, novelty_score, param_count)


def _insert_into_notebook(
    nb: Any,
    workflow: Dict[str, Any],
    fingerprint: str,
    graph_json: str,
    loss_ratio: float,
    novelty_score: Optional[float],
    param_count: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Insert or deduplicate a designer workflow into the research notebook.

    Returns a result dict on success, or ``None`` on failure.
    """
    existing = nb.conn.execute(
        "SELECT result_id FROM program_results WHERE graph_fingerprint = ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    if existing and existing[0]:
        result_id = str(existing[0])
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="designer_edit",
            architecture_desc=f"Manual edit: {workflow.get('name', fingerprint[:8])}",
            tier="screening",
            screening_passed=True,
            screening_loss_ratio=loss_ratio,
            screening_novelty=novelty_score,
        )
        return {"success": True, "result_id": result_id, "fingerprint": fingerprint, "deduped": True}

    exp_id = "designer_edits"
    existing_exp = nb.conn.execute(
        "SELECT 1 FROM experiments WHERE experiment_id = ?",
        (exp_id,),
    ).fetchone()
    if not existing_exp:
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) "
            "VALUES (?, ?, 'designer', 'completed', '{}')",
            (exp_id, _time_mod.time()),
        )
        nb.conn.commit()

    # Designer models have been tested in the designer — mark stage1 as passed
    # if the loss_ratio is reasonable (< 1.0 means better than random baseline)
    designer_tested = loss_ratio < 1.0

    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=fingerprint,
        graph_json=graph_json,
        model_source="designer_edit",
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=designer_tested,
        loss_ratio=loss_ratio,
        novelty_score=novelty_score,
        param_count=param_count,
    )
    if not result_id:
        logger.warning("Local auto-promotion rejected by quality gate (fingerprint=%s)", fingerprint)
        return None

    nb.upsert_leaderboard(
        result_id=result_id,
        model_source="designer_edit",
        architecture_desc=f"Manual edit: {workflow.get('name', fingerprint[:8])}",
        tier="screening",
        screening_passed=True,
        screening_loss_ratio=loss_ratio,
        screening_novelty=novelty_score,
    )
    return {"success": True, "result_id": result_id, "fingerprint": fingerprint, "deduped": False}


def _auto_promote_workflow_locally(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fallback promotion path: write directly to research notebook DB."""
    try:
        from research.scientist.notebook import LabNotebook  # noqa: F811
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    notebook_path = _PROJECT_ROOT / "research" / "lab_notebook.db"
    if not notebook_path.exists():
        logger.warning("Local auto-promotion unavailable (missing notebook): %s", notebook_path)
        return None

    converted = _convert_workflow_to_graph(workflow)
    if converted is None:
        return None
    _graph, fingerprint, graph_json, loss_ratio, novelty_score, param_count = converted

    nb = LabNotebook(str(notebook_path))
    try:
        return _insert_into_notebook(
            nb, workflow, fingerprint, graph_json, loss_ratio, novelty_score, param_count,
        )
    except Exception as exc:
        logger.warning("Local auto-promotion failed: %s", exc)
        return None
    finally:
        try:
            nb.close()
        except Exception:
            pass


# ── Workflow Semantic Warnings ────────────────────────────────────────


def _collect_workflow_semantic_warnings(workflow_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect approximate-mapping warnings for workflow components."""
    if not (HAS_BRIDGE and bridge_component_capability):
        return []
    warnings: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in workflow_json.get("nodes", []):
        node_id = str(node.get("id") or "")
        component_type = str(node.get("component_type") or "")
        if not component_type:
            continue
        try:
            cap = bridge_component_capability(component_type)
        except Exception:
            continue
        if not cap.get("bridge_supported"):
            continue
        semantic = str(cap.get("semantic_fidelity") or "exact")
        if semantic != "approximate":
            continue
        primitive_name = cap.get("primitive_name")
        for msg in cap.get("warnings") or [cap.get("reason")]:
            key = (node_id, component_type, str(msg))
            if key in seen:
                continue
            seen.add(key)
            warnings.append(
                {
                    "node_id": node_id,
                    "component_type": component_type,
                    "mapping_kind": cap.get("mapping_kind"),
                    "primitive_name": primitive_name,
                    "message": str(msg),
                }
            )
    return warnings


# ── Original Graph Fetch ─────────────────────────────────────────────


def _try_fetch_original_graph(
    metadata: dict[str, Any], model_dim: int,
) -> tuple[Any, dict[str, str]] | None:
    """Try to fetch the original ComputationGraph from research notebook.

    When a workflow was imported from research (has result_id in metadata),
    deserialize the original graph_json directly, bypassing the lossy
    workflow_to_graph() round-trip.

    Returns ``(graph, cg_to_aria_map)`` on success, ``None`` on failure.
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

        # Override model_dim to match the eval budget
        graph.model_dim = model_dim

        # Validate fingerprint if available
        expected_fp = metadata.get("graph_fingerprint")
        if expected_fp and graph.fingerprint() != expected_fp:
            logger.info(
                "Workflow has been modified (fingerprint changed from %s to %s). "
                "Bypassing original graph load.",
                graph.fingerprint(), expected_fp,
            )
            return None

        # Build cg_to_aria map by parsing workflow node IDs.
        # The importer creates IDs like "op_{cg_id}_{op_name}" and "input_{cg_id}".
        # We don't have the workflow nodes here, so build the canonical mapping
        # from the graph's topo order using the same convention as graph_to_workflow().
        cg_to_aria: dict[str, str] = {}
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


# ── Public API ────────────────────────────────────────────────────────

__all__ = [
    # Path constants
    "_PROJECT_ROOT",
    "_ARIA_DESIGNER_ROOT",
    "_ARIA_CORE_ROOT",
    "_RESEARCH_ROOT",
    # Import helper
    "_optional_import",
    # Optional runtime objects
    "KernelDispatcher",
    "runtime_compile",
    "find_unsupported_edge_dtype_pairings",
    "export_onnx",
    "bridge_evaluate",
    "bridge_validate",
    "bridge_estimate",
    "bridge_list_primitives",
    "bridge_analyze_compression",
    "bridge_analyze_routing",
    "bridge_component_capability",
    "bridge_profile",
    "import_survivors",
    "import_single",
    "graph_to_workflow",
    "check_compatibility",
    "compute_palette_constraints",
    "extract_block",
    "expand_block",
    "list_builtin_blocks",
    "BUILTIN_BLOCKS",
    # HAS_* flags
    "HAS_BRIDGE",
    "HAS_PROFILER",
    "HAS_IMPORTER",
    "HAS_CONSTRAINTS",
    "HAS_SUBGRAPH",
    # Eval run store
    "_EVAL_RUNS",
    "_EVAL_RUNS_LOCK",
    "_evict_old_runs",
    "_store_run",
    "_update_run",
    "_get_run",
    "_list_runs",
    "_stage_elapsed_metrics",
    # Perf bundle
    "_build_designer_perf_bundle",
    # Helpers
    "_fetch_research_recommendation_signals",
    "_discovery_url_for_fingerprint",
    "_compute_eval_composite_score",
    # Lookup helpers
    "_require_component",
    "_require_proposal",
    "_require_workflow",
    "_require_run",
    # Research integration
    "_sync_lineage_to_research",
    "_auto_promote_workflow_to_research",
    "_auto_promote_workflow_locally",
    # Semantic warnings
    "_collect_workflow_semantic_warnings",
    # Original graph fetch
    "_try_fetch_original_graph",
    # Re-exports from research
    "evaluate_perf_budget_gate",
    "build_duplicate_work_report",
    "build_perf_contract",
    "emit_perf_artifact",
    "list_recent_perf_artifacts",
    "summarize_perf_artifacts",
    # Re-exports from siblings
    "_utc_now",
]
