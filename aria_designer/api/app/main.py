from __future__ import annotations

from collections import deque
import importlib.util
import json
import logging
import re
import requests
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import database as db
from .config import settings
from .loader import scan_and_load
from .patcher import apply_patch_ops, PatchError
from .suggestions import suggest_components
from .mutation import refine_winner
from .marketplace import search_marketplace, install_component
from .diff import diff_graphs
from .collaboration import collab_manager
from .property_audit import audit_components
from .benchmark_targets import benchmark_target_catalog, build_benchmark_analysis
from .models import (
    ApplyPatchRequest,
    AskAriaPromptRequest,
    AriaPatchProposalModel,
    CompileWorkflowRequest,
    ComponentConfigValidateRequest,
    ComponentModel,
    RunWorkflowRequest,
    SuggestComponentsRequest,
    ValidateWorkflowRequest,
    ValidateWorkflowResponse,
    ValidationIssue,
    WorkflowGraphModel,
)

# Try to import runtime from parent directory
import sys
import os

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARIA_DESIGNER_ROOT = _PROJECT_ROOT / "aria_designer"
_ARIA_CORE_ROOT = _PROJECT_ROOT / "aria_core"
for _p in (_ARIA_DESIGNER_ROOT, _PROJECT_ROOT, _ARIA_CORE_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)


def _optional_import(module: str, names: list[str], *, aliases: dict[str, str] | None = None):
    """Import names from a module, returning None for each on ImportError."""
    aliases = aliases or {}
    try:
        mod = __import__(module, fromlist=names)
        return tuple(getattr(mod, n) for n in names)
    except ImportError:
        return tuple(None for _ in names)


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
HAS_BRIDGE = bridge_evaluate is not None

(bridge_profile,) = _optional_import("runtime.profiler", ["profile_workflow"])
HAS_PROFILER = bridge_profile is not None

(import_survivors, import_single, graph_to_workflow) = _optional_import(
    "runtime.importer", ["import_survivors", "import_single", "graph_to_workflow"])
HAS_IMPORTER = import_survivors is not None

# Research notebook path for fetching original graphs
_RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "research"

(check_compatibility, compute_palette_constraints) = _optional_import(
    "runtime.constraints", ["check_compatibility", "compute_palette_constraints"])
HAS_CONSTRAINTS = check_compatibility is not None

(extract_block, expand_block, list_builtin_blocks, BUILTIN_BLOCKS) = _optional_import(
    "runtime.subgraph", ["extract_block", "expand_block", "list_builtin_blocks", "BUILTIN_BLOCKS"])
HAS_SUBGRAPH = extract_block is not None

logger = logging.getLogger(__name__)
from .loader import COMPONENTS_ROOT  # noqa: E402 – single source of truth


# ── Eval Run Store ────────────────────────────────────────────────────
# In-memory store for evaluation run results, keyed by run_id.
# Each entry stores full stage-by-stage metrics so external apps (ARIA,
# dashboards, CI pipelines) can query results via REST after completion.

import threading
import time as _time_mod

_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
_EVAL_RUNS_LOCK = threading.Lock()
_EVAL_RUNS_MAX = 200          # max runs kept in memory
_EVAL_RUNS_TTL_S = 3600       # evict runs older than 1 hour
_RESEARCH_SIGNALS_CACHE_LOCK = threading.Lock()
_RESEARCH_SIGNALS_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "payload": None,
}

# Optional lineage sync to research notebook service.
from app.config import settings


def _evict_old_runs():
    """Remove expired runs. Called under lock."""
    cutoff = _time_mod.time() - _EVAL_RUNS_TTL_S
    expired = [k for k, v in _EVAL_RUNS.items() if v.get("_created_ts", 0) < cutoff]
    for k in expired:
        del _EVAL_RUNS[k]
    # If still over capacity, drop oldest
    if len(_EVAL_RUNS) > _EVAL_RUNS_MAX:
        by_ts = sorted(_EVAL_RUNS.items(), key=lambda kv: kv[1].get("_created_ts", 0))
        for k, _ in by_ts[: len(_EVAL_RUNS) - _EVAL_RUNS_MAX]:
            del _EVAL_RUNS[k]


def _store_run(run_id: str, data: Dict[str, Any]):
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        data["_created_ts"] = _time_mod.time()
        _EVAL_RUNS[run_id] = data


def _update_run(run_id: str, updates: Dict[str, Any]):
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
        out = []
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


def _fetch_research_recommendation_signals(force: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch and cache recommendation signals from research analytics API."""
    if not settings.RECOMMENDER_USE_RESEARCH_SIGNALS:
        return None

    now = _time_mod.time()
    with _RESEARCH_SIGNALS_CACHE_LOCK:
        cached_payload = _RESEARCH_SIGNALS_CACHE.get("payload")
        fetched_at = float(_RESEARCH_SIGNALS_CACHE.get("fetched_at") or 0.0)
        if not force and cached_payload and (now - fetched_at) <= max(1.0, settings.RECOMMENDER_SIGNALS_TTL_S):
            return cached_payload

    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/analytics/recommendation-signals"
    try:
        resp = requests.get(url, timeout=max(0.2, settings.RECOMMENDER_SIGNALS_TIMEOUT))
        if not resp.ok:
            return None
        payload = resp.json() if resp.content else {}
        if not isinstance(payload, dict):
            return None
        with _RESEARCH_SIGNALS_CACHE_LOCK:
            _RESEARCH_SIGNALS_CACHE["fetched_at"] = now
            _RESEARCH_SIGNALS_CACHE["payload"] = payload
        return payload
    except Exception:
        return None


from .models import utc_now_iso as _utc_now  # noqa: E402 – single source of truth


# ── Lookup helpers (DRY: eliminates repeated get→None→404 pattern) ───

def _require_component(component_id: str):
    comp = db.get_component(component_id)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
    return comp


def _require_proposal(proposal_id: str):
    proposal = db.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


def _require_workflow(workflow_id: str):
    wf = db.get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return wf


def _require_run(run_id: str):
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Evaluation run {run_id} not found")
    return run


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


def _auto_promote_workflow_locally(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fallback promotion path: write directly to research notebook DB."""
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        from research.synthesis.serializer import graph_to_json
        from research.scientist.notebook import LabNotebook
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    notebook_path = _PROJECT_ROOT / "research" / "lab_notebook.db"
    if not notebook_path.exists():
        logger.warning("Local auto-promotion unavailable (missing notebook): %s", notebook_path)
        return None

    try:
        model_dim = int((workflow.get("metadata") or {}).get("model_dim") or 256)
    except Exception:
        model_dim = 256

    try:
        graph, _ = _w2g(workflow, model_dim=model_dim, return_id_map=True)
        fingerprint = graph.fingerprint()
        graph_json = graph_to_json(graph)
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
    param_count = None
    try:
        from research.synthesis.compiler import compile_model
        model = compile_model(graph)
        param_count = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logger.debug("Could not compute param_count for designer graph: %s", exc)

    nb = LabNotebook(str(notebook_path))
    try:
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
    except Exception as exc:
        logger.warning("Local auto-promotion failed: %s", exc)
        return None
    finally:
        try:
            nb.close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load components from disk."""
    db.init_db()
    
    # Cleanup junk (abandoned workflows) on startup
    cleaned = db.cleanup_orphaned_workflows(max_age_hours=48)
    if cleaned > 0:
        logger.info("Startup cleanup: %d abandoned workflows removed", cleaned)

    count = scan_and_load()
    logger.info("Startup complete: %d components loaded", count)
    yield


app = FastAPI(title="Aria Designer API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ...

@app.websocket("/api/v1/collaboration/{workflow_id}")
async def collaboration_endpoint(websocket: WebSocket, workflow_id: str):
    await collab_manager.connect(workflow_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Broadcast update to other users
            await collab_manager.broadcast(workflow_id, data, sender=websocket)
    except WebSocketDisconnect:
        collab_manager.disconnect(workflow_id, websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        collab_manager.disconnect(workflow_id, websocket)


# ── Health ────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    counts = db.count_components()
    return {"status": "ok", "components": counts}


# ── Components ────────────────────────────────────────────────────────

@app.get("/api/v1/components")
def list_components(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Filter by status (default: approved)"),
) -> List[Dict[str, Any]]:
    """List registered components. Defaults to approved only."""
    if status is None:
        status = "approved"
    return db.list_components(category=category, status=status)


@app.get("/api/v1/components/{component_id}")
def get_component(component_id: str) -> Dict[str, Any]:
    """Get a single component by ID."""
    comp = _require_component(component_id)
    return comp


@app.get("/api/v1/components/{component_id}/properties")
def get_component_properties(component_id: str) -> Dict[str, Any]:
    """Return normalized property schema/defaults for one component."""
    comp = _require_component(component_id)

    params = comp.get("params") or {}
    properties = []
    for name, schema in params.items():
        schema = schema or {}
        properties.append({
            "name": name,
            "type": schema.get("type", "string"),
            "default": schema.get("default"),
            "description": schema.get("description", ""),
            "options": schema.get("options"),
            "constraints": schema.get("constraints"),
            "format": schema.get("format"),
            "required": bool(schema.get("required", False)),
        })

    return {
        "component_id": comp.get("id"),
        "component_name": comp.get("name"),
        "category": comp.get("category"),
        "description": comp.get("description", ""),
        "inputs": comp.get("inputs", []),
        "outputs": comp.get("outputs", []),
        "properties": properties,
    }


@app.get("/api/v1/components/{component_id}/execution-capability")
def get_component_execution_capability(component_id: str) -> Dict[str, Any]:
    """Return execution capability across native/runtime bridge paths."""
    comp = _require_component(component_id)

    category = comp.get("category", "")
    manifest_id = comp.get("id", component_id)
    component_type = f"{category}/{manifest_id}" if category else manifest_id

    component_dir = COMPONENTS_ROOT / str(category) / str(manifest_id)
    native_impl = []
    if (component_dir / "kernel.c").exists():
        native_impl.append("c")
    if (component_dir / "kernel.cpp").exists() or (component_dir / "kernel.cc").exists():
        native_impl.append("cpp")
    if (component_dir / "kernel.rs").exists():
        native_impl.append("rust")
    if (component_dir / "kernel.pyx").exists():
        native_impl.append("cython")

    python_fallback = (component_dir / "kernel_fallback.py").exists()

    bridge_info: Dict[str, Any] = {
        "bridge_supported": False,
        "primitive_name": None,
        "execution_class": "unknown",
        "reason": "Research bridge unavailable in this environment.",
    }
    if HAS_BRIDGE and bridge_component_capability:
        try:
            bridge_info = bridge_component_capability(component_type)
        except Exception as exc:
            bridge_info = {
                "bridge_supported": False,
                "primitive_name": None,
                "execution_class": "unknown",
                "reason": f"Capability check failed: {exc}",
            }

    return {
        "component_id": manifest_id,
        "component_type": component_type,
        "category": category,
        "native_impl": native_impl,
        "python_fallback": python_fallback,
        "preferred_backend": native_impl[0] if native_impl else ("python" if python_fallback else "none"),
        "bridge": bridge_info,
        "has_semantic_warnings": bool(bridge_info.get("warnings")),
    }


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


@app.get("/api/v1/integration/bridge-gap-report")
def get_bridge_gap_report() -> Dict[str, Any]:
    """Summarize components unsupported by the research primitive bridge."""
    comps = db.list_components(status="approved")
    gaps: List[Dict[str, Any]] = []
    by_class: Dict[str, int] = {}
    by_category: Dict[str, int] = {}

    for comp in comps:
        cid = comp.get("id")
        category = comp.get("category", "")
        ctype = f"{category}/{cid}" if category else str(cid)
        cap = (
            bridge_component_capability(ctype)
            if HAS_BRIDGE and bridge_component_capability
            else {
                "bridge_supported": False,
                "execution_class": "unknown",
                "reason": "Research bridge unavailable in this environment.",
                "primitive_name": None,
            }
        )
        if cap.get("bridge_supported"):
            continue

        execution_class = str(cap.get("execution_class", "unknown"))
        by_class[execution_class] = by_class.get(execution_class, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        gaps.append(
            {
                "component_id": cid,
                "component_type": ctype,
                "category": category,
                "execution_class": execution_class,
                "reason": cap.get("reason", ""),
            }
        )

    gaps.sort(key=lambda row: (row["category"], row["component_id"]))
    return {
        "total_components": len(comps),
        "unsupported_components": len(gaps),
        "by_execution_class": dict(sorted(by_class.items())),
        "by_category": dict(sorted(by_category.items())),
        "gaps": gaps,
    }


@app.post("/api/v1/components/{component_id}/validate-config")
def validate_component_config(component_id: str, req: ComponentConfigValidateRequest) -> Dict[str, Any]:
    """Validate a component config payload against manifest param schema/defaults."""
    comp = _require_component(component_id)

    params = comp.get("params") or {}
    raw_config = req.config or {}
    normalized = {}
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    def _type_ok(schema: Dict[str, Any], value: Any) -> bool:
        expected = schema.get("type")
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "float":
            return (isinstance(value, (int, float)) and not isinstance(value, bool))
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "string":
            return isinstance(value, str)
        if expected == "enum":
            if schema.get("multi_select") or schema.get("multiple"):
                if not isinstance(value, (list, tuple)):
                    return False
                return all(isinstance(v, (str, int, float, bool)) for v in value)
            return isinstance(value, (str, int, float, bool))
        return True

    for name, schema in params.items():
        schema = schema or {}
        has_value = name in raw_config
        value = raw_config.get(name, schema.get("default"))
        normalized[name] = value

        if schema.get("required", False) and (value is None or value == ""):
            errors.append({"param": name, "message": "Required parameter is missing"})
            continue

        if value is None:
            continue

        expected_type = schema.get("type")
        if expected_type and not _type_ok(schema, value):
            errors.append({
                "param": name,
                "message": f"Expected {expected_type}, got {type(value).__name__}",
            })
            continue

        if expected_type == "enum":
            options = schema.get("options") or []
            if options:
                if schema.get("multi_select") or schema.get("multiple"):
                    invalid_values = [v for v in (value or []) if v not in options]
                    if invalid_values:
                        errors.append({
                            "param": name,
                            "message": f"Invalid options {invalid_values}. Allowed: {options}",
                        })
                        continue
                elif value not in options:
                    errors.append({
                        "param": name,
                        "message": f"Invalid option '{value}'. Allowed: {options}",
                    })
                    continue

        constraints = schema.get("constraints") or {}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            min_v = constraints.get("min")
            max_v = constraints.get("max")
            if min_v is not None and value < min_v:
                errors.append({"param": name, "message": f"Must be >= {min_v}"})
            if max_v is not None and value > max_v:
                errors.append({"param": name, "message": f"Must be <= {max_v}"})

        if not has_value and schema.get("default") is not None:
            warnings.append({"param": name, "message": "Using default value"})

    for name in raw_config.keys():
        if name not in params:
            warnings.append({"param": name, "message": "Unknown parameter for this component"})

    category = str(comp.get("category") or "")
    manifest_id = str(comp.get("id") or component_id)
    fallback_path = COMPONENTS_ROOT / category / manifest_id / "kernel_fallback.py"
    if fallback_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                f"validate_handler_{category}_{manifest_id}",
                str(fallback_path),
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                handler_cls = getattr(module, "ComponentHandler", None)
                if handler_cls is not None:
                    handler = handler_cls()
                    validate_fn = getattr(handler, "validate_config", None)
                    if callable(validate_fn):
                        custom_errors = validate_fn(normalized) or []
                        for msg in custom_errors:
                            errors.append({"param": "__component__", "message": str(msg)})
        except Exception as exc:
            warnings.append({
                "param": "__component__",
                "message": f"Custom validation unavailable: {exc}",
            })

    return {
        "component_id": comp.get("id"),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_config": normalized,
    }


@app.get("/api/v1/components/property-audit/report")
def get_component_property_audit() -> Dict[str, Any]:
    """Audit property coverage/defaults/help for all components."""
    return audit_components(COMPONENTS_ROOT)


@app.post("/api/v1/components")
def create_component(component: ComponentModel) -> Dict[str, Any]:
    """Register a new component (status=draft)."""
    manifest = component.model_dump()
    if "params" not in manifest:
        manifest["params"] = manifest.get("params_schema") or {}
    manifest["status"] = "draft"
    now = _utc_now()
    db.upsert_component(manifest, created_at=now, updated_at=now)
    return manifest


@app.post("/api/v1/components/{component_id}/approve")
def approve_component(component_id: str) -> Dict[str, str]:
    """Approve a component for use in the palette."""
    if not db.update_component_status(component_id, "approved", _utc_now()):
        raise HTTPException(status_code=404, detail="Component not found")
    return {"status": "approved", "component_id": component_id}


@app.post("/api/v1/components/{component_id}/deprecate")
def deprecate_component(component_id: str) -> Dict[str, str]:
    """Deprecate a component (hidden from new workflows)."""
    if not db.update_component_status(component_id, "deprecated", _utc_now()):
        raise HTTPException(status_code=404, detail="Component not found")
    return {"status": "deprecated", "component_id": component_id}


@app.post("/api/v1/components/reload")
def reload_components() -> Dict[str, Any]:
    """Re-scan components/ directory and reload into DB."""
    count = scan_and_load()
    return {"reloaded": count, "totals": db.count_components()}


# ── Workflows ─────────────────────────────────────────────────────────

@app.post("/api/v1/workflows/validate", response_model=ValidateWorkflowResponse)
def validate_workflow(req: ValidateWorkflowRequest) -> ValidateWorkflowResponse:
    """Validate a workflow graph (structure, types, constraints)."""
    workflow = req.workflow
    issues: List[ValidationIssue] = []

    node_ids = {node.id for node in workflow.nodes}
    if len(node_ids) != len(workflow.nodes):
        issues.append(ValidationIssue(
            severity="error", code="duplicate_node_id",
            message="Workflow contains duplicate node ids.",
        ))

    for edge in workflow.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            issues.append(ValidationIssue(
                severity="error", code="dangling_edge",
                message=f"Edge {edge.id} references missing source/target node.",
            ))

    # Validate component types exist in registry and cache them
    comp_cache = {}
    for node in workflow.nodes:
        comp = db.get_component(node.component_type)
        if comp is None:
            issues.append(ValidationIssue(
                severity="warning", code="unknown_component",
                message=f"Node {node.id}: unknown component type '{node.component_type}'.",
            ))
        else:
            comp_cache[node.id] = comp

    # Validate manifest-port dtype compatibility for all edges.
    workflow_payload = workflow.model_dump()
    
    output_types = {"output", "output_head", "graph_output"}
    output_node_ids = {
        node.id
        for node in workflow.nodes
        if (node.component_type or "").split("/")[-1] in output_types
    }
    if output_node_ids:
        reverse_adj: Dict[str, List[str]] = {node.id: [] for node in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in node_ids and edge.target in node_ids:
                reverse_adj[edge.target].append(edge.source)

        reachable = set()
        queue = deque(output_node_ids)
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
                    message="Dead branch detected. Node does not connect to the final output.",
                ))

    if find_unsupported_edge_dtype_pairings is not None:
        dtype_issues = find_unsupported_edge_dtype_pairings(
            workflow_payload,
            lambda component_type: db.get_component(component_type),
        )
        for issue in dtype_issues:
            issues.append(ValidationIssue(
                severity="error",
                code="unsupported_edge_dtype_pairing",
                message=issue["message"],
                edge_id=issue.get("edge_id") or None,
            ))
    else:
        for edge in workflow.edges:
            src_comp = comp_cache.get(edge.source)
            tgt_comp = comp_cache.get(edge.target)
            if not src_comp or not tgt_comp:
                continue

            src_port = next((p for p in src_comp.get("outputs", []) if p["name"] == edge.source_port), None)
            tgt_port = next((p for p in tgt_comp.get("inputs", []) if p["name"] == edge.target_port), None)
            if src_port and tgt_port and src_port["dtype"] != tgt_port["dtype"]:
                pair = (src_port["dtype"], tgt_port["dtype"])
                # Allow implicit complex_tensor <-> tensor conversion
                if pair not in (("complex_tensor", "tensor"), ("tensor", "complex_tensor")):
                    issues.append(ValidationIssue(
                        severity="error",
                        code="unsupported_edge_dtype_pairing",
                        message=(
                            f"Unsupported edge dtype pairing on edge {edge.id}: "
                            f"{edge.source}.{edge.source_port} ({src_port['dtype']}) -> "
                            f"{edge.target}.{edge.target_port} ({tgt_port['dtype']}). "
                            "Supported pairings currently require matching source/target dtypes."
                        ),
                        edge_id=edge.id,
                    ))

    # Cycle detection and graph structure using native C validator if available
    if KernelDispatcher:
        try:
            dispatcher = KernelDispatcher()
            # Prepare data for C validator
            node_ids_list = [node.id for node in workflow.nodes]
            node_to_idx = {nid: i for i, nid in enumerate(node_ids_list)}
            c_edges = []
            for edge in workflow.edges:
                if edge.source in node_to_idx and edge.target in node_to_idx:
                    c_edges.append((node_to_idx[edge.source], node_to_idx[edge.target], 0, 0))

            res = dispatcher.validate_graph(node_ids_list, c_edges)
            if not res['valid']:
                code = "cycle_detected" if res.get('code') in [-3, -7] else "native_validator_error"
                issues.append(ValidationIssue(
                    severity="error", code=code,
                    message=f"Native validation failed: {res['error']}",
                ))
        except Exception as e:
            logger.error("Native validator failed, falling back: %s", e)
            # Fallback to current DFS implementation if native fails
            _validate_fallback_cycles(node_ids, workflow, issues)
    else:
        _validate_fallback_cycles(node_ids, workflow, issues)

    return ValidateWorkflowResponse(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
    )

def _validate_fallback_cycles(node_ids, workflow, issues):
    # Cycle detection (DFS)
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in workflow.edges:
        if edge.source in adj:
            adj[edge.source].append(edge.target)

    visited, in_stack = set(), set()

    def has_cycle(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in in_stack:
                return True
            if neighbor not in visited and has_cycle(neighbor):
                return True
        in_stack.discard(node)
        return False

    for nid in node_ids:
        if nid not in visited and has_cycle(nid):
            issues.append(ValidationIssue(
                severity="error", code="cycle_detected",
                message="Workflow contains a cycle.",
            ))
            break


@app.post("/api/v1/workflows/compile")
def compile_workflow(req: CompileWorkflowRequest) -> Dict[str, Any]:
    semantic_warnings = _collect_workflow_semantic_warnings(req.workflow.model_dump())
    if runtime_compile is None:
        return {
            "compiled": False,
            "error": "Runtime compiler not available",
            "workflow_id": req.workflow.workflow_id,
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }

    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))
        model = runtime_compile(req.workflow.model_dump(), components_dir)

        # For now, we don't save the compiled model to disk, just verify it compiles
        return {
            "compiled": True,
            "target": req.target,
            "workflow_id": req.workflow.workflow_id,
            "node_count": len(req.workflow.nodes),
            "submodule_count": len(model.submodules),
            "notes": "Workflow compiled successfully to torch.nn.Module",
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }
    except Exception as e:
        logger.error("Compilation failed: %s", e)
        return {
            "compiled": False,
            "error": str(e),
            "workflow_id": req.workflow.workflow_id,
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }


@app.post("/api/v1/workflows/preview")
def preview_workflow(req: CompileWorkflowRequest) -> Dict[str, Any]:
    """Run a forward pass with dummy data and return intermediate shapes/stats."""
    if not runtime_compile:
        return {"error": "Runtime not available"}
        
    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))
        model = runtime_compile(req.workflow.model_dump(), components_dir)
        
        # Generate dummy inputs
        inputs = {}
        # Find source nodes
        sources = [n.id for n in req.workflow.nodes if not any(e.target == n.id for e in req.workflow.edges)]
        import torch
        for nid in sources:
            inputs[nid] = torch.randn(1, 16, 64) # Default dummy
            
        # Hook into model execution to capture intermediate outputs?
        # WorkflowModule doesn't support hooks yet. 
        # But we can just run it and get outputs if we modify WorkflowModule or use a tracer.
        # For now, just run forward and return output shapes of sink nodes.
        
        outputs = model(inputs)
        
        results = {}
        for nid, val in outputs.items():
            if isinstance(val, torch.Tensor):
                results[nid] = {
                    "shape": list(val.shape), 
                    "mean": float(val.mean()) if val.numel() > 0 else 0.0, 
                    "std": float(val.std()) if val.numel() > 0 else 0.0
                }
            elif hasattr(val, "__len__"):
                results[nid] = {"type": type(val).__name__, "size": len(val)}
            else:
                results[nid] = {"type": type(val).__name__, "value": str(val)}
                
        return {"success": True, "results": results}
    except Exception as e:
        logger.error("Preview failed: %s", e)
        return {"success": False, "error": str(e)}


@app.post("/api/v1/workflows/run")
def run_workflow(req: RunWorkflowRequest) -> Dict[str, Any]:
    run_id = f"run_{uuid4().hex[:10]}"
    return {
        "accepted": True,
        "run_id": run_id,
        "workflow_id": req.workflow.workflow_id,
        "budget": req.budget,
        "notes": "Scaffold run path; executor integration pending.",
    }


@app.put("/api/v1/workflows/{workflow_id}")
def save_workflow(workflow_id: str, workflow: WorkflowGraphModel) -> Dict[str, Any]:
    """Save or update a workflow."""
    now = _utc_now()
    wf_dict = workflow.model_dump()
    old_fingerprint = None
    try:
        existing = db.get_workflow(workflow_id)
        if existing and existing.get("graph_json"):
            existing_graph = json.loads(existing["graph_json"])
            old_fingerprint = (existing_graph.get("metadata") or {}).get("graph_fingerprint")
    except Exception:
        old_fingerprint = None
    
    # Calculate fingerprint if bridge is available
    fingerprint = None
    if HAS_BRIDGE:
        try:
            from runtime.bridge import workflow_to_graph as _w2g
            model_dim = wf_dict.get("metadata", {}).get("model_dim", 256)
            graph, _ = _w2g(wf_dict, model_dim, return_id_map=True)
            fingerprint = graph.fingerprint()
            meta = wf_dict.setdefault("metadata", {})
            meta["graph_fingerprint"] = fingerprint
            if old_fingerprint and fingerprint and old_fingerprint != fingerprint:
                meta["parent_fingerprint"] = old_fingerprint
        except Exception as e:
            logger.warning("Could not calculate fingerprint for saved workflow: %s", e)

    version = db.save_workflow(
        workflow_id=workflow_id,
        name=workflow.name,
        graph_json=json.dumps(wf_dict),
        author="user",
        created_at=now,
        updated_at=now,
    )

    # Sync to research notebook if enabled
    if settings.LINEAGE_SYNC_ENABLED:
        try:
            # First sync the run lineage
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
            
            # Also record as a program result to get a result_id if it's a new fingerprint
            # but usually save_workflow is called often, so we might want a specific 'commit' action.
            # For now, let's just make sure the fingerprint is known.
        except Exception as e:
            logger.warning("Failed to sync saved workflow to research: %s", e)

    # Auto-promote fingerprints into research discoveries so manual designer
    # edits are visible without extra manual commit. Research endpoint dedupes
    # by fingerprint, so repeated saves are safe.
    promoted = None
    fingerprint_changed = bool(fingerprint and old_fingerprint and fingerprint != old_fingerprint)
    should_promote = bool(fingerprint)
    if should_promote:
        promoted = _auto_promote_workflow_to_research(wf_dict)

    return {
        "workflow_id": workflow_id,
        "version": version,
        "saved_at": now,
        "fingerprint": fingerprint,
        "fingerprint_changed": fingerprint_changed,
        "parent_fingerprint": old_fingerprint if fingerprint_changed else None,
        "auto_promoted": bool(promoted),
        "promoted_result_id": (promoted or {}).get("result_id") if promoted else None,
    }


@app.get("/api/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = _require_workflow(workflow_id)
    wf["graph"] = json.loads(wf.pop("graph_json"))
    return wf


@app.get("/api/v1/workflows")
def list_workflows() -> List[Dict[str, Any]]:
    return db.list_workflows()


@app.post("/api/v1/workflows/diff")
def post_diff_workflows(wf_a: WorkflowGraphModel, wf_b: WorkflowGraphModel) -> List[PatchOpModel]:
    return diff_graphs(wf_a.model_dump(), wf_b.model_dump())


# ── Aria Co-Design ────────────────────────────────────────────────────

@app.post("/api/v1/aria/propose-patch")
def propose_patch(patch: AriaPatchProposalModel) -> Dict[str, Any]:
    proposal_id = f"patch_{uuid4().hex[:10]}"
    now = _utc_now()
    db.save_proposal(
        proposal_id=proposal_id,
        workflow_id=patch.workflow_id,
        patch_json=json.dumps(patch.model_dump()),
        rationale=patch.rationale,
        created_at=now,
    )
    return {"proposal_id": proposal_id, "status": "pending", "proposal": patch.model_dump()}


def _edge_id() -> str:
    return f"aria_e_{uuid4().hex[:8]}"


def _append_edge(edges: List[Dict[str, Any]], source: str, target: str, source_port: str = "out", target_port: str = "in") -> None:
    if not source or not target or source == target:
        return
    exists = any(
        e.get("source") == source and
        e.get("target") == target and
        (e.get("source_port") or "out") == (source_port or "out") and
        (e.get("target_port") or "in") == (target_port or "in")
        for e in edges
    )
    if exists:
        return
    edges.append({
        "id": _edge_id(),
        "source": source,
        "source_port": source_port or "out",
        "target": target,
        "target_port": target_port or "in",
    })


def _contains_token(component_type: str, token: str) -> bool:
    return token in str(component_type or "").lower()


def _auto_connect_added_nodes(workflow: Dict[str, Any], added_node_ids: List[str]) -> None:
    """Connect added nodes into the main trunk (including partial insertions)."""
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    if not nodes:
        return

    nodes_by_id = {str(n.get("id")): n for n in nodes}

    for new_id in added_node_ids:
        node = nodes_by_id.get(str(new_id))
        if not node:
            continue
        ctype = str(node.get("component_type", ""))
        if _contains_token(ctype, "input") or _contains_token(ctype, "output"):
            continue

        incoming = [e for e in edges if e.get("target") == new_id]
        outgoing = [e for e in edges if e.get("source") == new_id]

        output_nodes = [n for n in nodes if _contains_token(n.get("component_type", ""), "output") and n.get("id") != new_id]
        output_node = output_nodes[-1] if output_nodes else None

        # Case 1: already fully wired
        if incoming and outgoing:
            continue

        # Case 2: outgoing-only (e.g. new -> output). Repair missing predecessor edge.
        if (not incoming) and outgoing:
            out_to_output = [e for e in outgoing if output_node and e.get("target") == output_node.get("id")]
            if out_to_output and output_node:
                # Find a predecessor candidate currently feeding output (excluding new).
                in_to_output = [
                    e for e in edges
                    if e.get("target") == output_node.get("id") and e.get("source") != new_id
                ]
                if in_to_output:
                    old = in_to_output[-1]
                    try:
                        edges.remove(old)
                    except ValueError:
                        pass
                    _append_edge(
                        edges,
                        str(old.get("source", "")),
                        str(new_id),
                        str(old.get("source_port") or "out"),
                        "in",
                    )
                    continue

                # Fallback predecessor: latest non-output sink.
                source_ids = {e.get("source") for e in edges}
                sinks = [
                    n for n in nodes
                    if n.get("id") not in source_ids
                    and n.get("id") not in {new_id, output_node.get("id")}
                    and not _contains_token(n.get("component_type", ""), "output")
                ]
                if sinks:
                    _append_edge(edges, str(sinks[-1].get("id", "")), str(new_id))
                    continue

        # Case 3: incoming-only. Prefer inserting between predecessor and output.
        if incoming and (not outgoing):
            src = str(incoming[-1].get("source", ""))
            if output_node:
                # Remove bypass source->output when present, then connect new->output.
                bypass = [
                    e for e in edges
                    if e.get("source") == src and e.get("target") == output_node.get("id")
                ]
                for b in bypass:
                    try:
                        edges.remove(b)
                    except ValueError:
                        pass
                _append_edge(edges, str(new_id), str(output_node.get("id", "")), "out", "in")
                continue

        # Case 4: isolated insertion — predecessor -> new -> output
        if output_node:
            inc_to_output = [e for e in edges if e.get("target") == output_node.get("id")]
            if inc_to_output:
                old = inc_to_output[-1]
                try:
                    edges.remove(old)
                except ValueError:
                    pass
                _append_edge(
                    edges,
                    str(old.get("source", "")),
                    str(new_id),
                    str(old.get("source_port") or "out"),
                    "in",
                )
                _append_edge(
                    edges,
                    str(new_id),
                    str(output_node.get("id", "")),
                    "out",
                    str(old.get("target_port") or "in"),
                )
                continue

        # Fallback append: sink -> new
        source_ids = {e.get("source") for e in edges}
        sinks = [
            n for n in nodes
            if n.get("id") not in source_ids
            and n.get("id") != new_id
            and not _contains_token(n.get("component_type", ""), "output")
        ]
        if sinks:
            _append_edge(edges, str(sinks[-1].get("id", "")), str(new_id))
            continue

        inputs = [n for n in nodes if _contains_token(n.get("component_type", ""), "input") and n.get("id") != new_id]
        if inputs:
            _append_edge(edges, str(inputs[0].get("id", "")), str(new_id))


def _auto_layout_workflow(workflow: Dict[str, Any]) -> None:
    """Deterministic layered auto-layout for workflow nodes."""
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    if not nodes:
        return

    node_ids = [str(n.get("id")) for n in nodes if n.get("id") is not None]
    if not node_ids:
        return

    indeg: Dict[str, int] = {nid: 0 for nid in node_ids}
    outs: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for e in edges:
        s = str(e.get("source", ""))
        t = str(e.get("target", ""))
        if s in outs and t in indeg:
            outs[s].append(t)
            indeg[t] += 1

    depths: Dict[str, int] = {nid: 0 for nid in node_ids}
    queue = [nid for nid in node_ids if indeg[nid] == 0]

    # Bias explicit inputs to depth 0.
    for n in nodes:
        nid = str(n.get("id"))
        if _contains_token(n.get("component_type", ""), "input"):
            depths[nid] = 0

    q_idx = 0
    while q_idx < len(queue):
        u = queue[q_idx]
        q_idx += 1
        for v in outs.get(u, []):
            depths[v] = max(depths.get(v, 0), depths.get(u, 0) + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    max_depth = max(depths.values()) if depths else 0
    # Push outputs to the far-right column.
    for n in nodes:
        nid = str(n.get("id"))
        if _contains_token(n.get("component_type", ""), "output"):
            depths[nid] = max_depth + 1

    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for n in nodes:
        nid = str(n.get("id"))
        d = int(depths.get(nid, 0))
        by_depth.setdefault(d, []).append(n)

    # Keep relative vertical order stable using previous y.
    for group in by_depth.values():
        def _y(node: Dict[str, Any]) -> float:
            pos = ((node.get("ui_meta") or {}).get("position") or {})
            try:
                return float(pos.get("y", 0))
            except Exception:
                return 0.0
        group.sort(key=_y)

    x_step = 260
    y_step = 140
    x0 = 90
    y0 = 120

    for d, group in sorted(by_depth.items(), key=lambda kv: kv[0]):
        for idx, n in enumerate(group):
            ui_meta = n.setdefault("ui_meta", {})
            ui_meta["position"] = {"x": x0 + d * x_step, "y": y0 + idx * y_step}


def _postprocess_patched_workflow(workflow: Dict[str, Any], added_node_ids: List[str]) -> Dict[str, Any]:
    """Connect added nodes if needed and realign the entire canvas."""
    if not isinstance(workflow, dict):
        return workflow
    _auto_connect_added_nodes(workflow, added_node_ids)
    _auto_layout_workflow(workflow)
    return workflow


@app.post("/api/v1/aria/apply-patch")
def apply_patch(req: ApplyPatchRequest) -> Dict[str, Any]:
    proposal = _require_proposal(req.proposal_id)

    if proposal.get("status") == "applied":
        raise HTTPException(status_code=409, detail="Proposal already applied")

    patch_data = json.loads(proposal["patch_json"])
    workflow_id = proposal["workflow_id"]
    ops = patch_data.get("ops", [])
    proposal_base_version = int(patch_data.get("base_version") or 0)

    # Load the current workflow
    wf_row = _require_workflow(workflow_id)
    current_version = int(wf_row.get("version") or 0)
    if proposal_base_version and current_version and proposal_base_version != current_version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Proposal is stale (base_version={proposal_base_version}, "
                f"current_version={current_version}). Regenerate a new proposal on the latest graph."
            ),
        )

    workflow = json.loads(wf_row["graph_json"])

    # Apply patch operations
    from .patcher import PatchError as _PE
    added_node_ids = [
        str(((op or {}).get("payload") or {}).get("id", ""))
        for op in ops
        if str((op or {}).get("op", "")) == "add_node"
        and ((op or {}).get("payload") or {}).get("id")
    ]
    try:
        patched_workflow = apply_patch_ops(workflow, ops)
    except _PE as e:
        raise HTTPException(status_code=422, detail=f"Patch application failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Unexpected error applying patch: {str(e)}")

    if added_node_ids:
        patched_workflow = _postprocess_patched_workflow(patched_workflow, added_node_ids)

    # Validate patched workflow with bridge if available
    validation_info = None
    new_fingerprint = None
    model_dim = patched_workflow.get("metadata", {}).get("model_dim", 256)
    if HAS_BRIDGE:
        validation_info = bridge_validate(patched_workflow, model_dim=model_dim)
        if not validation_info.get("valid", False):
            raise HTTPException(
                status_code=422,
                detail=f"Patched workflow invalid: {validation_info.get('error', 'unknown error')}",
            )

    # Recompute fingerprint for the patched graph to track lineage
    old_fingerprint = workflow.get("metadata", {}).get("graph_fingerprint")
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        patched_graph, _ = _w2g(patched_workflow, model_dim, return_id_map=True)
        new_fingerprint = patched_graph.fingerprint()
        # Update metadata with new fingerprint and lineage
        meta = patched_workflow.setdefault("metadata", {})
        meta["graph_fingerprint"] = new_fingerprint
        if old_fingerprint and old_fingerprint != new_fingerprint:
            meta["parent_fingerprint"] = old_fingerprint
    except Exception:
        logger.debug("Could not recompute fingerprint after patch", exc_info=True)

    # Save the patched workflow as a new version
    now = _utc_now()
    new_version = db.save_workflow(
        workflow_id=workflow_id,
        name=workflow.get("name", ""),
        graph_json=json.dumps(patched_workflow),
        author=f"aria (approved by {req.approved_by})",
        parent_id=f"{workflow_id}@v{wf_row.get('version', 0)}",
        created_at=now,
        updated_at=now,
    )

    # Mark proposal as applied
    db.resolve_proposal(req.proposal_id, "applied", req.approved_by, now)

    return {
        "applied": True,
        "proposal_id": req.proposal_id,
        "approved_by": req.approved_by,
        "workflow_id": workflow_id,
        "new_version": new_version,
        "ops_applied": len(ops),
        "validation": validation_info,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "patched_workflow": patched_workflow,
    }


@app.post("/api/v1/aria/reject-patch")
def reject_patch(req: ApplyPatchRequest) -> Dict[str, Any]:
    """Reject a pending patch proposal."""
    proposal = _require_proposal(req.proposal_id)
    if proposal.get("status") != "pending":
        raise HTTPException(status_code=409, detail=f"Proposal is already {proposal['status']}")
    now = _utc_now()
    db.resolve_proposal(req.proposal_id, "rejected", req.approved_by, now)
    return {
        "rejected": True,
        "proposal_id": req.proposal_id,
        "rejected_by": req.approved_by,
    }


@app.post("/api/v1/aria/suggest-components")
def get_suggestions(req: SuggestComponentsRequest) -> List[Dict[str, Any]]:
    """Suggest components based on current graph state."""
    research_signals = _fetch_research_recommendation_signals(force=False)
    return suggest_components(
        req.workflow.model_dump(),
        prompt=req.prompt,
        research_signals=research_signals,
    )


def _infer_component_from_prompt(prompt: str, fallback_suggestions: List[Dict[str, Any]]) -> Optional[str]:
    lower = prompt.lower()
    components_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))

    # Canonicalize common historical aliases to installed component IDs.
    aliases = {
        "normalization/layernorm": "normalization/layernorm_pre",
        "normalization/rmsnorm": "normalization/rmsnorm_pre",
    }

    def _is_installed(comp_type: str) -> bool:
        token = aliases.get(comp_type, comp_type)
        if "/" not in token:
            return False
        cat, cid = token.split("/", 1)
        return os.path.isdir(os.path.join(components_root, cat, cid))

    # Try to match component names directly against the approved component DB
    # This handles "add layernorm", "insert softmax_attention", etc.
    approved = db.list_components(status="approved")
    # Build lookup: component_id (leaf) → category/id
    comp_lookup = {}
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        comp_type = f"{cat}/{c['id']}" if cat else c["id"]
        comp_type = aliases.get(comp_type, comp_type)
        if not _is_installed(comp_type):
            continue
        comp_lookup[cid] = comp_type
        # Also index by name (space→underscore)
        name = str(c.get("name", "")).lower().replace(" ", "_")
        if name and name != cid:
            comp_lookup[name] = comp_type

    # Words that are action verbs / noise, not component names
    _skip_keys = {"add", "sub", "exp", "log", "set", "cos", "sin", "abs", "neg", "sign",
                  "sort", "mul", "div", "split", "join", "loop", "none", "filter",
                  "input", "output", "first", "last", "all"}

    # Check for exact component name mentions in prompt (longest match first)
    for key in sorted(comp_lookup.keys(), key=len, reverse=True):
        if key in _skip_keys:
            continue
        if key in lower and len(key) > 2:
            return comp_lookup[key]

    # Fallback heuristics for common terms (specific intents first)
    if "split pipeline" in lower or "parallel branch" in lower or "split branch" in lower:
        return "structural/split2"
    if "routing" in lower or "top-k" in lower or "topk" in lower or "early-exit" in lower:
        return "routing/mod_topk"
    if "compression" in lower or "low-rank" in lower or "low rank" in lower or "bottleneck" in lower:
        return "math_space/low_rank_proj"
    if "output" in lower:
        return "io/output_head"
    if "attention" in lower:
        return "mixing/softmax_attention"
    if "ffn" in lower or "feed forward" in lower or "mlp" in lower:
        return "channel_mixing/swiglu_mlp"
    if fallback_suggestions:
        comp = fallback_suggestions[0].get("component", {})
        cid = comp.get("id")
        cat = comp.get("category")
        if cid and "/" in cid:
            normalized = aliases.get(cid, cid)
            return normalized if _is_installed(normalized) else None
        if cid and cat:
            normalized = aliases.get(f"{cat}/{cid}", f"{cat}/{cid}")
            return normalized if _is_installed(normalized) else None
    return None


def _normalize_component_type(raw: str, approved: List[Dict[str, Any]]) -> Optional[str]:
    token = (raw or "").strip().lower().replace(" ", "_")
    if not token:
        return None
    if "/" in token:
        return token
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        name = str(c.get("name", "")).lower().replace(" ", "_")
        if token == cid or token == name:
            return f"{cat}/{cid}" if cat and cid else None
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        if token in cid:
            return f"{cat}/{cid}" if cat and cid else None
    return None


def _resolve_node_token(token: str, nodes: List[Dict[str, Any]]) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    by_id = {str(n.get("id", "")).lower(): str(n.get("id")) for n in nodes}
    if t in by_id:
        return by_id[t]
    for n in nodes:
        comp_type = str(n.get("component_type", "")).lower()
        cid = comp_type.split("/")[-1]
        if t == cid or t in cid:
            return str(n.get("id"))
    return None


@app.post("/api/v1/aria/generate-patch")
def generate_patch_from_prompt(req: AskAriaPromptRequest) -> Dict[str, Any]:
    """Generate and store a deterministic patch proposal from prompt + workflow."""
    import traceback as _tb
    try:
        return _generate_patch_impl(req)
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("generate-patch error: %s", _tb.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _generate_patch_impl(req: AskAriaPromptRequest) -> Dict[str, Any]:
    workflow = req.workflow.model_dump()
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt is required")

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    source_ids = {e.get("source") for e in edges}
    sink_nodes = [n for n in nodes if n.get("id") not in source_ids]
    last_node = sink_nodes[-1] if sink_nodes else (nodes[-1] if nodes else None)
    has_output = any("output" in str(n.get("component_type", "")) for n in nodes)

    suggestions = suggest_components(workflow)
    approved = db.list_components(status="approved")
    ops: List[Dict[str, Any]] = []
    lower = prompt.lower()

    # Build edge lookup for insertion operations
    incoming_edges = {}  # target_id → edge
    outgoing_edges = {}  # source_id → [edges]
    for e in edges:
        incoming_edges[e.get("target")] = e
        outgoing_edges.setdefault(e.get("source"), []).append(e)

    # Replace operation: "replace X with Y"
    for src_raw, dst_raw in re.findall(r"replace\s+([-\w/]+)\s+with\s+([-\w/ ]+)", lower):
        node_id = _resolve_node_token(src_raw, nodes)
        dst_type = _normalize_component_type(dst_raw, approved) or _infer_component_from_prompt(dst_raw, suggestions)
        if node_id and dst_type:
            ops.append({
                "op": "replace_node",
                "node_id": node_id,
                "payload": {"component_type": dst_type},
            })

    # Remove operation: "remove/delete X"
    for rem_raw in re.findall(r"(?:remove|delete)\s+(?:node\s+)?([-\w/]+)", lower):
        node_id = _resolve_node_token(rem_raw, nodes)
        if node_id:
            ops.append({"op": "remove_node", "node_id": node_id, "payload": {}})

    # Insert at beginning/first: "add X as the first component", "add X at the beginning"
    m_first = re.search(
        r"(?:add|insert)\s+([-\w/ ]+?)\s+(?:as\s+(?:the\s+)?first|at\s+(?:the\s+)?(?:beginning|start)|(?:to|at)\s+(?:the\s+)?front)",
        lower,
    )
    if m_first:
        comp_text = m_first.group(1).strip()
        component_type = _normalize_component_type(comp_text, approved) or \
                         _infer_component_from_prompt(comp_text, suggestions)
        if component_type:
            # Find the first non-input node
            input_node = next((n for n in nodes if "input" in str(n.get("component_type", ""))), None)
            if input_node:
                first_targets = outgoing_edges.get(input_node["id"], [])
                if first_targets:
                    first_edge = first_targets[0]
                    new_node_id = f"aria_{uuid4().hex[:8]}"
                    add_payload_first: Dict[str, Any] = {
                        "id": new_node_id,
                        "component_type": component_type,
                        "params": {},
                        "ui_meta": {"position": {"x": 160, "y": 220}},
                        "edges": [
                            {"source": input_node["id"], "source_port": "out",
                             "target": new_node_id, "target_port": "in"},
                            {"source": new_node_id, "source_port": "out",
                             "target": first_edge.get("target", ""),
                             "target_port": first_edge.get("target_port", "in")},
                        ],
                    }
                    ops.append({
                        "op": "rewire",
                        "payload": {"action": "remove", "source": input_node["id"],
                                    "target": first_edge["target"]},
                    })
                    ops.append({"op": "add_node", "payload": add_payload_first})

    # Insert operation: "add/insert X before/after Y"
    for add_raw, pos_raw, tgt_raw in re.findall(
        r"(?:add|insert)\s+([-\w/ ]+?)\s+(before|after)\s+([-\w/]+)", lower
    ):
        component_type = _normalize_component_type(add_raw.strip(), approved) or \
                         _infer_component_from_prompt(add_raw.strip(), suggestions)
        target_node_id = _resolve_node_token(tgt_raw, nodes)
        if component_type and target_node_id:
            new_node_id = f"aria_{uuid4().hex[:8]}"
            add_payload: Dict[str, Any] = {
                "id": new_node_id,
                "component_type": component_type,
                "params": {},
                "ui_meta": {"position": {"x": 360, "y": 220}},
                "edges": [],
            }
            if pos_raw == "before":
                # Wire: predecessor → new_node → target_node
                inc = incoming_edges.get(target_node_id)
                if inc:
                    add_payload["edges"].append({
                        "source": inc.get("source", ""),
                        "source_port": inc.get("source_port", "out"),
                        "target": new_node_id,
                        "target_port": "in",
                    })
                add_payload["edges"].append({
                    "source": new_node_id,
                    "source_port": "out",
                    "target": target_node_id,
                    "target_port": inc.get("target_port", "in") if inc else "in",
                })
                # Remove old edge (predecessor → target)
                if inc:
                    ops.append({
                        "op": "rewire",
                        "payload": {"action": "remove", "source": inc["source"], "target": target_node_id},
                    })
            else:  # after
                out_edges = outgoing_edges.get(target_node_id, [])
                add_payload["edges"].append({
                    "source": target_node_id,
                    "source_port": "out",
                    "target": new_node_id,
                    "target_port": "in",
                })
                for oe in out_edges:
                    add_payload["edges"].append({
                        "source": new_node_id,
                        "source_port": "out",
                        "target": oe.get("target", ""),
                        "target_port": oe.get("target_port", "in"),
                    })
                    ops.append({
                        "op": "rewire",
                        "payload": {"action": "remove", "source": target_node_id, "target": oe["target"]},
                    })
            ops.append({"op": "add_node", "payload": add_payload})

    # Connect operation: "connect X to Y"
    for src_raw, tgt_raw in re.findall(r"connect\s+([-\w/]+)\s+to\s+([-\w/]+)", lower):
        source = _resolve_node_token(src_raw, nodes)
        target = _resolve_node_token(tgt_raw, nodes)
        if source and target:
            ops.append({
                "op": "rewire",
                "payload": {
                    "action": "add",
                    "source": source,
                    "source_port": "out",
                    "target": target,
                    "target_port": "in",
                },
            })

    # Param mutation: "set PARAM of NODE to VALUE"
    for key_raw, node_raw, val_raw in re.findall(
        r"set\s+([\w]+)\s+of\s+([-\w/]+)\s+to\s+([-\w.]+)", lower
    ):
        node_id = _resolve_node_token(node_raw, nodes)
        if not node_id:
            continue
        value: Any = val_raw
        if val_raw in {"true", "false"}:
            value = val_raw == "true"
        else:
            try:
                value = int(val_raw)
            except ValueError:
                try:
                    value = float(val_raw)
                except ValueError:
                    value = val_raw
        ops.append({
            "op": "mutate_param",
            "node_id": node_id,
            "payload": {key_raw: value},
        })

    # Data/control optimization: "optimize data/control", "fix schema", etc.
    if "optimize" in lower and ("data" in lower or "control" in lower or "workflow" in lower):
        for node in nodes:
            ctype = str(node.get("component_type", ""))
            if "join" in ctype:
                ops.append({
                    "op": "mutate_param",
                    "node_id": node["id"],
                    "payload": {"join_type": "inner"},
                })
            if "filter" in ctype:
                ops.append({
                    "op": "mutate_param",
                    "node_id": node["id"],
                    "payload": {"filter_scope": "dataset_row"},
                })

    # Stability/brittleness fix: "fix brittle", "fix gradients", "stabilize"
    is_stability_fix = any(kw in lower for kw in (
        "brittle", "gradient", "unstable", "nan", "explod", "stabil", "zero grad",
    ))
    if is_stability_fix and not ops:
        existing_types = {str(n.get("component_type", "")).split("/")[-1] for n in nodes}
        has_norm = any(t in existing_types for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre"))

        if not has_norm:
            # Insert normalization before the output node to prevent logit explosion
            output_node = next((n for n in nodes if "output" in str(n.get("component_type", ""))), None)
            norm_type = _normalize_component_type("rmsnorm", approved) or "normalization/rmsnorm"
            new_node_id = f"aria_{uuid4().hex[:8]}"

            if output_node and output_node["id"] in incoming_edges:
                inc = incoming_edges[output_node["id"]]
                ops.append({
                    "op": "rewire",
                    "payload": {"action": "remove", "source": inc["source"], "target": output_node["id"]},
                })
                ops.append({
                    "op": "add_node",
                    "payload": {
                        "id": new_node_id,
                        "component_type": norm_type,
                        "params": {},
                        "ui_meta": {"position": {"x": 440, "y": 220}},
                        "edges": [
                            {"source": inc.get("source", ""), "source_port": inc.get("source_port", "out"),
                             "target": new_node_id, "target_port": "in"},
                            {"source": new_node_id, "source_port": "out",
                             "target": output_node["id"], "target_port": inc.get("target_port", "in")},
                        ],
                    },
                })
            elif last_node:
                ops.append({
                    "op": "add_node",
                    "payload": {
                        "id": new_node_id,
                        "component_type": norm_type,
                        "params": {},
                        "ui_meta": {"position": {"x": 440, "y": 220}},
                        "edges": [
                            {"source": last_node["id"], "source_port": "out",
                             "target": new_node_id, "target_port": "in"},
                        ],
                    },
                })

    # Structured split/merge intent: branch from the pre-output trunk and merge back.
    split_intent = any(kw in lower for kw in (
        "split pipeline", "parallel branch", "split branch", "parallelize", "two branches",
    ))
    if split_intent and not ops:
        output_node = next((n for n in nodes if "output" in str(n.get("component_type", ""))), None)
        if output_node and output_node["id"] in incoming_edges:
            inc = incoming_edges[output_node["id"]]
            trunk = str(inc.get("source", ""))
            if trunk and trunk in {str(n.get("id", "")) for n in nodes}:
                branch_node_id = f"aria_branch_{uuid4().hex[:6]}"
                merge_node_id = f"aria_merge_{uuid4().hex[:6]}"

                # Remove trunk -> output and insert: trunk -> merge -> output
                # plus parallel branch trunk -> relu -> merge.
                ops.append({
                    "op": "rewire",
                    "payload": {"action": "remove", "source": trunk, "target": output_node["id"]},
                })
                ops.append({
                    "op": "add_node",
                    "payload": {
                        "id": branch_node_id,
                        "component_type": "math/relu",
                        "params": {},
                        "ui_meta": {"position": {"x": 420, "y": 140}},
                        "edges": [],
                    },
                })
                ops.append({
                    "op": "add_node",
                    "payload": {
                        "id": merge_node_id,
                        "component_type": "math/add",
                        "params": {},
                        "ui_meta": {"position": {"x": 560, "y": 220}},
                        "edges": [
                            {"source": trunk, "source_port": inc.get("source_port", "y"), "target": merge_node_id, "target_port": "a"},
                            {"source": branch_node_id, "source_port": "y", "target": merge_node_id, "target_port": "b"},
                            {"source": merge_node_id, "source_port": "y", "target": output_node["id"], "target_port": inc.get("target_port", "x")},
                        ],
                    },
                })
                ops.append({
                    "op": "rewire",
                    "payload": {
                        "action": "add",
                        "source": trunk,
                        "source_port": inc.get("source_port", "y"),
                        "target": branch_node_id,
                        "target_port": "x",
                    },
                })

    # Smart fallback: analyze graph and suggest useful improvements when no
    # explicit edit pattern was matched (e.g. "beat benchmarks", "improve novelty").
    if not ops:
        # For benchmark/quality prompts, prefer graph analysis over prompt inference
        is_benchmark_prompt = any(kw in lower for kw in ("benchmark", "speed", "flop", "novelty", "quality", "stability"))
        component_type = None if is_benchmark_prompt else _infer_component_from_prompt(prompt, suggestions)

        # If we still can't infer, analyze the graph to propose useful changes
        if not component_type:
            existing_types = {str(n.get("component_type", "")).split("/")[-1] for n in nodes}

            # Check what the graph is missing
            has_norm = any(t in existing_types for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre"))
            has_attention = any(t in existing_types for t in ("softmax_attention", "linear_attention", "graph_attention"))
            has_activation = any(t in existing_types for t in ("gelu", "relu", "silu", "swiglu_mlp"))
            has_residual = "add" in existing_types
            is_benchmark = any(kw in lower for kw in ("benchmark", "speed", "flop", "novelty", "quality", "stability"))
            is_novelty = "novelty" in lower

            if is_benchmark or is_novelty:
                # Strategy: add components that reference architectures use
                if not has_norm:
                    component_type = "normalization/layernorm"
                elif not has_attention and len(nodes) > 3:
                    component_type = "mixing/softmax_attention"
                elif not has_residual and len(nodes) > 3:
                    # Suggest adding a residual connection as a rewire
                    # Find first and last non-IO nodes
                    non_io = [n for n in nodes if "input" not in str(n.get("component_type", "")) and "output" not in str(n.get("component_type", ""))]
                    if len(non_io) >= 2:
                        ops.append({
                            "op": "rewire",
                            "payload": {
                                "action": "add",
                                "source": non_io[0]["id"],
                                "source_port": "out",
                                "target": non_io[-1]["id"],
                                "target_port": "in",
                            },
                        })
                else:
                    # Graph already has basics; add a novel op for novelty
                    novel_ops = ["selective_scan", "fourier_mixing", "low_rank_proj",
                                 "tropical_gate", "poincare_add", "clifford_attention"]
                    for nop in novel_ops:
                        if nop not in existing_types:
                            component_type = _normalize_component_type(nop, approved) or nop
                            break

            if not component_type and not has_output:
                component_type = "io/output_head"
            if not component_type:
                component_type = _normalize_component_type("relu", approved) or "math/relu"

        if not ops:
            new_node_id = f"aria_{uuid4().hex[:8]}"
            payload = {
                "id": new_node_id,
                "component_type": component_type,
                "params": {},
                "ui_meta": {"position": {"x": 520, "y": 220}},
                "edges": [],
            }
            # Insert before output if there is one, otherwise append to last
            output_node = next((n for n in nodes if "output" in str(n.get("component_type", ""))), None)
            if output_node and output_node["id"] in incoming_edges:
                inc = incoming_edges[output_node["id"]]
                payload["edges"].append({
                    "source": inc.get("source", ""),
                    "source_port": inc.get("source_port", "out"),
                    "target": new_node_id,
                    "target_port": "in",
                })
                payload["edges"].append({
                    "source": new_node_id,
                    "source_port": "out",
                    "target": output_node["id"],
                    "target_port": inc.get("target_port", "in"),
                })
                ops.append({
                    "op": "rewire",
                    "payload": {"action": "remove", "source": inc["source"], "target": output_node["id"]},
                })
            elif last_node is not None:
                payload["edges"].append({
                    "source": last_node.get("id", ""),
                    "source_port": "out",
                    "target": new_node_id,
                    "target_port": "in",
                })
            ops.append({"op": "add_node", "payload": payload})

    conn = db._get_conn()
    existing = conn.execute(
        "SELECT version FROM workflows WHERE id = ?", (req.workflow.workflow_id,)
    ).fetchone()
    resolved_base_version = int(existing["version"]) if existing and existing["version"] is not None else int(req.base_version or 1)

    patch = AriaPatchProposalModel(
        workflow_id=req.workflow.workflow_id,
        base_version=resolved_base_version,
        author="aria",
        rationale=f"Prompt: {prompt}",
        expected_impact={"summary": "User-directed patch generated from Ask Aria prompt."},
        ops=ops,
    )

    proposal_id = f"patch_{uuid4().hex[:10]}"
    now = _utc_now()

    # Ensure workflow exists in DB (FK constraint on aria_proposals)
    existing = conn.execute(
        "SELECT 1 FROM workflows WHERE id = ?", (patch.workflow_id,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO workflows (id, name, graph_json, version, author, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (patch.workflow_id, workflow.get("name", patch.workflow_id),
             json.dumps(workflow), req.base_version, "aria", now, now),
        )
        conn.commit()

    db.save_proposal(
        proposal_id=proposal_id,
        workflow_id=patch.workflow_id,
        patch_json=json.dumps(patch.model_dump()),
        rationale=patch.rationale,
        created_at=now,
    )
    return {
        "proposal_id": proposal_id,
        "status": "pending",
        "proposal": patch.model_dump(),
        "ops_count": len(ops),
        "suggestions_used": suggestions[:3],
    }


@app.post("/api/v1/aria/refine-winner")
def refine_winner_endpoint(workflow_id: str, num_variations: int = 3) -> Dict[str, Any]:
    """Generate evolutionary variations for a workflow."""
    try:
        proposal_ids = refine_winner(workflow_id, num_variations)
        return {"success": True, "generated_proposals": proposal_ids}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/aria/proposals")
def list_proposals(
    status: Optional[str] = Query(None),
    workflow_id: Optional[str] = Query(None),
    fresh_only: bool = Query(False),
) -> List[Dict[str, Any]]:
    proposals = db.list_proposals(status=status)

    if workflow_id:
        proposals = [p for p in proposals if str(p.get("workflow_id") or "") == str(workflow_id)]

    if fresh_only:
        versions: Dict[str, int] = {}
        filtered: List[Dict[str, Any]] = []
        for proposal in proposals:
            wf_id = str(proposal.get("workflow_id") or "")
            if not wf_id:
                continue

            current_version = versions.get(wf_id)
            if current_version is None:
                wf = db.get_workflow(wf_id)
                current_version = int(wf.get("version") or 0) if wf else 0
                versions[wf_id] = current_version

            base_version = 0
            try:
                patch = json.loads(proposal.get("patch_json") or "{}")
                base_version = int(patch.get("base_version") or 0)
            except Exception:
                base_version = 0

            # Keep legacy proposals that do not carry base_version, but drop explicit stale ones.
            if base_version > 0 and current_version > 0 and base_version != current_version:
                continue
            filtered.append(proposal)
        proposals = filtered

    return proposals


@app.get("/api/v1/aria/proposals/{proposal_id}")
def get_proposal(proposal_id: str) -> Dict[str, Any]:
    """Get a single proposal by ID."""
    proposal = _require_proposal(proposal_id)
    proposal["patch"] = json.loads(proposal.pop("patch_json", "{}"))
    return proposal


# ── Survivor Import ──────────────────────────────────────────────────

@app.get("/api/v1/import/survivors")
def get_survivors(
    n: int = Query(10, ge=1, le=100),
    sort_by: str = Query("validation_loss_ratio"),
    min_novelty: float = Query(0.0, ge=0.0, le=1.0),
) -> List[Dict[str, Any]]:
    """List top survivors from the research pipeline as importable workflows."""
    if not HAS_IMPORTER:
        raise HTTPException(status_code=501, detail="Importer not available")
    try:
        return import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/import/survivors/{result_id}")
def import_survivor(result_id: str) -> Dict[str, Any]:
    """Import a single survivor by result_id, save it as a new workflow."""
    if not HAS_IMPORTER:
        raise HTTPException(status_code=501, detail="Importer not available")
    try:
        wf = import_single(result_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Save the imported workflow
    now = _utc_now()
    version = db.save_workflow(
        workflow_id=wf["workflow_id"],
        name=wf["name"],
        graph_json=json.dumps(wf),
        author="import:research",
        created_at=now,
        updated_at=now,
    )
    wf["version"] = version
    return wf


# ── Marketplace ──────────────────────────────────────────────────────

@app.get("/api/v1/marketplace/search")
def get_marketplace_components(q: str = "") -> List[Dict[str, Any]]:
    return search_marketplace(q)


@app.post("/api/v1/marketplace/install/{component_id}")
def post_install_component(component_id: str) -> Dict[str, Any]:
    success = install_component(component_id)
    if success:
        scan_and_load() # Reload
        return {"installed": True, "component_id": component_id}
    raise HTTPException(status_code=400, detail="Installation failed")


# ── Export ────────────────────────────────────────────────────────────

@app.post("/api/v1/export/onnx")
def export_workflow_onnx(req: CompileWorkflowRequest) -> Any:
    if not export_onnx:
        raise HTTPException(status_code=501, detail="ONNX export not available")
    
    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))
        onnx_bytes = export_onnx(req.workflow.model_dump(), components_dir)
        # Return as downloadable file
        from fastapi.responses import Response
        return Response(content=onnx_bytes, media_type="application/octet-stream", 
                        headers={"Content-Disposition": f"attachment; filename=model.onnx"})
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


# ── Research Bridge ───────────────────────────────────────────────────

@app.post("/api/v1/workflows/evaluate")
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
        model_dim=budget.get("model_dim", 256),
        vocab_size=budget.get("vocab_size", 32000),
        device=budget.get("device", "cpu"),
        run_fingerprint=budget.get("run_fingerprint", False),
        run_novelty=budget.get("run_novelty", False),
        batch_size=budget.get("batch_size", 2),
        seq_len=budget.get("seq_len", 128),
    )
    result_dict = result.to_dict()
    result_dict["benchmarking"] = build_benchmark_analysis(
        result_dict,
        external_observed=budget.get("benchmark_observed"),
    )
    run_id = f"eval_{uuid4().hex[:12]}"
    result_dict["run_id"] = run_id
    result_dict["semantic_warnings"] = semantic_warnings
    result_dict["semantic_warning_count"] = len(semantic_warnings)

    created_at = _utc_now()

    # Persist for observability
    _store_run(run_id, {
        "run_id": run_id,
        "workflow_id": wf.get("workflow_id"),
        "status": result_dict.get("status", "unknown"),
        "created_at": created_at,
        "total_time_ms": result_dict.get("total_time_ms"),
        "budget": budget,
        "stages": {},
        "result": result_dict,
    })

    lineage_payload = {
        "run_id": run_id,
        "workflow_id": wf.get("workflow_id"),
        "workflow_version": wf.get("version") or (wf.get("metadata") or {}).get("version"),
        "graph_fingerprint": result_dict.get("graph_fingerprint"),
        "status": result_dict.get("status", "unknown"),
        "source": "aria_designer",
        "total_time_ms": result_dict.get("total_time_ms"),
        "metrics": {
            "sandbox_passed": result_dict.get("sandbox_passed"),
            "overall_novelty": result_dict.get("overall_novelty"),
            "efficiency_score": result_dict.get("efficiency_score"),
            "benchmark_target_score": (result_dict.get("benchmarking") or {}).get("summary", {}).get("score"),
        },
        "payload": result_dict,
        "created_at": _time_mod.time(),
    }
    result_dict["lineage_sync"] = {
        "attempted": settings.LINEAGE_SYNC_ENABLED,
        "synced": _sync_lineage_to_research(lineage_payload) if settings.LINEAGE_SYNC_ENABLED else False,
    }
    return result_dict


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


@app.post("/api/v1/workflows/evaluate/stream")
async def evaluate_workflow_stream(req: RunWorkflowRequest):
    """Stream evaluation results via SSE as each pipeline stage completes.

    Also persists results server-side so they can be queried later via
    GET /api/v1/eval/runs/{run_id} and sub-endpoints.
    The run_id is emitted as the first SSE event.
    """
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")

    import asyncio

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

        # Init the stored run record
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
            """Write stage result to the run store for REST observability."""
            _update_run(run_id, {
                f"stages": {**(_get_run(run_id) or {}).get("stages", {}), stage_name: stage_data},
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
                    "payload": {
                        "error": error,
                        "error_stage": error_stage,
                        "result": accumulated,
                    },
                    "created_at": _time_mod.time(),
                }
                lineage_synced = _sync_lineage_to_research(lineage_payload)

        def _attach_benchmarking():
            accumulated["benchmarking"] = build_benchmark_analysis(
                accumulated,
                external_observed=budget.get("benchmark_observed"),
            )

        # Emit run_id so the client can poll REST later if the stream drops
        yield f"event: run_id\ndata: {_json({'run_id': run_id})}\n\n"
        if semantic_warnings:
            accumulated["semantic_warnings"] = semantic_warnings
            yield f"event: semantic_warnings\ndata: {_json({'count': len(semantic_warnings), 'warnings': semantic_warnings})}\n\n"

        # --- Stage 1: conversion ---
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
                # Invert id_map: cg_id -> aria_id
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

        # --- Stage 2: profiling ---
        yield f"event: stage\ndata: {_json({'stage': 'profiling', 'status': 'running'})}\n\n"
        t0 = _time_mod.monotonic()
        op_profiles_for_nodes = []
        try:
            if HAS_PROFILER:
                report = await asyncio.to_thread(
                    bridge_profile, wf, model_dim=model_dim, device=device,
                    runtime=False, vocab_size=vocab_size, batch_size=batch_size, seq_len=seq_len,
                )
                report_dict = report.to_dict()
                # Map op_profiles node_ids to aria IDs
                mapped_profiles = []
                for op in report_dict.get("op_profiles", []):
                    aria_id = cg_to_aria.get(op.get("node_id"), None)
                    entry = {**op, "aria_node_id": aria_id}
                    mapped_profiles.append(entry)
                op_profiles_for_nodes = mapped_profiles
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

        # --- Stage 3: compilation ---
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

        # --- Stage 4: sandbox ---
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
            
            # --- Stage 4.5: routing (Phase 5.1 live-sync) ---
            if bridge_analyze_routing:
                yield f"event: stage\ndata: {_json({'stage': 'routing', 'status': 'running'})}\n\n"
                t0 = _time_mod.monotonic()
                try:
                    # Use bridge helper
                    rt_results = await asyncio.to_thread(
                        bridge_analyze_routing, model, graph
                    )
                    
                    # Map back to aria node IDs
                    mapped_rt = []
                    for entry in rt_results:
                        cg_node_id = entry.get("node_id")
                        aria_node_id = cg_to_aria.get(cg_node_id)
                        if aria_node_id:
                            mapped_rt.append({**entry, "aria_node_id": aria_node_id})
                    
                    metrics = {"op_routing": mapped_rt}
                    accumulated["routing"] = metrics
                    _persist_stage("routing", {"status": "done", "metrics": metrics})
                    elapsed = (_time_mod.monotonic() - t0) * 1000
                    yield f"event: stage\ndata: {_json({'stage': 'routing', 'status': 'done', 'elapsed_ms': round(elapsed, 1), 'metrics': metrics})}\n\n"
                except Exception as e:
                    logger.error(f"Routing analysis failed: {e}")
                    # Don't fail the whole stream for telemetry
                    elapsed = (_time_mod.monotonic() - t0) * 1000
                    yield f"event: stage\ndata: {_json({'stage': 'routing', 'status': 'skipped', 'elapsed_ms': round(elapsed, 1), 'error': str(e)})}\n\n"

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

        # --- Stage 5: compression ---
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

        # --- Stage 6: fingerprint ---
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

        # --- Stage 7: novelty ---
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

        # --- Done ---
        total_ms = (_time_mod.monotonic() - total_t0) * 1000
        _attach_benchmarking()
        _persist_done("success", total_ms=round(total_ms, 1))
        yield f"event: done\ndata: {_json({'status': 'success', 'total_time_ms': round(total_ms, 1), 'result': accumulated, 'benchmarking': accumulated.get('benchmarking')})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Eval Observability Endpoints ──────────────────────────────────────

@app.get("/api/v1/eval/runs")
def list_eval_runs(
    status: Optional[str] = Query(None, description="Filter by status: running, success, error, failed_sandbox"),
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


@app.get("/api/v1/eval/runs/{run_id}")
def get_eval_run(run_id: str) -> Dict[str, Any]:
    """Get full evaluation results for a run.

    Includes all stage metrics, per-op profiles, fingerprint, novelty,
    and the original budget parameters. Available during and after the run.
    """
    run = _require_run(run_id)
    # Strip internal fields
    return {k: v for k, v in run.items() if not k.startswith("_")}


@app.get("/api/v1/eval/runs/{run_id}/stages")
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


@app.get("/api/v1/eval/runs/{run_id}/profile")
def get_eval_run_profile(run_id: str) -> Dict[str, Any]:
    """Get per-op profiling data for a run.

    Returns FLOPs, params, memory per op, category breakdown,
    bottleneck analysis, and native kernel coverage. Each op includes
    its aria_node_id for mapping back to the visual canvas.
    """
    run = _require_run(run_id)
    profiling = run.get("stages", {}).get("profiling", {})
    if not profiling or profiling.get("status") != "done":
        raise HTTPException(status_code=404, detail="Profiling data not available for this run")
    return {
        "run_id": run_id,
        **profiling.get("metrics", {}),
    }


@app.get("/api/v1/eval/runs/{run_id}/fingerprint")
def get_eval_run_fingerprint(run_id: str) -> Dict[str, Any]:
    """Get behavioral fingerprint for a run.

    Returns CKA similarity scores (vs transformer/SSM/conv),
    interaction locality, sparsity, intrinsic dimensionality, and isotropy.
    """
    run = _require_run(run_id)
    fingerprint = run.get("stages", {}).get("fingerprint", {})
    if not fingerprint or fingerprint.get("status") != "done":
        raise HTTPException(status_code=404, detail="Fingerprint data not available for this run")
    return {
        "run_id": run_id,
        **fingerprint.get("metrics", {}),
    }


@app.get("/api/v1/eval/runs/{run_id}/novelty")
def get_eval_run_novelty(run_id: str) -> Dict[str, Any]:
    """Get novelty scores for a run.

    Returns structural, behavioral, and overall novelty scores,
    plus the most similar known architecture.
    """
    run = _require_run(run_id)
    novelty = run.get("stages", {}).get("novelty", {})
    if not novelty or novelty.get("status") != "done":
        raise HTTPException(status_code=404, detail="Novelty data not available for this run")
    return {
        "run_id": run_id,
        **novelty.get("metrics", {}),
    }


@app.get("/api/v1/eval/runs/{run_id}/compression")
def get_eval_run_compression(run_id: str) -> Dict[str, Any]:
    """Get compression & efficiency analysis for a run.

    Returns pruning curve, sparse op coverage, compression ratio,
    theoretical sizes, and composite efficiency score.
    """
    run = _require_run(run_id)
    compression = run.get("stages", {}).get("compression", {})
    if not compression or compression.get("status") != "done":
        raise HTTPException(status_code=404, detail="Compression data not available for this run")
    return {
        "run_id": run_id,
        **compression.get("metrics", {}),
    }


@app.get("/api/v1/eval/runs/{run_id}/sandbox")
def get_eval_run_sandbox(run_id: str) -> Dict[str, Any]:
    """Get sandbox evaluation results for a run.

    Returns forward/backward timing, param count, peak memory,
    gradient norm, and stability score.
    """
    run = _require_run(run_id)
    sandbox = run.get("stages", {}).get("sandbox", {})
    if not sandbox or sandbox.get("status") != "done":
        raise HTTPException(status_code=404, detail="Sandbox data not available for this run")
    return {
        "run_id": run_id,
        **sandbox.get("metrics", {}),
    }


@app.get("/api/v1/eval/runs/{run_id}/benchmarking")
def get_eval_run_benchmarking(run_id: str) -> Dict[str, Any]:
    """Get benchmark target comparison for a run.

    Includes target table, on/off-target summary, and scaling projection.
    """
    run = _require_run(run_id)
    if run.get("status") == "running":
        raise HTTPException(status_code=409, detail="Benchmarking not final while run is still running")
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


@app.get("/api/v1/benchmarks/targets")
def get_benchmark_targets(run_id: Optional[str] = Query(None, description="Optional run_id for live target comparison")) -> Dict[str, Any]:
    """Return benchmark target catalog and optional run-specific comparison."""
    payload: Dict[str, Any] = benchmark_target_catalog()
    if run_id:
        run = _require_run(run_id)
        result_payload = run.get("result") or {}
        benchmarking = result_payload.get("benchmarking")
        if not benchmarking:
            stage_metrics = {
                name: (stage.get("metrics") if isinstance(stage, dict) else None)
                for name, stage in (run.get("stages") or {}).items()
                if isinstance(stage, dict)
            }
            benchmarking = build_benchmark_analysis(stage_metrics)
        payload["run_id"] = run_id
        payload["analysis"] = benchmarking
    return payload


@app.post("/api/v1/workflows/profile")
def profile_workflow_endpoint(req: RunWorkflowRequest) -> Dict[str, Any]:
    """Profile a workflow: FLOPs, memory, latency, bottleneck analysis."""
    if not HAS_PROFILER:
        raise HTTPException(status_code=501, detail="Profiler not available")
    wf = req.workflow.model_dump()
    budget = req.budget
    report = bridge_profile(
        wf,
        model_dim=budget.get("model_dim", 256),
        device=budget.get("device", "cpu"),
        runtime=budget.get("runtime", False),
        vocab_size=budget.get("vocab_size", 32000),
        batch_size=budget.get("batch_size", 2),
        seq_len=budget.get("seq_len", 128),
    )
    return report.to_dict()


@app.post("/api/v1/workflows/validate-graph")
def validate_workflow_graph_endpoint(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    """Validate that a workflow maps to a valid ComputationGraph in the research pipeline."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    wf = req.workflow.model_dump()
    return bridge_validate(wf, model_dim=req.workflow.metadata.get("model_dim", 256))


@app.get("/api/v1/primitives")
def list_primitives() -> List[Dict[str, Any]]:
    """List all available primitives from the research pipeline."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    return bridge_list_primitives()


# ── Estimate ──────────────────────────────────────────────────────────

@app.post("/api/v1/workflows/estimate")
def estimate_workflow(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    """Estimate params, FLOPs, and memory for a workflow."""
    # Use bridge for accurate graph-aware estimation if available
    if HAS_BRIDGE:
        wf = req.workflow.model_dump()
        model_dim = req.workflow.metadata.get("model_dim", 256)
        result = bridge_estimate(wf, model_dim=model_dim)
        result["workflow_id"] = req.workflow.workflow_id
        result["node_count"] = len(req.workflow.nodes)
        result["edge_count"] = len(req.workflow.edges)
        return result

    # Fallback: manifest-based estimation
    total_params = 0
    for node in req.workflow.nodes:
        comp = db.get_component(node.component_type)
        if comp and comp.get("performance", {}).get("has_params"):
            formula = comp["performance"].get("param_formula", "0")
            try:
                val = eval(formula, {"__builtins__": {}}, {"D": 256, "D_in": 256, "D_out": 256, "vocab_size": 32000})
                total_params += int(val)
            except Exception:
                pass

    return {
        "workflow_id": req.workflow.workflow_id,
        "estimated_params": total_params,
        "node_count": len(req.workflow.nodes),
        "edge_count": len(req.workflow.edges),
    }


# ── Blocks (Subgraph Composition) ────────────────────────────────────

@app.get("/api/v1/blocks/builtin")
def get_builtin_blocks(model_dim: int = Query(256, ge=1, le=65536)) -> List[Dict[str, Any]]:
    """List all built-in block templates."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    return list_builtin_blocks(model_dim=model_dim)


@app.get("/api/v1/blocks/builtin/{block_key}")
def get_builtin_block(block_key: str, model_dim: int = Query(256, ge=1, le=65536)) -> Dict[str, Any]:
    """Get a specific built-in block template by key."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    factory = BUILTIN_BLOCKS.get(block_key)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Block '{block_key}' not found")
    return factory(model_dim=model_dim)


@app.post("/api/v1/blocks/extract")
def extract_block_endpoint(
    workflow: WorkflowGraphModel,
    node_ids: List[str] = Query(...),
    block_name: str = Query("Custom Block"),
) -> Dict[str, Any]:
    """Extract a set of nodes from a workflow as a reusable block."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    wf = workflow.model_dump()
    block, modified_wf = extract_block(wf, set(node_ids), block_name)
    return {"block": block, "modified_workflow": modified_wf}


@app.post("/api/v1/blocks/expand")
def expand_block_endpoint(
    workflow: WorkflowGraphModel,
    block_node_id: str = Query(...),
    block: Dict[str, Any] = ...,
) -> Dict[str, Any]:
    """Expand a block node back into its constituent nodes."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    wf = workflow.model_dump()
    try:
        expanded = expand_block(wf, block_node_id, block)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return expanded


# ── Constraint Visualization ─────────────────────────────────────────

@app.post("/api/v1/constraints/check")
def check_constraints_endpoint(
    req: ValidateWorkflowRequest,
    candidate_id: str = Query(..., description="Component ID to check"),
) -> Dict[str, Any]:
    """Check if a candidate component is compatible with the current workflow."""
    if not HAS_CONSTRAINTS:
        raise HTTPException(status_code=501, detail="Constraints module not available")
    wf = req.workflow.model_dump()
    return check_compatibility(wf, candidate_id)


@app.post("/api/v1/constraints/palette")
def palette_constraints_endpoint(req: ValidateWorkflowRequest) -> Dict[str, Dict[str, Any]]:
    """Compute compatibility for all palette components against the current workflow."""
    if not HAS_CONSTRAINTS:
        raise HTTPException(status_code=501, detail="Constraints module not available")
    wf = req.workflow.model_dump()
    # Get all approved component IDs
    all_components = db.list_components(status="approved")
    component_ids = [c["id"] for c in all_components]
    return compute_palette_constraints(wf, component_ids, selected_node_id=req.selected_node_id)
