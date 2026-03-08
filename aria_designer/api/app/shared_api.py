from __future__ import annotations

import logging
import os
import threading
import time as _time_mod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException

from . import database as db
from .config import settings

logger = logging.getLogger(__name__)

# ── Import Helper ─────────────────────────────────────────────────────

def _optional_import(module: str, names: list[str], *, aliases: dict[str, str] | None = None):
    """Import names from a module, returning None for each on ImportError.
    
    Tries both direct import and aria_designer prefixed import for runtime flexibility.
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

# ── Eval Run Store (Shared State) ─────────────────────────────────────

_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
_EVAL_RUNS_LOCK = threading.Lock()
_EVAL_RUNS_MAX = 200          # max runs kept in memory
_EVAL_RUNS_TTL_S = 3600       # evict runs older than 1 hour

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

# ── Lookup Helpers ────────────────────────────────────────────────────

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

def _auto_promote_workflow_to_research(workflow: Dict[str, Any], project_root: Path) -> Optional[Dict[str, Any]]:
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
            return _auto_promote_workflow_locally(workflow, project_root)
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict) or not data.get("success"):
            logger.warning("Auto-promotion returned unexpected payload: %s", data)
            return _auto_promote_workflow_locally(workflow, project_root)
        return data
    except Exception as exc:
        logger.warning("Auto-promotion unavailable: %s", exc)
    return _auto_promote_workflow_locally(workflow, project_root)

def _auto_promote_workflow_locally(workflow: Dict[str, Any], project_root: Path) -> Optional[Dict[str, Any]]:
    """Fallback promotion path: write directly to research notebook DB."""
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        from research.synthesis.serializer import graph_to_json
        from research.scientist.notebook import LabNotebook
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    notebook_path = project_root / "research" / "lab_notebook.db"
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
