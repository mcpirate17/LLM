from __future__ import annotations

import logging
import time as _time_mod
from typing import Any, Dict, Optional

import requests

from .config import settings
from .runtime_features import _PROJECT_ROOT
from .type_utils import dig, safe_float
from .workflow_graph_cache import materialize_workflow_graph

logger = logging.getLogger(__name__)


def _sync_lineage_to_research(payload: Dict[str, Any]) -> bool:
    if not settings.LINEAGE_SYNC_ENABLED:
        return False
    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/designer/lineage/sync"
    try:
        resp = requests.post(url, json=payload, timeout=settings.LINEAGE_SYNC_TIMEOUT)
        if resp.status_code >= 400:
            logger.warning(
                "Lineage sync failed (%s): %s", resp.status_code, resp.text[:200]
            )
            return False
        return True
    except (requests.RequestException, OSError) as exc:
        logger.warning("Lineage sync unavailable: %s", exc)
        return False


def _auto_promote_workflow_to_research(
    workflow: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/designer/commit"
    try:
        resp = requests.post(
            url,
            json={"workflow": workflow},
            timeout=max(settings.LINEAGE_SYNC_TIMEOUT, 6.0),
        )
        if resp.status_code >= 400:
            logger.warning(
                "Auto-promotion failed (%s): %s", resp.status_code, resp.text[:300]
            )
            return _auto_promote_workflow_locally(workflow)
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict) or not data.get("success"):
            logger.warning("Auto-promotion returned unexpected payload: %s", data)
            return _auto_promote_workflow_locally(workflow)
        if not data.get("result_id"):
            logger.warning("Auto-promotion returned no result_id: %s", data)
            return _auto_promote_workflow_locally(workflow)
        return data
    except (requests.RequestException, OSError) as exc:
        logger.warning("Auto-promotion unavailable: %s", exc)
    return _auto_promote_workflow_locally(workflow)


def _convert_workflow_to_graph(
    workflow: Dict[str, Any],
) -> Optional[tuple[Any, str, str, float, Optional[float], Optional[int]]]:
    try:
        from research.synthesis.serializer import graph_to_json
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    model_dim = int(safe_float(dig(workflow, "metadata", "model_dim"), default=256))
    try:
        graph = materialize_workflow_graph(workflow, model_dim)
        fingerprint = graph.fingerprint()
        g_json = graph_to_json(graph)
    except Exception as exc:
        logger.warning("Local auto-promotion graph conversion failed: %s", exc)
        return None

    meta = (
        workflow.get("metadata") if isinstance(workflow.get("metadata"), dict) else {}
    )
    loss_ratio = meta.get("loss_ratio")
    try:
        loss_ratio = float(loss_ratio) if loss_ratio is not None else 1.0
    except Exception:
        logger.debug(
            "Failed to parse loss_ratio from workflow metadata, defaulting to 1.0",
            exc_info=True,
        )
        loss_ratio = 1.0
    novelty_score = meta.get("novelty_score")
    try:
        novelty_score = float(novelty_score) if novelty_score is not None else None
    except Exception:
        logger.debug(
            "Failed to parse novelty_score from workflow metadata", exc_info=True
        )
        novelty_score = None

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
        return {
            "success": True,
            "result_id": result_id,
            "fingerprint": fingerprint,
            "deduped": True,
        }

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
        logger.warning(
            "Local auto-promotion rejected by quality gate (fingerprint=%s)",
            fingerprint,
        )
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
    return {
        "success": True,
        "result_id": result_id,
        "fingerprint": fingerprint,
        "deduped": False,
    }


def _auto_promote_workflow_locally(
    workflow: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        from research.scientist.notebook import LabNotebook
    except Exception as exc:
        logger.warning("Local auto-promotion unavailable (imports): %s", exc)
        return None

    notebook_path = _PROJECT_ROOT / "research" / "lab_notebook.db"
    if not notebook_path.exists():
        logger.warning(
            "Local auto-promotion unavailable (missing notebook): %s", notebook_path
        )
        return None

    converted = _convert_workflow_to_graph(workflow)
    if converted is None:
        return None
    _graph, fingerprint, graph_json, loss_ratio, novelty_score, param_count = converted

    try:
        nb = LabNotebook(str(notebook_path))
    except Exception as exc:
        logger.warning(
            "Local auto-promotion unavailable (notebook open failed): %s", exc
        )
        return None
    try:
        return _insert_into_notebook(
            nb,
            workflow,
            fingerprint,
            graph_json,
            loss_ratio,
            novelty_score,
            param_count,
        )
    except Exception as exc:
        logger.warning("Local auto-promotion failed: %s", exc)
        return None
    finally:
        try:
            nb.close()
        except Exception:
            logger.debug("Failed to close notebook connection", exc_info=True)
