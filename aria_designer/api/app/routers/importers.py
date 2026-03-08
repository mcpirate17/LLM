from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from .. import database as db
from ..models import utc_now_iso as _utc_now

# Optional runtime imports
try:
    from runtime.importer import import_survivors, import_single
    HAS_IMPORTER = True
except ImportError:
    try:
        from aria_designer.runtime.importer import import_survivors, import_single
        HAS_IMPORTER = True
    except ImportError:
        import_survivors = import_single = None
        HAS_IMPORTER = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/import", tags=["import"])

@router.get("/survivors")
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/survivors/{result_id}")
def post_import_survivor(result_id: str) -> Dict[str, Any]:
    """Import a single survivor by result_id, save it as a new workflow."""
    if not HAS_IMPORTER:
        raise HTTPException(status_code=501, detail="Importer not available")
    try:
        wf = import_single(result_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Save the imported workflow
    now = _utc_now()
    workflow_id = wf.get("workflow_id") or f"imported_{result_id}"
    version = db.save_workflow(
        workflow_id=workflow_id,
        name=wf.get("name", f"Imported {result_id}"),
        graph_json=json.dumps(wf),
        author="importer",
        created_at=now,
        updated_at=now,
    )
    return {"workflow_id": workflow_id, "version": version, "workflow": wf}
