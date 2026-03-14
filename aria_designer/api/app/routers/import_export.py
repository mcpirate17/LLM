from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..models import utc_now_iso as _utc_now, CompileWorkflowRequest
from ..marketplace import search_marketplace, install_component
from ..loader import scan_and_load

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

try:
    from runtime.export import export_onnx
except ImportError:
    try:
        from aria_designer.runtime.export import export_onnx
    except ImportError:
        export_onnx = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["import_export"])


@router.get("/import/survivors")
def get_survivors(
    n: int = Query(10, ge=1, le=100),
    sort_by: str = Query("validation_loss_ratio"),
    min_novelty: float = Query(0.0, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """List top survivors from the research pipeline as importable workflows."""
    if not HAS_IMPORTER:
        raise HTTPException(status_code=501, detail="Importer not available")
    try:
        survivors = import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty)
        return {"survivors": survivors}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/import/survivors/{result_id}")
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


@router.get("/marketplace/search")
def get_marketplace_components(q: str = "") -> List[Dict[str, Any]]:
    return search_marketplace(q)


@router.post("/marketplace/install/{component_id}")
def post_install_component(component_id: str) -> Dict[str, Any]:
    success = install_component(component_id)
    if success:
        scan_and_load()  # Reload
        return {"installed": True, "component_id": component_id}
    raise HTTPException(status_code=400, detail="Installation failed")


@router.post("/export/onnx")
def export_workflow_onnx(req: CompileWorkflowRequest) -> Any:
    if not export_onnx:
        raise HTTPException(status_code=501, detail="ONNX export not available")

    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "components"))
        onnx_bytes = export_onnx(req.workflow.model_dump(), components_dir)
        # Return as downloadable file
        from fastapi.responses import Response
        return Response(content=onnx_bytes, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=model.onnx"})
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
