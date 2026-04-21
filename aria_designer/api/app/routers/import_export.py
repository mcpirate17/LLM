from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from aria_designer.component_identity import canonicalize_workflow_ids
from ..models import utc_now_iso as _utc_now, CompileWorkflowRequest
from ..workflow_support import collect_unresolved_nodes, get_approved_registry_ids

# Optional runtime imports
try:
    from aria_designer.runtime.importer import import_survivors, import_single

    HAS_IMPORTER = True
except ImportError:
    try:
        from aria_designer.runtime.importer import import_survivors, import_single

        HAS_IMPORTER = True
    except ImportError:
        import_survivors = import_single = None
        HAS_IMPORTER = False

try:
    from aria_designer.runtime.export import export_onnx
except ImportError:
    try:
        from aria_designer.runtime.export import export_onnx
    except ImportError:
        export_onnx = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["import_export"])


def _canonicalize_imported_workflow(
    workflow: Dict[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Canonicalize component IDs in an imported workflow.

    When *strict* is True (used by single-import), raises HTTPException 422
    with per-node diagnostics for any unresolved IDs.  When False (used by
    the list endpoint), returns the workflow with ``_unresolved_ids`` metadata
    so the caller can filter silently.
    """
    registry_ids = get_approved_registry_ids()
    canonicalize_workflow_ids(workflow, registry_ids, preserve_raw_ids=True)
    unresolved = collect_unresolved_nodes(workflow, registry_ids)
    if unresolved:
        if strict:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Imported workflow could not be normalized to canonical component IDs.",
                    "issues": unresolved,
                },
            )
        workflow.setdefault("metadata", {})["_unresolved_ids"] = unresolved
    return workflow


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
        survivors = [
            _canonicalize_imported_workflow(dict(s), strict=False) for s in survivors
        ]
        return {"survivors": survivors}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/import/survivors/{result_id}")
def import_survivor(result_id: str) -> Dict[str, Any]:
    """Import a single survivor by result_id, save it as a new workflow."""
    # Check if already imported into designer DB
    for prefix in ("imported_", "survivor_"):
        existing = db.get_workflow(f"{prefix}{result_id}")
        if existing is not None:
            wf = json.loads(existing["graph_json"])
            wf["version"] = existing.get("version", 1)
            return wf

    if not HAS_IMPORTER:
        raise HTTPException(status_code=501, detail="Importer not available")
    try:
        wf = import_single(result_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    wf = _canonicalize_imported_workflow(wf, strict=False)

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
    raise HTTPException(
        status_code=501,
        detail="Marketplace not implemented — no real backend exists yet",
    )


@router.post("/marketplace/install/{component_id}")
def post_install_component(component_id: str) -> Dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Marketplace install not implemented",
    )


@router.post("/export/onnx")
def export_workflow_onnx(req: CompileWorkflowRequest) -> Any:
    if not export_onnx:
        raise HTTPException(status_code=501, detail="ONNX export not available")

    try:
        components_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "components")
        )
        onnx_bytes = export_onnx(req.workflow.model_dump(), components_dir)
        # Return as downloadable file
        from fastapi.responses import Response

        return Response(
            content=onnx_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=model.onnx"},
        )
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
