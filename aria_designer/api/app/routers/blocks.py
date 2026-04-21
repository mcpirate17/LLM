from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..models import WorkflowGraphModel, ValidateWorkflowRequest
from ..runtime_features import (
    extract_block,
    expand_block,
    list_builtin_blocks,
    BUILTIN_BLOCKS,
    HAS_SUBGRAPH,
    check_compatibility,
    compute_palette_constraints,
    HAS_CONSTRAINTS,
)

router = APIRouter(prefix="/api/v1", tags=["blocks", "constraints"])


@router.get("/blocks/builtin")
def get_builtin_blocks(
    model_dim: int = Query(256, ge=1, le=65536),
) -> List[Dict[str, Any]]:
    """List all built-in block templates."""
    if not HAS_SUBGRAPH:
        raise HTTPException(
            status_code=501, detail="Subgraph composition not available"
        )
    return list_builtin_blocks(model_dim=model_dim)


@router.get("/blocks/builtin/{block_key}")
def get_builtin_block(
    block_key: str, model_dim: int = Query(256, ge=1, le=65536)
) -> Dict[str, Any]:
    """Get a specific built-in block template by key."""
    if not HAS_SUBGRAPH:
        raise HTTPException(
            status_code=501, detail="Subgraph composition not available"
        )
    factory = BUILTIN_BLOCKS.get(block_key)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Block '{block_key}' not found")
    return factory(model_dim=model_dim)


@router.post("/blocks/extract")
def extract_block_endpoint(
    workflow: WorkflowGraphModel,
    node_ids: List[str] = Query(...),
    block_name: str = Query("Custom Block"),
) -> Dict[str, Any]:
    """Extract a set of nodes from a workflow as a reusable block."""
    if not HAS_SUBGRAPH:
        raise HTTPException(
            status_code=501, detail="Subgraph composition not available"
        )
    wf = workflow.model_dump()
    block, modified_wf = extract_block(wf, set(node_ids), block_name)
    return {"block": block, "modified_workflow": modified_wf}


@router.post("/blocks/expand")
def expand_block_endpoint(
    workflow: WorkflowGraphModel,
    block_node_id: str = Query(...),
    block: Dict[str, Any] = ...,
) -> Dict[str, Any]:
    """Expand a block node back into its constituent nodes."""
    if not HAS_SUBGRAPH:
        raise HTTPException(
            status_code=501, detail="Subgraph composition not available"
        )
    wf = workflow.model_dump()
    try:
        expanded = expand_block(wf, block_node_id, block)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return expanded


@router.post("/constraints/check")
def check_constraints_endpoint(
    req: ValidateWorkflowRequest,
    candidate_id: str = Query(..., description="Component ID to check"),
) -> Dict[str, Any]:
    """Check if a candidate component is compatible with the current workflow."""
    if not HAS_CONSTRAINTS:
        raise HTTPException(status_code=501, detail="Constraints module not available")
    wf = req.workflow.model_dump()
    return check_compatibility(wf, candidate_id)


@router.post("/constraints/palette")
def palette_constraints_endpoint(
    req: ValidateWorkflowRequest,
) -> Dict[str, Dict[str, Any]]:
    """Compute compatibility for all palette components against the current workflow."""
    if not HAS_CONSTRAINTS:
        raise HTTPException(status_code=501, detail="Constraints module not available")
    wf = req.workflow.model_dump()
    # Get all approved component IDs
    all_components = db.list_components(status="approved")
    component_ids = [c["id"] for c in all_components]
    return compute_palette_constraints(
        wf, component_ids, selected_node_id=req.selected_node_id
    )
