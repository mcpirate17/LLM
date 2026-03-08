from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from .. import database as db
from ..models import WorkflowGraphModel, utc_now_iso as _utc_now

# Optional runtime imports
try:
    from runtime.subgraph import extract_block, expand_block, list_builtin_blocks, BUILTIN_BLOCKS
    HAS_SUBGRAPH = True
except ImportError:
    extract_block = expand_block = list_builtin_blocks = BUILTIN_BLOCKS = None
    HAS_SUBGRAPH = False

router = APIRouter(prefix="/api/v1/blocks", tags=["blocks"])

@router.get("/builtin")
def get_builtin_blocks(model_dim: int = Query(256, ge=1, le=65536)) -> List[Dict[str, Any]]:
    """List all built-in block templates."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    return list_builtin_blocks(model_dim=model_dim)

@router.get("/builtin/{block_key}")
def get_builtin_block(block_key: str, model_dim: int = Query(256, ge=1, le=65536)) -> Dict[str, Any]:
    """Get a specific built-in block template by key."""
    if not HAS_SUBGRAPH:
        raise HTTPException(status_code=501, detail="Subgraph composition not available")
    factory = BUILTIN_BLOCKS.get(block_key)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Block '{block_key}' not found")
    return factory(model_dim=model_dim)

@router.post("/extract")
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

@router.post("/expand")
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
