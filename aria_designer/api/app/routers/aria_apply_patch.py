from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from .. import database as db
from ..aria_patch_postprocess import postprocess_patched_workflow
from ..models import ApplyPatchRequest, utc_now_iso as _utc_now
from ..patcher import apply_patch_ops
from ..runtime_features import HAS_BRIDGE, bridge_validate
from ..type_utils import dig, safe_str
from ..workflow_support import (
    _require_proposal,
    _require_workflow,
    collect_unresolved_nodes,
)
from .aria_workflow_utils import canonicalize_workflow_payload

logger = logging.getLogger(__name__)
router = APIRouter()
# ---------------------------------------------------------------------------
# apply-patch / reject-patch
# ---------------------------------------------------------------------------


@router.post("/apply-patch")
def apply_patch(req: ApplyPatchRequest) -> Dict[str, Any]:
    proposal = _require_proposal(req.proposal_id)
    if proposal.get("status") == "applied":
        raise HTTPException(status_code=409, detail="Proposal already applied")
    patch_data = json.loads(proposal["patch_json"])
    workflow_id = proposal["workflow_id"]
    ops = patch_data.get("ops", [])
    proposal_base_version = int(patch_data.get("base_version") or 0)
    wf_row = _require_workflow(workflow_id)
    _raise_if_stale_patch(proposal_base_version, int(wf_row.get("version") or 0))
    workflow = json.loads(wf_row["graph_json"])
    patched_workflow = _apply_ops_to_workflow(workflow, ops)
    model_dim = patched_workflow.get("metadata", {}).get("model_dim", 256)
    validation_info = _validate_patched_workflow(patched_workflow, model_dim)
    old_fingerprint, new_fingerprint = _refresh_patch_fingerprint(
        workflow,
        patched_workflow,
        model_dim,
    )
    new_version = _persist_applied_patch(
        req,
        wf_row,
        workflow,
        patched_workflow,
        workflow_id,
    )
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


def _raise_if_stale_patch(proposal_base_version: int, current_version: int) -> None:
    if not proposal_base_version or not current_version:
        return
    if proposal_base_version == current_version:
        return
    raise HTTPException(
        status_code=409,
        detail=(
            f"Proposal is stale (base_version={proposal_base_version}, "
            f"current_version={current_version}). Regenerate a new proposal on the latest graph."
        ),
    )


def _apply_ops_to_workflow(
    workflow: Dict[str, Any],
    ops: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from ..patcher import PatchError as _PE

    added_node_ids = [
        safe_str(dig(op, "payload", "id"))
        for op in ops
        if safe_str(dig(op, "op")) == "add_node" and dig(op, "payload", "id")
    ]
    insertion_hints = {
        safe_str(dig(op, "payload", "id")): dict(
            dig(op, "payload", "insertion_hint", default={})
        )
        for op in ops
        if safe_str(dig(op, "op")) == "add_node"
        and dig(op, "payload", "id")
        and isinstance(dig(op, "payload", "insertion_hint"), dict)
    }
    try:
        patched_workflow = apply_patch_ops(workflow, ops)
    except _PE as e:
        raise HTTPException(
            status_code=422, detail=f"Patch application failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Unexpected error applying patch: {str(e)}"
        )
    if added_node_ids:
        patched_workflow = postprocess_patched_workflow(
            patched_workflow,
            added_node_ids,
            insertion_hints=insertion_hints,
        )
    canonicalize_workflow_payload(patched_workflow, preserve_raw_ids=True)
    unresolved = collect_unresolved_nodes(patched_workflow)
    if unresolved:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Patch produced unresolved component IDs.",
                "issues": unresolved,
            },
        )
    return patched_workflow


def _validate_patched_workflow(
    patched_workflow: Dict[str, Any],
    model_dim: int,
) -> Optional[Dict[str, Any]]:
    if not HAS_BRIDGE:
        return None
    validation_info = bridge_validate(patched_workflow, model_dim=model_dim)
    if not validation_info.get("valid", False):
        raise HTTPException(
            status_code=422,
            detail=f"Patched workflow invalid: {validation_info.get('error', 'unknown error')}",
        )
    return validation_info


def _refresh_patch_fingerprint(
    workflow: Dict[str, Any],
    patched_workflow: Dict[str, Any],
    model_dim: int,
) -> Tuple[Optional[str], Optional[str]]:
    old_fingerprint = workflow.get("metadata", {}).get("graph_fingerprint")
    new_fingerprint = None
    try:
        from research.synthesis.workflow_converter import (
            workflow_to_computation_graph as _w2g,
        )

        patched_graph, _ = _w2g(patched_workflow, model_dim, return_id_map=True)
        new_fingerprint = patched_graph.fingerprint()
        meta = patched_workflow.setdefault("metadata", {})
        meta["graph_fingerprint"] = new_fingerprint
        if old_fingerprint and old_fingerprint != new_fingerprint:
            meta["parent_fingerprint"] = old_fingerprint
    except Exception:
        logger.debug("Could not recompute fingerprint after patch", exc_info=True)
    return old_fingerprint, new_fingerprint


def _persist_applied_patch(
    req: ApplyPatchRequest,
    wf_row: Dict[str, Any],
    workflow: Dict[str, Any],
    patched_workflow: Dict[str, Any],
    workflow_id: str,
) -> int:
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
    db.resolve_proposal(req.proposal_id, "applied", req.approved_by, now)
    return new_version


@router.post("/reject-patch")
def reject_patch(req: ApplyPatchRequest) -> Dict[str, Any]:
    """Reject a pending patch proposal."""
    proposal = _require_proposal(req.proposal_id)
    if proposal.get("status") != "pending":
        raise HTTPException(
            status_code=409, detail=f"Proposal is already {proposal['status']}"
        )
    now = _utc_now()
    db.resolve_proposal(req.proposal_id, "rejected", req.approved_by, now)
    return {
        "rejected": True,
        "proposal_id": req.proposal_id,
        "rejected_by": req.approved_by,
    }
