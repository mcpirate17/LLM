from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import HTTPException

from . import database as db
from . import runtime_features as _rf
from .runtime_features import HAS_BRIDGE

logger = logging.getLogger(__name__)


def _require_component(component_id: str) -> Dict[str, Any]:
    comp = db.get_component(component_id)
    if comp is None:
        raise HTTPException(
            status_code=404, detail=f"Component {component_id} not found"
        )
    return comp


def _require_proposal(proposal_id: str) -> Dict[str, Any]:
    proposal = db.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


def _require_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = db.get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_id}' not found"
        )
    return wf


def _require_run(run_id: str) -> Dict[str, Any]:
    from .eval_run_store import _get_run

    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Evaluation run {run_id} not found"
        )
    return run


def get_approved_registry_ids() -> set[str]:
    return db.list_component_types(status="approved")


def collect_unresolved_nodes(
    workflow: Dict[str, Any],
    registry_ids: set[str] | None = None,
) -> List[Dict[str, str]]:
    if registry_ids is None:
        registry_ids = get_approved_registry_ids()
    issues: List[Dict[str, str]] = []
    for node in workflow.get("nodes", []):
        component_type = str(node.get("component_type") or "").strip().lower()
        if component_type and component_type not in registry_ids:
            issues.append(
                {
                    "node_id": str(node.get("id") or ""),
                    "component_type": str(node.get("component_type") or ""),
                    "message": (
                        f"Node {node.get('id')} uses unresolved component type "
                        f"'{node.get('component_type')}'."
                    ),
                }
            )
    return issues


def require_feature(flag: bool, name: str) -> None:
    if not flag:
        raise HTTPException(status_code=501, detail=f"{name} not available")


def _collect_workflow_semantic_warnings(
    workflow_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not (HAS_BRIDGE and _rf.bridge_component_capability):
        return []
    warnings: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in workflow_json.get("nodes", []):
        node_id = str(node.get("id") or "")
        component_type = str(node.get("component_type") or "")
        if not component_type:
            continue
        try:
            cap = _rf.bridge_component_capability(component_type)
        except Exception:
            logger.debug(
                "Failed to get capability for component %s",
                component_type,
                exc_info=True,
            )
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
