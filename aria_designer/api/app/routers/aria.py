from __future__ import annotations

def _append_edge(edges: list, source: str, target: str, source_port: str = "out", target_port: str = "in") -> None:
    if not source or not target or source == target:
        return
    exists = any(
        e.get("source") == source and
        e.get("target") == target and
        (e.get("source_port") or "out") == (source_port or "out") and
        (e.get("target_port") or "in") == (target_port or "in")
        for e in edges
    )
    if not exists:
        edges.append({
            "id": f"aria_e_{__import__('uuid').uuid4().hex[:6]}",
            "source": source,
            "target": target,
            "source_port": source_port,
            "target_port": target_port,
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




import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..models import (
    ApplyPatchRequest,
    AriaPatchProposalModel,
    SuggestComponentsRequest,
    utc_now_iso as _utc_now
)
from ..research_signals import fetch_research_recommendation_signals

# Optional runtime imports
try:
    from ..suggestions import suggest_components
    from ..mutation import refine_winner
    HAS_SUGGESTIONS = True
except ImportError:
    HAS_SUGGESTIONS = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/aria", tags=["aria"])

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

@router.post("/propose-patch")
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

@router.post("/suggest-components")
def post_suggest_components(req: SuggestComponentsRequest) -> List[Dict[str, Any]]:
    if not HAS_SUGGESTIONS:
        raise HTTPException(status_code=501, detail="Suggestions module not available")
    return suggest_components(
        req.workflow.model_dump(),
        prompt=req.prompt,
        research_signals=fetch_research_recommendation_signals(force=False),
    )

@router.get("/proposals")
def list_proposals(
    workflow_id: Optional[str] = None,
    status: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    # Note: currently database.list_proposals doesn't support workflow_id filtering,
    # but it DOES support status filtering.
    return db.list_proposals(status=status)

@router.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: str) -> Dict[str, Any]:
    return _require_proposal(proposal_id)

@router.post("/apply-patch")
def post_apply_patch(req: ApplyPatchRequest) -> Dict[str, Any]:
    proposal = _require_proposal(req.proposal_id)

    if proposal.get("status") == "applied":
        raise HTTPException(status_code=409, detail="Proposal already applied")

    patch_data = json.loads(proposal["patch_json"])
    workflow_id = proposal["workflow_id"]
    ops = patch_data.get("ops", [])

    # Load the current workflow
    wf_row = _require_workflow(workflow_id)
    workflow = json.loads(wf_row["graph_json"])

    # Apply patch operations
    from ..patcher import apply_patch_ops, PatchError as _PE
    
    added_node_ids = [
        str(((op or {}).get("payload") or {}).get("id", ""))
        for op in ops
        if str((op or {}).get("op", "")) == "add_node"
        and ((op or {}).get("payload") or {}).get("id")
    ]
    
    try:
        patched_workflow = apply_patch_ops(workflow, ops)
    except _PE as e:
        raise HTTPException(status_code=400, detail=f"Patch failed: {str(e)}")

    if added_node_ids:
        patched_workflow = _postprocess_patched_workflow(patched_workflow, added_node_ids)

    # Save the patched workflow
    now = _utc_now()
    version = db.save_workflow(
        workflow_id=workflow_id,
        name=wf_row["name"],
        graph_json=json.dumps(patched_workflow),
        author="aria",
        parent_id=f"{workflow_id}@v{wf_row.get('version', 0)}",
        created_at=now,
        updated_at=now,
    )

    # Update proposal status
    db.resolve_proposal(
        req.proposal_id,
        "applied",
        req.approved_by,
        now,
    )

    return {
        "applied": True,
        "proposal_id": req.proposal_id,
        "approved_by": req.approved_by,
        "workflow_id": workflow_id,
        "version": version,
        "status": "applied",
        "patched_workflow": patched_workflow,
    }

@router.post("/reject-patch")
def reject_patch(proposal_id: str) -> Dict[str, str]:
    """Reject a patch proposal."""
    now = _utc_now()
    if not db.resolve_proposal(proposal_id, "rejected", "manual", now):
        raise HTTPException(status_code=404, detail="Proposal not found")
    return {"status": "rejected", "proposal_id": proposal_id}

