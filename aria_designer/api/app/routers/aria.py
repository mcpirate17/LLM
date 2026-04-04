from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..aria_patch_postprocess import postprocess_patched_workflow
from ..historical_insights import build_historical_insights_response
from ..models import (
    ApplyPatchRequest,
    AriaPatchProposalModel,
    AskAriaPromptRequest,
    HistoricalInsightsResponse,
    RecordOutcomeRequest,
    SuggestComponentsRequest,
    utc_now_iso as _utc_now,
)
from ..patcher import apply_patch_ops
from ..suggestions import suggest_components
from ..mutation import refine_winner, _MUTATION_PENALTIES

HAS_SUGGESTIONS = True  # kept for test monkeypatching compatibility
from ..research_signals import (
    fetch_research_recommendation_signals,
)
from ..intent_parser import compute_insertion_point, parse_intent_constraints
from ..shared_api import (
    _require_proposal,
    _require_workflow,
    HAS_BRIDGE,
    bridge_validate,
    _PROJECT_ROOT,
    get_approved_registry_ids,
    collect_unresolved_nodes,
)
from ..component_identity import canonicalize_component_id, canonicalize_workflow_ids
from ..type_utils import dig, safe_str

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/aria", tags=["aria"])


def _canonicalize_workflow_payload(
    workflow: Dict[str, Any], *, preserve_raw_ids: bool = False
) -> Dict[str, Any]:
    canonicalize_workflow_ids(
        workflow,
        get_approved_registry_ids(),
        preserve_raw_ids=preserve_raw_ids,
    )
    return workflow


# ---------------------------------------------------------------------------
# Patch proposal CRUD
# ---------------------------------------------------------------------------


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
    return {
        "proposal_id": proposal_id,
        "status": "pending",
        "proposal": patch.model_dump(),
    }


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
    current_version = int(wf_row.get("version") or 0)
    if (
        proposal_base_version
        and current_version
        and proposal_base_version != current_version
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Proposal is stale (base_version={proposal_base_version}, "
                f"current_version={current_version}). Regenerate a new proposal on the latest graph."
            ),
        )
    workflow = json.loads(wf_row["graph_json"])
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
    _canonicalize_workflow_payload(patched_workflow, preserve_raw_ids=True)
    unresolved = collect_unresolved_nodes(patched_workflow)
    if unresolved:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Patch produced unresolved component IDs.",
                "issues": unresolved,
            },
        )
    validation_info = None
    new_fingerprint = None
    model_dim = patched_workflow.get("metadata", {}).get("model_dim", 256)
    if HAS_BRIDGE:
        validation_info = bridge_validate(patched_workflow, model_dim=model_dim)
        if not validation_info.get("valid", False):
            raise HTTPException(
                status_code=422,
                detail=f"Patched workflow invalid: {validation_info.get('error', 'unknown error')}",
            )
    old_fingerprint = workflow.get("metadata", {}).get("graph_fingerprint")
    try:
        from aria_designer.runtime.bridge import workflow_to_graph as _w2g

        patched_graph, _ = _w2g(patched_workflow, model_dim, return_id_map=True)
        new_fingerprint = patched_graph.fingerprint()
        meta = patched_workflow.setdefault("metadata", {})
        meta["graph_fingerprint"] = new_fingerprint
        if old_fingerprint and old_fingerprint != new_fingerprint:
            meta["parent_fingerprint"] = old_fingerprint
    except Exception:
        logger.debug("Could not recompute fingerprint after patch", exc_info=True)
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


@router.post("/suggest-components")
def post_suggest_components(req: SuggestComponentsRequest) -> List[Dict[str, Any]]:
    """Suggest components based on current graph state."""
    return suggest_components(
        req.workflow.model_dump(),
        prompt=req.prompt,
        research_signals=fetch_research_recommendation_signals(force=False),
    )


@router.get("/historical-insights", response_model=HistoricalInsightsResponse)
def get_historical_insights() -> HistoricalInsightsResponse:
    return build_historical_insights_response()


@router.post("/record-outcome")
def record_outcome(req: RecordOutcomeRequest) -> Dict[str, Any]:
    """Record user feedback on a suggestion (Task 3I)."""
    try:
        db.record_suggestion_outcome(
            suggestion_id=req.suggestion_id,
            outcome=req.outcome,
            timestamp=_utc_now(),
            fingerprint=req.fingerprint,
            intent=req.intent,
            details=req.details,
            session_id=req.session_id,
        )
        return {"success": True}
    except Exception as e:
        logger.error("Failed to record suggestion outcome: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Prompt inference helpers
# ---------------------------------------------------------------------------


def _infer_component_from_prompt(
    prompt: str, fallback_suggestions: List[Dict[str, Any]]
) -> Optional[str]:
    lower = prompt.lower()
    components_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "components")
    )
    registry_ids = get_approved_registry_ids()

    def _is_installed(comp_type: str) -> bool:
        resolved = canonicalize_component_id(comp_type, registry_ids)
        if "/" not in resolved:
            return False
        cat, cid = resolved.split("/", 1)
        return os.path.isdir(os.path.join(components_root, cat, cid))

    approved = db.list_components(status="approved")
    comp_lookup: Dict[str, str] = {}
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        comp_type = f"{cat}/{c['id']}" if cat else c["id"]
        comp_type = canonicalize_component_id(comp_type, registry_ids)
        if not _is_installed(comp_type):
            continue
        comp_lookup[cid] = comp_type
        name = str(c.get("name", "")).lower().replace(" ", "_")
        if name and name != cid:
            comp_lookup[name] = comp_type
    _skip_keys = {
        "add",
        "sub",
        "exp",
        "log",
        "set",
        "cos",
        "sin",
        "abs",
        "neg",
        "sign",
        "sort",
        "mul",
        "div",
        "split",
        "join",
        "loop",
        "none",
        "filter",
        "input",
        "output",
        "first",
        "last",
        "all",
    }
    for key in sorted(comp_lookup.keys(), key=len, reverse=True):
        if key in _skip_keys:
            continue
        if key in lower and len(key) > 2:
            return comp_lookup[key]
    if (
        "split pipeline" in lower
        or "parallel branch" in lower
        or "split branch" in lower
    ):
        return canonicalize_component_id("split2", registry_ids)
    if (
        "routing" in lower
        or "top-k" in lower
        or "topk" in lower
        or "early-exit" in lower
    ):
        return "routing/mod_topk"
    if (
        "compression" in lower
        or "low-rank" in lower
        or "low rank" in lower
        or "bottleneck" in lower
    ):
        return canonicalize_component_id("low_rank_proj", registry_ids)
    if "output" in lower:
        return canonicalize_component_id("output_head", registry_ids)
    if "attention" in lower:
        return canonicalize_component_id("softmax_attention", registry_ids)
    if "ffn" in lower or "feed forward" in lower or "mlp" in lower:
        return "channel_mixing/swiglu_mlp"
    if fallback_suggestions:
        comp = fallback_suggestions[0].get("component", {})
        cid_val, cat_val = comp.get("id"), comp.get("category")
        if cid_val and "/" in cid_val:
            normalized = canonicalize_component_id(cid_val, registry_ids)
            return normalized if _is_installed(normalized) else None
        if cid_val and cat_val:
            normalized = canonicalize_component_id(f"{cat_val}/{cid_val}", registry_ids)
            return normalized if _is_installed(normalized) else None
    return None


def _normalize_component_type(
    raw: str, approved: List[Dict[str, Any]]
) -> Optional[str]:
    token = (raw or "").strip().lower().replace(" ", "_")
    if not token:
        return None
    canonical = canonicalize_component_id(token, get_approved_registry_ids())
    if "/" in canonical and db.get_component(canonical):
        return canonical
    # Fallback: scan approved list for substring matches on unknown leaves
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        name = str(c.get("name", "")).lower().replace(" ", "_")
        if token == cid or token == name:
            return f"{cat}/{cid}" if cat and cid else None
    for c in approved:
        cid = str(c.get("id", "")).lower()
        cat = str(c.get("category", "")).lower()
        if token in cid:
            return f"{cat}/{cid}" if cat and cid else None
    return None


def _resolve_node_token(token: str, nodes: List[Dict[str, Any]]) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    by_id = {str(n.get("id", "")).lower(): str(n.get("id")) for n in nodes}
    if t in by_id:
        return by_id[t]
    for n in nodes:
        cid = str(n.get("component_type", "")).lower().split("/")[-1]
        if t == cid or t in cid:
            return str(n.get("id"))
    return None


def _insertion_hint_payload(
    workflow: Dict[str, Any], component_type: str | None
) -> Dict[str, str | None]:
    return compute_insertion_point(
        workflow.get("nodes") if isinstance(workflow, dict) else [],
        workflow.get("edges") if isinstance(workflow, dict) else [],
        component_type,
    )


def _resolve_component_type(
    raw: str,
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> Optional[str]:
    token = (raw or "").strip()
    if not token:
        return None
    canonical = canonicalize_component_id(token)
    if "/" in canonical:
        return canonical
    return _normalize_component_type(token, approved) or _infer_component_from_prompt(
        token, suggestions
    )


def _coerce_prompt_value(raw: str) -> Any:
    if raw in {"true", "false"}:
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _build_edge_maps(
    edges: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    incoming_edges: Dict[str, Any] = {}
    outgoing_edges: Dict[str, List[Any]] = {}
    for edge in edges:
        incoming_edges[edge.get("target")] = edge
        outgoing_edges.setdefault(edge.get("source"), []).append(edge)
    return incoming_edges, outgoing_edges


def _append_replace_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    nodes: List[Dict[str, Any]],
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> None:
    for src_raw, dst_raw in re.findall(
        r"replace\s+([-\w/]+)\s+with\s+([-\w/ ]+)", lower
    ):
        node_id = _resolve_node_token(src_raw, nodes)
        dst_type = _resolve_component_type(dst_raw, approved, suggestions)
        if node_id and dst_type:
            ops.append(
                {
                    "op": "replace_node",
                    "node_id": node_id,
                    "payload": {"component_type": dst_type},
                }
            )


def _append_remove_ops(
    ops: List[Dict[str, Any]], lower: str, nodes: List[Dict[str, Any]]
) -> None:
    for rem_raw in re.findall(r"(?:remove|delete)\s+(?:node\s+)?([-\w/]+)", lower):
        node_id = _resolve_node_token(rem_raw, nodes)
        if node_id:
            ops.append({"op": "remove_node", "node_id": node_id, "payload": {}})


def _append_insert_front_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    nodes: List[Dict[str, Any]],
    outgoing_edges: Dict[str, List[Any]],
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> None:
    match = re.search(
        r"(?:add|insert)\s+([-\w/ ]+?)\s+(?:as\s+(?:the\s+)?first|at\s+(?:the\s+)?(?:beginning|start)|(?:to|at)\s+(?:the\s+)?front)",
        lower,
    )
    if not match:
        return
    component_type = _resolve_component_type(match.group(1), approved, suggestions)
    if not component_type:
        return
    input_node = next(
        (n for n in nodes if "input" in str(n.get("component_type", ""))), None
    )
    if not input_node:
        return
    first_targets = outgoing_edges.get(input_node["id"], [])
    if not first_targets:
        return
    first_edge = first_targets[0]
    node_id = f"aria_{uuid4().hex[:8]}"
    ops.append(
        {
            "op": "rewire",
            "payload": {
                "action": "remove",
                "source": input_node["id"],
                "target": first_edge["target"],
            },
        }
    )
    ops.append(
        {
            "op": "add_node",
            "payload": {
                "id": node_id,
                "component_type": component_type,
                "params": {},
                "ui_meta": {"position": {"x": 160, "y": 220}},
                "edges": [
                    {
                        "source": input_node["id"],
                        "source_port": "out",
                        "target": node_id,
                        "target_port": "in",
                    },
                    {
                        "source": node_id,
                        "source_port": "out",
                        "target": first_edge.get("target", ""),
                        "target_port": first_edge.get("target_port", "in"),
                    },
                ],
            },
        }
    )


def _append_relative_insert_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    nodes: List[Dict[str, Any]],
    incoming_edges: Dict[str, Any],
    outgoing_edges: Dict[str, List[Any]],
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> None:
    for add_raw, position, target_raw in re.findall(
        r"(?:add|insert)\s+([-\w/ ]+?)\s+(before|after)\s+([-\w/]+)", lower
    ):
        component_type = _resolve_component_type(add_raw.strip(), approved, suggestions)
        target_id = _resolve_node_token(target_raw, nodes)
        if not component_type or not target_id:
            continue
        node_id = f"aria_{uuid4().hex[:8]}"
        payload: Dict[str, Any] = {
            "id": node_id,
            "component_type": component_type,
            "params": {},
            "ui_meta": {"position": {"x": 360, "y": 220}},
            "edges": [],
        }
        if position == "before":
            incoming = incoming_edges.get(target_id)
            if incoming:
                payload["edges"].append(
                    {
                        "source": incoming.get("source", ""),
                        "source_port": incoming.get("source_port", "out"),
                        "target": node_id,
                        "target_port": "in",
                    }
                )
                ops.append(
                    {
                        "op": "rewire",
                        "payload": {
                            "action": "remove",
                            "source": incoming["source"],
                            "target": target_id,
                        },
                    }
                )
            payload["edges"].append(
                {
                    "source": node_id,
                    "source_port": "out",
                    "target": target_id,
                    "target_port": incoming.get("target_port", "in")
                    if incoming
                    else "in",
                }
            )
        else:
            for outgoing in outgoing_edges.get(target_id, []):
                payload["edges"].append(
                    {
                        "source": node_id,
                        "source_port": "out",
                        "target": outgoing.get("target", ""),
                        "target_port": outgoing.get("target_port", "in"),
                    }
                )
                ops.append(
                    {
                        "op": "rewire",
                        "payload": {
                            "action": "remove",
                            "source": target_id,
                            "target": outgoing["target"],
                        },
                    }
                )
            payload["edges"].append(
                {
                    "source": target_id,
                    "source_port": "out",
                    "target": node_id,
                    "target_port": "in",
                }
            )
        ops.append({"op": "add_node", "payload": payload})


def _append_connect_ops(
    ops: List[Dict[str, Any]], lower: str, nodes: List[Dict[str, Any]]
) -> None:
    for source_raw, target_raw in re.findall(
        r"connect\s+([-\w/]+)\s+to\s+([-\w/]+)", lower
    ):
        source_id = _resolve_node_token(source_raw, nodes)
        target_id = _resolve_node_token(target_raw, nodes)
        if source_id and target_id:
            ops.append(
                {
                    "op": "rewire",
                    "payload": {
                        "action": "add",
                        "source": source_id,
                        "source_port": "out",
                        "target": target_id,
                        "target_port": "in",
                    },
                }
            )


def _append_param_mutation_ops(
    ops: List[Dict[str, Any]], lower: str, nodes: List[Dict[str, Any]]
) -> None:
    for key_raw, node_raw, value_raw in re.findall(
        r"set\s+([\w]+)\s+of\s+([-\w/]+)\s+to\s+([-\w.]+)", lower
    ):
        node_id = _resolve_node_token(node_raw, nodes)
        if node_id:
            ops.append(
                {
                    "op": "mutate_param",
                    "node_id": node_id,
                    "payload": {key_raw: _coerce_prompt_value(value_raw)},
                }
            )


def _append_optimize_ops(
    ops: List[Dict[str, Any]], lower: str, nodes: List[Dict[str, Any]]
) -> None:
    if "optimize" not in lower or not any(
        token in lower for token in ("data", "control", "workflow")
    ):
        return
    for node in nodes:
        component_type = str(node.get("component_type", ""))
        if "join" in component_type:
            ops.append(
                {
                    "op": "mutate_param",
                    "node_id": node["id"],
                    "payload": {"join_type": "inner"},
                }
            )
        if "filter" in component_type:
            ops.append(
                {
                    "op": "mutate_param",
                    "node_id": node["id"],
                    "payload": {"filter_scope": "dataset_row"},
                }
            )


def _append_stability_fix_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    workflow: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    incoming_edges: Dict[str, Any],
    last_node: Optional[Dict[str, Any]],
    approved: List[Dict[str, Any]],
) -> None:
    if (
        not any(
            kw in lower
            for kw in (
                "brittle",
                "gradient",
                "unstable",
                "nan",
                "explod",
                "stabil",
                "zero grad",
            )
        )
        or ops
    ):
        return
    existing_types = {str(n.get("component_type", "")).split("/")[-1] for n in nodes}
    if any(
        name in existing_types
        for name in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre")
    ):
        return
    output_node = next(
        (n for n in nodes if "output" in str(n.get("component_type", ""))), None
    )
    norm_type = (
        _normalize_component_type("rmsnorm", approved) or "normalization/rmsnorm"
    )
    node_id = f"aria_{uuid4().hex[:8]}"
    payload = {
        "id": node_id,
        "component_type": norm_type,
        "params": {},
        "insertion_hint": _insertion_hint_payload(workflow, norm_type),
        "ui_meta": {"position": {"x": 440, "y": 220}},
    }
    if output_node and output_node["id"] in incoming_edges:
        incoming = incoming_edges[output_node["id"]]
        ops.append(
            {
                "op": "rewire",
                "payload": {
                    "action": "remove",
                    "source": incoming["source"],
                    "target": output_node["id"],
                },
            }
        )
        ops.append(
            {
                "op": "add_node",
                "payload": {
                    **payload,
                    "edges": [
                        {
                            "source": incoming.get("source", ""),
                            "source_port": incoming.get("source_port", "out"),
                            "target": node_id,
                            "target_port": "in",
                        },
                        {
                            "source": node_id,
                            "source_port": "out",
                            "target": output_node["id"],
                            "target_port": incoming.get("target_port", "in"),
                        },
                    ],
                },
            }
        )
        return
    if last_node:
        ops.append(
            {
                "op": "add_node",
                "payload": {
                    **payload,
                    "edges": [
                        {
                            "source": last_node["id"],
                            "source_port": "out",
                            "target": node_id,
                            "target_port": "in",
                        }
                    ],
                },
            }
        )


def _append_split_branch_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    nodes: List[Dict[str, Any]],
    incoming_edges: Dict[str, Any],
) -> None:
    if (
        not any(
            kw in lower
            for kw in (
                "split pipeline",
                "parallel branch",
                "split branch",
                "parallelize",
                "two branches",
            )
        )
        or ops
    ):
        return
    output_node = next(
        (n for n in nodes if "output" in str(n.get("component_type", ""))), None
    )
    if not output_node or output_node["id"] not in incoming_edges:
        return
    incoming = incoming_edges[output_node["id"]]
    trunk = str(incoming.get("source", ""))
    if not trunk or trunk not in {str(n.get("id", "")) for n in nodes}:
        return
    branch_id = f"aria_branch_{uuid4().hex[:6]}"
    merge_id = f"aria_merge_{uuid4().hex[:6]}"
    ops.append(
        {
            "op": "rewire",
            "payload": {
                "action": "remove",
                "source": trunk,
                "target": output_node["id"],
            },
        }
    )
    ops.append(
        {
            "op": "add_node",
            "payload": {
                "id": branch_id,
                "component_type": "math/relu",
                "params": {},
                "ui_meta": {"position": {"x": 420, "y": 140}},
                "edges": [],
            },
        }
    )
    ops.append(
        {
            "op": "add_node",
            "payload": {
                "id": merge_id,
                "component_type": "math/add",
                "params": {},
                "ui_meta": {"position": {"x": 560, "y": 220}},
                "edges": [
                    {
                        "source": trunk,
                        "source_port": incoming.get("source_port", "y"),
                        "target": merge_id,
                        "target_port": "a",
                    },
                    {
                        "source": branch_id,
                        "source_port": "y",
                        "target": merge_id,
                        "target_port": "b",
                    },
                    {
                        "source": merge_id,
                        "source_port": "y",
                        "target": output_node["id"],
                        "target_port": incoming.get("target_port", "x"),
                    },
                ],
            },
        }
    )
    ops.append(
        {
            "op": "rewire",
            "payload": {
                "action": "add",
                "source": trunk,
                "source_port": incoming.get("source_port", "y"),
                "target": branch_id,
                "target_port": "x",
            },
        }
    )


def _select_fallback_component_type(
    lower: str,
    nodes: List[Dict[str, Any]],
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    has_output: bool,
    ops: List[Dict[str, Any]],
) -> Optional[str]:
    is_benchmark_prompt = any(
        kw in lower
        for kw in ("benchmark", "speed", "flop", "novelty", "quality", "stability")
    )
    component_type = (
        None
        if is_benchmark_prompt
        else _infer_component_from_prompt(lower, suggestions)
    )
    if component_type:
        return component_type
    existing_types = {str(n.get("component_type", "")).split("/")[-1] for n in nodes}
    has_norm = any(
        t in existing_types
        for t in ("layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre")
    )
    has_attention = any(
        t in existing_types
        for t in ("softmax_attention", "linear_attention", "graph_attention")
    )
    has_residual = "add" in existing_types
    if is_benchmark_prompt or "novelty" in lower:
        if not has_norm:
            return canonicalize_component_id("layernorm")
        if not has_attention and len(nodes) > 3:
            return canonicalize_component_id("softmax_attention")
        if not has_residual and len(nodes) > 3:
            non_io_nodes = [
                n
                for n in nodes
                if "input" not in str(n.get("component_type", ""))
                and "output" not in str(n.get("component_type", ""))
            ]
            if len(non_io_nodes) >= 2:
                ops.append(
                    {
                        "op": "rewire",
                        "payload": {
                            "action": "add",
                            "source": non_io_nodes[0]["id"],
                            "source_port": "out",
                            "target": non_io_nodes[-1]["id"],
                            "target_port": "in",
                        },
                    }
                )
                return None
        for novelty_op in (
            "selective_scan",
            "low_rank_proj",
            "tropical_gate",
            "poincare_add",
            "clifford_attention",
        ):
            if novelty_op not in existing_types:
                return _normalize_component_type(novelty_op, approved) or novelty_op
    if not has_output:
        return canonicalize_component_id("output_head")
    return _normalize_component_type("relu", approved) or canonicalize_component_id(
        "relu"
    )


def _append_fallback_ops(
    ops: List[Dict[str, Any]],
    lower: str,
    workflow: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    incoming_edges: Dict[str, Any],
    last_node: Optional[Dict[str, Any]],
    approved: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    has_output: bool,
) -> None:
    if ops:
        return
    component_type = _select_fallback_component_type(
        lower, nodes, approved, suggestions, has_output, ops
    )
    if ops or not component_type:
        return
    node_id = f"aria_{uuid4().hex[:8]}"
    payload: Dict[str, Any] = {
        "id": node_id,
        "component_type": component_type,
        "insertion_hint": _insertion_hint_payload(workflow, component_type),
        "params": {},
        "ui_meta": {"position": {"x": 520, "y": 220}},
        "edges": [],
    }
    output_node = next(
        (n for n in nodes if "output" in str(n.get("component_type", ""))), None
    )
    if output_node and output_node["id"] in incoming_edges:
        incoming = incoming_edges[output_node["id"]]
        payload["edges"].append(
            {
                "source": incoming.get("source", ""),
                "source_port": incoming.get("source_port", "out"),
                "target": node_id,
                "target_port": "in",
            }
        )
        payload["edges"].append(
            {
                "source": node_id,
                "source_port": "out",
                "target": output_node["id"],
                "target_port": incoming.get("target_port", "in"),
            }
        )
        ops.append(
            {
                "op": "rewire",
                "payload": {
                    "action": "remove",
                    "source": incoming["source"],
                    "target": output_node["id"],
                },
            }
        )
    elif last_node is not None:
        payload["edges"].append(
            {
                "source": last_node.get("id", ""),
                "source_port": "out",
                "target": node_id,
                "target_port": "in",
            }
        )
    ops.append({"op": "add_node", "payload": payload})


# ---------------------------------------------------------------------------
# generate-patch (prompt -> deterministic patch)
# ---------------------------------------------------------------------------


@router.post("/generate-patch")
def generate_patch_from_prompt(req: AskAriaPromptRequest) -> Dict[str, Any]:
    """Generate and store a deterministic patch proposal from prompt + workflow."""
    import traceback as _tb

    try:
        return _generate_patch_impl(req)
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error("generate-patch error: %s", _tb.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _save_patch_proposal(
    req: AskAriaPromptRequest,
    workflow: Dict[str, Any],
    prompt: str,
    ops: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Persist the generated patch proposal and return the API response."""
    conn = db._get_conn()
    existing = conn.execute(
        "SELECT version FROM workflows WHERE id = ?",
        (req.workflow.workflow_id,),
    ).fetchone()
    resolved_base_version = (
        int(existing["version"])
        if existing and existing["version"] is not None
        else int(req.base_version or 1)
    )
    patch = AriaPatchProposalModel(
        workflow_id=req.workflow.workflow_id,
        base_version=resolved_base_version,
        author="aria",
        rationale=f"Prompt: {prompt}",
        expected_impact={
            "summary": "User-directed patch generated from Ask Aria prompt."
        },
        ops=ops,
    )
    proposal_id = f"patch_{uuid4().hex[:10]}"
    now = _utc_now()
    existing = conn.execute(
        "SELECT 1 FROM workflows WHERE id = ?", (patch.workflow_id,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO workflows (id, name, graph_json, version, author, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                patch.workflow_id,
                workflow.get("name", patch.workflow_id),
                json.dumps(workflow),
                req.base_version,
                "aria",
                now,
                now,
            ),
        )
        conn.commit()
    db.save_proposal(
        proposal_id=proposal_id,
        workflow_id=patch.workflow_id,
        patch_json=json.dumps(patch.model_dump()),
        rationale=patch.rationale,
        created_at=now,
    )
    return {
        "proposal_id": proposal_id,
        "status": "pending",
        "proposal": patch.model_dump(),
        "ops_count": len(ops),
        "suggestions_used": suggestions[:3],
    }


def _generate_patch_impl(req: AskAriaPromptRequest) -> Dict[str, Any]:
    """Generate a deterministic patch proposal from prompt + workflow."""
    workflow = _canonicalize_workflow_payload(
        req.workflow.model_dump(), preserve_raw_ids=True
    )
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt is required")
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    source_ids = {e.get("source") for e in edges}
    sink_nodes = [n for n in nodes if n.get("id") not in source_ids]
    last_node = sink_nodes[-1] if sink_nodes else (nodes[-1] if nodes else None)
    has_output = any("output" in str(n.get("component_type", "")) for n in nodes)
    suggestions = suggest_components(workflow)
    approved = db.list_components(status="approved")
    incoming_edges, outgoing_edges = _build_edge_maps(edges)
    ops: List[Dict[str, Any]] = []
    lower = prompt.lower()

    _append_replace_ops(ops, lower, nodes, approved, suggestions)
    _append_remove_ops(ops, lower, nodes)
    _append_insert_front_ops(ops, lower, nodes, outgoing_edges, approved, suggestions)
    _append_relative_insert_ops(
        ops,
        lower,
        nodes,
        incoming_edges,
        outgoing_edges,
        approved,
        suggestions,
    )
    _append_connect_ops(ops, lower, nodes)
    _append_param_mutation_ops(ops, lower, nodes)
    _append_optimize_ops(ops, lower, nodes)
    _append_stability_fix_ops(
        ops,
        lower,
        workflow,
        nodes,
        incoming_edges,
        last_node,
        approved,
    )
    _append_split_branch_ops(ops, lower, nodes, incoming_edges)
    _append_fallback_ops(
        ops,
        lower,
        workflow,
        nodes,
        incoming_edges,
        last_node,
        approved,
        suggestions,
        has_output,
    )

    return _save_patch_proposal(req, workflow, prompt, ops, suggestions)


# ---------------------------------------------------------------------------
# Refine-winner + quality validation
# ---------------------------------------------------------------------------


def _fetch_parent_scores_for_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Fetch scores from the research leaderboard for the current workflow fingerprint."""
    wf = db.get_workflow(workflow_id)
    if not wf or not wf.get("graph_json"):
        return None
    graph = json.loads(wf["graph_json"])
    fingerprint = dig(graph, "metadata", "graph_fingerprint")
    if not fingerprint:
        return None
    try:
        from research.scientist.notebook import LabNotebook

        notebook_path = _PROJECT_ROOT / "research" / "lab_notebook.db"
        if not notebook_path.exists():
            return None
        nb = LabNotebook(str(notebook_path))
        try:
            res = nb.conn.execute(
                "SELECT result_id FROM program_results WHERE graph_fingerprint = ? ORDER BY timestamp DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
            if not res:
                return None
            return nb.get_leaderboard_entry(res[0])
        finally:
            nb.close()
    except Exception as e:
        logger.warning("Could not fetch parent scores from LabNotebook: %s", e)
        return None


@router.post("/refine-winner")
def refine_winner_endpoint(
    workflow_id: str, num_variations: int = 3, intent: Optional[str] = None
) -> Dict[str, Any]:
    """Generate evolutionary variations for a workflow with quality gate validation."""
    try:
        parent_scores = _fetch_parent_scores_for_workflow(workflow_id)
        raw_proposal_ids = refine_winner(
            workflow_id, num_variations * 2, intent=intent, parent_scores=parent_scores
        )
        valid_proposals: list[str] = []
        rejection_reasons: List[str] = []
        for pid in raw_proposal_ids:
            if len(valid_proposals) >= num_variations:
                break
            is_valid, error = _validate_proposal_quality(
                pid, parent_scores=parent_scores
            )
            if is_valid:
                valid_proposals.append(pid)
            else:
                logger.warning("Proposal %s failed quality gate: %s", pid, error)
                if error:
                    rejection_reasons.append(error)
        return {
            "success": True,
            "generated_proposals": valid_proposals,
            "parent_scores_found": bool(parent_scores),
            "validation_failures": len(raw_proposal_ids) - len(valid_proposals),
            "warning": (
                "All refinement candidates regressed against the parent guardrails."
                if raw_proposal_ids and not valid_proposals
                else None
            ),
            "rejection_reasons": rejection_reasons[:5],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _load_proposal_workflow_and_ops(
    proposal_id: str,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[str],
]:
    from ..database import get_proposal, get_workflow

    proposal = get_proposal(proposal_id)
    if not proposal:
        return None, None, [], "Proposal not found"
    workflow_row = get_workflow(proposal["workflow_id"])
    if not workflow_row:
        return None, None, [], "Workflow not found"
    workflow = json.loads(workflow_row["graph_json"])
    patch = json.loads(proposal["patch_json"])
    return proposal, workflow, patch.get("ops", []), None


def _compile_patched_refinement(
    workflow: Dict[str, Any], ops: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    from ..patcher import apply_patch_ops as _apply
    from aria_designer.runtime.compiler import compile_workflow as _rc

    try:
        patched_workflow = _apply(workflow, ops)
        components_dir = _PROJECT_ROOT / "aria_designer" / "components"
        if not components_dir.exists():
            components_dir = Path(__file__).parent.parent.parent / "components"
        _rc(patched_workflow, str(components_dir))
        return patched_workflow, None
    except Exception as exc:
        return None, f"Compilation failed: {exc}"


def _validate_snapshot_guardrails(
    parent_snapshot: Dict[str, Any],
    candidate_snapshot: Dict[str, Any],
    parent_scores: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not candidate_snapshot.get("valid", False):
        return f"Validation failed: {candidate_snapshot.get('error', 'unknown')}"
    smoke = candidate_snapshot.get("smoke_test")
    if smoke and not bool(smoke.get("ok", True)):
        return f"Smoke test failed: {smoke.get('error', 'unknown')}"
    if smoke and not bool(smoke.get("grad_flows", True)):
        return "Smoke test reported gradient regression"
    if smoke and not bool(smoke.get("no_nan", True)):
        return "Smoke test reported unstable outputs"
    if parent_snapshot.get("has_gradient_path") and not candidate_snapshot.get(
        "has_gradient_path"
    ):
        return "Refinement removed the parent gradient path"
    constraints = parse_intent_constraints(None, parent_scores)
    min_retention = getattr(constraints, "min_retention_ratio", 0.7) or 0.7
    parent_ops = int(parent_snapshot.get("n_ops", 0) or 0)
    if parent_ops <= 0:
        return None
    retention = float(candidate_snapshot.get("n_ops", 0)) / float(parent_ops)
    if retention < min_retention:
        return f"Refinement retained only {retention:.2%} of parent ops (floor {min_retention:.0%})"
    return None


def _count_non_io_ops(workflow: Dict[str, Any]) -> int:
    return len(
        [
            node
            for node in workflow.get("nodes", [])
            if str(node.get("component_type", "")).split("/")[0]
            not in ("graph_input", "graph_output", "io")
        ]
    )


def _validate_parent_regression_guardrails(
    workflow: Dict[str, Any],
    ops: List[Dict[str, Any]],
    parent_scores: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not parent_scores:
        return None
    parent_composite = float(parent_scores.get("composite_score") or 0.0)
    parent_tier = str(parent_scores.get("tier") or "screening")
    if (
        parent_tier not in ("investigation", "validation", "breakthrough")
        or parent_composite <= 100
    ):
        return None
    op_count = _count_non_io_ops(workflow)
    remove_ops = [op for op in ops if op.get("op") == "remove_node"]
    if op_count > 0 and len(remove_ops) / op_count > 0.3:
        return f"Rejected: removes {len(remove_ops)}/{op_count} ops from {parent_tier}-tier parent (composite {parent_composite:.1f})"
    predicted = parent_composite * _estimate_patch_quality_multiplier(
        ops, parent_scores
    )
    if predicted < (parent_composite * 0.95):
        return f"Rejected: predicted composite {predicted:.1f} falls below 95% of parent {parent_composite:.1f}"
    return None


def _validate_proposal_quality(
    proposal_id: str, parent_scores: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """Phase 0.2: Quality gate -- compile check + forward pass + regression check."""
    _, workflow, ops, load_error = _load_proposal_workflow_and_ops(proposal_id)
    if load_error:
        return False, load_error
    patched_workflow, compile_error = _compile_patched_refinement(workflow, ops)
    if compile_error:
        return False, compile_error
    parent_snapshot = _build_refinement_quality_snapshot(workflow)
    candidate_snapshot = _build_refinement_quality_snapshot(patched_workflow)
    guard_error = _validate_snapshot_guardrails(
        parent_snapshot, candidate_snapshot, parent_scores
    )
    if guard_error:
        return False, guard_error
    regression_error = _validate_parent_regression_guardrails(
        workflow, ops, parent_scores
    )
    if regression_error:
        return False, regression_error
    return True, None


def _build_refinement_quality_snapshot(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "valid": True,
        "n_ops": 0,
        "depth": 0,
        "has_gradient_path": True,
        "smoke_test": None,
    }
    try:
        if HAS_BRIDGE:
            from aria_designer.runtime.bridge import validate_workflow_graph

            result = validate_workflow_graph(workflow_json, model_dim=256)
            if not result.get("valid", False):
                return {
                    "valid": False,
                    "error": result.get("error", "bridge validation failed"),
                }
            gi = result.get("graph_info") or {}
            snapshot.update(
                {
                    "n_ops": int(gi.get("n_ops") or 0),
                    "depth": int(gi.get("depth") or 0),
                    "has_gradient_path": bool(gi.get("has_gradient_path", True)),
                }
            )
    except Exception as exc:
        logger.debug(
            "Bridge validation unavailable during refinement quality gate: %s", exc
        )
    snapshot["smoke_test"] = _run_native_refinement_smoke_test(workflow_json)
    return snapshot


def _run_native_refinement_smoke_test(
    workflow_json: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        from aria_core import smoke_test_graph  # type: ignore
    except Exception:
        logger.debug("aria_core smoke_test_graph not available", exc_info=True)
        return None
    try:
        result = smoke_test_graph(workflow_json, 256, 32)
    except TypeError:
        result = smoke_test_graph(workflow_json)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if isinstance(result, dict):
        return result
    return {
        k: getattr(result, k)
        for k in ("ok", "has_params", "grad_flows", "no_nan")
        if hasattr(result, k)
    }


def _estimate_patch_quality_multiplier(
    ops: List[Dict[str, Any]], parent_scores: Optional[Dict[str, Any]] = None
) -> float:
    """Estimate quality retention multiplier for a patch using shared penalty table."""
    penalty = sum(_MUTATION_PENALTIES.get(str(op.get("op") or ""), 0.04) for op in ops)
    constraints = parse_intent_constraints(None, parent_scores)
    if constraints.preserve_novelty:
        penalty *= 0.85
    tier = safe_str(dig(parent_scores, "tier"), lower=True)
    if tier in {"investigation", "validation", "breakthrough"}:
        penalty *= 1.15
    return max(0.75, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Proposals listing
# ---------------------------------------------------------------------------


@router.get("/proposals")
def list_proposals(
    status: Optional[str] = Query(None),
    workflow_id: Optional[str] = Query(None),
    fresh_only: bool = Query(False),
) -> List[Dict[str, Any]]:
    proposals = db.list_proposals(status=status)
    if workflow_id:
        proposals = [
            p for p in proposals if str(p.get("workflow_id") or "") == str(workflow_id)
        ]
    if fresh_only:
        versions: Dict[str, int] = {}
        filtered: List[Dict[str, Any]] = []
        for proposal in proposals:
            wf_id = str(proposal.get("workflow_id") or "")
            if not wf_id:
                continue
            cv = versions.get(wf_id)
            if cv is None:
                wf = db.get_workflow(wf_id)
                cv = int(wf.get("version") or 0) if wf else 0
                versions[wf_id] = cv
            bv = 0
            try:
                bv = int(
                    json.loads(proposal.get("patch_json") or "{}").get("base_version")
                    or 0
                )
            except Exception:
                logger.debug(
                    "Failed to parse base_version from proposal %s",
                    proposal.get("proposal_id", "unknown"),
                    exc_info=True,
                )
            if bv > 0 and cv > 0 and bv != cv:
                continue
            filtered.append(proposal)
        proposals = filtered
    return proposals


@router.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: str) -> Dict[str, Any]:
    """Get a single proposal by ID."""
    proposal = _require_proposal(proposal_id)
    proposal["patch"] = json.loads(proposal.pop("patch_json", "{}"))
    return proposal
