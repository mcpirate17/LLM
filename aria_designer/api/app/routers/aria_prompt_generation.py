from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from aria_designer.component_identity import canonicalize_component_id

from .. import database as db
from ..intent_parser import compute_insertion_point
from ..models import (
    AriaPatchProposalModel,
    AskAriaPromptRequest,
    utc_now_iso as _utc_now,
)
from ..suggestions import suggest_components
from ..workflow_support import get_approved_registry_ids
from .aria_workflow_utils import canonicalize_workflow_payload

logger = logging.getLogger(__name__)
router = APIRouter()
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
    workflow = canonicalize_workflow_payload(
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
