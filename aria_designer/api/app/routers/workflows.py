from __future__ import annotations

import json
import logging
import os
import time as _time_mod
from collections import deque
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..component_identity import (
    canonicalize_workflow_ids,
)
from ..config import settings
from ..benchmark_targets import benchmark_target_catalog, build_benchmark_analysis
from ..diff import diff_graphs
from ..models import (
    CompileWorkflowRequest,
    PatchOpModel,
    RunWorkflowRequest,
    ValidateWorkflowRequest,
    ValidateWorkflowResponse,
    ValidationIssue,
    WorkflowGraphModel,
    utc_now_iso as _utc_now,
)
from ..shared_api import (
    HAS_BRIDGE,
    HAS_PROFILER,
    KernelDispatcher,
    _auto_promote_workflow_to_research,
    _collect_workflow_semantic_warnings,
    _require_run,
    _require_workflow,
    _sync_lineage_to_research,
    bridge_estimate,
    bridge_list_primitives,
    bridge_profile,
    bridge_validate,
    find_unsupported_edge_dtype_pairings,
    runtime_compile,
)
from research.defaults import MODEL_DIM, VOCAB_SIZE
from research.perf_contract import emit_perf_artifact
from ..type_utils import dig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["workflows"])


def _canonicalize_request_workflow(workflow: WorkflowGraphModel) -> tuple[WorkflowGraphModel, dict, set[str]]:
    registry_ids = db.list_component_types(status="approved")
    payload = workflow.model_dump()
    canonicalize_workflow_ids(payload, registry_ids)
    return WorkflowGraphModel.model_validate(payload), payload, registry_ids


def _unresolved_component_issues(workflow_payload: dict, registry_ids: set[str]) -> List[ValidationIssue]:
    unresolved: List[ValidationIssue] = []
    for node in workflow_payload.get("nodes", []):
        component_type = str(node.get("component_type") or "").strip().lower()
        if component_type and component_type not in registry_ids:
            unresolved.append(ValidationIssue(
                severity="error",
                code="unknown_component",
                node_id=node.get("id"),
                message=(
                    f"Node {node.get('id')}: unresolved component type "
                    f"'{node.get('component_type')}'."
                ),
            ))
    return unresolved


# ── Helpers ────────────────────────────────────────────────────────────

def _validate_fallback_cycles(
    node_ids: set, workflow: Any, issues: List[ValidationIssue],
) -> None:
    """Cycle detection via DFS (fallback when native validator unavailable)."""
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in workflow.edges:
        if edge.source in adj:
            adj[edge.source].append(edge.target)

    visited, in_stack = set(), set()

    def has_cycle(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in in_stack:
                return True
            if neighbor not in visited and has_cycle(neighbor):
                return True
        in_stack.discard(node)
        return False

    for nid in node_ids:
        if nid not in visited and has_cycle(nid):
            issues.append(ValidationIssue(
                severity="error", code="cycle_detected",
                message="Workflow contains a cycle.",
            ))
            break



def _check_dead_branches(
    workflow: Any, node_ids: set, issues: List[ValidationIssue],
) -> None:
    """Detect nodes that do not connect to any output node."""
    output_types = {"output", "output_head", "graph_output"}
    output_node_ids = {
        node.id
        for node in workflow.nodes
        if (node.component_type or "").split("/")[-1] in output_types
    }
    if not output_node_ids:
        return

    reverse_adj: Dict[str, List[str]] = {node.id: [] for node in workflow.nodes}
    for edge in workflow.edges:
        if edge.source in node_ids and edge.target in node_ids:
            reverse_adj[edge.target].append(edge.source)

    reachable: set = set()
    queue = deque(output_node_ids)
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for source in reverse_adj.get(current, []):
            if source not in reachable:
                queue.append(source)

    for node in workflow.nodes:
        if node.id not in reachable:
            issues.append(ValidationIssue(
                node_id=node.id,
                severity="error",
                code="dead_branch",
                message="Dead branch detected. Node does not connect to the final output.",
            ))


def _check_dtype_compatibility(
    workflow: Any, workflow_payload: dict, comp_cache: dict,
    issues: List[ValidationIssue],
) -> None:
    """Validate manifest-port dtype compatibility for all edges."""
    if find_unsupported_edge_dtype_pairings is not None:
        dtype_issues = find_unsupported_edge_dtype_pairings(
            workflow_payload,
            lambda component_type: db.get_component(component_type),
        )
        for issue in dtype_issues:
            issues.append(ValidationIssue(
                severity="error",
                code="unsupported_edge_dtype_pairing",
                message=issue["message"],
                edge_id=issue.get("edge_id") or None,
            ))
    else:
        for edge in workflow.edges:
            src_comp = comp_cache.get(edge.source)
            tgt_comp = comp_cache.get(edge.target)
            if not src_comp or not tgt_comp:
                continue

            src_port = next((p for p in src_comp.get("outputs", []) if p["name"] == edge.source_port), None)
            tgt_port = next((p for p in tgt_comp.get("inputs", []) if p["name"] == edge.target_port), None)
            if src_port and tgt_port and src_port["dtype"] != tgt_port["dtype"]:
                pair = (src_port["dtype"], tgt_port["dtype"])
                # Allow implicit complex_tensor <-> tensor conversion
                if pair not in (("complex_tensor", "tensor"), ("tensor", "complex_tensor")):
                    issues.append(ValidationIssue(
                        severity="error",
                        code="unsupported_edge_dtype_pairing",
                        message=(
                            f"Unsupported edge dtype pairing on edge {edge.id}: "
                            f"{edge.source}.{edge.source_port} ({src_port['dtype']}) -> "
                            f"{edge.target}.{edge.target_port} ({tgt_port['dtype']}). "
                            "Supported pairings currently require matching source/target dtypes."
                        ),
                        edge_id=edge.id,
                    ))


# ── Workflows ─────────────────────────────────────────────────────────

@router.post("/workflows/validate", response_model=ValidateWorkflowResponse)
def validate_workflow(req: ValidateWorkflowRequest) -> ValidateWorkflowResponse:
    """Validate a workflow graph (structure, types, constraints)."""
    workflow, workflow_payload, registry_ids = _canonicalize_request_workflow(req.workflow)
    issues: List[ValidationIssue] = []

    node_ids = {node.id for node in workflow.nodes}
    if len(node_ids) != len(workflow.nodes):
        issues.append(ValidationIssue(
            severity="error", code="duplicate_node_id",
            message="Workflow contains duplicate node ids.",
        ))

    for edge in workflow.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            issues.append(ValidationIssue(
                severity="error", code="dangling_edge",
                message=f"Edge {edge.id} references missing source/target node.",
            ))

    issues.extend(_unresolved_component_issues(workflow_payload, registry_ids))

    # Validate component types exist in registry and cache them
    comp_cache = {}
    for node in workflow.nodes:
        comp = db.get_component(node.component_type)
        if comp is None:
            continue
        else:
            comp_cache[node.id] = comp

    # Validate manifest-port dtype compatibility for all edges.
    _check_dtype_compatibility(workflow, workflow_payload, comp_cache, issues)

    # Dead-branch detection
    _check_dead_branches(workflow, node_ids, issues)

    # Cycle detection and graph structure using native C validator if available
    if KernelDispatcher:
        try:
            dispatcher = KernelDispatcher()
            node_ids_list = [node.id for node in workflow.nodes]
            node_to_idx = {nid: i for i, nid in enumerate(node_ids_list)}
            c_edges = []
            for edge in workflow.edges:
                if edge.source in node_to_idx and edge.target in node_to_idx:
                    c_edges.append((node_to_idx[edge.source], node_to_idx[edge.target], 0, 0))

            res = dispatcher.validate_graph(node_ids_list, c_edges)
            if not res['valid']:
                code = "cycle_detected" if res.get('code') in [-3, -7] else "native_validator_error"
                issues.append(ValidationIssue(
                    severity="error", code=code,
                    message=f"Native validation failed: {res['error']}",
                ))
        except Exception as e:
            logger.error("Native validator failed, falling back: %s", e)
            _validate_fallback_cycles(node_ids, workflow, issues)
    else:
        _validate_fallback_cycles(node_ids, workflow, issues)

    return ValidateWorkflowResponse(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
    )


@router.post("/workflows/compile")
def compile_workflow(req: CompileWorkflowRequest) -> Dict[str, Any]:
    semantic_warnings = _collect_workflow_semantic_warnings(req.workflow.model_dump())
    if runtime_compile is None:
        return {
            "compiled": False,
            "error": "Runtime compiler not available",
            "workflow_id": req.workflow.workflow_id,
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }

    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "components"))
        model = runtime_compile(req.workflow.model_dump(), components_dir)

        return {
            "compiled": True,
            "target": req.target,
            "workflow_id": req.workflow.workflow_id,
            "node_count": len(req.workflow.nodes),
            "submodule_count": len(model.submodules),
            "notes": "Workflow compiled successfully to torch.nn.Module",
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }
    except Exception as e:
        logger.error("Compilation failed: %s", e)
        return {
            "compiled": False,
            "error": str(e),
            "workflow_id": req.workflow.workflow_id,
            "semantic_warnings": semantic_warnings,
            "semantic_warning_count": len(semantic_warnings),
        }


@router.post("/workflows/preview")
def preview_workflow(req: CompileWorkflowRequest) -> Dict[str, Any]:
    """Run a forward pass with dummy data and return intermediate shapes/stats."""
    if not runtime_compile:
        return {"error": "Runtime not available"}

    try:
        components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "components"))
        model = runtime_compile(req.workflow.model_dump(), components_dir)

        # Generate dummy inputs
        inputs = {}
        sources = [n.id for n in req.workflow.nodes if not any(e.target == n.id for e in req.workflow.edges)]
        import torch
        for nid in sources:
            inputs[nid] = torch.randn(1, 16, 64)  # Default dummy

        outputs = model(inputs)

        results = {}
        for nid, val in outputs.items():
            if isinstance(val, torch.Tensor):
                results[nid] = {
                    "shape": list(val.shape),
                    "mean": float(val.mean()) if val.numel() > 0 else 0.0,
                    "std": float(val.std()) if val.numel() > 0 else 0.0,
                }
            elif hasattr(val, "__len__"):
                results[nid] = {"type": type(val).__name__, "size": len(val)}
            else:
                results[nid] = {"type": type(val).__name__, "value": str(val)}

        return {"success": True, "results": results}
    except Exception as e:
        logger.error("Preview failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/workflows/run")
def run_workflow(req: RunWorkflowRequest) -> Dict[str, Any]:
    run_id = f"run_{uuid4().hex[:10]}"
    return {
        "accepted": True,
        "run_id": run_id,
        "workflow_id": req.workflow.workflow_id,
        "budget": req.budget,
        "notes": "Scaffold run path; executor integration pending.",
    }


@router.put("/workflows/{workflow_id}")
def save_workflow(workflow_id: str, workflow: WorkflowGraphModel) -> Dict[str, Any]:
    """Save or update a workflow."""
    now = _utc_now()
    registry_ids = db.list_component_types(status="approved")
    wf_dict = workflow.model_dump()
    canonicalize_workflow_ids(wf_dict, registry_ids, preserve_raw_ids=True)
    unresolved = _unresolved_component_issues(wf_dict, registry_ids)
    if unresolved:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Workflow contains unresolved component IDs.",
                "issues": [issue.model_dump() for issue in unresolved],
            },
        )
    old_fingerprint = None
    try:
        existing = db.get_workflow(workflow_id)
        if existing and existing.get("graph_json"):
            existing_graph = json.loads(existing["graph_json"])
            old_fingerprint = dig(existing_graph, "metadata", "graph_fingerprint")
    except Exception:
        logger.warning("Failed to load existing workflow fingerprint for %s", workflow_id, exc_info=True)
        old_fingerprint = None

    # Calculate fingerprint if bridge is available
    fingerprint = None
    if HAS_BRIDGE:
        try:
            from runtime.bridge import workflow_to_graph as _w2g
            model_dim = wf_dict.get("metadata", {}).get("model_dim", 256)
            graph, _ = _w2g(wf_dict, model_dim, return_id_map=True)
            fingerprint = graph.fingerprint()
            meta = wf_dict.setdefault("metadata", {})
            meta["graph_fingerprint"] = fingerprint
            if old_fingerprint and fingerprint and old_fingerprint != fingerprint:
                meta["parent_fingerprint"] = old_fingerprint
        except Exception as e:
            logger.warning("Could not calculate fingerprint for saved workflow: %s", e)

    version = db.save_workflow(
        workflow_id=workflow_id,
        name=workflow.name,
        graph_json=json.dumps(wf_dict),
        author="user",
        created_at=now,
        updated_at=now,
    )

    # Sync to research notebook if enabled
    if settings.LINEAGE_SYNC_ENABLED:
        try:
            lineage_payload = {
                "run_id": f"save_{uuid4().hex[:10]}",
                "workflow_id": workflow_id,
                "workflow_version": version,
                "graph_fingerprint": fingerprint,
                "status": "saved",
                "source": "aria_designer",
                "total_time_ms": 0,
                "metrics": {
                    "node_count": len(wf_dict.get("nodes", [])),
                    "edge_count": len(wf_dict.get("edges", [])),
                },
                "payload": wf_dict,
                "created_at": _time_mod.time(),
            }
            _sync_lineage_to_research(lineage_payload)
        except Exception as e:
            logger.warning("Failed to sync saved workflow to research: %s", e)

    # Auto-promote fingerprints into research discoveries
    promoted = None
    fingerprint_changed = bool(fingerprint and old_fingerprint and fingerprint != old_fingerprint)
    should_promote = bool(fingerprint)
    if should_promote:
        promoted = _auto_promote_workflow_to_research(wf_dict)

    return {
        "workflow_id": workflow_id,
        "version": version,
        "saved_at": now,
        "fingerprint": fingerprint,
        "fingerprint_changed": fingerprint_changed,
        "parent_fingerprint": old_fingerprint if fingerprint_changed else None,
        "auto_promoted": bool(promoted),
        "promoted_result_id": (promoted or {}).get("result_id") if promoted else None,
    }


@router.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = _require_workflow(workflow_id)
    wf["graph"] = json.loads(wf.pop("graph_json"))
    return wf


@router.get("/workflows")
def list_workflows() -> List[Dict[str, Any]]:
    return db.list_workflows()


@router.post("/workflows/diff")
def post_diff_workflows(wf_a: WorkflowGraphModel, wf_b: WorkflowGraphModel) -> List[PatchOpModel]:
    return diff_graphs(wf_a.model_dump(), wf_b.model_dump())


# ── Perf / Benchmarks / Profile / Estimate ────────────────────────────

@router.get("/perf/summary")
def get_perf_summary(limit: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    """Return recent designer perf artifacts and aggregate summary."""
    from research.perf_contract import list_recent_perf_artifacts, summarize_perf_artifacts
    artifacts = list_recent_perf_artifacts(component="aria_designer", limit=limit)
    return {
        "summary": summarize_perf_artifacts(artifacts, component="aria_designer"),
        "artifacts": artifacts,
    }


@router.get("/benchmarks/targets")
def get_benchmark_targets(
    run_id: Optional[str] = Query(None, description="Optional run_id for live target comparison"),
) -> Dict[str, Any]:
    """Return benchmark target catalog and optional run-specific comparison."""
    payload: Dict[str, Any] = benchmark_target_catalog()
    if run_id:
        run = _require_run(run_id)
        result_payload = run.get("result") or {}
        benchmarking = result_payload.get("benchmarking")
        if not benchmarking:
            stage_metrics = {
                name: (stage.get("metrics") if isinstance(stage, dict) else None)
                for name, stage in (run.get("stages") or {}).items()
                if isinstance(stage, dict)
            }
            benchmarking = build_benchmark_analysis(stage_metrics)
        payload["run_id"] = run_id
        payload["analysis"] = benchmarking
    return payload


@router.post("/workflows/profile")
def profile_workflow_endpoint(req: RunWorkflowRequest) -> Dict[str, Any]:
    """Profile a workflow: FLOPs, memory, latency, bottleneck analysis."""
    if not HAS_PROFILER:
        raise HTTPException(status_code=501, detail="Profiler not available")
    wf = req.workflow.model_dump()
    budget = req.budget
    report = bridge_profile(
        wf,
        model_dim=budget.get("model_dim", MODEL_DIM),
        device=budget.get("device", "cpu"),
        runtime=budget.get("runtime", False),
        vocab_size=budget.get("vocab_size", VOCAB_SIZE),
        batch_size=budget.get("batch_size", 2),
        seq_len=budget.get("seq_len", 128),
    )
    payload = report.to_dict()
    perf_contract = payload.get("perf_contract")
    if isinstance(perf_contract, dict):
        perf_contract["identity"] = {
            **(perf_contract.get("identity") or {}),
            "workflow_id": wf.get("workflow_id"),
            "runtime_enabled": bool(budget.get("runtime", False)),
        }
        perf_artifact_path = emit_perf_artifact(
            perf_contract,
            slug=f"profile_{wf.get('workflow_id') or uuid4().hex[:12]}",
        )
        perf_contract["artifact_path"] = perf_artifact_path
        payload["perf_contract"] = perf_contract
        payload["perf_artifact_path"] = perf_artifact_path
        payload["perf_budget_gate"] = perf_contract.get("budget_verdict")
    return payload


@router.post("/workflows/validate-graph")
def validate_workflow_graph_endpoint(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    """Validate that a workflow maps to a valid ComputationGraph in the research pipeline."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    wf = req.workflow.model_dump()
    return bridge_validate(wf, model_dim=req.workflow.metadata.get("model_dim", 256))


@router.get("/primitives")
def list_primitives() -> List[Dict[str, Any]]:
    """List all available primitives from the research pipeline."""
    if not HAS_BRIDGE:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    return bridge_list_primitives()


@router.post("/workflows/estimate")
def estimate_workflow(req: ValidateWorkflowRequest) -> Dict[str, Any]:
    """Estimate params, FLOPs, and memory for a workflow."""
    # Use bridge for accurate graph-aware estimation if available
    if HAS_BRIDGE:
        wf = req.workflow.model_dump()
        model_dim = req.workflow.metadata.get("model_dim", 256)
        result = bridge_estimate(wf, model_dim=model_dim)
        result["workflow_id"] = req.workflow.workflow_id
        result["node_count"] = len(req.workflow.nodes)
        result["edge_count"] = len(req.workflow.edges)
        return result

    # Fallback: manifest-based estimation
    total_params = 0
    for node in req.workflow.nodes:
        comp = db.get_component(node.component_type)
        if comp and comp.get("performance", {}).get("has_params"):
            formula = comp["performance"].get("param_formula", "0")
            try:
                val = eval(formula, {"__builtins__": {}}, {"D": 256, "D_in": 256, "D_out": 256, "vocab_size": 32000})
                total_params += int(val)
            except Exception:
                logger.warning("Failed to evaluate param_formula for component %s: %s", node.component_type, formula, exc_info=True)

    return {
        "workflow_id": req.workflow.workflow_id,
        "estimated_params": total_params,
        "node_count": len(req.workflow.nodes),
        "edge_count": len(req.workflow.edges),
    }
