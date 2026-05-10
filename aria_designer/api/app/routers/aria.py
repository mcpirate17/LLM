from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..historical_insights import build_historical_insights_response
from ..models import (
    AriaPatchProposalModel,
    HistoricalInsightsResponse,
    RecordOutcomeRequest,
    SuggestComponentsRequest,
    utc_now_iso as _utc_now,
)
from ..suggestions import suggest_components
from ..mutation import refine_winner, _MUTATION_PENALTIES

HAS_SUGGESTIONS = True  # kept for test monkeypatching compatibility
from ..research_signals import (
    fetch_research_recommendation_signals,
)
from ..intent_parser import parse_intent_constraints
from ..runtime_features import HAS_BRIDGE, _PROJECT_ROOT
from ..workflow_support import (
    _require_proposal,
)
from ..type_utils import dig, safe_str
from .aria_apply_patch import router as apply_patch_router
from .aria_prompt_generation import router as prompt_generation_router

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/aria", tags=["aria"])
router.include_router(apply_patch_router)
router.include_router(prompt_generation_router)


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
                "SELECT result_id FROM program_results_compat WHERE graph_fingerprint = ? ORDER BY timestamp DESC LIMIT 1",
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
