"""Experiment launch mode, eligibility, and preflight helpers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..trust_policy import TRUSTED_COMPARABILITY_LABELS, TRUSTED_TRUST_LABELS
from .deps import get_notebook

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..notebook import LabNotebook


_VALID_START_MODES = frozenset(
    {
        "single",
        "live_screening",
        "continuous",
        "evolve",
        "novelty",
        "investigation",
        "validation",
        "scale_up",
        "refine_fingerprint",
        "compact_synthesis",
        "sparse_morph",
    }
)


def resolve_scale_up_result_ids(
    nb: LabNotebook,
    result_ids: List[str],
    graph_fingerprints: List[str],
) -> Dict[str, Any]:
    """Resolve explicit result IDs and/or fingerprint prefixes for scale-up."""
    merged_result_ids: List[str] = []
    seen: set = set()
    for result_id in result_ids:
        if result_id in seen:
            continue
        seen.add(result_id)
        merged_result_ids.append(result_id)

    resolved: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for fingerprint in graph_fingerprints:
        rows = nb.conn.execute(
            """
            SELECT result_id, graph_fingerprint, experiment_id, stage1_passed,
                   loss_ratio, timestamp
            FROM program_results
            WHERE graph_fingerprint LIKE ?
            ORDER BY stage1_passed DESC,
                     (loss_ratio IS NULL) ASC,
                     loss_ratio ASC,
                     timestamp DESC
            LIMIT 5
            """,
            (f"{fingerprint}%",),
        ).fetchall()

        if not rows:
            unresolved.append(fingerprint)
            continue

        chosen = dict(rows[0])
        chosen_result_id = str(chosen.get("result_id") or "")
        if chosen_result_id and chosen_result_id not in seen:
            seen.add(chosen_result_id)
            merged_result_ids.append(chosen_result_id)

        candidates = [
            {
                "result_id": row["result_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "experiment_id": row["experiment_id"],
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
            }
            for row in rows
        ]
        resolved.append(
            {
                "requested_fingerprint": fingerprint,
                "selected_result_id": chosen.get("result_id"),
                "selected_graph_fingerprint": chosen.get("graph_fingerprint"),
                "selected_experiment_id": chosen.get("experiment_id"),
                "candidate_count": len(rows),
                "candidates": candidates,
            }
        )

    return {
        "result_ids": merged_result_ids,
        "resolved_fingerprints": resolved,
        "unresolved_fingerprints": unresolved,
    }


def _ineligible_from_missing_rows(
    result_id: str, program: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if program is None:
        return {
            "result_id": result_id,
            "reason": "result_not_found",
            "detail": "Result ID was not found in program results.",
        }
    if not bool(program.get("stage1_passed")):
        return {
            "result_id": result_id,
            "reason": "not_stage1_survivor",
            "detail": "Result exists but is not a Stage-1 survivor.",
        }
    return {
        "result_id": result_id,
        "reason": "not_in_leaderboard",
        "detail": "Result exists but has no leaderboard progression record.",
    }


def _program_is_screening_admissible(program: Optional[Dict[str, Any]]) -> bool:
    if not program or not bool(program.get("stage1_passed")):
        return False

    trust_label = str(program.get("trust_label") or "").strip().lower()
    comparability_label = str(program.get("comparability_label") or "").strip().lower()
    if (
        trust_label in TRUSTED_TRUST_LABELS
        and comparability_label in TRUSTED_COMPARABILITY_LABELS
    ):
        return True

    provenance_raw = program.get("data_provenance_json")
    if not provenance_raw:
        return False
    try:
        provenance = json.loads(provenance_raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(provenance.get("eligible_for_promotion"))


def _ensure_screening_leaderboard_entry(
    nb: LabNotebook,
    result_id: str,
    program: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    existing = nb.get_leaderboard_entry(result_id)
    if existing is not None:
        return existing

    graph_fingerprint = str((program or {}).get("graph_fingerprint") or "").strip()
    if graph_fingerprint:
        sibling = nb.get_leaderboard_entry_by_fingerprint(graph_fingerprint)
        if sibling is not None:
            return sibling

    if not _program_is_screening_admissible(program):
        return None

    program = program or {}
    if getattr(nb, "_read_only", False):
        return {
            "result_id": result_id,
            "tier": "screening",
            "investigation_passed": False,
            "validation_passed": False,
            "investigation_loss_ratio": None,
            "validation_loss_ratio": None,
        }

    entry_id = nb.upsert_leaderboard(
        result_id=result_id,
        model_source=str(program.get("model_source") or "screening_backfill"),
        architecture_desc=str(program.get("graph_fingerprint") or "")[:40],
        screening_loss_ratio=program.get("loss_ratio"),
        screening_novelty=program.get("novelty_score"),
        screening_passed=bool(program.get("stage1_passed")),
        tier="screening",
        trust_label=program.get("trust_label"),
        comparability_label=program.get("comparability_label"),
        novelty_confidence=program.get("novelty_confidence"),
        fp_jacobian_spectral_norm=program.get("fp_jacobian_spectral_norm"),
        routing_savings_ratio=program.get("routing_savings_ratio"),
        activation_sparsity_score=program.get("activation_sparsity_score"),
        depth_savings_ratio=program.get("depth_savings_ratio"),
        compression_ratio=program.get("compression_ratio"),
        wikitext_perplexity=program.get("wikitext_perplexity"),
        wikitext_score=program.get("wikitext_score"),
        notes="Auto-admitted to screening from trusted Stage-1 survivor.",
    )
    return nb.get_leaderboard_entry(result_id) or {
        "entry_id": entry_id,
        "tier": "screening",
    }


def _evaluate_mode_eligibility(
    mode: str, result_id: str, tier: str, lb: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if mode == "investigation":
        if tier == "screening":
            if lb.get("investigation_loss_ratio") is not None:
                return {
                    "result_id": result_id,
                    "reason": "already_investigated_unchanged",
                    "detail": "Investigation evidence already exists for this screening-tier result.",
                    "tier": tier,
                }
            return None
        return {
            "result_id": result_id,
            "reason": "not_screening_tier",
            "detail": f"Current tier is '{tier or 'unknown'}'; only screening tier can be investigated.",
            "tier": tier or None,
        }

    if mode == "validation":
        has_investigation_evidence = (
            lb.get("investigation_loss_ratio") is not None
            or lb.get("investigation_passed") is not None
        )
        if tier != "investigation" and not has_investigation_evidence:
            return {
                "result_id": result_id,
                "reason": "not_investigation_tier",
                "detail": f"Current tier is '{tier or 'unknown'}'; validation requires investigation tier.",
                "tier": tier or None,
            }
        if not bool(lb.get("investigation_passed")):
            return {
                "result_id": result_id,
                "reason": "not_investigation_passed",
                "detail": "Investigation evidence did not pass robustness gate.",
                "tier": tier,
            }
        return None

    return {
        "result_id": result_id,
        "reason": "unsupported_mode",
        "detail": f"Eligibility checks are not implemented for mode '{mode}'.",
    }


def build_start_mode_eligibility(
    nb: LabNotebook,
    mode: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """Validate candidate progression eligibility for start modes."""
    nb.flush_writes()
    payload: Dict[str, Any] = {
        "mode": mode,
        "requested_result_ids": list(result_ids),
        "eligible_result_ids": [],
        "ineligible": [],
        "all_eligible": False,
    }
    if not result_ids:
        return payload

    placeholders = ",".join("?" for _ in result_ids)
    leaderboard_rows = nb.conn.execute(
        f"""
        SELECT result_id, tier, investigation_passed, validation_passed,
               investigation_loss_ratio, validation_loss_ratio
        FROM leaderboard
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()
    program_rows = nb.conn.execute(
        f"""
        SELECT result_id, stage1_passed, graph_fingerprint, model_source, loss_ratio,
               novelty_score, novelty_confidence, fp_jacobian_spectral_norm,
               routing_savings_ratio, activation_sparsity_score,
               depth_savings_ratio, compression_ratio, wikitext_perplexity,
               wikitext_score, trust_label, comparability_label,
               data_provenance_json
        FROM program_results
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()

    leaderboard_by_id = {row["result_id"]: dict(row) for row in leaderboard_rows}
    program_by_id = {row["result_id"]: dict(row) for row in program_rows}

    for result_id in result_ids:
        lb = leaderboard_by_id.get(result_id)
        program = program_by_id.get(result_id)

        if lb is None:
            lb = _ensure_screening_leaderboard_entry(nb, result_id, program)
            if lb is not None:
                leaderboard_by_id[result_id] = lb
                tier = str(lb.get("tier") or "").lower()
                failure = _evaluate_mode_eligibility(mode, result_id, tier, lb)
                if failure is None:
                    payload["eligible_result_ids"].append(result_id)
                else:
                    payload["ineligible"].append(failure)
                continue
            payload["ineligible"].append(
                _ineligible_from_missing_rows(result_id, program)
            )
            continue

        tier = str(lb.get("tier") or "").lower()
        failure = _evaluate_mode_eligibility(mode, result_id, tier, lb)
        if failure is None:
            payload["eligible_result_ids"].append(result_id)
        else:
            payload["ineligible"].append(failure)

    payload["all_eligible"] = (
        len(payload["ineligible"]) == 0 and len(payload["eligible_result_ids"]) > 0
    )
    payload["summary"] = {
        "requested": len(result_ids),
        "eligible": len(payload["eligible_result_ids"]),
        "ineligible": len(payload["ineligible"]),
    }
    return payload


def normalize_start_mode(raw_mode: str) -> str:
    """Normalize and validate experiment start mode string."""
    mode = str(raw_mode or "single").strip().lower().replace("-", "_")
    if mode in _VALID_START_MODES:
        return mode
    return "single"


def run_launch_preflight(
    *,
    config,
    mode: str,
    prescreen: Dict[str, Any],
    notebook_path: str,
    sample_n: int = 4,
) -> Dict[str, Any]:
    """Run preflight checks before launching an experiment.

    Returns a dict with 'verdict' ('pass', 'warn', 'fail') and 'checks'.
    """
    checks: List[Dict[str, Any]] = []
    verdict = "pass"

    prescreen_warnings = prescreen.get("warnings", [])
    if prescreen_warnings:
        checks.append(
            {
                "name": "prescreen_warnings",
                "status": "warn",
                "details": prescreen_warnings,
            }
        )
        verdict = "warn"

    prescreen_blockers = prescreen.get("blockers", [])
    if prescreen_blockers:
        checks.append(
            {
                "name": "prescreen_blockers",
                "status": "fail",
                "details": prescreen_blockers,
            }
        )
        verdict = "fail"

    nb = get_notebook(notebook_path, read_only=True)
    try:
        active = nb.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'running'"
        ).fetchone()[0]
        if active > 0:
            checks.append(
                {
                    "name": "active_experiment",
                    "status": "warn",
                    "details": f"{active} experiment(s) marked as running",
                }
            )
            if verdict == "pass":
                verdict = "warn"
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)

    if not checks:
        checks.append({"name": "all_clear", "status": "pass", "details": None})

    return {"verdict": verdict, "checks": checks, "sample_n": sample_n}


def apply_compact_synthesis_bias(config) -> Dict[str, Any]:
    """Apply compact-synthesis mode biases to RunConfig.

    Returns dict of changes applied (for logging/response).
    """
    changes: Dict[str, Any] = {}
    if hasattr(config, "max_nodes") and (
        config.max_nodes is None or config.max_nodes > 12
    ):
        changes["max_nodes"] = {"from": config.max_nodes, "to": 12}
        config.max_nodes = 12
    if hasattr(config, "grammar_config") and config.grammar_config is not None:
        gc = config.grammar_config
        if hasattr(gc, "max_depth") and (gc.max_depth is None or gc.max_depth > 5):
            changes["grammar_max_depth"] = {"from": gc.max_depth, "to": 5}
            gc.max_depth = 5
    return changes


def apply_sparse_morph_bias(config) -> Dict[str, Any]:
    """Apply sparse-morph mode biases to RunConfig.

    Returns dict of changes applied.
    """
    changes: Dict[str, Any] = {}
    if hasattr(config, "grammar_config") and config.grammar_config is not None:
        gc = config.grammar_config
        if hasattr(gc, "sparsity_bias"):
            changes["sparsity_bias"] = {"from": gc.sparsity_bias, "to": 0.7}
            gc.sparsity_bias = 0.7
    return changes


def apply_live_screening_bias(config) -> Dict[str, Any]:
    """Apply live-screening biases to RunConfig.

    Enables the cheap pre-S1 gate for live screening experiments only.
    """
    changes: Dict[str, Any] = {}
    if hasattr(config, "enable_stage09_cheap_train_gate"):
        changes["enable_stage09_cheap_train_gate"] = {
            "from": bool(getattr(config, "enable_stage09_cheap_train_gate", False)),
            "to": True,
        }
        config.enable_stage09_cheap_train_gate = True
    return changes


def extract_hypothesis_missing_fields(critique: Optional[Dict[str, Any]]) -> List[str]:
    """Extract list of missing required fields from a hypothesis critique dict."""
    if not critique or not isinstance(critique, dict):
        return []
    missing = critique.get("missing_fields", [])
    if isinstance(missing, list):
        return [str(f) for f in missing if f]
    return []


_BRIEFING_MODE_MAP = {
    "synthesis": "single",
    "single": "single",
    "continuous": "continuous",
    "evolve": "evolve",
    "evolution": "evolve",
    "novelty": "novelty",
    "novelty_search": "novelty",
    "investigation": "investigation",
    "investigate": "investigation",
    "validation": "validation",
    "scale_up": "scale_up",
    "compact_synthesis": "compact_synthesis",
    "sparse_morph": "sparse_morph",
    "live_screening": "live_screening",
}


def normalize_briefing_mode(raw_mode: Optional[str]) -> Optional[str]:
    """Normalize LLM-suggested briefing mode to a valid start mode."""
    if not raw_mode:
        return None
    mode = str(raw_mode).strip().lower().replace("-", "_")
    return _BRIEFING_MODE_MAP.get(mode, mode if mode in _VALID_START_MODES else None)


def briefing_action_from_mode(mode: Optional[str]) -> Optional[str]:
    """Map a normalized mode to a briefing action key."""
    if not mode:
        return None
    action_map = {
        "single": "continuous",
        "continuous": "continuous",
        "evolve": "novelty_search",
        "novelty": "novelty_search",
        "live_screening": "continuous",
        "investigation": "investigate",
        "validation": "validate",
        "scale_up": "scale_up",
        "compact_synthesis": "compact_synthesis",
        "sparse_morph": "novelty_search",
    }
    return action_map.get(mode, mode)


def briefing_action_label(mode: Optional[str], hypothesis: Optional[str] = None) -> str:
    """Human-readable label for a briefing action."""
    label_map = {
        "single": "Run Synthesis",
        "continuous": "Continue Research",
        "evolve": "Run Evolution Search",
        "novelty": "Run Novelty Search",
        "live_screening": "Run Live Screening",
        "investigation": "Investigate Candidates",
        "validation": "Validate Candidates",
        "scale_up": "Scale Up",
        "compact_synthesis": "Run Compact Synthesis",
        "sparse_morph": "Run Sparse Morphology",
    }
    label = label_map.get(str(mode or ""), "Start Experiment")
    if hypothesis:
        label += f": {hypothesis[:60]}"
    return label


def augment_sparse_action_config(
    config: Optional[Dict[str, Any]],
    mode: Optional[str],
    sparse_coverage_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Augment a suggested config with sparse coverage hints when appropriate."""
    if config is None or not isinstance(config, dict):
        return config
    if mode not in ("novelty", "evolve", "single", "continuous"):
        return config

    sparse_share = float(sparse_coverage_data.get("sparse_share", 0))
    target_share = float(sparse_coverage_data.get("target_share", 0.15))
    if sparse_share < target_share:
        config.setdefault("morph_focus_sparse", True)
        config.setdefault("morph_sparse_weight_storage", "semi_structured_2_4")
    return config


def run_pipeline_sample_check(*, config, sample_n: int = 5) -> Dict[str, Any]:
    """Run a quick pipeline sample check: generate, compile, test S0.

    Returns dict with 'generated', 'compiled', 'passed_s0', 'errors'.
    """
    generated = 0
    compiled = 0
    passed_s0 = 0
    errors: List[str] = []

    try:
        from ...synthesis.grammar import GrammarConfig, random_graph
        from ...synthesis.compiler import compile_model

        gc = GrammarConfig()
        for _ in range(sample_n):
            try:
                graph = random_graph(gc)
                generated += 1
                model = compile_model([graph], vocab_size=256, max_seq_len=64)
                compiled += 1
                import torch

                x = torch.randint(0, 256, (1, 16))
                out = model(x)
                if out is not None:
                    passed_s0 += 1
            except Exception as exc:
                errors.append(str(exc)[:200])
    except ImportError as exc:
        errors.append(f"Import error: {exc}")

    return {
        "generated": generated,
        "compiled": compiled,
        "passed_s0": passed_s0,
        "errors": errors,
    }
