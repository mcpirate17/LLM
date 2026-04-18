from __future__ import annotations

"""Pure flow helpers for auto-escalation stages."""

import time
import uuid
from typing import Any, Dict, Iterable, List, Sequence

from ..thresholds import (
    HELLASWAG_RANDOM_CHANCE_GATE,
    UNDERSTANDING_MIN_BINDING,
    UNDERSTANDING_MIN_DIAGNOSTIC,
    UNDERSTANDING_MIN_HELLASWAG,
    UNDERSTANDING_MIN_SIGNALS,
    UNDERSTANDING_SOFT_BINDING,
    UNDERSTANDING_SOFT_DIAGNOSTIC,
    VALIDATION_BEST_LR_HARD,
)
from ._types import RunConfig


def merge_unique_result_rows(
    primary_rows: Sequence[Dict[str, Any]],
    extra_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = list(primary_rows)
    seen = {row.get("result_id") for row in merged if row.get("result_id")}
    for row in extra_rows:
        result_id = row.get("result_id")
        if not result_id or result_id in seen:
            continue
        merged.append(row)
        seen.add(result_id)
    return merged


def filter_uninvestigated_rows(
    rows: Sequence[Dict[str, Any]],
    investigated_fingerprints: Iterable[str],
) -> tuple[List[Dict[str, Any]], int]:
    investigated = set(investigated_fingerprints)
    if not investigated:
        return list(rows), 0
    kept = [row for row in rows if row.get("graph_fingerprint") not in investigated]
    return kept, len(rows) - len(kept)


def screening_candidates_above_threshold(
    rows: Sequence[Dict[str, Any]],
    score_map: Dict[str, float],
    threshold: float,
) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("result_id")
        and score_map.get(str(row["result_id"]), 0.0) >= threshold
    ]


def build_selected_screening_ids(
    ranked_rows: Sequence[Dict[str, Any]],
    rows_by_id: Dict[str, Dict[str, Any]],
    *,
    limit: int,
) -> List[str]:
    selected_ids: List[str] = []
    for item in ranked_rows:
        result_id = item.get("result_id")
        row = rows_by_id.get(str(result_id or ""))
        if row is None or not row.get("stage1_passed"):
            continue
        selected_ids.append(str(result_id))
        if len(selected_ids) >= limit:
            break
    return selected_ids


def build_selection_decision_payload(
    *,
    context: str,
    experiment_id: str | None,
    selection: Dict[str, Any],
    candidate_ids: Sequence[str],
    scored_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "decision_id": str(uuid.uuid4())[:12],
        "timestamp": time.time(),
        "context": context,
        "experiment_id": experiment_id,
        "candidate_pool_summary": selection.get("summary", {}),
        "score_breakdown": selection.get("scored", []),
        "policy": selection.get("policy", {}),
        "reason": selection.get("reason", ""),
        "chosen_experiments": [
            {
                "result_id": result_id,
                "family": (row := scored_by_id.get(result_id, {})).get("family"),
                "score": row.get("score"),
            }
            for result_id in candidate_ids
        ],
        "trigger": None,
    }


def sparse_dense_learning_signal(
    rows: Sequence[Dict[str, Any]],
) -> tuple[float, float] | None:
    sparse_rows = [row for row in rows if (row.get("sparsity_ratio") or 0) > 0.3]
    dense_rows = [row for row in rows if (row.get("sparsity_ratio") or 0) <= 0.3]
    if not sparse_rows or not dense_rows:
        return None
    avg_sparse = sum(row.get("loss_ratio", 1.0) for row in sparse_rows) / len(
        sparse_rows
    )
    avg_dense = sum(row.get("loss_ratio", 1.0) for row in dense_rows) / len(dense_rows)
    return avg_sparse, avg_dense


def understanding_gate_metrics(
    understanding: Dict[str, float],
) -> tuple[bool, float, float, float]:
    """Investigation→validation gate.

    Requires at least UNDERSTANDING_MIN_SIGNALS (default: 2) of the three
    capability signals to clear their *strict* thresholds. This replaces the
    pre-2026-04-17 OR-gate, which let a single barely-above-random signal
    promote a model that lacked all other capabilities.
    """
    diagnostic = float(understanding.get("diagnostic_score", 0.0) or 0.0)
    binding_composite = (
        0.4 * float(understanding.get("ar_auc", 0.0) or 0.0)
        + 0.3 * float(understanding.get("induction_auc", 0.0) or 0.0)
        + 0.3 * float(understanding.get("binding_auc", 0.0) or 0.0)
    )
    hellaswag = float(understanding.get("hellaswag_acc", 0.0) or 0.0)
    signals_passed = sum(
        (
            diagnostic >= UNDERSTANDING_MIN_DIAGNOSTIC,
            binding_composite >= UNDERSTANDING_MIN_BINDING,
            hellaswag >= UNDERSTANDING_MIN_HELLASWAG,
        )
    )
    passes = signals_passed >= UNDERSTANDING_MIN_SIGNALS
    return passes, diagnostic, binding_composite, hellaswag


def screening_understanding_filter(
    understanding: Dict[str, float],
) -> tuple[bool, str]:
    """Screening→investigation filter (best-effort).

    Probe scores are usually NULL at screening time (probes run during
    investigation). This filter only blocks a candidate when all three
    capability signals have been measured AND all are below the *soft*
    floors — i.e., this candidate has been investigated before and is
    known-incapable. Returns ``(allow, reason)``.

    Allow when:
      - any signal is None/missing (no measurement) — let normal screening proceed
      - any one signal exceeds its soft floor — capability evidence exists
    Block when:
      - all three measurements are present AND below their soft floors
    """
    diag_raw = understanding.get("diagnostic_score")
    ar_raw = understanding.get("ar_auc")
    ind_raw = understanding.get("induction_auc")
    bind_raw = understanding.get("binding_auc")
    hella_raw = understanding.get("hellaswag_acc")

    if diag_raw is None or hella_raw is None or (
        ar_raw is None and ind_raw is None and bind_raw is None
    ):
        return True, "no_probe_data"

    diagnostic = float(diag_raw or 0.0)
    binding_composite = (
        0.4 * float(ar_raw or 0.0)
        + 0.3 * float(ind_raw or 0.0)
        + 0.3 * float(bind_raw or 0.0)
    )
    hellaswag = float(hella_raw or 0.0)

    if (
        diagnostic < UNDERSTANDING_SOFT_DIAGNOSTIC
        and binding_composite < UNDERSTANDING_SOFT_BINDING
        and hellaswag <= HELLASWAG_RANDOM_CHANCE_GATE
    ):
        return False, (
            f"all_signals_near_zero(diag={diagnostic:.3f},"
            f"bind={binding_composite:.3f},hella={hellaswag:.3f})"
        )
    return True, "above_soft_floor"


def strong_investigation_candidates(
    *,
    inv_results: Sequence[Dict[str, Any]],
    novelty_meta: Dict[str, Dict[str, Any]],
    composite_scores: Dict[str, float],
    replication_info: Dict[str, Dict[str, Any]],
    understanding_data: Dict[str, Dict[str, float]],
    min_score: float,
    config: RunConfig,
    threshold_for_replication,
    meets_empirical_override,
    logger,
) -> tuple[List[Dict[str, Any]], int]:
    strong: List[Dict[str, Any]] = []
    blocked_incomplete_fingerprint = 0
    for row in inv_results:
        result_id = str(row.get("result_id") or "")
        if not result_id:
            continue
        meta = novelty_meta.get(result_id, {})
        if not bool(meta.get("fingerprint_completed_post_investigation")):
            blocked_incomplete_fingerprint += 1
            logger.warning(
                "escalation_blocked_fingerprint_incomplete: result_id=%s cka_source=%s",
                result_id[:12],
                meta.get("cka_source", "unknown"),
            )
            continue
        if not bool(meta.get("novelty_valid_for_promotion")):
            logger.info(
                "escalation_blocked_novelty_invalid: result_id=%s reason=%s cka_source=%s",
                result_id[:12],
                meta.get("novelty_validity_reason", "unknown"),
                meta.get("cka_source", "unknown"),
            )
            continue

        candidate_score = composite_scores.get(result_id, 0.0)
        replication = replication_info.get(result_id, {"n": 1, "loss_std": 0.0})
        replication_n = int(replication["n"])
        if min_score > 0:
            effective_threshold = threshold_for_replication(
                min_score=min_score,
                replication_n=replication_n,
                loss_std=float(replication["loss_std"]),
            )
            if candidate_score < effective_threshold:
                logger.info(
                    "Auto-validate: %s rejected (score %.1f < threshold %.1f, base=%.1f, n=%d, loss_std=%.4f)",
                    result_id[:12],
                    candidate_score,
                    effective_threshold,
                    min_score,
                    replication_n,
                    float(replication["loss_std"]),
                )
                continue

        best_loss_ratio = row.get("best_loss_ratio")
        if best_loss_ratio is not None and float(best_loss_ratio) > 1.5:
            logger.warning(
                "loss_ratio_sanity_check: result_id=%s best_loss_ratio=%.4f > 1.5 — possible NORM/RAW confusion. Skipping candidate.",
                result_id[:12],
                float(best_loss_ratio),
            )
            continue

        baseline_loss_ratio = row.get("baseline_loss_ratio")
        baseline_gate_passed = (
            baseline_loss_ratio is not None
            and float(baseline_loss_ratio) < config.auto_validate_max_baseline_ratio
        )
        empirical_override = meets_empirical_override(
            row,
            candidate_score,
            min_score,
        )
        passes_understanding, diagnostic, binding_composite, hellaswag = (
            understanding_gate_metrics(understanding_data.get(result_id, {}))
        )
        if not passes_understanding:
            logger.info(
                "escalation_blocked_no_understanding: result_id=%s diag=%.3f bind_comp=%.3f hella=%.3f",
                result_id[:12],
                diagnostic,
                binding_composite,
                hellaswag,
            )
            continue

        if (
            row.get("robustness", 0) >= config.auto_validate_min_robustness
            and (best_loss_ratio or 1.0) < VALIDATION_BEST_LR_HARD
            and (baseline_gate_passed or empirical_override)
            and not row.get("brittle_risk", False)
            and (
                row.get("loss_ratio_multiplier") is None
                or row.get("loss_ratio_multiplier")
                <= config.investigation_max_loss_ratio_multiplier
            )
        ):
            strong.append(row)
    return strong, blocked_incomplete_fingerprint


def prepare_validation_candidates(
    strong_rows: Sequence[Dict[str, Any]],
    graph_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for row in strong_rows:
        result_id = str(row.get("result_id") or "")
        if not result_id:
            continue
        meta = graph_meta.get(result_id, {})
        prepared.append(
            {
                "result_id": result_id,
                "graph_json": meta.get("graph_json"),
                "routing_mode": meta.get("routing_mode"),
                "loss_ratio": row.get("best_loss_ratio"),
                "baseline_loss_ratio": row.get("baseline_loss_ratio"),
                "novelty_score": row.get("novelty_confidence"),
                "throughput_tok_s": row.get("throughput_tok_s"),
                "flops_per_token": row.get("flops_per_token"),
                "peak_memory_mb": row.get("peak_memory_mb"),
                "stage0_passed": 1,
                "stage05_passed": 1,
                "stage1_passed": 1,
                "stability_score": row.get("robustness"),
                "has_nan_grad": 0,
                "has_zero_grad": 0,
            }
        )
    return prepared
