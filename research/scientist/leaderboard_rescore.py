"""Canonical leaderboard rescore helpers shared by API and maintenance scripts."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, Tuple

from .breakthrough_gates import passes_breakthrough_from_row
from .leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    get_scoring_version,
    prefetch_program_results,
)
from .notebook import LabNotebook


_AGG_METRIC_TO_SCORE_KWARGS = {
    "wikitext_perplexity": (
        "ppl_screening",
        "ppl_investigation",
        "ppl_validation",
    ),
    "blimp_overall_accuracy": ("blimp_accuracy",),
    "hellaswag_acc": (
        "hellaswag_acc_screening",
        "hellaswag_acc_investigation",
        "hellaswag_acc_validation",
    ),
    "tinystories_score": ("tinystories_score",),
    "cross_task_score": ("cross_task_score",),
    "diagnostic_score": ("diagnostic_score",),
    "fp_hierarchy_fitness": ("hierarchy_fitness",),
    "ar_auc": ("ar_auc",),
    "nano_ar_inv_score": ("nano_ar_inv_score",),
    "induction_auc": ("induction_auc",),
    "binding_auc": ("binding_auc",),
    "induction_v2_investigation_auc": ("induction_v2_inv_auc",),
    "binding_v2_investigation_auc": ("binding_v2_inv_auc",),
}


_DENORMALIZED_SCORE_METRIC_TO_KWARG = {
    "induction_v2_investigation_auc": "induction_v2_inv_auc",
    "binding_v2_investigation_auc": "binding_v2_inv_auc",
}

_DENORMALIZED_PROGRAM_RESULT_COLUMNS = (
    "hellaswag_metric_version",
    "hellaswag_tokenizer_mode",
    "hellaswag_tiktoken_encoding",
    "induction_v2_investigation_max_gap_acc",
    "induction_v2_investigation_protocol_version",
    "binding_v2_investigation_max_distance_acc",
    "binding_v2_investigation_protocol_version",
)


def _denormalized_score_metric_updates(
    pr_dict: Dict[str, Any],
    score_kw: Dict[str, Any],
) -> Dict[str, Any]:
    """Return leaderboard display metrics from the same inputs used for scoring."""
    updates: Dict[str, Any] = {}
    for col, kwarg_name in _DENORMALIZED_SCORE_METRIC_TO_KWARG.items():
        value = score_kw.get(kwarg_name)
        if value is None:
            value = pr_dict.get(col)
        if value is not None:
            updates[col] = value
    for col in _DENORMALIZED_PROGRAM_RESULT_COLUMNS:
        if pr_dict.get(col) is not None:
            updates[col] = pr_dict.get(col)
    return updates


def _apply_fingerprint_aggregates(
    nb: LabNotebook,
    pr_dict: Dict[str, Any],
    score_kw: Dict[str, Any],
) -> Dict[str, Any]:
    """Inject per-fingerprint means/CV into score kwargs and return DB updates."""
    graph_fingerprint = str(pr_dict.get("graph_fingerprint") or "").strip()
    if not graph_fingerprint:
        return {}

    updates: Dict[str, Any] = {}
    agg = nb.get_fingerprint_aggregates(graph_fingerprint) or {}
    if agg.get("n_runs", 0) > 0:
        updates["replication_n"] = agg.get("n_runs")
        updates["replication_loss_mean"] = agg.get("loss_mean")
        updates["replication_loss_std"] = agg.get("loss_std")
        updates["replication_best_vs_mean_gap"] = agg.get("best_vs_mean_gap")
        score_kw["replication_n"] = agg.get("n_runs")
        score_kw["replication_loss_mean"] = agg.get("loss_mean")
        score_kw["replication_loss_std"] = agg.get("loss_std")
        score_kw["replication_best_vs_mean_gap"] = agg.get("best_vs_mean_gap")

    metric_agg = nb.get_fingerprint_metric_aggregates(graph_fingerprint) or {}
    if not metric_agg:
        return updates

    for col, kwarg_names in _AGG_METRIC_TO_SCORE_KWARGS.items():
        stat = metric_agg.get(col) or {}
        if int(stat.get("n") or 0) < 2 or stat.get("mean") is None:
            continue
        mean_val = stat["mean"]
        for kwarg_name in kwarg_names:
            # Preserve stage semantics: only average a metric into a stage that
            # had that metric populated in the canonical row.
            if score_kw.get(kwarg_name) is not None:
                score_kw[kwarg_name] = mean_val

    tier_cv = metric_agg.get("_tier_cv") or {}
    updates["n_runs"] = metric_agg.get("_n_runs_max")
    updates["cv_loss"] = tier_cv.get("loss")
    updates["cv_understanding"] = tier_cv.get("und")
    updates["cv_capability"] = tier_cv.get("cap")
    score_kw["n_runs"] = updates["n_runs"]
    score_kw["cv_loss"] = updates["cv_loss"]
    score_kw["cv_understanding"] = updates["cv_understanding"]
    score_kw["cv_capability"] = updates["cv_capability"]
    return updates


def _stability_from_breakdown(breakdown: Dict[str, Any]) -> float:
    if not breakdown.get("_cv_penalty_applied"):
        return 1.0
    loss_pen = float(breakdown.get("_cv_penalty_loss") or 1.0)
    und_pen = float(breakdown.get("_cv_penalty_und") or 1.0)
    cap_pen = float(breakdown.get("_cv_penalty_cap") or 1.0)
    return (loss_pen * und_pen * cap_pen) ** (1.0 / 3.0)


def rescore_entry(
    nb: LabNotebook,
    entry_id: str,
    result_id: str,
    is_ref: bool,
    pr_cache: Dict[str, Dict],
    pr_updates: Optional[Dict[str, Any]] = None,
    *,
    reason: str = "canonical_rescore",
) -> Tuple[float, float]:
    """Recompute and persist one leaderboard composite score."""
    existing = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    if not existing:
        return 0.0, 0.0

    current_version = get_scoring_version()
    row = dict(existing)
    old_score = float(row.get("composite_score") or 0.0)
    old_version = str(row.get("scoring_config_hash") or "").strip()
    pr_dict = dict(pr_cache.get(result_id, {}))
    if pr_updates:
        pr_dict.update(pr_updates)
    score_kw = build_score_kwargs_from_prefetch(pr_dict, row, is_ref)
    aggregate_updates = _apply_fingerprint_aggregates(nb, pr_dict, score_kw)
    aggregate_updates.update(_denormalized_score_metric_updates(pr_dict, score_kw))
    score_result = compute_composite(decompose=True, **score_kw)
    if isinstance(score_result, dict):
        new_score = float(score_result.get("composite_score") or 0.0)
        aggregate_updates["score_stability_penalty"] = _stability_from_breakdown(
            score_result.get("breakdown") or {}
        )
    else:
        new_score = float(score_result or 0.0)
        aggregate_updates["score_stability_penalty"] = 1.0

    # Tier demotion: if a breakthrough row no longer satisfies the gates with
    # the new composite + the freshly aggregated capability metrics, drop it
    # back to validation. Reference rows are left alone (they are pinned).
    # Triggered by both the canonical rescore CLI and the post-rescreen
    # ``sync_fingerprint_leaderboard`` path that funnels through here.
    demote_to: Optional[str] = None
    demote_reason: Optional[str] = None
    if not is_ref and str(row.get("tier") or "").lower() == "breakthrough":
        gate_row = dict(row)
        for col, val in aggregate_updates.items():
            if val is not None:
                gate_row[col] = val
        passed, demote_reason = passes_breakthrough_from_row(
            gate_row, composite_score=new_score
        )
        if not passed:
            demote_to = "validation"
            aggregate_updates["tier"] = "validation"

    if new_score != old_score or old_version != current_version or aggregate_updates:
        columns = nb._get_leaderboard_columns()
        sets = ["composite_score = ?"]
        params: list[Any] = [new_score]
        if "scoring_config_hash" in columns:
            sets.append("scoring_config_hash = ?")
            params.append(current_version)
        if "rescore_status" in columns:
            sets.append("rescore_status = 'rescored'")
        if "rescore_timestamp" in columns:
            sets.append("rescore_timestamp = ?")
            params.append(time.time())
        if "old_composite_score" in columns:
            sets.append("old_composite_score = ?")
            params.append(old_score)
        if "rescore_reason" in columns:
            sets.append("rescore_reason = ?")
            demote_suffix = (
                f"; demoted_to={demote_to}({demote_reason})" if demote_to else ""
            )
            params.append(reason + demote_suffix)
        for col, val in aggregate_updates.items():
            if col in columns and val is not None:
                sets.append(f"{col} = ?")
                params.append(val)
        params.append(entry_id)
        nb.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
            params,
        )
    return new_score, old_score


def rescore_leaderboard(
    nb: LabNotebook,
    *,
    result_ids: Optional[Iterable[str]] = None,
    only_stale: bool = False,
    reason: str = "canonical_rescore",
) -> Tuple[int, int]:
    """Bulk rescore leaderboard rows against the active backend scoring version."""
    params: list[Any] = []
    where: list[str] = []
    normalized_ids = [str(result_id) for result_id in (result_ids or []) if result_id]
    if normalized_ids:
        placeholders = ",".join("?" for _ in normalized_ids)
        where.append(f"result_id IN ({placeholders})")
        params.extend(normalized_ids)
    if only_stale:
        where.append("(scoring_config_hash IS NULL OR scoring_config_hash != ?)")
        params.append(get_scoring_version())

    current_version = get_scoring_version()
    query = (
        "SELECT entry_id, result_id, is_reference, composite_score, "
        "scoring_config_hash "
        "FROM leaderboard"
    )
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY composite_score DESC"

    rows = nb.conn.execute(query, tuple(params)).fetchall()
    all_ids = [str(row["result_id"]) for row in rows if row["result_id"]]
    pr_cache = prefetch_program_results(nb.conn, all_ids)

    changed = 0
    for row in rows:
        new_score, old_score = rescore_entry(
            nb,
            str(row["entry_id"]),
            str(row["result_id"]),
            bool(row["is_reference"]),
            pr_cache,
            reason=reason,
        )
        old_version = str(row["scoring_config_hash"] or "").strip()
        if new_score != old_score or old_version != current_version:
            changed += 1
    nb.conn.commit()
    return len(rows), changed
