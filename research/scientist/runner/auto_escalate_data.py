from __future__ import annotations

"""Compact data-loading helpers for auto-escalation."""

import json
import math
from typing import Any, Dict, Iterable, List

from ..trust_policy import sql_trusted_clause


def trusted_screening_candidates(
    nb, *, experiment_id: str | None, limit: int
) -> List[Dict[str, Any]]:
    if experiment_id:
        rows = nb.conn.execute(
            f"""SELECT * FROM program_results
                WHERE experiment_id = ?
                  AND stage1_passed = 1
                  AND {sql_trusted_clause()}
                ORDER BY loss_ratio ASC NULLS LAST
                LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    return nb.get_top_programs(limit, sort_by="loss_ratio", trusted_only=True)


def trusted_global_screening_candidates(nb, *, limit: int) -> List[Dict[str, Any]]:
    rows = nb.conn.execute(
        f"""SELECT pr.* FROM leaderboard l
            JOIN program_results pr ON l.result_id = pr.result_id
            WHERE l.tier = 'screening'
              AND l.screening_passed = 1
              AND COALESCE(l.is_reference, 0) = 0
              AND {sql_trusted_clause(table_alias="l")}
              AND l.investigation_loss_ratio IS NULL
              AND (l.tags IS NULL OR l.tags NOT LIKE '%provisional_random_tokens%')
            ORDER BY l.composite_score DESC
            LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def composite_score_map(nb, result_ids: Iterable[str]) -> Dict[str, float]:
    ids = [rid for rid in result_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = nb.conn.execute(
        f"SELECT result_id, composite_score FROM leaderboard WHERE result_id IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return {row["result_id"]: float(row["composite_score"] or 0.0) for row in rows}


def novelty_metadata(nb, result_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = [rid for rid in result_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = nb.conn.execute(
        f"""SELECT result_id, novelty_valid_for_promotion, cka_source, fingerprint_json
            FROM program_results
            WHERE result_id IN ({placeholders})""",
        tuple(ids),
    ).fetchall()
    out = {}
    for row in rows:
        meta = dict(row)
        fp_json = meta.pop("fingerprint_json", None)
        if fp_json:
            try:
                parsed = json.loads(fp_json)
            except (ValueError, TypeError):
                parsed = {}
            meta["fingerprint_completed_post_investigation"] = bool(
                parsed.get("fingerprint_completed_post_investigation")
            )
        else:
            meta["fingerprint_completed_post_investigation"] = False
        out[row["result_id"]] = meta
    return out


def investigation_support_data(
    nb, result_ids: Iterable[str]
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    ids = [rid for rid in result_ids if rid]
    if not ids:
        return {}, {}, {}
    placeholders = ",".join("?" for _ in ids)
    score_rows = nb.conn.execute(
        f"""SELECT result_id, composite_score, replication_n, replication_loss_std
            FROM leaderboard WHERE result_id IN ({placeholders})""",
        tuple(ids),
    ).fetchall()
    composite_scores = {
        row["result_id"]: float(row["composite_score"] or 0.0) for row in score_rows
    }
    replication = {
        row["result_id"]: {
            "n": int(row["replication_n"] or 1),
            "loss_std": float(row["replication_loss_std"] or 0.0),
        }
        for row in score_rows
    }
    understanding_rows = nb.conn.execute(
        f"""SELECT result_id, ar_auc, induction_auc, binding_auc, diagnostic_score, hellaswag_acc
            FROM program_results WHERE result_id IN ({placeholders})""",
        tuple(ids),
    ).fetchall()
    understanding = {
        row["result_id"]: {
            "ar_auc": float(row["ar_auc"] or 0.0),
            "induction_auc": float(row["induction_auc"] or 0.0),
            "binding_auc": float(row["binding_auc"] or 0.0),
            "diagnostic_score": float(row["diagnostic_score"] or 0.0),
            "hellaswag_acc": float(row["hellaswag_acc"] or 0.0),
        }
        for row in understanding_rows
    }
    return composite_scores, replication, understanding


def graph_meta_by_result_id(nb, result_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = [rid for rid in result_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = nb.conn.execute(
        f"SELECT result_id, graph_json, routing_mode FROM program_results WHERE result_id IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return {row["result_id"]: dict(row) for row in rows}


def effective_validation_threshold(
    *, min_score: float, replication_n: int, loss_std: float
) -> float:
    if min_score <= 0:
        return min_score
    if replication_n <= 1:
        return min_score * 1.10
    if loss_std > 0 and replication_n >= 2:
        se_score = 160.0 * loss_std / math.sqrt(replication_n)
        return min_score + 1.28 * se_score
    return min_score
