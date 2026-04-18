#!/usr/bin/env python3
"""Backfill candidate-readiness provenance labels in place.

This tool does not move or delete any rows. It enriches existing
``program_results`` and ``leaderboard`` entries with cohort/trust/protocol
metadata and graph-derived structural provenance so the current notebook can
be queried safely by trust tier.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from research.scientist.leaderboard_scoring import SCORING_VERSION
from research.scientist.notebook import LabNotebook
from research.scientist.notebook.program_provenance import (
    merge_experiment_provenance_kwargs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def _table_columns(nb: LabNotebook, table: str) -> set[str]:
    return {row[1] for row in nb.conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _build_program_row_payload(nb: LabNotebook, row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload["experiment_type"] = row.get("experiment_type")
    raw_provenance = row.get("data_provenance_json")
    if isinstance(raw_provenance, str) and raw_provenance.strip():
        try:
            parsed_provenance = json.loads(raw_provenance)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_provenance = {}
        if isinstance(parsed_provenance, dict):
            for key, value in parsed_provenance.items():
                if payload.get(key) in (None, "") and value not in (None, ""):
                    payload[key] = value
    config_json = row.get("config_json")
    experiment_config: Dict[str, Any] = {}
    if isinstance(config_json, str) and config_json.strip():
        try:
            parsed = json.loads(config_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            experiment_config = parsed
    return merge_experiment_provenance_kwargs(payload, experiment_config)


def _target_program_results(
    nb: LabNotebook,
    *,
    where: str = "",
    limit: int = 0,
) -> list[Dict[str, Any]]:
    pr_cols = _table_columns(nb, "program_results")
    wanted = [
        "result_id",
        "experiment_id",
        "model_source",
        "graph_json",
        "stage0_passed",
        "stage05_passed",
        "stage1_passed",
        "validation_loss_ratio",
        "wikitext_perplexity",
        "hellaswag_acc",
        "induction_auc",
        "binding_auc",
        "binding_composite",
        "data_mode",
        "tokenizer_mode",
        "tokenizer_id",
        "tokenizer_version",
        "tiktoken_encoding",
        "corpus_path",
        "corpus_id",
        "corpus_version",
        "corpus_format",
        "corpus_text_key",
        "corpus_train_fraction",
        "corpus_val_fraction",
        "corpus_max_chars",
        "hf_dataset",
        "hf_subset",
        "hf_split",
        "hf_text_key",
        "hydra_dataset",
        "hydra_data_dir",
        "split_id",
        "split_policy_version",
        "provenance_complete",
        "screening_wikitext_metric_version",
        "induction_probe_metric_version",
        "novelty_reference_version",
        "novelty_scoring_policy_version",
        "result_cohort",
        "trust_label",
        "comparability_label",
        "evaluation_protocol_version",
        "init_regime",
        "data_provenance_json",
    ]
    selected = [c for c in wanted if c in pr_cols]
    sql = (
        f"SELECT {', '.join('pr.' + c for c in selected)}, "
        "e.experiment_type AS experiment_type "
        "FROM program_results pr "
        "LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id "
        ""
    )
    if "config_json" in _table_columns(nb, "experiments"):
        sql = sql.replace(
            "e.experiment_type AS experiment_type ",
            "e.experiment_type AS experiment_type, e.config_json AS config_json ",
        )
    if where:
        sql += f"WHERE {where} "
    sql += "ORDER BY pr.timestamp ASC "
    if limit > 0:
        sql += f"LIMIT {int(limit)}"
    return [dict(row) for row in nb.conn.execute(sql).fetchall()]


def _update_program_result(
    nb: LabNotebook, row: Dict[str, Any], *, dry_run: bool
) -> bool:
    payload = _build_program_row_payload(nb, row)
    result_cohort = nb._infer_result_cohort(payload)
    trust_label = nb._infer_trust_label(payload, result_cohort)
    comparability_label = nb._infer_comparability_label(
        payload, result_cohort, trust_label
    )
    evaluation_protocol_version = nb._infer_evaluation_protocol_version(
        payload, result_cohort, trust_label
    )
    init_regime = nb._infer_init_regime(payload, result_cohort)
    data_provenance_json = nb._build_data_provenance(
        payload,
        result_cohort=result_cohort,
        trust_label=trust_label,
        comparability_label=comparability_label,
        evaluation_protocol_version=evaluation_protocol_version,
        init_regime=init_regime,
    )

    changed = (
        row.get("result_cohort") != result_cohort
        or row.get("trust_label") != trust_label
        or row.get("comparability_label") != comparability_label
        or row.get("evaluation_protocol_version") != evaluation_protocol_version
        or row.get("init_regime") != init_regime
        or row.get("data_provenance_json") != data_provenance_json
    )
    if not changed:
        return False
    if dry_run:
        return True
    nb.conn.execute(
        """
        UPDATE program_results
        SET result_cohort = ?, trust_label = ?, comparability_label = ?,
            evaluation_protocol_version = ?, init_regime = ?, data_provenance_json = ?
        WHERE result_id = ?
        """,
        (
            result_cohort,
            trust_label,
            comparability_label,
            evaluation_protocol_version,
            init_regime,
            data_provenance_json,
            row["result_id"],
        ),
    )
    return True


def _sync_leaderboard(nb: LabNotebook, *, dry_run: bool) -> int:
    lb_cols = _table_columns(nb, "leaderboard")
    if "entry_id" not in lb_cols:
        return 0
    rows = nb.conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.result_cohort, l.trust_label,
               l.comparability_label, l.evaluation_protocol_version, l.scoring_version,
               pr.result_cohort AS pr_result_cohort, pr.trust_label AS pr_trust_label,
               pr.comparability_label AS pr_comparability_label,
               pr.evaluation_protocol_version AS pr_evaluation_protocol_version
        FROM leaderboard l
        LEFT JOIN program_results pr ON pr.result_id = l.result_id
        """
    ).fetchall()
    updated = 0
    for row in rows:
        data = dict(row)
        next_values = {
            "result_cohort": data.get("pr_result_cohort") or data.get("result_cohort"),
            "trust_label": data.get("pr_trust_label") or data.get("trust_label"),
            "comparability_label": data.get("pr_comparability_label")
            or data.get("comparability_label"),
            "evaluation_protocol_version": data.get("pr_evaluation_protocol_version")
            or data.get("evaluation_protocol_version"),
            "scoring_version": data.get("scoring_version") or SCORING_VERSION,
        }
        changed = any(data.get(key) != value for key, value in next_values.items())
        if not changed:
            continue
        updated += 1
        if dry_run:
            continue
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET result_cohort = ?, trust_label = ?, comparability_label = ?,
                evaluation_protocol_version = ?, scoring_version = ?
            WHERE entry_id = ?
            """,
            (
                next_values["result_cohort"],
                next_values["trust_label"],
                next_values["comparability_label"],
                next_values["evaluation_protocol_version"],
                next_values["scoring_version"],
                data["entry_id"],
            ),
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill provenance labels and graph-derived trust metadata in place"
    )
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--where",
        default="",
        help="Optional SQL WHERE clause over program_results alias 'pr'",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    nb = LabNotebook(args.db)
    nb.flush_writes()

    rows = _target_program_results(nb, where=args.where, limit=args.limit)
    logger.info("Loaded %d program_results rows", len(rows))

    updated_pr = 0
    for row in rows:
        if _update_program_result(nb, row, dry_run=args.dry_run):
            updated_pr += 1

    updated_lb = _sync_leaderboard(nb, dry_run=args.dry_run)

    if not args.dry_run:
        nb.conn.commit()

    summary = {
        "program_results_seen": len(rows),
        "program_results_updated": updated_pr,
        "leaderboard_updated": updated_lb,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
