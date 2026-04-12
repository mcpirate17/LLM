#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit unresolved partial candidate-grade rows."
    )
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pr.result_id,
               pr.experiment_id,
               COALESCE(ex.experiment_type, '') AS experiment_type,
               pr.timestamp,
               pr.stage0_passed,
               pr.stage05_passed,
               pr.stage1_passed,
               pr.wikitext_perplexity,
               pr.induction_auc,
               pr.hellaswag_acc,
               pr.validation_loss_ratio,
               pr.discovery_loss_ratio,
               pr.loss_ratio,
               pr.data_provenance_json
        FROM program_results pr
        LEFT JOIN experiments ex ON ex.experiment_id = pr.experiment_id
        WHERE pr.trust_label = 'candidate_grade'
          AND pr.comparability_label = 'partial'
        ORDER BY pr.timestamp DESC
        LIMIT ?
        """,
        (int(args.limit),),
    ).fetchall()
    conn.close()

    reason_counts: Counter[str] = Counter()
    for row in rows:
        payload = {}
        raw = row["data_provenance_json"]
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
        if (
            not payload.get("corpus_id")
            or not payload.get("tokenizer_id")
            or not payload.get("split_id")
        ):
            reason = "missing_data_identity"
        elif not any(
            payload.get(key)
            for key in (
                "screening_wikitext_metric_version",
                "induction_probe_metric_version",
                "novelty_reference_version",
            )
        ):
            reason = "missing_metric_version"
        else:
            reason = "other"
        reason_counts[reason] += 1

    print("partial_candidate_grade_summary")
    print(json.dumps(dict(reason_counts), indent=2, sort_keys=True))
    print("rows")
    for row in rows:
        payload = {}
        raw = row["data_provenance_json"]
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
        print(
            json.dumps(
                {
                    "result_id": row["result_id"],
                    "experiment_id": row["experiment_id"],
                    "experiment_type": row["experiment_type"],
                    "stage1_passed": bool(row["stage1_passed"]),
                    "wikitext_perplexity": row["wikitext_perplexity"],
                    "induction_auc": row["induction_auc"],
                    "hellaswag_acc": row["hellaswag_acc"],
                    "validation_loss_ratio": row["validation_loss_ratio"],
                    "discovery_loss_ratio": row["discovery_loss_ratio"],
                    "loss_ratio": row["loss_ratio"],
                    "corpus_id": payload.get("corpus_id"),
                    "tokenizer_id": payload.get("tokenizer_id"),
                    "split_id": payload.get("split_id"),
                    "screening_wikitext_metric_version": payload.get(
                        "screening_wikitext_metric_version"
                    ),
                    "induction_probe_metric_version": payload.get(
                        "induction_probe_metric_version"
                    ),
                    "novelty_reference_version": payload.get(
                        "novelty_reference_version"
                    ),
                    "comparability_reason": payload.get("comparability_reason"),
                    "comparability_gaps": payload.get("comparability_gaps"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
