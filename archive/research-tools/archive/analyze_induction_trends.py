#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

DB_PATH = "research/lab_notebook.db"


def _bucket(auc: float) -> str:
    if auc >= 0.05:
        return "learner"
    if auc >= 0.02:
        return "weak_learner"
    if auc == 0.0:
        return "non_learner_zero"
    return "non_learner_low"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize canonical induction trends."
    )
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT pr.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY pr.graph_fingerprint
                       ORDER BY pr.timestamp DESC
                   ) AS rn
            FROM program_results pr
            JOIN induction_metrics_v2 im ON im.graph_fingerprint = pr.graph_fingerprint
        )
        SELECT
            latest.graph_fingerprint,
            latest.model_source,
            latest.graph_json,
            latest.graph_n_ops,
            latest.graph_depth,
            latest.graph_n_unique_ops,
            latest.graph_category_histogram,
            im.auc
        FROM latest
        JOIN induction_metrics_v2 im ON im.graph_fingerprint = latest.graph_fingerprint
        WHERE latest.rn = 1
        """
    ).fetchall()

    buckets = Counter()
    model_source = defaultdict(list)
    category_means = defaultdict(list)
    op_token_means = defaultdict(list)
    for row in rows:
        auc = float(row["auc"] or 0.0)
        buckets[_bucket(auc)] += 1
        model_source[str(row["model_source"] or "unknown")].append(auc)
        hist = row["graph_category_histogram"]
        if hist:
            try:
                parsed = json.loads(hist)
            except json.JSONDecodeError:
                parsed = {}
            for key, value in parsed.items():
                if value:
                    category_means[str(key)].append(auc)
        graph_json = str(row["graph_json"] or "")
        for token in (
            "attention",
            "mamba",
            "rwkv",
            "conv",
            "matmul",
            "fourier",
            "fft",
            "moe",
            "router",
            "sparse",
        ):
            if token in graph_json.lower():
                op_token_means[token].append(auc)

    print("bucket_counts")
    for key, value in sorted(buckets.items()):
        print(key, value)

    print("\nmodel_source_mean_auc")
    for key, vals in sorted(model_source.items()):
        print(key, round(sum(vals) / max(len(vals), 1), 4), len(vals))

    print("\ncategory_mean_auc_top10")
    ranked_categories = sorted(
        (
            (sum(vals) / len(vals), key, len(vals))
            for key, vals in category_means.items()
            if vals
        ),
        reverse=True,
    )
    for mean_auc, key, n in ranked_categories[:10]:
        print(key, round(mean_auc, 4), n)

    print("\ncomponent_token_mean_auc")
    for key, vals in sorted(op_token_means.items()):
        if vals:
            print(key, round(sum(vals) / len(vals), 4), len(vals))


if __name__ == "__main__":
    main()
