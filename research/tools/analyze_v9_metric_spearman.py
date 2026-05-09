#!/usr/bin/env python
"""Spearman-rank analysis: each Gemini trajectory metric vs ground-truth labels.

After v9 backfill populates the four scoring metrics on 17k+ rows, run this
to see which metrics actually predict outcomes worth promoting. Output is the
empirical-weighting evidence we said v9.1 calibration would need (see
``tasks/gemini_metrics_scoring_spec.md`` §4 and the EZNAS precedent in the
industry survey).

Three label types compared per metric:

* ``stage1_passed`` — binary, trivial promotion test from screening.
* ``loss_ratio`` (lower-is-better) — continuous post-training quality.
* ``composite_score_v8_1`` — pre-v9 ranking we want to either match or
  beat empirically.
* ``induction_intermediate_auc`` — capability landmark; only populated
  on rows that reached investigation tier.

Outputs ``research/perf_artifacts/v9_spearman_<ts>.json`` plus a stdout
summary table. No DB writes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

from scipy.stats import spearmanr

from research.defaults import RUNS_DB

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / RUNS_DB
ARTIFACT_DIR = ROOT / "research" / "perf_artifacts"

GEMINI_METRICS = [
    ("erf_density", "fp_jacobian_erf_density", False),
    ("erf_variance_log", "fp_jacobian_erf_variance", True),
    ("erf_decay_slope", "fp_jacobian_erf_decay_slope", False),
    ("icld_velocity", "fp_icld_velocity", False),
    ("logit_margin_velocity", "fp_logit_margin_velocity", False),
    ("id_collapse_rate", "fp_id_collapse_rate", False),
    ("spec_norm_log", "fp_jacobian_spectral_norm", True),
]

LABELS = [
    ("stage1_passed", "pr.stage1_passed", "binary"),
    # Loss ratio: lower is better. We negate for correlation so positive ρ
    # means "metric predicts a good outcome".
    ("neg_loss_ratio", "-pr.loss_ratio", "continuous"),
    ("composite_score_v8_1", "l.composite_score", "continuous"),
    (
        "induction_intermediate_auc",
        "pr.induction_intermediate_auc",
        "continuous",
    ),
    ("binding_intermediate_auc", "pr.binding_intermediate_auc", "continuous"),
]


def _log_safe(values: list[float]) -> list[float]:
    return [math.log10(abs(v) + 1e-9) if v is not None else None for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default=None,
        help="Restrict to rows with this fp_metric_phase (e.g. 'init', "
        "'screening_750'). Default: any row with the metric populated.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default research/perf_artifacts/v9_spearman_<ts>.json)",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else ARTIFACT_DIR / f"v9_spearman_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )

    where_phase = f" AND pr.fp_metric_phase = '{args.phase}'" if args.phase else ""

    metric_cols = ", ".join(f"pr.{c}" for _, c, _ in GEMINI_METRICS)
    label_cols = ", ".join(f"({expr}) AS {name}" for name, expr, _ in LABELS)

    sql = (
        "SELECT pr.graph_fingerprint, "
        f"       {metric_cols}, "
        f"       {label_cols} "
        "FROM program_results pr "
        "LEFT JOIN leaderboard l ON l.result_id = pr.result_id "
        "WHERE pr.fp_jacobian_erf_density IS NOT NULL "
        "  AND pr.graph_json IS NOT NULL "
        f"{where_phase}"
    )

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql).fetchall()
    conn.close()
    print(f"[setup] loaded {len(rows)} rows with at least one Gemini metric populated")

    if len(rows) < 50:
        print("[abort] sample too small for meaningful correlation")
        sys.exit(1)

    # Build columns up-front to compute log/sign transforms.
    metric_columns: dict[str, list[float | None]] = {}
    for label, col, log_transform in GEMINI_METRICS:
        vals = [row[col] for row in rows]
        if log_transform:
            vals = _log_safe(vals)
        metric_columns[label] = vals

    label_columns: dict[str, list[float | None]] = {}
    label_kinds: dict[str, str] = {}
    for label_name, _expr, kind in LABELS:
        label_columns[label_name] = [row[label_name] for row in rows]
        label_kinds[label_name] = kind

    artifact: dict = {
        "timestamp": time.time(),
        "n_rows_total": len(rows),
        "phase_filter": args.phase,
        "metric_descriptions": {
            label: {"column": col, "log_transform": log}
            for label, col, log in GEMINI_METRICS
        },
        "label_descriptions": {
            label: {"expr": expr, "kind": kind} for label, expr, kind in LABELS
        },
        "spearman": {},
        "n_rows_per_label": {},
    }

    print()
    header = "metric".ljust(24) + "  ".join(f"{l:>20}" for l, _, _ in LABELS)
    print(header)
    print("-" * len(header))

    for metric_label in metric_columns:
        x = metric_columns[metric_label]
        row_parts = [metric_label.ljust(24)]
        artifact["spearman"][metric_label] = {}
        for label_name in label_columns:
            y = label_columns[label_name]
            paired = [
                (xv, yv) for xv, yv in zip(x, y) if xv is not None and yv is not None
            ]
            artifact["n_rows_per_label"][label_name] = len(paired)
            if len(paired) < 30:
                row_parts.append("n/a".rjust(20))
                artifact["spearman"][metric_label][label_name] = None
                continue
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            rho, pval = spearmanr(xs, ys)
            artifact["spearman"][metric_label][label_name] = {
                "rho": float(rho),
                "p": float(pval),
                "n": len(paired),
            }
            row_parts.append(f"ρ={rho:+.3f} (n={len(paired):>5})".rjust(20))
        print("  ".join(row_parts))

    # Recommended weight calibration: weight ∝ |ρ| against the most
    # decision-relevant label for each metric. Use composite_score_v8_1 as
    # the "this is what we ranked on before" baseline — metrics with high
    # |ρ| against composite are largely re-encoding existing signal;
    # metrics with high |ρ| against capability labels (induction_intermediate_auc,
    # binding_intermediate_auc, neg_loss_ratio) are adding orthogonal signal worth
    # weighting heavily.
    print()
    print("=== weight-calibration suggestion (v9.1 candidate) ===")
    print(
        "Heuristic: weight ∝ max(|ρ| vs neg_loss_ratio, |ρ| vs induction_intermediate_auc, "
        "|ρ| vs binding_intermediate_auc).\n"
        "This favors metrics that predict capability-relevant labels — "
        "what v9 was supposed to surface — over metrics that just re-encode "
        "the v8.1 composite.\n"
    )
    capability_labels = (
        "neg_loss_ratio",
        "induction_intermediate_auc",
        "binding_intermediate_auc",
    )
    raw_weights = {}
    for metric_label in metric_columns:
        rhos = [
            abs(artifact["spearman"][metric_label][lbl]["rho"])
            for lbl in capability_labels
            if artifact["spearman"][metric_label].get(lbl) is not None
        ]
        raw_weights[metric_label] = max(rhos) if rhos else 0.0

    total = sum(raw_weights.values())
    print(f"{'metric':<24}{'raw |ρ|max':>12}{'normalized':>14}")
    for label, w in sorted(raw_weights.items(), key=lambda kv: -kv[1]):
        norm = (w / total) if total > 0 else 0.0
        print(f"  {label:<22}{w:>12.3f}{norm:>14.1%}")
    artifact["proposed_v9_1_weights"] = {
        label: (w / total if total > 0 else 0.0) for label, w in raw_weights.items()
    }

    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\n[done] artifact written: {out_path}")


if __name__ == "__main__":
    main()
