"""A/B harness for the GBM predictor: with vs without Gemini features.

Trains two GBMs from the same corpus split — one with all features
(Gemini trajectory metrics + classic post-eval probes), one with the
Gemini features dropped — and reports the ROC AUC / PPV / NPV / rank
Spearman delta. The exclusion list is the v9 trajectory feature names
already wired into ``_POST_EVAL_FEATURE_NAMES``.

Usage::

    python -m research.tools.predictor_ab
    python -m research.tools.predictor_ab --out artifacts/predictor_ab.json
    python -m research.tools.predictor_ab --db /custom/lab.db --out my.json

Decision criterion (from v10 handoff section 6): any positive ROC AUC
delta is sufficient — but PPV improvement at the operating threshold
is the real signal. If neither moves, the Gemini features should be
dropped from the predictor regardless of what the composite does.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


GEMINI_FEATURE_NAMES = frozenset(
    {
        "fp_jacobian_erf_density_best",
        "fp_jacobian_erf_variance_best",
        "fp_icld_velocity_best",
        "fp_logit_margin_velocity_best",
        "fp_id_collapse_rate_best",
        "fp_jacobian_spectral_norm_best",
    }
)


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    try:
        return float(candidate) - float(baseline)
    except (TypeError, ValueError):
        return None


def _gate_metric(report: Dict[str, Any], key: str) -> float | None:
    metrics = (report.get("gate_metrics") or {}) if isinstance(report, dict) else {}
    val = metrics.get(key)
    if val is None:
        # Fall back to top-level for ROC AUC.
        val = report.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _rank_metric(report: Dict[str, Any], head: str, key: str) -> float | None:
    heads = (report.get("rank_heads") or {}) if isinstance(report, dict) else {}
    head_dict = heads.get(head) or {}
    val = head_dict.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def run_ab(db_path: str) -> Dict[str, Any]:
    """Train baseline (no Gemini) and candidate (with Gemini) GBMs.

    Returns a dict with both reports and a ``deltas`` summary.
    """
    from research.scientist.intelligence.predictor_gbm import evaluate_gbm

    logger.info("training baseline (Gemini features excluded)")
    baseline = evaluate_gbm(
        db_path=db_path,
        excluded_feature_names=set(GEMINI_FEATURE_NAMES),
    )
    logger.info("training candidate (Gemini features included)")
    candidate = evaluate_gbm(db_path=db_path)

    if "error" in baseline or "error" in candidate:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "error": "one or both models failed; see report bodies",
        }

    deltas = {
        "gate_auc": _delta(candidate.get("gate_auc"), baseline.get("gate_auc")),
        "gate_threshold": _delta(
            candidate.get("gate_threshold"), baseline.get("gate_threshold")
        ),
        "ppv": _delta(
            _gate_metric(candidate, "precision_ppv"),
            _gate_metric(baseline, "precision_ppv"),
        ),
        "npv": _delta(
            _gate_metric(candidate, "npv"), _gate_metric(baseline, "npv")
        ),
        "recall": _delta(
            _gate_metric(candidate, "recall_tpr_sensitivity"),
            _gate_metric(baseline, "recall_tpr_sensitivity"),
        ),
        "f1": _delta(_gate_metric(candidate, "f1"), _gate_metric(baseline, "f1")),
        "rank_spearman_ppl": _delta(
            _rank_metric(candidate, "ppl", "spearman"),
            _rank_metric(baseline, "ppl", "spearman"),
        ),
        "rank_spearman_composite": _delta(
            _rank_metric(candidate, "composite", "spearman"),
            _rank_metric(baseline, "composite", "spearman"),
        ),
        "rank_ndcg_ppl": _delta(
            _rank_metric(candidate, "ppl", "ndcg"),
            _rank_metric(baseline, "ppl", "ndcg"),
        ),
        "rank_ndcg_composite": _delta(
            _rank_metric(candidate, "composite", "ndcg"),
            _rank_metric(baseline, "composite", "ndcg"),
        ),
        "skip_rate": _delta(candidate.get("skip_rate"), baseline.get("skip_rate")),
        "false_skip_rate": _delta(
            candidate.get("false_skip_rate"), baseline.get("false_skip_rate")
        ),
    }

    # Acceptance: any positive delta on gate_auc, PPV, or rank Spearman
    # justifies keeping Gemini features. Gate metrics can't move while
    # Gemini features sit in ``_PROBE_FEATURE_NAMES`` (gate strips probes
    # to avoid label leakage), so rank Spearman is the relevant signal
    # for v9 trajectory metrics. If neither rank head improves, drop
    # them and re-promote to non-probes only after a separate experiment.
    keep_gemini_features = bool(
        (deltas.get("gate_auc") or 0.0) > 0.0
        or (deltas.get("ppv") or 0.0) > 0.0
        or (deltas.get("rank_spearman_ppl") or 0.0) > 0.0
        or (deltas.get("rank_spearman_composite") or 0.0) > 0.0
    )

    return {
        "baseline_no_gemini": baseline,
        "candidate_with_gemini": candidate,
        "deltas": deltas,
        "verdict": {
            "keep_gemini_features": keep_gemini_features,
            "reason": (
                "candidate beats baseline on at least one of "
                "{gate_auc, ppv, rank_spearman_ppl, rank_spearman_composite}"
                if keep_gemini_features
                else "candidate does not improve any tracked metric"
            ),
        },
        "excluded_in_baseline": sorted(GEMINI_FEATURE_NAMES),
    }


def _print_summary(result: Dict[str, Any]) -> None:
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return

    baseline = result["baseline_no_gemini"]
    candidate = result["candidate_with_gemini"]
    deltas = result["deltas"]
    verdict = result["verdict"]

    print("─" * 64)
    print("Predictor A/B — baseline (no Gemini) vs candidate (with Gemini)")
    print("─" * 64)
    print(f"corpus n_train={baseline.get('n_train')} n_test={baseline.get('n_test')}")
    print()
    print(
        f"  {'metric':<25}{'baseline':>14}{'candidate':>14}{'delta':>14}"
    )
    print("  " + "─" * 64)
    rows = [
        ("gate_auc", baseline.get("gate_auc"), candidate.get("gate_auc"), deltas["gate_auc"]),
        (
            "gate_precision (PPV)",
            _gate_metric(baseline, "precision_ppv"),
            _gate_metric(candidate, "precision_ppv"),
            deltas["ppv"],
        ),
        (
            "gate_npv",
            _gate_metric(baseline, "npv"),
            _gate_metric(candidate, "npv"),
            deltas["npv"],
        ),
        (
            "gate_recall",
            _gate_metric(baseline, "recall_tpr_sensitivity"),
            _gate_metric(candidate, "recall_tpr_sensitivity"),
            deltas["recall"],
        ),
        ("gate_f1", _gate_metric(baseline, "f1"), _gate_metric(candidate, "f1"), deltas["f1"]),
        (
            "rank_spearman_ppl",
            _rank_metric(baseline, "ppl", "spearman"),
            _rank_metric(candidate, "ppl", "spearman"),
            deltas["rank_spearman_ppl"],
        ),
        (
            "rank_spearman_composite",
            _rank_metric(baseline, "composite", "spearman"),
            _rank_metric(candidate, "composite", "spearman"),
            deltas["rank_spearman_composite"],
        ),
    ]
    for name, b, c, d in rows:
        b_str = f"{b:.4f}" if isinstance(b, (int, float)) else "n/a"
        c_str = f"{c:.4f}" if isinstance(c, (int, float)) else "n/a"
        d_str = (
            f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"
        )
        print(f"  {name:<25}{b_str:>14}{c_str:>14}{d_str:>14}")
    print()
    print(f"VERDICT: keep_gemini_features={verdict['keep_gemini_features']}")
    print(f"reason: {verdict['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab notebook (default: research/lab_notebook.db).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path. Default: print only.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    result = run_ab(args.db)
    _print_summary(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        logger.info("wrote A/B report to %s", out)


if __name__ == "__main__":
    main()
