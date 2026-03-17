"""Calibrate rapid screening thresholds against historical DB results.

Reads stored metrics from program_results to simulate which architectures
the rapid screening check would have killed, and reports TP/FP rates.

Usage:
    python -m research.tools.calibrate_screening [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.eval.screening_rapid import RapidScreeningCheck

logger = logging.getLogger(__name__)


def _resolve_db(db_path: str) -> Path:
    path = Path(db_path)
    if path.exists():
        return path.absolute()
    cwd = Path.cwd()
    if (cwd / db_path).exists():
        return (cwd / db_path).absolute()
    if cwd.name == "research" and (cwd.parent / db_path).exists():
        return (cwd.parent / db_path).absolute()
    return path.absolute()


def _load_results(db_path: Path) -> List[Dict[str, Any]]:
    """Load program results with metrics relevant to screening calibration."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT
                result_id,
                stage1_passed,
                loss_ratio,
                grad_norm,
                max_grad_norm,
                mean_grad_norm,
                routing_utilization_entropy,
                has_nan_grad
            FROM program_results
            WHERE loss_ratio IS NOT NULL
            ORDER BY rowid DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _simulate_kill(
    row: Dict[str, Any],
    checker: RapidScreeningCheck,
) -> Optional[str]:
    """Simulate whether the rapid check would have killed this result.

    Uses stored summary metrics as proxies for what the 50-step check would see.
    Returns the kill reason or None if it would have passed.
    """
    max_gn = row.get("max_grad_norm")
    gn = row.get("grad_norm")
    loss_ratio = row.get("loss_ratio")
    entropy = row.get("routing_utilization_entropy")
    has_nan_grad = row.get("has_nan_grad")

    # NaN proxy: has_nan_grad flag or non-finite grad_norm
    if has_nan_grad:
        return "grad_nan_inf (has_nan_grad=True)"
    if gn is not None:
        try:
            import math

            if not math.isfinite(gn):
                return "grad_nan_inf"
        except (TypeError, ValueError):
            pass

    # Simulate grad norm hard limit (uses max_grad_norm as proxy for step-10 norm)
    if max_gn is not None and max_gn > checker.grad_norm_hard_limit:
        return f"grad_norm_exploding (max={max_gn:.1f})"

    # Loss ratio proxy: loss_ratio close to 1.0 means loss didn't improve.
    # DB loss_ratio is from full S1 (200+ steps). For the 50-step rapid check,
    # we use 0.75 as the simulation threshold — architectures that can't reduce
    # loss by 25% in full training almost certainly can't do it in 50 steps.
    if loss_ratio is not None and loss_ratio > 0.75:
        return f"loss_stalled (ratio={loss_ratio:.3f})"

    # Routing collapse — skip exact 0.0 which typically means "not measured"
    if entropy is not None and 0.0 < entropy < checker.routing_entropy_minimum:
        return f"routing_collapse (entropy={entropy:.4f})"

    return None


def calibrate(db_path: Path) -> Dict[str, Any]:
    """Run calibration and return report."""
    results = _load_results(db_path)
    checker = RapidScreeningCheck()

    total = len(results)
    s1_passed = [r for r in results if r.get("stage1_passed")]
    s1_failed = [r for r in results if not r.get("stage1_passed")]

    true_positives = 0  # would kill, actually failed S1
    false_positives = 0  # would kill, actually passed S1
    true_negatives = 0  # would pass, actually passed S1
    false_negatives = 0  # would pass, actually failed S1

    kill_reasons: Dict[str, int] = {}
    fp_details: List[Dict[str, Any]] = []

    for row in results:
        kill = _simulate_kill(row, checker)
        passed_s1 = bool(row.get("stage1_passed"))

        if kill:
            kill_reasons[kill.split(" ")[0]] = (
                kill_reasons.get(kill.split(" ")[0], 0) + 1
            )
            if passed_s1:
                false_positives += 1
                fp_details.append(
                    {
                        "result_id": row["result_id"][:8]
                        if row.get("result_id")
                        else "?",
                        "kill_reason": kill,
                        "loss_ratio": row.get("loss_ratio"),
                    }
                )
            else:
                true_positives += 1
        else:
            if passed_s1:
                true_negatives += 1
            else:
                false_negatives += 1

    n_s1_failed = len(s1_failed)
    n_s1_passed = len(s1_passed)
    tp_rate = true_positives / n_s1_failed if n_s1_failed > 0 else 0
    fp_rate = false_positives / n_s1_passed if n_s1_passed > 0 else 0
    gpu_minutes_saved = true_positives * 2.5  # ~2.5 min per S1 run

    report = {
        "total_results": total,
        "s1_passed": n_s1_passed,
        "s1_failed": n_s1_failed,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "tp_rate": round(tp_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "precision": round(
            true_positives / max(1, true_positives + false_positives), 4
        ),
        "recall": round(tp_rate, 4),
        "gpu_minutes_saved": round(gpu_minutes_saved, 1),
        "kill_reason_counts": kill_reasons,
        "false_positive_details": fp_details[:20],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate rapid screening thresholds")
    parser.add_argument(
        "--db", default="research/lab_notebook.db", help="Path to lab notebook DB"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = _resolve_db(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return

    report = calibrate(db_path)

    print(f"\n{'=' * 60}")
    print("RAPID SCREENING CALIBRATION REPORT")
    print(f"{'=' * 60}")
    print(f"Total results analyzed: {report['total_results']}")
    print(f"  S1 passed: {report['s1_passed']}")
    print(f"  S1 failed: {report['s1_failed']}")
    print()
    print(f"True Positives  (would kill, actually failed):  {report['true_positives']}")
    print(
        f"False Positives (would kill, actually passed):  {report['false_positives']}"
    )
    print(f"True Negatives  (would pass, actually passed):  {report['true_negatives']}")
    print(
        f"False Negatives (would pass, actually failed):  {report['false_negatives']}"
    )
    print()
    print(f"TP Rate (recall):  {report['tp_rate']:.1%}")
    print(f"FP Rate:           {report['fp_rate']:.1%}")
    print(f"Precision:         {report['precision']:.1%}")
    print(f"GPU-minutes saved: ~{report['gpu_minutes_saved']:.0f} min")
    print()
    print("Kill reason breakdown:")
    for reason, count in sorted(
        report["kill_reason_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"  {reason}: {count}")

    if report["false_positive_details"]:
        print(f"\nFalse positives (first {len(report['false_positive_details'])}):")
        for fp in report["false_positive_details"]:
            print(
                f"  {fp['result_id']} loss_ratio={fp['loss_ratio']} reason={fp['kill_reason']}"
            )

    # Target: < 5% FP rate
    if report["fp_rate"] > 0.05:
        print(
            f"\n⚠ FP rate {report['fp_rate']:.1%} exceeds 5% target — thresholds need loosening"
        )
    else:
        print(f"\n✓ FP rate {report['fp_rate']:.1%} is within 5% target")


if __name__ == "__main__":
    main()
