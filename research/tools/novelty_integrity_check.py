#!/usr/bin/env python3
"""Novelty pipeline integrity checks.

Usage:
  PYTHONPATH=.. python tools/novelty_integrity_check.py --db runs.db
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from research.eval.novelty_calibration import calibrate_baseline_transformer_novelty
from research.scientist.notebook import LabNotebook


def _boolish(v) -> bool:
    return bool(int(v)) if isinstance(v, (int, bool)) else bool(v)


def run_integrity_check(
    nb: LabNotebook, calibrate_if_missing: bool = False, runs: int = 6
) -> Dict[str, object]:
    failures: List[str] = []
    warnings: List[str] = []

    rows = nb.conn.execute(
        """SELECT result_id, novelty_score, novelty_confidence, cka_source,
                  novelty_reference_version, novelty_valid_for_promotion,
                  novelty_validity_reason, novelty_requires_justification
           FROM program_results_compat
           WHERE novelty_score IS NOT NULL"""
    ).fetchall()

    used_refs = set()
    for row in rows:
        r = dict(row)
        used_refs.add(r.get("novelty_reference_version") or "")
        if not r.get("novelty_reference_version"):
            failures.append(f"{r.get('result_id')}: missing novelty_reference_version")
        if not r.get("cka_source"):
            failures.append(f"{r.get('result_id')}: missing cka_source")

        if r.get("cka_source") == "heuristic_fallback" and _boolish(
            r.get("novelty_valid_for_promotion")
        ):
            reason = str(r.get("novelty_validity_reason") or "")
            justified = _boolish(r.get("novelty_requires_justification"))
            if not (reason.startswith("override:") and justified):
                failures.append(
                    f"{r.get('result_id')}: heuristic novelty marked valid without explicit override+justification"
                )

    for ref in sorted(r for r in used_refs if r):
        cal = nb.get_latest_novelty_calibration(reference_version=ref)
        if cal is None:
            if calibrate_if_missing:
                calibration = calibrate_baseline_transformer_novelty(
                    n_runs=max(2, int(runs))
                )
                nb.record_novelty_calibration(
                    reference_version=ref,
                    cka_source=calibration.get("cka_source"),
                    cka_artifact_version=calibration.get("cka_artifact_version"),
                    probe_protocol_hash=calibration.get("probe_protocol_hash"),
                    n_runs=calibration.get("n_runs") or runs,
                    noise_floor_mean=calibration.get("noise_floor_mean"),
                    noise_floor_std=calibration.get("noise_floor_std"),
                    confidence_low=calibration.get("confidence_low"),
                    confidence_high=calibration.get("confidence_high"),
                    distribution=calibration.get("distribution") or {},
                    metadata={"generated_by": "novelty_integrity_check"},
                )
                warnings.append(f"Missing calibration for {ref}; generated one.")
            else:
                failures.append(
                    f"Missing novelty_calibration row for reference_version={ref}"
                )

    # Promotion integrity: higher tiers should not rely on invalid novelty entries.
    promoted = nb.conn.execute(
        """SELECT l.result_id, l.tier, pr.novelty_valid_for_promotion, pr.cka_source
           FROM leaderboard l
           JOIN program_results_compat pr ON pr.result_id = l.result_id
           WHERE l.tier IN ('validation', 'breakthrough')"""
    ).fetchall()
    for row in promoted:
        r = dict(row)
        valid = _boolish(r.get("novelty_valid_for_promotion"))
        if not valid and r.get("cka_source") != "artifact":
            failures.append(
                f"{r.get('result_id')}: tier={r.get('tier')} but novelty_valid_for_promotion is false"
            )

    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "checked_program_rows": len(rows),
        "checked_promoted_rows": len(promoted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Novelty scoring integrity checks")
    parser.add_argument("--db", default="runs.db", help="Path to notebook SQLite DB")
    parser.add_argument("--calibrate-if-missing", action="store_true")
    parser.add_argument(
        "--runs", type=int, default=6, help="Calibration runs when generating"
    )
    args = parser.parse_args()

    nb = LabNotebook(args.db)
    try:
        report = run_integrity_check(
            nb, calibrate_if_missing=args.calibrate_if_missing, runs=args.runs
        )
    finally:
        nb.close()

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
