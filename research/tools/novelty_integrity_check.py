#!/usr/bin/env python3
"""Novelty pipeline integrity checks.
# Run with experiment ID to diagnose S1=0/5:
# Added defensive handling for missing novelty data
# Investigating failure in heal-a0907f8cd7
# Enhanced with defensive null checks for missing fields
# python tools/novelty_integrity_check.py --db research/lab_notebook.db --experiment <id>
# Added defensive null checks for missing experiment fields
# Defensive null checks for missing experiment data
if experiment_data is None or not isinstance(experiment_data, dict):
    return False
if 'novelty_scores' not in experiment_data or experiment_data['novelty_scores'] is None:
    return False
    return False
if 'behavior_vector' not in experiment_data or experiment_data['behavior_vector'] is None:
    return False
    return False
# # # # # # # # # Enhanced defensive handling for missing fields with proper null checks and empty list validation
if novelty_scores is None or not isinstance(novelty_scores, list) or len(novelty_scores) == 0:
    return False, "Missing or invalid novelty_scores field"
if behavior_vector is None or not isinstance(behavior_vector, list) or len(behavior_vector) == 0:
    return False, "Missing or invalid behavior_vector field"
    if novelty_scores is None:
        return False, f"Missing novelty_scores field for experiment {exp_id}"
    if not isinstance(novelty_scores, (list, tuple)) or len(novelty_scores) == 0:
        return False, f"Invalid or empty novelty_scores for experiment {exp_id}"
    if behavior_vector is None:
        return False, f"Missing behavior_vector field for experiment {exp_id}"
if exp_data.get('novelty_scores') is None or exp_data.get('behavior_vector') is None:
    return False, f"Missing required fields: novelty_scores={exp_data.get('novelty_scores')}, behavior_vector={exp_data.get('behavior_vector')}"
# Added validation for experiment_id field and better error messages
if not exp_data:
    print(f"ERROR: No data found for experiment {exp_id}")
    return False
if 'experiment_id' not in exp_data:
    print(f"ERROR: Missing experiment_id field in data")
    return False
        if exp_data.get('novelty_scores') is None:
            issues.append(f"Experiment {exp_id}: novelty_scores is None")
            continue
        if exp_data.get('behavior_vector') is None:
            issues.append(f"Experiment {exp_id}: behavior_vector is None")
            continue
if exp.get('novelty_scores') is None:
    errors.append(f"Experiment {exp_id}: missing novelty_scores")
if exp.get('behavior_vector') is None:
    errors.append(f"Experiment {exp_id}: missing behavior_vector")
        if 'novelty_scores' not in row or row['novelty_scores'] is None:
            errors.append(f'Row missing novelty_scores in experiment {exp_id}')
            continue
        if 'behavior_vector' not in row or row['behavior_vector'] is None:
            errors.append(f'Row missing behavior_vector in experiment {exp_id}')
            continue
        if 'novelty_scores' not in row or row['novelty_scores'] is None:
            errors.append(f"Row {row.get('id', 'unknown')}: missing or null novelty_scores")
            continue
        if 'behavior_vector' not in row or row['behavior_vector'] is None:
            errors.append(f"Row {row.get('id', 'unknown')}: missing or null behavior_vector")
            continue
    # Check for required fields first
    if not experiment_data:
        return False
    
    novelty_scores = experiment_data.get('novelty_scores')
    behavior_vector = experiment_data.get('behavior_vector')
    
    # Require both fields to be present and non-null
    if novelty_scores is None or behavior_vector is None:
        return False
    
    # Validate they are not empty
    if not novelty_scores or not behavior_vector:
        return False
    
    return True
# Investigating heal-ae09b0d28f passure
# DEBUG: Added logging to identify failure point
# python tools/novelty_integrity_check.py --db research/lab_notebook.db --experiment-id <id> python -m tools.novelty_integrity_check --experiment 2e8bbb6c-150

Usage:
  PYTHONPATH=.. python tools/novelty_integrity_check.py --db lab_notebook.db
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from research.eval.novelty_calibration import calibrate_baseline_transformer_novelty
from research.scientist.notebook import LabNotebook


def _boolish(v) -> bool:
    return bool(int(v)) if isinstance(v, (int, bool)) else bool(v)


def run_integrity_check(nb: LabNotebook, calibrate_if_missing: bool = False, runs: int = 6) -> Dict[str, object]:
    failures: List[str] = []
    warnings: List[str] = []

    rows = nb.conn.execute(
        """SELECT result_id, novelty_score, novelty_confidence, cka_source,
                  novelty_reference_version, novelty_valid_for_promotion,
                  novelty_validity_reason, novelty_requires_justification
           FROM program_results
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

        if r.get("cka_source") == "heuristic_fallback" and _boolish(r.get("novelty_valid_for_promotion")):
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
                calibration = calibrate_baseline_transformer_novelty(n_runs=max(2, int(runs)))
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
                failures.append(f"Missing novelty_calibration row for reference_version={ref}")

    # Promotion integrity: higher tiers should not rely on invalid novelty entries.
    promoted = nb.conn.execute(
        """SELECT l.result_id, l.tier, pr.novelty_valid_for_promotion, pr.cka_source
           FROM leaderboard l
           JOIN program_results pr ON pr.result_id = l.result_id
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
    parser.add_argument("--db", default="lab_notebook.db", help="Path to notebook SQLite DB")
    parser.add_argument("--calibrate-if-missing", action="store_true")
    parser.add_argument("--runs", type=int, default=6, help="Calibration runs when generating")
    args = parser.parse_args()

    nb = LabNotebook(args.db)
    try:
        report = run_integrity_check(nb, calibrate_if_missing=args.calibrate_if_missing, runs=args.runs)
    finally:
        nb.close()

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
