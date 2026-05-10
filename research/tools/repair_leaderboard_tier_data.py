#!/usr/bin/env python3
"""Repair missing tier-specific leaderboard fields from canonical program results.

Usage:
    python -m research.tools.repair_leaderboard_tier_data --db research/runs.db
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from research.scientist.notebook import LabNotebook


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _repair_row(nb: LabNotebook, row: Dict[str, Any]) -> bool:
    tier = str(row.get("tier") or "screening")
    entry_id = str(row["entry_id"])
    updates: Dict[str, Any] = {}

    screening_loss = _coalesce(
        row.get("screening_loss_ratio"),
        row.get("discovery_loss_ratio"),
        row.get("loss_ratio"),
    )
    screening_nov = _coalesce(
        row.get("screening_novelty"),
        row.get("novelty_score"),
        row.get("novelty_score_legacy"),
    )
    screening_passed = row.get("screening_passed")
    if screening_passed is None:
        screening_passed = 1

    if row.get("screening_loss_ratio") is None and screening_loss is not None:
        updates["screening_loss_ratio"] = screening_loss
    if row.get("screening_novelty") is None and screening_nov is not None:
        updates["screening_novelty"] = screening_nov
    if row.get("screening_passed") is None:
        updates["screening_passed"] = int(bool(screening_passed))

    if tier in {"investigation", "validation", "breakthrough"}:
        inv_loss = _coalesce(
            row.get("investigation_loss_ratio"),
            row.get("discovery_loss_ratio"),
            row.get("loss_ratio"),
        )
        inv_robust = _coalesce(
            row.get("investigation_robustness"),
            1.0
            if tier in {"validation", "breakthrough"} and inv_loss is not None
            else None,
            1.0 if tier == "investigation" and inv_loss is not None else None,
            0.0
            if tier == "investigation" and row.get("investigation_passed") == 0
            else None,
        )
        inv_passed = _coalesce(
            row.get("investigation_passed"),
            1 if tier in {"investigation", "validation", "breakthrough"} else None,
        )
        if row.get("investigation_loss_ratio") is None and inv_loss is not None:
            updates["investigation_loss_ratio"] = inv_loss
        if row.get("investigation_robustness") is None and inv_robust is not None:
            updates["investigation_robustness"] = inv_robust
        if row.get("investigation_passed") is None and inv_passed is not None:
            updates["investigation_passed"] = int(bool(inv_passed))

    if tier in {"validation", "breakthrough"}:
        val_loss = _coalesce(
            row.get("validation_loss_ratio"), row.get("pr_validation_loss_ratio")
        )
        val_baseline = _coalesce(
            row.get("validation_baseline_ratio"),
            row.get("baseline_loss_ratio"),
        )
        val_std = _coalesce(
            row.get("validation_multi_seed_std"),
            row.get("lb_init_sensitivity_std"),
            row.get("init_sensitivity_std"),
        )
        val_passed = _coalesce(
            row.get("validation_passed"),
            1 if val_loss is not None else None,
        )

        if row.get("validation_loss_ratio") is None and val_loss is not None:
            updates["validation_loss_ratio"] = val_loss
        if row.get("validation_baseline_ratio") is None and val_baseline is not None:
            updates["validation_baseline_ratio"] = val_baseline
        if row.get("validation_multi_seed_std") is None and val_std is not None:
            updates["validation_multi_seed_std"] = val_std
        if row.get("validation_passed") is None and val_passed is not None:
            updates["validation_passed"] = int(bool(val_passed))

    if not updates:
        return False

    nb.promote_to_tier(entry_id=entry_id, tier=tier, **updates)
    return True


def repair_leaderboard_tier_data(db_path: str, dry_run: bool = False) -> Dict[str, int]:
    nb = LabNotebook(db_path)
    try:
        pr_cols = nb._get_program_results_columns()
        init_std_select = (
            "pr.init_sensitivity_std AS init_sensitivity_std"
            if "init_sensitivity_std" in pr_cols
            else "NULL AS init_sensitivity_std"
        )
        novelty_legacy_select = (
            "pr.novelty_score_legacy"
            if "novelty_score_legacy" in pr_cols
            else "NULL AS novelty_score_legacy"
        )
        rows = nb.conn.execute(
            f"""
            SELECT l.entry_id, l.result_id, l.tier,
                   l.screening_loss_ratio, l.screening_novelty, l.screening_passed,
                   l.investigation_loss_ratio, l.investigation_robustness, l.investigation_passed,
                   l.validation_loss_ratio, l.validation_baseline_ratio,
                   l.validation_multi_seed_std, l.validation_passed,
                   l.init_sensitivity_std AS lb_init_sensitivity_std,
                   pr.loss_ratio, pr.discovery_loss_ratio, pr.validation_loss_ratio AS pr_validation_loss_ratio,
                   pr.baseline_loss_ratio, pr.novelty_score, {novelty_legacy_select},
                   {init_std_select}
            FROM leaderboard l
            JOIN program_results_compat pr ON pr.result_id = l.result_id
            WHERE
                (l.tier = 'screening' AND (
                    l.screening_loss_ratio IS NULL OR
                    l.screening_novelty IS NULL OR
                    l.screening_passed IS NULL
                ))
                OR
                (l.tier = 'investigation' AND (
                    l.screening_loss_ratio IS NULL OR
                    l.screening_novelty IS NULL OR
                    l.screening_passed IS NULL OR
                    l.investigation_loss_ratio IS NULL OR
                    l.investigation_robustness IS NULL OR
                    l.investigation_passed IS NULL
                ))
                OR
                (l.tier IN ('validation', 'breakthrough') AND (
                    l.screening_loss_ratio IS NULL OR
                    l.screening_novelty IS NULL OR
                    l.screening_passed IS NULL OR
                    l.investigation_loss_ratio IS NULL OR
                    l.investigation_robustness IS NULL OR
                    l.investigation_passed IS NULL OR
                    l.validation_loss_ratio IS NULL OR
                    l.validation_baseline_ratio IS NULL OR
                    l.validation_multi_seed_std IS NULL OR
                    l.validation_passed IS NULL
                ))
            """
        ).fetchall()

        counts = {
            "candidates": len(rows),
            "repaired": 0,
            "screening": 0,
            "investigation": 0,
            "validation": 0,
        }
        for row in rows:
            tier = str(row["tier"] or "screening")
            bucket = "validation" if tier in {"validation", "breakthrough"} else tier
            if dry_run:
                counts[bucket] += 1
                counts["repaired"] += 1
                continue
            if _repair_row(nb, dict(row)):
                counts["repaired"] += 1
                counts[bucket] += 1

        if not dry_run:
            synced = nb.backfill_fingerprint_aggregates()
            counts["fingerprints_synced"] = synced
            nb.conn.commit()
        else:
            counts["fingerprints_synced"] = 0
        return counts
    finally:
        nb.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="research/runs.db", help="Path to notebook SQLite DB"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report affected rows without mutating"
    )
    args = parser.parse_args()

    counts = repair_leaderboard_tier_data(args.db, dry_run=args.dry_run)
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
