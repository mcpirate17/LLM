#!/usr/bin/env python3
"""One-shot migration: fix broken baseline ratios and re-evaluate leaderboard tiers.

Problems fixed:
  1. Baseline was trained with n_layers=config.n_layers (4) instead of 2.
     A 4-layer transformer on random data with 500 steps achieves loss ≈ ln(32000)
     (random chance), so any model that learns at all gets ratio ≈ 0.  All existing
     validation_baseline_ratio values are meaningless.

  2. Breakthrough thresholds tightened:
       - raw threshold: 0.90 → 0.70
       - normalized threshold: 0.85 (new)
       - FLOP gate: reject if flops_per_token > 5x baseline

This script:
  1. Nullifies all validation_baseline_ratio and normalized_baseline_ratio values
     (they were computed against a broken baseline)
  2. Demotes any breakthrough entries to validation (they must re-qualify)
  3. Clears the stale baseline cache so new 2-layer baselines are trained
  4. Reports what changed

Usage:
    python -m research.tools.migrate_leaderboard_tiers [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path


def run_migration(db_path: str, dry_run: bool = False, verbose: bool = True):
    """Invalidate broken baseline ratios, demote breakthroughs, clear cache."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Ensure new columns exist
    existing_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(leaderboard)").fetchall()
    }
    new_cols = {
        "normalized_baseline_ratio": "REAL",
        "param_efficiency": "REAL",
        "quant_int8_retention": "REAL",
        "quant_quality_per_byte": "REAL",
        "robustness_long_ctx_score": "REAL",
        "robustness_noise_score": "REAL",
        "init_sensitivity_std": "REAL",
    }
    for col_name, col_type in new_cols.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE leaderboard ADD COLUMN {col_name} {col_type}")
    conn.commit()

    # --- Count what we're about to change ---
    n_with_bl = conn.execute(
        "SELECT COUNT(*) FROM leaderboard WHERE validation_baseline_ratio IS NOT NULL"
    ).fetchone()[0]

    n_with_norm = conn.execute(
        "SELECT COUNT(*) FROM leaderboard WHERE normalized_baseline_ratio IS NOT NULL"
    ).fetchone()[0]

    n_breakthrough = conn.execute(
        "SELECT COUNT(*) FROM leaderboard WHERE tier = 'breakthrough'"
    ).fetchone()[0]

    # Also check program_results.baseline_loss_ratio
    pr_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(program_results)").fetchall()
    }
    n_pr_baseline = 0
    if "baseline_loss_ratio" in pr_cols:
        n_pr_baseline = conn.execute(
            "SELECT COUNT(*) FROM program_results WHERE baseline_loss_ratio IS NOT NULL"
        ).fetchone()[0]

    # --- Check baseline cache ---
    cache_path = Path(db_path).parent / "baseline_cache.db"
    n_cache_entries = 0
    n_stale_cache = 0
    if cache_path.exists():
        cache_conn = sqlite3.connect(str(cache_path))
        try:
            rows = cache_conn.execute(
                "SELECT config_key, final_loss FROM baseline_results"
            ).fetchall()
            n_cache_entries = len(rows)
            random_chance = math.log(32000)
            for key, loss in rows:
                if loss >= random_chance * 0.95:
                    n_stale_cache += 1
        except sqlite3.OperationalError:
            pass
        cache_conn.close()

    # --- Report ---
    if verbose:
        print("Leaderboard baseline ratio migration")
        print(f"  DB: {db_path}")
        print()
        print("Problem: baseline was trained with n_layers=4 instead of 2.")
        print(
            f"  All {n_cache_entries} cached baselines have loss >= ln(32000) (random chance)."
        )
        print(f"  All {n_with_bl} leaderboard baseline_ratio values are meaningless.")
        print()
        print("Actions:")
        print(
            f"  1. NULL out validation_baseline_ratio on {n_with_bl} leaderboard entries"
        )
        print(
            f"  2. NULL out normalized_baseline_ratio on {n_with_norm} leaderboard entries"
        )
        print(f"  3. NULL out param_efficiency on {n_with_norm} leaderboard entries")
        print(
            f"  4. NULL out baseline_loss_ratio on {n_pr_baseline} program_results entries"
        )
        print(f"  5. Demote {n_breakthrough} breakthrough entries → validation")
        print(
            f"  6. Delete {n_stale_cache}/{n_cache_entries} stale baseline cache entries"
        )
        print()

    if dry_run:
        print("DRY RUN — no changes applied")
        return {
            "bl_nullified": n_with_bl,
            "norm_nullified": n_with_norm,
            "pr_nullified": n_pr_baseline,
            "demoted": n_breakthrough,
            "cache_cleared": n_stale_cache,
        }

    # --- Apply ---

    # 1-3: Null out broken leaderboard ratios
    conn.execute("""
        UPDATE leaderboard
        SET validation_baseline_ratio = NULL,
            normalized_baseline_ratio = NULL,
            param_efficiency = NULL
        WHERE validation_baseline_ratio IS NOT NULL
           OR normalized_baseline_ratio IS NOT NULL
    """)

    # 4: Null out broken program_results baseline ratios
    if "baseline_loss_ratio" in pr_cols and n_pr_baseline > 0:
        conn.execute("""
            UPDATE program_results
            SET baseline_loss_ratio = NULL
            WHERE baseline_loss_ratio IS NOT NULL
        """)

    # 5: Demote breakthroughs
    if n_breakthrough > 0:
        conn.execute("""
            UPDATE leaderboard
            SET tier = 'validation'
            WHERE tier = 'breakthrough'
        """)

    conn.commit()
    conn.close()

    # 6: Clear stale baseline cache
    if cache_path.exists() and n_stale_cache > 0:
        cache_conn = sqlite3.connect(str(cache_path))
        random_chance = math.log(32000)
        cache_conn.execute(
            "DELETE FROM baseline_results WHERE final_loss >= ?",
            (random_chance * 0.95,),
        )
        cache_conn.commit()
        cache_conn.close()

    if verbose:
        print("Done.")
        print(f"  {n_with_bl} leaderboard baseline ratios nullified")
        print(f"  {n_pr_baseline} program_results baseline ratios nullified")
        print(f"  {n_breakthrough} breakthroughs demoted to validation")
        print(f"  {n_stale_cache} stale cache entries deleted")
        print()
        print("Next steps:")
        print("  - Run experiments normally; baseline will retrain with 2 layers")
        print(
            "  - Existing validation entries will get fresh baseline_ratio on re-validation"
        )
        print("  - The sanity check in baseline.compare() now returns 1.0 if baseline")
        print("    hasn't learned (loss >= 95% of random chance)")

    return {
        "bl_nullified": n_with_bl,
        "norm_nullified": n_with_norm,
        "pr_nullified": n_pr_baseline,
        "demoted": n_breakthrough,
        "cache_cleared": n_stale_cache,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix broken baseline ratios and re-evaluate leaderboard tiers"
    )
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab_notebook.db (default: research/lab_notebook.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying the database",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    run_migration(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
