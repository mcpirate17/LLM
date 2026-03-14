"""Rescore leaderboard: replace inflated random-token composites with honest metrics.

Problem: screening_loss_ratio was computed on random tokens (memorisation signal),
while reference architectures used investigation/validation metrics.  This made
screening-only entries appear to dominate GPT-2/Mamba when they don't.

This script:
  1. Snapshots current leaderboard into ``leaderboard_snapshot_<version>``
  2. Adds ``scoring_version`` and ``screening_metric_version`` columns
  3. For screening-only rows (no inv/val metrics), applies a confidence
     discount so they stop dominating ranking and grammar learning
  4. Recomputes ``wikitext_score`` with the fixed formula everywhere
  5. Recomputes ``composite_score`` from the best available *real* metric
  6. Tags screening-only rows as ``provisional_random_tokens``
  7. Emits a JSON audit artifact

Usage:
    python -m research.tools.rescore_leaderboard [--db PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCORING_VERSION = "wikitext_rescore_v2"

# Screening-only rows get this confidence factor on their performance term.
# Effectively: composite ≈ old_score * 0.15 for the performance component,
# reflecting that random-token screening_lr is low-confidence evidence.
_SCREENING_ONLY_CONFIDENCE = 0.15


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add scoring_version and screening_metric_version if missing."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(leaderboard)")}
    if "scoring_version" not in existing:
        conn.execute("ALTER TABLE leaderboard ADD COLUMN scoring_version TEXT")
    if "screening_metric_version" not in existing:
        conn.execute("ALTER TABLE leaderboard ADD COLUMN screening_metric_version TEXT")


def _snapshot(conn: sqlite3.Connection, version: str) -> str:
    """Create a full snapshot table of the leaderboard."""
    table_name = f"leaderboard_snapshot_{version}"
    conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
    conn.execute(f"CREATE TABLE [{table_name}] AS SELECT * FROM leaderboard")
    count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
    return table_name, count


def _fixed_wikitext_score(ppl: Optional[float], vocab_size: int = 32000) -> Optional[float]:
    """log(vocab/ppl) / log(vocab) — 1.0 for perfect, 0.0 for random."""
    if ppl is None or ppl <= 0:
        return None
    return max(0.0, min(1.0, math.log(vocab_size / ppl) / math.log(vocab_size)))


def _fixed_tinystories_score(ppl: Optional[float], vocab_size: int = 32000) -> Optional[float]:
    """Same formula as wikitext_score."""
    return _fixed_wikitext_score(ppl, vocab_size)


def _is_screening_only(row: Dict[str, Any]) -> bool:
    """True if row has no investigation or validation metrics."""
    inv_lr = row.get("investigation_loss_ratio")
    val_lr = row.get("validation_loss_ratio")
    val_base = row.get("validation_baseline_ratio")
    has_real = (
        (inv_lr is not None and inv_lr > 0)
        or (val_lr is not None and val_lr > 0)
        or (val_base is not None and val_base > 0)
    )
    return not has_real


def _recompute_composite(
    row: Dict[str, Any],
    is_screening_only: bool,
) -> float:
    """Recompute composite_score using best available real metric."""
    from research.scientist.leaderboard_scoring import compute_composite_score
    from research.scientist.leaderboard_schema import SCORE_COLUMN_MAP

    kw: Dict[str, Any] = {}
    for col, param in SCORE_COLUMN_MAP.items():
        kw[param] = row.get(col)

    # If screening-only, downgrade screening_lr confidence by passing it
    # through a separate path: we NULL out screening_lr and instead inject
    # a very low-confidence performance term manually via the val_lr slot
    # with a heavy discount.
    if is_screening_only:
        screen_lr = kw.get("screening_lr")
        # Clear all performance metrics — the only one was screening_lr
        kw["screening_lr"] = None
        kw["inv_lr"] = None
        kw["val_lr"] = None
        kw["val_baseline"] = None
        # Re-inject as val_lr with heavy confidence discount
        # The scoring function gives val_lr confidence=1.0, so we pre-discount
        # the metric itself: effective_lr = 1 - (1-lr) * confidence
        if screen_lr is not None and screen_lr > 0:
            raw_perf = 1.0 - screen_lr  # typically 0.99 for screening
            discounted_perf = raw_perf * _SCREENING_ONLY_CONFIDENCE
            kw["screening_lr"] = 1.0 - discounted_perf  # ~0.85

    kw["novelty_confidence"] = row.get("novelty_confidence")
    kw["is_reference"] = bool(row.get("is_reference"))
    kw["scaling_param_efficiency"] = (
        row.get("scaling_param_efficiency") or row.get("efficiency_multiple")
    )
    kw["loss_improvement_rate"] = row.get("loss_improvement_rate")
    kw["wikitext_perplexity"] = row.get("wikitext_perplexity")
    kw["wikitext_score"] = row.get("wikitext_score")
    kw["investigation_passed"] = row.get("investigation_passed")
    kw["validation_passed"] = row.get("validation_passed")

    return float(compute_composite_score(**kw))


def _rescore(db_path: str, dry_run: bool = False) -> Dict[str, Any]:
    """Run the rescore and return audit dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Snapshot
    _ensure_columns(conn)
    snap_table, snap_count = _snapshot(conn, SCORING_VERSION)
    print(f"Snapshot: {snap_table} ({snap_count} rows)")

    # 2. Fetch all rows
    rows = conn.execute("SELECT * FROM leaderboard").fetchall()
    rows = [dict(r) for r in rows]
    total = len(rows)

    # 3. Classify and rescore
    audit: Dict[str, Any] = {
        "version": SCORING_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "db_path": db_path,
        "snapshot_table": snap_table,
        "total_rows": total,
        "dry_run": dry_run,
    }

    tier_counts: Dict[str, int] = {}
    changed_count = 0
    screening_only_count = 0
    wikitext_fixed_count = 0
    no_real_metrics_count = 0

    before_top20: List[Dict[str, Any]] = []
    after_top20: List[Dict[str, Any]] = []

    # Collect before-top20
    sorted_before = sorted(rows, key=lambda r: r.get("composite_score") or 0, reverse=True)
    for r in sorted_before[:20]:
        before_top20.append({
            "result_id": r["result_id"],
            "tier": r.get("tier"),
            "composite_score": r.get("composite_score"),
            "screening_loss_ratio": r.get("screening_loss_ratio"),
            "investigation_loss_ratio": r.get("investigation_loss_ratio"),
            "wikitext_perplexity": r.get("wikitext_perplexity"),
            "is_reference": bool(r.get("is_reference")),
        })

    updates: List[Dict[str, Any]] = []

    for row in rows:
        tier = row.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        old_composite = row.get("composite_score") or 0
        old_wiki_score = row.get("wikitext_score")
        screening_only = _is_screening_only(row)

        if screening_only:
            screening_only_count += 1

        # Fix wikitext_score
        new_wiki_score = old_wiki_score
        wiki_ppl = row.get("wikitext_perplexity")
        if wiki_ppl is not None:
            new_wiki_score = _fixed_wikitext_score(wiki_ppl)
            if new_wiki_score != old_wiki_score:
                wikitext_fixed_count += 1

        # Fix tinystories_score
        new_ts_score = row.get("tinystories_score")
        ts_ppl = row.get("tinystories_perplexity")
        if ts_ppl is not None:
            new_ts_score = _fixed_tinystories_score(ts_ppl)

        if wiki_ppl is None and screening_only:
            no_real_metrics_count += 1

        # Recompute composite
        new_composite = _recompute_composite(row, screening_only)

        # Determine screening_metric_version
        if screening_only:
            metric_version = "random_tokens_discounted"
        elif row.get("validation_baseline_ratio") is not None:
            metric_version = "validation_baseline"
        elif row.get("validation_loss_ratio") is not None:
            metric_version = "validation_lr"
        elif row.get("investigation_loss_ratio") is not None:
            metric_version = "investigation_lr"
        else:
            metric_version = "unknown"

        # Determine tags update
        tags = row.get("tags") or ""
        if screening_only and "provisional_random_tokens" not in tags:
            tags = f"{tags},provisional_random_tokens" if tags else "provisional_random_tokens"

        changed = (
            abs(new_composite - old_composite) > 0.01
            or new_wiki_score != old_wiki_score
            or new_ts_score != row.get("tinystories_score")
        )

        if changed:
            changed_count += 1
            updates.append({
                "entry_id": row["entry_id"],
                "composite_score": round(new_composite, 4),
                "wikitext_score": round(new_wiki_score, 4) if new_wiki_score is not None else None,
                "tinystories_score": round(new_ts_score, 4) if new_ts_score is not None else None,
                "scoring_version": SCORING_VERSION,
                "screening_metric_version": metric_version,
                "tags": tags,
            })

        # Update row dict for after-top20 computation
        row["composite_score"] = new_composite

    # Collect after-top20
    sorted_after = sorted(rows, key=lambda r: r.get("composite_score") or 0, reverse=True)
    for r in sorted_after[:20]:
        after_top20.append({
            "result_id": r["result_id"],
            "tier": r.get("tier"),
            "composite_score": round(r.get("composite_score") or 0, 4),
            "screening_loss_ratio": r.get("screening_loss_ratio"),
            "investigation_loss_ratio": r.get("investigation_loss_ratio"),
            "wikitext_perplexity": r.get("wikitext_perplexity"),
            "is_reference": bool(r.get("is_reference")),
        })

    audit["tier_counts"] = tier_counts
    audit["rows_changed"] = changed_count
    audit["screening_only_count"] = screening_only_count
    audit["wikitext_score_fixed_count"] = wikitext_fixed_count
    audit["no_real_token_metrics_count"] = no_real_metrics_count
    audit["before_top20"] = before_top20
    audit["after_top20"] = after_top20

    # 4. Apply updates
    if not dry_run and updates:
        for upd in updates:
            conn.execute(
                """UPDATE leaderboard
                   SET composite_score = ?,
                       wikitext_score = ?,
                       tinystories_score = ?,
                       scoring_version = ?,
                       screening_metric_version = ?,
                       tags = ?
                   WHERE entry_id = ?""",
                (
                    upd["composite_score"],
                    upd["wikitext_score"],
                    upd["tinystories_score"],
                    upd["scoring_version"],
                    upd["screening_metric_version"],
                    upd["tags"],
                    upd["entry_id"],
                ),
            )
        conn.commit()
        print(f"Applied {len(updates)} updates")
    elif dry_run:
        print(f"DRY RUN: would update {len(updates)} rows")
    else:
        print("No changes needed")

    conn.close()
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore leaderboard with honest metrics")
    parser.add_argument("--db", default="lab_notebook.db", help="Path to database")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    audit = _rescore(args.db, dry_run=args.dry_run)

    # Write audit artifact
    artifact_dir = Path("rescore_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / f"rescore_audit_{SCORING_VERSION}_{int(time.time())}.json"
    artifact_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nAudit artifact: {artifact_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESCORE SUMMARY — {SCORING_VERSION}")
    print(f"{'='*60}")
    print(f"Total rows:              {audit['total_rows']}")
    print(f"Rows changed:            {audit['rows_changed']}")
    print(f"Screening-only:          {audit['screening_only_count']}")
    print(f"WikiText score fixed:    {audit['wikitext_score_fixed_count']}")
    print(f"No real-token metrics:   {audit['no_real_token_metrics_count']}")
    print(f"\nTier counts:")
    for tier, cnt in sorted(audit["tier_counts"].items()):
        print(f"  {tier:20s}: {cnt}")

    print(f"\n--- BEFORE top 20 ---")
    for i, r in enumerate(audit["before_top20"]):
        ref = " [REF]" if r["is_reference"] else ""
        print(f"  {i+1:2d}. composite={r['composite_score']:8.2f}  tier={r['tier']:14s}  "
              f"wiki_ppl={str(r.get('wikitext_perplexity') or 'None'):>8s}  "
              f"inv_lr={r.get('investigation_loss_ratio') or 'None'}{ref}")

    print(f"\n--- AFTER top 20 ---")
    for i, r in enumerate(audit["after_top20"]):
        ref = " [REF]" if r["is_reference"] else ""
        print(f"  {i+1:2d}. composite={r['composite_score']:8.4f}  tier={r['tier']:14s}  "
              f"wiki_ppl={str(r.get('wikitext_perplexity') or 'None'):>8s}  "
              f"inv_lr={r.get('investigation_loss_ratio') or 'None'}{ref}")


if __name__ == "__main__":
    main()
