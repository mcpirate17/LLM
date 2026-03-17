#!/usr/bin/env python3
"""Leaderboard rescorer v3 — CKA fix + scoring version unification.

Recomputes composite scores using:
  - Downgraded novelty confidence (CKA still all-zeros in DB, code fixed
    but entries not re-fingerprinted → novelty_confidence=0.2 structural-only)
  - tiktoken perplexity where available (from TIKTOKEN_RERUN_RESULTS)
  - Unified wikitext_rescore_v2 formula for all entries
  - provisional_random_tokens entries get composite_score=NULL
  - Preserves old_composite_score before overwriting
  - Tracks rescore_reason per entry

Usage:
  python -m research.tools.rescore_leaderboard --dry-run   # preview only
  python -m research.tools.rescore_leaderboard --apply      # write to DB
  python -m research.tools.rescore_leaderboard --entry-id <id>  # single entry
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from ..defaults import VOCAB_SIZE

SCORING_VERSION = "wikitext_rescore_v6"

# Screening-only rows get this confidence factor on their performance term.
_SCREENING_ONLY_CONFIDENCE = 0.15

# CKA is still all-zeros in program_results (code fixed, DB not re-fingerprinted).
# Downgrade novelty_confidence to structural-only level for all CKA-broken entries.
_CKA_BROKEN_NOVELTY_CONFIDENCE = 0.2

# Demotion threshold: if new score drops more than this below old, flag DEMOTION_RISK.
_DEMOTION_THRESHOLD = 20.0


def _fixed_wikitext_score(
    ppl: Optional[float],
    vocab_size: int = VOCAB_SIZE,
) -> Optional[float]:
    """log(vocab/ppl) / log(vocab) — 1.0 for perfect, 0.0 for random."""
    if ppl is None or ppl <= 0:
        return None
    return max(0.0, min(1.0, math.log(vocab_size / ppl) / math.log(vocab_size)))


def _is_screening_only(row: dict[str, Any]) -> bool:
    """True if row has no investigation or validation metrics."""
    return not any(
        row.get(c) is not None and row.get(c) > 0
        for c in (
            "investigation_loss_ratio",
            "validation_loss_ratio",
            "validation_baseline_ratio",
        )
    )


def _classify_rescore_reasons(
    row: dict[str, Any],
    pr: dict[str, Any],
) -> list[str]:
    """Determine which rescore reasons apply to this entry."""
    reasons: list[str] = []
    is_ref = bool(row.get("is_reference"))

    # REASON 1: CKA broken
    if (
        pr.get("fp_cka_vs_transformer") == 0.0
        and pr.get("cka_source") == "artifact"
        and not is_ref
    ):
        reasons.append("cka_broken")

    # REASON 2: Wrong scoring version
    if row.get("scoring_version") != SCORING_VERSION:
        reasons.append("scoring_version_mismatch")

    # REASON 3: Byte-era perplexity (no tiktoken rerun)
    # All non-reference entries currently use byte-era PPL
    # tiktoken reruns only exist for references (from TIKTOKEN_RERUN_RESULTS.md)
    if not is_ref and row.get("wikitext_perplexity") is not None:
        reasons.append("byte_era_perplexity")

    # REASON 4: provisional_random_tokens
    if row.get("screening_metric_version") == "random_tokens_discounted":
        reasons.append("provisional_random_tokens")

    # REASON 5: Explicit pending flag
    if row.get("rescore_status") in ("pending", "cka_broken_pending"):
        reasons.append("explicit_pending")

    # References always get rescored for consistency
    if is_ref:
        reasons.append("reference_baseline")

    return reasons


def _recompute_composite(
    row: dict[str, Any],
    pr: dict[str, Any],
    is_screening_only: bool,
    cka_broken: bool,
    gpt2_wikitext_ppl: Optional[float] = None,
    gpt2_raw_anchor: Optional[float] = None,
) -> Optional[float]:
    """Recompute composite_score using v6 formula (GPT-2 PPL = 100 anchor)."""
    from research.scientist.leaderboard_scoring import compute_composite_v6

    is_ref = bool(row.get("is_reference"))
    if is_screening_only and not is_ref and row.get("wikitext_perplexity") is None:
        return None

    kw: dict[str, Any] = {}

    # WikiText perplexity — the ONE performance anchor for v6
    kw["wikitext_perplexity"] = row.get("wikitext_perplexity") or pr.get(
        "wikitext_perplexity"
    )
    if gpt2_wikitext_ppl is not None:
        kw["gpt2_wikitext_ppl"] = gpt2_wikitext_ppl

    # Loss ratios (for hard learning gate only, not performance)
    kw["screening_lr"] = row.get("screening_loss_ratio")
    kw["inv_lr"] = row.get("investigation_loss_ratio")
    kw["val_lr"] = row.get("validation_loss_ratio")
    kw["val_baseline"] = row.get("validation_baseline_ratio")
    kw["val_std"] = row.get("validation_multi_seed_std")
    kw["inv_robust"] = row.get("investigation_robustness")
    kw["loss_ratio"] = pr.get("loss_ratio")

    # Novelty
    fresh_novelty = pr.get("novelty_score")
    kw["screening_nov"] = (
        fresh_novelty if fresh_novelty is not None else row.get("screening_novelty")
    )

    real_cka = (
        pr.get("fp_cka_vs_transformer") is not None
        and pr.get("fp_cka_vs_transformer") > 0.0
    )
    if cka_broken and not is_ref and not real_cka:
        kw["novelty_confidence"] = _CKA_BROKEN_NOVELTY_CONFIDENCE
    else:
        kw["novelty_confidence"] = pr.get("novelty_confidence") or row.get(
            "novelty_confidence"
        )

    kw["behavioral_novelty"] = pr.get("behavioral_novelty")
    kw["structural_novelty"] = pr.get("structural_novelty")
    kw["cka_reference_quality"] = real_cka

    # Convergence
    kw["loss_improvement_rate"] = pr.get("loss_improvement_rate")

    # Training budget (step gate)
    kw["n_train_steps"] = pr.get("n_train_steps")

    # Efficiency
    kw["param_count"] = pr.get("param_count")

    # Robustness
    kw["spectral_norm"] = row.get("fp_jacobian_spectral_norm")
    kw["investigation_passed"] = row.get("investigation_passed")
    kw["validation_passed"] = row.get("validation_passed")

    kw["is_reference"] = is_ref

    # Anchor 2: final_loss for wikitext103+tiktoken trained models
    kw["final_loss"] = pr.get("final_loss")
    tags = row.get("tags") or ""
    kw["is_wikitext_tiktoken"] = "tiktoken_native" in tags and "wikitext103" in tags

    # Normalization anchor (GPT-2's raw score = 100.0)
    if gpt2_raw_anchor is not None:
        kw["gpt2_raw_anchor"] = gpt2_raw_anchor

    return float(compute_composite_v6(**kw))


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add any missing columns needed for v3 rescore."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(leaderboard)")}
    for col, typ in [
        ("scoring_version", "TEXT"),
        ("screening_metric_version", "TEXT"),
        ("old_composite_score", "REAL"),
        ("rescore_status", "TEXT"),
        ("rescore_reason", "TEXT"),
        ("rescore_timestamp", "TEXT"),
        ("perplexity_tokenizer_penalty", "INTEGER"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE leaderboard ADD COLUMN {col} {typ}")


def _snapshot(conn: sqlite3.Connection, version: str) -> tuple[str, int]:
    """Create a full snapshot table of the leaderboard."""
    table_name = f"leaderboard_snapshot_{version}"
    conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
    conn.execute(f"CREATE TABLE [{table_name}] AS SELECT * FROM leaderboard")
    count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
    return table_name, count


def _load_program_results(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Load program_results keyed by result_id."""
    rows = conn.execute("SELECT * FROM program_results").fetchall()
    return {dict(r)["result_id"]: dict(r) for r in rows}


def _rescore(
    db_path: str,
    dry_run: bool = False,
    entry_id_filter: Optional[str] = None,
) -> dict[str, Any]:
    """Run the rescore and return audit dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _ensure_columns(conn)
    snap_table, snap_count = _snapshot(conn, SCORING_VERSION)
    print(f"Snapshot: {snap_table} ({snap_count} rows)")

    pr_map = _load_program_results(conn)

    # Fetch leaderboard rows
    if entry_id_filter:
        rows = conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?", (entry_id_filter,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM leaderboard").fetchall()
    rows = [dict(r) for r in rows]
    total = len(rows)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    audit: dict[str, Any] = {
        "version": SCORING_VERSION,
        "timestamp": now,
        "db_path": db_path,
        "snapshot_table": snap_table,
        "total_rows": total,
        "dry_run": dry_run,
    }

    # Collect before-top10
    sorted_before = sorted(
        rows, key=lambda r: r.get("composite_score") or 0, reverse=True
    )
    before_top10 = [
        {
            "entry_id": r["entry_id"],
            "arch": r.get("architecture_desc", "")[:60],
            "composite": r.get("composite_score"),
            "tier": r.get("tier"),
            "is_ref": bool(r.get("is_reference")),
        }
        for r in sorted_before[:10]
    ]

    updates: list[dict[str, Any]] = []
    stats = {
        "rescored": 0,
        "nulled": 0,
        "up": 0,
        "down": 0,
        "unchanged": 0,
        "cka_broken": 0,
        "version_mismatch": 0,
        "byte_era": 0,
        "random_tokens": 0,
        "demotion_risk": [],
    }
    deltas: list[float] = []

    # GPT-2 anchor for v6 scoring.
    # Use GPT-2-wikitext103 (tiktoken, 28.8M params) as the anchor reference.
    # Raw anchor = perf(60) + conv(20) + eff(10) + novelty(0) + robust(5) = 95
    # This is GPT-2's expected score when it matches itself on all terms.
    # No DB contamination — the anchor is the formula's identity point.
    gpt2_raw_anchor = 95.0
    gpt2_wikitext_ppl = None

    # Find GPT-2 references for PPL anchor
    gpt2_row = next(
        (
            r
            for r in rows
            if r.get("is_reference") and "gpt2" in (r.get("tags") or "").lower()
        ),
        None,
    )
    if gpt2_row:
        gpt2_wikitext_ppl = gpt2_row.get("wikitext_perplexity")
        if gpt2_wikitext_ppl:
            print(f"GPT-2 WikiText PPL anchor: {gpt2_wikitext_ppl:.2f}")

    gpt2_wiki_row = next(
        (
            r
            for r in rows
            if r.get("is_reference") and r.get("reference_name") == "GPT-2-wikitext103"
        ),
        None,
    )
    if gpt2_wiki_row:
        gpt2_wiki_pr = pr_map.get(gpt2_wiki_row.get("result_id", ""), {})
        print(
            f"GPT-2 WikiText-103: final_loss={gpt2_wiki_pr.get('final_loss')}, params={gpt2_wiki_pr.get('param_count')}"
        )

    print(f"GPT-2 raw anchor: {gpt2_raw_anchor:.2f} (normalizes to 100.0)")

    for row in rows:
        pr = pr_map.get(row.get("result_id", ""), {})
        is_ref = bool(row.get("is_reference"))
        screening_only = _is_screening_only(row)
        reasons = _classify_rescore_reasons(row, pr)

        if not reasons and not entry_id_filter:
            continue  # Nothing to fix for this entry

        cka_broken = "cka_broken" in reasons
        old_composite = row.get("composite_score") or 0.0
        new_composite = _recompute_composite(
            row,
            pr,
            screening_only,
            cka_broken,
            gpt2_wikitext_ppl=gpt2_wikitext_ppl,
            gpt2_raw_anchor=gpt2_raw_anchor,
        )

        # Determine rescore_status
        if new_composite is None:
            rescore_status = "needs_corpus_rescreen"
            stats["nulled"] += 1
        else:
            rescore_status = "rescored_v6"

        # Track stats
        if "cka_broken" in reasons:
            stats["cka_broken"] += 1
        if "scoring_version_mismatch" in reasons:
            stats["version_mismatch"] += 1
        if "byte_era_perplexity" in reasons:
            stats["byte_era"] += 1
        if "provisional_random_tokens" in reasons:
            stats["random_tokens"] += 1

        # Delta tracking
        if new_composite is not None:
            delta = new_composite - old_composite
            deltas.append(delta)
            if delta > 0.01:
                stats["up"] += 1
            elif delta < -0.01:
                stats["down"] += 1
            else:
                stats["unchanged"] += 1

            # Demotion risk check for validated entries
            if (
                row.get("tier") == "validation"
                and row.get("validation_passed")
                and delta < -_DEMOTION_THRESHOLD
            ):
                stats["demotion_risk"].append(
                    {
                        "entry_id": row["entry_id"],
                        "arch": row.get("architecture_desc", "")[:60],
                        "old": round(old_composite, 2),
                        "new": round(new_composite, 2),
                        "delta": round(delta, 2),
                    }
                )

        # Fix wikitext_score
        wiki_ppl = row.get("wikitext_perplexity")
        new_wiki_score = (
            _fixed_wikitext_score(wiki_ppl) if wiki_ppl else row.get("wikitext_score")
        )

        # Fix tinystories_score
        ts_ppl = row.get("tinystories_perplexity")
        new_ts_score = (
            _fixed_wikitext_score(ts_ppl) if ts_ppl else row.get("tinystories_score")
        )

        # Perplexity tokenizer penalty: non-ref entries without tiktoken rerun
        ppl_penalty = 1 if ("byte_era_perplexity" in reasons and not is_ref) else 0

        # Tags
        tags = row.get("tags") or ""
        if screening_only and "provisional_random_tokens" not in tags:
            tags = (
                f"{tags},provisional_random_tokens"
                if tags
                else "provisional_random_tokens"
            )
        if "needs_rescreening" in tags and rescore_status == "rescored_v6":
            tags = ",".join(t for t in tags.split(",") if t != "needs_rescreening")

        # Screening metric version
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

        stats["rescored"] += 1
        updates.append(
            {
                "entry_id": row["entry_id"],
                "old_composite_score": old_composite,
                "composite_score": round(new_composite, 4)
                if new_composite is not None
                else None,
                "wikitext_score": round(new_wiki_score, 4)
                if new_wiki_score is not None
                else None,
                "tinystories_score": round(new_ts_score, 4)
                if new_ts_score is not None
                else None,
                "scoring_version": SCORING_VERSION,
                "screening_metric_version": metric_version,
                "rescore_status": rescore_status,
                "rescore_reason": ",".join(reasons),
                "rescore_timestamp": now,
                "perplexity_tokenizer_penalty": ppl_penalty,
                "tags": tags,
            }
        )

        # Update row for after-top10
        row["composite_score"] = new_composite

    # After-top10
    sorted_after = sorted(
        rows,
        key=lambda r: (
            r.get("composite_score") if r.get("composite_score") is not None else -1
        ),
        reverse=True,
    )
    after_top10 = [
        {
            "entry_id": r["entry_id"],
            "arch": r.get("architecture_desc", "")[:60],
            "composite": round(r["composite_score"], 2)
            if r.get("composite_score") is not None
            else None,
            "tier": r.get("tier"),
            "is_ref": bool(r.get("is_reference")),
        }
        for r in sorted_after[:10]
    ]

    audit["stats"] = stats
    audit["before_top10"] = before_top10
    audit["after_top10"] = after_top10
    if deltas:
        audit["avg_delta"] = round(sum(deltas) / len(deltas), 4)
        audit["min_delta"] = round(min(deltas), 4)
        audit["max_delta"] = round(max(deltas), 4)

    # Apply
    if not dry_run and updates:
        for upd in updates:
            conn.execute(
                """UPDATE leaderboard
                   SET old_composite_score = ?,
                       composite_score = ?,
                       wikitext_score = ?,
                       tinystories_score = ?,
                       scoring_version = ?,
                       screening_metric_version = ?,
                       rescore_status = ?,
                       rescore_reason = ?,
                       rescore_timestamp = ?,
                       perplexity_tokenizer_penalty = ?,
                       tags = ?
                   WHERE entry_id = ?""",
                (
                    upd["old_composite_score"],
                    upd["composite_score"],
                    upd["wikitext_score"],
                    upd["tinystories_score"],
                    upd["scoring_version"],
                    upd["screening_metric_version"],
                    upd["rescore_status"],
                    upd["rescore_reason"],
                    upd["rescore_timestamp"],
                    upd["perplexity_tokenizer_penalty"],
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


def _print_report(audit: dict[str, Any]) -> None:
    """Print human-readable summary."""
    s = audit.get("stats", {})
    print(f"\n{'=' * 70}")
    print(f"RESCORE SUMMARY — {audit['version']}")
    print(f"{'=' * 70}")
    print(f"Total rows:          {audit['total_rows']}")
    print(f"Entries rescored:    {s.get('rescored', 0)}")
    print(f"  Score went UP:     {s.get('up', 0)}")
    print(f"  Score went DOWN:   {s.get('down', 0)}")
    print(f"  Unchanged:         {s.get('unchanged', 0)}")
    print(f"  Nulled (pending):  {s.get('nulled', 0)}")
    print("\nReason breakdown:")
    print(f"  CKA broken:              {s.get('cka_broken', 0)}")
    print(f"  Scoring version mismatch:{s.get('version_mismatch', 0)}")
    print(f"  Byte-era perplexity:     {s.get('byte_era', 0)}")
    print(f"  Random tokens:           {s.get('random_tokens', 0)}")

    if "avg_delta" in audit:
        print("\nScore deltas:")
        print(f"  Average: {audit['avg_delta']:+.2f}")
        print(f"  Min:     {audit['min_delta']:+.2f}")
        print(f"  Max:     {audit['max_delta']:+.2f}")

    print("\n--- OLD Top 10 ---")
    for i, r in enumerate(audit.get("before_top10", [])):
        ref = " [REF]" if r.get("is_ref") else ""
        cs = r.get("composite")
        print(f"  {i + 1:2d}. {cs:8.2f}  {r['tier']:14s}  {r['arch']}{ref}")

    print("\n--- NEW Top 10 ---")
    for i, r in enumerate(audit.get("after_top10", [])):
        ref = " [REF]" if r.get("is_ref") else ""
        cs = r.get("composite")
        cs_str = f"{cs:8.2f}" if cs is not None else "    NULL"
        print(f"  {i + 1:2d}. {cs_str}  {r['tier']:14s}  {r['arch']}{ref}")

    demotion = s.get("demotion_risk", [])
    if demotion:
        print(f"\n{'!' * 70}")
        print(f"DEMOTION RISK — {len(demotion)} validated entries:")
        for d in demotion:
            print(
                f"  {d['entry_id']}: {d['old']:.2f} → {d['new']:.2f} ({d['delta']:+.2f})  {d['arch']}"
            )
        print(f"{'!' * 70}")
    else:
        print("\nNo DEMOTION_RISK entries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore leaderboard v3")
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to database",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    parser.add_argument("--entry-id", default=None, help="Rescore single entry")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("Must specify --dry-run or --apply")

    audit = _rescore(args.db, dry_run=args.dry_run, entry_id_filter=args.entry_id)
    _print_report(audit)

    # Write audit artifact
    artifact_dir = Path("research/rescore_artifacts")
    artifact_dir.mkdir(exist_ok=True, parents=True)
    artifact_path = (
        artifact_dir / f"rescore_audit_{SCORING_VERSION}_{int(time.time())}.json"
    )
    artifact_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nAudit artifact: {artifact_path}")


if __name__ == "__main__":
    main()
