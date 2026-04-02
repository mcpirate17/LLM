#!/usr/bin/env python3
"""Backfill HellaSwag commonsense reasoning eval for top leaderboard entries.

Evaluates existing models and stores hellaswag_acc in program_results +
leaderboard. After backfill, prints a rescore recommendation.

NOTE: HellaSwag is informational only at nano scale — all architectures
score ~25% (random chance). Gates are disabled. Data is collected for
dashboard display and future analysis at larger scale.

Usage:
    python -m research.tools.backfill_hellaswag [--top N] [--tier validation,investigation] [--dry-run] [--device cuda]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.eval.hellaswag_eval import (
    INVESTIGATION_N_EXAMPLES,
    SCREENING_N_EXAMPLES,
    VALIDATION_N_EXAMPLES,
    evaluate_hellaswag,
)
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.scientist.notebook import LabNotebook
from research.scientist.thresholds import HELLASWAG_RANDOM_CHANCE_GATE
from research.tools._backfill_shared import DB_PATH, reconstruct_model

_TIER_N_EXAMPLES = {
    "validation": VALIDATION_N_EXAMPLES,
    "breakthrough": VALIDATION_N_EXAMPLES,
    "investigation": INVESTIGATION_N_EXAMPLES,
    "screening": SCREENING_N_EXAMPLES,
}


def _ensure_leaderboard_column(nb):
    """Add hellaswag_acc to leaderboard if missing (not auto-migrated)."""
    cols = {
        row[1] for row in nb.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
    }
    if "hellaswag_acc" not in cols:
        nb.conn.execute("ALTER TABLE leaderboard ADD COLUMN hellaswag_acc REAL")
        print("Added hellaswag_acc column to leaderboard table")


def _query_candidates(nb, tiers: list[str], top: int, force: bool):
    """Query and filter entries needing HellaSwag eval."""
    tier_ph = ",".join("?" for _ in tiers)
    rows = nb.conn.execute(
        f"SELECT l.entry_id, l.result_id, l.tier, l.composite_score, "
        f"l.is_reference, pr.graph_json, pr.hellaswag_acc, pr.graph_fingerprint "
        f"FROM leaderboard l "
        f"LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        f"WHERE l.tier IN ({tier_ph}) ORDER BY l.composite_score DESC",
        tuple(tiers),
    ).fetchall()

    if not force:
        rows = [r for r in rows if r["hellaswag_acc"] is None]

    by_tier: dict[str, list] = {}
    for r in rows:
        tier_list = by_tier.setdefault(r["tier"], [])
        if len(tier_list) < top:
            tier_list.append(r)

    result = []
    for t in tiers:
        result.extend(by_tier.get(t, []))
    return result, by_tier


def _store_and_rescore(nb, entry_id, result_id, acc, status, n_total, is_ref, pr_cache):
    """Store HellaSwag result and recompute composite score. Returns (new_score, old_score) or None."""
    nb.conn.execute(
        "UPDATE program_results SET hellaswag_acc=?, hellaswag_status=?, hellaswag_n_examples=? WHERE result_id=?",
        (acc, status, n_total, result_id),
    )
    nb.conn.execute(
        "UPDATE leaderboard SET hellaswag_acc=? WHERE result_id=?", (acc, result_id)
    )

    existing = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE entry_id=?", (entry_id,)
    ).fetchone()
    if not existing:
        return None
    d = dict(existing)
    d["hellaswag_acc"] = acc
    pr_dict = dict(pr_cache.get(result_id, {}))
    pr_dict["hellaswag_acc"] = acc
    score_kw = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
    new_score = compute_composite(**score_kw)
    old_score = float(d.get("composite_score") or 0)
    nb.conn.execute(
        "UPDATE leaderboard SET composite_score=? WHERE entry_id=?",
        (new_score, entry_id),
    )
    return new_score, old_score


def main():
    parser = argparse.ArgumentParser(
        description="Backfill HellaSwag eval for top leaderboard entries"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--tier", default="validation,investigation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tier.split(",")]
    nb = LabNotebook(DB_PATH)
    _ensure_leaderboard_column(nb)
    rows, by_tier = _query_candidates(nb, tiers, args.top, args.force)

    total = len(rows)
    print(f"Entries to backfill: {total}  (device={args.device})")
    for t in tiers:
        n = len(by_tier.get(t, []))
        if n:
            print(f"  {t}: {n}")
    print()

    if total == 0:
        print("Nothing to backfill.")
        return

    if args.dry_run:
        for r in rows:
            fp = (r["graph_fingerprint"] or "")[:12]
            print(
                f"  [{fp}] tier={r['tier']} score={r['composite_score']:.1f} ref={bool(r['is_reference'])}"
            )
        print(f"\nDry run: would evaluate {total} entries.")
        return

    pr_cache = prefetch_program_results(nb.conn, [r["result_id"] for r in rows])
    evaluated, failed, skipped, at_random = 0, 0, 0, 0
    t0 = time.time()

    for i, row in enumerate(rows):
        entry_id, result_id = row["entry_id"], row["result_id"]
        graph_json = row["graph_json"]
        fp = (row["graph_fingerprint"] or "")[:12]
        is_ref = bool(row["is_reference"])

        if is_ref:
            print(f"  [{fp}] skip: reference model")
            skipped += 1
            continue
        if not graph_json or graph_json == "{}":
            skipped += 1
            print(f"  [{fp}] skip: no graph_json")
            continue

        n_examples = _TIER_N_EXAMPLES.get(row["tier"], 100)
        try:
            model = reconstruct_model(graph_json, args.device)
            hs = evaluate_hellaswag(
                model, VOCAB_SIZE, args.device, n_examples=n_examples
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

            acc = hs.get("hellaswag_acc")
            if acc is not None:
                result = _store_and_rescore(
                    nb,
                    entry_id,
                    result_id,
                    acc,
                    hs.get("hellaswag_status", "ok"),
                    hs.get("hellaswag_total"),
                    is_ref,
                    pr_cache,
                )
                evaluated += 1
                if acc <= HELLASWAG_RANDOM_CHANCE_GATE:
                    at_random += 1
                if result:
                    new_score, old_score = result
                    print(
                        f"  [{fp}] acc={acc:.1%} score={old_score:.1f}->{new_score:.1f} ({new_score - old_score:+.1f})"
                    )
            else:
                failed += 1
                print(f"  [{fp}] status={hs.get('hellaswag_status')}")

        except (RuntimeError, KeyError, ValueError) as e:
            failed += 1
            print(f"  [{fp}] error: {e}")
            if args.device == "cuda":
                torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            nb.conn.commit()
            print(f"  ... {i + 1}/{total} ({time.time() - t0:.0f}s)")

    nb.conn.commit()
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("BACKFILL COMPLETE")
    print(f"  Evaluated: {evaluated}, Failed: {failed}, Skipped: {skipped}")
    print(f"  At random chance (<=28%): {at_random}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
