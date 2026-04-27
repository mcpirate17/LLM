"""Force-rescore every leaderboard row under v10 with the fresh module.

Use this when an in-process rescore (POST /api/leaderboard/rescore) would
hit the cached module in a long-running dashboard server. This script
imports leaderboard_scoring fresh, so the v9 trajectory plumbing fix in
``_pr_dict_to_score_kwargs`` / ``_PR_SELECT_COLS`` is guaranteed in
scope.

It does NOT honour ``only_stale`` because the post-bug scoring_version
column is already 'v10' on every row — only_stale would skip them all.
Instead it walks every leaderboard row, reads program_results via the
patched select, and writes back ``composite_score`` plus
``old_composite_score`` for diff visibility.

Usage::

    python -m research.tools.rescore_leaderboard_v10
    python -m research.tools.rescore_leaderboard_v10 --version v10 --commit-every 200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab notebook (default: research/lab_notebook.db).",
    )
    parser.add_argument(
        "--version",
        default="v10",
        help="Scoring version to rescore under (default: v10).",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=200,
        help="Commit interval in rows (default: 200).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on rows processed (default: no cap).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from research.scientist.leaderboard_scoring import (
        build_score_kwargs_from_prefetch,
        compute_composite,
        prefetch_program_results,
        set_scoring_version,
    )
    from research.scientist.notebook import LabNotebook

    set_scoring_version(args.version)

    # LabNotebook with use_native=True acquires the writer flock via
    # aria-db for the lifetime of the connection — no need to wrap
    # this loop in acquire_writer_lock (and doing so would deadlock
    # against ourselves since the lock is already held).
    nb = LabNotebook(args.db)
    try:
        cols = nb._get_leaderboard_columns()
        # SELECT * because _pr_dict_to_score_kwargs reads several
        # leaderboard fields by name (tier, wikitext_perplexity,
        # screening_loss_ratio, ...). A narrow SELECT silently
        # passes None for those fields and the scorer happily
        # produces a wrong-but-positive number — that is exactly
        # how the previous rescore-with-narrow-SELECT shaved
        # ~150pts off frontier rows.
        sql = "SELECT * FROM leaderboard ORDER BY composite_score DESC"
        if args.limit:
            sql += f" LIMIT {int(args.limit)}"
        rows = nb.conn.execute(sql).fetchall()
        logger.info("rescoring %d leaderboard rows", len(rows))

        ids = [str(row["result_id"]) for row in rows if row["result_id"]]
        pr_cache = prefetch_program_results(nb.conn, ids)

        # Per-fingerprint metric aggregation — injects cross-run means
        # into the score kwargs and computes per-tier CVs that drive the
        # score-stability penalty inside compute_composite_v10.  Without
        # this, the rescore would re-score on the single row in the
        # leaderboard (cherry-picked best-of-runs) and miss the variance
        # signal entirely.
        metric_to_kwarg = {
            "wikitext_perplexity": ("ppl_screening", "ppl_investigation", "ppl_validation"),
            "blimp_overall_accuracy": ("blimp_accuracy",),
            "hellaswag_acc": (
                "hellaswag_acc_screening",
                "hellaswag_acc_investigation",
                "hellaswag_acc_validation",
            ),
            "tinystories_score": ("tinystories_score",),
            "cross_task_score": ("cross_task_score",),
            "diagnostic_score": ("diagnostic_score",),
            "fp_hierarchy_fitness": ("hierarchy_fitness",),
            "ar_auc": ("ar_auc",),
            "induction_auc": ("induction_auc",),
            "binding_auc": ("binding_auc",),
            "induction_v2_investigation_auc": ("induction_v2_inv_auc",),
            "binding_v2_investigation_auc": ("binding_v2_inv_auc",),
        }

        # Update widening — these new cols may not exist on older DBs.
        agg_cols = {
            "n_runs", "cv_loss", "cv_understanding", "cv_capability",
            "score_stability_penalty",
        } & cols

        changed = 0
        unchanged = 0
        t0 = time.time()
        for idx, row in enumerate(rows, start=1):
            rid = str(row["result_id"]) if row["result_id"] else ""
            if not rid:
                continue
            lb_row = dict(row)
            pr = dict(pr_cache.get(rid, {}))
            kw = build_score_kwargs_from_prefetch(
                pr, lb_row, bool(row["is_reference"])
            )
            # Cross-run aggregation for this fingerprint.
            fp = lb_row.get("graph_fingerprint") or pr.get("graph_fingerprint")
            metric_agg = nb.get_fingerprint_metric_aggregates(fp) if fp else {}
            tier_cv = (metric_agg.get("_tier_cv") or {}) if metric_agg else {}
            n_runs_max = int((metric_agg.get("_n_runs_max") if metric_agg else 0) or 0)
            if metric_agg:
                for mcol, kwarg_names in metric_to_kwarg.items():
                    stat = metric_agg.get(mcol) or {}
                    if int(stat.get("n") or 0) >= 2 and stat.get("mean") is not None:
                        for kn in kwarg_names:
                            if kw.get(kn) is not None:
                                kw[kn] = stat["mean"]
                kw["cv_loss"] = tier_cv.get("loss")
                kw["cv_understanding"] = tier_cv.get("und")
                kw["cv_capability"] = tier_cv.get("cap")
                kw["n_runs"] = n_runs_max
            scored = compute_composite(decompose=True, **kw)
            if isinstance(scored, dict):
                new_score = float(scored.get("composite_score") or 0.0)
                bd = scored.get("breakdown") or {}
                if bd.get("_cv_penalty_applied"):
                    pl = float(bd.get("_cv_penalty_loss") or 1.0)
                    pu = float(bd.get("_cv_penalty_und") or 1.0)
                    pc = float(bd.get("_cv_penalty_cap") or 1.0)
                    stability = (pl * pu * pc) ** (1.0 / 3.0)
                else:
                    stability = 1.0
            else:
                new_score = float(scored or 0.0)
                stability = 1.0
            old_score = float(row["composite_score"] or 0.0)
            if abs(new_score - old_score) < 1e-9:
                unchanged += 1
                continue
            sets = ["composite_score = ?"]
            params: list = [new_score]
            if "scoring_version" in cols:
                sets.append("scoring_version = ?")
                params.append(args.version)
            if "rescore_status" in cols:
                sets.append("rescore_status = 'rescored'")
            if "rescore_timestamp" in cols:
                sets.append("rescore_timestamp = ?")
                params.append(time.time())
            if "old_composite_score" in cols:
                sets.append("old_composite_score = ?")
                params.append(old_score)
            if "rescore_reason" in cols:
                sets.append("rescore_reason = ?")
                params.append("rescore_v10_v9_plumbing_fix")
            # Persist aggregation columns when the schema supports them.
            if "n_runs" in agg_cols:
                sets.append("n_runs = ?")
                params.append(n_runs_max if n_runs_max else None)
            if "cv_loss" in agg_cols:
                sets.append("cv_loss = ?")
                params.append(tier_cv.get("loss"))
            if "cv_understanding" in agg_cols:
                sets.append("cv_understanding = ?")
                params.append(tier_cv.get("und"))
            if "cv_capability" in agg_cols:
                sets.append("cv_capability = ?")
                params.append(tier_cv.get("cap"))
            if "score_stability_penalty" in agg_cols:
                sets.append("score_stability_penalty = ?")
                params.append(stability)
            params.append(row["entry_id"])
            nb.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )
            changed += 1
            if changed % args.commit_every == 0:
                nb._maybe_commit()
                elapsed = time.time() - t0
                logger.info(
                    "  [%d/%d] changed=%d unchanged=%d rate=%.0f rows/s",
                    idx,
                    len(rows),
                    changed,
                    unchanged,
                    idx / max(elapsed, 1e-6),
                )
        nb._maybe_commit()
        logger.info(
            "[done] total=%d changed=%d unchanged=%d elapsed=%.1fs",
            len(rows),
            changed,
            unchanged,
            time.time() - t0,
        )
    finally:
        nb.close()


if __name__ == "__main__":
    main()
