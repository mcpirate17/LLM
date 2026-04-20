#!/usr/bin/env python3
"""Rank backlog candidates by their plausible top-25 composite score.

Read-only. For each unpromoted program_results row with ``stage1_passed=1``,
computes:

- ``score_actual``: composite with measured probe values (NULLs → treated as
  conservative defaults by the scorer).
- ``score_max``: composite with optimistic upper-bound probes in place of
  NULLs (iv2/bv2/ar/hellaswag/blimp at near-top-percentile). Represents
  the best-case the candidate could reach if the full screening/probe
  suite yielded excellent results.

Candidates whose ``score_max`` clears the current top-25 threshold are
flagged for promotion. Output: ``research/reports/promote_backlog_candidates_YYYY-MM-DD.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from research.scientist.leaderboard_scoring import (  # noqa: E402
    _pr_dict_to_score_kwargs,
    compute_composite,
)
from research.tools._db_maintenance import connect_readonly

DEFAULT_DB = Path("research/lab_notebook.db")

# Optimistic upper bounds calibrated against the *actual* top-25 leaderboard.
# Measured 2026-04-19: top-25 probe distributions are much narrower than
# naive percentiles. Fills below are ~top-25 max; scoring higher than this
# would require hitting a genuine outlier like the c9c7075e/10cfec26 cohort.
OPTIMISTIC_FILL: Dict[str, float] = {
    "induction_auc": 0.15,  # top-25 max ~0.44, but >0.15 is rare
    "induction_v2_investigation_auc": 0.15,  # top-25 avg 0.052, max 0.808 (1 outlier)
    "binding_auc": 0.10,  # top-25 avg 0.006
    "binding_v2_investigation_auc": 0.10,  # top-25 max 0.095
    "ar_auc": 0.02,  # top-25 max 0.009
    "hellaswag_acc": 0.35,  # top-25 max 0.34
    "blimp_overall_accuracy": 0.55,  # modest
    "robustness_long_ctx_combined_score": 0.40,  # v7 frontier anchor
    "validation_loss_ratio": 0.35,
    "ncd_description_length_per_param": 1.5e-6,
}


def _top_k_threshold(conn, k: int) -> float:
    rows = conn.execute(
        """
        SELECT composite_score FROM leaderboard
        WHERE composite_score IS NOT NULL AND (is_reference = 0 OR is_reference IS NULL)
        ORDER BY composite_score DESC LIMIT ?
        """,
        (k,),
    ).fetchall()
    if not rows or len(rows) < k:
        return 0.0
    return float(rows[-1]["composite_score"])


def _candidate_rows(conn) -> List[Any]:
    return conn.execute(
        """
        SELECT pr.*
        FROM program_results pr
        LEFT JOIN leaderboard l ON pr.result_id = l.result_id
        WHERE l.entry_id IS NULL
          AND pr.stage1_passed = 1
          AND TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
        """
    ).fetchall()


def _score(pr_row: Any, optimistic: bool) -> float:
    pr_dict: Dict[str, Any] = {k: pr_row[k] for k in pr_row.keys()}
    if optimistic:
        for key, val in OPTIMISTIC_FILL.items():
            if pr_dict.get(key) is None:
                pr_dict[key] = val
    # Empty leaderboard dict — backlog rows have no leaderboard row
    kw = _pr_dict_to_score_kwargs(pr_dict, d={}, is_reference=False)
    try:
        score = compute_composite(**kw)
    except Exception:
        return 0.0
    return float(score) if score is not None and math.isfinite(float(score)) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument(
        "--top-k", type=int, default=25, help="Threshold rank on leaderboard"
    )
    ap.add_argument(
        "--out",
        default="research/reports/promote_backlog_candidates_2026-04-19.jsonl",
    )
    args = ap.parse_args()

    conn = connect_readonly(Path(args.db))
    try:
        threshold = _top_k_threshold(conn, args.top_k)
        rows = _candidate_rows(conn)
    finally:
        conn.close()

    print(f"Top-{args.top_k} leaderboard threshold: composite_score >= {threshold:.2f}")
    print(f"Backlog candidates (stage1_passed, unpromoted): {len(rows)}")

    ranked: List[Dict[str, Any]] = []
    for r in rows:
        s_actual = _score(r, optimistic=False)
        s_max = _score(r, optimistic=True)
        if s_max < threshold:
            continue
        ranked.append(
            {
                "graph_fingerprint": r["graph_fingerprint"],
                "result_id": r["result_id"],
                "model_source": r["model_source"],
                "score_actual": round(s_actual, 2),
                "score_max": round(s_max, 2),
                "gap_to_top25": round(threshold - s_actual, 2),
                "loss_ratio": r["loss_ratio"],
                "wikitext_perplexity": r["wikitext_perplexity"],
                "hellaswag_acc": r["hellaswag_acc"],
                "induction_v2_investigation_auc": r["induction_v2_investigation_auc"],
                "binding_v2_investigation_auc": r["binding_v2_investigation_auc"],
                "induction_auc": r["induction_auc"],
                "binding_auc": r["binding_auc"],
                "ar_auc": r["ar_auc"],
                "blimp_overall_accuracy": r["blimp_overall_accuracy"],
                "timestamp": r["timestamp"],
            }
        )

    ranked.sort(key=lambda c: c["score_max"], reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in ranked:
            fh.write(json.dumps(row) + "\n")

    print(f"\nCandidates clearing top-{args.top_k} threshold: {len(ranked)}")
    print(f"Written to: {out_path}\n")
    print(
        f"{'rank':>4}  {'fp':<16}  {'score_max':>9}  {'score_actual':>12}  {'source':<18}  {'loss':>7}  {'ppl':>8}  {'hs':>5}"
    )
    for i, row in enumerate(ranked[:30], 1):
        fp = row["graph_fingerprint"][:16]
        ppl = row.get("wikitext_perplexity")
        ppl_str = f"{ppl:8.2f}" if ppl is not None else "       —"
        hs = row.get("hellaswag_acc")
        hs_str = f"{hs:5.2f}" if hs is not None else "    —"
        loss = row.get("loss_ratio")
        loss_str = f"{loss:7.3f}" if loss is not None else "      —"
        src = (row.get("model_source") or "-")[:18]
        print(
            f"{i:>4}  {fp:<16}  {row['score_max']:9.2f}  {row['score_actual']:12.2f}  "
            f"{src:<18}  {loss_str}  {ppl_str}  {hs_str}"
        )


if __name__ == "__main__":
    main()
