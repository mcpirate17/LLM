#!/usr/bin/env python3
"""Rank fingerprints for the next induction_intermediate/binding_intermediate backfill wave.

Read-only — safe to run while the current backfill holds the writer lock.

Priority score per candidate (higher = run sooner):

    priority = w_v1  * v1_signal              # leading indicator (iv1 AUC / max gap)
             + w_tpl * template_capability    # prior P(v2_strong | template)
             + w_gap * high_s1_low_lb_bonus   # template with many s1 but few lb rows
             + w_cov * missing_v2_fraction    # tier-level v2 coverage deficit

v1 induction is a strong leading indicator of v2 induction (confirmed
2026-04-19: iv1>=0.35 → 100% iv2>0.3 on backfilled rows, iv1<0.05 →
<1% iv2>0.3). Reading from `induction_metrics_v2` (12k+ fingerprints,
schema v2 of the metric; predates the investigation-tier v2 probe).

Usage:
    python -m research.tools.probe_backfill_priority --top 3000 \\
        --out research/reports/probe_priority_next.jsonl
    python -m research.tools.probe_backfill_priority --top 500 --tier screening
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "runs.db"

# v1→v2 buckets measured 2026-04-19 on 680 backfilled rows.
# P(iv2>0.3 | iv1_bucket): null=0.15%, weak=0%, medium=0%, high=0%, top=100%.
# Binding: bv1>=0.35 → 100% bv2>0.3.
_V1_STRONG_THRESHOLD = 0.35
_V1_MEDIUM_THRESHOLD = 0.10

# Weights — tuned so that a top-v1 fingerprint beats any pure template-prior
# bonus, and a template with zero leaderboard representation but high s1-pass
# rate gets a meaningful boost.
_W_V1 = 4.0
_W_TPL = 1.5
_W_GAP = 1.0
_W_COV = 0.5


def _load_v1_induction(conn) -> Dict[str, Dict[str, float]]:
    """Map fingerprint → {auc, max_gap_acc} from induction_metrics_v2."""
    out: Dict[str, Dict[str, float]] = {}
    rows = conn.execute(
        "SELECT graph_fingerprint, auc, gap_4, gap_8, gap_16, gap_32, gap_64 "
        "FROM induction_metrics_v2 WHERE graph_fingerprint IS NOT NULL"
    ).fetchall()
    for r in rows:
        fp = str(r["graph_fingerprint"])
        gaps = [r["gap_4"], r["gap_8"], r["gap_16"], r["gap_32"], r["gap_64"]]
        max_gap = max((float(g) for g in gaps if g is not None), default=0.0)
        out[fp] = {
            "iv1_auc": float(r["auc"]) if r["auc"] is not None else 0.0,
            "iv1_max_gap": max_gap,
        }
    return out


def _template_capability_prior(
    conn,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """Per templates_json key: prior P(v2_strong) + s1/lb coverage counts.

    Falls back to v1 strong when v2 is missing (leading-indicator substitution).
    """
    rows = conn.execute(
        """
        SELECT pgf.templates_json AS tpl,
               COUNT(*) AS s1_n,
               SUM(CASE WHEN l.entry_id IS NULL THEN 1 ELSE 0 END) AS backlog,
               SUM(CASE WHEN l.induction_intermediate_auc > 0.3
                          OR l.binding_intermediate_auc > 0.3
                        THEN 1 ELSE 0 END) AS v2_strong,
               SUM(CASE WHEN l.induction_intermediate_auc IS NOT NULL
                        THEN 1 ELSE 0 END) AS v2_observed,
               SUM(CASE WHEN imv2.auc >= ? THEN 1 ELSE 0 END) AS v1_strong
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
        LEFT JOIN induction_metrics_v2 imv2 ON imv2.graph_fingerprint = pr.graph_fingerprint
        WHERE pr.stage1_passed = 1
          AND pgf.templates_json IS NOT NULL AND pgf.templates_json <> '[]'
        GROUP BY pgf.templates_json
        """,
        (_V1_STRONG_THRESHOLD,),
    ).fetchall()

    prior: Dict[str, float] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        tpl = str(r["tpl"])
        s1_n = int(r["s1_n"] or 0)
        # Evidence-weighted prior: (strong_v2 + strong_v1_proxy) / (observed + 1)
        # +1 Laplace smoothing so templates with zero observations get a small prior.
        strong = int(r["v2_strong"] or 0) + 0.5 * int(r["v1_strong"] or 0)
        observed = max(int(r["v2_observed"] or 0), int(r["v1_strong"] or 0))
        prior[tpl] = strong / max(observed + 1, 1)
        counts[tpl] = {
            "s1_n": s1_n,
            "backlog": int(r["backlog"] or 0),
            "v2_observed": int(r["v2_observed"] or 0),
            "v2_strong": int(r["v2_strong"] or 0),
            "v1_strong": int(r["v1_strong"] or 0),
        }
    return prior, counts


def _load_candidates(
    conn,
    tiers: Optional[List[str]],
    include_backlog: bool,
) -> List[Dict[str, Any]]:
    """Pull all fingerprints that lack v2 induction/binding AUCs."""
    clauses = [
        "pr.graph_fingerprint <> ''",
        "pr.graph_fingerprint IS NOT NULL",
        "pr.stage1_passed = 1",
        "(l.induction_intermediate_auc IS NULL "
        "OR l.binding_intermediate_auc IS NULL "
        "OR l.entry_id IS NULL)",
    ]
    params: Tuple[Any, ...] = ()
    if tiers:
        placeholders = ",".join("?" for _ in tiers)
        clauses.append(f"(l.tier IN ({placeholders}) OR l.entry_id IS NULL)")
        params = tuple(tiers)
    if not include_backlog:
        clauses.append("l.entry_id IS NOT NULL")

    q = f"""
        SELECT DISTINCT pr.graph_fingerprint AS fp,
                        pr.result_id AS result_id,
                        COALESCE(l.tier, 'backlog') AS tier,
                        l.entry_id AS entry_id,
                        pgf.templates_json AS tpl,
                        l.composite_score AS score,
                        l.induction_intermediate_auc AS iv2,
                        l.binding_intermediate_auc AS bv2
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
        WHERE {" AND ".join(clauses)}
    """
    rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def _tier_coverage(conn) -> Dict[str, float]:
    """Fraction of rows per tier still missing v2 induction."""
    rows = conn.execute(
        """
        SELECT tier,
               COUNT(*) AS n,
               SUM(CASE WHEN induction_intermediate_auc IS NULL
                        THEN 1 ELSE 0 END) AS missing
        FROM leaderboard
        GROUP BY tier
        """
    ).fetchall()
    out: Dict[str, float] = {}
    for r in rows:
        n = max(int(r["n"] or 0), 1)
        out[str(r["tier"])] = float(int(r["missing"] or 0)) / n
    out["backlog"] = 1.0  # all backlog rows are missing v2 by definition
    return out


def _score(
    candidate: Dict[str, Any],
    v1_data: Dict[str, Dict[str, float]],
    tpl_prior: Dict[str, float],
    tpl_counts: Dict[str, Dict[str, int]],
    tier_cov: Dict[str, float],
) -> Dict[str, float]:
    fp = str(candidate["fp"])
    tpl = str(candidate["tpl"] or "")
    iv1 = v1_data.get(fp, {}).get("iv1_auc", 0.0)
    gap = v1_data.get(fp, {}).get("iv1_max_gap", 0.0)
    v1_sig = max(iv1, gap * 0.5)  # max-gap is accuracy-like, damp slightly

    tpl_cap = tpl_prior.get(tpl, 0.0)

    gap_bonus = 0.0
    tc = tpl_counts.get(tpl)
    if tc and tc["s1_n"] >= 15:
        # Normalise by s1_n so a template with 200 s1 / 0 lb outranks 20 s1 / 0 lb
        lb_rep = (tc["s1_n"] - tc["backlog"]) / max(tc["s1_n"], 1)
        gap_bonus = (1.0 - lb_rep) * min(tc["s1_n"] / 100.0, 2.0)

    cov_bonus = tier_cov.get(str(candidate["tier"]), 0.0)

    priority = (
        _W_V1 * v1_sig + _W_TPL * tpl_cap + _W_GAP * gap_bonus + _W_COV * cov_bonus
    )
    return {
        "priority": priority,
        "v1_signal": v1_sig,
        "tpl_capability": tpl_cap,
        "gap_bonus": gap_bonus,
        "cov_bonus": cov_bonus,
    }


def rank(
    db_path: Path,
    top: int,
    tiers: Optional[List[str]],
    include_backlog: bool,
) -> List[Dict[str, Any]]:
    conn = connect_readonly(db_path)
    try:
        v1_data = _load_v1_induction(conn)
        tpl_prior, tpl_counts = _template_capability_prior(conn)
        tier_cov = _tier_coverage(conn)
        cands = _load_candidates(conn, tiers, include_backlog)
    finally:
        conn.close()

    # Dedupe by fingerprint — keep the richest annotation
    seen: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        fp = str(c["fp"])
        existing = seen.get(fp)
        if existing is None or (c.get("entry_id") and not existing.get("entry_id")):
            seen[fp] = c

    ranked: List[Dict[str, Any]] = []
    for c in seen.values():
        s = _score(c, v1_data, tpl_prior, tpl_counts, tier_cov)
        ranked.append({**c, **s})
    ranked.sort(key=lambda r: r["priority"], reverse=True)
    return ranked[: max(top, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_DEFAULT_DB), help="lab notebook path")
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument(
        "--tier",
        default="",
        help="comma-separated tiers (screening,investigation,validation)",
    )
    ap.add_argument(
        "--no-backlog",
        action="store_true",
        help="exclude 1.6k unpromoted program_results rows",
    )
    ap.add_argument(
        "--out",
        default="research/reports/probe_priority_next.jsonl",
        help="output jsonl path",
    )
    ap.add_argument(
        "--limit-per-template",
        type=int,
        default=0,
        help="cap candidates per templates_json key (0=unlimited)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tiers = [t.strip() for t in args.tier.split(",") if t.strip()] or None
    out = rank(
        Path(args.db),
        args.top,
        tiers,
        include_backlog=not args.no_backlog,
    )

    if args.limit_per_template > 0:
        capped: List[Dict[str, Any]] = []
        per_tpl: Dict[str, int] = defaultdict(int)
        for row in out:
            tpl = str(row.get("tpl") or "")
            if per_tpl[tpl] >= args.limit_per_template:
                continue
            per_tpl[tpl] += 1
            capped.append(row)
        out = capped

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")

    # Summary rollup
    by_tier: Dict[str, int] = defaultdict(int)
    top_score = out[0]["priority"] if out else 0.0
    for row in out:
        by_tier[str(row["tier"])] += 1
    logger.info(
        "wrote %d candidates → %s (top priority=%.3f)", len(out), out_path, top_score
    )
    for t, n in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        logger.info("  %-20s %d", t, n)


if __name__ == "__main__":
    main()
