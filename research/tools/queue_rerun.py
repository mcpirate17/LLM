"""Queue validation/breakthrough reruns under the score-stability rule.

Two modes:
  manual:  --fingerprint FP [--n N]
           Queue N follow-up tasks for the given fingerprint regardless of
           rank.  Use when you want a confirmation rerun on something
           specific (a candidate you just promoted, a regression suspect).

  auto:    --auto [--top-n 15]
           For every leaderboard fingerprint with n_runs < cap, compute
           an upper-bound CI on its composite score.  Queue a rerun if
           the upper bound reaches the score at rank ``top_n`` (i.e. it
           is "in striking distance" of that boundary).

Sigma for the CI:
  - n >= 2:   sigma = std(composite-equivalent) / sqrt(n)
              We approximate composite std by the score-stability
              penalty's effect on the loss tier (the dominant variance
              source) — specifically: sigma = composite * cv_loss / sqrt(n_runs).
              When cv_loss is null but n_runs >= 2 we fall back to
              prior_CV * composite.
  - n == 1:   sigma = prior_CV * composite, where prior_CV is the cohort
              median CV(composite-proxy) over multi-run fingerprints,
              with a hard fallback of 0.10 when cohort data is sparse.

The script writes a dry-run report by default; use ``--apply`` to push
the tasks onto the followup_tasks queue for the runner to pick up.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional

from research.defaults import RUNS_DB

DB_PATH_DEFAULT = RUNS_DB
REPORT_DIR = Path("research/reports")
HARD_PRIOR_CV_FALLBACK = 0.10
PRIOR_CV_MIN_COHORT_N = 20  # need at least this many multi-run fps to
# trust the cohort median; otherwise the multi-run cohort is biased
# (architectures get re-run for a reason — usually high variance —
# so a small cohort over-estimates "typical" CV).  Below the threshold
# we use HARD_PRIOR_CV_FALLBACK.
N_RUNS_CAP_DEFAULT = 4  # max total runs (initial + reruns) before stopping
Z_95 = 1.96


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _cohort_prior_cv(conn: sqlite3.Connection) -> tuple[float, dict]:
    """Median composite-equivalent CV across multi-run val/breakthrough fps.

    The single-run-prior CV needs to estimate the spread of *composite
    score* across hypothetical reruns, so we compute a per-row
    composite-equivalent CV (weighted-tier-CV) and take the cohort
    median.  Falls back to ``HARD_PRIOR_CV_FALLBACK`` when fewer than
    3 multi-run fingerprints exist.
    """
    rows = conn.execute(
        """SELECT cv_loss, cv_understanding, cv_capability FROM leaderboard
           WHERE tier IN ('validation','breakthrough')
             AND COALESCE(n_runs, 0) >= 2"""
    ).fetchall()
    cvs: list[float] = []
    for r in rows:
        cv = _composite_cv(dict(r))
        if cv is not None:
            cvs.append(cv)
    if len(cvs) >= PRIOR_CV_MIN_COHORT_N:
        return median(cvs), {"source": "cohort_median", "n": len(cvs)}
    return HARD_PRIOR_CV_FALLBACK, {
        "source": "fallback",
        "n_observed": len(cvs),
        "fallback": HARD_PRIOR_CV_FALLBACK,
        "reason": (
            f"multi-run cohort too small (need >={PRIOR_CV_MIN_COHORT_N} fps, "
            f"have {len(cvs)}); cohort is biased toward high-variance fps."
        ),
    }


def _composite_at_rank(conn: sqlite3.Connection, rank: int) -> Optional[float]:
    """Composite score at the given rank within validation/breakthrough.

    rank is 1-based (rank=1 = highest score).
    """
    rows = conn.execute(
        """SELECT composite_score FROM leaderboard
           WHERE tier IN ('validation','breakthrough')
             AND COALESCE(is_reference, 0) = 0
             AND composite_score IS NOT NULL
           ORDER BY composite_score DESC LIMIT ?""",
        (max(1, int(rank)),),
    ).fetchall()
    if len(rows) < rank:
        return None
    return float(rows[rank - 1]["composite_score"])


def _candidates(conn: sqlite3.Connection, n_runs_cap: int) -> list[dict]:
    rows = conn.execute(
        """SELECT entry_id, result_id, graph_fingerprint, tier,
                  composite_score, COALESCE(n_runs, 1) AS n_runs,
                  cv_loss, cv_understanding, cv_capability,
                  score_stability_penalty
           FROM leaderboard
           WHERE tier IN ('validation','breakthrough')
             AND COALESCE(is_reference, 0) = 0
             AND graph_fingerprint IS NOT NULL
             AND composite_score IS NOT NULL
             AND COALESCE(n_runs, 1) < ?""",
        (int(n_runs_cap),),
    ).fetchall()
    return [dict(r) for r in rows]


_TIER_WEIGHTS = {
    # Approximate share of total max points each tier contributes to
    # the composite.  Loss=225 / Understanding=175 / Capability=175 /
    # Aux+legacy=~210 (no CV penalty).  Used to convert per-tier CVs
    # into a composite-equivalent CV without re-scoring per run.
    "loss": 225.0 / 785.0,
    "und": 175.0 / 785.0,
    "cap": 175.0 / 785.0,
}


def _composite_cv(row: dict) -> Optional[float]:
    """Approximate CV(composite) from per-tier CVs.

    We treat tier CVs as independent contributors and weight by tier
    point share:  CV_composite ≈ Σ w_t · CV_t.  This is a coarse
    estimate but much tighter than treating CV(loss) as if it were the
    full composite CV (which over-estimates by 3-4×).  Returns None if
    no tier CV is populated.
    """
    parts: list[float] = []
    weight_sum = 0.0
    for key, weight in _TIER_WEIGHTS.items():
        cv = row.get(f"cv_{key}") if key != "loss" else row.get("cv_loss")
        # Map "und"/"cap" -> column names cv_understanding/cv_capability.
        if key == "und":
            cv = row.get("cv_understanding")
        elif key == "cap":
            cv = row.get("cv_capability")
        if cv is None:
            continue
        parts.append(float(cv) * weight)
        weight_sum += weight
    if not parts or weight_sum == 0.0:
        return None
    return sum(parts) / weight_sum


def _sigma(row: dict, prior_cv: float) -> tuple[float, str]:
    """Sigma on the composite score for the upper-bound CI.

    Returns (sigma, source-tag).  Sigma = composite × CV / sqrt(n).
    For multi-run fingerprints we use the observed composite-equivalent
    CV.  For n=1 we fall back to ``prior_cv`` (cohort median or hard
    fallback).
    """
    composite = float(row["composite_score"] or 0.0)
    n = max(1, int(row["n_runs"] or 1))
    cv_composite = _composite_cv(row) if n >= 2 else None
    if cv_composite is not None:
        return (
            composite * cv_composite / math.sqrt(n),
            f"observed_cv_composite={cv_composite:.3f}",
        )
    return composite * prior_cv, f"prior_cv={prior_cv:.3f}"


def evaluate(
    db_path: str,
    *,
    top_n: int,
    n_runs_cap: int,
    explicit_fingerprints: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Score every eligible fingerprint against the striking-distance rule."""
    conn = _connect(db_path)
    try:
        boundary = _composite_at_rank(conn, top_n)
        prior_cv, prior_dbg = _cohort_prior_cv(conn)

        if explicit_fingerprints:
            fps = [fp.strip() for fp in explicit_fingerprints if fp and fp.strip()]
            placeholders = ",".join("?" * len(fps))
            rows = conn.execute(
                f"""SELECT entry_id, result_id, graph_fingerprint, tier,
                           composite_score, COALESCE(n_runs, 1) AS n_runs,
                           cv_loss, cv_understanding, cv_capability,
                           score_stability_penalty
                    FROM leaderboard
                    WHERE graph_fingerprint IN ({placeholders})""",
                fps,
            ).fetchall()
            candidates = [dict(r) for r in rows]
            mode = "manual"
        else:
            candidates = _candidates(conn, n_runs_cap)
            mode = "auto"

        eligible: list[dict] = []
        skipped: list[dict] = []
        for row in candidates:
            sigma, src = _sigma(row, prior_cv)
            composite = float(row["composite_score"] or 0.0)
            upper = composite + Z_95 * sigma
            entry: dict = {
                "graph_fingerprint": row["graph_fingerprint"],
                "result_id": row["result_id"],
                "tier": row["tier"],
                "composite": round(composite, 2),
                "n_runs": int(row["n_runs"]),
                "cv_loss": row.get("cv_loss"),
                "sigma_source": src,
                "sigma": round(sigma, 2),
                "upper_bound_95": round(upper, 2),
                "boundary_top_n": (
                    round(boundary, 2) if boundary is not None else None
                ),
            }
            if mode == "manual":
                eligible.append(entry)
                continue
            if boundary is None:
                skipped.append({**entry, "reason": "no_boundary"})
                continue
            if upper >= boundary:
                eligible.append(entry)
            else:
                skipped.append({**entry, "reason": "below_boundary"})

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db_path": db_path,
            "mode": mode,
            "top_n": int(top_n),
            "n_runs_cap": int(n_runs_cap),
            "boundary_top_n": (round(boundary, 2) if boundary is not None else None),
            "prior_cv": round(prior_cv, 4),
            "prior_cv_source": prior_dbg,
            "eligible": sorted(eligible, key=lambda e: -e["upper_bound_95"]),
            "skipped": sorted(skipped, key=lambda e: -e["upper_bound_95"]),
        }
    finally:
        conn.close()


def queue(
    db_path: str,
    *,
    eligible: list[dict],
    n_per_fp: int,
    apply: bool,
    investigation_steps: Optional[int] = None,
    device: str = "cuda",
) -> list[dict]:
    """Push followup_tasks for each eligible fingerprint (or dry-run)."""
    if not eligible:
        return []

    if not apply:
        return [
            {
                "graph_fingerprint": e["graph_fingerprint"],
                "result_id": e["result_id"],
                "n_per_fp": int(n_per_fp),
                "task_id": None,
                "dry_run": True,
            }
            for e in eligible
        ]

    from research.scientist.notebook import LabNotebook
    from research.scientist.runner import RunConfig

    config = RunConfig()
    config.device = device
    config.gbm_prescreener_enabled = False
    config.allow_unproven_ml_influence = False
    # Score-stability reruns: one independent draw per rerun.  Default
    # to single-seed at investigation-tier budget; multi-seed averaging
    # within one rerun reduces intra-sample variance but doesn't add a
    # cross-rerun sample, which is what CV needs.
    config.validation_n_seeds = 1
    config.validation_steps = 2500
    if investigation_steps is not None:
        config.investigation_steps = max(1, int(investigation_steps))

    nb = LabNotebook(db_path)
    out: list[dict] = []
    n_each = max(1, int(n_per_fp))
    try:
        for e in eligible:
            task_ids: list[str] = []
            # One queued row per requested rerun.  The runner claims them
            # one at a time via claim_followup_task("validation"), so N
            # separate tasks = N sequential validation passes.
            for i in range(n_each):
                task_id = nb.enqueue_followup_task(
                    stage="validation",
                    result_ids=[str(e["result_id"])],
                    hypothesis=(
                        "Score-stability rerun: candidate is within striking "
                        "distance of the top-N boundary; queue a confirmation "
                        "run to tighten its CI before crowning."
                    ),
                    config=config.to_dict(),
                    evidence_pack={
                        "score_stability_eval": e,
                        "rerun_index": i + 1,
                        "rerun_total": n_each,
                    },
                    source_context="queue_rerun",
                    priority_score=float(e.get("upper_bound_95") or 0.0),
                    priority_reasons={
                        "policy": "score_stability_ci",
                        "fingerprint": e["graph_fingerprint"],
                        "boundary_top_n": e.get("boundary_top_n"),
                        "upper_bound_95": e.get("upper_bound_95"),
                        "sigma_source": e.get("sigma_source"),
                    },
                    metadata={
                        "source_tool": "queue_rerun",
                        "rerun_index": i + 1,
                        "rerun_total": n_each,
                    },
                    bypass_dedup=True,
                )
                if task_id:
                    task_ids.append(task_id)
            out.append(
                {
                    "graph_fingerprint": e["graph_fingerprint"],
                    "result_id": e["result_id"],
                    "n_per_fp": n_each,
                    "task_ids": task_ids,
                    "task_id": task_ids[0] if task_ids else None,
                    "dry_run": False,
                }
            )
    finally:
        nb.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH_DEFAULT)
    parser.add_argument(
        "--fingerprint",
        action="append",
        default=None,
        help="Manual mode: queue reruns for these fingerprint(s).  May be "
        "repeated.  Skips the striking-distance check.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: scan leaderboard for striking-distance candidates.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Boundary rank for auto mode (default 15).",
    )
    parser.add_argument(
        "--n-runs-cap",
        type=int,
        default=N_RUNS_CAP_DEFAULT,
        help=f"Max total runs before excluding (default {N_RUNS_CAP_DEFAULT}).",
    )
    parser.add_argument(
        "--n", type=int, default=2, help="Reruns per eligible fingerprint (default 2)."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--investigation-steps", type=int, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Push tasks onto followup_tasks; default is dry-run.",
    )
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    if not args.auto and not args.fingerprint:
        parser.error("must pass either --auto or --fingerprint FP (repeatable)")
    if args.auto and args.fingerprint:
        parser.error("--auto and --fingerprint are mutually exclusive")

    eval_report = evaluate(
        args.db,
        top_n=int(args.top_n),
        n_runs_cap=int(args.n_runs_cap),
        explicit_fingerprints=args.fingerprint,
    )
    queued = queue(
        args.db,
        eligible=eval_report["eligible"],
        n_per_fp=int(args.n),
        apply=bool(args.apply),
        investigation_steps=args.investigation_steps,
        device=str(args.device or "cuda"),
    )
    eval_report["queued"] = queued

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = REPORT_DIR / f"queue_rerun_{stamp}"
    json_path = prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(eval_report, indent=2))

    n_elig = len(eval_report["eligible"])
    n_skip = len(eval_report["skipped"])
    print(
        f"{'Queued' if args.apply else 'Dry-run'}: {n_elig} eligible, "
        f"{n_skip} skipped, n_per_fp={args.n}"
    )
    print(f"  boundary @ rank {args.top_n}: {eval_report['boundary_top_n']}")
    print(f"  prior_cv: {eval_report['prior_cv']} ({eval_report['prior_cv_source']})")
    print(f"  report: {json_path}")
    if eval_report["eligible"]:
        print("  top eligible:")
        for e in eval_report["eligible"][:8]:
            print(
                f"    {e['graph_fingerprint']}  "
                f"composite={e['composite']:.1f}  n={e['n_runs']}  "
                f"sigma={e['sigma']:.1f} ({e['sigma_source']})  "
                f"upper95={e['upper_bound_95']:.1f}"
            )


if __name__ == "__main__":
    main()
