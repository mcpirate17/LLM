"""NAS decision-report CLI.

Two modes:

* default  — read a candidate JSONL (or published-sanity .json), run the explicit
  gate policy, and print *per-candidate gate decisions* (which gate failed, value,
  threshold, head version, reason) plus a stage/gate summary. This is the thing the
  old single-blended-score rank could not produce.
* --backtest — run the policy over the four named corpora and report, where ground
  truth exists, false-reject / false-accept rates:
    1. the cascade clean shortlist (decision distribution),
    2. the 7 published-family sanity graphs (admit vs real S1 induction),
    3. runs.db known-good *capable* rows (false-reject),
    4. runs.db known-bad rows (false-accept).

Read-only: safe to run while the DB writer lock is held (opens runs.db ``mode=ro``).
There are currently zero >100M-param / >20K-step capable rows in runs.db, so the
"large-run" cohort falls back to capability (induction_screening_auc >= 0.35) at any
scale; this is reported explicitly and is also why the cheap-to-large heads stay blocked.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.tools.nas_gate_policy import (
    DEFAULT_THRESHOLDS,
    GateDecision,
    GatePolicyConfig,
    Stage,
    evaluate_candidates,
    load_thresholds,
    read_rows,
    summarize,
)

SHORTLIST = "research/reports/cpu_cascade_million_shortlist_clean.jsonl"
PUBLISHED = "research/reports/published_nas_sanity_summary.json"
CAPABLE_SQL = (
    "SELECT graph_fingerprint, MAX(induction_screening_auc) "
    "FROM graph_runs WHERE induction_screening_auc >= 0.35 GROUP BY graph_fingerprint"
)
KNOWN_BAD_SQL = (
    "SELECT graph_fingerprint, MAX(induction_screening_auc) FROM graph_runs "
    "WHERE stage1_passed = 0 AND induction_screening_auc IS NOT NULL "
    "AND induction_screening_auc < 0.05 GROUP BY graph_fingerprint"
)


def _resolve_thresholds(use_defaults: bool) -> dict[str, float]:
    return dict(DEFAULT_THRESHOLDS) if use_defaults else load_thresholds()


# --------------------------------------------------------------------------- #
# Per-candidate decision report
# --------------------------------------------------------------------------- #
def render_decisions_markdown(
    decisions: list[GateDecision], summary: dict[str, Any], title: str
) -> str:
    lines = [
        f"# {title}",
        "",
        f"`{json.dumps(summary)}`",
        "",
        "| fingerprint | accepted | stage | first_failed | value | threshold | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in decisions:
        rej = d.rejections[0] if d.rejections else None
        val = "" if rej is None or rej.value is None else f"{rej.value:.4f}"
        thr = "" if rej is None or rej.threshold is None else f"{rej.threshold:.4f}"
        lines.append(
            f"| {d.fingerprint} | {d.accepted} | {d.stage.value} | "
            f"{d.first_failed_gate or ''} | {val} | {thr} | {d.reason} |"
        )
    return "\n".join(lines) + "\n"


def report_decisions(
    rows: list[dict[str, Any]], thresholds: dict[str, float], config: GatePolicyConfig
) -> tuple[list[GateDecision], dict[str, Any]]:
    decisions = evaluate_candidates(rows, thresholds, config)
    return decisions, summarize(decisions)


# --------------------------------------------------------------------------- #
# Backtest cohorts from runs.db
# --------------------------------------------------------------------------- #
def _fetch_db_cohort(
    conn: sqlite3.Connection, sql: str, sample: int
) -> list[tuple[str, float]]:
    rows = conn.execute(sql).fetchall()
    return [(fp, float(v)) for fp, v in rows if fp][:sample]


def _score_cohort(
    cohort: list[tuple[str, float]], conn: sqlite3.Connection, scorer: Any
) -> list[dict[str, Any]]:
    """Score each non-placeholder graph through the oracle into a policy row."""
    out: list[dict[str, Any]] = []
    for fp, capable_val in cohort:
        g = conn.execute(
            "SELECT graph_json, lit_match_type FROM graphs "
            "WHERE graph_fingerprint = ? AND graph_json_is_placeholder = 0 "
            "AND graph_json IS NOT NULL LIMIT 1",
            (fp,),
        ).fetchone()
        if g is None:
            continue  # placeholder/missing graph: oracle-invisible, cannot score
        try:
            scored = scorer.score_graph_dict(json.loads(g[0]))
        except Exception:  # noqa: BLE001 — a single bad graph must not abort the cohort
            continue
        out.append(
            {
                "fingerprint": fp,
                "label_free_probe_predictions": scored.get(
                    "label_free_probe_predictions"
                ),
                "lit_match_type": g[1],
                "_measured_induction": capable_val,
            }
        )
    return out


def _stage_rates(decisions: list[GateDecision]) -> dict[str, Any]:
    """Split admissions into exploit (predictor merit) vs rescue (blind-spot quota)."""
    n = len(decisions)
    exploit = sum(1 for d in decisions if d.stage is Stage.EXPLOIT)
    rescue = sum(1 for d in decisions if d.stage is Stage.RESCUE)
    rejected = n - exploit - rescue
    return {
        "n": n,
        "exploit_n": exploit,
        "rescue_n": rescue,
        "rejected_n": rejected,
        "exploit_rate": round(exploit / n, 4) if n else 0.0,
        "rescue_rate": round(rescue / n, 4) if n else 0.0,
        "rejected_rate": round(rejected / n, 4) if n else 0.0,
    }


def backtest(
    thresholds: dict[str, float], config: GatePolicyConfig, sample: int
) -> dict[str, Any]:
    out: dict[str, Any] = {"thresholds": thresholds, "config": config.model_dump()}

    # 1. cascade clean shortlist — decision distribution.
    sl_rows = read_rows(SHORTLIST)
    out["shortlist"] = summarize(evaluate_candidates(sl_rows, thresholds, config))

    # 2. published-family sanity — admit vs real S1 induction.
    pub_rows = read_rows(PUBLISHED)
    pub_dec = {
        d.fingerprint: d for d in evaluate_candidates(pub_rows, thresholds, config)
    }
    pub_detail = []
    for r in pub_rows:
        fp = str(r.get("fingerprint") or "")
        s1 = (r.get("s1_actual") or {}).get("s1_induction_auc")
        d = pub_dec.get(fp)
        pub_detail.append(
            {
                "fingerprint": fp,
                "published_key": r.get("published_key"),
                "stage": d.stage.value if d else None,
                "admitted": bool(d and d.stage in (Stage.EXPLOIT, Stage.RESCUE)),
                "s1_induction_auc": s1,
                "s1_capable": bool(s1 is not None and s1 >= 0.35),
            }
        )
    capable = [x for x in pub_detail if x["s1_capable"]]
    out["published"] = {
        "n": len(pub_detail),
        "n_capable": len(capable),
        "capable_admitted": sum(1 for x in capable if x["admitted"]),
        "detail": pub_detail,
    }

    # 3 + 4. runs.db cohorts via oracle scoring.
    from research.tools.label_free_probe_oracle import LabelFreeProbeOracleScorer

    scorer = LabelFreeProbeOracleScorer.try_load()
    if scorer is None:
        out["db_cohorts"] = {"skipped": "probe oracle unavailable"}
        return out
    conn = sqlite3.connect(f"file:{RUNS_DB}?mode=ro", uri=True)
    try:
        good = _score_cohort(_fetch_db_cohort(conn, CAPABLE_SQL, sample), conn, scorer)
        bad = _score_cohort(_fetch_db_cohort(conn, KNOWN_BAD_SQL, sample), conn, scorer)
    finally:
        conn.close()
    good_dec = evaluate_candidates(good, thresholds, config)
    bad_dec = evaluate_candidates(bad, thresholds, config)
    good_rates = _stage_rates(good_dec)
    bad_rates = _stage_rates(bad_dec)
    out["known_good_capable"] = {
        # false-reject = capable graph rejected outright; exploit_recall = admitted on predictor merit.
        "false_reject_rate": good_rates["rejected_rate"],
        "exploit_recall": good_rates["exploit_rate"],
        "rescue_rate": good_rates["rescue_rate"],
        "note": "induction>=0.35 at any scale; zero >100M/>20K capable rows exist in runs.db",
        "stage_counts": good_rates,
        "by_first_failed_gate": summarize(good_dec)["by_first_failed_gate"],
    }
    out["known_bad"] = {
        # predictor false-accept = bad graph admitted on exploit merit (rescue admits are intentional).
        "predictor_false_accept_rate": bad_rates["exploit_rate"],
        "rescue_quota_admits": bad_rates["rescue_n"],
        "stage_counts": bad_rates,
        "by_first_failed_gate": summarize(bad_dec)["by_first_failed_gate"],
    }
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", help="candidate JSONL or published-sanity .json")
    ap.add_argument(
        "--backtest", action="store_true", help="run the 4-corpus policy backtest"
    )
    ap.add_argument(
        "--use-defaults",
        action="store_true",
        help="use spec default thresholds, skip oracle",
    )
    ap.add_argument("--rescue-quota", type=int, default=GatePolicyConfig().rescue_quota)
    ap.add_argument(
        "--ar-gate-advisory",
        action="store_true",
        help="demote ar_gate from hard no-go to advisory (for induction/binding-target passes)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=300,
        help="per-cohort sample cap for runs.db backtest",
    )
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    args = ap.parse_args()

    thresholds = _resolve_thresholds(args.use_defaults)
    config = GatePolicyConfig(
        rescue_quota=args.rescue_quota, ar_gate_hard=not args.ar_gate_advisory
    )

    if args.backtest:
        result = backtest(thresholds, config, args.sample)
        payload = json.dumps(result, indent=1, default=str)
        print(payload)
        if args.json_out:
            Path(args.json_out).write_text(payload + "\n")
        if args.markdown_out:
            Path(args.markdown_out).write_text(
                "# NAS gate-policy backtest\n\n```json\n" + payload + "\n```\n"
            )
        return

    if not args.jsonl:
        ap.error("provide --jsonl or --backtest")
    rows = read_rows(args.jsonl)
    decisions, summ = report_decisions(rows, thresholds, config)
    title = f"NAS gate decisions — {Path(args.jsonl).name}"
    md = render_decisions_markdown(decisions, summ, title)
    print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"summary": summ, "decisions": [d.to_dict() for d in decisions]},
                indent=1,
            )
            + "\n"
        )
    if args.markdown_out:
        Path(args.markdown_out).write_text(md)


if __name__ == "__main__":
    main()
