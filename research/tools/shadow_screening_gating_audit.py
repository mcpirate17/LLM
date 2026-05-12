#!/usr/bin/env python
"""Read-only shadow audit for deferred screening probes.

The main question is whether expensive post-S1 probes, especially BLiMP, can
be deferred behind cheaper signals without losing candidates that the current
scorer would escalate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.scientist.thresholds import (
    HELLASWAG_RANDOM_CHANCE_GATE,
    UNDERSTANDING_SOFT_BINDING,
    V7_SCREENING_THRESHOLD,
    V8_SCREENING_THRESHOLD,
)


DEFAULT_DB = Path("research/runs.db")
DEFAULT_TIMING = Path("tasks/audit/screening_gating_standard.json")
DEFAULT_JSON_OUT = Path("tasks/audit/shadow_screening_gating.json")
DEFAULT_MD_OUT = Path("tasks/audit/shadow_screening_gating.md")


@dataclass(slots=True)
class TimingEstimate:
    current_ms: float
    no_blimp_ms: float
    blimp_ms: float


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_timing_estimate(path: Path) -> TimingEstimate:
    fallback = TimingEstimate(
        current_ms=11_890.2,
        no_blimp_ms=5_820.8,
        blimp_ms=6_069.4,
    )
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    variants = data.get("variants")
    if not isinstance(variants, list):
        variants = (data.get("gating") or {}).get("variants")
    if not isinstance(variants, list):
        return fallback
    current = next(
        (row for row in variants if row.get("variant") == "current"),
        None,
    )
    if not isinstance(current, dict):
        return fallback
    current_ms = _safe_float(current.get("elapsed_ms"), fallback.current_ms)
    probes = current.get("probe_timings_ms") or {}
    blimp_ms = _safe_float(probes.get("blimp_elapsed_ms"), fallback.blimp_ms)
    if current_ms <= 0.0 or blimp_ms <= 0.0:
        return fallback
    return TimingEstimate(
        current_ms=current_ms,
        no_blimp_ms=max(0.0, current_ms - blimp_ms),
        blimp_ms=blimp_ms,
    )


def _load_rows(conn: sqlite3.Connection, *, limit: int) -> list[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT l.*,
               pr.rowid AS pr_rowid,
               pr.stage1_passed AS pr_stage1_passed
          FROM leaderboard l
          JOIN program_results pr ON pr.result_id = l.result_id
         WHERE pr.stage1_passed = 1
           AND pr.blimp_overall_accuracy IS NOT NULL
           AND pr.hellaswag_acc IS NOT NULL
           AND pr.binding_screening_composite IS NOT NULL
         ORDER BY pr.rowid DESC
         LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def _score_rows(
    conn: sqlite3.Connection,
    rows: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    prefetched = prefetch_program_results(conn, [str(row["result_id"]) for row in rows])
    scored: list[Dict[str, Any]] = []
    for row in rows:
        result_id = str(row["result_id"])
        pr = prefetched.get(result_id)
        if not pr:
            continue
        is_reference = bool(row.get("is_reference"))
        kw = build_score_kwargs_from_prefetch(pr, row, is_reference)
        current = compute_composite(decompose=True, **kw)
        no_blimp_kw = dict(kw)
        no_blimp_kw["blimp_accuracy"] = None
        no_blimp = compute_composite(decompose=True, **no_blimp_kw)
        current_score = _safe_float(current.get("composite_score"))
        no_blimp_score = _safe_float(no_blimp.get("composite_score"))
        scored.append(
            {
                "result_id": result_id,
                "tier": row.get("tier"),
                "current_score": current_score,
                "score_without_blimp": no_blimp_score,
                "blimp_delta": current_score - no_blimp_score,
                "stored_composite_score": _safe_float(row.get("composite_score")),
                "hellaswag_acc": _safe_float(kw.get("hellaswag_acc_screening")),
                "binding_screening_composite": _safe_float(
                    row.get("binding_screening_composite")
                ),
                "blimp_accuracy": _safe_float(kw.get("blimp_accuracy")),
                "wikitext_score": _safe_float(kw.get("ppl_screening")),
                "cheap_soft_signal": bool(
                    _safe_float(kw.get("hellaswag_acc_screening"))
                    > HELLASWAG_RANDOM_CHANCE_GATE
                    or _safe_float(row.get("binding_screening_composite"))
                    >= UNDERSTANDING_SOFT_BINDING
                ),
            }
        )
    return scored


def _policy_run_blimp(
    row: Dict[str, Any],
    *,
    threshold: float,
    policy: str,
) -> bool:
    cheap_score = _safe_float(row.get("score_without_blimp"))
    has_soft_signal = bool(row.get("cheap_soft_signal"))
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "soft_signal_only":
        return has_soft_signal
    if policy == "soft_signal_or_within_10":
        return has_soft_signal or cheap_score >= threshold - 10.0
    if policy == "soft_signal_or_within_25":
        return has_soft_signal or cheap_score >= threshold - 25.0
    if policy == "within_25":
        return cheap_score >= threshold - 25.0
    if policy == "within_50":
        return cheap_score >= threshold - 50.0
    raise ValueError(f"unknown policy: {policy}")


def _summarize_policy(
    rows: list[Dict[str, Any]],
    *,
    threshold: float,
    policy: str,
    timing: TimingEstimate,
) -> Dict[str, Any]:
    current_passes = [
        row for row in rows if _safe_float(row.get("current_score")) >= threshold
    ]
    run_rows = [
        row
        for row in rows
        if _policy_run_blimp(row, threshold=threshold, policy=policy)
    ]
    run_ids = {row["result_id"] for row in run_rows}
    shadow_passes = []
    missed = []
    for row in current_passes:
        shadow_score = (
            _safe_float(row.get("current_score"))
            if row["result_id"] in run_ids
            else _safe_float(row.get("score_without_blimp"))
        )
        if shadow_score >= threshold:
            shadow_passes.append(row)
        else:
            missed.append(row)
    total_ms = len(rows) * timing.no_blimp_ms + len(run_rows) * timing.blimp_ms
    current_ms = len(rows) * timing.current_ms
    saved_ms = max(0.0, current_ms - total_ms)
    missed_sorted = sorted(
        missed,
        key=lambda row: _safe_float(row.get("current_score")),
        reverse=True,
    )
    return {
        "policy": policy,
        "threshold": threshold,
        "n_rows": len(rows),
        "run_blimp_count": len(run_rows),
        "skip_blimp_count": len(rows) - len(run_rows),
        "run_blimp_pct": round(100.0 * len(run_rows) / max(len(rows), 1), 2),
        "current_pass_count": len(current_passes),
        "shadow_pass_count": len(shadow_passes),
        "missed_current_pass_count": len(missed),
        "missed_current_pass_pct": round(
            100.0 * len(missed) / max(len(current_passes), 1),
            2,
        ),
        "estimated_total_ms": round(total_ms, 1),
        "estimated_saved_ms": round(saved_ms, 1),
        "estimated_saved_pct": round(100.0 * saved_ms / max(current_ms, 1.0), 2),
        "max_missed_current_score": round(
            max((_safe_float(row.get("current_score")) for row in missed), default=0.0),
            4,
        ),
        "missed_examples": [
            {
                "result_id": row["result_id"],
                "current_score": round(_safe_float(row.get("current_score")), 4),
                "score_without_blimp": round(
                    _safe_float(row.get("score_without_blimp")), 4
                ),
                "blimp_delta": round(_safe_float(row.get("blimp_delta")), 4),
                "hellaswag_acc": round(_safe_float(row.get("hellaswag_acc")), 4),
                "binding_screening_composite": round(
                    _safe_float(row.get("binding_screening_composite")), 4
                ),
                "blimp_accuracy": round(_safe_float(row.get("blimp_accuracy")), 4),
            }
            for row in missed_sorted[:20]
        ],
    }


def _distribution(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = sorted(_safe_float(row.get(key)) for row in rows if row.get(key) is not None)
    if not vals:
        return {"n": 0}

    def pct(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return round(vals[idx], 4)

    return {
        "n": len(vals),
        "min": round(vals[0], 4),
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "max": round(vals[-1], 4),
        "mean": round(sum(vals) / len(vals), 4),
    }


def build_audit(
    *,
    db_path: Path,
    timing_path: Path,
    limit: int,
) -> Dict[str, Any]:
    timing = _load_timing_estimate(timing_path)
    with _connect(db_path) as conn:
        rows = _load_rows(conn, limit=limit)
        scored = _score_rows(conn, rows)
    policies = (
        "always",
        "never",
        "soft_signal_only",
        "soft_signal_or_within_10",
        "soft_signal_or_within_25",
        "within_25",
        "within_50",
    )
    thresholds = {
        "v7_screening": float(V7_SCREENING_THRESHOLD),
        "v8_screening": float(V8_SCREENING_THRESHOLD),
    }
    return {
        "source": {
            "db_path": str(db_path),
            "timing_path": str(timing_path),
            "limit": int(limit),
            "n_rows": len(scored),
        },
        "timing_estimate_ms": {
            "current": round(timing.current_ms, 4),
            "no_blimp": round(timing.no_blimp_ms, 4),
            "blimp": round(timing.blimp_ms, 4),
        },
        "thresholds": thresholds,
        "metric_distributions": {
            "current_score": _distribution(scored, "current_score"),
            "score_without_blimp": _distribution(scored, "score_without_blimp"),
            "blimp_delta": _distribution(scored, "blimp_delta"),
            "hellaswag_acc": _distribution(scored, "hellaswag_acc"),
            "binding_screening_composite": _distribution(
                scored, "binding_screening_composite"
            ),
            "blimp_accuracy": _distribution(scored, "blimp_accuracy"),
        },
        "policy_results": {
            threshold_name: [
                _summarize_policy(
                    scored,
                    threshold=threshold,
                    policy=policy,
                    timing=timing,
                )
                for policy in policies
            ]
            for threshold_name, threshold in thresholds.items()
        },
    }


def _markdown_table(rows: list[Dict[str, Any]]) -> str:
    lines = [
        "| Policy | Run BLiMP | Save | Current Passes Missed | Max Missed Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {run_blimp_count}/{n_rows} ({run_blimp_pct:.2f}%) "
            "| {estimated_saved_pct:.2f}% | {missed_current_pass_count}/{current_pass_count} "
            "({missed_current_pass_pct:.2f}%) | {max_missed_current_score:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def render_markdown(audit: Dict[str, Any]) -> str:
    timing = audit["timing_estimate_ms"]
    lines = [
        "# Shadow Screening Gating Audit",
        "",
        "Read-only audit over persisted screening rows. Estimates BLiMP deferral "
        "using the controlled scheduling timing artifact.",
        "",
        "## Inputs",
        "",
        f"- Rows: {audit['source']['n_rows']}",
        f"- Timing current: {timing['current']:.1f} ms",
        f"- Timing without BLiMP: {timing['no_blimp']:.1f} ms",
        f"- BLiMP marginal estimate: {timing['blimp']:.1f} ms",
        "",
        "## Metric Distributions",
        "",
        "```json",
        json.dumps(audit["metric_distributions"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    for threshold_name, rows in audit["policy_results"].items():
        lines.extend(
            [
                f"## {threshold_name}",
                "",
                _markdown_table(rows),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--timing", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    audit = build_audit(
        db_path=args.db,
        timing_path=args.timing,
        limit=max(1, int(args.limit)),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["policy_results"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
