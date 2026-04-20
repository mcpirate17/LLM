#!/usr/bin/env python3
"""Batch-rescreen leaderboard entries: re-run rapid + fingerprint + full
post-train probes, then rescore composite. Fingerprint-aware dedup — one
recompute per unique graph_fingerprint, rescore propagates to all siblings
via screening_recompute.recompute_screening_metrics.

Mirrors the Program Detail "re-screen" flow but operates over many entries so
you do not have to click per-program.

Dry-run default. --apply executes. Writes audit JSONL to
research/reports/rescreen_batch_YYYY-MM-DD.jsonl with before/after composite
diffs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from research.scientist.discovery_scoring import discovery_score
from research.scientist.notebook import LabNotebook
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("research/lab_notebook.db")
DEFAULT_AUDIT = Path(
    f"research/reports/rescreen_batch_{date.today().isoformat()}.jsonl"
)
DEFAULT_LOG = Path(f"research/reports/rescreen_batch_{date.today().isoformat()}.log")
DEFAULT_PROGRESS = Path(
    f"research/reports/rescreen_batch_{date.today().isoformat()}.progress.json"
)


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def _sys_avail_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def _gpu_mem_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
    except Exception:
        pass
    return None


def _write_progress(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic breadcrumb so a hard-kill still leaves last-known state."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(path)
    except Exception:
        pass


PROBE_COLUMNS = (
    "fp_jacobian_spectral_norm",
    "fp_interaction_locality",
    "activation_sparsity_score",
    "fp_isotropy",
    "fp_rank_ratio",
    "fp_sensitivity_uniformity",
    "hellaswag_acc",
    "induction_auc",
    "binding_auc",
)

# Tiers considered "below screening" (backlog / pre-screening / failed states).
# An entry in one of these that rescores above V7_SCREENING_THRESHOLD gets
# promoted up to 'screening'.
_BACKLOG_TIERS = frozenset(
    {
        "screened_out",
        "investigation_failed",
        "investigation_fingerprint_incomplete",
        "",  # no tier / NULL
    }
)


def _apply_tier_decision(
    nb,
    entry_id: str,
    current_tier: str,
    new_composite: Optional[float],
) -> Dict[str, Any]:
    """Promote-to-screening or fail based on tier-appropriate threshold.

    Rules:
      - Reference rows (is_reference=1) are never failed. They're pinned baselines.
      - If current tier is below screening (backlog/fail) and composite clears
        V7_SCREENING_THRESHOLD → promote_to_tier('screening').
      - If current tier == 'screening' and composite < V7_SCREENING_THRESHOLD
        → tier = 'screened_out'.
      - If current tier == 'investigation' and composite < V7_INVESTIGATION_THRESHOLD
        → tier = 'investigation_failed'.
      - If current tier == 'validation' and composite < V7_INVESTIGATION_THRESHOLD
        → tier = 'validation_failed'. (No separate V7_VALIDATION_THRESHOLD in
        thresholds.py; investigation threshold is the floor a validation entry
        must still clear.)
      - No auto-promotion above 'screening'.
    """
    from research.scientist.thresholds import (
        V7_SCREENING_THRESHOLD,
        V7_INVESTIGATION_THRESHOLD,
        TIER_RANK,
    )

    action = "unchanged"
    new_tier = current_tier
    reason = None

    if new_composite is None:
        return {
            "tier_action": "unchanged",
            "tier_before": current_tier,
            "tier_after": current_tier,
            "reason": "no_composite",
        }

    is_ref_row = nb.conn.execute(
        "SELECT COALESCE(is_reference, 0) FROM leaderboard WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    is_reference = bool(is_ref_row[0]) if is_ref_row else False
    if is_reference:
        return {
            "tier_action": "unchanged_reference",
            "tier_before": current_tier,
            "tier_after": current_tier,
            "reason": "is_reference",
        }

    below_screening_rank = TIER_RANK.get(current_tier, 0) < TIER_RANK.get(
        "screening", 1
    )

    # Promote up to screening (only direction we auto-promote).
    if new_composite >= V7_SCREENING_THRESHOLD and (
        current_tier in _BACKLOG_TIERS or below_screening_rank
    ):
        nb.promote_to_tier(
            entry_id,
            "screening",
            notes="rescreen_leaderboard_batch: cleared V7_SCREENING_THRESHOLD",
        )
        return {
            "tier_action": "promoted_to_screening",
            "tier_before": current_tier,
            "tier_after": "screening",
            "threshold": float(V7_SCREENING_THRESHOLD),
        }

    # Tier-aware fail — each tier has its own floor it must still clear.
    fail_tier: Optional[str] = None
    threshold_used: Optional[float] = None
    if current_tier == "screening" and new_composite < V7_SCREENING_THRESHOLD:
        fail_tier = "screened_out"
        threshold_used = V7_SCREENING_THRESHOLD
    elif current_tier == "investigation" and new_composite < V7_INVESTIGATION_THRESHOLD:
        fail_tier = "investigation_failed"
        threshold_used = V7_INVESTIGATION_THRESHOLD
    elif current_tier == "validation" and new_composite < V7_INVESTIGATION_THRESHOLD:
        fail_tier = "validation_failed"
        threshold_used = V7_INVESTIGATION_THRESHOLD
    elif current_tier == "breakthrough" and new_composite < V7_INVESTIGATION_THRESHOLD:
        fail_tier = "breakthrough_failed"
        threshold_used = V7_INVESTIGATION_THRESHOLD

    if fail_tier is not None:
        # Direct UPDATE — promote_to_tier would block a downgrade.
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET tier = ?,
                notes = COALESCE(notes, '') ||
                        ' | rescreen_leaderboard_batch: composite ' ||
                        printf('%.2f', ?) ||
                        ' below threshold ' || printf('%.2f', ?) ||
                        ' (was ' || COALESCE(?, 'unknown') || ')'
            WHERE entry_id = ?
            """,
            (
                fail_tier,
                float(new_composite),
                float(threshold_used),
                current_tier,
                entry_id,
            ),
        )
        nb.conn.commit()
        return {
            "tier_action": "failed",
            "tier_before": current_tier,
            "tier_after": fail_tier,
            "threshold": float(threshold_used),
        }

    return {
        "tier_action": action,
        "tier_before": current_tier,
        "tier_after": new_tier,
        "reason": reason,
    }


def _select_candidates(
    db: Path,
    tier: Optional[str],
    min_composite: Optional[float],
    max_composite: Optional[float],
    missing_probes: bool,
    require_stage1: bool,
    rank_by: str,
    cohort: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Select leaderboard entries matching filter, dedup by graph_fingerprint.

    rank_by: 'composite' orders by stored l.composite_score (V7).
             'dashboard' orders by the shared backend discovery score used for
             Discoveries ranking, then applies fingerprint dedup and the final
             unique-candidate limit.
    cohort: None = no cohort filter. 'backfill' = entries the UI tags as
            Backfill (trust_label='backfill_observation' OR
            comparability_label='reconstructed_init_variant' OR
            result_cohort='backfill').
    """
    con = connect_readonly(db)
    try:
        clauses = [
            "TRIM(COALESCE(pr.graph_json, '')) <> ''",
            "pr.graph_json <> '{}'",
        ]
        params: List[Any] = []
        if tier:
            clauses.append("l.tier = ?")
            params.append(tier)
        if min_composite is not None:
            clauses.append("COALESCE(l.composite_score, -1e9) >= ?")
            params.append(float(min_composite))
        if max_composite is not None:
            clauses.append("COALESCE(l.composite_score, 1e9) <= ?")
            params.append(float(max_composite))
        if require_stage1:
            clauses.append("COALESCE(pr.stage1_passed, 0) = 1")
        if missing_probes:
            null_checks = " OR ".join(f"pr.{c} IS NULL" for c in PROBE_COLUMNS)
            clauses.append(f"({null_checks})")
        if cohort == "backfill":
            clauses.append(
                "("
                "LOWER(COALESCE(pr.trust_label, '')) = 'backfill_observation' OR "
                "LOWER(COALESCE(pr.comparability_label, '')) = 'reconstructed_init_variant' OR "
                "LOWER(COALESCE(pr.result_cohort, '')) = 'backfill'"
                ")"
            )

        if rank_by == "dashboard":
            order_clause = ""
            limit_clause = ""
        else:
            order_clause = "ORDER BY COALESCE(l.composite_score, -1e9) DESC"
            oversample = int(limit) * 4 if int(limit) > 0 else 0
            limit_clause = (
                f" LIMIT {max(int(limit), oversample, 500)}" if int(limit) > 0 else ""
            )
        sql = f"""
            SELECT
                l.entry_id, l.result_id, l.composite_score, l.tier,
                pr.graph_fingerprint, pr.loss_ratio, pr.stage1_passed,
                pr.hellaswag_acc, pr.induction_auc, pr.binding_auc,
                pr.fp_jacobian_spectral_norm, pr.activation_sparsity_score,
                pr.validation_loss_ratio, pr.discovery_loss_ratio, pr.novelty_score,
                pr.trust_label, pr.comparability_label,
                pr.baseline_loss_ratio, pr.most_similar_to,
                pr.param_count, pr.graph_n_params_estimate,
                pr.loss_improvement_rate, pr.throughput_tok_s,
                pr.forward_time_ms, pr.flops_forward, pr.flops_per_param,
                pr.sparsity_ratio, pr.peak_memory_mb,
                pr.routing_utilization_entropy, pr.routing_drop_rate,
                pr.routing_capacity_overflow_count, pr.routing_confidence_mean,
                pr.routing_confidence_std, pr.routing_tokens_total,
                pr.routing_tokens_processed, pr.routing_expert_count,
                pr.routing_expert_utilization_json, pr.routing_savings_ratio,
                pr.depth_savings_ratio, pr.effective_depth_ratio,
                pr.recursion_savings_ratio,
                pr.blimp_overall_accuracy, pr.ar_auc,
                pr.graph_json, pr.routing_mode,
                l.investigation_loss_ratio, l.screening_loss_ratio,
                l.validation_baseline_ratio, l.scaling_param_efficiency,
                l.scaling_gate_passed, l.robustness_noise_score,
                l.init_sensitivity_std,
                l.quant_int8_retention
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE {" AND ".join(clauses)}
            {order_clause}
            {limit_clause}
        """
        rows = [dict(row) for row in con.execute(sql, params).fetchall()]

        if rank_by == "dashboard":
            for row in rows:
                row["architecture_family"] = LabNotebook._classify_architecture_family(
                    row.get("graph_json"),
                    row.get("routing_mode"),
                )
                row["dashboard_score"] = discovery_score(row)
            rows.sort(
                key=lambda row: (
                    float(row.get("dashboard_score") or 0.0),
                    float(row.get("composite_score") or -1e9),
                ),
                reverse=True,
            )
        else:
            for row in rows:
                row["dashboard_score"] = None

        seen_fps: set = set()
        unique: List[Dict[str, Any]] = []
        dupes = 0
        for r in rows:
            fp = (r.get("graph_fingerprint") or "").strip()
            if fp and fp in seen_fps:
                dupes += 1
                continue
            if fp:
                seen_fps.add(fp)
            r.pop("graph_json", None)
            unique.append(r)
        if dupes:
            logger.info("Fingerprint dedup: %d duplicate siblings collapsed", dupes)
        if int(limit) > 0:
            return unique[: int(limit)]
        return unique
    finally:
        con.close()


def _missing_probe_count(row: Dict[str, Any]) -> int:
    return sum(1 for c in PROBE_COLUMNS if row.get(c) is None)


def _query_composite(conn: sqlite3.Connection, entry_id: str) -> Optional[float]:
    r = conn.execute(
        "SELECT composite_score FROM leaderboard WHERE entry_id = ?",
        (str(entry_id),),
    ).fetchone()
    if r is None:
        return None
    val = r[0] if not isinstance(r, sqlite3.Row) else r["composite_score"]
    return float(val) if val is not None else None


def _rescreen_one(
    nb,
    db_path: Path,
    cand: Dict[str, Any],
    device: str,
    allow_insufficient_learning_metrics: bool,
) -> Dict[str, Any]:
    """Call recompute_screening_metrics and capture before/after diff."""
    from research.scientist.screening_recompute import recompute_screening_metrics

    result_id = str(cand["result_id"])
    entry_id = str(cand["entry_id"])
    fp = (cand.get("graph_fingerprint") or "").strip()

    before = _query_composite(nb.conn, entry_id)
    sibling_entries_before: List[Dict[str, Any]] = []
    if fp:
        rows = nb.conn.execute(
            """
            SELECT l.entry_id, l.composite_score
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint = ? AND l.entry_id != ?
            """,
            (fp, entry_id),
        ).fetchall()
        sibling_entries_before = [dict(r) for r in rows]

    t0 = time.time()
    payload = recompute_screening_metrics(
        nb=nb,
        notebook_path=db_path,
        result_id=result_id,
        device=device,
        allow_insufficient_learning_metrics=allow_insufficient_learning_metrics,
        provenance_source="rescreen_leaderboard_batch",
    )
    elapsed = time.time() - t0

    after = _query_composite(nb.conn, entry_id)

    # Tier decision for the primary entry.
    current_tier_row = nb.conn.execute(
        "SELECT tier FROM leaderboard WHERE entry_id = ?", (entry_id,)
    ).fetchone()
    current_tier = str(current_tier_row["tier"] or "") if current_tier_row else ""
    tier_result = _apply_tier_decision(nb, entry_id, current_tier, after)

    sibling_diffs: List[Dict[str, Any]] = []
    for s in sibling_entries_before:
        sib_entry_id = str(s["entry_id"])
        after_s = _query_composite(nb.conn, sib_entry_id)
        sib_tier_row = nb.conn.execute(
            "SELECT tier FROM leaderboard WHERE entry_id = ?", (sib_entry_id,)
        ).fetchone()
        sib_current_tier = str(sib_tier_row["tier"] or "") if sib_tier_row else ""
        sib_tier = _apply_tier_decision(nb, sib_entry_id, sib_current_tier, after_s)
        sibling_diffs.append(
            {
                "entry_id": sib_entry_id,
                "before": s["composite_score"],
                "after": after_s,
                "tier_action": sib_tier["tier_action"],
                "tier_before": sib_tier["tier_before"],
                "tier_after": sib_tier["tier_after"],
            }
        )

    updates = payload.get("updates") or {}
    errors = payload.get("errors") or {}
    return {
        "result_id": result_id,
        "entry_id": entry_id,
        "graph_fingerprint": fp,
        "status": payload.get("status"),
        "composite_before": before,
        "composite_after": after,
        "composite_diff": (
            (after - before) if (before is not None and after is not None) else None
        ),
        "probe_fields_updated": sorted(updates.keys()),
        "n_probe_fields_updated": len(updates),
        "errors": errors,
        "sibling_updates": sibling_diffs,
        "tier_action": tier_result["tier_action"],
        "tier_before": tier_result["tier_before"],
        "tier_after": tier_result["tier_after"],
        "elapsed_sec": round(elapsed, 2),
    }


def _fix_tiers_only(nb, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply tier decision to each candidate without rescreening.

    Uses the composite_score already stored in the leaderboard. Intended for
    cleaning up entries that were rescored by a previous run but missed the
    tier update logic.
    """
    audit: List[Dict[str, Any]] = []
    for cand in candidates:
        entry_id = str(cand["entry_id"])
        composite = _query_composite(nb.conn, entry_id)
        current_tier_row = nb.conn.execute(
            "SELECT tier FROM leaderboard WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        current_tier = str(current_tier_row["tier"] or "") if current_tier_row else ""
        result = _apply_tier_decision(nb, entry_id, current_tier, composite)
        audit.append(
            {
                "entry_id": entry_id,
                "result_id": cand.get("result_id"),
                "graph_fingerprint": cand.get("graph_fingerprint"),
                "composite": composite,
                "tier_action": result["tier_action"],
                "tier_before": result["tier_before"],
                "tier_after": result["tier_after"],
            }
        )
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--tier",
        default="screening",
        help="Leaderboard tier to select (default: screening). "
        "Pass empty string to disable tier filter.",
    )
    ap.add_argument(
        "--rank-by",
        choices=["composite", "dashboard"],
        default="composite",
        help="composite: order by stored V7 composite_score (default). "
        "dashboard: order by the shared backend discovery score "
        "used by Discoveries, then dedup by fingerprint before "
        "applying the final limit.",
    )
    ap.add_argument(
        "--cohort",
        choices=["backfill"],
        default=None,
        help="Restrict to a cohort. 'backfill' selects entries the UI tags "
        "as Backfill (trust_label/comparability_label/result_cohort).",
    )
    ap.add_argument("--min-composite", type=float, default=None)
    ap.add_argument("--max-composite", type=float, default=None)
    ap.add_argument(
        "--missing-probes",
        action="store_true",
        help="Only select entries with at least one NULL probe field",
    )
    ap.add_argument(
        "--require-stage1",
        action="store_true",
        help="Restrict to program_results with stage1_passed=1",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Cap selection count (0 = no cap)"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Execute rescreen. Without --apply, dry-run prints selection only.",
    )
    ap.add_argument(
        "--fix-tiers-only",
        action="store_true",
        help="Skip rescreening; just apply promote/fail tier logic using "
        "current composite_score. Use this to clean up entries rescored "
        "by a prior run that missed tier adjustment.",
    )
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help="Path for the persistent log file (timestamped, "
        "append mode — survives terminal/process crash).",
    )
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_PROGRESS,
        help="Atomic breadcrumb file containing last-known iter "
        "state. Survives even SIGKILL/OOM-kill.",
    )
    ap.add_argument(
        "--memory-cap-gb",
        type=float,
        default=100.0,
        help="Bail gracefully if process RSS exceeds this many GB. "
        "Box has 128GB; default 100GB leaves headroom so the "
        "kernel OOM killer never fires (which freezes the "
        "system). Set 0 to disable.",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Skip fingerprints already present in this audit JSONL. "
        "Defaults to --audit path if it exists.",
    )
    ap.add_argument(
        "--strict-learning-gate",
        action="store_true",
        help="If set, treat insufficient-learning gate failures as errors "
        "(no probes captured). Default is to tolerate the gate and "
        "capture hellaswag/blimp/binding probes anyway — the tier "
        "logic here handles promote/fail, not the gate.",
    )
    args = ap.parse_args()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(args.log_file, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(sh)
    logger.info(
        "rescreen_batch start pid=%d rss=%.2fGB sys_avail=%.2fGB args=%s",
        os.getpid(),
        _rss_gb(),
        _sys_avail_gb(),
        vars(args),
    )

    tier_arg = args.tier if args.tier else None
    candidates = _select_candidates(
        db=args.db,
        tier=tier_arg,
        min_composite=args.min_composite,
        max_composite=args.max_composite,
        missing_probes=args.missing_probes,
        require_stage1=args.require_stage1,
        rank_by=args.rank_by,
        cohort=args.cohort,
        limit=args.limit,
    )
    print(
        f"Selected {len(candidates)} unique-fingerprint candidates "
        f"(tier={tier_arg or 'any'}, rank_by={args.rank_by}, "
        f"cohort={args.cohort or 'any'})"
    )
    print(
        f"{'rank':>4}  {'fp':<16}  {'composite':>10}  {'dashboard':>10}  "
        f"{'missing':>7}  {'val_lr':>7}  {'s1':>3}  {'hs':>5}  {'trust':<24}"
    )
    for i, c in enumerate(candidates, 1):
        fp = (c.get("graph_fingerprint") or "")[:16]
        comp = c.get("composite_score")
        dash = c.get("dashboard_score")
        loss = c.get("validation_loss_ratio") or c.get("loss_ratio")
        s1 = c.get("stage1_passed")
        hs = c.get("hellaswag_acc")
        trust = (c.get("trust_label") or "-")[:24]
        print(
            f"{i:>4}  {fp:<16}  "
            f"{(comp if comp is not None else float('nan')):>10.2f}  "
            f"{(dash if dash is not None else float('nan')):>10.2f}  "
            f"{_missing_probe_count(c):>7}  "
            f"{(loss if loss is not None else float('nan')):>7.3f}  "
            f"{(s1 or 0):>3}  "
            f"{(hs if hs is not None else float('nan')):>5.2f}  "
            f"{trust:<24}"
        )

    if not args.apply:
        mode = "fix-tiers-only" if args.fix_tiers_only else "rescreen"
        print(f"\nDry-run only (mode={mode}). Re-run with --apply to execute.")
        return

    # Resume support — skip fingerprints already done.
    resume_path = args.resume_from or (args.audit if args.audit.exists() else None)
    done_fps: set = set()
    if resume_path and resume_path.exists():
        with resume_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fp = str(row.get("graph_fingerprint") or "").strip()
                if fp:
                    done_fps.add(fp)
        if done_fps:
            pre = len(candidates)
            candidates = [
                c
                for c in candidates
                if str(c.get("graph_fingerprint") or "").strip() not in done_fps
            ]
            print(
                f"Resume: skipping {pre - len(candidates)} entries already in "
                f"{resume_path} ({len(candidates)} remaining)"
            )

    # Write path — aria-db writer flock.
    from research.scientist.runner._helpers_gate import clear_gpu_memory
    import gc

    # Open audit in append mode so incremental writes survive crashes.
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_fh = args.audit.open("a", buffering=1)  # line-buffered
    nb = LabNotebook(str(args.db))
    audit_rows: List[Dict[str, Any]] = []
    try:
        if args.fix_tiers_only:
            audit_rows = _fix_tiers_only(nb, candidates)
            for r in audit_rows:
                print(
                    f"  [{(r.get('graph_fingerprint') or '')[:16]}] "
                    f"composite={r.get('composite')} "
                    f"tier {r.get('tier_before')}→{r.get('tier_after')} "
                    f"({r.get('tier_action')})"
                )
            nb.flush_writes()
            _write_audit(args.audit, audit_rows)
            n_promoted = sum(
                1 for r in audit_rows if r.get("tier_action") == "promoted_to_screening"
            )
            n_failed = sum(1 for r in audit_rows if r.get("tier_action") == "failed")
            print(
                f"\nDone (fix-tiers-only). "
                f"promoted={n_promoted} failed={n_failed} "
                f"unchanged={len(audit_rows) - n_promoted - n_failed}. "
                f"Audit: {args.audit}"
            )
            return
        for i, cand in enumerate(candidates, 1):
            fp_short = (cand.get("graph_fingerprint") or "")[:16]
            rss_before = _rss_gb()
            sys_avail = _sys_avail_gb()
            gpu_before = _gpu_mem_gb()
            _write_progress(
                args.progress_file,
                {
                    "phase": "starting",
                    "iter": i,
                    "total": len(candidates),
                    "result_id": cand.get("result_id"),
                    "entry_id": cand.get("entry_id"),
                    "graph_fingerprint": cand.get("graph_fingerprint"),
                    "rss_gb": round(rss_before, 2),
                    "sys_avail_gb": round(sys_avail, 2),
                    "gpu_alloc_gb": round(gpu_before, 2)
                    if gpu_before is not None
                    else None,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                },
            )
            logger.info(
                "iter=%d/%d fp=%s result_id=%s rss=%.2fGB sys_avail=%.2fGB gpu=%s",
                i,
                len(candidates),
                fp_short,
                cand.get("result_id"),
                rss_before,
                sys_avail,
                f"{gpu_before:.2f}GB" if gpu_before is not None else "n/a",
            )

            # Memory ceiling — bail BEFORE the kernel OOM-killer fires (which
            # freezes the box). Audit JSONL is line-buffered, so resume works.
            if args.memory_cap_gb > 0 and rss_before >= args.memory_cap_gb:
                logger.error(
                    "Memory cap hit: rss=%.2fGB >= cap=%.2fGB. Bailing at iter=%d "
                    "to avoid OOM-kill. Resume with same --audit path.",
                    rss_before,
                    args.memory_cap_gb,
                    i,
                )
                _write_progress(
                    args.progress_file,
                    {
                        "phase": "bail_memory_cap",
                        "iter": i,
                        "total": len(candidates),
                        "rss_gb": round(rss_before, 2),
                        "cap_gb": args.memory_cap_gb,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                break

            try:
                result = _rescreen_one(
                    nb=nb,
                    db_path=args.db,
                    cand=cand,
                    device=str(args.device),
                    # Inverted: flag name is --strict-learning-gate but the
                    # underlying API expects "allow" semantics.
                    allow_insufficient_learning_metrics=not bool(
                        args.strict_learning_gate
                    ),
                )
            except Exception as e:
                logger.warning(
                    "Rescreen failed for %s (fp=%s): %s",
                    cand.get("result_id"),
                    fp_short,
                    e,
                )
                result = {
                    "result_id": cand.get("result_id"),
                    "entry_id": cand.get("entry_id"),
                    "graph_fingerprint": cand.get("graph_fingerprint"),
                    "status": "error",
                    "error": str(e),
                }
            audit_rows.append(result)
            # Incremental audit write — survives crash/freeze.
            audit_fh.write(json.dumps(result, default=str) + "\n")
            audit_fh.flush()

            diff = result.get("composite_diff")
            diff_str = f"{diff:+.2f}" if isinstance(diff, (int, float)) else "n/a"
            tier_str = (
                f"{result.get('tier_before')}→{result.get('tier_after')}"
                if result.get("tier_action") != "unchanged"
                else result.get("tier_before") or "-"
            )
            rss_after = _rss_gb()
            gpu_after = _gpu_mem_gb()
            logger.info(
                "  [%d/%d] %s %s composite %s→%s (%s) tier=%s (%s) "
                "fields=%s errs=%s elapsed=%ss rss=%.2fGB(Δ%+.2f) gpu=%s",
                i,
                len(candidates),
                fp_short,
                result.get("status"),
                result.get("composite_before"),
                result.get("composite_after"),
                diff_str,
                tier_str,
                result.get("tier_action"),
                result.get("n_probe_fields_updated", 0),
                ",".join(sorted((result.get("errors") or {}).keys())) or "-",
                result.get("elapsed_sec", 0),
                rss_after,
                rss_after - rss_before,
                f"{gpu_after:.2f}GB" if gpu_after is not None else "n/a",
            )

            # Aggressive per-entry cleanup to prevent memory creep.
            # Without this, ~100 entries will saturate GPU+CPU memory.
            clear_gpu_memory()
            gc.collect()
            # Flush DB writes periodically so memory doesn't pile up in aria-db.
            if i % 10 == 0:
                nb.flush_writes()
            _write_progress(
                args.progress_file,
                {
                    "phase": "completed",
                    "iter": i,
                    "total": len(candidates),
                    "result_id": cand.get("result_id"),
                    "graph_fingerprint": cand.get("graph_fingerprint"),
                    "status": result.get("status"),
                    "rss_gb": round(_rss_gb(), 2),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                },
            )

        nb.flush_writes()
    finally:
        nb.close()
        audit_fh.close()

    n_ok = sum(1 for r in audit_rows if r.get("status") == "ok")
    n_err = sum(1 for r in audit_rows if r.get("status") == "error")
    n_promoted = sum(
        1 for r in audit_rows if r.get("tier_action") == "promoted_to_screening"
    )
    n_failed = sum(1 for r in audit_rows if r.get("tier_action") == "failed")
    diffs = [
        r["composite_diff"]
        for r in audit_rows
        if isinstance(r.get("composite_diff"), (int, float))
    ]
    print(
        f"\nDone. {n_ok} ok, {n_err} errors. "
        f"Tier changes: promoted={n_promoted} failed={n_failed}. "
        f"Audit: {args.audit}"
    )
    if diffs:
        print(
            f"Composite diff stats: n={len(diffs)} "
            f"min={min(diffs):+.2f} max={max(diffs):+.2f} "
            f"mean={sum(diffs) / len(diffs):+.2f}"
        )


def _write_audit(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
