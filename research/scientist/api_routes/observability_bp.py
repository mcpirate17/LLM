"""Observability API routes — component health, alerts, training SSE stream,
error log, experiment lifecycle, throughput, op analytics, resource utilization,
grammar evolution, failure patterns, leaderboard dynamics, insight effectiveness,
DB health, and API health."""

from __future__ import annotations

import json as _json
import logging
import math as _math
import os
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Response, jsonify, request
from ..json_utils import fast_dumps as _json_dumps
from ._helpers import get_runner
from ._utils import with_notebook_context
from .deps import ApiRouteContext, get_notebook

logger = logging.getLogger(__name__)

# ── Alert thresholds (configurable via /api/observability/alerts/config) ──
_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "s0_pass_rate_min": 0.30,
    "s1_pass_rate_min": 0.05,
    "grad_norm_max": 50000.0,
    "routing_collapse_score_min": 0.3,
    "op_failure_rate_max": 0.90,
    "stale_experiment_hours": 6,
}

# ── Component health cache (TTL-based) ──
_health_cache: Dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 120.0  # 2 minutes

# ── OpIndex cache (5-minute TTL, keyed by window label) ──
_op_index_caches: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_OP_INDEX_TTL = 300.0  # 5 minutes

# Window label → seconds mapping (reused across endpoints)
_WINDOW_SECONDS: Dict[str, Optional[int]] = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
    "all": None,
}

# ── Throughput cache (60s TTL) ──
_throughput_cache: Optional[Dict[str, Any]] = None
_throughput_cache_ts: float = 0.0
_THROUGHPUT_TTL = 60.0


def _build_op_index(notebook_path: str, window: str = "all") -> Dict[str, Any]:
    """Parse graph_json, build op co-occurrence and loss data.

    Args:
        window: Time window label ("1h", "6h", "24h", "7d", "all").

    Returns dict with:
      pair_counts: {(op_a, op_b): {n, s0, s1}}
      loss_by_op: {op_name: [loss_ratio, ...]}
      failure_groups: {error_type: {ops: Counter, count: int}}
      stored_rates: {op_name: {n, s0, s1}}
      corrected_rates: {op_name: {n, s0, s1, excluded}} — excludes non-op-specific errors
    """
    now = time.monotonic()
    cached = _op_index_caches.get(window)
    if cached and (now - cached[0]) < _OP_INDEX_TTL:
        return cached[1]

    # Error types that are NOT the fault of individual ops:
    #   RuntimeError — implementation bug (dtype mismatch, shape error in compiler)
    #   causality_violation — graph-level constraint, not op-specific
    _NON_OP_ERRORS = frozenset({"RuntimeError", "causality_violation"})

    pair_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "s0": 0, "s1": 0}
    )
    loss_by_op: Dict[str, List[float]] = defaultdict(list)
    failure_groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"ops": defaultdict(int), "count": 0}
    )
    stored_rates: Dict[str, Dict[str, int]] = {}
    # corrected_rates: same as stored_rates but excludes programs whose failure
    # was caused by non-op-specific errors (implementation bugs, graph issues)
    corrected_rates: Dict[str, Dict[str, int]] = {}

    nb = get_notebook(notebook_path)
    try:
        window_seconds = _WINDOW_SECONDS.get(window)
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            rows = nb.conn.execute(
                "SELECT graph_json, stage0_passed, stage1_passed, loss_ratio, error_type "
                "FROM program_results WHERE graph_json IS NOT NULL AND timestamp > ?",
                (cutoff,),
            ).fetchall()
        else:
            rows = nb.conn.execute(
                "SELECT graph_json, stage0_passed, stage1_passed, loss_ratio, error_type "
                "FROM program_results WHERE graph_json IS NOT NULL"
            ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.debug("op_index query failed: %s", exc)
        rows = []

    for r in rows:
        try:
            g = _json.loads(r["graph_json"])
        except (ValueError, KeyError) as exc:
            logger.debug("Skipping unparseable graph_json: %s", exc)
            continue
        raw_nodes = g.get("nodes")
        if not isinstance(raw_nodes, dict):
            continue
        ops = sorted(
            {
                n.get("op_name", n.get("op", ""))
                for n in raw_nodes.values()
                if isinstance(n, dict)
            }
            - {"", "input"}
        )
        s0 = bool(r["stage0_passed"])
        s1 = bool(r["stage1_passed"])
        lr = r["loss_ratio"]
        et = r["error_type"] or ""

        # Is this a non-op-specific failure?
        is_non_op_failure = not s0 and et in _NON_OP_ERRORS

        # Per-op stored rates (raw — shared with _get_component_health)
        for op in ops:
            s = stored_rates.setdefault(op, {"n": 0, "s0": 0, "s1": 0})
            s["n"] += 1
            if s0:
                s["s0"] += 1
            if s1:
                s["s1"] += 1

            # Corrected rates: exclude non-op-specific failures entirely
            c = corrected_rates.setdefault(
                op, {"n": 0, "s0": 0, "s1": 0, "excluded": 0}
            )
            if is_non_op_failure:
                c["excluded"] += 1
            else:
                c["n"] += 1
                if s0:
                    c["s0"] += 1
                if s1:
                    c["s1"] += 1

        # Pair co-occurrence
        for i, a in enumerate(ops):
            for b in ops[i + 1 :]:
                key = (a, b)
                pair_counts[key]["n"] += 1
                if s0:
                    pair_counts[key]["s0"] += 1
                if s1:
                    pair_counts[key]["s1"] += 1

        # Loss by op
        if lr is not None and s0:
            for op in ops:
                loss_by_op[op].append(float(lr))

        # Failure grouping (S0 failures)
        if not s0 and et:
            failure_groups[et]["count"] += 1
            for op in ops:
                failure_groups[et]["ops"][op] += 1

        # S1 failure grouping (passed S0 but failed S1)
        if s0 and not s1 and et:
            s1_et = "s1_" + et
            failure_groups[s1_et]["count"] += 1
            for op in ops:
                failure_groups[s1_et]["ops"][op] += 1

    result = {
        "pair_counts": dict(pair_counts),
        "loss_by_op": dict(loss_by_op),
        "failure_groups": {
            k: {"ops": dict(v["ops"]), "count": v["count"]}
            for k, v in failure_groups.items()
        },
        "stored_rates": stored_rates,
        "corrected_rates": corrected_rates,
    }
    _op_index_caches[window] = (now, result)
    return result


def _get_throughput(notebook_path: str) -> Dict[str, Any]:
    """Compute throughput metrics with 60s TTL cache."""
    global _throughput_cache, _throughput_cache_ts
    now_mono = time.monotonic()
    if _throughput_cache and (now_mono - _throughput_cache_ts) < _THROUGHPUT_TTL:
        return _throughput_cache

    now = time.time()
    windows = {"1h": 3600, "6h": 21600, "24h": 86400}
    result: Dict[str, Any] = {}

    nb = get_notebook(notebook_path)
    try:
        for label, seconds in windows.items():
            cutoff = now - seconds
            row = nb.conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) as s0, "
                "SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1 "
                "FROM program_results WHERE timestamp > ?",
                (cutoff,),
            ).fetchone()
            result[label] = {
                "total": row["total"] or 0,
                "s0_passed": row["s0"] or 0,
                "s1_passed": row["s1"] or 0,
                "s0_rate": round((row["s0"] or 0) / max(row["total"] or 1, 1), 3),
                "s1_rate": round((row["s1"] or 0) / max(row["s0"] or 1, 1), 3)
                if (row["s0"] or 0) > 0
                else 0.0,
            }
    except sqlite3.OperationalError as e:
        logger.error("Throughput query error: %s", e)

    result["computed_at"] = now
    _throughput_cache = result
    _throughput_cache_ts = now_mono
    return result


def _get_component_health(notebook_path: str, window: str = "all") -> Dict[str, Any]:
    """Build component health report from op_success_rates + profiling data."""
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    # Cache is only valid for the same window
    if (
        _health_cache
        and (now - _health_cache_ts) < _HEALTH_CACHE_TTL
        and _health_cache.get("_window") == window
    ):
        return _health_cache

    nb = get_notebook(notebook_path)
    try:
        window_seconds = _WINDOW_SECONDS.get(window)
        if window_seconds is not None:
            since_ts = time.time() - window_seconds
            op_rates = nb.get_op_success_rates_windowed(since_ts)
        else:
            op_rates = nb.get_op_success_rates()
    except sqlite3.OperationalError as exc:
        logger.debug("op_success_rates query failed: %s", exc)
        op_rates = []

    # ── TF-IDF failure attribution ──
    # Raw S0 pass rate blames ubiquitous ops (linear_proj, add, layernorm) for
    # failures they didn't cause — they appear in nearly every graph, so their
    # "failure rate" tracks the overall failure rate, not op-specific problems.
    #
    # TF-IDF separates signal from noise:
    #   TF  = P(fail | op present)      — op's failure rate
    #   IDF = log(N / n_containing_op)   — rarity of the op
    #   blame = TF * IDF                 — high only for RARE ops that ALSO fail
    #
    # Data sources:
    #   op_success_rates — includes ALL generated graphs (S0 pass AND fail),
    #       accumulated via merge_op_failure_counts.  This is the right source
    #       for TF-IDF because it captures the full population of generated
    #       graphs, not just the survivors stored in program_results.
    #   program_results  — only S0+ survivors.  Used for ground-truth S1 rate.
    # Ground-truth S1 from stored program_results — reuse op_index cache
    # to avoid a second full-table scan of graph_json
    stored_rates: Dict[str, Dict[str, int]] = {}
    corrected_rates: Dict[str, Dict[str, int]] = {}
    try:
        idx = _build_op_index(notebook_path, window=window)
        stored_rates = idx.get("stored_rates", {})
        corrected_rates = idx.get("corrected_rates", {})
    except (sqlite3.OperationalError, KeyError, TypeError) as exc:
        logger.debug(
            "Failed to build op index for observability health window=%s: %s",
            window,
            exc,
            exc_info=True,
        )

    # Total generated graphs (from op_success_rates): sum of n_used is an
    # over-count (each graph has multiple ops), but the MAX n_used across
    # all ops is a lower bound on total graphs.  For IDF we need the number
    # of "documents" (graphs).
    max_n_used = max((r.get("n_used") or 0 for r in op_rates), default=0)

    def _compute_blame(
        op_name: str, n_used: int, n_s0: int
    ) -> Tuple[float, float, float]:
        """Return (blame_score, tf, idf) for an op.

        Uses op_success_rates data (includes failure accumulation) for TF,
        and max_n_used as document count for IDF.

        blame = tf * idf where:
          tf  = 1 - (n_s0 / n_used) = failure rate when op present
          idf = log(max_n_used / n_used) = rarity of op in generated graphs
        """
        if n_used == 0 or max_n_used == 0:
            return 0.0, 0.0, 0.0
        tf = 1.0 - (n_s0 / n_used)
        idf = _math.log(max(max_n_used, 1) / n_used) if n_used < max_n_used else 0.0
        return tf * idf, tf, idf

    # Load profiling data for gradient health
    grad_health: Dict[str, Dict] = {}
    try:
        profiling_db = Path("research/profiling/component_profiles.db")
        if profiling_db.exists():
            conn = sqlite3.connect(str(profiling_db), timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT op_name, grad_norm, grad_exploding, grad_vanishing, "
                    "output_has_nan, output_has_inf, forward_time_us, backward_time_us, "
                    "lipschitz_estimate, error FROM op_profiles"
                ).fetchall()
            finally:
                conn.close()
            for r in rows:
                grad_health[r["op_name"]] = {
                    "grad_norm": float(r["grad_norm"])
                    if r["grad_norm"] is not None
                    else None,
                    "grad_exploding": bool(r["grad_exploding"])
                    if r["grad_exploding"] is not None
                    else False,
                    "grad_vanishing": bool(r["grad_vanishing"])
                    if r["grad_vanishing"] is not None
                    else False,
                    "has_nan": bool(r["output_has_nan"])
                    if r["output_has_nan"] is not None
                    else False,
                    "has_inf": bool(r["output_has_inf"])
                    if r["output_has_inf"] is not None
                    else False,
                    "fwd_us": float(r["forward_time_us"])
                    if r["forward_time_us"] is not None
                    else None,
                    "bwd_us": float(r["backward_time_us"])
                    if r["backward_time_us"] is not None
                    else None,
                    "lipschitz": float(r["lipschitz_estimate"])
                    if r["lipschitz_estimate"] is not None
                    else None,
                    "profile_error": r["error"],
                }
    except (sqlite3.OperationalError, KeyError, TypeError, OSError) as exc:
        logger.debug(
            "Failed to load component profiling health data: %s", exc, exc_info=True
        )

    # Build per-component health
    components: List[Dict[str, Any]] = []
    total_healthy = 0
    total_degraded = 0
    total_broken = 0

    for row in op_rates:
        op = row["op_name"]

        # TF-IDF blame: use corrected rates that exclude non-op-specific errors
        # (RuntimeError = dtype/shape bugs, causality_violation = graph-level).
        # This prevents ops from being blamed for failures they didn't cause.
        cr = corrected_rates.get(op)
        if cr and cr["n"] > 0:
            blame, tf, idf = _compute_blame(op, cr["n"], cr["s0"])
        else:
            # Fallback to raw op_success_rates if no corrected data
            raw_n = row.get("n_used") or 0
            raw_s0 = row.get("n_stage0_passed") or 0
            blame, tf, idf = _compute_blame(op, raw_n, raw_s0)

        # Raw blame for transparency (includes all errors)
        raw_n = row.get("n_used") or 0
        raw_s0 = row.get("n_stage0_passed") or 0
        raw_blame, raw_tf, _ = _compute_blame(op, raw_n, raw_s0)
        n_excluded = cr["excluded"] if cr else 0

        # For display (s0_rate, s1_rate), prefer ground-truth from stored
        # program_results so the pass rates reflect actual outcomes.
        sr = stored_rates.get(op)
        if sr and sr["n"] > 0:
            n_used = sr["n"]
            n_s0 = sr["s0"]
            n_s1 = sr["s1"]
        else:
            n_used = raw_n
            n_s0 = raw_s0
            n_s1 = row.get("n_stage1_passed") or 0
        n_s05 = row.get("n_stage05_passed") or 0

        s0_rate = n_s0 / max(n_used, 1)
        s1_rate = n_s1 / max(n_s0, 1) if n_s0 > 0 else 0.0

        prof = grad_health.get(op, {})
        grad_norm = prof.get("grad_norm")
        has_nan = prof.get("has_nan", False)
        lipschitz = prof.get("lipschitz") or 0.0

        # ── Structural ops: exempt from S1 blame ──
        # Ops with no learnable parameters (splits, masks, reduce ops)
        # should not be blamed for low S1 — they are scaffolding.
        from research.synthesis.context_rules import S1_EXEMPT_OPS

        if op in S1_EXEMPT_OPS:
            status = "structural"
            reasons = ["scaffolding op — not a standalone learner"]
            total_healthy += 1
            components.append(
                {
                    "op": op,
                    "status": status,
                    "reasons": reasons,
                    "n_used": n_used,
                    "n_s0": n_s0,
                    "n_s05": n_s05,
                    "n_s1": n_s1,
                    "s0_rate": round(s0_rate, 4),
                    "s1_rate": round(s1_rate, 4),
                    "blame": round(blame, 3),
                    "raw_blame": round(raw_blame, 3),
                    "n_excluded": n_excluded,
                    "grad_norm": grad_norm,
                    "lipschitz": lipschitz,
                    "data_source": "search+profiling" if prof else "search",
                }
            )
            continue

        # ── Classify health using TF-IDF blame ──
        #
        # blame = TF * IDF measures failure-specificity:
        #   high blame → rare op that consistently appears in failing graphs
        #   low blame  → common op whose failure rate mirrors the base rate
        #
        # Redemption: if the op has high S1 rate in stored program_results,
        # it clearly works when composed correctly — blame is from bad
        # pairings, not the op itself.  s1_rate > 50% redeems any blame.
        #
        # Sample-size scaling: rare ops with few samples get inflated IDF,
        # so we require proportionally higher blame to classify as broken.
        # At n=50+ the thresholds are baseline; below that they scale up
        # to avoid false positives on under-sampled ops.
        #
        # Thresholds:
        #   broken:   blame > scaled_threshold AND n >= 20 AND NOT redeemed
        #   degraded: blame > scaled_threshold/2 AND n >= 10 AND NOT redeemed
        #             OR lipschitz > 2.0   (gradient amplifier)
        #             OR grad_norm > 50000 (extreme gradient)
        #             OR NaN/Inf in profiling (always broken)
        # Redeem if high S1 rate OR if profiling shows the op is clean
        # (no NaN, reasonable gradient, lipschitz <= 1.0 means the op
        # itself is fine — blame comes from bad graph composition)
        _profile_clean = (
            bool(prof)
            and not has_nan
            and not prof.get("has_inf", False)
            and not prof.get("profile_error")
            and (grad_norm is None or grad_norm < 5000)
            and lipschitz <= 1.01
        )
        redeemed = s1_rate > 0.5 or _profile_clean
        # Scale blame thresholds: base 2.0 at n>=50, up to 4.0 at n=20
        _confidence = min(raw_n / 50.0, 1.0)
        _broken_threshold = 2.0 + 2.0 * (1.0 - _confidence)
        _degraded_threshold = _broken_threshold / 2.0
        status = "healthy"
        reasons: List[str] = []

        if has_nan or prof.get("has_inf", False):
            status = "broken"
            reasons.append("NaN/Inf in output")
        elif prof.get("profile_error"):
            status = "broken"
            reasons.append(f"profile error: {prof['profile_error'][:60]}")
        elif blame > _broken_threshold and raw_n >= 20 and not redeemed:
            status = "broken"
            reasons.append(
                f"TF-IDF blame={blame:.2f} "
                f"(fail_rate={tf:.0%}, rarity={idf:.1f}, n={raw_n})"
            )
        elif grad_norm is not None and grad_norm > 50000:
            status = "degraded"
            reasons.append(f"grad_norm={grad_norm:.0f}")
        elif blame > _degraded_threshold and raw_n >= 10 and not redeemed:
            status = "degraded"
            reasons.append(
                f"TF-IDF blame={blame:.2f} "
                f"(fail_rate={tf:.0%}, rarity={idf:.1f}, n={raw_n})"
            )
        elif lipschitz > 2.0:
            status = "degraded"
            reasons.append(f"gradient amplifier (lipschitz={lipschitz:.1f})")
        elif s1_rate < 0.05 and n_s0 >= 10:
            status = "degraded"
            reasons.append(f"S1 pass rate {s1_rate:.0%}")

        if status == "healthy":
            total_healthy += 1
        elif status == "degraded":
            total_degraded += 1
        else:
            total_broken += 1

        data_source = "search+profiling" if prof else "search"
        components.append(
            {
                "op": op,
                "status": status,
                "reasons": reasons,
                "n_used": n_used,
                "s0_rate": round(s0_rate, 3),
                "s1_rate": round(s1_rate, 3),
                "blame": round(blame, 3),
                "fail_rate": round(tf, 3),
                "rarity": round(idf, 3),
                "raw_blame": round(raw_blame, 3),
                "raw_fail_rate": round(raw_tf, 3),
                "n_excluded": n_excluded,
                "lipschitz": round(lipschitz, 2) if lipschitz else None,
                "grad_norm": round(grad_norm, 1) if grad_norm is not None else None,
                "has_nan": has_nan,
                "fwd_us": prof.get("fwd_us"),
                "bwd_us": prof.get("bwd_us"),
                "data_source": data_source,
            }
        )

    # Add profiled-only ops (not in op_success_rates yet)
    rated_ops = {row["op_name"] for row in op_rates}
    for op_name, prof in grad_health.items():
        if op_name in rated_ops:
            continue
        status = "healthy"
        reasons = []
        if prof.get("has_nan") or prof.get("has_inf"):
            status = "broken"
            reasons.append("NaN/Inf in profiling")
        elif prof.get("profile_error"):
            status = "broken"
            reasons.append("profile error")
        elif prof.get("grad_norm") and prof["grad_norm"] > 50000:
            status = "degraded"
            reasons.append(f"grad_norm={prof['grad_norm']:.0f}")
        if status == "healthy":
            total_healthy += 1
        elif status == "degraded":
            total_degraded += 1
        else:
            total_broken += 1
        components.append(
            {
                "op": op_name,
                "status": status,
                "reasons": reasons,
                "n_used": 0,
                "s0_rate": None,
                "s1_rate": None,
                "grad_norm": round(prof["grad_norm"], 1)
                if prof.get("grad_norm") is not None
                else None,
                "grad_exploding": prof.get("grad_exploding", False),
                "has_nan": prof.get("has_nan", False),
                "fwd_us": prof.get("fwd_us"),
                "bwd_us": prof.get("bwd_us"),
                "data_source": "profiling_only",
            }
        )

    components.sort(
        key=lambda c: (
            {"broken": 0, "degraded": 1, "structural": 2, "healthy": 3}.get(
                c["status"], 2
            ),
            -(c["n_used"] or 0),
        )
    )

    result = {
        "components": components,
        "total": len(components),
        "healthy": total_healthy,
        "degraded": total_degraded,
        "broken": total_broken,
        "cached_at": time.time(),
        "window": window,
        "_window": window,
    }
    _health_cache = result
    _health_cache_ts = now
    return result


def _evaluate_alerts(
    notebook_path: str, thresholds: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Evaluate alert conditions against current state."""
    alerts: List[Dict[str, Any]] = []
    now = time.time()

    nb = get_notebook(notebook_path)
    try:
        nb.get_dashboard_summary()
    except (sqlite3.OperationalError, KeyError, TypeError) as exc:
        logger.debug("Dashboard summary prefetch failed for alerts: %s", exc)

    # Alert 1: S0 pass rate too low
    try:
        row = nb.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) as passed "
            "FROM program_results WHERE timestamp > ?",
            (now - 3600,),
        ).fetchone()
        if row and row["total"] >= 10:
            rate = row["passed"] / row["total"]
            if rate < thresholds.get("s0_pass_rate_min", 0.30):
                alerts.append(
                    {
                        "id": "low_s0_rate",
                        "severity": "critical",
                        "title": "Low S0 pass rate",
                        "message": f"S0 pass rate is {rate:.0%} in the last hour ({row['passed']}/{row['total']})",
                        "value": round(rate, 3),
                        "threshold": thresholds["s0_pass_rate_min"],
                        "timestamp": now,
                    }
                )
    except sqlite3.OperationalError:
        logger.debug("Low S0 pass-rate alert evaluation failed", exc_info=True)

    # Alert 2: S1 pass rate critically low
    try:
        row = nb.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as passed "
            "FROM program_results WHERE stage0_passed = 1 AND timestamp > ?",
            (now - 7200,),
        ).fetchone()
        if row and row["total"] >= 20:
            rate = row["passed"] / row["total"]
            if rate < thresholds.get("s1_pass_rate_min", 0.05):
                alerts.append(
                    {
                        "id": "low_s1_rate",
                        "severity": "critical",
                        "title": "Low S1 pass rate",
                        "message": f"S1 pass rate is {rate:.0%} in the last 2 hours ({row['passed']}/{row['total']})",
                        "value": round(rate, 3),
                        "threshold": thresholds["s1_pass_rate_min"],
                        "timestamp": now,
                    }
                )
    except sqlite3.OperationalError:
        logger.debug("Low S1 pass-rate alert evaluation failed", exc_info=True)

    # Alert 3: Routing collapse detected
    try:
        row = nb.conn.execute(
            "SELECT AVG(CAST(json_extract(starvation_report_json, '$.collapse_score') AS REAL)) as avg_collapse "
            "FROM program_results "
            "WHERE starvation_report_json IS NOT NULL AND timestamp > ?",
            (now - 7200,),
        ).fetchone()
        if row and row["avg_collapse"] is not None:
            score = float(row["avg_collapse"])
            threshold = thresholds.get("routing_collapse_score_min", 0.3)
            if score < threshold:
                alerts.append(
                    {
                        "id": "routing_collapse",
                        "severity": "warning",
                        "title": "Routing collapse detected",
                        "message": f"Average routing health score is {score:.2f} (threshold: {threshold})",
                        "value": round(score, 3),
                        "threshold": threshold,
                        "timestamp": now,
                    }
                )
    except sqlite3.OperationalError:
        logger.debug("Routing collapse alert evaluation failed", exc_info=True)

    # Alert 4: Broken components
    health = _get_component_health(notebook_path)
    if health["broken"] > 0:
        broken_ops = [c["op"] for c in health["components"] if c["status"] == "broken"][
            :5
        ]
        alerts.append(
            {
                "id": "broken_components",
                "severity": "warning" if health["broken"] <= 3 else "critical",
                "title": f"{health['broken']} broken component(s)",
                "message": f"Broken ops: {', '.join(broken_ops)}"
                + (" ..." if health["broken"] > 5 else ""),
                "value": health["broken"],
                "threshold": 0,
                "timestamp": now,
            }
        )

    # Alert 5: Stale experiment (no results in N hours)
    try:
        row = nb.conn.execute(
            "SELECT MAX(timestamp) as last_ts FROM program_results"
        ).fetchone()
        if row and row["last_ts"]:
            hours_since = (now - row["last_ts"]) / 3600
            threshold_hours = thresholds.get("stale_experiment_hours", 6)
            if hours_since > threshold_hours:
                alerts.append(
                    {
                        "id": "stale_pipeline",
                        "severity": "info",
                        "title": "Pipeline idle",
                        "message": f"No new results in {hours_since:.1f} hours",
                        "value": round(hours_since, 1),
                        "threshold": threshold_hours,
                        "timestamp": now,
                    }
                )
    except sqlite3.OperationalError:
        logger.debug("Stale pipeline alert evaluation failed", exc_info=True)

    alerts.sort(key=lambda a: {"critical": 0, "warning": 1, "info": 2}[a["severity"]])
    return alerts


def register_observability_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/observability/health")
    def api_component_health():
        """Component health grid — all ops with status/metrics."""
        try:
            window = request.args.get("window", "all")
            if window not in _WINDOW_SECONDS:
                window = "all"
            health = _get_component_health(notebook_path, window=window)
            return jsonify(health)
        except Exception as e:
            logger.error("Error in /api/observability/health: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/observability/health/refresh", methods=["POST"])
    def api_component_health_refresh():
        """Force-refresh component health + OpIndex caches."""
        global _health_cache_ts
        _health_cache_ts = 0.0
        _op_index_caches.clear()
        health = _get_component_health(notebook_path)
        return jsonify(health)

    @app.route("/api/observability/alerts")
    def api_alerts():
        """Active alerts based on threshold evaluation."""
        try:
            alerts = _evaluate_alerts(notebook_path, _DEFAULT_THRESHOLDS)
            return jsonify({"alerts": alerts, "thresholds": _DEFAULT_THRESHOLDS})
        except Exception as e:
            logger.error("Error in /api/observability/alerts: %s", e)
            return jsonify({"alerts": [], "error": str(e)}), 500

    @app.route("/api/observability/alerts/config", methods=["GET", "POST"])
    def api_alert_config():
        """Get/set alert thresholds."""
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            for k, v in body.items():
                if k in _DEFAULT_THRESHOLDS:
                    try:
                        _DEFAULT_THRESHOLDS[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        return jsonify(_DEFAULT_THRESHOLDS)

    @app.route("/api/observability/stream")
    def api_training_stream():
        """SSE stream for real-time training progress + routing telemetry."""
        runner = get_runner(notebook_path)

        def event_stream():
            last_step = -1
            while True:
                try:
                    progress = runner.progress
                    if progress is None:
                        time.sleep(2)
                        yield "event: keepalive\ndata: {}\n\n"
                        continue

                    prog_dict = (
                        progress.to_dict() if hasattr(progress, "to_dict") else {}
                    )
                    current_step = prog_dict.get("current_program", 0)

                    if current_step != last_step:
                        last_step = current_step
                        # Include live loss curve tail (last 20 points)
                        try:
                            curve = runner.get_live_loss_curve()
                            if curve:
                                prog_dict["loss_curve_tail"] = curve[-20:]
                        except (AttributeError, KeyError, TypeError) as exc:
                            logger.debug(
                                "Failed to read live loss curve for observability stream: %s",
                                exc,
                            )
                        data = _json_dumps(prog_dict, safe=True)
                        yield f"event: progress\ndata: {data}\n\n"

                    # Check for alerts on each tick
                    try:
                        alerts = _evaluate_alerts(notebook_path, _DEFAULT_THRESHOLDS)
                        if alerts:
                            alert_data = _json_dumps({"alerts": alerts}, safe=True)
                            yield f"event: alerts\ndata: {alert_data}\n\n"
                    except (sqlite3.OperationalError, KeyError, TypeError) as exc:
                        logger.debug(
                            "Failed to emit observability alerts event: %s",
                            exc,
                            exc_info=True,
                        )

                    time.sleep(3)
                except GeneratorExit:
                    return
                except Exception as exc:
                    # Outer SSE loop: keep broad to prevent stream death
                    logger.debug(
                        "Observability event stream tick failed: %s", exc, exc_info=True
                    )
                    time.sleep(5)
                    yield "event: keepalive\ndata: {}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/observability/failure-blocklist")
    @wnb
    def api_failure_blocklist(nb=None):
        """Op-pair failure signatures that should be auto-disabled."""
        blocklist = nb.get_failure_signature_blocklist(
            min_seen=int(request.args.get("min_seen", 10)),
            max_fail_rate=float(request.args.get("max_fail_rate", 0.90)),
        )
        return jsonify({"blocklist": blocklist, "count": len(blocklist)})

    # ── P0: Error log ──────────────────────────────────────────────────

    @app.route("/api/observability/error-log")
    @wnb
    def api_error_log(nb=None):
        """Recent error events from learning_log."""
        limit = int(request.args.get("limit", 50))
        error_types = (
            "error",
            "compile_error",
            "training_error",
            "eval_error",
            "runtime_error",
            "stage0_fail",
        )
        placeholders = ",".join("?" for _ in error_types)
        rows = nb.conn.execute(
            f"SELECT id, timestamp, event_type, description, evidence "
            f"FROM learning_log WHERE event_type IN ({placeholders}) "
            f"ORDER BY timestamp DESC LIMIT ?",
            (*error_types, limit),
        ).fetchall()
        entries = [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "description": r["description"],
                "evidence": r["evidence"][:500] if r["evidence"] else None,
            }
            for r in rows
        ]
        return jsonify({"errors": entries, "count": len(entries)})

    # ── P0: Experiment lifecycle ───────────────────────────────────────

    @app.route("/api/observability/experiment-lifecycle")
    @wnb
    def api_experiment_lifecycle(nb=None):
        """Recent experiments with orphan detection."""
        limit = int(request.args.get("limit", 20))
        now = time.time()
        rows = nb.conn.execute(
            "SELECT experiment_id, experiment_type, status, "
            "started_at, completed_at, duration_seconds, "
            "n_programs_generated, n_stage0_passed, n_stage1_passed, "
            "best_loss_ratio "
            "FROM experiments ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        exp_ids = [str(r["experiment_id"]) for r in rows if r["experiment_id"]]
        persisted_by_exp = {}
        if exp_ids:
            placeholders = ",".join("?" for _ in exp_ids)
            persisted_rows = nb.conn.execute(
                f"""
                SELECT
                    experiment_id,
                    COUNT(*) AS persisted_program_rows,
                    SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) AS persisted_stage0_passed,
                    SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) AS persisted_stage1_passed
                FROM program_results
                WHERE experiment_id IN ({placeholders})
                GROUP BY experiment_id
                """,
                exp_ids,
            ).fetchall()
            persisted_by_exp = {
                str(r["experiment_id"]): dict(r) for r in persisted_rows
            }
        experiments = []
        for r in rows:
            entry = dict(r)
            persisted = persisted_by_exp.get(str(r["experiment_id"])) or {}
            persisted_rows = int(persisted.get("persisted_program_rows") or 0)
            persisted_s0 = int(persisted.get("persisted_stage0_passed") or 0)
            persisted_s1 = int(persisted.get("persisted_stage1_passed") or 0)
            stored_total = int(r["n_programs_generated"] or 0)
            stored_s0 = int(r["n_stage0_passed"] or 0)
            stored_s1 = int(r["n_stage1_passed"] or 0)
            entry["persisted_program_rows"] = persisted_rows
            entry["persisted_stage0_passed"] = persisted_s0
            entry["persisted_stage1_passed"] = persisted_s1
            entry["count_discrepancy"] = {
                "programs": persisted_rows - stored_total,
                "stage0": persisted_s0 - stored_s0,
                "stage1": persisted_s1 - stored_s1,
            }
            entry["count_mismatch"] = any(
                value != 0 for value in entry["count_discrepancy"].values()
            )
            # Flag orphans: running > 2 hours
            if (
                r["status"] == "running"
                and r["started_at"]
                and (now - r["started_at"]) > 7200
            ):
                entry["orphan"] = True
                entry["running_hours"] = round((now - r["started_at"]) / 3600, 1)
            else:
                entry["orphan"] = False
            experiments.append(entry)
        return jsonify({"experiments": experiments, "count": len(experiments)})

    @app.route("/api/observability/experiment-lifecycle/cleanup", methods=["POST"])
    @wnb
    def api_experiment_lifecycle_cleanup(nb=None):
        """Mark stale running experiments as failed."""
        timeout = int(request.args.get("timeout_minutes", 60))
        cleaned = nb.cleanup_stale_experiments(timeout_minutes=timeout)
        return jsonify({"cleaned": cleaned})

    # ── P0: Throughput ─────────────────────────────────────────────────

    @app.route("/api/observability/throughput")
    def api_throughput():
        """Program evaluation throughput by time window."""
        try:
            data = _get_throughput(notebook_path)
            return jsonify(data)
        except Exception as e:
            logger.error("Error in /api/observability/throughput: %s", e)
            return jsonify({"error": str(e)}), 500

    # ── P1: Op pairs ───────────────────────────────────────────────────

    @app.route("/api/observability/op-pairs")
    def api_op_pairs():
        """Top op pairs by co-occurrence with s0/s1 rates."""
        top_n = int(request.args.get("top", 30))
        try:
            idx = _build_op_index(notebook_path)
            pairs = []
            for (a, b), counts in idx["pair_counts"].items():
                if counts["n"] < 3:
                    continue
                pairs.append(
                    {
                        "op_a": a,
                        "op_b": b,
                        "n": counts["n"],
                        "s0_rate": round(counts["s0"] / max(counts["n"], 1), 3),
                        "s1_rate": round(counts["s1"] / max(counts["s0"], 1), 3)
                        if counts["s0"] > 0
                        else 0.0,
                    }
                )
            pairs.sort(key=lambda p: p["n"], reverse=True)
            return jsonify({"pairs": pairs[:top_n], "total_pairs": len(pairs)})
        except Exception as e:
            logger.error("Error in /api/observability/op-pairs: %s", e)
            return jsonify({"pairs": [], "error": str(e)}), 500

    # ── P1: Loss distribution ──────────────────────────────────────────

    @app.route("/api/observability/loss-distribution")
    def api_loss_distribution():
        """Per-op loss ratio distribution (box plot data)."""
        try:
            idx = _build_op_index(notebook_path)
            dist = []
            for op, values in idx["loss_by_op"].items():
                if len(values) < 3:
                    continue
                sv = sorted(values)
                n = len(sv)
                dist.append(
                    {
                        "op": op,
                        "n": n,
                        "min": round(sv[0], 4),
                        "q1": round(sv[n // 4], 4),
                        "median": round(statistics.median(sv), 4),
                        "q3": round(sv[3 * n // 4], 4),
                        "max": round(sv[-1], 4),
                        "mean": round(statistics.mean(sv), 4),
                    }
                )
            dist.sort(key=lambda d: d["median"])
            return jsonify({"distributions": dist})
        except Exception as e:
            logger.error("Error in /api/observability/loss-distribution: %s", e)
            return jsonify({"distributions": [], "error": str(e)}), 500

    # ── P1: Resource utilization ───────────────────────────────────────

    @app.route("/api/observability/resource-utilization")
    def api_resource_utilization():
        """Live CPU%, RAM%, GPU allocation."""
        result: Dict[str, Any] = {}
        try:
            import psutil

            result["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            result["ram_percent"] = mem.percent
            result["ram_used_gb"] = round(mem.used / (1024**3), 2)
            result["ram_total_gb"] = round(mem.total / (1024**3), 2)
        except ImportError:
            result["cpu_percent"] = None
            result["ram_percent"] = None

        try:
            import torch

            if torch.cuda.is_available():
                result["gpu_allocated_gb"] = round(
                    torch.cuda.memory_allocated() / (1024**3), 3
                )
                result["gpu_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / (1024**3), 3
                )
                result["gpu_name"] = torch.cuda.get_device_name(0)
            else:
                result["gpu_allocated_gb"] = None
                result["gpu_reserved_gb"] = None
        except (ImportError, RuntimeError) as exc:
            logger.debug("GPU info unavailable: %s", exc)
            result["gpu_allocated_gb"] = None
            result["gpu_reserved_gb"] = None

        return jsonify(result)

    # ── P1: API health ─────────────────────────────────────────────────

    @app.route("/api/observability/api-health")
    def api_api_health():
        """API request counters by endpoint × status bucket."""
        try:
            from ..api import _api_health_counters, _api_health_lock

            with _api_health_lock:
                snapshot = dict(_api_health_counters)
            return jsonify({"counters": snapshot})
        except ImportError:
            return jsonify({"counters": {}, "note": "counters not available"})

    # ── P2: Grammar evolution ──────────────────────────────────────────

    @app.route("/api/observability/grammar-evolution")
    @wnb
    def api_grammar_evolution(nb=None):
        """Timeline of grammar weight changes from learning_log."""
        limit = int(request.args.get("limit", 30))
        rows = nb.conn.execute(
            "SELECT id, timestamp, description, old_weights, new_weights, evidence "
            "FROM learning_log "
            "WHERE event_type IN ('grammar_weights_applied', 'chat_grammar_overrides_applied') "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        entries = []
        for r in rows:
            entry: Dict[str, Any] = {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "description": r["description"],
            }
            # Parse weight diffs
            try:
                old_w = _json.loads(r["old_weights"]) if r["old_weights"] else None
                new_w = _json.loads(r["new_weights"]) if r["new_weights"] else None
                if (
                    old_w
                    and new_w
                    and isinstance(old_w, dict)
                    and isinstance(new_w, dict)
                ):
                    changes = {}
                    for k in set(old_w) | set(new_w):
                        ov = old_w.get(k, 1.0)
                        nv = new_w.get(k, 1.0)
                        if ov != nv:
                            changes[k] = {"old": ov, "new": nv}
                    entry["changes"] = changes
                else:
                    entry["changes"] = {}
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Failed to parse grammar weight diff: %s", exc)
                entry["changes"] = {}
            entries.append(entry)
        return jsonify({"events": entries, "count": len(entries)})

    # ── P2: Failure patterns ───────────────────────────────────────────

    @app.route("/api/observability/failure-patterns")
    def api_obs_failure_patterns():
        """Failed graphs grouped by error_type with top co-occurring ops."""
        top_ops = int(request.args.get("top_ops", 5))
        try:
            idx = _build_op_index(notebook_path)
            patterns = []
            for error_type, data in idx["failure_groups"].items():
                sorted_ops = sorted(
                    data["ops"].items(), key=lambda x: x[1], reverse=True
                )[:top_ops]
                patterns.append(
                    {
                        "error_type": error_type,
                        "count": data["count"],
                        "top_ops": [
                            {"op": op, "occurrences": cnt} for op, cnt in sorted_ops
                        ],
                    }
                )
            patterns.sort(key=lambda p: p["count"], reverse=True)
            return jsonify({"patterns": patterns})
        except Exception as e:
            logger.error("Error in /api/observability/failure-patterns: %s", e)
            return jsonify({"patterns": [], "error": str(e)}), 500

    # ── P2: Leaderboard dynamics ───────────────────────────────────────

    @app.route("/api/observability/leaderboard-dynamics")
    @wnb
    def api_leaderboard_dynamics(nb=None):
        """Tier counts per day + recent promotions."""
        # Daily tier counts
        rows = nb.conn.execute(
            "SELECT date(timestamp, 'unixepoch') as day, tier, COUNT(*) as cnt "
            "FROM leaderboard GROUP BY day, tier ORDER BY day"
        ).fetchall()
        daily: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            day = r["day"]
            if day is None:
                continue
            daily[day][r["tier"]] = r["cnt"]

        # Recent promotions (last 20)
        promos = nb.conn.execute(
            "SELECT entry_id, result_id, tier, timestamp, "
            "screening_loss_ratio, investigation_loss_ratio, composite_score "
            "FROM leaderboard ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()

        return jsonify(
            {
                "daily": {d: dict(tiers) for d, tiers in sorted(daily.items())},
                "recent_promotions": [dict(r) for r in promos],
            }
        )

    # ── P2: Insight effectiveness ──────────────────────────────────────

    @app.route("/api/observability/insight-effectiveness")
    @wnb
    def api_insight_effectiveness(nb=None):
        """Insights with prediction accuracy and Bayesian posterior mean."""
        rows = nb.conn.execute(
            "SELECT insight_id, category, insight_type, subject_key, "
            "content, confidence, status, n_predictions, n_correct, "
            "alpha, beta_ "
            "FROM insights WHERE n_predictions > 0 "
            "ORDER BY n_predictions DESC LIMIT 50"
        ).fetchall()
        entries = []
        for r in rows:
            n_pred = r["n_predictions"] or 0
            n_corr = r["n_correct"] or 0
            alpha = r["alpha"] or 1.0
            beta = r["beta_"] or 1.0
            entries.append(
                {
                    "insight_id": r["insight_id"],
                    "category": r["category"],
                    "insight_type": r["insight_type"],
                    "subject_key": r["subject_key"],
                    "content": (r["content"] or "")[:200],
                    "confidence": r["confidence"],
                    "status": r["status"],
                    "n_predictions": n_pred,
                    "n_correct": n_corr,
                    "accuracy": round(n_corr / max(n_pred, 1), 3),
                    "bayesian_mean": round(alpha / (alpha + beta), 3),
                }
            )
        return jsonify({"insights": entries, "count": len(entries)})

    # ── P3: DB health ──────────────────────────────────────────────────

    @app.route("/api/observability/db-health")
    @wnb
    def api_db_health(nb=None):
        """Database file size, table row counts, WAL size."""
        result: Dict[str, Any] = {}
        try:
            db_path = notebook_path
            result["db_size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
            wal_path = db_path + "-wal"
            if os.path.exists(wal_path):
                result["wal_size_mb"] = round(
                    os.path.getsize(wal_path) / (1024 * 1024), 2
                )
            else:
                result["wal_size_mb"] = 0.0
        except OSError as exc:
            logger.debug("DB file size check failed: %s", exc)
            result["db_size_mb"] = None
            result["wal_size_mb"] = None

        try:
            tables = [
                "program_results",
                "experiments",
                "leaderboard",
                "learning_log",
                "insights",
                "training_curves",
                "entries",
            ]
            row_counts = {}
            for t in tables:
                try:
                    row = nb.conn.execute(f"SELECT COUNT(*) as c FROM {t}").fetchone()
                    row_counts[t] = row["c"] if row else 0
                except sqlite3.OperationalError:
                    row_counts[t] = None
            result["row_counts"] = row_counts
        except (sqlite3.OperationalError, KeyError, TypeError) as e:
            logger.debug("DB health row count query failed: %s", e)
            result["row_counts"] = {}
            result["error"] = str(e)

        return jsonify(result)
