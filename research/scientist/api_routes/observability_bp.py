"""Observability API routes — component health, alerts, training SSE stream."""
from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from flask import Response, jsonify, request
from ..notebook import LabNotebook
from ..json_utils import fast_dumps as _json_dumps
from ._helpers import get_runner
from .deps import ApiRouteContext

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


def _get_component_health(notebook_path: str) -> Dict[str, Any]:
    """Build component health report from op_success_rates + profiling data."""
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return _health_cache

    nb = LabNotebook(notebook_path)
    try:
        op_rates = nb.get_op_success_rates()
    except Exception:
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
    import json as _json
    import math as _math

    # Ground-truth S1 from stored program_results
    stored_rates: Dict[str, Dict[str, int]] = {}
    try:
        rows = nb.conn.execute(
            "SELECT graph_json, stage0_passed, stage1_passed "
            "FROM program_results WHERE graph_json IS NOT NULL"
        ).fetchall()
        for r in rows:
            try:
                g = _json.loads(r[0])
            except Exception:
                continue
            ops = {
                n.get("op_name", n.get("op", ""))
                for n in g.get("nodes", {}).values()
                if isinstance(n, dict)
            } - {"", "input"}
            for op_name in ops:
                s = stored_rates.setdefault(op_name, {"n": 0, "s0": 0, "s1": 0})
                s["n"] += 1
                if r[1]:
                    s["s0"] += 1
                if r[2]:
                    s["s1"] += 1
    except Exception:
        pass
    finally:
        nb.close()

    # Total generated graphs (from op_success_rates): sum of n_used is an
    # over-count (each graph has multiple ops), but the MAX n_used across
    # all ops is a lower bound on total graphs.  For IDF we need the number
    # of "documents" (graphs).
    max_n_used = max((r.get("n_used") or 0 for r in op_rates), default=0)

    def _compute_blame(op_name: str, n_used: int, n_s0: int) -> Tuple[float, float, float]:
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
        from research.profiling.schema import ComponentDB
        with ComponentDB() as cdb:
            rows = cdb.query(
                "SELECT op_name, grad_norm, grad_exploding, grad_vanishing, "
                "output_has_nan, output_has_inf, forward_time_us, backward_time_us, "
                "lipschitz_estimate, error FROM op_profiles"
            )
            for r in rows:
                grad_health[r["op_name"]] = {
                    "grad_norm": float(r["grad_norm"]) if r["grad_norm"] is not None else None,
                    "grad_exploding": bool(r["grad_exploding"]) if r["grad_exploding"] is not None else False,
                    "grad_vanishing": bool(r["grad_vanishing"]) if r["grad_vanishing"] is not None else False,
                    "has_nan": bool(r["output_has_nan"]) if r["output_has_nan"] is not None else False,
                    "has_inf": bool(r["output_has_inf"]) if r["output_has_inf"] is not None else False,
                    "fwd_us": float(r["forward_time_us"]) if r["forward_time_us"] is not None else None,
                    "bwd_us": float(r["backward_time_us"]) if r["backward_time_us"] is not None else None,
                    "lipschitz": float(r["lipschitz_estimate"]) if r["lipschitz_estimate"] is not None else None,
                    "profile_error": r["error"],
                }
    except Exception:
        pass

    # Build per-component health
    components: List[Dict[str, Any]] = []
    total_healthy = 0
    total_degraded = 0
    total_broken = 0

    for row in op_rates:
        op = row["op_name"]

        # TF-IDF uses the full op_success_rates (includes co-occurrence counts
        # from failed graphs) to compute blame — that's the signal we need.
        raw_n = row.get("n_used") or 0
        raw_s0 = row.get("n_stage0_passed") or 0
        blame, tf, idf = _compute_blame(op, raw_n, raw_s0)

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
        # Thresholds (require n >= 5 to avoid noise from small samples):
        #   broken:   blame > 2.0 AND n >= 5 AND NOT redeemed
        #   degraded: blame > 1.0 AND n >= 5 AND NOT redeemed
        #             OR lipschitz > 2.0   (gradient amplifier)
        #             OR grad_norm > 50000 (extreme gradient)
        #             OR NaN/Inf in profiling (always broken)
        redeemed = s1_rate > 0.5  # works well when composed correctly
        status = "healthy"
        reasons: List[str] = []

        if has_nan or prof.get("has_inf", False):
            status = "broken"
            reasons.append("NaN/Inf in output")
        elif prof.get("profile_error"):
            status = "broken"
            reasons.append(f"profile error: {prof['profile_error'][:60]}")
        elif blame > 2.0 and raw_n >= 5 and not redeemed:
            status = "broken"
            reasons.append(
                f"TF-IDF blame={blame:.2f} "
                f"(fail_rate={tf:.0%}, rarity={idf:.1f}, n={raw_n})"
            )
        elif grad_norm is not None and grad_norm > 50000:
            status = "degraded"
            reasons.append(f"grad_norm={grad_norm:.0f}")
        elif blame > 1.0 and raw_n >= 5 and not redeemed:
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

        components.append({
            "op": op,
            "status": status,
            "reasons": reasons,
            "n_used": n_used,
            "s0_rate": round(s0_rate, 3),
            "s1_rate": round(s1_rate, 3),
            "blame": round(blame, 3),
            "fail_rate": round(tf, 3),
            "rarity": round(idf, 3),
            "lipschitz": round(lipschitz, 2) if lipschitz else None,
            "grad_norm": round(grad_norm, 1) if grad_norm is not None else None,
            "has_nan": has_nan,
            "fwd_us": prof.get("fwd_us"),
            "bwd_us": prof.get("bwd_us"),
        })

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
        components.append({
            "op": op_name,
            "status": status,
            "reasons": reasons,
            "n_used": 0,
            "s0_rate": None,
            "s1_rate": None,
            "grad_norm": round(prof["grad_norm"], 1) if prof.get("grad_norm") is not None else None,
            "grad_exploding": prof.get("grad_exploding", False),
            "has_nan": prof.get("has_nan", False),
            "fwd_us": prof.get("fwd_us"),
            "bwd_us": prof.get("bwd_us"),
        })

    components.sort(key=lambda c: (
        {"broken": 0, "degraded": 1, "healthy": 2}[c["status"]],
        -(c["n_used"] or 0),
    ))

    result = {
        "components": components,
        "total": len(components),
        "healthy": total_healthy,
        "degraded": total_degraded,
        "broken": total_broken,
        "cached_at": time.time(),
    }
    _health_cache = result
    _health_cache_ts = now
    return result


def _evaluate_alerts(notebook_path: str, thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate alert conditions against current state."""
    alerts: List[Dict[str, Any]] = []
    now = time.time()

    nb = LabNotebook(notebook_path)
    try:
        summary = nb.get_dashboard_summary()
    except Exception:
        summary = {}

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
                alerts.append({
                    "id": "low_s0_rate",
                    "severity": "critical",
                    "title": "Low S0 pass rate",
                    "message": f"S0 pass rate is {rate:.0%} in the last hour ({row['passed']}/{row['total']})",
                    "value": round(rate, 3),
                    "threshold": thresholds["s0_pass_rate_min"],
                    "timestamp": now,
                })
    except Exception:
        pass

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
                alerts.append({
                    "id": "low_s1_rate",
                    "severity": "critical",
                    "title": "Low S1 pass rate",
                    "message": f"S1 pass rate is {rate:.0%} in the last 2 hours ({row['passed']}/{row['total']})",
                    "value": round(rate, 3),
                    "threshold": thresholds["s1_pass_rate_min"],
                    "timestamp": now,
                })
    except Exception:
        pass

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
                alerts.append({
                    "id": "routing_collapse",
                    "severity": "warning",
                    "title": "Routing collapse detected",
                    "message": f"Average routing health score is {score:.2f} (threshold: {threshold})",
                    "value": round(score, 3),
                    "threshold": threshold,
                    "timestamp": now,
                })
    except Exception:
        pass

    # Alert 4: Broken components
    health = _get_component_health(notebook_path)
    if health["broken"] > 0:
        broken_ops = [c["op"] for c in health["components"] if c["status"] == "broken"][:5]
        alerts.append({
            "id": "broken_components",
            "severity": "warning" if health["broken"] <= 3 else "critical",
            "title": f"{health['broken']} broken component(s)",
            "message": f"Broken ops: {', '.join(broken_ops)}" + (" ..." if health["broken"] > 5 else ""),
            "value": health["broken"],
            "threshold": 0,
            "timestamp": now,
        })

    # Alert 5: Stale experiment (no results in N hours)
    try:
        row = nb.conn.execute(
            "SELECT MAX(timestamp) as last_ts FROM program_results"
        ).fetchone()
        if row and row["last_ts"]:
            hours_since = (now - row["last_ts"]) / 3600
            threshold_hours = thresholds.get("stale_experiment_hours", 6)
            if hours_since > threshold_hours:
                alerts.append({
                    "id": "stale_pipeline",
                    "severity": "info",
                    "title": "Pipeline idle",
                    "message": f"No new results in {hours_since:.1f} hours",
                    "value": round(hours_since, 1),
                    "threshold": threshold_hours,
                    "timestamp": now,
                })
    except Exception:
        pass

    nb.close()
    alerts.sort(key=lambda a: {"critical": 0, "warning": 1, "info": 2}[a["severity"]])
    return alerts


def register_observability_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/observability/health")
    def api_component_health():
        """Component health grid — all ops with status/metrics."""
        try:
            health = _get_component_health(notebook_path)
            return jsonify(health)
        except Exception as e:
            logger.error("Error in /api/observability/health: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/observability/health/refresh", methods=["POST"])
    def api_component_health_refresh():
        """Force-refresh component health cache."""
        global _health_cache_ts
        _health_cache_ts = 0.0
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

                    prog_dict = progress.to_dict() if hasattr(progress, "to_dict") else {}
                    current_step = prog_dict.get("current_program", 0)

                    if current_step != last_step:
                        last_step = current_step
                        # Include live loss curve tail (last 20 points)
                        try:
                            curve = runner.get_live_loss_curve()
                            if curve:
                                prog_dict["loss_curve_tail"] = curve[-20:]
                        except Exception:
                            pass
                        data = _json_dumps(prog_dict, safe=True)
                        yield f"event: progress\ndata: {data}\n\n"

                    # Check for alerts on each tick
                    try:
                        alerts = _evaluate_alerts(notebook_path, _DEFAULT_THRESHOLDS)
                        if alerts:
                            alert_data = _json_dumps({"alerts": alerts}, safe=True)
                            yield f"event: alerts\ndata: {alert_data}\n\n"
                    except Exception:
                        pass

                    time.sleep(3)
                except GeneratorExit:
                    return
                except Exception:
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
    def api_failure_blocklist():
        """Op-pair failure signatures that should be auto-disabled."""
        nb = LabNotebook(notebook_path)
        try:
            blocklist = nb.get_failure_signature_blocklist(
                min_seen=int(request.args.get("min_seen", 10)),
                max_fail_rate=float(request.args.get("max_fail_rate", 0.90)),
            )
            return jsonify({"blocklist": blocklist, "count": len(blocklist)})
        except Exception as e:
            logger.error("Error in /api/observability/failure-blocklist: %s", e)
            return jsonify({"blocklist": {}, "error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/observability/monitor")
    def api_monitor():
        """Compact CLI monitoring endpoint — single JSON with all key telemetry.

        Designed for: `curl -s .../api/observability/monitor | jq .`
        or `watch -n 3 'curl -s .../api/observability/monitor | jq .'`
        """
        result: Dict[str, Any] = {"ts": time.time()}

        # 1. Runner progress
        runner = get_runner(notebook_path)
        try:
            progress = runner.progress
            if progress is not None:
                p = progress.to_dict() if hasattr(progress, "to_dict") else {}
                result["run"] = {
                    "status": p.get("status", "idle"),
                    "program": f"{p.get('current_program', 0)}/{p.get('total_programs', '?')}",
                    "s0": p.get("stage0_passed", 0),
                    "s1": p.get("stage1_passed", 0),
                    "best_lr": p.get("best_loss_ratio"),
                    "elapsed_m": round(p.get("elapsed_seconds", 0) / 60, 1),
                    "stage": p.get("current_stage", ""),
                }
            else:
                result["run"] = {"status": "idle"}
        except Exception:
            result["run"] = {"status": "unknown"}

        # 2. Live training step (most recent from loss curve)
        try:
            curve = runner.get_live_loss_curve()
            if curve:
                last = curve[-1]
                result["train"] = {
                    "step": last.get("step"),
                    "loss": last.get("loss"),
                    "total_steps": last.get("total_steps"),
                    "routing_aux_loss": last.get("routing_aux_loss"),
                    "grad_norm": last.get("grad_norm"),
                    "phase": last.get("phase", ""),
                }
        except Exception:
            pass

        # 3. Alerts (compact)
        try:
            alerts = _evaluate_alerts(notebook_path, _DEFAULT_THRESHOLDS)
            if alerts:
                result["alerts"] = [
                    {"severity": a["severity"][0].upper(), "msg": a["title"]}
                    for a in alerts
                ]
            else:
                result["alerts"] = []
        except Exception:
            result["alerts"] = []

        # 4. Component health (counts only)
        try:
            health = _get_component_health(notebook_path)
            result["components"] = {
                "total": health["total"],
                "ok": health["healthy"],
                "warn": health["degraded"],
                "fail": health["broken"],
            }
        except Exception:
            pass

        # 5. Recent routing telemetry from last S1 result
        nb = LabNotebook(notebook_path)
        try:
            row = nb.conn.execute(
                "SELECT routing_mode, routing_confidence_mean, routing_drop_rate, "
                "routing_utilization_entropy, routing_aux_loss_mean, routing_tokens_total "
                "FROM program_results "
                "WHERE stage1_passed = 1 AND routing_mode IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row and row["routing_mode"]:
                result["routing"] = {
                    "mode": row["routing_mode"],
                    "confidence": round(float(row["routing_confidence_mean"]), 3) if row["routing_confidence_mean"] else None,
                    "drop_rate": round(float(row["routing_drop_rate"]), 3) if row["routing_drop_rate"] else None,
                    "entropy": round(float(row["routing_utilization_entropy"]), 3) if row["routing_utilization_entropy"] else None,
                    "aux_loss": round(float(row["routing_aux_loss_mean"]), 4) if row["routing_aux_loss_mean"] else None,
                }
        except Exception:
            pass
        finally:
            nb.close()

        # Support text format for plain CLI: ?format=text
        fmt = request.args.get("format", "json")
        if fmt == "text":
            lines = []
            r = result.get("run", {})
            lines.append(f"status={r.get('status','?')}  prog={r.get('program','?')}  s0={r.get('s0',0)}  s1={r.get('s1',0)}  best_lr={r.get('best_lr','?')}  elapsed={r.get('elapsed_m',0)}m")
            t = result.get("train", {})
            if t:
                parts = [f"step={t.get('step','?')}/{t.get('total_steps','?')}  loss={t.get('loss','?')}"]
                if t.get("routing_aux_loss") is not None:
                    parts.append(f"raux={t['routing_aux_loss']}")
                if t.get("grad_norm") is not None:
                    parts.append(f"gnorm={t['grad_norm']}")
                lines.append("  ".join(parts))
            c = result.get("components", {})
            if c:
                lines.append(f"components: {c.get('ok',0)} ok / {c.get('warn',0)} warn / {c.get('fail',0)} fail")
            rt = result.get("routing", {})
            if rt:
                parts = [f"routing={rt.get('mode','?')}"]
                if rt.get("confidence") is not None:
                    parts.append(f"conf={rt['confidence']}")
                if rt.get("drop_rate") is not None:
                    parts.append(f"drop={rt['drop_rate']}")
                if rt.get("entropy") is not None:
                    parts.append(f"ent={rt['entropy']}")
                lines.append("  ".join(parts))
            al = result.get("alerts", [])
            if al:
                lines.append(f"ALERTS: {', '.join(a['msg'] for a in al)}")
            return Response("\n".join(lines) + "\n", mimetype="text/plain")

        return jsonify(result)
