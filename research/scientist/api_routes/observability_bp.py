"""Observability API routes — component health, alerts, training SSE stream,
error log, experiment lifecycle, throughput, op analytics, resource utilization,
grammar evolution, failure patterns, leaderboard dynamics, insight effectiveness,
DB health, and API health."""

from __future__ import annotations

import json as _json
import logging
import os
import sqlite3
import statistics
import threading
import time
from collections import defaultdict
from typing import Any, Dict
from flask import Response, jsonify, request
from ..json_utils import fast_dumps as _json_dumps
from ..trust_policy import sql_trusted_clause
from ._api_health import API_HEALTH_COUNTERS, API_HEALTH_LOCK
from ._helpers import get_runner
from ._observability_core import (
    _DEFAULT_THRESHOLDS,
    _WINDOW_SECONDS,
    build_op_index,
    get_cached_alerts,
    get_component_health,
    get_throughput,
    refresh_observability_caches,
)
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)
_ALERT_WORKERS: dict[str, threading.Thread] = {}
_LATEST_ALERTS: dict[str, list[dict[str, Any]]] = {}
_ALERTS_LOCK = threading.Lock()
_DB_HEALTH_TABLES = (
    "program_results",
    "experiments",
    "leaderboard",
    "learning_log",
    "insights",
    "training_curves",
    "entries",
)
_INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES = (
    "exact_graph_replay",
    "reference",
    "reference_registration",
    "validation",
    "backfill",
)


def _ensure_alert_worker(notebook_path: str) -> None:
    with _ALERTS_LOCK:
        existing = _ALERT_WORKERS.get(notebook_path)
        if existing is not None and existing.is_alive():
            return

        def _loop() -> None:
            while True:
                try:
                    alerts = get_cached_alerts(notebook_path, _DEFAULT_THRESHOLDS)
                    with _ALERTS_LOCK:
                        _LATEST_ALERTS[notebook_path] = alerts
                except Exception as exc:
                    logger.debug("Shared alert worker failed: %s", exc, exc_info=True)
                time.sleep(3)

        worker = threading.Thread(
            target=_loop,
            name=f"observability-alerts:{notebook_path}",
            daemon=True,
        )
        _ALERT_WORKERS[notebook_path] = worker
        worker.start()


def _latest_alerts_snapshot(notebook_path: str) -> list[dict[str, Any]]:
    with _ALERTS_LOCK:
        cached = _LATEST_ALERTS.get(notebook_path)
    if cached is not None:
        return cached
    alerts = get_cached_alerts(notebook_path, _DEFAULT_THRESHOLDS)
    with _ALERTS_LOCK:
        _LATEST_ALERTS[notebook_path] = alerts
    return alerts


def _fetch_db_health_row_counts(nb) -> dict[str, Any]:
    query = " UNION ALL ".join(
        f"SELECT '{table}' AS table_name, COUNT(*) AS c FROM {table}"
        for table in _DB_HEALTH_TABLES
    )
    try:
        rows = nb.conn.execute(query).fetchall()
    except sqlite3.OperationalError:
        row_counts: dict[str, Any] = {}
        for table in _DB_HEALTH_TABLES:
            try:
                row = nb.conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
                row_counts[table] = row["c"] if row else 0
            except sqlite3.OperationalError:
                row_counts[table] = None
        return row_counts
    row_counts = {table: None for table in _DB_HEALTH_TABLES}
    for row in rows:
        row_counts[row["table_name"]] = row["c"]
    return row_counts


def _register_health_routes(app, notebook_path: str, wnb) -> None:
    """P0: health, alerts, SSE stream, error-log, experiment-lifecycle."""

    @app.route("/api/observability/health")
    def api_component_health():
        """Component health grid — all ops with status/metrics."""
        try:
            window = request.args.get("window", "all")
            if window not in _WINDOW_SECONDS:
                window = "all"
            health = get_component_health(notebook_path, window=window)
            return jsonify(health)
        except Exception as e:
            logger.error("Error in /api/observability/health: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/observability/health/refresh", methods=["POST"])
    def api_component_health_refresh():
        """Force-refresh component health + OpIndex caches."""
        refresh_observability_caches()
        health = get_component_health(notebook_path)
        return jsonify(health)

    @app.route("/api/observability/alerts")
    def api_alerts():
        """Active alerts based on threshold evaluation."""
        try:
            alerts = get_cached_alerts(notebook_path, _DEFAULT_THRESHOLDS)
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
        _ensure_alert_worker(notebook_path)

        def event_stream():
            last_step = -1
            last_alert_signature = None
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

                    try:
                        alerts = _latest_alerts_snapshot(notebook_path)
                        alert_signature = _json_dumps({"alerts": alerts}, safe=True)
                        if alerts and alert_signature != last_alert_signature:
                            last_alert_signature = alert_signature
                            yield f"event: alerts\ndata: {alert_signature}\n\n"
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
                FROM program_results_compat
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


def _register_analytics_routes(app, notebook_path: str) -> None:
    """P1: throughput, op-pairs, loss-dist, resource-util, api-health."""

    @app.route("/api/observability/throughput")
    def api_throughput():
        """Program evaluation throughput by time window."""
        try:
            data = get_throughput(notebook_path)
            return jsonify(data)
        except Exception as e:
            logger.error("Error in /api/observability/throughput: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/observability/op-pairs")
    def api_op_pairs():
        """Top op pairs by co-occurrence with s0/s1 rates."""
        top_n = int(request.args.get("top", 30))
        try:
            idx = build_op_index(notebook_path)
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

    @app.route("/api/observability/loss-distribution")
    def api_loss_distribution():
        """Per-op loss ratio distribution (box plot data)."""
        try:
            idx = build_op_index(notebook_path)
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

    @app.route("/api/observability/api-health")
    def api_api_health():
        """API request counters by endpoint x status bucket."""
        with API_HEALTH_LOCK:
            snapshot = dict(API_HEALTH_COUNTERS)
        return jsonify({"counters": snapshot})


def _register_patterns_routes(app, notebook_path: str, wnb) -> None:
    """P2: grammar-evolution, failure-patterns, leaderboard-dynamics, insight-effectiveness."""

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

    @app.route("/api/observability/failure-patterns")
    def api_obs_failure_patterns():
        """Failed graphs grouped by error_type with top co-occurring ops."""
        top_ops = int(request.args.get("top_ops", 5))
        try:
            idx = build_op_index(notebook_path)
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

    @app.route("/api/observability/leaderboard-dynamics")
    @wnb
    def api_leaderboard_dynamics(nb=None):
        """Tier counts per day + recent promotions."""
        trusted_only = str(request.args.get("trusted_only", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        where = ""
        params = []
        if trusted_only:
            where = f" WHERE {sql_trusted_clause()}"
        # Daily tier counts
        rows = nb.conn.execute(
            "SELECT date(timestamp, 'unixepoch') as day, tier, COUNT(*) as cnt "
            f"FROM leaderboard{where} GROUP BY day, tier ORDER BY day",
            params,
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
            f"FROM leaderboard{where} ORDER BY timestamp DESC LIMIT 20",
            params,
        ).fetchall()

        return jsonify(
            {
                "daily": {d: dict(tiers) for d, tiers in sorted(daily.items())},
                "recent_promotions": [dict(r) for r in promos],
                "trusted_only": trusted_only,
            }
        )

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


def _register_db_health_routes(app, notebook_path: str, wnb) -> None:
    """P3: db-health."""

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
            result["row_counts"] = _fetch_db_health_row_counts(nb)
        except (sqlite3.OperationalError, KeyError, TypeError) as e:
            logger.debug("DB health row count query failed: %s", e)
            result["row_counts"] = {}
            result["error"] = str(e)

        try:
            result["entity_counts"] = nb.get_data_accounting_summary()
        except (sqlite3.OperationalError, KeyError, TypeError) as e:
            logger.debug("DB health entity count query failed: %s", e)
            result["entity_counts"] = {}
            result.setdefault("error", str(e))

        return jsonify(result)


def _duplicate_fingerprint_limit() -> int:
    limit = int(request.args.get("limit", 50))
    return max(1, min(limit, 500))


def _load_within_experiment_duplicate_fingerprints(nb, limit: int):
    return nb.conn.execute(
        """
        SELECT
            pr.graph_fingerprint AS fp,
            pr.experiment_id,
            COUNT(*) AS n_runs,
            MIN(pr.timestamp) AS first_ts,
            MAX(pr.timestamp) AS last_ts,
            MAX(e.experiment_type) AS experiment_type,
            MAX(pr.model_source) AS model_source,
            MAX(pr.result_cohort) AS result_cohort,
            SUM(CASE WHEN pr.stage1_passed = 1 THEN 1 ELSE 0 END) AS n_passed
        FROM program_results_compat pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
        GROUP BY pr.graph_fingerprint, pr.experiment_id
        HAVING n_runs > 1
        ORDER BY n_runs DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _load_cross_experiment_duplicate_fingerprints(nb, limit: int):
    placeholders = ",".join("?" for _ in _INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES)
    return nb.conn.execute(
        f"""
        SELECT
            pr.graph_fingerprint AS fp,
            COUNT(DISTINCT pr.experiment_id) AS n_experiments,
            COUNT(*) AS n_rows,
            MIN(pr.timestamp) AS first_ts,
            MAX(pr.timestamp) AS last_ts,
            GROUP_CONCAT(DISTINCT e.experiment_type) AS experiment_types
        FROM program_results_compat pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
          AND e.experiment_type NOT IN ({placeholders})
        GROUP BY pr.graph_fingerprint
        HAVING n_experiments > 1
        ORDER BY n_experiments DESC, n_rows DESC
        LIMIT ?
        """,
        (*_INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES, limit),
    ).fetchall()


def _within_duplicate_summary(nb):
    return nb.conn.execute(
        """
        SELECT COUNT(*) AS dup_groups, SUM(n_runs - 1) AS excess_rows
        FROM (
          SELECT COUNT(*) AS n_runs
          FROM program_results_compat
          WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
          GROUP BY graph_fingerprint, experiment_id
          HAVING n_runs > 1
        )
        """
    ).fetchone()


def _cross_duplicate_summary(nb):
    placeholders = ",".join("?" for _ in _INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES)
    return nb.conn.execute(
        f"""
        SELECT COUNT(*) AS fingerprints, SUM(n - 1) AS excess_experiments
        FROM (
          SELECT pr.graph_fingerprint, COUNT(DISTINCT pr.experiment_id) AS n
          FROM program_results_compat pr
          JOIN experiments e ON e.experiment_id = pr.experiment_id
          WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
            AND e.experiment_type NOT IN ({placeholders})
          GROUP BY pr.graph_fingerprint
          HAVING n > 1
        )
        """,
        _INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES,
    ).fetchone()


def _duplicate_fingerprint_guards(nb) -> dict[str, bool]:
    return {
        "idx_pr_fp_per_experiment": bool(
            nb.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_pr_fp_per_experiment'"
            ).fetchone()
        ),
        "reject_dup_fingerprint_no_reason": bool(
            nb.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='reject_dup_fingerprint_no_reason'"
            ).fetchone()
        ),
    }


def _duplicate_fingerprint_rejection_count() -> int:
    from research.scientist.notebook import LabNotebook

    return int(getattr(LabNotebook, "_dup_rejection_count", 0) or 0)


def _duplicate_fingerprints_payload(nb, limit: int) -> dict[str, Any]:
    within = _load_within_experiment_duplicate_fingerprints(nb, limit)
    cross = _load_cross_experiment_duplicate_fingerprints(nb, limit)
    agg_within = _within_duplicate_summary(nb)
    agg_cross = _cross_duplicate_summary(nb)
    return {
        "within_experiment": {
            "summary": {
                "duplicate_groups": agg_within["dup_groups"] or 0,
                "excess_rows": agg_within["excess_rows"] or 0,
            },
            "groups": [dict(row) for row in within],
        },
        "cross_experiment_unintentional": {
            "summary": {
                "fingerprints": agg_cross["fingerprints"] or 0,
                "excess_experiments": agg_cross["excess_experiments"] or 0,
            },
            "intentional_types_excluded": list(
                _INTENTIONAL_DUP_FINGERPRINT_EXPERIMENT_TYPES
            ),
            "fingerprints": [dict(row) for row in cross],
        },
        "rejection_counter": {
            "since_dashboard_boot": _duplicate_fingerprint_rejection_count(),
            "guards_installed": _duplicate_fingerprint_guards(nb),
        },
    }


def _register_governance_routes(app, notebook_path: str, wnb) -> None:
    """Slice 3c: surface fingerprint dedup violations for the dashboard."""

    @app.route("/api/governance/duplicate-fingerprints")
    @wnb
    def api_governance_dup_fps(nb=None):
        """Top fingerprints with multiple rows.

        Buckets:
          - within_experiment: SAME (fingerprint, experiment_id) appears > 1 row.
            These are unconditional violations — the dedup gate should have
            caught them.
          - cross_experiment_unintentional: same fp across multiple
            experiments where the experiment_type is NOT in INTENTIONAL_TYPES
            (i.e., evolution / novelty / synthesis re-evaluating something
            the system has already seen).
        """
        return jsonify(
            _duplicate_fingerprints_payload(nb, _duplicate_fingerprint_limit())
        )


def register_observability_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    _register_health_routes(app, notebook_path, wnb)
    _register_analytics_routes(app, notebook_path)
    _register_patterns_routes(app, notebook_path, wnb)
    _register_db_health_routes(app, notebook_path, wnb)
    _register_governance_routes(app, notebook_path, wnb)
