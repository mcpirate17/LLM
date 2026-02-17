"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from .notebook import LabNotebook
from .persona import get_aria
from .runner import ExperimentRunner, RunConfig
from .llm.context import build_program_context

logger = logging.getLogger(__name__)

# Singleton runner shared across requests
_runner: Optional[ExperimentRunner] = None


def _insight_dedup_key(content: str) -> str:
    """Normalize numeric values to create a stable dedup key for insights.

    Replaces decimals/percentages and multi-digit integers so that
    'appears in 144 survivors' matches 'appears in 145 survivors'.
    Preserves single-digit suffixes in op names like 'split2'.
    """
    import re
    s = re.sub(r'\d+\.\d+%?', '#', content)   # decimals / pcts
    s = re.sub(r'\b\d{2,}\b', '#', s)           # multi-digit ints
    return s


def _deduplicate_insights(insights: list) -> list:
    """Keep only the most recent insight per semantic dedup key."""
    seen: dict = {}
    for ins in insights:
        key = _insight_dedup_key(ins.get("content", ""))
        if key not in seen:
            seen[key] = ins
    return list(seen.values())


def _normalize_entry(entry: dict) -> dict:
    """Normalize notebook entry shape for UI consumers.

    Ensures ``metadata`` is available as a parsed dict while preserving
    original ``metadata_json`` for compatibility.
    """
    normalized = dict(entry)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        return normalized

    raw = normalized.get("metadata_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            normalized["metadata"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            normalized["metadata"] = {}
    else:
        normalized["metadata"] = {}
    return normalized


def _normalize_entries(entries: list) -> list:
    return [_normalize_entry(entry) for entry in entries]


def _entry_to_live_feed_event(entry: dict) -> Optional[dict]:
    """Convert a persisted notebook live-feed entry into UI event shape."""
    normalized = _normalize_entry(entry)
    metadata = normalized.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None

    live_type = metadata.get("live_feed_type")
    payload = metadata.get("payload")
    if not live_type or not isinstance(payload, dict):
        return None

    event = {"type": live_type, **payload}
    ts = normalized.get("timestamp")
    if isinstance(ts, (int, float)):
        event["ts"] = int(ts * 1000)
    return event


def _normalize_hypothesis(hypothesis: dict) -> dict:
    """Normalize campaign hypothesis shape for UI consumers."""
    normalized = dict(hypothesis)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        return normalized

    raw = normalized.get("metadata_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            normalized["metadata"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            normalized["metadata"] = {}
    else:
        normalized["metadata"] = {}
    return normalized


def _normalize_hypotheses(hypotheses: list) -> list:
    return [_normalize_hypothesis(hypothesis) for hypothesis in hypotheses]


def _rank_label(delta: Optional[int], seen_runs: int) -> str:
    if seen_runs <= 1:
        return "new"
    if delta is None:
        return "unknown"
    if delta <= -2:
        return "up"
    if delta >= 2:
        return "down"
    return "stable"


def _compute_cross_run_stability(nb: LabNotebook, top_programs: list) -> dict:
    """Compute rank movement for top candidates across recent experiments.

    Uses graph fingerprint as the architecture key and tracks its rank
    among stage-1-passing programs for each completed experiment.
    """
    experiments = [
        exp for exp in nb.get_recent_experiments(40)
        if exp.get("status") == "completed"
    ]
    if not top_programs or not experiments:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": len(experiments),
        }

    fingerprint_ranks_by_experiment: dict[str, dict[str, int]] = {}
    for exp in experiments:
        experiment_id = exp.get("experiment_id")
        if not experiment_id:
            continue
        programs = nb.get_program_results(experiment_id)
        ranked = sorted(
            [
                p for p in programs
                if p.get("stage1_passed") and p.get("loss_ratio") is not None
            ],
            key=lambda p: p.get("loss_ratio", float("inf")),
        )
        ranks = {}
        for idx, program in enumerate(ranked, start=1):
            fp = program.get("graph_fingerprint")
            if fp and fp not in ranks:
                ranks[fp] = idx
        fingerprint_ranks_by_experiment[experiment_id] = ranks

    candidates = []
    summary = {"stable": 0, "up": 0, "down": 0, "new": 0}
    for index, program in enumerate(top_programs[:20], start=1):
        fp = program.get("graph_fingerprint")
        if not fp:
            continue

        history = []
        for exp in experiments:
            experiment_id = exp.get("experiment_id")
            if not experiment_id:
                continue
            rank = fingerprint_ranks_by_experiment.get(experiment_id, {}).get(fp)
            if rank is None:
                continue
            history.append({
                "experiment_id": experiment_id,
                "timestamp": exp.get("timestamp"),
                "rank": rank,
            })

        seen_runs = len(history)
        latest_rank = history[0]["rank"] if history else None
        previous_rank = history[1]["rank"] if len(history) > 1 else None
        delta = None
        if latest_rank is not None and previous_rank is not None:
            delta = latest_rank - previous_rank
        trend = _rank_label(delta, seen_runs)
        summary[trend] = summary.get(trend, 0) + 1

        candidates.append({
            "result_id": program.get("result_id"),
            "graph_fingerprint": fp,
            "current_overall_rank": index,
            "seen_runs": seen_runs,
            "latest_rank": latest_rank,
            "previous_rank": previous_rank,
            "rank_delta": delta,
            "trend": trend,
        })

    return {
        "summary": summary,
        "candidates": candidates,
        "window_size": len(experiments),
    }


def _get_sse_timeout_seconds() -> float:
    """Get SSE stream polling timeout from env with safe fallback."""
    raw = os.environ.get("ARIA_SSE_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    if timeout <= 0:
        logger.warning("Non-positive ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    return timeout


def _get_runner(notebook_path: str) -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(notebook_path)
    return _runner


def _compute_recommendation(program: dict, leaderboard_entry: Optional[dict]) -> dict:
    """Deterministic next-action recommendation based on tier and pass/fail."""
    tier = (leaderboard_entry or {}).get("tier", "screening")
    s1 = program.get("stage1_passed", False)

    if not s1:
        return {
            "action": "archive",
            "rationale": "Program did not pass Stage 1 learning evaluation.",
            "confidence": "high",
        }

    if tier == "breakthrough":
        return {
            "action": "publish",
            "rationale": "Breakthrough-tier architecture with validated performance.",
            "confidence": "high",
        }

    if tier == "validation":
        passed = (leaderboard_entry or {}).get("validation_passed", False)
        if passed:
            return {
                "action": "scale up or publish",
                "rationale": "Validation passed with multi-seed stability confirmed.",
                "confidence": "high",
            }
        return {
            "action": "re-validate",
            "rationale": "Validation tier but not yet passed; may need more seeds or longer training.",
            "confidence": "medium",
        }

    if tier == "investigation":
        passed = (leaderboard_entry or {}).get("investigation_passed", False)
        if passed:
            return {
                "action": "validate",
                "rationale": "Investigation passed; promote to validation for multi-seed confirmation.",
                "confidence": "high",
            }
        return {
            "action": "re-investigate or archive",
            "rationale": "Investigation tier but not yet passed; re-run or archive if stale.",
            "confidence": "medium",
        }

    # screening (default)
    return {
        "action": "investigate",
        "rationale": "Screening-tier candidate; needs deeper investigation to confirm potential.",
        "confidence": "medium",
    }


def _annotate_qkv_usage(programs: list, analytics) -> None:
    for program in programs:
        if not isinstance(program, dict):
            continue
        qkv_usage = analytics.qkv_usage_enum(program)
        program["qkv_usage"] = qkv_usage
        program["uses_qkv"] = qkv_usage != "qkv_free"
        program["compression_metrics"] = analytics.canonical_compression_metrics(program)
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)


def _normalize_result_ids(raw_ids: Any) -> List[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        if value is None:
            continue
        result_id = str(value).strip()
        if not result_id or result_id in seen:
            continue
        seen.add(result_id)
        normalized.append(result_id)
    return normalized


def _build_start_mode_eligibility(
    nb: LabNotebook,
    mode: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """Validate candidate progression eligibility for start modes.

    Returns a structured payload containing per-candidate reasons.
    """
    payload: Dict[str, Any] = {
        "mode": mode,
        "requested_result_ids": list(result_ids),
        "eligible_result_ids": [],
        "ineligible": [],
        "all_eligible": False,
    }
    if not result_ids:
        return payload

    placeholders = ",".join("?" for _ in result_ids)
    leaderboard_rows = nb.conn.execute(
        f"""
        SELECT result_id, tier, investigation_passed, validation_passed,
               investigation_loss_ratio, validation_loss_ratio
        FROM leaderboard
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()
    program_rows = nb.conn.execute(
        f"""
        SELECT result_id, stage1_passed
        FROM program_results
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()

    leaderboard_by_id = {row["result_id"]: dict(row) for row in leaderboard_rows}
    program_by_id = {row["result_id"]: dict(row) for row in program_rows}

    for result_id in result_ids:
        lb = leaderboard_by_id.get(result_id)
        program = program_by_id.get(result_id)

        if lb is None:
            if program is None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "result_not_found",
                    "detail": "Result ID was not found in program results.",
                })
            elif not bool(program.get("stage1_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_stage1_survivor",
                    "detail": "Result exists but is not a Stage-1 survivor.",
                })
            else:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_in_leaderboard",
                    "detail": "Result exists but has no leaderboard progression record.",
                })
            continue

        tier = str(lb.get("tier") or "").lower()

        if mode == "investigation":
            if tier != "screening":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_screening_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; only screening tier can be investigated.",
                    "tier": tier or None,
                })
                continue
            if lb.get("investigation_loss_ratio") is not None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "already_investigated_unchanged",
                    "detail": "Candidate already has investigation evidence; provide a changed-condition trigger before re-investigating.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        if mode == "validation":
            if tier != "investigation":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; validation requires investigation tier.",
                    "tier": tier or None,
                })
                continue
            if not bool(lb.get("investigation_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_passed",
                    "detail": "Investigation evidence did not pass robustness gate.",
                    "tier": tier,
                })
                continue
            if bool(lb.get("validation_passed")) or lb.get("validation_loss_ratio") is not None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "already_validated",
                    "detail": "Candidate already has validation evidence.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        payload["ineligible"].append({
            "result_id": result_id,
            "reason": "unsupported_mode",
            "detail": f"Eligibility checks are not implemented for mode '{mode}'.",
        })

    payload["all_eligible"] = len(payload["ineligible"]) == 0 and len(payload["eligible_result_ids"]) > 0
    payload["summary"] = {
        "requested": len(result_ids),
        "eligible": len(payload["eligible_result_ids"]),
        "ineligible": len(payload["ineligible"]),
    }
    return payload


def _build_report_action_eligibility(
    nb: LabNotebook,
    result_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-result report action eligibility aligned with start guardrails."""
    normalized_ids = _normalize_result_ids(result_ids)
    if not normalized_ids:
        return {}

    inv = _build_start_mode_eligibility(nb, "investigation", normalized_ids)
    val = _build_start_mode_eligibility(nb, "validation", normalized_ids)

    inv_eligible = set(inv.get("eligible_result_ids") or [])
    val_eligible = set(val.get("eligible_result_ids") or [])
    inv_reason = {
        row.get("result_id"): row.get("reason")
        for row in (inv.get("ineligible") or [])
        if row.get("result_id")
    }
    val_reason = {
        row.get("result_id"): row.get("reason")
        for row in (val.get("ineligible") or [])
        if row.get("result_id")
    }

    eligibility_by_id: Dict[str, Dict[str, Any]] = {}
    for result_id in normalized_ids:
        investigation_eligible = result_id in inv_eligible
        validation_eligible = result_id in val_eligible
        queue_eligible = investigation_eligible or validation_eligible
        queue_reason = None
        if not queue_eligible:
            queue_reason = inv_reason.get(result_id) or val_reason.get(result_id) or "not_progression_eligible"

        eligibility_by_id[result_id] = {
            "investigationEligible": investigation_eligible,
            "validationEligible": validation_eligible,
            "queueEligible": queue_eligible,
            "queueReason": queue_reason,
            "investigationReason": inv_reason.get(result_id),
            "validationReason": val_reason.get(result_id),
        }

    return eligibility_by_id


def _llm_config_path(notebook_path: str) -> Path:
    """Path for persisted LLM configuration, next to the notebook DB."""
    return Path(notebook_path).parent / "llm_config.json"


def _load_persisted_llm_config(notebook_path: str):
    """Auto-load LLM config from disk if present."""
    config_path = _llm_config_path(notebook_path)
    if not config_path.exists():
        return
    try:
        import json as _json
        data = _json.loads(config_path.read_text())
        backend = str(data.get("backend", "")).strip()
        if not backend:
            return
        aria = get_aria()
        aria.configure_llm(
            backend_name=backend,
            api_key=str(data.get("api_key", "")).strip(),
            model=str(data.get("model", "")).strip(),
            host=str(data.get("host", "")).strip(),
        )
        logger.info(f"Loaded persisted LLM config: {backend}")
    except Exception as e:
        logger.warning(f"Failed to load persisted LLM config: {e}")


def _save_llm_config(notebook_path: str, config: Dict):
    """Persist LLM config to disk so it survives restarts."""
    config_path = _llm_config_path(notebook_path)
    try:
        import json as _json
        config_path.write_text(_json.dumps(config, indent=2))
        logger.info(f"Saved LLM config to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save LLM config: {e}")


def create_app(
    notebook_path: str = "research/lab_notebook.db",
    static_folder: Optional[str] = None,
) -> Flask:
    """Create the Flask API app."""

    if static_folder is None:
        static_folder = str(Path(__file__).parent.parent / "dashboard" / "build")

    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    CORS(app)

    # Auto-load persisted LLM config
    _load_persisted_llm_config(notebook_path)

    # ── Global error handlers ──

    @app.errorhandler(404)
    def not_found(e):
        # Only return JSON for API routes; let static files 404 naturally
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(app.static_folder, "index.html")

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"500 error on {request.method} {request.path}: {e}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.error(f"Unhandled exception on {request.method} {request.path}: "
                     f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    @app.after_request
    def log_response(response):
        if request.path.startswith("/api/") and response.status_code >= 400:
            logger.warning(f"{request.method} {request.path} -> {response.status_code}")
        return response

    # ── Dashboard routes ──

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(app.static_folder, path)

    # ── Read-only API routes ──

    @app.route("/api/status")
    def api_status():
        """Get Aria's current status and dashboard summary."""
        nb = LabNotebook(notebook_path)
        runner = _get_runner(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()
            return jsonify({
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            })
        except Exception as e:
            logger.error(f"Error in /api/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments")
    def api_experiments():
        """List recent experiments."""
        n = request.args.get("n", 20, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n))
        except Exception as e:
            logger.error(f"Error in /api/experiments: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>")
    def api_experiment_detail(experiment_id):
        """Get experiment details with entries and per-experiment programs."""
        nb = LabNotebook(notebook_path)
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404
            entries = nb.get_entries(experiment_id=experiment_id)
            programs = nb.get_program_results(experiment_id)
            return jsonify({
                "experiment": exp,
                "entries": entries,
                "programs": programs,
            })
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/programs")
    def api_experiment_programs(experiment_id):
        """All programs for an experiment (not just S1 survivors)."""
        nb = LabNotebook(notebook_path)
        try:
            programs = nb.get_program_results(experiment_id)
            return jsonify(programs)
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>")
    def api_program_detail(result_id):
        """Full program detail with parsed graph JSON + fingerprint + all metrics."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            # Include training curve availability flag
            try:
                curve = nb.get_training_curve(result_id)
                program["has_training_curve"] = len(curve) > 0
            except Exception:
                program["has_training_curve"] = False

            # Try LLM explanation of fingerprint (non-critical)
            try:
                ctx = build_program_context(program)
                explanation = aria.explain_fingerprint(ctx)
                if explanation:
                    program["llm_explanation"] = explanation
            except Exception as e:
                logger.debug(f"LLM fingerprint explanation failed for {result_id}: {e}")

            try:
                from .analytics import ExperimentAnalytics
                analytics = ExperimentAnalytics(nb)
                qkv_usage = analytics.qkv_usage_enum(program)
                program["qkv_usage"] = qkv_usage
                program["uses_qkv"] = qkv_usage != "qkv_free"
                program["compression_metrics"] = analytics.canonical_compression_metrics(program)
                program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)
            except Exception as e:
                logger.debug("QKV usage classification failed for %s: %s", result_id, e)

            return jsonify(program)
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/failures")
    def api_failure_analysis(experiment_id):
        """Failure analysis: error distribution, stage funnel."""
        nb = LabNotebook(notebook_path)
        try:
            analysis = nb.get_failure_analysis(experiment_id)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/analysis")
    def api_experiment_analysis(experiment_id):
        """LLM-generated analysis (stored or on-demand)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404

            # Return stored analysis if available
            stored = exp.get("llm_analysis")
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

            # Try generating on-demand
            results = exp.get("results") or {}
            from .llm.context import build_experiment_context
            ctx = build_experiment_context(results)
            analysis = aria.analyze_results(results, context=ctx)

            if analysis:
                # Cache it
                try:
                    nb.conn.execute(
                        "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                        (analysis, experiment_id),
                    )
                    nb.conn.commit()
                except Exception as e:
                    logger.warning("Failed caching llm_analysis for %s: %s",
                                   experiment_id, e)
                return jsonify({"analysis": analysis, "source": "generated"})

            return jsonify({"analysis": None, "source": "unavailable",
                            "reason": "No LLM backend configured"})
        except Exception as e:
            logger.error(f"Error in experiment analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
        except Exception as e:
            logger.error(f"Error in /api/trends: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends/context")
    def api_trends_context():
        """Trend data plus adaptation-event deltas for inline linkage UI."""
        nb = LabNotebook(notebook_path)

        def _event_delta_payload(trends: List[Dict[str, Any]], event: Dict[str, Any]) -> Dict[str, Any]:
            timestamp = float(event.get("timestamp") or 0.0)
            previous = [row for row in trends if float(row.get("timestamp") or 0.0) < timestamp]
            following = [row for row in trends if float(row.get("timestamp") or 0.0) >= timestamp]

            before = previous[-3:]
            after = following[:3]

            before_ids = [str(row.get("experiment_id")) for row in before if row.get("experiment_id")]
            after_ids = [str(row.get("experiment_id")) for row in after if row.get("experiment_id")]

            def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if not values:
                    return None
                return sum(values) / len(values)

            before_adj_s1 = _avg(before, "adjusted_s1_pass_rate")
            after_adj_s1 = _avg(after, "adjusted_s1_pass_rate")
            before_novelty = _avg(before, "best_novelty_score")
            after_novelty = _avg(after, "best_novelty_score")
            before_loss = _avg(before, "best_loss_ratio")
            after_loss = _avg(after, "best_loss_ratio")

            return {
                "timestamp": timestamp,
                "event_type": event.get("event_type"),
                "description": event.get("description") or "Grammar weights adjusted",
                "before_window": {
                    "n_experiments": len(before),
                    "experiment_ids": before_ids,
                    "adjusted_s1_rate": before_adj_s1,
                    "best_novelty": before_novelty,
                    "best_loss_ratio": before_loss,
                },
                "after_window": {
                    "n_experiments": len(after),
                    "experiment_ids": after_ids,
                    "adjusted_s1_rate": after_adj_s1,
                    "best_novelty": after_novelty,
                    "best_loss_ratio": after_loss,
                },
                "delta": {
                    "adjusted_s1_rate": (
                        after_adj_s1 - before_adj_s1
                        if after_adj_s1 is not None and before_adj_s1 is not None
                        else None
                    ),
                    "best_novelty": (
                        after_novelty - before_novelty
                        if after_novelty is not None and before_novelty is not None
                        else None
                    ),
                    "best_loss_ratio": (
                        after_loss - before_loss
                        if after_loss is not None and before_loss is not None
                        else None
                    ),
                },
            }

        try:
            trends = nb.get_experiment_trends()
            learning_log = nb.get_learning_log(limit=300)
            adaptation_events = [
                _event_delta_payload(trends, event)
                for event in learning_log
                if event.get("event_type") == "grammar_weights_applied"
            ]
            return jsonify({
                "trends": trends,
                "adaptation_events": adaptation_events,
                "generated_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Error in /api/trends/context: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs")
    def api_programs():
        """List top programs."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort", "novelty_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            programs = nb.get_top_programs(n, sort_by)
            _annotate_qkv_usage(programs, analytics)
            return jsonify(programs)
        except Exception as e:
            logger.error(f"Error in /api/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights, deduplicated by content (keeps latest)."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            raw = nb.get_insights(category=category, limit=200)
            return jsonify(_deduplicate_insights(raw))
        except Exception as e:
            logger.error(f"Error in /api/insights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/entries")
    def api_entries():
        """List notebook entries."""
        exp_id = request.args.get("experiment_id")
        entry_type = request.args.get("type")
        n = request.args.get("n", 50, type=int)
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type=entry_type, limit=n
            )
            return jsonify(_normalize_entries(entries))
        except Exception as e:
            logger.error(f"Error in /api/entries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/live-feed")
    def api_live_feed():
        """List persisted live-feed events for replay in the dashboard."""
        exp_id = request.args.get("experiment_id")
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id,
                entry_type="live_feed",
                limit=n,
            )
            events = []
            for entry in reversed(entries):
                evt = _entry_to_live_feed_event(entry)
                if evt is not None:
                    events.append(evt)
            return jsonify(events)
        except Exception as e:
            logger.error(f"Error in /api/live-feed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/metrics/<metric_name>")
    def api_metrics(metric_name):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))
        except Exception as e:
            logger.error(f"Error in /api/metrics/{metric_name}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/dashboard")
    def api_dashboard():
        """Get all dashboard data in one call."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()

            # Add campaign/hypothesis/knowledge counts
            try:
                active_campaigns = nb.get_active_campaigns()
                total_hypotheses = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses"
                ).fetchone()[0]
                knowledge_entries = nb.conn.execute(
                    "SELECT COUNT(*) FROM knowledge_base WHERE status = 'active'"
                ).fetchone()[0]
                summary["active_campaigns"] = len(active_campaigns)
                summary["total_hypotheses"] = total_hypotheses
                summary["knowledge_entries"] = knowledge_entries
            except Exception as e:
                logger.warning("Failed enriching dashboard campaign metadata: %s", e)

            recent_experiments = nb.get_recent_experiments(10)
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            top_programs = nb.get_top_programs(10)
            _annotate_qkv_usage(top_programs, analytics)

            data = {
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "recent_experiments": recent_experiments,
                "top_programs": top_programs,
                "insights": _deduplicate_insights(nb.get_insights(limit=50)),
                "recent_entries": _normalize_entries(nb.get_entries(limit=20)),
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            }

            # Compute deltas from latest completed experiment
            try:
                completed = [e for e in recent_experiments
                             if e.get("status") == "completed"]
                if len(completed) >= 2:
                    latest = completed[0]
                    previous = completed[1]
                    data["deltas"] = {
                        "experiment_id": latest.get("experiment_id"),
                        "programs": (latest.get("n_programs_generated") or 0)
                                    - (previous.get("n_programs_generated") or 0),
                        "stage1": (latest.get("n_stage1_passed") or 0)
                                  - (previous.get("n_stage1_passed") or 0),
                        "best_loss": round(
                            (latest.get("best_loss_ratio") or 1)
                            - (previous.get("best_loss_ratio") or 1), 4
                        ) if latest.get("best_loss_ratio") else None,
                        "best_novelty": round(
                            (latest.get("best_novelty_score") or 0)
                            - (previous.get("best_novelty_score") or 0), 4
                        ) if latest.get("best_novelty_score") else None,
                    }
            except Exception:
                pass

            # Include learning trajectory trend in summary
            try:
                trajectory = analytics.learning_trajectory()
                if trajectory and trajectory.get("trend") != "insufficient_data":
                    summary["learning_trend"] = trajectory.get("trend")
                    summary["learning_slope"] = trajectory.get("slope")
                    summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
            except Exception:
                pass

            # Include latest auto-recommendation if experiment just completed
            last_rec = runner.last_recommendation
            if last_rec:
                data["last_recommendation"] = last_rec

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/dashboard: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Report endpoint ──

    @app.route("/api/report")
    def api_report():
        """Consolidated research report with all data."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(20, sort_by="loss_ratio"),
                "top_programs_expanded": nb.get_top_programs(80, sort_by="loss_ratio"),
                "recent_experiments": nb.get_recent_experiments(100),
                "op_success_rates": analytics.op_success_rates(),
                "math_family_coverage": analytics.math_family_coverage(),
                "mathspace_operator_impact": analytics.mathspace_operator_impact(),
                "routing_mode_comparison": analytics.routing_mode_comparison(),
                "gating_behavior_diagnostics": analytics.gating_behavior_diagnostics(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "experiment_clusters": analytics.experiment_clusters(),
                "grammar_weights": {
                    "learned": analytics.compute_grammar_weights(),
                    "default": analytics.get_current_grammar_weights(),
                    "control_comparison": analytics.control_experiment_comparison(),
                    "holdout_validation": analytics.holdout_validation(),
                    "learning_diagnostics": analytics.grammar_weight_learning_diagnostics(),
                },
                "learning_log": nb.get_learning_log(limit=50),
                "insights": nb.get_insights(),
            }
            learning_diagnostics = data["grammar_weights"].get("learning_diagnostics") or {}
            data["architecture_rerun_telemetry"] = {
                "unique_fingerprint_count": int(learning_diagnostics.get("unique_fingerprints") or 0),
                "total_result_rows": int(learning_diagnostics.get("total_rows") or 0),
                "repeat_result_rows": int(learning_diagnostics.get("repeat_rows") or 0),
                "rerun_ratio": float(learning_diagnostics.get("rerun_ratio") or 0.0),
                "top_fingerprint_concentration": float(learning_diagnostics.get("top_fingerprint_concentration") or 0.0),
                "weighting_mode": str(learning_diagnostics.get("mode") or "unknown"),
            }
            data["action_eligibility"] = _build_report_action_eligibility(
                nb,
                [
                    row.get("result_id")
                    for row in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]
                    if row.get("result_id")
                ],
            )
            _annotate_qkv_usage(data["top_programs"], analytics)
            _annotate_qkv_usage(data["top_programs_expanded"], analytics)

            expanded_by_fingerprint: Dict[str, List[Dict[str, Any]]] = {}
            for row in data["top_programs_expanded"]:
                fp = row.get("graph_fingerprint")
                if not fp:
                    continue
                expanded_by_fingerprint.setdefault(fp, []).append(row)

            grouped_rank_by_fingerprint = {
                row.get("graph_fingerprint"): index
                for index, row in enumerate(data["top_programs"], start=1)
                if row.get("graph_fingerprint")
            }
            for fp, rows in expanded_by_fingerprint.items():
                repeat_count = len(rows)
                grouped_rank = grouped_rank_by_fingerprint.get(fp)
                for repeat_index, row in enumerate(rows, start=1):
                    row["group_repeat_count"] = repeat_count
                    row["group_repeat_index"] = repeat_index
                    row["grouped_fingerprint_rank"] = grouped_rank

            data["cross_run_stability"] = _compute_cross_run_stability(
                nb, data["top_programs"]
            )
            stability_by_result = {
                candidate.get("result_id"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("result_id")
            }
            stability_by_fingerprint = {
                candidate.get("graph_fingerprint"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("graph_fingerprint")
            }

            fallback_stability = {
                "trend": "unknown",
                "seen_runs": 0,
                "latest_rank": None,
                "previous_rank": None,
                "rank_delta": None,
            }
            for program in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]:
                by_result = stability_by_result.get(program.get("result_id"))
                by_fingerprint = stability_by_fingerprint.get(program.get("graph_fingerprint"))
                program["cross_run_stability"] = by_result or by_fingerprint or fallback_stability

            # Generate narrative (optional, non-blocking)
            try:
                narrative = aria.generate_report_narrative(data)
                data["narrative"] = narrative
            except Exception as e:
                logger.debug(f"Report narrative generation failed: {e}")
                data["narrative"] = None

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/report: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Analytics endpoints ──

    @app.route("/api/analytics/op-success")
    def api_op_success():
        """Op success rate table."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.op_success_rates())
        except Exception as e:
            logger.error(f"Error in op-success: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/failure-patterns")
    def api_failure_patterns():
        """Failure analysis by error type and stage."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.failure_patterns())
        except Exception as e:
            logger.error(f"Error in failure-patterns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/grammar-weights")
    def api_grammar_weights():
        """Current vs learned grammar weights."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            defaults = analytics.get_current_grammar_weights()
            learned = analytics.compute_grammar_weights()
            control_comparison = analytics.control_experiment_comparison()
            holdout = analytics.holdout_validation()
            explanation = aria.explain_grammar_weights(defaults, learned)
            diagnostics = analytics.grammar_weight_learning_diagnostics()
            return jsonify({
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
                "learning_diagnostics": diagnostics,
                "architecture_rerun_telemetry": {
                    "unique_fingerprint_count": int(diagnostics.get("unique_fingerprints") or 0),
                    "total_result_rows": int(diagnostics.get("total_rows") or 0),
                    "repeat_result_rows": int(diagnostics.get("repeat_rows") or 0),
                    "rerun_ratio": float(diagnostics.get("rerun_ratio") or 0.0),
                    "top_fingerprint_concentration": float(diagnostics.get("top_fingerprint_concentration") or 0.0),
                    "weighting_mode": str(diagnostics.get("mode") or "unknown"),
                },
                "explanation": explanation,
            })
        except Exception as e:
            logger.error(f"Error in grammar-weights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier")
    def api_efficiency_frontier():
        """Pareto-optimal programs on loss vs FLOPs."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/experiment-clusters")
    def api_experiment_clusters():
        """Deterministic experiment clustering summary and stability signal."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.experiment_clusters())
        except Exception as e:
            logger.error(f"Error in experiment-clusters: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-health")
    def api_routing_health():
        """Routing telemetry health summary grouped by routing mode."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_health() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("explanation", "Routing telemetry is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-comparison")
    def api_routing_comparison():
        """Consolidated routing-mode comparison with confidence/sample labels."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_mode_comparison() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("n_modes", 0)
            payload.setdefault("total_programs", 0)
            payload.setdefault("routed_programs", 0)
            payload.setdefault("uniform_programs", 0)
            payload.setdefault("explanation", "Routing comparison data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gating-diagnostics")
    def api_gating_diagnostics():
        """Canonical gating behavior diagnostics (entropy/collapse/retention)."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.gating_behavior_diagnostics() or {}
            payload.setdefault("available", False)
            payload.setdefault("total_routed_programs", 0)
            payload.setdefault("avg_gate_entropy", None)
            payload.setdefault("collapse_risk_counts", {"low": 0, "medium": 0, "high": 0, "unknown": 0})
            payload.setdefault("by_mode", [])
            payload.setdefault("token_retention_curve_overall", [])
            payload.setdefault("explanation", "Gating diagnostics are unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in gating-diagnostics: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/math-family-coverage")
    def api_math_family_coverage():
        """Coverage of evaluated/surviving programs by mathematical family."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.math_family_coverage())
        except Exception as e:
            logger.error(f"Error in math-family-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/mathspace-impact")
    def api_mathspace_impact():
        """Impact of math-space operators/families on S1/validation/novelty."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.mathspace_operator_impact() or {}
            payload.setdefault("available", False)
            payload.setdefault("totals", {
                "n_programs_with_graph": 0,
                "n_programs_with_mathspace": 0,
                "n_mathspace_ops_observed": 0,
            })
            payload.setdefault("by_operator", [])
            payload.setdefault("by_family", [])
            payload.setdefault("top_trustworthy_operators", [])
            payload.setdefault("explanation", "Math-space impact data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in mathspace-impact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-coverage")
    def api_compression_coverage():
        """Coverage of compression techniques across tested and surviving programs."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.compression_coverage())
        except Exception as e:
            logger.error(f"Error in compression-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/negative-results")
    def api_negative_results():
        """Aggregated negative results: failed ops, error types, anti-patterns."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.negative_results_synthesis())
        except Exception as e:
            logger.error(f"Error in negative-results: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-trajectory")
    def api_learning_trajectory():
        """S1 rate trend over time with regression analysis."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.learning_trajectory())
        except Exception as e:
            logger.error(f"Error in learning-trajectory: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/control-comparison")
    def api_control_comparison():
        """Compare control (default weights) vs learned-weight experiments."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            result = analytics.control_experiment_comparison()
            if result is None:
                return jsonify({"status": "insufficient_data",
                                "message": "Need at least 2 control and 2 learned experiments"})
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in control-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-summary")
    def api_learning_summary():
        """Aria-generated 3-5 bullet summary of what the system has learned."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            payload = aria.summarize_learning_bullets({
                "summary": nb.get_dashboard_summary(),
                "grammar_default": analytics.get_current_grammar_weights(),
                "grammar_learned": analytics.compute_grammar_weights(),
                "frontier": analytics.efficiency_frontier(),
                "clusters": analytics.experiment_clusters(),
                "recent_experiments": nb.get_recent_experiments(10),
                "trajectory": analytics.learning_trajectory(),
            })
            payload.setdefault("bullets", [])
            payload.setdefault("source", "rule-based")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in learning-summary: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-log")
    def api_learning_log():
        """Audit trail of grammar weight changes."""
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_learning_log(limit=n))
        except Exception as e:
            logger.error(f"Error in learning-log: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/decision-packet/<result_id>")
    def api_decision_packet(result_id):
        """One-click evidence bundle for promotion decisions."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            fingerprint = program.get("graph_fingerprint", "")
            experiment_id = program.get("experiment_id")

            # Leaderboard entry
            leaderboard_entry = None
            try:
                rows = nb.get_leaderboard(limit=200)
                for entry in rows:
                    if entry.get("result_id") == result_id:
                        leaderboard_entry = entry
                        break
            except Exception:
                pass

            # Experiment data + failure analysis
            experiment = None
            failure_analysis = {"funnel": {}, "errors": {}, "stage_deaths": {}}
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass
                try:
                    failure_analysis = nb.get_failure_analysis(experiment_id)
                except Exception:
                    pass

            # Hypothesis chain — find hypothesis linked to this experiment
            hypothesis_chain = []
            if experiment_id:
                try:
                    hyp_row = nb.conn.execute(
                        "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
                        (experiment_id,),
                    ).fetchone()
                    if hyp_row:
                        hypothesis_chain = nb.get_hypothesis_chain(
                            hyp_row["hypothesis_id"] if isinstance(hyp_row, dict)
                            else hyp_row[0]
                        )
                except Exception:
                    pass

            # Cross-run stability for this specific result
            cross_run = {"trend": "unknown", "seen_runs": 0}
            try:
                top = nb.get_top_programs(20, sort_by="loss_ratio")
                stability = _compute_cross_run_stability(nb, top)
                for c in stability.get("candidates", []):
                    if c.get("result_id") == result_id:
                        cross_run = {
                            "trend": c.get("trend", "unknown"),
                            "seen_runs": c.get("seen_runs", 0),
                        }
                        break
            except Exception:
                pass

            # Build outcomes by phase
            tier = (leaderboard_entry or {}).get("tier", "screening")
            outcomes = {
                "screening": {
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty": program.get("novelty_score"),
                },
                "investigation": None,
                "validation": None,
            }
            if leaderboard_entry:
                inv_lr = leaderboard_entry.get("investigation_loss_ratio")
                if inv_lr is not None:
                    outcomes["investigation"] = {
                        "loss_ratio": inv_lr,
                        "robustness": leaderboard_entry.get("investigation_robustness"),
                        "passed": bool(leaderboard_entry.get("investigation_passed")),
                    }
                val_lr = leaderboard_entry.get("validation_loss_ratio")
                if val_lr is not None:
                    outcomes["validation"] = {
                        "loss_ratio": val_lr,
                        "baseline_ratio": leaderboard_entry.get("validation_baseline_ratio"),
                        "multi_seed_std": leaderboard_entry.get("validation_multi_seed_std"),
                        "passed": bool(leaderboard_entry.get("validation_passed")),
                    }

            # Baseline comparison
            bl_ratio = program.get("baseline_loss_ratio")
            baseline_comparison = {"ratio": bl_ratio, "interpretation": "unknown"}
            if bl_ratio is not None:
                if bl_ratio < 0.95:
                    baseline_comparison["interpretation"] = "outperforms"
                elif bl_ratio <= 1.05:
                    baseline_comparison["interpretation"] = "comparable"
                else:
                    baseline_comparison["interpretation"] = "underperforms"

            # Failure context
            failure_context = {
                "stage_at_death": program.get("stage_at_death"),
                "error_type": program.get("error_type"),
                "experiment_errors": failure_analysis.get("errors", {}),
                "experiment_funnel": failure_analysis.get("funnel", {}),
            }

            # Recommendation
            recommendation = _compute_recommendation(program, leaderboard_entry)

            # Evidence flags
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            packet_status = analytics.reproducibility_packet_status(
                leaderboard_entry if leaderboard_entry else program
            )
            evidence_flags = {
                "has_baseline": bl_ratio is not None,
                "has_cka_artifact": program.get("cka_source") == "artifact",
                "has_multi_seed": outcomes["validation"] is not None,
                "has_hypothesis": len(hypothesis_chain) > 0,
                "repro_packet_ready": packet_status.get("status") == "ready",
            }

            return jsonify({
                "result_id": result_id,
                "fingerprint": fingerprint,
                "experiment_id": experiment_id,
                "hypothesis_chain": hypothesis_chain,
                "outcomes": outcomes,
                "baseline_comparison": baseline_comparison,
                "failure_context": failure_context,
                "cross_run_stability": cross_run,
                "recommendation": recommendation,
                "evidence_flags": evidence_flags,
                "compression_metrics": analytics.canonical_compression_metrics(
                    leaderboard_entry if leaderboard_entry else program
                ),
                "reproducibility_packet": packet_status,
            })
        except Exception as e:
            logger.error(f"Error in /api/decision-packet/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/reproducibility-manifest/<result_id>")
    def api_reproducibility_manifest(result_id):
        """Exportable reproducibility manifest for a program result."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            experiment_id = program.get("experiment_id")
            experiment = None
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass

            config = (experiment or {}).get("config", {}) or {}
            training = {}
            try:
                tp = json.loads(program.get("training_program_json") or "{}")
                training = tp
            except (json.JSONDecodeError, TypeError):
                pass

            # Grammar weights snapshot from experiment config
            grammar_weights = config.get("applied_grammar_weights") or config.get("grammar_weights")
            grammar_config = config.get("grammar_config", {})

            manifest = {
                "result_id": result_id,
                "graph_fingerprint": program.get("graph_fingerprint"),
                "experiment_id": experiment_id,
                "experiment_type": (experiment or {}).get("experiment_type"),
                "timestamp": program.get("timestamp"),
                "code_version": config.get("code_version"),
                "seeds": {
                    "experiment_seed": config.get("seed"),
                    "training_seed": training.get("seed"),
                },
                "data": {
                    "data_mode": config.get("data_mode"),
                    "dataset": config.get("dataset"),
                    "seq_len": training.get("seq_len") or config.get("seq_len"),
                    "batch_size": training.get("batch_size") or config.get("batch_size"),
                    "vocab_size": training.get("vocab_size") or config.get("vocab_size"),
                },
                "grammar": {
                    "max_ops": grammar_config.get("max_ops"),
                    "max_depth": grammar_config.get("max_depth"),
                    "weights_snapshot": grammar_weights,
                },
                "training": {
                    "learning_rate": training.get("learning_rate") or training.get("lr"),
                    "steps": training.get("steps") or training.get("n_steps"),
                    "warmup_steps": training.get("warmup_steps"),
                },
                "architecture": {
                    "param_count": program.get("param_count"),
                    "graph_json": program.get("graph_json"),
                },
                "outcomes": {
                    "stage0_passed": bool(program.get("stage0_passed")),
                    "stage05_passed": bool(program.get("stage05_passed")),
                    "stage1_passed": bool(program.get("stage1_passed")),
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty_score": program.get("novelty_score"),
                    "baseline_loss_ratio": program.get("baseline_loss_ratio"),
                },
                "canonical_metrics": {
                    "compression": analytics.canonical_compression_metrics(program),
                },
                "packet_status": analytics.reproducibility_packet_status(program),
            }
            return jsonify(manifest)
        except Exception as e:
            logger.error(f"Error in /api/reproducibility-manifest/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/training-curve")
    def api_training_curve(result_id):
        """Per-step training data for a program."""
        nb = LabNotebook(notebook_path)
        try:
            curve = nb.get_training_curve(result_id)
            return jsonify(curve)
        except Exception as e:
            logger.error(f"Error in training-curve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Leaderboard endpoints ──

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            stability = _compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {
                        "trend": "unknown",
                        "seen_runs": 0,
                        "latest_rank": None,
                        "previous_rank": None,
                        "rank_delta": None,
                    },
                )
            _annotate_qkv_usage(entries, analytics)
            # Group by tier for the dashboard
            tiers = {}
            for entry in entries:
                t = entry.get("tier", "screening")
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(entry)
            return jsonify({
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Control endpoints ──

    @app.route("/api/experiments/start", methods=["POST"])
    def api_start_experiment():
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = _get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        body = request.get_json(silent=True) or {}
        hypothesis = body.pop("hypothesis", None)
        mode = body.pop("mode", "single")  # "single", "continuous", "evolve", "novelty"

        config = RunConfig.from_dict(body) if body else RunConfig()

        eligibility: Optional[Dict[str, Any]] = None

        try:
            if mode == "continuous":
                config.continuous = True
                exp_id = runner.start_continuous(config)
            elif mode == "evolve":
                exp_id = runner.start_evolution(config, hypothesis=hypothesis)
            elif mode == "novelty":
                exp_id = runner.start_novelty_search(config, hypothesis=hypothesis)
            elif mode == "investigation":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for investigation mode"}), 400
                nb = LabNotebook(notebook_path)
                try:
                    eligibility = _build_start_mode_eligibility(nb, "investigation", result_ids)
                finally:
                    nb.close()
                if not eligibility.get("all_eligible"):
                    return jsonify({
                        "error": "Ineligible result_ids for investigation mode",
                        "eligibility": eligibility,
                    }), 409
                exp_id = runner.start_investigation(result_ids, config, hypothesis=hypothesis)
            elif mode == "validation":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for validation mode"}), 400
                nb = LabNotebook(notebook_path)
                try:
                    eligibility = _build_start_mode_eligibility(nb, "validation", result_ids)
                finally:
                    nb.close()
                if not eligibility.get("all_eligible"):
                    return jsonify({
                        "error": "Ineligible result_ids for validation mode",
                        "eligibility": eligibility,
                    }), 409
                exp_id = runner.start_validation(result_ids, config, hypothesis=hypothesis)
            elif mode == "scale_up":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for scale_up mode"}), 400
                config.scale_up = True
                config.scale_up_result_ids = ",".join(result_ids)
                exp_id = runner.start_scale_up(result_ids, config, hypothesis=hypothesis)
            else:
                exp_id = runner.start_experiment(config, hypothesis=hypothesis)

            return jsonify({
                "experiment_id": exp_id,
                "status": "started",
                "config": config.to_dict(),
                "aria_message": runner.progress.aria_message,
                "hypothesis_critique": runner.progress.hypothesis_critique,
                "hypothesis_review_gate": (
                    runner.progress.hypothesis_critique.get("gate")
                    if isinstance(runner.progress.hypothesis_critique, dict)
                    else None
                ),
                "eligibility": eligibility,
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error starting experiment: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/experiments/stop", methods=["POST"])
    def api_stop_experiment():
        """Stop the currently running experiment."""
        runner = _get_runner(notebook_path)
        if not runner.is_running:
            return jsonify({"error": "No experiment is running"}), 409

        runner.stop()
        return jsonify({
            "status": "stopping",
            "aria_message": runner.progress.aria_message,
        })

    @app.route("/api/experiments/<experiment_id>/cancel", methods=["POST"])
    def api_cancel_experiment(experiment_id):
        """Cancel a stuck/running experiment by marking it as failed."""
        nb = LabNotebook(notebook_path)
        try:
            cancelled = nb.cancel_experiment(experiment_id)
            if not cancelled:
                return jsonify({
                    "error": "Experiment not found or not in running state",
                }), 404
            return jsonify({"status": "cancelled", "experiment_id": experiment_id})
        finally:
            nb.close()

    @app.route("/api/experiments/cleanup-stale", methods=["POST"])
    def api_cleanup_stale():
        """Clean up stale running experiments that are no longer active."""
        nb = LabNotebook(notebook_path)
        try:
            count = nb.cleanup_stale_experiments()
            return jsonify({"cleaned": count})
        finally:
            nb.close()

    @app.route("/api/progress")
    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = _get_runner(notebook_path)
        return jsonify({
            "is_running": runner.is_running,
            "progress": runner.progress.to_dict(),
        })

    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = _get_runner(notebook_path)
        sse_timeout = _get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = json.dumps(event["data"])
                    yield f"event: {event['type']}\ndata: {data}\n\n"
                # After timeout, check if client is still connected
                yield f"event: keepalive\ndata: {{}}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        """Get the default RunConfig."""
        return jsonify(RunConfig().to_dict())

    # ── LLM Configuration endpoints ──

    @app.route("/api/llm/config")
    def api_llm_config():
        """Get current LLM backend configuration."""
        aria = get_aria()
        return jsonify(aria.get_llm_config())

    @app.route("/api/llm/config", methods=["POST"])
    def api_llm_configure():
        """Configure the LLM backend at runtime and persist to disk."""
        aria = get_aria()
        body = request.get_json(silent=True) or {}

        backend_name = str(body.get("backend", "")).strip()
        if not backend_name:
            return jsonify({"error": "backend is required (anthropic, openai, ollama)"}), 400

        api_key = str(body.get("api_key", "")).strip()
        model = str(body.get("model", "")).strip()
        host = str(body.get("host", "")).strip()

        success = aria.configure_llm(
            backend_name=backend_name,
            api_key=api_key,
            model=model,
            host=host,
        )

        if success:
            # Quick health check: try a minimal LLM call to verify the key works
            health_ok = True
            health_error = None
            llm = aria._get_llm()
            if llm:
                try:
                    test_resp = llm.generate(
                        "Respond with exactly: OK",
                        max_tokens=10, temperature=0,
                    )
                    if not (test_resp and test_resp.text):
                        health_ok = False
                        health_error = "LLM returned empty response"
                except Exception as e:
                    health_ok = False
                    health_error = f"{type(e).__name__}: {str(e)[:150]}"
                    logger.warning(f"LLM health check failed: {health_error}")

            # Persist config so it survives server restarts
            _save_llm_config(notebook_path, {
                "backend": backend_name,
                "api_key": api_key,
                "model": model,
                "host": host,
            })

            # Clear any cached deterministic briefing so AI takes over
            if hasattr(aria, "_briefing_cache"):
                aria._briefing_cache = None

            result = {
                "status": "configured",
                "config": aria.get_llm_config(),
            }
            if not health_ok:
                result["status"] = "configured_with_warning"
                result["warning"] = health_error
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to configure LLM backend"}), 500

    # ── Strategy Briefing endpoint ──

    def _normalize_briefing_mode(mode: Optional[str]) -> Optional[str]:
        if not mode:
            return None
        normalized = str(mode).strip().lower()
        aliases = {
            "evolution": "evolve",
            "evolve": "evolve",
            "novelty_search": "novelty",
            "novelty": "novelty",
            "investigate": "investigation",
            "investigation": "investigation",
            "validate": "validation",
            "validation": "validation",
            "scale-up": "scale_up",
            "scale_up": "scale_up",
            "continuous": "continuous",
            "single": "single",
        }
        return aliases.get(normalized, normalized)

    def _briefing_action_from_mode(mode: Optional[str]) -> Optional[str]:
        if not mode:
            return None
        actions = {
            "investigation": "investigate",
            "validation": "validate",
            "continuous": "continuous",
            "novelty": "novelty_search",
            "evolve": "evolve",
            "scale_up": "scale_up",
        }
        return actions.get(mode)

    def _briefing_action_label(mode: Optional[str], hypothesis: Optional[str] = None) -> str:
        """Human-readable label for an LLM-suggested action."""
        labels = {
            "continuous": "Run Continuous Research",
            "evolve": "Run Evolution Search",
            "novelty": "Run Novelty Search",
            "investigation": "Investigate Candidates",
            "validation": "Run Validation",
            "scale_up": "Scale Up Training",
        }
        return labels.get(mode, f"Run {mode or 'experiment'}")

    @app.route("/api/strategy/briefing")
    def api_strategy_briefing():
        """Data-driven strategy briefing for the overview page.

        Tries LLM-powered briefing first (via Aria), falls back to
        deterministic rules.  Always returns a valid response.
        """
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            summary = nb.get_dashboard_summary()
            recent = nb.get_recent_experiments(10)
            trajectory = analytics.learning_trajectory() or {}

            # Optional: highlight a just-completed experiment
            just_completed_id = request.args.get("just_completed")
            just_completed_exp = None
            if just_completed_id:
                for e in recent:
                    if (e.get("experiment_id") or "").startswith(just_completed_id):
                        just_completed_exp = e
                        break
                # Clear briefing cache so LLM sees the new context
                aria_inst = get_aria()
                if hasattr(aria_inst, "_briefing_cache"):
                    aria_inst._briefing_cache = None

            # --- Pipeline counts ---
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}
            screening = tiers.get("screening", 0)
            investigation = tiers.get("investigation", 0)
            validation = tiers.get("validation", 0)
            breakthrough = tiers.get("breakthrough", 0)

            # --- Recent outcomes ---
            completed = [e for e in recent if e.get("status") == "completed"]
            recent_s1_rates = []
            for e in completed[:5]:
                gen = e.get("n_programs_generated") or 0
                passed = e.get("n_stage1_passed") or 0
                if gen > 0:
                    recent_s1_rates.append(passed / gen)

            avg_recent_s1 = (
                sum(recent_s1_rates) / len(recent_s1_rates)
                if recent_s1_rates
                else None
            )

            # --- Learning trend ---
            trend = trajectory.get("trend", "insufficient_data")
            slope = trajectory.get("slope")

            # --- Common data block (used by both LLM and deterministic) ---
            total_exp = summary.get("total_experiments", 0)
            total_progs = summary.get("total_programs_evaluated", 0)
            s1_survivors = summary.get("stage1_survivors", 0)

            pipeline_data = {
                "screening": screening,
                "investigation": investigation,
                "validation": validation,
                "breakthrough": breakthrough,
            }
            data_block = {
                "total_experiments": total_exp,
                "total_programs": total_progs,
                "s1_survivors": s1_survivors,
                "avg_recent_s1_rate": avg_recent_s1,
                "learning_trend": trend,
                "learning_slope": slope,
                "pipeline": pipeline_data,
            }

            recent_window = recent[:10]
            recent_cancelled = 0
            recent_failed = 0
            for exp in recent_window:
                status = str(exp.get("status") or "").strip().lower()
                if status in {"cancelled", "canceled"}:
                    recent_cancelled += 1
                elif status == "failed":
                    recent_failed += 1

            recent_completed_window = completed[:5]
            recent_zero_s1_runs = 0
            for exp in recent_completed_window:
                gen = exp.get("n_programs_generated") or 0
                passed = exp.get("n_stage1_passed") or 0
                if gen > 0 and passed == 0:
                    recent_zero_s1_runs += 1

            recommendation_evidence = {
                "learning_trend": trend,
                "learning_slope": slope,
                "avg_recent_s1_rate": avg_recent_s1,
                "recent_completed_runs": len(recent_completed_window),
                "recent_zero_s1_runs": recent_zero_s1_runs,
                "recent_cancelled_runs": recent_cancelled,
                "recent_failed_runs": recent_failed,
                "pipeline": pipeline_data,
            }

            # --- Try LLM-powered briefing first ---
            aria = get_aria()
            fallback_reason: Optional[str] = None
            llm = aria._get_llm()
            llm_reachable = False
            if llm is None:
                fallback_reason = "llm_not_configured"
            else:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
                if not llm_reachable:
                    fallback_reason = "llm_unreachable"
            try:
                from .llm.context import build_briefing_context

                # Gather extra context for LLM
                try:
                    active_campaigns = nb.get_active_campaigns()
                    campaign = active_campaigns[0] if active_campaigns else None
                except Exception:
                    campaign = None

                try:
                    dw = analytics.get_current_grammar_weights() or {}
                except Exception:
                    dw = {}

                try:
                    gw = analytics.compute_grammar_weights() or {}
                except Exception:
                    gw = {}

                try:
                    top_programs = nb.conn.execute(
                        "SELECT graph_fingerprint, loss_ratio, novelty_score, tier "
                        "FROM leaderboard ORDER BY composite_score DESC LIMIT 3"
                    ).fetchall()
                    top_progs = [dict(r) for r in top_programs] if top_programs else None
                except Exception:
                    top_progs = None

                try:
                    briefing_context = build_briefing_context(
                        recent_experiments=recent,
                        pipeline_tiers=tiers,
                        learning_trajectory=trajectory,
                        campaign=campaign,
                        grammar_weights=gw,
                        default_weights=dw,
                        top_programs=top_progs,
                        just_completed=just_completed_exp,
                    )
                except Exception:
                    briefing_context = {
                        "pipeline": pipeline_data,
                        "learning": {
                            "trend": trend,
                            "slope": slope,
                            "avg_recent_s1_rate": avg_recent_s1,
                        },
                        "recent_experiments": recent[:5],
                        "campaign": campaign,
                    }

                ai_briefing = aria.generate_briefing(context=briefing_context)
                if ai_briefing and ai_briefing.get("briefing_text"):
                    suggested = ai_briefing.get("suggested_action") or {}
                    normalized_mode = _normalize_briefing_mode(suggested.get("mode"))
                    action_key = _briefing_action_from_mode(normalized_mode)
                    suggested_config = dict(suggested.get("config") or {})
                    hypothesis = suggested.get("hypothesis")
                    if normalized_mode:
                        suggested_config["mode"] = normalized_mode
                    if hypothesis:
                        suggested_config["hypothesis"] = hypothesis
                    return jsonify({
                        "briefing": ai_briefing["briefing_text"],
                        "action": action_key or normalized_mode or "continuous",
                        "action_label": _briefing_action_label(
                            normalized_mode, hypothesis),
                        "action_rationale": suggested.get("reasoning", ""),
                        "ai_powered": True,
                        "confidence": ai_briefing.get("confidence", 0.5),
                        "suggested_config": suggested_config or None,
                        "evidence": recommendation_evidence,
                        "data": data_block,
                    })
                if fallback_reason is None:
                    fallback_reason = "llm_empty_response"
            except Exception as e:
                logger.warning(f"LLM briefing unavailable, using deterministic: {e}")
                err_msg = str(e)[:120]
                fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"

            # --- Deterministic fallback: build briefing sentences ---
            sentences = []
            if total_exp > 0:
                sentences.append(
                    f"Across {total_exp} experiments, {total_progs:,} architectures "
                    f"have been evaluated with {s1_survivors} stage-1 survivors "
                    f"({s1_survivors / max(total_progs, 1) * 100:.1f}% overall pass rate)."
                )

            # 2. Recent performance
            if avg_recent_s1 is not None:
                n_recent = len(recent_s1_rates)
                sentences.append(
                    f"The last {n_recent} completed experiment{'s' if n_recent != 1 else ''} "
                    f"averaged a {avg_recent_s1 * 100:.1f}% S1 pass rate."
                )

            # 3. Learning trajectory
            if trend == "improving" and slope is not None:
                sentences.append(
                    f"The system is learning — S1 rate is improving at "
                    f"+{abs(slope) * 100:.2f} percentage points per experiment."
                )
            elif trend == "declining" and slope is not None:
                sentences.append(
                    f"S1 rate is declining ({slope * 100:.2f} pp/experiment). "
                    f"Consider switching search strategy or trying evolution mode."
                )
            elif trend == "plateaued":
                sentences.append(
                    "S1 rate has plateaued — a novelty search or evolution run "
                    "could help escape the current local optimum."
                )

            # 4. Pipeline state
            pipeline_parts = []
            if screening > 0:
                pipeline_parts.append(f"{screening} at screening")
            if investigation > 0:
                pipeline_parts.append(f"{investigation} under investigation")
            if validation > 0:
                pipeline_parts.append(f"{validation} in validation")
            if breakthrough > 0:
                pipeline_parts.append(
                    f"{breakthrough} breakthrough{'s' if breakthrough != 1 else ''}"
                )
            if pipeline_parts:
                sentences.append(
                    f"Candidate pipeline: {', '.join(pipeline_parts)}."
                )

            # 5. Last experiment outcome
            if completed:
                last = completed[0]
                last_s1 = last.get("n_stage1_passed") or 0
                last_gen = last.get("n_programs_generated") or 0
                last_loss = last.get("best_loss_ratio")
                last_id = last.get("experiment_id", "")[:8]
                parts = [
                    f"Last experiment ({last_id}): "
                    f"{last_s1}/{last_gen} passed S1"
                ]
                if last_loss is not None:
                    parts.append(f"best loss {last_loss:.4f}")
                aria_sum = last.get("aria_summary")
                if aria_sum:
                    parts.append(f"— {aria_sum}")
                sentences.append(". ".join(parts) + ".")

            briefing = " ".join(sentences)

            # --- Determine recommended action ---
            action = None
            action_label = None
            action_rationale = None

            if breakthrough > 0:
                action = "export_breakthrough"
                action_label = "Export Breakthrough Report"
                action_rationale = (
                    f"{breakthrough} candidate{'s have' if breakthrough != 1 else ' has'} "
                    f"reached breakthrough tier — ready for publication review."
                )
            elif validation > 0 and screening == 0 and investigation == 0:
                action = "monitor_validation"
                action_label = "Review Validation Progress"
                action_rationale = (
                    f"{validation} candidate{'s are' if validation != 1 else ' is'} "
                    f"in validation. Monitor results before starting new experiments."
                )
            elif screening > 0:
                inv_failed = nb.conn.execute(
                    "SELECT COUNT(*) FROM leaderboard "
                    "WHERE tier = 'investigation' AND investigation_passed = 0"
                ).fetchone()[0]
                action = "investigate"
                action_label = (
                    f"Investigate {screening} Screening "
                    f"Survivor{'s' if screening != 1 else ''}"
                )
                rationale_parts = [
                    f"{screening} candidate{'s' if screening != 1 else ''} passed "
                    f"screening and "
                    f"{'are' if screening != 1 else 'is'} awaiting deeper investigation"
                ]
                if inv_failed > 0:
                    rationale_parts.append(
                        f"({inv_failed} prior investigation"
                        f"{'s' if inv_failed != 1 else ''} "
                        f"failed — fresh candidates may outperform)"
                    )
                if avg_recent_s1 is not None:
                    rationale_parts.append(
                        f"with recent {avg_recent_s1 * 100:.0f}% hit rate"
                    )
                action_rationale = ", ".join(rationale_parts) + "."
            elif total_exp == 0:
                action = "start_first"
                action_label = "Run First Experiment"
                action_rationale = (
                    "No experiments yet. Start a mixed continuous run to begin "
                    "exploring the architecture space."
                )
            elif trend == "declining" or (
                len(recent_s1_rates) >= 3
                and all(r == 0 for r in recent_s1_rates[:3])
            ):
                action = "novelty_search"
                action_label = "Try Evolution / Novelty Search"
                action_rationale = (
                    "Recent experiments are underperforming. An evolution or "
                    "novelty-driven search can escape the current local minimum."
                )
            else:
                action = "continuous"
                action_label = "Continue Research"
                action_rationale = (
                    "The pipeline is active and the system is "
                    + ("learning" if trend == "improving" else "exploring")
                    + ". Continue generating and evaluating new architectures."
                )

            # Build deterministic suggested_config from action
            det_mode_map = {
                "investigate": "investigation",
                "continuous": "continuous",
                "start_first": "continuous",
                "novelty_search": "novelty",
                "export_breakthrough": None,
                "monitor_validation": None,
            }
            det_mode = det_mode_map.get(action, "continuous")
            det_config = (
                {"mode": det_mode, "model_source": "mixed"}
                if det_mode
                else None
            )

            return jsonify({
                "briefing": briefing,
                "action": action,
                "action_label": action_label,
                "action_rationale": action_rationale,
                "ai_powered": False,
                "fallback_reason": fallback_reason,
                "suggested_config": det_config,
                "evidence": recommendation_evidence,
                "data": data_block,
            })
        except Exception as e:
            logger.error(f"Error in /api/strategy/briefing: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Aria Intelligence endpoints ──

    @app.route("/api/aria/cycle-status")
    def api_aria_cycle_status():
        """Get Aria continuous-cycle status (planning/running/analyzing)."""
        runner = _get_runner(notebook_path)
        try:
            return jsonify(runner.get_aria_cycle_status())
        except Exception as e:
            logger.error(f"Error in /api/aria/cycle-status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/recommendation")
    def api_aria_recommendation():
        """Get Aria's experiment recommendation based on all data."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            suggestion = aria.suggest_experiment(context)
            return jsonify(suggestion)
        except Exception as e:
            logger.error(f"Error in /api/aria/recommendation: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/strategy")
    def api_aria_strategy():
        """Get Aria's research strategy recommendation."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            strategy = aria.plan_strategy(context)
            return jsonify({
                "strategy": strategy,
                "available": strategy is not None,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/strategy: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat", methods=["POST"])
    def api_aria_chat():
        """Interactive Aria chat response grounded in current research context."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()

        def _normalize_chat_sections(text: str) -> str:
            raw = (text or "").strip()
            if not raw:
                return ""
            if (
                "Evidence:" in raw
                and "Recommendation:" in raw
                and "Next Action:" in raw
            ):
                return raw
            return (
                "Evidence:\n"
                f"- {raw}\n\n"
                "Recommendation:\n"
                "- Continue with the strongest data-supported direction from recent runs.\n\n"
                "Next Action:\n"
                "- Launch the suggested experiment and review post-run briefing for iteration."
            )

        try:
            body = request.get_json(silent=True) or {}
            question = str(body.get("message") or "").strip()
            history_raw = body.get("history") or []
            session_id = str(body.get("session_id") or "").strip()
            fallback_reason: Optional[str] = None

            if not question:
                return jsonify({"error": "message is required"}), 400

            # Persist user message to DB if session_id provided
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="user", text=question,
                        label="You",
                    )
                except Exception:
                    pass  # Non-fatal — don't block chat on persistence failure

            # Build history lines: prefer DB history when session_id given
            history_lines: List[str] = []
            if session_id:
                try:
                    db_messages = nb.get_chat_history(session_id, limit=12)
                    for msg in db_messages:
                        role = str(msg.get("role") or "user").strip().lower()
                        text = str(msg.get("text") or "").strip()
                        if not text:
                            continue
                        label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                        history_lines.append(f"{label}: {text}")
                except Exception:
                    pass  # Fall through to request-body history
            if not history_lines and isinstance(history_raw, list):
                for entry in history_raw[-8:]:
                    if not isinstance(entry, dict):
                        continue
                    role = str(entry.get("role") or "user").strip().lower()
                    if role not in {"user", "aria", "assistant", "system"}:
                        role = "user"
                    text = str(entry.get("text") or "").strip()
                    if not text:
                        continue
                    label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                    history_lines.append(f"{label}: {text}")

            try:
                analytics_data = runner._gather_analytics_data(nb)
            except Exception:
                analytics_data = {}

            try:
                history = nb.get_recent_experiments(10)
            except Exception:
                history = []

            try:
                past_hypotheses = runner._get_past_hypotheses(nb)
            except Exception:
                past_hypotheses = []

            try:
                from .llm.context import build_rich_context
                context = build_rich_context(
                    results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                             "stage1_passed": 0, "novel_count": 0},
                    analytics_data=analytics_data,
                    history=history,
                    past_hypotheses=past_hypotheses,
                )
            except Exception:
                context = (
                    "Context fallback:\n"
                    f"- Recent experiments: {len(history)}\n"
                    f"- Analytics keys: {len(analytics_data) if isinstance(analytics_data, dict) else 0}\n"
                    f"- Past hypotheses: {len(past_hypotheses) if isinstance(past_hypotheses, list) else 0}"
                )

            llm = aria._get_llm()
            if llm:
                try:
                    if hasattr(llm, "is_available") and not llm.is_available():
                        fallback_reason = "llm_unreachable"
                except Exception:
                    fallback_reason = "llm_unreachable"
                try:
                    from .llm.prompts import SYSTEM_PROMPT, CHAT_PROMPT
                    prompt = CHAT_PROMPT.format(
                        context=context,
                        history="\n".join(history_lines) if history_lines else "(none)",
                        question=question,
                    )
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=768)
                    aria._track_cost(resp)
                    text = (resp.text or "").strip()
                    if text:
                        reply_text = _normalize_chat_sections(text)
                        if session_id:
                            try:
                                nb.save_chat_message(
                                    session_id=session_id, role="aria",
                                    text=reply_text, label="Aria",
                                )
                            except Exception:
                                pass
                        return jsonify({
                            "reply": reply_text,
                            "ai_powered": True,
                            "used_context": True,
                            "fallback_reason": None,
                        })
                    fallback_reason = fallback_reason or "llm_empty_response"
                except Exception as e:
                    logger.warning(f"Aria chat LLM failed, using fallback: {e}")
                    err_msg = str(e)[:120]
                    fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"
            else:
                fallback_reason = "llm_not_configured"

            strategy = aria.plan_strategy(context) or ""
            suggestion = aria.suggest_experiment(context) or {}
            evidence_lines = [
                "- LLM chat backend is unavailable; using deterministic analysis.",
            ]
            if strategy:
                evidence_lines.append(f"- Current strategy signal: {strategy}")
            reasoning = suggestion.get("reasoning") if isinstance(suggestion, dict) else None
            recommendation_lines = []
            if reasoning:
                recommendation_lines.append(f"- {reasoning}")
            else:
                recommendation_lines.append("- Use the current Strategy Advisor recommendation to stay aligned with latest metrics.")
            next_action = "- Run one suggested experiment, then ask again to refresh with new evidence."

            fallback_reply = (
                "Evidence:\n"
                + "\n".join(evidence_lines)
                + "\n\nRecommendation:\n"
                + "\n".join(recommendation_lines)
                + "\n\nNext Action:\n"
                + next_action
            )
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="aria",
                        text=fallback_reply,
                        label=f"Aria (fallback: {fallback_reason})",
                    )
                except Exception:
                    pass
            return jsonify({
                "reply": fallback_reply,
                "ai_powered": False,
                "used_context": True,
                "fallback_reason": fallback_reason,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat/history")
    def api_aria_chat_history():
        """Load chat history from the database."""
        nb = LabNotebook(notebook_path)
        try:
            session_id = request.args.get("session_id", "default")
            limit = min(int(request.args.get("limit", 50)), 200)
            messages = nb.get_chat_history(session_id, limit=limit)
            return jsonify({"messages": messages, "session_id": session_id})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat/message", methods=["POST"])
    def api_aria_chat_message():
        """Save a single chat message to the database."""
        nb = LabNotebook(notebook_path)
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            role = body.get("role", "user")
            text = body.get("text", "")
            label = body.get("label")
            message_id = body.get("message_id")
            metadata = body.get("metadata")
            if not text:
                return jsonify({"error": "text is required"}), 400
            mid = nb.save_chat_message(
                session_id=session_id, role=role, text=text,
                label=label, message_id=message_id, metadata=metadata,
            )
            return jsonify({"message_id": mid, "saved": True})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/message: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    def _estimate_tokens(text: str) -> int:
        """Rough token count: ~4 chars per token."""
        return len(text or "") // 4

    @app.route("/api/aria/chat/compact", methods=["POST"])
    def api_aria_chat_compact():
        """Compact older chat messages into a summary when token budget exceeded."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            token_budget = int(body.get("token_budget", 4000))

            messages = nb.get_chat_history(session_id, limit=200)
            if not messages:
                return jsonify({"compacted": False, "reason": "no messages"})

            # Calculate tokens for active messages
            total_tokens = sum(_estimate_tokens(m.get("text", "")) for m in messages)
            if total_tokens <= token_budget:
                return jsonify({"compacted": False, "reason": "within budget",
                                "total_tokens": total_tokens})

            # Find oldest messages that exceed the budget
            # Keep recent messages within budget, compact the rest
            keep_tokens = 0
            keep_from = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = _estimate_tokens(messages[i].get("text", ""))
                if keep_tokens + msg_tokens > token_budget * 0.7:  # Keep 70% budget for recent
                    keep_from = i + 1
                    break
                keep_tokens += msg_tokens

            to_compact = messages[:keep_from]
            if not to_compact:
                return jsonify({"compacted": False, "reason": "nothing to compact"})

            # Build text for summarization
            compact_text = "\n".join(
                f"{m.get('role', 'unknown').upper()}: {m.get('text', '')}"
                for m in to_compact
            )

            # Try LLM summarization, fall back to first-sentence extraction
            summary_text = None
            llm = aria._get_llm()
            if llm:
                try:
                    from .llm.prompts import SYSTEM_PROMPT, CHAT_COMPACTION_PROMPT
                    prompt = CHAT_COMPACTION_PROMPT.format(messages=compact_text[:3000])
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=300)
                    aria._track_cost(resp)
                    summary_text = (resp.text or "").strip()
                except Exception as e:
                    logger.warning(f"Chat compaction LLM failed: {e}")

            if not summary_text:
                # Fallback: extract first sentence from each message
                lines = []
                for m in to_compact:
                    text = (m.get("text") or "").strip()
                    first_sentence = text.split(".")[0].strip()
                    if first_sentence and len(first_sentence) > 10:
                        role = m.get("role", "?").upper()
                        lines.append(f"- [{role}] {first_sentence}.")
                    if len(lines) >= 5:
                        break
                summary_text = "\n".join(lines) if lines else "Previous conversation summarized."

            # Save summary message
            import uuid as _uuid
            summary_id = f"summary-{_uuid.uuid4().hex[:8]}"
            compact_ids = [m["message_id"] for m in to_compact if m.get("message_id")]

            nb.save_chat_message(
                session_id=session_id, role="system",
                text=summary_text, label="Summary",
                message_id=summary_id,
                metadata={"compaction": True, "summarized_count": len(compact_ids)},
            )
            nb.mark_messages_compacted(compact_ids, summary_id)

            return jsonify({
                "compacted": True,
                "messages_compacted": len(compact_ids),
                "summary_id": summary_id,
                "summary_tokens": _estimate_tokens(summary_text),
                "original_tokens": total_tokens,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/compact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/system/status")
    def api_system_status():
        """Report system status: CUDA, LLM, database, runner state."""
        import torch
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            # CUDA info
            cuda_available = torch.cuda.is_available()
            cuda_info = {}
            if cuda_available:
                try:
                    cuda_info = {
                        "device_name": torch.cuda.get_device_name(0),
                        "device_count": torch.cuda.device_count(),
                    }
                    mem = torch.cuda.mem_get_info(0)
                    cuda_info["memory_free_gb"] = round(mem[0] / 1e9, 1)
                    cuda_info["memory_total_gb"] = round(mem[1] / 1e9, 1)
                except Exception as e:
                    logger.warning("Failed collecting CUDA details: %s", e)

            # LLM backend
            llm = aria._get_llm()
            llm_reachable = False
            if llm is not None:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
            llm_info = {
                "available": llm_reachable,
                "configured": llm is not None,
                "backend": llm.name if llm else None,
            }

            # Database stats
            summary = nb.get_dashboard_summary()
            db_info = {
                "path": notebook_path,
                "total_experiments": summary.get("total_experiments", 0),
                "total_programs": summary.get("total_programs_evaluated", 0),
            }

            return jsonify({
                "cuda": {"available": cuda_available, **cuda_info},
                "llm": llm_info,
                "database": db_info,
                "is_running": runner.is_running,
            })
        except Exception as e:
            logger.error(f"Error in /api/system/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/validate", methods=["POST"])
    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = body.get("n", 5)
        n = min(n, 20)  # cap at 20

        try:
            from ..synthesis.grammar import GrammarConfig, batch_generate
            from ..synthesis.compiler import compile_model
            from ..synthesis.validator import validate_graph
            from ..eval.sandbox import safe_eval

            grammar = GrammarConfig(model_dim=256, max_depth=8, max_ops=12)
            graphs = batch_generate(n, grammar)

            generated = len(graphs)
            compiled = 0
            passed_s0 = 0
            errors = []

            for graph in graphs:
                val = validate_graph(graph)
                if not val.valid:
                    errors.append(f"validation: {val.errors[0] if val.errors else 'unknown'}")
                    continue

                try:
                    model = compile_model(
                        [graph] * 2,
                        vocab_size=1000,
                        max_seq_len=128,
                    )
                    compiled += 1

                    result = safe_eval(model, batch_size=1, seq_len=64,
                                       vocab_size=1000, device="cpu")
                    if result.passed:
                        passed_s0 += 1
                    else:
                        errors.append(f"sandbox: {result.error or 'failed'}")
                    del model
                except Exception as e:
                    errors.append(f"compile: {str(e)[:60]}")

            healthy = compiled > 0 and passed_s0 > 0
            return jsonify({
                "generated": generated,
                "compiled": compiled,
                "passed_s0": passed_s0,
                "errors": errors[:5],
                "healthy": healthy,
            })
        except Exception as e:
            logger.error(f"Error in pipeline validation: {e}")
            return jsonify({
                "generated": 0,
                "compiled": 0,
                "passed_s0": 0,
                "errors": [str(e)],
                "healthy": False,
            })

    # ── Campaign endpoints ──

    @app.route("/api/campaigns")
    def api_campaigns():
        """List all campaigns with summary stats."""
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC"
            ).fetchall()
            campaigns = []
            for r in rows:
                d = dict(r)
                # Add summary stats
                d["n_experiments"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_hypotheses"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_decisions"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                campaigns.append(d)
            return jsonify(campaigns)
        except Exception as e:
            logger.error(f"Error in /api/campaigns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>")
    def api_campaign_detail(campaign_id):
        """Full campaign detail with experiments, hypotheses, decisions."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404
            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = _normalize_hypotheses(nb.get_campaign_hypotheses(campaign_id))
            decisions = nb.get_campaign_decisions(campaign_id)
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )
            return jsonify({
                "campaign": campaign,
                "experiments": experiments,
                "hypotheses": hypotheses,
                "decisions": decisions,
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/report")
    def api_campaign_report(campaign_id):
        """Compiled campaign report (LLM-generated narrative)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404

            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = _normalize_hypotheses(nb.get_campaign_hypotheses(campaign_id))
            decisions = nb.get_campaign_decisions(campaign_id)
            knowledge = nb.get_knowledge()
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )

            from .llm.context import build_campaign_report_context
            context = build_campaign_report_context(
                campaign, experiments, hypotheses, decisions, knowledge)
            report = aria.compile_campaign_report(
                campaign, experiments, hypotheses, decisions, knowledge,
                context=context)

            return jsonify({
                "campaign": campaign,
                "report": report,
                "stats": {
                    "n_experiments": len(experiments),
                    "n_hypotheses": len(hypotheses),
                    "n_confirmed": sum(1 for h in hypotheses if h.get("status") == "confirmed"),
                    "n_refuted": sum(1 for h in hypotheses if h.get("status") == "refuted"),
                    "n_decisions": len(decisions),
                },
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}/report: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/hypotheses")
    def api_campaign_hypotheses(campaign_id):
        """Hypothesis chain for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            return jsonify(hypotheses)
        except Exception as e:
            logger.error(f"Error in campaign hypotheses: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/decisions")
    def api_campaign_decisions(campaign_id):
        """Decision log for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            decisions = nb.get_campaign_decisions(campaign_id)
            return jsonify(decisions)
        except Exception as e:
            logger.error(f"Error in campaign decisions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns", methods=["POST"])
    def api_create_campaign():
        """Create a new campaign manually."""
        body = request.get_json(silent=True) or {}
        title = body.get("title", "")
        objective = body.get("objective", "")
        success_criteria = body.get("success_criteria", "")

        if not title or not objective or not success_criteria:
            return jsonify({"error": "title, objective, and success_criteria required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            campaign_id = nb.create_campaign(
                title=title, objective=objective,
                success_criteria=success_criteria,
                parent_id=body.get("parent_campaign_id"),
            )
            return jsonify({
                "campaign_id": campaign_id,
                "status": "created",
            })
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/pause", methods=["POST"])
    def api_pause_campaign(campaign_id):
        """Pause a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            nb.update_campaign(campaign_id, status="paused")
            return jsonify({"status": "paused"})
        except Exception as e:
            logger.error(f"Error pausing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/complete", methods=["POST"])
    def api_complete_campaign(campaign_id):
        """Complete a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            nb.update_campaign(campaign_id, status="completed",
                               completed_at=time.time())
            runner = _get_runner(notebook_path)
            runner._emit_event("campaign_completed", {
                "campaign_id": campaign_id,
                "title": (campaign or {}).get("title", ""),
            })
            return jsonify({"status": "completed"})
        except Exception as e:
            logger.error(f"Error completing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Hypothesis endpoints ──

    @app.route("/api/hypotheses/<hypothesis_id>/chain")
    def api_hypothesis_chain(hypothesis_id):
        """Hypothesis lineage chain."""
        nb = LabNotebook(notebook_path)
        try:
            chain = nb.get_hypothesis_chain(hypothesis_id)
            return jsonify(chain)
        except Exception as e:
            logger.error(f"Error in hypothesis chain: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Knowledge base endpoints ──

    @app.route("/api/knowledge")
    def api_knowledge():
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_knowledge(category=category)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in /api/knowledge: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/search")
    def api_knowledge_search():
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.search_knowledge(q)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    return app


def _setup_logging(log_dir: Optional[str] = None):
    """Configure logging with console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent)
    log_path = Path(log_dir) / "aria_dashboard.log"
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")


def run_server(
    notebook_path: str = "research/lab_notebook.db",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    _setup_logging()
    app = create_app(notebook_path)
    logger.info(f"Starting Aria's Dashboard API on http://{host}:{port}")
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
