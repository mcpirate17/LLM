"""experiments API route registration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..code_agent import _should_autospawn_self_repair, _spawn_code_agent_task
from .deps import ApiRouteContext
from ._utils import with_notebook_context
from ..runner._types import RunConfig
from ._experiment_launch import (
    build_start_error_response,
    build_start_success_response,
    create_launch_lifecycle_context,
    launch_experiment_mode,
    load_rerun_source,
    maybe_block_preflight,
    parse_start_request,
    publish_launch_failed,
    publish_launch_requested,
    start_batch_rerun,
    start_rerun_from_source,
)
from ._helpers import (
    get_runner,
    _BATCH_RERUN_STATE,
)
from ._strategy_preflight import (
    normalize_start_mode,
    run_launch_preflight,
    apply_live_screening_bias,
)

logger = logging.getLogger(__name__)


def _extract_ops_summary(graph_json_str: str | None) -> str | None:
    """Extract unique op names from a graph_json string for display."""
    if not graph_json_str:
        return None
    try:
        graph = (
            json.loads(graph_json_str)
            if isinstance(graph_json_str, str)
            else graph_json_str
        )
        nodes = graph.get("nodes", {})
        ops = sorted(
            {
                n["op_name"]
                for n in (nodes.values() if isinstance(nodes, dict) else nodes)
                if isinstance(n, dict) and n.get("op_name") and not n.get("is_input")
            }
        )
        if not ops:
            return None
        if len(ops) <= 5:
            return ", ".join(ops)
        return ", ".join(ops[:4]) + f" +{len(ops) - 4}"
    except Exception as exc:
        logger.debug("Returning default due to error: %s", exc)
        return None


def _is_backfill_experiment_type(experiment_type: Any) -> bool:
    if not experiment_type:
        return False
    return "backfill" in str(experiment_type).lower()


def _get_cached_experiment_analysis(exp: Dict[str, Any]) -> Optional[str]:
    stored = exp.get("llm_analysis")
    return stored if stored else None


def _generate_experiment_analysis(
    nb, experiment_id: str, exp: Dict[str, Any]
) -> Optional[str]:
    from ..llm.context_experiment import build_experiment_context
    from ._helpers import get_aria_for_notebook

    results = exp.get("results") or {}
    analysis = get_aria_for_notebook(str(nb.db_path)).analyze_results(
        results,
        context=build_experiment_context(results),
    )
    if not analysis:
        return None
    nb.conn.execute(
        "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
        (analysis, experiment_id),
    )
    nb.conn.commit()
    return analysis


def _experiment_delete_impact(nb, experiment_id: str) -> Dict[str, int]:
    """Return counts that make experiment deletion unsafe by default."""
    row = nb.conn.execute(
        """
        SELECT
            COUNT(*) AS program_results,
            COALESCE(SUM(CASE WHEN COALESCE(stage1_passed, 0) = 1 THEN 1 ELSE 0 END), 0)
                AS stage1_results,
            COALESCE(SUM(CASE
                WHEN COALESCE(induction_auc, 0) > 0
                  OR COALESCE(binding_auc, 0) > 0
                  OR COALESCE(induction_v2_investigation_auc, 0) > 0
                  OR COALESCE(binding_v2_investigation_auc, 0) > 0
                  OR COALESCE(hellaswag_acc, 0) > 0
                  OR COALESCE(blimp_overall_accuracy, 0) > 0
                THEN 1 ELSE 0 END), 0) AS diagnostic_results
        FROM program_results
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    result = {
        "program_results": int(row["program_results"] or 0) if row else 0,
        "stage1_results": int(row["stage1_results"] or 0) if row else 0,
        "diagnostic_results": int(row["diagnostic_results"] or 0) if row else 0,
    }
    lb_row = nb.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM leaderboard l
        JOIN program_results pr ON pr.result_id = l.result_id
        WHERE pr.experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    result["leaderboard_rows"] = int(lb_row["n"] or 0) if lb_row else 0
    tc_row = nb.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM training_curves tc
        JOIN program_results pr ON pr.result_id = tc.result_id
        WHERE pr.experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    result["training_curve_rows"] = int(tc_row["n"] or 0) if tc_row else 0
    return result


def register_experiments_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/experiments")
    @wnb
    def api_experiments(nb=None):
        """List experiments (newest first)."""
        n = request.args.get("n", type=int)
        if n is None:
            n = request.args.get("limit", type=int)
        if n is None:
            n = 200
        n = max(1, min(n, 5000))
        offset = request.args.get("offset", 0, type=int)
        offset = max(0, min(offset, 1_000_000))
        return jsonify(nb.get_recent_experiments(n, offset=offset))

    @app.route("/api/experiments/<experiment_id>")
    @wnb
    def api_experiment_detail(experiment_id, nb=None):
        """Get experiment details with entries and per-experiment programs."""
        exp = nb.get_experiment(experiment_id)
        if exp is None:
            return jsonify({"error": "Not found"}), 404
        entries = nb.get_entries(experiment_id=experiment_id)
        programs = nb.get_program_results(experiment_id)
        # Add compact op summary, strip heavy graph_json from list response
        for p in programs:
            p["ops_summary"] = _extract_ops_summary(p.get("graph_json"))
            p.pop("graph_json", None)
        prereg = nb.get_preregistration_for_experiment(experiment_id)
        deviations = nb.get_preregistration_deviations(experiment_id)
        payload = {
            "experiment": exp,
            "entries": entries,
            "programs": programs,
            "preregistration": prereg,
            "preregistration_deviations": deviations,
        }
        return jsonify(_json_safe(payload))

    @app.route("/api/experiments/<experiment_id>/programs")
    @wnb
    def api_experiment_programs(experiment_id, nb=None):
        """All programs for an experiment (not just S1 survivors)."""
        programs = nb.get_program_results(experiment_id)
        for p in programs:
            p["ops_summary"] = _extract_ops_summary(p.get("graph_json"))
            p.pop("graph_json", None)
        return jsonify(_json_safe(programs))

    @app.route("/api/experiments/<experiment_id>/failures")
    @wnb
    def api_failure_analysis(experiment_id, nb=None):
        """Failure analysis: error distribution, stage funnel."""
        analysis = nb.get_failure_analysis(experiment_id)
        return jsonify(analysis)

    @app.route("/api/experiments/<experiment_id>/analysis")
    @wnb
    def api_experiment_analysis(experiment_id, nb=None):
        """Stored experiment analysis."""
        exp = nb.get_experiment(experiment_id)
        if exp is None:
            return jsonify({"error": "Not found"}), 404

        stored = _get_cached_experiment_analysis(exp)
        if stored:
            return jsonify({"analysis": stored, "source": "stored"})

        if _is_backfill_experiment_type(exp.get("experiment_type")):
            return jsonify(
                {
                    "analysis": None,
                    "source": "unavailable",
                    "reason": "Backfill experiments skip LLM analysis",
                }
            )

        return jsonify(
            {
                "analysis": None,
                "source": "unavailable",
                "reason": "No cached analysis. Use POST /api/experiments/<id>/analysis to generate one.",
            }
        )

    @app.route("/api/experiments/<experiment_id>/analysis", methods=["POST"])
    @wnb
    def api_experiment_analysis_refresh(experiment_id, nb=None):
        """Generate or refresh experiment analysis explicitly."""
        exp = nb.get_experiment(experiment_id)
        if exp is None:
            return jsonify({"error": "Not found"}), 404
        if _is_backfill_experiment_type(exp.get("experiment_type")):
            return jsonify(
                {
                    "analysis": None,
                    "source": "unavailable",
                    "reason": "Backfill experiments skip LLM analysis",
                }
            )

        body = request.get_json(silent=True) or {}
        force = bool(body.get("force", False))
        if not force:
            stored = _get_cached_experiment_analysis(exp)
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

        try:
            analysis = _generate_experiment_analysis(nb, experiment_id, exp)
        except Exception as exc:
            logger.warning(
                "Experiment analysis generation failed for %s: %s",
                experiment_id,
                exc,
            )
            return jsonify(
                {
                    "analysis": None,
                    "source": "unavailable",
                    "reason": str(exc),
                }
            ), 503

        if analysis:
            return jsonify({"analysis": analysis, "source": "generated"})

        return jsonify(
            {
                "analysis": None,
                "source": "unavailable",
                "reason": "No LLM backend configured",
            }
        )

    @app.route("/api/experiments/preflight", methods=["POST"])
    def api_preflight_experiment():
        """Run preflight checks without launching an experiment."""
        runner = get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        mode = normalize_start_mode(body.pop("mode", "single"))
        sample_n = int(body.pop("preflight_sample_n", body.pop("sample_n", 4)) or 4)
        config = RunConfig.from_dict(body) if body else RunConfig()
        if mode == "live_screening":
            apply_live_screening_bias(config)
            mode = "single"
        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )
        preflight = run_launch_preflight(
            config=config,
            mode=mode,
            prescreen=prescreen,
            notebook_path=notebook_path,
            sample_n=sample_n,
        )
        return jsonify(
            {
                "status": "ok",
                "mode": mode,
                "config": config.to_dict(),
                "prescreen": prescreen,
                "preflight": preflight,
                "can_start_without_override": preflight.get("verdict") == "pass",
            }
        )

    @app.route("/api/experiments/start", methods=["POST"])
    @wnb
    def api_start_experiment(nb=None):
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        start = parse_start_request(request.get_json(silent=True) or {})
        config, prescreen = runner.prescreen_run_config(
            start.config,
            mode=start.mode,
            auto_harden=start.auto_harden,
        )
        preflight = run_launch_preflight(
            config=config,
            mode=start.mode,
            prescreen=prescreen,
            notebook_path=notebook_path,
            sample_n=start.preflight_sample_n,
        )
        blocked = maybe_block_preflight(start, prescreen, preflight)
        if blocked is not None:
            return blocked

        launch_context = create_launch_lifecycle_context()
        try:
            publish_launch_requested(
                start,
                notebook_path=notebook_path,
                context=launch_context,
            )
            exp_id, eligibility, scale_up_resolution, refine_resolution, error = (
                launch_experiment_mode(start, nb=nb, runner=runner)
            )
            if error is not None:
                return error
            start.config = config
            return build_start_success_response(
                start,
                exp_id,
                runner=runner,
                prescreen=prescreen,
                preflight=preflight,
                eligibility=eligibility,
                scale_up_resolution=scale_up_resolution,
                refine_resolution=refine_resolution,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            publish_launch_failed(
                e,
                mode=start.mode,
                notebook_path=notebook_path,
                context=launch_context,
            )
            return build_start_error_response(
                e,
                mode=start.mode,
                notebook_path=notebook_path,
                runner=runner,
                should_autospawn_self_repair=_should_autospawn_self_repair,
                spawn_code_agent_task=_spawn_code_agent_task,
            )

    @app.route("/api/experiments/stop", methods=["POST"])
    def api_stop_experiment():
        """Stop the currently running experiment."""
        runner = get_runner(notebook_path)
        if not runner.is_running:
            return jsonify({"error": "No experiment is running"}), 409

        runner.stop()
        return jsonify(
            {
                "status": "stopping",
                "aria_message": runner.progress.aria_message,
            }
        )

    @app.route("/api/experiments/<experiment_id>/cancel", methods=["POST"])
    @wnb
    def api_cancel_experiment(experiment_id, nb=None):
        """Cancel a stuck/running experiment by marking it as failed."""
        cancelled = nb.cancel_experiment(experiment_id)
        if not cancelled:
            return jsonify(
                {
                    "error": "Experiment not found or not in running state",
                }
            ), 404
        return jsonify({"status": "cancelled", "experiment_id": experiment_id})

    @app.route("/api/experiments/<experiment_id>", methods=["DELETE"])
    @wnb
    def api_delete_experiment(experiment_id, nb=None):
        """Delete an experiment and all its child rows."""
        exp = nb.get_experiment(experiment_id)
        if exp is None:
            return jsonify({"error": "Experiment not found"}), 404
        if str(exp.get("status", "")).strip().lower() == "running":
            return jsonify(
                {"error": "Cannot delete a running experiment — cancel it first"}
            ), 409
        impact = _experiment_delete_impact(nb, experiment_id)
        if any(impact.values()):
            return jsonify(
                {
                    "error": (
                        "Refusing to delete a non-empty experiment. "
                        "Program results, leaderboard rows, training curves, "
                        "and diagnostic metrics are preserved by default."
                    ),
                    "experiment_id": experiment_id,
                    "delete_impact": impact,
                }
            ), 409
        nb._delete_experiment_cascade(experiment_id)
        return jsonify({"status": "deleted", "experiment_id": experiment_id})

    @app.route("/api/experiments/<experiment_id>/rerun", methods=["POST"])
    @wnb
    def api_rerun_experiment(experiment_id, nb=None):
        """Relaunch an experiment using its stored config and mode."""
        runner = get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        try:
            source = load_rerun_source(nb, experiment_id)
            if source is None:
                return jsonify({"error": "Experiment not found"}), 404

            if str(source.get("status") or "").strip().lower() == "running":
                nb.cancel_experiment(experiment_id)

            new_id, mode, config = start_rerun_from_source(
                source=source,
                experiment_id=experiment_id,
                notebook_path=notebook_path,
            )
            return jsonify(
                {
                    "status": "started",
                    "source_experiment_id": experiment_id,
                    "experiment_id": new_id,
                    "mode": mode,
                    "config": config.to_dict(),
                }
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/experiments/batch-rerun", methods=["POST"])
    @wnb
    def api_batch_rerun(nb=None):
        """Queue multiple experiments for sequential rerun."""
        data = request.get_json(silent=True) or {}
        experiment_ids = data.get("experiment_ids", [])
        if not experiment_ids or not isinstance(experiment_ids, list):
            return jsonify({"error": "experiment_ids must be a non-empty list"}), 400

        if _BATCH_RERUN_STATE["active"]:
            return jsonify({"error": "A batch rerun is already in progress"}), 409

        runner = get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        for eid in experiment_ids:
            exp = load_rerun_source(nb, eid)
            if exp is None:
                return jsonify({"error": f"Experiment {eid} not found"}), 404

        return jsonify(start_batch_rerun(experiment_ids, notebook_path=notebook_path))

    @app.route("/api/experiments/batch-rerun/status", methods=["GET"])
    def api_batch_rerun_status():
        """Poll batch rerun progress."""
        return jsonify(
            {
                "active": _BATCH_RERUN_STATE["active"],
                "total": _BATCH_RERUN_STATE["total"],
                "completed": _BATCH_RERUN_STATE["completed"],
                "current": _BATCH_RERUN_STATE["current"],
                "remaining": _BATCH_RERUN_STATE["remaining"],
                "results": _BATCH_RERUN_STATE["results"],
            }
        )

    @app.route("/api/experiments/batch-rerun/cancel", methods=["POST"])
    def api_batch_rerun_cancel():
        """Cancel remaining batch reruns. Current experiment keeps running."""
        if not _BATCH_RERUN_STATE["active"]:
            return jsonify({"status": "no_batch_active"})
        cancelled = list(_BATCH_RERUN_STATE["remaining"])
        _BATCH_RERUN_STATE["remaining"] = []
        return jsonify(
            {
                "status": "cancelled",
                "cancelled_count": len(cancelled),
                "completed_so_far": _BATCH_RERUN_STATE["completed"],
            }
        )

    @app.route("/api/experiments/<experiment_id>/fill-gaps", methods=["POST"])
    @wnb
    def api_fill_experiment_gaps(experiment_id, nb=None):
        """Backfill missing summary metrics for an existing experiment row."""
        result = nb.backfill_experiment_metrics(experiment_id)
        if not result.get("found"):
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(
            {
                "status": "ok",
                "experiment_id": experiment_id,
                **result,
            }
        )

    @app.route("/api/experiments/cleanup-stale", methods=["POST"])
    @wnb
    def api_cleanup_stale(nb=None):
        """Clean up stale running experiments that are no longer active."""
        count = nb.cleanup_stale_experiments()
        return jsonify({"cleaned": count})
