"""experiments API route registration."""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from typing import Any, Dict, Optional

from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from ..runner import RunConfig
from ..persona import get_aria
from ..code_agent import _should_autospawn_self_repair, _spawn_code_agent_task
from ._helpers import (
    get_runner, normalize_result_ids, record_run_trigger,
    _BATCH_RERUN_STATE,
)
from ._strategy_preflight import (
    normalize_start_mode, run_launch_preflight,
    apply_compact_synthesis_bias, apply_sparse_morph_bias,
    extract_hypothesis_missing_fields,
    build_start_mode_eligibility, resolve_scale_up_result_ids,
)
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_experiments_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/experiments")
    def api_experiments():
        """List experiments (newest first)."""
        n = request.args.get("n", type=int)
        if n is None:
            n = request.args.get("limit", type=int)
        if n is None:
            n = 200
        n = max(1, min(n, 5000))
        offset = request.args.get("offset", 0, type=int)
        offset = max(0, min(offset, 1_000_000))
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n, offset=offset))
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
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}/programs: {e}")
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

            stored = exp.get("llm_analysis")
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

            results = exp.get("results") or {}
            from ..llm.context_experiment import build_experiment_context
            ctx = build_experiment_context(results)
            analysis = aria.analyze_results(results, context=ctx)

            if analysis:
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

    @app.route("/api/experiments/preflight", methods=["POST"])
    def api_preflight_experiment():
        """Run preflight checks without launching an experiment."""
        runner = get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        mode = normalize_start_mode(body.pop("mode", "single"))
        sample_n = int(body.pop("preflight_sample_n", body.pop("sample_n", 4)) or 4)
        config = RunConfig.from_dict(body) if body else RunConfig()
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
        return jsonify({
            "status": "ok",
            "mode": mode,
            "config": config.to_dict(),
            "prescreen": prescreen,
            "preflight": preflight,
            "can_start_without_override": preflight.get("verdict") == "pass",
        })

    @app.route("/api/experiments/start", methods=["POST"])
    def api_start_experiment():
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        preflight_override = bool(body.pop("preflight_override", False))
        enforce_preflight = bool(body.pop("enforce_preflight", True))
        preflight_sample_n = int(body.pop("preflight_sample_n", 4) or 4)
        hypothesis = body.pop("hypothesis", None)
        preregistration = body.pop("preregistration", None)
        exploratory = bool(body.pop("exploratory", False))
        refine_analysis_json = body.pop("refine_analysis_json", "")
        mode = normalize_start_mode(body.pop("mode", "single"))

        config = RunConfig.from_dict(body) if body else RunConfig()
        if refine_analysis_json:
            config.refine_analysis_json = (
                refine_analysis_json if isinstance(refine_analysis_json, str)
                else json.dumps(refine_analysis_json)
            )
        compact_changes: Dict[str, Any] = {}
        sparse_morph_changes: Dict[str, Any] = {}
        if mode == "compact_synthesis":
            compact_changes = apply_compact_synthesis_bias(config)
            mode = "single"
        if mode == "sparse_morph":
            sparse_morph_changes = apply_sparse_morph_bias(config)
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
            sample_n=preflight_sample_n,
        )
        if enforce_preflight and preflight.get("verdict") in {"warn", "fail"} and not preflight_override:
            return jsonify({
                "error": (
                    "Preflight gate blocked launch."
                    if preflight.get("verdict") == "fail"
                    else "Preflight produced warnings; override required to start."
                ),
                "preflight_blocked": True,
                "preflight": preflight,
                "config": config.to_dict(),
                "prescreen": prescreen,
            }), 409

        eligibility: Optional[Dict[str, Any]] = None
        scale_up_resolution: Optional[Dict[str, Any]] = None
        refine_resolution: Optional[Dict[str, Any]] = None

        try:
            if mode == "continuous":
                config.continuous = True
                exp_id = runner.start_continuous(config)
            elif mode == "evolve":
                exp_id = runner.start_evolution(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "novelty":
                exp_id = runner.start_novelty_search(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "investigation":
                result_ids = normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for investigation mode"}), 400
                force_reinvestigate = bool(body.get("force") or body.get("force_reinvestigate"))
                if not force_reinvestigate:
                    nb = LabNotebook(notebook_path)
                    try:
                        eligibility = build_start_mode_eligibility(nb, "investigation", result_ids)
                    finally:
                        nb.close()
                    if not eligibility.get("all_eligible"):
                        return jsonify({
                            "error": "Ineligible result_ids for investigation mode",
                            "eligibility": eligibility,
                        }), 409
                exp_id = runner.start_investigation(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                    force=force_reinvestigate,
                )
            elif mode == "validation":
                result_ids = normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for validation mode"}), 400
                force_validation = bool(
                    body.get("force")
                    or body.get("force_validation")
                    or body.get("force_override")
                    or body.get("allow_ineligible")
                    or body.get("override_ineligible")
                )
                if not force_validation:
                    nb = LabNotebook(notebook_path)
                    try:
                        eligibility = build_start_mode_eligibility(nb, "validation", result_ids)
                    finally:
                        nb.close()
                    if not eligibility.get("all_eligible"):
                        return jsonify({
                            "error": "Ineligible result_ids for validation mode",
                            "eligibility": eligibility,
                        }), 409
                exp_id = runner.start_validation(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                    force=force_validation,
                )
            elif mode == "scale_up":
                result_ids = normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    scale_up_resolution = resolve_scale_up_result_ids(
                        nb,
                        result_ids=result_ids,
                        graph_fingerprints=graph_fingerprints,
                    )
                finally:
                    nb.close()
                result_ids = scale_up_resolution.get("result_ids", [])
                if not result_ids:
                    return jsonify({
                        "error": "result_ids or graph_fingerprints required for scale_up mode",
                        "scale_up_resolution": scale_up_resolution,
                    }), 400
                config.scale_up = True
                config.scale_up_result_ids = ",".join(result_ids)
                exp_id = runner.start_scale_up(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "refine_fingerprint":
                result_ids = normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    refine_resolution = resolve_scale_up_result_ids(
                        nb,
                        result_ids=result_ids,
                        graph_fingerprints=graph_fingerprints,
                    )
                finally:
                    nb.close()

                result_ids = refine_resolution.get("result_ids", [])
                if not result_ids:
                    return jsonify({
                        "error": "result_ids or graph_fingerprints required for refine_fingerprint mode",
                        "refine_resolution": refine_resolution,
                    }), 400

                exp_id = runner.start_fingerprint_refinement(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                )
            else:
                exp_id = runner.start_experiment(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )

            record_run_trigger(
                experiment_id=exp_id,
                source="ui_start",
                mode=mode,
                details={
                    "endpoint": "/api/experiments/start",
                    "auto_harden": auto_harden,
                },
            )
            critique = (
                runner.progress.hypothesis_critique
                if isinstance(runner.progress.hypothesis_critique, dict)
                else None
            )
            missing_fields = extract_hypothesis_missing_fields(critique)

            return jsonify({
                "experiment_id": exp_id,
                "status": "started",
                "config": config.to_dict(),
                "prescreen": prescreen,
                "compact_synthesis_bias": compact_changes,
                "sparse_morph_bias": sparse_morph_changes,
                "scale_up_resolution": scale_up_resolution,
                "refine_resolution": refine_resolution,
                "aria_message": runner.progress.aria_message,
                "hypothesis_critique": critique,
                "hypothesis_review_gate": critique.get("gate") if critique else None,
                "hypothesis_missing_fields": missing_fields,
                "preflight": preflight,
                "preflight_override": preflight_override,
                "eligibility": eligibility,
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error starting experiment: {e}\n{traceback.format_exc()}")
            error_text = str(e)
            auto_repair_task: Optional[Dict[str, Any]] = None
            if _should_autospawn_self_repair(error_text):
                try:
                    auto_repair_task = _spawn_code_agent_task(
                        goal=(
                            "Experiment start failed with runtime/code error. "
                            f"mode={mode}, error={error_text}. "
                            "Identify root cause, apply safe code/config fixes, and report validation."
                        ),
                        notebook_path=notebook_path,
                        allow_write=True,
                        session_id="",
                    )
                except Exception as spawn_err:
                    logger.warning("Auto self-repair spawn failed: %s", spawn_err)
            return jsonify({
                "error": error_text,
                "auto_repair_started": bool(auto_repair_task),
                "auto_repair_task": auto_repair_task,
            }), 500

    @app.route("/api/experiments/stop", methods=["POST"])
    def api_stop_experiment():
        """Stop the currently running experiment."""
        runner = get_runner(notebook_path)
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

    @app.route("/api/experiments/<experiment_id>/rerun", methods=["POST"])
    def api_rerun_experiment(experiment_id):
        """Relaunch an experiment using its stored config and mode."""
        runner = get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        nb = LabNotebook(notebook_path)
        try:
            source = nb.get_resumable_experiment(experiment_id)
            if source is None:
                source = nb.get_experiment(experiment_id)
            if source is None:
                return jsonify({"error": "Experiment not found"}), 404

            try:
                config_dict = json.loads(source.get("config_json") or "{}")
            except Exception:
                config_dict = {}
            config = RunConfig.from_dict(config_dict)
            hypothesis = source.get("hypothesis")
            exp_type = str(source.get("experiment_type") or "synthesis").strip().lower()

            if str(source.get("status") or "").strip().lower() == "running":
                nb.cancel_experiment(experiment_id)

            if exp_type == "continuous":
                config.continuous = True
                new_id = runner.start_continuous(config)
                mode = "continuous"
            elif exp_type == "evolution":
                new_id = runner.start_evolution(config, hypothesis=hypothesis)
                mode = "evolve"
            elif exp_type == "novelty":
                new_id = runner.start_novelty_search(config, hypothesis=hypothesis)
                mode = "novelty"
            else:
                new_id = runner.start_experiment(config, hypothesis=hypothesis)
                mode = "single"

            record_run_trigger(
                experiment_id=new_id,
                source="ui_rerun",
                mode=mode,
                details={
                    "endpoint": f"/api/experiments/{experiment_id}/rerun",
                    "source_experiment_id": experiment_id,
                },
            )

            return jsonify({
                "status": "started",
                "source_experiment_id": experiment_id,
                "experiment_id": new_id,
                "mode": mode,
                "config": config.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error rerunning experiment {experiment_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/batch-rerun", methods=["POST"])
    def api_batch_rerun():
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

        nb = LabNotebook(notebook_path)
        try:
            for eid in experiment_ids:
                exp = nb.get_resumable_experiment(eid) or nb.get_experiment(eid)
                if exp is None:
                    return jsonify({"error": f"Experiment {eid} not found"}), 404
        finally:
            nb.close()

        queue = list(experiment_ids)
        first_id = queue.pop(0)

        _BATCH_RERUN_STATE.update({
            "active": True,
            "total": len(experiment_ids),
            "completed": 0,
            "current": first_id,
            "remaining": queue,
            "results": [],
        })

        def _run_single(eid):
            """Rerun a single experiment, return new_id or None on error."""
            r = get_runner(notebook_path)
            nb2 = LabNotebook(notebook_path)
            try:
                source = nb2.get_resumable_experiment(eid) or nb2.get_experiment(eid)
                if source is None:
                    return None
                try:
                    config_dict = json.loads(source.get("config_json") or "{}")
                except Exception:
                    config_dict = {}
                config = RunConfig.from_dict(config_dict)
                hypothesis = source.get("hypothesis")
                exp_type = str(source.get("experiment_type") or "synthesis").strip().lower()

                if str(source.get("status") or "").strip().lower() == "running":
                    nb2.cancel_experiment(eid)

                if exp_type == "continuous":
                    config.continuous = True
                    new_id = r.start_continuous(config)
                elif exp_type == "evolution":
                    new_id = r.start_evolution(config, hypothesis=hypothesis)
                elif exp_type == "novelty":
                    new_id = r.start_novelty_search(config, hypothesis=hypothesis)
                else:
                    new_id = r.start_experiment(config, hypothesis=hypothesis)

                record_run_trigger(
                    experiment_id=new_id,
                    source="ui_batch_rerun",
                    mode=exp_type,
                    details={"source_experiment_id": eid},
                )
                return new_id
            except Exception as e:
                logger.error(f"Batch rerun error for {eid}: {e}\n{traceback.format_exc()}")
                return None
            finally:
                nb2.close()

        def _batch_worker():
            """Background thread: run first, then poll and run remaining."""
            try:
                new_id = _run_single(first_id)
                _BATCH_RERUN_STATE["results"].append({
                    "source_id": first_id,
                    "new_id": new_id,
                    "ok": new_id is not None,
                })

                for next_id in list(_BATCH_RERUN_STATE["remaining"]):
                    r = get_runner(notebook_path)
                    while r.is_running:
                        time.sleep(5)

                    _BATCH_RERUN_STATE["completed"] += 1
                    _BATCH_RERUN_STATE["current"] = next_id
                    _BATCH_RERUN_STATE["remaining"] = [
                        x for x in _BATCH_RERUN_STATE["remaining"] if x != next_id
                    ]

                    new_id = _run_single(next_id)
                    _BATCH_RERUN_STATE["results"].append({
                        "source_id": next_id,
                        "new_id": new_id,
                        "ok": new_id is not None,
                    })

                r = get_runner(notebook_path)
                while r.is_running:
                    time.sleep(5)
                _BATCH_RERUN_STATE["completed"] += 1

            except Exception as e:
                logger.error(f"Batch rerun worker error: {e}\n{traceback.format_exc()}")
            finally:
                _BATCH_RERUN_STATE["active"] = False
                _BATCH_RERUN_STATE["current"] = None
                _BATCH_RERUN_STATE["remaining"] = []

        t = threading.Thread(target=_batch_worker, daemon=True)
        t.start()

        return jsonify({
            "status": "queued",
            "total": len(experiment_ids),
            "started": first_id,
            "queued": queue,
        })

    @app.route("/api/experiments/batch-rerun/status", methods=["GET"])
    def api_batch_rerun_status():
        """Poll batch rerun progress."""
        return jsonify({
            "active": _BATCH_RERUN_STATE["active"],
            "total": _BATCH_RERUN_STATE["total"],
            "completed": _BATCH_RERUN_STATE["completed"],
            "current": _BATCH_RERUN_STATE["current"],
            "remaining": _BATCH_RERUN_STATE["remaining"],
            "results": _BATCH_RERUN_STATE["results"],
        })

    @app.route("/api/experiments/batch-rerun/cancel", methods=["POST"])
    def api_batch_rerun_cancel():
        """Cancel remaining batch reruns. Current experiment keeps running."""
        if not _BATCH_RERUN_STATE["active"]:
            return jsonify({"status": "no_batch_active"})
        cancelled = list(_BATCH_RERUN_STATE["remaining"])
        _BATCH_RERUN_STATE["remaining"] = []
        return jsonify({
            "status": "cancelled",
            "cancelled_count": len(cancelled),
            "completed_so_far": _BATCH_RERUN_STATE["completed"],
        })

    @app.route("/api/experiments/<experiment_id>/fill-gaps", methods=["POST"])
    def api_fill_experiment_gaps(experiment_id):
        """Backfill missing summary metrics for an existing experiment row."""
        nb = LabNotebook(notebook_path)
        try:
            result = nb.backfill_experiment_metrics(experiment_id)
            if not result.get("found"):
                return jsonify({"error": "Experiment not found"}), 404
            return jsonify({
                "status": "ok",
                "experiment_id": experiment_id,
                **result,
            })
        except Exception as e:
            logger.error(f"Error filling gaps for experiment {experiment_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
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
