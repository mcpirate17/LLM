"""Control/experiment mutation route registration for scientist API."""
from __future__ import annotations

from ..json_utils import json_safe as _json_safe

from .deps import ApiRouteContext, install_legacy_symbols

def register_control_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)
    @app.route("/api/experiments/preflight", methods=["POST"])
    def api_preflight_experiment():
        """Run preflight checks without launching an experiment."""
        runner = _get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        mode = _normalize_start_mode(body.pop("mode", "single"))
        sample_n = int(body.pop("preflight_sample_n", body.pop("sample_n", 4)) or 4)
        config = RunConfig.from_dict(body) if body else RunConfig()
        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )
        preflight = _run_launch_preflight(
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

    @app.route("/api/worker/evaluate", methods=["POST"])
    def api_worker_evaluate():
        """Z12: Distributed worker endpoint for evaluating a computation graph."""
        runner = _get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        
        graph_json = body.get("graph_json")
        config_dict = body.get("config")
        seed = body.get("seed", 42)
        
        if not graph_json or not config_dict:
            return jsonify({"error": "Missing graph_json or config"}), 400
            
        try:
            from ..synthesis.graph import json_to_graph
            from ..synthesis.compiler import compile_model
            import torch
            
            graph = json_to_graph(graph_json)
            config = RunConfig.from_dict(config_dict)
            
            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)
            
            # Compile model locally on worker
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            ).to(dev)
            
            # Use the runner's async-friendly training method
            result = runner._micro_train_async(model, config, seed, dev)
            
            return jsonify({
                "status": "ok",
                "result": result,
                "device": dev_str,
                "worker_id": os.environ.get("ARIA_WORKER_ID", "anonymous")
            })
            
        except Exception as e:
            logger.error("Worker evaluation failed: %s", e)
            return jsonify({"error": str(e), "passed": False}), 500

    @app.route("/api/experiments/start", methods=["POST"])
    def api_start_experiment():
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = _get_runner(notebook_path)

        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        preflight_override = bool(body.pop("preflight_override", False))
        enforce_preflight = bool(body.pop("enforce_preflight", True))
        preflight_sample_n = int(body.pop("preflight_sample_n", 4) or 4)
        hypothesis = body.pop("hypothesis", None)
        preregistration = body.pop("preregistration", None)
        exploratory = bool(body.pop("exploratory", False))
        refine_analysis_json = body.pop("refine_analysis_json", "")
        mode = _normalize_start_mode(body.pop("mode", "single"))
        if mode in {"investigation", "validation"}:
            requested_ids = _normalize_result_ids(body.get("result_ids", []))
            if not requested_ids:
                return jsonify({"error": f"result_ids required for {mode} mode"}), 400
        if mode in {"scale_up", "refine_fingerprint"}:
            requested_ids = _normalize_result_ids(body.get("result_ids", []))
            requested_fps = _normalize_result_ids(
                body.get("graph_fingerprints", body.get("fingerprints", []))
            )
            if not requested_ids and not requested_fps:
                payload = {
                    "error": f"result_ids or graph_fingerprints required for {mode} mode",
                }
                if mode == "refine_fingerprint":
                    payload["refine_resolution"] = {"result_ids": [], "missing_graph_fingerprints": []}
                else:
                    payload["scale_up_resolution"] = {"result_ids": [], "missing_graph_fingerprints": []}
                return jsonify(payload), 400

        config = RunConfig.from_dict(body) if body else RunConfig()
        if refine_analysis_json:
            config.refine_analysis_json = (
                refine_analysis_json if isinstance(refine_analysis_json, str)
                else json.dumps(refine_analysis_json)
            )
        compact_changes: Dict[str, Any] = {}
        sparse_morph_changes: Dict[str, Any] = {}
        if mode == "compact_synthesis":
            compact_changes = _apply_compact_synthesis_bias(config)
            mode = "single"
        if mode == "sparse_morph":
            sparse_morph_changes = _apply_sparse_morph_bias(config)
            mode = "single"

        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )
        preflight = _run_launch_preflight(
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

        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

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
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for investigation mode"}), 400
                force_reinvestigate = bool(body.get("force") or body.get("force_reinvestigate"))
                if not force_reinvestigate:
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
                exp_id = runner.start_investigation(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                    force=force_reinvestigate,
                )
            elif mode == "validation":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
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
                        eligibility = _build_start_mode_eligibility(nb, "validation", result_ids)
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
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = _normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    scale_up_resolution = _resolve_scale_up_result_ids(
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
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = _normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    refine_resolution = _resolve_scale_up_result_ids(
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

            _record_run_trigger(
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
            missing_fields = _extract_hypothesis_missing_fields(critique)

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

    @app.route("/api/experiments/<experiment_id>/rerun", methods=["POST"])
    def api_rerun_experiment(experiment_id):
        """Relaunch an experiment using its stored config and mode."""
        runner = _get_runner(notebook_path)
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

            # If it is still marked running from a stale reboot state, mark it cancelled first.
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
                # Fallback to single synthesis-style rerun.
                new_id = runner.start_experiment(config, hypothesis=hypothesis)
                mode = "single"

            _record_run_trigger(
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

    # ── Batch rerun state ────────────────────────────────────────────
    _batch_rerun_state: Dict[str, Any] = {
        "active": False,
        "total": 0,
        "completed": 0,
        "current": None,
        "remaining": [],
        "results": [],
    }

    @app.route("/api/experiments/batch-rerun", methods=["POST"])
    def api_batch_rerun():
        """Queue multiple experiments for sequential rerun."""
        data = request.get_json(silent=True) or {}
        experiment_ids = data.get("experiment_ids", [])
        if not experiment_ids or not isinstance(experiment_ids, list):
            return jsonify({"error": "experiment_ids must be a non-empty list"}), 400

        if _batch_rerun_state["active"]:
            return jsonify({"error": "A batch rerun is already in progress"}), 409

        runner = _get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        # Validate all experiment IDs exist before starting
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

        _batch_rerun_state.update({
            "active": True,
            "total": len(experiment_ids),
            "completed": 0,
            "current": first_id,
            "remaining": queue,
            "results": [],
        })

        def _run_single(eid):
            """Rerun a single experiment, return new_id or None on error."""
            r = _get_runner(notebook_path)
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

                _record_run_trigger(
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
                # Run the first experiment
                new_id = _run_single(first_id)
                _batch_rerun_state["results"].append({
                    "source_id": first_id,
                    "new_id": new_id,
                    "ok": new_id is not None,
                })

                for next_id in list(_batch_rerun_state["remaining"]):
                    # Wait for current experiment to finish
                    r = _get_runner(notebook_path)
                    while r.is_running:
                        time.sleep(5)

                    _batch_rerun_state["completed"] += 1
                    _batch_rerun_state["current"] = next_id
                    _batch_rerun_state["remaining"] = [
                        x for x in _batch_rerun_state["remaining"] if x != next_id
                    ]

                    new_id = _run_single(next_id)
                    _batch_rerun_state["results"].append({
                        "source_id": next_id,
                        "new_id": new_id,
                        "ok": new_id is not None,
                    })

                # Wait for the last one to finish
                r = _get_runner(notebook_path)
                while r.is_running:
                    time.sleep(5)
                _batch_rerun_state["completed"] += 1

            except Exception as e:
                logger.error(f"Batch rerun worker error: {e}\n{traceback.format_exc()}")
            finally:
                _batch_rerun_state["active"] = False
                _batch_rerun_state["current"] = None
                _batch_rerun_state["remaining"] = []

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
            "active": _batch_rerun_state["active"],
            "total": _batch_rerun_state["total"],
            "completed": _batch_rerun_state["completed"],
            "current": _batch_rerun_state["current"],
            "remaining": _batch_rerun_state["remaining"],
            "results": _batch_rerun_state["results"],
        })

    @app.route("/api/experiments/batch-rerun/cancel", methods=["POST"])
    def api_batch_rerun_cancel():
        """Cancel remaining batch reruns. Current experiment keeps running."""
        if not _batch_rerun_state["active"]:
            return jsonify({"status": "no_batch_active"})
        cancelled = list(_batch_rerun_state["remaining"])
        _batch_rerun_state["remaining"] = []
        return jsonify({
            "status": "cancelled",
            "cancelled_count": len(cancelled),
            "completed_so_far": _batch_rerun_state["completed"],
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

    @app.route("/api/programs/purge-junk", methods=["POST"])
    def api_purge_junk_programs():
        """Purge Stage 0 failure program results that carry no useful data."""
        dry_run = True
        if request.is_json and request.json:
            dry_run = request.json.get("dry_run", True)
        nb = LabNotebook(notebook_path)
        try:
            result = nb.purge_junk_programs(dry_run=dry_run)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error purging junk programs: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/progress")
    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = _get_runner(notebook_path)
        progress_payload = _with_native_runner_progress(runner.progress.to_dict())
        trigger = _get_run_trigger_snapshot(progress_payload.get("experiment_id"))
        progress_payload["run_trigger_source"] = trigger.get("source")
        progress_payload["run_trigger"] = trigger
        return jsonify({
            "is_running": runner.is_running,
            "progress": progress_payload,
            "native_runner": progress_payload.get("native_runner"),
            "run_trigger_source": trigger.get("source"),
            "run_trigger": trigger,
        })

    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = _get_runner(notebook_path)
        sse_timeout = _get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = json.dumps(
                        _json_safe(event.get("data", {})),
                        allow_nan=False,
                    )
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

    @app.route("/api/diagnostics/fingerprint")
    def api_fingerprint_diagnostics():
        """Expose lightweight runtime diagnostics for fingerprint analysis."""
        reset = str(request.args.get("reset", "0")).strip().lower() in {"1", "true", "yes"}
        try:
            from research.eval.fingerprint import get_sensitivity_skip_stats

            stats = get_sensitivity_skip_stats(reset=reset)
            return jsonify({
                "sensitivity_skips": stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/fingerprint: {e}")
            return jsonify({
                "sensitivity_skips": {
                    "total": 0,
                    "by_reason": {},
                },
                "error": str(e),
            }), 500

    @app.route("/api/diagnostics/report-cache")
    def api_report_cache_diagnostics():
        """Expose report snapshot cache usage and retention diagnostics."""
        nb = LabNotebook(notebook_path)
        try:
            cleanup = str(request.args.get("cleanup", "0")).strip().lower() in {"1", "true", "yes"}
            try:
                ttl_seconds = int(os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)))
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400"))
            except Exception:
                max_rows_per_scope = 400

            cleanup_stats = None
            if cleanup:
                cleanup_stats = nb.cleanup_report_snapshots(
                    ttl_seconds=max(60, ttl_seconds),
                    max_rows_per_scope=max(20, max_rows_per_scope),
                )

            snapshot_stats = nb.get_report_snapshot_stats()
            return jsonify({
                "snapshot_cache": snapshot_stats,
                "retention": {
                    "ttl_seconds": max(60, int(ttl_seconds or 0)),
                    "max_rows_per_scope": max(20, int(max_rows_per_scope or 0)),
                },
                "cleanup_triggered": bool(cleanup),
                "cleanup": cleanup_stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/report-cache: {e}")
            return jsonify({
                "snapshot_cache": {
                    "total_snapshots": 0,
                    "n_scopes": 0,
                    "oldest_age_seconds": None,
                    "newest_age_seconds": None,
                    "scopes": [],
                },
                "error": str(e),
            }), 500
        finally:
            nb.close()

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

    def _sparse_coverage_summary(sparse_coverage_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        summary = sparse_coverage_data or {}
        sparse_share = summary.get("sparse_share")
        sparse_survival_rate = summary.get("sparse_survival_rate")
        target_share = 0.15
        sparse_share_value = float(sparse_share) if isinstance(sparse_share, (int, float)) else None
        sparse_survival_value = float(sparse_survival_rate) if isinstance(sparse_survival_rate, (int, float)) else None
        below_target = bool(sparse_share_value is not None and sparse_share_value < target_share)
        return {
            "sparse_share": sparse_share_value,
            "sparse_survival_rate": sparse_survival_value,
            "target_share": target_share,
            "below_target": below_target,
        }

    def _augment_sparse_action_config(
        suggested_config: Optional[Dict[str, Any]],
        normalized_mode: Optional[str],
        sparse_coverage_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        config = dict(suggested_config or {})
        sparse_summary = _sparse_coverage_summary(sparse_coverage_data)
        if not sparse_summary.get("below_target"):
            return config

        mode = str(normalized_mode or config.get("mode") or "").strip().lower()
        if mode not in {"novelty", "evolve", "continuous", "single", "synthesis"}:
            return config

        config.setdefault("model_source", "mixed")
        config.setdefault("morph_focus_sparse", True)
        config.setdefault("morph_ratio", 0.8)
        config.setdefault("use_synthesized_training", True)
        config.setdefault("morph_sparse_weight_storage", "semi_structured_2_4")
        config.setdefault("math_space_weight", 2.2)
        config.setdefault("n_programs", 120)
        if mode in {"novelty", "evolve"}:
            config.setdefault("max_depth", 6)
            config.setdefault("max_ops", 10)
        return config

