"""Designer, campaign, knowledge, and proxy route registration."""
from __future__ import annotations

from .deps import ApiRouteContext, install_legacy_symbols

def register_designer_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)
    @app.route("/api/validate", methods=["POST"])
    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = min(int(body.get("n", body.get("sample_n", 5)) or 5), 20)
        mode = _normalize_start_mode(body.pop("mode", "single"))
        auto_harden = bool(body.pop("auto_harden", True))
        runner = _get_runner(notebook_path)
        config = RunConfig.from_dict(body) if body else RunConfig()
        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )

        try:
            sample = _run_pipeline_sample_check(config=config, sample_n=n)
            preflight = _run_launch_preflight(
                config=config,
                mode=mode,
                prescreen=prescreen,
                notebook_path=notebook_path,
                sample_n=n,
            )
            healthy = preflight.get("verdict") != "fail"
            return jsonify({
                "generated": sample.get("generated", 0),
                "compiled": sample.get("compiled", 0),
                "passed_s0": sample.get("passed_s0", 0),
                "errors": sample.get("errors", [])[:5],
                "healthy": healthy,
                "mode": mode,
                "config": config.to_dict(),
                "prescreen": prescreen,
                "preflight": preflight,
            })
        except Exception as e:
            logger.error(f"Error in pipeline validation: {e}")
            return jsonify({
                "generated": 0,
                "compiled": 0,
                "passed_s0": 0,
                "errors": [str(e)],
                "healthy": False,
                "mode": mode,
                "config": config.to_dict(),
                "prescreen": prescreen,
            })

    # ── Designer endpoints (proxy-first → aria_designer API) ──
    #
    # Each endpoint tries the aria_designer backend first via HTTP proxy.
    # If the proxy is unavailable or disabled (ARIA_DESIGNER_PROXY_ENABLED=0),
    # the legacy local implementation is used as fallback.

    @app.route("/api/designer/lifecycle")
    def api_designer_lifecycle():
        """Return current aria_designer service status."""
        payload = _designer_service_status()
        payload.update(_designer_idle_state())
        return jsonify(payload)

    @app.route("/api/designer/ensure-running", methods=["POST"])
    def api_designer_ensure_running():
        """Ensure aria_designer API+UI are running for seamless UX."""
        body = request.get_json(silent=True) or {}
        force_restart = bool(body.get("force_restart", False))
        result = _start_designer_services(force_restart=force_restart)
        if result.get("ok"):
            result.update(_designer_touch_activity("ensure-running"))
        status = 200 if result.get("ok") else 503
        return jsonify(result), status

    @app.route("/api/designer/stop", methods=["POST"])
    def api_designer_stop():
        """Stop aria_designer API+UI services."""
        result = _stop_designer_services()
        status = 200 if result.get("ok") else 500
        return jsonify(result), status

    @app.route("/api/designer/touch", methods=["POST"])
    def api_designer_touch():
        """Refresh designer activity for idle auto-stop policy."""
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "manual-touch")
        payload = {"ok": True}
        payload.update(_designer_touch_activity(reason))
        payload.update(_designer_idle_state())
        return jsonify(payload), 200

    @app.route("/api/designer/compile", methods=["POST"])
    def api_designer_compile():
        """Accept graph JSON from designer and return compiled module info."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/compile
        proxy_body = {"workflow": workflow_json, "target": "auto"}
        proxied = _proxy_or_error(
            _designer_proxy("POST", "/api/v1/workflows/compile", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        # Local fallback for offline/unit-test mode.
        try:
            out = compile_designer_graph(workflow_json)
            return jsonify(out), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/designer/validate", methods=["POST"])
    def api_designer_validate():
        """Accept graph JSON from designer and return validation results."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/validate
        proxy_body = {"workflow": workflow_json}
        proxied = _proxy_or_error(
            _designer_proxy("POST", "/api/v1/workflows/validate", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        try:
            out = validate_designer_graph(workflow_json)
            return jsonify(out), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/designer/run", methods=["POST"])
    def api_designer_run():
        """Accept graph JSON from designer, run forward pass, and return metrics."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        device = request.args.get("device", "cpu")

        # Proxy: POST /api/v1/workflows/run
        proxy_body = {"workflow": workflow_json, "budget": {"device": device}}
        proxied = _proxy_or_error(
            _designer_proxy("POST", "/api/v1/workflows/run", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        try:
            out = run_designer_graph(workflow_json, device=device)
            return jsonify(out), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/designer/components", methods=["GET"])
    def api_designer_components():
        """Return all available primitives formatted for the designer."""
        # Proxy: GET /api/v1/components
        proxied = _proxy_or_error(
            _designer_proxy("GET", "/api/v1/components")
        )
        if proxied is not None:
            return proxied

        try:
            return jsonify(get_designer_components()), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/designer/save", methods=["POST"])
    def api_designer_save():
        """Save a workflow definition to the notebook."""
        body = request.get_json(silent=True) or {}
        workflow_id = body.get("workflow_id")
        name = body.get("name", "Untitled Workflow")
        if not workflow_id:
            return jsonify({"success": False, "error": "Missing workflow_id"}), 400

        # Proxy: PUT /api/v1/workflows/{workflow_id}
        proxy_body = {
            "schema_version": "workflow_graph.v1",
            "workflow_id": workflow_id,
            "name": name,
            "nodes": body.get("nodes", []),
            "edges": body.get("edges", []),
            "metadata": body.get("metadata", {}),
        }
        proxied = _proxy_or_error(
            _designer_proxy("PUT", f"/api/v1/workflows/{workflow_id}", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/commit", methods=["POST"])
    def api_designer_commit():
        """Commit a designer architecture as a new program result in the research pipeline."""
        body = request.get_json(silent=True) or {}
        workflow = body.get("workflow")
        if not workflow:
            return jsonify({"success": False, "error": "Missing workflow data"}), 400

        # Proxy: POST /api/v1/workflows/evaluate
        # Note: evaluate is effectively a commit to the evaluation database in the designer
        # which our dashboard syncs from.
        proxied = _proxy_or_error(
            _designer_proxy("POST", "/api/v1/workflows/evaluate", json_body={"workflow": workflow})
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
    def designer_v1_proxy(path):
        """Catch-all proxy for designer API v1 routes when embedded."""
        result = _proxy_or_error(
            _designer_proxy(request.method, f"/api/v1/{path}",
                            json_body=request.get_json(silent=True) if request.method in ("POST", "PUT") else None,
                            params=request.args)
        )
        if result is not None:
            return result
        return jsonify({"error": "Designer API proxy failed"}), 502

    @app.route("/api/designer/load/<workflow_id>")
    def api_designer_load(workflow_id):
        """Load a specific workflow definition."""
        # Proxy: GET /api/v1/workflows/{workflow_id}
        proxied = _proxy_or_error(
            _designer_proxy("GET", f"/api/v1/workflows/{workflow_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/list")
    def api_designer_list_workflows():
        """List all saved workflows."""
        # Proxy: GET /api/v1/workflows
        proxied = _proxy_or_error(
            _designer_proxy("GET", "/api/v1/workflows")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/templates")
    def api_designer_templates():
        """Return hardcoded starter templates for the designer.

        No proxy equivalent — templates are served locally.
        """
        templates = [
            {
                "id": "tpl_linear",
                "name": "Simple Linear",
                "description": "Single linear projection.",
                "workflow": {
                    "nodes": [
                        {"id": "n0", "component_type": "io/input", "params": {}, "ui_meta": {"position": {"x": 100, "y": 100}}},
                        {"id": "n1", "component_type": "linear_algebra/linear_proj", "params": {}, "ui_meta": {"position": {"x": 100, "y": 200}}},
                        {"id": "n2", "component_type": "io/output", "params": {}, "ui_meta": {"position": {"x": 100, "y": 300}}}
                    ],
                    "edges": [
                        {"id": "e0", "source": "n0", "target": "n1"},
                        {"id": "e1", "source": "n1", "target": "n2"}
                    ]
                }
            },
            {
                "id": "tpl_mlp",
                "name": "Standard MLP",
                "description": "Two-layer MLP with ReLU.",
                "workflow": {
                    "nodes": [
                        {"id": "in", "component_type": "io/input", "params": {}, "ui_meta": {"position": {"x": 100, "y": 50}}},
                        {"id": "l1", "component_type": "linear_algebra/linear_proj", "params": {"out_dim": 512}, "ui_meta": {"position": {"x": 100, "y": 150}}},
                        {"id": "act", "component_type": "math/relu", "params": {}, "ui_meta": {"position": {"x": 100, "y": 250}}},
                        {"id": "l2", "component_type": "linear_algebra/linear_proj", "params": {"out_dim": 256}, "ui_meta": {"position": {"x": 100, "y": 350}}},
                        {"id": "out", "component_type": "io/output", "params": {}, "ui_meta": {"position": {"x": 100, "y": 450}}}
                    ],
                    "edges": [
                        {"id": "e1", "source": "in", "target": "l1"},
                        {"id": "e2", "source": "l1", "target": "act"},
                        {"id": "e3", "source": "act", "target": "l2"},
                        {"id": "e4", "source": "l2", "target": "out"}
                    ]
                }
            }
        ]
        return jsonify(templates)

    @app.route("/api/designer/export/python", methods=["POST"])
    def api_designer_export_python():
        """Generate standalone PyTorch module code for a workflow.

        No proxy equivalent — uses local generation.
        """
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        from ..designer_utils import generate_python_module
        code = generate_python_module(workflow_json)
        return jsonify({"success": True, "code": code})

    @app.route("/api/designer/import/survivors")
    def api_designer_survivors():
        """List top survivors from the research pipeline for importing."""
        n = request.args.get("n", 20, type=int)

        # Proxy: GET /api/v1/import/survivors
        proxied = _proxy_or_error(
            _designer_proxy("GET", "/api/v1/import/survivors", params={"n": n})
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/import", methods=["POST"])
    def api_designer_import():
        """Import a computation graph from the research pipeline by result_id."""
        body = request.get_json(silent=True) or {}
        result_id = body.get("result_id")
        if not result_id:
            return jsonify({"success": False, "error": "Missing result_id"}), 400

        # Proxy: POST /api/v1/import/survivors/{result_id}
        proxied = _proxy_or_error(
            _designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/lineage/sync", methods=["POST"])
    def api_designer_lineage_sync():
        """Upsert Aria Designer run-lineage metadata into the research notebook."""
        body = request.get_json(silent=True) or {}
        run_id = str(body.get("run_id") or "").strip()
        workflow_id = str(body.get("workflow_id") or "").strip()
        if not run_id or not workflow_id:
            return jsonify({
                "success": False,
                "error": "run_id and workflow_id are required",
            }), 400

        workflow_version = body.get("workflow_version")
        try:
            workflow_version = int(workflow_version) if workflow_version is not None else None
        except Exception:
            workflow_version = None

        total_time_ms = body.get("total_time_ms")
        try:
            total_time_ms = float(total_time_ms) if total_time_ms is not None else None
        except Exception:
            total_time_ms = None

        created_at = body.get("created_at")
        try:
            created_at = float(created_at) if created_at is not None else None
        except Exception:
            created_at = None

        nb = LabNotebook(notebook_path)
        try:
            nb.save_designer_run_lineage(
                run_id=run_id,
                workflow_id=workflow_id,
                workflow_version=workflow_version,
                graph_fingerprint=body.get("graph_fingerprint"),
                status=str(body.get("status") or "unknown"),
                source=str(body.get("source") or "aria_designer"),
                total_time_ms=total_time_ms,
                metrics=body.get("metrics") if isinstance(body.get("metrics"), dict) else {},
                payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
                created_at=created_at,
            )
            row = nb.get_designer_run_lineage(run_id)
            return jsonify({
                "success": True,
                "run_id": run_id,
                "workflow_id": workflow_id,
                "stored": bool(row),
            })
        finally:
            nb.close()

    @app.route("/api/designer/lineage/<run_id>")
    def api_designer_lineage_get(run_id):
        """Get one designer run-lineage record."""
        nb = LabNotebook(notebook_path)
        try:
            row = nb.get_designer_run_lineage(run_id)
            if row is None:
                return jsonify({"error": "Lineage run not found"}), 404
            return jsonify(row)
        finally:
            nb.close()

    @app.route("/api/designer/lineage")
    def api_designer_lineage_list():
        """List designer run-lineage rows, optionally filtered by workflow_id."""
        workflow_id = request.args.get("workflow_id")
        limit = request.args.get("limit", 100, type=int)
        limit = max(1, min(int(limit or 100), 500))
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.list_designer_run_lineage(workflow_id=workflow_id, limit=limit)
            return jsonify(rows)
        finally:
            nb.close()

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
            from ..analytics import ExperimentAnalytics
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
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )

            from ..llm.context import build_campaign_report_context
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

    @app.route("/api/knowledge/backfill", methods=["POST"])
    def api_knowledge_backfill():
        """Backfill missing knowledge categories from measured experiment data."""
        nb = LabNotebook(notebook_path)
        try:
            result = _backfill_knowledge_from_real_data(nb)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in /api/knowledge/backfill: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Action Queue endpoints ──

    @app.route("/api/actions")
    def api_actions():
        """Aggregated prioritized action list for the dashboard."""
        nb = LabNotebook(notebook_path)
        try:
            actions = _compute_action_queue(nb)
            return jsonify(actions)
        except Exception as e:
            logger.error(f"Error in /api/actions: {e}")
            return jsonify([]), 500
        finally:
            nb.close()

    @app.route("/api/actions/<action_id>/dismiss", methods=["POST"])
    def api_action_dismiss(action_id):
        """Dismiss an action card (ephemeral, resets on server restart)."""
        clean_id = str(action_id or "").strip()[:64]
        if not clean_id:
            return jsonify({"error": "Missing action_id"}), 400
        _DISMISSED_ACTIONS.add(clean_id)
        return jsonify({"dismissed": clean_id, "total_dismissed": len(_DISMISSED_ACTIONS)})

    @app.route("/api/actions/<action_id>/approve", methods=["POST"])
    def api_action_approve(action_id):
        """User approves a pending autonomous action."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            action = autonomy.approve(action_id)
            if not action:
                return jsonify({"error": "Action not found or not pending"}), 404
            store.update_status(
                action_id, action.status,
                executed_at=action.executed_at,
                undo_snapshot=action.undo_snapshot,
            )
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error approving action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/actions/<action_id>/undo", methods=["POST"])
    def api_action_undo(action_id):
        """Undo a recently executed autonomous action (within 5 min window)."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            action = autonomy.undo(action_id)
            if not action:
                return jsonify({"error": "Action not found or undo window expired"}), 404
            store.update_status(action_id, action.status)
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error undoing action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    # ── Aria Autonomy endpoints ─────────────────────────────────────

    @app.route("/api/aria/autonomy")
    def api_aria_autonomy_get():
        """Get current autonomy trust level and per-decision-type settings."""
        try:
            autonomy, _ = _get_autonomy(notebook_path)
            return jsonify(autonomy.get_config())
        except Exception as e:
            logger.error(f"Error in GET /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/autonomy", methods=["PUT"])
    def api_aria_autonomy_put():
        """Update autonomy trust level or per-decision-type overrides."""
        try:
            autonomy, _ = _get_autonomy(notebook_path)
            body = request.get_json(force=True, silent=True) or {}
            config = autonomy.update_config(body)
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error in PUT /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/activity")
    def api_aria_activity():
        """Get Aria's recent autonomous decisions and their outcomes."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            limit = request.args.get("limit", 20, type=int)
            # Combine in-memory actions with persisted ones
            memory_actions = autonomy.get_recent_activity(limit)
            stored_actions = store.get_recent(limit)

            # Merge: prefer in-memory (fresher), fill with stored
            seen_ids = {a["action_id"] for a in memory_actions}
            merged = list(memory_actions)
            for sa in stored_actions:
                if sa["action_id"] not in seen_ids:
                    merged.append(sa)
                    seen_ids.add(sa["action_id"])

            merged.sort(key=lambda a: a.get("created_at", 0), reverse=True)
            return jsonify(merged[:limit])
        except Exception as e:
            logger.error(f"Error in /api/aria/activity: {e}")
            return jsonify([]), 500

    # ── /api/v1/ aliases for embedded aria_designer iframe ──
    # The designer app uses /api/v1/... routes (its native API paths).
    # When embedded via /designer-proxy/, RESEARCH_API_BASE resolves to the
    # dashboard origin, so these requests land here instead of on port 8091.

    # Import survivor routes removed — the catch-all proxy below forwards
    # these to the aria_designer backend which has the canonical importer
    # (runtime/importer.py:import_single). This ensures both embedded and
    # full-view modes produce identical workflows with the same node IDs,
    # metadata (result_id, fingerprint), and component type mappings.

    @app.route("/api/v1/components", methods=["GET"])
    def api_v1_components():
        """Return designer components — proxy to designer API or fallback to local DB."""
        proxied = _proxy_or_error(
            _designer_proxy("GET", "/api/v1/components", params=dict(request.args))
        )
        if proxied is not None:
            return proxied
        # Fallback: read directly from the designer component database
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from api.app import database as _designer_db
            comps = _designer_db.list_components(
                category=request.args.get("category"),
                status=request.args.get("status"),
            )
            if comps:
                return jsonify(comps)
        except Exception:
            logger.debug("Could not load components from designer DB, falling back to primitives")
        return jsonify(get_designer_components())

    @app.route("/api/v1/import/survivors", methods=["GET"])
    def api_v1_import_survivors():
        """List importable survivors — proxy to designer or local fallback."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort_by", "loss_ratio")
        min_novelty = request.args.get("min_novelty", 0.0, type=float)

        proxied = _proxy_or_error(
            _designer_proxy("GET", "/api/v1/import/survivors",
                            params={"n": n, "sort_by": sort_by, "min_novelty": min_novelty})
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_survivors as _import_survivors
            return jsonify(_import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty))
        except ImportError:
            nb = LabNotebook(notebook_path)
            try:
                survivors = nb.get_top_programs(n, sort_by=sort_by)
                return jsonify(survivors)
            finally:
                nb.close()

    @app.route("/api/v1/import/survivors/<result_id>", methods=["POST"])
    def api_v1_import_single(result_id):
        """Import a single survivor — proxy to designer or local fallback."""
        proxied = _proxy_or_error(
            _designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_single as _import_single
            wf = _import_single(result_id)
            return jsonify(wf)
        except ImportError:
            return jsonify({"error": "Importer not available"}), 501
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    # ── Catch-all proxy for remaining /api/v1/ designer routes ──
    # The embedded designer iframe makes requests to /api/v1/... which hit
    # this Flask server (same origin).  Specific aliases above handle some
    # routes; everything else is proxied to the aria_designer backend.

    @app.route("/api/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    def api_v1_catchall(subpath):
        """Proxy unhandled /api/v1/ requests to the aria_designer backend."""
        method = request.method
        json_body = request.get_json(silent=True)
        params = dict(request.args) if request.args else None

        # SSE streaming endpoints need special handling — don't buffer the response
        if "stream" in subpath:
            return _proxy_stream(method, f"/api/v1/{subpath}", json_body=json_body, params=params)

        resp = _designer_proxy(method, f"/api/v1/{subpath}", json_body=json_body, params=params)
        result = _proxy_or_error(resp)
        if result is not None:
            return result
        return jsonify({"error": f"Designer backend unavailable for /api/v1/{subpath}"}), 502

    # ── Designer static files (same-origin iframe) ──
