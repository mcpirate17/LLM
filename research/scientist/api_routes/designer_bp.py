"""Designer integration and proxy API routes."""
from __future__ import annotations

from . import misc_bp as misc
from .deps import ApiRouteContext

def register_designer_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    _dashboard_index_path = context.dashboard_index_path
    _dashboard_missing_response = context.dashboard_missing_response
    _is_asset_path = context.is_asset_path

    @app.route("/api/designer/lifecycle")
    def api_designer_lifecycle():
        """Return current aria_designer service status."""
        payload = misc.designer_service_status()
        payload.update(misc.designer_idle_state())
        return misc.jsonify(payload)


    @app.route("/api/designer/ensure-running", methods=["POST"])
    def api_designer_ensure_running():
        """Ensure aria_designer API+UI are running for seamless UX."""
        body = misc.request.get_json(silent=True) or {}
        force_restart = bool(body.get("force_restart", False))
        result = misc.start_designer_services(force_restart=force_restart)
        if result.get("ok"):
            result.update(misc.designer_touch_activity("ensure-running"))
        status = 200 if result.get("ok") else 503
        return misc.jsonify(result), status


    @app.route("/api/designer/stop", methods=["POST"])
    def api_designer_stop():
        """Stop aria_designer API+UI services."""
        result = misc.stop_designer_services()
        status = 200 if result.get("ok") else 500
        return misc.jsonify(result), status


    @app.route("/api/designer/touch", methods=["POST"])
    def api_designer_touch():
        """Refresh designer activity for idle auto-stop policy."""
        body = misc.request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "manual-touch")
        payload = {"ok": True}
        payload.update(misc.designer_touch_activity(reason))
        payload.update(misc.designer_idle_state())
        return misc.jsonify(payload), 200


    @app.route("/api/designer/compile", methods=["POST"])
    def api_designer_compile():
        """Accept graph JSON from designer and return compiled module info."""
        workflow_json = misc.request.get_json(silent=True)
        if not workflow_json:
            return misc.jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/compile
        proxy_body = {"workflow": workflow_json, "target": "auto"}
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", "/api/v1/workflows/compile", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        # Fallback: local compilation
        result = misc.compile_designer_graph(workflow_json)
        return misc.jsonify(result)


    @app.route("/api/designer/validate", methods=["POST"])
    def api_designer_validate():
        """Accept graph JSON from designer and return validation results."""
        workflow_json = misc.request.get_json(silent=True)
        if not workflow_json:
            return misc.jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/validate
        proxy_body = {"workflow": workflow_json}
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", "/api/v1/workflows/validate", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        # Fallback: local validation
        result = misc.validate_designer_graph(workflow_json)
        return misc.jsonify(result)


    @app.route("/api/designer/run", methods=["POST"])
    def api_designer_run():
        """Accept graph JSON from designer, run forward pass, and return metrics."""
        workflow_json = misc.request.get_json(silent=True)
        if not workflow_json:
            return misc.jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        device = misc.request.args.get("device", "cpu")

        # Proxy: POST /api/v1/workflows/run
        proxy_body = {"workflow": workflow_json, "budget": {"device": device}}
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", "/api/v1/workflows/run", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        # Fallback: local execution
        result = misc.run_designer_graph(workflow_json, device=device)
        return misc.jsonify(result)


    @app.route("/api/designer/components", methods=["GET"])
    def api_designer_components():
        """Return all available primitives formatted for the designer."""
        # Proxy: GET /api/v1/components
        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", "/api/v1/components")
        )
        if proxied is not None:
            return proxied

        # Fallback: local component list
        return misc.jsonify(misc.get_designer_components())


    @app.route("/api/designer/save", methods=["POST"])
    def api_designer_save():
        """Save a workflow definition to the notebook."""
        body = misc.request.get_json(silent=True) or {}
        workflow_id = body.get("workflow_id")
        name = body.get("name", "Untitled Workflow")
        if not workflow_id:
            return misc.jsonify({"success": False, "error": "Missing workflow_id"}), 400

        # Proxy: PUT /api/v1/workflows/{workflow_id}
        proxy_body = {
            "schema_version": "workflow_graph.v1",
            "workflow_id": workflow_id,
            "name": name,
            "nodes": body.get("nodes", []),
            "edges": body.get("edges", []),
            "metadata": body.get("metadata", {}),
        }
        proxied = misc.proxy_or_error(
            misc.designer_proxy("PUT", f"/api/v1/workflows/{workflow_id}", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/commit", methods=["POST"])
    def api_designer_commit():
        """Commit a designer architecture as a new program result in the research pipeline."""
        body = misc.request.get_json(silent=True) or {}
        workflow = body.get("workflow")
        if not workflow:
            return misc.jsonify({"success": False, "error": "Missing workflow data"}), 400

        # Proxy: POST /api/v1/workflows/evaluate
        # Note: evaluate is effectively a commit to the evaluation database in the designer
        # which our dashboard syncs from.
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", "/api/v1/workflows/evaluate", json_body={"workflow": workflow})
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    def designer_v1_proxy(path):
        """Catch-all proxy for designer API v1 routes when embedded.

        This route is registered earlier than the later catch-all below, so it
        must preserve SSE semantics for streaming endpoints like
        ``/api/v1/workflows/evaluate/stream`` instead of routing them through
        the JSON proxy adapter.
        """
        json_body = misc.request.get_json(silent=True) if misc.request.method in ("POST", "PUT", "OPTIONS") else None
        params = misc.request.args

        if "stream" in path:
            return misc.proxy_stream(misc.request.method, f"/api/v1/{path}", json_body=json_body, params=params)

        result = misc.proxy_or_error(
            misc.designer_proxy(
                misc.request.method,
                f"/api/v1/{path}",
                json_body=json_body,
                params=params,
            )
        )
        if result is not None:
            return result
        return misc.jsonify({"error": "Designer API proxy failed"}), 502


    @app.route("/api/designer/load/<workflow_id>")
    def api_designer_load(workflow_id):
        """Load a specific workflow definition."""
        # Proxy: GET /api/v1/workflows/{workflow_id}
        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", f"/api/v1/workflows/{workflow_id}")
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/list")
    def api_designer_list_workflows():
        """List all saved workflows."""
        # Proxy: GET /api/v1/workflows
        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", "/api/v1/workflows")
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


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
        return misc.jsonify(templates)


    @app.route("/api/designer/export/python", methods=["POST"])
    def api_designer_export_python():
        """Generate standalone PyTorch module code for a workflow.

        No proxy equivalent — uses local generation.
        """
        workflow_json = misc.request.get_json(silent=True)
        if not workflow_json:
            return misc.jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        from .. import designer_utils as _designer_utils

        code = _designer_utils.generate_python_module(workflow_json)
        return misc.jsonify({"success": True, "code": code})


    @app.route("/api/designer/import/survivors")
    def api_designer_survivors():
        """List top survivors from the research pipeline for importing."""
        n = misc.request.args.get("n", 20, type=int)

        # Proxy: GET /api/v1/import/survivors
        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", "/api/v1/import/survivors", params={"n": n})
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/import", methods=["POST"])
    def api_designer_import():
        """Import a computation graph from the research pipeline by result_id."""
        body = misc.request.get_json(silent=True) or {}
        result_id = body.get("result_id")
        if not result_id:
            return misc.jsonify({"success": False, "error": "Missing result_id"}), 400

        # Proxy: POST /api/v1/import/survivors/{result_id}
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        return misc.jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/lineage/sync", methods=["POST"])
    def api_designer_lineage_sync():
        """Upsert Aria Designer run-lineage metadata into the research notebook."""
        body = misc.request.get_json(silent=True) or {}
        run_id = str(body.get("run_id") or "").strip()
        workflow_id = str(body.get("workflow_id") or "").strip()
        if not run_id or not workflow_id:
            return misc.jsonify({
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

        nb = misc.LabNotebook(notebook_path)
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
            return misc.jsonify({
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
        nb = misc.LabNotebook(notebook_path)
        try:
            row = nb.get_designer_run_lineage(run_id)
            if row is None:
                return misc.jsonify({"error": "Lineage run not found"}), 404
            return misc.jsonify(row)
        finally:
            nb.close()


    @app.route("/api/designer/lineage")
    def api_designer_lineage_list():
        """List designer run-lineage rows, optionally filtered by workflow_id."""
        workflow_id = misc.request.args.get("workflow_id")
        limit = misc.request.args.get("limit", 100, type=int)
        limit = max(1, min(int(limit or 100), 500))
        nb = misc.LabNotebook(notebook_path)
        try:
            rows = nb.list_designer_run_lineage(workflow_id=workflow_id, limit=limit)
            return misc.jsonify(rows)
        finally:
            nb.close()

    @app.route("/api/v1/components", methods=["GET"])
    def api_v1_components():
        """Return designer components — proxy to designer API or fallback to local DB."""
        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", "/api/v1/components", params=dict(misc.request.args))
        )
        if proxied is not None:
            return proxied
        # Fallback: read directly from the designer component database
        try:
            import sys as _sys
            _designer_root = str(misc.Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from api.app import database as _designer_db
            comps = _designer_db.list_components(
                category=misc.request.args.get("category"),
                status=misc.request.args.get("status"),
            )
            if comps:
                return misc.jsonify(comps)
        except Exception:
            misc.logger.debug("Could not load components from designer DB, falling back to primitives")
        return misc.jsonify(misc.get_designer_components())


    @app.route("/api/v1/import/survivors", methods=["GET"])
    def api_v1_import_survivors():
        """List importable survivors — proxy to designer or local fallback."""
        n = misc.request.args.get("n", 20, type=int)
        sort_by = misc.request.args.get("sort_by", "loss_ratio")
        min_novelty = misc.request.args.get("min_novelty", 0.0, type=float)

        proxied = misc.proxy_or_error(
            misc.designer_proxy("GET", "/api/v1/import/survivors",
                            params={"n": n, "sort_by": sort_by, "min_novelty": min_novelty})
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(misc.Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_survivors as _import_survivors
            return misc.jsonify(_import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty))
        except ImportError:
            nb = misc.LabNotebook(notebook_path)
            try:
                survivors = nb.get_top_programs(n, sort_by=sort_by)
                return misc.jsonify(survivors)
            finally:
                nb.close()


    @app.route("/api/v1/import/survivors/<result_id>", methods=["POST"])
    def api_v1_import_single(result_id):
        """Import a single survivor — proxy to designer or local fallback."""
        proxied = misc.proxy_or_error(
            misc.designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(misc.Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_single as _import_single
            wf = _import_single(result_id)
            return misc.jsonify(wf)
        except ImportError:
            return misc.jsonify({"error": "Importer not available"}), 501
        except ValueError as e:
            return misc.jsonify({"error": str(e)}), 404


    @app.route("/api/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    def api_v1_catchall(subpath):
        """Proxy unhandled /api/v1/ requests to the aria_designer backend."""
        method = misc.request.method
        json_body = misc.request.get_json(silent=True)
        params = dict(misc.request.args) if misc.request.args else None

        # SSE streaming endpoints need special handling — don't buffer the response
        if "stream" in subpath:
            return misc.proxy_stream(method, f"/api/v1/{subpath}", json_body=json_body, params=params)

        resp = misc.designer_proxy(method, f"/api/v1/{subpath}", json_body=json_body, params=params)
        result = misc.proxy_or_error(resp)
        if result is not None:
            return result
        return misc.jsonify({"error": f"Designer backend unavailable for /api/v1/{subpath}"}), 502
