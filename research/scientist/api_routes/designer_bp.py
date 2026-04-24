"""Designer integration and proxy API routes."""

from __future__ import annotations

import importlib
import logging
from typing import Callable

from flask import jsonify, request

from . import _designer as _des
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def _load_designer_importer(*names: str) -> tuple[Callable, ...]:
    """Load importer functions from the canonical aria_designer package path."""
    module = importlib.import_module("aria_designer.runtime.importer")
    try:
        return tuple(getattr(module, name) for name in names)
    except AttributeError as exc:
        missing = ", ".join(name for name in names if not hasattr(module, name))
        raise ImportError(
            f"aria_designer.runtime.importer is missing: {missing}"
        ) from exc


def _register_lifecycle_routes(app) -> None:
    """Lifecycle: status, ensure-running, stop, touch."""

    @app.route("/api/designer/lifecycle")
    def api_designer_lifecycle():
        """Return current aria_designer service status."""
        payload = _des.designer_service_status()
        payload.update(_des.designer_idle_state())
        return jsonify(payload)

    @app.route("/api/designer/ensure-running", methods=["POST"])
    def api_designer_ensure_running():
        """Ensure aria_designer API+UI are running for seamless UX."""
        body = request.get_json(silent=True) or {}
        force_restart = bool(body.get("force_restart", False))
        result = _des.start_designer_services(force_restart=force_restart)
        if result.get("ok"):
            result.update(_des.designer_touch_activity("ensure-running"))
        status = 200 if result.get("ok") else 503
        return jsonify(result), status

    @app.route("/api/designer/stop", methods=["POST"])
    def api_designer_stop():
        """Stop aria_designer API+UI services."""
        result = _des.stop_designer_services()
        status = 200 if result.get("ok") else 500
        return jsonify(result), status

    @app.route("/api/designer/touch", methods=["POST"])
    def api_designer_touch():
        """Refresh designer activity for idle auto-stop policy."""
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "manual-touch")
        payload = {"ok": True}
        payload.update(_des.designer_touch_activity(reason))
        payload.update(_des.designer_idle_state())
        return jsonify(payload), 200


def _register_workflow_routes(app) -> None:
    """Workflow: compile, validate, run, components, save, commit."""

    @app.route("/api/designer/compile", methods=["POST"])
    def api_designer_compile():
        """Accept graph JSON from designer and return compiled module info."""
        from ..designer_utils import compile_designer_graph

        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        proxy_body = {"workflow": workflow_json, "target": "auto"}
        proxied = _des.proxy_or_error(
            _des.designer_proxy(
                "POST", "/api/v1/workflows/compile", json_body=proxy_body
            )
        )
        if proxied is not None:
            return proxied

        result = compile_designer_graph(workflow_json)
        return jsonify(result)

    @app.route("/api/designer/validate", methods=["POST"])
    def api_designer_validate():
        """Accept graph JSON from designer and return validation results."""
        from ..designer_utils import validate_designer_graph

        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        proxy_body = {"workflow": workflow_json}
        proxied = _des.proxy_or_error(
            _des.designer_proxy(
                "POST", "/api/v1/workflows/validate", json_body=proxy_body
            )
        )
        if proxied is not None:
            return proxied

        result = validate_designer_graph(workflow_json)
        return jsonify(result)

    @app.route("/api/designer/run", methods=["POST"])
    def api_designer_run():
        """Accept graph JSON from designer, run forward pass, and return metrics."""
        from ..designer_utils import run_designer_graph

        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        device = request.args.get("device", "cpu")

        proxy_body = {"workflow": workflow_json, "budget": {"device": device}}
        proxied = _des.proxy_or_error(
            _des.designer_proxy("POST", "/api/v1/workflows/run", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        result = run_designer_graph(workflow_json, device=device)
        return jsonify(result)

    @app.route("/api/designer/components", methods=["GET"])
    def api_designer_components():
        """Return all available primitives formatted for the designer."""
        from ..designer_utils import get_designer_components

        proxied = _des.proxy_or_error(_des.designer_proxy("GET", "/api/v1/components"))
        if proxied is not None:
            return proxied

        return jsonify(get_designer_components())

    @app.route("/api/designer/save", methods=["POST"])
    def api_designer_save():
        """Save a workflow definition to the notebook."""
        body = request.get_json(silent=True) or {}
        workflow_id = body.get("workflow_id")
        name = body.get("name", "Untitled Workflow")
        if not workflow_id:
            return jsonify({"success": False, "error": "Missing workflow_id"}), 400

        proxy_body = {
            "schema_version": "workflow_graph.v1",
            "workflow_id": workflow_id,
            "name": name,
            "nodes": body.get("nodes", []),
            "edges": body.get("edges", []),
            "metadata": body.get("metadata", {}),
        }
        proxied = _des.proxy_or_error(
            _des.designer_proxy(
                "PUT", f"/api/v1/workflows/{workflow_id}", json_body=proxy_body
            )
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

        proxied = _des.proxy_or_error(
            _des.designer_proxy(
                "POST", "/api/v1/workflows/evaluate", json_body={"workflow": workflow}
            )
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


def _register_crud_routes(app) -> None:
    """CRUD: load, list, templates, export."""

    @app.route("/api/designer/load/<workflow_id>")
    def api_designer_load(workflow_id):
        """Load a specific workflow definition."""
        proxied = _des.proxy_or_error(
            _des.designer_proxy("GET", f"/api/v1/workflows/{workflow_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503

    @app.route("/api/designer/list")
    def api_designer_list_workflows():
        """List all saved workflows."""
        proxied = _des.proxy_or_error(_des.designer_proxy("GET", "/api/v1/workflows"))
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
                        {
                            "id": "n0",
                            "component_type": "io/input",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 100}},
                        },
                        {
                            "id": "n1",
                            "component_type": "linear_algebra/linear_proj",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 200}},
                        },
                        {
                            "id": "n2",
                            "component_type": "io/output",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 300}},
                        },
                    ],
                    "edges": [
                        {"id": "e0", "source": "n0", "target": "n1"},
                        {"id": "e1", "source": "n1", "target": "n2"},
                    ],
                },
            },
            {
                "id": "tpl_mlp",
                "name": "Standard MLP",
                "description": "Two-layer MLP with ReLU.",
                "workflow": {
                    "nodes": [
                        {
                            "id": "in",
                            "component_type": "io/input",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 50}},
                        },
                        {
                            "id": "l1",
                            "component_type": "linear_algebra/linear_proj",
                            "params": {"out_dim": 512},
                            "ui_meta": {"position": {"x": 100, "y": 150}},
                        },
                        {
                            "id": "act",
                            "component_type": "math/relu",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 250}},
                        },
                        {
                            "id": "l2",
                            "component_type": "linear_algebra/linear_proj",
                            "params": {"out_dim": 256},
                            "ui_meta": {"position": {"x": 100, "y": 350}},
                        },
                        {
                            "id": "out",
                            "component_type": "io/output",
                            "params": {},
                            "ui_meta": {"position": {"x": 100, "y": 450}},
                        },
                    ],
                    "edges": [
                        {"id": "e1", "source": "in", "target": "l1"},
                        {"id": "e2", "source": "l1", "target": "act"},
                        {"id": "e3", "source": "act", "target": "l2"},
                        {"id": "e4", "source": "l2", "target": "out"},
                    ],
                },
            },
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

        from .. import designer_utils as _designer_utils

        code = _designer_utils.generate_python_module(workflow_json)
        return jsonify({"success": True, "code": code})


def _register_import_routes(app) -> None:
    """Import: import/survivors, import."""

    @app.route("/api/designer/import/survivors")
    def api_designer_survivors():
        """List top survivors from the research pipeline for importing."""
        n = request.args.get("n", 20, type=int)

        proxied = _des.proxy_or_error(
            _des.designer_proxy("GET", "/api/v1/import/survivors", params={"n": n})
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

        proxied = _des.proxy_or_error(
            _des.designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


def _register_lineage_routes(app, notebook_path: str, wnb) -> None:
    """Lineage: sync, get, list (uses @wnb)."""

    @app.route("/api/designer/lineage/sync", methods=["POST"])
    @wnb
    def api_designer_lineage_sync(nb=None):
        """Upsert Aria Designer run-lineage metadata into the research notebook."""
        body = request.get_json(silent=True) or {}
        run_id = str(body.get("run_id") or "").strip()
        workflow_id = str(body.get("workflow_id") or "").strip()
        if not run_id or not workflow_id:
            return jsonify(
                {
                    "success": False,
                    "error": "run_id and workflow_id are required",
                }
            ), 400

        workflow_version = body.get("workflow_version")
        try:
            workflow_version = (
                int(workflow_version) if workflow_version is not None else None
            )
        except (TypeError, ValueError):
            workflow_version = None

        total_time_ms = body.get("total_time_ms")
        try:
            total_time_ms = float(total_time_ms) if total_time_ms is not None else None
        except (TypeError, ValueError):
            total_time_ms = None

        created_at = body.get("created_at")
        try:
            created_at = float(created_at) if created_at is not None else None
        except (TypeError, ValueError):
            created_at = None

        nb.save_designer_run_lineage(
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            graph_fingerprint=body.get("graph_fingerprint"),
            status=str(body.get("status") or "unknown"),
            source=str(body.get("source") or "aria_designer"),
            total_time_ms=total_time_ms,
            metrics=body.get("metrics")
            if isinstance(body.get("metrics"), dict)
            else {},
            payload=body.get("payload")
            if isinstance(body.get("payload"), dict)
            else {},
            created_at=created_at,
        )
        row = nb.get_designer_run_lineage(run_id)
        return jsonify(
            {
                "success": True,
                "run_id": run_id,
                "workflow_id": workflow_id,
                "stored": bool(row),
            }
        )

    @app.route("/api/designer/lineage/<run_id>")
    @wnb
    def api_designer_lineage_get(run_id, nb=None):
        """Get one designer run-lineage record."""
        row = nb.get_designer_run_lineage(run_id)
        if row is None:
            return jsonify({"error": "Lineage run not found"}), 404
        return jsonify(row)

    @app.route("/api/designer/lineage")
    @wnb
    def api_designer_lineage_list(nb=None):
        """List designer run-lineage rows, optionally filtered by workflow_id."""
        workflow_id = request.args.get("workflow_id")
        limit = request.args.get("limit", 100, type=int)
        limit = max(1, min(int(limit or 100), 500))
        rows = nb.list_designer_run_lineage(workflow_id=workflow_id, limit=limit)
        return jsonify(rows)


def _register_v1_proxy_routes(app, notebook_path: str, wnb) -> None:
    """/api/v1/* proxy routes."""

    @app.route(
        "/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    def designer_v1_proxy(path):
        """Catch-all proxy for designer API v1 routes when embedded.

        This route is registered earlier than the later catch-all below, so it
        must preserve SSE semantics for streaming endpoints like
        ``/api/v1/workflows/evaluate/stream`` instead of routing them through
        the JSON proxy adapter.
        """
        json_body = (
            request.get_json(silent=True)
            if request.method in ("POST", "PUT", "OPTIONS")
            else None
        )
        params = request.args

        if "stream" in path:
            return _des.proxy_stream(
                request.method, f"/api/v1/{path}", json_body=json_body, params=params
            )

        result = _des.proxy_or_error(
            _des.designer_proxy(
                request.method,
                f"/api/v1/{path}",
                json_body=json_body,
                params=params,
            )
        )
        if result is not None:
            return result
        return jsonify({"error": "Designer API proxy failed"}), 502

    @app.route("/api/v1/components", methods=["GET"])
    def api_v1_components():
        """Return designer components — proxy to designer API or fallback to local DB."""
        proxied = _des.proxy_or_error(
            _des.designer_proxy("GET", "/api/v1/components", params=dict(request.args))
        )
        if proxied is not None:
            return proxied
        # Fallback: read directly from the designer component database
        try:
            from aria_designer.api.app import database as _designer_db

            comps = _designer_db.list_components(
                category=request.args.get("category"),
                status=request.args.get("status"),
            )
            if comps:
                return jsonify(comps)
        except Exception:  # noqa: BLE001 — graceful fallback to primitives
            logger.debug(
                "Could not load components from designer DB, falling back to primitives"
            )
        from ..designer_utils import get_designer_components

        return jsonify(get_designer_components())

    @app.route("/api/v1/import/survivors", methods=["GET"])
    @wnb
    def api_v1_import_survivors(nb=None):
        """List importable survivors — proxy to designer or local fallback."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort_by", "loss_ratio")
        min_novelty = request.args.get("min_novelty", 0.0, type=float)

        proxied = _des.proxy_or_error(
            _des.designer_proxy(
                "GET",
                "/api/v1/import/survivors",
                params={"n": n, "sort_by": sort_by, "min_novelty": min_novelty},
            )
        )
        if proxied is not None:
            return proxied

        try:
            (_import_survivors,) = _load_designer_importer("import_survivors")
        except ImportError as exc:
            logger.error("Importer import failed: %s", exc, exc_info=True)
            return jsonify({"error": f"Importer not available: {exc}"}), 501
        return jsonify(_import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty))

    @app.route("/api/v1/import/survivors/<result_id>", methods=["POST"])
    def api_v1_import_single(result_id):
        """Import a single survivor — proxy to designer or local fallback."""
        proxied = _des.proxy_or_error(
            _des.designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        try:
            (_import_single,) = _load_designer_importer("import_single")
        except ImportError as exc:
            logger.error("Importer import failed: %s", exc, exc_info=True)
            return jsonify({"error": f"Importer not available: {exc}"}), 501
        try:
            wf = _import_single(result_id)
            return jsonify(wf)
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 404

    @app.route(
        "/api/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    def api_v1_catchall(subpath):
        """Proxy unhandled /api/v1/ requests to the aria_designer backend."""
        method = request.method
        json_body = request.get_json(silent=True)
        params = dict(request.args) if request.args else None

        # SSE streaming endpoints need special handling — don't buffer the response
        if "stream" in subpath:
            return _des.proxy_stream(
                method, f"/api/v1/{subpath}", json_body=json_body, params=params
            )

        resp = _des.designer_proxy(
            method, f"/api/v1/{subpath}", json_body=json_body, params=params
        )
        result = _des.proxy_or_error(resp)
        if result is not None:
            return result
        return jsonify(
            {"error": f"Designer backend unavailable for /api/v1/{subpath}"}
        ), 502


def register_designer_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    _register_lifecycle_routes(app)
    _register_workflow_routes(app)
    _register_crud_routes(app)
    _register_import_routes(app)
    _register_lineage_routes(app, notebook_path, wnb)
    _register_v1_proxy_routes(app, notebook_path, wnb)
