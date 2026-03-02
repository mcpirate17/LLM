"""
Tests for designer endpoint proxy-first mode.

Verifies that /api/designer/* endpoints:
1. Forward to aria_designer API when proxy is enabled and available
2. Fall back to legacy local implementation when proxy is unavailable
3. Return clear error semantics for proxy timeouts
4. Preserve backward-compatible response shapes

Run: cd /path/to/LLM/research && python -m pytest tests/test_designer_proxy.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure the research package is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


def _make_mock_response(status_code=200, json_body=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = json.dumps(json_body or {})
    return resp


# Sample workflow for testing
_SAMPLE_WORKFLOW = {
    "schema_version": "workflow_graph.v1",
    "workflow_id": "test_wf",
    "name": "Test Workflow",
    "nodes": [
        {"id": "n0", "component_type": "io/input", "params": {}, "ui_meta": {"position": {"x": 0, "y": 0}}},
        {"id": "n1", "component_type": "linear_algebra/linear_proj", "params": {}, "ui_meta": {"position": {"x": 0, "y": 100}}},
        {"id": "n2", "component_type": "io/output", "params": {}, "ui_meta": {"position": {"x": 0, "y": 200}}},
    ],
    "edges": [
        {"id": "e0", "source": "n0", "target": "n1"},
        {"id": "e1", "source": "n1", "target": "n2"},
    ],
}


@unittest.skipUnless(HAS_FLASK, "Flask required")
class TestDesignerProxyHelpers(unittest.TestCase):
    """Test the _designer_proxy and _proxy_or_error helpers."""

    def setUp(self):
        import research.scientist.api as api_mod
        self.api_mod = api_mod

    def test_proxy_disabled_returns_none(self):
        """When proxy is disabled, _designer_proxy returns None."""
        with patch.object(self.api_mod, "_DESIGNER_PROXY_ENABLED", False):
            result = self.api_mod._designer_proxy("GET", "/api/v1/components")
            self.assertIsNone(result)

    @patch("research.scientist.api._requests")
    def test_proxy_connection_error_returns_none(self, mock_requests):
        """When designer backend is unreachable, returns None for fallback."""
        import requests
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout
        mock_requests.request.side_effect = requests.ConnectionError("refused")

        with patch.object(self.api_mod, "_DESIGNER_PROXY_ENABLED", True):
            result = self.api_mod._designer_proxy("GET", "/api/v1/components")
            self.assertIsNone(result)

    @patch("research.scientist.api._requests")
    def test_proxy_timeout_returns_none(self, mock_requests):
        """When proxy times out, returns None for fallback."""
        import requests
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout
        mock_requests.request.side_effect = requests.Timeout("timed out")

        with patch.object(self.api_mod, "_DESIGNER_PROXY_ENABLED", True):
            result = self.api_mod._designer_proxy("GET", "/api/v1/components")
            self.assertIsNone(result)

    @patch("research.scientist.api._requests")
    def test_proxy_success_returns_response(self, mock_requests):
        """On success, returns the Response object."""
        import requests
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout
        mock_resp = _make_mock_response(200, {"valid": True})
        mock_requests.request.return_value = mock_resp

        with patch.object(self.api_mod, "_DESIGNER_PROXY_ENABLED", True):
            result = self.api_mod._designer_proxy("POST", "/api/v1/workflows/validate",
                                                   json_body={"workflow": _SAMPLE_WORKFLOW})
            self.assertIsNotNone(result)
            self.assertEqual(result.status_code, 200)

    def test_proxy_or_error_none_passthrough(self):
        """_proxy_or_error(None) returns None (for fallback)."""
        result = self.api_mod._proxy_or_error(None)
        self.assertIsNone(result)

    def test_proxy_or_error_success(self):
        """_proxy_or_error converts response to Flask (body, status) tuple."""
        # Need a Flask app context for jsonify
        from flask import Flask
        app = Flask(__name__)
        with app.app_context():
            mock_resp = _make_mock_response(200, {"valid": True, "issues": []})
            result = self.api_mod._proxy_or_error(mock_resp)
            self.assertIsNotNone(result)
            body, status = result
            self.assertEqual(status, 200)

    def test_proxy_or_error_4xx(self):
        """_proxy_or_error forwards 4xx status from proxy."""
        from flask import Flask
        app = Flask(__name__)
        with app.app_context():
            mock_resp = _make_mock_response(404, {"detail": "Not found"})
            result = self.api_mod._proxy_or_error(mock_resp)
            self.assertIsNotNone(result)
            _, status = result
            self.assertEqual(status, 404)


@unittest.skipUnless(HAS_FLASK, "Flask required")
class TestDesignerEndpointProxyMode(unittest.TestCase):
    """Test that designer endpoints properly proxy and fallback."""

    @classmethod
    def setUpClass(cls):
        """Create a Flask test client with mocked notebook."""
        import research.scientist.api as api_mod
        cls.api_mod = api_mod

        # Create app with a temp notebook
        import tempfile
        cls._tmpdir = tempfile.mkdtemp()
        cls._nb_path = os.path.join(cls._tmpdir, "test.db")

        # We need to mock the runner and notebook to avoid heavy deps
        with patch.object(api_mod, "_DESIGNER_PROXY_ENABLED", True):
            cls.app = api_mod.create_app(notebook_path=cls._nb_path)
            cls.client = cls.app.test_client()

    def _proxy_validate_response(self):
        return _make_mock_response(200, {"valid": True, "issues": []})

    def _proxy_compile_response(self):
        return _make_mock_response(200, {
            "compiled": True,
            "workflow_id": "test_wf",
            "module_class": "WorkflowModule",
            "param_count": 1024,
        })

    def _proxy_run_response(self):
        return _make_mock_response(200, {
            "accepted": True,
            "run_id": "run_abc123",
            "workflow_id": "test_wf",
        })

    def _proxy_components_response(self):
        return _make_mock_response(200, [
            {"id": "linear_proj", "name": "Linear Projection", "category": "linear_algebra"},
        ])

    def _proxy_save_response(self):
        return _make_mock_response(200, {
            "workflow_id": "test_wf",
            "version": 1,
            "saved_at": "2026-02-21T00:00:00Z",
        })

    def _proxy_load_response(self):
        return _make_mock_response(200, {
            "id": "test_wf",
            "name": "Test Workflow",
            "graph": _SAMPLE_WORKFLOW,
        })

    def _proxy_list_response(self):
        return _make_mock_response(200, [
            {"id": "test_wf", "name": "Test Workflow"},
        ])

    def _proxy_survivors_response(self):
        return _make_mock_response(200, [
            {"result_id": "abc123", "loss_ratio": 0.95},
        ])

    def _proxy_import_response(self):
        return _make_mock_response(200, {
            "success": True,
            "workflow": _SAMPLE_WORKFLOW,
        })

    # --- Proxy success tests ---

    @patch("research.scientist.api._designer_proxy")
    def test_validate_proxies_to_designer(self, mock_proxy):
        """POST /api/designer/validate forwards to designer API."""
        mock_proxy.return_value = self._proxy_validate_response()
        resp = self.client.post("/api/designer/validate",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["valid"])
        # Verify the proxy was called with correct path
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][0], "POST")
        self.assertEqual(call_args[0][1], "/api/v1/workflows/validate")

    @patch("research.scientist.api._designer_proxy")
    def test_compile_proxies_to_designer(self, mock_proxy):
        """POST /api/designer/compile forwards to designer API."""
        mock_proxy.return_value = self._proxy_compile_response()
        resp = self.client.post("/api/designer/compile",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["compiled"])
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/workflows/compile")

    @patch("research.scientist.api._designer_proxy")
    def test_run_proxies_to_designer(self, mock_proxy):
        """POST /api/designer/run forwards to designer API."""
        mock_proxy.return_value = self._proxy_run_response()
        resp = self.client.post("/api/designer/run",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["accepted"])
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/workflows/run")

    @patch("research.scientist.api._designer_proxy")
    def test_components_proxies_to_designer(self, mock_proxy):
        """GET /api/designer/components forwards to designer API."""
        mock_proxy.return_value = self._proxy_components_response()
        resp = self.client.get("/api/designer/components")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/components")

    @patch("research.scientist.api._designer_proxy")
    def test_save_proxies_to_designer(self, mock_proxy):
        """POST /api/designer/save forwards to designer API."""
        mock_proxy.return_value = self._proxy_save_response()
        body = {**_SAMPLE_WORKFLOW, "workflow_id": "test_wf", "name": "Test"}
        resp = self.client.post("/api/designer/save",
                                json=body,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][0], "PUT")
        self.assertIn("/api/v1/workflows/test_wf", call_args[0][1])

    @patch("research.scientist.api._designer_proxy")
    def test_load_proxies_to_designer(self, mock_proxy):
        """GET /api/designer/load/<id> forwards to designer API."""
        mock_proxy.return_value = self._proxy_load_response()
        resp = self.client.get("/api/designer/load/test_wf")
        self.assertEqual(resp.status_code, 200)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/workflows/test_wf")

    @patch("research.scientist.api._designer_proxy")
    def test_list_proxies_to_designer(self, mock_proxy):
        """GET /api/designer/list forwards to designer API."""
        mock_proxy.return_value = self._proxy_list_response()
        resp = self.client.get("/api/designer/list")
        self.assertEqual(resp.status_code, 200)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/workflows")

    @patch("research.scientist.api._designer_proxy")
    def test_survivors_proxies_to_designer(self, mock_proxy):
        """GET /api/designer/import/survivors forwards to designer API."""
        mock_proxy.return_value = self._proxy_survivors_response()
        resp = self.client.get("/api/designer/import/survivors")
        self.assertEqual(resp.status_code, 200)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/import/survivors")

    @patch("research.scientist.api._designer_proxy")
    def test_import_proxies_to_designer(self, mock_proxy):
        """POST /api/designer/import forwards to designer API."""
        mock_proxy.return_value = self._proxy_import_response()
        resp = self.client.post("/api/designer/import",
                                json={"result_id": "abc123"},
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        mock_proxy.assert_called_once()
        call_args = mock_proxy.call_args
        self.assertEqual(call_args[0][1], "/api/v1/import/survivors/abc123")

    # --- Fallback tests (proxy unavailable) ---

    @patch("research.scientist.api._designer_proxy", return_value=None)
    @patch("research.scientist.api.validate_designer_graph")
    def test_validate_falls_back_when_proxy_unavailable(self, mock_validate, mock_proxy):
        """When proxy returns None, validate falls back to local."""
        mock_validate.return_value = {"valid": True, "issues": []}
        resp = self.client.post("/api/designer/validate",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        mock_validate.assert_called_once()

    @patch("research.scientist.api._designer_proxy", return_value=None)
    @patch("research.scientist.api.compile_designer_graph")
    def test_compile_falls_back_when_proxy_unavailable(self, mock_compile, mock_proxy):
        """When proxy returns None, compile falls back to local."""
        mock_compile.return_value = {"compiled": True, "workflow_id": "test_wf"}
        resp = self.client.post("/api/designer/compile",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        mock_compile.assert_called_once()

    @patch("research.scientist.api._designer_proxy", return_value=None)
    @patch("research.scientist.api.run_designer_graph")
    def test_run_falls_back_when_proxy_unavailable(self, mock_run, mock_proxy):
        """When proxy returns None, run falls back to local."""
        mock_run.return_value = {"success": True}
        resp = self.client.post("/api/designer/run",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        mock_run.assert_called_once()

    @patch("research.scientist.api._designer_proxy", return_value=None)
    @patch("research.scientist.api.get_designer_components")
    def test_components_falls_back_when_proxy_unavailable(self, mock_comps, mock_proxy):
        """When proxy returns None, components falls back to local."""
        mock_comps.return_value = [{"id": "relu"}]
        resp = self.client.get("/api/designer/components")
        self.assertEqual(resp.status_code, 200)
        mock_comps.assert_called_once()

    # --- Templates and export/python have no proxy (local-only) ---

    def test_templates_always_local(self):
        """GET /api/designer/templates always returns local templates."""
        resp = self.client.get("/api/designer/templates")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) >= 2)

    @patch("research.scientist.api.generate_python_module")
    def test_export_python_always_local(self, mock_gen):
        """POST /api/designer/export/python always uses local generation."""
        mock_gen.return_value = "import torch\n"
        resp = self.client.post("/api/designer/export/python",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("code", data)

    # --- Error semantics ---

    @patch("research.scientist.api._designer_proxy")
    def test_proxy_4xx_forwarded(self, mock_proxy):
        """4xx from proxy is forwarded to client."""
        mock_proxy.return_value = _make_mock_response(
            422, {"detail": "Validation error: missing nodes"}
        )
        resp = self.client.post("/api/designer/validate",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 422)
        data = resp.get_json()
        self.assertIn("detail", data)

    @patch("research.scientist.api._designer_proxy")
    def test_proxy_5xx_forwarded(self, mock_proxy):
        """5xx from proxy is forwarded to client."""
        mock_proxy.return_value = _make_mock_response(
            500, {"detail": "Internal server error"}
        )
        resp = self.client.post("/api/designer/compile",
                                json=_SAMPLE_WORKFLOW,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 500)

    def test_missing_body_returns_400(self):
        """POST without JSON body returns 400."""
        resp = self.client.post("/api/designer/validate",
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_save_missing_workflow_id_returns_400(self):
        """POST /api/designer/save without workflow_id returns 400."""
        resp = self.client.post("/api/designer/save",
                                json={"name": "Test"},
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_import_missing_result_id_returns_400(self):
        """POST /api/designer/import without result_id returns 400."""
        resp = self.client.post("/api/designer/import",
                                json={},
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)


@unittest.skipUnless(HAS_FLASK, "Flask required")
class TestDesignerProxyConfig(unittest.TestCase):
    """Test proxy configuration via environment variables."""

    def test_default_proxy_base(self):
        import research.scientist.api as api_mod
        self.assertEqual(api_mod._DESIGNER_PROXY_BASE, "http://127.0.0.1:8091")

    def test_default_proxy_enabled(self):
        import research.scientist.api as api_mod
        self.assertTrue(api_mod._DESIGNER_PROXY_ENABLED)

    def test_default_proxy_timeout(self):
        import research.scientist.api as api_mod
        self.assertEqual(api_mod._DESIGNER_PROXY_TIMEOUT, 10.0)


if __name__ == "__main__":
    unittest.main()
