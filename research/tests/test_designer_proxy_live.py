"""Live integration test for research -> aria_designer proxy path.

This test is intentionally lightweight and only runs when aria_designer API is
reachable at http://127.0.0.1:8091.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import requests


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _designer_is_up() -> bool:
    try:
        resp = requests.get("http://127.0.0.1:8091/health", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.designer,
    pytest.mark.skipif(
        not _designer_is_up(),
        reason="aria_designer backend is not running on 127.0.0.1:8091",
    ),
]


def _sample_workflow() -> dict:
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_live_proxy_validate",
        "name": "Live Proxy Validate",
        "nodes": [
            {"id": "n0", "component_type": "io/input", "params": {}, "ui_meta": {}},
            {"id": "n1", "component_type": "math/relu", "params": {}, "ui_meta": {}},
            {
                "id": "n2",
                "component_type": "io/output_head",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "n0",
                "source_port": "y",
                "target": "n1",
                "target_port": "x",
            },
            {
                "id": "e1",
                "source": "n1",
                "source_port": "y",
                "target": "n2",
                "target_port": "x",
            },
        ],
    }


def test_validate_uses_live_designer_proxy(monkeypatch: pytest.MonkeyPatch):
    import research.scientist.api as api_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_path = os.path.join(tmpdir, "live_proxy_test.db")
        app = api_mod.create_app(notebook_path=notebook_path)
        client = app.test_client()

        monkeypatch.setattr(api_mod, "_DESIGNER_PROXY_ENABLED", True)
        monkeypatch.setattr(api_mod, "_DESIGNER_PROXY_BASE", "http://127.0.0.1:8091")

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError(
                "Fallback validate_designer_graph should not be called"
            )

        monkeypatch.setattr(api_mod, "validate_designer_graph", _unexpected_fallback)

        resp = client.post("/api/designer/validate", json=_sample_workflow())
        assert resp.status_code == 200

        data = resp.get_json()
        assert isinstance(data, dict)
        assert data.get("valid") is True
