from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from flask import Response

import pytest

import research.scientist.api as api_mod


pytestmark = pytest.mark.designer


def test_api_v1_stream_routes_use_sse_proxy():
    """Embedded designer stream routes must preserve SSE responses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = api_mod.create_app(notebook_path=os.path.join(tmpdir, "stream_proxy.db"))
        client = app.test_client()

        with patch("research.scientist.api_routes.misc_bp.proxy_stream") as mock_stream, patch(
            "research.scientist.api_routes.misc_bp.designer_proxy"
        ) as mock_proxy:
            mock_stream.return_value = Response(
                'event: run_id\ndata: {"run_id":"eval_test"}\n\n',
                status=200,
                content_type="text/event-stream",
            )

            resp = client.post(
                "/api/v1/workflows/evaluate/stream",
                json={
                    "workflow": {
                        "schema_version": "workflow_graph.v1",
                        "workflow_id": "wf_stream_proxy",
                        "name": "Stream Proxy Test",
                        "nodes": [],
                        "edges": [],
                    },
                    "budget": {},
                },
            )

            assert resp.status_code == 200
            assert resp.content_type == "text/event-stream"
            assert b"event: run_id" in resp.data
            mock_stream.assert_called_once()
            mock_proxy.assert_not_called()
