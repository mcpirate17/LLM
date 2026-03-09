"""Tests for the /api/native-profile/v2/data endpoints."""

import pytest
import json
import unittest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.api


class TestNativeProfileApiEndpoints(unittest.TestCase):
    """Test GET /api/native-profile/v2/data and POST /api/native-profile/v2/enable."""

    @classmethod
    def setUpClass(cls):
        from research.scientist.api import create_app

        cls.app = create_app(notebook_path=":memory:")
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    # ── GET /api/native-profile/v2/data ────────────────────────────────

    def test_get_profile_empty_when_no_execution(self):
        """GET returns empty profile when no profiled execution has happened."""
        with patch("research.scientist.native_runner.get_native_profile", return_value=None), \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=None):
            resp = self.client.get("/api/native-profile/v2/data")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["enabled"], False)
        self.assertEqual(data["node_profiles"], [])
        self.assertEqual(data["peak_memory_bytes"], 0)
        self.assertEqual(data["total_duration_us"], 0.0)

    def test_get_profile_returns_real_data_after_execution(self):
        """GET returns node_profiles and timing data after a profiled execution."""
        fake_profile = {
            "node_profiles": [
                {"node_id": 0, "op_name": "input", "duration_us": 1.5, "start_ns": 100, "end_ns": 1600},
                {"node_id": 1, "op_name": "relu", "duration_us": 3.2, "start_ns": 1700, "end_ns": 4900},
                {"node_id": 2, "op_name": "add", "duration_us": 2.0, "start_ns": 5000, "end_ns": 7000},
            ],
            "peak_memory_bytes": 4096,
        }

        mock_rust = MagicMock()
        mock_rust.profiler_enabled.return_value = True

        with patch("research.scientist.native_runner.get_native_profile", return_value=fake_profile), \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=mock_rust):
            resp = self.client.get("/api/native-profile/v2/data")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["enabled"])
        self.assertEqual(len(data["node_profiles"]), 3)
        self.assertEqual(data["peak_memory_bytes"], 4096)
        self.assertAlmostEqual(data["total_duration_us"], 6.7, places=5)

        # Verify individual node profile fields
        relu_profile = data["node_profiles"][1]
        self.assertEqual(relu_profile["node_id"], 1)
        self.assertEqual(relu_profile["op_name"], "relu")
        self.assertAlmostEqual(relu_profile["duration_us"], 3.2)

    # ── POST /api/native-profile/v2/enable ────────────────────────

    def test_post_enable_profiling_on(self):
        """POST with enable=true toggles profiling on."""
        mock_rust = MagicMock()
        mock_rust.profiler_enabled.return_value = True

        with patch("research.scientist.native_runner.enable_native_profiling", return_value=True) as mock_enable, \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=mock_rust):
            resp = self.client.post(
                "/api/native-profile/v2/enable",
                data=json.dumps({"enable": True}),
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["requested"])
        self.assertTrue(data["enabled"])
        self.assertTrue(data["accepted"])
        mock_enable.assert_called_once_with(True)

    def test_post_disable_profiling(self):
        """POST with enable=false toggles profiling off."""
        mock_rust = MagicMock()
        mock_rust.profiler_enabled.return_value = False

        with patch("research.scientist.native_runner.enable_native_profiling", return_value=True) as mock_enable, \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=mock_rust):
            resp = self.client.post(
                "/api/native-profile/v2/enable",
                data=json.dumps({"enable": False}),
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["requested"])
        self.assertFalse(data["enabled"])
        mock_enable.assert_called_once_with(False)

    def test_post_enable_no_rust_scheduler(self):
        """POST returns accepted=False when Rust scheduler is unavailable."""
        with patch("research.scientist.native_runner.enable_native_profiling", return_value=False) as mock_enable, \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=None):
            resp = self.client.post(
                "/api/native-profile/v2/enable",
                data=json.dumps({"enable": True}),
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["requested"])
        self.assertFalse(data["enabled"])
        self.assertFalse(data["accepted"])

    def test_post_enable_defaults_to_true(self):
        """POST without body defaults enable to True."""
        mock_rust = MagicMock()
        mock_rust.profiler_enabled.return_value = True

        with patch("research.scientist.native_runner.enable_native_profiling", return_value=True) as mock_enable, \
             patch("research.scientist.native_runner._try_import_rust_scheduler", return_value=mock_rust):
            resp = self.client.post(
                "/api/native-profile/v2/enable",
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["requested"])
        mock_enable.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
