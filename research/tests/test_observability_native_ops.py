from __future__ import annotations

import json

from research.scientist.api_routes import _observability_core as obs


def test_build_op_index_prefers_native_bridge(monkeypatch):
    class FakeRust:
        def build_op_index_from_rows(self, rows_json):
            rows = json.loads(rows_json)
            assert rows == [
                {
                    "graph_json": '{"nodes":[{"op_name":"gelu"},{"op_name":"linear_proj"}]}',
                    "stage0_passed": True,
                    "stage1_passed": False,
                    "loss_ratio": 0.2,
                    "error_type": "shape_mismatch",
                    "failure_op": "gelu",
                    "failure_details_json": '{"failure_op":"gelu"}',
                }
            ]
            return json.dumps(
                {
                    "pair_counts": [
                        {
                            "op_a": "gelu",
                            "op_b": "linear_proj",
                            "n": 4,
                            "s0": 3,
                            "s1": 2,
                        }
                    ],
                    "loss_by_op": [
                        {"op": "gelu", "values": [0.2, 0.4]},
                    ],
                    "failure_groups": [
                        {
                            "name": "shape_mismatch",
                            "count": 2,
                            "ops": [{"op": "gelu", "count": 2}],
                        }
                    ],
                    "stored_rates": [
                        {"op": "gelu", "n": 4, "s0": 3, "s1": 2},
                    ],
                    "corrected_rates": [
                        {"op": "gelu", "n": 3, "s0": 3, "s1": 2, "excluded": 1},
                    ],
                }
            )

    obs.refresh_observability_caches()
    monkeypatch.setattr(obs, "_try_import_rust_scheduler", lambda: FakeRust())
    monkeypatch.setattr(
        obs,
        "_load_program_rows",
        lambda nb, window: [
            {
                "graph_json": '{"nodes":[{"op_name":"gelu"},{"op_name":"linear_proj"}]}',
                "stage0_passed": 1,
                "stage1_passed": 0,
                "loss_ratio": 0.2,
                "error_type": "shape_mismatch",
                "failure_op": "gelu",
                "failure_details_json": '{"failure_op":"gelu"}',
            }
        ],
    )

    class DummyNotebook:
        pass

    monkeypatch.setattr(obs, "get_notebook", lambda path: DummyNotebook())

    result = obs.build_op_index("/tmp/native.sqlite", window="all")

    assert result["pair_counts"] == {
        ("gelu", "linear_proj"): {"n": 4, "s0": 3, "s1": 2}
    }
    assert result["loss_by_op"] == {"gelu": [0.2, 0.4]}
    assert result["failure_groups"] == {
        "shape_mismatch": {"count": 2, "ops": {"gelu": 2}}
    }
    assert result["stored_rates"] == {"gelu": {"n": 4, "s0": 3, "s1": 2}}
    assert result["corrected_rates"] == {
        "gelu": {"n": 3, "s0": 3, "s1": 2, "excluded": 1}
    }


def test_get_cached_alerts_reuses_recent_snapshot(monkeypatch):
    obs.refresh_observability_caches()
    calls = {"n": 0}

    def fake_evaluate(notebook_path, thresholds):
        calls["n"] += 1
        return [{"id": "x", "severity": "info"}]

    monkeypatch.setattr(obs, "evaluate_alerts", fake_evaluate)

    first = obs.get_cached_alerts("/tmp/native.sqlite", {"s0_pass_rate_min": 0.3})
    second = obs.get_cached_alerts("/tmp/native.sqlite", {"s0_pass_rate_min": 0.3})

    assert first == second == [{"id": "x", "severity": "info"}]
    assert calls["n"] == 1
