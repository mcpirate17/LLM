"""Tests for the runtime profiler."""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.profiler import profile_static, profile_runtime, ProfileReport


def _simple_mlp():
    return {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "n2", "component_type": "gelu", "params": {}},
            {"id": "n3", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "n4", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1", "source_port": "out", "target_port": "in"},
            {"id": "e1", "source": "n1", "target": "n2", "source_port": "out", "target_port": "in"},
            {"id": "e2", "source": "n2", "target": "n3", "source_port": "out", "target_port": "in"},
            {"id": "e3", "source": "n3", "target": "n4", "source_port": "out", "target_port": "in"},
        ],
    }


def _attention_workflow():
    return {
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}},
            {"id": "q", "component_type": "linear_proj", "params": {"out_dim": 128}},
            {"id": "k", "component_type": "linear_proj", "params": {"out_dim": 128}},
            {"id": "v", "component_type": "linear_proj", "params": {"out_dim": 128}},
            {"id": "attn", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "av", "component_type": "matmul", "params": {}},
            {"id": "proj", "component_type": "linear_proj", "params": {"out_dim": 128}},
            {"id": "out", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "in", "target": "q"},
            {"id": "e1", "source": "in", "target": "k"},
            {"id": "e2", "source": "in", "target": "v"},
            {"id": "e3", "source": "q", "target": "attn"},
            {"id": "e4", "source": "k", "target": "attn"},
            {"id": "e5", "source": "attn", "target": "sm"},
            {"id": "e6", "source": "sm", "target": "av"},
            {"id": "e7", "source": "v", "target": "av"},
            {"id": "e8", "source": "av", "target": "proj"},
            {"id": "e9", "source": "proj", "target": "out"},
        ],
    }


# ── Static profiling tests ───────────────────────────────────────────

def test_static_profile_mlp():
    report = profile_static(_simple_mlp(), model_dim=256)
    assert report.total_params > 0
    assert report.total_flops_per_token > 0
    assert report.total_memory_bytes > 0
    assert len(report.op_profiles) == 3  # linear, gelu, linear
    assert report.native_coverage > 0


def test_static_profile_attention():
    report = profile_static(_attention_workflow(), model_dim=128)
    assert report.total_params > 0
    assert len(report.op_profiles) == 7


def test_static_flops_breakdown():
    report = profile_static(_simple_mlp(), model_dim=256)
    assert "parameterized" in report.flops_by_category
    assert report.flops_by_category["parameterized"] > 0


def test_static_params_breakdown():
    report = profile_static(_simple_mlp(), model_dim=256)
    assert "parameterized" in report.params_by_category
    # Only parameterized ops should have params
    for cat, count in report.params_by_category.items():
        if cat != "parameterized":
            assert count == 0, f"Non-parameterized category {cat} has {count} params"


def test_static_bottleneck_detection():
    report = profile_static(_simple_mlp(), model_dim=256)
    assert len(report.bottleneck_ops) > 0
    assert "linear_proj" in report.bottleneck_ops[0]


def test_static_native_coverage():
    report = profile_static(_simple_mlp(), model_dim=256)
    # linear_proj, gelu are all native
    assert report.native_coverage == 1.0


def test_static_json_serializable():
    report = profile_static(_simple_mlp(), model_dim=256)
    d = report.to_dict()
    json_str = json.dumps(d)
    assert "total_params" in json_str


# ── Runtime profiling tests ──────────────────────────────────────────

def test_runtime_profile_mlp():
    report = profile_runtime(
        _simple_mlp(),
        model_dim=256,
        device="cpu",
        warmup_iters=1,
        bench_iters=2,
    )
    assert report.forward_time_ms > 0
    assert report.backward_time_ms > 0
    assert report.throughput_tokens_per_sec > 0
    # Static fields should also be filled
    assert report.total_params > 0
    assert report.total_flops_per_token > 0
