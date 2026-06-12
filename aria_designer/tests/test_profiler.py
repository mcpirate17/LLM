"""Tests for the runtime profiler."""

from _workflows import make_attention_workflow, make_mlp_workflow
import json

from aria_designer.runtime.bridge import workflow_to_graph
from aria_designer.runtime.profiler import (
    profile_static,
    profile_static_graph,
    profile_runtime,
)


# ── Static profiling tests ───────────────────────────────────────────


def test_static_profile_mlp():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    assert report.total_params > 0
    assert report.total_flops_per_token > 0
    assert report.total_memory_bytes > 0
    assert len(report.op_profiles) == 3  # linear, gelu, linear
    assert report.native_coverage > 0


def test_static_profile_attention():
    report = profile_static(
        make_attention_workflow(out_dim=128, ports=False), model_dim=128
    )
    assert report.total_params > 0
    assert len(report.op_profiles) == 7


def test_static_profile_graph_reuses_existing_graph():
    graph = workflow_to_graph(make_mlp_workflow(activation="gelu"), model_dim=256)
    report = profile_static_graph(graph, model_dim=256)
    assert report.total_params > 0
    assert len(report.op_profiles) == 3


def test_static_flops_breakdown():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    assert "parameterized" in report.flops_by_category
    assert report.flops_by_category["parameterized"] > 0


def test_static_params_breakdown():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    assert "parameterized" in report.params_by_category
    # Only parameterized ops should have params
    for cat, count in report.params_by_category.items():
        if cat != "parameterized":
            assert count == 0, f"Non-parameterized category {cat} has {count} params"


def test_static_bottleneck_detection():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    assert len(report.bottleneck_ops) > 0
    assert "linear_proj" in report.bottleneck_ops[0]


def test_static_native_coverage():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    # linear_proj, gelu are all native
    assert report.native_coverage == 1.0


def test_static_json_serializable():
    report = profile_static(make_mlp_workflow(activation="gelu"), model_dim=256)
    d = report.to_dict()
    json_str = json.dumps(d)
    assert "total_params" in json_str
    assert "perf_contract" in d
    assert d["perf_contract"]["component"] == "aria_designer"


# ── Runtime profiling tests ──────────────────────────────────────────


def test_runtime_profile_mlp():
    report = profile_runtime(
        make_mlp_workflow(activation="gelu"),
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
