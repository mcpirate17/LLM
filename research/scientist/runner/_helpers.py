"""Shared helper functions for the runner package — aggregator.

Previously a 2,160-line god file. The implementations now live in three
focused sub-modules; this module re-exports their public API so existing
``from ._helpers import X`` call sites continue to work unchanged.
"""

from __future__ import annotations

from ._helpers_gate import (  # noqa: F401
    InflightState,
    _build_source_map,
    _corpus_type_from_config,
    _headroom_ratio_threshold,
    _unload_ollama_if_running,
    check_inflight_health,
    clear_gpu_memory,
    get_reference_losses,
    normalized_loss_ratio,
    resolve_stage1_gate_metrics,
    stage1_learning_gate,
)
from ._helpers_metrics import (  # noqa: F401
    _graph_matching_ops,
    _load_best_reference_probe_ppl,
    _native_proactive_gating,
    _native_runner_progress_report,
    _rebuild_graph_with_overrides,
    _trajectory_probe_capability_tier,
    apply_adaptive_grad_clip,
    compute_seed_metrics,
    graph_observed_routing_ops,
    graph_routing_ops,
    propose_ablation_suite,
    routing_fast_lane_fields,
    screening_probe_fields,
    screening_wikitext_fields,
    trajectory_probe_fields,
)
from ._helpers_benchmark import (  # noqa: F401
    SSELogHandler,
    _build_benchmark_model,
    _evaluate_investigation_benchmarks,
    _record_investigation_result,
    _safe_tier,
    _submit_benchmark_eval,
    _upsert_screening_entry,
    build_validation_entry,
    handle_breakthrough,
    promote_validation_candidate,
    run_baseline_comparison,
    run_trajectory_probe,
)

__all__ = [
    "InflightState",
    "_build_source_map",
    "_corpus_type_from_config",
    "_headroom_ratio_threshold",
    "_unload_ollama_if_running",
    "check_inflight_health",
    "clear_gpu_memory",
    "get_reference_losses",
    "normalized_loss_ratio",
    "resolve_stage1_gate_metrics",
    "stage1_learning_gate",
    "_graph_matching_ops",
    "_load_best_reference_probe_ppl",
    "_native_proactive_gating",
    "_native_runner_progress_report",
    "_rebuild_graph_with_overrides",
    "_trajectory_probe_capability_tier",
    "apply_adaptive_grad_clip",
    "compute_seed_metrics",
    "graph_observed_routing_ops",
    "graph_routing_ops",
    "propose_ablation_suite",
    "routing_fast_lane_fields",
    "screening_probe_fields",
    "screening_wikitext_fields",
    "trajectory_probe_fields",
    "SSELogHandler",
    "_build_benchmark_model",
    "_evaluate_investigation_benchmarks",
    "_record_investigation_result",
    "_safe_tier",
    "_submit_benchmark_eval",
    "_upsert_screening_entry",
    "build_validation_entry",
    "handle_breakthrough",
    "promote_validation_candidate",
    "run_baseline_comparison",
    "run_trajectory_probe",
]
