from __future__ import annotations
"""
Auto-extracted mixin for LabNotebook.
"""
"""
Electronic Lab Notebook

Persistent, structured record of all experiments, hypotheses,
observations, and conclusions. Stored as SQLite for queryability
and served to the React dashboard via API.
"""



import json
import logging
import math
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..preregistration import PreregistrationError, validate_preregistration
except Exception:  # direct-module loading fallback for test harness
    import importlib.util as _importlib_util
    import sys as _sys

    _prereg_path = Path(__file__).parent.parent / "preregistration.py"
    _prereg_spec = _importlib_util.spec_from_file_location(
        "_notebook_preregistration_fallback",
        str(_prereg_path),
    )
    _prereg_mod = _importlib_util.module_from_spec(_prereg_spec)
    assert _prereg_spec is not None and _prereg_spec.loader is not None
    _sys.modules[_prereg_spec.name] = _prereg_mod
    _prereg_spec.loader.exec_module(_prereg_mod)
    PreregistrationError = _prereg_mod.PreregistrationError
    validate_preregistration = _prereg_mod.validate_preregistration

LOGGER = logging.getLogger(__name__)

_INSIGHT_TOP_OPS_RE = re.compile(r"^Top-performing ops \(S1 rate\):\s*(.+?)\.\s")
_INSIGHT_WINNING_COMBO_RE = re.compile(r"^Winning combination:\s*(.+?)\s+appears in\s+\d+\s+survivors")
_INSIGHT_FAILING_OPS_RE = re.compile(r"^Consistently failing ops:\s*(.+?)\.\s")
_INSIGHT_GRAPH_CORR_RE = re.compile(r"^Graph\s+(.+?)\s+is\s+(?:positively|negatively)\s+correlated")
_INSIGHT_COMMON_FAILURE_RE = re.compile(r"^Most common failure:\s*(.+?)\s+\(")
_INSIGHT_STANDALONE_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _insight_ops_key(text: str) -> str:
    ops = []
    for token in (text or "").split(","):
        name = token.strip()
        if not name:
            continue
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if name:
            ops.append(name)
    if not ops:
        return ""
    return "+".join(sorted(set(ops)))


def infer_insight_identity(category: str, content: str) -> Tuple[str, str, str]:
    """Infer ``(insight_type, subject_key, semantic_key)`` from insight text."""
    text = str(content or "").strip()
    cat = str(category or "pattern").strip() or "pattern"

    m = _INSIGHT_TOP_OPS_RE.match(text)
    if m:
        subject_key = _insight_ops_key(m.group(1))
        if subject_key:
            return ("top_ops", subject_key, f"top_ops:{subject_key}")

    m = _INSIGHT_WINNING_COMBO_RE.match(text)
    if m:
        raw_ops = [part.strip() for part in m.group(1).split("+")]
        subject_key = "+".join(sorted({op for op in raw_ops if op}))
        if subject_key:
            return ("winning_combo", subject_key, f"winning_combo:{subject_key}")

    m = _INSIGHT_FAILING_OPS_RE.match(text)
    if m:
        subject_key = _insight_ops_key(m.group(1))
        if subject_key:
            return ("failing_ops", subject_key, f"failing_ops:{subject_key}")

    m = _INSIGHT_GRAPH_CORR_RE.match(text)
    if m:
        subject_key = re.sub(r"\s+", "_", m.group(1).strip().lower())
        return ("graph_correlation", subject_key, f"graph_correlation:{subject_key}")

    m = _INSIGHT_COMMON_FAILURE_RE.match(text)
    if m:
        subject_key = re.sub(r"\s+", "_", m.group(1).strip().lower())
        return ("common_failure", subject_key, f"common_failure:{subject_key}")

    if text.startswith("Overall survival rate:"):
        return ("overall_survival_rate", "global", "overall_survival_rate:global")

    if text.startswith("Found ") and "genuinely novel survivors" in text:
        return ("novel_survivors", "global", "novel_survivors:global")

    if text.startswith("Strong Stage 1 pass rate"):
        return ("stage1_pass_rate", "global", "stage1_pass_rate:global")

    normalized = _INSIGHT_STANDALONE_NUM_RE.sub("#", text)
    semantic = f"text:{cat}:{normalized[:240]}"
    return (f"text_{cat}", normalized[:120], semantic)


NOTEBOOK_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    experiment_type TEXT NOT NULL,  -- 'synthesis', 'morphological', 'training', 'evolution'
    status TEXT NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'failed', 'aborted'

    -- Hypothesis
    hypothesis TEXT,
    research_question TEXT,
    preregistration_id TEXT,

    -- Configuration
    config_json TEXT NOT NULL,

    -- Results (filled after completion)
    results_json TEXT,
    n_programs_generated INTEGER DEFAULT 0,
    n_stage0_passed INTEGER DEFAULT 0,
    n_stage05_passed INTEGER DEFAULT 0,
    n_stage1_passed INTEGER DEFAULT 0,
    best_loss_ratio REAL,
    best_novelty_score REAL,

    -- Aria's analysis
    aria_summary TEXT,
    aria_mood TEXT,
    insights_json TEXT,
    llm_analysis TEXT,

    -- Timing
    started_at REAL,
    completed_at REAL,
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS hypothesis_preregistrations (
    preregistration_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    experiment_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'registered', -- 'registered' | 'linked' | 'completed'
    hypothesis_json TEXT NOT NULL,
    analysis_plan_json TEXT NOT NULL,
    falsification_json TEXT NOT NULL,
    confounders_json TEXT NOT NULL,
    exploratory INTEGER DEFAULT 0,
    created_by TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS preregistration_deviations (
    deviation_id TEXT PRIMARY KEY,
    preregistration_id TEXT REFERENCES hypothesis_preregistrations(preregistration_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp REAL NOT NULL,
    deviation_type TEXT NOT NULL, -- 'exploratory'
    rationale TEXT NOT NULL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS healer_tasks (
    task_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    trigger_type TEXT NOT NULL,
    trigger_payload_json TEXT,
    scope TEXT NOT NULL,
    reproduction_steps_json TEXT,
    acceptance_tests_json TEXT,
    model_endpoint TEXT,
    sandbox_policy_json TEXT,
    state TEXT NOT NULL, -- 'open'|'reproducing'|'patch_proposed'|'verifying'|'completed'|'failed'|'blocked'
    patch_summary TEXT,
    risk_assessment TEXT,
    result_json TEXT,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS healer_task_events (
    event_id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES healer_tasks(task_id),
    timestamp REAL NOT NULL,
    state TEXT,
    message TEXT NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp REAL NOT NULL,
    entry_type TEXT NOT NULL,  -- 'hypothesis', 'observation', 'result', 'analysis',
                               -- 'error', 'insight', 'decision', 'note'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    tags TEXT  -- comma-separated
);

CREATE TABLE IF NOT EXISTS program_results (
    result_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp REAL NOT NULL,
    graph_fingerprint TEXT NOT NULL,
    graph_json TEXT NOT NULL,

    -- Stage results
    stage0_passed INTEGER,
    stage05_passed INTEGER,
    stage1_passed INTEGER,
    stage0_error TEXT,

    -- Metrics
    param_count INTEGER,
    loss_ratio REAL,
    final_loss REAL,
    throughput_tok_s REAL,

    -- Novelty
    novelty_score REAL,
    structural_novelty REAL,
    behavioral_novelty REAL,
    most_similar_to TEXT,
    novelty_confidence REAL,

    -- Fingerprint
    fingerprint_json TEXT,

    -- Training program used
    training_program_json TEXT,

    -- Sandbox metrics (Stage 0)
    compile_time_ms REAL,
    forward_time_ms REAL,
    backward_time_ms REAL,
    peak_memory_mb REAL,
    grad_norm REAL,
    stability_score REAL,
    extreme_input_passed INTEGER,
    random_input_passed INTEGER,
    output_range_min REAL,
    output_range_max REAL,
    has_nan_output INTEGER,
    has_inf_output INTEGER,
    has_nan_grad INTEGER,
    has_zero_grad INTEGER,
    error_type TEXT,
    error_message TEXT,
    stage_at_death TEXT,
    regression_gate_pass INTEGER,
    regression_gate_reason TEXT,

    -- Training metrics (Stage 1)
    initial_loss REAL,
    min_loss REAL,
    loss_improvement_rate REAL,
    discovery_loss REAL,
    discovery_loss_ratio REAL,
    validation_loss REAL,
    validation_loss_ratio REAL,
    generalization_gap REAL,
    avg_step_time_ms REAL,
    total_train_time_ms REAL,
    max_grad_norm REAL,
    mean_grad_norm REAL,
    grad_norm_std REAL,
    n_train_steps INTEGER,
    final_lr REAL,
    perf_traces_json TEXT,
    gpu_starvation_json TEXT,
    kernel_timing_json TEXT,
    queue_telemetry_json TEXT,

    -- Fingerprint metrics
    fp_interaction_locality REAL,
    fp_interaction_sparsity REAL,
    fp_interaction_symmetry REAL,
    fp_interaction_hierarchy REAL,
    fp_intrinsic_dim REAL,
    fp_isotropy REAL,
    fp_rank_ratio REAL,
    fp_jacobian_spectral_norm REAL,
    fp_jacobian_effective_rank REAL,
    fp_sensitivity_uniformity REAL,
    fp_cka_vs_transformer REAL,
    fp_cka_vs_ssm REAL,
    fp_cka_vs_conv REAL,

    -- Graph structural metrics
    graph_n_ops INTEGER,
    graph_depth INTEGER,
    graph_n_params_estimate INTEGER,
    graph_has_gradient_path INTEGER,
    graph_n_edges INTEGER,
    graph_n_unique_ops INTEGER,
    graph_category_histogram TEXT,
    graph_uses_math_spaces INTEGER,
    graph_uses_frequency_domain INTEGER,

    -- FLOP metrics
    flops_forward INTEGER,
    flops_per_param REAL,
    flops_per_token REAL,

    -- Sparse telemetry metrics
    sparse_density_mean REAL,
    sparse_density_last REAL,
    sparse_fallback_calls INTEGER,
    sparse_kernel_fallback_calls INTEGER,
    sparse_nm_compliance REAL,
    sparse_active_params_estimate INTEGER,
    sparse_telemetry_json TEXT,
    sparsity_ratio REAL,

    -- One-shot pruning baseline metrics
    pruning_method TEXT,
    pruning_target_sparsity REAL,
    pruning_actual_sparsity REAL,
    pruning_n_params_total INTEGER,
    pruning_n_params_pruned INTEGER,
    pruning_dense_eval_loss REAL,
    pruning_pruned_eval_loss REAL,
    pruning_quality_retention REAL,
    pruning_active_params_estimate INTEGER,
    pruning_error TEXT,

    -- Performance telemetry
    perf_report_json TEXT,
    kernel_timings_json TEXT,
    starvation_report_json TEXT,

    -- Baseline comparison
    baseline_loss_ratio REAL,

    -- Novelty calibration + validity
    novelty_raw_score REAL,
    novelty_z_score REAL,
    novelty_reference_version TEXT,
    novelty_valid_for_promotion INTEGER,
    novelty_validity_reason TEXT,
    novelty_requires_justification INTEGER,
    cka_probe_protocol_hash TEXT,
    cka_reference_quality TEXT,

    -- Adaptive compute telemetry (MoD/MoR)
    depth_savings_ratio REAL,
    effective_depth_ratio REAL,
    recursion_savings_ratio REAL,
    recursion_depth_ratio REAL,

    -- External benchmarks
    external_benchmarks_json TEXT
);

CREATE TABLE IF NOT EXISTS metrics_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    experiment_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS insights (
    insight_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    experiment_id TEXT,
    category TEXT NOT NULL,  -- 'pattern', 'failure_mode', 'success_factor', 'hypothesis'
    insight_type TEXT,
    subject_key TEXT,
    semantic_key TEXT,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    supporting_evidence TEXT,
    status TEXT DEFAULT 'active',  -- 'active', 'confirmed', 'refuted', 'superseded'
    confirmation_strength REAL DEFAULT 0.0,  -- quantitative confidence score
    independent_validations INTEGER DEFAULT 0  -- count of independent confirmation attempts
);

CREATE TABLE IF NOT EXISTS training_curves (
    result_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    loss REAL,
    grad_norm REAL,
    step_time_ms REAL,
    PRIMARY KEY (result_id, step)
);

CREATE TABLE IF NOT EXISTS op_success_rates (
    op_name TEXT PRIMARY KEY,
    n_used INTEGER DEFAULT 0,
    n_stage0_passed INTEGER DEFAULT 0,
    n_stage05_passed INTEGER DEFAULT 0,
    n_stage1_passed INTEGER DEFAULT 0,
    avg_loss_ratio REAL,
    avg_novelty REAL,
    avg_novelty_confidence REAL,
    last_updated REAL
);

CREATE TABLE IF NOT EXISTS failure_signatures (
    signature TEXT PRIMARY KEY,
    n_failures INTEGER DEFAULT 0,
    n_successes INTEGER DEFAULT 0,
    error_types TEXT,
    last_updated REAL
);

CREATE TABLE IF NOT EXISTS op_rehabilitation_cache (
    op_name TEXT PRIMARY KEY,
    compile_passed INTEGER,
    forward_passed INTEGER,
    error_message TEXT,
    tested_at REAL,
    model_dim INTEGER
);

CREATE TABLE IF NOT EXISTS learning_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT,
    old_weights TEXT,
    new_weights TEXT,
    evidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_experiment ON entries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_programs_experiment ON program_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_programs_novelty ON program_results(novelty_score);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_log(metric_name);
CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
CREATE INDEX IF NOT EXISTS idx_training_curves_result ON training_curves(result_id);
CREATE INDEX IF NOT EXISTS idx_learning_log_type ON learning_log(event_type);

CREATE TABLE IF NOT EXISTS leaderboard (
    entry_id TEXT PRIMARY KEY,
    result_id TEXT REFERENCES program_results(result_id),
    timestamp REAL NOT NULL,
    -- Source
    model_source TEXT NOT NULL,  -- 'graph_synthesis' or 'morphological_box'
    architecture_desc TEXT,       -- human-readable description
    -- Screening results (Phase 1)
    screening_loss_ratio REAL,
    screening_novelty REAL,
    screening_passed INTEGER DEFAULT 0,
    -- Investigation results (Phase 2)
    investigation_loss_ratio REAL,
    investigation_robustness REAL,  -- fraction of training programs that work
    investigation_best_training TEXT,  -- JSON of best training program
    investigation_passed INTEGER DEFAULT 0,
    -- Validation results (Phase 3)
    validation_loss_ratio REAL,
    validation_baseline_ratio REAL,
    validation_multi_seed_std REAL,
    validation_passed INTEGER DEFAULT 0,
    -- Robustness & Efficiency (Detailed)
    normalized_baseline_ratio REAL,
    param_efficiency REAL,
    quant_int8_retention REAL,
    quant_quality_per_byte REAL,
    robustness_long_ctx_score REAL,
    robustness_noise_score REAL,
    init_sensitivity_std REAL,
    fp_jacobian_spectral_norm REAL,
    -- Scaling
    scaling_param_efficiency REAL,
    scaling_flop_efficiency REAL,
    scaling_gate_passed INTEGER,
    scaling_best_family TEXT,
    scaling_d512_param_efficiency REAL,
    scaling_confidence TEXT,
    -- Composite score
    composite_score REAL,
    tier TEXT DEFAULT 'screening',  -- 'screening', 'investigation', 'validation', 'breakthrough'
    -- Metadata
    tags TEXT,
    notes TEXT,
    is_reference INTEGER DEFAULT 0,
    reference_name TEXT DEFAULT NULL,
    is_pinned INTEGER DEFAULT 0,
    routing_savings_ratio REAL,
    compression_ratio REAL,
    discovery_loss_ratio REAL
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_tier ON leaderboard(tier);
CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(composite_score);
CREATE INDEX IF NOT EXISTS idx_leaderboard_result ON leaderboard(result_id);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    success_criteria TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    parent_campaign_id TEXT,
    findings_summary TEXT,
    started_at REAL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    campaign_id TEXT REFERENCES campaigns(campaign_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp REAL NOT NULL,
    prediction TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    test_method TEXT NOT NULL,
    success_metric TEXT NOT NULL,
    parent_hypothesis_id TEXT,
    status TEXT DEFAULT 'pending',
    outcome_evidence TEXT,
    outcome_summary TEXT,
    child_hypotheses TEXT,
    confidence_before REAL,
    confidence_after REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    campaign_id TEXT REFERENCES campaigns(campaign_id),
    timestamp REAL NOT NULL,
    decision_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence_ids TEXT,
    alternatives_considered TEXT,
    evidence_pack_json TEXT,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS attribution_reports (
    report_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    hypothesis_id TEXT REFERENCES hypotheses(hypothesis_id),
    supporting_experiments TEXT,
    ablation_experiments TEXT,
    outcome TEXT,
    report_json TEXT
);

CREATE TABLE IF NOT EXISTS novelty_calibration (
    calibration_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    reference_version TEXT NOT NULL,
    cka_source TEXT,
    cka_artifact_version TEXT,
    probe_protocol_hash TEXT,
    n_runs INTEGER NOT NULL,
    noise_floor_mean REAL,
    noise_floor_std REAL,
    confidence_low REAL,
    confidence_high REAL,
    distribution_json TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS selection_decisions (
    decision_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    context TEXT NOT NULL,
    experiment_id TEXT,
    candidate_pool_summary_json TEXT,
    score_breakdown_json TEXT,
    policy_json TEXT,
    reason TEXT,
    chosen_experiments_json TEXT,
    trigger_json TEXT
);

CREATE TABLE IF NOT EXISTS selection_family_stats (
    family TEXT PRIMARY KEY,
    n_trials INTEGER DEFAULT 0,
    cumulative_reward REAL DEFAULT 0.0,
    mean_reward REAL DEFAULT 0.0,
    last_reward REAL,
    last_updated REAL
);

CREATE TABLE IF NOT EXISTS selection_insight_trials (
    trial_id TEXT PRIMARY KEY,
    decision_id TEXT REFERENCES selection_decisions(decision_id),
    timestamp REAL NOT NULL,
    context TEXT NOT NULL,
    source_experiment_id TEXT,
    insight_ids_json TEXT NOT NULL,
    chosen_result_ids_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reward REAL,
    outcome TEXT,
    resolved_timestamp REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS selection_insight_interactions (
    insight_a TEXT NOT NULL,
    insight_b TEXT NOT NULL,
    n_trials INTEGER DEFAULT 0,
    n_supported INTEGER DEFAULT 0,
    n_not_supported INTEGER DEFAULT 0,
    cumulative_reward REAL DEFAULT 0.0,
    mean_reward REAL DEFAULT 0.0,
    last_reward REAL,
    last_outcome TEXT,
    last_updated REAL,
    PRIMARY KEY (insight_a, insight_b)
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    entry_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    supporting_evidence TEXT,
    times_validated INTEGER DEFAULT 1,
    last_validated REAL,
    status TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_campaign ON hypotheses(campaign_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_experiment ON hypotheses(experiment_id);
CREATE INDEX IF NOT EXISTS idx_prereg_experiment ON hypothesis_preregistrations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_healer_tasks_experiment ON healer_tasks(experiment_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_healer_tasks_state ON healer_tasks(state, timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_campaign ON decisions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_attribution_hypothesis ON attribution_reports(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_novelty_calibration_ref ON novelty_calibration(reference_version, timestamp);
CREATE INDEX IF NOT EXISTS idx_selection_decisions_context ON selection_decisions(context);
CREATE INDEX IF NOT EXISTS idx_selection_insight_trials_status ON selection_insight_trials(status, timestamp);
CREATE INDEX IF NOT EXISTS idx_selection_insight_trials_context ON selection_insight_trials(context, timestamp);
CREATE INDEX IF NOT EXISTS idx_selection_insight_interactions_reward ON selection_insight_interactions(mean_reward DESC, n_trials DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_base(status);

CREATE TABLE IF NOT EXISTS aria_chat (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    role TEXT NOT NULL,          -- 'user', 'aria', 'system'
    text TEXT NOT NULL,
    label TEXT,
    compacted INTEGER DEFAULT 0,
    summary_of TEXT,             -- JSON list of message_ids this summarizes
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_aria_chat_session ON aria_chat(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_aria_chat_compacted ON aria_chat(session_id, compacted);

CREATE TABLE IF NOT EXISTS report_snapshots (
    snapshot_key TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    query_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    latest_completed_ts REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_report_snapshots_scope_updated ON report_snapshots(scope, updated_at);

CREATE TABLE IF NOT EXISTS workflow_definitions (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    graph_json TEXT NOT NULL,
    metadata_json TEXT,
    author TEXT DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS designer_run_lineage (
    run_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    workflow_version INTEGER,
    graph_fingerprint TEXT,
    status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'aria_designer',
    total_time_ms REAL,
    metrics_json TEXT,
    payload_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_designer_lineage_workflow ON designer_run_lineage(workflow_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_designer_lineage_status ON designer_run_lineage(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_digests (
    digest_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    cycle_number INTEGER,
    digest_json TEXT NOT NULL,
    narrative_summary TEXT,
    n_experiments_analyzed INTEGER,
    n_curves_analyzed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_knowledge_digests_ts ON knowledge_digests(timestamp DESC);
"""

# Columns added in the schema expansion — used for migration
_PROGRAM_RESULTS_NEW_COLUMNS = {
    "compile_time_ms": "REAL",
    "forward_time_ms": "REAL",
    "backward_time_ms": "REAL",
    "peak_memory_mb": "REAL",
    "grad_norm": "REAL",
    "stability_score": "REAL",
    "extreme_input_passed": "INTEGER",
    "random_input_passed": "INTEGER",
    "output_range_min": "REAL",
    "output_range_max": "REAL",
    "has_nan_output": "INTEGER",
    "has_inf_output": "INTEGER",
    "has_nan_grad": "INTEGER",
    "has_zero_grad": "INTEGER",
    "error_type": "TEXT",
    "error_message": "TEXT",
    "stage_at_death": "TEXT",
    "initial_loss": "REAL",
    "min_loss": "REAL",
    "loss_improvement_rate": "REAL",
    "discovery_loss": "REAL",
    "discovery_loss_ratio": "REAL",
    "validation_loss": "REAL",
    "validation_loss_ratio": "REAL",
    "generalization_gap": "REAL",
    "avg_step_time_ms": "REAL",
    "total_train_time_ms": "REAL",
    "max_grad_norm": "REAL",
    "mean_grad_norm": "REAL",
    "grad_norm_std": "REAL",
    "n_train_steps": "INTEGER",
    "final_lr": "REAL",
    "perf_traces_json": "TEXT",
    "gpu_starvation_json": "TEXT",
    "kernel_timing_json": "TEXT",
    "queue_telemetry_json": "TEXT",
    "perf_report_json": "TEXT",
    "kernel_timings_json": "TEXT",
    "starvation_report_json": "TEXT",
    "fp_interaction_locality": "REAL",
    "fp_interaction_sparsity": "REAL",
    "fp_interaction_symmetry": "REAL",
    "regression_gate_pass": "INTEGER",
    "regression_gate_reason": "TEXT",
    "fp_interaction_hierarchy": "REAL",
    "fp_intrinsic_dim": "REAL",
    "fp_isotropy": "REAL",
    "fp_rank_ratio": "REAL",
    "fp_jacobian_spectral_norm": "REAL",
    "fp_jacobian_effective_rank": "REAL",
    "fp_sensitivity_uniformity": "REAL",
    "fp_cka_vs_transformer": "REAL",
    "fp_cka_vs_ssm": "REAL",
    "fp_cka_vs_conv": "REAL",
    "graph_n_ops": "INTEGER",
    "graph_depth": "INTEGER",
    "graph_n_params_estimate": "INTEGER",
    "graph_has_gradient_path": "INTEGER",
    "graph_n_edges": "INTEGER",
    "graph_n_unique_ops": "INTEGER",
    "graph_category_histogram": "TEXT",
    "graph_uses_math_spaces": "INTEGER",
    "graph_uses_frequency_domain": "INTEGER",
    "flops_forward": "INTEGER",
    "flops_per_param": "REAL",
    "flops_per_token": "REAL",
    "sparse_density_mean": "REAL",
    "sparse_density_last": "REAL",
    "sparse_fallback_calls": "INTEGER",
    "sparse_kernel_fallback_calls": "INTEGER",
    "sparse_nm_compliance": "REAL",
    "sparse_active_params_estimate": "INTEGER",
    "sparse_telemetry_json": "TEXT",
    "sparsity_ratio": "REAL",
    "pruning_method": "TEXT",
    "pruning_target_sparsity": "REAL",
    "pruning_actual_sparsity": "REAL",
    "pruning_n_params_total": "INTEGER",
    "pruning_n_params_pruned": "INTEGER",
    "pruning_dense_eval_loss": "REAL",
    "pruning_pruned_eval_loss": "REAL",
    "pruning_quality_retention": "REAL",
    "pruning_active_params_estimate": "INTEGER",
    "pruning_error": "TEXT",
    "baseline_loss_ratio": "REAL",
    # Routing telemetry (Track A)
    "routing_mode": "TEXT",
    "routing_tokens_total": "INTEGER",
    "routing_tokens_processed": "INTEGER",
    "routing_tokens_skipped": "INTEGER",
    "routing_drop_rate": "REAL",
    "routing_utilization_entropy": "REAL",
    "routing_capacity_overflow_count": "INTEGER",
    "routing_confidence_mean": "REAL",
    "routing_confidence_std": "REAL",
    "routing_expert_utilization_json": "TEXT",
    "routing_expert_count": "INTEGER",
    "routing_savings_ratio": "REAL",
    "compression_ratio": "REAL",
    # Novelty calibration
    "novelty_confidence": "REAL",
    "novelty_raw_score": "REAL",
    "novelty_z_score": "REAL",
    "novelty_reference_version": "TEXT",
    "novelty_valid_for_promotion": "INTEGER",
    "novelty_validity_reason": "TEXT",
    "novelty_requires_justification": "INTEGER",
    # CKA provenance
    "cka_source": "TEXT",
    "cka_artifact_version": "TEXT",
    "cka_probe_protocol_hash": "TEXT",
    "cka_reference_quality": "TEXT",
    # Diagnostic tasks
    "diagnostic_tasks_json": "TEXT",
    "diagnostic_score": "REAL",
    # Adaptive compute telemetry (MoD/MoR)
    "depth_savings_ratio": "REAL",
    "effective_depth_ratio": "REAL",
    "recursion_savings_ratio": "REAL",
    "recursion_depth_ratio": "REAL",
    # External benchmarks
    "external_benchmarks_json": "TEXT",
    # Activation sparsity & routing collapse (interpretability)
    "activation_sparsity_score": "REAL",
    "dead_neuron_ratio": "REAL",
    "routing_collapse_score": "REAL",
    # WikiText perplexity (domain generalization)
    "wikitext_perplexity": "REAL",
    "wikitext_score": "REAL",
    # TinyStories (domain generalization)
    "tinystories_perplexity": "REAL",
    "tinystories_score": "REAL",
    # Cross-task robustness (code vs NL)
    "cross_task_score": "REAL",
    # Efficiency wall (memory/FLOP scaling)
    "efficiency_wall_score": "REAL",
    "max_viable_seq_len": "INTEGER",
    "scaling_regime": "TEXT",
    # Hierarchy detection (Gromov delta)
    "fp_hierarchy_fitness": "REAL",
    "fp_gromov_delta": "REAL",
    # NCD reward signal
    "ncd_score": "REAL",
    "ncd_description_length": "INTEGER",
    "ncd_description_length_per_param": "REAL",
    # Long-context sub-scores (RULER-style)
    "robustness_long_ctx_scaling_score": "REAL",
    "robustness_long_ctx_assoc_score": "REAL",
    "robustness_long_ctx_multi_hop_score": "REAL",
    "robustness_long_ctx_passkey_score": "REAL",
    "robustness_long_ctx_retrieval_aggregate": "REAL",
    "robustness_long_ctx_combined_score": "REAL",
    # Efficiency multiple (geomean of per-dimension ratios vs GPT-2)
    "efficiency_multiple": "REAL",
}


@dataclass
class ExperimentEntry:
    """A single lab notebook entry."""
    entry_type: str
    title: str
    content: str
    experiment_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)



class _AnalyticsMixin:
    """Analytics operations for the Lab Notebook."""

    # ── Op Success Rates ──

    def update_op_success_rates(self, experiment_id: str) -> None:
        """Recompute op success rates from program results in this experiment.

        Uses a targeted query (only needed columns) and avoids dict(r)
        conversion overhead from get_program_results.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage0_passed, stage05_passed, stage1_passed,
                      loss_ratio, novelty_score, novelty_confidence
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL""",
            (experiment_id,),
        ).fetchall()

        op_stats: Dict[str, Dict] = {}
        # Reusable reference to avoid repeated dict key hashing
        _OP_NAME = "op_name"

        for r in rows:
            graph_json = r[0]  # access by index — faster than by name
            if not graph_json:
                continue
            try:
                graph_data = json.loads(graph_json)
                nodes = graph_data.get("nodes", {})
            except (json.JSONDecodeError, TypeError):
                continue

            ops_in_graph = set()
            for node_data in nodes.values():
                op_name = node_data.get(_OP_NAME, "")
                if op_name and op_name != "input":
                    ops_in_graph.add(op_name)

            s0 = r[1]   # stage0_passed
            s05 = r[2]  # stage05_passed
            s1 = r[3]   # stage1_passed
            lr = r[4]   # loss_ratio
            nov = r[5]  # novelty_score
            nov_conf = r[6]  # novelty_confidence

            for op_name in ops_in_graph:
                if op_name not in op_stats:
                    op_stats[op_name] = {
                        "n_used": 0, "n_s0": 0, "n_s05": 0, "n_s1": 0,
                        "lr_sum": 0.0, "lr_n": 0,
                        "nov_sum": 0.0, "nov_n": 0,
                        "nov_conf_sum": 0.0, "nov_conf_n": 0,
                    }
                stats = op_stats[op_name]
                stats["n_used"] += 1
                if s0:
                    stats["n_s0"] += 1
                if s05:
                    stats["n_s05"] += 1
                if s1:
                    stats["n_s1"] += 1
                if lr is not None:
                    stats["lr_sum"] += lr
                    stats["lr_n"] += 1
                if nov is not None:
                    stats["nov_sum"] += nov
                    stats["nov_n"] += 1
                if nov_conf is not None:
                    stats["nov_conf_sum"] += nov_conf
                    stats["nov_conf_n"] += 1

        now = time.time()
        for op_name, stats in op_stats.items():
            avg_lr = stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
            avg_nov = stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
            avg_nov_conf = (stats["nov_conf_sum"] / stats["nov_conf_n"]
                           if stats["nov_conf_n"] else None)
            self.conn.execute(
                """INSERT INTO op_success_rates
                   (op_name, n_used, n_stage0_passed, n_stage05_passed,
                    n_stage1_passed, avg_loss_ratio, avg_novelty,
                    avg_novelty_confidence, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(op_name) DO UPDATE SET
                    n_used = n_used + excluded.n_used,
                    n_stage0_passed = n_stage0_passed + excluded.n_stage0_passed,
                    n_stage05_passed = n_stage05_passed + excluded.n_stage05_passed,
                    n_stage1_passed = n_stage1_passed + excluded.n_stage1_passed,
                    avg_loss_ratio = excluded.avg_loss_ratio,
                    avg_novelty = excluded.avg_novelty,
                    avg_novelty_confidence = excluded.avg_novelty_confidence,
                    last_updated = excluded.last_updated""",
                (op_name, stats["n_used"], stats["n_s0"], stats["n_s05"],
                 stats["n_s1"], avg_lr, avg_nov, avg_nov_conf, now),
            )
        self._maybe_commit()


    def get_op_success_rates(self) -> List[Dict]:
        """Get all op success rates."""
        rows = self.conn.execute(
            """SELECT * FROM op_success_rates
               ORDER BY n_stage1_passed DESC, n_used DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


    def update_failure_signatures(self, experiment_id: str) -> None:
        """Update failure_signatures table from program results in this experiment.

        Extracts op-pair bigrams from each graph and tracks how often
        each bigram appears in failed vs successful programs.  This gives
        Aria a compact memory of which structural patterns to avoid.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1""",
            (experiment_id,),
        ).fetchall()

        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)

        now = time.time()
        for sig, st in sig_stats.items():
            # Keep error_types compact: top 3, comma-separated
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(signature) DO UPDATE SET
                    n_failures = n_failures + excluded.n_failures,
                    n_successes = n_successes + excluded.n_successes,
                    error_types = COALESCE(excluded.error_types, error_types),
                    last_updated = excluded.last_updated""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()


    def backfill_failure_signatures(self) -> int:
        """One-time backfill of failure_signatures from all existing results.

        Skips if the table already has data.  Returns count of signatures created.
        """
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM failure_signatures"
        ).fetchone()[0]
        if existing > 0:
            return 0
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
        ).fetchall()
        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)
        now = time.time()
        for sig, st in sig_stats.items():
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()
        LOGGER.info("Backfilled %d failure signatures from existing results", len(sig_stats))
        return len(sig_stats)


    def recompute_failure_signatures(self) -> int:
        """Delete and rebuild failure_signatures from scratch using S1-only failures.

        Unlike backfill_failure_signatures(), this always runs (even if data exists)
        and only counts programs that passed S0+S0.5 but failed at S1 as failures.
        This cleans up historically contaminated data from S0.5 causality failures.
        """
        self.conn.execute("DELETE FROM failure_signatures")
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
        ).fetchall()
        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)
        now = time.time()
        for sig, st in sig_stats.items():
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()
        LOGGER.info("Recomputed %d failure signatures (S1-only failures)", len(sig_stats))
        return len(sig_stats)


    def get_failure_signature_blocklist(self, min_seen: int = 20,
                                        max_fail_rate: float = 0.95) -> Dict[str, float]:
        """Return op-pair bigrams that consistently fail.

        Returns {signature: penalty} where penalty is 0.0 (hard block) for
        100% failure bigrams and scales up to 1.0.  Only includes bigrams
        seen at least ``min_seen`` times with failure rate >= ``max_fail_rate``.
        """
        rows = self.conn.execute(
            """SELECT signature, n_failures, n_successes
               FROM failure_signatures
               WHERE (n_failures + n_successes) >= ?""",
            (min_seen,),
        ).fetchall()
        blocklist: Dict[str, float] = {}
        for r in rows:
            total = r[1] + r[2]
            fail_rate = r[1] / total if total else 0
            if fail_rate >= max_fail_rate:
                # Scale: 100% fail → 0.0, max_fail_rate → 0.3
                penalty = max(0.0, 0.3 * (1.0 - fail_rate) / (1.0 - max_fail_rate))
                blocklist[r[0]] = round(penalty, 2)
        return blocklist


    # ── Op Rehabilitation Cache ──

    def get_op_rehabilitation_cache(self, max_age_hours: float = 24.0) -> Dict[str, Dict]:
        """Return cached op rehabilitation results, filtered by recency.

        Returns {op_name: {compile_passed, forward_passed, error_message, tested_at, model_dim}}.
        """
        cutoff = time.time() - max_age_hours * 3600
        rows = self.conn.execute(
            """SELECT op_name, compile_passed, forward_passed, error_message, tested_at, model_dim
               FROM op_rehabilitation_cache
               WHERE tested_at >= ?""",
            (cutoff,),
        ).fetchall()
        cache: Dict[str, Dict] = {}
        for r in rows:
            cache[r[0]] = {
                "compile_passed": bool(r[1]),
                "forward_passed": bool(r[2]),
                "error_message": r[3],
                "tested_at": r[4],
                "model_dim": r[5],
            }
        return cache


    # ── Learning Log ──

    def log_learning_event(self, event_type: str, description: str,
                           old_weights: Optional[Dict] = None,
                           new_weights: Optional[Dict] = None,
                           evidence: Optional[str] = None,
                           **event_data: Any) -> None:
        """Log a grammar weight change or learning decision.

        Backward-compatible with callers that pass extra structured keyword
        fields (e.g. ``changes=...``, ``excluded_ops=...``).
        """
        if old_weights is None and "old_weights" in event_data:
            old_weights = event_data.pop("old_weights")
        if new_weights is None and "new_weights" in event_data:
            new_weights = event_data.pop("new_weights")

        if event_data:
            serialized_extra = json.dumps(event_data, sort_keys=True, default=str)
            if evidence:
                evidence = f"{evidence}\n\nmeta={serialized_extra}"
            else:
                evidence = serialized_extra

        self.conn.execute(
            """INSERT INTO learning_log
               (timestamp, event_type, description, old_weights,
                new_weights, evidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (time.time(), event_type, description,
             json.dumps(old_weights) if old_weights else None,
             json.dumps(new_weights) if new_weights else None,
             evidence),
        )
        self._maybe_commit()


    def get_learning_log(self, limit: int = 100) -> List[Dict]:
        """Get recent learning log entries."""
        rows = self.conn.execute(
            "SELECT * FROM learning_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            for f in ("old_weights", "new_weights"):
                if d.get(f):
                    try:
                        d[f] = json.loads(d[f])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results


    def save_effective_weights(self, weights: Dict[str, float],
                               s1_rate: float,
                               experiment_id: Optional[str] = None) -> None:
        """Save the final applied grammar weights and S1 outcome for EMA continuity."""
        self.log_learning_event(
            "effective_weights_snapshot",
            f"Effective weights after {experiment_id or 'unknown'} (S1={s1_rate:.3f})",
            new_weights=weights,
            evidence=json.dumps({"s1_rate": s1_rate, "experiment_id": experiment_id}),
        )


    def load_last_effective_weights(self) -> Optional[tuple]:
        """Load the most recent effective weights snapshot.

        Returns (weights_dict, s1_rate) or None if no snapshot exists.
        """
        row = self.conn.execute(
            "SELECT new_weights, evidence FROM learning_log "
            "WHERE event_type='effective_weights_snapshot' "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return None
        try:
            weights = json.loads(row[0])
            meta = json.loads(row[1]) if row[1] else {}
            return (weights, meta.get("s1_rate", 0.0))
        except (json.JSONDecodeError, TypeError):
            return None


    # ── Workflow Definitions ──

    def save_workflow_definition(
        self,
        workflow_id: str,
        name: str,
        graph_json: str,
        metadata: Optional[Dict] = None,
        author: str = "user",
    ) -> None:
        """Save a visual designer workflow definition."""
        now = time.time()
        self.conn.execute(
            """INSERT INTO workflow_definitions
               (workflow_id, name, timestamp, graph_json, metadata_json, author)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(workflow_id) DO UPDATE SET
                 name = excluded.name,
                 timestamp = excluded.timestamp,
                 graph_json = excluded.graph_json,
                 metadata_json = excluded.metadata_json,
                 author = excluded.author""",
            (workflow_id, name, now, graph_json, json.dumps(metadata or {}), author),
        )
        self._maybe_commit()


    def get_workflow_definition(self, workflow_id: str) -> Optional[Dict]:
        """Get a specific workflow definition."""
        row = self.conn.execute(
            "SELECT * FROM workflow_definitions WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata_json"):
            try:
                d["metadata"] = json.loads(d["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d


    def list_workflow_definitions(self, limit: int = 50) -> List[Dict]:
        """List recent workflow definitions."""
        rows = self.conn.execute(
            """SELECT workflow_id, name, timestamp, author
               FROM workflow_definitions
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


    # ── Designer Run Lineage ──

    def save_designer_run_lineage(
        self,
        run_id: str,
        workflow_id: str,
        *,
        workflow_version: Optional[int] = None,
        graph_fingerprint: Optional[str] = None,
        status: str = "unknown",
        source: str = "aria_designer",
        total_time_ms: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> None:
        """Upsert lineage metadata for runs produced by Aria Designer."""
        now = time.time()
        created_ts = float(created_at) if created_at is not None else now
        self.conn.execute(
            """INSERT INTO designer_run_lineage
               (run_id, workflow_id, workflow_version, graph_fingerprint, status, source,
                total_time_ms, metrics_json, payload_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET
                 workflow_id = excluded.workflow_id,
                 workflow_version = excluded.workflow_version,
                 graph_fingerprint = excluded.graph_fingerprint,
                 status = excluded.status,
                 source = excluded.source,
                 total_time_ms = excluded.total_time_ms,
                 metrics_json = excluded.metrics_json,
                 payload_json = excluded.payload_json,
                 updated_at = excluded.updated_at""",
            (
                run_id,
                workflow_id,
                workflow_version,
                graph_fingerprint,
                status,
                source,
                total_time_ms,
                json.dumps(metrics or {}),
                json.dumps(payload or {}),
                created_ts,
                now,
            ),
        )
        self._maybe_commit()


    def get_designer_run_lineage(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM designer_run_lineage WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["metrics"] = {}
        try:
            d["payload"] = json.loads(d.get("payload_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["payload"] = {}
        return d


    def list_designer_run_lineage(
        self, *, workflow_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM designer_run_lineage"
        params: List[Any] = []
        if workflow_id:
            query += " WHERE workflow_id = ?"
            params.append(workflow_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(max(1, limit)))
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                d["metrics"] = {}
            out.append(d)
        return out


    def get_report_snapshot(
        self,
        snapshot_key: str,
        scope: str,
        min_latest_completed_ts: float,
    ) -> Optional[Dict[str, Any]]:
        if not snapshot_key:
            return None
        row = self.conn.execute(
            """SELECT payload_json, latest_completed_ts
               FROM report_snapshots
               WHERE snapshot_key = ? AND scope = ?""",
            (snapshot_key, scope),
        ).fetchone()
        if not row:
            return None
        cached_latest = float(row["latest_completed_ts"] or 0.0)
        if cached_latest < float(min_latest_completed_ts or 0.0):
            return None
        payload = row["payload_json"]
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


    def save_report_snapshot(
        self,
        snapshot_key: str,
        scope: str,
        query: Dict[str, Any],
        payload: Dict[str, Any],
        latest_completed_ts: float,
    ) -> None:
        if not snapshot_key or not scope:
            return
        now = time.time()
        self.conn.execute(
            """INSERT INTO report_snapshots (
                   snapshot_key, scope, query_json, payload_json,
                   latest_completed_ts, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(snapshot_key) DO UPDATE SET
                   scope = excluded.scope,
                   query_json = excluded.query_json,
                   payload_json = excluded.payload_json,
                   latest_completed_ts = excluded.latest_completed_ts,
                   updated_at = excluded.updated_at""",
            (
                snapshot_key,
                scope,
                json.dumps(query or {}, sort_keys=True, separators=(",", ":")),
                json.dumps(payload or {}, separators=(",", ":")),
                float(latest_completed_ts or 0.0),
                now,
                now,
            ),
        )
        self._maybe_commit()

        cleanup_interval_seconds = 300.0
        last_cleanup = float(self.__class__._last_report_snapshot_cleanup_at or 0.0)
        if (now - last_cleanup) >= cleanup_interval_seconds:
            try:
                ttl_seconds = int(os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)))
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400"))
            except Exception:
                max_rows_per_scope = 400
            self.cleanup_report_snapshots(
                ttl_seconds=max(60, ttl_seconds),
                max_rows_per_scope=max(20, max_rows_per_scope),
            )
            self.__class__._last_report_snapshot_cleanup_at = now


    def cleanup_report_snapshots(
        self,
        ttl_seconds: int = 7 * 24 * 3600,
        max_rows_per_scope: int = 400,
    ) -> Dict[str, int]:
        ttl = max(60, int(ttl_seconds or 0))
        cap = max(1, int(max_rows_per_scope or 0))
        cutoff = time.time() - float(ttl)

        stats = {
            "deleted_expired": 0,
            "deleted_capped": 0,
            "remaining": 0,
        }

        cur = self.conn.execute(
            "DELETE FROM report_snapshots WHERE updated_at < ?",
            (cutoff,),
        )
        stats["deleted_expired"] = int(cur.rowcount or 0)

        scopes = self.conn.execute(
            "SELECT DISTINCT scope FROM report_snapshots"
        ).fetchall()
        for row in scopes:
            scope = row[0]
            if not scope:
                continue
            cur = self.conn.execute(
                """DELETE FROM report_snapshots
                   WHERE snapshot_key IN (
                       SELECT snapshot_key
                       FROM report_snapshots
                       WHERE scope = ?
                       ORDER BY updated_at DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (scope, cap),
            )
            stats["deleted_capped"] += int(cur.rowcount or 0)

        remaining_row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM report_snapshots"
        ).fetchone()
        stats["remaining"] = int(remaining_row["n"] or 0) if remaining_row else 0
        self._maybe_commit()
        return stats


    def get_report_snapshot_stats(self) -> Dict[str, Any]:
        now = time.time()
        rows = self.conn.execute(
            """SELECT scope,
                      COUNT(*) AS count,
                      MIN(updated_at) AS oldest_updated_at,
                      MAX(updated_at) AS newest_updated_at
               FROM report_snapshots
               GROUP BY scope
               ORDER BY count DESC, scope ASC"""
        ).fetchall()

        scopes: List[Dict[str, Any]] = []
        total = 0
        oldest_seen: Optional[float] = None
        newest_seen: Optional[float] = None
        for row in rows:
            count = int(row["count"] or 0)
            oldest = float(row["oldest_updated_at"] or 0.0)
            newest = float(row["newest_updated_at"] or 0.0)
            total += count
            if oldest > 0 and (oldest_seen is None or oldest < oldest_seen):
                oldest_seen = oldest
            if newest > 0 and (newest_seen is None or newest > newest_seen):
                newest_seen = newest

            scopes.append({
                "scope": row["scope"],
                "count": count,
                "oldest_age_seconds": round(max(0.0, now - oldest), 2) if oldest > 0 else None,
                "newest_age_seconds": round(max(0.0, now - newest), 2) if newest > 0 else None,
            })

        return {
            "total_snapshots": total,
            "n_scopes": len(scopes),
            "oldest_age_seconds": round(max(0.0, now - oldest_seen), 2) if oldest_seen else None,
            "newest_age_seconds": round(max(0.0, now - newest_seen), 2) if newest_seen else None,
            "scopes": scopes,
        }


    # ── Attribution Reports ──

    def record_attribution_report(
        self,
        hypothesis_id: Optional[str],
        supporting_experiments: Optional[List[str]],
        ablation_experiments: Optional[List[str]],
        outcome: str,
        report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an attribution report row linking evidence and ablations."""
        report_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO attribution_reports
            (report_id, timestamp, hypothesis_id, supporting_experiments,
             ablation_experiments, outcome, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                report_id,
                now,
                hypothesis_id,
                json.dumps(supporting_experiments or []),
                json.dumps(ablation_experiments or []),
                outcome,
                json.dumps(report or {}),
            ),
        )
        self._maybe_commit()
        return report_id


    def get_attribution_reports(self, hypothesis_id: Optional[str] = None,
                                limit: int = 100) -> List[Dict]:
        """Return attribution reports, newest first."""
        query = "SELECT * FROM attribution_reports WHERE 1=1"
        params: List[Any] = []
        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict] = []
        for row in rows:
            item = dict(row)
            for key in ("supporting_experiments", "ablation_experiments", "report_json"):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out


    # ── Report Markdown Export ──

    def save_report_markdown(self, content: str, reason: str,
                             summary: Optional[Dict] = None) -> Optional[Path]:
        """Save a report as a markdown file alongside the database.

        Creates a reports/ directory next to lab_notebook.db and writes
        the report content as a .md file with a frontmatter-style header.

        Returns the path to the created file, or None on failure.
        """
        logger = logging.getLogger(__name__)
        try:
            reports_dir = self.db_path.parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
            safe_reason = reason.replace(" ", "_").replace("/", "-")[:40]
            filename = f"report_{timestamp_str}_{safe_reason}.md"
            filepath = reports_dir / filename

            # Build frontmatter header
            header_lines = [
                "---",
                f"generated: {now.isoformat()}",
                f"reason: {reason}",
            ]
            if summary:
                header_lines.append(
                    f"experiments: {summary.get('total_experiments', '?')}")
                total_prog = summary.get("total_programs_evaluated", 0)
                s1 = summary.get("stage1_survivors", 0)
                rate = s1 / max(total_prog, 1) * 100
                header_lines.append(f"s1_pass_rate: {rate:.1f}%")
                header_lines.append(f"stage1_survivors: {s1}")
            header_lines.append("---")
            header_lines.append("")

            full_content = "\n".join(header_lines) + content

            filepath.write_text(full_content, encoding="utf-8")
            logger.info(f"Report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"Failed to save report markdown: {e}")
            return None

