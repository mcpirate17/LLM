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



class _NotebookCore:
    """Core operations for the Lab Notebook."""
    """Electronic lab notebook for the AI scientist."""

    _cached_code_version: Optional[str] = None
    _last_report_snapshot_cleanup_at: float = 0.0

    # GPT-2 reference metrics (measured on our d_model=256, 6-layer config)
    _GPT2_REF = {
        "loss_ratio": 0.2646,
        "param_count": 9_767_424,
        "flops_forward": 19_534_848,
        "throughput_tok_s": 1_200_845,
        "peak_memory_mb": 115.0,
        "forward_time_ms": 0.43,
    }

    @staticmethod
    def resolve_db_path(db_path: str | Path) -> Path:
        """Resolve a database path to its absolute path, handling nested research/ cases.

        Ensures that if we are currently inside the research/ directory,
        a path like 'research/lab_notebook.db' refers to the one in the parent.
        """
        path = Path(db_path)
        if not path.is_absolute():
            # If we are in /some/path/LLM/research and db_path is 'research/lab_notebook.db'
            # then path.resolve() would be /some/path/LLM/research/research/lab_notebook.db.
            # We want /some/path/LLM/research/lab_notebook.db.
            cwd = Path.cwd()
            if cwd.name == "research" and path.parts and path.parts[0] == "research":
                # db_path starts with 'research/' and we are already in research/
                # assume the user meant the parent's research/ directory
                return (cwd.parent / db_path).absolute()
        return path.resolve()


    def __init__(self, db_path: str | Path = "research/lab_notebook.db"):
        self.db_path = self.resolve_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        # Enable WAL mode for high-concurrency performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._batch_depth = 0
        self._program_results_columns: Optional[set[str]] = None
        self._leaderboard_columns: Optional[set[str]] = None
        self.conn.executescript(NOTEBOOK_SCHEMA)
        self._maybe_commit()
        self._migrate()

        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()


    def _writer_loop(self):
        """Background thread that handles all database writes."""
        # Use a separate connection for the writer thread
        writer_conn = sqlite3.connect(str(self.db_path))
        writer_conn.execute("PRAGMA journal_mode=WAL")
        writer_conn.execute("PRAGMA synchronous=NORMAL")
        
        batch = []
        last_commit = time.time()
        
        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                item = self._write_queue.get(timeout=0.1)
                if item is None: # Sentinel
                    break
                
                sql, params = item
                if sql == "__flush__":
                    # Flush request: commit pending batch and signal caller
                    if batch:
                        writer_conn.commit()
                        batch = []
                        last_commit = time.time()
                    params.set()  # params is a threading.Event
                    continue
                if isinstance(params, list) and params and isinstance(params[0], (list, tuple)):
                    writer_conn.executemany(sql, params)
                else:
                    writer_conn.execute(sql, params)
                batch.append(item)

                if len(batch) >= 50 or (time.time() - last_commit > 1.0 and batch):
                    writer_conn.commit()
                    batch = []
                    last_commit = time.time()
                    
            except queue.Empty:
                if batch:
                    writer_conn.commit()
                    batch = []
                    last_commit = time.time()
                continue
            except Exception as e:
                LOGGER.error(f"LabNotebook async writer error: {e}")
        
        if batch:
            writer_conn.commit()
        writer_conn.close()


    def _submit_write(self, sql: str, params: Any):
        """Submit a write task to the background queue."""
        self._write_queue.put((sql, params))


    def flush_writes(self, timeout: float = 5.0):
        """Block until the async write queue is drained and committed.

        Useful in tests and any code that writes via ``_submit_write`` then
        immediately reads back via the main ``self.conn``.
        """
        # Put a sentinel-like marker and wait for drain
        flush_event = threading.Event()
        self._write_queue.put(("__flush__", flush_event))
        flush_event.wait(timeout=timeout)


    def _migrate(self):
        """Add any missing columns to existing databases."""
        # Migrate experiments table
        try:
            self.conn.execute("SELECT llm_analysis FROM experiments LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN llm_analysis TEXT")
            self._maybe_commit()

        # Migrate program_results: add new columns if missing
        existing = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(program_results)").fetchall()
        }
        for col_name, col_type in _PROGRAM_RESULTS_NEW_COLUMNS.items():
            if col_name not in existing:
                try:
                    self.conn.execute(
                        f"ALTER TABLE program_results ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    # Column may already exist in older DBs with partial migrations.
                    pass

        # Migrate program_results: add arch_spec_json if missing
        if "arch_spec_json" not in existing:
            self.conn.execute(
                "ALTER TABLE program_results ADD COLUMN arch_spec_json TEXT"
            )
        if "model_source" not in existing:
            self.conn.execute(
                "ALTER TABLE program_results ADD COLUMN model_source TEXT"
            )

        # Ensure leaderboard table exists (created in schema but needed for old DBs)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                entry_id TEXT PRIMARY KEY,
                result_id TEXT REFERENCES program_results(result_id),
                timestamp REAL NOT NULL,
                model_source TEXT NOT NULL,
                architecture_desc TEXT,
                screening_loss_ratio REAL,
                screening_novelty REAL,
                screening_passed INTEGER DEFAULT 0,
                investigation_loss_ratio REAL,
                investigation_robustness REAL,
                investigation_best_training TEXT,
                investigation_passed INTEGER DEFAULT 0,
                validation_loss_ratio REAL,
                validation_baseline_ratio REAL,
                validation_multi_seed_std REAL,
                validation_passed INTEGER DEFAULT 0,
                composite_score REAL,
                tier TEXT DEFAULT 'screening',
                tags TEXT,
                notes TEXT,
                is_reference INTEGER DEFAULT 0,
                reference_name TEXT DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_leaderboard_tier ON leaderboard(tier);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(composite_score);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_result ON leaderboard(result_id);
        """)
        # Migrate decisions: add evidence_pack_json if missing
        try:
            decision_cols = {
                row[1] for row in
                self.conn.execute("PRAGMA table_info(decisions)").fetchall()
            }
        except sqlite3.OperationalError:
            decision_cols = set()
        if "evidence_pack_json" not in decision_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE decisions ADD COLUMN evidence_pack_json TEXT"
                )
            except sqlite3.OperationalError:
                pass
        # Migrate op_success_rates: add avg_novelty_confidence if missing
        osr_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(op_success_rates)").fetchall()
        }
        if "avg_novelty_confidence" not in osr_cols:
            self.conn.execute(
                "ALTER TABLE op_success_rates ADD COLUMN avg_novelty_confidence REAL"
            )

        # Migrate experiments: add campaign_id if missing
        exp_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "campaign_id" not in exp_cols:
            self.conn.execute(
                "ALTER TABLE experiments ADD COLUMN campaign_id TEXT"
            )
        if "preregistration_id" not in exp_cols:
            self.conn.execute(
                "ALTER TABLE experiments ADD COLUMN preregistration_id TEXT"
            )

        # Migrate hypotheses: add metadata_json if missing
        hyp_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(hypotheses)").fetchall()
        }
        if "metadata_json" not in hyp_cols:
            self.conn.execute(
                "ALTER TABLE hypotheses ADD COLUMN metadata_json TEXT"
            )

        # Migrate campaigns: add completion_reason and successor_campaign_id
        camp_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(campaigns)").fetchall()
        }
        if "completion_reason" not in camp_cols:
            self.conn.execute(
                "ALTER TABLE campaigns ADD COLUMN completion_reason TEXT"
            )
        if "successor_campaign_id" not in camp_cols:
            self.conn.execute(
                "ALTER TABLE campaigns ADD COLUMN successor_campaign_id TEXT"
            )

        # Migrate insights: add semantic identity columns and collapse duplicates.
        insight_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(insights)").fetchall()
        }
        if "insight_type" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN insight_type TEXT")
        if "subject_key" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN subject_key TEXT")
        if "semantic_key" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN semantic_key TEXT")

        rows = self.conn.execute(
            """SELECT insight_id, category, content, insight_type, subject_key, semantic_key
               FROM insights"""
        ).fetchall()
        for row in rows:
            existing_type = str(row["insight_type"] or "").strip() if isinstance(row, sqlite3.Row) else str(row[3] or "").strip()
            existing_subject = str(row["subject_key"] or "").strip() if isinstance(row, sqlite3.Row) else str(row[4] or "").strip()
            existing_semantic = str(row["semantic_key"] or "").strip() if isinstance(row, sqlite3.Row) else str(row[5] or "").strip()
            if existing_type and existing_subject and existing_semantic:
                continue
            category = row["category"] if isinstance(row, sqlite3.Row) else row[1]
            content = row["content"] if isinstance(row, sqlite3.Row) else row[2]
            inferred_type, inferred_subject, inferred_semantic = infer_insight_identity(
                str(category or ""),
                str(content or ""),
            )
            self.conn.execute(
                """UPDATE insights
                   SET insight_type = COALESCE(NULLIF(insight_type, ''), ?),
                       subject_key = COALESCE(NULLIF(subject_key, ''), ?),
                       semantic_key = COALESCE(NULLIF(semantic_key, ''), ?)
                   WHERE insight_id = ?""",
                (inferred_type, inferred_subject, inferred_semantic, row["insight_id"] if isinstance(row, sqlite3.Row) else row[0]),
            )

        def _supersede_active_semantic_duplicates() -> None:
            active_rows = self.conn.execute(
                """SELECT insight_id, semantic_key
                   FROM insights
                   WHERE status = 'active'
                     AND semantic_key IS NOT NULL
                     AND semantic_key != ''
                   ORDER BY confidence DESC, timestamp DESC"""
            ).fetchall()
            seen_semantic: set[str] = set()
            for row in active_rows:
                sem = str(row["semantic_key"] if isinstance(row, sqlite3.Row) else row[1])
                insight_id = row["insight_id"] if isinstance(row, sqlite3.Row) else row[0]
                if sem in seen_semantic:
                    self.conn.execute(
                        "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
                        (insight_id,),
                    )
                    continue
                seen_semantic.add(sem)

        _supersede_active_semantic_duplicates()

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_semantic_key ON insights(semantic_key)"
        )
        try:
            self.conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_insights_active_semantic_unique
                   ON insights(semantic_key)
                   WHERE status = 'active' AND semantic_key IS NOT NULL AND semantic_key != ''"""
            )
        except sqlite3.IntegrityError:
            _supersede_active_semantic_duplicates()
            self.conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_insights_active_semantic_unique
                   ON insights(semantic_key)
                   WHERE status = 'active' AND semantic_key IS NOT NULL AND semantic_key != ''"""
            )

        # Migrate leaderboard: add efficiency and robustness columns
        lb_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
        }
        for col in (
            "normalized_baseline_ratio REAL",
            "param_efficiency REAL",
            "quant_int8_retention REAL",
            "quant_quality_per_byte REAL",
            "robustness_long_ctx_score REAL",
            "robustness_noise_score REAL",
            "init_sensitivity_std REAL",
            "fp_jacobian_spectral_norm REAL",
            "scaling_param_efficiency REAL",
            "scaling_flop_efficiency REAL",
            "scaling_gate_passed INTEGER",
            "scaling_best_family TEXT",
            "scaling_d512_param_efficiency REAL",
            "scaling_confidence TEXT",
            "campaign_id TEXT",
            "is_pinned INTEGER DEFAULT 0",
            "routing_savings_ratio REAL",
            "compression_ratio REAL",
            "activation_sparsity_score REAL",
            "dead_neuron_ratio REAL",
            "routing_collapse_score REAL",
            "wikitext_perplexity REAL",
            "wikitext_score REAL",
            "tinystories_perplexity REAL",
            "tinystories_score REAL",
            "cross_task_score REAL",
            "efficiency_wall_score REAL",
            "max_viable_seq_len INTEGER",
            "scaling_regime TEXT",
            "discovery_loss_ratio REAL",
            "pre_inv_score REAL",
            "ncd_score REAL",
            "robustness_long_ctx_scaling_score REAL",
            "robustness_long_ctx_assoc_score REAL",
            "robustness_long_ctx_multi_hop_score REAL",
            "robustness_long_ctx_passkey_score REAL",
            "robustness_long_ctx_retrieval_aggregate REAL",
            "robustness_long_ctx_combined_score REAL",
            "depth_savings_ratio REAL",
            "recursion_savings_ratio REAL",
            "activation_sparsity_score REAL",
            "routing_expert_count INTEGER",
            "routing_confidence_mean REAL",
            "routing_drop_rate REAL",
            "efficiency_multiple REAL",
        ):
            col_name = col.split()[0]
            if col_name not in lb_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE leaderboard ADD COLUMN {col}"
                    )
                except sqlite3.OperationalError:
                    pass

        # Migrate leaderboard: add reference/pin columns
        if "is_reference" not in lb_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE leaderboard ADD COLUMN is_reference INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
        if "reference_name" not in lb_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE leaderboard ADD COLUMN reference_name TEXT DEFAULT NULL"
                )
            except sqlite3.OperationalError:
                pass

        self._program_results_columns = None
        self._leaderboard_columns = None
        self._maybe_commit()


    def _get_program_results_columns(self) -> set[str]:
        """Return current program_results columns for defensive inserts."""
        if self._program_results_columns is None:
            rows = self.conn.execute("PRAGMA table_info(program_results)").fetchall()
            self._program_results_columns = {str(row[1]) for row in rows}
        return self._program_results_columns


    def _get_leaderboard_columns(self) -> set[str]:
        """Return current leaderboard columns for defensive updates."""
        if self._leaderboard_columns is None:
            rows = self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
            self._leaderboard_columns = {str(row[1]) for row in rows}
        return self._leaderboard_columns


    @classmethod
    def _detect_code_version(cls) -> str:
        """Detect code version for experiment traceability."""
        if cls._cached_code_version:
            return cls._cached_code_version

        env_version = os.environ.get("RESEARCH_CODE_VERSION")
        if env_version:
            cls._cached_code_version = env_version
            return cls._cached_code_version

        repo_root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                text=True,
            ).strip()
            if commit:
                cls._cached_code_version = commit
                return cls._cached_code_version
        except Exception:
            pass

        cls._cached_code_version = "unknown"
        return cls._cached_code_version


    # ── Knowledge Digests ──

    def store_digest(self, digest_dict: Dict) -> str:
        """Store a knowledge digest and return its ID."""
        digest_id = str(uuid.uuid4())
        ts = digest_dict.get("timestamp", time.time())
        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge_digests
               (digest_id, timestamp, cycle_number, digest_json,
                narrative_summary, n_experiments_analyzed, n_curves_analyzed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                digest_id,
                ts,
                digest_dict.get("cycle_number"),
                json.dumps(digest_dict),
                digest_dict.get("narrative", "")[:2000],
                digest_dict.get("n_experiments_analyzed"),
                digest_dict.get("n_curves_analyzed"),
            ),
        )
        self._maybe_commit()
        return digest_id


    def get_latest_digest(self) -> Optional[Dict]:
        """Return the most recent knowledge digest, or None."""
        try:
            row = self.conn.execute(
                "SELECT digest_json FROM knowledge_digests ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            LOGGER.debug("Failed to load latest digest: %s", e)
        return None


    def close(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_write_queue"):
            self._write_queue.put(None) # Sentinel
        if hasattr(self, "_writer_thread") and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        self.conn.close()


    def _compress(self, data: Any) -> bytes:
        """JSON-encode and zlib-compress data."""
        return zlib.compress(json.dumps(data).encode("utf-8"))


    def _decompress(self, blob: Any) -> Any:
        """Decompress zlib blob and JSON-decode with fallback for raw strings."""
        if not blob:
            return None
        if not isinstance(blob, bytes):
            # Already a string (old data)
            try:
                return json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                return blob
        try:
            return json.loads(zlib.decompress(blob).decode("utf-8"))
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
            # Fallback for old uncompressed bytes data if any
            return json.loads(blob.decode("utf-8"))


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()


    @contextmanager
    def batch(self):
        """Context manager to batch multiple writes into a single commit."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._maybe_commit()


    def _maybe_commit(self):
        """Commit unless inside a batch() context."""
        if self._batch_depth == 0:
            self.conn.commit()


    def _sanitize_numeric(self, value: Any) -> Any:
        """Deep sanitize values for SQLite: convert NumPy/Torch scalars to Python types.
        
        Prevents binary blob corruption when NumPy float32/int64 values are 
        inserted into REAL/INTEGER columns.
        """
        if value is None:
            return None
        
        # Handle NumPy/Torch scalars
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                # Returns a standard Python float or int
                return value.item()
            except Exception:
                pass
        
        # Handle explicit NumPy types
        if hasattr(value, 'dtype'):
            try:
                return float(value)
            except Exception:
                pass

        if isinstance(value, dict):
            return {k: self._sanitize_numeric(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_numeric(v) for v in value]
        
        # Final pass for floating point safety
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
            
        return value

