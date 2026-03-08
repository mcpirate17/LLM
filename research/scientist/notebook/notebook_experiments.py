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



class _ExperimentsMixin:
    """Experiments operations for the Lab Notebook."""

    def cleanup_stale_experiments(
        self,
        timeout_minutes: int = 60,
        startup_failure_minutes: int = 15,
    ) -> int:
        """Mark stale or startup-failed running experiments as failed.

        - Long-running stale experiments are cleaned after ``timeout_minutes``.
        - Runs with no progress signals are cleaned after
          ``startup_failure_minutes`` to handle interrupted startup paths.

        Returns the number of experiments cleaned up.
        """
        now = time.time()
        cutoff = now - (timeout_minutes * 60)
        startup_cutoff = now - (startup_failure_minutes * 60)

        stale_rows = self.conn.execute(
            "SELECT experiment_id FROM experiments "
            "WHERE status = 'running' AND started_at < ?",
            (cutoff,),
        ).fetchall()
        stale_ids = {r["experiment_id"] for r in stale_rows}

        startup_failed_rows = self.conn.execute(
            """
            SELECT e.experiment_id
            FROM experiments e
            WHERE e.status = 'running'
              AND e.started_at < ?
              AND NOT EXISTS (
                SELECT 1 FROM program_results pr WHERE pr.experiment_id = e.experiment_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM metrics_log ml WHERE ml.experiment_id = e.experiment_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM entries en
                WHERE en.experiment_id = e.experiment_id
                  AND en.entry_type != 'hypothesis'
              )
            """,
            (startup_cutoff,),
        ).fetchall()
        startup_failed_ids = {r["experiment_id"] for r in startup_failed_rows}

        if not stale_ids and not startup_failed_ids:
            return 0

        updates = []
        all_ids = stale_ids | startup_failed_ids
        for experiment_id in all_ids:
            if experiment_id in startup_failed_ids and experiment_id not in stale_ids:
                reason = "Startup failed before any progress was recorded"
            else:
                reason = "Process terminated while running"
            updates.append((reason, experiment_id))

        self.conn.executemany(
            "UPDATE experiments SET status = 'failed', "
            "results_json = json_set(COALESCE(results_json, '{}'), '$.failure_reason', ?) "
            "WHERE experiment_id = ?",
            updates,
        )
        self._maybe_commit()
        return len(all_ids)


    def get_resumable_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment data for resume if status is 'running' or 'failed'.

        Returns dict with config_json, experiment_type, hypothesis, started_at,
        or None if the experiment doesn't exist or isn't resumable.
        """
        row = self.conn.execute(
            "SELECT experiment_id, experiment_type, status, config_json, "
            "hypothesis, started_at FROM experiments "
            "WHERE experiment_id = ? AND status IN ('running', 'failed')",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "experiment_id": row["experiment_id"],
            "experiment_type": row["experiment_type"],
            "status": row["status"],
            "config_json": row["config_json"],
            "hypothesis": row["hypothesis"],
            "started_at": row["started_at"],
        }


    # ── Hypothesis Preregistration ──

    def create_preregistration(
        self,
        experiment_type: str,
        preregistration: Dict[str, Any],
        created_by: str = "runner",
        notes: Optional[str] = None,
    ) -> str:
        """Create a structured preregistration entry."""
        validate_preregistration(preregistration)
        prereg_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO hypothesis_preregistrations
            (preregistration_id, timestamp, experiment_type, status,
             hypothesis_json, analysis_plan_json, falsification_json,
             confounders_json, exploratory, created_by, notes)
            VALUES (?, ?, ?, 'registered', ?, ?, ?, ?, ?, ?, ?)""",
            (
                prereg_id,
                now,
                experiment_type,
                json.dumps(preregistration.get("hypothesis") or {}),
                json.dumps(preregistration.get("analysis_plan") or {}),
                json.dumps(preregistration.get("falsification_conditions") or []),
                json.dumps(preregistration.get("confounders_checklist") or []),
                int(bool(preregistration.get("exploratory"))),
                created_by,
                notes,
            ),
        )
        self._maybe_commit()
        return prereg_id


    def get_preregistration(self, preregistration_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM hypothesis_preregistrations WHERE preregistration_id = ?",
            (preregistration_id,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        for field in ("hypothesis_json", "analysis_plan_json", "falsification_json", "confounders_json"):
            raw = out.get(field)
            if raw:
                try:
                    out[field] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out


    def get_preregistration_for_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT preregistration_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if not row or not row["preregistration_id"]:
            return None
        return self.get_preregistration(row["preregistration_id"])


    def get_preregistration_deviations(self, experiment_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM preregistration_deviations
               WHERE experiment_id = ?
               ORDER BY timestamp DESC""",
            (experiment_id,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            if d.get("details_json"):
                try:
                    d["details_json"] = json.loads(d["details_json"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(d)
        return out


    def log_preregistration_deviation(
        self,
        experiment_id: str,
        rationale: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record explicit exploratory deviation from preregistered plan."""
        exp = self.conn.execute(
            "SELECT preregistration_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        prereg_id = exp["preregistration_id"] if exp else None
        dev_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO preregistration_deviations
            (deviation_id, preregistration_id, experiment_id, timestamp,
             deviation_type, rationale, details_json)
            VALUES (?, ?, ?, ?, 'exploratory', ?, ?)""",
            (
                dev_id,
                prereg_id,
                experiment_id,
                time.time(),
                rationale,
                json.dumps(details or {}),
            ),
        )
        self._maybe_commit()
        return dev_id


    # ── Experiments ──

    def start_experiment(
        self,
        experiment_type: str,
        config: Dict,
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict] = None,
        preregistration_id: Optional[str] = None,
        require_preregistration: bool = False,
    ) -> str:
        """Start a new experiment. Returns experiment ID."""
        if require_preregistration and not preregistration_id:
            raise PreregistrationError("Experiment start blocked: missing preregistration_id.")
        exp_id = str(uuid.uuid4())[:12]
        now = time.time()
        config_payload = dict(config)
        config_payload.setdefault("code_version", self._detect_code_version())

        self.conn.execute(
            """INSERT INTO experiments
            (experiment_id, timestamp, experiment_type, status, hypothesis,
             research_question, preregistration_id, config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
            (exp_id, now, experiment_type, hypothesis, research_question, preregistration_id,
             json.dumps(config_payload), now),
        )
        if preregistration_id:
            self.conn.execute(
                """UPDATE hypothesis_preregistrations
                   SET experiment_id = ?, status = 'linked'
                   WHERE preregistration_id = ?""",
                (exp_id, preregistration_id),
            )
        self._maybe_commit()

        # Log entry
        source = (hypothesis_metadata or {}).get("source", "unknown")
        confidence = (hypothesis_metadata or {}).get("confidence")
        critique_confidence = (hypothesis_metadata or {}).get("critique_confidence")
        critique = (hypothesis_metadata or {}).get("critique")
        effective_confidence = confidence if confidence is not None else critique_confidence
        confidence_text = effective_confidence if effective_confidence is not None else "not provided"
        if isinstance(critique, dict):
            verdict = critique.get("verdict") or "unknown"
            gate = critique.get("gate") or "n/a"
            concerns = critique.get("concerns") or []
            concern_hint = concerns[0] if concerns else "no concerns recorded"
            critique_text = f"{verdict} (gate={gate}) — {concern_hint}"
        else:
            critique_text = critique if critique else "not provided"
        self.add_entry(ExperimentEntry(
            entry_type="hypothesis",
            title=f"Experiment {exp_id} started",
            content=(
                f"Type: {experiment_type}\n"
                f"Hypothesis: {hypothesis or 'exploratory'}\n"
                f"Provenance: {source}\n"
                f"Confidence: {confidence_text}\n"
                f"Critique: {critique_text}"
            ),
            experiment_id=exp_id,
            tags=["experiment_start"],
            metadata=hypothesis_metadata or {},
        ))

        return exp_id


    def complete_experiment(
        self,
        experiment_id: str,
        results: Dict,
        aria_summary: str = "",
        aria_mood: str = "contemplative",
        insights: Optional[List[str]] = None,
        llm_analysis: Optional[str] = None,
        exploratory_deviation_reason: Optional[str] = None,
    ):
        """Mark an experiment as completed with results."""
        n_total = results.get("total", 0)
        if n_total == 0:
            return self.fail_experiment(
                experiment_id, 
                error="Experiment completed with 0 programs generated (possible synthesis failure).",
                results=results
            )

        now = time.time()
        started = self.conn.execute(
            "SELECT started_at FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        ).fetchone()
        duration = now - started["started_at"] if started else 0

        self.conn.execute(
            """UPDATE experiments SET
                status = 'completed',
                results_json = ?,
                n_programs_generated = ?,
                n_stage0_passed = ?,
                n_stage05_passed = ?,
                n_stage1_passed = ?,
                best_loss_ratio = ?,
                best_novelty_score = ?,
                aria_summary = ?,
                aria_mood = ?,
                insights_json = ?,
                llm_analysis = ?,
                completed_at = ?,
                duration_seconds = ?
            WHERE experiment_id = ?""",
            (self._compress(results),
             results.get("total", 0),
             results.get("stage0_passed", 0),
             results.get("stage05_passed", 0),
             results.get("stage1_passed", 0),
             results.get("best_loss_ratio"),
             results.get("best_novelty_score"),
             aria_summary, aria_mood,
             self._compress(insights or []),
             llm_analysis,
             now, duration,
             experiment_id),
        )
        self._maybe_commit()

        prereg = self.get_preregistration_for_experiment(experiment_id)
        is_exploratory = bool(exploratory_deviation_reason)
        if is_exploratory:
            self.log_preregistration_deviation(
                experiment_id,
                rationale=exploratory_deviation_reason or "Post-hoc exploratory deviation.",
                details={"source": "complete_experiment"},
            )
        self.add_entry(ExperimentEntry(
            entry_type="analysis",
            title="Post-hoc Analysis Link",
            content=(
                "Analysis linked to preregistration."
                if prereg
                else "Analysis has no preregistration link and is exploratory."
            ),
            experiment_id=experiment_id,
            tags=["analysis_traceability"],
            metadata={
                "preregistration_id": prereg.get("preregistration_id") if prereg else None,
                "analysis_mode": "exploratory" if is_exploratory or not prereg else "confirmatory",
                "deviation_reason": exploratory_deviation_reason,
            },
        ))


    def fail_experiment(self, experiment_id: str, error: str, results: Optional[Dict] = None):
        """Mark an experiment as failed. Deletes record if it contains no useful information."""
        results_blob = self._compress(results) if results else None
        n_prog = results.get("total", 0) if results else 0
        
        # First update so we have the state
        self.conn.execute(
            """UPDATE experiments SET 
               status = 'failed', 
               completed_at = ?,
               aria_summary = ?,
               results_json = ?,
               n_programs_generated = ?
               WHERE experiment_id = ?""",
            (time.time(), f"FAILED: {error}", results_blob, n_prog, experiment_id),
        )
        self._maybe_commit()

        # Delete if it's total junk (no programs AND no LLM insights)
        row = self.conn.execute(
            "SELECT llm_analysis FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        ).fetchone()
        
        if n_prog == 0 and (not row or not row["llm_analysis"]):
            self.conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
            self.conn.execute("DELETE FROM entries WHERE experiment_id = ?", (experiment_id,))
            self._maybe_commit()
            LOGGER.info("Deleted zero-value failed experiment %s", experiment_id)


    def purge_empty_experiments(self) -> int:
        """Delete failed experiments that produced no program_results.

        Call periodically (e.g. between experiment cycles) to prevent
        empty experiments from accumulating.  Returns count deleted.
        """
        rows = self.conn.execute("""
            SELECT experiment_id FROM experiments
            WHERE status = 'failed'
            AND NOT EXISTS (
                SELECT 1 FROM program_results p
                WHERE p.experiment_id = experiments.experiment_id
            )
        """).fetchall()
        if not rows:
            return 0
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM experiments WHERE experiment_id IN ({placeholders})", ids
        )
        for table in ("entries", "insights", "hypotheses"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE experiment_id IN ({placeholders})", ids
            )
        self._maybe_commit()
        LOGGER.debug("Purged %d empty failed experiments", len(ids))
        return len(ids)


    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment by marking it as failed.

        Returns True if the experiment was cancelled, False if not found or
        not in a cancellable state.
        """
        row = self.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if not row or row["status"] != "running":
            return False
        self.conn.execute(
            """UPDATE experiments SET status = 'failed', completed_at = ?,
               aria_summary = 'Cancelled by user'
               WHERE experiment_id = ?""",
            (time.time(), experiment_id),
        )
        self._maybe_commit()
        return True


    # ── Queries ──

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("results_json"):
            d["results"] = self._decompress(d["results_json"])
        if d.get("insights_json"):
            d["insights"] = self._decompress(d["insights_json"])
        return d


    def backfill_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Backfill missing summary metrics on an existing experiment row.

        Uses already-recorded program_results/results_json only (no rerun).
        """
        exp = self.conn.execute(
            "SELECT experiment_id, best_loss_ratio, best_novelty_score, results_json "
            "FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if exp is None:
            return {"found": False, "updated_fields": [], "updated": False}

        agg = self.conn.execute(
            """SELECT
                    MIN(loss_ratio) AS min_loss_ratio,
                    MAX(novelty_score) AS max_novelty_score,
                    AVG(throughput_tok_s) AS avg_throughput_tok_s,
                    COUNT(*) AS n_results
               FROM program_results
               WHERE experiment_id = ?""",
            (experiment_id,),
        ).fetchone()

        min_loss = agg["min_loss_ratio"] if agg else None
        max_novelty = agg["max_novelty_score"] if agg else None
        avg_tp = agg["avg_throughput_tok_s"] if agg else None
        n_results = int(agg["n_results"] or 0) if agg else 0

        perf_tp = None
        raw_results = exp["results_json"]
        if isinstance(raw_results, str) and raw_results:
            try:
                parsed = self._decompress(raw_results)
                perf = parsed.get("perf_report") if isinstance(parsed, dict) else None
                if isinstance(perf, dict):
                    perf_tp = perf.get("avg_throughput_tok_s")
            except Exception:
                perf_tp = None

        updates: List[str] = []
        params: List[Any] = []
        updated_fields: List[str] = []

        if exp["best_loss_ratio"] is None and min_loss is not None:
            updates.append("best_loss_ratio = ?")
            params.append(float(min_loss))
            updated_fields.append("best_loss_ratio")

        if exp["best_novelty_score"] is None and max_novelty is not None:
            updates.append("best_novelty_score = ?")
            params.append(float(max_novelty))
            updated_fields.append("best_novelty_score")

        if updates:
            params.append(experiment_id)
            self.conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?",
                tuple(params),
            )
            self._maybe_commit()

        throughput_available = (
            (avg_tp is not None and float(avg_tp) > 0)
            or (perf_tp is not None and float(perf_tp) > 0)
        )

        return {
            "found": True,
            "updated": bool(updated_fields),
            "updated_fields": updated_fields,
            "n_program_results": n_results,
            "throughput_available": bool(throughput_available),
        }


    def get_recent_experiments(self, n: int = 20, offset: int = 0) -> List[Dict]:
        n = max(1, int(n))
        offset = max(0, int(offset))
        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, status,
                      hypothesis, research_question,
                      n_programs_generated, n_stage0_passed, n_stage05_passed,
                      n_stage1_passed,
                      best_loss_ratio, best_novelty_score, aria_mood,
                      aria_summary, duration_seconds
               FROM experiments ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            (n, offset)
        ).fetchall()
        return [dict(r) for r in rows]


    def get_latest_completed_experiment_timestamp(self) -> float:
        row = self.conn.execute(
            "SELECT MAX(timestamp) AS latest_ts FROM experiments WHERE status = 'completed'"
        ).fetchone()
        if not row:
            return 0.0
        try:
            return float(row["latest_ts"] or 0.0)
        except Exception:
            return 0.0


    def get_experiment_trends(self, limit: int = 50) -> List[Dict]:
        """Get cross-experiment trend data for charts."""

        def _mode_factor(mode: Optional[str]) -> float:
            normalized = str(mode or "").strip().lower()
            if normalized in {"investigation", "validation", "single"}:
                return 0.55
            if normalized in {"continuous", "evolution", "synthesis", "morphological", "training"}:
                return 1.0
            return 0.8

        def _resolve_mode(row: Dict) -> str:
            config_mode = None
            raw_config = row.get("config_json")
            if isinstance(raw_config, str) and raw_config.strip():
                try:
                    parsed = json.loads(raw_config)
                    if isinstance(parsed, dict):
                        config_mode = (
                            parsed.get("mode")
                            or parsed.get("run_mode")
                            or parsed.get("experiment_mode")
                        )
                except (json.JSONDecodeError, TypeError):
                    config_mode = None
            return str(config_mode or row.get("experiment_type") or "unknown")

        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, config_json, results_json,
                      n_programs_generated, n_stage0_passed, n_stage05_passed, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, duration_seconds
               FROM experiments
               WHERE status = 'completed'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        exp_ids = [row["experiment_id"] for row in rows if row["experiment_id"]]
        program_metrics_by_exp: Dict[str, Dict[str, Any]] = {}
        if exp_ids:
            available_cols = self._get_program_results_columns()
            select_parts = ["experiment_id"]

            def _avg(col: str, alias: Optional[str] = None) -> None:
                if col in available_cols:
                    select_parts.append(f"AVG({col}) as {alias or 'avg_' + col}")

            _avg("throughput_tok_s", "avg_throughput_tok_s_programs")
            _avg("routing_tokens_total", "avg_routing_tokens_total")
            _avg("routing_tokens_processed", "avg_routing_tokens_processed")
            _avg("routing_drop_rate", "avg_routing_drop_rate")
            _avg("routing_utilization_entropy", "avg_routing_utilization_entropy")
            _avg("routing_confidence_mean", "avg_routing_confidence_mean")
            _avg("routing_capacity_overflow_count", "avg_routing_capacity_overflow_count")
            _avg("discovery_loss_ratio", "avg_discovery_loss_ratio")
            _avg("validation_loss_ratio", "avg_validation_loss_ratio")
            _avg("generalization_gap", "avg_generalization_gap")
            if "routing_tokens_total" in available_cols and "routing_tokens_processed" in available_cols:
                select_parts.append(
                    "AVG(CASE WHEN routing_tokens_total > 0 "
                    "THEN CAST(routing_tokens_processed AS REAL) / routing_tokens_total END) "
                    "as avg_routing_token_retention"
                )
            _avg("depth_savings_ratio", "avg_depth_savings_ratio")
            _avg("effective_depth_ratio", "avg_effective_depth_ratio")
            _avg("recursion_savings_ratio", "avg_recursion_savings_ratio")
            _avg("recursion_depth_ratio", "avg_recursion_depth_ratio")

            if len(select_parts) > 1:
                placeholders = ",".join("?" for _ in exp_ids)
                query = (
                    f"SELECT {', '.join(select_parts)} "
                    f"FROM program_results WHERE experiment_id IN ({placeholders}) "
                    f"GROUP BY experiment_id"
                )
                agg_rows = self.conn.execute(query, exp_ids).fetchall()
                program_metrics_by_exp = {row["experiment_id"]: dict(row) for row in agg_rows}
        trends = []
        total_programs = 0
        total_stage1 = 0
        for r in rows:
            d = dict(r)
            exp_id = d.get("experiment_id")
            if exp_id and exp_id in program_metrics_by_exp:
                d.update(program_metrics_by_exp[exp_id])
            
            # Extract perf report if available
            results_json = d.get("results_json")
            if results_json:
                try:
                    res = self._decompress(results_json)
                    perf = res.get("perf_report")
                    if isinstance(perf, dict):
                        d["avg_step_time_ms"] = perf.get("trace_avg_ms", {}).get("forward_pass", 0) + \
                                               perf.get("trace_avg_ms", {}).get("backward_pass", 0)
                        d["avg_throughput_tok_s"] = perf.get("avg_throughput_tok_s", 0)
                        d["gpu_starvation_ms"] = perf.get("gpu_starvation", {}).get("total_stall_ms", 0)
                except Exception:
                    pass
            if d.get("avg_throughput_tok_s") in (None, 0) and d.get("avg_throughput_tok_s_programs") is not None:
                d["avg_throughput_tok_s"] = d.get("avg_throughput_tok_s_programs")

            n_programs = max(int(d.get("n_programs_generated") or 0), 0)
            n_stage1 = max(int(d.get("n_stage1_passed") or 0), 0)
            total = max(n_programs, 1)
            raw_s1_rate = n_stage1 / total

            trend_mode = _resolve_mode(d)
            mode_factor = _mode_factor(trend_mode)
            effective_n = max(1.0, n_programs * mode_factor)
            trend_weight = min(1.0, effective_n / 20.0)

            d["s1_pass_rate"] = raw_s1_rate
            d["trend_mode"] = trend_mode
            d["_effective_n"] = effective_n
            d["_trend_weight"] = trend_weight

            total_programs += n_programs
            total_stage1 += n_stage1
            trends.append(d)

        if not trends:
            return trends

        overall_rate = total_stage1 / max(total_programs, 1)
        prior_strength = 12.0

        for d in trends:
            raw_rate = d.get("s1_pass_rate") or 0.0
            effective_n = d.get("_effective_n") or 1.0
            trend_weight = d.get("_trend_weight") or 0.0

            shrinkage = effective_n / (effective_n + prior_strength)
            adjusted_rate = overall_rate + shrinkage * (raw_rate - overall_rate)

            variance = max(adjusted_rate * (1.0 - adjusted_rate), 0.0)
            stderr = math.sqrt(variance / max(effective_n, 1.0))
            halfwidth = 1.96 * stderr
            lower = max(0.0, adjusted_rate - halfwidth)
            upper = min(1.0, adjusted_rate + halfwidth)

            if effective_n >= 20:
                confidence = "high"
            elif effective_n >= 8:
                confidence = "medium"
            else:
                confidence = "low"

            d["adjusted_s1_pass_rate"] = round(adjusted_rate, 6)
            d["s1_confidence_lower"] = round(lower, 6)
            d["s1_confidence_upper"] = round(upper, 6)
            d["s1_confidence_halfwidth"] = round(halfwidth, 6)
            d["trend_weight"] = round(trend_weight, 4)
            d["trend_confidence"] = confidence

            d.pop("_effective_n", None)
            d.pop("_trend_weight", None)
            # Remove raw blob columns that are not JSON-serializable
            d.pop("results_json", None)
            d.pop("config_json", None)

        return self._json_clean(trends)


    def get_campaign_experiments(self, campaign_id: str) -> List[Dict]:
        """Get all experiments for a campaign."""
        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, status,
                      hypothesis, n_programs_generated, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, aria_mood,
                      duration_seconds
               FROM experiments WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]


    # ── Hypotheses ──

    def record_hypothesis(self, campaign_id: Optional[str],
                          prediction: str, reasoning: str,
                          test_method: str, success_metric: str,
                          parent_id: Optional[str] = None,
                          confidence: float = 0.5,
                          experiment_id: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> str:
        """Record a structured hypothesis. Returns hypothesis_id."""
        hypothesis_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO hypotheses
            (hypothesis_id, campaign_id, experiment_id, timestamp,
             prediction, reasoning, test_method, success_metric,
             parent_hypothesis_id, status, confidence_before, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (hypothesis_id, campaign_id, experiment_id, now,
             prediction, reasoning, test_method, success_metric,
             parent_id, confidence,
             json.dumps(metadata) if metadata else None),
        )
        # Update parent's child list
        if parent_id:
            parent = self.conn.execute(
                "SELECT child_hypotheses FROM hypotheses WHERE hypothesis_id = ?",
                (parent_id,),
            ).fetchone()
            if parent:
                children = json.loads(parent["child_hypotheses"] or "[]")
                children.append(hypothesis_id)
                self.conn.execute(
                    "UPDATE hypotheses SET child_hypotheses = ? WHERE hypothesis_id = ?",
                    (json.dumps(children), parent_id),
                )
        self._maybe_commit()
        return hypothesis_id


    def resolve_hypothesis(self, hypothesis_id: str, status: str,
                           evidence: str, summary: str,
                           confidence_after: float) -> None:
        """Resolve a hypothesis with outcome."""
        self.conn.execute(
            """UPDATE hypotheses SET
                status = ?, outcome_evidence = ?, outcome_summary = ?,
                confidence_after = ?
            WHERE hypothesis_id = ?""",
            (status, evidence, summary, confidence_after, hypothesis_id),
        )
        self._maybe_commit()


    def get_hypothesis_chain(self, hypothesis_id: str,
                             max_depth: int = 500) -> List[Dict]:
        """Trace lineage from root to all descendants."""
        # Find root (with cycle detection)
        current = hypothesis_id
        visited = {current}
        for _ in range(max_depth):
            row = self.conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            parent = row["parent_hypothesis_id"]
            if parent is None or parent in visited:
                break
            visited.add(parent)
            current = parent

        # BFS from root (with max nodes limit)
        chain: List[Dict] = []
        queue_ids = [current]
        seen: set = set()
        while queue_ids and len(chain) < max_depth:
            hid = queue_ids.pop(0)
            if hid in seen:
                continue
            seen.add(hid)
            row = self.conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (hid,),
            ).fetchone()
            if row is None:
                continue
            d = dict(row)
            chain.append(d)
            children = json.loads(d.get("child_hypotheses") or "[]")
            queue_ids.extend(children)
        return chain

