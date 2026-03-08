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



class _KnowledgeMixin:
    """Knowledge operations for the Lab Notebook."""

    # ── Insights ──

    def record_insight(
        self,
        category: str,
        content: str,
        experiment_id: Optional[str] = None,
        confidence: float = 0.5,
        evidence: Optional[str] = None,
        insight_type: Optional[str] = None,
        subject_key: Optional[str] = None,
        semantic_key: Optional[str] = None,
    ) -> str:
        """Record an insight/learning, superseding active semantic duplicates."""
        inferred_type, inferred_subject, inferred_semantic = infer_insight_identity(
            category,
            content,
        )
        insight_type = str(insight_type or inferred_type).strip() or inferred_type
        subject_key = str(subject_key or inferred_subject).strip() or inferred_subject
        semantic_key = str(semantic_key or inferred_semantic).strip() or inferred_semantic

        if semantic_key:
            existing = self.conn.execute(
                """SELECT insight_id FROM insights
                   WHERE status = 'active' AND semantic_key = ?
                   ORDER BY confidence DESC, timestamp DESC""",
                (semantic_key,),
            ).fetchall()
            for row in existing:
                self.conn.execute(
                    "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
                    (row["insight_id"],),
                )

        insight_id = str(uuid.uuid4())[:12]
        insert_sql = """INSERT INTO insights
            (insight_id, timestamp, experiment_id, category, insight_type,
             subject_key, semantic_key, content, confidence, supporting_evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        insert_params = (
            insight_id,
            time.time(),
            experiment_id,
            category,
            insight_type,
            subject_key,
            semantic_key,
            content,
            confidence,
            evidence,
        )
        try:
            self.conn.execute(insert_sql, insert_params)
        except sqlite3.IntegrityError:
            if semantic_key:
                self.conn.execute(
                    "UPDATE insights SET status = 'superseded' WHERE status = 'active' AND semantic_key = ?",
                    (semantic_key,),
                )
                self.conn.execute(insert_sql, insert_params)
            else:
                raise
        self._maybe_commit()
        return insight_id


    def supersede_insight(self, insight_id: str) -> None:
        """Mark an insight as superseded (replaced by a newer version)."""
        self.conn.execute(
            "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
            (insight_id,),
        )
        self._maybe_commit()


    def get_insights(self, category: Optional[str] = None,
                     status: str = "active", limit: int = 50) -> List[Dict]:
        query = "SELECT * FROM insights WHERE status = ?"
        params = [status]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


    def record_selection_insight_trial(
        self,
        decision_id: str,
        context: str,
        insight_ids: List[str],
        chosen_result_ids: List[str],
        source_experiment_id: Optional[str] = None,
    ) -> str:
        """Record one insight-bundle trial tied to a selection decision."""
        trial_id = str(uuid.uuid4())[:12]
        now = time.time()
        cleaned_insights = sorted({
            str(i).strip() for i in (insight_ids or []) if str(i).strip()
        })
        cleaned_results = sorted({
            str(r).strip() for r in (chosen_result_ids or []) if str(r).strip()
        })
        self.conn.execute(
            """INSERT INTO selection_insight_trials
               (trial_id, decision_id, timestamp, context, source_experiment_id,
                insight_ids_json, chosen_result_ids_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                trial_id,
                decision_id,
                now,
                context or "",
                source_experiment_id,
                json.dumps(cleaned_insights),
                json.dumps(cleaned_results),
            ),
        )
        self._maybe_commit()
        return trial_id


    def get_pending_selection_insight_trials(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return unresolved insight-bundle trials."""
        rows = self.conn.execute(
            """SELECT * FROM selection_insight_trials
               WHERE status = 'pending'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("insight_ids_json", "chosen_result_ids_json", "metadata_json"):
                raw = item.get(key)
                if not raw:
                    continue
                try:
                    item[key] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(item)
        return out


    def resolve_selection_insight_trial(
        self,
        trial_id: str,
        reward: float,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Resolve a trial and update pairwise insight interaction stats."""
        row = self.conn.execute(
            "SELECT * FROM selection_insight_trials WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        if row is None:
            return
        trial = dict(row)
        if str(trial.get("status") or "") == "resolved":
            return
        now = time.time()
        reward_value = float(reward or 0.0)
        outcome_text = str(outcome or "inconclusive").strip() or "inconclusive"
        self.conn.execute(
            """UPDATE selection_insight_trials
               SET status = 'resolved',
                   reward = ?,
                   outcome = ?,
                   resolved_timestamp = ?,
                   metadata_json = ?
               WHERE trial_id = ?""",
            (
                reward_value,
                outcome_text,
                now,
                json.dumps(metadata or {}),
                trial_id,
            ),
        )

        try:
            insight_ids = json.loads(trial.get("insight_ids_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            insight_ids = []
        cleaned = sorted({
            str(i).strip() for i in (insight_ids or []) if str(i).strip()
        })
        if not cleaned:
            self._maybe_commit()
            return

        # Track singleton and pair interactions. Singleton uses (id, id).
        pairs: List[Tuple[str, str]] = []
        for insight_id in cleaned:
            pairs.append((insight_id, insight_id))
        for i in range(len(cleaned)):
            for j in range(i + 1, len(cleaned)):
                a, b = cleaned[i], cleaned[j]
                if a > b:
                    a, b = b, a
                pairs.append((a, b))

        supported_inc = 1 if outcome_text == "supported" else 0
        not_supported_inc = 1 if outcome_text == "not_supported" else 0
        for insight_a, insight_b in pairs:
            self.conn.execute(
                """INSERT INTO selection_insight_interactions
                   (insight_a, insight_b, n_trials, n_supported, n_not_supported,
                    cumulative_reward, mean_reward, last_reward, last_outcome, last_updated)
                   VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(insight_a, insight_b) DO UPDATE SET
                     n_trials = n_trials + 1,
                     n_supported = n_supported + excluded.n_supported,
                     n_not_supported = n_not_supported + excluded.n_not_supported,
                     cumulative_reward = cumulative_reward + excluded.last_reward,
                     mean_reward = (cumulative_reward + excluded.last_reward) / (n_trials + 1),
                     last_reward = excluded.last_reward,
                     last_outcome = excluded.last_outcome,
                     last_updated = excluded.last_updated""",
                (
                    insight_a,
                    insight_b,
                    supported_inc,
                    not_supported_inc,
                    reward_value,
                    reward_value,
                    reward_value,
                    outcome_text,
                    now,
                ),
            )
        self._maybe_commit()


    def get_selection_insight_interactions(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return learned insight interaction stats sorted by confidence/reward."""
        rows = self.conn.execute(
            """SELECT * FROM selection_insight_interactions
               ORDER BY n_trials DESC, mean_reward DESC
               LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
        return [dict(r) for r in rows]


    # ── Knowledge Base ──

    def add_knowledge(self, category: str, title: str, content: str,
                      evidence: Optional[List[str]] = None,
                      confidence: float = 0.5) -> str:
        """Add a knowledge base entry. Returns entry_id."""
        entry_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO knowledge_base
            (entry_id, timestamp, category, title, content, confidence,
             supporting_evidence, times_validated, last_validated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'active')""",
            (entry_id, now, category, title, content, confidence,
             json.dumps(evidence) if evidence else None, now),
        )
        self._maybe_commit()
        return entry_id


    def get_knowledge(self, category: Optional[str] = None) -> List[Dict]:
        """Get knowledge base entries, optionally filtered by category."""
        query = "SELECT * FROM knowledge_base WHERE status = 'active'"
        params: List[Any] = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY timestamp DESC"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            base_conf = float(d.get("confidence") or 0.5)
            validated = int(d.get("times_validated") or 0)
            # Validation bonus saturates; repeated confirmations help but are capped.
            val_bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
            effective_conf = min(0.95, max(0.0, base_conf) + val_bonus)
            d["effective_confidence"] = round(effective_conf, 4)
            d["validation_bonus"] = round(val_bonus, 4)
            d["confidence_capped"] = effective_conf >= 0.95
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        results.sort(
            key=lambda row: (
                float(row.get("effective_confidence") or 0.0),
                int(row.get("times_validated") or 0),
                float(row.get("timestamp") or 0.0),
            ),
            reverse=True,
        )
        return results


    def validate_knowledge(self, entry_id: str) -> None:
        """Increment times_validated and update last_validated."""
        now = time.time()
        self.conn.execute(
            """UPDATE knowledge_base SET
                times_validated = times_validated + 1,
                last_validated = ?
            WHERE entry_id = ?""",
            (now, entry_id),
        )
        self._maybe_commit()


    def search_knowledge(self, query: str) -> List[Dict]:
        """Simple LIKE search on title + content."""
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM knowledge_base
               WHERE status = 'active'
               AND (title LIKE ? OR content LIKE ?)
               ORDER BY timestamp DESC""",
            (pattern, pattern),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            base_conf = float(d.get("confidence") or 0.5)
            validated = int(d.get("times_validated") or 0)
            val_bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
            effective_conf = min(0.95, max(0.0, base_conf) + val_bonus)
            d["effective_confidence"] = round(effective_conf, 4)
            d["validation_bonus"] = round(val_bonus, 4)
            d["confidence_capped"] = effective_conf >= 0.95
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        results.sort(
            key=lambda row: (
                float(row.get("effective_confidence") or 0.0),
                int(row.get("times_validated") or 0),
                float(row.get("timestamp") or 0.0),
            ),
            reverse=True,
        )
        return results

