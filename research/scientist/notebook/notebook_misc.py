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



class _MiscMixin:
    """Misc operations for the Lab Notebook."""

    # ── Training Curves ──

    def store_training_curve(self, result_id: str,
                             curve: List[Dict]) -> None:
        """Store per-step training data for survivors only.

        curve: list of dicts with keys step, loss, grad_norm, step_time_ms
        """
        if not curve or not result_id:
            return
        # Only store curves for results that passed S1 (survivors).
        # S1 failure learning signal is captured in loss_ratio, not per-step curves.
        row = self.conn.execute(
            "SELECT stage1_passed FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row and row[0] == 0:
            return
        self.conn.executemany(
            """INSERT OR REPLACE INTO training_curves
               (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            [(result_id, d.get("step", i), d.get("loss"),
              d.get("grad_norm"), d.get("step_time_ms"))
             for i, d in enumerate(curve)],
        )
        self._maybe_commit()


    def get_training_curve(self, result_id: str) -> List[Dict]:
        """Get per-step training data for a program."""
        rows = self.conn.execute(
            """SELECT step, loss, grad_norm, step_time_ms
               FROM training_curves WHERE result_id = ?
               ORDER BY step""",
            (result_id,),
        ).fetchall()
        return [dict(r) for r in rows]


    def strip_graph_json_for_failures(self, experiment_id: str) -> int:
        """Clear graph_json for S1 failures with no loss data.

        Called after update_op_success_rates() has already consumed the graphs.
        Sets to empty string (NOT NULL constraint on column).
        Returns the number of rows stripped.
        """
        cur = self.conn.execute(
            """UPDATE program_results SET graph_json = ''
               WHERE experiment_id = ?
                 AND stage0_passed = 1 AND stage1_passed = 0
                 AND loss_ratio IS NULL AND length(graph_json) > 0""",
            (experiment_id,),
        )
        n = cur.rowcount
        if n:
            self._maybe_commit()
        return n


    def merge_op_failure_counts(self, op_counts: Dict[str, Dict[str, int]]) -> None:
        """Merge S0 failure op counts into op_success_rates.

        Called after update_op_success_rates() to incorporate ops from programs
        that failed S0/S0.5 and were not stored in program_results.

        Args:
            op_counts: {op_name: {"n_used": int, "n_s0": int, "n_s05": int}}
        """
        if not op_counts:
            return
        now = time.time()
        for op_name, counts in op_counts.items():
            self.conn.execute(
                """INSERT INTO op_success_rates
                   (op_name, n_used, n_stage0_passed, n_stage05_passed,
                    n_stage1_passed, last_updated)
                   VALUES (?, ?, ?, ?, 0, ?)
                   ON CONFLICT(op_name) DO UPDATE SET
                    n_used = n_used + excluded.n_used,
                    n_stage0_passed = n_stage0_passed + excluded.n_stage0_passed,
                    n_stage05_passed = n_stage05_passed + excluded.n_stage05_passed,
                    last_updated = excluded.last_updated""",
                (op_name, counts.get("n_used", 0), counts.get("n_s0", 0),
                 counts.get("n_s05", 0), now),
            )
        self._maybe_commit()


    # ── Failure Signatures ──

    
    def _extract_op_bigrams(graph_json: str) -> List[str]:
        """Extract sorted op-pair bigrams from a graph JSON.

        A bigram is "opA->opB" for each edge in the graph.  Returns a
        sorted deduplicated list, giving a compact structural fingerprint
        of what-connects-to-what.
        """
        try:
            data = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return []
        nodes = data.get("nodes", {})
        bigrams: set = set()
        for nid, nd in nodes.items():
            op = nd.get("op_name", "")
            if not op or op == "input":
                continue
            for inp in nd.get("input_ids", []):
                parent = nodes.get(str(inp), {})
                pop = parent.get("op_name", "")
                if pop and pop != "input":
                    bigrams.add(f"{pop}->{op}")
        return sorted(bigrams)


    def get_entries(self, experiment_id: Optional[str] = None,
                    entry_type: Optional[str] = None,
                    limit: int = 50) -> List[Dict]:
        query = "SELECT * FROM entries WHERE 1=1"
        params = []
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


    def set_external_benchmarks(self, result_id: str, payload: Any) -> bool:
        """Store external benchmark payload for a program result."""
        if not result_id:
            return False
        serialized = None
        try:
            if payload is None:
                serialized = None
            elif isinstance(payload, dict):
                # Merge partial benchmark updates (for example, scaling-only writes)
                # with any previously stored benchmark families (for example, long_context).
                existing = self.conn.execute(
                    "SELECT external_benchmarks_json FROM program_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                merged: Dict[str, Any] = {}
                if existing and existing["external_benchmarks_json"]:
                    try:
                        parsed = json.loads(existing["external_benchmarks_json"])
                        if isinstance(parsed, dict):
                            merged.update(parsed)
                    except Exception:
                        pass
                merged.update(payload)
                serialized = json.dumps(merged)
            else:
                serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return False
        cur = self.conn.execute(
            "UPDATE program_results SET external_benchmarks_json = ? WHERE result_id = ?",
            (serialized, result_id),
        )
        self._maybe_commit()
        return cur.rowcount > 0


    def get_failure_analysis(self, experiment_id: str) -> Dict:
        """Get failure analysis data for an experiment."""
        programs = self.get_program_results(experiment_id)
        total = len(programs)
        if total == 0:
            return {"total": 0, "funnel": {}, "errors": {}, "stage_deaths": {}}

        s0_pass = sum(1 for p in programs if p.get("stage0_passed"))
        s05_pass = sum(1 for p in programs if p.get("stage05_passed"))
        s1_pass = sum(1 for p in programs if p.get("stage1_passed"))

        # Error type distribution (use classified error_type if available)
        errors: Dict[str, int] = {}
        for p in programs:
            err_type = p.get("error_type") or ""
            err_msg = p.get("error_message") or p.get("stage0_error") or ""
            key = err_type if err_type else err_msg[:80].strip()
            if key:
                errors[key] = errors.get(key, 0) + 1

        # Stage-at-death histogram
        stage_deaths = {"validation": 0, "stage0": 0, "stage0.5": 0, "stage1": 0}
        for p in programs:
            sad = p.get("stage_at_death")
            if sad and sad in stage_deaths:
                stage_deaths[sad] += 1
            elif not p.get("stage0_passed"):
                stage_deaths["stage0"] += 1
            elif not p.get("stage05_passed"):
                stage_deaths["stage0.5"] += 1
            elif not p.get("stage1_passed"):
                stage_deaths["stage1"] += 1

        return {
            "total": total,
            "funnel": {
                "generated": total,
                "stage0_passed": s0_pass,
                "stage05_passed": s05_pass,
                "stage1_passed": s1_pass,
            },
            "errors": dict(sorted(errors.items(), key=lambda x: -x[1])[:10]),
            "stage_deaths": stage_deaths,
        }


    def _json_clean(self, obj: Any) -> Any:
        """Deep clean object for JSON serialization (handles NaN/Inf)."""
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            return None
        if isinstance(obj, dict):
            return {k: self._json_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_clean(x) for x in obj]
        return obj


    def get_dashboard_summary(self) -> Dict:
        """Get aggregate stats for the dashboard."""
        total_exp = self.conn.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]
        completed = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'completed'"
        ).fetchone()[0]
        total_programs = self.conn.execute(
            "SELECT COUNT(*) FROM program_results"
        ).fetchone()[0]
        stage1_survivors = self.conn.execute(
            "SELECT COUNT(*) FROM program_results WHERE stage1_passed = 1"
        ).fetchone()[0]
        avg_novelty = self.conn.execute(
            "SELECT AVG(novelty_score) FROM program_results WHERE novelty_score IS NOT NULL"
        ).fetchone()[0]
        top_novelty = self.conn.execute(
            "SELECT MAX(novelty_score) FROM program_results"
        ).fetchone()[0]
        n_insights = self.conn.execute(
            "SELECT COUNT(*) FROM insights WHERE status = 'active'"
        ).fetchone()[0]

        # Learning summary
        n_learning_events = self.conn.execute(
            "SELECT COUNT(*) FROM learning_log"
        ).fetchone()[0]
        latest_learning = self.conn.execute(
            "SELECT description FROM learning_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        avg_step_time = self.conn.execute(
            "SELECT AVG(avg_step_time_ms) FROM program_results WHERE avg_step_time_ms IS NOT NULL"
        ).fetchone()[0]
        avg_throughput = self.conn.execute(
            "SELECT AVG(throughput_tok_s) FROM program_results WHERE throughput_tok_s IS NOT NULL"
        ).fetchone()[0]
        avg_entropy = self.conn.execute(
            "SELECT AVG(routing_utilization_entropy) FROM program_results WHERE routing_utilization_entropy IS NOT NULL"
        ).fetchone()[0]
        avg_savings = self.conn.execute(
            "SELECT AVG(depth_savings_ratio) FROM program_results WHERE depth_savings_ratio IS NOT NULL"
        ).fetchone()[0]
        avg_recursion_savings = self.conn.execute(
            "SELECT AVG(recursion_savings_ratio) FROM program_results WHERE recursion_savings_ratio IS NOT NULL"
        ).fetchone()[0]
        avg_token_retention = self.conn.execute(
            """SELECT AVG(CASE WHEN routing_tokens_total > 0
                               THEN CAST(routing_tokens_processed AS REAL) / routing_tokens_total END)
               FROM program_results"""
        ).fetchone()[0]
        avg_sparsity = self.conn.execute(
            "SELECT AVG(sparsity_ratio) FROM program_results WHERE sparsity_ratio IS NOT NULL"
        ).fetchone()[0]
        
        # Unique fingerprint count for grammar diversity tracking
        unique_fingerprints = self.conn.execute(
            "SELECT COUNT(DISTINCT graph_fingerprint) FROM program_results"
        ).fetchone()[0]

        latest_perf_report = None
        latest_dedup = None
        latest_perf_row = self.conn.execute(
            """SELECT experiment_id, completed_at, results_json
               FROM experiments
               WHERE status = 'completed'
                 AND results_json IS NOT NULL
               ORDER BY completed_at DESC
               LIMIT 1"""
        ).fetchone()
        if latest_perf_row and latest_perf_row["results_json"]:
            try:
                latest_results = json.loads(latest_perf_row["results_json"])
                perf_report = latest_results.get("perf_report") if isinstance(latest_results, dict) else None
                if isinstance(perf_report, dict):
                    queue = perf_report.get("queue_telemetry") or {}
                    kernel_hotspots = perf_report.get("kernel_hotspots") or []
                    top_kernel = kernel_hotspots[0] if kernel_hotspots else None
                    latest_perf_report = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "completed_at": latest_perf_row["completed_at"],
                        "programs_profiled": int(perf_report.get("programs_profiled", 0) or 0),
                        "avg_submit_wait_ms": float(queue.get("submit_wait_avg_ms", 0.0) or 0.0),
                        "avg_scheduling_wait_ms": float(queue.get("scheduling_wait_avg_ms", 0.0) or 0.0),
                        "gpu_starvation_events": int((perf_report.get("gpu_starvation") or {}).get("event_count", 0) or 0),
                        "top_kernel": top_kernel,
                    }
                # Extract dedup stats from latest experiment
                if isinstance(latest_results, dict) and "dedup_rate" in latest_results:
                    latest_dedup = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "dedup_rate": latest_results.get("dedup_rate", 0),
                        "skipped_dedup": latest_results.get("skipped_dedup", 0),
                        "novel_count": latest_results.get("dedup_novel_count", 0),
                        "known_fingerprints": latest_results.get("dedup_known_fingerprints", 0),
                    }
            except (TypeError, ValueError, json.JSONDecodeError):
                latest_perf_report = None

        return {
            "total_experiments": total_exp,
            "completed_experiments": completed,
            "total_programs_evaluated": total_programs,
            "stage1_survivors": stage1_survivors,
            "survival_rate": stage1_survivors / max(total_programs, 1),
            "avg_novelty_score": avg_novelty or 0,
            "top_novelty_score": top_novelty or 0,
            "active_insights": n_insights,
            "learning_events": n_learning_events,
            "latest_learning": latest_learning[0] if latest_learning else None,
            "avg_step_time_ms": avg_step_time or 0,
            "avg_throughput_tok_s": avg_throughput or 0,
            "avg_routing_entropy": avg_entropy,
            "avg_depth_savings": avg_savings,
            "avg_recursion_savings": avg_recursion_savings,
            "avg_routing_token_retention": avg_token_retention,
            "avg_sparsity_ratio": avg_sparsity,
            "latest_perf_report": latest_perf_report,
            "unique_fingerprints": unique_fingerprints,
            "latest_dedup": latest_dedup,
        }


    # ── Leaderboard ──

    # Ops considered "routing" for the structural complexity bonus
    _ROUTING_OPS = frozenset({
        "route_topk", "route_lanes", "route_recursion", "token_merge",
        "mod_topk", "early_exit", "adaptive_recursion", "token_merging",
        "cascade", "speculative", "moe_topk", "adaptive_lane_mixer",
        "mixed_recursion_gate", "relu_gate_routing", "routing_conditioned_compression",
        "token_type_classifier", "entropy_router", "progressive_compression_gate",
        "compression_mixture_experts", "latent_attention_compressor",
    })

    _SPARSE_OPS = frozenset({
        "nm_sparse_linear", "block_sparse_linear", "semi_structured_2_4_linear",
        "structured_sparse", "block_sparse", "semi_structured_2_4",
        "hash_trick", "sparse_topk", "latent_attention_compressor",
        "routing_conditioned_compression", "compression_mixture_experts",
        "progressive_compression_gate",
    })

    _MOE_OPS = frozenset({
        "moe_topk", "route_topk", "route_lanes", "adaptive_lane_mixer",
        "compression_mixture_experts", "entropy_router",
    })
    _TIER_ORDER = {
        "screening": 0,
        "investigation": 1,
        "validation": 2,
        "breakthrough": 3,
    }

    def _count_routing_ops(self, result_id: str) -> Optional[int]:
        """Count routing/branching ops in the graph for a program result."""
        try:
            row = self.conn.execute(
                "SELECT graph_json FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row[0]:
                return None
            graph_data = json.loads(row[0])
            nodes = graph_data.get("nodes", [])
            count = sum(1 for n in nodes if n.get("op") in self._ROUTING_OPS)
            return count if count > 0 else None
        except Exception:
            return None


    def _count_sparse_ops(self, result_id: str) -> Optional[int]:
        """Count sparsity/compression ops in the graph for a program result."""
        try:
            row = self.conn.execute(
                "SELECT graph_json FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row[0]:
                return None
            graph_data = json.loads(row[0])
            nodes = graph_data.get("nodes", [])
            count = sum(1 for n in nodes if n.get("op") in self._SPARSE_OPS)
            return count if count > 0 else None
        except Exception:
            return None


    def _count_moe_ops(self, result_id: str) -> Optional[int]:
        """Count MoE-specific ops in the graph for a program result."""
        try:
            row = self.conn.execute(
                "SELECT graph_json FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row[0]:
                return None
            graph_data = json.loads(row[0])
            nodes = graph_data.get("nodes", [])
            count = sum(1 for n in nodes if n.get("op") in self._MOE_OPS)
            return count if count > 0 else None
        except Exception:
            return None


    
    def _best_min(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        try:
            return float(min(vals))
        except Exception:
            return None


    
    def _best_max(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        try:
            return float(max(vals))
        except Exception:
            return None


    
    def _best_bool(rows: List[Dict[str, Any]], key: str) -> Optional[int]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return int(any(bool(v) for v in vals))


    
    def compute_efficiency_multiple(self, 
        loss_ratio: Optional[float] = None,
        param_count: Optional[float] = None,
        flops_forward: Optional[float] = None,
        throughput_tok_s: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
        forward_time_ms: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Geometric mean of per-dimension ratios vs GPT-2.

        All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
        dimensions to return a result (graceful with missing data).
        Returns dict with per-dimension ratios and ``geomean``, or None.
        """
        ref = self._GPT2_REF
        ratios: Dict[str, float] = {}

        # x_quality: ref_loss / cand_loss (lower loss = better)
        if loss_ratio is not None and loss_ratio > 0:
            ratios["x_quality"] = ref["loss_ratio"] / loss_ratio

        # x_params: ref_params / cand_params (fewer = better)
        if param_count is not None and param_count > 0:
            ratios["x_params"] = ref["param_count"] / param_count

        # x_flops: ref_flops / cand_flops (fewer = better)
        if flops_forward is not None and flops_forward > 0:
            ratios["x_flops"] = ref["flops_forward"] / flops_forward

        # x_throughput: cand_tput / ref_tput (higher = better)
        if throughput_tok_s is not None and throughput_tok_s > 0:
            ratios["x_throughput"] = throughput_tok_s / ref["throughput_tok_s"]

        # x_memory: ref_mem / cand_mem (less = better)
        if peak_memory_mb is not None and peak_memory_mb > 0:
            ratios["x_memory"] = ref["peak_memory_mb"] / peak_memory_mb

        # x_latency: ref_lat / cand_lat (lower = better)
        if forward_time_ms is not None and forward_time_ms > 0:
            ratios["x_latency"] = ref["forward_time_ms"] / forward_time_ms

        if len(ratios) < 3:
            return None

        geomean = 1.0
        for v in ratios.values():
            geomean *= v
        geomean = geomean ** (1.0 / len(ratios))
        ratios["geomean"] = geomean
        ratios["n_dimensions"] = float(len(ratios) - 1)  # exclude geomean itself
        return ratios


    
    def compute_composite_score(self, 
        screening_lr: Optional[float] = None,
        screening_nov: Optional[float] = None,
        inv_lr: Optional[float] = None,
        inv_robust: Optional[float] = None,
        val_lr: Optional[float] = None,
        val_baseline: Optional[float] = None,
        val_std: Optional[float] = None,
        novelty_confidence: Optional[float] = None,
        scaling_param_efficiency: Optional[float] = None,
        is_reference: bool = False,
        routing_savings: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        entropy: Optional[float] = None,
        discovery_lr: Optional[float] = None,
        spectral_norm: Optional[float] = None,
        robustness_noise: Optional[float] = None,
        quant_retention: Optional[float] = None,
        long_ctx_score: Optional[float] = None,
        init_std: Optional[float] = None,
        loss_improvement_rate: Optional[float] = None,
        quant_quality_per_byte: Optional[float] = None,
        ncd_score: Optional[float] = None,
        n_routing_ops: Optional[int] = None,
        n_sparse_ops: Optional[int] = None,
        n_moe_ops: Optional[int] = None,
        recursion_savings: Optional[float] = None,
        depth_savings: Optional[float] = None,
        activation_sparsity: Optional[float] = None,
        max_viable_seq_len: Optional[int] = None,
        long_ctx_scaling: Optional[float] = None,
        long_ctx_passkey: Optional[float] = None,
        long_ctx_multi_hop: Optional[float] = None,
        long_ctx_assoc: Optional[float] = None,
        routing_expert_count: Optional[int] = None,
        routing_confidence_mean: Optional[float] = None,
        routing_drop_rate: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Compute "Total Scientific Utility" — an open-ended additive score.
        ...
        """
        score = 0.0

        # 1. Performance Utility (Primary)
        # Use validation_baseline_ratio if available, otherwise fallback
        perf_lr = val_baseline if val_baseline is not None else (
            val_lr if val_lr is not None else (
                inv_lr if inv_lr is not None else screening_lr
            )
        )
        if perf_lr is not None:
            score += 100.0 * max(0, 1.0 - perf_lr)
        
        # Discovery channel (random tokens)
        if discovery_lr is not None:
            score += 20.0 * max(0, 1.0 - discovery_lr)

        # Learning Efficiency: How fast did it learn?
        if loss_improvement_rate is not None:
            # High improvement rate per step is efficient learning
            # Up to 20 points
            score += 20.0 * max(0, min(1.0, loss_improvement_rate))

        # 2. Novelty Utility
        eff_nov = 1.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
        conf = 1.0 if is_reference else (novelty_confidence if novelty_confidence is not None else 1.0)
        score += 40.0 * eff_nov * conf

        # 3. Efficiency & Scaling Utility
        if scaling_param_efficiency is not None:
            score += 10.0 * max(0, scaling_param_efficiency - 1.0)
        
        if routing_savings is not None:
            score += 50.0 * routing_savings
            
        if compression_ratio is not None:
            # Reward compression: 4x (0.25) -> 20 utility
            # Weight compression ratio + maintained quality
            comp_score = 20.0 * max(0, 1.0 - (compression_ratio / 1.0))
            if quant_quality_per_byte is not None:
                # Reward high quality per compressed byte
                comp_score += 10.0 * max(0, quant_quality_per_byte)
            score += comp_score

        # NCD: reward compact graph descriptions that explain training behavior
        if ncd_score is not None:
            # Low NCD = graph structure predicts training dynamics (good)
            # Max 15 points when NCD = 0
            score += 15.0 * max(0, 1.0 - ncd_score)

        # 3b. Structural complexity bonus: reward routing/branching architectures
        # Counterbalances MDL and NCD penalties for exotic architectures
        if n_routing_ops is not None and n_routing_ops > 0:
            # Up to 15 points (reduced from 25, replaced by MoE quality)
            score += min(15.0, n_routing_ops * 5.0)

        # 3c. Sparsity bonus (max 30pts: 20 structural + 10 activation)
        if n_sparse_ops is not None and n_sparse_ops > 0:
            score += min(20.0, n_sparse_ops * 6.0)
        if activation_sparsity is not None and activation_sparsity > 0.3:
            score += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)

        # 3d. MoE quality bonus (max ~25pts)
        if n_moe_ops is not None and n_moe_ops > 0:
            moe_base = min(10.0, n_moe_ops * 5.0)
            # Expert diversity multiplier: more experts = higher potential
            if routing_expert_count is not None and routing_expert_count > 1:
                expert_mult = min(1.5, 1.0 + math.log2(routing_expert_count) / 6.0)
                moe_base *= expert_mult
            # Confidence bonus: high confidence = routing is working
            if routing_confidence_mean is not None and routing_confidence_mean > 0.5:
                moe_base *= 1.0 + 0.3 * (routing_confidence_mean - 0.5)
            # Drop rate penalty: high drop = wasted compute
            if routing_drop_rate is not None and routing_drop_rate > 0.3:
                moe_base *= max(0.5, 1.0 - (routing_drop_rate - 0.3))
            score += moe_base

        # 3e. Adaptive computation bonus (max 25pts)
        if recursion_savings is not None and recursion_savings > 0:
            score += 15.0 * min(1.0, recursion_savings / 0.5)
        if depth_savings is not None and depth_savings > 0:
            score += 10.0 * min(1.0, depth_savings / 0.5)

        # 4. Robustness & Stability Utility
        if spectral_norm is not None:
            score += 10.0 * max(0, 1.0 - (spectral_norm / 20.0))

        if robustness_noise is not None:
            score += 15.0 * max(0, 1.0 - robustness_noise)

        if quant_retention is not None:
            score += 15.0 * max(0, quant_retention - 0.5) / 0.5

        # 4b. Expanded long-context scoring (total budget 50pts, up from 20)
        if long_ctx_score is not None:
            # Base combined score: 20pts (unchanged)
            score += 20.0 * long_ctx_score
            # Sub-score bonuses: reward specific long-context capabilities
            if long_ctx_passkey is not None:
                score += 10.0 * long_ctx_passkey
            if long_ctx_multi_hop is not None:
                score += 10.0 * long_ctx_multi_hop
            if long_ctx_scaling is not None:
                score += 5.0 * long_ctx_scaling
            if long_ctx_assoc is not None:
                score += 5.0 * long_ctx_assoc

        # Bonus for viable long sequences (log-scale, max 20pts)
        if max_viable_seq_len is not None and max_viable_seq_len > 512:
            seq_bonus = 5.0 * min(4.0, math.log2(max_viable_seq_len / 512))
            score += seq_bonus

        # 5. Generalization Utility (The "Anti-Cheat")
        # Wikitext/TinyStories scores are normalized 0-1, where 1 is good (low perplexity)
        # We also look at raw perplexity for severe penalties.
        # Note: These values might be in the future, but we add them now.
        
        # If we have raw perplexity data, apply severe penalties for "Zombie" models
        # We assume 10^6 is the cutoff for total failure to generalize.
        # We use wikitext_perplexity as the primary proxy.
        # This function signature might need updating or we use kwargs
        wikitext_perplexity = kwargs.get("wikitext_perplexity")
        if wikitext_perplexity is not None:
            if wikitext_perplexity > 1000000:
                return 0.0 # Instant disqualification for non-generalizing models
            if wikitext_perplexity > 1000:
                # Logarithmic penalty for high perplexity
                score -= 50.0 * math.log10(wikitext_perplexity / 1000.0)

        # 6. Numerical Integrity (Spectral Floor)
        if spectral_norm is not None and spectral_norm < 0.01:
            # Gradients are likely not propagating (numerical collapse)
            score -= 40.0

        # 7. Penalties
        if val_std is not None and val_std > 0.1:
            # High variance across seeds is a major red flag
            score -= 50.0 * min(2.0, val_std / 0.5)
            
        if entropy is not None and entropy > 0.95:
            # Only penalize truly unfocused routing, not healthy multi-lane distribution
            score -= 5.0 * (entropy - 0.95)

        # Sanity floor
        return max(0.0, score)


    
    def _reference_novelty_for_display(novelty: Optional[float]) -> Optional[float]:
        """Compress reference novelty values for dashboard display.

        Reference architectures are anchor points, so we intentionally present
        their novelty on a reduced scale to avoid implying they are frontier
        discoveries in the same sense as synthesized candidates.
        """
        if novelty is None:
            return None
        try:
            value = float(novelty)
        except (TypeError, ValueError):
            return None
        value = max(0.0, min(1.0, value))
        return min(0.35, value * 0.4)


    def pin_reference(self, entry_id: str, reference_name: str) -> None:
        """Pin a leaderboard entry as a reference architecture."""
        self.conn.execute(
            """UPDATE leaderboard
               SET is_reference = 1,
                   reference_name = ?,
                   model_source = 'reference'
               WHERE entry_id = ?""",
            (reference_name, entry_id),
        )
        self._maybe_commit()


    # ── Pre-investigation gate helpers ──────────────────────────────────

    def get_investigation_eligible(
        self,
        max_lr: float,
        min_stability: float,
        min_spectral_norm: float,
        max_spectral_norm: float,
        min_improvement_rate: float,
        ref_lr_ceiling: Optional[float] = None,
    ) -> List[Dict]:
        """Stage A hard reject: return screening candidates that pass all hard filters.

        Joins program_results with leaderboard to return full metric rows for
        candidates eligible for investigation.
        """
        lr_ceiling = ref_lr_ceiling if ref_lr_ceiling is not None else max_lr
        rows = self.conn.execute(
            """SELECT pr.*, l.entry_id, l.tier, l.composite_score,
                      l.screening_loss_ratio, l.screening_novelty,
                      l.pre_inv_score, l.is_reference, l.reference_name
               FROM program_results pr
               JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE l.tier = 'screening'
                 AND COALESCE(l.is_reference, 0) = 0
                 AND pr.stage1_passed = 1
                 AND COALESCE(pr.has_nan_grad, 0) = 0
                 AND COALESCE(pr.has_nan_output, 0) = 0
                 AND COALESCE(pr.has_inf_output, 0) = 0
                 AND COALESCE(pr.has_zero_grad, 0) = 0
                 AND COALESCE(pr.graph_has_gradient_path, 1) = 1
                 AND COALESCE(pr.stability_score, 0) >= ?
                 AND (pr.fp_jacobian_spectral_norm IS NULL
                      OR (pr.fp_jacobian_spectral_norm >= ? AND pr.fp_jacobian_spectral_norm <= ?))
                 AND COALESCE(pr.loss_improvement_rate, 0) >= ?
                 AND COALESCE(pr.loss_ratio, 1.0) < ?
               ORDER BY pr.loss_ratio ASC NULLS LAST""",
            (min_stability, min_spectral_norm, max_spectral_norm,
             min_improvement_rate, lr_ceiling),
        ).fetchall()
        return [dict(r) for r in rows]


    
    def compute_pre_investigation_score(row: Dict, best_ref_lr: Optional[float] = None) -> float:
        """Stage B composite readiness score (0-100 scale).

        Components:
        - Performance (40pts): loss_ratio, discovery_loss_ratio, loss_improvement_rate
        - Stability (20pts): stability_score, spectral_norm (Gaussian around 1.0), grad_norm_std
        - Novelty (20pts): novelty_score * confidence, structural_novelty, behavioral_novelty
        - Fingerprint quality (10pts): fp_intrinsic_dim, fp_isotropy, fp_rank_ratio
        - Efficiency (10pts): throughput_tok_s, peak_memory_mb
        - Reference penalty (-20pts): if loss_ratio > 1.5 * best_reference_lr
        """
        import math
        score = 0.0

        # ── Performance (40 pts) ──
        lr = row.get("loss_ratio")
        if lr is not None and lr > 0:
            # Lower LR is better; LR=0.1 → 40pts, LR=0.8 → ~8pts
            score += max(0, min(40, 40 * (1.0 - float(lr))))

        dlr = row.get("discovery_loss_ratio")
        if dlr is not None and dlr > 0:
            # Bonus: up to 5pts from discovery loss (replaces top of performance)
            score += max(0, min(5, 5 * (1.0 - float(dlr))))

        lir = row.get("loss_improvement_rate")
        if lir is not None and float(lir) > 0:
            # Up to 5pts for improvement rate
            score += min(5, float(lir) * 10)

        # Cap performance at 40
        score = min(40, score)

        # ── Stability (20 pts) ──
        stab = row.get("stability_score")
        if stab is not None:
            score += min(10, float(stab) * 10)

        sn = row.get("fp_jacobian_spectral_norm")
        if sn is not None and float(sn) > 0:
            # Gaussian centered on 1.0: score = 6 * exp(-(log(sn))^2 / 2)
            log_sn = math.log(float(sn))
            score += max(0, min(6, 6 * math.exp(-log_sn * log_sn / 2.0)))

        gns = row.get("grad_norm_std")
        if gns is not None:
            # Lower grad_norm_std is better; up to 4pts
            score += max(0, min(4, 4 * max(0, 1.0 - float(gns))))

        # ── Novelty (20 pts) ──
        ns = row.get("novelty_score")
        nc = row.get("novelty_confidence")
        if ns is not None:
            conf = float(nc) if nc is not None else 0.5
            score += min(10, float(ns) * conf * 10)

        sn_nov = row.get("structural_novelty")
        if sn_nov is not None:
            score += min(5, float(sn_nov) * 5)

        bn = row.get("behavioral_novelty")
        if bn is not None:
            score += min(5, float(bn) * 5)

        # ── Fingerprint quality (10 pts) ──
        fid = row.get("fp_intrinsic_dim")
        if fid is not None and float(fid) > 0:
            # Higher intrinsic dim → better; up to 4pts, cap at dim=20
            score += min(4, float(fid) / 5.0)

        fiso = row.get("fp_isotropy")
        if fiso is not None:
            score += min(3, float(fiso) * 3)

        frr = row.get("fp_rank_ratio")
        if frr is not None:
            score += min(3, float(frr) * 3)

        # ── Efficiency (10 pts) ──
        tp = row.get("throughput_tok_s")
        if tp is not None and float(tp) > 0:
            # Higher throughput → better; up to 5pts, 10k tok/s → 5pts
            score += min(5, float(tp) / 2000.0)

        mem = row.get("peak_memory_mb")
        if mem is not None and float(mem) > 0:
            # Lower memory → better; up to 5pts, 100MB → 5pts, 500MB → 1pt
            score += max(0, min(5, 5 * (1.0 - float(mem) / 600.0)))

        # ── Reference penalty (-20 pts) ──
        if best_ref_lr is not None and lr is not None:
            if float(lr) > 1.5 * float(best_ref_lr):
                score -= 20

        return max(0, min(100, round(score, 2)))


    def get_references(self) -> List[Dict]:
        """Get all pinned reference architectures."""
        rows = self.conn.execute(
            """SELECT l.*, pr.graph_json AS _graph_json,
                      pr.routing_mode AS _routing_mode,
                      pr.graph_fingerprint AS _graph_fingerprint
               FROM leaderboard l
               LEFT JOIN program_results pr ON pr.result_id = l.result_id
               WHERE COALESCE(l.is_reference, 0) = 1
               ORDER BY l.composite_score DESC NULLS LAST, l.reference_name ASC, l.timestamp DESC"""
        ).fetchall()
        refs: List[Dict] = []
        for row in rows:
            entry = dict(row)
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
            entry["architecture_family"] = self._classify_architecture_family(
                graph_json=entry.pop("_graph_json", None),
                routing_mode=entry.pop("_routing_mode", None),
            )
            entry["screening_novelty"] = self._reference_novelty_for_display(
                entry.get("screening_novelty")
            )
            refs.append(entry)
        return refs


    
    def _classify_architecture_family(self, 
        graph_json: Optional[str],
        routing_mode: Optional[str],
    ) -> str:
        """Map graph structure to a compact architecture family label."""
        if routing_mode:
            return "Routed-MoE"
        if not graph_json:
            return "Unknown"

        try:
            graph = json.loads(graph_json)
            nodes = graph.get("nodes")
            if isinstance(nodes, dict):
                node_iter = [n for n in nodes.values() if isinstance(n, dict)]
            elif isinstance(nodes, list):
                node_iter = [n for n in nodes if isinstance(n, dict)]
            else:
                node_iter = []
            ops = {str(n.get("op_name", "")).strip() for n in node_iter}
        except (json.JSONDecodeError, TypeError, ValueError):
            return "Unknown"

        if not ops:
            return "Unknown"

        attention_ops = {
            "attention", "self_attention", "mha", "multihead_attention", "qkv_attention",
            "softmax_attention", "linear_attention",
        }
        conv_ops = {"conv1d", "conv1d_seq", "depthwise_conv1d", "conv_only"}
        spectral_ops = {"sin", "cos", "fft", "ifft", "fourier_mix", "fourier_mixing", "rfft_seq", "irfft_seq"}
        gating_ops = {"sigmoid", "tanh", "silu", "gelu", "maximum", "minimum", "swiglu", "topk_gate", "moe_topk"}
        mlp_ops = {"linear_proj", "linear_proj_up", "linear_proj_down", "learnable_bias", "swiglu_mlp"}
        ssm_ops = {"state_space", "selective_scan"}
        adaptive_ops = {"mod_topk", "early_exit", "adaptive_recursion", "fixed_point_iter"}

        has_attention = bool(ops & attention_ops)
        has_conv = bool(ops & conv_ops)
        has_spectral = bool(ops & spectral_ops)
        has_gating = bool(ops & gating_ops)
        has_mlp = bool(ops & mlp_ops)
        has_ssm = bool(ops & ssm_ops)
        has_adaptive = bool(ops & adaptive_ops) or routing_mode in ("mod_topk", "early_exit", "adaptive_recursion")

        family = "Hybrid-Mixer"
        if has_ssm:
            family = "Mamba-SSM" if not has_attention else "Hybrid-SSM"
        elif has_attention:
            if has_conv or has_spectral or has_gating:
                family = "Hybrid-Attention"
            else:
                family = "Attention"
        elif has_conv and has_spectral:
            family = "Spectral-Conv"
        elif has_spectral:
            family = "Spectral-Mixer"
        elif has_conv:
            family = "Conv-Mixer"
        elif has_gating and has_mlp:
            family = "Gated-MLP"
        elif has_mlp:
            family = "MLP-Mixer"
        elif has_gating:
            family = "Nonlinear-Mixer"

        # Apply modifiers
        if routing_mode == "moe_topk" or "moe_topk" in ops:
            family = f"MoE-{family}"
        if has_adaptive:
            family = f"Adaptive-{family}"

        return family


    def get_unresolved_hypotheses(self,
                                  campaign_id: Optional[str] = None) -> List[Dict]:
        """Get pending/testing hypotheses."""
        query = "SELECT * FROM hypotheses WHERE status IN ('pending', 'testing')"
        params: List[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        query += " ORDER BY timestamp DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


    # ── Decisions ──

    def record_decision(self, campaign_id: Optional[str],
                        decision_type: str, subject: str,
                        rationale: str,
                        evidence_ids: Optional[List[str]] = None,
                        alternatives: Optional[List[Dict]] = None,
                        evidence_pack: Optional[Dict] = None) -> str:
        """Record a go/no-go or other decision. Returns decision_id."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO decisions
            (decision_id, campaign_id, timestamp, decision_type,
             subject, rationale, evidence_ids, alternatives_considered,
             evidence_pack_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (decision_id, campaign_id, now, decision_type, subject,
             rationale,
             json.dumps(evidence_ids) if evidence_ids else None,
             json.dumps(alternatives) if alternatives else None,
             json.dumps(evidence_pack) if evidence_pack else None),
        )
        self._maybe_commit()
        return decision_id


    def get_decisions(self, campaign_id: Optional[str] = None,
                      decision_type: Optional[str] = None) -> List[Dict]:
        """Get decisions, optionally filtered."""
        query = "SELECT * FROM decisions WHERE 1=1"
        params: List[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        if decision_type:
            query += " AND decision_type = ?"
            params.append(decision_type)
        query += " ORDER BY timestamp DESC"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            for f in ("evidence_ids", "alternatives_considered"):
                if d.get(f):
                    try:
                        d[f] = json.loads(d[f])
                    except (json.JSONDecodeError, TypeError):
                        pass
            if d.get("evidence_pack_json"):
                try:
                    d["evidence_pack"] = json.loads(d["evidence_pack_json"])
                except (json.JSONDecodeError, TypeError):
                    d["evidence_pack"] = None
            results.append(d)
        return results


    # ── Selection Decisions / Family Bandit Stats ──

    def record_selection_decision(
        self,
        context: str,
        candidate_pool_summary: Dict[str, Any],
        score_breakdown: List[Dict[str, Any]],
        policy: Dict[str, Any],
        reason: str,
        chosen_experiments: List[Dict[str, Any]],
        experiment_id: Optional[str] = None,
        trigger: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record one evidence-based experiment-selection decision."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_decisions
            (decision_id, timestamp, context, experiment_id,
             candidate_pool_summary_json, score_breakdown_json,
             policy_json, reason, chosen_experiments_json, trigger_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                now,
                context,
                experiment_id,
                json.dumps(candidate_pool_summary or {}),
                json.dumps(score_breakdown or []),
                json.dumps(policy or {}),
                reason or "",
                json.dumps(chosen_experiments or []),
                json.dumps(trigger or {}),
            ),
        )
        self._maybe_commit()
        return decision_id


    def get_selection_decisions(
        self,
        context: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return selection decisions newest first."""
        query = "SELECT * FROM selection_decisions WHERE 1=1"
        params: List[Any] = []
        if context:
            query += " AND context = ?"
            params.append(context)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in (
                "candidate_pool_summary_json",
                "score_breakdown_json",
                "policy_json",
                "chosen_experiments_json",
                "trigger_json",
            ):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out


    def get_selection_family_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return family bandit stats keyed by family name."""
        rows = self.conn.execute(
            "SELECT * FROM selection_family_stats"
        ).fetchall()
        return {r["family"]: dict(r) for r in rows}


    def update_selection_family_stats(self, family: str, reward: float) -> None:
        """Update per-family running reward estimate for UCB/uncertainty."""
        family_name = (family or "Unknown").strip() or "Unknown"
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_family_stats
            (family, n_trials, cumulative_reward, mean_reward, last_reward, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(family) DO UPDATE SET
                n_trials = n_trials + 1,
                cumulative_reward = cumulative_reward + excluded.last_reward,
                mean_reward = (cumulative_reward + excluded.last_reward) * 1.0
                              / (n_trials + 1),
                last_reward = excluded.last_reward,
                last_updated = excluded.last_updated
            """,
            (family_name, float(reward), float(reward), float(reward), now),
        )
        self._maybe_commit()


    # ── Novelty Calibration ──

    def record_novelty_calibration(
        self,
        reference_version: str,
        n_runs: int,
        distribution: Dict[str, Any],
        noise_floor_mean: Optional[float] = None,
        noise_floor_std: Optional[float] = None,
        confidence_low: Optional[float] = None,
        confidence_high: Optional[float] = None,
        cka_source: Optional[str] = None,
        cka_artifact_version: Optional[str] = None,
        probe_protocol_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist novelty baseline calibration stats."""
        calibration_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO novelty_calibration
            (calibration_id, timestamp, reference_version, cka_source,
             cka_artifact_version, probe_protocol_hash, n_runs,
             noise_floor_mean, noise_floor_std, confidence_low, confidence_high,
             distribution_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id,
                now,
                reference_version,
                cka_source,
                cka_artifact_version,
                probe_protocol_hash,
                int(max(1, n_runs)),
                noise_floor_mean,
                noise_floor_std,
                confidence_low,
                confidence_high,
                json.dumps(distribution or {}),
                json.dumps(metadata or {}),
            ),
        )
        self._maybe_commit()
        return calibration_id


    def get_latest_novelty_calibration(
        self,
        reference_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest novelty calibration row, optionally by reference version."""
        query = "SELECT * FROM novelty_calibration WHERE 1=1"
        params: List[Any] = []
        if reference_version:
            query += " AND reference_version = ?"
            params.append(reference_version)
        query += " ORDER BY timestamp DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        out = dict(row)
        for key in ("distribution_json", "metadata_json"):
            raw = out.get(key)
            if raw:
                try:
                    out[key] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out

