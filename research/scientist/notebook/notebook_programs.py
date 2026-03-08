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



class _ProgramsMixin:
    """Programs operations for the Lab Notebook."""

    def purge_junk_programs(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Delete Stage 0 failure program results that carry no useful data.

        Targets results where stage0_passed = 0 or NULL, excluding any that
        somehow passed stage1 (safety guard).

        Returns dict with 'deleted' or 'would_delete' count and 'dry_run' flag.
        """
        junk_query = """
            SELECT result_id, experiment_id FROM program_results
            WHERE (stage0_passed = 0 OR stage0_passed IS NULL)
              AND (stage1_passed != 1 OR stage1_passed IS NULL)
        """
        junk_rows = self.conn.execute(junk_query).fetchall()
        count = len(junk_rows)

        if dry_run or count == 0:
            return {"would_delete": count, "dry_run": True}

        junk_ids = [r["result_id"] for r in junk_rows]
        affected_experiments = {r["experiment_id"] for r in junk_rows if r["experiment_id"]}

        # Cascade delete in foreign-key dependency order
        batch_size = 500
        for i in range(0, len(junk_ids), batch_size):
            batch = junk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            self.conn.execute(
                f"DELETE FROM training_curves WHERE result_id IN ({placeholders})", batch
            )
            self.conn.execute(
                f"DELETE FROM leaderboard WHERE result_id IN ({placeholders})", batch
            )
            self.conn.execute(
                f"DELETE FROM program_results WHERE result_id IN ({placeholders})", batch
            )

        self._maybe_commit()

        # Recalculate op success rates for affected experiments
        for exp_id in affected_experiments:
            try:
                self.update_op_success_rates(exp_id)
            except Exception:
                pass  # non-critical

        return {"deleted": count, "dry_run": False}


    # ── Entries ──

    def add_entry(self, entry: ExperimentEntry) -> str:
        """Add a notebook entry."""
        entry_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO entries
            (entry_id, experiment_id, timestamp, entry_type, title, content,
             metadata_json, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, entry.experiment_id, time.time(),
             entry.entry_type, entry.title, entry.content,
             json.dumps(entry.metadata), ",".join(entry.tags)),
        )
        self._maybe_commit()
        return entry_id


    # ── Program Results ──

    def has_fingerprint(self, graph_fingerprint: str) -> bool:
        """Check if a computation graph has already been evaluated."""
        if not graph_fingerprint:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM program_results WHERE graph_fingerprint = ? LIMIT 1",
            (graph_fingerprint,),
        ).fetchone()
        return row is not None


    def record_program_result(self, experiment_id: str,
                              graph_fingerprint: str, graph_json: str,
                              result_id: Optional[str] = None,
                              **kwargs) -> str:
        """Record results for a single synthesized program.

        Accepts all program_results columns as keyword arguments.
        Boolean fields (stage0_passed, etc.) are converted to int.

        Quality gate: rejects results that provide no learning signal —
        S0 failures, S1 failures with no loss data, and results with
        errors — to keep the database lean and focused.
        """
        # ── Quality gate: reject noise ──
        s0 = kwargs.get("stage0_passed")
        s1 = kwargs.get("stage1_passed")
        loss_ratio = kwargs.get("loss_ratio")
        err = kwargs.get("error_message")

        # Reject S0 failures that carry no error classification.
        # S0 failures WITH error_type inform compile-failure clustering.
        if s0 is not None and not s0:
            error_type = kwargs.get("error_type")
            if not error_type:
                LOGGER.debug("Quality gate: dropping S0 failure with no error_type (fp=%s)", graph_fingerprint)
                return ""

        # Reject S1 failures that carry no learning signal at all:
        # no loss data AND no error classification AND no novelty data.
        # Failures WITH loss_ratio inform grammar weights; failures WITH
        # error_type inform failure-pattern clustering; failures WITH
        # novelty data inform op success rates — all are valuable.
        if s0 and not s1:
            error_type = kwargs.get("error_type")
            novelty = kwargs.get("novelty_score") or kwargs.get("novelty_confidence")
            if loss_ratio is None and not error_type and not novelty:
                LOGGER.debug("Quality gate: dropping S1 failure with no signal (fp=%s)", graph_fingerprint)
                return ""

        if not result_id:
            result_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Convert booleans to int for SQLite
        bool_fields = {
            "stage0_passed", "stage05_passed", "stage1_passed",
            "extreme_input_passed", "random_input_passed",
            "has_nan_output", "has_inf_output", "has_nan_grad", "has_zero_grad",
            "graph_has_gradient_path", "graph_uses_math_spaces",
            "graph_uses_frequency_domain", "regression_gate_pass",
        }
        for f in bool_fields:
            if f in kwargs and kwargs[f] is not None:
                kwargs[f] = int(kwargs[f])

        # Sanitize numeric types (NumPy/Torch scalars) → native Python to prevent blob storage
        kwargs = self._sanitize_numeric(kwargs)

        # Handle legacy 'throughput' -> 'throughput_tok_s' alias
        if "throughput" in kwargs:
            kwargs.setdefault("throughput_tok_s", kwargs.pop("throughput"))
        valid_columns = self._get_program_results_columns()
        unknown_cols: List[str] = []
        filtered_kwargs: Dict[str, Any] = {}
        for col, val in kwargs.items():
            if col in valid_columns:
                filtered_kwargs[col] = val
            else:
                unknown_cols.append(col)
        if unknown_cols:
            LOGGER.debug(
                "Dropping unknown program_results columns: %s",
                ", ".join(sorted(unknown_cols)),
            )

        # Build column list dynamically from what's provided
        base_cols = ["result_id", "experiment_id", "timestamp",
                     "graph_fingerprint", "graph_json"]
        base_vals = [result_id, experiment_id, now,
                     graph_fingerprint, graph_json]

        extra_cols = []
        extra_vals = []
        for col, val in filtered_kwargs.items():
            extra_cols.append(col)
            extra_vals.append(val)

        all_cols = base_cols + extra_cols
        all_vals = base_vals + extra_vals
        placeholders = ", ".join(["?"] * len(all_cols))
        col_str = ", ".join(all_cols)

        self._submit_write(
            f"INSERT INTO program_results ({col_str}) VALUES ({placeholders})",
            all_vals,
        )
        return result_id


    def save_op_rehabilitation_result(self, op_name: str, compile_passed: bool,
                                       forward_passed: bool, error_message: Optional[str],
                                       model_dim: int) -> None:
        """Store a rehabilitation test result."""
        self.conn.execute(
            """INSERT INTO op_rehabilitation_cache
               (op_name, compile_passed, forward_passed, error_message, tested_at, model_dim)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(op_name) DO UPDATE SET
                compile_passed = excluded.compile_passed,
                forward_passed = excluded.forward_passed,
                error_message = excluded.error_message,
                tested_at = excluded.tested_at,
                model_dim = excluded.model_dim""",
            (op_name, int(compile_passed), int(forward_passed), error_message, time.time(), model_dim),
        )
        self._maybe_commit()


    def get_top_programs(self, n: int = 20,
                         sort_by: str = "novelty_score") -> List[Dict]:
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty",
                       "behavioral_novelty", "validation_loss_ratio",
                       "discovery_loss_ratio"}
        if sort_by not in valid_sorts:
            sort_by = "novelty_score"

        order = "DESC" if sort_by == "novelty_score" else "ASC"
        if sort_by in ("structural_novelty", "behavioral_novelty"):
            order = "DESC"

        rows = self.conn.execute(
            f"""SELECT * FROM program_results
                WHERE stage1_passed = 1
                ORDER BY {sort_by} {order} NULLS LAST
                LIMIT ?""",
            (n,)
        ).fetchall()
        rows_dicts = [dict(r) for r in rows]
        for d in rows_dicts:
            if not d.get("architecture_family"):
                d["architecture_family"] = self._classify_architecture_family(
                    graph_json=d.get("graph_json"),
                    routing_mode=d.get("routing_mode"),
                )
        return rows_dicts


    def get_report_top_programs_grouped_by_fingerprint(
        self,
        n: int = 20,
        sort_by: str = "loss_ratio",
    ) -> List[Dict]:
        """Get report ranking rows grouped by graph fingerprint.

        Returns one representative survivor per fingerprint, enriched with
        repeat-count and run-spread metadata across all stage1 survivors.
        """
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty",
                       "behavioral_novelty", "validation_loss_ratio",
                       "discovery_loss_ratio"}
        if sort_by not in valid_sorts:
            sort_by = "loss_ratio"

        order = "DESC" if sort_by == "novelty_score" else "ASC"
        if sort_by in ("structural_novelty", "behavioral_novelty"):
            order = "DESC"

        # Pull enough candidates to fill n unique fingerprints.
        rows = self.conn.execute(
            f"""SELECT * FROM program_results
                WHERE stage1_passed = 1
                ORDER BY {sort_by} {order} NULLS LAST, timestamp DESC
                LIMIT ?""",
            (max(n * 12, 200),),
        ).fetchall()

        spread_rows = self.conn.execute(
            """SELECT
                   graph_fingerprint,
                   COUNT(*) AS repeat_count,
                   COUNT(DISTINCT experiment_id) AS repeat_experiment_span,
                   MIN(timestamp) AS repeat_first_seen_ts,
                   MAX(timestamp) AS repeat_last_seen_ts,
                   MIN(loss_ratio) AS repeat_loss_min,
                   MAX(loss_ratio) AS repeat_loss_max,
                   AVG(loss_ratio) AS repeat_loss_mean,
                   MIN(novelty_score) AS repeat_novelty_min,
                   MAX(novelty_score) AS repeat_novelty_max
               FROM program_results
               WHERE stage1_passed = 1
                 AND graph_fingerprint IS NOT NULL
                 AND TRIM(graph_fingerprint) != ''
               GROUP BY graph_fingerprint"""
        ).fetchall()
        spread_by_fp = {row["graph_fingerprint"]: dict(row) for row in spread_rows}

        grouped: List[Dict] = []
        seen_fingerprints = set()
        for row in rows:
            record = dict(row)
            fingerprint = record.get("graph_fingerprint")
            if not fingerprint or fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            spread = spread_by_fp.get(fingerprint, {})
            record["repeat_count"] = int(spread.get("repeat_count") or 1)
            record["repeat_experiment_span"] = int(spread.get("repeat_experiment_span") or 1)
            record["repeat_first_seen_ts"] = spread.get("repeat_first_seen_ts")
            record["repeat_last_seen_ts"] = spread.get("repeat_last_seen_ts")
            record["repeat_loss_min"] = spread.get("repeat_loss_min")
            record["repeat_loss_max"] = spread.get("repeat_loss_max")
            record["repeat_loss_mean"] = spread.get("repeat_loss_mean")
            record["repeat_novelty_min"] = spread.get("repeat_novelty_min")
            record["repeat_novelty_max"] = spread.get("repeat_novelty_max")
            grouped.append(record)

            if len(grouped) >= n:
                break

        return grouped


    def get_program_results(self, experiment_id: str,
                            limit: int = 500) -> List[Dict]:
        """Get ALL program results for an experiment (not just survivors)."""
        rows = self.conn.execute(
            """SELECT * FROM program_results
               WHERE experiment_id = ?
               ORDER BY novelty_score DESC NULLS LAST
               LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


    def get_program_detail(self, result_id: str) -> Optional[Dict]:
        """Get full detail for a single program result."""
        row = self.conn.execute(
            "SELECT * FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        return self._parse_program_json_fields(dict(row))


    def get_program_details(self, result_ids: List[str]) -> List[Dict]:
        """Batch fetch full details for multiple program results."""
        ids = [rid for rid in result_ids if rid]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM program_results WHERE result_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {}
        for row in rows:
            d = self._parse_program_json_fields(dict(row))
            by_id[d.get("result_id")] = d
        return [by_id.get(rid) for rid in ids]


    @staticmethod
    def _parse_program_json_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        """Parse known JSON fields for program results in-place."""
        json_fields = (
            "graph_json", "fingerprint_json", "training_program_json",
            "graph_category_histogram", "external_benchmarks_json",
            "perf_report_json", "kernel_timings_json", "starvation_report_json",
            "diagnostic_tasks_json", "sparsity_report_json"
        )
        for json_field in json_fields:
            val = d.get(json_field)
            if val and isinstance(val, str):
                try:
                    d[json_field + "_parsed"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


    def _sync_fingerprint_leaderboard(self, result_id: str) -> None:
        """Aggregate leaderboard evidence across all runs of a fingerprint.

        This ensures repeated training runs for the same architecture contribute
        to one coherent fingerprint-level score/tier rather than fragmenting
        across per-result rows.
        """
        fp_row = self.conn.execute(
            "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if not fp_row or not fp_row["graph_fingerprint"]:
            return
        graph_fingerprint = str(fp_row["graph_fingerprint"])

        lb_rows_raw = self.conn.execute(
            """
            SELECT l.*
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint = ?
            """,
            (graph_fingerprint,),
        ).fetchall()
        if not lb_rows_raw:
            return
        lb_rows = [dict(r) for r in lb_rows_raw]

        pr_cols_all = self._get_program_results_columns()
        wanted_pr_cols = [
            "result_id", "novelty_confidence", "loss_improvement_rate",
            "discovery_loss_ratio", "validation_loss_ratio", "efficiency_multiple",
            "max_viable_seq_len", "robustness_long_ctx_scaling_score",
            "robustness_long_ctx_assoc_score", "robustness_long_ctx_multi_hop_score",
            "robustness_long_ctx_passkey_score", "robustness_long_ctx_retrieval_aggregate",
            "robustness_long_ctx_combined_score", "robustness_noise_score",
            "activation_sparsity_score", "depth_savings_ratio",
            "recursion_savings_ratio", "routing_expert_count",
            "routing_confidence_mean", "routing_drop_rate",
            "wikitext_perplexity", "wikitext_score", "tinystories_perplexity",
            "tinystories_score", "cross_task_score", "efficiency_wall_score",
        ]
        pr_select_cols = [c for c in wanted_pr_cols if c in pr_cols_all]
        if not pr_select_cols:
            pr_select_cols = ["result_id"]
        pr_rows_raw = self.conn.execute(
            f"SELECT {', '.join(pr_select_cols)} FROM program_results WHERE graph_fingerprint = ?",
            (graph_fingerprint,),
        ).fetchall()
        pr_rows = [dict(r) for r in pr_rows_raw]

        # Use current best composite entry as the anchor for stable metadata.
        anchor = max(
            lb_rows,
            key=lambda r: (float(r.get("composite_score") or -1e9), float(r.get("timestamp") or 0.0)),
        )
        merged = dict(anchor)

        # Best-of-run metrics used directly by scoring.
        min_cols = (
            "screening_loss_ratio",
            "investigation_loss_ratio",
            "validation_loss_ratio",
            "validation_baseline_ratio",
            "validation_multi_seed_std",
            "discovery_loss_ratio",
            "compression_ratio",
            "routing_drop_rate",
            "robustness_noise_score",
            "wikitext_perplexity",
            "tinystories_perplexity",
            "ncd_score",
        )
        max_cols = (
            "screening_novelty",
            "investigation_robustness",
            "normalized_baseline_ratio",
            "param_efficiency",
            "quant_int8_retention",
            "quant_quality_per_byte",
            "robustness_long_ctx_score",
            "init_sensitivity_std",
            "scaling_param_efficiency",
            "scaling_flop_efficiency",
            "scaling_d512_param_efficiency",
            "routing_savings_ratio",
            "activation_sparsity_score",
            "depth_savings_ratio",
            "recursion_savings_ratio",
            "routing_expert_count",
            "routing_confidence_mean",
            "efficiency_multiple",
            "wikitext_score",
            "tinystories_score",
            "cross_task_score",
            "efficiency_wall_score",
            "max_viable_seq_len",
            "robustness_long_ctx_scaling_score",
            "robustness_long_ctx_assoc_score",
            "robustness_long_ctx_multi_hop_score",
            "robustness_long_ctx_passkey_score",
            "robustness_long_ctx_retrieval_aggregate",
            "robustness_long_ctx_combined_score",
            "loss_improvement_rate",
        )
        bool_cols = (
            "screening_passed",
            "investigation_passed",
            "validation_passed",
            "scaling_gate_passed",
        )

        # Combine leaderboard + program rows where useful.
        combo_rows = lb_rows + pr_rows
        for col in min_cols:
            best = self._best_min(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in max_cols:
            best = self._best_max(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in bool_cols:
            best = self._best_bool(combo_rows, col)
            if best is not None:
                merged[col] = best

        # Tier is fingerprint-level progression.
        highest_tier = self._highest_tier(lb_rows)
        if highest_tier:
            merged["tier"] = highest_tier

        nov_conf = self._best_max(pr_rows, "novelty_confidence")
        n_routing = self._count_routing_ops(result_id)
        n_sparse = self._count_sparse_ops(result_id)
        n_moe = self._count_moe_ops(result_id)
        composite = self.compute_composite_score(
            screening_lr=merged.get("screening_loss_ratio"),
            screening_nov=merged.get("screening_novelty"),
            inv_lr=merged.get("investigation_loss_ratio"),
            inv_robust=merged.get("investigation_robustness"),
            val_lr=merged.get("validation_loss_ratio"),
            val_baseline=merged.get("validation_baseline_ratio"),
            val_std=merged.get("validation_multi_seed_std"),
            novelty_confidence=nov_conf,
            scaling_param_efficiency=merged.get("scaling_param_efficiency"),
            is_reference=bool(merged.get("is_reference")),
            routing_savings=merged.get("routing_savings_ratio"),
            compression_ratio=merged.get("compression_ratio"),
            discovery_lr=merged.get("discovery_loss_ratio"),
            spectral_norm=merged.get("fp_jacobian_spectral_norm"),
            robustness_noise=merged.get("robustness_noise_score"),
            quant_retention=merged.get("quant_int8_retention"),
            long_ctx_score=merged.get("robustness_long_ctx_score"),
            init_std=merged.get("init_sensitivity_std"),
            loss_improvement_rate=merged.get("loss_improvement_rate"),
            quant_quality_per_byte=merged.get("quant_quality_per_byte"),
            ncd_score=merged.get("ncd_score"),
            n_routing_ops=n_routing,
            n_sparse_ops=n_sparse,
            n_moe_ops=n_moe,
            recursion_savings=merged.get("recursion_savings_ratio"),
            depth_savings=merged.get("depth_savings_ratio"),
            activation_sparsity=merged.get("activation_sparsity_score"),
            max_viable_seq_len=merged.get("max_viable_seq_len"),
            long_ctx_scaling=merged.get("robustness_long_ctx_scaling_score"),
            long_ctx_passkey=merged.get("robustness_long_ctx_passkey_score"),
            long_ctx_multi_hop=merged.get("robustness_long_ctx_multi_hop_score"),
            long_ctx_assoc=merged.get("robustness_long_ctx_assoc_score"),
            routing_expert_count=merged.get("routing_expert_count"),
            routing_confidence_mean=merged.get("routing_confidence_mean"),
            routing_drop_rate=merged.get("routing_drop_rate"),
            wikitext_perplexity=merged.get("wikitext_perplexity"),
        )
        # Monotonic safeguard: fingerprint aggregate should not score below its
        # historical best leaderboard score when incorporating additional runs.
        prior_best = self._best_max(lb_rows, "composite_score")
        if prior_best is not None:
            composite = max(float(composite), float(prior_best))

        update_cols = [
            "tier",
            "composite_score",
            "screening_loss_ratio",
            "screening_novelty",
            "screening_passed",
            "investigation_loss_ratio",
            "investigation_robustness",
            "investigation_passed",
            "validation_loss_ratio",
            "validation_baseline_ratio",
            "validation_multi_seed_std",
            "validation_passed",
            "discovery_loss_ratio",
            "loss_improvement_rate",
            "normalized_baseline_ratio",
            "param_efficiency",
            "quant_int8_retention",
            "quant_quality_per_byte",
            "robustness_long_ctx_score",
            "robustness_noise_score",
            "init_sensitivity_std",
            "scaling_param_efficiency",
            "scaling_flop_efficiency",
            "scaling_gate_passed",
            "scaling_d512_param_efficiency",
            "routing_savings_ratio",
            "compression_ratio",
            "activation_sparsity_score",
            "wikitext_perplexity",
            "wikitext_score",
            "tinystories_perplexity",
            "tinystories_score",
            "cross_task_score",
            "efficiency_wall_score",
            "max_viable_seq_len",
            "robustness_long_ctx_scaling_score",
            "robustness_long_ctx_assoc_score",
            "robustness_long_ctx_multi_hop_score",
            "robustness_long_ctx_passkey_score",
            "robustness_long_ctx_retrieval_aggregate",
            "robustness_long_ctx_combined_score",
            "depth_savings_ratio",
            "recursion_savings_ratio",
            "routing_expert_count",
            "routing_confidence_mean",
            "routing_drop_rate",
            "ncd_score",
            "efficiency_multiple",
            "timestamp",
        ]
        update_cols = [c for c in update_cols if c in self._get_leaderboard_columns()]
        sets = [f"{c} = ?" for c in update_cols]

        # Keep all rows for traceability but synchronize fingerprint-level evidence.
        now_ts = time.time()
        params_template = []
        for col in update_cols:
            if col == "composite_score":
                params_template.append(composite)
            elif col == "timestamp":
                params_template.append(now_ts)
            else:
                val = merged.get(col)
                if isinstance(val, bool):
                    val = int(val)
                params_template.append(val)

        for row in lb_rows:
            params = list(params_template)
            params.append(row["entry_id"])
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )


    def backfill_fingerprint_aggregates(self) -> int:
        """Recompute fingerprint-level leaderboard aggregates for all entries."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT l.result_id
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint IS NOT NULL
            """
        ).fetchall()
        synced = 0
        seen_fp: set[str] = set()
        for row in rows:
            rid = row["result_id"]
            fp_row = self.conn.execute(
                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                (rid,),
            ).fetchone()
            fp = str(fp_row["graph_fingerprint"]) if fp_row and fp_row["graph_fingerprint"] else ""
            if not fp or fp in seen_fp:
                continue
            seen_fp.add(fp)
            self._sync_fingerprint_leaderboard(rid)
            synced += 1
        self._maybe_commit()
        return synced


    def get_leaderboard_entry(self, result_id: str) -> Optional[Dict]:
        """Fetch a single leaderboard entry by result_id."""
        if not result_id:
            return None
        rows = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(rows) if rows else None


    def get_investigated_fingerprints(self) -> set:
        """Return fingerprints that have already been investigated or beyond.

        Checks both leaderboard tiers AND program_results from investigation/
        ablation experiments, so candidates tested in failed/interrupted
        investigations are not re-queued indefinitely.
        """
        fps = set()
        # Tier-based: candidates promoted in leaderboard
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM leaderboard l "
            "JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE l.tier IN ('investigation', 'validation', 'breakthrough')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        # History-based: fingerprints tested in investigation/ablation experiments
        # (catches failed/interrupted investigations that never reached leaderboard)
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM program_results pr "
            "JOIN experiments e ON e.experiment_id = pr.experiment_id "
            "WHERE e.experiment_type IN ('investigation', 'ablation')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        return fps


    def get_tiers_for_result_ids(self, result_ids: List[str]) -> Dict[str, str]:
        """Return {result_id: tier} for given result IDs that have leaderboard entries."""
        if not result_ids:
            return {}
        placeholders = ",".join("?" for _ in result_ids)
        rows = self.conn.execute(
            f"SELECT result_id, tier FROM leaderboard WHERE result_id IN ({placeholders})",
            result_ids,
        ).fetchall()
        return {r["result_id"]: r["tier"] for r in rows}

