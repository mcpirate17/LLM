"""
Electronic Lab Notebook

Persistent, structured record of all experiments, hypotheses,
observations, and conclusions. Stored as SQLite for queryability
and served to the React dashboard via API.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import sqlite3
import subprocess
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .preregistration import PreregistrationError, validate_preregistration
except Exception:  # direct-module loading fallback for test harness
    import importlib.util as _importlib_util
    import sys as _sys

    _prereg_path = Path(__file__).with_name("preregistration.py")
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
    -- Composite score
    composite_score REAL,
    tier TEXT DEFAULT 'screening',  -- 'screening', 'investigation', 'validation', 'breakthrough'
    -- Metadata
    tags TEXT,
    notes TEXT
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
    source TEXT NOT NULL DEFAULT 'aria-designer',
    total_time_ms REAL,
    metrics_json TEXT,
    payload_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_designer_lineage_workflow ON designer_run_lineage(workflow_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_designer_lineage_status ON designer_run_lineage(status, updated_at DESC);
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


class LabNotebook:
    """Electronic lab notebook for the AI scientist."""

    _cached_code_version: Optional[str] = None
    _last_report_snapshot_cleanup_at: float = 0.0

    def __init__(self, db_path: str | Path = "research/lab_notebook.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        # Enable WAL mode for high-concurrency performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._batch_depth = 0
        self._program_results_columns: Optional[set[str]] = None
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
                self.conn.execute(
                    f"ALTER TABLE program_results ADD COLUMN {col_name} {col_type}"
                )

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
                notes TEXT
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
        self._program_results_columns = None

    def _get_program_results_columns(self) -> set[str]:
        """Return current program_results columns for defensive inserts."""
        if self._program_results_columns is None:
            rows = self.conn.execute("PRAGMA table_info(program_results)").fetchall()
            self._program_results_columns = {str(row[1]) for row in rows}
        return self._program_results_columns

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

        # Migrate leaderboard: add campaign_id if missing
        lb_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
        }
        if "campaign_id" not in lb_cols:
            self.conn.execute(
                "ALTER TABLE leaderboard ADD COLUMN campaign_id TEXT"
            )
        return cls._cached_code_version

        self._maybe_commit()

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

    # ── Code Healer ──

    def create_healer_task(
        self,
        experiment_id: Optional[str],
        trigger_type: str,
        scope: str,
        reproduction_steps: List[str],
        acceptance_tests: List[str],
        model_endpoint: Optional[str],
        sandbox_policy: Dict[str, Any],
        trigger_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = f"heal-{uuid.uuid4().hex[:10]}"
        now = time.time()
        self.conn.execute(
            """INSERT INTO healer_tasks
            (task_id, timestamp, experiment_id, trigger_type, trigger_payload_json,
             scope, reproduction_steps_json, acceptance_tests_json, model_endpoint,
             sandbox_policy_json, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (
                task_id,
                now,
                experiment_id,
                trigger_type,
                json.dumps(trigger_payload or {}),
                scope,
                json.dumps(reproduction_steps or []),
                json.dumps(acceptance_tests or []),
                model_endpoint,
                json.dumps(sandbox_policy or {}),
            ),
        )
        self._maybe_commit()
        return task_id

    def update_healer_task(
        self,
        task_id: str,
        state: Optional[str] = None,
        patch_summary: Optional[str] = None,
        risk_assessment: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        completed: bool = False,
    ) -> None:
        sets: List[str] = []
        params: List[Any] = []
        if state is not None:
            sets.append("state = ?")
            params.append(state)
        if patch_summary is not None:
            sets.append("patch_summary = ?")
            params.append(patch_summary)
        if risk_assessment is not None:
            sets.append("risk_assessment = ?")
            params.append(risk_assessment)
        if result is not None:
            sets.append("result_json = ?")
            params.append(json.dumps(result))
        if completed:
            sets.append("completed_at = ?")
            params.append(time.time())
        if not sets:
            return
        params.append(task_id)
        self.conn.execute(
            f"UPDATE healer_tasks SET {', '.join(sets)} WHERE task_id = ?",
            params,
        )
        self._maybe_commit()

    def add_healer_event(
        self,
        task_id: str,
        message: str,
        state: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO healer_task_events
            (event_id, task_id, timestamp, state, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                task_id,
                time.time(),
                state,
                message,
                json.dumps(payload or {}),
            ),
        )
        self._maybe_commit()
        return event_id

    def get_healer_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM healer_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        for field in (
            "trigger_payload_json",
            "reproduction_steps_json",
            "acceptance_tests_json",
            "sandbox_policy_json",
            "result_json",
        ):
            raw = out.get(field)
            if raw:
                try:
                    out[field] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out

    def get_recent_healer_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM healer_tasks
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in (
                "trigger_payload_json",
                "reproduction_steps_json",
                "acceptance_tests_json",
                "sandbox_policy_json",
                "result_json",
            ):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out

    def get_healer_events(self, task_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM healer_task_events
               WHERE task_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (task_id, limit),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.get("payload_json")
            if raw:
                try:
                    item["payload_json"] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(item)
        return out

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

    def record_program_result(self, experiment_id: str,
                              graph_fingerprint: str, graph_json: str,
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

    def get_op_success_rates(self) -> List[Dict]:
        """Get all op success rates."""
        rows = self.conn.execute(
            """SELECT * FROM op_success_rates
               ORDER BY n_stage1_passed DESC, n_used DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Failure Signatures ──

    @staticmethod
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

    def update_failure_signatures(self, experiment_id: str) -> None:
        """Update failure_signatures table from program results in this experiment.

        Extracts op-pair bigrams from each graph and tracks how often
        each bigram appears in failed vs successful programs.  This gives
        Aria a compact memory of which structural patterns to avoid.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL""",
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
               FROM program_results WHERE graph_json IS NOT NULL"""
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

    def get_failure_signature_blocklist(self, min_seen: int = 5,
                                        max_fail_rate: float = 0.85) -> Dict[str, float]:
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

    # ── Metrics ──

    def log_metric(self, metric_name: str, value: float,
                   experiment_id: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Log a time-series metric."""
        self._submit_write(
            """INSERT INTO metrics_log
            (timestamp, experiment_id, metric_name, metric_value, metadata_json)
            VALUES (?, ?, ?, ?, ?)""",
            (time.time(), experiment_id, metric_name, value,
             json.dumps(metadata) if metadata else None),
        )

    # ── Insights ──

    def record_insight(
        self,
        category: str,
        content: str,
        experiment_id: Optional[str] = None,
        confidence: float = 0.5,
        evidence: Optional[str] = None,
    ) -> str:
        """Record an insight/learning."""
        insight_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO insights
            (insight_id, timestamp, experiment_id, category, content,
             confidence, supporting_evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, time.time(), experiment_id, category,
             content, confidence, evidence),
        )
        self._maybe_commit()
        return insight_id

    def supersede_insight(self, insight_id: str) -> None:
        """Mark an insight as superseded (replaced by a newer version)."""
        self.conn.execute(
            "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
            (insight_id,),
        )
        self._maybe_commit()

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

    def get_top_programs(self, n: int = 20,
                         sort_by: str = "novelty_score") -> List[Dict]:
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty",
                       "behavioral_novelty"}
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
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty", "behavioral_novelty"}
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

    def get_metrics(self, metric_name: str,
                    experiment_id: Optional[str] = None,
                    limit: int = 1000) -> List[Dict]:
        query = "SELECT * FROM metrics_log WHERE metric_name = ?"
        params = [metric_name]
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

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
        source: str = "aria-designer",
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
            "diagnostic_tasks_json"
        )
        for json_field in json_fields:
            val = d.get(json_field)
            if val and isinstance(val, str):
                try:
                    d[json_field + "_parsed"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def set_external_benchmarks(self, result_id: str, payload: Any) -> bool:
        """Store external benchmark payload for a program result."""
        if not result_id:
            return False
        try:
            serialized = json.dumps(payload) if payload is not None else None
        except (TypeError, ValueError):
            return False
        cur = self.conn.execute(
            "UPDATE program_results SET external_benchmarks_json = ? WHERE result_id = ?",
            (serialized, result_id),
        )
        self._maybe_commit()
        return cur.rowcount > 0

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

    @staticmethod
    def compute_composite_score(
        screening_lr: Optional[float] = None,
        screening_nov: Optional[float] = None,
        inv_lr: Optional[float] = None,
        inv_robust: Optional[float] = None,
        val_lr: Optional[float] = None,
        val_baseline: Optional[float] = None,
        val_std: Optional[float] = None,
        novelty_confidence: Optional[float] = None,
    ) -> float:
        """Compute normalized composite score across research phases.

        Each tier produces a score in [0, 1], so programs at different
        stages are comparable without later stages dominating by
        construction (#49).

        Validation std uses hard threshold: std > 0.5 caps the
        validation score at 0.3 to strongly penalize instability (#50).

        Novelty contribution is scaled by novelty_confidence so that
        low-confidence scores (structural-only, failed fingerprint)
        don't inflate rankings.
        """
        # Screening tier: [0, 1]
        screening_score = 0.0
        if screening_lr is not None:
            screening_score += 0.6 * max(0, 1 - screening_lr)
        if screening_nov is not None:
            conf = novelty_confidence if novelty_confidence is not None else 1.0
            screening_score += 0.4 * screening_nov * conf

        # Investigation tier: [0, 1]
        inv_score = 0.0
        has_inv = inv_lr is not None or inv_robust is not None
        if inv_lr is not None:
            inv_score += 0.6 * max(0, 1 - inv_lr)
        if inv_robust is not None:
            inv_score += 0.4 * inv_robust

        # Validation tier: [0, 1]
        val_score = 0.0
        has_val = val_baseline is not None
        if val_baseline is not None:
            val_score += 0.6 * max(0, 1 - val_baseline)
        if val_std is not None:
            # Hard threshold: high variability strongly penalized
            if val_std > 0.5:
                val_score = min(val_score, 0.3)
            else:
                val_score += 0.4 * (1 / (1 + val_std))

        # Weighted combination — later tiers matter more but are
        # normalized so a screening-only program isn't artificially low
        if has_val:
            return 0.2 * screening_score + 0.3 * inv_score + 0.5 * val_score
        elif has_inv:
            return 0.4 * screening_score + 0.6 * inv_score
        else:
            return screening_score

    def upsert_leaderboard(
        self,
        result_id: str,
        model_source: str,
        architecture_desc: str = "",
        screening_loss_ratio: Optional[float] = None,
        screening_novelty: Optional[float] = None,
        screening_passed: bool = False,
        investigation_loss_ratio: Optional[float] = None,
        investigation_robustness: Optional[float] = None,
        investigation_best_training: Optional[str] = None,
        investigation_passed: bool = False,
        validation_loss_ratio: Optional[float] = None,
        validation_baseline_ratio: Optional[float] = None,
        validation_multi_seed_std: Optional[float] = None,
        validation_passed: bool = False,
        tier: str = "screening",
        tags: Optional[str] = None,
        notes: Optional[str] = None,
        novelty_confidence: Optional[float] = None,
    ) -> str:
        """Insert or update a leaderboard entry."""
        # Check if entry exists for this result_id
        existing = self.conn.execute(
            "SELECT entry_id FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()

        composite = self.compute_composite_score(
            screening_lr=screening_loss_ratio,
            screening_nov=screening_novelty,
            inv_lr=investigation_loss_ratio,
            inv_robust=investigation_robustness,
            val_lr=validation_loss_ratio,
            val_baseline=validation_baseline_ratio,
            val_std=validation_multi_seed_std,
            novelty_confidence=novelty_confidence,
        )

        if existing:
            entry_id = existing["entry_id"]
            self.conn.execute(
                """UPDATE leaderboard SET
                    timestamp = ?, model_source = ?, architecture_desc = ?,
                    screening_loss_ratio = ?, screening_novelty = ?,
                    screening_passed = ?,
                    investigation_loss_ratio = ?, investigation_robustness = ?,
                    investigation_best_training = ?, investigation_passed = ?,
                    validation_loss_ratio = ?, validation_baseline_ratio = ?,
                    validation_multi_seed_std = ?, validation_passed = ?,
                    composite_score = ?, tier = ?, tags = ?, notes = ?
                WHERE entry_id = ?""",
                (time.time(), model_source, architecture_desc,
                 screening_loss_ratio, screening_novelty,
                 int(screening_passed),
                 investigation_loss_ratio, investigation_robustness,
                 investigation_best_training, int(investigation_passed),
                 validation_loss_ratio, validation_baseline_ratio,
                 validation_multi_seed_std, int(validation_passed),
                 composite, tier, tags, notes,
                 entry_id),
            )
        else:
            entry_id = str(uuid.uuid4())[:12]
            self.conn.execute(
                """INSERT INTO leaderboard
                (entry_id, result_id, timestamp, model_source, architecture_desc,
                 screening_loss_ratio, screening_novelty, screening_passed,
                 investigation_loss_ratio, investigation_robustness,
                 investigation_best_training, investigation_passed,
                 validation_loss_ratio, validation_baseline_ratio,
                 validation_multi_seed_std, validation_passed,
                 composite_score, tier, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, result_id, time.time(), model_source,
                 architecture_desc,
                 screening_loss_ratio, screening_novelty,
                 int(screening_passed),
                 investigation_loss_ratio, investigation_robustness,
                 investigation_best_training, int(investigation_passed),
                 validation_loss_ratio, validation_baseline_ratio,
                 validation_multi_seed_std, int(validation_passed),
                 composite, tier, tags, notes),
            )

        self._maybe_commit()
        return entry_id

    def get_leaderboard(self, tier: Optional[str] = None,
                        limit: int = 50,
                        sort_by: str = "composite_score",
                        include_family: bool = True) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {"composite_score", "screening_loss_ratio",
                       "investigation_loss_ratio", "validation_loss_ratio",
                       "screening_novelty", "timestamp"}
        if sort_by not in valid_sorts:
            sort_by = "composite_score"

        query = (
            "SELECT l.*, pr.graph_json AS _graph_json, "
            "pr.routing_mode AS _routing_mode, "
            "pr.graph_fingerprint AS _graph_fingerprint, "
            "pr.arch_spec_json AS _arch_spec_json, "
            "pr.param_count AS _param_count, "
            "pr.graph_n_params_estimate AS _graph_n_params_estimate, "
            "pr.novelty_confidence AS _novelty_confidence, "
            "pr.cka_source AS _cka_source, "
            "pr.routing_confidence_mean AS _routing_confidence_mean "
            "FROM leaderboard l "
            "LEFT JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE 1=1"
        )
        params: List[Any] = []
        if tier:
            query += " AND l.tier = ?"
            params.append(tier)
        oversample = max(limit * 6, 200)
        query += f" ORDER BY l.{sort_by} DESC NULLS LAST LIMIT ?"
        params.append(oversample)

        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if include_family:
                d["architecture_family"] = self._classify_architecture_family(
                    graph_json=d.get("_graph_json"),
                    routing_mode=d.get("_routing_mode"),
                )
            d.pop("_graph_json", None)
            d.pop("_routing_mode", None)
            d["arch_spec_json"] = d.pop("_arch_spec_json", None)
            d["param_count"] = d.pop("_param_count", None)
            d["graph_n_params_estimate"] = d.pop("_graph_n_params_estimate", None)
            d["novelty_confidence"] = d.pop("_novelty_confidence", None)
            d["cka_source"] = d.pop("_cka_source", None)
            d["routing_confidence_mean"] = d.pop("_routing_confidence_mean", None)
            
            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = json.loads(
                        d["investigation_best_training"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)

        # Deduplicate by graph fingerprint: keep best composite_score per arch
        seen_fingerprints: Dict[str, int] = {}
        deduped = []
        for entry in results:
            fp = entry.get("_graph_fingerprint")
            if fp:
                if fp in seen_fingerprints:
                    # Keep the one with higher composite_score
                    existing_idx = seen_fingerprints[fp]
                    existing_score = deduped[existing_idx].get("composite_score") or 0
                    new_score = entry.get("composite_score") or 0
                    if new_score > existing_score:
                        deduped[existing_idx] = entry
                    continue
                seen_fingerprints[fp] = len(deduped)
            deduped.append(entry)

        # Clean up internal field
        for entry in deduped:
            entry.pop("_graph_fingerprint", None)

        return deduped[:limit]

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

    @staticmethod
    def _classify_architecture_family(
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

    def promote_to_tier(self, entry_id: str, tier: str,
                        **kwargs) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        sets = ["tier = ?"]
        params: List[Any] = [tier]

        for col in ("investigation_loss_ratio", "investigation_robustness",
                     "investigation_best_training", "investigation_passed",
                     "validation_loss_ratio", "validation_baseline_ratio",
                     "validation_multi_seed_std", "validation_passed",
                     "notes"):
            if col in kwargs:
                sets.append(f"{col} = ?")
                val = kwargs[col]
                if isinstance(val, bool):
                    val = int(val)
                params.append(val)

        # Recompute composite score
        row = self.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row:
            d = dict(row)
            d.update(kwargs)
            # Look up novelty_confidence from linked program_results
            nov_conf = None
            if d.get("result_id"):
                pr = self.conn.execute(
                    "SELECT novelty_confidence FROM program_results WHERE result_id = ?",
                    (d["result_id"],),
                ).fetchone()
                if pr:
                    nov_conf = pr["novelty_confidence"]
            composite = self.compute_composite_score(
                screening_lr=d.get("screening_loss_ratio"),
                screening_nov=d.get("screening_novelty"),
                inv_lr=d.get("investigation_loss_ratio"),
                inv_robust=d.get("investigation_robustness"),
                val_lr=d.get("validation_loss_ratio"),
                val_baseline=d.get("validation_baseline_ratio"),
                val_std=d.get("validation_multi_seed_std"),
                novelty_confidence=nov_conf,
            )
            sets.append("composite_score = ?")
            params.append(composite)

        sets.append("timestamp = ?")
        params.append(time.time())
        params.append(entry_id)

        self.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
            params,
        )
        self._maybe_commit()

    # ── Campaigns ──

    def create_campaign(self, title: str, objective: str,
                        success_criteria: str,
                        parent_id: Optional[str] = None) -> str:
        """Create a new research campaign. Returns campaign_id."""
        campaign_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO campaigns
            (campaign_id, timestamp, title, objective, success_criteria,
             status, parent_campaign_id, started_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
            (campaign_id, now, title, objective, success_criteria,
             parent_id, now),
        )
        self._maybe_commit()
        return campaign_id

    def get_campaign(self, campaign_id: str) -> Optional[Dict]:
        """Get a campaign by ID."""
        row = self.conn.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_active_campaigns(self) -> List[Dict]:
        """Get all active campaigns."""
        rows = self.conn.execute(
            "SELECT * FROM campaigns WHERE status = 'active' ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_campaign(self, campaign_id: str, **kwargs) -> None:
        """Update campaign fields."""
        allowed = {"title", "objective", "success_criteria", "status",
                    "findings_summary", "completed_at",
                    "completion_reason", "successor_campaign_id"}
        sets = []
        params: List[Any] = []
        for k, v in kwargs.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                params.append(v)
        if not sets:
            return
        params.append(campaign_id)
        self.conn.execute(
            f"UPDATE campaigns SET {', '.join(sets)} WHERE campaign_id = ?",
            params,
        )
        self._maybe_commit()

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

    def get_campaign_hypotheses(self, campaign_id: str) -> List[Dict]:
        """Get all hypotheses for a campaign."""
        rows = self.conn.execute(
            """SELECT * FROM hypotheses WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        ).fetchall()
        hypotheses = []
        for row in rows:
            hypothesis = dict(row)
            raw_meta = hypothesis.get("metadata_json")
            if isinstance(raw_meta, str) and raw_meta.strip():
                try:
                    parsed = json.loads(raw_meta)
                    hypothesis["metadata"] = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    hypothesis["metadata"] = {}
            else:
                hypothesis["metadata"] = {}
            hypotheses.append(hypothesis)
        return hypotheses

    def get_campaign_decisions(self, campaign_id: str) -> List[Dict]:
        """Get all decisions for a campaign."""
        rows = self.conn.execute(
            """SELECT * FROM decisions WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def evaluate_campaign_criteria(self, campaign_id: str) -> Dict:
        """Evaluate campaign success criteria against measured data.

        Returns {
            all_met: bool,        # True if every parseable criterion passes
            n_criteria: int,
            n_passing: int,
            n_at_risk: int,
            n_not_yet: int,
            stale: bool,          # True if 10+ experiments with no progress
            tracker: List[Dict],  # per-criterion status from analytics
        }
        """
        from .analytics import ExperimentAnalytics

        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {"all_met": False, "n_criteria": 0, "n_passing": 0,
                    "n_at_risk": 0, "n_not_yet": 0, "stale": False,
                    "tracker": []}

        experiments = self.get_campaign_experiments(campaign_id)
        hypotheses = self.get_campaign_hypotheses(campaign_id)
        decisions = self.get_campaign_decisions(campaign_id)

        analytics = ExperimentAnalytics(self)
        tracker = analytics.campaign_success_criteria_tracker(
            campaign, experiments, hypotheses, decisions,
        )

        n_passing = sum(1 for t in tracker if t.get("status") == "pass")
        n_at_risk = sum(1 for t in tracker if t.get("status") == "at_risk")
        n_not_yet = sum(1 for t in tracker if t.get("status") == "not_yet")
        n_criteria = len(tracker)

        # All parseable criteria must pass (ignore unknown/not_yet-only)
        all_met = n_criteria > 0 and n_passing == n_criteria

        # Stale: 10+ experiments but zero criteria passing
        stale = len(experiments) >= 10 and n_passing == 0 and n_at_risk > 0

        return {
            "all_met": all_met,
            "n_criteria": n_criteria,
            "n_passing": n_passing,
            "n_at_risk": n_at_risk,
            "n_not_yet": n_not_yet,
            "stale": stale,
            "tracker": tracker,
        }

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
        query += " ORDER BY confidence DESC, times_validated DESC"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
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
               ORDER BY confidence DESC""",
            (pattern, pattern),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

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

    # ── Aria Chat Persistence ──────────────────────────────────────

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        label: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Persist a single chat message and return its message_id."""
        mid = message_id or str(uuid.uuid4())
        self.conn.execute(
            """INSERT OR REPLACE INTO aria_chat
               (message_id, session_id, timestamp, role, text, label, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mid, session_id, time.time(), role, text, label,
             json.dumps(metadata) if metadata else None),
        )
        self._maybe_commit()
        return mid

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
        include_compacted: bool = False,
    ) -> List[Dict]:
        """Return chat messages for a session, newest last."""
        if include_compacted:
            rows = self.conn.execute(
                """SELECT * FROM aria_chat
                   WHERE session_id = ?
                   ORDER BY timestamp ASC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM aria_chat
                   WHERE session_id = ? AND compacted = 0
                   ORDER BY timestamp ASC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_messages_compacted(
        self, message_ids: List[str], summary_message_id: str
    ) -> None:
        """Mark messages as compacted (replaced by a summary)."""
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self.conn.execute(
            f"UPDATE aria_chat SET compacted = 1 WHERE message_id IN ({placeholders})",
            message_ids,
        )
        # Update the summary message to reference what it summarizes
        self.conn.execute(
            "UPDATE aria_chat SET summary_of = ? WHERE message_id = ?",
            (json.dumps(message_ids), summary_message_id),
        )
        self._maybe_commit()

    def compact_old_chat(self, max_chars: int = 160) -> int:
        """Truncate and purge old chat messages to keep the DB lean.

        - Truncates all compacted messages to max_chars
        - Deletes compacted messages older than 7 days
        Returns number of rows affected.
        """
        cutoff = time.time() - 7 * 86400
        # Delete old compacted messages
        n1 = self.conn.execute(
            "DELETE FROM aria_chat WHERE compacted = 1 AND timestamp < ?",
            (cutoff,),
        ).rowcount
        # Truncate long messages that are already compacted
        self.conn.execute(
            """UPDATE aria_chat SET text = SUBSTR(text, 1, ?) || '...'
               WHERE compacted = 1 AND LENGTH(text) > ?""",
            (max_chars, max_chars),
        )
        self._maybe_commit()
        return n1
