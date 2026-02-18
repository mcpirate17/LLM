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
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


NOTEBOOK_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    experiment_type TEXT NOT NULL,  -- 'synthesis', 'morphological', 'training', 'evolution'
    status TEXT NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'failed', 'aborted'

    -- Hypothesis
    hypothesis TEXT,
    research_question TEXT,

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

    -- Baseline comparison
    baseline_loss_ratio REAL
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
    outcome TEXT
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
CREATE INDEX IF NOT EXISTS idx_decisions_campaign ON decisions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
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
    "fp_interaction_locality": "REAL",
    "fp_interaction_sparsity": "REAL",
    "fp_interaction_symmetry": "REAL",
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
    # Novelty calibration
    "novelty_confidence": "REAL",
    # CKA provenance
    "cka_source": "TEXT",
    "cka_artifact_version": "TEXT",
    # Diagnostic tasks
    "diagnostic_tasks_json": "TEXT",
    "diagnostic_score": "REAL",
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

    def __init__(self, db_path: str | Path = "research/lab_notebook.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(NOTEBOOK_SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add any missing columns to existing databases."""
        # Migrate experiments table
        try:
            self.conn.execute("SELECT llm_analysis FROM experiments LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN llm_analysis TEXT")
            self.conn.commit()

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

        # Migrate leaderboard: add campaign_id if missing
        lb_cols = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
        }
        if "campaign_id" not in lb_cols:
            self.conn.execute(
                "ALTER TABLE leaderboard ADD COLUMN campaign_id TEXT"
            )

        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
        self.conn.commit()
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

    # ── Experiments ──

    def start_experiment(
        self,
        experiment_type: str,
        config: Dict,
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict] = None,
    ) -> str:
        """Start a new experiment. Returns experiment ID."""
        exp_id = str(uuid.uuid4())[:12]
        now = time.time()
        config_payload = dict(config)
        config_payload.setdefault("code_version", self._detect_code_version())

        self.conn.execute(
            """INSERT INTO experiments
            (experiment_id, timestamp, experiment_type, status, hypothesis,
             research_question, config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
            (exp_id, now, experiment_type, hypothesis, research_question,
             json.dumps(config_payload), now),
        )
        self.conn.commit()

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
    ):
        """Mark an experiment as completed with results."""
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
            (json.dumps(results),
             results.get("total", 0),
             results.get("stage0_passed", 0),
             results.get("stage05_passed", 0),
             results.get("stage1_passed", 0),
             results.get("best_loss_ratio"),
             results.get("best_novelty_score"),
             aria_summary, aria_mood,
             json.dumps(insights or []),
             llm_analysis,
             now, duration,
             experiment_id),
        )
        self.conn.commit()

    def fail_experiment(self, experiment_id: str, error: str):
        """Mark an experiment as failed."""
        self.conn.execute(
            """UPDATE experiments SET status = 'failed', completed_at = ?,
               aria_summary = ? WHERE experiment_id = ?""",
            (time.time(), f"FAILED: {error}", experiment_id),
        )
        self.conn.commit()

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
        self.conn.commit()
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
        self.conn.commit()
        return entry_id

    # ── Program Results ──

    def record_program_result(self, experiment_id: str,
                              graph_fingerprint: str, graph_json: str,
                              **kwargs) -> str:
        """Record results for a single synthesized program.

        Accepts all program_results columns as keyword arguments.
        Boolean fields (stage0_passed, etc.) are converted to int.
        """
        result_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Convert booleans to int for SQLite
        bool_fields = {
            "stage0_passed", "stage05_passed", "stage1_passed",
            "extreme_input_passed", "random_input_passed",
            "has_nan_output", "has_inf_output", "has_nan_grad", "has_zero_grad",
            "graph_has_gradient_path", "graph_uses_math_spaces",
            "graph_uses_frequency_domain",
        }
        for f in bool_fields:
            if f in kwargs and kwargs[f] is not None:
                kwargs[f] = int(kwargs[f])

        # Handle legacy 'throughput' -> 'throughput_tok_s' alias
        if "throughput" in kwargs:
            kwargs.setdefault("throughput_tok_s", kwargs.pop("throughput"))

        # Build column list dynamically from what's provided
        base_cols = ["result_id", "experiment_id", "timestamp",
                     "graph_fingerprint", "graph_json"]
        base_vals = [result_id, experiment_id, now,
                     graph_fingerprint, graph_json]

        extra_cols = []
        extra_vals = []
        for col, val in kwargs.items():
            extra_cols.append(col)
            extra_vals.append(val)

        all_cols = base_cols + extra_cols
        all_vals = base_vals + extra_vals
        placeholders = ", ".join(["?"] * len(all_cols))
        col_str = ", ".join(all_cols)

        self.conn.execute(
            f"INSERT INTO program_results ({col_str}) VALUES ({placeholders})",
            all_vals,
        )
        self.conn.commit()
        return result_id

    # ── Training Curves ──

    def store_training_curve(self, result_id: str,
                             curve: List[Dict]) -> None:
        """Store per-step training data.

        curve: list of dicts with keys step, loss, grad_norm, step_time_ms
        """
        if not curve:
            return
        self.conn.executemany(
            """INSERT OR REPLACE INTO training_curves
               (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            [(result_id, d.get("step", i), d.get("loss"),
              d.get("grad_norm"), d.get("step_time_ms"))
             for i, d in enumerate(curve)],
        )
        self.conn.commit()

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
        self.conn.commit()

    def get_op_success_rates(self) -> List[Dict]:
        """Get all op success rates."""
        rows = self.conn.execute(
            """SELECT * FROM op_success_rates
               ORDER BY n_stage1_passed DESC, n_used DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Learning Log ──

    def log_learning_event(self, event_type: str, description: str,
                           old_weights: Optional[Dict] = None,
                           new_weights: Optional[Dict] = None,
                           evidence: Optional[str] = None) -> None:
        """Log a grammar weight change or learning decision."""
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
        self.conn.commit()

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

    # ── Metrics ──

    def log_metric(self, metric_name: str, value: float,
                   experiment_id: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Log a time-series metric."""
        self.conn.execute(
            """INSERT INTO metrics_log
            (timestamp, experiment_id, metric_name, metric_value, metadata_json)
            VALUES (?, ?, ?, ?, ?)""",
            (time.time(), experiment_id, metric_name, value,
             json.dumps(metadata) if metadata else None),
        )
        self.conn.commit()

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
        self.conn.commit()
        return insight_id

    def supersede_insight(self, insight_id: str) -> None:
        """Mark an insight as superseded (replaced by a newer version)."""
        self.conn.execute(
            "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
            (insight_id,),
        )
        self.conn.commit()

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
            d["results"] = json.loads(d["results_json"])
        if d.get("insights_json"):
            d["insights"] = json.loads(d["insights_json"])
        return d

    def get_recent_experiments(self, n: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, status,
                      hypothesis, research_question,
                      n_programs_generated, n_stage0_passed, n_stage05_passed,
                      n_stage1_passed,
                      best_loss_ratio, best_novelty_score, aria_mood,
                      aria_summary, duration_seconds
               FROM experiments ORDER BY timestamp DESC LIMIT ?""",
            (n,)
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
        return [dict(r) for r in rows]

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
        d = dict(row)
        # Parse stored JSON fields
        for json_field in ("graph_json", "fingerprint_json",
                           "training_program_json", "graph_category_histogram"):
            val = d.get(json_field)
            if val and isinstance(val, str):
                try:
                    d[json_field + "_parsed"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

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
            """SELECT experiment_id, timestamp, experiment_type, config_json, n_programs_generated,
                      n_stage0_passed, n_stage05_passed, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, duration_seconds
               FROM experiments
               WHERE status = 'completed'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        trends = []
        total_programs = 0
        total_stage1 = 0
        for r in rows:
            d = dict(r)
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

        return trends

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

        self.conn.commit()
        return entry_id

    def get_leaderboard(self, tier: Optional[str] = None,
                        limit: int = 50,
                        sort_by: str = "composite_score") -> List[Dict]:
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
        query += f" ORDER BY l.{sort_by} DESC NULLS LAST LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
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

        return deduped

    def get_investigated_fingerprints(self) -> set:
        """Return fingerprints that have already been investigated or beyond."""
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM leaderboard l "
            "JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE l.tier IN ('investigation', 'validation', 'breakthrough')"
        ).fetchall()
        return {r[0] for r in rows if r[0]}

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
            "softmax_attention",
        }
        conv_ops = {"conv1d", "conv1d_seq", "depthwise_conv1d"}
        spectral_ops = {"sin", "cos", "fft", "ifft", "fourier_mix"}
        gating_ops = {"sigmoid", "tanh", "silu", "gelu", "maximum", "minimum", "swiglu"}
        mlp_ops = {"linear_proj", "linear_proj_up", "linear_proj_down", "learnable_bias"}

        has_attention = bool(ops & attention_ops)
        has_conv = bool(ops & conv_ops)
        has_spectral = bool(ops & spectral_ops)
        has_gating = bool(ops & gating_ops)
        has_mlp = bool(ops & mlp_ops)

        if has_attention:
            if has_conv or has_spectral or has_gating:
                return "Hybrid-Attention"
            return "Attention"
        if has_conv and has_spectral:
            return "Spectral-Conv"
        if has_spectral:
            return "Spectral-Mixer"
        if has_conv:
            return "Conv-Mixer"
        if has_gating and has_mlp:
            return "Gated-MLP"
        if has_mlp:
            return "MLP-Mixer"
        if has_gating:
            return "Nonlinear-Mixer"
        return "Hybrid-Mixer"

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
        self.conn.commit()

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
        self.conn.commit()
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
        self.conn.commit()

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
        self.conn.commit()
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
        self.conn.commit()

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
                        alternatives: Optional[List[Dict]] = None) -> str:
        """Record a go/no-go or other decision. Returns decision_id."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO decisions
            (decision_id, campaign_id, timestamp, decision_type,
             subject, rationale, evidence_ids, alternatives_considered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (decision_id, campaign_id, now, decision_type, subject,
             rationale,
             json.dumps(evidence_ids) if evidence_ids else None,
             json.dumps(alternatives) if alternatives else None),
        )
        self.conn.commit()
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
            results.append(d)
        return results

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
        self.conn.commit()
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
        self.conn.commit()

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
        self.conn.commit()
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
        self.conn.commit()
