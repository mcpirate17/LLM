"""
Electronic Lab Notebook

Persistent, structured record of all experiments, hypotheses,
observations, and conclusions. Stored as SQLite for queryability
and served to the React dashboard via API.
"""

from __future__ import annotations

import json
import logging
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
    status TEXT DEFAULT 'active'  -- 'active', 'confirmed', 'refuted', 'superseded'
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
    confidence_after REAL
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

    def cleanup_stale_experiments(self, timeout_minutes: int = 60) -> int:
        """Mark experiments stuck in 'running' status as failed.

        Returns the number of experiments cleaned up.
        """
        cutoff = time.time() - (timeout_minutes * 60)
        rows = self.conn.execute(
            "SELECT experiment_id FROM experiments "
            "WHERE status = 'running' AND started_at < ?",
            (cutoff,),
        ).fetchall()

        if not rows:
            return 0

        exp_ids = [r["experiment_id"] for r in rows]
        self.conn.executemany(
            "UPDATE experiments SET status = 'failed', "
            "results_json = json_set(COALESCE(results_json, '{}'), '$.failure_reason', ?) "
            "WHERE experiment_id = ?",
            [("Process terminated while running", eid) for eid in exp_ids],
        )
        self.conn.commit()
        return len(exp_ids)

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
        self.add_entry(ExperimentEntry(
            entry_type="hypothesis",
            title=f"Experiment {exp_id} started",
            content=f"Type: {experiment_type}\nHypothesis: {hypothesis or 'exploratory'}",
            experiment_id=exp_id,
            tags=["experiment_start"],
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
                      hypothesis, n_programs_generated, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, aria_mood,
                      duration_seconds
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
        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, n_programs_generated,
                      n_stage0_passed, n_stage05_passed, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, duration_seconds
               FROM experiments
               WHERE status = 'completed'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        trends = []
        for r in rows:
            d = dict(r)
            total = d.get("n_programs_generated") or 1
            d["s1_pass_rate"] = (d.get("n_stage1_passed") or 0) / total
            trends.append(d)
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

        query = "SELECT * FROM leaderboard WHERE 1=1"
        params: List[Any] = []
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        query += f" ORDER BY {sort_by} DESC NULLS LAST LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = json.loads(
                        d["investigation_best_training"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

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
                    "findings_summary", "completed_at"}
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
        return [dict(r) for r in rows]

    def get_campaign_decisions(self, campaign_id: str) -> List[Dict]:
        """Get all decisions for a campaign."""
        rows = self.conn.execute(
            """SELECT * FROM decisions WHERE campaign_id = ?
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
                          experiment_id: Optional[str] = None) -> str:
        """Record a structured hypothesis. Returns hypothesis_id."""
        hypothesis_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO hypotheses
            (hypothesis_id, campaign_id, experiment_id, timestamp,
             prediction, reasoning, test_method, success_metric,
             parent_hypothesis_id, status, confidence_before)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (hypothesis_id, campaign_id, experiment_id, now,
             prediction, reasoning, test_method, success_metric,
             parent_id, confidence),
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

    def get_hypothesis_chain(self, hypothesis_id: str) -> List[Dict]:
        """Trace lineage from root to all descendants."""
        # Find root
        current = hypothesis_id
        while True:
            row = self.conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            parent = row["parent_hypothesis_id"]
            if parent is None:
                break
            current = parent

        # BFS from root
        chain = []
        queue_ids = [current]
        seen = set()
        while queue_ids:
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
