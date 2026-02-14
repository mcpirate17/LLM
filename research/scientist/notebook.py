"""
Electronic Lab Notebook

Persistent, structured record of all experiments, hypotheses,
observations, and conclusions. Stored as SQLite for queryability
and served to the React dashboard via API.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
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

    -- Fingerprint
    fingerprint_json TEXT,

    -- Training program used
    training_program_json TEXT
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

CREATE INDEX IF NOT EXISTS idx_entries_experiment ON entries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_programs_experiment ON program_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_programs_novelty ON program_results(novelty_score);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_log(metric_name);
CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
"""


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

    def __init__(self, db_path: str | Path = "research/lab_notebook.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(NOTEBOOK_SCHEMA)
        self.conn.commit()
        # Migrate: add llm_analysis column if missing (safe to repeat)
        try:
            self.conn.execute("SELECT llm_analysis FROM experiments LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN llm_analysis TEXT")
            self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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

        self.conn.execute(
            """INSERT INTO experiments
            (experiment_id, timestamp, experiment_type, status, hypothesis,
             research_question, config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
            (exp_id, now, experiment_type, hypothesis, research_question,
             json.dumps(config), now),
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

    def record_program_result(
        self,
        experiment_id: str,
        graph_fingerprint: str,
        graph_json: str,
        stage0_passed: bool = False,
        stage05_passed: bool = False,
        stage1_passed: bool = False,
        stage0_error: Optional[str] = None,
        param_count: int = 0,
        loss_ratio: Optional[float] = None,
        final_loss: Optional[float] = None,
        throughput: Optional[float] = None,
        novelty_score: Optional[float] = None,
        structural_novelty: Optional[float] = None,
        behavioral_novelty: Optional[float] = None,
        most_similar_to: Optional[str] = None,
        fingerprint_json: Optional[str] = None,
        training_program_json: Optional[str] = None,
    ) -> str:
        """Record results for a single synthesized program."""
        result_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO program_results
            (result_id, experiment_id, timestamp, graph_fingerprint, graph_json,
             stage0_passed, stage05_passed, stage1_passed, stage0_error,
             param_count, loss_ratio, final_loss, throughput_tok_s,
             novelty_score, structural_novelty, behavioral_novelty,
             most_similar_to, fingerprint_json, training_program_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result_id, experiment_id, time.time(), graph_fingerprint, graph_json,
             int(stage0_passed), int(stage05_passed), int(stage1_passed),
             stage0_error, param_count, loss_ratio, final_loss, throughput,
             novelty_score, structural_novelty, behavioral_novelty,
             most_similar_to, fingerprint_json, training_program_json),
        )
        self.conn.commit()
        return result_id

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
        for json_field in ("graph_json", "fingerprint_json", "training_program_json"):
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

        # Error type distribution
        errors: Dict[str, int] = {}
        for p in programs:
            err = p.get("stage0_error", "")
            if err:
                key = err[:80].strip()
                errors[key] = errors.get(key, 0) + 1

        # Stage-at-death histogram
        stage_deaths = {"validation": 0, "stage0": 0, "stage0.5": 0, "stage1": 0}
        for p in programs:
            if not p.get("stage0_passed"):
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

        return {
            "total_experiments": total_exp,
            "completed_experiments": completed,
            "total_programs_evaluated": total_programs,
            "stage1_survivors": stage1_survivors,
            "survival_rate": stage1_survivors / max(total_programs, 1),
            "avg_novelty_score": avg_novelty or 0,
            "top_novelty_score": top_novelty or 0,
            "active_insights": n_insights,
        }
