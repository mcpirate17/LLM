"""
Experiment Database

SQLite database for tracking all architecture exploration experiments.
Stores specs, evaluation results, and lineage (parent→child mutations).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .morphological_box import ArchSpec
from .evaluator import Stage0Result, Stage1Result


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    spec_id TEXT PRIMARY KEY,
    short_name TEXT NOT NULL,
    choices TEXT NOT NULL,  -- JSON
    seed INTEGER NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT,
    created_at REAL NOT NULL,
    tags TEXT  -- JSON list of all tags
);

CREATE TABLE IF NOT EXISTS stage0_results (
    spec_id TEXT PRIMARY KEY REFERENCES experiments(spec_id),
    passed INTEGER NOT NULL,
    error TEXT,
    error_type TEXT,
    param_count INTEGER,
    forward_time_ms REAL,
    backward_time_ms REAL,
    peak_memory_mb REAL,
    output_shape TEXT,
    grad_norm REAL,
    has_nan_grad INTEGER,
    has_zero_grad INTEGER,
    evaluated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stage1_results (
    spec_id TEXT PRIMARY KEY REFERENCES experiments(spec_id),
    passed INTEGER NOT NULL,
    error TEXT,
    steps_completed INTEGER,
    initial_loss REAL,
    final_loss REAL,
    best_loss REAL,
    loss_ratio REAL,
    discovery_loss REAL,
    discovery_loss_ratio REAL,
    validation_loss REAL,
    validation_loss_ratio REAL,
    generalization_gap REAL,
    avg_step_time_ms REAL,
    throughput_tok_s REAL,
    peak_memory_mb REAL,
    loss_curve TEXT,  -- JSON
    avg_grad_norm REAL,
    max_grad_norm REAL,
    loss_decreasing INTEGER,
    loss_stable INTEGER,
    converges INTEGER,
    evaluated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_generation ON experiments(generation);
CREATE INDEX IF NOT EXISTS idx_experiments_parent ON experiments(parent_id);
CREATE INDEX IF NOT EXISTS idx_stage0_passed ON stage0_results(passed);
CREATE INDEX IF NOT EXISTS idx_stage1_passed ON stage1_results(passed);
CREATE INDEX IF NOT EXISTS idx_stage1_loss_ratio ON stage1_results(loss_ratio);
CREATE INDEX IF NOT EXISTS idx_stage1_passed_loss ON stage1_results(passed, loss_ratio DESC);
"""

# Cache keys affected by each table write
_CACHE_KEYS_BY_TABLE = {
    "experiments": ("count_experiments", "unevaluated_specs_0", "unevaluated_specs_1"),
    "stage0_results": (
        "count_experiments",
        "unevaluated_specs_0",
        "unevaluated_specs_1",
    ),
    "stage1_results": ("count_experiments", "unevaluated_specs_1"),
}


class ExperimentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

        self._batch_depth = 0  # >0 means inside a batch() context
        self._lock = threading.Lock()

        # In-memory cache for frequent queries (TTL + max size)
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key → (expires_at, value)
        self._cache_lock = threading.Lock()
        self._cache_ttl = 30.0  # seconds
        self._cache_max_size = 64

    def _cache_get(self, key: str) -> Tuple[bool, Any]:
        """Return (hit, value) from cache, respecting TTL."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None and time.time() < entry[0]:
                return True, entry[1]
            if entry is not None:
                del self._cache[key]
            return False, None

    def _cache_set(self, key: str, value: Any) -> None:
        """Store value in cache with TTL, evicting oldest if at max size."""
        with self._cache_lock:
            if len(self._cache) >= self._cache_max_size and key not in self._cache:
                # Evict the entry closest to expiry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time() + self._cache_ttl, value)

    def _invalidate_cache_for(self, table: str) -> None:
        """Invalidate only cache keys affected by writes to *table*."""
        keys = _CACHE_KEYS_BY_TABLE.get(table, ())
        if not keys:
            return
        with self._cache_lock:
            for key in keys:
                self._cache.pop(key, None)
            # Also invalidate any top_architectures_* keys for stage1 writes
            if table == "stage1_results":
                to_drop = [k for k in self._cache if k.startswith("top_architectures_")]
                for k in to_drop:
                    del self._cache[k]

    def _compress(self, data: Any) -> bytes:
        """JSON-encode and zlib-compress data."""
        return zlib.compress(json.dumps(data).encode("utf-8"))

    def _decompress(self, blob: bytes) -> Any:
        """Decompress zlib blob and JSON-decode."""
        if not blob:
            return None
        try:
            return json.loads(zlib.decompress(blob).decode("utf-8"))
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
            # Fallback for old uncompressed data
            if isinstance(blob, bytes):
                return json.loads(blob.decode("utf-8"))
            return json.loads(blob)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def batch(self):
        """Context manager to batch multiple writes into a single commit.

        Usage:
            with db.batch():
                db.save_spec(spec)
                db.save_stage0(result)
            # single commit happens here
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self.conn.commit()

    def flush(self) -> None:
        """Force-commit any pending writes."""
        self.conn.commit()

    def _maybe_commit(self):
        """Commit unless inside a batch() context."""
        if self._batch_depth == 0:
            self.conn.commit()

    # ── Write ──

    def save_spec(self, spec: ArchSpec):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO experiments
                (spec_id, short_name, choices, seed, generation, parent_id, created_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    spec.id,
                    spec.short_name,
                    self._compress(spec.choices),
                    spec.seed,
                    spec.generation,
                    spec.parent_id,
                    time.time(),
                    json.dumps(sorted(spec.all_tags())),
                ),
            )
            self._maybe_commit()
        self._invalidate_cache_for("experiments")

    def save_stage0(self, result: Stage0Result):
        d = result.to_dict()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO stage0_results
                (spec_id, passed, error, error_type, param_count, forward_time_ms,
                 backward_time_ms, peak_memory_mb, output_shape, grad_norm,
                 has_nan_grad, has_zero_grad, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d["spec_id"],
                    int(d["passed"]),
                    d["error"],
                    d["error_type"],
                    d["param_count"],
                    d["forward_time_ms"],
                    d["backward_time_ms"],
                    d["peak_memory_mb"],
                    d["output_shape"],
                    d["grad_norm"],
                    int(d["has_nan_grad"]),
                    int(d["has_zero_grad"]),
                    time.time(),
                ),
            )
            self._maybe_commit()
        self._invalidate_cache_for("stage0_results")

    def save_stage1(self, result: Stage1Result):
        d = result.to_dict()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO stage1_results
                (spec_id, passed, error, steps_completed, initial_loss, final_loss,
                 best_loss, loss_ratio, discovery_loss, discovery_loss_ratio,
                 validation_loss, validation_loss_ratio, generalization_gap,
                 avg_step_time_ms, throughput_tok_s,
                 peak_memory_mb, loss_curve, avg_grad_norm, max_grad_norm,
                 loss_decreasing, loss_stable, converges, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d["spec_id"],
                    int(d["passed"]),
                    d["error"],
                    d["steps_completed"],
                    d["initial_loss"],
                    d["final_loss"],
                    d["best_loss"],
                    d["loss_ratio"],
                    d.get("discovery_loss"),
                    d.get("discovery_loss_ratio"),
                    d.get("validation_loss"),
                    d.get("validation_loss_ratio"),
                    d.get("generalization_gap"),
                    d["avg_step_time_ms"],
                    d["throughput_tok_s"],
                    d["peak_memory_mb"],
                    self._compress(d["loss_curve"]),
                    d["avg_grad_norm"],
                    d["max_grad_norm"],
                    int(d["loss_decreasing"]),
                    int(d["loss_stable"]),
                    int(d["converges"]),
                    time.time(),
                ),
            )
            self._maybe_commit()
        self._invalidate_cache_for("stage1_results")

    # ── Read ──

    def get_spec(self, spec_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE spec_id = ?", (spec_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["choices"] = self._decompress(d["choices"])
        d["tags"] = json.loads(d["tags"]) if d["tags"] else []
        return d

    def get_stage0(self, spec_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM stage0_results WHERE spec_id = ?", (spec_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_stage1(self, spec_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM stage1_results WHERE spec_id = ?", (spec_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["loss_curve"] = self._decompress(d["loss_curve"]) if d["loss_curve"] else []
        return d

    def has_stage0(self, spec_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM stage0_results WHERE spec_id = ?", (spec_id,)
        ).fetchone()
        return row is not None

    def has_stage1(self, spec_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM stage1_results WHERE spec_id = ?", (spec_id,)
        ).fetchone()
        return row is not None

    # ── Queries ──

    def count_experiments(self) -> Dict[str, int]:
        """Summary counts (single query)."""
        cache_key = "count_experiments"
        hit, val = self._cache_get(cache_key)
        if hit:
            return val

        row = self.conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM experiments) AS total_specs,
                (SELECT COUNT(*) FROM stage0_results) AS s0_total,
                (SELECT COUNT(*) FROM stage0_results WHERE passed=1) AS s0_pass,
                (SELECT COUNT(*) FROM stage1_results) AS s1_total,
                (SELECT COUNT(*) FROM stage1_results WHERE passed=1) AS s1_pass
        """).fetchone()
        res = {
            "total_specs": row[0],
            "stage0_evaluated": row[1],
            "stage0_passed": row[2],
            "stage1_evaluated": row[3],
            "stage1_passed": row[4],
        }
        self._cache_set(cache_key, res)
        return res

    def top_architectures(self, n: int = 10) -> List[Dict]:
        """Get the best performing architectures by loss ratio."""
        cache_key = f"top_architectures_{n}"
        hit, val = self._cache_get(cache_key)
        if hit:
            return val

        rows = self.conn.execute(
            """SELECT e.spec_id, e.short_name, e.choices, e.generation,
                      s1.loss_ratio, s1.final_loss, s1.throughput_tok_s,
                      s0.param_count, s0.peak_memory_mb
               FROM experiments e
               JOIN stage1_results s1 ON e.spec_id = s1.spec_id
               JOIN stage0_results s0 ON e.spec_id = s0.spec_id
               WHERE s1.passed = 1
               ORDER BY s1.loss_ratio ASC
               LIMIT ?""",
            (n,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["choices"] = self._decompress(d["choices"])
            results.append(d)

        self._cache_set(cache_key, results)
        return results

    def stage0_passed_ids(self) -> List[str]:
        """Get all spec IDs that passed Stage 0."""
        rows = self.conn.execute(
            "SELECT spec_id FROM stage0_results WHERE passed=1"
        ).fetchall()
        return [r[0] for r in rows]

    def stage0_failure_analysis(self) -> List[Dict]:
        """Analyze which choices cause Stage 0 failures."""
        rows = self.conn.execute(
            """SELECT e.choices, s0.error_type, COUNT(*) as cnt
               FROM experiments e
               JOIN stage0_results s0 ON e.spec_id = s0.spec_id
               WHERE s0.passed = 0
               GROUP BY s0.error_type
               ORDER BY cnt DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def choice_success_rates(self) -> Dict[str, Dict[str, float]]:
        """For each dimension+option, what fraction passed Stage 1?"""
        rows = self.conn.execute(
            """SELECT e.choices, s1.passed
               FROM experiments e
               JOIN stage1_results s1 ON e.spec_id = s1.spec_id"""
        ).fetchall()

        counts: Dict[str, Dict[str, List[int]]] = {}
        for row in rows:
            choices = self._decompress(row["choices"])
            passed = row["passed"]
            for dim_name, opt_name in choices.items():
                if dim_name not in counts:
                    counts[dim_name] = {}
                if opt_name not in counts[dim_name]:
                    counts[dim_name][opt_name] = []
                counts[dim_name][opt_name].append(passed)

        rates = {}
        for dim_name, opts in counts.items():
            rates[dim_name] = {}
            for opt_name, results in opts.items():
                rates[dim_name][opt_name] = (
                    sum(results) / len(results) if results else 0.0
                )
        return rates

    def unevaluated_specs(self, stage: int = 0) -> List[str]:
        """Get spec IDs not yet evaluated at the given stage."""
        cache_key = f"unevaluated_specs_{stage}"
        hit, val = self._cache_get(cache_key)
        if hit:
            return val

        if stage == 0:
            rows = self.conn.execute(
                """SELECT e.spec_id FROM experiments e
                   LEFT JOIN stage0_results s0 ON e.spec_id = s0.spec_id
                   WHERE s0.spec_id IS NULL"""
            ).fetchall()
        elif stage == 1:
            rows = self.conn.execute(
                """SELECT e.spec_id FROM experiments e
                   JOIN stage0_results s0 ON e.spec_id = s0.spec_id
                   LEFT JOIN stage1_results s1 ON e.spec_id = s1.spec_id
                   WHERE s0.passed = 1 AND s1.spec_id IS NULL"""
            ).fetchall()
        else:
            return []

        res = [r[0] for r in rows]
        self._cache_set(cache_key, res)
        return res

    def reconstruct_spec(self, spec_id: str) -> Optional[ArchSpec]:
        """Reconstruct an ArchSpec from the database."""
        d = self.get_spec(spec_id)
        if d is None:
            return None
        return ArchSpec(
            choices=d["choices"],
            seed=d["seed"],
            generation=d["generation"],
            parent_id=d["parent_id"],
        )
