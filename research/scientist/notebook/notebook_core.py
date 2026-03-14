from __future__ import annotations
"""Auto-extracted mixin for LabNotebook."""

import json
import os
import queue
import sqlite3
import subprocess
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from ._shared import LOGGER, NOTEBOOK_SCHEMA, _PROGRAM_RESULTS_NEW_COLUMNS, infer_insight_identity


class _NotebookCore:
    """Core operations for the Lab Notebook."""
    """Electronic lab notebook for the AI scientist."""
    __slots__ = ()

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
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Enable WAL mode for high-concurrency performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._batch_depth = 0
        self._program_results_columns: Optional[set[str]] = None
        self._leaderboard_columns: Optional[set[str]] = None
        self._dashboard_summary_cache: Optional[Dict[str, Any]] = None
        self._dashboard_summary_cache_expires_at: float = 0.0
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
        writer_conn.execute("PRAGMA foreign_keys=ON")
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
        self._invalidate_dashboard_summary_cache()
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
        # Refresh the reader connection so subsequent reads observe the
        # writer thread's committed WAL snapshot immediately.
        self.conn.commit()


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
            CREATE INDEX IF NOT EXISTS idx_leaderboard_model_source ON leaderboard(model_source);
        """)
        self.conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_programs_stage1_passed ON program_results(stage1_passed);
            CREATE INDEX IF NOT EXISTS idx_programs_graph_fingerprint ON program_results(graph_fingerprint);
            CREATE INDEX IF NOT EXISTS idx_programs_routing_mode ON program_results(routing_mode);
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

        training_curves_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'training_curves'"
        ).fetchone()
        training_curves_sql = str(training_curves_row[0] or "") if training_curves_row else ""
        if "REFERENCES program_results" not in training_curves_sql:
            self.conn.execute(
                "DELETE FROM training_curves "
                "WHERE NOT EXISTS (SELECT 1 FROM program_results pr WHERE pr.result_id = training_curves.result_id)"
            )
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_curves_new (
                    result_id TEXT NOT NULL REFERENCES program_results(result_id) ON DELETE CASCADE,
                    step INTEGER NOT NULL,
                    loss REAL,
                    grad_norm REAL,
                    step_time_ms REAL,
                    PRIMARY KEY (result_id, step)
                )
            """)
            self.conn.execute("""
                INSERT OR REPLACE INTO training_curves_new (result_id, step, loss, grad_norm, step_time_ms)
                SELECT tc.result_id, tc.step, tc.loss, tc.grad_norm, tc.step_time_ms
                FROM training_curves tc
                JOIN program_results pr ON pr.result_id = tc.result_id
            """)
            self.conn.execute("DROP TABLE training_curves")
            self.conn.execute("ALTER TABLE training_curves_new RENAME TO training_curves")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_curves_result ON training_curves(result_id)"
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
            "wikitext_pre_perplexity REAL",
            "wikitext_ppl_improvement REAL",
            "screening_wikitext_status TEXT",
            "screening_wikitext_metric_version TEXT",
            "screening_wikitext_variant TEXT",
            "screening_wikitext_elapsed_ms REAL",
            "screening_wikitext_budget_json TEXT",
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
            # Real-token eval trajectory (action plan Phase 0)
            "robustness_grade TEXT",
            "evaluation_stage TEXT",
            "eval_budget_steps INTEGER",
            "capability_tier TEXT",
            "wikitext_ppl_improvement_ratio REAL",
            # Trajectory probe v2 metrics
            "peak_ppl REAL",
            "peak_step INTEGER",
            "steps_to_divergence INTEGER",
            "ppl_500 REAL",
            # Recipe re-roll tracking (Phase 5)
            "reinvestigation_count INTEGER DEFAULT 0",
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
            self._invalidate_dashboard_summary_cache()
            self.conn.commit()


    def _invalidate_dashboard_summary_cache(self) -> None:
        """Clear the short-lived dashboard summary cache after writes."""
        self._dashboard_summary_cache = None
        self._dashboard_summary_cache_expires_at = 0.0
