"""
Experiment Analytics — Learning Feedback Engine

Analyzes experiment history to learn which operations, structures, and
combinations correlate with success. Feeds back into grammar weights
to improve synthesis over time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..eval.utils import safe_json_load, safe_parse_float
from ..synthesis.grammar import GrammarConfig
from ..synthesis.primitives import get_primitive
from .notebook import LabNotebook


class ExperimentAnalytics:
    """Data-driven analytics over experiment history."""

    LEARNING_TRAJECTORY_MIN_EXPERIMENTS = 5
    FINGERPRINT_WEIGHT_CAP = 3.0

    def __init__(self, notebook: LabNotebook):
        self.nb = notebook
        self._last_grammar_weight_diagnostics: Optional[Dict] = None

    _OP_NAME_PATTERN = re.compile(r'"op_name"\s*:\s*"([^"]+)"')
    _OP_KEY_PATTERN = re.compile(r'"op"\s*:\s*"([^"]+)"')

    _FULL_QKV_TOKEN_MIXERS: Set[str] = {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "random_feature_attention",
        "compressed_attention",
        "cross_attention_pool",
    }
    _Q_EQ_K_EQ_V_TOKEN_MIXERS: Set[str] = {
        "shared_qk_attention",
    }
    _QKV_FREE_TOKEN_MIXERS: Set[str] = {
        "conv_only",
        "state_space",
        "fourier_mixing",
        "differentiable_sort",
        "integral_kernel_mixing",
    }
    _COMPRESSION_FACTORS: Dict[str, float] = {
        "low_rank": 0.55,
        "shared_basis": 0.5,
        "hash_trick": 0.35,
        "structured_sparse": 0.4,
        "kronecker": 0.5,
        "polynomial": 0.6,
        "residual_quantized": 0.3,
        "compressed_attention": 0.7,
        "bottleneck": 0.55,
        "grouped_linear": 0.3,
        "tied_proj": 0.3,
    }

    # Map graph op names to compression mechanism labels
    _OP_TO_COMPRESSION: Dict[str, str] = {
        "low_rank_proj": "low_rank",
        "grouped_linear": "grouped_linear",
        "bottleneck_proj": "bottleneck",
        "shared_basis_proj": "shared_basis",
        "tied_proj": "tied_proj",
        "nm_sparse_linear": "structured_sparse",
        "block_sparse_linear": "structured_sparse",
        "semi_structured_2_4_linear": "structured_sparse",
    }

    @staticmethod
    def _as_float(value) -> Optional[float]:
        return safe_parse_float(value)

    @classmethod
    def _extract_ops_fast(cls, graph_json: str) -> Optional[List[str]]:
        """Fast-path op extraction from JSON string without full decode."""
        if not graph_json:
            return []
        if '"op_name"' not in graph_json and '"op"' not in graph_json:
            return []
        ops = set()
        ops.update(op for op in cls._OP_NAME_PATTERN.findall(graph_json) if op and op != "input")
        ops.update(op for op in cls._OP_KEY_PATTERN.findall(graph_json) if op and op != "input")
        return sorted(ops)

    @staticmethod
    def _extract_ops_fallback(graph_json: str) -> Optional[List[str]]:
        """Robust fallback extraction using JSON decode."""
        try:
            graph_data = json.loads(graph_json)
            nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
            return sorted({
                nd.get("op_name") or nd.get("op")
                for nd in nodes.values()
                if isinstance(nd, dict)
                and (nd.get("op_name") or nd.get("op"))
                and (nd.get("op_name") or nd.get("op")) != "input"
            })
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

    @staticmethod
    def _extract_arch_choices(arch_spec_json: Optional[str]) -> Dict[str, str]:
        if not arch_spec_json:
            return {}
        try:
            arch = json.loads(arch_spec_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(arch, dict):
            return {}
        choices = arch.get("choices", {})
        if not isinstance(choices, dict):
            return {}
        return choices

    def get_efficiency_frontier(self) -> List[Dict[str, Any]]:
        """
        Compute the Pareto-optimal efficiency frontier (Parameter Count vs Loss Ratio).
        Returns list of programs on the frontier.
        """
        rows = self.nb.conn.execute("""
            SELECT fingerprint, param_count, graph_n_params_estimate,
                   loss_ratio, validation_loss_ratio, graph_json
            FROM program_results
            WHERE (loss_ratio IS NOT NULL OR validation_loss_ratio IS NOT NULL)
              AND (param_count IS NOT NULL OR graph_n_params_estimate IS NOT NULL)
        """).fetchall()
        
        programs = []
        for r in rows:
            p = dict(r)
            # Use real params if available, else estimate
            p["params"] = p.get("param_count") or p.get("graph_n_params_estimate") or 1e9
            # Use validation loss if available, else standard loss
            p["loss"] = p.get("validation_loss_ratio") or p.get("loss_ratio") or 2.0
            programs.append(p)
            
        if not programs:
            return []
            
        # Simple Pareto Sort: O(N^2) but N is usually small (< 5000)
        frontier = []
        for i, p1 in enumerate(programs):
            is_dominated = False
            for j, p2 in enumerate(programs):
                if i == j: continue
                # p2 dominates p1 if it's better in both and strictly better in one
                if p2["params"] <= p1["params"] and p2["loss"] <= p1["loss"]:
                    if p2["params"] < p1["params"] or p2["loss"] < p1["loss"]:
                        is_dominated = True
                        break
            if not is_dominated:
                frontier.append(p1)
                
        return sorted(frontier, key=lambda x: x["params"])

    def qkv_usage_enum(self, program: Dict) -> str:
        """Classify token-mixing QKV usage for a program.

        Returns one of: ``full_qkv``, ``q_eq_k_eq_v``, ``qkv_free``.
        """
        if not isinstance(program, dict):
            return "qkv_free"

        choices = self._extract_arch_choices(program.get("arch_spec_json"))
        token_mixing = choices.get("token_mixing")

        if token_mixing in self._FULL_QKV_TOKEN_MIXERS:
            return "full_qkv"
        if token_mixing in self._Q_EQ_K_EQ_V_TOKEN_MIXERS:
            return "q_eq_k_eq_v"
        if token_mixing in self._QKV_FREE_TOKEN_MIXERS:
            return "qkv_free"

        graph_json = program.get("graph_json")
        ops: Set[str] = set()
        if isinstance(graph_json, str) and graph_json:
            fast_ops = self._extract_ops_fast(graph_json)
            if fast_ops is None:
                fast_ops = self._extract_ops_fallback(graph_json)
            ops = set(fast_ops or [])

        if ops & {
            "attention",
            "self_attention",
            "mha",
            "multihead_attention",
            "qkv_attention",
            "softmax_attention",
            "linear_attention",
            "random_feature_attention",
            "graph_attention",
            "compressed_attention",
            "cross_attention_pool",
        }:
            return "full_qkv"

        if ops & {
            "shared_qk_attention",
            "shared_qkv_attention",
        }:
            return "q_eq_k_eq_v"

        return "qkv_free"

    def _detect_compression_ops(self, program: Dict) -> List[str]:
        """Detect compression primitives used in a program's graph."""
        graph_json = program.get("graph_json")
        if not isinstance(graph_json, str) or not graph_json:
            return []
        ops = self._extract_ops_fast(graph_json)
        if ops is None:
            ops = self._extract_ops_fallback(graph_json) or []
        return [op for op in ops if op in self._OP_TO_COMPRESSION]

    def canonical_compression_metrics(self, program: Dict) -> Dict:
        """Compute canonical compression metrics for API payloads."""
        choices = self._extract_arch_choices(program.get("arch_spec_json"))
        mechanism = (
            choices.get("weight_storage")
            or choices.get("token_representation")
            or (
                choices.get("token_mixing")
                if choices.get("token_mixing") == "compressed_attention"
                else None
            )
        )
        # If arch_spec didn't indicate compression, check actual graph ops
        compression_ops = self._detect_compression_ops(program)
        if not mechanism or mechanism == "dense":
            if compression_ops:
                # Use the first compression op's mechanism label
                mechanism = self._OP_TO_COMPRESSION.get(compression_ops[0], "dense")
            else:
                mechanism = mechanism or "dense"
        factor = self._COMPRESSION_FACTORS.get(mechanism, 1.0)

        raw_params = self._as_float(
            program.get("param_count")
            if program.get("param_count") is not None
            else program.get("graph_n_params_estimate")
        )
        compressed_params = (
            int(max(1.0, round(raw_params * factor)))
            if raw_params is not None and raw_params > 0
            else None
        )
        compression_ratio = (
            max(0.01, min(1.0, compressed_params / raw_params))
            if raw_params and compressed_params is not None
            else None
        )

        baseline_ratio = self._as_float(
            program.get("validation_baseline_ratio")
            if program.get("validation_baseline_ratio") is not None
            else program.get("baseline_loss_ratio")
        )
        validation_loss = self._as_float(
            program.get("validation_loss_ratio")
            if program.get("validation_loss_ratio") is not None
            else program.get("loss_ratio")
        )
        investigation_loss = self._as_float(program.get("investigation_loss_ratio"))
        screening_loss = self._as_float(program.get("screening_loss_ratio"))

        if baseline_ratio is not None:
            quality_retention = max(0.0, min(1.0, 1.25 - baseline_ratio))
        elif validation_loss is not None:
            quality_retention = max(0.0, min(1.0, 1.0 - validation_loss))
        elif investigation_loss is not None:
            quality_retention = max(0.0, min(1.0, 1.1 - investigation_loss))
        elif screening_loss is not None:
            quality_retention = max(0.0, min(1.0, 1.0 - screening_loss))
        else:
            quality_retention = None

        compressed_memory_mb = (
            (compressed_params * 4) / (1024 * 1024)
            if compressed_params is not None
            else None
        )
        dense_memory_mb = (
            (raw_params * 4) / (1024 * 1024)
            if raw_params is not None
            else None
        )

        return {
            "compression_mechanism": mechanism,
            "compression_factor": factor,
            "raw_param_count": int(raw_params) if raw_params is not None else None,
            "compressed_param_estimate": compressed_params,
            "compression_ratio": compression_ratio,
            "estimated_memory_mb": compressed_memory_mb,
            "dense_estimated_memory_mb": dense_memory_mb,
            "quality_retention_score": quality_retention,
            "compression_ops": compression_ops,
        }

    def compression_primitive_effectiveness(self) -> Dict:
        """Compute per-primitive compression effectiveness from experiment history.

        Returns success rates and parameter efficiency for each compression
        primitive observed in program graphs.
        """
        rows = self.nb.conn.execute("""
            SELECT graph_json, stage1_passed, loss_ratio, validation_loss_ratio,
                   param_count, graph_n_params_estimate
            FROM program_results
            WHERE graph_json IS NOT NULL
        """).fetchall()

        per_op: Dict[str, Dict] = {}
        for row in rows:
            record = dict(row)
            ops = self._detect_compression_ops(record)
            if not ops:
                continue
            passed = bool(record.get("stage1_passed"))
            # Prefer validation_loss_ratio if available
            loss = self._as_float(
                record.get("validation_loss_ratio")
                if record.get("validation_loss_ratio") is not None
                else record.get("loss_ratio")
            )
            params = self._as_float(
                record.get("param_count")
                if record.get("param_count") is not None
                else record.get("graph_n_params_estimate")
            )
            for op_name in ops:
                bucket = per_op.setdefault(op_name, {
                    "op_name": op_name,
                    "mechanism": self._OP_TO_COMPRESSION.get(op_name, "unknown"),
                    "n_tested": 0,
                    "n_survived": 0,
                    "sum_loss": 0.0,
                    "n_loss": 0,
                    "best_loss": None,
                    "sum_params": 0.0,
                    "n_params": 0,
                })
                bucket["n_tested"] += 1
                if passed:
                    bucket["n_survived"] += 1
                if loss is not None:
                    bucket["sum_loss"] += loss
                    bucket["n_loss"] += 1
                    if bucket["best_loss"] is None or loss < bucket["best_loss"]:
                        bucket["best_loss"] = loss
                if params is not None:
                    bucket["sum_params"] += params
                    bucket["n_params"] += 1

        primitives = []
        for op_name, bucket in sorted(
            per_op.items(), key=lambda kv: kv[1]["n_tested"], reverse=True
        ):
            n = bucket["n_tested"]
            primitives.append({
                "op_name": op_name,
                "mechanism": bucket["mechanism"],
                "n_tested": n,
                "n_survived": bucket["n_survived"],
                "survival_rate": round(bucket["n_survived"] / n, 4) if n > 0 else 0.0,
                "avg_loss_ratio": (
                    round(bucket["sum_loss"] / bucket["n_loss"], 4)
                    if bucket["n_loss"] > 0 else None
                ),
                "best_loss_ratio": (
                    round(bucket["best_loss"], 4)
                    if bucket["best_loss"] is not None else None
                ),
                "avg_param_count": (
                    int(bucket["sum_params"] / bucket["n_params"])
                    if bucket["n_params"] > 0 else None
                ),
            })
        return {"primitives": primitives, "n_programs_with_compression": sum(1 for _ in primitives)}

    def compression_coverage(self) -> Dict:
        """Summarize compression-technique coverage across tested and surviving programs."""
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, arch_spec_json, loss_ratio, baseline_loss_ratio,
                   param_count, graph_n_params_estimate
            FROM program_results
            WHERE arch_spec_json IS NOT NULL OR loss_ratio IS NOT NULL
        """).fetchall()

        aggregates: Dict[str, Dict] = {}
        total_tested = 0
        total_survived = 0
        compressed_tested = 0
        compressed_survived = 0

        dense_markers = {"dense", "dense_matrix", "standard_float"}

        for row in rows:
            record = dict(row)
            metrics = self.canonical_compression_metrics(record)
            mechanism = metrics.get("compression_mechanism") or "dense"
            bucket = aggregates.setdefault(
                mechanism,
                {
                    "technique": mechanism,
                    "n_tested": 0,
                    "n_survived": 0,
                    "sum_loss": 0.0,
                    "n_loss": 0,
                    "best_loss": None,
                    "sum_quality": 0.0,
                    "n_quality": 0,
                    "sum_ratio": 0.0,
                    "n_ratio": 0,
                    "sum_memory_mb": 0.0,
                    "n_memory": 0,
                },
            )

            bucket["n_tested"] += 1
            total_tested += 1
            if mechanism not in dense_markers:
                compressed_tested += 1

            stage1_passed = bool(record.get("stage1_passed"))
            if stage1_passed:
                bucket["n_survived"] += 1
                total_survived += 1
                if mechanism not in dense_markers:
                    compressed_survived += 1

            loss_ratio = self._as_float(record.get("loss_ratio"))
            if loss_ratio is not None:
                bucket["sum_loss"] += loss_ratio
                bucket["n_loss"] += 1
                if bucket["best_loss"] is None or loss_ratio < bucket["best_loss"]:
                    bucket["best_loss"] = loss_ratio

            quality_retention = self._as_float(metrics.get("quality_retention_score"))
            if quality_retention is not None:
                bucket["sum_quality"] += quality_retention
                bucket["n_quality"] += 1

            ratio = self._as_float(metrics.get("compression_ratio"))
            if ratio is not None:
                bucket["sum_ratio"] += ratio
                bucket["n_ratio"] += 1

            memory_mb = self._as_float(metrics.get("estimated_memory_mb"))
            if memory_mb is not None:
                bucket["sum_memory_mb"] += memory_mb
                bucket["n_memory"] += 1

        techniques = []
        for mechanism, bucket in sorted(
            aggregates.items(), key=lambda item: item[1]["n_tested"], reverse=True
        ):
            n_tested = bucket["n_tested"]
            n_survived = bucket["n_survived"]
            techniques.append({
                "technique": mechanism,
                "n_tested": n_tested,
                "n_survived": n_survived,
                "survival_rate": round(n_survived / n_tested, 4) if n_tested > 0 else 0.0,
                "tested_share": round(n_tested / total_tested, 4) if total_tested > 0 else 0.0,
                "survivor_share": round(n_survived / total_survived, 4) if total_survived > 0 else 0.0,
                "avg_loss_ratio": round(bucket["sum_loss"] / bucket["n_loss"], 4) if bucket["n_loss"] > 0 else None,
                "best_loss_ratio": round(bucket["best_loss"], 4) if bucket["best_loss"] is not None else None,
                "avg_quality_retention": round(bucket["sum_quality"] / bucket["n_quality"], 4) if bucket["n_quality"] > 0 else None,
                "avg_compression_ratio": round(bucket["sum_ratio"] / bucket["n_ratio"], 4) if bucket["n_ratio"] > 0 else None,
                "avg_estimated_memory_mb": round(bucket["sum_memory_mb"] / bucket["n_memory"], 4) if bucket["n_memory"] > 0 else None,
            })

        return {
            "techniques": techniques,
            "totals": {
                "n_tested": total_tested,
                "n_survived": total_survived,
                "n_compressed_tested": compressed_tested,
                "n_compressed_survived": compressed_survived,
            },
        }

    def sparse_coverage(self) -> Dict:
        """Summarize sparsity exploration coverage across tested programs.

        Returns counts of programs using sparse ops, sparse weight storage,
        RigL training, or one-shot pruning baselines, plus survival rates
        and average density.
        """
        rows = self.nb.conn.execute("""
            SELECT COUNT(*) AS n_total,
                   SUM(CASE WHEN sparse_density_mean IS NOT NULL
                            OR graph_json LIKE '%sparse%'
                            OR graph_json LIKE '%block_sparse%'
                       THEN 1 ELSE 0 END) AS n_sparse,
                   SUM(CASE WHEN (sparse_density_mean IS NOT NULL
                            OR graph_json LIKE '%sparse%'
                            OR graph_json LIKE '%block_sparse%')
                            AND stage1_passed = 1
                       THEN 1 ELSE 0 END) AS n_sparse_survived,
                   AVG(CASE WHEN sparse_density_mean IS NOT NULL
                       THEN sparse_density_mean END) AS avg_density,
                   SUM(CASE WHEN pruning_method IS NOT NULL
                       THEN 1 ELSE 0 END) AS n_pruning,
                   SUM(CASE WHEN stage1_passed = 1
                       THEN 1 ELSE 0 END) AS n_total_survived
            FROM program_results
            WHERE loss_ratio IS NOT NULL OR stage0_passed IS NOT NULL
        """).fetchone()

        if not rows or not rows[0]:
            return {}

        n_total = int(rows[0] or 0)
        n_sparse = int(rows[1] or 0)
        n_sparse_survived = int(rows[2] or 0)
        avg_density = float(rows[3]) if rows[3] is not None else None
        n_pruning = int(rows[4] or 0)
        n_total_survived = int(rows[5] or 0)

        # Count RigL runs (optimizer recipe stored in training_program_json)
        rigl_row = self.nb.conn.execute("""
            SELECT COUNT(*) FROM program_results
            WHERE training_program_json LIKE '%rigl%'
        """).fetchone()
        n_rigl = int(rigl_row[0]) if rigl_row else 0

        return {
            "n_total_tested": n_total,
            "n_sparse_tested": n_sparse,
            "n_sparse_survived": n_sparse_survived,
            "n_total_survived": n_total_survived,
            "avg_density": round(avg_density, 4) if avg_density is not None else None,
            "n_pruning_runs": n_pruning,
            "n_rigl_runs": n_rigl,
            "sparse_share": round(n_sparse / n_total, 4) if n_total > 0 else 0.0,
            "sparse_survival_rate": (
                round(n_sparse_survived / n_sparse, 4) if n_sparse > 0 else 0.0
            ),
        }

    def reproducibility_packet_status(self, program: Dict) -> Dict:
        """Evaluate reproducibility packet completeness for a program."""
        arch_choices = self._extract_arch_choices(program.get("arch_spec_json"))
        checks = [
            ("result_id", bool(program.get("result_id"))),
            ("graph_fingerprint", bool(program.get("graph_fingerprint"))),
            ("arch_spec", bool(arch_choices)),
            (
                "baseline_ratio",
                program.get("validation_baseline_ratio") is not None
                or program.get("baseline_loss_ratio") is not None,
            ),
            (
                "multi_seed_std",
                program.get("validation_multi_seed_std") is not None,
            ),
            ("cka_artifact", program.get("cka_source") == "artifact"),
        ]
        ready_count = sum(1 for _, ok in checks if ok)
        total_checks = len(checks)
        if ready_count == total_checks:
            status = "ready"
        elif ready_count >= 4:
            status = "partial"
        else:
            status = "sparse"
        return {
            "status": status,
            "ready_count": ready_count,
            "total_checks": total_checks,
            "missing": [name for name, ok in checks if not ok],
        }

    def op_success_rates(self) -> Dict[str, Dict]:
        """Get per-op success rates from the op_success_rates table."""
        rows = self.nb.get_op_success_rates()
        result = {}
        for row in rows:
            op = row["op_name"]
            n_used = row["n_used"] or 1
            result[op] = {
                "n_used": n_used,
                "s0_rate": (row.get("n_stage0_passed") or 0) / n_used,
                "s05_rate": (row.get("n_stage05_passed") or 0) / n_used,
                "s1_rate": (row.get("n_stage1_passed") or 0) / n_used,
                "avg_loss_ratio": row.get("avg_loss_ratio"),
                "avg_novelty": row.get("avg_novelty"),
                "avg_novelty_confidence": row.get("avg_novelty_confidence"),
            }
        return result

    def gate_performance_summary(self) -> Dict:
        """Analyze Stage 0.5 (Causality Gate) vs Stage 1 (Micro-Corpus) efficiency.
        
        Tracks how many 'cheaters' (random-token hackers) are caught by the 
        causality gate and how well discovery loss predicts validation loss.
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, stage05_passed, stage1_passed, discovery_loss_ratio, 
                   validation_loss_ratio, error_type
            FROM program_results
            WHERE stage0_passed = 1
        """).fetchall()

        if not rows:
            return {}

        total = len(rows)
        s05_passed = sum(1 for r in rows if r["stage05_passed"])
        s05_failed = total - s05_passed
        
        causality_violations = sum(1 for r in rows if r["error_type"] == "causality_violation")
        
        # Correlation between discovery and validation
        discovery = []
        validation = []
        for r in rows:
            if r["discovery_loss_ratio"] is not None and r["validation_loss_ratio"] is not None:
                discovery.append(r["discovery_loss_ratio"])
                validation.append(r["validation_loss_ratio"])
        
        correlation = None
        if len(discovery) > 5:
            try:
                import numpy as np
                correlation = float(np.corrcoef(discovery, validation)[0, 1])
            except Exception:
                pass

        return {
            "total_screened": total,
            "stage05_pass_rate": round(s05_passed / total, 4) if total > 0 else 0.0,
            "causality_violations": causality_violations,
            "discovery_validation_correlation": round(correlation, 4) if correlation is not None else None,
            "n_correlation_samples": len(discovery)
        }

    def gate_health_daily(self, n_days: int = 14) -> Dict:
        """Daily breakdown of causality gate metrics for monitoring dashboards.

        Returns per-day stats: models screened, gate pass rate, causality
        violations, and discovery-vs-validation correlation.
        """
        import time as _time
        cutoff = _time.time() - (n_days * 86400)
        rows = self.nb.conn.execute("""
            SELECT result_id, stage05_passed, stage1_passed,
                   discovery_loss_ratio, validation_loss_ratio,
                   error_type, timestamp
            FROM program_results
            WHERE stage0_passed = 1 AND timestamp > ?
            ORDER BY timestamp
        """, (cutoff,)).fetchall()

        if not rows:
            return {"daily": [], "summary": self.gate_performance_summary()}

        from collections import defaultdict
        from datetime import datetime

        buckets: dict = defaultdict(list)
        for r in rows:
            day = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d")
            buckets[day].append(r)

        daily = []
        for day in sorted(buckets):
            day_rows = buckets[day]
            n = len(day_rows)
            passed = sum(1 for r in day_rows if r["stage05_passed"])
            violations = sum(1 for r in day_rows if r["error_type"] == "causality_violation")

            disc, val = [], []
            for r in day_rows:
                if r["discovery_loss_ratio"] is not None and r["validation_loss_ratio"] is not None:
                    disc.append(r["discovery_loss_ratio"])
                    val.append(r["validation_loss_ratio"])

            corr = None
            if len(disc) > 3:
                try:
                    import numpy as np
                    corr = round(float(np.corrcoef(disc, val)[0, 1]), 4)
                except Exception:
                    pass

            daily.append({
                "date": day,
                "models_screened": n,
                "gate_pass_rate": round(passed / n, 4) if n else 0.0,
                "causality_violations": violations,
                "gate_failure_rate": round((n - passed) / n, 4) if n else 0.0,
                "discovery_validation_correlation": corr,
                "n_correlation_samples": len(disc),
            })

        return {"daily": daily, "summary": self.gate_performance_summary()}

    def structural_correlations(self) -> Dict[str, float]:
        """Analyze which graph properties correlate with Stage 1 success.

        Returns correlation-like scores for graph metrics vs success.
        Vectorized via NumPy for high-performance orchestration.
        """
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, graph_n_ops, graph_depth,
                   graph_n_params_estimate, graph_n_unique_ops,
                   graph_uses_math_spaces, graph_uses_frequency_domain,
                   graph_has_gradient_path
            FROM program_results
            WHERE graph_n_ops IS NOT NULL
        """).fetchall()

        if len(rows) < 10:
            return {}

        metrics = ["graph_n_ops", "graph_depth", "graph_n_params_estimate",
                    "graph_n_unique_ops", "graph_uses_math_spaces",
                    "graph_uses_frequency_domain", "graph_has_gradient_path"]

        data = np.array([[float(r[m] or 0) for m in metrics] for r in rows], dtype=np.float32)
        passed = np.array([bool(r["stage1_passed"]) for r in rows], dtype=bool)

        if not np.any(passed) or np.all(passed):
            return {m: 0.0 for m in metrics}

        success_data = data[passed]
        fail_data = data[~passed]

        avg_success = np.mean(success_data, axis=0)
        avg_fail = np.mean(fail_data, axis=0)
        std_all = np.std(data, axis=0)

        correlations = {}
        for i, m in enumerate(metrics):
            if std_all[i] > 1e-9:
                correlations[m] = float((avg_success[i] - avg_fail[i]) / std_all[i])
            else:
                correlations[m] = 0.0

        return correlations

    def _gather_category_stats(
        self, op_rates: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Group op success rates by category."""
        cat_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "s1_total": 0, "novelty_sum": 0.0, "count": 0,
            "conf_sum": 0.0, "conf_count": 0,
        })
        for op_name, stats in op_rates.items():
            try:
                op = get_primitive(op_name)
                cat = op.category.value
            except (KeyError, Exception):
                continue
            cat_stats[cat]["total"] += stats["n_used"]
            cat_stats[cat]["s1_total"] += int(stats["s1_rate"] * stats["n_used"])
            if stats.get("avg_novelty"):
                cat_stats[cat]["novelty_sum"] += stats["avg_novelty"] * stats["n_used"]
                cat_stats[cat]["count"] += stats["n_used"]
            if stats.get("avg_novelty_confidence"):
                cat_stats[cat]["conf_sum"] += stats["avg_novelty_confidence"] * stats["n_used"]
                cat_stats[cat]["conf_count"] += stats["n_used"]
        return cat_stats

    def _compute_weights_from_stats(
        self, cat_stats: Dict[str, Dict],
    ) -> Optional[Dict[str, float]]:
        """Compute grammar weights from per-category statistics."""
        default_weights = GrammarConfig().category_weights

        cat_s1_rates = {}
        cat_novelties = {}
        cat_confidences = {}
        for cat, stats in cat_stats.items():
            if stats["total"] < 2:
                continue
            cat_s1_rates[cat] = stats["s1_total"] / max(stats["total"], 1)
            cat_novelties[cat] = (stats["novelty_sum"] / stats["count"]
                                  if stats["count"] > 0 else 0.0)
            cat_confidences[cat] = (stats["conf_sum"] / stats["conf_count"]
                                    if stats["conf_count"] > 0 else 0.0)

        if not cat_s1_rates:
            return None

        mean_s1 = sum(cat_s1_rates.values()) / len(cat_s1_rates)

        learned = {}
        stability_multipliers = self.instability_attribution()
        
        for cat, s1_rate in cat_s1_rates.items():
            n = cat_stats[cat]["total"]
            relative = s1_rate / max(mean_s1, 0.01)

            # Statistical guard (#42): skip noisy differences
            se = math.sqrt(s1_rate * (1 - s1_rate) / n) if n > 0 and 0 < s1_rate < 1 else 0.0
            effect = abs(s1_rate - mean_s1)
            if se > 0 and effect < se:
                default = default_weights.get(cat, 1.0)
                tentative = default * (relative ** 2)
                learned[cat] = round(0.5 * tentative + 0.5 * default, 2)
                continue

            amplified = relative ** 2
            # Discount novelty factor by average confidence for this category
            # Low-confidence novelty (e.g. structural-only at 0.2) contributes
            # much less than high-confidence (full behavioral at 0.9)
            raw_novelty = cat_novelties.get(cat, 0.0)
            confidence = cat_confidences.get(cat, 0.0)
            novelty_factor = 1.0 + raw_novelty * confidence
            base = default_weights.get(cat, 1.0)
            
            # Apply stability multiplier
            stab = stability_multipliers.get(cat, 1.0)
            weight = base * amplified * novelty_factor * stab
            
            if cat == "frequency_domain":
                learned[cat] = round(max(0.0, min(0.1, weight)), 2)
            else:
                learned[cat] = round(max(0.5, min(8.0, weight)), 2)

        # EMA blending if last_applied exists (moved from compute_grammar_weights)
        # This function should just return the raw learned weights from stats.
        # Actually, the caller handles EMA blending.
        
        # Hard cap: frequency_domain shows strong negative correlation with S1 (-0.33).
        # Cap at 0.1 to suppress generation of frequency-domain ops.
        _SUPPRESSED_CATEGORIES = {"frequency_domain": 0.1}
        for cat, cap in _SUPPRESSED_CATEGORIES.items():
            if cat in learned:
                learned[cat] = min(learned[cat], cap)

        return learned if learned else None

    def _collect_fingerprint_capped_op_rates(
        self,
        per_fingerprint_cap: float,
    ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """Build op success rates with capped contribution per architecture fingerprint."""
        rows = self.nb.conn.execute(
            """SELECT result_id, graph_fingerprint, graph_json, stage1_passed,
                      novelty_score, novelty_confidence
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        ).fetchall()

        extracted_rows: List[Dict] = []
        fingerprint_counts: Dict[str, int] = defaultdict(int)
        
        # Z13: Identify Pareto winners for weighting boost
        pareto_ids = set(self.pareto_optimal_programs())
        
        for row in rows:
            graph_json = row["graph_json"]
            if not graph_json:
                continue

            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue

            graph_fingerprint = str(row["graph_fingerprint"] or "").strip()
            fp_key = graph_fingerprint or f"result:{row['result_id']}"
            fingerprint_counts[fp_key] += 1
            
            # Pareto boost: 5x weight for non-dominated models
            is_pareto = row["result_id"] in pareto_ids
            
            extracted_rows.append(
                {
                    "fingerprint": fp_key,
                    "ops": set(ops),
                    "stage1_passed": bool(row["stage1_passed"]),
                    "novelty_score": self._as_float(row["novelty_score"]),
                    "novelty_confidence": self._as_float(row["novelty_confidence"]),
                    "weight_multiplier": 5.0 if is_pareto else 1.0
                }
            )

        op_stats: Dict[str, Dict] = {}
        effective_rows = 0.0
        for extracted in extracted_rows:
            fp_key = extracted["fingerprint"]
            fp_count = max(fingerprint_counts.get(fp_key, 1), 1)
            row_weight = min(1.0, max(per_fingerprint_cap, 0.1) / float(fp_count))
            # Apply Pareto multiplier
            row_weight *= extracted["weight_multiplier"]
            
            effective_rows += row_weight

            for op_name in extracted["ops"]:
                stats = op_stats.setdefault(
                    op_name,
                    {
                        "n_used": 0.0,
                        "n_s1": 0.0,
                        "nov_sum": 0.0,
                        "nov_n": 0.0,
                        "conf_sum": 0.0,
                        "conf_n": 0.0,
                    },
                )
                stats["n_used"] += row_weight
                if extracted["stage1_passed"]:
                    stats["n_s1"] += row_weight
                novelty = extracted["novelty_score"]
                if novelty is not None:
                    stats["nov_sum"] += novelty * row_weight
                    stats["nov_n"] += row_weight
                confidence = extracted["novelty_confidence"]
                if confidence is not None:
                    stats["conf_sum"] += confidence * row_weight
                    stats["conf_n"] += row_weight

        op_rates: Dict[str, Dict] = {}
        for op_name, stats in op_stats.items():
            n_used = stats["n_used"]
            if n_used <= 0:
                continue
            op_rates[op_name] = {
                "n_used": n_used,
                "n_stage1_passed": stats["n_s1"],
                "s1_rate": stats["n_s1"] / n_used,
                "avg_novelty": (stats["nov_sum"] / stats["nov_n"]) if stats["nov_n"] > 0 else None,
                "avg_novelty_confidence": (stats["conf_sum"] / stats["conf_n"]) if stats["conf_n"] > 0 else None,
            }

        total_rows = len(extracted_rows)
        unique_fingerprints = len(fingerprint_counts)
        repeat_rows = sum(max(0, count - 1) for count in fingerprint_counts.values())
        top_fingerprint_count = max(fingerprint_counts.values()) if fingerprint_counts else 0
        diagnostics: Dict[str, float] = {
            "total_rows": float(total_rows),
            "effective_rows": float(round(effective_rows, 4)),
            "unique_fingerprints": float(unique_fingerprints),
            "repeat_rows": float(repeat_rows),
            "rerun_ratio": (repeat_rows / total_rows) if total_rows > 0 else 0.0,
            "top_fingerprint_concentration": (top_fingerprint_count / total_rows) if total_rows > 0 else 0.0,
            "fingerprint_cap": float(per_fingerprint_cap),
        }
        return op_rates, diagnostics

    @staticmethod
    def _wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score interval for Bernoulli proportion."""
        if total <= 0:
            return (0.0, 0.0)
        p = successes / total
        z2 = z * z
        denom = 1.0 + z2 / total
        center = (p + z2 / (2.0 * total)) / denom
        margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    @staticmethod
    def _two_prop_pvalue(successes_a: int, total_a: int,
                         successes_b: int, total_b: int) -> float:
        """Two-sided z-test p-value for proportion difference."""
        if total_a <= 0 or total_b <= 0:
            return 1.0
        p_a = successes_a / total_a
        p_b = successes_b / total_b
        pooled = (successes_a + successes_b) / max(total_a + total_b, 1)
        if pooled <= 0.0 or pooled >= 1.0:
            return 1.0
        se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b))
        if se <= 1e-12:
            return 1.0
        z = abs((p_a - p_b) / se)
        # 2 * (1 - Phi(|z|)) expressed via erfc for determinism without scipy
        return min(1.0, max(0.0, math.erfc(z / math.sqrt(2.0))))

    @staticmethod
    def _apply_fdr_bh(rows: List[Dict[str, Any]], p_key: str = "p_value",
                      q_key: str = "q_value") -> List[Dict[str, Any]]:
        """Apply Benjamini-Hochberg FDR correction in-place and return rows."""
        indexed = []
        for idx, row in enumerate(rows):
            p = row.get(p_key)
            if isinstance(p, (int, float)):
                indexed.append((idx, float(p)))
        m = len(indexed)
        if m == 0:
            return rows
        ranked = sorted(indexed, key=lambda t: t[1])
        q_vals = [1.0] * m
        prev = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            p = ranked[i][1]
            q = min(prev, p * m / rank)
            q_vals[i] = q
            prev = q
        for i, (orig_idx, _p) in enumerate(ranked):
            rows[orig_idx][q_key] = float(q_vals[i])
        for row in rows:
            row.setdefault(q_key, 1.0)
        return rows

    @staticmethod
    def _depth_bucket(depth: Optional[int]) -> str:
        if depth is None:
            return "unknown"
        d = int(depth)
        if d <= 3:
            return "shallow"
        if d <= 6:
            return "medium"
        return "deep"

    def pareto_optimal_programs(self) -> List[str]:
        """Find result_ids of non-dominated programs (Accuracy vs Efficiency).
        
        Uses NumPy for high-performance vectorized dominance checking.
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, loss_ratio, validation_loss_ratio, 
                   graph_n_params_estimate, param_count
            FROM program_results
            WHERE stage1_passed = 1
        """).fetchall()
        
        if not rows: return []
        
        # Criteria 1: Accuracy (1 - Loss Ratio). Higher is better.
        # Criteria 2: Efficiency (1 / Params). Higher is better.
        data = []
        ids = []
        for r in rows:
            lr = r["validation_loss_ratio"] if r["validation_loss_ratio"] is not None else r["loss_ratio"]
            params = r["param_count"] if r["param_count"] is not None else r["graph_n_params_estimate"]
            if lr is not None and params is not None:
                data.append([1.0 - lr, 1.0 / max(1, params)])
                ids.append(r["result_id"])
        
        if not data: return []
        
        costs = np.array(data, dtype=np.float32)
        n_points = costs.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            # A point is dominated if another point is >= in all criteria AND > in at least one
            # Here we simplify to: is there any point better than me in both?
            dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
            if np.any(dominated):
                # Wait, logic is: am I dominated? 
                # costs[j] dominates costs[i] if costs[j] is better.
                pass
        
        # Proper O(N^2) vectorized Pareto (fine for N < 1000)
        is_pareto = np.ones(n_points, dtype=bool)
        for i, c in enumerate(costs):
            if is_pareto[i]:
                # Keep points that are not dominated by any other point
                # j dominates i if costs[j] >= costs[i] and any(costs[j] > costs[i])
                is_pareto[i] = not np.any(np.all(costs >= c, axis=1) & np.any(costs > c, axis=1))
                
        return [ids[i] for i in range(n_points) if is_pareto[i]]

    def _load_program_factor_rows(self) -> List[Dict[str, Any]]:
        """Load per-program factors for attribution analysis."""
        rows = self.nb.conn.execute(
            """SELECT result_id, experiment_id, graph_json, stage1_passed,
                      graph_depth, graph_uses_math_spaces
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        ).fetchall()
        parsed: List[Dict[str, Any]] = []
        for row in rows:
            graph_json = row["graph_json"]
            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue
            op_set = set(ops)
            families: Set[str] = set()
            for op_name in op_set:
                try:
                    families.add(get_primitive(op_name).category.value)
                except Exception:
                    continue
            parsed.append({
                "result_id": row["result_id"],
                "experiment_id": row["experiment_id"],
                "stage1_passed": int(bool(row["stage1_passed"])),
                "ops": op_set,
                "families": families,
                "math_space": bool(row["graph_uses_math_spaces"]),
                "depth_bucket": self._depth_bucket(row["graph_depth"]),
            })
        parsed.sort(key=lambda r: (str(r["experiment_id"]), str(r["result_id"])))
        return parsed

    def _factor_success_stats(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute per-factor success rates, uncertainty, and p-values."""
        total_n = len(rows)
        if total_n <= 1:
            return []
        total_s = sum(r["stage1_passed"] for r in rows)

        factors: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            for op_name in row["ops"]:
                factors[("op", op_name)].append(idx)
            for fam in row["families"]:
                factors[("family", fam)].append(idx)
            factors[("math_space", "enabled" if row["math_space"] else "disabled")].append(idx)
            factors[("depth_bucket", row["depth_bucket"])].append(idx)

        out: List[Dict[str, Any]] = []
        for (factor_type, factor_name), indices in sorted(factors.items()):
            with_n = len(indices)
            without_n = total_n - with_n
            if with_n <= 0 or without_n <= 0:
                continue
            with_s = sum(rows[i]["stage1_passed"] for i in indices)
            without_s = total_s - with_s
            with_rate = with_s / with_n
            without_rate = without_s / without_n
            ci_low, ci_high = self._wilson_interval(with_s, with_n)
            p_val = self._two_prop_pvalue(with_s, with_n, without_s, without_n)
            out.append({
                "factor_type": factor_type,
                "factor_name": factor_name,
                "n_with": with_n,
                "n_without": without_n,
                "success_with": with_s,
                "success_without": without_s,
                "rate_with": with_rate,
                "rate_without": without_rate,
                "delta_rate": with_rate - without_rate,
                "ci_with_low": ci_low,
                "ci_with_high": ci_high,
                "p_value": p_val,
            })
        self._apply_fdr_bh(out, p_key="p_value", q_key="q_value")
        out.sort(key=lambda r: (r["factor_type"], r["factor_name"]))
        return out

    def _matched_control_stats(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Matched-control comparisons for single-factor contrasts."""
        comparisons: List[Dict[str, Any]] = []
        if len(rows) < 4:
            return comparisons

        # Math-space effect, matched on depth bucket
        for bucket in sorted({r["depth_bucket"] for r in rows}):
            group_a = [r for r in rows if r["depth_bucket"] == bucket and r["math_space"]]
            group_b = [r for r in rows if r["depth_bucket"] == bucket and not r["math_space"]]
            if not group_a or not group_b:
                continue
            s_a = sum(r["stage1_passed"] for r in group_a)
            s_b = sum(r["stage1_passed"] for r in group_b)
            n_a = len(group_a)
            n_b = len(group_b)
            p_val = self._two_prop_pvalue(s_a, n_a, s_b, n_b)
            delta = (s_a / n_a) - (s_b / n_b)
            se = math.sqrt((s_a / n_a) * (1 - (s_a / n_a)) / n_a + (s_b / n_b) * (1 - (s_b / n_b)) / n_b)
            comparisons.append({
                "factor": "math_space",
                "match_on": f"depth_bucket={bucket}",
                "n_a": n_a,
                "n_b": n_b,
                "delta_rate": delta,
                "ci_low": delta - 1.96 * se,
                "ci_high": delta + 1.96 * se,
                "p_value": p_val,
            })

        # Op family effect, matched on depth bucket and math-space
        families = sorted({f for r in rows for f in r["families"]})
        strata_keys = sorted({(r["depth_bucket"], r["math_space"]) for r in rows})
        for family in families:
            total_a = total_b = succ_a = succ_b = 0
            for depth_bucket, math_space in strata_keys:
                stratum = [r for r in rows if r["depth_bucket"] == depth_bucket and r["math_space"] == math_space]
                if not stratum:
                    continue
                a = [r for r in stratum if family in r["families"]]
                b = [r for r in stratum if family not in r["families"]]
                if not a or not b:
                    continue
                total_a += len(a)
                total_b += len(b)
                succ_a += sum(r["stage1_passed"] for r in a)
                succ_b += sum(r["stage1_passed"] for r in b)
            if total_a <= 0 or total_b <= 0:
                continue
            rate_a = succ_a / total_a
            rate_b = succ_b / total_b
            delta = rate_a - rate_b
            p_val = self._two_prop_pvalue(succ_a, total_a, succ_b, total_b)
            se = math.sqrt(rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b)
            comparisons.append({
                "factor": f"family:{family}",
                "match_on": "depth_bucket,math_space",
                "n_a": total_a,
                "n_b": total_b,
                "delta_rate": delta,
                "ci_low": delta - 1.96 * se,
                "ci_high": delta + 1.96 * se,
                "p_value": p_val,
            })

        self._apply_fdr_bh(comparisons, p_key="p_value", q_key="q_value")
        comparisons.sort(key=lambda r: (r["factor"], r["match_on"]))
        return comparisons

    def grammar_weight_attribution_report(self) -> Dict[str, Any]:
        """Attribution report separating correlation from stronger evidence."""
        rows = self._load_program_factor_rows()
        factor_stats = self._factor_success_stats(rows)
        matched_controls = self._matched_control_stats(rows)

        def _is_interpretable_factor(signal: Dict[str, Any]) -> bool:
            factor_name = str(signal.get("factor_name") or "").strip().lower()
            return bool(factor_name and factor_name not in {"unknown", "none", "null", "nan"})

        strong_correlational = [
            s for s in factor_stats
            if s["n_with"] >= 20
            and s["delta_rate"] > 0.05
            and s.get("q_value", 1.0) <= 0.10
            and s["ci_with_low"] > s["rate_without"]
        ]
        strong_correlational_interpretable = [
            s for s in strong_correlational if _is_interpretable_factor(s)
        ]
        matched_positive = [
            m for m in matched_controls
            if m["n_a"] >= 12
            and m["n_b"] >= 12
            and m["delta_rate"] > 0.05
            and m.get("q_value", 1.0) <= 0.10
            and m["ci_low"] > 0.0
        ]

        correlational_ok = bool(strong_correlational_interpretable or matched_positive)
        top_signal = None
        if strong_correlational_interpretable:
            top_signal = sorted(
                strong_correlational_interpretable,
                key=lambda s: (s["q_value"], -s["delta_rate"], -s["n_with"]),
            )[0]
        elif matched_positive:
            top_signal = sorted(
                matched_positive,
                key=lambda m: (m["q_value"], -m["delta_rate"], -(m["n_a"] + m["n_b"])),
            )[0]

        uncertainty = {
            "n_programs": len(rows),
            "n_factors_tested": len(factor_stats),
            "n_matched_tests": len(matched_controls),
            "fdr_method": "benjamini_hochberg",
            "correlational_signal_count": len(strong_correlational),
            "interpretable_correlational_signal_count": len(strong_correlational_interpretable),
            "matched_signal_count": len(matched_positive),
        }

        return {
            "factors": factor_stats,
            "matched_controls": matched_controls,
            "strong_correlational_evidence": correlational_ok,
            "top_signal": top_signal,
            "uncertainty": uncertainty,
            "requires_ablation": correlational_ok,
        }

    def compute_grammar_weights(
        self,
        last_applied: Optional[Dict[str, float]] = None,
        alpha: float = 0.6,
    ) -> Optional[Dict[str, float]]:
        """Compute learned category weights from historical success data.

        Uses the aggregate op_success_rates table. For holdout validation
        of the learned weights, call ``holdout_validation()`` separately.

        Args:
            last_applied: Previous effective weights for EMA blending.
                When None (first run, tests), no blending is applied.
            alpha: EMA weight for new signal vs last_applied (default 0.6).

        Returns a dict of category -> weight, or None if insufficient data.
        """
        fingerprint_rates, fingerprint_diag = self._collect_fingerprint_capped_op_rates(
            per_fingerprint_cap=self.FINGERPRINT_WEIGHT_CAP,
        )
        weighting_mode = "fingerprint_capped"
        op_rates = fingerprint_rates
        if len(op_rates) < 5:
            op_rates = self.op_success_rates()
            weighting_mode = "legacy_op_aggregate"
        if len(op_rates) < 5:
            self._last_grammar_weight_diagnostics = {
                "mode": weighting_mode,
                "insufficient_op_coverage": True,
                "op_count": len(op_rates),
                **fingerprint_diag,
            }
            return None

        cat_stats = self._gather_category_stats(op_rates)
        if not cat_stats:
            self._last_grammar_weight_diagnostics = {
                "mode": weighting_mode,
                "insufficient_category_coverage": True,
                "op_count": len(op_rates),
                **fingerprint_diag,
            }
            return None

        learned = self._compute_weights_from_stats(cat_stats)
        
        # Z13: Search Health Guard (Step 3)
        # If discovery and validation are uncorrelated, we are likely 'reward hacking'.
        # In this case, we should revert towards uniform (default) weights to find 
        # a new region of the search space.
        gate_stats = self.gate_performance_summary()
        correlation = gate_stats.get("discovery_validation_correlation")
        n_samples = gate_stats.get("n_correlation_samples", 0)
        
        # If correlation is low (< 0.3) and we have enough data, dampen learned signal
        if learned and correlation is not None and n_samples > 10 and correlation < 0.3:
            logger.info(f"Low discovery-validation correlation detected ({correlation:.2f}); dampening learned weights to increase diversity.")
            for cat in learned:
                default = default_weights.get(cat, 1.0)
                # Blend 70% default, 30% learned
                learned[cat] = round(0.7 * default + 0.3 * learned[cat], 2)

        if learned and last_applied:
            for cat in learned:
                if cat in last_applied:
                    learned[cat] = round(
                        alpha * learned[cat] + (1 - alpha) * last_applied[cat], 2
                    )
        self._last_grammar_weight_diagnostics = {
            "mode": weighting_mode,
            "insufficient_op_coverage": False,
            "op_count": len(op_rates),
            "category_count": len(cat_stats),
            "used_fingerprint_capping": weighting_mode == "fingerprint_capped",
            **fingerprint_diag,
        }
        return learned

    def grammar_weight_learning_diagnostics(self) -> Dict:
        """Return diagnostics for grammar-weight learning robustness."""
        if self._last_grammar_weight_diagnostics is None:
            self.compute_grammar_weights()
        diagnostics = dict(self._last_grammar_weight_diagnostics or {})
        if "mode" not in diagnostics:
            diagnostics["mode"] = "unknown"
        if "used_fingerprint_capping" not in diagnostics:
            diagnostics["used_fingerprint_capping"] = False
        if "fingerprint_cap" not in diagnostics:
            diagnostics["fingerprint_cap"] = float(self.FINGERPRINT_WEIGHT_CAP)
        return diagnostics

    def grammar_weight_audit_info(self) -> Dict[str, Any]:
        """Return reproducible query info for grammar weight learning."""
        return {
            "query": (
                "SELECT result_id, graph_fingerprint, graph_json, stage1_passed, "
                "novelty_score, novelty_confidence FROM program_results "
                "WHERE graph_json IS NOT NULL"
            ),
            "params": [],
            "fingerprint_cap": float(self.FINGERPRINT_WEIGHT_CAP),
            "weighting_mode": "fingerprint_capped",
            "source": "program_results",
        }

    def holdout_validation(self, holdout_fraction: float = 0.2) -> Optional[Dict]:
        """Evaluate grammar quality on holdout experiments.

        Returns s1_rate and program count for holdout experiments.
        """
        experiments = self.nb.conn.execute(
            "SELECT experiment_id FROM experiments WHERE status = 'completed'"
        ).fetchall()

        if len(experiments) < 5:
            return None

        holdout_ids = []
        for row in experiments:
            eid = row["experiment_id"]
            h = int(hashlib.md5(eid.encode()).hexdigest()[:8], 16)
            if (h % 100) < int(holdout_fraction * 100):
                holdout_ids.append(eid)

        if not holdout_ids:
            return None

        placeholders = ",".join("?" * len(holdout_ids))
        row = self.nb.conn.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1_passed
            FROM program_results
            WHERE experiment_id IN ({placeholders})
        """, tuple(holdout_ids)).fetchone()

        total = row["total"] or 0
        s1 = row["s1_passed"] or 0
        return {
            "holdout_experiments": len(holdout_ids),
            "holdout_programs": total,
            "holdout_s1_passed": s1,
            "holdout_s1_rate": s1 / max(total, 1),
        }

    def instability_attribution(self) -> Dict[str, float]:
        """Correlate architectural categories with high Jacobian spectral norm.
        
        Returns a penalty multiplier [0.5, 1.0] for each category.
        Categories that frequently cause instability get lower multipliers.
        """
        rows = self.nb.conn.execute("""
            SELECT graph_json, fp_jacobian_spectral_norm
            FROM program_results
            WHERE fp_jacobian_spectral_norm IS NOT NULL
        """).fetchall()
        
        if len(rows) < 10: return {}
        
        cat_norms = defaultdict(list)
        for r in rows:
            ops = self._extract_ops_fast(r["graph_json"])
            norm = float(r["fp_jacobian_spectral_norm"])
            if not ops: continue
            
            seen_cats = set()
            for op in ops:
                try:
                    cat = get_primitive(op).category.value
                    if cat not in seen_cats:
                        cat_norms[cat].append(norm)
                        seen_cats.add(cat)
                except Exception: continue
        
        penalties = {}
        for cat, norms in cat_norms.items():
            if len(norms) < 5: continue
            avg_norm = np.mean(norms)
            # Threshold: > 15 is risky, > 50 is toxic
            if avg_norm > 15.0:
                # Scale penalty from 1.0 down to 0.5
                penalty = max(0.5, 1.0 - (avg_norm - 15.0) / 70.0)
                penalties[cat] = round(float(penalty), 2)
            else:
                penalties[cat] = 1.0
                
        return penalties

    def experiment_clusters(self, n_clusters: int = 3) -> Optional[Dict]:
        """Cluster completed experiments by outcome profile.

        Uses high-performance NumPy vectorization for k-means clustering,
        silhouette scores, and model selection.
        """
        rows = self.nb.conn.execute("""
            SELECT experiment_id, n_programs_generated, n_stage1_passed,
                   best_novelty_score, best_loss_ratio, duration_seconds
            FROM experiments
            WHERE status = 'completed' AND n_programs_generated > 0
        """).fetchall()

        if len(rows) < 3: return None

        experiments = []
        for row in rows:
            total = row["n_programs_generated"] or 0
            if total <= 0: continue
            experiments.append({
                "experiment_id": row["experiment_id"],
                "s1_rate": (row["n_stage1_passed"] or 0) / total,
                "best_novelty": float(row["best_novelty_score"] or 0.0),
                "best_loss_ratio": float(row["best_loss_ratio"] or 1.0),
                "duration_seconds": float(row["duration_seconds"] or 0.0),
            })

        if len(experiments) < 3: return None

        exp_ids = [e["experiment_id"] for e in experiments]
        placeholders = ",".join("?" * len(exp_ids))
        
        # Load failures and errors
        failure_rows = self.nb.conn.execute(f"""
            SELECT experiment_id, COUNT(*) as n_total,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 0 THEN 1 ELSE 0 END) as n_compile_fail,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 1 AND COALESCE(stage05_passed, 0) = 0 THEN 1 ELSE 0 END) as n_train_fail,
                   SUM(CASE WHEN COALESCE(stage05_passed, 0) = 1 AND COALESCE(stage1_passed, 0) = 0 THEN 1 ELSE 0 END) as n_stage1_fail
            FROM program_results WHERE experiment_id IN ({placeholders}) GROUP BY experiment_id
        """, tuple(exp_ids)).fetchall()
        
        fail_map = {r["experiment_id"]: r for r in failure_rows}
        
        error_rows = self.nb.conn.execute(f"""
            SELECT experiment_id, error_type, COUNT(*) as n
            FROM program_results WHERE experiment_id IN ({placeholders})
            AND error_type IS NOT NULL AND TRIM(error_type) != '' GROUP BY experiment_id, error_type
        """, tuple(exp_ids)).fetchall()
        
        error_map = defaultdict(dict)
        for r in error_rows: error_map[r["experiment_id"]][r["error_type"]] = int(r["n"] or 0)

        for e in experiments:
            f = fail_map.get(e["experiment_id"], {"n_total": 1, "n_compile_fail": 0, "n_train_fail": 0, "n_stage1_fail": 0})
            n = float(f["n_total"] or 1)
            e.update({
                "compile_fail_rate": f["n_compile_fail"] / n,
                "train_fail_rate": f["n_train_fail"] / n,
                "stage1_fail_rate": f["n_stage1_fail"] / n,
                "error_diversity": 0.0
            })
            errs = error_map.get(e["experiment_id"], {})
            total_err = float(sum(errs.values()))
            if total_err > 0 and len(errs) > 1:
                probs = np.array(list(errs.values())) / total_err
                e["error_diversity"] = -np.sum(probs * np.log(probs)) / np.log(len(errs))

        # Load trajectories
        seq_rows = self.nb.conn.execute(f"""
            SELECT experiment_id, stage1_passed, loss_ratio, novelty_score
            FROM program_results WHERE experiment_id IN ({placeholders})
            ORDER BY experiment_id ASC, timestamp ASC
        """, tuple(exp_ids)).fetchall()
        
        per_exp_seq = defaultdict(list)
        for r in seq_rows:
            per_exp_seq[r["experiment_id"]].append((float(r["stage1_passed"] or 0), float(r["novelty_score"] or 0), float(r["loss_ratio"] or 1.0)))

        for e in experiments:
            seq = np.array(per_exp_seq.get(e["experiment_id"], []), dtype=np.float32)
            if len(seq) < 2:
                e.update({k: 0.0 for k in ["stage1_momentum", "novelty_momentum", "loss_improvement_momentum", "outcome_volatility", "outcome_peak_timing", "recovery_lag", "stage1_transition_timing", "primary_change_point_timing", "stage1_transition_density", "change_point_confidence", "windowed_change_dispersion", "window_change_localization", "transition_gap_entropy"]})
                continue
            
            # Vectorized momentum and statistics
            window = max(1, len(seq) // 3)
            e["stage1_momentum"] = np.mean(seq[-window:, 0]) - np.mean(seq[:window, 0])
            e["novelty_momentum"] = np.mean(seq[-window:, 1]) - np.mean(seq[:window, 1])
            e["loss_improvement_momentum"] = np.mean(seq[:window, 2]) - np.mean(seq[-window:, 2])
            
            proxy = 0.5 * seq[:, 0] + 0.3 * seq[:, 1] + 0.2 * (1.0 / (1.0 + np.maximum(seq[:, 2], 1e-9)))
            e["outcome_volatility"] = np.std(proxy)
            e["outcome_peak_timing"] = np.argmax(proxy) / max(len(seq) - 1, 1)
            
            # Transitions
            transitions = np.where(seq[1:, 0] != seq[:-1, 0])[0] + 1
            e["stage1_transition_timing"] = transitions[0] / (len(seq) - 1) if len(transitions) > 0 else 0.0
            e["stage1_transition_density"] = len(transitions) / max(len(seq) - 1, 1)
            if len(transitions) >= 2:
                gaps = np.diff(transitions).astype(np.float32)
                p = gaps / gaps.sum()
                e["transition_gap_entropy"] = -np.sum(p * np.log(p + 1e-10)) / np.log(len(transitions))
            else:
                e["transition_gap_entropy"] = 0.0

            deltas = np.abs(np.diff(proxy))
            if len(deltas) > 0:
                e["primary_change_point_timing"] = (np.argmax(deltas) + 1) / max(len(seq) - 1, 1)
                e["change_point_confidence"] = np.max(deltas) / (np.sum(deltas) + 1e-10)
                
                # Windowed change dispersion and localization
                n_deltas = len(deltas)
                seg = max(1, n_deltas // 3)
                window_means = [np.mean(deltas[i*seg : (i+1)*seg]) if i*seg < n_deltas else 0.0 for i in range(3)]
                e["windowed_change_dispersion"] = np.std(window_means)
                total_window_change = np.sum(window_means)
                e["window_change_localization"] = np.max(window_means) / total_window_change if total_window_change > 1e-9 else 0.0
            else:
                e.update({"primary_change_point_timing": 0.0, "change_point_confidence": 0.0, 
                          "windowed_change_dispersion": 0.0, "window_change_localization": 0.0})

            # Recovery lag
            early_baseline = np.mean(proxy[:window])
            trough_idx = np.argmin(proxy)
            recovery_idx = np.where(proxy[trough_idx+1:] >= early_baseline)[0]
            e["recovery_lag"] = (recovery_idx[0] + 1) / (len(seq) - 1) if len(recovery_idx) > 0 else (1.0 if len(seq) > 1 else 0.0)

        # Prepare for Vectorized K-Means
        feature_keys = ["s1_rate", "best_novelty", "best_loss_ratio", "duration_seconds", "compile_fail_rate", "train_fail_rate", "stage1_fail_rate", "error_diversity", "stage1_momentum", "novelty_momentum", "loss_improvement_momentum", "outcome_volatility", "outcome_peak_timing", "recovery_lag", "stage1_transition_timing", "primary_change_point_timing", "stage1_transition_density", "change_point_confidence", "windowed_change_dispersion", "window_change_localization", "transition_gap_entropy"]
        X = np.array([[e[k] for k in feature_keys] for e in experiments], dtype=np.float32)
        # Normalize and invert loss_ratio
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = X_max - X_min
        X_norm = np.zeros_like(X)
        mask = X_range > 1e-9
        X_norm[:, mask] = (X[:, mask] - X_min[mask]) / X_range[mask]
        X_norm[:, feature_keys.index("best_loss_ratio")] = 1.0 - X_norm[:, feature_keys.index("best_loss_ratio")]

        def _vectorized_kmeans(k, salt):
            # Exact match of original deterministic init
            seed_hex = hashlib.md5(f"{dataset_signature}:{salt}".encode()).hexdigest()
            first_idx = int(seed_hex[:8], 16) % len(X_norm)
            centroids = [X_norm[first_idx]]
            chosen_idxs = {first_idx}
            
            for _ in range(1, k):
                # Farthest point initialization (K-Means++)
                dists = np.min(cdist(X_norm, np.array(centroids)), axis=1)
                # Filter out already chosen to be safe (though dist would be 0)
                next_idx = np.argmax(dists)
                centroids.append(X_norm[next_idx])
                chosen_idxs.add(next_idx)
            centroids = np.array(centroids)
            
            for _ in range(30):
                dists = cdist(X_norm, centroids)
                assignments = np.argmin(dists, axis=1)
                new_centroids = np.array([X_norm[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i] for i in range(k)])
                if np.allclose(centroids, new_centroids): break
                centroids = new_centroids
            
            inertia = np.sum(np.min(cdist(X_norm, centroids), axis=1)**2)
            return assignments, centroids, inertia

        def _vectorized_silhouette(assignments, dist_matrix):
            unique = np.unique(assignments)
            if len(unique) < 2: return 0.0
            sil = []
            for i in range(len(X_norm)):
                c_i = assignments[i]
                mask_same = (assignments == c_i)
                mask_same[i] = False
                if not np.any(mask_same):
                    sil.append(0.0)
                    continue
                a_i = dist_matrix[i, mask_same].mean()
                b_i = min(dist_matrix[i, assignments == c].mean() for c in unique if c != c_i)
                sil.append((b_i - a_i) / max(a_i, b_i, 1e-9))
            return np.mean(sil)

        dataset_signature = "|".join(sorted(exp_ids))
        dist_matrix = cdist(X_norm, X_norm)
        max_k = min(max(2, n_clusters), len(X_norm) - 1)
        if max_k < 2: return None

        candidates = []
        for k_val in range(2, max_k + 1):
            runs = []
            for salt in range(4):
                assign, cents, inertia = _vectorized_kmeans(k_val, salt)
                sil = _vectorized_silhouette(assign, dist_matrix)
                counts = np.bincount(assign, minlength=k_val)
                imbalance = np.sum(np.abs(counts - len(X_norm)/k_val)) / (2.0 * len(X_norm))
                runs.append({"assignments": assign, "centroids": cents, "inertia": inertia, "silhouette": sil, "quality": sil - 0.15 * imbalance})
            
            best = max(runs, key=lambda r: (r["quality"], -r["inertia"]))
            candidates.append({"k": k_val, "best": best, "runs": runs, "score": best["quality"]})

        selected = max(candidates, key=lambda c: (c["score"], -c["k"]))
        k, best_run = selected["k"], selected["best"]
        assign, cents = best_run["assignments"], best_run["centroids"]

        # Consensus and Stability
        def _agreement(a1, a2):
            m1 = (a1[:, None] == a1[None, :])
            m2 = (a2[:, None] == a2[None, :])
            return np.mean(m1 == m2)

        cons_scores = [_agreement(r1["assignments"], r2["assignments"]) for i, r1 in enumerate(selected["runs"]) for r2 in selected["runs"][i+1:]]
        consensus = np.mean(cons_scores) if cons_scores else 1.0
        
        intra = np.mean([np.mean(dist_matrix[i, assign == assign[i]]) for i in range(len(X_norm))])
        inter = np.min(cdist(cents, cents) + np.eye(k)*1e9)
        stability = 0.6 * (inter / (inter + intra + 1e-9)) + 0.4 * consensus

        # Summary and Description
        clusters = []
        for ci in range(k):
            members = [experiments[i] for i in range(len(experiments)) if assign[i] == ci]
            if not members: continue
            summary = {k: round(float(np.mean([m[k] for m in members])), 4) for k in feature_keys if k != "duration_seconds"}
            summary["avg_duration_seconds"] = round(float(np.mean([m["duration_seconds"] for m in members])), 2)
            summary.update({"cluster_id": ci, "size": len(members), "experiment_ids": [m["experiment_id"] for m in members[:10]]})
            # Map avg_s1_rate, etc back to required keys
            for fk in ["s1_rate", "best_novelty", "best_loss_ratio"]:
                summary[f"avg_{fk}"] = summary.pop(fk)
            for fk in feature_keys:
                if fk not in ["s1_rate", "best_novelty", "best_loss_ratio", "duration_seconds"] and fk in summary:
                    summary[f"avg_{fk}"] = summary.pop(fk)
            clusters.append(summary)

        clusters.sort(key=lambda c: c["avg_s1_rate"], reverse=True)
        self._describe_clusters(clusters)

        return {
            "n_experiments": len(experiments), "n_clusters": len(clusters),
            "feature_keys": feature_keys, "stability_score": round(float(np.clip(stability, 0, 1)), 4),
            "model_selection": {"candidate_ks": [c["k"] for c in candidates], "selected_k": k, 
                                "silhouette": round(float(best_run["silhouette"]), 4), "consensus": round(float(consensus), 4),
                                "selection_margin": round(float(sorted(candidates, key=lambda c: -c["score"])[0]["score"] - sorted(candidates, key=lambda c: -c["score"])[1]["score"]), 4) if len(candidates) > 1 else 0.0},
            "clusters": clusters
        }

    @staticmethod
    def _describe_clusters(clusters: List[Dict]) -> None:
        """Generate contrastive plain-language descriptions for clusters.

        Ranks clusters against each other so labels are mutually exclusive
        (e.g., "the most productive", "moderate", "the least productive").
        """
        if not clusters:
            return

        # Rank by S1 rate descending to assign relative labels
        ranked = sorted(
            enumerate(clusters),
            key=lambda ic: (ic[1].get("avg_s1_rate", 0) or 0),
            reverse=True,
        )

        for rank_idx, (orig_idx, c) in enumerate(ranked):
            size = c.get("size", 0)
            s1_pct = (c.get("avg_s1_rate", 0) or 0) * 100
            novelty = c.get("avg_best_novelty", 0) or 0
            loss_ratio = c.get("avg_best_loss_ratio", 0) or 0
            compile_fail = (c.get("avg_compile_fail_rate", 0) or 0) * 100
            duration = c.get("avg_duration_seconds", 0) or 0

            # S1 description
            if s1_pct >= 30:
                s1_desc = f"high S1 pass rate ({s1_pct:.0f}%)"
            elif s1_pct >= 10:
                s1_desc = f"moderate S1 pass rate ({s1_pct:.0f}%)"
            elif s1_pct > 0:
                s1_desc = f"low S1 pass rate ({s1_pct:.0f}%)"
            else:
                s1_desc = "no S1 survivors"

            # Novelty description
            if novelty >= 0.7:
                nov_desc = "high novelty"
            elif novelty >= 0.3:
                nov_desc = "moderate novelty"
            else:
                nov_desc = "low novelty"

            # Find distinguishing feature for this cluster
            distinguisher = ""
            if compile_fail >= 50:
                distinguisher = f" High compile failure ({compile_fail:.0f}%) suggests grammar is exploring risky territory."
            elif novelty >= 0.5:
                distinguisher = f" High novelty ({novelty:.2f}) means these explore unfamiliar architecture space."
            elif duration > 600:
                distinguisher = f" Long average duration ({duration:.0f}s) indicates deeper investigation runs."

            # Relative character label
            n_clusters = len(clusters)
            if n_clusters == 1:
                character = "the only cluster"
            elif rank_idx == 0:
                character = "the most productive cluster"
            elif rank_idx == n_clusters - 1:
                if s1_pct == 0:
                    character = "the failing cluster"
                else:
                    character = "the least productive cluster"
            else:
                character = "a mid-tier cluster"

            clusters[orig_idx]["description"] = (
                f"{size} experiments with {s1_desc}, {nov_desc}."
                f" {character.capitalize()}.{distinguisher}"
            )

    @staticmethod
    def _explain_routing_health(by_mode: List[Dict], total_programs: int,
                                overall_stage1_pass_rate: float) -> str:
        """Generate deterministic plain-language routing interpretation."""
        if not by_mode:
            return (
                "No routing telemetry available yet. Run routed architectures "
                "to estimate drop rate, utilization balance, and confidence."
            )

        def weighted_mean(key: str) -> Optional[float]:
            weighted_sum = 0.0
            weighted_n = 0
            for row in by_mode:
                value = row.get(key)
                n_programs = int(row.get("n_programs") or 0)
                if value is None or n_programs <= 0:
                    continue
                weighted_sum += float(value) * n_programs
                weighted_n += n_programs
            if weighted_n == 0:
                return None
            return weighted_sum / weighted_n

        avg_drop = weighted_mean("avg_drop_rate")
        avg_entropy = weighted_mean("avg_utilization_entropy")
        avg_confidence = weighted_mean("avg_confidence_mean")

        if avg_drop is None:
            drop_desc = "unknown"
            drop_text = "drop rate is unavailable"
        elif avg_drop <= 0.05:
            drop_desc = "low"
            drop_text = f"drop rate is low ({avg_drop * 100:.1f}%)"
        elif avg_drop <= 0.15:
            drop_desc = "moderate"
            drop_text = f"drop rate is moderate ({avg_drop * 100:.1f}%)"
        else:
            drop_desc = "high"
            drop_text = f"drop rate is high ({avg_drop * 100:.1f}%)"

        if avg_entropy is None:
            entropy_text = "utilization balance is unavailable"
        elif avg_entropy >= 1.2:
            entropy_text = f"utilization appears balanced (entropy {avg_entropy:.2f})"
        elif avg_entropy >= 0.8:
            entropy_text = f"utilization is moderately balanced (entropy {avg_entropy:.2f})"
        else:
            entropy_text = f"utilization looks concentrated (entropy {avg_entropy:.2f})"

        if avg_confidence is None:
            confidence_desc = "unknown"
            confidence_text = "confidence is unavailable"
        elif avg_confidence >= 0.7:
            confidence_desc = "strong"
            confidence_text = f"confidence is strong ({avg_confidence:.2f})"
        elif avg_confidence >= 0.5:
            confidence_desc = "moderate"
            confidence_text = f"confidence is moderate ({avg_confidence:.2f})"
        else:
            confidence_desc = "weak"
            confidence_text = f"confidence is weak ({avg_confidence:.2f})"

        best_mode = max(by_mode, key=lambda r: float(r.get("stage1_pass_rate") or 0.0))
        sentence_1 = (
            f"Across {total_programs} routed programs, overall Stage 1 pass rate is "
            f"{overall_stage1_pass_rate * 100:.1f}% and {drop_text}."
        )
        sentence_2 = (
            f"Routing {entropy_text}, while {confidence_text}. "
            f"Best-performing mode is '{best_mode.get('routing_mode', 'unknown')}' "
            f"at {float(best_mode.get('stage1_pass_rate') or 0.0) * 100:.1f}% S1 pass."
        )

        if drop_desc == "high":
            sentence_3 = "High drop suggests routing-capacity pressure; consider reducing overflow and token skipping."
        elif confidence_desc == "weak":
            sentence_3 = "Low confidence suggests uncertain routing choices; prioritize modes with steadier confidence and lower variance."
        elif avg_entropy is not None and avg_entropy < 0.8:
            sentence_3 = "Low entropy suggests expert over-concentration; improve utilization balance to reduce mode-collapse risk."
        else:
            sentence_3 = "Telemetry suggests routing is reasonably healthy and balanced across current modes."

        return f"{sentence_1} {sentence_2} {sentence_3}"

    @staticmethod
    def _routing_sample_size_label(n_programs: int) -> str:
        if n_programs >= 80:
            return "high"
        if n_programs >= 30:
            return "medium"
        return "low"

    @staticmethod
    def _routing_confidence_label(avg_confidence_mean: Optional[float], avg_confidence_std: Optional[float]) -> str:
        if avg_confidence_mean is None:
            return "unknown"
        adjusted = avg_confidence_mean - 0.5 * (avg_confidence_std or 0.0)
        if adjusted >= 0.7:
            return "high"
        if adjusted >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _routing_stability_label(avg_confidence_std: Optional[float]) -> str:
        if avg_confidence_std is None:
            return "unknown"
        if avg_confidence_std <= 0.08:
            return "stable"
        if avg_confidence_std <= 0.16:
            return "moderate"
        return "volatile"

    @staticmethod
    def _routing_efficiency_label(token_retention: Optional[float]) -> str:
        if token_retention is None:
            return "unknown"
        if token_retention >= 0.9:
            return "high"
        if token_retention >= 0.75:
            return "medium"
        return "low"

    def routing_mode_comparison(self) -> Dict:
        """Compare routing modes with sample-size and confidence labels."""
        rows = self.nb.conn.execute("""
            SELECT
                COALESCE(NULLIF(routing_mode, ''), 'uniform') as routing_mode,
                COUNT(*) as n_programs,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as n_stage1_passed,
                AVG(loss_ratio) as avg_loss_ratio,
                AVG(routing_tokens_total) as avg_tokens_total,
                AVG(routing_tokens_processed) as avg_tokens_processed,
                AVG(routing_tokens_skipped) as avg_tokens_skipped,
                AVG(routing_drop_rate) as avg_drop_rate,
                AVG(routing_utilization_entropy) as avg_utilization_entropy,
                AVG(routing_capacity_overflow_count) as avg_capacity_overflow_count,
                AVG(routing_confidence_mean) as avg_confidence_mean,
                AVG(routing_confidence_std) as avg_confidence_std
            FROM program_results
            GROUP BY COALESCE(NULLIF(routing_mode, ''), 'uniform')
            ORDER BY n_programs DESC
        """).fetchall()

        if not rows:
            return {
                "available": False,
                "n_modes": 0,
                "total_programs": 0,
                "routed_programs": 0,
                "uniform_programs": 0,
                "by_mode": [],
                "explanation": (
                    "No routing telemetry available yet. Run routed architectures "
                    "to compare routing modes and confidence stability."
                ),
            }

        by_mode = []
        total_programs = 0
        total_stage1 = 0
        routed_programs = 0
        uniform_programs = 0

        for row in rows:
            n_programs = row["n_programs"] or 0
            n_stage1 = row["n_stage1_passed"] or 0
            total_programs += n_programs
            total_stage1 += n_stage1
            mode = row["routing_mode"] or "uniform"

            if mode == "uniform":
                uniform_programs += n_programs
            else:
                routed_programs += n_programs

            stage1_pass_rate = n_stage1 / max(n_programs, 1)
            avg_drop_rate = row["avg_drop_rate"]
            avg_tokens_total = row["avg_tokens_total"]
            avg_tokens_processed = row["avg_tokens_processed"]
            token_retention = None
            if avg_tokens_total and avg_tokens_total > 0 and avg_tokens_processed is not None:
                token_retention = avg_tokens_processed / avg_tokens_total
            elif avg_drop_rate is not None:
                token_retention = max(0.0, min(1.0, 1.0 - avg_drop_rate))

            avg_conf_mean = row["avg_confidence_mean"]
            avg_conf_std = row["avg_confidence_std"]

            by_mode.append({
                "routing_mode": mode,
                "n_programs": n_programs,
                "stage1_pass_rate": stage1_pass_rate,
                "avg_loss_ratio": row["avg_loss_ratio"],
                "avg_tokens_total": avg_tokens_total,
                "avg_tokens_processed": avg_tokens_processed,
                "avg_tokens_skipped": row["avg_tokens_skipped"],
                "avg_drop_rate": avg_drop_rate,
                "avg_utilization_entropy": row["avg_utilization_entropy"],
                "avg_capacity_overflow_count": row["avg_capacity_overflow_count"],
                "avg_confidence_mean": avg_conf_mean,
                "avg_confidence_std": avg_conf_std,
                "token_retention": token_retention,
                "sample_size_label": self._routing_sample_size_label(n_programs),
                "confidence_label": self._routing_confidence_label(avg_conf_mean, avg_conf_std),
                "stability_label": self._routing_stability_label(avg_conf_std),
                "efficiency_label": self._routing_efficiency_label(token_retention),
            })

        overall_stage1_pass_rate = total_stage1 / max(total_programs, 1)

        return {
            "available": True,
            "n_modes": len(by_mode),
            "total_programs": total_programs,
            "routed_programs": routed_programs,
            "uniform_programs": uniform_programs,
            "overall_stage1_pass_rate": overall_stage1_pass_rate,
            "by_mode": by_mode,
            "explanation": self._explain_routing_health(
                by_mode=by_mode,
                total_programs=total_programs,
                overall_stage1_pass_rate=overall_stage1_pass_rate,
            ),
        }

    def routing_health(self) -> Dict:
        """Aggregate routing telemetry by routing mode.

        Returns structured defaults when routing telemetry is not yet available,
        so dashboard/API consumers can safely render partial states.
        """
        comparison = self.routing_mode_comparison()
        if not comparison.get("available"):
            return {
                "available": False,
                "n_modes": 0,
                "total_programs": 0,
                "by_mode": [],
                "explanation": comparison.get("explanation", "Routing telemetry is unavailable."),
            }
        return {
            "available": True,
            "n_modes": comparison.get("n_modes", 0),
            "total_programs": comparison.get("total_programs", 0),
            "overall_stage1_pass_rate": comparison.get("overall_stage1_pass_rate", 0.0),
            "by_mode": comparison.get("by_mode", []),
            "explanation": comparison.get("explanation", ""),
        }

    @staticmethod
    def _routing_collapse_risk_label(avg_entropy: Optional[float]) -> str:
        if avg_entropy is None:
            return "unknown"
        if avg_entropy >= 1.2:
            return "low"
        if avg_entropy >= 0.8:
            return "medium"
        return "high"

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * percentile
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    def gating_behavior_diagnostics(self) -> Dict:
        """Canonical diagnostics for gated/recursive routing behavior."""
        rows = self.nb.conn.execute("""
            SELECT
                COALESCE(NULLIF(routing_mode, ''), 'uniform') AS routing_mode,
                routing_tokens_total,
                routing_tokens_processed,
                routing_drop_rate,
                routing_utilization_entropy,
                routing_capacity_overflow_count
            FROM program_results
            WHERE routing_mode IS NOT NULL
               OR routing_drop_rate IS NOT NULL
               OR routing_utilization_entropy IS NOT NULL
               OR routing_tokens_total IS NOT NULL
               OR routing_tokens_processed IS NOT NULL
        """).fetchall()

        if not rows:
            return {
                "available": False,
                "total_routed_programs": 0,
                "avg_gate_entropy": None,
                "collapse_risk_counts": {"low": 0, "medium": 0, "high": 0, "unknown": 0},
                "by_mode": [],
                "token_retention_curve_overall": [],
                "explanation": (
                    "No gating telemetry available yet. Run routed/recursive candidates "
                    "to estimate entropy, collapse risk, and token retention behavior."
                ),
            }

        by_mode_raw: Dict[str, Dict[str, Any]] = {}
        total_entropy = 0.0
        entropy_count = 0
        overall_retentions: List[float] = []
        collapse_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

        for row in rows:
            mode = row["routing_mode"] or "uniform"
            bucket = by_mode_raw.setdefault(mode, {
                "n_programs": 0,
                "entropies": [],
                "retentions": [],
                "drop_rates": [],
                "overflows": [],
            })
            bucket["n_programs"] += 1

            entropy = row["routing_utilization_entropy"]
            if entropy is not None:
                entropy = float(entropy)
                bucket["entropies"].append(entropy)
                total_entropy += entropy
                entropy_count += 1

            drop_rate = row["routing_drop_rate"]
            if drop_rate is not None:
                bucket["drop_rates"].append(float(drop_rate))

            overflow = row["routing_capacity_overflow_count"]
            if overflow is not None:
                bucket["overflows"].append(float(overflow))

            retention = None
            total_tokens = row["routing_tokens_total"]
            processed_tokens = row["routing_tokens_processed"]
            if total_tokens is not None and processed_tokens is not None and float(total_tokens) > 0:
                retention = float(processed_tokens) / float(total_tokens)
            elif drop_rate is not None:
                retention = max(0.0, min(1.0, 1.0 - float(drop_rate)))

            if retention is not None:
                retention = max(0.0, min(1.0, retention))
                bucket["retentions"].append(retention)
                overall_retentions.append(retention)

        by_mode: List[Dict[str, Any]] = []
        for mode, bucket in sorted(by_mode_raw.items(), key=lambda item: item[1]["n_programs"], reverse=True):
            entropies = bucket["entropies"]
            retentions = bucket["retentions"]
            avg_entropy = (sum(entropies) / len(entropies)) if entropies else None
            collapse_label = self._routing_collapse_risk_label(avg_entropy)
            collapse_counts[collapse_label] += 1

            avg_retention = (sum(retentions) / len(retentions)) if retentions else None
            token_curve = []
            for quantile, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
                value = self._percentile(retentions, quantile)
                if value is not None:
                    token_curve.append({"quantile": name, "retention": value})

            avg_drop = (sum(bucket["drop_rates"]) / len(bucket["drop_rates"])) if bucket["drop_rates"] else None
            avg_overflow = (sum(bucket["overflows"]) / len(bucket["overflows"])) if bucket["overflows"] else None

            by_mode.append({
                "routing_mode": mode,
                "n_programs": bucket["n_programs"],
                "avg_gate_entropy": avg_entropy,
                "collapse_risk_label": collapse_label,
                "avg_token_retention": avg_retention,
                "token_retention_curve": token_curve,
                "avg_drop_rate": avg_drop,
                "avg_capacity_overflow_count": avg_overflow,
            })

        overall_curve = []
        for quantile, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
            value = self._percentile(overall_retentions, quantile)
            if value is not None:
                overall_curve.append({"quantile": name, "retention": value})

        avg_gate_entropy = (total_entropy / entropy_count) if entropy_count > 0 else None
        explanation = (
            f"Gating diagnostics over {len(rows)} routed candidates: "
            f"avg gate entropy is {avg_gate_entropy:.2f}. "
            f"Collapse-risk modes (high) = {collapse_counts['high']}."
            if avg_gate_entropy is not None
            else "Gating diagnostics collected, but gate entropy is not yet available."
        )

        return {
            "available": True,
            "total_routed_programs": len(rows),
            "avg_gate_entropy": avg_gate_entropy,
            "collapse_risk_counts": collapse_counts,
            "by_mode": by_mode,
            "token_retention_curve_overall": overall_curve,
            "explanation": explanation,
        }

    def control_experiment_comparison(self) -> Optional[Dict]:
        """Compare control experiments (default weights) vs learned-weight experiments.

        Control experiments are flagged with ``control_experiment = true`` in
        config_json.  Uses time-matched comparison: each control experiment is
        paired with temporally adjacent learned experiments (the nearest before
        and after) to avoid confounding from increasing search difficulty over
        a session.

        Returns None if fewer than 2 control experiments exist.
        """
        rows = self.nb.conn.execute("""
            SELECT experiment_id, config_json, timestamp
            FROM experiments WHERE status = 'completed' AND config_json IS NOT NULL
            ORDER BY timestamp
        """).fetchall()

        control_exps: list[tuple[str, str]] = []  # (exp_id, created_at)
        learned_exps: list[tuple[str, str]] = []
        for row in rows:
            try:
                cfg = json.loads(row["config_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            ts = row["timestamp"] or ""
            if cfg.get("control_experiment"):
                control_exps.append((row["experiment_id"], ts))
            else:
                learned_exps.append((row["experiment_id"], ts))

        if len(control_exps) < 2 or len(learned_exps) < 2:
            return None

        def _s1_for_exp(exp_id: str) -> tuple[int, int]:
            """Return (total_programs, s1_passed) for one experiment."""
            r = self.nb.conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1
                FROM program_results WHERE experiment_id = ?
            """, (exp_id,)).fetchone()
            return (r["total"] or 0, r["s1"] or 0)

        def _s1_stats(exp_ids: list[str]) -> dict:
            total = s1 = 0
            for eid in exp_ids:
                t, s = _s1_for_exp(eid)
                total += t
                s1 += s
            return {"experiments": len(exp_ids), "programs": total,
                    "s1_passed": s1, "s1_rate": s1 / max(total, 1)}

        # Time-matched comparison: for each control, find nearest learned
        # experiments (one before, one after) and compare within that window
        learned_ts = [(eid, ts) for eid, ts in learned_exps]
        pair_diffs: list[float] = []
        matched_control_ids: list[str] = []
        matched_learned_ids: list[str] = []

        for ctrl_id, ctrl_ts in control_exps:
            # Find nearest learned experiments by timestamp
            before = [e for e in learned_ts if e[1] <= ctrl_ts]
            after = [e for e in learned_ts if e[1] > ctrl_ts]
            neighbors = []
            if before:
                neighbors.append(before[-1][0])  # latest before
            if after:
                neighbors.append(after[0][0])     # earliest after
            if not neighbors:
                continue

            ctrl_total, ctrl_s1 = _s1_for_exp(ctrl_id)
            if ctrl_total == 0:
                continue
            ctrl_rate = ctrl_s1 / ctrl_total

            nbr_total = nbr_s1 = 0
            for nid in neighbors:
                t, s = _s1_for_exp(nid)
                nbr_total += t
                nbr_s1 += s
            if nbr_total == 0:
                continue
            nbr_rate = nbr_s1 / nbr_total

            pair_diffs.append(nbr_rate - ctrl_rate)
            matched_control_ids.append(ctrl_id)
            matched_learned_ids.extend(neighbors)

        # Also compute overall stats for display
        all_control_ids = [eid for eid, _ in control_exps]
        all_learned_ids = [eid for eid, _ in learned_exps]
        control = _s1_stats(all_control_ids)
        learned = _s1_stats(all_learned_ids)

        # Time-matched effect: mean of per-window differences
        if pair_diffs:
            matched_diff = sum(pair_diffs) / len(pair_diffs)
            # Paired z-test (approximate)
            if len(pair_diffs) > 1:
                diff_std = (sum((d - matched_diff) ** 2 for d in pair_diffs) / (len(pair_diffs) - 1)) ** 0.5
                matched_se = diff_std / len(pair_diffs) ** 0.5 if diff_std > 0 else 0.0
                matched_z = matched_diff / matched_se if matched_se > 0 else 0.0
            else:
                matched_z = 0.0
        else:
            matched_diff = 0.0
            matched_z = 0.0

        return {
            "control": control,
            "learned": learned,
            "s1_rate_difference": round(matched_diff, 4),
            "z_score": round(matched_z, 3),
            "significant_at_p05": abs(matched_z) > 1.96,
            "learned_is_better": matched_diff > 0,
            "time_matched": True,
            "matched_pairs": len(pair_diffs),
            "interpretation": (
                "Learned weights significantly outperform controls (time-matched)"
                if matched_z > 1.96
                else "Learned weights significantly underperform controls (time-matched)"
                if matched_z < -1.96
                else "No significant difference between learned and control weights (time-matched)"
            ),
            "caveat": (
                "Comparison is time-matched: each control is compared only with "
                "temporally adjacent learned experiments to account for increasing "
                "search difficulty over a session."
            ),
        }

    def failure_patterns(self) -> Dict[str, Dict]:
        """Analyze common failure modes by error type and stage."""
        rows = self.nb.conn.execute("""
            SELECT error_type, stage_at_death, COUNT(*) as count
            FROM program_results
            WHERE error_type IS NOT NULL
            GROUP BY error_type, stage_at_death
            ORDER BY count DESC
        """).fetchall()

        patterns: Dict[str, Dict] = {}
        for r in rows:
            error_type = r["error_type"] or "unknown"
            if error_type not in patterns:
                patterns[error_type] = {"total": 0, "by_stage": {}}
            patterns[error_type]["total"] += r["count"]
            stage = r["stage_at_death"] or "unknown"
            patterns[error_type]["by_stage"][stage] = r["count"]

        return patterns

    def top_op_combinations(self, n: int = 10) -> List[Dict]:
        """Find op combinations that co-occur in Stage 1 survivors."""
        rows = self.nb.conn.execute("""
            SELECT graph_json, novelty_score, loss_ratio
            FROM program_results
            WHERE stage1_passed = 1 AND graph_json IS NOT NULL
            ORDER BY novelty_score DESC NULLS LAST
            LIMIT 200
        """).fetchall()

        # Count op pair co-occurrences
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_novelty: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        ops_cache: Dict[str, Optional[List[str]]] = {}

        for r in rows:
            graph_json = r["graph_json"]
            if graph_json in ops_cache:
                ops = ops_cache[graph_json]
            else:
                ops = self._extract_ops_fast(graph_json)
                if ops is None:
                    ops = self._extract_ops_fallback(graph_json)
                ops_cache[graph_json] = ops

            if not ops:
                continue

            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    pair = (ops[i], ops[j])
                    pair_counts[pair] += 1
                    if r["novelty_score"]:
                        pair_novelty[pair].append(r["novelty_score"])

        # Sort by frequency
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:n]
        results = []
        for (op_a, op_b), count in top_pairs:
            novelties = pair_novelty.get((op_a, op_b), [])
            results.append({
                "ops": [op_a, op_b],
                "count": count,
                "avg_novelty": sum(novelties) / len(novelties) if novelties else 0,
            })

        return results

    def learning_trajectory(self) -> Dict:
        """Compute learning trajectory: S1 rate trend with regression.

        Returns:
            {
                "points": [{experiment_id, timestamp, s1_rate, n_programs}, ...],
                "trend": "improving" | "plateaued" | "declining",
                "slope": float,           # S1-rate change per experiment
                "recent_s1_rate": float,   # avg of last 5
                "overall_s1_rate": float,
                "n_experiments": int,
                "min_experiments_required": int,
                "weight_adjustments": int, # count of grammar_weights_applied events
            }
        """
        experiments = self.nb.get_recent_experiments(100)
        # Reverse to chronological order
        experiments = list(reversed(experiments))

        points = []
        for exp in experiments:
            n_gen = exp.get("n_programs_generated") or 0
            n_s1 = exp.get("n_stage1_passed") or 0
            if n_gen == 0:
                continue
            points.append({
                "experiment_id": exp.get("experiment_id", ""),
                "timestamp": exp.get("timestamp", 0),
                "s1_rate": n_s1 / n_gen,
                "n_stage1_passed": n_s1,
                "n_programs": n_gen,
            })

        if not points:
            return {
                "points": [],
                "trend": "insufficient_data",
                "slope": 0.0,
                "recent_s1_rate": 0.0,
                "overall_s1_rate": 0.0,
                "n_experiments": 0,
                "min_experiments_required": self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS,
                "trend_confidence": "low",
                "overall_s1_confidence_halfwidth": 0.0,
                "weight_adjustments": 0,
            }

        total_programs = sum(max(int(p.get("n_programs") or 0), 0) for p in points)
        total_stage1 = sum(max(int(p.get("n_stage1_passed") or 0), 0) for p in points)
        global_rate = total_stage1 / max(total_programs, 1)
        prior_strength = 12.0

        for point in points:
            n_programs = max(int(point.get("n_programs") or 0), 0)
            effective_n = max(1.0, float(n_programs))
            raw_rate = float(point.get("s1_rate") or 0.0)
            shrinkage = effective_n / (effective_n + prior_strength)
            adjusted_rate = global_rate + shrinkage * (raw_rate - global_rate)
            variance = max(adjusted_rate * (1.0 - adjusted_rate), 0.0)
            halfwidth = 1.96 * math.sqrt(variance / max(effective_n, 1.0))

            point["adjusted_s1_rate"] = adjusted_rate
            point["s1_confidence_lower"] = max(0.0, adjusted_rate - halfwidth)
            point["s1_confidence_upper"] = min(1.0, adjusted_rate + halfwidth)
            point["s1_confidence_halfwidth"] = halfwidth
            point["trend_weight"] = min(1.0, effective_n / 20.0)

        # Linear regression on S1 rate vs experiment index
        n = len(points)
        rates = [p.get("adjusted_s1_rate", p["s1_rate"]) for p in points]
        mean_x = (n - 1) / 2.0
        mean_y = sum(rates) / n
        num = sum((i - mean_x) * (r - mean_y) for i, r in enumerate(rates))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        # Trend classification
        # Use slope relative to mean to avoid noise at tiny scales
        relative_slope = slope / max(mean_y, 0.01)
        if n < self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS:
            trend = "insufficient_data"
        elif relative_slope > 0.05:
            trend = "improving"
        elif relative_slope < -0.05:
            trend = "declining"
        else:
            trend = "plateaued"

        recent = rates[-5:] if len(rates) >= 5 else rates
        recent_rate = sum(recent) / len(recent)
        avg_trend_weight = sum(p.get("trend_weight", 0.0) for p in points) / max(len(points), 1)
        avg_conf_halfwidth = sum(p.get("s1_confidence_halfwidth", 0.0) for p in points) / max(len(points), 1)

        if avg_trend_weight >= 0.75:
            trend_confidence = "high"
        elif avg_trend_weight >= 0.45:
            trend_confidence = "medium"
        else:
            trend_confidence = "low"

        # Count grammar weight adjustments
        try:
            log = self.nb.get_learning_log(limit=200)
            weight_adjustments = sum(
                1 for entry in log
                if entry.get("event_type") == "grammar_weights_applied"
            )
        except Exception:
            weight_adjustments = 0

        return {
            "points": points,
            "trend": trend,
            "slope": round(slope, 6),
            "recent_s1_rate": round(recent_rate, 4),
            "overall_s1_rate": round(mean_y, 4),
            "n_experiments": n,
            "min_experiments_required": self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS,
            "trend_confidence": trend_confidence,
            "overall_s1_confidence_halfwidth": round(avg_conf_halfwidth, 6),
            "weight_adjustments": weight_adjustments,
        }

    def math_family_coverage(self) -> Dict:
        """Summarize evaluated/surviving coverage by mathematical family."""
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, graph_json, arch_spec_json
            FROM program_results
            WHERE graph_json IS NOT NULL OR arch_spec_json IS NOT NULL
        """).fetchall()

        family_order = ["euclidean", "hyperbolic", "tropical", "p-adic", "clifford", "functional"]
        stats = {
            fam: {"family": fam, "n_tested": 0, "n_survived": 0}
            for fam in family_order
        }

        hyperbolic_ops = {"poincare_add", "exp_map", "log_map", "hyp_linear", "hyp_distance", "hyp_tangent_nonlinear"}
        tropical_ops = {"tropical_matmul", "tropical_add", "tropical_attention", "tropical_center"}
        padic_ops = {"padic_expand", "ultrametric_attention", "padic_gate"}
        clifford_ops = {"geometric_product", "rotor_transform", "grade_select", "grade_mix"}
        functional_ops = {"basis_expansion", "integral_kernel", "fixed_point_iter"}

        def _family_from_row(graph_json: Optional[str], arch_spec_json: Optional[str]) -> str:
            op_names: set[str] = set()
            if graph_json:
                try:
                    graph = json.loads(graph_json)
                    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
                    node_iter = nodes.values() if isinstance(nodes, dict) else nodes if isinstance(nodes, list) else []
                    for node in node_iter:
                        if isinstance(node, dict):
                            op_name = node.get("op_name") or node.get("op")
                            if op_name:
                                op_names.add(op_name)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            token_mixing = None
            channel_mixing = None
            if arch_spec_json:
                try:
                    arch = json.loads(arch_spec_json)
                    choices = arch.get("choices", {}) if isinstance(arch, dict) else {}
                    if isinstance(choices, dict):
                        token_mixing = choices.get("token_mixing")
                        channel_mixing = choices.get("channel_mixing")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            if (op_names & functional_ops) or token_mixing == "integral_kernel_mixing" or channel_mixing in {
                "basis_expansion_layer", "implicit_fixed_point"
            }:
                return "functional"
            if op_names & hyperbolic_ops:
                return "hyperbolic"
            if op_names & tropical_ops:
                return "tropical"
            if op_names & padic_ops:
                return "p-adic"
            if op_names & clifford_ops:
                return "clifford"
            return "euclidean"

        total_tested = 0
        total_survived = 0
        for row in rows:
            family = _family_from_row(row["graph_json"], row["arch_spec_json"])
            bucket = stats.get(family, stats["euclidean"])
            bucket["n_tested"] += 1
            total_tested += 1

            if row["stage1_passed"]:
                bucket["n_survived"] += 1
                total_survived += 1

        families = []
        for fam in family_order:
            entry = stats[fam]
            n_tested = entry["n_tested"]
            n_survived = entry["n_survived"]
            families.append({
                "family": fam,
                "n_tested": n_tested,
                "n_survived": n_survived,
                "survival_rate": round(n_survived / n_tested, 4) if n_tested > 0 else 0.0,
                "tested_share": round(n_tested / total_tested, 4) if total_tested > 0 else 0.0,
                "survivor_share": round(n_survived / total_survived, 4) if total_survived > 0 else 0.0,
            })

        return {
            "families": families,
            "totals": {
                "n_tested": total_tested,
                "n_survived": total_survived,
            },
        }

    def mathspace_operator_impact(self) -> Dict:
        """Impact summary for math-space operators and families.

        Reports tested counts, S1/validation pass rates, novelty signal,
        and baseline-win rates for each math-space operator family.
        """
        columns = {
            row["name"]
            for row in self.nb.conn.execute("PRAGMA table_info(program_results)").fetchall()
            if row and row["name"]
        }
        validation_col = "validation_passed" if "validation_passed" in columns else "NULL AS validation_passed"
        baseline_col = "validation_baseline_ratio" if "validation_baseline_ratio" in columns else "NULL AS validation_baseline_ratio"

        rows = self.nb.conn.execute(f"""
            SELECT graph_json, stage1_passed, {validation_col}, novelty_score, {baseline_col}
            FROM program_results
            WHERE graph_json IS NOT NULL
        """).fetchall()

        op_family = {
            "poincare_add": "hyperbolic",
            "exp_map": "hyperbolic",
            "log_map": "hyperbolic",
            "hyp_linear": "hyperbolic",
            "hyp_distance": "hyperbolic",
            "hyp_tangent_nonlinear": "hyperbolic",
            "tropical_matmul": "tropical",
            "tropical_add": "tropical",
            "tropical_attention": "tropical",
            "tropical_center": "tropical",
            "padic_expand": "p-adic",
            "ultrametric_attention": "p-adic",
            "padic_gate": "p-adic",
            "geometric_product": "clifford",
            "rotor_transform": "clifford",
            "grade_select": "clifford",
            "grade_mix": "clifford",
        }
        tracked_ops = set(op_family.keys())

        if not rows:
            return {
                "available": False,
                "totals": {
                    "n_programs_with_graph": 0,
                    "n_programs_with_mathspace": 0,
                    "n_mathspace_ops_observed": 0,
                },
                "by_operator": [],
                "by_family": [],
                "explanation": "No graph-level program data available for math-space impact analysis.",
            }

        by_operator: Dict[str, Dict[str, float]] = {}
        by_family: Dict[str, Dict[str, float]] = {}
        programs_with_mathspace = 0

        def _ensure_bucket(store: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
            if key not in store:
                store[key] = {
                    "n_tested": 0,
                    "n_stage1_passed": 0,
                    "n_validation_passed": 0,
                    "n_baseline_wins": 0,
                    "novelty_sum": 0.0,
                    "novelty_count": 0,
                }
            return store[key]

        for row in rows:
            graph_json = row["graph_json"]
            if not graph_json:
                continue

            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue

            used_ops = sorted(tracked_ops.intersection(set(ops)))
            if not used_ops:
                continue

            programs_with_mathspace += 1
            used_families = sorted({op_family[op] for op in used_ops})
            stage1_passed = bool(row["stage1_passed"])
            validation_passed = bool(row["validation_passed"])
            novelty = self._as_float(row["novelty_score"])
            baseline_ratio = self._as_float(row["validation_baseline_ratio"])
            baseline_win = baseline_ratio is not None and baseline_ratio < 1.0

            for op_name in used_ops:
                bucket = _ensure_bucket(by_operator, op_name)
                bucket["n_tested"] += 1
                if stage1_passed:
                    bucket["n_stage1_passed"] += 1
                if validation_passed:
                    bucket["n_validation_passed"] += 1
                if baseline_win:
                    bucket["n_baseline_wins"] += 1
                if novelty is not None:
                    bucket["novelty_sum"] += novelty
                    bucket["novelty_count"] += 1

            for family in used_families:
                bucket = _ensure_bucket(by_family, family)
                bucket["n_tested"] += 1
                if stage1_passed:
                    bucket["n_stage1_passed"] += 1
                if validation_passed:
                    bucket["n_validation_passed"] += 1
                if baseline_win:
                    bucket["n_baseline_wins"] += 1
                if novelty is not None:
                    bucket["novelty_sum"] += novelty
                    bucket["novelty_count"] += 1

        def _finalize(rows_by_key: Dict[str, Dict[str, float]], label_key: str) -> List[Dict[str, float]]:
            finalized: List[Dict[str, float]] = []
            for key, bucket in rows_by_key.items():
                n_tested = int(bucket["n_tested"])
                if n_tested <= 0:
                    continue
                novelty_count = int(bucket["novelty_count"])
                stage1_rate = float(bucket["n_stage1_passed"]) / n_tested
                validation_rate = float(bucket["n_validation_passed"]) / n_tested
                baseline_win_rate = float(bucket["n_baseline_wins"]) / n_tested
                sample_weight = min(1.0, n_tested / 25.0)
                trust_score = (0.5 * stage1_rate + 0.3 * validation_rate + 0.2 * baseline_win_rate) * sample_weight
                if trust_score >= 0.6 and n_tested >= 20:
                    trust_label = "high"
                elif trust_score >= 0.35 and n_tested >= 8:
                    trust_label = "medium"
                else:
                    trust_label = "low"
                finalized.append({
                    label_key: key,
                    "n_tested": n_tested,
                    "n_stage1_passed": int(bucket["n_stage1_passed"]),
                    "n_validation_passed": int(bucket["n_validation_passed"]),
                    "n_baseline_wins": int(bucket["n_baseline_wins"]),
                    "stage1_pass_rate": round(stage1_rate, 4),
                    "validation_pass_rate": round(validation_rate, 4),
                    "baseline_win_rate": round(baseline_win_rate, 4),
                    "trust_score": round(trust_score, 4),
                    "trust_label": trust_label,
                    "avg_novelty_score": (
                        round(float(bucket["novelty_sum"]) / novelty_count, 4)
                        if novelty_count > 0 else None
                    ),
                })
            return sorted(finalized, key=lambda row: (-row["n_tested"], row[label_key]))

        by_operator_rows = _finalize(by_operator, "op_name")
        by_family_rows = _finalize(by_family, "family")
        top_trustworthy_ops = sorted(
            by_operator_rows,
            key=lambda row: (-(row.get("trust_score") or 0.0), -(row.get("n_tested") or 0), row.get("op_name") or ""),
        )[:3]

        top_op = by_operator_rows[0]["op_name"] if by_operator_rows else None
        explanation = (
            f"Observed {len(by_operator_rows)} math-space ops across {programs_with_mathspace}/{len(rows)} programs with graph traces. "
            f"Most common op: {top_op}."
            if top_op
            else "No math-space operators were observed in current graph traces."
        )

        return {
            "available": len(by_operator_rows) > 0,
            "totals": {
                "n_programs_with_graph": len(rows),
                "n_programs_with_mathspace": programs_with_mathspace,
                "n_mathspace_ops_observed": len(by_operator_rows),
            },
            "by_operator": by_operator_rows,
            "by_family": by_family_rows,
            "top_trustworthy_operators": top_trustworthy_ops,
            "explanation": explanation,
        }

    def compute_insights(self) -> List[Dict]:
        """Generate data-driven insights from experiment history.

        Returns structured insight dicts with varied category and confidence:
        ``[{"content": str, "category": str, "confidence": float}, ...]``

        Confidence is scaled by sample size and effect strength so that
        the dashboard scoring formula produces differentiated scores.
        """
        insights: List[Dict] = []

        # 1. Op success rate insights
        op_rates = self.op_success_rates()
        if op_rates:
            rated_ops = [(op, s["s1_rate"], s["n_used"])
                         for op, s in op_rates.items() if s["n_used"] >= 5]
            if rated_ops:
                rated_ops.sort(key=lambda x: -x[1])
                best_ops = rated_ops[:3]
                worst_ops = rated_ops[-3:]

                if best_ops[0][1] > 0:
                    op_names = ", ".join(f"{op}({rate:.0%})" for op, rate, _ in best_ops)
                    # Confidence scales with total usage of the best ops
                    total_usage = sum(n for _, _, n in best_ops)
                    conf = min(0.9, 0.5 + total_usage / 500)
                    insights.append({
                        "content": (
                            f"Top-performing ops (S1 rate): {op_names}. "
                            f"These compose well into learnable architectures."
                        ),
                        "category": "success_factor",
                        "confidence": round(conf, 2),
                    })

                if worst_ops and worst_ops[-1][1] == 0 and worst_ops[-1][2] >= 10:
                    failing = [(op, n) for op, rate, n in worst_ops if rate == 0]
                    if failing:
                        op_names = ", ".join(op for op, _ in failing)
                        total_usage = sum(n for _, n in failing)
                        conf = min(0.85, 0.4 + total_usage / 300)
                        insights.append({
                            "content": (
                                f"Consistently failing ops: {op_names}. "
                                f"Consider reducing their grammar weight."
                            ),
                            "category": "failure_mode",
                            "confidence": round(conf, 2),
                        })

        # 2. Structural correlation insights
        correlations = self.structural_correlations()
        if correlations:
            for metric, effect in sorted(correlations.items(),
                                         key=lambda x: -abs(x[1])):
                if abs(effect) > 0.3:
                    direction = "positively" if effect > 0 else "negatively"
                    name = metric.replace("graph_", "").replace("_", " ")
                    # Stronger effect → higher confidence
                    conf = min(0.9, 0.3 + abs(effect) * 0.6)
                    insights.append({
                        "content": (
                            f"Graph {name} is {direction} correlated with "
                            f"Stage 1 success (effect={effect:.2f})."
                        ),
                        "category": "hypothesis",
                        "confidence": round(conf, 2),
                    })
                    break  # just the strongest

        # 3. Failure pattern insights
        failures = self.failure_patterns()
        if failures:
            top_failure = max(failures.items(), key=lambda x: x[1]["total"])
            if top_failure[1]["total"] >= 10:
                total_failures = top_failure[1]["total"]
                conf = min(0.85, 0.45 + total_failures / 500)
                insights.append({
                    "content": (
                        f"Most common failure: {top_failure[0]} "
                        f"({total_failures} occurrences). "
                        f"Stages: {top_failure[1]['by_stage']}"
                    ),
                    "category": "failure_mode",
                    "confidence": round(conf, 2),
                })

        # 4. Op combination insights
        combos = self.top_op_combinations(5)
        if combos and combos[0]["count"] >= 3:
            top = combos[0]
            conf = min(0.9, 0.5 + top["count"] / 200)
            insights.append({
                "content": (
                    f"Winning combination: {' + '.join(top['ops'])} "
                    f"appears in {top['count']} survivors "
                    f"(avg novelty {top['avg_novelty']:.3f})."
                ),
                "category": "success_factor",
                "confidence": round(conf, 2),
            })

        # 5. Overall progress insight
        summary = self.nb.get_dashboard_summary()
        total = summary.get("total_programs_evaluated", 0)
        survivors = summary.get("stage1_survivors", 0)
        if total > 0:
            rate = survivors / total
            conf = min(0.95, 0.4 + total / 1000)
            insights.append({
                "content": (
                    f"Overall survival rate: {rate:.1%} "
                    f"({survivors}/{total} programs). "
                    f"{'Grammar is productive.' if rate > 0.03 else 'Grammar needs tuning.'}"
                ),
                "category": "pattern",
                "confidence": round(conf, 2),
            })

        return insights

    @staticmethod
    def _parse_criteria_text(success_criteria: str) -> List[str]:
        if not isinstance(success_criteria, str) or not success_criteria.strip():
            return []
        items: List[str] = []
        for part in re.split(r"\n|;|\|", success_criteria):
            text = part.strip()
            if not text:
                continue
            text = re.sub(r"^[-*•]\s*", "", text)
            text = re.sub(r"^\d+[.)]\s*", "", text)
            if text:
                items.append(text)
        return items

    @staticmethod
    def _parse_threshold(text: str) -> Optional[Dict]:
        symbol_match = re.search(r"(<=|>=|<|>|=)\s*(\d+(?:\.\d+)?)(\s*%)?", text)
        if symbol_match:
            return {
                "op": symbol_match.group(1),
                "value": float(symbol_match.group(2)),
                "is_percent": bool(symbol_match.group(3)),
            }

        phrase_patterns = [
            (r"at least\s*(\d+(?:\.\d+)?)(\s*%)?", ">="),
            (r"no more than\s*(\d+(?:\.\d+)?)(\s*%)?", "<="),
            (r"less than\s*(\d+(?:\.\d+)?)(\s*%)?", "<"),
            (r"greater than\s*(\d+(?:\.\d+)?)(\s*%)?", ">"),
        ]
        for pattern, op in phrase_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    "op": op,
                    "value": float(match.group(1)),
                    "is_percent": bool(match.group(2)),
                }
        return None

    @staticmethod
    def _infer_criterion_type(text: str) -> str:
        if "baseline" in text or "loss ratio" in text:
            return "baseline"
        if "novelty" in text:
            return "novelty"
        if "stage 1" in text or "stage1" in text or "s1" in text or "survivor" in text:
            return "stage1"
        if "decision" in text or "go/no-go" in text or "go no-go" in text:
            return "decision"
        return "unknown"

    @staticmethod
    def _normalize_threshold(criterion_type: str, threshold: Optional[Dict]) -> Optional[Dict]:
        if not threshold or criterion_type == "decision":
            return threshold
        should_normalize_as_ratio = bool(threshold.get("is_percent")) or float(threshold.get("value", 0)) > 1
        if not should_normalize_as_ratio:
            return threshold
        value = float(threshold.get("value", 0))
        normalized = dict(threshold)
        normalized["value"] = value / 100.0 if value <= 100 else value
        return normalized

    @staticmethod
    def _compare_threshold(observed: Optional[float], threshold: Optional[Dict]) -> Optional[bool]:
        if observed is None or not threshold:
            return None
        op = threshold.get("op")
        value = float(threshold.get("value", 0))
        if op == "<":
            return observed < value
        if op == "<=":
            return observed <= value
        if op == ">":
            return observed > value
        if op == ">=":
            return observed >= value
        if op == "=":
            return abs(observed - value) < 1e-9
        return None

    @staticmethod
    def _threshold_label(criterion_type: str, threshold: Optional[Dict]) -> str:
        if not threshold:
            return ""
        op = threshold.get("op", "")
        value = float(threshold.get("value", 0))
        if criterion_type in {"baseline", "novelty", "stage1"}:
            if criterion_type == "stage1":
                return f" (target {op} {value * 100:.1f}%)"
            return f" (target {op} {value:.3f})"
        return f" (target {op} {value:g})"

    def campaign_success_criteria_tracker(
        self,
        campaign: Dict,
        experiments: List[Dict],
        hypotheses: List[Dict],
        decisions: List[Dict],
    ) -> List[Dict]:
        criteria = self._parse_criteria_text((campaign or {}).get("success_criteria", ""))
        if not criteria:
            return []

        baseline_values = [
            float(exp.get("best_baseline_ratio"))
            for exp in experiments
            if isinstance(exp.get("best_baseline_ratio"), (int, float))
        ]
        novelty_values = [
            float(exp.get("best_novelty_score"))
            for exp in experiments
            if isinstance(exp.get("best_novelty_score"), (int, float))
        ]
        stage1_values = []
        for exp in experiments:
            total = exp.get("n_programs_generated") or exp.get("n_programs") or 0
            passed = exp.get("n_stage1_passed") or 0
            if total:
                stage1_values.append(float(passed) / float(total))

        best_baseline_ratio = min(baseline_values) if baseline_values else None
        best_novelty = max(novelty_values) if novelty_values else None
        best_stage1_rate = max(stage1_values) if stage1_values else None
        experiment_count = len(experiments)
        hypothesis_count = len(hypotheses)
        decision_count = len(decisions)

        tracker: List[Dict] = []
        for index, criterion in enumerate(criteria):
            text = criterion.lower()
            criterion_type = self._infer_criterion_type(text)
            threshold = self._normalize_threshold(criterion_type, self._parse_threshold(text))
            item = {
                "id": f"{index}-{criterion}",
                "criterion": criterion,
                "criterion_type": criterion_type,
                "status": "not_yet",
                "observed_text": "No mapped metric yet (criterion type not recognized).",
            }

            if criterion_type == "baseline":
                observed = best_baseline_ratio
                passed = self._compare_threshold(observed, threshold) if threshold else (
                    observed < 1.0 if observed is not None else None
                )
                item["status"] = (
                    "not_yet" if passed is None else "pass" if passed else "at_risk" if experiment_count > 0 else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best baseline ratio {observed:.3f}{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "baseline ratio not yet measured"

            elif criterion_type == "novelty":
                observed = best_novelty
                passed = self._compare_threshold(observed, threshold) if threshold else (
                    observed >= 0.7 if observed is not None else None
                )
                item["status"] = (
                    "not_yet" if passed is None else "pass" if passed else "at_risk" if experiment_count > 0 else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best novelty {observed:.3f}{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "novelty signal not yet available"

            elif criterion_type == "stage1":
                observed = best_stage1_rate
                passed = self._compare_threshold(observed, threshold) if threshold else (
                    observed >= 0.05 if observed is not None else None
                )
                item["status"] = (
                    "not_yet" if passed is None else "pass" if passed else "at_risk" if experiment_count > 0 else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best S1 rate {observed * 100:.1f}%{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "S1 evidence not yet available"

            elif criterion_type == "decision":
                observed = float(decision_count)
                passed = self._compare_threshold(observed, threshold) if threshold else observed > 0
                item["status"] = "pass" if passed else "at_risk" if hypothesis_count > 0 else "not_yet"
                item["observed_text"] = (
                    f"{decision_count} decision{'s' if decision_count != 1 else ''} logged"
                    f"{self._threshold_label(criterion_type, threshold)}"
                )

            tracker.append(item)

        return tracker

    def negative_results_synthesis(self) -> Dict:
        """Aggregate repeatedly failed patterns into a "do not pursue" list.

        Combines zero-success ops, dominant error types, anti-correlated
        structural features, and refuted hypotheses into a single report.
        """
        result: Dict = {
            "failed_ops": [],
            "weak_ops": [],
            "dominant_errors": [],
            "anti_patterns": [],
            "toxic_bigrams": [],
            "refuted_hypotheses": [],
            "summary": "",
        }

        # 1. Ops with 0% S1 rate and sufficient sample size
        op_rates = self.op_success_rates()
        min_usage = 5
        for op_name, stats in sorted(
            op_rates.items(), key=lambda x: -(x[1].get("n_used", 0))
        ):
            n_used = stats.get("n_used", 0)
            s1_rate = stats.get("s1_rate", 0)
            if n_used >= min_usage and s1_rate == 0:
                s0_rate = stats.get("s0_rate", 0)
                result["failed_ops"].append({
                    "op_name": op_name,
                    "n_used": n_used,
                    "s0_rate": round(s0_rate, 3),
                    "s1_rate": 0.0,
                    "failure_stage": (
                        "compilation" if s0_rate < 0.5
                        else "learning"
                    ),
                    "confidence": round(min(0.95, 0.4 + n_used / 100), 2),
                })

        # 1b. Weak ops: nonzero but poor S1 rate (soft penalty candidates).
        # These shouldn't be hard-excluded but should be selected less often.
        mean_s1 = 0.0
        s1_vals = [s.get("s1_rate", 0) for s in op_rates.values()
                   if s.get("n_used", 0) >= min_usage]
        if s1_vals:
            mean_s1 = sum(s1_vals) / len(s1_vals)
        weak_threshold = max(mean_s1 * 0.5, 0.20)
        for op_name, stats in sorted(
            op_rates.items(), key=lambda x: x[1].get("s1_rate", 0)
        ):
            n_used = stats.get("n_used", 0)
            s1_rate = stats.get("s1_rate", 0)
            if n_used >= min_usage and 0 < s1_rate <= weak_threshold:
                # Soft penalty: linearly scale from 0.2 (at s1=0) to 1.0 (at threshold)
                penalty = round(max(0.2, s1_rate / weak_threshold), 2)
                result["weak_ops"].append({
                    "op_name": op_name,
                    "n_used": n_used,
                    "s1_rate": round(s1_rate, 3),
                    "penalty_weight": penalty,
                    "threshold": round(weak_threshold, 3),
                })

        # 2. Dominant error types (top 10 by count)
        failures = self.failure_patterns()
        total_failures = sum(v["total"] for v in failures.values())
        for error_type, info in sorted(
            failures.items(), key=lambda x: -x[1]["total"]
        )[:10]:
            pct = info["total"] / total_failures if total_failures > 0 else 0
            top_stage = max(
                info.get("by_stage", {}).items(),
                key=lambda x: x[1],
                default=("unknown", 0),
            )
            result["dominant_errors"].append({
                "error_type": error_type,
                "count": info["total"],
                "percentage": round(pct * 100, 1),
                "primary_stage": top_stage[0],
                "by_stage": info.get("by_stage", {}),
            })

        # 3. Anti-correlated structural features (negative correlations)
        correlations = self.structural_correlations()
        for metric, effect in sorted(
            correlations.items(), key=lambda x: x[1]
        ):
            if effect < -0.15:
                name = metric.replace("graph_", "").replace("_", " ")
                result["anti_patterns"].append({
                    "feature": name,
                    "metric": metric,
                    "correlation": round(effect, 3),
                    "interpretation": (
                        f"Higher {name} is associated with lower S1 success"
                    ),
                })

        # 4. Refuted hypotheses from insights table
        try:
            rows = self.nb.conn.execute("""
                SELECT content, confidence, supporting_evidence, timestamp
                FROM insights
                WHERE status = 'refuted'
                ORDER BY timestamp DESC
                LIMIT 20
            """).fetchall()
            for r in rows:
                result["refuted_hypotheses"].append({
                    "content": r["content"],
                    "confidence": r["confidence"],
                    "evidence": r["supporting_evidence"],
                    "timestamp": r["timestamp"],
                })
        except Exception:
            pass

        # 5. Toxic op-pair bigrams from failure_signatures table
        try:
            blocklist = self.nb.get_failure_signature_blocklist()
            for sig, penalty in sorted(blocklist.items(), key=lambda x: x[1]):
                op1, op2 = sig.split("->") if "->" in sig else (sig, "unknown")
                cat1, cat2 = "unknown", "unknown"
                try:
                    cat1 = get_primitive(op1).category.value
                except Exception:
                    pass
                try:
                    cat2 = get_primitive(op2).category.value
                except Exception:
                    pass

                result["toxic_bigrams"].append({
                    "pattern": sig,
                    "op1": op1,
                    "op2": op2,
                    "cat1": cat1,
                    "cat2": cat2,
                    "penalty": penalty,
                })
        except Exception:
            pass

        # 6. Summary text
        n_ops = len(result["failed_ops"])
        n_weak = len(result["weak_ops"])
        n_errs = len(result["dominant_errors"])
        n_anti = len(result["anti_patterns"])
        n_toxic = len(result["toxic_bigrams"])
        n_ref = len(result["refuted_hypotheses"])
        parts = []
        if n_ops:
            op_names = ", ".join(o["op_name"] for o in result["failed_ops"][:5])
            parts.append(f"{n_ops} ops with 0% S1 rate ({op_names})")
        if n_weak:
            weak_names = ", ".join(o["op_name"] for o in result["weak_ops"][:5])
            parts.append(f"{n_weak} weak ops soft-penalized ({weak_names})")
        if n_errs:
            parts.append(
                f"{n_errs} error types, top: {result['dominant_errors'][0]['error_type']}"
                f" ({result['dominant_errors'][0]['count']} occurrences)"
            )
        if n_anti:
            parts.append(f"{n_anti} anti-correlated structural features")
        if n_toxic:
            top_toxic = ", ".join(t["pattern"] for t in result["toxic_bigrams"][:3])
            parts.append(f"{n_toxic} toxic op-pair patterns ({top_toxic})")
        if n_ref:
            parts.append(f"{n_ref} refuted hypotheses")
        result["summary"] = "; ".join(parts) if parts else "No negative results to report yet."

        return result

    def decision_outcome_analysis(self, lookback: int = 30) -> Dict:
        """Analyze which selection decisions led to successful vs failed experiments.

        Joins mode_selection decisions with subsequent experiment outcomes to
        compute per-mode success rates.  Returns a dict with per-mode stats and
        a ``mode_penalties`` dict mapping mode names to penalty multipliers
        (< 1.0 for consistently failing modes).
        """
        result: Dict = {
            "mode_stats": {},
            "mode_penalties": {},
            "total_decisions": 0,
            "analysis_window": lookback,
        }

        try:
            # Get recent mode_selection decisions with their chosen mode
            rows = self.nb.conn.execute(
                """SELECT decision_id, timestamp, chosen_experiments_json
                   FROM selection_decisions
                   WHERE context = 'mode_selection'
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (lookback,),
            ).fetchall()
        except Exception:
            return result

        if not rows:
            return result

        # For each decision, find the experiment that started shortly after
        # and check its outcome
        mode_outcomes: Dict[str, Dict] = {}  # mode -> {total, s1_any, s1_total}
        for row in rows:
            chosen_json = row["chosen_experiments_json"]
            if not chosen_json:
                continue
            try:
                chosen = json.loads(chosen_json) if isinstance(chosen_json, str) else chosen_json
            except (json.JSONDecodeError, TypeError):
                continue
            if not chosen:
                continue
            mode = chosen[0].get("mode", "synthesis") if isinstance(chosen[0], dict) else "synthesis"
            decision_ts = row["timestamp"]

            # Find the next completed experiment after this decision
            exp_row = self.nb.conn.execute(
                """SELECT experiment_type, n_stage1_passed, n_programs_generated,
                          best_loss_ratio, best_novelty_score
                   FROM experiments
                   WHERE timestamp >= ? AND status = 'completed'
                   ORDER BY timestamp ASC
                   LIMIT 1""",
                (decision_ts,),
            ).fetchone()

            if exp_row is None:
                continue

            if mode not in mode_outcomes:
                mode_outcomes[mode] = {"total": 0, "s1_any": 0, "s1_total": 0,
                                       "programs_total": 0}
            stats = mode_outcomes[mode]
            stats["total"] += 1
            s1 = exp_row["n_stage1_passed"] or 0
            stats["s1_total"] += s1
            stats["programs_total"] += exp_row["n_programs_generated"] or 0
            if s1 > 0:
                stats["s1_any"] += 1

        result["total_decisions"] = sum(s["total"] for s in mode_outcomes.values())

        # Compute per-mode statistics and penalties
        overall_success_rate = 0.0
        total_with_s1 = sum(s["s1_any"] for s in mode_outcomes.values())
        total_decisions = result["total_decisions"]
        if total_decisions > 0:
            overall_success_rate = total_with_s1 / total_decisions

        for mode, stats in mode_outcomes.items():
            n = stats["total"]
            success_rate = stats["s1_any"] / n if n > 0 else 0
            s1_per_program = (stats["s1_total"] / stats["programs_total"]
                              if stats["programs_total"] > 0 else 0)
            result["mode_stats"][mode] = {
                "n_decisions": n,
                "success_rate": round(success_rate, 3),
                "s1_per_program": round(s1_per_program, 4),
                "s1_total": stats["s1_total"],
                "consecutive_failures": 0,  # filled below
            }

            # Count consecutive recent failures for this mode
            consec = 0
            for row2 in rows:
                chosen2 = row2["chosen_experiments_json"]
                try:
                    c2 = json.loads(chosen2) if isinstance(chosen2, str) else chosen2
                except (json.JSONDecodeError, TypeError):
                    continue
                if not c2 or not isinstance(c2[0], dict):
                    continue
                if c2[0].get("mode") != mode:
                    continue
                exp2 = self.nb.conn.execute(
                    """SELECT n_stage1_passed FROM experiments
                       WHERE timestamp >= ? AND status = 'completed'
                       ORDER BY timestamp ASC LIMIT 1""",
                    (row2["timestamp"],),
                ).fetchone()
                if exp2 and (exp2["n_stage1_passed"] or 0) == 0:
                    consec += 1
                else:
                    break
            result["mode_stats"][mode]["consecutive_failures"] = consec

            # Penalty: reduce weight for modes that consistently fail.
            # Minimum 5 decisions before penalizing to avoid noise.
            # Penalty scales from 1.0 (at or above average) to 0.3 (at 0% success).
            if n >= 5 and overall_success_rate > 0:
                relative = success_rate / max(overall_success_rate, 0.01)
                penalty = round(max(0.3, min(1.0, relative)), 2)
            else:
                penalty = 1.0
            # Extra penalty for recent consecutive failures (3+ in a row)
            if consec >= 3:
                penalty = round(max(0.3, penalty * 0.7), 2)
            result["mode_penalties"][mode] = penalty

        return result

    def efficiency_frontier(self) -> List[Dict]:
        """Find Pareto-optimal programs on loss vs FLOPs/params.

        Returns programs that are not dominated by any other program
        (lower loss AND lower FLOPs simultaneously).
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, graph_fingerprint, final_loss,
                   flops_forward, param_count, novelty_score,
                   loss_ratio, baseline_loss_ratio, graph_json
            FROM program_results
            WHERE stage1_passed = 1
              AND final_loss IS NOT NULL
              AND flops_forward IS NOT NULL
              AND flops_forward > 0
            ORDER BY final_loss ASC
        """).fetchall()

        if not rows:
            return []

        programs = [dict(r) for r in rows]

        # Find Pareto frontier: minimize (loss, flops) in O(n log n)
        programs.sort(key=lambda r: (r["flops_forward"], r["final_loss"]))
        frontier = []
        best_loss = float("inf")
        for p in programs:
            loss = p["final_loss"]
            if loss < best_loss:
                frontier.append(p)
                best_loss = loss

        # Extract ops list only for the final frontier programs
        for p in frontier:
            ops = None
            graph_json = p.get("graph_json")
            if graph_json:
                ops = self._extract_ops_fast(graph_json)
                if ops is None:
                    ops = self._extract_ops_fallback(graph_json)
            p["ops"] = ops or []
            p.pop("graph_json", None)

        return frontier

    def efficiency_frontier_3d(self) -> Dict:
        """Find Pareto-optimal programs on loss vs FLOPs vs compression.

        Extends the 2D frontier to three objectives: minimize loss,
        minimize FLOPs, minimize effective params (compression).
        A program is Pareto-optimal if no other program is better on
        ALL three dimensions simultaneously.

        Returns dict with frontier programs, summary stats, and
        dominated count.
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, graph_fingerprint, final_loss,
                   flops_forward, param_count, novelty_score,
                   loss_ratio, baseline_loss_ratio, graph_json,
                   routing_savings_ratio, compression_ratio
            FROM program_results
            WHERE stage1_passed = 1
              AND final_loss IS NOT NULL
              AND flops_forward IS NOT NULL
              AND flops_forward > 0
            ORDER BY final_loss ASC
        """).fetchall()

        if not rows:
            return {"frontier": [], "total_candidates": 0,
                    "frontier_count": 0, "dominated_count": 0}

        programs = [dict(r) for r in rows]

        # Effective params: param_count * compression_ratio (or just param_count)
        for p in programs:
            cr = p.get("compression_ratio")
            pc = p.get("param_count") or 0
            p["effective_params"] = pc * cr if (cr and cr > 0) else pc

        # 3D Pareto: minimize (final_loss, flops_forward, effective_params)
        # A point is dominated if another point is <= on all 3 and < on at least 1
        n = len(programs)
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            li, fi, ei = (programs[i]["final_loss"],
                          programs[i]["flops_forward"],
                          programs[i]["effective_params"])
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                lj, fj, ej = (programs[j]["final_loss"],
                              programs[j]["flops_forward"],
                              programs[j]["effective_params"])
                # j dominates i if j <= i on all and j < i on at least one
                if lj <= li and fj <= fi and ej <= ei:
                    if lj < li or fj < fi or ej < ei:
                        dominated[i] = True
                        break

        frontier = []
        for i, p in enumerate(programs):
            p["is_pareto_optimal"] = not dominated[i]
            if not dominated[i]:
                frontier.append(p)

        # Extract ops for frontier programs
        for p in frontier:
            ops = None
            graph_json = p.get("graph_json")
            if graph_json:
                ops = self._extract_ops_fast(graph_json)
                if ops is None:
                    ops = self._extract_ops_fallback(graph_json)
            p["ops"] = ops or []
            p.pop("graph_json", None)

        # Clean graph_json from non-frontier programs too (not returned but tidy)
        dominated_count = sum(1 for d in dominated if d)

        return {
            "frontier": frontier,
            "total_candidates": n,
            "frontier_count": len(frontier),
            "dominated_count": dominated_count,
        }

    def moe_activation_telemetry(self) -> Dict:
        """Aggregate MoE expert utilization and routing quality from experiment history.

        Analyzes programs with routing telemetry to compute:
        - Per-expert utilization distribution
        - Load balance score (entropy-based)
        - Routing quality correlation with loss
        - Capacity overflow trends
        """
        rows = self.nb.conn.execute("""
            SELECT routing_mode, routing_tokens_total, routing_tokens_processed,
                   routing_tokens_skipped, routing_drop_rate,
                   routing_utilization_entropy, routing_capacity_overflow_count,
                   routing_confidence_mean, routing_confidence_std,
                   routing_expert_utilization_json,
                   loss_ratio, stage1_passed, graph_json
            FROM program_results
            WHERE routing_mode IS NOT NULL
        """).fetchall()

        if not rows:
            return {"n_programs": 0, "experts": [], "summary": {}}

        n_programs = len(rows)
        sum_entropy = 0.0
        n_entropy = 0
        sum_drop_rate = 0.0
        n_drop = 0
        sum_overflow = 0
        n_overflow = 0
        sum_confidence = 0.0
        n_confidence = 0
        n_survived = 0
        sum_loss = 0.0
        n_loss = 0
        best_loss = None

        # Per-expert aggregation
        expert_totals: Dict[int, Dict] = {}

        for row in rows:
            record = dict(row)

            if record.get("stage1_passed"):
                n_survived += 1

            loss = self._as_float(record.get("loss_ratio"))
            if loss is not None:
                sum_loss += loss
                n_loss += 1
                if best_loss is None or loss < best_loss:
                    best_loss = loss

            entropy = self._as_float(record.get("routing_utilization_entropy"))
            if entropy is not None:
                sum_entropy += entropy
                n_entropy += 1

            drop_rate = self._as_float(record.get("routing_drop_rate"))
            if drop_rate is not None:
                sum_drop_rate += drop_rate
                n_drop += 1

            overflow = record.get("routing_capacity_overflow_count")
            if overflow is not None:
                sum_overflow += int(overflow)
                n_overflow += 1

            conf = self._as_float(record.get("routing_confidence_mean"))
            if conf is not None:
                sum_confidence += conf
                n_confidence += 1

            # Parse per-expert utilization
            util_json = record.get("routing_expert_utilization_json")
            if util_json:
                try:
                    utilizations = json.loads(util_json)
                    if isinstance(utilizations, list):
                        for eidx, util_val in enumerate(utilizations):
                            bucket = expert_totals.setdefault(eidx, {
                                "expert_id": eidx,
                                "sum_utilization": 0.0,
                                "n_samples": 0,
                                "max_utilization": 0.0,
                                "min_utilization": 1.0,
                            })
                            u = float(util_val)
                            bucket["sum_utilization"] += u
                            bucket["n_samples"] += 1
                            bucket["max_utilization"] = max(bucket["max_utilization"], u)
                            bucket["min_utilization"] = min(bucket["min_utilization"], u)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Build per-expert summary
        experts = []
        for eidx in sorted(expert_totals.keys()):
            b = expert_totals[eidx]
            n = b["n_samples"]
            avg = b["sum_utilization"] / n if n > 0 else 0.0
            experts.append({
                "expert_id": eidx,
                "avg_utilization": round(avg, 4),
                "max_utilization": round(b["max_utilization"], 4),
                "min_utilization": round(b["min_utilization"], 4),
                "n_samples": n,
            })

        # Compute load balance score from average utilizations
        # Perfect balance = all experts have equal utilization
        # Score = 1 - coefficient of variation (capped at 1.0)
        load_balance_score = None
        if experts:
            utils = [e["avg_utilization"] for e in experts]
            mean_u = sum(utils) / len(utils) if utils else 0
            if mean_u > 0 and len(utils) > 1:
                var = sum((u - mean_u) ** 2 for u in utils) / len(utils)
                cv = (var ** 0.5) / mean_u
                load_balance_score = round(max(0.0, min(1.0, 1.0 - cv)), 4)

        summary = {
            "n_programs": n_programs,
            "n_survived": n_survived,
            "survival_rate": round(n_survived / n_programs, 4) if n_programs > 0 else 0.0,
            "avg_loss_ratio": round(sum_loss / n_loss, 4) if n_loss > 0 else None,
            "best_loss_ratio": round(best_loss, 4) if best_loss is not None else None,
            "avg_utilization_entropy": round(sum_entropy / n_entropy, 4) if n_entropy > 0 else None,
            "avg_drop_rate": round(sum_drop_rate / n_drop, 4) if n_drop > 0 else None,
            "avg_capacity_overflow": round(sum_overflow / n_overflow, 2) if n_overflow > 0 else None,
            "avg_routing_confidence": round(sum_confidence / n_confidence, 4) if n_confidence > 0 else None,
            "load_balance_score": load_balance_score,
            "n_experts_observed": len(experts),
        }

        return {
            "n_programs": n_programs,
            "experts": experts,
            "summary": summary,
        }

    def sparse_quant_codesign_summary(self) -> Dict:
        """Aggregate quality-retention-per-byte across programs with sparse+quant data.

        Looks for programs that have both sparsity metrics (sparse_density_*)
        and pruning quality retention data, then estimates combined
        sparse+quant efficiency potential.

        Returns summary with per-program and aggregate statistics.
        """
        rows = self.nb.conn.execute(
            "SELECT result_id, graph_json, stage1_passed, final_loss, "
            "sparse_density_mean, sparse_density_last, "
            "pruning_method, pruning_target_sparsity, pruning_actual_sparsity, "
            "pruning_quality_retention, pruning_dense_eval_loss, "
            "pruning_pruned_eval_loss, pruning_n_params_total "
            "FROM program_results "
            "WHERE (sparse_density_mean IS NOT NULL OR pruning_method IS NOT NULL)"
        ).fetchall()
        if not rows:
            return {"n_programs": 0, "programs": [], "summary": {}}

        programs = []
        sum_retention = 0.0
        sum_compression = 0.0
        n_with_retention = 0
        best_qpb = None  # quality-per-byte
        best_qpb_id = None

        for row in rows:
            pid = row[0]
            graph_json = row[1]
            survived = row[2]
            best_loss = row[3]
            density_mean = row[4]
            density_last = row[5]
            prune_method = row[6]
            prune_target = row[7]
            prune_actual = row[8]
            prune_retention = row[9]
            dense_loss = row[10]
            pruned_loss = row[11]
            n_params = row[12]

            # Detect compression ops in graph
            record = {"graph_json": graph_json or ""}
            compression_ops = self._detect_compression_ops(record)

            # Estimate effective density
            effective_density = density_last or density_mean or 1.0
            if prune_actual is not None and prune_actual > 0:
                effective_density = min(effective_density, 1.0 - prune_actual)

            # Estimate bytes per param under INT8 quant + sparsity
            bytes_sparse_quant_int8 = effective_density * 1.0  # 8-bit = 1 byte
            bytes_sparse_quant_int4 = effective_density * 0.5  # 4-bit = 0.5 byte
            bytes_original = 4.0  # float32

            compression_int8 = bytes_original / max(bytes_sparse_quant_int8, 0.01)
            compression_int4 = bytes_original / max(bytes_sparse_quant_int4, 0.01)

            # Quality retention
            retention = None
            if prune_retention is not None:
                retention = float(prune_retention)
            elif dense_loss is not None and pruned_loss is not None and pruned_loss > 0:
                retention = float(dense_loss) / float(pruned_loss)

            # Quality-per-byte (higher = better)
            qpb_int8 = None
            if retention is not None:
                qpb_int8 = retention * compression_int8
                sum_retention += retention
                sum_compression += compression_int8
                n_with_retention += 1
                if best_qpb is None or qpb_int8 > best_qpb:
                    best_qpb = qpb_int8
                    best_qpb_id = pid

            entry = {
                "program_id": pid,
                "survived": bool(survived),
                "best_loss": round(float(best_loss), 6) if best_loss else None,
                "effective_density": round(float(effective_density), 4),
                "compression_ops": compression_ops,
                "pruning_method": prune_method,
                "quality_retention": round(retention, 4) if retention else None,
                "compression_ratio_int8": round(compression_int8, 2),
                "compression_ratio_int4": round(compression_int4, 2),
                "quality_per_byte_int8": round(qpb_int8, 4) if qpb_int8 else None,
                "n_params": n_params,
            }
            programs.append(entry)

        summary = {
            "n_programs": len(programs),
            "n_with_retention": n_with_retention,
            "avg_quality_retention": (
                round(sum_retention / n_with_retention, 4) if n_with_retention > 0 else None
            ),
            "avg_compression_ratio_int8": (
                round(sum_compression / n_with_retention, 2) if n_with_retention > 0 else None
            ),
            "best_quality_per_byte": round(best_qpb, 4) if best_qpb else None,
            "best_qpb_program_id": best_qpb_id,
        }

        return {
            "n_programs": len(programs),
            "programs": programs,
            "summary": summary,
        }

    def get_current_grammar_weights(self) -> Dict[str, float]:
        """Get the default grammar weights for comparison."""
        return dict(GrammarConfig().category_weights)


class RefinementAnalyzer:
    """Data-driven refinement advisor that examines programs against population success data.

    Produces concrete recommendations (swap/add/remove ops) with evidence,
    behavioral gap analysis, and grammar hints for smarter mutations.
    """

    # Behavioral gap → improvement op suggestions
    _GAP_HINTS: Dict[str, List[str]] = {
        "fp_isotropy": ["rmsnorm", "layer_norm", "batch_norm"],
        "fp_interaction_locality": ["softmax_attention", "state_space", "fourier_mixing"],
        "fp_sensitivity_uniformity": ["residual", "skip_connection", "highway_gate"],
        "fp_rank_ratio": ["low_rank_proj", "linear_proj", "bottleneck_proj"],
        "fp_interaction_sparsity": ["topk_gate", "threshold_gate", "sparse_linear"],
        "fp_intrinsic_dim": ["grouped_linear", "low_rank_proj", "bottleneck_proj"],
        "fp_jacobian_spectral_norm": ["rmsnorm", "layer_norm", "residual", "highway_gate"],
    }

    # Human-readable metric labels
    _METRIC_LABELS: Dict[str, str] = {
        "fp_isotropy": "Isotropy",
        "fp_interaction_locality": "Interaction Locality",
        "fp_interaction_sparsity": "Interaction Sparsity",
        "fp_interaction_symmetry": "Interaction Symmetry",
        "fp_interaction_hierarchy": "Interaction Hierarchy",
        "fp_intrinsic_dim": "Intrinsic Dimensionality",
        "fp_rank_ratio": "Rank Ratio",
        "fp_jacobian_spectral_norm": "Jacobian Spectral Norm",
        "fp_jacobian_effective_rank": "Jacobian Effective Rank",
        "fp_sensitivity_uniformity": "Sensitivity Uniformity",
    }

    # Fingerprint metrics to analyze
    _FP_METRICS = [
        "fp_isotropy", "fp_interaction_locality", "fp_interaction_sparsity",
        "fp_interaction_symmetry", "fp_interaction_hierarchy", "fp_intrinsic_dim",
        "fp_rank_ratio", "fp_sensitivity_uniformity", "fp_jacobian_spectral_norm",
    ]

    def __init__(self, analytics: ExperimentAnalytics):
        self.analytics = analytics
        self.nb = analytics.nb

    def analyze_program_for_refinement(
        self, result_id: str, program_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze a program against population data and produce refinement recommendations."""
        graph_json = program_row.get("graph_json")
        if isinstance(graph_json, dict):
            graph_json = json.dumps(graph_json)
        program_ops = ExperimentAnalytics._extract_ops_fast(graph_json or "") or []

        # Map ops to categories
        op_categories: Dict[str, str] = {}
        for op_name in program_ops:
            try:
                prim = get_primitive(op_name)
                op_categories[op_name] = prim.category.value
            except Exception:
                op_categories[op_name] = "unknown"

        # Fetch population op stats
        op_stats_rows = self.nb.get_op_success_rates()
        pop_s1_rates: Dict[str, float] = {}
        pop_s0_rates: Dict[str, float] = {}
        pop_n_used: Dict[str, int] = {}
        pop_categories: Dict[str, str] = {}

        for row in op_stats_rows:
            op = str(row.get("op_name", ""))
            n_used = int(row.get("n_used") or 0)
            n_s1 = int(row.get("n_stage1_passed") or 0)
            n_s0 = int(row.get("n_stage0_passed") or 0)
            pop_n_used[op] = n_used
            pop_s1_rates[op] = n_s1 / n_used if n_used > 0 else 0.0
            pop_s0_rates[op] = n_s0 / n_used if n_used > 0 else 0.0
            try:
                prim = get_primitive(op)
                pop_categories[op] = prim.category.value
            except Exception:
                pop_categories[op] = "unknown"

        # Compute mean S1 rate across ops with sufficient data
        qualified_rates = [
            pop_s1_rates[op] for op, n in pop_n_used.items() if n >= 5
        ]
        mean_s1_rate = (
            sum(qualified_rates) / len(qualified_rates) if qualified_rates else 0.0
        )

        n_programs_total = sum(pop_n_used.values()) // max(1, len(pop_n_used)) if pop_n_used else 0
        n_stage1_passed = sum(
            int(r.get("n_stage1_passed") or 0) for r in op_stats_rows
        ) // max(1, len(op_stats_rows)) if op_stats_rows else 0

        if not program_ops and not op_stats_rows:
            return self._empty_analysis(result_id, "no_data")

        # Per-op health cards
        op_health = self._build_op_health(
            program_ops, op_categories, pop_s1_rates, pop_s0_rates,
            pop_n_used, pop_categories, mean_s1_rate,
        )

        # Recommended additions
        recommended_additions = self._build_recommended_additions(
            program_ops, pop_s1_rates, pop_n_used, pop_categories, mean_s1_rate,
        )

        # Behavioral gap analysis
        behavioral_gaps = self._build_behavioral_gaps(program_row)

        # Build recipe
        recipe = self._build_recipe(
            op_health, behavioral_gaps, program_row, mean_s1_rate,
        )

        # Brittleness advice (Aria's technical insight)
        brittleness_advice = self._build_brittleness_advice(program_row)

        analysis_quality = "full" if qualified_rates else ("partial" if op_stats_rows else "no_data")

        return {
            "result_id": result_id,
            "graph_fingerprint": program_row.get("graph_fingerprint", ""),
            "program_ops": program_ops,
            "op_health": op_health,
            "recommended_additions": recommended_additions,
            "behavioral_gaps": behavioral_gaps,
            "recipe": recipe,
            "brittleness_advice": brittleness_advice,
            "population_stats": {
                "n_programs_total": n_programs_total,
                "n_stage1_passed": n_stage1_passed,
                "mean_s1_rate": round(mean_s1_rate, 4),
            },
            "analysis_quality": analysis_quality,
        }

    def _build_op_health(
        self,
        program_ops: List[str],
        op_categories: Dict[str, str],
        pop_s1_rates: Dict[str, float],
        pop_s0_rates: Dict[str, float],
        pop_n_used: Dict[str, int],
        pop_categories: Dict[str, str],
        mean_s1_rate: float,
    ) -> List[Dict[str, Any]]:
        """Build per-op health cards with recommendations."""
        cards: List[Dict[str, Any]] = []
        program_op_set = set(program_ops)

        for op_name in program_ops:
            n = pop_n_used.get(op_name, 0)
            s1 = pop_s1_rates.get(op_name, 0.0)
            s0 = pop_s0_rates.get(op_name, 0.0)
            cat = op_categories.get(op_name, "unknown")

            # Classify health
            if n < 5:
                health = "untested"
                recommendation = "investigate"
            elif s0 < 0.5 and n >= 5:
                health = "risky"
                recommendation = "swap"
            elif s1 < mean_s1_rate * 0.5 and n >= 10:
                health = "weak"
                recommendation = "swap"
            elif s1 >= mean_s1_rate * 1.2 and n >= 5:
                health = "strong"
                recommendation = "keep"
            else:
                health = "neutral"
                recommendation = "keep"

            # Find swap candidates: same category, higher S1, not in program
            swap_candidates: List[Dict[str, Any]] = []
            if recommendation in ("swap", "investigate"):
                same_cat_ops = [
                    (op, pop_s1_rates[op])
                    for op, c in pop_categories.items()
                    if c == cat and op not in program_op_set
                    and pop_n_used.get(op, 0) >= 5
                    and pop_s1_rates.get(op, 0.0) > s1
                ]
                same_cat_ops.sort(key=lambda x: x[1], reverse=True)
                swap_candidates = [
                    {"op_name": op, "s1_rate": round(rate, 4)}
                    for op, rate in same_cat_ops[:3]
                ]

            cards.append({
                "op_name": op_name,
                "category": cat,
                "global_s1_rate": round(s1, 4),
                "global_s0_rate": round(s0, 4),
                "n_used": n,
                "health": health,
                "recommendation": recommendation,
                "swap_candidates": swap_candidates,
            })

        return cards

    def _build_recommended_additions(
        self,
        program_ops: List[str],
        pop_s1_rates: Dict[str, float],
        pop_n_used: Dict[str, int],
        pop_categories: Dict[str, str],
        mean_s1_rate: float,
    ) -> List[Dict[str, Any]]:
        """Find high-performing ops not in this program."""
        program_op_set = set(program_ops)
        candidates: List[Dict[str, Any]] = []

        for op_name, s1 in pop_s1_rates.items():
            if op_name in program_op_set:
                continue
            n = pop_n_used.get(op_name, 0)
            if n < 5 or s1 < mean_s1_rate:
                continue
            freq_score = s1 * math.log1p(n)
            candidates.append({
                "op_name": op_name,
                "category": pop_categories.get(op_name, "unknown"),
                "global_s1_rate": round(s1, 4),
                "top_performer_frequency": n,
                "score": freq_score,
                "rationale": f"S1 rate {s1:.1%} across {n} uses, above population mean",
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for c in candidates:
            del c["score"]
        return candidates[:5]

    def _build_behavioral_gaps(
        self, program_row: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare program fingerprint metrics to population S1 survivor means."""
        gaps: List[Dict[str, Any]] = []

        # Fetch S1 survivor stats from recent experiments
        try:
            rows = self.nb.conn.execute(
                """SELECT {} FROM program_results
                   WHERE stage1_passed = 1""".format(
                    ", ".join(self._FP_METRICS)
                )
            ).fetchall()
        except Exception:
            return gaps

        if len(rows) < 3:
            return gaps

        # Compute population means and stds
        pop_stats: Dict[str, Tuple[float, float]] = {}
        for metric in self._FP_METRICS:
            values = [
                float(r[metric]) for r in rows
                if r[metric] is not None
                and not math.isnan(float(r[metric]))
                and not math.isinf(float(r[metric]))
            ]
            if len(values) >= 3:
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                pop_stats[metric] = (mean_val, max(std_val, 1e-8))

        for metric, (pop_mean, pop_std) in pop_stats.items():
            prog_val = program_row.get(metric)
            if prog_val is None:
                continue
            try:
                prog_val = float(prog_val)
                if math.isnan(prog_val) or math.isinf(prog_val):
                    continue
            except (TypeError, ValueError):
                continue

            z_score = (prog_val - pop_mean) / pop_std
            abs_z = abs(z_score)
            if abs_z < 1.0:
                continue

            severity = "low" if abs_z < 1.5 else ("medium" if abs_z < 2.0 else "high")
            improvement_ops = self._GAP_HINTS.get(metric, [])

            gaps.append({
                "metric": metric,
                "label": self._METRIC_LABELS.get(metric, metric),
                "program_value": round(prog_val, 4),
                "population_mean": round(pop_mean, 4),
                "population_std": round(pop_std, 4),
                "z_score": round(z_score, 2),
                "severity": severity,
                "improvement_ops": improvement_ops,
            })

        gaps.sort(key=lambda g: abs(g["z_score"]), reverse=True)
        return gaps

    def _build_recipe(
        self,
        op_health: List[Dict[str, Any]],
        behavioral_gaps: List[Dict[str, Any]],
        program_row: Dict[str, Any],
        mean_s1_rate: float,
    ) -> Dict[str, Any]:
        """Combine analysis into actionable recipe with grammar hints."""
        risky_ops = [h for h in op_health if h["health"] == "risky" and h["n_used"] >= 8]
        weak_ops = [h for h in op_health if h["health"] == "weak"]
        significant_gaps = [g for g in behavioral_gaps if g["severity"] in ("medium", "high")]

        # Determine exclude/boost ops
        exclude_ops = [h["op_name"] for h in risky_ops]
        boost_ops: Dict[str, float] = {}
        for h in op_health:
            if h["health"] == "strong":
                boost_ops[h["op_name"]] = min(3.0, 1.5)
            for sc in h.get("swap_candidates", []):
                boost_ops[sc["op_name"]] = min(3.0, 2.0)

        # Categories to boost from gap hints
        add_categories: Dict[str, float] = {}
        for gap in significant_gaps:
            for op in gap.get("improvement_ops", []):
                try:
                    prim = get_primitive(op)
                    cat = prim.category.value
                    add_categories[cat] = max(add_categories.get(cat, 1.0), 1.5)
                except Exception:
                    pass

        # Determine intent
        loss_ratio = program_row.get("loss_ratio")
        param_count = program_row.get("param_count") or program_row.get("graph_n_params_estimate")
        has_reasonable_loss = isinstance(loss_ratio, (int, float)) and float(loss_ratio) < 1.1
        has_high_params = isinstance(param_count, (int, float)) and float(param_count) > 100000

        if risky_ops:
            recommended_intent = "quality"
            primary_target = f"Replace {len(risky_ops)} risky op(s) with proven alternatives"
            confidence = "high" if len(risky_ops) >= 2 else "medium"
        elif len(weak_ops) >= 2:
            recommended_intent = "quality"
            primary_target = f"Improve {len(weak_ops)} underperforming ops"
            confidence = "medium"
        elif significant_gaps:
            recommended_intent = "novelty"
            top_gap = significant_gaps[0]
            primary_target = f"Address {top_gap['label']} gap (z={top_gap['z_score']:+.1f})"
            confidence = "medium" if len(significant_gaps) >= 2 else "low"
        elif has_high_params and has_reasonable_loss:
            recommended_intent = "compression"
            primary_target = "Reduce parameter count while maintaining quality"
            confidence = "medium"
        else:
            recommended_intent = "balanced"
            primary_target = "General improvement across all dimensions"
            confidence = "low"

        # Human summary
        parts: List[str] = []
        if risky_ops:
            parts.append(f"{len(risky_ops)} risky op(s) should be replaced")
        if weak_ops:
            parts.append(f"{len(weak_ops)} weak op(s) could be improved")
        if significant_gaps:
            gap_names = [g["label"] for g in significant_gaps[:3]]
            parts.append(f"behavioral gaps in {', '.join(gap_names)}")
        if not parts:
            parts.append("No major issues detected; balanced refinement recommended")
        human_summary = "; ".join(parts) + "."

        return {
            "recommended_intent": recommended_intent,
            "confidence": confidence,
            "primary_target": primary_target,
            "grammar_hints": {
                "exclude_ops": exclude_ops,
                "boost_ops": boost_ops,
                "add_categories": add_categories,
            },
            "human_summary": human_summary,
        }

    def _build_brittleness_advice(self, program_row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Aria's technical advice for making architectures less brittle."""
        robustness = program_row.get("investigation_robustness")
        spectral_norm = program_row.get("fp_jacobian_spectral_norm")
        init_std = program_row.get("init_sensitivity_std")
        
        advice_parts = []
        is_brittle = False
        
        if robustness is not None and robustness < 0.5:
            is_brittle = True
            advice_parts.append(
                f"Low robustness ({robustness:.2f}) indicates high sensitivity to training recipes. "
                "Only a fraction of hyperparameter seeds converge successfully."
            )
            
        if spectral_norm is not None and spectral_norm > 15.0:
            is_brittle = True
            advice_parts.append(
                f"High Jacobian spectral norm ({spectral_norm:.1f}) suggests 'exploding' gradient paths "
                "or poor signal propagation. This often leads to training instability."
            )
            
        if init_std is not None and init_std > 0.1:
            is_brittle = True
            advice_parts.append(
                f"High initialization sensitivity ({init_std:.3f}) means the architecture's success "
                "depends heavily on lucky weight initialization."
            )

        if not is_brittle:
            return None

        # Practical remedies
        remedies = [
            "Add 'rmsnorm' or 'layer_norm' before non-linear ops to bound activations.",
            "Use 'residual' or 'skip_connection' paths to improve gradient flow.",
            "Verify that learning rates are sufficiently low for this topology.",
            "If using 'Spectral-Conv', consider adding a small epsilon or weight decay to fourier mixing."
        ]
        
        return {
            "summary": "This architecture is marked as 'Brittle' because it lacks training stability.",
            "diagnosis": " ".join(advice_parts),
            "remedies": remedies,
            "aria_insight": (
                "To make this front-runner reliable, we must tame its variance. "
                "A 0.33 robustness means it's a great 'idea' that currently requires 'perfect' conditions. "
                "Stabilizing it will likely improve its validation score even further."
            )
        }

    def _empty_analysis(self, result_id: str, quality: str) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "result_id": result_id,
            "graph_fingerprint": "",
            "program_ops": [],
            "op_health": [],
            "recommended_additions": [],
            "behavioral_gaps": [],
            "recipe": {
                "recommended_intent": "balanced",
                "confidence": "low",
                "primary_target": "Insufficient data for analysis",
                "grammar_hints": {
                    "exclude_ops": [],
                    "boost_ops": {},
                    "add_categories": {},
                },
                "human_summary": "Insufficient population data for analysis.",
            },
            "population_stats": {
                "n_programs_total": 0,
                "n_stage1_passed": 0,
                "mean_s1_rate": 0.0,
            },
            "analysis_quality": quality,
        }
