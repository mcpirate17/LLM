from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Set
from ...eval.utils import safe_parse_float


class _OpsMixin:
    """Op extraction, compression analysis, and efficiency frontiers."""
    __slots__ = ()

    _OP_NAME_PATTERN = re.compile(r'"op_name"\s*:\s*"([^"]+)"')
    _OP_KEY_PATTERN = re.compile(r'"op"\s*:\s*"([^"]+)"')

    _FULL_QKV_TOKEN_MIXERS: Set[str] = {
        "softmax_attention", "linear_attention", "graph_attention",
        "random_feature_attention", "compressed_attention", "cross_attention_pool",
    }
    _Q_EQ_K_EQ_V_TOKEN_MIXERS: Set[str] = {"shared_qk_attention"}
    _QKV_FREE_TOKEN_MIXERS: Set[str] = {
        "conv_only", "state_space", "fourier_mixing",
        "differentiable_sort", "integral_kernel_mixing",
    }
    _COMPRESSION_FACTORS: Dict[str, float] = {
        "low_rank": 0.55, "shared_basis": 0.5, "hash_trick": 0.35,
        "structured_sparse": 0.4, "kronecker": 0.5, "polynomial": 0.6,
        "residual_quantized": 0.3, "compressed_attention": 0.7,
        "bottleneck": 0.55, "grouped_linear": 0.3, "tied_proj": 0.3,
    }
    _OP_TO_COMPRESSION: Dict[str, str] = {
        "low_rank_proj": "low_rank", "grouped_linear": "grouped_linear",
        "bottleneck_proj": "bottleneck", "shared_basis_proj": "shared_basis",
        "tied_proj": "tied_proj", "nm_sparse_linear": "structured_sparse",
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
                   param_count, graph_n_params_estimate, graph_json
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

