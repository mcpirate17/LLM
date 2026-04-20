from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import heapq
import json
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional


# Lazy-loaded to avoid circular imports at module level
_OP_CATEGORY_CACHE: Dict[str, str] = {}


@lru_cache(maxsize=8192)
def _cached_extract_op_names(graph_json: str) -> tuple[str, ...]:
    from research.scientist.intelligence.graph_ops import extract_unique_graph_ops

    return tuple(extract_unique_graph_ops(graph_json))


@lru_cache(maxsize=8192)
def _cached_extract_template_name(graph_json: str) -> str:
    try:
        data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    except (json.JSONDecodeError, TypeError):
        return ""
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("template") or metadata.get("template_name") or "")


@lru_cache(maxsize=8192)
def _cached_extract_unique_ops(graph_json: str) -> tuple[str, ...]:
    return _cached_extract_op_names(graph_json)


def _get_op_category(op_name: str) -> str:
    """Map an op name to its OpCategory string. Cached for speed."""
    if op_name in _OP_CATEGORY_CACHE:
        return _OP_CATEGORY_CACHE[op_name]
    try:
        from research.synthesis.primitives import get_primitive

        prim = get_primitive(op_name)
        cat = prim.category.value
    except (KeyError, ValueError):
        cat = "unknown"
    _OP_CATEGORY_CACHE[op_name] = cat
    return cat


# Canonical routing/compression/MoE ops — mirrored from grammar.py
_ROUTING_OPS: frozenset = frozenset(
    {
        "entropy_score",
        "token_type_classifier",
        "route_topk",
        "route_lanes",
        "route_recursion",
        "adaptive_lane_mixer",
        "mixed_recursion_gate",
        "early_exit",
        "cascade",
        "speculative",
        "adaptive_recursion",
        "mod_topk",
        "token_merge",
        "relu_gate_routing",
        "moe_topk",
        "moe_2expert",
        "n_way_sparse_router",
        "tropical_moe",
        "topk_gate",
        "tropical_gate",
        "tropical_router",
        "sparse_threshold",
        "lif_neuron",
        "padic_gate",
        "routing_conditioned_compression",
        "progressive_compression_gate",
        "compression_mixture_experts",
        "latent_attention_compressor",
    }
)

# All 11 OpCategory values for the distribution vector
_ALL_CATEGORIES: tuple = (
    "elementwise_unary",
    "elementwise_binary",
    "reduction",
    "linear_algebra",
    "structural",
    "parameterized",
    "mixing",
    "sequence",
    "frequency",
    "math_space",
    "functional",
)


class _AnalyticsMixin:
    """Analytics operations for the Lab Notebook."""

    __slots__ = ()

    def _query_op_stats_sql(
        self, where_sql: str, params: tuple[Any, ...]
    ) -> Optional[List[Dict[str, Any]]]:
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            f"""
            WITH op_rows AS (
                SELECT DISTINCT
                    pr.result_id AS result_id,
                    gpo.op_name AS op_name,
                    pr.stage0_passed AS stage0_passed,
                    pr.stage05_passed AS stage05_passed,
                    pr.stage1_passed AS stage1_passed,
                    pr.loss_ratio AS loss_ratio,
                    pr.novelty_score AS novelty_score,
                    pr.novelty_confidence AS novelty_confidence
                FROM program_results pr
                JOIN program_graph_ops gpo ON gpo.result_id = pr.result_id
                WHERE {where_sql}
            )
            SELECT
                op_name,
                COUNT(*) AS n_used,
                SUM(CASE WHEN stage0_passed THEN 1 ELSE 0 END) AS n_stage0_passed,
                SUM(CASE WHEN stage05_passed THEN 1 ELSE 0 END) AS n_stage05_passed,
                SUM(CASE WHEN stage1_passed THEN 1 ELSE 0 END) AS n_stage1_passed,
                AVG(loss_ratio) AS avg_loss_ratio,
                AVG(novelty_score) AS avg_novelty,
                AVG(novelty_confidence) AS avg_novelty_confidence
            FROM op_rows
            WHERE op_name IS NOT NULL AND op_name <> '' AND op_name <> 'input'
            GROUP BY op_name
            ORDER BY n_stage1_passed DESC, n_used DESC
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def _query_bigram_stats_sql(
        self,
        where_sql: str,
        params: tuple[Any, ...],
        *,
        include_error_types: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        error_select = (
            ", substr(group_concat(DISTINCT CASE "
            "WHEN stage1_passed = 0 AND error_type IS NOT NULL AND error_type <> '' "
            "THEN error_type END), 1, 255) AS error_types"
            if include_error_types
            else ""
        )
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            f"""
            signatures AS (
                SELECT DISTINCT
                    pr.result_id AS result_id,
                    gpp.signature AS signature,
                    pr.stage1_passed AS stage1_passed,
                    pr.loss_ratio AS loss_ratio,
                    pr.novelty_score AS novelty_score,
                    pr.error_type AS error_type
                FROM program_results pr
                JOIN program_graph_pairs gpp ON gpp.result_id = pr.result_id
                WHERE {where_sql}
            )
            SELECT
                signature,
                COUNT(*) AS support,
                SUM(CASE WHEN stage1_passed THEN 1 ELSE 0 END) AS n_stage1_passed,
                SUM(CASE WHEN stage1_passed THEN 0 ELSE 1 END) AS n_failures,
                SUM(CASE WHEN stage1_passed THEN 1 ELSE 0 END) AS n_successes,
                AVG(loss_ratio) AS avg_loss_ratio,
                AVG(novelty_score) AS avg_novelty
                {error_select}
            FROM signatures
            GROUP BY signature
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def _query_graph_feature_rows_sql(self) -> Optional[List[Dict[str, Any]]]:
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                pr.result_id,
                pr.stage1_passed,
                pr.novelty_score,
                pr.graph_category_histogram,
                pr.fp_interaction_sparsity,
                pr.fp_cka_vs_transformer,
                pr.fp_cka_vs_ssm,
                pr.fp_cka_vs_conv,
                COALESCE(gf.template_name, '') AS template_name,
                COALESCE(
                    (SELECT group_concat(op_name, char(31)) FROM (
                        SELECT op_name
                        FROM program_graph_ops
                        WHERE result_id = pr.result_id
                        ORDER BY op_name
                    )),
                    ''
                ) AS ops_blob,
                COALESCE(
                    (SELECT group_concat(signature, char(31)) FROM (
                        SELECT signature
                        FROM program_graph_pairs
                        WHERE result_id = pr.result_id
                        ORDER BY signature
                    )),
                    ''
                ) AS pairs_blob
            FROM program_results pr
            JOIN program_graph_features gf ON gf.result_id = pr.result_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _query_nearest_peers_sql(
        self, graph_fingerprint: str, limit: int = 500
    ) -> Optional[List[Dict[str, Any]]]:
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            WITH fingerprint_rows AS (
                SELECT
                    pr.result_id,
                    pr.graph_fingerprint,
                    pr.loss_ratio,
                    pr.novelty_score,
                    pr.stage1_passed,
                    pr.timestamp,
                    l.tier,
                    l.composite_score
                FROM program_results pr
                LEFT JOIN leaderboard l ON l.result_id = pr.result_id
                WHERE pr.graph_fingerprint IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM program_graph_features gf WHERE gf.result_id = pr.result_id
                  )
            ),
            latest_rows AS (
                SELECT fr.*
                FROM fingerprint_rows fr
                WHERE fr.result_id = (
                    SELECT fr2.result_id
                    FROM fingerprint_rows fr2
                    WHERE fr2.graph_fingerprint = fr.graph_fingerprint
                    ORDER BY fr2.timestamp DESC, fr2.result_id DESC
                    LIMIT 1
                )
            ),
            node_ops AS (
                SELECT DISTINCT
                    graph_fingerprint,
                    op_name
                FROM program_graph_ops
                WHERE graph_fingerprint IS NOT NULL
            ),
            target_ops AS (
                SELECT op_name
                FROM node_ops
                WHERE graph_fingerprint = ?
            ),
            target_count AS (
                SELECT COUNT(*) AS n FROM target_ops
            ),
            peer_counts AS (
                SELECT graph_fingerprint, COUNT(*) AS peer_op_count
                FROM node_ops
                WHERE graph_fingerprint <> ?
                GROUP BY graph_fingerprint
            ),
            intersections AS (
                SELECT
                    n.graph_fingerprint AS graph_fingerprint,
                    COUNT(*) AS overlap
                FROM node_ops n
                JOIN target_ops t ON t.op_name = n.op_name
                WHERE n.graph_fingerprint <> ?
                GROUP BY n.graph_fingerprint
            )
            SELECT
                lr.graph_fingerprint AS fingerprint,
                ROUND(
                    CAST(i.overlap AS FLOAT) /
                    CAST((pc.peer_op_count + tc.n - i.overlap) AS FLOAT),
                    4
                ) AS jaccard_similarity,
                lr.loss_ratio,
                lr.novelty_score,
                lr.stage1_passed,
                COALESCE(lr.tier, '') AS tier,
                lr.composite_score
            FROM intersections i
            JOIN peer_counts pc ON pc.graph_fingerprint = i.graph_fingerprint
            JOIN target_count tc
            JOIN latest_rows lr ON lr.graph_fingerprint = i.graph_fingerprint
            WHERE tc.n > 0
              AND (pc.peer_op_count + tc.n - i.overlap) > 0
              AND CAST(i.overlap AS FLOAT) / CAST((pc.peer_op_count + tc.n - i.overlap) AS FLOAT) >= 0.1
            ORDER BY jaccard_similarity DESC, lr.timestamp DESC
            LIMIT ?
            """,
            (graph_fingerprint, graph_fingerprint, graph_fingerprint, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Op Success Rates ──

    def update_op_success_rates(self, experiment_id: str) -> None:
        """Recompute op success rates from program results in this experiment.

        Uses a targeted query (only needed columns) and avoids dict(r)
        conversion overhead from get_program_results.
        """
        sql_rows = self._query_op_stats_sql(
            "pr.experiment_id = ? AND pr.graph_json IS NOT NULL",
            (experiment_id,),
        )
        if sql_rows is None:
            rows = self.conn.execute(
                """SELECT graph_json, stage0_passed, stage05_passed, stage1_passed,
                          loss_ratio, novelty_score, novelty_confidence
                   FROM program_results
                   WHERE experiment_id = ? AND graph_json IS NOT NULL""",
                (experiment_id,),
            ).fetchall()

            op_stats: Dict[str, Dict] = {}

            for r in rows:
                graph_json = r[0]
                if not graph_json:
                    continue
                ops_in_graph = _cached_extract_unique_ops(graph_json)
                if not ops_in_graph:
                    continue

                s0 = r[1]
                s05 = r[2]
                s1 = r[3]
                lr = r[4]
                nov = r[5]
                nov_conf = r[6]

                for op_name in ops_in_graph:
                    if op_name not in op_stats:
                        op_stats[op_name] = {
                            "n_used": 0,
                            "n_s0": 0,
                            "n_s05": 0,
                            "n_s1": 0,
                            "lr_sum": 0.0,
                            "lr_n": 0,
                            "nov_sum": 0.0,
                            "nov_n": 0,
                            "nov_conf_sum": 0.0,
                            "nov_conf_n": 0,
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
            sql_rows = [
                {
                    "op_name": op_name,
                    "n_used": stats["n_used"],
                    "n_stage0_passed": stats["n_s0"],
                    "n_stage05_passed": stats["n_s05"],
                    "n_stage1_passed": stats["n_s1"],
                    "avg_loss_ratio": (
                        stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
                    ),
                    "avg_novelty": (
                        stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
                    ),
                    "avg_novelty_confidence": (
                        stats["nov_conf_sum"] / stats["nov_conf_n"]
                        if stats["nov_conf_n"]
                        else None
                    ),
                }
                for op_name, stats in op_stats.items()
            ]
        now = time.time()
        rows_to_write = []
        for stats in sql_rows:
            rows_to_write.append(
                (
                    stats["op_name"],
                    stats["n_used"],
                    stats["n_stage0_passed"],
                    stats["n_stage05_passed"],
                    stats["n_stage1_passed"],
                    stats["avg_loss_ratio"],
                    stats["avg_novelty"],
                    stats["avg_novelty_confidence"],
                    now,
                )
            )
        self.conn.executemany(
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
                avg_loss_ratio = CASE
                    WHEN op_success_rates.n_used = 0 THEN excluded.avg_loss_ratio
                    WHEN excluded.avg_loss_ratio IS NULL THEN op_success_rates.avg_loss_ratio
                    ELSE (op_success_rates.avg_loss_ratio * op_success_rates.n_used
                          + excluded.avg_loss_ratio * excluded.n_used)
                         / (op_success_rates.n_used + excluded.n_used)
                END,
                avg_novelty = CASE
                    WHEN op_success_rates.n_used = 0 THEN excluded.avg_novelty
                    WHEN excluded.avg_novelty IS NULL THEN op_success_rates.avg_novelty
                    ELSE (op_success_rates.avg_novelty * op_success_rates.n_used
                          + excluded.avg_novelty * excluded.n_used)
                         / (op_success_rates.n_used + excluded.n_used)
                END,
                avg_novelty_confidence = CASE
                    WHEN op_success_rates.n_used = 0 THEN excluded.avg_novelty_confidence
                    WHEN excluded.avg_novelty_confidence IS NULL THEN op_success_rates.avg_novelty_confidence
                    ELSE (op_success_rates.avg_novelty_confidence * op_success_rates.n_used
                          + excluded.avg_novelty_confidence * excluded.n_used)
                         / (op_success_rates.n_used + excluded.n_used)
                END,
                last_updated = excluded.last_updated""",
            rows_to_write,
        )
        self._maybe_commit()

    def get_op_success_rates(self) -> List[Dict]:
        """Get all op success rates."""
        rows = self.conn.execute(
            """SELECT * FROM op_success_rates
               ORDER BY n_stage1_passed DESC, n_used DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_op_success_rates_windowed(self, since_ts: float) -> List[Dict]:
        """Compute op success rates from program_results within a time window.

        Read-only windowed view — does not write to the accumulated table.
        """
        sql_rows = self._query_op_stats_sql(
            "pr.timestamp > ? AND pr.graph_json IS NOT NULL",
            (since_ts,),
        )
        if sql_rows is not None:
            return sql_rows

        rows = self.conn.execute(
            """SELECT graph_json, stage0_passed, stage05_passed, stage1_passed,
                      loss_ratio, novelty_score, novelty_confidence
               FROM program_results
               WHERE timestamp > ? AND graph_json IS NOT NULL""",
            (since_ts,),
        ).fetchall()
        op_stats: Dict[str, Dict] = {}
        for r in rows:
            graph_json = r[0]
            if not graph_json:
                continue
            ops_in_graph = _cached_extract_unique_ops(graph_json)
            if not ops_in_graph:
                continue
            s0, s05, s1, lr, nov, nov_conf = r[1], r[2], r[3], r[4], r[5], r[6]
            for op_name in ops_in_graph:
                stats = op_stats.setdefault(
                    op_name,
                    {
                        "n_used": 0,
                        "n_s0": 0,
                        "n_s05": 0,
                        "n_s1": 0,
                        "lr_sum": 0.0,
                        "lr_n": 0,
                        "nov_sum": 0.0,
                        "nov_n": 0,
                        "nov_conf_sum": 0.0,
                        "nov_conf_n": 0,
                    },
                )
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
        result = []
        for op_name, stats in sorted(
            op_stats.items(), key=lambda x: (-x[1]["n_s1"], -x[1]["n_used"])
        ):
            result.append(
                {
                    "op_name": op_name,
                    "n_used": stats["n_used"],
                    "n_stage0_passed": stats["n_s0"],
                    "n_stage05_passed": stats["n_s05"],
                    "n_stage1_passed": stats["n_s1"],
                    "avg_loss_ratio": (
                        stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
                    ),
                    "avg_novelty": (
                        stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
                    ),
                    "avg_novelty_confidence": (
                        stats["nov_conf_sum"] / stats["nov_conf_n"]
                        if stats["nov_conf_n"]
                        else None
                    ),
                }
            )
        return result

    def get_op_pair_priors(
        self, min_support: int = 5, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Aggregate op bigram priors from program results."""
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                gp.signature,
                COUNT(*) AS support,
                SUM(CASE WHEN pr.stage1_passed THEN 1 ELSE 0 END) AS n_stage1_passed,
                AVG(pr.loss_ratio) AS avg_loss_ratio,
                AVG(pr.novelty_score) AS avg_novelty
            FROM program_graph_pairs gp
            JOIN program_results pr ON pr.result_id = gp.result_id
            GROUP BY gp.signature
            HAVING COUNT(*) >= ?
            """,
            (int(min_support),),
        ).fetchall()
        priors = [
            {
                "signature": row["signature"],
                "success_rate": round(
                    float(row["n_stage1_passed"]) / int(row["support"]), 4
                ),
                "support": int(row["support"]),
                "avg_loss_ratio": (
                    round(float(row["avg_loss_ratio"]), 4)
                    if row["avg_loss_ratio"] is not None
                    else None
                ),
                "avg_novelty": (
                    round(float(row["avg_novelty"]), 4)
                    if row["avg_novelty"] is not None
                    else None
                ),
            }
            for row in rows
        ]
        return heapq.nlargest(
            limit, priors, key=lambda item: (item["success_rate"], item["support"])
        )

    def get_fingerprint_buckets(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Bucket fingerprints into structural families with rich feature vectors.

        Each bucket includes:
        - Original fields: bucket, n_graphs, s1_rate, avg_novelty, top_ops, top_pairs
        - op_category_distribution: normalized 11-dim vector over OpCategory
        - top_routing_ops: top-3 routing/compression/MoE ops by frequency
        - template_signature: most common template name in the bucket
        """
        rows = self._query_graph_feature_rows_sql() or []
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            record = dict(row)
            bucket_name = self._assign_fingerprint_bucket(record)
            bucket = buckets.setdefault(
                bucket_name, self._new_fingerprint_bucket(bucket_name)
            )
            if "ops_blob" in record:
                ops = (
                    [op for op in str(record.get("ops_blob") or "").split("\x1f") if op]
                    if record.get("ops_blob")
                    else []
                )
                pairs = (
                    [
                        sig
                        for sig in str(record.get("pairs_blob") or "").split("\x1f")
                        if sig
                    ]
                    if record.get("pairs_blob")
                    else []
                )
                template_name = str(record.get("template_name") or "")
            else:
                ops = self._extract_op_names(record["graph_json"])
                pairs = self._extract_op_bigrams(record["graph_json"])
                template_name = self._extract_template_name(record["graph_json"])
            bucket["n_graphs"] += 1
            bucket["n_stage1_passed"] += int(bool(record["stage1_passed"]))
            if record["novelty_score"] is not None:
                bucket["novelty_sum"] += float(record["novelty_score"])
                bucket["novelty_n"] += 1
            for op_name in ops:
                bucket["op_counts"][op_name] = bucket["op_counts"].get(op_name, 0) + 1
                # Accumulate category distribution
                cat = _get_op_category(op_name)
                if cat in bucket["category_counts"]:
                    bucket["category_counts"][cat] += 1
                # Accumulate routing op counts
                if op_name in _ROUTING_OPS:
                    bucket["routing_op_counts"][op_name] = (
                        bucket["routing_op_counts"].get(op_name, 0) + 1
                    )
            for signature in pairs:
                bucket["pair_counts"][signature] = (
                    bucket["pair_counts"].get(signature, 0) + 1
                )
            if template_name:
                bucket["template_counts"][template_name] = (
                    bucket["template_counts"].get(template_name, 0) + 1
                )
        ranked = [
            self._finalize_fingerprint_bucket(bucket) for bucket in buckets.values()
        ]
        return heapq.nlargest(limit, ranked, key=lambda item: item["n_graphs"])

    def get_nearest_peers(
        self, graph_fingerprint: str, n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find the n most structurally similar historical fingerprints via Jaccard similarity.

        Compares the op-set of the target fingerprint against all other fingerprints
        in program_results. Returns peers sorted by descending Jaccard similarity,
        each with loss_ratio, novelty_score, tier, and composite_score.
        """
        sql_rows = self._query_nearest_peers_sql(graph_fingerprint, limit=max(n, 500))
        if sql_rows is not None:
            return [
                {
                    "fingerprint": str(row["fingerprint"]),
                    "jaccard_similarity": float(row["jaccard_similarity"]),
                    "loss_ratio": (
                        float(row["loss_ratio"])
                        if row["loss_ratio"] is not None
                        else None
                    ),
                    "novelty_score": (
                        float(row["novelty_score"])
                        if row["novelty_score"] is not None
                        else None
                    ),
                    "stage1_passed": bool(row["stage1_passed"]),
                    "tier": str(row["tier"] or ""),
                    "composite_score": (
                        float(row["composite_score"])
                        if row["composite_score"] is not None
                        else None
                    ),
                }
                for row in sql_rows[:n]
            ]

        return []
