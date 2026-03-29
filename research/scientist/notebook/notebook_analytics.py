from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import heapq
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._shared import LOGGER

# Lazy-loaded to avoid circular imports at module level
_OP_CATEGORY_CACHE: Dict[str, str] = {}


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

            s0 = r[1]  # stage0_passed
            s05 = r[2]  # stage05_passed
            s1 = r[3]  # stage1_passed
            lr = r[4]  # loss_ratio
            nov = r[5]  # novelty_score
            nov_conf = r[6]  # novelty_confidence

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

        now = time.time()
        for op_name, stats in op_stats.items():
            avg_lr = stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
            avg_nov = stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
            avg_nov_conf = (
                stats["nov_conf_sum"] / stats["nov_conf_n"]
                if stats["nov_conf_n"]
                else None
            )
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
                (
                    op_name,
                    stats["n_used"],
                    stats["n_s0"],
                    stats["n_s05"],
                    stats["n_s1"],
                    avg_lr,
                    avg_nov,
                    avg_nov_conf,
                    now,
                ),
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
            try:
                graph_data = json.loads(graph_json)
                nodes = graph_data.get("nodes", {})
            except (json.JSONDecodeError, TypeError):
                continue

            ops_in_graph = set()
            for node_data in nodes.values():
                op_name = node_data.get("op_name", "")
                if op_name and op_name != "input":
                    ops_in_graph.add(op_name)

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

        result = []
        for op_name, stats in sorted(
            op_stats.items(), key=lambda x: (-x[1]["n_s1"], -x[1]["n_used"])
        ):
            avg_lr = stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
            avg_nov = stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
            avg_nov_conf = (
                stats["nov_conf_sum"] / stats["nov_conf_n"]
                if stats["nov_conf_n"]
                else None
            )
            result.append(
                {
                    "op_name": op_name,
                    "n_used": stats["n_used"],
                    "n_stage0_passed": stats["n_s0"],
                    "n_stage05_passed": stats["n_s05"],
                    "n_stage1_passed": stats["n_s1"],
                    "avg_loss_ratio": avg_lr,
                    "avg_novelty": avg_nov,
                    "avg_novelty_confidence": avg_nov_conf,
                }
            )
        return result

    def get_op_pair_priors(
        self, min_support: int = 5, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Aggregate op bigram priors from program results."""
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, loss_ratio, novelty_score
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        ).fetchall()
        aggregates: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for signature in self._extract_op_bigrams(row["graph_json"]):
                bucket = aggregates.setdefault(
                    signature, self._new_pair_bucket(signature)
                )
                bucket["support"] += 1
                bucket["n_stage1_passed"] += int(bool(row["stage1_passed"]))
                if row["loss_ratio"] is not None:
                    bucket["loss_sum"] += float(row["loss_ratio"])
                    bucket["loss_n"] += 1
                if row["novelty_score"] is not None:
                    bucket["novelty_sum"] += float(row["novelty_score"])
                    bucket["novelty_n"] += 1
        priors = [
            self._finalize_pair_bucket(signature, bucket)
            for signature, bucket in aggregates.items()
            if bucket["support"] >= min_support
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
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, novelty_score, graph_category_histogram,
                      fp_interaction_sparsity, fp_cka_vs_transformer, fp_cka_vs_ssm, fp_cka_vs_conv
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        ).fetchall()
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            record = dict(row)
            bucket_name = self._assign_fingerprint_bucket(record)
            bucket = buckets.setdefault(
                bucket_name, self._new_fingerprint_bucket(bucket_name)
            )
            ops = self._extract_op_names(record["graph_json"])
            pairs = self._extract_op_bigrams(record["graph_json"])
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
            # Extract template signature from graph metadata
            template_name = self._extract_template_name(record["graph_json"])
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
        # Get target ops
        target_row = self.conn.execute(
            """SELECT graph_json FROM program_results
               WHERE graph_fingerprint = ? AND graph_json IS NOT NULL
               LIMIT 1""",
            (graph_fingerprint,),
        ).fetchone()
        if not target_row:
            return []
        target_ops = frozenset(self._extract_op_names(target_row[0]))
        if not target_ops:
            return []

        # Get all distinct fingerprints with their ops and scores
        rows = self.conn.execute(
            """SELECT pr.graph_fingerprint, pr.graph_json, pr.loss_ratio,
                      pr.novelty_score, pr.stage1_passed,
                      l.tier, l.composite_score
               FROM program_results pr
               LEFT JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE pr.graph_fingerprint IS NOT NULL
                 AND pr.graph_fingerprint != ?
                 AND pr.graph_json IS NOT NULL
               GROUP BY pr.graph_fingerprint
               ORDER BY pr.timestamp DESC
               LIMIT 500""",
            (graph_fingerprint,),
        ).fetchall()

        peers: List[Dict[str, Any]] = []
        for row in rows:
            peer_ops = frozenset(self._extract_op_names(row[1]))
            if not peer_ops:
                continue
            intersection = len(target_ops & peer_ops)
            union = len(target_ops | peer_ops)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard < 0.1:
                continue
            peers.append(
                {
                    "fingerprint": str(row[0]),
                    "jaccard_similarity": round(jaccard, 4),
                    "loss_ratio": float(row[2]) if row[2] is not None else None,
                    "novelty_score": float(row[3]) if row[3] is not None else None,
                    "stage1_passed": bool(row[4]),
                    "tier": str(row[5] or ""),
                    "composite_score": float(row[6]) if row[6] is not None else None,
                }
            )

        peers.sort(key=lambda p: p["jaccard_similarity"], reverse=True)
        return peers[:n]

    def get_lineage_successor_stats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Aggregate parent→child fingerprint transitions from designer lineage."""
        rows = self.conn.execute(
            """SELECT workflow_id, workflow_version, graph_fingerprint, payload_json, updated_at
               FROM designer_run_lineage
               WHERE graph_fingerprint IS NOT NULL
               ORDER BY workflow_id ASC, workflow_version ASC, updated_at ASC"""
        ).fetchall()
        leaderboard = self._leaderboard_by_fingerprint()
        transitions: Dict[str, Dict[str, Any]] = {}
        previous_by_workflow: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            record = dict(row)
            payload = self._json_dict(record.get("payload_json"))
            previous = previous_by_workflow.get(str(record.get("workflow_id"))) or {}
            child_fp = str(record.get("graph_fingerprint") or "").strip()
            if not child_fp:
                continue
            parent_fp = (
                self._payload_parent_fingerprint(payload)
                or str(previous.get("graph_fingerprint") or "").strip()
            )
            previous_by_workflow[str(record.get("workflow_id"))] = {
                "graph_fingerprint": child_fp,
                "payload": payload,
            }
            if not parent_fp or parent_fp == child_fp:
                continue
            key = f"{parent_fp}->{child_fp}"
            bucket = transitions.setdefault(
                key, self._new_lineage_bucket(parent_fp, child_fp)
            )
            bucket["support"] += 1
            parent_metrics = leaderboard.get(parent_fp, {})
            child_metrics = leaderboard.get(child_fp, {})
            parent_score = float(parent_metrics.get("composite_score") or 0.0)
            child_score = float(child_metrics.get("composite_score") or 0.0)
            improved = child_score > parent_score
            bucket["improved"] += int(improved)
            bucket["child_successes"] += int(bool(child_metrics.get("stage1_passed")))
            bucket["delta_sum"] += child_score - parent_score
            parent_payload = previous.get("payload")
            for change in self._summarize_workflow_changes(parent_payload, payload):
                bucket["change_counts"][change] = (
                    bucket["change_counts"].get(change, 0) + 1
                )
        results = [
            self._finalize_lineage_bucket(bucket) for bucket in transitions.values()
        ]
        results.sort(
            key=lambda item: (item["improved_rate"], item["support"]), reverse=True
        )
        return results[:limit]

    def get_failure_risk_signatures(
        self, limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compute soft failure-risk penalties with positive-evidence filtering."""
        support = self._top_performer_bigram_support()
        rows = self.conn.execute(
            """SELECT signature, n_failures, n_successes, error_types
               FROM failure_signatures"""
        ).fetchall()
        risk_signatures: List[Dict[str, Any]] = []
        critical: List[Dict[str, Any]] = []
        for row in rows:
            positive_support = int(support.get(str(row["signature"]), 0))
            if positive_support < 3:
                continue
            total = int(row["n_failures"] or 0) + int(row["n_successes"] or 0)
            if total <= 0:
                continue
            fail_rate = float(row["n_failures"] or 0) / float(total)
            weight = self._failure_penalty_weight(fail_rate, total, positive_support)
            if weight is None:
                continue
            item = {
                "signature": row["signature"],
                "support": total,
                "fail_rate": round(fail_rate, 4),
                "positive_support": positive_support,
                "weight": weight,
                "error_types": row["error_types"],
            }
            risk_signatures.append(item)
            if fail_rate >= 0.98 and total >= 50 and positive_support >= 5:
                critical.append(item)
        risk_signatures.sort(key=lambda item: (item["weight"], -item["support"]))
        critical.sort(key=lambda item: (-item["fail_rate"], -item["support"]))
        return {
            "failure_risk_signatures": risk_signatures[:limit],
            "critical_failures": critical[: min(limit, 25)],
        }

    @staticmethod
    def _new_pair_bucket(signature: str) -> Dict[str, Any]:
        return {
            "signature": signature,
            "support": 0,
            "n_stage1_passed": 0,
            "loss_sum": 0.0,
            "loss_n": 0,
            "novelty_sum": 0.0,
            "novelty_n": 0,
        }

    @staticmethod
    def _finalize_pair_bucket(signature: str, bucket: Dict[str, Any]) -> Dict[str, Any]:
        support = int(bucket["support"])
        avg_loss = (bucket["loss_sum"] / bucket["loss_n"]) if bucket["loss_n"] else None
        avg_novelty = (
            (bucket["novelty_sum"] / bucket["novelty_n"])
            if bucket["novelty_n"]
            else None
        )
        return {
            "signature": signature,
            "success_rate": round(float(bucket["n_stage1_passed"]) / support, 4),
            "support": support,
            "avg_loss_ratio": round(avg_loss, 4) if avg_loss is not None else None,
            "avg_novelty": round(avg_novelty, 4) if avg_novelty is not None else None,
        }

    @staticmethod
    def _new_fingerprint_bucket(bucket_name: str) -> Dict[str, Any]:
        return {
            "bucket": bucket_name,
            "n_graphs": 0,
            "n_stage1_passed": 0,
            "novelty_sum": 0.0,
            "novelty_n": 0,
            "op_counts": {},
            "pair_counts": {},
            "category_counts": {cat: 0 for cat in _ALL_CATEGORIES},
            "routing_op_counts": {},
            "template_counts": {},
        }

    @staticmethod
    def _finalize_fingerprint_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
        novelty = (
            (bucket["novelty_sum"] / bucket["novelty_n"])
            if bucket["novelty_n"]
            else None
        )
        # Normalize category distribution to sum=1
        cat_counts = bucket.get("category_counts", {})
        cat_total = max(1, sum(cat_counts.values()))
        op_category_distribution = {
            cat: round(cat_counts.get(cat, 0) / cat_total, 4) for cat in _ALL_CATEGORIES
        }
        # Top-3 routing ops by frequency
        routing_counts = bucket.get("routing_op_counts", {})
        top_routing_ops = [
            op
            for op, _ in sorted(
                routing_counts.items(), key=lambda item: item[1], reverse=True
            )[:3]
        ]
        # Most common template
        template_counts = bucket.get("template_counts", {})
        template_signature = ""
        if template_counts:
            template_signature = max(template_counts, key=template_counts.get)

        return {
            "bucket": bucket["bucket"],
            "n_graphs": int(bucket["n_graphs"]),
            "s1_rate": round(
                float(bucket["n_stage1_passed"]) / max(int(bucket["n_graphs"]), 1), 4
            ),
            "avg_novelty": round(novelty, 4) if novelty is not None else None,
            "top_ops": [
                {"op_name": op_name, "count": count}
                for op_name, count in sorted(
                    bucket["op_counts"].items(), key=lambda item: item[1], reverse=True
                )[:10]
            ],
            "top_pairs": [
                {"signature": signature, "count": count}
                for signature, count in sorted(
                    bucket["pair_counts"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
            "op_category_distribution": op_category_distribution,
            "top_routing_ops": top_routing_ops,
            "template_signature": template_signature,
        }

    @staticmethod
    def _new_lineage_bucket(parent_fp: str, child_fp: str) -> Dict[str, Any]:
        return {
            "parent_fingerprint": parent_fp,
            "child_fingerprint": child_fp,
            "support": 0,
            "improved": 0,
            "child_successes": 0,
            "delta_sum": 0.0,
            "change_counts": {},
        }

    @staticmethod
    def _finalize_lineage_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
        support = max(int(bucket["support"]), 1)
        return {
            "parent_fingerprint": bucket["parent_fingerprint"],
            "child_fingerprint": bucket["child_fingerprint"],
            "support": int(bucket["support"]),
            "improved_rate": round(float(bucket["improved"]) / support, 4),
            "child_success_rate": round(float(bucket["child_successes"]) / support, 4),
            "avg_composite_delta": round(float(bucket["delta_sum"]) / support, 4),
            "change_patterns": [
                {"change": change, "count": count}
                for change, count in sorted(
                    bucket["change_counts"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:6]
            ],
        }

    def _assign_fingerprint_bucket(self, row: Dict[str, Any]) -> str:
        ops = set(self._extract_op_names(row.get("graph_json") or ""))
        hist = self._json_dict(row.get("graph_category_histogram"))
        sparse = (
            float(row.get("fp_interaction_sparsity") or 0.0) >= 0.55
            or self._hist_score(hist, "sparse") >= 0.2
        )
        attention = (
            float(row.get("fp_cka_vs_transformer") or 0.0) >= 0.5
            or any("attention" in op for op in ops)
            or self._hist_score(hist, "attention") >= 0.2
        )
        mixing = (
            max(
                float(row.get("fp_cka_vs_ssm") or 0.0),
                float(row.get("fp_cka_vs_conv") or 0.0),
            )
            >= 0.5
            or any(
                token in op
                for op in ops
                for token in ("state_space", "scan", "conv", "mix")
            )
            or self._hist_score(hist, "mix") >= 0.2
        )
        if attention and mixing:
            return "hybrid"
        if sparse:
            return "sparse"
        if attention:
            return "attention-heavy"
        if mixing:
            return "mixing-heavy"
        return "exotic"

    @staticmethod
    def _hist_score(histogram: Dict[str, Any], token: str) -> float:
        total = sum(float(value or 0.0) for value in histogram.values())
        if total <= 0.0:
            return 0.0
        matched = sum(
            float(value or 0.0)
            for key, value in histogram.items()
            if token in str(key).lower()
        )
        return matched / total

    @staticmethod
    def _json_dict(payload: Any) -> Dict[str, Any]:
        try:
            loaded = json.loads(payload) if isinstance(payload, str) else payload
        except (json.JSONDecodeError, TypeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _payload_parent_fingerprint(payload: Dict[str, Any]) -> str:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("parent_fingerprint"):
            return str(metadata["parent_fingerprint"])
        if payload.get("parent_fingerprint"):
            return str(payload["parent_fingerprint"])
        return ""

    def _extract_op_names(self, graph_json: str) -> List[str]:
        from research.scientist.analytics.analytics_ops import _OpsMixin

        ops = _OpsMixin._extract_ops_fast(graph_json)
        if ops is None:
            ops = _OpsMixin._extract_ops_fallback(graph_json)
        return list(ops or [])

    @staticmethod
    def _extract_template_name(graph_json: str) -> str:
        """Extract the template name from graph JSON metadata, if present."""
        try:
            data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        except (json.JSONDecodeError, TypeError):
            return ""
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("template") or metadata.get("template_name") or "")

    def _leaderboard_by_fingerprint(self) -> Dict[str, Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT pr.graph_fingerprint, pr.stage1_passed, l.composite_score
               FROM program_results pr
               LEFT JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE pr.graph_fingerprint IS NOT NULL"""
        ).fetchall()
        scores: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            fingerprint = str(row["graph_fingerprint"] or "").strip()
            if not fingerprint:
                continue
            existing = scores.get(fingerprint)
            composite = float(row["composite_score"] or 0.0)
            if existing is None or composite >= float(
                existing.get("composite_score") or 0.0
            ):
                scores[fingerprint] = {
                    "composite_score": composite,
                    "stage1_passed": int(bool(row["stage1_passed"])),
                }
        return scores

    def _top_performer_bigram_support(self) -> Dict[str, int]:
        # Compute 25th-percentile threshold in SQL instead of sorting all losses in Python
        threshold_row = self.conn.execute(
            """SELECT loss_ratio FROM program_results
               WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL
               ORDER BY loss_ratio ASC
               LIMIT 1 OFFSET (
                   SELECT MAX(0, COUNT(*) / 4 - 1) FROM program_results
                   WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL
               )"""
        ).fetchone()
        threshold = float(threshold_row["loss_ratio"]) if threshold_row else None
        rows = self.conn.execute(
            """SELECT pr.graph_json, pr.loss_ratio, l.tier
               FROM program_results pr
               LEFT JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE pr.stage1_passed = 1 AND pr.graph_json IS NOT NULL"""
        ).fetchall()
        support: Dict[str, int] = defaultdict(int)
        for row in rows:
            tier = str(row["tier"] or "").lower()
            in_top_loss = (
                threshold is not None
                and row["loss_ratio"] is not None
                and float(row["loss_ratio"]) <= threshold
            )
            in_top_tier = tier in {"investigation", "validation", "breakthrough"}
            if not in_top_loss and not in_top_tier:
                continue
            for signature in self._extract_op_bigrams(row["graph_json"]):
                support[signature] += 1
        return dict(support)

    @staticmethod
    def _failure_penalty_weight(
        fail_rate: float, total: int, positive_support: int
    ) -> Optional[float]:
        # Soft penalties only — many high-fail-rate pairs are standard
        # transformer patterns (norm→routing, routing→residual, proj→routing)
        # that fail due to broader template composition, not the pair itself.
        # Floor at 0.25 so no pair is nearly killed.
        if fail_rate >= 0.95 and total >= 20 and positive_support >= 5:
            return 0.25
        if fail_rate >= 0.85 and total >= 10 and positive_support >= 3:
            return 0.45
        if fail_rate >= 0.70 and total >= 5 and positive_support >= 3:
            return 0.65
        return None

    @staticmethod
    def _summarize_workflow_changes(
        parent_payload: Any, child_payload: Any
    ) -> List[str]:
        parent_nodes = _AnalyticsMixin._workflow_nodes(parent_payload)
        child_nodes = _AnalyticsMixin._workflow_nodes(child_payload)
        if not parent_nodes or not child_nodes:
            return []
        changes: List[str] = []
        parent_types = {node_id: comp for node_id, comp in parent_nodes.items()}
        child_types = {node_id: comp for node_id, comp in child_nodes.items()}
        for node_id, parent_type in parent_types.items():
            child_type = child_types.get(node_id)
            if child_type and child_type != parent_type:
                changes.append(f"swap:{parent_type}->{child_type}")
        added = sorted(set(child_types.values()) - set(parent_types.values()))
        removed = sorted(set(parent_types.values()) - set(child_types.values()))
        changes.extend(f"add:{comp}" for comp in added[:3])
        changes.extend(f"remove:{comp}" for comp in removed[:3])
        return changes

    @staticmethod
    def _workflow_nodes(payload: Any) -> Dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            return {}
        return {
            str(node.get("id")): str(node.get("component_type"))
            for node in nodes
            if isinstance(node, dict) and node.get("id") and node.get("component_type")
        }

    def update_failure_signatures(self, experiment_id: str) -> None:
        """Update failure_signatures table from program results in this experiment.

        Extracts op-pair bigrams from each graph and tracks how often
        each bigram appears in failed vs successful programs.  This gives
        Aria a compact memory of which structural patterns to avoid.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1""",
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
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
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
        LOGGER.info(
            "Backfilled %d failure signatures from existing results", len(sig_stats)
        )
        return len(sig_stats)

    def recompute_failure_signatures(self) -> int:
        """Delete and rebuild failure_signatures from scratch using S1-only failures.

        Unlike backfill_failure_signatures(), this always runs (even if data exists)
        and only counts programs that passed S0+S0.5 but failed at S1 as failures.
        This cleans up historically contaminated data from S0.5 causality failures.
        """
        self.conn.execute("DELETE FROM failure_signatures")
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
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
        LOGGER.info(
            "Recomputed %d failure signatures (S1-only failures)", len(sig_stats)
        )
        return len(sig_stats)

    def get_failure_signature_blocklist(
        self, min_seen: int = 20, max_fail_rate: float = 0.95
    ) -> Dict[str, float]:
        """Return op-pair bigrams that consistently fail.

        Returns {signature: penalty} where penalty is a soft deweight factor.
        100% failure bigrams get 0.05 (95% deweight), scaling up to 0.3 at
        the ``max_fail_rate`` threshold.  No hard blocks — all pairs retain
        a small chance of being selected.  Only includes bigrams seen at
        least ``min_seen`` times with failure rate >= ``max_fail_rate``.
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
                # Soft deweight: 100% fail → 0.05, max_fail_rate → 0.3
                penalty = 0.05 + 0.25 * (1.0 - fail_rate) / (1.0 - max_fail_rate)
                blocklist[r[0]] = round(penalty, 2)
        return blocklist

    # ── Op Rehabilitation Cache ──

    def get_op_rehabilitation_cache(
        self, max_age_hours: float = 24.0
    ) -> Dict[str, Dict]:
        """Return cached op rehabilitation results, filtered by recency.

        Returns {op_name: {compile_passed, forward_passed, error_message, tested_at, model_dim}}.
        """
        cutoff = time.time() - max_age_hours * 3600
        rows = self.conn.execute(
            """SELECT op_name, compile_passed, forward_passed, error_message, tested_at, model_dim
               FROM op_rehabilitation_cache
               WHERE tested_at >= ?""",
            (cutoff,),
        ).fetchall()
        cache: Dict[str, Dict] = {}
        for r in rows:
            cache[r[0]] = {
                "compile_passed": bool(r[1]),
                "forward_passed": bool(r[2]),
                "error_message": r[3],
                "tested_at": r[4],
                "model_dim": r[5],
            }
        return cache

    # ── Learning Log ──

    def log_learning_event(
        self,
        event_type: str,
        description: str,
        old_weights: Optional[Dict] = None,
        new_weights: Optional[Dict] = None,
        evidence: Optional[str] = None,
        **event_data: Any,
    ) -> None:
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
            (
                time.time(),
                event_type,
                description,
                json.dumps(old_weights) if old_weights else None,
                json.dumps(new_weights) if new_weights else None,
                evidence,
            ),
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

    def save_effective_weights(
        self,
        weights: Dict[str, float],
        s1_rate: float,
        experiment_id: Optional[str] = None,
    ) -> None:
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
        source: str = "aria_designer",
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
                ttl_seconds = int(
                    os.environ.get(
                        "ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)
                    )
                )
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(
                    os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400")
                )
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

            scopes.append(
                {
                    "scope": row["scope"],
                    "count": count,
                    "oldest_age_seconds": round(max(0.0, now - oldest), 2)
                    if oldest > 0
                    else None,
                    "newest_age_seconds": round(max(0.0, now - newest), 2)
                    if newest > 0
                    else None,
                }
            )

        return {
            "total_snapshots": total,
            "n_scopes": len(scopes),
            "oldest_age_seconds": round(max(0.0, now - oldest_seen), 2)
            if oldest_seen
            else None,
            "newest_age_seconds": round(max(0.0, now - newest_seen), 2)
            if newest_seen
            else None,
            "scopes": scopes,
        }

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

    def get_attribution_reports(
        self, hypothesis_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
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
            for key in (
                "supporting_experiments",
                "ablation_experiments",
                "report_json",
            ):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out

    # ── Report Markdown Export ──

    def save_report_markdown(
        self, content: str, reason: str, summary: Optional[Dict] = None
    ) -> Optional[Path]:
        """Save a report as a markdown file alongside the database.

        Creates a reports/ directory next to lab_notebook.db and writes
        the report content as a .md file with a frontmatter-style header.

        Returns the path to the created file, or None on failure.
        """
        logger = LOGGER
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
                    f"experiments: {summary.get('total_experiments', '?')}"
                )
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
