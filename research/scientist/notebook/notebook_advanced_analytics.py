from __future__ import annotations

"""Extended analytics and reporting mixin for LabNotebook."""

import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..runtime_events import publish_runtime_event
from ._shared import LOGGER
from .failure_signature_audits import AUDITED_FALSE_FAILURE_SIGNATURE_SET
from .notebook_analytics import (
    _ALL_CATEGORIES,
    _cached_extract_op_names,
    _cached_extract_template_name,
)


class _AdvancedAnalyticsMixin:
    """Advanced analytics, scaffold profiling, and report snapshot methods."""

    __slots__ = ()

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
        cat_counts = bucket.get("category_counts", {})
        cat_total = max(1, sum(cat_counts.values()))
        op_category_distribution = {
            cat: round(cat_counts.get(cat, 0) / cat_total, 4) for cat in _ALL_CATEGORIES
        }
        routing_counts = bucket.get("routing_op_counts", {})
        top_routing_ops = [
            op
            for op, _ in sorted(
                routing_counts.items(), key=lambda item: item[1], reverse=True
            )[:3]
        ]
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

    def _assign_fingerprint_bucket(self, row: Dict[str, Any]) -> str:
        if row.get("ops_blob") is not None:
            ops = {op for op in str(row.get("ops_blob") or "").split("\x1f") if op}
        else:
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
        if not isinstance(graph_json, str) or not graph_json:
            return []
        return list(_cached_extract_op_names(graph_json))

    @staticmethod
    def _extract_template_name(graph_json: str) -> str:
        """Extract the template name from graph JSON metadata, if present."""
        if not isinstance(graph_json, str) or not graph_json:
            return ""
        return _cached_extract_template_name(graph_json)

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
            signature = str(row["signature"] or "")
            if signature in AUDITED_FALSE_FAILURE_SIGNATURE_SET:
                continue
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
                "signature": signature,
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
        self.flush_writes()
        self._ensure_graph_features()
        survivor_row = self.conn.execute(
            """SELECT COUNT(*) AS n
               FROM program_results
               WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL"""
        ).fetchone()
        survivor_count = int(survivor_row["n"] or 0) if survivor_row else 0
        threshold = None
        if survivor_count >= 20:
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
            """SELECT gp.signature, pr.loss_ratio, l.tier
               FROM program_graph_pairs gp
               JOIN program_results pr ON pr.result_id = gp.result_id
               LEFT JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE pr.stage1_passed = 1"""
        ).fetchall()
        support: Dict[str, int] = defaultdict(int)
        for row in rows:
            tier = str(row["tier"] or "").lower()
            in_top_loss = survivor_count < 20 or (
                threshold is not None
                and row["loss_ratio"] is not None
                and float(row["loss_ratio"]) <= threshold
            )
            in_top_tier = tier in {"investigation", "validation", "breakthrough"}
            if not in_top_loss and not in_top_tier:
                continue
            support[str(row["signature"])] += 1
        return dict(support)

    @staticmethod
    def _failure_penalty_weight(
        fail_rate: float, total: int, positive_support: int
    ) -> Optional[float]:
        if fail_rate >= 0.95 and total >= 20 and positive_support >= 5:
            return 0.05
        if fail_rate >= 0.85 and total >= 10 and positive_support >= 3:
            return 0.15
        if fail_rate >= 0.70 and total >= 5 and positive_support >= 3:
            return 0.30
        return None

    @staticmethod
    def _summarize_workflow_changes(
        parent_payload: Any, child_payload: Any
    ) -> List[str]:
        parent_nodes = _AdvancedAnalyticsMixin._workflow_nodes(parent_payload)
        child_nodes = _AdvancedAnalyticsMixin._workflow_nodes(child_payload)
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
        """Update failure_signatures table from program results in this experiment."""
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                gp.signature,
                SUM(CASE WHEN pr.stage1_passed THEN 0 ELSE 1 END) AS n_failures,
                SUM(CASE WHEN pr.stage1_passed THEN 1 ELSE 0 END) AS n_successes,
                substr(group_concat(DISTINCT CASE
                    WHEN pr.stage1_passed = 0 AND pr.error_type IS NOT NULL AND pr.error_type <> ''
                    THEN pr.error_type
                END), 1, 255) AS error_types
            FROM program_graph_pairs gp
            JOIN program_results pr ON pr.result_id = gp.result_id
            WHERE pr.experiment_id = ?
              AND pr.stage0_passed = 1
              AND pr.stage05_passed = 1
            GROUP BY gp.signature
            """,
            (experiment_id,),
        ).fetchall()
        if not rows:
            return
        now = time.time()
        self.conn.executemany(
            """INSERT INTO failure_signatures
               (signature, n_failures, n_successes, error_types, last_updated)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(signature) DO UPDATE SET
                n_failures = n_failures + excluded.n_failures,
                n_successes = n_successes + excluded.n_successes,
                error_types = COALESCE(excluded.error_types, error_types),
                last_updated = excluded.last_updated""",
            [
                (
                    row["signature"],
                    int(row["n_failures"] or 0),
                    int(row["n_successes"] or 0),
                    row["error_types"],
                    now,
                )
                for row in rows
            ],
        )
        self._maybe_commit()

    def backfill_failure_signatures(self) -> int:
        """One-time backfill of failure_signatures from all existing results."""
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM failure_signatures"
        ).fetchone()[0]
        if existing > 0:
            return 0
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                gp.signature,
                SUM(CASE WHEN pr.stage1_passed THEN 0 ELSE 1 END) AS n_failures,
                SUM(CASE WHEN pr.stage1_passed THEN 1 ELSE 0 END) AS n_successes,
                substr(group_concat(DISTINCT CASE
                    WHEN pr.stage1_passed = 0 AND pr.error_type IS NOT NULL AND pr.error_type <> ''
                    THEN pr.error_type
                END), 1, 255) AS error_types
            FROM program_graph_pairs gp
            JOIN program_results pr ON pr.result_id = gp.result_id
            WHERE pr.stage0_passed = 1
              AND pr.stage05_passed = 1
            GROUP BY gp.signature
            """
        ).fetchall()
        now = time.time()
        self.conn.executemany(
            """INSERT INTO failure_signatures
               (signature, n_failures, n_successes, error_types, last_updated)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    row["signature"],
                    int(row["n_failures"] or 0),
                    int(row["n_successes"] or 0),
                    row["error_types"],
                    now,
                )
                for row in rows
            ],
        )
        self._maybe_commit()
        LOGGER.info("Backfilled %d failure signatures from existing results", len(rows))
        return len(rows)

    def recompute_failure_signatures(self) -> int:
        """Delete and rebuild failure_signatures from scratch using S1-only failures."""
        self.conn.execute("DELETE FROM failure_signatures")
        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                gp.signature,
                SUM(CASE WHEN pr.stage1_passed THEN 0 ELSE 1 END) AS n_failures,
                SUM(CASE WHEN pr.stage1_passed THEN 1 ELSE 0 END) AS n_successes,
                substr(group_concat(DISTINCT CASE
                    WHEN pr.stage1_passed = 0 AND pr.error_type IS NOT NULL AND pr.error_type <> ''
                    THEN pr.error_type
                END), 1, 255) AS error_types
            FROM program_graph_pairs gp
            JOIN program_results pr ON pr.result_id = gp.result_id
            WHERE pr.stage0_passed = 1
              AND pr.stage05_passed = 1
            GROUP BY gp.signature
            """
        ).fetchall()
        now = time.time()
        self.conn.executemany(
            """INSERT INTO failure_signatures
               (signature, n_failures, n_successes, error_types, last_updated)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    row["signature"],
                    int(row["n_failures"] or 0),
                    int(row["n_successes"] or 0),
                    row["error_types"],
                    now,
                )
                for row in rows
            ],
        )
        self._maybe_commit()
        LOGGER.info("Recomputed %d failure signatures (S1-only failures)", len(rows))
        return len(rows)

    def get_failure_signature_blocklist(
        self, min_seen: int = 20, max_fail_rate: float = 0.95
    ) -> Dict[str, float]:
        """Return op-pair bigrams that consistently fail."""
        rows = self.conn.execute(
            """SELECT signature, n_failures, n_successes
               FROM failure_signatures
               WHERE (n_failures + n_successes) >= ?""",
            (min_seen,),
        ).fetchall()
        blocklist: Dict[str, float] = {}
        for row in rows:
            signature = str(row[0] or "")
            if signature in AUDITED_FALSE_FAILURE_SIGNATURE_SET:
                continue
            total = row[1] + row[2]
            fail_rate = row[1] / total if total else 0
            if fail_rate >= max_fail_rate:
                penalty = 0.05 + 0.25 * (1.0 - fail_rate) / (1.0 - max_fail_rate)
                blocklist[signature] = round(penalty, 2)
        return blocklist

    def get_op_rehabilitation_cache(
        self, max_age_hours: float = 24.0
    ) -> Dict[str, Dict]:
        """Return cached op rehabilitation results, filtered by recency."""
        cutoff = time.time() - max_age_hours * 3600
        rows = self.conn.execute(
            """SELECT op_name, compile_passed, forward_passed, error_message, tested_at, model_dim
               FROM op_rehabilitation_cache
               WHERE tested_at >= ?""",
            (cutoff,),
        ).fetchall()
        cache: Dict[str, Dict] = {}
        for row in rows:
            cache[row[0]] = {
                "compile_passed": bool(row[1]),
                "forward_passed": bool(row[2]),
                "error_message": row[3],
                "tested_at": row[4],
                "model_dim": row[5],
            }
        return cache

    def log_learning_event(
        self,
        event_type: str,
        description: str,
        old_weights: Optional[Dict] = None,
        new_weights: Optional[Dict] = None,
        evidence: Optional[str] = None,
        **event_data: Any,
    ) -> None:
        """Log a grammar weight change or learning decision."""
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

        try:
            publish_runtime_event(
                notebook_path=self.db_path,
                event_type="learning_event_logged",
                producer="notebook.advanced_analytics",
                run_id=str(event_data.get("experiment_id") or "").strip() or None,
                payload={
                    "log_event_type": event_type,
                    "description": description,
                    "old_weights": old_weights,
                    "new_weights": new_weights,
                    "evidence": evidence,
                    "event_data": event_data,
                },
            )
        except Exception as exc:
            LOGGER.warning(
                "Runtime telemetry publish failed for %s: %s", event_type, exc
            )

        try:
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
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Learning log write failed for %s; continuing without SQLite persistence: %s",
                event_type,
                exc,
            )

    def get_learning_log(self, limit: int = 100) -> List[Dict]:
        """Get recent learning log entries."""
        rows = self.conn.execute(
            "SELECT * FROM learning_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            data = dict(row)
            for field in ("old_weights", "new_weights"):
                if data.get(field):
                    try:
                        data[field] = json.loads(data[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(data)
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
        """Load the most recent effective weights snapshot."""
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
        data = dict(row)
        try:
            data["metrics"] = json.loads(data.get("metrics_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            data["metrics"] = {}
        try:
            data["payload"] = json.loads(data.get("payload_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            data["payload"] = {}
        return data

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
        results: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            try:
                data["metrics"] = json.loads(data.get("metrics_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                data["metrics"] = {}
            results.append(data)
        return results

    def save_scaffold_profile_run(
        self,
        *,
        run_id: str,
        config: Dict[str, Any],
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        self.conn.execute(
            """INSERT OR REPLACE INTO scaffold_profile_runs
               (run_id, timestamp, device, config_json, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                now,
                device,
                json.dumps(config or {}),
                json.dumps(metadata or {}),
            ),
        )
        self._maybe_commit()

    def save_scaffold_profile_result(
        self,
        *,
        run_id: str,
        family: str,
        case_name: str,
        status: str,
        metrics: Dict[str, Any],
        graph_json: Optional[str] = None,
        graph_fingerprint: Optional[str] = None,
        op_a: Optional[str] = None,
        op_b: Optional[str] = None,
    ) -> str:
        now = time.time()
        result_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO scaffold_profile_results
               (profile_result_id, run_id, timestamp, family, case_name, op_a, op_b,
                status, graph_json, graph_fingerprint, compile_time_ms,
                sandbox_passed, stability_score, causality_passed, param_count,
                passed, loss_ratio, validation_loss_ratio, discovery_loss_ratio,
                final_loss, avg_step_time_ms, throughput_tok_s, elapsed_s, error,
                metrics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result_id,
                run_id,
                now,
                family,
                case_name,
                op_a,
                op_b,
                status,
                graph_json,
                graph_fingerprint,
                metrics.get("compile_time_ms"),
                int(bool(metrics.get("sandbox_passed")))
                if metrics.get("sandbox_passed") is not None
                else None,
                metrics.get("stability_score"),
                int(bool(metrics.get("causality_passed")))
                if metrics.get("causality_passed") is not None
                else None,
                metrics.get("param_count"),
                int(bool(metrics.get("passed")))
                if metrics.get("passed") is not None
                else None,
                metrics.get("loss_ratio"),
                metrics.get("validation_loss_ratio"),
                metrics.get("discovery_loss_ratio"),
                metrics.get("final_loss"),
                metrics.get("avg_step_time_ms"),
                metrics.get("throughput_tok_s"),
                metrics.get("elapsed_s"),
                metrics.get("error"),
                json.dumps(metrics or {}),
            ),
        )
        self._maybe_commit()
        return result_id

    def list_scaffold_profile_results(
        self,
        *,
        run_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM scaffold_profile_results"
        params: List[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(max(1, limit)))
        rows = self.conn.execute(query, params).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            try:
                data["metrics"] = json.loads(data.get("metrics_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                data["metrics"] = {}
            results.append(data)
        return results

    def get_scaffold_component_stats(
        self,
        *,
        since_ts: float = 0.0,
        min_support: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-op scaffold profiling evidence for governance."""
        query = (
            "SELECT family, status, op_a, op_b, sandbox_passed, passed, "
            "loss_ratio, validation_loss_ratio, throughput_tok_s "
            "FROM scaffold_profile_results"
        )
        params: List[Any] = []
        if since_ts > 0:
            query += " WHERE timestamp >= ?"
            params.append(float(since_ts))
        rows = self.conn.execute(query, params).fetchall()

        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            record = dict(row)
            ops = {
                str(record.get("op_a") or "").strip(),
                str(record.get("op_b") or "").strip(),
            }
            ops.discard("")
            if not ops:
                continue
            family = str(record.get("family") or "").strip()
            status = str(record.get("status") or "").strip()
            sandbox_passed = int(bool(record.get("sandbox_passed")))
            passed = int(bool(record.get("passed")))
            loss_ratio = record.get("validation_loss_ratio")
            if loss_ratio is None:
                loss_ratio = record.get("loss_ratio")
            throughput = record.get("throughput_tok_s")
            for op_name in ops:
                bucket = buckets.setdefault(
                    op_name,
                    {
                        "support": 0,
                        "n_ok": 0,
                        "n_screen_fail": 0,
                        "n_error": 0,
                        "n_sandbox_passed": 0,
                        "n_passed": 0,
                        "loss_sum": 0.0,
                        "loss_n": 0,
                        "throughput_sum": 0.0,
                        "throughput_n": 0,
                        "family_counts": defaultdict(int),
                    },
                )
                bucket["support"] += 1
                if status == "ok":
                    bucket["n_ok"] += 1
                elif status == "screen_fail":
                    bucket["n_screen_fail"] += 1
                elif status == "error":
                    bucket["n_error"] += 1
                bucket["n_sandbox_passed"] += sandbox_passed
                bucket["n_passed"] += passed
                if isinstance(loss_ratio, (int, float)):
                    bucket["loss_sum"] += float(loss_ratio)
                    bucket["loss_n"] += 1
                if isinstance(throughput, (int, float)):
                    bucket["throughput_sum"] += float(throughput)
                    bucket["throughput_n"] += 1
                if family:
                    bucket["family_counts"][family] += 1

        result: Dict[str, Dict[str, Any]] = {}
        for op_name, bucket in buckets.items():
            support = int(bucket["support"])
            if support < max(1, int(min_support)):
                continue
            ok_rate = bucket["n_ok"] / support
            sandbox_rate = bucket["n_sandbox_passed"] / support
            pass_rate = bucket["n_passed"] / support
            avg_loss = (
                bucket["loss_sum"] / bucket["loss_n"] if bucket["loss_n"] else None
            )
            avg_tp = (
                bucket["throughput_sum"] / bucket["throughput_n"]
                if bucket["throughput_n"]
                else None
            )
            loss_term = 0.5
            if isinstance(avg_loss, float):
                loss_term = max(0.0, min(1.0, 1.0 - (avg_loss / 1.5)))
            throughput_term = 0.5
            if isinstance(avg_tp, float):
                throughput_term = max(0.0, min(1.0, avg_tp / 5000.0))
            raw_quality = (
                0.35 * ok_rate
                + 0.30 * pass_rate
                + 0.20 * sandbox_rate
                + 0.10 * loss_term
                + 0.05 * throughput_term
            )
            confidence = min(1.0, support / 12.0)
            prior_rate = (confidence * raw_quality) + ((1.0 - confidence) * 0.5)
            result[op_name] = {
                "support": support,
                "ok_rate": round(ok_rate, 4),
                "sandbox_rate": round(sandbox_rate, 4),
                "pass_rate": round(pass_rate, 4),
                "avg_loss_ratio": round(avg_loss, 6)
                if isinstance(avg_loss, float)
                else None,
                "avg_throughput_tok_s": round(avg_tp, 3)
                if isinstance(avg_tp, float)
                else None,
                "quality_score": round(raw_quality, 4),
                "prior_rate": round(prior_rate, 4),
                "families": dict(bucket["family_counts"]),
            }
        return result

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
        except (json.JSONDecodeError, TypeError):
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
            except (TypeError, ValueError):
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(
                    os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400")
                )
            except (TypeError, ValueError):
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
        results: List[Dict] = []
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
            results.append(item)
        return results

    def save_report_markdown(
        self, content: str, reason: str, summary: Optional[Dict] = None
    ) -> Optional[Path]:
        """Save a report as a markdown file alongside the database."""
        logger = LOGGER
        try:
            reports_dir = self.db_path.parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
            safe_reason = reason.replace(" ", "_").replace("/", "-")[:40]
            filename = f"report_{timestamp_str}_{safe_reason}.md"
            filepath = reports_dir / filename

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
            logger.info("Report saved to %s", filepath)
            return filepath
        except OSError as exc:
            logger.warning("Failed to save report markdown: %s", exc)
            return None
