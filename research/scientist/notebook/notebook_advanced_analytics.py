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
from typing import Any, Dict, List, Optional, Sequence

from ..json_utils import json_safe
from ..runtime_events import publish_runtime_event
from ._shared import LOGGER
from .failure_signature_audits import (
    AUDITED_FALSE_FAILURE_SIGNATURES,
    AUDITED_FALSE_FAILURE_SIGNATURE_SET,
)
from .notebook_analytics import (
    _ALL_CATEGORIES,
    _cached_extract_op_names,
    _cached_extract_template_name,
)


class _AdvancedAnalyticsMixin:
    """Advanced analytics, scaffold profiling, and report snapshot methods."""

    __slots__ = ()

    @staticmethod
    def _decode_json_field(
        data: Dict[str, Any], source_key: str, target_key: str
    ) -> None:
        try:
            data[target_key] = json.loads(data.get(source_key) or "{}")
        except (TypeError, json.JSONDecodeError):
            data[target_key] = {}

    @classmethod
    def _hydrate_designer_run_lineage_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        cls._decode_json_field(data, "metrics_json", "metrics")
        cls._decode_json_field(data, "payload_json", "payload")
        return data

    @classmethod
    def _hydrate_scaffold_profile_result_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        cls._decode_json_field(data, "metrics_json", "metrics")
        return data

    @staticmethod
    def _hydrate_attribution_report_row(row: Any) -> Dict[str, Any]:
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
        return item

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
        rows = list(
            self.conn.execute("""SELECT signature, n_failures, n_successes, error_types
               FROM failure_signatures""").fetchall()
        )
        suppressions = self._effective_failure_signature_suppressions(
            rows=rows,
            min_seen=5,
        )
        risk_signatures: List[Dict[str, Any]] = []
        critical: List[Dict[str, Any]] = []
        for row in rows:
            signature = str(row["signature"] or "")
            if signature in suppressions:
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
        survivor_row = self.conn.execute("""SELECT COUNT(*) AS n
               FROM program_results
               WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL""").fetchone()
        survivor_count = int(survivor_row["n"] or 0) if survivor_row else 0
        threshold = None
        if survivor_count >= 20:
            threshold_row = self.conn.execute("""SELECT loss_ratio FROM program_results
                   WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL
                   ORDER BY loss_ratio ASC
                   LIMIT 1 OFFSET (
                       SELECT MAX(0, COUNT(*) / 4 - 1) FROM program_results
                       WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL
                   )""").fetchone()
            threshold = float(threshold_row["loss_ratio"]) if threshold_row else None
        rows = self.conn.execute("""SELECT gp.signature, pr.loss_ratio, l.tier
               FROM program_graph_pairs gp
               JOIN program_results pr ON pr.result_id = gp.result_id
               LEFT JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE pr.stage1_passed = 1""").fetchall()
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

    def _get_failure_signature_suppressions(self) -> Dict[str, Dict[str, str]]:
        suppressions = {
            signature: {"reason": reason, "source": "audit"}
            for signature, reason in AUDITED_FALSE_FAILURE_SIGNATURES.items()
        }
        try:
            rows = self.conn.execute("""SELECT signature, reason, source
                   FROM failure_signature_suppressions
                   WHERE active = 1""").fetchall()
        except sqlite3.OperationalError:
            return suppressions
        for row in rows:
            signature = str(row["signature"] or "").strip()
            if not signature:
                continue
            suppressions[signature] = {
                "reason": str(row["reason"] or "").strip(),
                "source": str(row["source"] or "").strip() or "manual",
            }
        return suppressions

    def sync_failure_signature_suppressions(self) -> int:
        """Persist audited signature suppressions into the notebook database."""
        now = time.time()
        rows = [
            (signature, reason, "audit", 1, now, now)
            for signature, reason in AUDITED_FALSE_FAILURE_SIGNATURES.items()
        ]
        if not rows:
            return 0
        self.conn.executemany(
            """INSERT INTO failure_signature_suppressions
               (signature, reason, source, active, created_at, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(signature) DO UPDATE SET
                   reason = excluded.reason,
                   source = excluded.source,
                   active = 1,
                   last_updated = excluded.last_updated""",
            rows,
        )
        self._maybe_commit()
        return len(rows)

    def _failure_signature_op_fail_rates(
        self, ops: Sequence[str]
    ) -> Dict[str, Dict[str, float]]:
        op_names = sorted({str(op).strip() for op in ops if str(op).strip()})
        if not op_names:
            return {}
        placeholders = ",".join("?" for _ in op_names)
        rows = self.conn.execute(
            f"""
            SELECT
                go.op_name AS op_name,
                COUNT(DISTINCT pr.result_id) AS total,
                COUNT(DISTINCT CASE WHEN pr.stage1_passed = 1 THEN pr.result_id END) AS successes
            FROM program_graph_ops go
            JOIN program_results pr ON pr.result_id = go.result_id
            WHERE go.op_name IN ({placeholders})
              AND (
                  pr.stage1_passed = 1
                  OR (pr.stage0_passed = 1 AND pr.stage05_passed = 1)
              )
            GROUP BY go.op_name
            """,
            op_names,
        ).fetchall()
        stats: Dict[str, Dict[str, float]] = {}
        for row in rows:
            total = int(row["total"] or 0)
            successes = int(row["successes"] or 0)
            failures = max(0, total - successes)
            stats[str(row["op_name"])] = {
                "total": total,
                "successes": successes,
                "fail_rate": (float(failures) / float(total)) if total else 0.0,
            }
        return stats

    def _failure_signature_template_dominance(
        self, signatures: Sequence[str]
    ) -> Dict[str, Dict[str, Any]]:
        sigs = sorted({str(sig).strip() for sig in signatures if str(sig).strip()})
        if not sigs:
            return {}
        placeholders = ",".join("?" for _ in sigs)
        rows = self.conn.execute(
            f"""
            SELECT
                gp.signature AS signature,
                COALESCE(NULLIF(gf.template_name, ''), '') AS template_name,
                COUNT(*) AS support
            FROM program_graph_pairs gp
            JOIN program_results pr ON pr.result_id = gp.result_id
            LEFT JOIN program_graph_features gf ON gf.result_id = gp.result_id
            WHERE gp.signature IN ({placeholders})
              AND (
                  pr.stage1_passed = 1
                  OR (pr.stage0_passed = 1 AND pr.stage05_passed = 1)
              )
            GROUP BY gp.signature, COALESCE(NULLIF(gf.template_name, ''), '')
            """,
            sigs,
        ).fetchall()
        buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"known_total": 0, "best_template": "", "best_support": 0}
        )
        template_names: set[str] = set()
        for row in rows:
            signature = str(row["signature"] or "").strip()
            template_name = str(row["template_name"] or "").strip()
            support = int(row["support"] or 0)
            if not signature or not template_name or support <= 0:
                continue
            entry = buckets[signature]
            entry["known_total"] += support
            if support > int(entry["best_support"]):
                entry["best_support"] = support
                entry["best_template"] = template_name
            template_names.add(template_name)
        if not template_names:
            return {}
        template_placeholders = ",".join("?" for _ in template_names)
        template_rows = self.conn.execute(
            f"""
            SELECT template_name, eval_count, s1_pass_count
            FROM template_stats
            WHERE template_name IN ({template_placeholders})
            """,
            sorted(template_names),
        ).fetchall()
        template_rates = {}
        for row in template_rows:
            total = int(row["eval_count"] or 0)
            passed = int(row["s1_pass_count"] or 0)
            template_rates[str(row["template_name"] or "")] = (
                float(passed) / float(total) if total else 0.0
            )
        dominance: Dict[str, Dict[str, Any]] = {}
        for signature, entry in buckets.items():
            known_total = int(entry["known_total"] or 0)
            best_support = int(entry["best_support"] or 0)
            best_template = str(entry["best_template"] or "")
            if known_total <= 0 or best_support <= 0 or not best_template:
                continue
            dominance[signature] = {
                "template_name": best_template,
                "template_share": float(best_support) / float(known_total),
                "template_support": best_support,
                "template_s1_rate": float(template_rates.get(best_template, 0.0)),
            }
        return dominance

    def _heuristic_failure_signature_suppressions(
        self,
        *,
        rows: Optional[Sequence[Any]] = None,
        min_seen: int = 5,
    ) -> Dict[str, str]:
        source_rows = (
            list(rows)
            if rows is not None
            else list(
                self.conn.execute(
                    """SELECT signature, n_failures, n_successes, error_types
                   FROM failure_signatures"""
                ).fetchall()
            )
        )
        candidate_rows = []
        ops: set[str] = set()
        signatures: set[str] = set()
        for row in source_rows:
            signature = str(row["signature"] or "").strip()
            if not signature or signature in AUDITED_FALSE_FAILURE_SIGNATURE_SET:
                continue
            total = int(row["n_failures"] or 0) + int(row["n_successes"] or 0)
            if total < max(1, int(min_seen)):
                continue
            tokens = [tok.strip() for tok in signature.split("->") if tok.strip()]
            if len(tokens) != 2:
                continue
            src, dst = tokens
            candidate_rows.append((row, src, dst, total))
            ops.add(src)
            ops.add(dst)
            signatures.add(signature)
        op_rates = self._failure_signature_op_fail_rates(sorted(ops))
        template_dominance = self._failure_signature_template_dominance(
            sorted(signatures)
        )
        suppressions: Dict[str, str] = {}
        for row, src, dst, total in candidate_rows:
            signature = str(row["signature"] or "").strip()
            pair_fail_rate = float(row["n_failures"] or 0) / float(total)
            src_fail = float(op_rates.get(src, {}).get("fail_rate", 0.0))
            dst_fail = float(op_rates.get(dst, {}).get("fail_rate", 0.0))
            dominant_fail = max(src_fail, dst_fail)
            dominant_op = src if src_fail >= dst_fail else dst
            if dominant_fail >= 0.90 and pair_fail_rate <= dominant_fail + 0.05:
                suppressions[signature] = (
                    f"Pair fail rate is dominated by globally weak endpoint '{dominant_op}' "
                    f"({dominant_fail:.0%} fail), so this is not clean adjacency evidence."
                )
                continue
            tpl = template_dominance.get(signature)
            if (
                tpl
                and float(tpl.get("template_share") or 0.0) >= 0.75
                and int(tpl.get("template_support") or 0) >= 6
                and float(tpl.get("template_s1_rate") or 0.0) <= 0.15
            ):
                suppressions[signature] = (
                    f"Failures are dominated by weak template '{tpl['template_name']}' "
                    f"({float(tpl['template_share']):.0%} of labeled rows), not by a stable "
                    "pair-level incompatibility."
                )
        return suppressions

    def _effective_failure_signature_suppressions(
        self,
        *,
        rows: Optional[Sequence[Any]] = None,
        min_seen: int = 5,
    ) -> Dict[str, Dict[str, str]]:
        suppressions = dict(self._get_failure_signature_suppressions())
        heuristics = self._heuristic_failure_signature_suppressions(
            rows=rows,
            min_seen=min_seen,
        )
        for signature, reason in heuristics.items():
            suppressions.setdefault(
                signature,
                {"reason": reason, "source": "heuristic"},
            )
        return suppressions

    def prune_suppressed_failure_signatures(self) -> int:
        suppressions = self._get_failure_signature_suppressions()
        if not suppressions:
            return 0
        signatures = sorted(suppressions.keys())
        placeholders = ",".join("?" for _ in signatures)
        deleted = self.conn.execute(
            f"SELECT COUNT(*) FROM failure_signatures WHERE signature IN ({placeholders})",
            signatures,
        ).fetchone()[0]
        self.conn.execute(
            f"DELETE FROM failure_signatures WHERE signature IN ({placeholders})",
            signatures,
        )
        self._maybe_commit()
        return int(deleted or 0)

    def refresh_failure_signature_suppressions(
        self,
        *,
        include_heuristics: bool = True,
        prune_rows: bool = True,
        min_seen: int = 5,
    ) -> Dict[str, int]:
        """Persist current suppression decisions and optionally prune raw rows."""
        seeded = self.sync_failure_signature_suppressions()
        heuristic_added = 0
        if include_heuristics:
            now = time.time()
            heuristic_rows = [
                (signature, reason, "heuristic", 1, now, now)
                for signature, reason in self._heuristic_failure_signature_suppressions(
                    min_seen=min_seen
                ).items()
                if signature not in AUDITED_FALSE_FAILURE_SIGNATURE_SET
            ]
            if heuristic_rows:
                self.conn.executemany(
                    """INSERT INTO failure_signature_suppressions
                       (signature, reason, source, active, created_at, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(signature) DO UPDATE SET
                           reason = excluded.reason,
                           source = excluded.source,
                           active = 1,
                           last_updated = excluded.last_updated""",
                    heuristic_rows,
                )
                heuristic_added = len(heuristic_rows)
        deleted = self.prune_suppressed_failure_signatures() if prune_rows else 0
        self._maybe_commit()
        return {
            "seeded": int(seeded),
            "heuristic_added": int(heuristic_added),
            "deleted": int(deleted),
        }

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
        suppressions = self._get_failure_signature_suppressions()
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
        filtered_rows = [
            row for row in rows if str(row["signature"] or "") not in suppressions
        ]
        if not filtered_rows:
            return
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
                for row in filtered_rows
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
        suppressions = self._get_failure_signature_suppressions()
        rows = self.conn.execute("""
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
            """).fetchall()
        rows = [row for row in rows if str(row["signature"] or "") not in suppressions]
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
        suppressions = self._get_failure_signature_suppressions()
        rows = self.conn.execute("""
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
            """).fetchall()
        rows = [row for row in rows if str(row["signature"] or "") not in suppressions]
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
        rows = list(
            self.conn.execute(
                """SELECT signature, n_failures, n_successes
               FROM failure_signatures
               WHERE (n_failures + n_successes) >= ?""",
                (min_seen,),
            ).fetchall()
        )
        suppressions = self._effective_failure_signature_suppressions(
            rows=rows,
            min_seen=min_seen,
        )
        blocklist: Dict[str, float] = {}
        for row in rows:
            signature = str(row[0] or "")
            if signature in suppressions:
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
        return self._hydrate_designer_run_lineage_row(row)

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
        cursor = self.conn.execute(query, params)
        return [self._hydrate_designer_run_lineage_row(row) for row in cursor]

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
                (
                    int(bool(metrics.get("sandbox_passed")))
                    if metrics.get("sandbox_passed") is not None
                    else None
                ),
                metrics.get("stability_score"),
                (
                    int(bool(metrics.get("causality_passed")))
                    if metrics.get("causality_passed") is not None
                    else None
                ),
                metrics.get("param_count"),
                (
                    int(bool(metrics.get("passed")))
                    if metrics.get("passed") is not None
                    else None
                ),
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
        cursor = self.conn.execute(query, params)
        return [self._hydrate_scaffold_profile_result_row(row) for row in cursor]

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
        cursor = self.conn.execute(query, params)

        buckets: Dict[str, Dict[str, Any]] = {}
        for record in cursor:
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
                "avg_loss_ratio": (
                    round(avg_loss, 6) if isinstance(avg_loss, float) else None
                ),
                "avg_throughput_tok_s": (
                    round(avg_tp, 3) if isinstance(avg_tp, float) else None
                ),
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
                json.dumps(
                    json_safe(query or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(json_safe(payload or {}), separators=(",", ":")),
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
        rows = self.conn.execute("""SELECT scope,
                      COUNT(*) AS count,
                      MIN(updated_at) AS oldest_updated_at,
                      MAX(updated_at) AS newest_updated_at
               FROM report_snapshots
               GROUP BY scope
               ORDER BY count DESC, scope ASC""").fetchall()

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
                    "oldest_age_seconds": (
                        round(max(0.0, now - oldest), 2) if oldest > 0 else None
                    ),
                    "newest_age_seconds": (
                        round(max(0.0, now - newest), 2) if newest > 0 else None
                    ),
                }
            )

        return {
            "total_snapshots": total,
            "n_scopes": len(scopes),
            "oldest_age_seconds": (
                round(max(0.0, now - oldest_seen), 2) if oldest_seen else None
            ),
            "newest_age_seconds": (
                round(max(0.0, now - newest_seen), 2) if newest_seen else None
            ),
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

    def record_causal_rule_evidence(self, evidence: Dict[str, Any]) -> str:
        """Persist one causal ablation evidence row."""
        evidence_id = str(evidence.get("evidence_id") or uuid.uuid4())[:12]
        now = float(evidence.get("timestamp") or time.time())
        self.conn.execute(
            """INSERT OR REPLACE INTO causal_rule_evidence
               (evidence_id, timestamp, parent_experiment_id, parent_result_id,
                parent_fingerprint, ablation_experiment_id, rule_type, rule_key,
                rule_context, original_loss_ratio, ablation_best_loss_ratio,
                effect_size, original_stage1_passed, ablation_stage1_pass_count,
                ablation_total, outcome, confidence, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence_id,
                now,
                evidence.get("parent_experiment_id"),
                evidence.get("parent_result_id"),
                evidence.get("parent_fingerprint"),
                evidence.get("ablation_experiment_id"),
                str(evidence.get("rule_type") or "unknown"),
                str(evidence.get("rule_key") or "unknown"),
                evidence.get("rule_context"),
                evidence.get("original_loss_ratio"),
                evidence.get("ablation_best_loss_ratio"),
                evidence.get("effect_size"),
                int(bool(evidence.get("original_stage1_passed", True))),
                int(evidence.get("ablation_stage1_pass_count") or 0),
                int(evidence.get("ablation_total") or 0),
                str(evidence.get("outcome") or "inconclusive"),
                float(evidence.get("confidence") or 0.0),
                evidence.get("evidence_json"),
            ),
        )
        self._maybe_commit()
        return evidence_id

    def record_causal_ablation_child_observations(
        self,
        evidence_id: str,
        observations: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist child/provenance rows supporting one causal evidence item."""
        if not evidence_id or not observations:
            return 0
        now = time.time()
        rows = []
        for observation in observations:
            child_fingerprint = str(
                observation.get("child_fingerprint")
                or observation.get("graph_fingerprint")
                or ""
            )
            if not child_fingerprint:
                continue
            observation_id = str(observation.get("observation_id") or uuid.uuid4())[:12]
            rows.append(
                (
                    observation_id,
                    evidence_id,
                    float(observation.get("timestamp") or now),
                    observation.get("parent_result_id"),
                    observation.get("parent_experiment_id"),
                    observation.get("parent_fingerprint"),
                    observation.get("child_result_id") or observation.get("result_id"),
                    observation.get("child_experiment_id")
                    or observation.get("experiment_id"),
                    child_fingerprint,
                    observation.get("ablation_experiment_id"),
                    str(observation.get("source") or "unknown"),
                    str(observation.get("rule_type") or "unknown"),
                    str(observation.get("rule_key") or "unknown"),
                    observation.get("stage0_passed"),
                    observation.get("stage05_passed"),
                    observation.get("stage1_passed"),
                    observation.get("loss_ratio"),
                    observation.get("final_loss"),
                    observation.get("model_source"),
                    observation.get("trust_label"),
                    observation.get("comparability_label"),
                    json.dumps(json_safe(observation.get("provenance") or {})),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """INSERT OR IGNORE INTO causal_ablation_child_observations
               (observation_id, evidence_id, timestamp, parent_result_id,
                parent_experiment_id, parent_fingerprint, child_result_id,
                child_experiment_id, child_fingerprint, ablation_experiment_id,
                source, rule_type, rule_key, stage0_passed, stage05_passed,
                stage1_passed, loss_ratio, final_loss, model_source, trust_label,
                comparability_label, provenance_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._maybe_commit()
        return len(rows)

    def get_causal_ablation_child_observations(
        self,
        *,
        evidence_id: Optional[str] = None,
        result_id: Optional[str] = None,
        rule_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return ablation child observations with provenance JSON hydrated."""
        clauses = ["1=1"]
        params: List[Any] = []
        if evidence_id:
            clauses.append("evidence_id = ?")
            params.append(evidence_id)
        if result_id:
            clauses.append("parent_result_id = ?")
            params.append(result_id)
        if rule_type:
            clauses.append("rule_type = ?")
            params.append(rule_type)
        params.append(max(1, min(int(limit or 200), 2000)))
        rows = self.conn.execute(
            f"""SELECT * FROM causal_ablation_child_observations
                WHERE {" AND ".join(clauses)}
                ORDER BY timestamp DESC
                LIMIT ?""",
            tuple(params),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.get("provenance_json")
            if isinstance(raw, str) and raw.strip():
                try:
                    item["provenance"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            out.append(item)
        return out

    def get_causal_component_interaction_summary(
        self,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Aggregate causal evidence by rule with compact provenance samples."""
        capped_limit = max(1, min(int(limit or 50), 500))
        rows = self.conn.execute(
            """WITH evidence AS (
                   SELECT rule_type,
                          rule_key,
                          COUNT(*) AS evidence_count,
                          SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                              AS supported_count,
                          SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                              AS refuted_count,
                          SUM(CASE WHEN outcome = 'inconclusive' THEN 1 ELSE 0 END)
                              AS inconclusive_count,
                          AVG(confidence) AS avg_confidence,
                          AVG(effect_size) AS avg_effect_size,
                          MIN(effect_size) AS min_effect_size,
                          MAX(effect_size) AS max_effect_size
                   FROM causal_rule_evidence
                   GROUP BY rule_type, rule_key
               ),
               children AS (
                   SELECT rule_type,
                          rule_key,
                          COUNT(DISTINCT child_result_id) AS child_result_count,
                          COUNT(DISTINCT child_fingerprint)
                              AS child_fingerprint_count,
                          GROUP_CONCAT(DISTINCT source) AS child_sources,
                          GROUP_CONCAT(DISTINCT child_experiment_id)
                              AS child_experiments
                   FROM causal_ablation_child_observations
                   GROUP BY rule_type, rule_key
               ),
               metric_rows AS (
                   SELECT obs.rule_type,
                          obs.rule_key,
                          cp.result_id AS child_result_id,
                          CASE
                              WHEN cp.hellaswag_acc IS NOT NULL
                               AND cp.blimp_overall_accuracy IS NOT NULL
                               AND cp.induction_auc IS NOT NULL
                               AND cp.binding_auc IS NOT NULL
                               AND cp.binding_composite IS NOT NULL
                               AND cp.ar_auc IS NOT NULL
                               AND cp.wikitext_perplexity IS NOT NULL
                               AND cp.wikitext_score IS NOT NULL
                               AND cp.fp_jacobian_erf_density IS NOT NULL
                               AND cp.fp_icld_delta_loss IS NOT NULL
                               AND cp.fp_logit_margin_delta IS NOT NULL
                              THEN 1 ELSE 0
                          END AS metric_complete,
                          CASE
                              WHEN cp.loss_ratio IS NOT NULL
                               AND pp.loss_ratio IS NOT NULL
                              THEN cp.loss_ratio - pp.loss_ratio
                          END AS loss_support_effect,
                          CASE
                              WHEN cp.hellaswag_acc IS NOT NULL
                               AND pp.hellaswag_acc IS NOT NULL
                              THEN pp.hellaswag_acc - cp.hellaswag_acc
                          END AS hellaswag_support_effect,
                          CASE
                              WHEN cp.blimp_overall_accuracy IS NOT NULL
                               AND pp.blimp_overall_accuracy IS NOT NULL
                              THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy
                          END AS blimp_support_effect,
                          CASE
                              WHEN cp.induction_auc IS NOT NULL
                               AND pp.induction_auc IS NOT NULL
                              THEN pp.induction_auc - cp.induction_auc
                          END AS induction_support_effect,
                          CASE
                              WHEN cp.binding_composite IS NOT NULL
                               AND pp.binding_composite IS NOT NULL
                              THEN pp.binding_composite - cp.binding_composite
                          END AS binding_support_effect,
                          CASE
                              WHEN cp.ar_auc IS NOT NULL
                               AND pp.ar_auc IS NOT NULL
                              THEN pp.ar_auc - cp.ar_auc
                          END AS ar_support_effect,
                          CASE
                              WHEN cp.wikitext_score IS NOT NULL
                               AND pp.wikitext_score IS NOT NULL
                              THEN pp.wikitext_score - cp.wikitext_score
                          END AS wikitext_support_effect
                   FROM causal_ablation_child_observations obs
                   LEFT JOIN program_results cp
                     ON cp.result_id = obs.child_result_id
                   LEFT JOIN program_results pp
                     ON pp.result_id = obs.parent_result_id
               ),
               metric_scored AS (
                   SELECT *,
                          (
                              CASE WHEN loss_support_effect IS NOT NULL THEN 0.35 ELSE 0 END
                            + CASE WHEN hellaswag_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                            + CASE WHEN blimp_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                            + CASE WHEN induction_support_effect IS NOT NULL THEN 0.15 ELSE 0 END
                            + CASE WHEN binding_support_effect IS NOT NULL THEN 0.15 ELSE 0 END
                            + CASE WHEN ar_support_effect IS NOT NULL THEN 0.05 ELSE 0 END
                            + CASE WHEN wikitext_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                          ) AS metric_weight,
                          (
                              COALESCE(0.35 * loss_support_effect, 0.0)
                            + COALESCE(0.10 * hellaswag_support_effect, 0.0)
                            + COALESCE(0.10 * blimp_support_effect, 0.0)
                            + COALESCE(0.15 * induction_support_effect, 0.0)
                            + COALESCE(0.15 * binding_support_effect, 0.0)
                            + COALESCE(0.05 * ar_support_effect, 0.0)
                            + COALESCE(0.10 * wikitext_support_effect, 0.0)
                          ) AS metric_effect_numerator
                   FROM metric_rows
               ),
               metrics AS (
                   SELECT rule_type,
                          rule_key,
                          COUNT(child_result_id) AS metric_observation_count,
                          SUM(metric_complete) AS metric_complete_count,
                          SUM(CASE WHEN metric_weight > 0 THEN 1 ELSE 0 END)
                              AS metric_comparable_count,
                          AVG(loss_support_effect) AS avg_loss_support_effect,
                          AVG(hellaswag_support_effect)
                              AS avg_hellaswag_support_effect,
                          AVG(blimp_support_effect) AS avg_blimp_support_effect,
                          AVG(induction_support_effect)
                              AS avg_induction_support_effect,
                          AVG(binding_support_effect)
                              AS avg_binding_support_effect,
                          AVG(ar_support_effect) AS avg_ar_support_effect,
                          AVG(wikitext_support_effect)
                              AS avg_wikitext_support_effect,
                          AVG(
                              CASE WHEN metric_weight > 0
                                   THEN metric_effect_numerator / metric_weight
                              END
                          ) AS composite_support_effect
                   FROM metric_scored
                   GROUP BY rule_type, rule_key
               )
               SELECT evidence.*,
                      COALESCE(children.child_result_count, 0)
                          AS child_result_count,
                      COALESCE(children.child_fingerprint_count, 0)
                          AS child_fingerprint_count,
                      CASE
                          WHEN evidence.evidence_count > 0
                          THEN CAST(evidence.supported_count AS REAL)
                               / evidence.evidence_count
                          ELSE 0.0
                      END AS support_rate,
                      CASE
                          WHEN evidence.evidence_count > 0
                          THEN CAST(evidence.refuted_count AS REAL)
                               / evidence.evidence_count
                          ELSE 0.0
                      END AS refute_rate,
                      (evidence.supported_count - evidence.refuted_count)
                          AS net_support,
                      (
                          ABS(COALESCE(evidence.avg_effect_size, 0.0))
                          * SQRT(CAST(evidence.evidence_count AS REAL))
                          * COALESCE(evidence.avg_confidence, 0.0)
                      ) AS stability_score,
                      COALESCE(metrics.metric_observation_count, 0)
                          AS metric_observation_count,
                      COALESCE(metrics.metric_complete_count, 0)
                          AS metric_complete_count,
                      COALESCE(metrics.metric_comparable_count, 0)
                          AS metric_comparable_count,
                      CASE
                          WHEN COALESCE(metrics.metric_observation_count, 0) > 0
                          THEN CAST(metrics.metric_complete_count AS REAL)
                               / metrics.metric_observation_count
                          ELSE 0.0
                      END AS metric_complete_rate,
                      metrics.avg_loss_support_effect,
                      metrics.avg_hellaswag_support_effect,
                      metrics.avg_blimp_support_effect,
                      metrics.avg_induction_support_effect,
                      metrics.avg_binding_support_effect,
                      metrics.avg_ar_support_effect,
                      metrics.avg_wikitext_support_effect,
                      metrics.composite_support_effect,
                      children.child_sources,
                      children.child_experiments
               FROM evidence
               LEFT JOIN children
                 ON children.rule_type = evidence.rule_type
                AND children.rule_key = evidence.rule_key
               LEFT JOIN metrics
                 ON metrics.rule_type = evidence.rule_type
                AND metrics.rule_key = evidence.rule_key
               ORDER BY
                        CASE
                            WHEN evidence.evidence_count >= 3
                             AND COALESCE(children.child_fingerprint_count, 0) >= 3
                             AND COALESCE(metrics.metric_complete_count, 0) >= 3
                            THEN 0 ELSE 1
                        END,
                        ABS(COALESCE(metrics.composite_support_effect, 0.0)) DESC,
                        stability_score DESC,
                        evidence.evidence_count DESC,
                        ABS(COALESCE(evidence.avg_effect_size, 0.0)) DESC
               LIMIT ?""",
            (capped_limit,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            experiments = [
                token
                for token in str(item.pop("child_experiments") or "").split(",")
                if token
            ]
            item["child_experiment_samples"] = experiments[:12]
            item["child_sources"] = [
                token
                for token in str(item.get("child_sources") or "").split(",")
                if token
            ]
            out.append(item)
        return out

    def get_causal_rule_evidence(
        self,
        *,
        result_id: Optional[str] = None,
        rule_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return causal rule evidence rows, newest first."""
        clauses = ["1=1"]
        params: List[Any] = []
        if result_id:
            clauses.append("parent_result_id = ?")
            params.append(result_id)
        if rule_type:
            clauses.append("rule_type = ?")
            params.append(rule_type)
        params.append(max(1, min(int(limit or 50), 500)))
        rows = self.conn.execute(
            f"""SELECT * FROM causal_rule_evidence
                WHERE {" AND ".join(clauses)}
                ORDER BY timestamp DESC
                LIMIT ?""",
            tuple(params),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("rule_context", "evidence_json"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    try:
                        item[f"{key}_parsed"] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
            out.append(item)
        return out

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
        cursor = self.conn.execute(query, params)
        return [self._hydrate_attribution_report_row(row) for row in cursor]

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
