from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import math
from functools import lru_cache
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from ..json_utils import fast_loads as _json_loads
from ..thresholds import GPT2_REF
from ...synthesis.templates import TEMPLATES


@lru_cache(maxsize=8192)
def _cached_extract_op_bigrams(graph_json: str) -> tuple[str, ...]:
    try:
        data = _json_loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return ()
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return ()
    bigrams: set[str] = set()
    for nd in nodes.values():
        if not isinstance(nd, dict):
            continue
        op = nd.get("op_name", "")
        if not op or op == "input":
            continue
        for inp in nd.get("input_ids", ()):
            parent = nodes.get(str(inp), {})
            if not isinstance(parent, dict):
                continue
            pop = parent.get("op_name", "")
            if pop and pop != "input":
                bigrams.add(f"{pop}->{op}")
    return tuple(sorted(bigrams))


class _MiscMixin:
    """Misc operations for the Lab Notebook."""

    __slots__ = ()
    _DASHBOARD_SUMMARY_TTL_S = 2.0

    @staticmethod
    def _percentile(values: List[float], pct: float) -> Optional[float]:
        clean = sorted(v for v in values if v is not None and math.isfinite(v))
        if not clean:
            return None
        if len(clean) == 1:
            return float(clean[0])
        pos = max(0.0, min(1.0, pct)) * (len(clean) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(clean[lo])
        frac = pos - lo
        return float(clean[lo] + (clean[hi] - clean[lo]) * frac)

    @staticmethod
    def _infer_template_slot_counts() -> Dict[str, int]:
        counts: Dict[str, int] = {}
        template_dir = Path(__file__).resolve().parents[2] / "synthesis"
        template_files = sorted(template_dir.glob("_templates*.py"))
        for template_file in template_files:
            try:
                source = template_file.read_text(encoding="utf-8")
            except OSError:
                continue
            matches = list(
                re.finditer(r"^def\s+(tpl_[A-Za-z0-9_]+)\s*\(", source, re.M)
            )
            for idx, match in enumerate(matches):
                name = match.group(1)
                start = match.start()
                end = (
                    matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
                )
                body = source[start:end]
                counts[name.removeprefix("tpl_")] = body.count(
                    "_pick_compatible_motif("
                ) + body.count("_pick_compatible_motif_from_classes(")
        return counts

    def get_template_slot_observability(self, limit: int = 8) -> Dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT
                experiment_id,
                timestamp,
                graph_json,
                stage0_passed,
                stage05_passed,
                stage1_passed,
                loss_ratio,
                discovery_loss_ratio,
                validation_loss_ratio,
                novelty_score,
                error_type,
                stage_at_death,
                failure_details_json,
                routing_fast_lane_applied,
                routing_fast_lane_status,
                routing_fast_lane_score,
                routing_fast_lane_ppl_improvement,
                routing_fast_lane_slope,
                routing_fast_lane_slope_consistent
            FROM program_results
            WHERE graph_json IS NOT NULL
            """
        ).fetchall()
        if not rows:
            return {
                "top_templates": [],
                "struggling_templates": [],
                "motif_slots": [],
                "loss_distribution": {},
                "recommendations": [],
                "summary": {},
            }

        slot_counts = self._infer_template_slot_counts()
        template_stats: Dict[str, Dict[str, Any]] = {}
        motif_stats: Dict[str, Dict[str, Any]] = {}
        slot_stats: Dict[str, Dict[str, Any]] = {}
        experiment_buckets: Dict[str, Dict[str, Any]] = {}
        loss_values: List[float] = []
        validation_losses: List[float] = []
        discovery_losses: List[float] = []
        motifs_per_graph: List[float] = []
        templates_per_graph: List[float] = []

        for row in rows:
            try:
                graph = _json_loads(row["graph_json"]) if row["graph_json"] else {}
            except (json.JSONDecodeError, TypeError, KeyError):
                graph = {}
            metadata = graph.get("metadata", {}) if isinstance(graph, dict) else {}
            templates = metadata.get("templates_used") or []
            motifs = metadata.get("motifs_used") or []
            slot_usage = metadata.get("template_slot_usage") or []
            experiment_id = str(row["experiment_id"] or "")
            exp_bucket = experiment_buckets.setdefault(
                experiment_id or f"exp_{len(experiment_buckets)}",
                {
                    "experiment_id": experiment_id or None,
                    "timestamp": float(row["timestamp"] or 0.0),
                    "templates": {},
                    "slots": {},
                    "training_losses": [],
                    "validation_losses": [],
                    "discovery_losses": [],
                },
            )
            exp_bucket["timestamp"] = max(
                float(exp_bucket.get("timestamp") or 0.0),
                float(row["timestamp"] or 0.0),
            )
            if not isinstance(templates, list):
                templates = []
            if not isinstance(motifs, list):
                motifs = []
            if not isinstance(slot_usage, list):
                slot_usage = []

            motifs_per_graph.append(float(len(motifs)))
            templates_per_graph.append(float(len(templates)))

            loss_ratio = row["loss_ratio"]
            validation_lr = row["validation_loss_ratio"]
            discovery_lr = row["discovery_loss_ratio"]
            novelty = row["novelty_score"]
            if loss_ratio is not None and math.isfinite(loss_ratio):
                loss_values.append(float(loss_ratio))
                exp_bucket["training_losses"].append(float(loss_ratio))
            if validation_lr is not None and math.isfinite(validation_lr):
                validation_losses.append(float(validation_lr))
                exp_bucket["validation_losses"].append(float(validation_lr))
            if discovery_lr is not None and math.isfinite(discovery_lr):
                discovery_losses.append(float(discovery_lr))
                exp_bucket["discovery_losses"].append(float(discovery_lr))

            failure_details = {}
            raw_failure = row["failure_details_json"]
            if raw_failure:
                try:
                    failure_details = (
                        _json_loads(raw_failure)
                        if isinstance(raw_failure, str)
                        else raw_failure
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    failure_details = {}
            root_cause = (
                failure_details.get("root_cause_code")
                or row["error_type"]
                or row["stage_at_death"]
                or "unknown"
            )

            for template in templates:
                stat = template_stats.setdefault(
                    str(template),
                    {
                        "name": str(template),
                        "n_used": 0,
                        "n_stage0": 0,
                        "n_stage05": 0,
                        "n_stage1": 0,
                        "losses": [],
                        "validation_losses": [],
                        "discovery_losses": [],
                        "novelties": [],
                        "failure_reasons": {},
                        "slot_count": slot_counts.get(str(template), 0),
                        "routing_fast_lane_runs": 0,
                        "routing_fast_lane_ok": 0,
                        "routing_fast_lane_positive": 0,
                        "routing_fast_lane_scores": [],
                        "routing_fast_lane_improvements": [],
                        "routing_fast_lane_slopes": [],
                    },
                )
                stat["n_used"] += 1
                stat["n_stage0"] += 1 if row["stage0_passed"] else 0
                stat["n_stage05"] += 1 if row["stage05_passed"] else 0
                stat["n_stage1"] += 1 if row["stage1_passed"] else 0
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    stat["losses"].append(float(loss_ratio))
                if validation_lr is not None and math.isfinite(validation_lr):
                    stat["validation_losses"].append(float(validation_lr))
                if discovery_lr is not None and math.isfinite(discovery_lr):
                    stat["discovery_losses"].append(float(discovery_lr))
                if novelty is not None and math.isfinite(novelty):
                    stat["novelties"].append(float(novelty))
                if row["routing_fast_lane_applied"]:
                    stat["routing_fast_lane_runs"] += 1
                    if row["routing_fast_lane_status"] == "ok":
                        stat["routing_fast_lane_ok"] += 1
                    lane_score = row["routing_fast_lane_score"]
                    lane_improvement = row["routing_fast_lane_ppl_improvement"]
                    lane_slope = row["routing_fast_lane_slope"]
                    slope_consistent = bool(row["routing_fast_lane_slope_consistent"])
                    if lane_score is not None and math.isfinite(lane_score):
                        stat["routing_fast_lane_scores"].append(float(lane_score))
                    if lane_improvement is not None and math.isfinite(lane_improvement):
                        stat["routing_fast_lane_improvements"].append(
                            float(lane_improvement)
                        )
                    if lane_slope is not None and math.isfinite(lane_slope):
                        stat["routing_fast_lane_slopes"].append(float(lane_slope))
                    if (
                        lane_improvement is not None and float(lane_improvement) < 0.98
                    ) or (
                        lane_slope is not None
                        and float(lane_slope) > 0
                        and slope_consistent
                    ):
                        stat["routing_fast_lane_positive"] += 1
                if not row["stage1_passed"]:
                    stat["failure_reasons"][root_cause] = (
                        stat["failure_reasons"].get(root_cause, 0) + 1
                    )
                exp_tpl = exp_bucket["templates"].setdefault(
                    str(template),
                    {"n": 0, "s1": 0, "losses": []},
                )
                exp_tpl["n"] += 1
                exp_tpl["s1"] += 1 if row["stage1_passed"] else 0
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    exp_tpl["losses"].append(float(loss_ratio))

            for motif in motifs:
                mstat = motif_stats.setdefault(
                    str(motif),
                    {
                        "name": str(motif),
                        "n_used": 0,
                        "n_stage1": 0,
                        "losses": [],
                        "failure_reasons": {},
                    },
                )
                mstat["n_used"] += 1
                mstat["n_stage1"] += 1 if row["stage1_passed"] else 0
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    mstat["losses"].append(float(loss_ratio))
                if not row["stage1_passed"]:
                    mstat["failure_reasons"][root_cause] = (
                        mstat["failure_reasons"].get(root_cause, 0) + 1
                    )

            for slot in slot_usage:
                if not isinstance(slot, dict):
                    continue
                slot_key = str(
                    slot.get("slot_key")
                    or f"{slot.get('template_name', 'unknown')}.slot{slot.get('slot_index', 0)}"
                )
                sstat = slot_stats.setdefault(
                    slot_key,
                    {
                        "slot_key": slot_key,
                        "template_name": str(slot.get("template_name") or "unknown"),
                        "slot_index": int(slot.get("slot_index") or 0),
                        "slot_classes": list(slot.get("slot_classes") or []),
                        "n_used": 0,
                        "n_stage1": 0,
                        "losses": [],
                        "failure_reasons": {},
                        "selected_motifs": {},
                    },
                )
                sstat["n_used"] += 1
                sstat["n_stage1"] += 1 if row["stage1_passed"] else 0
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    sstat["losses"].append(float(loss_ratio))
                motif_name = slot.get("selected_motif")
                if motif_name:
                    sstat["selected_motifs"][str(motif_name)] = (
                        sstat["selected_motifs"].get(str(motif_name), 0) + 1
                    )
                if not row["stage1_passed"]:
                    sstat["failure_reasons"][root_cause] = (
                        sstat["failure_reasons"].get(root_cause, 0) + 1
                    )
                exp_slot = exp_bucket["slots"].setdefault(
                    slot_key,
                    {"n": 0, "s1": 0, "losses": []},
                )
                exp_slot["n"] += 1
                exp_slot["s1"] += 1 if row["stage1_passed"] else 0
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    exp_slot["losses"].append(float(loss_ratio))

        def summarize_template(stat: Dict[str, Any]) -> Dict[str, Any]:
            losses = stat["losses"]
            validation_vals = stat["validation_losses"]
            discovery_vals = stat["discovery_losses"]
            novelties = stat["novelties"]
            reasons = stat["failure_reasons"]
            fast_lane_scores = stat["routing_fast_lane_scores"]
            fast_lane_improvements = stat["routing_fast_lane_improvements"]
            fast_lane_slopes = stat["routing_fast_lane_slopes"]
            top_reason = None
            if reasons:
                top_reason = max(reasons.items(), key=lambda item: item[1])[0]
            return {
                "name": stat["name"],
                "n_used": stat["n_used"],
                "s0_rate": stat["n_stage0"] / max(stat["n_used"], 1),
                "s05_rate": stat["n_stage05"] / max(stat["n_used"], 1),
                "s1_rate": stat["n_stage1"] / max(stat["n_used"], 1),
                "avg_loss_ratio": (sum(losses) / len(losses) if losses else None),
                "best_loss_ratio": min(losses) if losses else None,
                "avg_validation_loss_ratio": (
                    sum(validation_vals) / len(validation_vals)
                    if validation_vals
                    else None
                ),
                "avg_discovery_loss_ratio": (
                    sum(discovery_vals) / len(discovery_vals)
                    if discovery_vals
                    else None
                ),
                "avg_novelty": (sum(novelties) / len(novelties) if novelties else None),
                "slot_count": int(stat.get("slot_count") or 0),
                "routing_fast_lane_runs": int(stat.get("routing_fast_lane_runs") or 0),
                "routing_fast_lane_ok_rate": (
                    stat["routing_fast_lane_ok"]
                    / max(int(stat.get("routing_fast_lane_runs") or 0), 1)
                    if stat.get("routing_fast_lane_runs")
                    else None
                ),
                "routing_fast_lane_positive_rate": (
                    stat["routing_fast_lane_positive"]
                    / max(int(stat.get("routing_fast_lane_runs") or 0), 1)
                    if stat.get("routing_fast_lane_runs")
                    else None
                ),
                "routing_fast_lane_avg_score": (
                    sum(fast_lane_scores) / len(fast_lane_scores)
                    if fast_lane_scores
                    else None
                ),
                "routing_fast_lane_avg_improvement": (
                    sum(fast_lane_improvements) / len(fast_lane_improvements)
                    if fast_lane_improvements
                    else None
                ),
                "routing_fast_lane_avg_slope": (
                    sum(fast_lane_slopes) / len(fast_lane_slopes)
                    if fast_lane_slopes
                    else None
                ),
                "top_failure_reason": top_reason,
                "failure_reasons": dict(
                    sorted(reasons.items(), key=lambda item: -item[1])[:3]
                ),
            }

        template_rows = [summarize_template(stat) for stat in template_stats.values()]
        template_rows = [row for row in template_rows if row["n_used"] > 0]
        active_template_names = frozenset(TEMPLATES)
        active_template_rows = [
            row for row in template_rows if row["name"] in active_template_names
        ]
        inactive_template_rows = [
            row for row in template_rows if row["name"] not in active_template_names
        ]
        top_templates = sorted(
            active_template_rows,
            key=lambda row: (
                -(row["s1_rate"] or 0.0),
                row["avg_validation_loss_ratio"]
                if row["avg_validation_loss_ratio"] is not None
                else row["avg_loss_ratio"]
                if row["avg_loss_ratio"] is not None
                else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit]
        struggling_templates = sorted(
            [row for row in active_template_rows if row["n_used"] >= 3],
            key=lambda row: (
                row["s1_rate"] or 0.0,
                row["avg_validation_loss_ratio"]
                if row["avg_validation_loss_ratio"] is not None
                else row["avg_loss_ratio"]
                if row["avg_loss_ratio"] is not None
                else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit]

        motif_rows = []
        for stat in motif_stats.values():
            losses = stat["losses"]
            reasons = stat["failure_reasons"]
            top_reason = (
                max(reasons.items(), key=lambda item: item[1])[0] if reasons else None
            )
            motif_rows.append(
                {
                    "name": stat["name"],
                    "n_used": stat["n_used"],
                    "s1_rate": stat["n_stage1"] / max(stat["n_used"], 1),
                    "avg_loss_ratio": sum(losses) / len(losses) if losses else None,
                    "top_failure_reason": top_reason,
                }
            )
        motif_rows = sorted(
            [row for row in motif_rows if row["n_used"] >= 2],
            key=lambda row: (-(row["n_used"] or 0), row["avg_loss_ratio"] or 999.0),
        )[:limit]

        slot_rows = []
        for stat in slot_stats.values():
            reasons = stat["failure_reasons"]
            selected = stat["selected_motifs"]
            top_reason = (
                max(reasons.items(), key=lambda item: item[1])[0] if reasons else None
            )
            top_motif = (
                max(selected.items(), key=lambda item: item[1])[0] if selected else None
            )
            slot_rows.append(
                {
                    "slot_key": stat["slot_key"],
                    "template_name": stat["template_name"],
                    "slot_index": stat["slot_index"],
                    "slot_classes": stat["slot_classes"],
                    "n_used": stat["n_used"],
                    "s1_rate": stat["n_stage1"] / max(stat["n_used"], 1),
                    "avg_loss_ratio": (
                        sum(stat["losses"]) / len(stat["losses"])
                        if stat["losses"]
                        else None
                    ),
                    "top_failure_reason": top_reason,
                    "top_selected_motif": top_motif,
                }
            )
        slot_rows = sorted(
            [
                row
                for row in slot_rows
                if row["n_used"] >= 2
                and row["template_name"] in active_template_names
                and row["slot_index"]
                < int(slot_counts.get(row["template_name"], 0) or 0)
            ],
            key=lambda row: (
                row["s1_rate"] if row["s1_rate"] is not None else 1.0,
                row["avg_loss_ratio"] if row["avg_loss_ratio"] is not None else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit]

        loss_distribution = {
            "training": {
                "median": self._percentile(loss_values, 0.5),
                "p25": self._percentile(loss_values, 0.25),
                "p75": self._percentile(loss_values, 0.75),
            },
            "validation": {
                "median": self._percentile(validation_losses, 0.5),
                "p25": self._percentile(validation_losses, 0.25),
                "p75": self._percentile(validation_losses, 0.75),
            },
            "discovery": {
                "median": self._percentile(discovery_losses, 0.5),
                "p25": self._percentile(discovery_losses, 0.25),
                "p75": self._percentile(discovery_losses, 0.75),
            },
        }

        recommendations: List[str] = []
        weak = next(
            (row for row in struggling_templates if (row["s1_rate"] or 0) < 0.15), None
        )
        if weak:
            recommendations.append(
                f"{weak['name']} is over-sampled relative to quality: S1 {(weak['s1_rate'] * 100):.1f}% over {weak['n_used']} runs. Reduce weight or harden motifs for {weak['top_failure_reason'] or 'unknown failures'}."
            )
        slot_heavy = next(
            (
                row
                for row in struggling_templates
                if (row.get("slot_count") or 0) >= 3
                and (row.get("s1_rate") or 0) < 0.25
            ),
            None,
        )
        if slot_heavy:
            recommendations.append(
                f"Slot-heavy template {slot_heavy['name']} underperforms with {slot_heavy['slot_count']} inferred motif slots. Tighten slot compatibility checks or narrow allowed motifs."
            )
        val_median = loss_distribution["validation"]["median"]
        train_median = loss_distribution["training"]["median"]
        if (
            val_median is not None
            and train_median is not None
            and val_median > train_median * 1.15
        ):
            recommendations.append(
                f"Validation loss ratio median ({val_median:.3f}) is materially worse than training median ({train_median:.3f}). Improve generalization gates or reduce brittle template/motif combinations."
            )
        best = top_templates[0] if top_templates else None
        if best:
            best_loss = (
                f"{best['avg_loss_ratio']:.3f}"
                if best["avg_loss_ratio"] is not None
                else "n/a"
            )
            recommendations.append(
                f"Exploit {best['name']} more aggressively: S1 {(best['s1_rate'] * 100):.1f}% with avg loss {best_loss}."
            )

        weak_slot = next(
            (row for row in slot_rows if (row["s1_rate"] or 0) < 0.15), None
        )
        if weak_slot:
            recommendations.append(
                f"Weak slot {weak_slot['slot_key']} is collapsing candidate quality: S1 {(weak_slot['s1_rate'] * 100):.1f}% with motif {weak_slot['top_selected_motif'] or 'none'} and failures dominated by {weak_slot['top_failure_reason'] or 'unknown'}."
            )

        routing_reprieve = next(
            (
                row
                for row in struggling_templates
                if (row.get("routing_fast_lane_runs") or 0) >= 3
                and (row.get("routing_fast_lane_positive_rate") or 0) >= 0.5
            ),
            None,
        )
        if routing_reprieve:
            recommendations.append(
                f"{routing_reprieve['name']} looks under-credited by short S1: fast lane positive on {(routing_reprieve['routing_fast_lane_positive_rate'] * 100):.1f}% of routing probes across {routing_reprieve['routing_fast_lane_runs']} runs. Treat it as a slow starter, not a dead template."
            )

        zero_slot_templates = sorted(
            [
                name
                for name, count in slot_counts.items()
                if count == 0 and name in active_template_names
            ]
        )[:10]

        sorted_buckets = sorted(
            experiment_buckets.values(),
            key=lambda item: float(item.get("timestamp") or 0.0),
        )[-20:]
        top_template_names = [row["name"] for row in top_templates[:3]]
        weak_slot_keys = [row["slot_key"] for row in slot_rows[:3]]

        template_trends = []
        for name in top_template_names:
            points = []
            for bucket in sorted_buckets:
                item = bucket["templates"].get(name)
                if not item or not item["n"]:
                    continue
                points.append(
                    {
                        "timestamp": bucket["timestamp"],
                        "experiment_id": bucket.get("experiment_id"),
                        "s1_rate": item["s1"] / max(item["n"], 1),
                        "avg_loss_ratio": (
                            sum(item["losses"]) / len(item["losses"])
                            if item["losses"]
                            else None
                        ),
                    }
                )
            if points:
                template_trends.append({"name": name, "points": points})

        slot_trends = []
        for key in weak_slot_keys:
            points = []
            for bucket in sorted_buckets:
                item = bucket["slots"].get(key)
                if not item or not item["n"]:
                    continue
                points.append(
                    {
                        "timestamp": bucket["timestamp"],
                        "experiment_id": bucket.get("experiment_id"),
                        "s1_rate": item["s1"] / max(item["n"], 1),
                        "avg_loss_ratio": (
                            sum(item["losses"]) / len(item["losses"])
                            if item["losses"]
                            else None
                        ),
                    }
                )
            if points:
                slot_trends.append({"slot_key": key, "points": points})

        loss_trends = []
        for bucket in sorted_buckets:
            loss_trends.append(
                {
                    "timestamp": bucket["timestamp"],
                    "experiment_id": bucket.get("experiment_id"),
                    "training_median": self._percentile(bucket["training_losses"], 0.5),
                    "validation_median": self._percentile(
                        bucket["validation_losses"], 0.5
                    ),
                    "discovery_median": self._percentile(
                        bucket["discovery_losses"], 0.5
                    ),
                }
            )

        return {
            "top_templates": top_templates,
            "struggling_templates": struggling_templates,
            "motif_slots": motif_rows,
            "slot_observability": slot_rows,
            "loss_distribution": loss_distribution,
            "template_trends": template_trends,
            "slot_trends": slot_trends,
            "loss_trends": loss_trends,
            "recommendations": recommendations[:4],
            "summary": {
                "avg_templates_per_graph": (
                    sum(templates_per_graph) / len(templates_per_graph)
                    if templates_per_graph
                    else 0.0
                ),
                "avg_motifs_per_graph": (
                    sum(motifs_per_graph) / len(motifs_per_graph)
                    if motifs_per_graph
                    else 0.0
                ),
                "templates_tracked": len(active_template_rows),
                "motifs_tracked": len(motif_rows),
                "zero_slot_templates": zero_slot_templates,
                "inactive_templates_tracked": len(inactive_template_rows),
                "inactive_template_names": sorted(
                    row["name"] for row in inactive_template_rows
                )[:10],
                "routing_fast_lane_templates": sum(
                    1
                    for row in active_template_rows
                    if (row.get("routing_fast_lane_runs") or 0) > 0
                ),
                "routing_fast_lane_runs": sum(
                    int(row.get("routing_fast_lane_runs") or 0)
                    for row in active_template_rows
                ),
                "routing_fast_lane_positive_templates": sum(
                    1
                    for row in active_template_rows
                    if (row.get("routing_fast_lane_positive_rate") or 0) >= 0.5
                    and (row.get("routing_fast_lane_runs") or 0) >= 3
                ),
            },
        }

    # ── Training Curves ──

    def store_training_curve(self, result_id: str, curve: List[Dict]) -> None:
        """Store per-step training data for survivors only.

        curve: list of dicts with keys step, loss, grad_norm, step_time_ms
        """
        if not curve or not result_id:
            return
        self.flush_writes()
        # Only store curves for results that passed S1 (survivors).
        # S1 failure learning signal is captured in loss_ratio, not per-step curves.
        row = self.conn.execute(
            "SELECT stage1_passed FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None or row[0] != 1:
            return
        self.conn.executemany(
            """INSERT OR REPLACE INTO training_curves
               (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    result_id,
                    d.get("step", i),
                    d.get("loss"),
                    d.get("grad_norm"),
                    d.get("step_time_ms"),
                )
                for i, d in enumerate(curve)
            ],
        )
        self._maybe_commit()

    def get_training_curve(self, result_id: str) -> List[Dict]:
        """Get per-step training data for a program."""
        rows = self.conn.execute(
            """SELECT step, loss, grad_norm, step_time_ms
               FROM training_curves WHERE result_id = ?
               ORDER BY step""",
            (result_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def strip_graph_json_for_failures(self, experiment_id: str) -> int:
        """Clear graph_json for S1 failures with no loss data.

        Called after update_op_success_rates() has already consumed the graphs.
        Sets to empty string (NOT NULL constraint on column).
        Returns the number of rows stripped.
        """
        cur = self.conn.execute(
            """UPDATE program_results SET graph_json = ''
               WHERE experiment_id = ?
                 AND stage0_passed = 1 AND stage1_passed = 0
                 AND loss_ratio IS NULL AND length(graph_json) > 0""",
            (experiment_id,),
        )
        n = cur.rowcount
        if n:
            self._maybe_commit()
        return n

    def merge_op_failure_counts(self, op_counts: Dict[str, Dict[str, int]]) -> None:
        """Merge S0 failure op counts into op_success_rates.

        Called after update_op_success_rates() to incorporate ops from programs
        that failed S0/S0.5 and were not stored in program_results.

        Args:
            op_counts: {op_name: {"n_used": int, "n_s0": int, "n_s05": int}}
        """
        if not op_counts:
            return
        now = time.time()
        for op_name, counts in op_counts.items():
            self.conn.execute(
                """INSERT INTO op_success_rates
                   (op_name, n_used, n_stage0_passed, n_stage05_passed,
                    n_stage1_passed, last_updated)
                   VALUES (?, ?, ?, ?, 0, ?)
                   ON CONFLICT(op_name) DO UPDATE SET
                    n_used = n_used + excluded.n_used,
                    n_stage0_passed = n_stage0_passed + excluded.n_stage0_passed,
                    n_stage05_passed = n_stage05_passed + excluded.n_stage05_passed,
                    last_updated = excluded.last_updated""",
                (
                    op_name,
                    counts.get("n_used", 0),
                    counts.get("n_s0", 0),
                    counts.get("n_s05", 0),
                    now,
                ),
            )
        self._maybe_commit()

    # ── Failure Signatures ──

    @staticmethod
    def _extract_op_bigrams(graph_json: str) -> List[str]:
        """Extract sorted op-pair bigrams from a graph JSON.

        A bigram is "opA->opB" for each edge in the graph.  Returns a
        sorted deduplicated list, giving a compact structural fingerprint
        of what-connects-to-what.
        """
        if not isinstance(graph_json, str) or not graph_json:
            return []
        return list(_cached_extract_op_bigrams(graph_json))

    def get_entries(
        self,
        experiment_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
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

    def set_external_benchmarks(self, result_id: str, payload: Any) -> bool:
        """Store external benchmark payload for a program result."""
        if not result_id:
            return False
        serialized = None
        try:
            if payload is None:
                serialized = None
            elif isinstance(payload, dict):
                # Merge partial benchmark updates (for example, scaling-only writes)
                # with any previously stored benchmark families (for example, long_context).
                existing = self.conn.execute(
                    "SELECT external_benchmarks_json FROM program_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                merged: Dict[str, Any] = {}
                if existing and existing["external_benchmarks_json"]:
                    try:
                        parsed = _json_loads(existing["external_benchmarks_json"])
                        if isinstance(parsed, dict):
                            merged.update(parsed)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                merged.update(payload)
                serialized = json.dumps(merged)
            else:
                serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return False
        cur = self.conn.execute(
            "UPDATE program_results SET external_benchmarks_json = ? WHERE result_id = ?",
            (serialized, result_id),
        )
        self._maybe_commit()
        return cur.rowcount > 0

    def get_failure_analysis(self, experiment_id: str) -> Dict:
        """Get failure analysis data for an experiment."""
        programs = self.get_program_results(experiment_id)
        total = len(programs)
        if total == 0:
            return {
                "total": 0,
                "funnel": {},
                "errors": {},
                "stage_deaths": {},
                "root_causes": {},
                "exemplars": [],
            }

        s0_pass = sum(1 for p in programs if p.get("stage0_passed"))
        s05_pass = sum(1 for p in programs if p.get("stage05_passed"))
        s1_pass = sum(1 for p in programs if p.get("stage1_passed"))

        # Error type distribution (use classified error_type if available)
        errors: Dict[str, int] = {}
        root_causes: Dict[str, int] = {}
        exemplars: List[Dict[str, Any]] = []
        for p in programs:
            err_type = p.get("error_type") or ""
            err_msg = p.get("error_message") or p.get("stage0_error") or ""
            key = err_type if err_type else err_msg[:80].strip()
            if key:
                errors[key] = errors.get(key, 0) + 1
            failure_details = {}
            raw_failure = p.get("failure_details_json")
            if raw_failure:
                try:
                    failure_details = (
                        _json_loads(raw_failure)
                        if isinstance(raw_failure, str)
                        else raw_failure
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    failure_details = {}
            root_cause = (
                failure_details.get("root_cause_code")
                or err_type
                or p.get("stage_at_death")
                or "unknown"
            )
            root_causes[root_cause] = root_causes.get(root_cause, 0) + 1
            if failure_details and len(exemplars) < 10:
                exemplars.append(
                    {
                        "result_id": p.get("result_id"),
                        "graph_fingerprint": p.get("graph_fingerprint"),
                        "stage": failure_details.get("stage")
                        or p.get("stage_at_death"),
                        "root_cause_code": root_cause,
                        "error_type": failure_details.get("error_type") or err_type,
                        "error_message": failure_details.get("error_message")
                        or err_msg,
                        "failure_op": failure_details.get("failure_op"),
                        "traceback_excerpt": failure_details.get("traceback_excerpt"),
                    }
                )

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
            "root_causes": dict(sorted(root_causes.items(), key=lambda x: -x[1])[:10]),
            "stage_deaths": stage_deaths,
            "exemplars": exemplars,
        }

    def get_dashboard_summary(self) -> Dict:
        """Get aggregate stats for the dashboard."""
        now = time.time()
        cached = getattr(self, "_dashboard_summary_cache", None)
        expires_at = float(
            getattr(self, "_dashboard_summary_cache_expires_at", 0.0) or 0.0
        )
        if cached is not None and now < expires_at:
            return dict(cached)

        exp_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_experiments,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_experiments
            FROM experiments
            """
        ).fetchone()
        program_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_programs_evaluated,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) AS stage1_survivors,
                AVG(novelty_score) AS avg_novelty_score,
                MAX(novelty_score) AS top_novelty_score,
                AVG(avg_step_time_ms) AS avg_step_time_ms,
                AVG(throughput_tok_s) AS avg_throughput_tok_s,
                AVG(routing_utilization_entropy) AS avg_routing_entropy,
                AVG(depth_savings_ratio) AS avg_depth_savings,
                AVG(recursion_savings_ratio) AS avg_recursion_savings,
                AVG(CASE WHEN routing_tokens_total > 0
                         THEN CAST(routing_tokens_processed AS REAL) / routing_tokens_total END) AS avg_routing_token_retention,
                AVG(sparsity_ratio) AS avg_sparsity_ratio,
                COUNT(DISTINCT graph_fingerprint) AS unique_fingerprints
            FROM program_results
            """
        ).fetchone()
        insight_row = self.conn.execute(
            "SELECT COUNT(*) AS active_insights FROM insights WHERE status = 'active'"
        ).fetchone()
        learning_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS learning_events,
                (
                    SELECT description
                    FROM learning_log ll2
                    ORDER BY ll2.timestamp DESC
                    LIMIT 1
                ) AS latest_learning
            FROM learning_log
            """
        ).fetchone()

        latest_perf_report = None
        latest_dedup = None
        latest_perf_row = self.conn.execute(
            """SELECT experiment_id, completed_at, results_json
               FROM experiments
               WHERE status = 'completed'
                 AND results_json IS NOT NULL
               ORDER BY completed_at DESC
               LIMIT 1"""
        ).fetchone()
        if latest_perf_row and latest_perf_row["results_json"]:
            try:
                rj = latest_perf_row["results_json"]
                latest_results = (
                    self._decompress(rj) if isinstance(rj, bytes) else _json_loads(rj)
                )
                perf_report = (
                    latest_results.get("perf_report")
                    if isinstance(latest_results, dict)
                    else None
                )
                if isinstance(perf_report, dict):
                    queue = perf_report.get("queue_telemetry") or {}
                    kernel_hotspots = perf_report.get("kernel_hotspots") or []
                    top_kernel = kernel_hotspots[0] if kernel_hotspots else None
                    latest_perf_report = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "completed_at": latest_perf_row["completed_at"],
                        "programs_profiled": int(
                            perf_report.get("programs_profiled", 0) or 0
                        ),
                        "avg_submit_wait_ms": float(
                            queue.get("submit_wait_avg_ms", 0.0) or 0.0
                        ),
                        "avg_scheduling_wait_ms": float(
                            queue.get("scheduling_wait_avg_ms", 0.0) or 0.0
                        ),
                        "gpu_starvation_events": int(
                            (perf_report.get("gpu_starvation") or {}).get(
                                "event_count", 0
                            )
                            or 0
                        ),
                        "top_kernel": top_kernel,
                    }
                # Extract dedup stats from latest experiment
                if isinstance(latest_results, dict) and "dedup_rate" in latest_results:
                    latest_dedup = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "dedup_rate": latest_results.get("dedup_rate", 0),
                        "skipped_dedup": latest_results.get("skipped_dedup", 0),
                        "novel_count": latest_results.get("dedup_novel_count", 0),
                        "known_fingerprints": latest_results.get(
                            "dedup_known_fingerprints", 0
                        ),
                    }
            except (TypeError, ValueError, json.JSONDecodeError):
                latest_perf_report = None

        total_programs = int(
            (program_row["total_programs_evaluated"] or 0) if program_row else 0
        )
        stage1_survivors = int(
            (program_row["stage1_survivors"] or 0) if program_row else 0
        )
        summary = {
            "total_experiments": int(
                (exp_row["total_experiments"] or 0) if exp_row else 0
            ),
            "completed_experiments": int(
                (exp_row["completed_experiments"] or 0) if exp_row else 0
            ),
            "total_programs_evaluated": total_programs,
            "stage1_survivors": stage1_survivors,
            "survival_rate": stage1_survivors / max(total_programs, 1),
            "avg_novelty_score": float(
                (program_row["avg_novelty_score"] or 0.0) if program_row else 0.0
            ),
            "top_novelty_score": float(
                (program_row["top_novelty_score"] or 0.0) if program_row else 0.0
            ),
            "active_insights": int(
                (insight_row["active_insights"] or 0) if insight_row else 0
            ),
            "learning_events": int(
                (learning_row["learning_events"] or 0) if learning_row else 0
            ),
            "latest_learning": (
                learning_row["latest_learning"] if learning_row else None
            ),
            "avg_step_time_ms": float(
                (program_row["avg_step_time_ms"] or 0.0) if program_row else 0.0
            ),
            "avg_throughput_tok_s": float(
                (program_row["avg_throughput_tok_s"] or 0.0) if program_row else 0.0
            ),
            "avg_routing_entropy": (
                float(program_row["avg_routing_entropy"])
                if program_row and program_row["avg_routing_entropy"] is not None
                else None
            ),
            "avg_depth_savings": (
                float(program_row["avg_depth_savings"])
                if program_row and program_row["avg_depth_savings"] is not None
                else None
            ),
            "avg_recursion_savings": (
                float(program_row["avg_recursion_savings"])
                if program_row and program_row["avg_recursion_savings"] is not None
                else None
            ),
            "avg_routing_token_retention": (
                float(program_row["avg_routing_token_retention"])
                if program_row
                and program_row["avg_routing_token_retention"] is not None
                else None
            ),
            "avg_sparsity_ratio": (
                float(program_row["avg_sparsity_ratio"])
                if program_row and program_row["avg_sparsity_ratio"] is not None
                else None
            ),
            "latest_perf_report": latest_perf_report,
            "unique_fingerprints": int(
                (program_row["unique_fingerprints"] or 0) if program_row else 0
            ),
            "latest_dedup": latest_dedup,
            "template_observability": self.get_template_slot_observability(),
        }
        self._dashboard_summary_cache = dict(summary)
        self._dashboard_summary_cache_expires_at = now + self._DASHBOARD_SUMMARY_TTL_S
        return summary

    # ── Leaderboard ──

    # Ops considered "routing" for the structural complexity bonus
    _ROUTING_OPS = frozenset(
        {
            "route_topk",
            "route_lanes",
            "route_recursion",
            "token_merge",
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
            "token_merging",
            "cascade",
            "speculative",
            "moe_topk",
            "adaptive_lane_mixer",
            "mixed_recursion_gate",
            "relu_gate_routing",
            "routing_conditioned_compression",
            "token_type_classifier",
            "entropy_score",
            "progressive_compression_gate",
            "compression_mixture_experts",
            "latent_attention_compressor",
        }
    )

    _SPARSE_OPS = frozenset(
        {
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "structured_sparse",
            "block_sparse",
            "semi_structured_2_4",
            "hash_trick",
            "sparse_topk",
            "latent_attention_compressor",
            "routing_conditioned_compression",
            "compression_mixture_experts",
            "progressive_compression_gate",
        }
    )

    _MOE_OPS = frozenset(
        {
            "moe_topk",
            "route_topk",
            "route_lanes",
            "adaptive_lane_mixer",
            "compression_mixture_experts",
            "entropy_score",
        }
    )
    _TIER_ORDER = {
        "screening": 0,
        "investigation": 1,
        "validation": 2,
        "breakthrough": 3,
    }

    def _graph_structural_counts(
        self, result_id: str, graph_json: Optional[str] = None
    ) -> Dict[str, Optional[int]]:
        """Return routing/sparse/MoE op counts with at most one graph lookup."""
        try:
            raw_graph_json = graph_json
            if raw_graph_json is None:
                row = self.conn.execute(
                    "SELECT graph_json FROM program_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                raw_graph_json = row[0] if row else None
            if not raw_graph_json:
                return {"routing": None, "sparse": None, "moe": None}
            graph_data = _json_loads(raw_graph_json)
            nodes = graph_data.get("nodes", [])
            if isinstance(nodes, dict):
                node_iter = nodes.values()
            elif isinstance(nodes, list):
                node_iter = nodes
            else:
                node_iter = []

            routing = 0
            sparse = 0
            moe = 0
            for node in node_iter:
                if not isinstance(node, dict):
                    continue
                op_name = node.get("op")
                if not op_name:
                    op_name = node.get("op_name")
                if op_name in self._ROUTING_OPS:
                    routing += 1
                if op_name in self._SPARSE_OPS:
                    sparse += 1
                if op_name in self._MOE_OPS:
                    moe += 1
            return {
                "routing": routing if routing > 0 else None,
                "sparse": sparse if sparse > 0 else None,
                "moe": moe if moe > 0 else None,
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {"routing": None, "sparse": None, "moe": None}

    def _count_routing_ops(self, result_id: str) -> Optional[int]:
        """Count routing/branching ops in the graph for a program result."""
        return self._graph_structural_counts(result_id).get("routing")

    def _count_sparse_ops(self, result_id: str) -> Optional[int]:
        """Count sparsity/compression ops in the graph for a program result."""
        return self._graph_structural_counts(result_id).get("sparse")

    def _count_moe_ops(self, result_id: str) -> Optional[int]:
        """Count MoE-specific ops in the graph for a program result."""
        return self._graph_structural_counts(result_id).get("moe")

    @staticmethod
    def _best_min(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        try:
            return float(min(vals))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _best_max(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        try:
            return float(max(vals))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _best_bool(rows: List[Dict[str, Any]], key: str) -> Optional[int]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return int(any(bool(v) for v in vals))

    def compute_efficiency_multiple(
        self,
        loss_ratio: Optional[float] = None,
        param_count: Optional[float] = None,
        flops_forward: Optional[float] = None,
        throughput_tok_s: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
        forward_time_ms: Optional[float] = None,
        is_moe: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Geometric mean of per-dimension ratios vs GPT-2.

        All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
        dimensions to return a result (graceful with missing data).

        For MoE models (is_moe=True), total param count is excluded since
        MoE activates only a fraction of params per token.
        Returns dict with per-dimension ratios and ``geomean``, or None.
        """
        ref = GPT2_REF
        ratios: Dict[str, float] = {}

        # x_quality: ref_loss / cand_loss (lower loss = better)
        if loss_ratio is not None and loss_ratio > 0:
            ratios["x_quality"] = ref["loss_ratio"] / loss_ratio

        # x_params: ref_params / cand_params (fewer = better)
        # MoE: skip — total params != active params
        if param_count is not None and param_count > 0 and not is_moe:
            ratios["x_params"] = ref["param_count"] / param_count

        # x_flops: ref_flops / cand_flops (fewer = better)
        if flops_forward is not None and flops_forward > 0:
            ratios["x_flops"] = ref["flops_forward"] / flops_forward

        # x_throughput: cand_tput / ref_tput (higher = better)
        if throughput_tok_s is not None and throughput_tok_s > 0:
            ratios["x_throughput"] = throughput_tok_s / ref["throughput_tok_s"]

        # x_memory: ref_mem / cand_mem (less = better)
        if peak_memory_mb is not None and peak_memory_mb > 0:
            ratios["x_memory"] = ref["peak_memory_mb"] / peak_memory_mb

        # x_latency: ref_lat / cand_lat (lower = better)
        if forward_time_ms is not None and forward_time_ms > 0:
            ratios["x_latency"] = ref["forward_time_ms"] / forward_time_ms

        if len(ratios) < 3:
            return None

        geomean = 1.0
        for v in ratios.values():
            geomean *= v
        geomean = geomean ** (1.0 / len(ratios))
        ratios["geomean"] = geomean
        ratios["n_dimensions"] = float(len(ratios) - 1)  # exclude geomean itself
        return ratios

    @staticmethod
    def compute_composite_score(**kwargs) -> float:
        """Delegate to v7 scoring (14-component, 565pt max).

        Translates legacy parameter names to v7 equivalents for backward
        compatibility with callers that still use ``wikitext_perplexity``.
        """
        from ..leaderboard_scoring import compute_composite

        # Translate legacy kwargs → current scoring parameter names
        if "wikitext_perplexity" in kwargs and "ppl_screening" not in kwargs:
            kwargs["ppl_screening"] = kwargs.pop("wikitext_perplexity")
        # wikitext_score has no direct scoring equivalent — drop silently
        kwargs.pop("wikitext_score", None)

        result = compute_composite(**kwargs)
        return result if isinstance(result, (int, float)) else float(result)

    @staticmethod
    def _reference_novelty_for_display(novelty: Optional[float]) -> Optional[float]:
        """Compress reference novelty values for dashboard display.

        Reference architectures are anchor points, so we intentionally present
        their novelty on a reduced scale to avoid implying they are frontier
        discoveries in the same sense as synthesized candidates.
        """
        if novelty is None:
            return None
        try:
            value = float(novelty)
        except (TypeError, ValueError):
            return None
        value = max(0.0, min(1.0, value))
        return min(0.35, value * 0.4)

    @staticmethod
    def _reference_family_fallback(reference_name: Optional[str]) -> str:
        """Infer a stable family label for named baselines when graph data is absent."""
        name = str(reference_name or "").strip().lower()
        if not name:
            return "Unknown"
        if "gpt" in name or "transformer" in name:
            return "Attention"
        if "mamba" in name or "ssm" in name:
            return "Mamba-SSM"
        if "rwkv" in name:
            return "Hybrid-Mixer"
        if "retrieval" in name or "rag" in name:
            return "Hybrid-Attention"
        return "Unknown"

    def pin_reference(self, entry_id: str, reference_name: str) -> None:
        """Pin a leaderboard entry as a reference architecture."""
        self.conn.execute(
            """UPDATE leaderboard
               SET is_reference = 1,
                   reference_name = ?,
                   model_source = 'reference'
               WHERE entry_id = ?""",
            (reference_name, entry_id),
        )
        self._maybe_commit()

    # ── Pre-investigation gate helpers ──────────────────────────────────

    def get_investigation_eligible(
        self,
        _max_lr: float,
        min_stability: float,
        min_spectral_norm: float,
        max_spectral_norm: float,
        min_improvement_rate: float,
        _ref_lr_ceiling: Optional[float] = None,
        min_composite_score: Optional[float] = None,
    ) -> List[Dict]:
        """Stage A hard reject: return screening candidates that pass health filters.

        Joins program_results with leaderboard to return full metric rows for
        candidates eligible for investigation.  Uses composite_score as the
        primary worthiness gate instead of loss_ratio — a model with excellent
        efficiency/novelty/stability deserves investigation even with moderate loss.
        """
        # Dynamic score floor: 25th percentile of investigation tier,
        # excluding reference architectures which inflate the floor.
        # Falls back to 75th percentile of screening tier when no
        # non-reference investigation entries exist yet.
        if min_composite_score is None:
            inv_scores = self.conn.execute(
                "SELECT composite_score FROM leaderboard"
                " WHERE tier IN ('investigation', 'validation')"
                " AND COALESCE(is_reference, 0) = 0"
                " AND composite_score IS NOT NULL"
                " ORDER BY composite_score ASC"
            ).fetchall()
            if inv_scores:
                min_composite_score = inv_scores[len(inv_scores) // 4][0]
            else:
                # No non-reference investigation entries: use 75th percentile
                # of screening tier as a reasonable promotion threshold
                scr_scores = self.conn.execute(
                    "SELECT composite_score FROM leaderboard"
                    " WHERE tier = 'screening'"
                    " AND composite_score IS NOT NULL"
                    " ORDER BY composite_score ASC"
                ).fetchall()
                min_composite_score = (
                    scr_scores[3 * len(scr_scores) // 4][0] if scr_scores else 0.0
                )
        rows = self.conn.execute(
            """SELECT pr.*, l.entry_id, l.tier, l.composite_score,
                      l.screening_loss_ratio, l.screening_novelty,
                      l.pre_inv_score, l.is_reference, l.reference_name
               FROM program_results pr
               JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE l.tier = 'screening'
                 AND COALESCE(l.is_reference, 0) = 0
                 AND pr.stage1_passed = 1
                 AND COALESCE(pr.has_nan_grad, 0) = 0
                 AND COALESCE(pr.has_nan_output, 0) = 0
                 AND COALESCE(pr.has_inf_output, 0) = 0
                 AND COALESCE(pr.has_zero_grad, 0) = 0
                 AND COALESCE(pr.graph_has_gradient_path, 1) = 1
                 AND COALESCE(pr.stability_score, 0) >= ?
                 AND (pr.fp_jacobian_spectral_norm IS NULL
                      OR (pr.fp_jacobian_spectral_norm >= ? AND pr.fp_jacobian_spectral_norm <= ?))
                 AND COALESCE(pr.loss_improvement_rate, 0) >= ?
                 AND COALESCE(l.composite_score, 0) >= ?
               ORDER BY l.composite_score DESC""",
            (
                min_stability,
                min_spectral_norm,
                max_spectral_norm,
                min_improvement_rate,
                min_composite_score,
            ),
        ).fetchall()
        return [dict(r) for r in rows]

    def compute_pre_investigation_score(
        row: Dict, best_ref_lr: Optional[float] = None
    ) -> float:
        """Stage B composite readiness score (0-100 scale).

        Components:
        - Performance (40pts): loss_ratio, discovery_loss_ratio, loss_improvement_rate
        - Stability (20pts): stability_score, spectral_norm (Gaussian around 1.0), grad_norm_std
        - Novelty (20pts): novelty_score * confidence, structural_novelty, behavioral_novelty
        - Fingerprint quality (10pts): fp_intrinsic_dim, fp_isotropy, fp_rank_ratio
        - Efficiency (10pts): throughput_tok_s, peak_memory_mb
        - Reference penalty (-20pts): if loss_ratio > 1.5 * best_reference_lr
        """
        import math

        score = 0.0

        # ── Performance (40 pts) ──
        lr = row.get("loss_ratio")
        if lr is not None and lr > 0:
            # Lower LR is better; LR=0.1 → 40pts, LR=0.8 → ~8pts
            score += max(0, min(40, 40 * (1.0 - float(lr))))

        dlr = row.get("discovery_loss_ratio")
        if dlr is not None and dlr > 0:
            # Bonus: up to 5pts from discovery loss (replaces top of performance)
            score += max(0, min(5, 5 * (1.0 - float(dlr))))

        lir = row.get("loss_improvement_rate")
        if lir is not None and float(lir) > 0:
            # Up to 5pts for improvement rate
            score += min(5, float(lir) * 10)

        # Cap performance at 40
        score = min(40, score)

        # ── Stability (20 pts) ──
        stab = row.get("stability_score")
        if stab is not None:
            score += min(10, float(stab) * 10)

        sn = row.get("fp_jacobian_spectral_norm")
        if sn is not None and float(sn) > 0:
            # Gaussian centered on 1.0: score = 6 * exp(-(log(sn))^2 / 2)
            log_sn = math.log(float(sn))
            score += max(0, min(6, 6 * math.exp(-log_sn * log_sn / 2.0)))

        gns = row.get("grad_norm_std")
        if gns is not None:
            # Lower grad_norm_std is better; up to 4pts
            score += max(0, min(4, 4 * max(0, 1.0 - float(gns))))

        # ── Novelty (20 pts) ──
        ns = row.get("novelty_score")
        nc = row.get("novelty_confidence")
        if ns is not None:
            conf = float(nc) if nc is not None else 0.5
            score += min(10, float(ns) * conf * 10)

        sn_nov = row.get("structural_novelty")
        if sn_nov is not None:
            score += min(5, float(sn_nov) * 5)

        bn = row.get("behavioral_novelty")
        if bn is not None:
            score += min(5, float(bn) * 5)

        # ── Fingerprint quality (10 pts) ──
        fid = row.get("fp_intrinsic_dim")
        if fid is not None and float(fid) > 0:
            # Higher intrinsic dim → better; up to 4pts, cap at dim=20
            score += min(4, float(fid) / 5.0)

        fiso = row.get("fp_isotropy")
        if fiso is not None:
            score += min(3, float(fiso) * 3)

        frr = row.get("fp_rank_ratio")
        if frr is not None:
            score += min(3, float(frr) * 3)

        # ── Efficiency (10 pts) ──
        tp = row.get("throughput_tok_s")
        if tp is not None and float(tp) > 0:
            # Higher throughput → better; up to 5pts, 10k tok/s → 5pts
            score += min(5, float(tp) / 2000.0)

        mem = row.get("peak_memory_mb")
        if mem is not None and float(mem) > 0:
            # Lower memory → better; up to 5pts, 100MB → 5pts, 500MB → 1pt
            score += max(0, min(5, 5 * (1.0 - float(mem) / 600.0)))

        # ── Reference penalty (-20 pts) ──
        if best_ref_lr is not None and lr is not None:
            if float(lr) > 1.5 * float(best_ref_lr):
                score -= 20

        return max(0, min(100, round(score, 2)))

    def get_references(self) -> List[Dict]:
        """Get all pinned reference architectures."""
        rows = self.conn.execute(
            """SELECT l.*, pr.graph_json AS _graph_json,
                      pr.routing_mode AS _routing_mode,
                      pr.graph_fingerprint AS _graph_fingerprint
               FROM leaderboard l
               LEFT JOIN program_results pr ON pr.result_id = l.result_id
               WHERE COALESCE(l.is_reference, 0) = 1
               ORDER BY l.composite_score DESC NULLS LAST, l.reference_name ASC, l.timestamp DESC"""
        ).fetchall()
        refs: List[Dict] = []
        for row in rows:
            entry = dict(row)
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
            family = self._classify_architecture_family(
                graph_json=entry.pop("_graph_json", None),
                routing_mode=entry.pop("_routing_mode", None),
            )
            if family == "Unknown":
                family = self._reference_family_fallback(entry.get("reference_name"))
            entry["architecture_family"] = family
            entry["screening_novelty"] = self._reference_novelty_for_display(
                entry.get("screening_novelty")
            )
            refs.append(entry)
        return refs

    @staticmethod
    def _classify_architecture_family(
        graph_json: Optional[str],
        routing_mode: Optional[str],
    ) -> str:
        """Map graph structure to a compact architecture family label."""
        if routing_mode:
            return "Routed-MoE"
        if not graph_json:
            return "Unknown"

        try:
            graph = _json_loads(graph_json)
            nodes = graph.get("nodes")
            if isinstance(nodes, dict):
                node_iter = [n for n in nodes.values() if isinstance(n, dict)]
            elif isinstance(nodes, list):
                node_iter = [n for n in nodes if isinstance(n, dict)]
            else:
                node_iter = []
            ops = {str(n.get("op_name", "")).strip() for n in node_iter}
        except (json.JSONDecodeError, TypeError, ValueError):
            return "Unknown"

        if not ops:
            return "Unknown"

        attention_ops = {
            "attention",
            "self_attention",
            "mha",
            "multihead_attention",
            "qkv_attention",
            "softmax_attention",
            "linear_attention",
        }
        conv_ops = {"conv1d", "conv1d_seq", "depthwise_conv1d", "conv_only"}
        spectral_ops = {
            "sin",
            "cos",
            "fft",
            "ifft",
            "fourier_mix",
            "fourier_mixing",
            "rfft_seq",
            "irfft_seq",
        }
        gating_ops = {
            "sigmoid",
            "tanh",
            "silu",
            "gelu",
            "maximum",
            "minimum",
            "swiglu",
            "topk_gate",
            "moe_topk",
        }
        mlp_ops = {
            "linear_proj",
            "linear_proj_up",
            "linear_proj_down",
            "learnable_bias",
            "swiglu_mlp",
        }
        ssm_ops = {"state_space", "selective_scan"}
        adaptive_ops = {
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
            "fixed_point_iter",
        }

        has_attention = bool(ops & attention_ops)
        has_conv = bool(ops & conv_ops)
        has_spectral = bool(ops & spectral_ops)
        has_gating = bool(ops & gating_ops)
        has_mlp = bool(ops & mlp_ops)
        has_ssm = bool(ops & ssm_ops)
        has_adaptive = bool(ops & adaptive_ops) or routing_mode in (
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
        )

        family = "Hybrid-Mixer"
        if has_ssm:
            family = "Mamba-SSM" if not has_attention else "Hybrid-SSM"
        elif has_attention:
            if has_conv or has_spectral or has_gating:
                family = "Hybrid-Attention"
            else:
                family = "Attention"
        elif has_conv and has_spectral:
            family = "Spectral-Conv"
        elif has_spectral:
            family = "Spectral-Mixer"
        elif has_conv:
            family = "Conv-Mixer"
        elif has_gating and has_mlp:
            family = "Gated-MLP"
        elif has_mlp:
            family = "MLP-Mixer"
        elif has_gating:
            family = "Nonlinear-Mixer"

        # Apply modifiers
        if routing_mode == "moe_topk" or "moe_topk" in ops:
            family = f"MoE-{family}"
        if has_adaptive:
            family = f"Adaptive-{family}"

        return family

    def get_unresolved_hypotheses(
        self, campaign_id: Optional[str] = None
    ) -> List[Dict]:
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

    def record_decision(
        self,
        campaign_id: Optional[str],
        decision_type: str,
        subject: str,
        rationale: str,
        evidence_ids: Optional[List[str]] = None,
        alternatives: Optional[List[Dict]] = None,
        evidence_pack: Optional[Dict] = None,
    ) -> str:
        """Record a go/no-go or other decision. Returns decision_id."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO decisions
            (decision_id, campaign_id, timestamp, decision_type,
             subject, rationale, evidence_ids, alternatives_considered,
             evidence_pack_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                campaign_id,
                now,
                decision_type,
                subject,
                rationale,
                json.dumps(evidence_ids) if evidence_ids else None,
                json.dumps(alternatives) if alternatives else None,
                json.dumps(evidence_pack) if evidence_pack else None,
            ),
        )
        self._maybe_commit()
        return decision_id

    def get_decisions(
        self, campaign_id: Optional[str] = None, decision_type: Optional[str] = None
    ) -> List[Dict]:
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
                        d[f] = _json_loads(d[f])
                    except (json.JSONDecodeError, TypeError):
                        pass
            if d.get("evidence_pack_json"):
                try:
                    d["evidence_pack"] = _json_loads(d["evidence_pack_json"])
                except (json.JSONDecodeError, TypeError):
                    d["evidence_pack"] = None
            results.append(d)
        return results

    # ── Selection Decisions / Family Bandit Stats ──

    def record_selection_decision(
        self,
        context: str,
        candidate_pool_summary: Dict[str, Any],
        score_breakdown: List[Dict[str, Any]],
        policy: Dict[str, Any],
        reason: str,
        chosen_experiments: List[Dict[str, Any]],
        experiment_id: Optional[str] = None,
        trigger: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record one evidence-based experiment-selection decision."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_decisions
            (decision_id, timestamp, context, experiment_id,
             candidate_pool_summary_json, score_breakdown_json,
             policy_json, reason, chosen_experiments_json, trigger_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                now,
                context,
                experiment_id,
                json.dumps(candidate_pool_summary or {}),
                json.dumps(score_breakdown or []),
                json.dumps(policy or {}),
                reason or "",
                json.dumps(chosen_experiments or []),
                json.dumps(trigger or {}),
            ),
        )
        self._maybe_commit()
        return decision_id

    def get_selection_decisions(
        self,
        context: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return selection decisions newest first."""
        query = "SELECT * FROM selection_decisions WHERE 1=1"
        params: List[Any] = []
        if context:
            query += " AND context = ?"
            params.append(context)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in (
                "candidate_pool_summary_json",
                "score_breakdown_json",
                "policy_json",
                "chosen_experiments_json",
                "trigger_json",
            ):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = _json_loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out

    def get_selection_family_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return family bandit stats keyed by family name."""
        rows = self.conn.execute("SELECT * FROM selection_family_stats").fetchall()
        return {r["family"]: dict(r) for r in rows}

    def update_selection_family_stats(self, family: str, reward: float) -> None:
        """Update per-family running reward estimate for UCB/uncertainty."""
        family_name = (family or "Unknown").strip() or "Unknown"
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_family_stats
            (family, n_trials, cumulative_reward, mean_reward, last_reward, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(family) DO UPDATE SET
                n_trials = n_trials + 1,
                cumulative_reward = cumulative_reward + excluded.last_reward,
                mean_reward = (cumulative_reward + excluded.last_reward) * 1.0
                              / (n_trials + 1),
                last_reward = excluded.last_reward,
                last_updated = excluded.last_updated
            """,
            (family_name, float(reward), float(reward), float(reward), now),
        )
        self._maybe_commit()

    # ── Novelty Calibration ──

    def record_novelty_calibration(
        self,
        reference_version: str,
        n_runs: int,
        distribution: Dict[str, Any],
        noise_floor_mean: Optional[float] = None,
        noise_floor_std: Optional[float] = None,
        confidence_low: Optional[float] = None,
        confidence_high: Optional[float] = None,
        cka_source: Optional[str] = None,
        cka_artifact_version: Optional[str] = None,
        probe_protocol_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist novelty baseline calibration stats."""
        calibration_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO novelty_calibration
            (calibration_id, timestamp, reference_version, cka_source,
             cka_artifact_version, probe_protocol_hash, n_runs,
             noise_floor_mean, noise_floor_std, confidence_low, confidence_high,
             distribution_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id,
                now,
                reference_version,
                cka_source,
                cka_artifact_version,
                probe_protocol_hash,
                int(max(1, n_runs)),
                noise_floor_mean,
                noise_floor_std,
                confidence_low,
                confidence_high,
                json.dumps(distribution or {}),
                json.dumps(metadata or {}),
            ),
        )
        self._maybe_commit()
        return calibration_id

    def get_latest_novelty_calibration(
        self,
        reference_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest novelty calibration row, optionally by reference version."""
        query = "SELECT * FROM novelty_calibration WHERE 1=1"
        params: List[Any] = []
        if reference_version:
            query += " AND reference_version = ?"
            params.append(reference_version)
        query += " ORDER BY timestamp DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        out = dict(row)
        for key in ("distribution_json", "metadata_json"):
            raw = out.get(key)
            if raw:
                try:
                    out[key] = _json_loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out
