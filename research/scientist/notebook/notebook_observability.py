"""Mixin for LabNotebook — split from notebook_misc."""

from __future__ import annotations

import json
import math
import statistics
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from ._notebook_misc_shared import (
    _cached_extract_op_bigrams,
    _cached_extract_observability_metadata,
    _ObservabilityAccumulator,
    _classify_template_structural,
    _capability_signal_count,
    _reference_metric_baselines,
    _reference_beating_metrics,
    _template_label_from_evidence,
    _summarize_template_stat,
    _empty_template_stat,
    _load_eval_native_module,
    _TEMPLATE_DEF_RE,
    _EMPTY_DATA_ACCOUNTING_SHAPE,
)
from ..json_utils import fast_loads as _json_loads
from ..leaderboard_scoring import (
    compute_efficiency_multiple as _compute_efficiency_multiple,
    compute_pre_investigation_score as _compute_pre_investigation_score,
)
from ...synthesis.templates import TEMPLATES



class _ObservabilityMixin:
    """Template observability + slot statistics."""

    __slots__ = ()
    _DASHBOARD_SUMMARY_TTL_S = 2.0
    _TEMPLATE_OBSERVABILITY_TTL_S = 10.0

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
    @lru_cache(maxsize=1)
    def _infer_template_slot_counts() -> Dict[str, int]:
        counts: Dict[str, int] = {}
        structural_overrides = {
            # 1 motif slot (norm_wrap) + 3 semantic router slots tracked explicitly
            "hybrid_sparse_triplet_router": 4,
            # 1 motif slot (norm_wrap) + 8 structural routing slots
            "multiscale_difficulty_router": 9,
            # 1 motif slot (norm_wrap) + 6 structural routing slots
            "multiscale_rich_lane_router": 7,
            # 1 motif slot (norm_wrap) + 10 structural stem/lane/merge slots
            "intelligent_multilane_router": 11,
            # 1 norm slot + role slots for trunk/controller/write/read/merge/mix/stabilize
            "typed_slot_memory_block": 7,
            # 1 norm slot + trunk/controller/retrieval/merge/mix/stabilize
            "sparse_relation_graph_block": 6,
            # 1 norm slot + trunk/controller/write/read/merge/mix/stabilize
            "token_program_interpreter_block": 7,
            # codex capability-first templates emit explicit structural slots only
            "codex_ssm_retention_block": 4,
            "codex_ssm_delta_memory_block": 2,
        }
        template_dir = Path(__file__).resolve().parents[2] / "synthesis"
        template_files = sorted(template_dir.glob("_templates*.py"))
        for template_file in template_files:
            try:
                source = template_file.read_text(encoding="utf-8")
            except OSError:
                continue
            matches = list(_TEMPLATE_DEF_RE.finditer(source))
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
        counts.update(structural_overrides)
        return counts

    def get_template_slot_observability(self, limit: int = 8) -> Dict[str, Any]:
        now = time.time()
        cached = self._template_observability_cache.get(limit)
        if cached is not None and now < float(
            self._template_observability_cache_expires_at or 0.0
        ):
            return dict(cached)

        self.flush_writes()
        self._ensure_graph_features()
        rows = self.conn.execute(
            """
            SELECT
                pr.experiment_id,
                pr.timestamp,
                pr.graph_fingerprint,
                gf.templates_json,
                gf.motifs_json,
                gf.slot_usage_json,
                pr.stage0_passed,
                pr.stage05_passed,
                pr.stage1_passed,
                pr.loss_ratio,
                pr.discovery_loss_ratio,
                pr.validation_loss_ratio,
                pr.novelty_score,
                pr.novelty_confidence,
                pr.error_type,
                pr.stage_at_death,
                pr.failure_details_json,
                pr.induction_auc,
                pr.binding_auc,
                pr.binding_auc_curriculum,
                pr.ar_auc,
                pr.hellaswag_acc,
                pr.screening_hellaswag_correct,
                pr.screening_hellaswag_total,
                pr.screening_wikitext_status,
                pr.routing_fast_lane_applied,
                pr.routing_fast_lane_status,
                pr.routing_fast_lane_score,
                pr.routing_fast_lane_ppl_improvement,
                pr.routing_fast_lane_slope,
                pr.routing_fast_lane_slope_consistent
            FROM program_results pr
            JOIN program_graph_features gf ON gf.result_id = pr.result_id
            """
        ).fetchall()
        slot_counts = self._infer_template_slot_counts()
        if not rows:
            result = self._assemble_observability_result(
                _ObservabilityAccumulator(
                    template_stats={},
                    motif_stats={},
                    slot_stats={},
                    experiment_buckets={},
                    loss_values=[],
                    validation_losses=[],
                    discovery_losses=[],
                    motifs_per_graph=[],
                    templates_per_graph=[],
                ),
                slot_counts,
                limit,
            )
            self._template_observability_cache[limit] = dict(result)
            self._template_observability_cache_expires_at = (
                now + self._TEMPLATE_OBSERVABILITY_TTL_S
            )
            return result

        acc = self._accumulate_observability_stats(rows, slot_counts)
        result = self._assemble_observability_result(acc, slot_counts, limit)
        self._template_observability_cache[limit] = dict(result)
        self._template_observability_cache_expires_at = (
            now + self._TEMPLATE_OBSERVABILITY_TTL_S
        )
        return result

    def _accumulate_observability_stats(
        self, rows: list, slot_counts: Dict[str, int]
    ) -> _ObservabilityAccumulator:
        """Parse rows and accumulate per-template/motif/slot statistics."""
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
            if row["templates_json"] is not None:
                try:
                    templates = tuple(
                        str(item)
                        for item in (json.loads(row["templates_json"]) or [])
                        if item is not None
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    templates = ()
                try:
                    motifs = tuple(
                        str(item)
                        for item in (json.loads(row["motifs_json"]) or [])
                        if item is not None
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    motifs = ()
                try:
                    loaded_slots = json.loads(row["slot_usage_json"]) or []
                    slot_usage = tuple(
                        item for item in loaded_slots if isinstance(item, dict)
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    slot_usage = ()
            else:
                templates, motifs, slot_usage = _cached_extract_observability_metadata(
                    str(row.get("graph_json") or "")
                )
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

            motifs_per_graph.append(float(len(motifs)))
            templates_per_graph.append(float(len(templates)))

            loss_ratio = row["loss_ratio"]
            validation_lr = row["validation_loss_ratio"]
            discovery_lr = row["discovery_loss_ratio"]
            novelty = row["novelty_score"]
            novelty_confidence = row["novelty_confidence"]
            induction_auc = row["induction_auc"]
            binding_auc = (
                row["binding_auc_curriculum"]
                if row["binding_auc_curriculum"] is not None
                else row["binding_auc"]
            )
            ar_auc = row["ar_auc"]
            hellaswag_acc = row["hellaswag_acc"]
            screening_hs_correct = row["screening_hellaswag_correct"]
            screening_hs_total = row["screening_hellaswag_total"]
            screening_wikitext_status = row["screening_wikitext_status"]
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
                    _empty_template_stat(
                        name=str(template),
                        slot_count=slot_counts.get(str(template), 0),
                    ),
                )
                stat["n_used"] += 1
                stat["n_stage0"] += 1 if row["stage0_passed"] else 0
                stat["n_stage05"] += 1 if row["stage05_passed"] else 0
                stat["n_stage1"] += 1 if row["stage1_passed"] else 0
                fingerprint = str(row["graph_fingerprint"] or "").strip()
                if fingerprint:
                    stat["fingerprints"].add(fingerprint)
                    if row["stage1_passed"]:
                        stat["stage1_fingerprints"].add(fingerprint)
                if loss_ratio is not None and math.isfinite(loss_ratio):
                    stat["losses"].append(float(loss_ratio))
                    if row["stage1_passed"]:
                        stat["stage1_losses"].append(float(loss_ratio))
                if validation_lr is not None and math.isfinite(validation_lr):
                    stat["validation_losses"].append(float(validation_lr))
                if discovery_lr is not None and math.isfinite(discovery_lr):
                    stat["discovery_losses"].append(float(discovery_lr))
                if novelty is not None and math.isfinite(novelty):
                    stat["novelties"].append(float(novelty))
                if novelty_confidence is not None and math.isfinite(novelty_confidence):
                    stat["novelty_confidences"].append(float(novelty_confidence))
                if induction_auc is not None and math.isfinite(induction_auc):
                    stat["induction_aucs"].append(float(induction_auc))
                if binding_auc is not None and math.isfinite(binding_auc):
                    stat["binding_aucs"].append(float(binding_auc))
                if ar_auc is not None and math.isfinite(ar_auc):
                    stat["ar_aucs"].append(float(ar_auc))
                if hellaswag_acc is not None and math.isfinite(hellaswag_acc):
                    stat["hellaswag_accs"].append(float(hellaswag_acc))
                if (
                    screening_hs_correct is not None
                    and screening_hs_total is not None
                    and screening_hs_total
                ):
                    stat["screening_hellaswag_accs"].append(
                        float(screening_hs_correct)
                        / max(float(screening_hs_total), 1.0)
                    )
                if screening_wikitext_status is not None:
                    stat["screening_wikitext_runs"] += 1
                    if str(screening_wikitext_status) == "ok":
                        stat["screening_wikitext_ok"] += 1
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

        return _ObservabilityAccumulator(
            template_stats=template_stats,
            motif_stats=motif_stats,
            slot_stats=slot_stats,
            experiment_buckets=experiment_buckets,
            loss_values=loss_values,
            validation_losses=validation_losses,
            discovery_losses=discovery_losses,
            motifs_per_graph=motifs_per_graph,
            templates_per_graph=templates_per_graph,
        )

    def _assemble_observability_result(
        self,
        acc: _ObservabilityAccumulator,
        slot_counts: Dict[str, int],
        limit: int,
    ) -> Dict[str, Any]:
        """Sort, rank, and assemble the final observability result dict."""
        active_template_names = frozenset(TEMPLATES)
        template_stats = dict(acc.template_stats)
        for name in active_template_names:
            template_stats.setdefault(
                str(name),
                _empty_template_stat(
                    name=str(name),
                    slot_count=slot_counts.get(str(name), 0),
                ),
            )
        template_rows = [
            _summarize_template_stat(stat) for stat in template_stats.values()
        ]
        reference_baselines = _reference_metric_baselines(template_rows)
        for row in template_rows:
            row["capability_signal_count"] = _capability_signal_count(row)
            row["reference_beating_metrics"] = _reference_beating_metrics(
                row, reference_baselines
            )
            row["structural_category"] = _template_label_from_evidence(
                row, reference_baselines
            )
        active_template_rows = [
            row for row in template_rows if row["name"] in active_template_names
        ]
        inactive_template_rows = [
            row
            for row in template_rows
            if row["name"] not in active_template_names and row["n_used"] > 0
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
        all_templates = sorted(
            active_template_rows,
            key=lambda row: (
                {
                    "insufficient": 0,
                    "sparse": 1,
                    "building": 2,
                    "established": 3,
                }.get(str(row.get("evidence_level") or ""), 0),
                row["s1_rate"] if row["s1_rate"] is not None else -1.0,
                -(row["n_used"] or 0),
                row["name"],
            ),
        )
        inactive_templates = sorted(
            inactive_template_rows,
            key=lambda row: (-(row["n_used"] or 0), row["name"]),
        )
        low_loss_template_families = sorted(
            [
                row
                for row in active_template_rows
                if row.get("repeated_low_loss_family")
            ],
            key=lambda row: (
                -(row.get("repeated_low_loss_count") or 0),
                row["best_loss_ratio"] if row["best_loss_ratio"] is not None else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit]

        motif_rows = []
        for stat in acc.motif_stats.values():
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
        for stat in acc.slot_stats.values():
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
        all_slot_rows = sorted(
            [
                row
                for row in slot_rows
                if row["template_name"] in active_template_names
                and (
                    int(slot_counts.get(row["template_name"], 0) or 0) <= 0
                    or row["slot_index"]
                    < int(slot_counts.get(row["template_name"], 0) or 0)
                )
            ],
            key=lambda row: (
                row["template_name"],
                row["slot_index"],
                row["slot_key"],
            ),
        )
        slot_rows = sorted(
            [row for row in all_slot_rows if row["n_used"] >= 2],
            key=lambda row: (
                row["s1_rate"] if row["s1_rate"] is not None else 1.0,
                row["avg_loss_ratio"] if row["avg_loss_ratio"] is not None else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit]

        loss_distribution = {
            "training": {
                "median": self._percentile(acc.loss_values, 0.5),
                "p25": self._percentile(acc.loss_values, 0.25),
                "p75": self._percentile(acc.loss_values, 0.75),
            },
            "validation": {
                "median": self._percentile(acc.validation_losses, 0.5),
                "p25": self._percentile(acc.validation_losses, 0.25),
                "p75": self._percentile(acc.validation_losses, 0.75),
            },
            "discovery": {
                "median": self._percentile(acc.discovery_losses, 0.5),
                "p25": self._percentile(acc.discovery_losses, 0.25),
                "p75": self._percentile(acc.discovery_losses, 0.75),
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
        sparse_attention = next(
            (
                row
                for row in all_templates
                if row["name"].startswith("attn_")
                and str(row.get("evidence_level")) in {"insufficient", "sparse"}
            ),
            None,
        )
        if sparse_attention:
            recommendations.append(
                f"{sparse_attention['name']} is still data-sparse. Continue randomized weighting/backfills before trusting its rank or slot guidance."
            )
        induction_gap = next(
            (
                row
                for row in active_template_rows
                if (row.get("avg_induction_auc") or 0.0) < 0.02
                and (row.get("n_used") or 0) >= 5
                and (row.get("s1_rate") or 0.0) >= 0.2
            ),
            None,
        )
        if induction_gap:
            recommendations.append(
                f"{induction_gap['name']} survives screening but still shows weak induction signal. Keep it in data-building mode, not champion mode."
            )
        repeated_low_loss = next(
            (
                row
                for row in low_loss_template_families
                if (row.get("repeated_low_loss_count") or 0) >= 3
            ),
            None,
        )
        if repeated_low_loss:
            recommendations.append(
                f"{repeated_low_loss['name']} is a repeated low-loss family: {repeated_low_loss['repeated_low_loss_count']} S1 survivors at loss_ratio <= 0.45. Track it separately from benchmark-champion templates."
            )

        zero_slot_templates = sorted(
            [
                name
                for name, count in slot_counts.items()
                if count == 0 and name in active_template_names
            ]
        )[:10]

        sorted_buckets = sorted(
            acc.experiment_buckets.values(),
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
            "all_templates": all_templates,
            "low_loss_template_families": low_loss_template_families,
            "inactive_templates": inactive_templates,
            "all_slots": all_slot_rows,
            "motif_slots": motif_rows,
            "slot_observability": slot_rows,
            "loss_distribution": loss_distribution,
            "template_trends": template_trends,
            "slot_trends": slot_trends,
            "loss_trends": loss_trends,
            "recommendations": recommendations[:6],
            "summary": {
                "avg_templates_per_graph": (
                    sum(acc.templates_per_graph) / len(acc.templates_per_graph)
                    if acc.templates_per_graph
                    else 0.0
                ),
                "avg_motifs_per_graph": (
                    sum(acc.motifs_per_graph) / len(acc.motifs_per_graph)
                    if acc.motifs_per_graph
                    else 0.0
                ),
                "templates_tracked": len(active_template_rows),
                "templates_observed_total": len(template_rows),
                "motifs_tracked": len(motif_rows),
                "zero_slot_templates": zero_slot_templates,
                "inactive_templates_tracked": len(inactive_template_rows),
                "inactive_template_names": sorted(
                    row["name"] for row in inactive_template_rows
                )[:10],
                "insufficient_templates": sum(
                    1
                    for row in active_template_rows
                    if str(row.get("evidence_level")) == "insufficient"
                ),
                "sparse_templates": sum(
                    1
                    for row in active_template_rows
                    if str(row.get("evidence_level")) == "sparse"
                ),
                "established_templates": sum(
                    1
                    for row in active_template_rows
                    if str(row.get("evidence_level")) == "established"
                ),
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
                "repeated_low_loss_templates": [
                    row["name"] for row in low_loss_template_families
                ],
            },
        }

    # ── Training Curves ──

