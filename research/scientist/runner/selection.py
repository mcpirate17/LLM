"""
Experiment Runner

The autonomous experiment execution engine. Aria uses this to:
1. Generate batches of synthesized programs
2. Evaluate them through the funnel
3. Record results in the lab notebook
4. Analyze patterns and formulate new hypotheses
5. Adjust strategy based on outcomes

Supports background execution controlled from the dashboard.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..notebook import LabNotebook
from ..evidence import build_evidence_pack

import logging

logger = logging.getLogger(__name__)

from ._helpers import clear_gpu_memory, normalized_loss_ratio
from ._types import RunConfig


class _SelectionMixin:
    """Candidate scoring, selection, novelty calibration."""

    def _score_candidate_pool(
        self,
        candidates: List[Dict[str, Any]],
        config: RunConfig,
        nb: LabNotebook,
        context: str,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-objective scoring + family uncertainty policy for candidate selection."""
        if not candidates:
            return {
                "summary": {"candidate_count": 0},
                "scored": [],
                "selected": [],
                "reason": "No candidates available.",
                "policy": {"name": config.selection_policy, "exploration": False},
            }

        self._resolve_pending_selection_insight_trials(nb)

        weights = {
            "quality": float(config.selection_quality_weight),
            "novelty": float(config.selection_novelty_weight),
            "efficiency": float(config.selection_efficiency_weight),
            "feasibility": float(config.selection_feasibility_weight),
        }
        weight_sum = sum(max(0.0, w) for w in weights.values()) or 1.0
        for k in list(weights):
            weights[k] = max(0.0, weights[k]) / weight_sum

        quality_raw: Dict[str, float] = {}
        novelty_raw: Dict[str, float] = {}
        efficiency_raw: Dict[str, float] = {}
        feasibility_raw: Dict[str, float] = {}
        families: Dict[str, str] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        for row in candidates:
            rid = str(row.get("result_id") or "")
            if not rid:
                continue
            by_id[rid] = row
            family = nb._classify_architecture_family(
                row.get("graph_json"),
                row.get("routing_mode"),
            )
            families[rid] = family

            loss_ratio = self._to_float(row.get("loss_ratio"), default=1.0)
            baseline_ratio = self._to_float(row.get("baseline_loss_ratio"), default=1.0)
            ncd = self._to_float(row.get("ncd_score"), default=1.0)

            # Quality: Weighted combination of Loss and NCD (Information Redundancy)
            # Higher NCD = structure does not explain the behavior (memorization).
            # Lower NCD = structure captures the underlying rules (learning).
            loss_score = max(0.0, (1.0 - loss_ratio) + max(0.0, 1.0 - baseline_ratio))
            quality_raw[rid] = (0.7 * loss_score) + (0.3 * (1.0 - ncd))

            novelty_raw[rid] = self._to_float(row.get("novelty_score"), default=0.0)

            throughput = self._to_float(row.get("throughput_tok_s"), default=0.0)
            flops = self._to_float(row.get("flops_per_token"), default=0.0)
            mem = self._to_float(row.get("peak_memory_mb"), default=0.0)
            dl = self._to_float(row.get("ncd_description_length"), default=1000.0)

            # Baseline targets from research
            is_efficient_arch = (
                family.startswith("MoE-")
                or family.startswith("Adaptive-")
                or "Mamba" in family
            )
            target_throughput = 10000.0 if is_efficient_arch else 5000.0
            throughput_bonus = max(0.0, throughput / target_throughput)

            # Efficiency: Throughput, FLOPs, and MDL (Description Length)
            # We penalize large description lengths (complex IR)
            mdl_penalty = math.log(max(1.0, dl)) * 0.04
            efficiency_raw[rid] = (
                (throughput_bonus * 5.0) - (0.35 * flops) - (0.15 * mem) - mdl_penalty
            )

            # Add adaptive savings bonus if available
            savings = self._to_float(row.get("depth_savings_ratio"), default=0.0)
            if savings > 0:
                efficiency_raw[rid] += savings * 10.0

            stage0 = 1.0 if int(row.get("stage0_passed") or 0) == 1 else 0.0
            stage05 = 1.0 if int(row.get("stage05_passed") or 0) == 1 else 0.0
            stage1 = 1.0 if int(row.get("stage1_passed") or 0) == 1 else 0.0
            stability = self._to_float(row.get("stability_score"), default=0.0)
            grad_penalty = 0.0
            if int(row.get("has_nan_grad") or 0) == 1:
                grad_penalty += 0.5
            if int(row.get("has_zero_grad") or 0) == 1:
                grad_penalty += 0.3
            feasibility_raw[rid] = max(
                0.0,
                (0.2 * stage0 + 0.2 * stage05 + 0.3 * stage1 + 0.3 * stability)
                - grad_penalty,
            )

        qn = self._norm_map(quality_raw, higher_is_better=True)
        nn = self._norm_map(novelty_raw, higher_is_better=True)
        en = self._norm_map(efficiency_raw, higher_is_better=True)
        fn = self._norm_map(feasibility_raw, higher_is_better=True)

        family_stats = nb.get_selection_family_stats()
        total_trials = sum(int(s.get("n_trials") or 0) for s in family_stats.values())
        family_bonus_raw: Dict[str, float] = {}
        family_uncertainty: Dict[str, float] = {}
        for rid, fam in families.items():
            stat = family_stats.get(fam, {})
            n_trials = int(stat.get("n_trials") or 0)
            mean_reward = self._to_float(stat.get("mean_reward"), default=0.0)
            uncertainty = 1.0 / math.sqrt(n_trials + 1.0)
            ucb = mean_reward + float(config.selection_ucb_c) * math.sqrt(
                math.log(max(total_trials, 1) + 1.0) / (n_trials + 1.0)
            )
            family_uncertainty[rid] = uncertainty
            family_bonus_raw[rid] = (
                uncertainty if config.selection_policy == "epsilon_greedy" else ucb
            )
        family_bonus = self._norm_map(family_bonus_raw, higher_is_better=True)
        unc_norm = self._norm_map(family_uncertainty, higher_is_better=True)
        insight_by_result, supporting_insight_ids = self._selection_supporting_insights(
            nb, candidates
        )
        interaction_rows = nb.get_selection_insight_interactions(limit=500)
        interaction_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in interaction_rows:
            key = (
                str(row.get("insight_a") or ""),
                str(row.get("insight_b") or ""),
            )
            interaction_map[key] = row
        insight_interaction_raw: Dict[str, float] = {}
        for rid in by_id:
            matched = insight_by_result.get(rid) or []
            rewards: List[float] = []
            for insight_id in matched:
                stat = interaction_map.get((insight_id, insight_id))
                if stat and int(stat.get("n_trials") or 0) >= 2:
                    rewards.append(self._to_float(stat.get("mean_reward"), default=0.5))
            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    a, b = matched[i], matched[j]
                    if a > b:
                        a, b = b, a
                    stat = interaction_map.get((a, b))
                    if stat and int(stat.get("n_trials") or 0) >= 2:
                        rewards.append(
                            self._to_float(stat.get("mean_reward"), default=0.5)
                        )
            if rewards:
                insight_interaction_raw[rid] = float(sum(rewards) / len(rewards))
            else:
                insight_interaction_raw[rid] = 0.5
        insight_interaction = self._norm_map(
            insight_interaction_raw, higher_is_better=True
        )

        scored: List[Dict[str, Any]] = []
        for rid in by_id:
            base_score = (
                weights["quality"] * qn.get(rid, 0.0)
                + weights["novelty"] * nn.get(rid, 0.0)
                + weights["efficiency"] * en.get(rid, 0.0)
                + weights["feasibility"] * fn.get(rid, 0.0)
            )
            bonus = family_bonus.get(rid, 0.0)
            total = (1.0 - float(config.selection_family_bonus_weight)) * base_score + (
                float(config.selection_family_bonus_weight) * bonus
            )
            # Additive term to prefer insight bundles with positive historical interactions.
            interaction_term = (insight_interaction.get(rid, 0.5) - 0.5) * 0.20
            total += interaction_term
            scored.append(
                {
                    "result_id": rid,
                    "family": families.get(rid, "Unknown"),
                    "score": round(total, 6),
                    "base_score": round(base_score, 6),
                    "components": {
                        "quality": round(qn.get(rid, 0.0), 6),
                        "novelty": round(nn.get(rid, 0.0), 6),
                        "efficiency": round(en.get(rid, 0.0), 6),
                        "feasibility": round(fn.get(rid, 0.0), 6),
                        "insight_interaction": round(
                            insight_interaction.get(rid, 0.5), 6
                        ),
                    },
                    "family_bonus": round(bonus, 6),
                    "family_uncertainty": round(unc_norm.get(rid, 0.0), 6),
                    "supporting_insight_ids": insight_by_result.get(rid, []),
                    "raw": {
                        "loss_ratio": self._to_float(
                            by_id[rid].get("loss_ratio"), default=1.0
                        ),
                        "baseline_loss_ratio": self._to_float(
                            by_id[rid].get("baseline_loss_ratio"), default=1.0
                        ),
                        "novelty_score": self._to_float(
                            by_id[rid].get("novelty_score"), default=0.0
                        ),
                        "throughput_tok_s": self._to_float(
                            by_id[rid].get("throughput_tok_s"), default=0.0
                        ),
                        "flops_per_token": self._to_float(
                            by_id[rid].get("flops_per_token"), default=0.0
                        ),
                        "peak_memory_mb": self._to_float(
                            by_id[rid].get("peak_memory_mb"), default=0.0
                        ),
                        "stability_score": self._to_float(
                            by_id[rid].get("stability_score"), default=0.0
                        ),
                    },
                }
            )

        scored.sort(
            key=lambda x: (x["score"], x["base_score"], x["components"]["novelty"]),
            reverse=True,
        )
        seed = self._stable_seed(
            context, experiment_id or "none", len(scored), total_trials
        )
        rng = random.Random(seed)
        exploration = rng.random() < max(0.0, min(1.0, float(config.selection_epsilon)))

        if exploration:
            ranked = sorted(
                scored,
                key=lambda x: (
                    x["family_uncertainty"],
                    x["components"]["novelty"],
                    x["base_score"],
                ),
                reverse=True,
            )
            reason = f"Explore: epsilon trigger in {context}; prioritized high-uncertainty families."
        else:
            ranked = scored
            reason = f"Exploit: selected highest evidence-weighted scores in {context}."

        summary = {
            "candidate_count": len(scored),
            "families": sorted({s["family"] for s in scored}),
            "weights": weights,
            "supporting_insight_ids": supporting_insight_ids,
            "policy": config.selection_policy,
            "epsilon": float(config.selection_epsilon),
            "ucb_c": float(config.selection_ucb_c),
            "exploration": exploration,
            "seed": seed,
        }
        policy = {
            "name": config.selection_policy,
            "exploration": exploration,
            "reason": reason,
            "family_stats": family_stats,
            "supporting_insight_ids": supporting_insight_ids,
        }
        return {
            "summary": summary,
            "scored": scored,
            "selected": ranked,
            "reason": reason,
            "policy": policy,
            "supporting_insight_ids": supporting_insight_ids,
            "supporting_insights_by_result": insight_by_result,
        }

    def _selection_safety_valve(
        self, nb: LabNotebook, config: RunConfig
    ) -> Optional[Dict[str, Any]]:
        """Trigger novelty/ablation-heavy fallback after repeated stagnation."""
        window = max(3, int(config.safety_plateau_window))
        recent = nb.get_recent_experiments(window)
        if len(recent) < window:
            return None
        ordered = list(reversed(recent))
        loss_vals = [
            self._to_float(e.get("best_loss_ratio"), default=float("nan"))
            for e in ordered
        ]
        loss_vals = [v for v in loss_vals if not math.isnan(v)]
        n_stage1 = [int(e.get("n_stage1_passed") or 0) for e in ordered]
        if not loss_vals:
            return None

        first_loss = loss_vals[0]
        best_recent = min(loss_vals)
        loss_gain = max(0.0, first_loss - best_recent)
        no_survivor_progress = all(v <= 0 for v in n_stage1)
        plateau = (
            loss_gain < float(config.safety_plateau_min_delta) and no_survivor_progress
        )
        if not plateau:
            return None

        with self._lock:
            recent_modes = [
                c.get("mode", "synthesis") for c in self._aria_cycle_history[-window:]
            ]
        novelty_share = sum(1 for m in recent_modes if m == "novelty") / max(
            1, len(recent_modes)
        )
        mode = "ablation_heavy" if novelty_share >= 0.5 else "novelty"
        return {
            "triggered": True,
            "mode": mode,
            "window": window,
            "loss_gain": round(loss_gain, 6),
            "min_required_gain": float(config.safety_plateau_min_delta),
            "no_survivor_progress": no_survivor_progress,
            "reason": (
                f"No measurable progress over {window} experiments "
                f"(loss gain={loss_gain:.4f}, S1 survivors unchanged)."
            ),
        }

    def _selection_supporting_insights(
        self,
        nb: LabNotebook,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Match active insights to candidates using structured matching.

        Filters out display_only insights and those with confidence <= 0.55.
        Uses insight_level + subject_key for structured matching instead of
        naive token-in-content overlap.
        """
        insights = nb.get_insights(limit=120, exclude_display_only=True)
        if not insights:
            return {}, []

        # Filter: only insights better than coin flip
        insights = [
            i
            for i in insights
            if (
                self._to_float(i.get("alpha"), default=1.0)
                / (
                    self._to_float(i.get("alpha"), default=1.0)
                    + self._to_float(i.get("beta_"), default=1.0)
                )
            )
            > 0.55
        ]
        if not insights:
            return {}, []

        by_result: Dict[str, List[str]] = {}
        global_scores: Dict[str, float] = {}
        for row in candidates:
            rid = str(row.get("result_id") or "")
            if not rid:
                continue
            tokens = self._candidate_tokens(row)
            self._to_float(row.get("graph_n_ops"), default=0)
            if not tokens:
                continue
            scored: List[Tuple[float, str]] = []
            for insight in insights:
                insight_id = str(insight.get("insight_id") or "")
                if not insight_id:
                    continue
                confidence = self._to_float(insight.get("alpha"), default=1.0) / (
                    self._to_float(insight.get("alpha"), default=1.0)
                    + self._to_float(insight.get("beta_"), default=1.0)
                )
                level = str(insight.get("insight_level") or "op")
                subject = str(insight.get("subject_key") or "")

                matched = False
                if level == "composition":
                    # Match if candidate graph contains the ops in subject_key
                    subject_ops = {s.strip() for s in subject.split("+") if s.strip()}
                    if subject_ops and subject_ops.issubset(tokens):
                        matched = True
                elif level == "structural":
                    # Match based on graph properties
                    if "graph_size" in subject:
                        matched = True  # structural size insights apply globally
                    elif subject in tokens:
                        matched = True
                elif level == "template":
                    # Match if any subject token appears in candidate tokens
                    subject_parts = {
                        s.strip().lower()
                        for s in subject.replace("+", " ").replace("_", " ").split()
                        if len(s.strip()) >= 3
                    }
                    if subject_parts & tokens:
                        matched = True
                else:
                    # Fallback: op-level, match subject_key directly
                    if subject and subject in tokens:
                        matched = True

                if not matched:
                    # Token fallback for backwards compat with legacy insights
                    content = str(insight.get("content") or "").lower()
                    hit_count = sum(1 for token in tokens if token in content)
                    if hit_count >= 2:
                        matched = True
                        confidence *= 0.8  # Discount token-matched insights

                if matched:
                    scored.append((confidence, insight_id))

            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            matched_ids = [insight_id for _, insight_id in scored[:3]]
            by_result[rid] = matched_ids
            for score, insight_id in scored[:5]:
                global_scores[insight_id] = max(
                    global_scores.get(insight_id, 0.0), score
                )

        global_ids = [
            iid
            for iid, _ in sorted(
                global_scores.items(), key=lambda kv: kv[1], reverse=True
            )[:6]
        ]
        return by_result, global_ids

    def _rule_based_insights(
        self, results: Dict, exp_id: str, nb: LabNotebook
    ) -> List[str]:
        """Rule-based insight generation (always runs)."""
        insights = []

        s0_rate = results["stage0_passed"] / max(results["total"], 1)
        s1_rate = results["stage1_passed"] / max(results["total"], 1)

        if s0_rate < 0.2:
            insight = "Low Stage 0 pass rate — grammar produces too many invalid programs. Consider tightening shape constraints."
            insights.append(insight)
            nb.record_insight("failure_mode", insight, exp_id, confidence=0.7)
            # failure_mode insights are display-only — not fed to Aria's decision state

        if s0_rate > 0.5 and s1_rate < 0.01:
            insight = "Programs compile but don't learn. The operations may not compose into learnable functions. Need more parameterized ops."
            insights.append(insight)
            nb.record_insight("failure_mode", insight, exp_id, confidence=0.6)
            # failure_mode insights are display-only — not fed to Aria's decision state

        if results["novel_count"] > 0:
            insight = f"Found {results['novel_count']} genuinely novel survivors! Behaviorally distinct from known architectures."
            insights.append(insight)
            nb.record_insight("success_factor", insight, exp_id, confidence=0.8)

        if s1_rate > 0.05:
            insight = f"Strong Stage 1 pass rate ({s1_rate:.0%}). Current grammar configuration is productive."
            insights.append(insight)
            nb.record_insight("pattern", insight, exp_id, confidence=0.7)

        return insights

    # ── Rich Context Helpers ──

    def _ensure_novelty_calibration(
        self,
        nb: LabNotebook,
        config: RunConfig,
        fp: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Fetch or create baseline novelty calibration for the active reference version."""
        if fp is None:
            return None
        reference_version = getattr(fp, "novelty_reference_version", None)
        if not reference_version:
            return None
        row = nb.get_latest_novelty_calibration(reference_version=reference_version)
        if row is not None:
            return row
        if not config.auto_novelty_calibration:
            return None

        try:
            from ...eval.novelty_calibration import (
                calibrate_baseline_transformer_novelty,
            )

            calibration = calibrate_baseline_transformer_novelty(
                n_runs=max(2, int(config.novelty_calibration_runs)),
                seq_len=min(32, int(config.max_seq_len)),
                model_dim=max(16, int(config.model_dim)),
                vocab_size=max(256, min(4096, int(config.vocab_size))),
                device="cpu",
                seed=self._stable_seed("novelty_calibration", reference_version),
            )
            nb.record_novelty_calibration(
                reference_version=calibration.get("reference_version")
                or reference_version,
                cka_source=calibration.get("cka_source"),
                cka_artifact_version=calibration.get("cka_artifact_version"),
                probe_protocol_hash=calibration.get("probe_protocol_hash"),
                n_runs=int(
                    calibration.get("n_runs") or config.novelty_calibration_runs
                ),
                noise_floor_mean=calibration.get("noise_floor_mean"),
                noise_floor_std=calibration.get("noise_floor_std"),
                confidence_low=calibration.get("confidence_low"),
                confidence_high=calibration.get("confidence_high"),
                distribution=calibration.get("distribution") or {},
                metadata=calibration.get("metadata") or {},
            )
            return nb.get_latest_novelty_calibration(
                reference_version=reference_version
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Novelty calibration failed for %s: %s", reference_version, e)
            return None

    def _populate_refuted_cache(self, nb: LabNotebook) -> None:
        """Populate the persona's refuted hypothesis cache for similarity gating.

        Merges refuted insights from the insights table with refuted hypotheses
        from negative_results_synthesis so the persona can reject new hypotheses
        that are too similar to proven failures.
        """
        refuted: List[Dict] = []
        try:
            # Source 1: Formally refuted insights
            for ins in nb.get_insights(status="refuted", limit=20):
                content = ins.get("content", "")
                if content:
                    refuted.append(
                        {
                            "content": content,
                            "confidence": ins.get("confidence", 0),
                        }
                    )

            # Source 2: Refuted hypotheses from negative_results_synthesis
            try:
                from ..analytics import ExperimentAnalytics

                analytics = ExperimentAnalytics(nb)
                neg = analytics.negative_results_synthesis()
                for rh in neg.get("refuted_hypotheses", []):
                    content = rh.get("content", "")
                    if content and not any(
                        r.get("content", "")[:80] == content[:80] for r in refuted
                    ):
                        refuted.append(
                            {
                                "content": content,
                                "confidence": rh.get("confidence", 0),
                            }
                        )
            except (ImportError, RuntimeError) as e:
                logger.debug("Negative results synthesis failed: %s", e)
        except (sqlite3.OperationalError, RuntimeError) as e:
            logger.debug("Refuted hypothesis cache population failed: %s", e)

        self.aria.set_refuted_hypotheses(refuted)

    def _is_control_experiment(self, config: RunConfig, n_experiments: int) -> bool:
        """Whether this continuous synthesis run should be a control experiment."""
        interval = int(getattr(config, "control_experiment_interval", 0) or 0)
        return interval > 0 and n_experiments > 0 and (n_experiments % interval == 0)

    def _candidate_tokens(self, row: Dict[str, Any]) -> Set[str]:
        """Extract lightweight semantic tokens for insight matching."""
        tokens: Set[str] = set()
        family = LabNotebook._classify_architecture_family(
            row.get("graph_json"),
            row.get("routing_mode"),
        )
        if family:
            tokens.update(
                part
                for part in family.lower().replace("-", " ").split()
                if len(part) >= 3
            )

        graph_json = row.get("graph_json")
        if isinstance(graph_json, str) and graph_json:
            try:
                graph_data = json.loads(graph_json)
                nodes = (
                    graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
                )
                for nd in nodes.values():
                    if not isinstance(nd, dict):
                        continue
                    op_name = str(nd.get("op_name") or "").strip().lower()
                    if not op_name or op_name == "input":
                        continue
                    tokens.add(op_name)
                    tokens.update(
                        part
                        for part in op_name.replace("-", "_").split("_")
                        if len(part) >= 3
                    )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("Failed to parse graph_json for tokens: %s", e)
        return tokens

    def _op_success_lookup(self, nb: LabNotebook) -> Dict[str, float]:
        """Return per-op Stage1 success rates for learning-guided refinement.

        Uses 7d windowed rates to avoid death spiral from stale lifetime data.
        """
        import time as _time

        lookup: Dict[str, float] = {}
        try:
            since_ts = _time.time() - 604800  # 7 days
            for row in nb.get_op_success_rates_windowed(since_ts):
                n_used = float(row.get("n_used") or 0.0)
                n_s1 = float(row.get("n_stage1_passed") or 0.0)
                if n_used > 0:
                    lookup[str(row.get("op_name"))] = n_s1 / n_used
        except (sqlite3.OperationalError, RuntimeError) as e:
            logger.debug("Op success rate lookup failed: %s", e)
        return lookup

    @staticmethod
    def _safe_build_evidence_pack(
        nb: LabNotebook,
        recommendation: Dict[str, Any],
        decision_type: str,
    ) -> Dict[str, Any]:
        try:
            return build_evidence_pack(
                nb,
                analytics=None,
                recommendation=recommendation,
                decision_type=decision_type,
            )
        except (RuntimeError, ValueError, sqlite3.OperationalError) as e:
            logger.debug("Evidence pack build failed: %s", e)
            return {
                "hypothesis": "Insufficient metrics; gather more evidence before confident action.",
                "supporting_metrics": [
                    {
                        "name": "evidence_unavailable",
                        "value": 0.0,
                        "baseline": 0.0,
                        "delta_vs_baseline": 0.0,
                    }
                ],
                "uncertainty": {
                    "note": "Evidence pack fallback due to sparse metrics."
                },
                "confounders": ["Sparse or missing recent experiment metrics."],
                "falsification": [
                    "If next experiment still yields sparse metrics, block automation."
                ],
            }

    def _resolve_baseline_recipe(
        self,
        train_result: Dict[str, Any],
        default_lr: float,
        default_weight_decay: float = 0.01,
    ) -> Dict[str, Any]:
        """Resolve baseline training recipe from observed candidate metadata."""
        optimizer_name = "adamw"

        optimizer_class = str(train_result.get("optimizer_class") or "").lower()
        if "sgd" in optimizer_class:
            optimizer_name = "sgd"

        lr = float(
            train_result.get("final_lr")
            or train_result.get("optimizer_lr")
            or default_lr
        )
        weight_decay = float(
            train_result.get("optimizer_weight_decay", default_weight_decay)
        )
        momentum = float(train_result.get("optimizer_momentum", 0.0))

        beta1 = train_result.get("optimizer_beta1")
        beta2 = train_result.get("optimizer_beta2")
        betas: Optional[Tuple[float, float]] = None
        if beta1 is not None and beta2 is not None:
            betas = (float(beta1), float(beta2))

        tp_json = train_result.get("training_program_json")
        if tp_json and not optimizer_class:
            try:
                tp = json.loads(tp_json)
                opt = tp.get("optimizer") or {}
                opt_name = str(opt.get("name") or "").lower()
                comps = [str(c).lower() for c in (opt.get("components") or [])]
                if "sgd" in opt_name or "sgd" in comps:
                    optimizer_name = "sgd"
                if "lr" in opt:
                    lr = float(opt["lr"])
                if "weight_decay" in opt:
                    weight_decay = float(opt["weight_decay"])
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                logger.debug(
                    "Failed to parse training_program_json for baseline recipe: %s", e
                )

        return {
            "optimizer_name": optimizer_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "betas": betas,
        }

    def _ood_robustness_check(
        self,
        model_factory: Callable[[], nn.Module],
        config: RunConfig,
        dev: torch.device,
        n_steps: int = 300,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test a candidate against hand-designed reference training recipes.

        Returns a dict with per-recipe results and an overall robustness score
        (fraction of recipes that achieved loss_ratio < 0.9).
        """
        recipe_results = []

        for recipe in self._REFERENCE_RECIPES:
            if self._stop_event.is_set():
                break

            try:
                model = model_factory().to(dev)
                model.train()

                if recipe["optimizer"] == "sgd":
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=recipe["lr"],
                        momentum=recipe.get("momentum", 0.0),
                        weight_decay=recipe.get("weight_decay", 0.0),
                    )
                else:  # adamw
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=recipe["lr"],
                        weight_decay=recipe.get("weight_decay", 0.01),
                        betas=(0.9, 0.95),
                    )

                seq_len = min(128, config.max_seq_len)
                initial_loss = None
                final_loss = None

                for step in range(n_steps):
                    if self._stop_event.is_set():
                        break

                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=config.stage1_batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )

                    with torch.amp.autocast(
                        device_type=dev.type,
                        dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val

                loss_ratio = (
                    normalized_loss_ratio(final_loss, config.vocab_size)
                    if initial_loss and final_loss
                    else None
                )
                recipe_results.append(
                    {
                        "recipe": recipe["name"],
                        "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                        "passed": loss_ratio is not None and loss_ratio < 0.9,
                        "initial_loss": initial_loss,
                        "final_loss": final_loss,
                    }
                )

                del model
                clear_gpu_memory()

            except (RuntimeError, ValueError, TypeError) as e:
                recipe_results.append(
                    {
                        "recipe": recipe["name"],
                        "loss_ratio": None,
                        "passed": False,
                        "error": str(e),
                    }
                )

        n_passed = sum(1 for r in recipe_results if r.get("passed"))
        return {
            "recipes_tested": len(recipe_results),
            "recipes_passed": n_passed,
            "ood_robustness": n_passed / max(len(recipe_results), 1),
            "recipe_results": recipe_results,
        }

    # ── Hyperparameter Sensitivity (#57) ──

    # Perturbations to test: each is (label, param_overrides) where overrides
    # are multipliers applied to the base config values.
    def _sensitivity_check(
        self,
        model_factory: Callable[[], nn.Module],
        config: RunConfig,
        dev: torch.device,
        base_loss_ratio: float,
        n_steps: int = 300,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test whether a candidate's performance is sensitive to hyperparameter changes.

        Trains the model with ±2x learning rate and ±2x training steps.
        Returns per-perturbation loss ratios and an overall sensitivity score.
        A robust candidate should learn under all perturbations (loss_ratio < 1.0).
        """
        perturbation_results = []
        base_lr = config.stage1_lr

        for label, overrides in self._SENSITIVITY_PERTURBATIONS:
            if self._stop_event.is_set():
                break

            lr = base_lr * overrides.get("lr_mult", 1.0)
            steps = int(n_steps * overrides.get("steps_mult", 1.0))

            try:
                model = model_factory().to(dev)
                model.train()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
                )

                seq_len = min(128, config.max_seq_len)
                initial_loss = None
                final_loss = None

                for step in range(steps):
                    if self._stop_event.is_set():
                        break

                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=config.stage1_batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )

                    with torch.amp.autocast(
                        device_type=dev.type,
                        dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val

                loss_ratio = (
                    normalized_loss_ratio(final_loss, config.vocab_size)
                    if initial_loss and final_loss
                    else None
                )

                # How much did loss_ratio change vs the base run?
                deviation = (
                    abs(loss_ratio - base_loss_ratio) / max(base_loss_ratio, 1e-6)
                    if loss_ratio is not None
                    else None
                )

                perturbation_results.append(
                    {
                        "perturbation": label,
                        "lr": lr,
                        "steps": steps,
                        "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                        "deviation_from_base": round(deviation, 4)
                        if deviation is not None
                        else None,
                        "still_learns": loss_ratio is not None and loss_ratio < 1.0,
                    }
                )

                del model
                clear_gpu_memory()

            except (RuntimeError, ValueError, TypeError) as e:
                perturbation_results.append(
                    {
                        "perturbation": label,
                        "loss_ratio": None,
                        "still_learns": False,
                        "error": str(e),
                    }
                )

        n_learns = sum(1 for r in perturbation_results if r.get("still_learns"))
        deviations = [
            r["deviation_from_base"]
            for r in perturbation_results
            if r.get("deviation_from_base") is not None
        ]
        avg_deviation = sum(deviations) / len(deviations) if deviations else None

        return {
            "perturbations_tested": len(perturbation_results),
            "perturbations_learn": n_learns,
            "hp_robustness": n_learns / max(len(perturbation_results), 1),
            "avg_deviation": round(avg_deviation, 4)
            if avg_deviation is not None
            else None,
            "perturbation_results": perturbation_results,
        }

    # ── Investigation Phase ──

    @staticmethod
    def _routing_stability_from_curve(
        training_curve: List[Dict[str, Any]],
    ) -> Optional[float]:
        """Compute a simple stability score from per-step loss trajectory."""
        if not training_curve:
            return None
        losses = [
            float(row.get("loss"))
            for row in training_curve
            if row.get("loss") is not None
        ]
        if len(losses) < 2:
            return None
        tail = losses[max(0, len(losses) // 2) :]
        if len(tail) < 2:
            return None
        mean_loss = sum(tail) / len(tail)
        if mean_loss <= 1e-8:
            return 1.0
        variance = sum((v - mean_loss) ** 2 for v in tail) / len(tail)
        std = variance**0.5
        cv = std / mean_loss
        return 1.0 / (1.0 + cv)

    @staticmethod
    def _validation_config_with_result_ids(
        config: RunConfig,
        result_ids: List[str],
        trigger: str,
    ) -> Dict[str, Any]:
        """Attach validation candidate metadata to persisted experiment config."""
        cfg = config.to_dict()
        ids = [rid for rid in result_ids if rid]
        cfg["validation_result_ids"] = ids
        cfg["validation_candidate_count"] = len(ids)
        cfg["validation_trigger"] = trigger
        return cfg
