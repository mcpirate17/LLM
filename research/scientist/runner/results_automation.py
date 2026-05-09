"""Results automation mixin: recommendations, reports, scale-up, campaigns."""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple


from ..evidence import build_evidence_pack
from ..llm.decision import NextExperimentDecisionPlanner
from ..notebook import ExperimentEntry, LabNotebook
from ..shared_utils import clamp
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ResultsAutomationMixin:
    """Auto-recommendation, auto-report, auto-scale-up, campaign evaluation."""

    __slots__ = ()

    def _auto_recommend(
        self, results: Dict, config: RunConfig, hypothesis: str, nb: LabNotebook
    ):
        """Auto-generate a recommendation after experiment completion and APPLY it."""
        try:
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
            _analytics = self._gather_analytics_data(nb)
            op_rates = _analytics.get("op_success_rates")
            comp_cov = _analytics.get("compression_coverage")
            heuristic = (
                self.aria.suggest_experiment(
                    context, op_success_rates=op_rates, compression_coverage=comp_cov
                )
                or {}
            )
            summary_payload = self._build_next_experiment_summary(nb, results)
            planner = NextExperimentDecisionPlanner.from_run_config(config)
            plan = planner.propose_plan(
                summary_payload,
                current_cost_dollars=float(self.aria.total_cost or 0.0),
                fallback_plan=heuristic,
            )
            suggestion = {
                "mode": plan.get("mode", heuristic.get("mode", "synthesis")),
                "reasoning": plan.get("reasoning", heuristic.get("reasoning", "")),
                "confidence": float(
                    plan.get("confidence", heuristic.get("confidence", 0.5)) or 0.5
                ),
                "config": plan.get("config", heuristic.get("config", {})),
                "planner": plan.get("planner", {}),
                "guardrails": plan.get("guardrails", {}),
                "summary_excerpt": plan.get("summary_excerpt", {}),
            }
            if suggestion:
                evidence_pack = build_evidence_pack(
                    nb,
                    analytics=None,
                    recommendation=suggestion,
                    decision_type="experiment_recommendation",
                )
                suggestion["evidence_pack"] = evidence_pack
                with self._lock:
                    self._last_recommendation = suggestion
                self._emit_event(
                    "aria_recommendation",
                    {
                        "mode": suggestion.get("mode"),
                        "reasoning": suggestion.get("reasoning", ""),
                        "confidence": suggestion.get("confidence", 0),
                        "config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "evidence_pack": evidence_pack,
                    },
                )
                # Store as notebook entry
                nb.add_entry(
                    ExperimentEntry(
                        entry_type="decision",
                        title="Aria's Next Experiment Recommendation",
                        content=suggestion.get("reasoning", ""),
                        metadata={
                            "mode": suggestion.get("mode"),
                            "confidence": suggestion.get("confidence", 0),
                            "suggested_config": suggestion.get("config", {}),
                            "planner": suggestion.get("planner", {}),
                            "guardrails": suggestion.get("guardrails", {}),
                            "summary_payload": summary_payload,
                            "evidence_pack": evidence_pack,
                        },
                    )
                )
                nb.record_decision(
                    campaign_id=self._active_campaign_id,
                    decision_type="next_experiment_plan",
                    subject=f"experiment:{summary_payload.get('recent_experiment_id') or 'latest'}",
                    rationale=suggestion.get("reasoning", ""),
                    alternatives=[
                        {
                            "heuristic_fallback": heuristic,
                        }
                    ],
                    evidence_pack={
                        "mode": suggestion.get("mode"),
                        "confidence": suggestion.get("confidence", 0),
                        "config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "guardrails": suggestion.get("guardrails", {}),
                        "summary_payload": summary_payload,
                    },
                )
                # PROACTIVE: Apply suggested config/grammar changes immediately
                self._apply_recommendation(suggestion, nb)
        except Exception as e:
            logger.debug(f"Auto-recommendation failed: {e}")

    def _apply_recommendation(self, suggestion: Dict, nb: LabNotebook):
        """Proactively apply Aria's recommended config and grammar changes.

        Also detects code-level issues in reasoning and spawns repair agents.
        """
        if not suggestion.get("evidence_pack"):
            logger.warning(
                "Skipping recommendation application: missing Evidence Pack."
            )
            return
        confidence = suggestion.get("confidence", 0)
        reasoning = str(suggestion.get("reasoning") or "")

        # Detect code-level issues in reasoning and spawn agent
        if confidence >= 0.3 and reasoning:
            self._maybe_spawn_agent_from_reasoning(reasoning, nb)

        if confidence < 0.4:
            return  # Low confidence — don't auto-apply config

        suggested_config = suggestion.get("config") or {}
        if not suggested_config:
            return

        # Categorize suggested keys into bins
        GRAMMAR_WEIGHT_KEYS = {"math_space_weight"}
        CATEGORY_WEIGHT_KEY = "category_weights"
        CONFIG_OVERRIDE_KEYS = {
            "n_programs",
            "model_dim",
            "max_depth",
            "max_ops",
            "model_source",
            "morph_focus_sparse",
            "use_synthesized_training",
            "novelty_weight",
            "selection_family_bonus_weight",
            "refinement_top_k",
            "refinement_generations",
            "refinement_budget_programs",
            "grammar_split_prob",
            "grammar_merge_prob",
            "grammar_risky_op_prob",
            "grammar_freq_domain_prob",
            "structured_sparsity_bias",
            "residual_prob",
            "optimizer_preference",
        }

        # Sanity clamps for numeric config values
        CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
            "grammar_split_prob": (0.0, 1.0),
            "grammar_merge_prob": (0.0, 1.0),
            "grammar_risky_op_prob": (0.0, 1.0),
            "grammar_freq_domain_prob": (0.0, 1.0),
            "structured_sparsity_bias": (0.0, 1.0),
            "residual_prob": (0.0, 1.0),
            "n_programs": (4, 500),
            "max_depth": (2, 30),
            "max_ops": (3, 40),
            "model_dim": (32, 1024),
        }
        GRAMMAR_WEIGHT_CLAMP = (0.1, 10.0)  # category weights & math_space_weight
        OP_WEIGHT_CLAMP = (0.01, 10.0)

        grammar_overrides = {}
        config_overrides = {}
        for k, v in suggested_config.items():
            if k in GRAMMAR_WEIGHT_KEYS:
                if isinstance(v, (int, float)):
                    grammar_overrides[k] = clamp(float(v), *GRAMMAR_WEIGHT_CLAMP)
            elif k == CATEGORY_WEIGHT_KEY and isinstance(v, dict):
                # Category weights dict → merge into grammar weight overrides
                for cat_name, weight in v.items():
                    if isinstance(weight, (int, float)):
                        grammar_overrides[cat_name] = clamp(
                            float(weight), *GRAMMAR_WEIGHT_CLAMP
                        )
            elif k == "op_weights" and isinstance(v, dict):
                new_op_weights = {
                    str(op): clamp(float(w), *OP_WEIGHT_CLAMP)
                    for op, w in v.items()
                    if isinstance(op, str) and isinstance(w, (int, float))
                }
                if new_op_weights:
                    self._op_weights_overrides.update(new_op_weights)
                    self._log_learning_event_compat(
                        nb,
                        "auto_op_weights",
                        f"Aria adjusted op weights: {new_op_weights}",
                        op_weights=new_op_weights,
                    )
                    logger.info("Aria auto-applied op weights: %s", new_op_weights)
            elif k in CONFIG_OVERRIDE_KEYS:
                if k in CLAMP_RANGES and isinstance(v, (int, float)):
                    lo, hi = CLAMP_RANGES[k]
                    v = type(v)(max(lo, min(hi, v)))
                config_overrides[k] = v

        if grammar_overrides:
            self._grammar_weight_overrides.update(grammar_overrides)
            self._log_learning_event_compat(
                nb,
                "auto_grammar_adjusted",
                f"Aria proactively adjusted grammar weights: {grammar_overrides}",
                weights=grammar_overrides,
            )
            logger.info("Aria auto-applied grammar overrides: %s", grammar_overrides)

        if config_overrides:
            self._last_chat_config_overrides = {
                **(self._last_chat_config_overrides or {}),
                **config_overrides,
            }
            self._log_learning_event_compat(
                nb,
                "auto_config_adjusted",
                f"Aria proactively adjusted config: {config_overrides}",
                changes=config_overrides,
            )
            logger.info("Aria auto-applied config overrides: %s", config_overrides)

    def _maybe_auto_report(
        self, config: RunConfig, nb: LabNotebook, reason: str = "session_end"
    ):
        """Auto-generate and store a research report."""
        if not config.auto_report:
            return

        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)

            report_data = {
                "summary": nb.get_dashboard_summary(
                    include_data_accounting=False,
                    include_template_observability=False,
                ),
                "top_programs": nb.get_top_programs(20, sort_by="loss_ratio"),
                "recent_experiments": nb.get_recent_experiments(100),
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "efficiency_frontier_3d": analytics.efficiency_frontier_3d(),
                "grammar_weights": analytics.compute_grammar_weights() or {},
                "default_weights": analytics.get_current_grammar_weights(),
            }

            narrative = self.aria.generate_report_narrative(report_data)

            nb.add_entry(
                ExperimentEntry(
                    entry_type="report",
                    title=f"Research Report ({reason})",
                    content=narrative,
                    metadata={
                        "trigger": reason,
                        "total_experiments": report_data["summary"].get(
                            "total_experiments", 0
                        ),
                        "stage1_survivors": report_data["summary"].get(
                            "stage1_survivors", 0
                        ),
                    },
                )
            )

            # Save as markdown file for human/LLM consumption
            nb.save_report_markdown(narrative, reason, report_data["summary"])

            self._emit_event(
                "auto_report_generated",
                {
                    "reason": reason,
                    "narrative_length": len(narrative),
                    "summary": report_data["summary"],
                },
            )

            logger.info(f"Auto-report generated ({reason}): {len(narrative)} chars")
        except Exception as e:
            logger.warning(f"Auto-report generation failed: {e}")

    def _maybe_auto_scale_up(self, results: Dict, config: RunConfig, nb: LabNotebook):
        """Check if we should auto-trigger scale-up after an experiment.

        Criteria:
        1. auto_scale_up is enabled in config
        2. Enough S1 survivors (>= auto_scale_up_min_survivors)
        3. Survivors have sufficient novelty (>= auto_scale_up_min_novelty avg)
        4. Not already a scale_up experiment (avoid recursion)
        5. No experiment currently running
        """
        if not config.auto_scale_up:
            return
        if config.scale_up:
            return  # don't chain scale-ups

        survivors = results.get("survivors", [])
        s1_count = results.get("stage1_passed", 0)

        if s1_count < config.auto_scale_up_min_survivors:
            return

        # Check novelty
        if survivors:
            # Don't gate on novelty_valid_for_promotion — missing novelty
            # data is a quality flag, not a disqualifier for scale-up.
            avg_novelty = sum(s.get("novelty", 0) for s in survivors) / len(survivors)
            if avg_novelty < config.auto_scale_up_min_novelty:
                return

        # Select top programs by loss ratio
        top_programs = nb.get_top_programs(
            config.auto_scale_up_top_n, sort_by="loss_ratio"
        )
        result_ids = [p["result_id"] for p in top_programs if p.get("stage1_passed")][
            : config.auto_scale_up_top_n
        ]

        if not result_ids:
            return

        logger.info(
            f"Auto-scale-up triggered: {len(result_ids)} programs qualify "
            f"(s1={s1_count}, survivors={len(survivors)})"
        )

        # Store the intent — can't start immediately since thread is still
        # running. Schedule via a flag the main thread can pick up.
        self._pending_scale_up = {
            "result_ids": result_ids,
            "config": config,
            "hypothesis": (
                f"Auto-scale-up: validating top {len(result_ids)} performers "
                f"at {config.scale_up_steps} steps to confirm they work at scale."
            ),
        }
        evidence_pack = build_evidence_pack(
            nb,
            analytics=None,
            recommendation={"mode": "scale_up"},
            decision_type="auto_scale_up",
        )
        self._pending_scale_up["evidence_pack"] = evidence_pack

        self._emit_event(
            "auto_scale_up_queued",
            {
                "result_ids": result_ids,
                "n_programs": len(result_ids),
                "reason": f"{s1_count} S1 survivors with avg novelty >= {config.auto_scale_up_min_novelty}",
                "evidence_pack": evidence_pack,
            },
        )

        nb.add_entry(
            ExperimentEntry(
                entry_type="decision",
                title="Auto-Scale-Up Triggered",
                content=(
                    f"Automatically queuing scale-up validation for {len(result_ids)} "
                    f"top performers. Criteria met: {s1_count} S1 survivors."
                ),
                metadata={"result_ids": result_ids, "evidence_pack": evidence_pack},
            )
        )

    def _run_pending_scale_up(self):
        """Launch pending auto-scale-up, auto-investigation, or auto-validation."""
        # Replay first: it resolves ambiguous frontier cases before we spend
        # heavier investigation/validation budget on them.
        if self._run_pending_replay():
            return

        # Check investigation first (higher priority)
        self._run_pending_investigation()
        if self.is_running:
            return

        # Then selective capability rankers before validation.
        self._run_pending_capability_ranking()
        if self.is_running:
            return

        # Then validation
        self._run_pending_validation()
        if self.is_running:
            return

        # Then scale-up
        pending = getattr(self, "_pending_scale_up", None)
        if pending is None:
            return
        self._pending_scale_up = None

        if self.is_running:
            return  # something else started, skip

        try:
            self.start_scale_up(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-scale-up: {e}")

    # ── Model Source Abstraction ──

    def _maybe_evaluate_campaign(self, config: RunConfig, nb: LabNotebook) -> None:
        """Evaluate campaign success criteria after an experiment.

        Auto-completes the campaign if criteria are met or the campaign is
        stale (10+ experiments with no criteria passing).  When a campaign
        completes, a successor campaign is formulated based on pipeline state.
        """
        if not config.enable_campaigns or not self._active_campaign_id:
            return

        try:
            evaluation = nb.evaluate_campaign_criteria(self._active_campaign_id)

            if not evaluation["all_met"] and not evaluation["stale"]:
                return  # still in progress

            campaign = nb.get_campaign(self._active_campaign_id)
            if not campaign or campaign.get("status") != "active":
                return

            # --- Complete the campaign ---
            if evaluation["all_met"]:
                reason = "criteria_met"
                findings = (
                    f"All {evaluation['n_criteria']} success criteria met. "
                    f"{evaluation['n_passing']} criteria passing."
                )
            else:
                reason = "stale"
                findings = (
                    f"Campaign stale after {len(nb.get_campaign_experiments(self._active_campaign_id))} "
                    f"experiments: {evaluation['n_at_risk']} criteria at risk, "
                    f"{evaluation['n_passing']} passing."
                )

            nb.update_campaign(
                self._active_campaign_id,
                status="completed",
                completed_at=time.time(),
                completion_reason=reason,
                findings_summary=findings,
            )

            self._emit_event(
                "campaign_completed",
                {
                    "campaign_id": self._active_campaign_id,
                    "title": campaign.get("title", ""),
                    "reason": reason,
                    "findings": findings,
                },
            )
            logger.info(
                f"Campaign completed ({reason}): "
                f"{campaign.get('title', '')} ({self._active_campaign_id})"
            )

            # --- Formulate successor campaign ---
            completed_id = self._active_campaign_id
            self._active_campaign_id = None

            # Determine next focus from pipeline state
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}

            recent = nb.get_recent_experiments(10)
            knowledge = nb.get_knowledge()
            all_campaigns = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            previous = [dict(r) for r in all_campaigns]

            # Build context that includes pipeline state for Aria
            from ..llm.context_hypothesis import build_campaign_formulation_context

            context = build_campaign_formulation_context(
                recent_experiments=recent,
                knowledge=knowledge,
                previous_campaigns=previous,
            )
            pipeline_hint = (
                f"\n\nPipeline state: "
                f"{tiers.get('screening', 0)} screening, "
                f"{tiers.get('investigation', 0)} investigation, "
                f"{tiers.get('validation', 0)} validation, "
                f"{tiers.get('breakthrough', 0)} breakthrough. "
            )
            if reason == "criteria_met":
                pipeline_hint += (
                    "Previous campaign succeeded — evolve to a more ambitious "
                    "objective (deeper investigation, validation, or scale-up)."
                )
            else:
                pipeline_hint += (
                    "Previous campaign stalled — pivot to a different approach "
                    "(novelty search, different architecture families, or "
                    "relaxed criteria)."
                )

            camp_data = self.aria.formulate_campaign(context=context + pipeline_hint)

            # Rule-based fallback: evolve based on pipeline state
            if camp_data["title"] == "Architecture Discovery Campaign":
                camp_data = self._pipeline_driven_campaign(tiers, reason)

            successor_id = nb.create_campaign(
                title=camp_data["title"],
                objective=camp_data["objective"],
                success_criteria=camp_data["success_criteria"],
                parent_id=completed_id,
            )

            # Link successor to completed campaign
            nb.update_campaign(
                completed_id,
                successor_campaign_id=successor_id,
            )

            self._active_campaign_id = successor_id
            self._emit_event(
                "campaign_created",
                {
                    "campaign_id": successor_id,
                    "title": camp_data["title"],
                    "objective": camp_data["objective"],
                    "predecessor": completed_id,
                },
            )
            logger.info(
                f"Successor campaign: {camp_data['title']} ({successor_id}) "
                f"→ replacing {completed_id}"
            )

        except Exception as e:
            logger.debug(f"Campaign evaluation failed: {e}")
