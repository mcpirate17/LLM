"""Control mixin: experiment start methods."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import json_safe

from ..notebook import LabNotebook
from ..llm.context_experiment import (
    build_mode_selection_context,
    build_manual_start_fallback_context,
)

from ._types import RunConfig, LiveProgress

logger = logging.getLogger(__name__)


class _ControlStartMixin:
    """Start-experiment family of methods for ExperimentRunner."""

    __slots__ = ()

    def start_experiment(
        self,
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
    ) -> str:
        """Start an experiment in a background thread. Returns experiment ID."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        config, prescreen = self.prescreen_run_config(
            config,
            mode="single",
            auto_harden=True,
        )

        self._ensure_math_spaces()
        self._stop_event.clear()
        self._set_aria_cycle_phase(
            "idle",
            continuous_active=False,
            cycle_index=0,
            selected_mode=None,
            note="Single-run experiment started.",
            emit_event=False,
        )

        # Pre-generate experiment ID
        nb = self._make_notebook()

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

        hypothesis_metadata = {
            "source": "user_input" if hypothesis is not None else "unknown",
            "llm_used": False,
            "fallback_used": False,
            "used_context": False,
            "review_status": "not_reviewed",
            "confidence": None,
            "critique": None,
        }
        if hypothesis is None:
            context = self._build_start_experiment_hypothesis_context(nb, config)
            llm_available = self.aria._get_llm() is not None
            if llm_available and not (context or "").strip():
                context = build_manual_start_fallback_context(config.to_dict())
            result = None
            if context:
                result = self.aria.formulate_hypothesis(
                    context=context,
                    return_metadata=True,
                )
                hypothesis_metadata["used_context"] = True
            else:
                result = self.aria.formulate_hypothesis(return_metadata=True)

            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = (
                    "rule_based_fallback" if context else "rule_based"
                )

            if context:
                hypothesis_metadata["context_char_count"] = len(context)

        # Preflight hypothesis critique
        critique = None
        if hypothesis:
            try:
                critique_context = (
                    self._build_start_experiment_hypothesis_context(
                        nb,
                        config,
                    )
                    if hypothesis_metadata.get("source") == "user_input"
                    else ""
                )
                critique = self.aria.critique_hypothesis(
                    hypothesis,
                    context=critique_context,
                )
                hypothesis_metadata["preflight_critique"] = critique
                hypothesis_metadata["critique"] = critique
                hypothesis_metadata["critique_confidence"] = critique.get("confidence")
                hypothesis_metadata["review_status"] = (
                    f"preflight_{critique.get('gate', 'warn')}"
                )
            except Exception as e:
                logger.warning(f"Hypothesis critique failed: {e}")

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_experiment",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=self.aria.greet(),
                hypothesis_critique=critique,
            )

        self._emit_event(
            "experiment_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "config": config.to_dict(),
                "prescreen": prescreen,
                "aria_greeting": self.aria.greet(),
                "hypothesis_critique": critique,
            },
        )

        self._thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _build_start_experiment_hypothesis_context(
        self,
        nb: LabNotebook,
        config: RunConfig,
    ) -> str:
        """Build context for hypothesis generation in manual start_experiment.

        Ensures manual starts use the same context-aware hypothesis pathway as
        continuous mode whenever history/analytics are available.
        """
        try:
            recent = nb.get_recent_experiments(10)
            leaderboard = nb.get_leaderboard(limit=20)
            analytics_data = self._gather_analytics_data(nb)
            context = build_mode_selection_context(
                recent_experiments=recent,
                leaderboard=leaderboard,
                analytics_data=analytics_data,
                current_mode="synthesis",
                n_experiments_in_session=len(recent),
                cost_spent=self.aria.total_cost,
                budget=config.max_cost_dollars,
            )
            if config.max_cost_dollars > 0:
                context += (
                    f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                    f"of ${config.max_cost_dollars:.2f}"
                )
            return context
        except Exception as e:
            logger.debug("Failed to build manual hypothesis context: %s", e)
            return build_manual_start_fallback_context(config.to_dict())

    def start_continuous(self, config: RunConfig) -> str:
        """Start continuous experiment mode in background."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        config, _ = self.prescreen_run_config(
            config,
            mode="continuous",
            auto_harden=True,
        )

        self._ensure_math_spaces()
        self._stop_event.clear()
        with self._lock:
            self._aria_cycle_paused = False

        config.continuous = True
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Continuous session initialized.",
        )

        limits = []
        if config.max_experiments > 0:
            limits.append(f"max_experiments={config.max_experiments}")
        if config.max_time_minutes > 0:
            limits.append(f"max_time={config.max_time_minutes}min")
        if config.max_cost_dollars > 0:
            limits.append(f"max_cost=${config.max_cost_dollars:.2f}")
        logger.info(
            "Starting continuous session: %d programs/cycle, dim=%d, "
            "depth=%d, ops=%d, device=%s [%s]",
            config.n_programs,
            config.model_dim,
            config.max_depth,
            config.max_ops,
            config.device,
            ", ".join(limits) if limits else "no limits",
        )

        with self._lock:
            self._progress = LiveProgress(
                status="generating",
                aria_message=f"{self.aria.NAME} entering continuous research mode...",
            )

        self._thread = threading.Thread(
            target=self._run_continuous_thread,
            args=(config,),
            daemon=True,
        )
        self._thread.start()
        return "continuous"

    def start_fingerprint_refinement(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
    ) -> str:
        """Start local mutation refinement around selected fingerprint sources."""
        ids = [rid.strip() for rid in result_ids if str(rid).strip()]
        if not ids:
            raise ValueError("result_ids required for fingerprint refinement")

        refine_config = config.copy()
        refine_config.model_source = "fingerprint_refine"
        refine_config.refine_source_result_ids = ",".join(ids)
        if refine_config.refine_mutations_per_source <= 0:
            refine_config.refine_mutations_per_source = 1

        source_stage1_passed = 0
        recent_synthesis_s1_rate = 0.0
        source_rows: List[Dict[str, Any]] = []
        recommendation: Optional[Dict[str, Any]] = None
        try:
            nb = self._make_notebook()
            recent = self._recent_synthesis_health(nb, window=5)
            recent_synthesis_s1_rate = float(recent.get("s1_rate") or 0.0)
            for rid in ids:
                row = nb.get_program_detail(rid)
                if row and row.get("stage1_passed"):
                    source_stage1_passed += 1
                if isinstance(row, dict):
                    source_rows.append(row)

            requested_intent = (
                str(refine_config.refine_intent or "balanced").strip().lower()
            )
            if requested_intent in {"recommended", "auto"}:
                # Auto-run RefinementAnalyzer if no pre-computed analysis
                if not refine_config.refine_analysis_json and source_rows:
                    try:
                        from ..analytics import ExperimentAnalytics, RefinementAnalyzer

                        analytics = ExperimentAnalytics(nb)
                        analyzer = RefinementAnalyzer(analytics)
                        primary_row = source_rows[0]
                        primary_id = primary_row.get("result_id", ids[0])
                        analysis = analyzer.analyze_program_for_refinement(
                            primary_id, primary_row
                        )
                        recipe = analysis.get("recipe", {})
                        resolved_intent = recipe.get("recommended_intent", "balanced")
                        recommendation = {
                            "intent": resolved_intent,
                            "rationale": recipe.get("primary_target", ""),
                            "evidence": recipe.get("grammar_hints", {}),
                        }
                        refine_config.refine_analysis_json = json.dumps(
                            json_safe(analysis)
                        )
                    except Exception as e:
                        logger.warning("RefinementAnalyzer failed, falling back: %s", e)
                        resolved_intent, recommendation = (
                            self._recommend_refinement_intent(
                                nb,
                                source_rows,
                            )
                        )
                else:
                    resolved_intent, recommendation = self._recommend_refinement_intent(
                        nb,
                        source_rows,
                    )
                refine_config.refine_intent = resolved_intent
            nb.close()
        except Exception as exc:
            logger.debug("Failed to compute recent synthesis S1 rate: %s", exc)
            recent_synthesis_s1_rate = 0.0

        if hypothesis is None:
            intent_spec = self._refinement_intent_spec(refine_config.refine_intent)
            source_rule = (
                f"source_selection_rule=result_ids({len(ids)}) with "
                f"stage1_survivor_sources={source_stage1_passed}/{len(ids)}"
            )
            mutation_plan = (
                "mutation_mechanism=evolution_local_neighborhood("
                f"operators=op_replace|config_tweak|edge_rewire, mutation_rate={refine_config.mutation_rate:.2f}, "
                f"mutations_per_source={refine_config.refine_mutations_per_source}, "
                f"pool_multiplier={max(1, int(refine_config.refine_pool_multiplier or 1))})"
            )
            baseline_s1 = f"recent_synthesis_s1_rate={recent_synthesis_s1_rate:.3f}"
            success_criteria = (
                "success_criteria=(stage0_pass_rate>=0.95 AND stage05_pass_rate>=0.70) "
                "AND (delta_s1_rate>=+0.03_vs_recent OR best_loss_ratio<=0.98*parent_loss_ratio)"
            )
            fallback_plan = (
                "fallback_plan=if(no_stage1_improvement OR no_stage1_sources) "
                "queue_ablation_suite_and_novelty_mode"
            )
            recommendation_clause = ""
            if recommendation:
                recommendation_clause = (
                    " recommended_intent="
                    f"{recommendation.get('intent')}"
                    f" rationale={recommendation.get('rationale')}"
                    f" evidence={recommendation.get('evidence')}"
                    ";"
                )
            hypothesis = (
                "Fingerprint refinement hypothesis: "
                f"{source_rule}; "
                f"{mutation_plan}; "
                f"intent={intent_spec['name']} weights={intent_spec['weights']} "
                f"score={intent_spec['formula']}; "
                f"{recommendation_clause} "
                f"{baseline_s1}; "
                f"{success_criteria}; "
                f"{fallback_plan}."
            )

        return self.start_experiment(refine_config, hypothesis=hypothesis)

    def _recommend_refinement_intent(
        self,
        nb: LabNotebook,
        source_rows: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Recommend refinement intent from historical quality/novelty/compression evidence."""
        if not source_rows:
            return "balanced", {
                "intent": "balanced",
                "rationale": "no_source_rows",
                "evidence": {"source_count": 0},
            }

        op_success = self._op_success_lookup(nb)
        sparse_hint_ops = (
            "sparse",
            "gate",
            "topk",
            "mask",
            "threshold",
            "skip",
            "mixture",
        )

        loss_values: List[float] = []
        novelty_values: List[float] = []
        param_values: List[float] = []
        op_success_values: List[float] = []
        sparse_ratios: List[float] = []

        for row in source_rows:
            loss = row.get("loss_ratio")
            novelty = row.get("novelty_score")
            params = row.get("param_count") or row.get("graph_n_params_estimate")

            if isinstance(loss, (int, float)):
                loss_values.append(float(loss))
            if isinstance(novelty, (int, float)):
                novelty_values.append(float(novelty))
            if isinstance(params, (int, float)) and float(params) > 0:
                param_values.append(float(params))

            ops: List[str] = []
            graph_json = row.get("graph_json")
            if isinstance(graph_json, str) and graph_json.strip():
                try:
                    graph_data = json.loads(graph_json)
                    nodes = (
                        graph_data.get("nodes", {})
                        if isinstance(graph_data, dict)
                        else {}
                    )
                    for nd in nodes.values():
                        if not isinstance(nd, dict):
                            continue
                        op_name = str(nd.get("op_name") or "").strip().lower()
                        if not op_name or op_name == "input":
                            continue
                        ops.append(op_name)
                except Exception as exc:
                    logger.debug("Falling back to default: %s", exc)
                    ops = []

            if ops:
                scores = [float(op_success.get(op, 0.5)) for op in ops]
                op_success_values.append(sum(scores) / len(scores))
                sparse_ratio = sum(
                    1.0 for op in ops if any(token in op for token in sparse_hint_ops)
                ) / len(ops)
                sparse_ratios.append(float(sparse_ratio))

        mean_loss = (sum(loss_values) / len(loss_values)) if loss_values else None
        mean_novelty = (
            (sum(novelty_values) / len(novelty_values)) if novelty_values else None
        )
        mean_params = (sum(param_values) / len(param_values)) if param_values else None
        mean_op_success = (
            (sum(op_success_values) / len(op_success_values))
            if op_success_values
            else None
        )
        mean_sparse_ratio = (
            (sum(sparse_ratios) / len(sparse_ratios)) if sparse_ratios else None
        )

        intent = "balanced"
        rationale = "mixed_signals"
        if (mean_loss is not None and mean_loss >= 0.75) or (
            mean_op_success is not None and mean_op_success < 0.35
        ):
            intent = "quality"
            rationale = "weak_quality_signal"
        elif (
            mean_params is not None
            and mean_params >= 500_000
            and (mean_loss is None or mean_loss <= 0.80)
        ):
            intent = "compression"
            rationale = "high_parameter_budget"
        elif mean_novelty is not None and mean_novelty < 0.45:
            intent = "novelty"
            rationale = "low_novelty_signal"
        elif (
            mean_sparse_ratio is not None
            and mean_sparse_ratio < 0.10
            and mean_params is not None
            and mean_params >= 1_000_000
        ):
            intent = "sparsity"
            rationale = "sparse_operator_gap"
        elif (
            mean_params is not None
            and mean_params > 0
            and mean_loss is not None
            and mean_loss < 0.60
        ):
            # Good quality but check FLOP efficiency
            baseline_params = 6 * 256**2  # ~393K for a minimal 2-layer transformer
            if mean_params > 3 * baseline_params:
                intent = "compression"
                rationale = "low_flop_efficiency"

        recommendation = {
            "intent": intent,
            "rationale": rationale,
            "evidence": {
                "source_count": len(source_rows),
                "mean_loss_ratio": mean_loss,
                "mean_novelty": mean_novelty,
                "mean_params": mean_params,
                "mean_op_success": mean_op_success,
                "mean_sparse_op_ratio": mean_sparse_ratio,
            },
        }
        return intent, recommendation

    def start_investigation(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        force: bool = False,
    ) -> str:
        """Start investigation phase for selected candidates.

        Args:
            force: Skip tier and already-investigated guards.  Allows
                   re-investigating candidates with different config
                   (e.g. longer steps, different data mode).
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        if not force:
            # Tier guard: reject result IDs already at investigation tier or beyond
            tiers = nb.get_tiers_for_result_ids(result_ids)
            already_done = {
                rid: tier
                for rid, tier in tiers.items()
                if tier in ("investigation", "validation", "breakthrough")
            }
            if already_done:
                nb.close()
                labels = ", ".join(
                    f"{rid} ({tier})" for rid, tier in already_done.items()
                )
                raise ValueError(
                    f"Cannot investigate: {len(already_done)} candidate(s) already "
                    f"at or beyond investigation tier: {labels}"
                )
        else:
            logger.info(
                "Force re-investigation: skipping tier/fingerprint guards for %s",
                ", ".join(r[:8] for r in result_ids),
            )
            # Force re-investigation: don't clear existing leaderboard data.
            # The investigation results will only be updated if the new run
            # produces better results (enforced in _record_investigation_result).
            try:
                nb.conn.commit()
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Investigation: deep study of {len(result_ids)} screening survivors "
                f"with multiple training programs to test robustness."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_investigation",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting investigation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event(
            "investigation_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "result_ids": result_ids,
                "n_training_programs": config.n_training_programs,
            },
        )

        self._thread = threading.Thread(
            target=self._run_investigation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_validation(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        trigger: str = "manual",
        force: bool = False,
    ) -> str:
        """Start validation phase for investigation survivors."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        # Tier guards can be bypassed explicitly for manual override workflows.
        tiers = nb.get_tiers_for_result_ids(result_ids)
        if not force:
            already_validated = {
                rid: tier
                for rid, tier in tiers.items()
                if tier in ("validation", "breakthrough")
            }
            if already_validated:
                nb.close()
                labels = ", ".join(
                    f"{rid} ({tier})" for rid, tier in already_validated.items()
                )
                raise ValueError(
                    f"Cannot validate: {len(already_validated)} candidate(s) already "
                    f"at or beyond validation tier: {labels}"
                )
            # Warn if known-screening candidates haven't been investigated
            # (result_ids without leaderboard entries are allowed — they may
            # come from auto-escalation paths that create entries mid-flight)
            not_investigated = {
                rid for rid in result_ids if tiers.get(rid) == "screening"
            }
            if not_investigated:
                nb.close()
                raise ValueError(
                    f"Cannot validate: {len(not_investigated)} candidate(s) are still "
                    f"at screening tier (not investigated): {', '.join(not_investigated)}"
                )

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Validation: publication-grade testing of {len(result_ids)} "
                f"investigation survivors with multi-seed evaluation."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="validation",
            config=self._validation_config_with_result_ids(config, result_ids, trigger),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_validation",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting validation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event(
            "validation_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "result_ids": result_ids,
            },
        )

        self._thread = threading.Thread(
            target=self._run_validation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_scale_up(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
    ) -> str:
        """Start scale-up validation of specific programs in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Scale-up validation: testing whether {len(result_ids)} "
                f"top performer(s) maintain their advantage at 10x training scale."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="scale_up",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_scale_up",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting scale-up validation of {len(result_ids)} program(s)...",
            )

        self._emit_event(
            "scale_up_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "result_ids": result_ids,
                "config": {
                    "steps": config.scale_up_steps,
                    "batch_size": config.scale_up_batch_size,
                    "seq_len": config.scale_up_seq_len,
                },
            },
        )

        self._thread = threading.Thread(
            target=self._run_scale_up_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_evolution(
        self,
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
    ) -> str:
        """Start evolutionary search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_evolution",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting evolutionary search...",
            )

        self._emit_event(
            "evolution_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "config": config.to_dict(),
            },
        )

        self._thread = threading.Thread(
            target=self._run_evolution_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_novelty_search(
        self,
        config: RunConfig,
        hypothesis: Optional[str] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
    ) -> str:
        """Start novelty search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_novelty_search",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting novelty search...",
            )

        self._emit_event(
            "novelty_started",
            {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "config": config.to_dict(),
            },
        )

        self._thread = threading.Thread(
            target=self._run_novelty_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_resume(
        self, experiment_id: str, config: Optional[RunConfig] = None
    ) -> str:
        """Resume an interrupted experiment from its last checkpoint.

        Looks up the experiment in the notebook, reconstructs config if needed,
        and dispatches to the appropriate thread based on experiment type.
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        exp_data = nb.get_resumable_experiment(experiment_id)
        if exp_data is None:
            nb.close()
            raise ValueError(
                f"Experiment {experiment_id} not found or not resumable "
                "(must be 'running' or 'failed')"
            )

        exp_type = exp_data["experiment_type"]
        exp_data.get("hypothesis", "")

        # Reconstruct config from stored config_json
        if config is None:
            try:
                config_dict = json.loads(exp_data["config_json"])
                config = RunConfig.from_dict(config_dict)
            except (json.JSONDecodeError, TypeError, ValueError):
                nb.close()
                raise ValueError(
                    f"Cannot reconstruct config for experiment {experiment_id}"
                )

        config.resume_experiment_id = experiment_id

        # Mark experiment as running again if it was failed
        if exp_data["status"] == "failed":
            nb.conn.execute(
                "UPDATE experiments SET status = 'running' WHERE experiment_id = ?",
                (experiment_id,),
            )
            nb.conn.commit()
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=experiment_id,
                status="resuming",
                aria_message=f"Resuming {exp_type} experiment {experiment_id}...",
            )

        self._emit_event(
            "experiment_resuming",
            {
                "experiment_id": experiment_id,
                "experiment_type": exp_type,
            },
        )

        if exp_type == "continuous" or config.continuous:
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )
        else:
            logger.warning(
                "Resume for experiment type '%s' not yet supported, "
                "falling back to continuous",
                exp_type,
            )
            config.continuous = True
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )

        self._thread.start()
        return experiment_id
