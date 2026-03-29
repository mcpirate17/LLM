"""Continuous mode execution methods (synthesis, evolution, novelty, refinement), split from continuous.py."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


from ..native_runner import compile_model_native_first as compile_model
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ..notebook import LabNotebook, ExperimentEntry
from ..llm.context_experiment import (
    build_mode_selection_context,
)
from ..llm.context_hypothesis import build_hypothesis_context
from ..shared_utils import resolve_device
from ._helpers import clear_gpu_memory

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress


class _ContinuousModesMixin:
    """Synthesis, evolution, novelty, refinement mode runners."""

    __slots__ = ()

    def _run_continuous_synthesis(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run a single synthesis experiment within continuous mode."""
        is_control = self._is_control_experiment(config, n_experiments)

        # Build context so Aria's hypothesis is informed by recent results
        recent = nb.get_recent_experiments(5)
        leaderboard = nb.get_leaderboard(limit=20)
        context = build_mode_selection_context(
            recent_experiments=recent,
            leaderboard=leaderboard,
            current_mode="synthesis",
            n_experiments_in_session=n_experiments,
        )
        if config.max_cost_dollars > 0:
            context += (
                f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                f"of ${config.max_cost_dollars:.2f}"
            )

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

        # Structured hypothesis (campaign-aware)
        structured_hyp = None
        hypothesis_id = None
        if config.enable_campaigns:
            try:
                knowledge = nb.get_knowledge()
                recent_hyps = []
                if self._active_campaign_id:
                    recent_hyps = nb.get_campaign_hypotheses(self._active_campaign_id)[
                        -5:
                    ]
                hyp_context = build_hypothesis_context(
                    campaign=nb.get_campaign(self._active_campaign_id)
                    if self._active_campaign_id
                    else None,
                    recent_hypotheses=recent_hyps,
                    knowledge=knowledge,
                    leaderboard=leaderboard,
                    recent_experiments=recent,
                )
                structured_hyp = self.aria.formulate_structured_hypothesis(
                    context=hyp_context
                )
                hypothesis = structured_hyp["prediction"]

                # Record structured hypothesis
                # Find parent: last unresolved hypothesis in chain
                parent_id = None
                nb.get_unresolved_hypotheses(self._active_campaign_id)
                # Also check if previous hypothesis suggested a follow-up
                if (
                    hasattr(self, "_next_follow_up_parent")
                    and self._next_follow_up_parent
                ):
                    parent_id = self._next_follow_up_parent
                    self._next_follow_up_parent = None

                hypothesis_id = nb.record_hypothesis(
                    campaign_id=self._active_campaign_id,
                    prediction=structured_hyp["prediction"],
                    reasoning=structured_hyp["reasoning"],
                    test_method=structured_hyp["test_method"],
                    success_metric=structured_hyp["success_metric"],
                    parent_id=parent_id,
                    confidence=structured_hyp["confidence"],
                    metadata={
                        "source": "structured_hypothesis",
                        "llm_used": True,
                        "fallback_used": False,
                        "used_context": True,
                        "review_status": "not_reviewed",
                        "confidence": structured_hyp.get("confidence"),
                        "critique": structured_hyp.get("critique"),
                    },
                )
                self._current_hypothesis_id = hypothesis_id

                self._emit_event(
                    "hypothesis_recorded",
                    {
                        "hypothesis_id": hypothesis_id,
                        "prediction": structured_hyp["prediction"],
                        "confidence": structured_hyp["confidence"],
                        "campaign_id": self._active_campaign_id,
                    },
                )
            except Exception as e:
                logger.debug(f"Structured hypothesis failed, using basic: {e}")
                structured_hyp = None

        if structured_hyp is None:
            result = self.aria.formulate_hypothesis(
                context=context,
                return_metadata=True,
            )
            if isinstance(result, tuple):
                hypothesis, basic_hyp_meta = result
            else:
                hypothesis = result
                basic_hyp_meta = {
                    "source": "rule_based_fallback",
                    "llm_used": False,
                    "fallback_used": True,
                    "used_context": True,
                    "review_status": "not_reviewed",
                    "confidence": None,
                    "critique": None,
                }
            hypothesis_metadata = {
                **basic_hyp_meta,
                "context_char_count": len(context),
            }
        else:
            hypothesis_metadata = {
                "source": "structured_hypothesis",
                "llm_used": True,
                "fallback_used": False,
                "used_context": True,
                "review_status": "not_reviewed",
                "confidence": structured_hyp.get("confidence"),
                "critique": structured_hyp.get("critique"),
                "hypothesis_id": hypothesis_id,
            }

        exp_config = config.to_dict()
        if is_control:
            exp_config["control_experiment"] = True
            exp_config["use_learned_grammar_weights"] = False

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=exp_config,
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            created_by="continuous_synthesis",
        )
        logger.info(
            "Experiment %s started (synthesis, %d programs) — hypothesis: %s",
            exp_id[:8],
            config.n_programs,
            (hypothesis or "none")[:150],
        )

        if is_control:
            nb.log_learning_event(
                "grammar_control_experiment",
                f"Experiment {exp_id} is a control run using default grammar weights",
                evidence=f"interval={config.control_experiment_interval}, experiment_number={n_experiments}",
            )

        # Link experiment to campaign
        if config.enable_campaigns and self._active_campaign_id:
            try:
                nb.conn.execute(
                    "UPDATE experiments SET campaign_id = ? WHERE experiment_id = ?",
                    (self._active_campaign_id, exp_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Campaign linking failed for %s: %s", exp_id, e)

        # Link hypothesis to experiment
        if hypothesis_id:
            try:
                nb.conn.execute(
                    "UPDATE hypotheses SET experiment_id = ?, status = 'testing' "
                    "WHERE hypothesis_id = ?",
                    (exp_id, hypothesis_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Hypothesis linking failed for %s: %s", exp_id, e)

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|synthesis] {hypothesis}",
            )

        self._emit_event(
            "experiment_started",
            {
                "experiment_id": exp_id,
                "experiment_number": n_experiments,
                "hypothesis": hypothesis,
                "mode": "synthesis",
                "is_control_experiment": is_control,
            },
        )

        # Diversify grammar config based on experiment number
        synth_config = self._diversify_grammar_config(config, n_experiments)

        results = self._execute_experiment(
            exp_id,
            synth_config,
            nb,
            use_learned_grammar=not is_control,
        )
        self._persist_applied_grammar_weights(nb, exp_id, results)

        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)

        # Structured hypothesis validation
        if structured_hyp and hypothesis_id:
            try:
                validation = self.aria.validate_structured_hypothesis(
                    structured_hyp, results, context=context
                )
                nb.resolve_hypothesis(
                    hypothesis_id=hypothesis_id,
                    status=validation["status"],
                    evidence=validation["evidence"],
                    summary=validation["explanation"],
                    confidence_after=validation["confidence_after"],
                )
                nb.add_entry(
                    ExperimentEntry(
                        entry_type="analysis",
                        title=f"Hypothesis {validation['status'].upper()}",
                        content=validation["explanation"],
                        experiment_id=exp_id,
                        metadata={
                            "hypothesis_id": hypothesis_id,
                            "status": validation["status"],
                            "confidence_after": validation["confidence_after"],
                        },
                    )
                )
                self._emit_event(
                    "hypothesis_resolved",
                    {
                        "hypothesis_id": hypothesis_id,
                        "status": validation["status"],
                        "evidence": validation["evidence"][:200],
                        "confidence_after": validation["confidence_after"],
                    },
                )
                # If follow-up suggested, queue it for next experiment
                if validation.get("follow_up"):
                    self._next_follow_up_parent = hypothesis_id
            except Exception as e:
                logger.debug(f"Structured validation failed: {e}")
        else:
            # Fallback to old-style validation
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="analysis",
                            title="Hypothesis Validation",
                            content=validation.get("explanation", ""),
                            experiment_id=exp_id,
                            metadata={"validated": validation.get("validated", False)},
                        )
                    )
            except Exception as e:
                logger.warning("Hypothesis validation logging failed: %s", e)

        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=summary,
            aria_mood=self.aria.state.mood,
            insights=insights,
            llm_analysis=llm_analysis,
        )
        if summary:
            logger.info("Aria summary: %s", summary[:200])
        nb.update_op_success_rates(exp_id)
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)
        self._auto_recommend(results, config, hypothesis, nb)

        if (
            config.auto_report
            and config.auto_report_every_n > 0
            and n_experiments % config.auto_report_every_n == 0
        ):
            self._maybe_auto_report(
                config,
                nb,
                reason=f"periodic (every {config.auto_report_every_n}, "
                f"after exp #{n_experiments})",
            )

        # Knowledge extraction
        self._maybe_extract_knowledge(config, nb, n_experiments)

        # Flush async writes so auto-escalate can read back S1 survivors
        nb.flush_writes()

        # Auto-escalation: promote S1 survivors to leaderboard and
        # queue investigation/validation if criteria met
        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event(
            "experiment_completed",
            {
                "experiment_id": exp_id,
                "results": results,
                "mode": "synthesis",
            },
        )

    def _run_continuous_evolution(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run evolution search within continuous mode (inline, not threaded)."""
        from ...search.evolution import evolutionary_search, EvolutionConfig

        hypothesis = f"Evolution search: {mode_reasoning}"
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            created_by="continuous_evolution",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="evolving",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|evolution] {hypothesis[:80]}",
            )

        self._emit_event(
            "experiment_started",
            {
                "experiment_id": exp_id,
                "experiment_number": n_experiments,
                "hypothesis": hypothesis,
                "mode": "evolution",
            },
        )

        # Cap depth/ops for evolution to prevent recursion overflow
        evo_config = EvolutionConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=self._build_grammar_config(config),
            exploit_prob=config.exploit_prob,
            local_mutation_prob=config.local_mutation_prob,
            exploit_top_k=config.exploit_top_k,
        )

        fitness_cache: dict = {}
        eval_counters = {"total": 0, "s0": 0, "s1": 0}

        def on_evaluate(graph, fitness, sandbox_result, s1_result):
            self._on_program_evaluated(
                graph,
                fitness,
                sandbox_result,
                s1_result,
                eval_counters,
                nb,
                exp_id,
                model_source="evolution",
            )

        fitness_fn = self._make_fitness_fn(
            config, on_evaluate=on_evaluate, fitness_cache=fitness_cache
        )

        def novelty_fn(graph, all_graphs):
            nov = novelty_score(graph)
            my_fp = graph.fingerprint()
            dup_count = sum(1 for g in all_graphs if g.fingerprint() == my_fp) - 1
            penalty = max(0, 1 - dup_count * 0.3)
            return nov.structural_novelty * penalty

        population = evolutionary_search(
            fitness_fn=fitness_fn,
            novelty_fn=novelty_fn,
            config=evo_config,
            stop_check=self._stop_event.is_set,
        )

        results = {
            "total": eval_counters["total"],
            "stage0_passed": eval_counters["s0"],
            "stage05_passed": eval_counters["s0"],
            "stage1_passed": eval_counters["s1"],
            "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
            "best_loss_ratio": 1.0
            - max((ind.fitness for ind in population), default=0),
            "best_novelty_score": max((ind.novelty for ind in population), default=0),
            "survivors": [],
        }

        for ind in population[:20]:
            if ind.fitness > 0.2:
                results["survivors"].append(
                    {
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    }
                )

        nb.update_op_success_rates(exp_id)
        nb.update_failure_signatures(exp_id)
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=summary,
            aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        nb.flush_writes()
        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event(
            "experiment_completed",
            {
                "experiment_id": exp_id,
                "results": results,
                "mode": "evolution",
            },
        )

    def _run_continuous_novelty(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run novelty search within continuous mode (inline, not threaded)."""
        from ...search.novelty_search import novelty_search, NoveltySearchConfig

        hypothesis = f"Novelty search: {mode_reasoning}"
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            exploratory=True,
            created_by="continuous_novelty",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="novelty_search",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|novelty] {hypothesis[:80]}",
            )

        self._emit_event(
            "experiment_started",
            {
                "experiment_id": exp_id,
                "experiment_number": n_experiments,
                "hypothesis": hypothesis,
                "mode": "novelty",
            },
        )

        # Note: depth/ops caps removed — user config is authoritative.
        # The grammar and validator enforce structural limits.

        grammar = self._build_grammar_config(config)
        ns_config = NoveltySearchConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=grammar,
            exploit_prob=config.exploit_prob,
            local_mutation_prob=config.local_mutation_prob,
            debug=config.debug,
        )
        dev = resolve_device(config.device)
        dev_str = str(dev)

        fitness_cache: dict = {}
        fingerprint_cache: dict = {}
        eval_counters = {"total": 0, "s0": 0, "s1": 0}

        _debug = config.debug

        def on_evaluate(graph, fitness, sandbox_result, s1_result):
            bfp = fingerprint_cache.get(graph.fingerprint())
            self._on_program_evaluated(
                graph,
                fitness,
                sandbox_result,
                s1_result,
                eval_counters,
                nb,
                exp_id,
                model_source="novelty",
                behavioral_fingerprint=bfp,
                debug=_debug,
            )

        def combined_fitness_fn(graph):
            """Compile once, run sandbox + micro-train + fingerprint in one pass."""
            gfp = graph.fingerprint()

            if gfp in fitness_cache:
                return fitness_cache[gfp]

            sandbox_result = None
            s1_result = None
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="novelty_fitness",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if not sandbox_result.passed:
                    del model
                    fitness = 0.0
                    fitness_cache[gfp] = fitness
                    on_evaluate(graph, fitness, sandbox_result, s1_result)
                    return fitness

                # Compute fingerprint with behavioral probes for novelty archive;
                # CKA deferred to post-investigation.
                try:
                    bfp = compute_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=dev_str,
                        include_cka=False,
                        include_behavioral_probes=True,
                    )
                    fingerprint_cache[gfp] = bfp
                except Exception as e:
                    if _debug:
                        logger.exception(
                            "DEBUG: Fingerprint computation failed for %s", gfp[:16]
                        )
                    else:
                        logger.debug("Fingerprint computation failed: %s", e)

                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed("fitness", gfp),
                )
                del model
                clear_gpu_memory()

                if s1_result.get("passed"):
                    fitness, _components = self._compute_multi_objective_fitness(
                        s1_result, sandbox_result, graph, config
                    )
                else:
                    fitness = 0.1
            except Exception:
                fitness = 0.0

            fitness_cache[gfp] = fitness
            on_evaluate(graph, fitness, sandbox_result, s1_result)
            return fitness

        def fingerprint_fn(graph):
            return fingerprint_cache.get(graph.fingerprint())

        ns_result = novelty_search(
            fitness_fn=combined_fitness_fn,
            fingerprint_fn=fingerprint_fn,
            config=ns_config,
            stop_check=self._stop_event.is_set,
        )

        results = {
            "total": eval_counters["total"],
            "stage0_passed": eval_counters["s0"],
            "stage05_passed": eval_counters["s0"],
            "stage1_passed": eval_counters["s1"],
            "novel_count": sum(
                1 for ind in ns_result.best_individuals if ind.novelty > 0.5
            ),
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "archive_size": ns_result.archive_size,
        }

        for ind in ns_result.best_individuals[:20]:
            lr = 1.0 - ind.fitness if ind.fitness > 0 else None
            if lr is not None and (
                results["best_loss_ratio"] is None or lr < results["best_loss_ratio"]
            ):
                results["best_loss_ratio"] = lr
            if ind.novelty and (
                results["best_novelty_score"] is None
                or ind.novelty > results["best_novelty_score"]
            ):
                results["best_novelty_score"] = ind.novelty
            if ind.fitness > 0.2:
                results["survivors"].append(
                    {
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    }
                )

        nb.update_op_success_rates(exp_id)
        nb.update_failure_signatures(exp_id)
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=summary,
            aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        nb.flush_writes()
        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event(
            "experiment_completed",
            {
                "experiment_id": exp_id,
                "results": results,
                "mode": "novelty",
            },
        )

    def _run_continuous_refinement(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run recursive local winner-tweak refinement with plateau stopping."""
        plan = self._build_refinement_plan(nb, config)
        if not plan:
            logger.info(
                "Refinement requested but no eligible Stage-1 winners found; falling back to synthesis."
            )
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning
            )
            return

        source_ids = list(plan.get("source_result_ids", []))
        total_generations = max(
            1, int(plan.get("generations") or config.refinement_generations or 1)
        )
        budget_remaining = max(
            int(plan.get("budget_programs") or 0), int(config.n_programs)
        )
        plateau_patience = max(1, int(config.refinement_plateau_patience or 1))
        mutation_radius = max(
            0.05, min(1.0, float(config.refinement_mutation_radius or 0.35))
        )
        novelty_pressure = max(
            0.0, min(1.0, float(config.refinement_novelty_pressure or 0.35))
        )

        best_loss_seen: Optional[float] = None
        plateau_count = 0
        executed_generations = 0
        history: List[Dict[str, Any]] = []

        for generation in range(total_generations):
            if self._stop_event.is_set() or budget_remaining <= 0 or not source_ids:
                break

            gen_cfg = config.copy()
            gen_cfg.model_source = "fingerprint_refine"
            gen_cfg.refine_source_result_ids = ",".join(source_ids)
            gen_cfg.refine_mutations_per_source = max(
                1, int(round(2 + 4 * mutation_radius))
            )
            gen_cfg.refine_pool_multiplier = max(
                2, int(round(2 + 3 * novelty_pressure))
            )
            gen_cfg.mutation_rate = max(
                0.10, min(0.95, float(config.mutation_rate) * (0.5 + mutation_radius))
            )
            gen_cfg.n_programs = max(4, min(int(config.n_programs), budget_remaining))

            generation_reason = (
                f"{mode_reasoning} | recursive_refine gen {generation + 1}/{total_generations} "
                f"from {len(source_ids)} seed(s)"
            )
            self._run_continuous_synthesis(
                gen_cfg,
                nb,
                n_experiments,
                limit_str,
                generation_reason,
            )
            budget_remaining -= int(gen_cfg.n_programs)
            executed_generations += 1

            recent = nb.get_recent_experiments(1)
            if not recent:
                break
            current_exp_id = str(recent[0].get("experiment_id") or "")
            if not current_exp_id:
                break
            rows = nb.get_program_results(current_exp_id, limit=400)
            survivors = [row for row in rows if row.get("stage1_passed")]
            if not survivors:
                history.append(
                    {
                        "generation": generation + 1,
                        "experiment_id": current_exp_id,
                        "stage1_survivors": 0,
                        "best_loss_ratio": None,
                    }
                )
                break

            cur_best = min(
                (
                    float(r.get("loss_ratio"))
                    for r in survivors
                    if isinstance(r.get("loss_ratio"), (int, float))
                ),
                default=None,
            )
            history.append(
                {
                    "generation": generation + 1,
                    "experiment_id": current_exp_id,
                    "stage1_survivors": len(survivors),
                    "best_loss_ratio": cur_best,
                }
            )
            if cur_best is not None:
                if best_loss_seen is None or cur_best < best_loss_seen - 1e-4:
                    best_loss_seen = cur_best
                    plateau_count = 0
                else:
                    plateau_count += 1

            selected = self._select_diverse_refinement_sources(
                survivors,
                top_k=max(1, int(config.refinement_top_k or 1)),
                min_distance=max(0.01, float(config.refinement_min_distance or 0.01)),
                novelty_pressure=novelty_pressure,
            )
            source_ids = [
                str(r.get("result_id") or "") for r in selected if r.get("result_id")
            ]
            if plateau_count >= plateau_patience:
                break

        nb.record_decision(
            campaign_id=self._active_campaign_id,
            decision_type="recursive_refinement",
            subject=f"cycle_{n_experiments}",
            rationale=(
                f"Executed recursive local refinement for {executed_generations}/{total_generations} generation(s); "
                f"plateau_count={plateau_count}, budget_remaining={budget_remaining}."
            ),
            evidence_pack={
                "seed_count": int(plan.get("source_count") or 0),
                "initial_source_result_ids": plan.get("source_result_ids", []),
                "history": history,
                "plateau_patience": plateau_patience,
                "min_distance": float(config.refinement_min_distance),
                "novelty_pressure": novelty_pressure,
            },
        )

    # ── Pre-investigation gate ─────────────────────────────────────────

    def _get_reference_baseline_lr(self, nb: LabNotebook) -> Optional[float]:
        """Fetch best screening_loss_ratio from registered reference architectures."""
        try:
            refs = nb.get_references()
            if not refs:
                return None
            lrs = [
                float(r["screening_loss_ratio"])
                for r in refs
                if r.get("screening_loss_ratio") is not None
            ]
            return min(lrs) if lrs else None
        except Exception:
            return None

    def _run_continuous_phase(
        self,
        phase: str,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run investigation or validation phase inline within continuous mode."""
        leaderboard = nb.get_leaderboard(limit=50)

        if phase == "investigation":
            self._run_inline_investigation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning
            )
        elif phase == "validation":
            self._run_inline_validation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning
            )
