"""Execution mixin: evolution + novelty search threads."""

from __future__ import annotations

import sqlite3
import time
import traceback

from ..notebook import ExperimentEntry
from ..shared_utils import resolve_device
from ._lifecycle import _LifecycleMixin
from .search_common import (
    analyze_search_results,
    make_cached_fingerprint_fn,
    make_combined_novelty_fitness_fn,
    make_program_evaluation_callback,
    publish_search_completion,
    structural_population_novelty,
    summarize_evolution_population,
    summarize_novelty_result,
)

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ExecutionSearchMixin:
    """Evolution and novelty search execution."""

    __slots__ = ()
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _publish_search_terminal_event = _LifecycleMixin._publish_terminal_event
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _log_learning_event_compat = _LifecycleMixin._log_learning_event_compat

    # Ops considered "routing" for dashboard template stats
    _ROUTING_OPS = frozenset(
        {
            "token_entropy",
            "token_class_proj",
            "route_topk",
            "route_lanes",
            "route_recursion",
            "difficulty_blend_3way",
            "score_depth_blend",
            "confidence_token_gate",
            "learned_token_gate",
            "cheap_verify_blend",
            "depth_weighted_proj",
            "depth_token_mask",
            "token_merging",
            "adjacent_token_merge",
            "relu_gated_moe",
            "moe_topk",
            "moe_2expert",
            "signal_conditioned_compression",
            "mixture_of_experts",
            "moe",
            "conditional_compute",
            "routing_node",
            "difficulty_routed_block",
        }
    )

    def _run_evolution_thread(self, exp_id: str, config: RunConfig, hypothesis: str):
        """Execute evolutionary search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ...search.evolution import EvolutionConfig, evolutionary_search

            grammar = self._build_grammar_config(config)

            evo_config = EvolutionConfig(
                population_size=config.population_size,
                n_generations=config.n_generations,
                tournament_size=config.tournament_size,
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate,
                elitism=config.elitism,
                fitness_weight=config.fitness_weight,
                novelty_weight=config.novelty_weight,
                grammar_config=grammar,
                exploit_prob=config.exploit_prob,
                local_mutation_prob=config.local_mutation_prob,
                exploit_top_k=config.exploit_top_k,
            )

            fitness_cache: dict = {}
            eval_counters = {"total": 0, "s0": 0, "s1": 0}
            _evo_debug = config.debug
            on_evaluate = make_program_evaluation_callback(
                self,
                eval_counters=eval_counters,
                nb=nb,
                exp_id=exp_id,
                model_source="evolution",
                debug=_evo_debug,
            )

            fitness_fn = self._make_fitness_fn(
                config, on_evaluate=on_evaluate, fitness_cache=fitness_cache
            )

            def gen_callback(gen, population):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0

                # Template stats (Task 1I)
                n_routing = sum(
                    1
                    for ind in population
                    if any(
                        node.op_name in self._ROUTING_OPS
                        for node in ind.graph.nodes.values()
                        if not node.is_input
                    )
                )
                n_standard = len(population) - n_routing

                self._update_progress(
                    current_generation=gen + 1,
                    status="evaluating",
                    best_fitness=best_fit,
                    avg_fitness=avg_fit,
                    elapsed_seconds=time.time() - t_start,
                    aria_message=(
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"best={best_fit:.3f}, avg={avg_fit:.3f}, routing={n_routing}/{len(population)}"
                    ),
                )
                self._emit_event(
                    "evolution_generation",
                    {
                        "experiment_id": exp_id,
                        "generation": gen + 1,
                        "total_generations": config.n_generations,
                        "best_fitness": best_fit,
                        "avg_fitness": avg_fit,
                        "population_size": len(population),
                        "n_routing": n_routing,
                        "n_standard": n_standard,
                    },
                )
                try:
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="live_feed",
                            title=f"Evolution generation {gen + 1}/{config.n_generations}",
                            content=(
                                f"Gen {gen + 1}/{config.n_generations}: "
                                f"best={best_fit:.3f}, avg={avg_fit:.3f}, "
                                f"pop={len(population)}, routing={n_routing}/{len(population)}"
                            ),
                            experiment_id=exp_id,
                            metadata={
                                "live_feed_type": "evo_gen",
                                "payload": {
                                    "experiment_id": exp_id,
                                    "generation": gen + 1,
                                    "total_generations": config.n_generations,
                                    "best_fitness": best_fit,
                                    "avg_fitness": avg_fit,
                                    "population_size": len(population),
                                    "n_routing": n_routing,
                                    "n_standard": n_standard,
                                },
                            },
                        )
                    )
                except (sqlite3.OperationalError, RuntimeError) as e:
                    logger.debug(
                        "Failed to persist evolution generation feed entry: %s", e
                    )

            population = evolutionary_search(
                fitness_fn=fitness_fn,
                novelty_fn=structural_population_novelty,
                config=evo_config,
                callback=gen_callback,
            )

            results = summarize_evolution_population(population, eval_counters)

            context, summary, llm_analysis, insights = analyze_search_results(
                self,
                exp_id=exp_id,
                results=results,
                config=config,
                hypothesis=hypothesis,
                nb=nb,
            )

            # Validate hypothesis
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
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning("Hypothesis validation failed for %s: %s", exp_id, e)

            publish_search_completion(
                self,
                nb=nb,
                exp_id=exp_id,
                results=results,
                summary=summary,
                insights=insights,
                llm_analysis=llm_analysis,
                producer="runner.execution_search",
                mode="evolution",
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="evolution_complete")

            self._update_progress(
                status="completed",
                elapsed_seconds=time.time() - t_start,
                aria_message=summary.split("\n")[-1]
                if summary
                else "Evolution complete.",
            )

            self._emit_event(
                "evolution_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Evolution failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Evolution failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "evolution" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "evolution" -x --tb=short'
                    ],
                    trigger_payload={"mode": "evolution", "error": str(e)},
                )
            except (RuntimeError, OSError) as heal_err:
                logger.warning(
                    "code_healer failed during evolution error handling: %s",
                    heal_err,
                    exc_info=True,
                )
            self._publish_terminal_event(
                producer="runner.execution_search",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "evolution",
                },
            )
            self._fail_experiment_compat(nb=nb, experiment_id=exp_id, error=str(e))
            self._update_progress(
                status="failed",
                error=str(e),
                aria_message=self.aria.react_to_failure(str(e)),
            )
            self._emit_event(
                "experiment_failed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        except BaseException as e:
            logger.critical(
                "Evolution thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                self._publish_terminal_event(
                    producer="runner.execution_search",
                    event_type="experiment_failed",
                    exp_id=exp_id,
                    payload={
                        "completed_at": time.time(),
                        "error": f"FATAL: {e}",
                        "results": None,
                        "mode": "evolution",
                        "fatal": True,
                    },
                )
                self._fail_experiment_compat(
                    nb=nb,
                    experiment_id=exp_id,
                    error=f"FATAL: {e}",
                )
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": exp_id, "error": f"FATAL: {e}"},
                )
            except RuntimeError:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
        finally:
            nb.close()
            self._run_pending_scale_up()

    def _run_novelty_thread(self, exp_id: str, config: RunConfig, hypothesis: str):
        """Execute novelty search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ...search.novelty_search import NoveltySearchConfig, novelty_search

            grammar = self._build_grammar_config(config)

            ns_config = NoveltySearchConfig(
                archive_size=config.archive_size,
                k_nearest=config.k_nearest,
                archive_threshold=config.archive_threshold,
                novelty_weight=config.novelty_weight,
                fitness_weight=config.fitness_weight,
                population_size=config.population_size,
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
            on_evaluate = make_program_evaluation_callback(
                self,
                eval_counters=eval_counters,
                nb=nb,
                exp_id=exp_id,
                model_source="novelty",
                fingerprint_cache=fingerprint_cache,
                debug=_debug,
            )
            combined_fitness_fn = make_combined_novelty_fitness_fn(
                self,
                config=config,
                device=dev,
                device_str=dev_str,
                fitness_cache=fitness_cache,
                fingerprint_cache=fingerprint_cache,
                on_evaluate=on_evaluate,
                stage_tag="evolution_combined_fitness",
                debug=_debug,
                log_successful_fingerprint=True,
            )
            fingerprint_fn = make_cached_fingerprint_fn(fingerprint_cache)

            def gen_callback(gen, population, archive):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                novelties = [ind.novelty for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0

                # Template stats (Task 1I)
                n_routing = sum(
                    1
                    for ind in population
                    if any(
                        node.op_name in self._ROUTING_OPS
                        for node in ind.graph.nodes.values()
                        if not node.is_input
                    )
                )
                n_standard = len(population) - n_routing

                self._update_progress(
                    current_generation=gen + 1,
                    status="evaluating",
                    best_fitness=best_fit,
                    avg_fitness=avg_fit,
                    archive_size=archive.size(),
                    elapsed_seconds=time.time() - t_start,
                    aria_message=(
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"archive={archive.size()}, best_fit={best_fit:.3f}, routing={n_routing}/{len(population)}"
                    ),
                )
                self._emit_event(
                    "novelty_generation",
                    {
                        "experiment_id": exp_id,
                        "generation": gen + 1,
                        "total_generations": config.n_generations,
                        "best_fitness": best_fit,
                        "avg_fitness": avg_fit,
                        "archive_size": archive.size(),
                        "best_novelty": max(novelties) if novelties else 0,
                        "n_routing": n_routing,
                        "n_standard": n_standard,
                    },
                )
                try:
                    best_novelty = max(novelties) if novelties else 0
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="live_feed",
                            title=f"Novelty generation {gen + 1}/{config.n_generations}",
                            content=(
                                f"Gen {gen + 1}/{config.n_generations}: "
                                f"best_fit={best_fit:.3f}, archive={archive.size()}, "
                                f"novelty={best_novelty:.3f}, routing={n_routing}/{len(population)}"
                            ),
                            experiment_id=exp_id,
                            metadata={
                                "live_feed_type": "nov_gen",
                                "payload": {
                                    "experiment_id": exp_id,
                                    "generation": gen + 1,
                                    "total_generations": config.n_generations,
                                    "best_fitness": best_fit,
                                    "avg_fitness": avg_fit,
                                    "archive_size": archive.size(),
                                    "best_novelty": best_novelty,
                                    "n_routing": n_routing,
                                    "n_standard": n_standard,
                                },
                            },
                        )
                    )
                except (sqlite3.OperationalError, RuntimeError) as e:
                    logger.debug(
                        "Failed to persist novelty generation feed entry: %s", e
                    )

            ns_result = novelty_search(
                fitness_fn=combined_fitness_fn,
                fingerprint_fn=fingerprint_fn,
                config=ns_config,
                callback=gen_callback,
                stop_check=self._stop_event.is_set,
            )

            results = summarize_novelty_result(
                ns_result,
                eval_counters,
                best_from_all_individuals=False,
            )

            context, summary, llm_analysis, insights = analyze_search_results(
                self,
                exp_id=exp_id,
                results=results,
                config=config,
                hypothesis=hypothesis,
                nb=nb,
            )

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
            except (RuntimeError, ValueError, KeyError) as e:
                logger.debug("Hypothesis validation failed in novelty search: %s", e)

            publish_search_completion(
                self,
                nb=nb,
                exp_id=exp_id,
                results=results,
                summary=summary,
                insights=insights,
                llm_analysis=llm_analysis,
                producer="runner.execution_search",
                mode="novelty",
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="novelty_search_complete")

            self._update_progress(
                status="completed",
                elapsed_seconds=time.time() - t_start,
                aria_message=summary.split("\n")[-1]
                if summary
                else "Novelty search complete.",
            )

            self._emit_event(
                "novelty_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                    "archive_size": ns_result.archive_size,
                },
            )

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Novelty search failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Novelty search failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "novelty" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "novelty" -x --tb=short'
                    ],
                    trigger_payload={"mode": "novelty", "error": str(e)},
                )
            except (RuntimeError, OSError) as heal_err:
                logger.warning(
                    "code_healer failed during novelty error handling: %s",
                    heal_err,
                    exc_info=True,
                )
            self._publish_terminal_event(
                producer="runner.execution_search",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "novelty",
                },
            )
            self._fail_experiment_compat(nb=nb, experiment_id=exp_id, error=str(e))
            self._update_progress(
                status="failed",
                error=str(e),
                aria_message=self.aria.react_to_failure(str(e)),
            )
            self._emit_event(
                "experiment_failed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        except BaseException as e:
            logger.critical(
                "Novelty thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                self._publish_terminal_event(
                    producer="runner.execution_search",
                    event_type="experiment_failed",
                    exp_id=exp_id,
                    payload={
                        "completed_at": time.time(),
                        "error": f"FATAL: {e}",
                        "results": None,
                        "mode": "novelty",
                        "fatal": True,
                    },
                )
                self._fail_experiment_compat(
                    nb=nb,
                    experiment_id=exp_id,
                    error=f"FATAL: {e}",
                )
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": exp_id, "error": f"FATAL: {e}"},
                )
            except RuntimeError:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
        finally:
            nb.close()
            self._run_pending_scale_up()
