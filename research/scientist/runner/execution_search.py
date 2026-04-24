"""Execution mixin: evolution + novelty search threads."""

from __future__ import annotations

import sqlite3
import time
import traceback

from ..native_runner import compile_model_native_first as compile_model
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ..notebook import ExperimentEntry
from ..shared_utils import resolve_device
from ._helpers import clear_gpu_memory
from ._lifecycle import _LifecycleMixin

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

            def novelty_fn(graph, all_graphs):
                """Structural novelty relative to current population."""
                nov = novelty_score(graph)
                # Penalize duplicates within population
                my_fp = graph.fingerprint()
                dup_count = sum(1 for g in all_graphs if g.fingerprint() == my_fp) - 1
                penalty = max(0, 1 - dup_count * 0.3)
                return nov.structural_novelty * penalty

            population = evolutionary_search(
                fitness_fn=fitness_fn,
                novelty_fn=novelty_fn,
                config=evo_config,
                callback=gen_callback,
            )

            results = {
                "total": eval_counters["total"],
                "stage0_passed": eval_counters["s0"],
                "stage05_passed": eval_counters["s0"],
                "stage1_passed": eval_counters["s1"],
                "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
                "best_loss_ratio": 1.0
                - max((ind.fitness for ind in population), default=0),
                "best_novelty_score": max(
                    (ind.novelty for ind in population), default=0
                ),
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

            # Rich context for Aria
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

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

            self._publish_terminal_event(
                producer="runner.execution_search",
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "aria_summary": summary,
                    "aria_mood": self.aria.state.mood,
                    "insights": insights,
                    "llm_analysis": llm_analysis,
                    "mode": "evolution",
                },
            )
            self._complete_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                insights=insights,
                llm_analysis=llm_analysis,
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
                        stage_tag="evolution_combined_fitness",
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
                        if _debug:
                            logger.info(
                                "DEBUG fingerprint: %s quality=%s probes_ok=%d locality=%s isotropy=%s",
                                gfp[:16],
                                bfp.quality,
                                bfp.analyses_succeeded,
                                bfp.interaction_locality,
                                bfp.isotropy,
                            )
                    except (RuntimeError, ValueError, TypeError) as e:
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
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Fitness computation failed: %s", e)
                    fitness = 0.0

                fitness_cache[gfp] = fitness
                on_evaluate(graph, fitness, sandbox_result, s1_result)
                return fitness

            def fingerprint_fn(graph):
                return fingerprint_cache.get(graph.fingerprint())

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
                if ind.fitness > 0.2:
                    results["survivors"].append(
                        {
                            "fingerprint": ind.fingerprint,
                            "novelty": ind.novelty,
                            "loss_ratio": 1.0 - ind.fitness,
                        }
                    )

            if results["survivors"]:
                results["best_loss_ratio"] = min(
                    s["loss_ratio"] for s in results["survivors"]
                )
                results["best_novelty_score"] = max(
                    s["novelty"] for s in results["survivors"]
                )

            nb.update_op_success_rates(exp_id)
            nb.update_failure_signatures(exp_id)

            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

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

            self._publish_terminal_event(
                producer="runner.execution_search",
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "aria_summary": summary,
                    "aria_mood": self.aria.state.mood,
                    "insights": insights,
                    "llm_analysis": llm_analysis,
                    "mode": "novelty",
                },
            )
            self._complete_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                insights=insights,
                llm_analysis=llm_analysis,
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
