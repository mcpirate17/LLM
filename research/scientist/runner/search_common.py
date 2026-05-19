"""Shared mechanics for evolution and novelty search runners."""

from __future__ import annotations

import logging
import time
from typing import Any

from ...eval.fingerprint import compute_fingerprint
from ...eval.metrics import novelty_score
from ..native_runner import compile_model_native_first as compile_model
from ._helpers import clear_gpu_memory

logger = logging.getLogger(__name__)


def make_program_evaluation_callback(
    runner: Any,
    *,
    eval_counters: dict[str, int],
    nb: Any,
    exp_id: str,
    model_source: str,
    fingerprint_cache: dict[str, Any] | None = None,
    debug: bool = False,
):
    """Build the callback passed into search fitness functions."""

    def on_evaluate(graph, fitness, sandbox_result, s1_result):
        behavioral_fingerprint = (
            fingerprint_cache.get(graph.fingerprint()) if fingerprint_cache else None
        )
        runner._on_program_evaluated(
            graph,
            fitness,
            sandbox_result,
            s1_result,
            eval_counters,
            nb,
            exp_id,
            model_source=model_source,
            behavioral_fingerprint=behavioral_fingerprint,
            debug=debug,
        )

    return on_evaluate


def structural_population_novelty(graph: Any, all_graphs: list[Any]) -> float:
    """Structural novelty with duplicate penalty inside the current population."""
    novelty = novelty_score(graph)
    fingerprint = graph.fingerprint()
    dup_count = sum(1 for item in all_graphs if item.fingerprint() == fingerprint) - 1
    penalty = max(0, 1 - dup_count * 0.3)
    return novelty.structural_novelty * penalty


def summarize_evolution_population(
    population: list[Any],
    eval_counters: dict[str, int],
) -> dict[str, Any]:
    results = {
        "total": eval_counters["total"],
        "stage0_passed": eval_counters["s0"],
        "stage05_passed": eval_counters["s0"],
        "stage1_passed": eval_counters["s1"],
        "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
        "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
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
    return results


def summarize_novelty_result(
    ns_result: Any,
    eval_counters: dict[str, int],
    *,
    best_from_all_individuals: bool,
) -> dict[str, Any]:
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
        loss_ratio = 1.0 - ind.fitness if ind.fitness > 0 else None
        if best_from_all_individuals:
            if loss_ratio is not None and (
                results["best_loss_ratio"] is None
                or loss_ratio < results["best_loss_ratio"]
            ):
                results["best_loss_ratio"] = loss_ratio
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

    if not best_from_all_individuals and results["survivors"]:
        results["best_loss_ratio"] = min(
            survivor["loss_ratio"] for survivor in results["survivors"]
        )
        results["best_novelty_score"] = max(
            survivor["novelty"] for survivor in results["survivors"]
        )
    return results


def analyze_search_results(
    runner: Any,
    *,
    exp_id: str,
    results: dict[str, Any],
    config: Any,
    hypothesis: str,
    nb: Any,
) -> tuple[Any, Any, Any, Any]:
    nb.update_op_success_rates(exp_id)
    nb.update_failure_signatures(exp_id)
    context = runner._build_rich_context_for_experiment(results, config, hypothesis, nb)
    summary = runner.aria.experiment_summary(results, context=context)
    llm_analysis = runner.aria.analyze_results(results, context=context)
    insights = runner._analyze_results(results, exp_id, nb, context=context)
    return context, summary, llm_analysis, insights


def publish_search_completion(
    runner: Any,
    *,
    nb: Any,
    exp_id: str,
    results: dict[str, Any],
    summary: Any,
    insights: Any,
    llm_analysis: Any,
    producer: str,
    mode: str,
) -> None:
    runner._publish_terminal_event(
        producer=producer,
        event_type="experiment_completed",
        exp_id=exp_id,
        payload={
            "completed_at": time.time(),
            "results": results,
            "aria_summary": summary,
            "aria_mood": runner.aria.state.mood,
            "insights": insights,
            "llm_analysis": llm_analysis,
            "mode": mode,
        },
    )
    runner._complete_experiment_compat(
        nb=nb,
        experiment_id=exp_id,
        results=results,
        aria_summary=summary,
        insights=insights,
        llm_analysis=llm_analysis,
    )


def make_combined_novelty_fitness_fn(
    runner: Any,
    *,
    config: Any,
    device: Any,
    device_str: str,
    fitness_cache: dict[str, float],
    fingerprint_cache: dict[str, Any],
    on_evaluate: Any,
    stage_tag: str,
    debug: bool,
    log_successful_fingerprint: bool = False,
):
    """Compile once, run sandbox, behavioral fingerprint, and micro-train."""

    def combined_fitness_fn(graph):
        graph_fingerprint = graph.fingerprint()
        if graph_fingerprint in fitness_cache:
            return fitness_cache[graph_fingerprint]

        sandbox_result = None
        s1_result = None
        try:
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            sandbox_result = runner._safe_eval_for_stage(
                model,
                stage_tag=stage_tag,
                batch_size=2,
                seq_len=min(128, config.max_seq_len),
                vocab_size=config.vocab_size,
                device=device_str,
            )
            if not sandbox_result.passed:
                del model
                fitness = 0.0
                fitness_cache[graph_fingerprint] = fitness
                on_evaluate(graph, fitness, sandbox_result, s1_result)
                return fitness

            try:
                behavioral_fingerprint = compute_fingerprint(
                    model,
                    seq_len=min(64, config.max_seq_len),
                    model_dim=config.model_dim,
                    vocab_size=config.vocab_size,
                    device=device_str,
                    include_cka=False,
                    include_behavioral_probes=True,
                )
                fingerprint_cache[graph_fingerprint] = behavioral_fingerprint
                if debug and log_successful_fingerprint:
                    logger.info(
                        "DEBUG fingerprint: %s quality=%s probes_ok=%d locality=%s isotropy=%s",
                        graph_fingerprint[:16],
                        behavioral_fingerprint.quality,
                        behavioral_fingerprint.analyses_succeeded,
                        behavioral_fingerprint.interaction_locality,
                        behavioral_fingerprint.isotropy,
                    )
            except (RuntimeError, ValueError, TypeError) as exc:
                if debug:
                    logger.exception(
                        "DEBUG: Fingerprint computation failed for %s",
                        graph_fingerprint[:16],
                    )
                else:
                    logger.debug("Fingerprint computation failed: %s", exc)

            s1_result = runner._micro_train(
                model,
                config,
                device,
                seed=runner._stable_seed("fitness", graph_fingerprint),
            )
            del model
            clear_gpu_memory()

            if s1_result.get("passed"):
                fitness, _components = runner._compute_multi_objective_fitness(
                    s1_result, sandbox_result, graph, config
                )
            else:
                fitness = 0.1
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Fitness computation failed: %s", exc)
            fitness = 0.0

        fitness_cache[graph_fingerprint] = fitness
        on_evaluate(graph, fitness, sandbox_result, s1_result)
        return fitness

    return combined_fitness_fn


def make_cached_fingerprint_fn(fingerprint_cache: dict[str, Any]):
    def fingerprint_fn(graph):
        return fingerprint_cache.get(graph.fingerprint())

    return fingerprint_fn
