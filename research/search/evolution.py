"""
Evolutionary Search over Computation Graphs

Uses evolutionary algorithms to search the space of synthesized programs:
- Tournament selection
- Graph mutation (add/remove/swap ops)
- Graph crossover (splice subgraphs)
- Fitness-based and novelty-based selection
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import (
    GrammarConfig,
    generate_layer_graph,
)
from ..synthesis.generation_runtime import (
    build_generation_runtime_context,
    normalize_generation_config,
)
from ._mutation import (
    crossover_graphs as _mutation_crossover_graphs,
    local_mutate_graph as _local_mutate_graph,
    mutate_graph as _mutation_mutate_graph,
    spawn_crossover_individual as _spawn_crossover_individual,
    spawn_fresh_individual as _spawn_fresh_individual,
    spawn_mutation_individual as _spawn_mutation_individual,
)
from ._nsga import (
    PARETO_FRONT_RANK as _PARETO_FRONT_RANK,
    nsga2_rank,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Individual:
    """An individual in the evolutionary population."""

    graph: ComputationGraph
    fitness: float = 0.0
    novelty: float = 0.0
    generation: int = 0
    parent_fingerprint: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    pareto_rank: int = 0
    crowding_dist: float = 0.0
    _cached_fingerprint: Optional[str] = field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        if self._cached_fingerprint is None:
            self._cached_fingerprint = self.graph.fingerprint()
        return self._cached_fingerprint


@dataclass(slots=True)
class EvolutionConfig:
    """Configuration for evolutionary search."""

    population_size: int = 50
    n_generations: int = 20
    tournament_size: int = 5
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    fresh_injection_rate: float = 0.1
    elitism: int = 5  # top N carry over unchanged
    # Fitness weighting
    fitness_weight: float = 0.5
    novelty_weight: float = 0.5
    grammar_config: Optional[GrammarConfig] = None
    # Exploitation parameters
    local_mutation_prob: float = 0.3
    exploit_prob: float = 0.2
    exploit_top_k: int = 5


def _generate_context_valid_graph(
    grammar: GrammarConfig,
    rng: random.Random,
    max_attempts: int = 4,
    runtime_context=None,
) -> ComputationGraph:
    """Retry generation a few times under stricter context validation."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return generate_layer_graph(
                grammar,
                seed=rng.randint(0, 2**32),
                _runtime_context=runtime_context,
            )
        except (ValueError, RuntimeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("graph generation failed without an explicit error")


def evolutionary_search(
    fitness_fn: Callable[[ComputationGraph], float],
    novelty_fn: Optional[
        Callable[[ComputationGraph, List[ComputationGraph]], float]
    ] = None,
    config: Optional[EvolutionConfig] = None,
    seed: int = 42,
    callback: Optional[Callable[[int, List[Individual]], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    archive: Optional[object] = None,
) -> List[Individual]:
    """Run evolutionary search over computation graphs.

    Args:
        fitness_fn: Evaluates a graph, returns fitness score (higher = better)
        novelty_fn: Evaluates novelty of a graph relative to population
        config: Evolution configuration
        seed: Random seed
        callback: Called after each generation with (gen_num, population)
        stop_check: Optional callable that returns True to abort early

    Returns:
        Final population sorted by combined score
    """
    if config is None:
        config = EvolutionConfig()

    rng = random.Random(seed)
    grammar = normalize_generation_config(config.grammar_config or GrammarConfig())
    base_runtime_context = build_generation_runtime_context(grammar)

    def generate_context_valid_graph(
        target_grammar: GrammarConfig,
        target_rng: random.Random,
        max_attempts: int = 4,
    ) -> ComputationGraph:
        if target_grammar is grammar:
            return _generate_context_valid_graph(
                target_grammar,
                target_rng,
                max_attempts=max_attempts,
                runtime_context=base_runtime_context,
            )
        target_grammar = normalize_generation_config(target_grammar)
        return _generate_context_valid_graph(
            target_grammar,
            target_rng,
            max_attempts=max_attempts,
            runtime_context=build_generation_runtime_context(target_grammar),
        )

    # Initialize population
    population = []
    init_failures = 0
    for i in range(config.population_size):
        try:
            graph = generate_layer_graph(
                grammar,
                seed=seed + i * 137,
                _runtime_context=base_runtime_context,
            )
            ind = Individual(graph=graph, generation=0)
            population.append(ind)
        except (ValueError, RuntimeError) as e:
            init_failures += 1
            if init_failures <= 3:
                logger.debug("Initial population gen failed (%d): %s", init_failures, e)

    if not population:
        logger.error(
            "Evolution aborted: failed to generate any initial individuals "
            "(%d attempts all failed)",
            config.population_size,
        )
        return []

    if init_failures > 0:
        logger.info(
            "Initial population: %d/%d succeeded (%d failures)",
            len(population),
            config.population_size,
            init_failures,
        )

    # Evaluate initial population
    _evaluate_population(population, fitness_fn, novelty_fn)
    nsga2_rank(population)
    population = _enforce_population_diversity(
        population=population,
        fitness_fn=fitness_fn,
        novelty_fn=novelty_fn,
        config=config,
        grammar=grammar,
        rng=rng,
        generation=0,
        generate_context_valid_graph=generate_context_valid_graph,
    )

    # Evolve
    for gen in range(config.n_generations):
        if stop_check and stop_check():
            logger.info("Evolution stopped early at gen %d by external signal", gen)
            break

        new_population = []
        local_mutation_fitness_threshold = _fitness_top_k_threshold(
            population, config.exploit_top_k
        )

        # Elitism: keep top individuals
        population.sort(key=lambda x: _combined_score(x, config), reverse=True)
        new_population.extend(population[: config.elitism])

        # Fill rest with offspring (with max attempts to prevent infinite loop)
        max_fill_attempts = config.population_size * 10
        fill_attempts = 0
        fill_failures = 0
        while len(new_population) < config.population_size:
            if stop_check and stop_check():
                break
            fill_attempts += 1
            if fill_attempts > max_fill_attempts:
                logger.warning(
                    "Gen %d: hit max fill attempts (%d) with %d/%d individuals "
                    "(%d failures). Proceeding with smaller population.",
                    gen,
                    max_fill_attempts,
                    len(new_population),
                    config.population_size,
                    fill_failures,
                )
                break

            # Exploitation: with exploit_prob, select from archive exploit targets
            use_exploit = (
                archive is not None
                and hasattr(archive, "suggest_exploit_target")
                and rng.random() < config.exploit_prob
            )
            if use_exploit:
                try:
                    targets = archive.suggest_exploit_target(k=config.exploit_top_k)
                    if targets:
                        parent = rng.choice(targets)
                        child_graph = _local_mutate_graph(parent.graph, rng)
                        child = Individual(
                            graph=child_graph,
                            generation=gen + 1,
                            parent_fingerprint=parent.fingerprint,
                            metadata={"mutation_type": "local", "exploit": True},
                        )
                        new_population.append(child)
                        logger.debug(
                            "exploit_mutation: parent_fp=%s, gen=%d",
                            parent.fingerprint[:16],
                            gen + 1,
                        )
                        continue
                except Exception as exc:
                    logger.debug(
                        "exploit_mutation failed for parent %s; falling back to standard reproduction: %s",
                        parent.fingerprint[:16],
                        exc,
                        exc_info=True,
                    )

            try:
                reproduction_mode = _choose_reproduction_mode(config, rng)
                if reproduction_mode == "crossover":
                    child = _spawn_crossover_individual(
                        population,
                        config,
                        grammar,
                        rng,
                        gen + 1,
                        tournament_select=_tournament_select,
                        individual_cls=Individual,
                        generate_context_valid_graph=generate_context_valid_graph,
                    )
                elif reproduction_mode == "mutation":
                    child = _spawn_mutation_individual(
                        population,
                        config,
                        grammar,
                        rng,
                        gen + 1,
                        local_mutation_fitness_threshold,
                        tournament_select=_tournament_select,
                        individual_cls=Individual,
                        generate_context_valid_graph=generate_context_valid_graph,
                    )
                else:
                    child = _spawn_fresh_individual(
                        grammar,
                        rng,
                        gen + 1,
                        generate_context_valid_graph=generate_context_valid_graph,
                        individual_cls=Individual,
                    )
                new_population.append(child)
            except (ValueError, RuntimeError):
                if reproduction_mode != "fresh":
                    try:
                        new_population.append(
                            _spawn_fresh_individual(
                                grammar,
                                rng,
                                gen + 1,
                                generate_context_valid_graph=generate_context_valid_graph,
                                individual_cls=Individual,
                            )
                        )
                    except (ValueError, RuntimeError):
                        fill_failures += 1
                else:
                    fill_failures += 1

        if not new_population:
            logger.error(
                "Gen %d: produced 0 individuals after %d attempts. "
                "Aborting evolution early.",
                gen,
                fill_attempts,
            )
            break

        population = new_population

        # Evaluate and rank with NSGA-II
        _evaluate_population(population, fitness_fn, novelty_fn)
        nsga2_rank(population)
        population = _enforce_population_diversity(
            population=population,
            fitness_fn=fitness_fn,
            novelty_fn=novelty_fn,
            config=config,
            grammar=grammar,
            rng=rng,
            generation=gen + 1,
            generate_context_valid_graph=generate_context_valid_graph,
        )

        if gen % 5 == 0 or gen == config.n_generations - 1:
            best = max(population, key=lambda x: x.fitness) if population else None
            best_fit = f"{best.fitness:.4f}" if best else "N/A"
            logger.info(
                "Evolution gen %d/%d: pop=%d, best_fitness=%s",
                gen + 1,
                config.n_generations,
                len(population),
                best_fit,
            )

            # Update grammar weights from Pareto front every 5 generations
            front_weights = pareto_front_op_weights(population)
            if front_weights:
                for op_name, w in front_weights.items():
                    grammar.op_weights[op_name] = max(0.1, w)

        if callback:
            callback(gen, population)

    # Final sort
    population.sort(key=lambda x: _combined_score(x, config), reverse=True)
    return population


def _evaluate_population(
    population: List[Individual],
    fitness_fn: Callable,
    novelty_fn: Optional[Callable],
    _config: Optional[EvolutionConfig] = None,
):
    """Evaluate fitness and novelty for all individuals.

    Individuals with ``metadata["_evaluated"] == True`` skip the fitness_fn
    call (their fitness is already set from a prior generation).  Novelty is
    always recomputed because it depends on the current population.
    """
    for ind in population:
        if ind.metadata.get("_evaluated"):
            continue
        try:
            ind.fitness = fitness_fn(ind.graph)
            ind.metadata.pop("fitness_error_type", None)
            ind.metadata.pop("fitness_error", None)
        except Exception as exc:
            ind.fitness = 0.0
            ind.metadata["fitness_error_type"] = type(exc).__name__
            ind.metadata["fitness_error"] = str(exc)[:240]
        ind.metadata["_evaluated"] = True

    # Novelty always recomputed — it depends on the current population mix.
    _evaluate_novelty(population, novelty_fn)


def _evaluate_novelty(
    population: List[Individual], novelty_fn: Optional[Callable]
) -> None:
    if not novelty_fn:
        return
    all_graphs = [ind.graph for ind in population]
    batch_scores = getattr(novelty_fn, "batch_scores", None)
    if callable(batch_scores):
        try:
            scores = batch_scores(all_graphs)
            if len(scores) != len(population):
                raise ValueError("batch novelty score count does not match population")
            for ind, novelty in zip(population, scores, strict=True):
                ind.novelty = float(novelty)
                ind.metadata.pop("novelty_error_type", None)
                ind.metadata.pop("novelty_error", None)
            return
        except Exception as exc:
            for ind in population:
                ind.metadata["novelty_batch_error_type"] = type(exc).__name__
                ind.metadata["novelty_batch_error"] = str(exc)[:240]

    for ind in population:
        try:
            ind.novelty = novelty_fn(ind.graph, all_graphs)
            ind.metadata.pop("novelty_error_type", None)
            ind.metadata.pop("novelty_error", None)
        except Exception as exc:
            ind.novelty = 0.0
            ind.metadata["novelty_error_type"] = type(exc).__name__
            ind.metadata["novelty_error"] = str(exc)[:240]


def _mutate_graph(
    graph: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
) -> ComputationGraph:
    return _mutation_mutate_graph(
        graph,
        grammar,
        rng,
        generate_context_valid_graph=_generate_context_valid_graph,
    )


def _crossover_graphs(
    g1: ComputationGraph,
    g2: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
) -> ComputationGraph:
    return _mutation_crossover_graphs(
        g1,
        g2,
        grammar,
        rng,
        generate_context_valid_graph=_generate_context_valid_graph,
    )


def _enforce_population_diversity(
    population: List[Individual],
    fitness_fn: Callable,
    novelty_fn: Optional[Callable],
    config: EvolutionConfig,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
    generate_context_valid_graph=_generate_context_valid_graph,
) -> List[Individual]:
    """Reduce fingerprint duplicates by replacing clone overflow with fresh individuals.
    Uses vectorized fingerprint comparisons for speed.
    """
    if not population:
        return population

    def score(ind: Individual) -> float:
        return _combined_score(ind, config)

    ranked = sorted(population, key=score, reverse=True)

    seen_fingerprints = set()
    deduped: List[Individual] = []
    duplicates = 0

    for ind in ranked:
        fp = ind.fingerprint
        if fp in seen_fingerprints:
            duplicates += 1
            continue

        seen_fingerprints.add(fp)
        deduped.append(ind)

    if duplicates == 0:
        return ranked

    # Fill back to population size with fresh unique individuals.
    new_replacements: List[Individual] = []
    max_attempts = max(10, config.population_size * 5)
    attempts = 0
    while (
        len(deduped) + len(new_replacements) < config.population_size
        and attempts < max_attempts
    ):
        attempts += 1
        try:
            graph = generate_context_valid_graph(grammar, rng)
            fp = graph.fingerprint()
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            ind = Individual(graph=graph, generation=generation)
            ind.metadata["diversity_replacement"] = True
            new_replacements.append(ind)
        except (ValueError, RuntimeError):
            continue

    # Evaluate only newly generated replacements (deduped survivors already evaluated).
    if new_replacements:
        _evaluate_population(new_replacements, fitness_fn, novelty_fn)

    deduped.extend(new_replacements)

    # If generation failed to refill entirely, append highest-ranked unique leftovers.
    if len(deduped) < config.population_size:
        for ind in ranked:
            if len(deduped) >= config.population_size:
                break
            fp = ind.fingerprint
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            deduped.append(ind)

    deduped = deduped[: config.population_size]

    # Recompute novelty for everyone now that population composition changed.
    _evaluate_novelty(deduped, novelty_fn)

    for ind in deduped:
        ind.metadata["dedupe_duplicates_replaced"] = duplicates

    return sorted(deduped, key=score, reverse=True)


def _tournament_select(
    population: List[Individual],
    tournament_size: int,
    rng: random.Random,
    fitness_weight: float = 0.5,
    novelty_weight: float = 0.5,
) -> Individual:
    """Tournament selection with NSGA-II crowded comparison when available."""
    candidates = rng.sample(population, min(tournament_size, len(population)))

    def _cmp_key(x: Individual) -> Tuple[float, float]:
        if x.pareto_rank != 0:
            # Lower rank better (negate so max works), higher crowding better
            return (-x.pareto_rank, x.crowding_dist)
        return (x.fitness * fitness_weight + x.novelty * novelty_weight, 0.0)

    return max(candidates, key=_cmp_key)


def pareto_front_op_weights(
    population: List[Individual],
    baseline_weight: float = 1.0,
    boost: float = 1.5,
    penalty: float = 0.7,
) -> Dict[str, float]:
    """Extract grammar op weight adjustments from Pareto front analysis.

    Ops that appear disproportionately in rank-0 individuals get boosted;
    ops that appear only in dominated individuals get penalized.

    Args:
        population: NSGA-II ranked population (must have pareto_rank set).
        baseline_weight: Default weight for unaffected ops.
        boost: Weight multiplier for Pareto-front-enriched ops.
        penalty: Weight multiplier for dominated-only ops.

    Returns:
        Dict mapping op_name → weight multiplier.
    """
    if not population:
        return {}

    front_ops: Dict[str, int] = {}
    dominated_ops: Dict[str, int] = {}

    for ind in population:
        counter = front_ops if ind.pareto_rank == _PARETO_FRONT_RANK else dominated_ops
        for node in ind.graph.nodes.values():
            if node.is_input:
                continue
            op_name = node.op_name
            counter[op_name] = counter.get(op_name, 0) + 1

    all_ops = set(front_ops) | set(dominated_ops)
    weights: Dict[str, float] = {}

    for op in all_ops:
        f_count = front_ops.get(op, 0)
        d_count = dominated_ops.get(op, 0)
        total = f_count + d_count
        if total == 0:
            continue
        front_ratio = f_count / total
        if front_ratio > 0.6:
            weights[op] = baseline_weight * boost
        elif front_ratio < 0.2 and d_count >= 3:
            weights[op] = baseline_weight * penalty
        # else: leave at default (no entry = no override)

    return weights


def _combined_score(ind: Individual, config: EvolutionConfig) -> float:
    """Weighted population score used consistently across selection and ranking."""
    if ind.pareto_rank != 0:
        # Lower pareto_rank is better, higher crowding_dist is better
        return -ind.pareto_rank + ind.crowding_dist * 0.001
    return ind.fitness * config.fitness_weight + ind.novelty * config.novelty_weight


def _fitness_top_k_threshold(population: List[Individual], top_k: int) -> float | None:
    if top_k <= 0 or not population:
        return None
    top_k = min(top_k, len(population))
    return float(np.partition([ind.fitness for ind in population], -top_k)[-top_k])


def _choose_reproduction_mode(
    config: EvolutionConfig,
    rng: random.Random,
) -> str:
    """Sample a reproduction mode from configured probabilities."""
    modes = (
        ("fresh", max(0.0, float(config.fresh_injection_rate))),
        ("crossover", max(0.0, float(config.crossover_rate))),
        ("mutation", max(0.0, float(config.mutation_rate))),
    )
    total = sum(weight for _, weight in modes)
    if total <= 0:
        return "mutation"
    draw = rng.random() * total
    cumulative = 0.0
    for mode, weight in modes:
        cumulative += weight
        if draw <= cumulative:
            return mode
    return "mutation"
