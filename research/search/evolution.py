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
from typing import Callable, Dict, List, Optional

from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig, generate_layer_graph
from ..synthesis.primitives import get_primitive

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    graph: ComputationGraph
    fitness: float = 0.0
    novelty: float = 0.0
    generation: int = 0
    parent_fingerprint: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return self.graph.fingerprint()

    @property
    def lineage_hash(self) -> str:
        """Deterministic hash of the ancestry to avoid redundant fingerprinting."""
        # Use existing lineage if present, otherwise compute it
        lh = self.metadata.get("lineage_hash")
        if lh:
            return lh
        
        if not self.parent_fingerprint:
            # Root individual (gen 0)
            res = f"root:{self.fingerprint[:16]}"
        else:
            # Derived individual
            res = f"gen{self.generation}:{self.parent_fingerprint[:32]}"
        
        self.metadata["lineage_hash"] = res
        return res


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary search."""
    population_size: int = 50
    n_generations: int = 20
    tournament_size: int = 5
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    elitism: int = 5  # top N carry over unchanged
    # Fitness weighting
    fitness_weight: float = 0.5
    novelty_weight: float = 0.5
    grammar_config: Optional[GrammarConfig] = None


def evolutionary_search(
    fitness_fn: Callable[[ComputationGraph], float],
    novelty_fn: Optional[Callable[[ComputationGraph, List[ComputationGraph]], float]] = None,
    config: Optional[EvolutionConfig] = None,
    seed: int = 42,
    callback: Optional[Callable[[int, List[Individual]], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
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
    grammar = config.grammar_config or GrammarConfig()

    # Initialize population
    population = []
    init_failures = 0
    for i in range(config.population_size):
        try:
            graph = generate_layer_graph(grammar, seed=seed + i * 137)
            ind = Individual(graph=graph, generation=0)
            population.append(ind)
        except (ValueError, RuntimeError) as e:
            init_failures += 1
            if init_failures <= 3:
                logger.debug("Initial population gen failed (%d): %s", init_failures, e)

    if not population:
        logger.error(
            "Evolution aborted: failed to generate any initial individuals "
            "(%d attempts all failed)", config.population_size,
        )
        return []

    if init_failures > 0:
        logger.info(
            "Initial population: %d/%d succeeded (%d failures)",
            len(population), config.population_size, init_failures,
        )

    # Evaluate initial population
    _evaluate_population(population, fitness_fn, novelty_fn, config)
    population = _enforce_population_diversity(
        population=population,
        fitness_fn=fitness_fn,
        novelty_fn=novelty_fn,
        config=config,
        grammar=grammar,
        rng=rng,
        generation=0,
    )

    # Evolve
    for gen in range(config.n_generations):
        if stop_check and stop_check():
            logger.info("Evolution stopped early at gen %d by external signal", gen)
            break

        new_population = []

        # Elitism: keep top individuals
        population.sort(key=lambda x: x.fitness * config.fitness_weight +
                        x.novelty * config.novelty_weight, reverse=True)
        new_population.extend(population[:config.elitism])

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
                    gen, max_fill_attempts, len(new_population),
                    config.population_size, fill_failures,
                )
                break

            if rng.random() < config.crossover_rate and len(population) >= 2:
                # Crossover
                p1 = _tournament_select(population, config.tournament_size, rng)
                p2 = _tournament_select(population, config.tournament_size, rng)
                try:
                    child_graph = _crossover_graphs(p1.graph, p2.graph, grammar, rng)
                    # Deterministic parent ordering for lineage hash stability
                    parents = sorted([p1.fingerprint, p2.fingerprint])
                    child = Individual(
                        graph=child_graph,
                        generation=gen + 1,
                        parent_fingerprint=f"{parents[0]}x{parents[1]}",
                    )
                    new_population.append(child)
                except (ValueError, RuntimeError):
                    fill_failures += 1
            else:
                # Mutation
                parent = _tournament_select(population, config.tournament_size, rng)
                try:
                    child_graph = _mutate_graph(parent.graph, grammar, rng)
                    child = Individual(
                        graph=child_graph,
                        generation=gen + 1,
                        parent_fingerprint=parent.fingerprint,
                    )
                    new_population.append(child)
                except (ValueError, RuntimeError):
                    # If mutation fails, generate fresh
                    try:
                        graph = generate_layer_graph(grammar, seed=rng.randint(0, 2**32))
                        child = Individual(graph=graph, generation=gen + 1)
                        new_population.append(child)
                    except (ValueError, RuntimeError):
                        fill_failures += 1

        if not new_population:
            logger.error(
                "Gen %d: produced 0 individuals after %d attempts. "
                "Aborting evolution early.", gen, fill_attempts,
            )
            break

        population = new_population

        # Evaluate
        _evaluate_population(population, fitness_fn, novelty_fn, config)
        population = _enforce_population_diversity(
            population=population,
            fitness_fn=fitness_fn,
            novelty_fn=novelty_fn,
            config=config,
            grammar=grammar,
            rng=rng,
            generation=gen + 1,
        )

        if gen % 5 == 0 or gen == config.n_generations - 1:
            best = max(population, key=lambda x: x.fitness) if population else None
            best_fit = f"{best.fitness:.4f}" if best else "N/A"
            logger.info(
                "Evolution gen %d/%d: pop=%d, best_fitness=%s",
                gen + 1, config.n_generations, len(population), best_fit,
            )

        if callback:
            callback(gen, population)

    # Final sort
    population.sort(key=lambda x: x.fitness * config.fitness_weight +
                    x.novelty * config.novelty_weight, reverse=True)
    return population


def _evaluate_population(
    population: List[Individual],
    fitness_fn: Callable,
    novelty_fn: Optional[Callable],
    config: EvolutionConfig,
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
    if novelty_fn:
        all_graphs = [ind.graph for ind in population]
        for ind in population:
            try:
                ind.novelty = novelty_fn(ind.graph, all_graphs)
                ind.metadata.pop("novelty_error_type", None)
                ind.metadata.pop("novelty_error", None)
            except Exception as exc:
                ind.novelty = 0.0
                ind.metadata["novelty_error_type"] = type(exc).__name__
                ind.metadata["novelty_error"] = str(exc)[:240]


def _enforce_population_diversity(
    population: List[Individual],
    fitness_fn: Callable,
    novelty_fn: Optional[Callable],
    config: EvolutionConfig,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
) -> List[Individual]:
    """Reduce fingerprint duplicates by replacing clone overflow with fresh individuals.
    Uses vectorized fingerprint comparisons for speed.
    """
    if not population:
        return population

    def score(ind: Individual) -> float:
        return ind.fitness * config.fitness_weight + ind.novelty * config.novelty_weight

    ranked = sorted(population, key=score, reverse=True)
    
    # Fast-path: use lineage_hash for structural identity check without IR lowering
    seen_hashes = set()
    seen_fingerprints = set()
    deduped: List[Individual] = []
    duplicates = 0

    for ind in ranked:
        lh = ind.lineage_hash
        if lh in seen_hashes:
            duplicates += 1
            continue
        
        # Fallback to fingerprint for potential convergence/root collisions
        fp = ind.fingerprint
        if fp in seen_fingerprints:
            duplicates += 1
            continue
            
        seen_hashes.add(lh)
        seen_fingerprints.add(fp)
        deduped.append(ind)

    if duplicates == 0:
        return ranked

    # Fill back to population size with fresh unique individuals.
    new_replacements: List[Individual] = []
    max_attempts = max(10, config.population_size * 5)
    attempts = 0
    while len(deduped) + len(new_replacements) < config.population_size and attempts < max_attempts:
        attempts += 1
        try:
            graph = generate_layer_graph(grammar, seed=rng.randint(0, 2**32))
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
        _evaluate_population(new_replacements, fitness_fn, novelty_fn, config)

    deduped.extend(new_replacements)

    # If generation failed to refill entirely, append highest-ranked leftovers.
    if len(deduped) < config.population_size:
        for ind in ranked:
            if len(deduped) >= config.population_size:
                break
            # Skip if already in deduped
            if any(d.graph is ind.graph for d in deduped):
                continue
            deduped.append(ind)

    deduped = deduped[:config.population_size]

    # Recompute novelty for everyone now that population composition changed.
    if novelty_fn:
        from ..eval.metrics import batch_novelty_scores
        all_graphs = [ind.graph for ind in deduped]
        novelty_metrics = batch_novelty_scores(all_graphs)
        for ind, metrics in zip(deduped, novelty_metrics):
            ind.novelty = metrics.overall_novelty

    for ind in deduped:
        ind.metadata["dedupe_duplicates_replaced"] = duplicates

    return sorted(deduped, key=score, reverse=True)


def _tournament_select(
    population: List[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    """Tournament selection."""
    candidates = rng.sample(population, min(tournament_size, len(population)))
    return max(candidates, key=lambda x: x.fitness + x.novelty)


def _mutate_graph(
    graph: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
) -> ComputationGraph:
    """Mutate a computation graph using parent-informed grammar perturbation."""
    parent_fp = graph.fingerprint()
    mut_grammar = _derive_mutation_grammar(graph, grammar, rng)

    # Try more than once to avoid trivially reproducing parent structure.
    for _ in range(3):
        new_graph = generate_layer_graph(mut_grammar, seed=rng.randint(0, 2**32))
        if new_graph.fingerprint() != parent_fp:
            break

    new_graph.metadata["lineage"] = {
        "type": "mutation",
        "parent": parent_fp,
        "parent_depth": graph.depth(),
        "parent_ops": graph.n_ops(),
    }
    return new_graph


def _crossover_graphs(
    g1: ComputationGraph,
    g2: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
) -> ComputationGraph:
    """Crossover two graphs by blending parent structure statistics."""
    p1_fp = g1.fingerprint()
    p2_fp = g2.fingerprint()
    cross_grammar = _derive_crossover_grammar(g1, g2, grammar, rng)

    child = generate_layer_graph(cross_grammar, seed=rng.randint(0, 2**32))
    child.metadata["lineage"] = {
        "type": "crossover",
        "parents": [p1_fp, p2_fp],
        "parent_depths": [g1.depth(), g2.depth()],
        "parent_ops": [g1.n_ops(), g2.n_ops()],
    }
    return child


def _derive_mutation_grammar(
    graph: ComputationGraph,
    base: GrammarConfig,
    rng: random.Random,
) -> GrammarConfig:
    """Create a lightly perturbed grammar centered on a parent graph.

    Caps max_depth and max_ops to hard limits to prevent unbounded growth
    across generations which causes Python recursion depth exceeded errors.
    """
    # Hard caps to prevent recursion depth overflow across generations
    HARD_MAX_DEPTH = 12
    HARD_MAX_OPS = 20

    parent_depth = max(1, graph.depth())
    parent_ops = max(1, graph.n_ops())
    parent_cat = _category_histogram(graph)

    min_depth = max(2, min(base.min_depth, parent_depth))
    max_depth = min(
        HARD_MAX_DEPTH,
        max(min_depth + 1, min(max(base.max_depth, parent_depth + 2), parent_depth + 4)),
    )
    max_ops = min(
        HARD_MAX_OPS,
        max(parent_ops + 2, min(max(base.max_ops, parent_ops + 4), parent_ops + 8)),
    )

    category_weights = dict(base.category_weights)
    for cat_name in category_weights:
        if parent_cat.get(cat_name, 0) > 0:
            category_weights[cat_name] = category_weights[cat_name] * 1.25
        else:
            category_weights[cat_name] = max(0.1, category_weights[cat_name] * 0.9)
        category_weights[cat_name] = max(0.1, category_weights[cat_name] * rng.uniform(0.9, 1.1))

    return GrammarConfig(
        model_dim=graph.model_dim,
        min_depth=min_depth,
        max_depth=max_depth,
        max_width=base.max_width,
        max_ops=max_ops,
        max_params_ratio=base.max_params_ratio,
        residual_prob=_clamp(base.residual_prob + rng.uniform(-0.1, 0.1), 0.0, 1.0),
        split_prob=_clamp(base.split_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        merge_prob=_clamp(base.merge_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        risky_op_prob=_clamp(base.risky_op_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0),
        freq_domain_prob=_clamp(base.freq_domain_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0),
        category_weights=category_weights,
        excluded_ops=set(base.excluded_ops),
    )


def _derive_crossover_grammar(
    g1: ComputationGraph,
    g2: ComputationGraph,
    base: GrammarConfig,
    rng: random.Random,
) -> GrammarConfig:
    """Create a blended grammar from two parents.

    Caps max_depth and max_ops to hard limits to prevent unbounded growth.
    """
    HARD_MAX_DEPTH = 12
    HARD_MAX_OPS = 20

    d1, d2 = max(1, g1.depth()), max(1, g2.depth())
    o1, o2 = max(1, g1.n_ops()), max(1, g2.n_ops())

    target_depth = max(2, int(round((d1 + d2) / 2 + rng.choice([-1, 0, 1]))))
    target_ops = max(3, int(round((o1 + o2) / 2 + rng.choice([-2, -1, 0, 1, 2]))))

    min_depth = max(2, min(base.min_depth, target_depth))
    max_depth = min(
        HARD_MAX_DEPTH,
        max(min_depth + 1, min(max(base.max_depth, target_depth + 2), target_depth + 4)),
    )
    max_ops = min(
        HARD_MAX_OPS,
        max(target_ops + 2, min(max(base.max_ops, target_ops + 4), target_ops + 10)),
    )

    cat1 = _category_histogram(g1)
    cat2 = _category_histogram(g2)
    category_weights = dict(base.category_weights)
    for cat_name, weight in category_weights.items():
        used = cat1.get(cat_name, 0) + cat2.get(cat_name, 0)
        if used > 0:
            category_weights[cat_name] = max(0.1, weight * 1.2)
        else:
            category_weights[cat_name] = max(0.1, weight * 0.85)
        category_weights[cat_name] = max(0.1, category_weights[cat_name] * rng.uniform(0.92, 1.08))

    return GrammarConfig(
        model_dim=g1.model_dim,
        min_depth=min_depth,
        max_depth=max_depth,
        max_width=max(base.max_width, 2),
        max_ops=max_ops,
        max_params_ratio=base.max_params_ratio,
        residual_prob=_clamp((base.residual_prob + 0.65) / 2 + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        split_prob=_clamp((base.split_prob + 0.35) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0),
        merge_prob=_clamp((base.merge_prob + 0.45) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0),
        risky_op_prob=_clamp(base.risky_op_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0),
        freq_domain_prob=_clamp(base.freq_domain_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0),
        category_weights=category_weights,
        excluded_ops=set(base.excluded_ops),
    )


def _category_histogram(graph: ComputationGraph) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for node in graph.nodes.values():
        if node.is_input:
            continue
        try:
            cat = get_primitive(node.op_name).category.value
        except KeyError:
            continue
        hist[cat] = hist.get(cat, 0) + 1
    return hist


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
