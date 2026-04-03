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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig, generate_layer_graph
from ..synthesis.primitives import get_primitive, list_primitives
from ..scientist.shared_utils import clamp
from .native_nsga import compute_crowding_distances

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
) -> ComputationGraph:
    """Retry generation a few times under stricter context validation."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return generate_layer_graph(grammar, seed=rng.randint(0, 2**32))
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
    _evaluate_population(population, fitness_fn, novelty_fn, config)
    nsga2_rank(population)
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

            reproduction_mode = _choose_reproduction_mode(config, rng)
            mode_handlers = {
                "crossover": lambda: _spawn_crossover_individual(
                    population, config, grammar, rng, gen + 1
                ),
                "mutation": lambda: _spawn_mutation_individual(
                    population, config, grammar, rng, gen + 1
                ),
                "fresh": lambda: _spawn_fresh_individual(grammar, rng, gen + 1),
            }
            try:
                child = mode_handlers[reproduction_mode]()
                new_population.append(child)
            except (ValueError, RuntimeError):
                if reproduction_mode != "fresh":
                    try:
                        new_population.append(
                            _spawn_fresh_individual(grammar, rng, gen + 1)
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
        _evaluate_population(population, fitness_fn, novelty_fn, config)
        nsga2_rank(population)
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
        return _combined_score(ind, config)

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
    while (
        len(deduped) + len(new_replacements) < config.population_size
        and attempts < max_attempts
    ):
        attempts += 1
        try:
            graph = _generate_context_valid_graph(grammar, rng)
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
        deduped_ids = {id(d.graph) for d in deduped}
        for ind in ranked:
            if len(deduped) >= config.population_size:
                break
            if id(ind.graph) in deduped_ids:
                continue
            deduped_ids.add(id(ind.graph))
            deduped.append(ind)

    deduped = deduped[: config.population_size]

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
    fitness_weight: float = 0.5,
    novelty_weight: float = 0.5,
) -> Individual:
    """Tournament selection with NSGA-II crowded comparison when available."""
    candidates = rng.sample(population, min(tournament_size, len(population)))

    def _cmp_key(x: Individual) -> Tuple[float, float]:
        if x.pareto_rank > 0:
            # Lower rank better (negate so max works), higher crowding better
            return (-x.pareto_rank, x.crowding_dist)
        return (x.fitness * fitness_weight + x.novelty * novelty_weight, 0.0)

    return max(candidates, key=_cmp_key)


_DEFAULT_OBJECTIVES: List[Tuple[str, str]] = [("fitness", "max"), ("novelty", "max")]


def fast_non_dominated_sort(
    population: List[Individual],
    objectives: Sequence[Tuple[str, str]] = _DEFAULT_OBJECTIVES,
) -> List[List[Individual]]:
    """NSGA-II fast non-dominated sort.

    Uses NumPy broadcasting for vectorized dominance comparison.

    Args:
        population: Individuals to rank.
        objectives: List of ``(attr_name, "min"|"max")`` tuples.

    Returns:
        List of Pareto fronts (front 0 = non-dominated).
    """
    n = len(population)
    if n == 0:
        return []

    # Stack all objective values into (n, m) array, sign-flipped so higher=better.
    signs = np.array(
        [1.0 if d == "max" else -1.0 for _, d in objectives], dtype=np.float64
    )
    attr_names = [o[0] for o in objectives]
    vals = (
        np.array(
            [[getattr(ind, a) for a in attr_names] for ind in population],
            dtype=np.float64,
        )
        * signs
    )  # (n, m)

    # Vectorized dominance: i dominates j iff all(vals[i] >= vals[j]) and any(vals[i] > vals[j])
    # Compare every pair via broadcasting: (n, 1, m) vs (1, n, m)
    diff = vals[:, np.newaxis, :] - vals[np.newaxis, :, :]  # (n, n, m)
    ge_all = np.all(diff >= 0, axis=2)  # (n, n) — i >= j on all objectives
    gt_any = np.any(diff > 0, axis=2)  # (n, n) — i > j on at least one
    dominates = ge_all & gt_any  # (n, n) — i dominates j

    # domination_count[j] = number of individuals that dominate j
    domination_count = dominates.sum(axis=0)  # sum over i dimension

    # Build fronts iteratively
    fronts: List[List[Individual]] = []
    rank = 1
    remaining = np.ones(n, dtype=bool)

    while True:
        # Current front: individuals with domination_count == 0 among remaining
        front_mask = remaining & (domination_count == 0)
        front_indices = np.where(front_mask)[0]
        if len(front_indices) == 0:
            break

        front = []
        for i in front_indices:
            population[i].pareto_rank = rank
            front.append(population[i])
        fronts.append(front)

        # Remove front members and decrement counts for those they dominated
        remaining[front_indices] = False
        for i in front_indices:
            dominated_by_i = np.where(dominates[i] & remaining)[0]
            domination_count[dominated_by_i] -= 1

        rank += 1

    return fronts


def assign_crowding_distance(
    front: List[Individual],
    objectives: Sequence[Tuple[str, str]] = _DEFAULT_OBJECTIVES,
) -> None:
    """Compute and assign crowding distance for a single Pareto front."""
    n = len(front)
    if n <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    attr_names = [attr for attr, _ in objectives]
    objective_matrix = np.asarray(
        [[getattr(ind, attr) for attr in attr_names] for ind in front],
        dtype=np.float32,
    )
    native_distances = compute_crowding_distances(objective_matrix)
    if native_distances is not None:
        for ind, distance in zip(front, native_distances):
            ind.crowding_dist = float(distance)
        return

    _assign_crowding_distance_in_python(front, objective_matrix)


def _assign_crowding_distance_in_python(
    front: List[Individual],
    objective_matrix: np.ndarray,
) -> None:
    """Reference crowding-distance implementation used as fallback and benchmark baseline."""
    n = len(front)
    for ind in front:
        ind.crowding_dist = 0.0

    for objective_idx in range(objective_matrix.shape[1]):
        order = np.argsort(objective_matrix[:, objective_idx], kind="mergesort")
        obj_values = objective_matrix[order, objective_idx]
        obj_min = float(obj_values[0])
        obj_max = float(obj_values[-1])
        span = obj_max - obj_min
        front[int(order[0])].crowding_dist = float("inf")
        front[int(order[-1])].crowding_dist = float("inf")
        if span <= 0.0:
            continue
        inv_span = 1.0 / span
        for pos in range(1, n - 1):
            idx = int(order[pos])
            if np.isinf(front[idx].crowding_dist):
                continue
            front[idx].crowding_dist += float(
                (obj_values[pos + 1] - obj_values[pos - 1]) * inv_span
            )


def nsga2_rank(
    population: List[Individual],
    objectives: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Individual]:
    """Rank population using NSGA-II non-dominated sort + crowding distance.

    Args:
        population: Individuals to rank.
        objectives: Objective specs; defaults to fitness(max) + novelty(max).

    Returns:
        Population sorted by (pareto_rank ASC, crowding_dist DESC).
    """
    if not population:
        return population

    objs = objectives if objectives is not None else _DEFAULT_OBJECTIVES
    fronts = fast_non_dominated_sort(population, objs)
    for front in fronts:
        assign_crowding_distance(front, objs)

    population.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))
    return population


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
        ops = [n.op_name for n in ind.graph.nodes.values() if not n.is_input]
        counter = front_ops if ind.pareto_rank == 0 else dominated_ops
        for op in ops:
            counter[op] = counter.get(op, 0) + 1

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
    if ind.pareto_rank > 0:
        # Lower pareto_rank is better, higher crowding_dist is better
        return -ind.pareto_rank + ind.crowding_dist * 0.001
    return ind.fitness * config.fitness_weight + ind.novelty * config.novelty_weight


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


def _spawn_fresh_individual(
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
) -> Individual:
    """Generate a fresh individual directly from the grammar."""
    graph = _generate_context_valid_graph(grammar, rng)
    child = Individual(graph=graph, generation=generation)
    child.metadata["fresh_injection"] = True
    return child


def _spawn_mutation_individual(
    population: List[Individual],
    config: EvolutionConfig,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
) -> Individual:
    """Sample a parent and return a mutation child.

    Top-K parents by fitness use local mutation (single-op swap) with
    probability local_mutation_prob, preserving winning topology.
    """
    parent = _tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )

    # Check if parent is in top-K by fitness
    top_k_fitness = sorted((ind.fitness for ind in population), reverse=True)[
        : config.exploit_top_k
    ]
    is_top_k = parent.fitness >= top_k_fitness[-1] if top_k_fitness else False

    if is_top_k and rng.random() < config.local_mutation_prob:
        child_graph = _local_mutate_graph(parent.graph, rng)
        mutation_type = "local"
    else:
        child_graph = _mutate_graph(parent.graph, grammar, rng)
        mutation_type = "standard"

    return Individual(
        graph=child_graph,
        generation=generation,
        parent_fingerprint=parent.fingerprint,
        metadata={"mutation_type": mutation_type},
    )


def _spawn_crossover_individual(
    population: List[Individual],
    config: EvolutionConfig,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
) -> Individual:
    """Sample two parents and return a crossover child."""
    if len(population) < 2:
        raise ValueError("crossover requires at least two parents")
    p1 = _tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )
    p2 = _tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )
    child_graph = _crossover_graphs(p1.graph, p2.graph, grammar, rng)
    parents = sorted([p1.fingerprint, p2.fingerprint])
    return Individual(
        graph=child_graph,
        generation=generation,
        parent_fingerprint=f"{parents[0]}x{parents[1]}",
    )


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
        new_graph = _generate_context_valid_graph(mut_grammar, rng)
        if new_graph.fingerprint() != parent_fp:
            break

    new_graph.prune_unreachable_nodes()

    new_graph.metadata["lineage"] = {
        "type": "mutation",
        "parent": parent_fp,
        "parent_depth": graph.depth(),
        "parent_ops": graph.n_ops(),
    }
    return new_graph


def _local_mutate_graph(
    graph: ComputationGraph,
    rng: random.Random,
) -> ComputationGraph:
    """Local mutation: swap one random op with a same-category alternative.

    Preserves the winning topology (all edges/connections intact) while
    exploring nearby alternatives in the op space. Validates context rules
    to avoid producing graphs with known-fatal op sequences (e.g.
    full-dim ops after split2, double-routing chains).
    """
    import copy
    from ..synthesis.context_rules import validate_context_rules

    new_graph = copy.deepcopy(graph)
    non_input_ids = [nid for nid, node in new_graph.nodes.items() if not node.is_input]
    if not non_input_ids:
        return new_graph

    # Shuffle node order so we try different targets if first fails
    targets = list(non_input_ids)
    rng.shuffle(targets)

    for target_id in targets:
        target_node = new_graph.nodes[target_id]

        try:
            current_op = get_primitive(target_node.op_name)
        except KeyError:
            continue

        # Find same-category alternatives, shuffled
        same_cat_ops = [
            op
            for op in list_primitives(current_op.category)
            if op.name != target_node.op_name and op.n_inputs == current_op.n_inputs
        ]
        if not same_cat_ops:
            continue
        rng.shuffle(same_cat_ops)

        # Try each candidate until one passes context validation
        original_op = target_node.op_name
        for candidate in same_cat_ops:
            target_node.op_name = candidate.name
            new_graph._cache.clear()

            violation = validate_context_rules(new_graph)
            if violation is None:
                new_graph.metadata["lineage"] = {
                    "type": "local_mutation",
                    "parent": graph.fingerprint(),
                    "swapped_node": target_id,
                    "old_op": original_op,
                    "new_op": candidate.name,
                }
                return new_graph

        # All candidates for this node failed — restore and try next node
        target_node.op_name = original_op
        new_graph._cache.clear()

    # No valid swap found — return unmodified copy
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

    child = _generate_context_valid_graph(cross_grammar, rng)
    child.prune_unreachable_nodes()
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
    HARD_MAX_DEPTH = 18
    HARD_MAX_OPS = 28

    parent_depth = max(1, graph.depth())
    parent_ops = max(1, graph.n_ops())
    parent_cat = _category_histogram(graph)

    max_depth = min(
        HARD_MAX_DEPTH,
        max(3, min(max(base.max_depth, parent_depth + 2), parent_depth + 4)),
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
        category_weights[cat_name] = max(
            0.1, category_weights[cat_name] * rng.uniform(0.9, 1.1)
        )

    # Propagate template/motif weights from parent grammar
    template_weights = dict(base.template_weights) if base.template_weights else {}
    motif_weights = dict(base.motif_weights) if base.motif_weights else {}

    # If parent has sparse/routing ops, bias child toward efficiency
    _SPARSE_ROUTING_OPS = frozenset(
        {
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
            "token_entropy",
            "moe_topk",
            "moe_2expert",
            "adjacent_token_merge",
        }
    )
    parent_has_efficiency = any(
        n.op_name in _SPARSE_ROUTING_OPS for n in graph.nodes.values() if not n.is_input
    )
    sparsity_bias = base.structured_sparsity_bias
    if parent_has_efficiency:
        sparsity_bias = max(sparsity_bias, 0.6)

    op_weights = dict(base.op_weights)

    # Binding range bias: if parent has only local-range mixers, boost
    # templates and ops that provide full-range sequence mixing.
    from ..synthesis.primitives import graph_binding_range_class

    parent_binding = graph_binding_range_class(graph)
    if parent_binding in ("local", "none"):
        for tpl_key in ("transformer_block", "state_space_block"):
            if tpl_key in template_weights:
                template_weights[tpl_key] *= 2.5
        _FULL_RANGE_OPS = frozenset(
            {
                "softmax_attention",
                "linear_attention",
                "graph_attention",
                "diff_attention",
                "state_space",
                "selective_scan",
                "rwkv_time_mixing",
                "gated_delta",
            }
        )
        for op_name in _FULL_RANGE_OPS:
            op_weights[op_name] = op_weights.get(op_name, 1.0) * 3.0

    return GrammarConfig(
        model_dim=graph.model_dim,
        max_depth=max_depth,
        max_width=base.max_width,
        max_ops=max_ops,
        residual_prob=clamp(base.residual_prob + rng.uniform(-0.1, 0.1), 0.0, 1.0),
        split_prob=clamp(base.split_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        merge_prob=clamp(base.merge_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        risky_op_prob=clamp(base.risky_op_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0),
        freq_domain_prob=clamp(
            base.freq_domain_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0
        ),
        category_weights=category_weights,
        op_weights=op_weights,
        template_weights=template_weights,
        motif_weights=motif_weights,
        structured_sparsity_bias=sparsity_bias,
        routing_mandatory=base.routing_mandatory,
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
    HARD_MAX_DEPTH = 18
    HARD_MAX_OPS = 28

    d1, d2 = max(1, g1.depth()), max(1, g2.depth())
    o1, o2 = max(1, g1.n_ops()), max(1, g2.n_ops())

    target_depth = max(2, int(round((d1 + d2) / 2 + rng.choice([-1, 0, 1]))))
    target_ops = max(3, int(round((o1 + o2) / 2 + rng.choice([-2, -1, 0, 1, 2]))))

    max_depth = min(
        HARD_MAX_DEPTH,
        max(3, min(max(base.max_depth, target_depth + 2), target_depth + 4)),
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
        category_weights[cat_name] = max(
            0.1, category_weights[cat_name] * rng.uniform(0.92, 1.08)
        )

    # Propagate template/motif weights from parent grammar
    template_weights = dict(base.template_weights) if base.template_weights else {}
    motif_weights = dict(base.motif_weights) if base.motif_weights else {}

    # If either parent has sparse/routing ops, bias child toward efficiency
    _SPARSE_ROUTING_OPS = frozenset(
        {
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
            "token_entropy",
            "moe_topk",
            "moe_2expert",
            "adjacent_token_merge",
        }
    )
    sparsity_bias = base.structured_sparsity_bias
    for g in (g1, g2):
        if any(
            n.op_name in _SPARSE_ROUTING_OPS for n in g.nodes.values() if not n.is_input
        ):
            sparsity_bias = max(sparsity_bias, 0.6)
            break

    op_weights = dict(base.op_weights)

    # Binding range bias: if both parents are local-only, boost full-range ops
    from ..synthesis.primitives import graph_binding_range_class

    g1_binding = graph_binding_range_class(g1)
    g2_binding = graph_binding_range_class(g2)
    if g1_binding in ("local", "none") and g2_binding in ("local", "none"):
        for tpl_key in ("transformer_block", "state_space_block"):
            if tpl_key in template_weights:
                template_weights[tpl_key] *= 2.5
        _FULL_RANGE_OPS = frozenset(
            {
                "softmax_attention",
                "linear_attention",
                "graph_attention",
                "diff_attention",
                "state_space",
                "selective_scan",
                "rwkv_time_mixing",
                "gated_delta",
            }
        )
        for op_name in _FULL_RANGE_OPS:
            op_weights[op_name] = op_weights.get(op_name, 1.0) * 3.0

    return GrammarConfig(
        model_dim=g1.model_dim,
        max_depth=max_depth,
        max_width=max(base.max_width, 2),
        max_ops=max_ops,
        residual_prob=clamp(
            (base.residual_prob + 0.65) / 2 + rng.uniform(-0.08, 0.08), 0.0, 1.0
        ),
        split_prob=clamp(
            (base.split_prob + 0.35) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0
        ),
        merge_prob=clamp(
            (base.merge_prob + 0.45) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0
        ),
        risky_op_prob=clamp(base.risky_op_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0),
        freq_domain_prob=clamp(
            base.freq_domain_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0
        ),
        category_weights=category_weights,
        op_weights=op_weights,
        template_weights=template_weights,
        motif_weights=motif_weights,
        structured_sparsity_bias=sparsity_bias,
        routing_mandatory=base.routing_mandatory,
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
