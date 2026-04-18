from __future__ import annotations

import random
from typing import Dict, Tuple

from ..synthesis.context_rules import validate_context_rules
from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig
from ..synthesis.primitives import REVERSE_OPCODE_MAP, get_primitive, list_primitives
from ._evolution_grammar import derive_crossover_grammar, derive_mutation_grammar
from .native_graph_mutation import plan_local_mutation_trials

_LOCAL_MUTATION_CHOICES_CACHE: Dict[Tuple[object, int], Tuple[str, ...]] = {}


def spawn_fresh_individual(
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
    *,
    generate_context_valid_graph,
    individual_cls,
):
    graph = generate_context_valid_graph(grammar, rng)
    child = individual_cls(graph=graph, generation=generation)
    child.metadata["fresh_injection"] = True
    return child


def spawn_mutation_individual(
    population,
    config,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
    local_mutation_fitness_threshold: float | None,
    *,
    tournament_select,
    individual_cls,
    generate_context_valid_graph,
):
    parent = tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )
    is_top_k = (
        local_mutation_fitness_threshold is not None
        and parent.fitness >= local_mutation_fitness_threshold
    )
    if is_top_k and rng.random() < config.local_mutation_prob:
        child_graph = local_mutate_graph(parent.graph, rng)
        mutation_type = "local"
    else:
        child_graph = mutate_graph(
            parent.graph,
            grammar,
            rng,
            generate_context_valid_graph=generate_context_valid_graph,
        )
        mutation_type = "standard"

    return individual_cls(
        graph=child_graph,
        generation=generation,
        parent_fingerprint=parent.fingerprint,
        metadata={"mutation_type": mutation_type},
    )


def spawn_crossover_individual(
    population,
    config,
    grammar: GrammarConfig,
    rng: random.Random,
    generation: int,
    *,
    tournament_select,
    individual_cls,
    generate_context_valid_graph,
):
    if len(population) < 2:
        raise ValueError("crossover requires at least two parents")
    p1 = tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )
    p2 = tournament_select(
        population,
        config.tournament_size,
        rng,
        config.fitness_weight,
        config.novelty_weight,
    )
    child_graph = crossover_graphs(
        p1.graph,
        p2.graph,
        grammar,
        rng,
        generate_context_valid_graph=generate_context_valid_graph,
    )
    parents = sorted([p1.fingerprint, p2.fingerprint])
    return individual_cls(
        graph=child_graph,
        generation=generation,
        parent_fingerprint=f"{parents[0]}x{parents[1]}",
    )


def mutate_graph(
    graph: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
    *,
    generate_context_valid_graph,
) -> ComputationGraph:
    parent_fp = graph.fingerprint()
    mut_grammar = derive_mutation_grammar(graph, grammar, rng)
    for _ in range(3):
        new_graph = generate_context_valid_graph(mut_grammar, rng)
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


def local_mutate_graph(
    graph: ComputationGraph,
    rng: random.Random,
) -> ComputationGraph:
    new_graph = graph.copy()
    native_plan = _native_local_mutation_plan(graph, rng)
    if native_plan is not None:
        node_ids, candidate_opcodes = native_plan
        for target_id, candidate_opcode in zip(
            node_ids, candidate_opcodes, strict=True
        ):
            candidate_name = REVERSE_OPCODE_MAP[int(candidate_opcode)]
            target_node = new_graph.nodes[int(target_id)]
            original_op = target_node.op_name
            target_node.op_name = candidate_name
            new_graph._ir_version += 1
            new_graph._cache.clear()
            if validate_context_rules(new_graph) is None:
                new_graph.metadata["lineage"] = {
                    "type": "local_mutation",
                    "parent": graph.fingerprint(),
                    "swapped_node": int(target_id),
                    "old_op": original_op,
                    "new_op": candidate_name,
                }
                return new_graph
            target_node.op_name = original_op
            new_graph._ir_version += 1
            new_graph._cache.clear()
        return new_graph

    for target_id, candidate_name in _python_local_mutation_trials(graph, rng):
        target_node = new_graph.nodes[target_id]
        original_op = target_node.op_name
        target_node.op_name = candidate_name
        new_graph._ir_version += 1
        new_graph._cache.clear()
        if validate_context_rules(new_graph) is None:
            new_graph.metadata["lineage"] = {
                "type": "local_mutation",
                "parent": graph.fingerprint(),
                "swapped_node": target_id,
                "old_op": original_op,
                "new_op": candidate_name,
            }
            return new_graph
        target_node.op_name = original_op
        new_graph._ir_version += 1
        new_graph._cache.clear()
    return new_graph


def crossover_graphs(
    g1: ComputationGraph,
    g2: ComputationGraph,
    grammar: GrammarConfig,
    rng: random.Random,
    *,
    generate_context_valid_graph,
) -> ComputationGraph:
    p1_fp = g1.fingerprint()
    p2_fp = g2.fingerprint()
    cross_grammar = derive_crossover_grammar(g1, g2, grammar, rng)
    child = generate_context_valid_graph(cross_grammar, rng)
    child.prune_unreachable_nodes()
    child.metadata["lineage"] = {
        "type": "crossover",
        "parents": [p1_fp, p2_fp],
        "parent_depths": [g1.depth(), g2.depth()],
        "parent_ops": [g1.n_ops(), g2.n_ops()],
    }
    return child


def _local_mutation_choice_names(category: object, n_inputs: int) -> Tuple[str, ...]:
    key = (category, n_inputs)
    cached = _LOCAL_MUTATION_CHOICES_CACHE.get(key)
    if cached is not None:
        return cached
    names = tuple(
        dict.fromkeys(
            op.name for op in list_primitives(category) if op.n_inputs == n_inputs
        )
    )
    _LOCAL_MUTATION_CHOICES_CACHE[key] = names
    return names


def _native_local_mutation_trials(
    graph: ComputationGraph,
    rng: random.Random,
):
    native_plan = _native_local_mutation_plan(graph, rng)
    if native_plan is None:
        return None
    node_ids, candidate_opcodes = native_plan
    return tuple(
        (
            int(target_id),
            REVERSE_OPCODE_MAP[int(candidate_opcode)],
        )
        for target_id, candidate_opcode in zip(node_ids, candidate_opcodes, strict=True)
    )


def _native_local_mutation_plan(
    graph: ComputationGraph,
    rng: random.Random,
):
    analysis_ir = graph._analysis_ir()
    planned = plan_local_mutation_trials(analysis_ir.op_codes, seed=rng.getrandbits(64))
    if planned is None:
        return None

    graph_node_indices, candidate_opcodes = planned
    if graph_node_indices.size == 0:
        return (
            graph_node_indices,
            candidate_opcodes,
        )

    node_ids = analysis_ir.node_ids
    if node_ids is None:
        return None
    return (
        node_ids[graph_node_indices],
        candidate_opcodes,
    )


def _python_local_mutation_trials(
    graph: ComputationGraph,
    rng: random.Random,
):
    non_input_ids = [nid for nid, node in graph.nodes.items() if not node.is_input]
    if not non_input_ids:
        return ()

    targets = list(non_input_ids)
    rng.shuffle(targets)
    planned_trials = []
    for target_id in targets:
        target_node = graph.nodes[target_id]
        try:
            current_op = get_primitive(target_node.op_name)
        except KeyError:
            continue
        same_cat_ops = [
            op_name
            for op_name in _local_mutation_choice_names(
                current_op.category, current_op.n_inputs
            )
            if op_name != target_node.op_name
        ]
        if not same_cat_ops:
            continue
        rng.shuffle(same_cat_ops)
        planned_trials.extend(
            (target_id, candidate_name) for candidate_name in same_cat_ops
        )
    return tuple(planned_trials)
