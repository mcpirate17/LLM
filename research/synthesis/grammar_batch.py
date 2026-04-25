"""Batch-generation compatibility helpers for the motif grammar."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchGenerateResult:
    """Result of batch_generate with generation statistics."""

    graphs: List["ComputationGraph"]
    n_attempted: int
    n_rejected_grammar: int
    n_rejected_dedup: int


def _candidate_batch_size(remaining: int) -> int:
    return min(32, max(8, remaining))


def _try_batch_packed_validation(candidates: list) -> list:
    from .graph_validator import build_dim_flow_validation_inputs
    from .native_analysis import validate_packed_ir_batch_natively
    from .validation_opcode_tables import validation_opcode_tables

    if not candidates:
        return []

    tables = validation_opcode_tables()
    dim_inputs = []
    input_node_indices = []
    for graph in candidates:
        analysis_ir = graph._analysis_ir()
        inputs = build_dim_flow_validation_inputs(
            graph,
            analysis_ir=analysis_ir,
            compute_analysis=False,
        )
        dim_inputs.append(inputs)
        input_node_indices.append(
            inputs.node_id_to_analysis_idx.get(graph._input_node_id, -1)
        )

    packed_results = validate_packed_ir_batch_natively(
        op_codes=[inputs.analysis_ir.op_codes for inputs in dim_inputs],
        input_indices=[inputs.analysis_ir.input_indices for inputs in dim_inputs],
        output_node_indices=[
            int(inputs.analysis_ir.output_node_idx) for inputs in dim_inputs
        ],
        param_estimates=[inputs.param_estimates for inputs in dim_inputs],
        has_params_flags=[inputs.has_params_flags for inputs in dim_inputs],
        nontrivial_flags=[inputs.nontrivial_flags for inputs in dim_inputs],
        kv_breaking_flags=[inputs.kv_breaking_flags for inputs in dim_inputs],
        node_dims=[inputs.node_dims for inputs in dim_inputs],
        node_seq_flags=[inputs.node_seq_flags for inputs in dim_inputs],
        op_kind_flags=[inputs.op_kind_flags for inputs in dim_inputs],
        full_dim_flags=[inputs.full_dim_flags for inputs in dim_inputs],
        model_dims=[graph.model_dim for graph in candidates],
        input_node_indices=input_node_indices,
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )
    if packed_results is None:
        return [(graph, inputs, None) for graph, inputs in zip(candidates, dim_inputs)]
    return list(zip(candidates, dim_inputs, packed_results))


def _forced_template_name(config) -> str | None:
    if config.forced_template:
        return str(config.forced_template)
    positive_templates = [
        str(name)
        for name, weight in config.template_weights.items()
        if float(weight) > 0.0
    ]
    return positive_templates[0] if len(positive_templates) == 1 else None


def _validate_candidate_batch(grammar, candidates: list, config) -> tuple[list, int]:
    valid_graphs = []
    rejected = 0
    forced_template = _forced_template_name(config)
    for graph, dim_inputs, packed_result in _try_batch_packed_validation(candidates):
        if forced_template is not None and graph.metadata.get("templates_used") != [
            forced_template
        ]:
            rejected += 1
            continue
        try:
            grammar._validate_graph(
                graph,
                config,
                dim_flow_inputs=dim_inputs,
                packed_validation=packed_result,
            )
        except (ValueError, RuntimeError):
            rejected += 1
            continue
        valid_graphs.append(graph)
    return valid_graphs, rejected


def _generate_unvalidated_candidate(grammar, config, seed: int):
    return grammar.generate_layer_graph(config, seed=seed, validate=False)


def _generate_candidate_batch(
    *,
    grammar,
    config,
    base_seed: int,
    start_attempt: int,
    max_attempts: int,
    target_count: int,
) -> tuple[list, int, int]:
    candidates = []
    attempts = 0
    rejected = 0
    while attempts < max_attempts and len(candidates) < target_count:
        attempts += 1
        try:
            graph = _generate_unvalidated_candidate(
                grammar,
                config,
                base_seed + (start_attempt + attempts) * 137,
            )
            candidates.append(graph)
        except (ValueError, RuntimeError):
            rejected += 1
    return candidates, attempts, rejected


def _add_unique_graphs(
    valid_graphs: list, graphs: list, fingerprints: set, n: int
) -> int:
    rejected_dedup = 0
    for graph in valid_graphs:
        fingerprint = graph.fingerprint()
        if fingerprint in fingerprints:
            rejected_dedup += 1
            continue
        fingerprints.add(fingerprint)
        graphs.append(graph)
        if len(graphs) >= n:
            break
    return rejected_dedup


def _fill_unique_graphs(
    *,
    grammar,
    config,
    base_seed: int,
    attempts: int,
    max_attempts: int,
    n: int,
    graphs: list,
    fingerprints: set,
) -> tuple[int, int, int]:
    n_rejected_grammar = 0
    n_rejected_dedup = 0
    while len(graphs) < n and attempts < max_attempts:
        candidates, attempted, rejected = _generate_candidate_batch(
            grammar=grammar,
            config=config,
            base_seed=base_seed,
            start_attempt=attempts,
            max_attempts=max_attempts - attempts,
            target_count=_candidate_batch_size(n - len(graphs)),
        )
        attempts += attempted
        n_rejected_grammar += rejected
        valid_graphs, rejected_validation = _validate_candidate_batch(
            grammar, candidates, config
        )
        n_rejected_grammar += rejected_validation
        n_rejected_dedup += _add_unique_graphs(valid_graphs, graphs, fingerprints, n)
    return attempts, n_rejected_grammar, n_rejected_dedup


def _relaxed_config(grammar, config):
    return grammar.replace(
        config,
        composition_depth=max(1, config.composition_depth - 1),
        max_ops=config.max_ops + 6,
        max_depth=config.max_depth + 6,
        template_exploration_budget=max(config.template_exploration_budget, 0.25),
        forced_template=None,
    )


def batch_generate(
    n: int,
    config=None,
    base_seed: int = 42,
    _use_adaptive_synthesis: bool = False,
    prior=None,
) -> BatchGenerateResult:
    """Generate N unique computation graphs."""
    from . import grammar

    if config is None:
        config = grammar.GrammarConfig()
    config = grammar._config_with_efficiency_prior(config, prior)

    graphs: list = []
    fingerprints: set = set()
    attempts = 0
    n_rejected_grammar = 0
    n_rejected_dedup = 0
    max_attempts = n * 10

    attempts, rejected_grammar, rejected_dedup = _fill_unique_graphs(
        grammar=grammar,
        config=config,
        base_seed=base_seed,
        attempts=attempts,
        max_attempts=max_attempts,
        n=n,
        graphs=graphs,
        fingerprints=fingerprints,
    )
    n_rejected_grammar += rejected_grammar
    n_rejected_dedup += rejected_dedup

    if len(graphs) == 0 and n_rejected_grammar > 0:
        relaxed = _relaxed_config(grammar, config)
        logger.info(
            "batch_generate: exhaustion recovery - relaxing constraints "
            "(depth %d->%d, ops %d->%d, composition %d->%d, exploration %.0f%%)",
            config.max_depth,
            relaxed.max_depth,
            config.max_ops,
            relaxed.max_ops,
            config.composition_depth,
            relaxed.composition_depth,
            relaxed.template_exploration_budget * 100,
        )
        retry_start = attempts
        attempts, rejected_grammar, rejected_dedup = _fill_unique_graphs(
            grammar=grammar,
            config=relaxed,
            base_seed=base_seed + 99999,
            attempts=attempts,
            max_attempts=retry_start + (n * 5),
            n=n,
            graphs=graphs,
            fingerprints=fingerprints,
        )
        n_rejected_grammar += rejected_grammar
        n_rejected_dedup += rejected_dedup

    rejection_rate = (n_rejected_grammar + n_rejected_dedup) / max(attempts, 1)
    logger.info(
        "batch_generate: %d graphs from %d attempts "
        "(%d grammar failures, %d duplicates, %.0f%% rejection rate)",
        len(graphs),
        attempts,
        n_rejected_grammar,
        n_rejected_dedup,
        rejection_rate * 100,
    )

    return BatchGenerateResult(
        graphs=graphs,
        n_attempted=attempts,
        n_rejected_grammar=n_rejected_grammar,
        n_rejected_dedup=n_rejected_dedup,
    )


class AdaptiveGenerator:
    """Adaptive generator - delegates to motif-based generation."""

    __slots__ = ("config", "prior", "model_dim", "max_params", "max_flops")

    def __init__(self, config, prior: Optional[object] = None):
        self.config = config
        self.prior = prior
        self.model_dim = config.model_dim
        self.max_params = 4 * self.model_dim * self.model_dim * 12
        self.max_flops = 4 * (12 * self.model_dim * self.model_dim * 128)

    def generate(self, seed: Optional[int] = None):
        from . import grammar

        return grammar.generate_layer_graph(
            grammar._config_with_efficiency_prior(self.config, self.prior),
            seed=seed,
        )
