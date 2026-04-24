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


def _generate_unique_graph(
    *,
    grammar,
    config,
    seed: int,
    graphs: list,
    fingerprints: set,
) -> bool:
    graph = grammar.generate_layer_graph(config, seed=seed)
    fingerprint = graph.fingerprint()
    if fingerprint in fingerprints:
        return False
    fingerprints.add(fingerprint)
    graphs.append(graph)
    return True


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

    while len(graphs) < n and attempts < max_attempts:
        attempts += 1
        try:
            if not _generate_unique_graph(
                grammar=grammar,
                config=config,
                seed=base_seed + attempts * 137,
                graphs=graphs,
                fingerprints=fingerprints,
            ):
                n_rejected_dedup += 1
        except (ValueError, RuntimeError):
            n_rejected_grammar += 1

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
        retry_attempts = 0
        while len(graphs) < n and retry_attempts < n * 5:
            retry_attempts += 1
            attempts += 1
            try:
                if not _generate_unique_graph(
                    grammar=grammar,
                    config=relaxed,
                    seed=base_seed + attempts * 137 + 99999,
                    graphs=graphs,
                    fingerprints=fingerprints,
                ):
                    n_rejected_dedup += 1
            except (ValueError, RuntimeError):
                n_rejected_grammar += 1

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
