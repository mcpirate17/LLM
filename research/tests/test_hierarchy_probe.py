"""Tests for hierarchy probe (Phase 3 of Three-Pillar Upgrade)."""

import pytest
import numpy as np
import torch


def test_gromov_delta_tree_metric():
    """Gromov delta of a perfect tree metric should be ~0."""
    from research.eval.hierarchy_probe import gromov_delta

    # Build a simple tree distance matrix (star graph: center + leaves)
    n = 10
    d = np.zeros((n, n))
    # Node 0 is center, all others at distance 1 from center
    for i in range(1, n):
        d[0, i] = d[i, 0] = 1.0
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = 2.0  # Through center

    delta = gromov_delta(d)
    assert delta >= 0.0
    # Star graphs are 0-hyperbolic (tree metric)
    assert delta < 0.5, f"Tree metric should have low delta, got {delta}"


def test_gromov_delta_random_euclidean():
    """Gromov delta of random Euclidean points should be > 0."""
    from research.eval.hierarchy_probe import gromov_delta
    from scipy.spatial.distance import pdist, squareform

    np.random.seed(42)
    points = np.random.randn(30, 10)
    d = squareform(pdist(points))

    delta = gromov_delta(d)
    assert delta > 0.0, "Random Euclidean points should have nonzero delta"


def test_hierarchy_fitness_shape():
    """hierarchy_fitness returns expected keys and value ranges."""
    from research.eval.hierarchy_probe import hierarchy_fitness

    reps = torch.randn(2, 16, 64)
    result = hierarchy_fitness(reps)

    assert "hierarchy_fitness" in result
    assert "gromov_delta" in result
    assert "n_tokens_sampled" in result
    assert 0.0 <= result["hierarchy_fitness"] <= 1.0
    assert result["gromov_delta"] >= 0.0
    assert result["n_tokens_sampled"] > 0


def test_hierarchy_fitness_small_input():
    """hierarchy_fitness handles very small inputs gracefully."""
    from research.eval.hierarchy_probe import hierarchy_fitness

    reps = torch.randn(1, 2, 32)  # Only 2 tokens
    result = hierarchy_fitness(reps)
    assert 0.0 <= result["hierarchy_fitness"] <= 1.0


def test_grammar_boost_activates():
    """Grammar hyperbolic boost should activate when hierarchy_fitness > threshold."""
    from research.synthesis.grammar import GrammarConfig

    config = GrammarConfig()
    config._hierarchy_fitness = 0.8  # Above threshold of 0.6
    assert config._hierarchy_fitness > config.hyperbolic_promotion_threshold


def test_grammar_boost_deactivates():
    """Grammar hyperbolic boost should not activate below threshold."""
    from research.synthesis.grammar import GrammarConfig

    config = GrammarConfig()
    config._hierarchy_fitness = 0.3  # Below threshold of 0.6
    assert config._hierarchy_fitness <= config.hyperbolic_promotion_threshold
