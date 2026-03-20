"""Tests for BatchGenerateResult statistics from batch_generate."""

from __future__ import annotations

from unittest.mock import patch

from research.synthesis.grammar import (
    BatchGenerateResult,
    GrammarConfig,
    batch_generate,
)


def test_batch_generate_returns_result_dataclass():
    """batch_generate returns BatchGenerateResult, not a bare list."""
    result = batch_generate(5, GrammarConfig(model_dim=64), base_seed=42)
    assert isinstance(result, BatchGenerateResult)
    assert isinstance(result.graphs, list)
    assert result.n_attempted > 0
    assert result.n_attempted >= len(result.graphs)
    assert result.n_rejected_grammar >= 0
    assert result.n_rejected_dedup >= 0


def test_batch_generate_counts_grammar_failures():
    """If generate_layer_graph raises ValueError on the first 5 attempts
    then succeeds, n_attempted=6, n_rejected_grammar=5, graphs has length 1."""
    call_count = 0

    def _mock_generate(config, seed=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            raise ValueError("mock grammar failure")
        # Return a minimal valid graph
        from research.synthesis.graph import ComputationGraph

        g = ComputationGraph(config.model_dim)
        inp = g.add_input()
        proj = g.add_op("linear_proj", [inp], config={"out_dim": config.model_dim})
        g.set_output(proj)
        return g

    with patch(
        "research.synthesis.grammar.generate_layer_graph",
        side_effect=_mock_generate,
    ):
        result = batch_generate(1, GrammarConfig(model_dim=64), base_seed=99)

    assert len(result.graphs) == 1
    assert result.n_attempted == 6, f"Expected 6 attempts, got {result.n_attempted}"
    assert result.n_rejected_grammar == 5, (
        f"Expected 5 grammar failures, got {result.n_rejected_grammar}"
    )


def test_batch_generate_counts_dedup_rejections():
    """Duplicate fingerprints should be counted in n_rejected_dedup."""
    call_count = 0

    def _mock_generate(config, seed=None):
        nonlocal call_count
        call_count += 1
        from research.synthesis.graph import ComputationGraph

        g = ComputationGraph(config.model_dim)
        inp = g.add_input()
        # Use different seeds to make unique-ish graphs on odd calls,
        # duplicate on even calls (same structure = same fingerprint)
        if call_count % 2 == 0:
            # Duplicate of the first graph
            proj = g.add_op("linear_proj", [inp], config={"out_dim": config.model_dim})
        else:
            proj = g.add_op("linear_proj", [inp], config={"out_dim": config.model_dim})
            proj = g.add_op("rmsnorm", [proj])
        g.set_output(proj)
        return g

    with patch(
        "research.synthesis.grammar.generate_layer_graph",
        side_effect=_mock_generate,
    ):
        result = batch_generate(5, GrammarConfig(model_dim=64), base_seed=99)

    # We should get at most 2 unique graphs (odd vs even structure)
    assert len(result.graphs) <= 2
    assert result.n_rejected_dedup > 0
    assert (
        result.n_rejected_grammar + result.n_rejected_dedup + len(result.graphs)
        == result.n_attempted
    )
