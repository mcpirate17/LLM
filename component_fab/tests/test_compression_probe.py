"""Tests for the compression compress/restore contract + module scorer.

Covers the WS-1 persistence half: a compression lane exposes its real
compress/restore bottleneck, the scorer pulls it out of a (fab-built) compiled
module and produces a real scorecard, and non-compression modules score to
``{}`` (never a fabricated weakness).
"""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.memory_primitives import HierarchicalResidualCompressorLane
from component_fab.metrics.compression_probe import (
    SupportsCompressionProbe,
    score_compression_in_module,
)
from component_fab.runner.grading import metadata_for_grade
from component_fab.tests.conftest import make_spec


def test_compressor_forward_equals_read_of_summaries() -> None:
    torch.manual_seed(0)
    lane = HierarchicalResidualCompressorLane(16)
    x = torch.randn(2, 8, 16)
    # The forward refactor must be exactly read(_summaries(x)).
    assert torch.allclose(lane(x), lane.read(lane._summaries(x)))


def test_compressor_implements_contract_with_consistent_pair() -> None:
    torch.manual_seed(0)
    lane = HierarchicalResidualCompressorLane(16, n_levels=4)
    assert isinstance(lane, SupportsCompressionProbe)
    compress, restore, latent_dim = lane.compression_probe_pair()
    assert latent_dim == 16 * 4
    x = torch.randn(2, 8, 16)
    # restore(compress(x)) is exactly the lane's readout path.
    assert torch.allclose(restore(compress(x)), lane(x))


def test_scorer_returns_empty_for_non_compression_module() -> None:
    plain = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    assert score_compression_in_module(plain, dim=16, seq_len=8) == {}


def test_scorer_scores_real_compression_lane() -> None:
    torch.manual_seed(0)
    wrapped = nn.Sequential(nn.Linear(16, 16), HierarchicalResidualCompressorLane(16))
    meta = score_compression_in_module(wrapped, dim=16, seq_len=8)
    assert meta["compression_declared"] is True
    assert meta["compression_n_ops"] == 1
    for key in (
        "compression_quality",
        "compression_effective_rank_ratio",
        "compression_reconstruct_mse",
        "compression_ratio",
    ):
        assert isinstance(meta[key], float)
    assert 0.0 <= meta["compression_effective_rank_ratio"] <= 1.0


def test_fab_built_compression_candidate_is_scored() -> None:
    # The fab dispatches op_invention_mechanism=hierarchical_residual_compressor
    # to the real lane; the scorer must find it inside the compiled module.
    module = generate_module(
        {"op_invention_mechanism": "hierarchical_residual_compressor"}, dim=32
    )
    torch.manual_seed(0)
    meta = score_compression_in_module(module, dim=32, seq_len=16)
    assert meta.get("compression_declared") is True
    assert meta["compression_n_ops"] >= 1


def test_metadata_for_grade_merges_compression_metrics() -> None:
    spec = make_spec({"op_invention_mechanism": "hierarchical_residual_compressor"})
    compression = {
        "compression_declared": True,
        "compression_effective_rank_ratio": 0.16,
        "compression_reconstruct_mse": 1.02,
    }
    meta = metadata_for_grade(spec, {"can_bind": True}, None, compression=compression)
    assert meta["compression_declared"] is True
    assert meta["compression_effective_rank_ratio"] == 0.16
    assert meta["compression_reconstruct_mse"] == 1.02

    # No compression op -> the keys are simply absent (never fabricated).
    bare = metadata_for_grade(spec, {"can_bind": True}, None)
    assert "compression_declared" not in bare
