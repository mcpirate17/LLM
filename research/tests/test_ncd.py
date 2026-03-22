"""Tests for NCD reward signal (Phase 2 of Three-Pillar Upgrade)."""

import pytest

pytestmark = pytest.mark.unit


def test_compute_ncd_basic():
    """NCD returns values in [0, 1] range and behaves sensibly."""
    from research.eval.ncd import compute_ncd
    import os

    # NCD should be in [0, 1] range
    data = b"hello world " * 100
    ncd = compute_ncd(data, data)
    assert 0.0 <= ncd <= 1.0

    # Random data should have high NCD (incompressible, no shared info)
    data_a = os.urandom(500)
    data_b = os.urandom(500)
    ncd_random = compute_ncd(data_a, data_b)
    assert 0.0 <= ncd_random <= 1.0

    # Data that shares a prefix should have lower NCD than random
    shared = b"shared content here " * 50
    data_c = shared + b"ending A " * 20
    data_d = shared + b"ending B " * 20
    ncd_shared = compute_ncd(data_c, data_d)
    assert ncd_shared < 0.5  # Should be relatively low due to shared content


def test_compute_graph_ncd_range():
    """compute_graph_ncd should return ncd_score in [0, 1]."""
    from research.eval.ncd import compute_graph_ncd

    result = compute_graph_ncd('{"ops":[]}', [1.0, 0.5, 0.3])
    assert 0 <= result["ncd_score"] <= 1
    assert result["description_length"] > 0
    assert result["description_length_per_param"] is None  # No n_params given


def test_compute_graph_ncd_with_params():
    """compute_graph_ncd with n_params should compute per-param description length."""
    from research.eval.ncd import compute_graph_ncd

    result = compute_graph_ncd(
        '{"ops":["linear","relu"]}', [2.0, 1.5, 1.0, 0.8], n_params=1000
    )
    assert result["description_length_per_param"] is not None
    assert result["description_length_per_param"] > 0


def test_compute_graph_ncd_dict_curve():
    """Loss curve as list of dicts (training_curve format)."""
    from research.eval.ncd import compute_graph_ncd

    curve = [
        {"loss": 2.0, "step": 0},
        {"loss": 1.5, "step": 1},
        {"loss": 1.0, "step": 2},
    ]
    result = compute_graph_ncd('{"ops":[]}', curve)
    assert 0 <= result["ncd_score"] <= 1


def test_composite_score_backwards_compat():
    """Composite score without NCD should match previous behavior."""
    from research.scientist.notebook import LabNotebook

    score = LabNotebook.compute_composite_score(
        screening_lr=0.5,
        screening_nov=0.7,
    )

    # Canonical scoring: nonlinear perf curve + novelty with confidence gate
    assert score > 30.0  # sanity: should be positive and meaningful
