import pytest
import torch

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE

from research.eval.fingerprint import (
    _collect_position_sensitivities,
    _sensitivity_metrics,
)

pytestmark = pytest.mark.unit


def test_sensitivity_metrics_native_returns_valid():
    sens_matrix = torch.tensor(
        [
            [0.5, 0.2, 0.1, 0.0],
            [0.1, 0.7, 0.2, 0.1],
            [0.0, 0.1, 0.6, 0.3],
        ],
        dtype=torch.float32,
    )

    result = _sensitivity_metrics(sens_matrix)

    for key in ("spectral_norm", "uniformity", "effective_rank"):
        assert key in result
        assert isinstance(result[key], float)
    assert result["spectral_norm"] > 0.0
    assert 0.0 <= result["uniformity"] <= 1.0
    assert result["effective_rank"] >= 1.0


def test_sensitivity_metrics_helper_uses_single_path():
    sens_matrix = torch.rand(3, 5, dtype=torch.float32)
    result = _sensitivity_metrics(sens_matrix)
    for key in ("spectral_norm", "uniformity", "effective_rank"):
        assert key in result
        assert isinstance(result[key], float)


def test_sensitivity_collection_returns_correct_shape():
    base = torch.randn(1, 5, 3, dtype=torch.float32)
    embed = base.clone().requires_grad_(True)
    x = embed * 0.5 + embed.roll(shifts=1, dims=1) * 0.25
    positions = torch.tensor([0, 2, 4], dtype=torch.int64)

    result = _collect_position_sensitivities(x, embed, positions)

    assert result is not None
    assert result.shape[0] == 3  # n_positions
    assert result.shape[1] == 5  # seq_len
