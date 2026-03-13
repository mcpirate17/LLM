import pytest
import torch

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE
import torch.nn as nn

from research.eval.fingerprint import (
    _interaction_influence_matrix,
    _interaction_metrics,
)

pytestmark = pytest.mark.unit


def test_interaction_metrics_native_returns_valid():
    influence = torch.tensor(
        [
            [1.0, 0.4, 0.1, 0.0, 0.0],
            [0.2, 1.2, 0.5, 0.1, 0.0],
            [0.0, 0.3, 1.1, 0.4, 0.1],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor([0, 2, 4], dtype=torch.int64)

    result = _interaction_metrics(influence, positions)

    for key in ("locality", "sparsity", "symmetry", "hierarchy"):
        assert key in result
        val = result[key]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_interaction_metrics_helper_uses_single_path():
    influence = torch.rand(4, 6, dtype=torch.float32)
    positions = torch.tensor([0, 1, 3, 5], dtype=torch.int64)

    result = _interaction_metrics(influence, positions)

    for key in ("locality", "sparsity", "symmetry", "hierarchy"):
        assert key in result
        assert isinstance(result[key], float)


class _TinyInteractionModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


def test_interaction_influence_matrix_matches_two_pass_behavior():
    torch.manual_seed(0)
    model = _TinyInteractionModel(vocab_size=32, hidden_dim=8)
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor([0, 2, 4], dtype=torch.long)

    combined = _interaction_influence_matrix(model, ids, positions, vocab_size=32)

    base_out = model(ids)
    perturbed = ids.expand(len(positions), -1).clone()
    row_idx = torch.arange(len(positions))
    perturbed[row_idx, positions] = (perturbed[row_idx, positions] + 1) % 32
    two_pass = (model(perturbed) - base_out).abs().mean(dim=-1)

    assert combined.shape == two_pass.shape
    assert torch.allclose(combined, two_pass, atol=1e-6, rtol=1e-6)
