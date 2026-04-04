import pytest
import torch

import torch.nn as nn

from research.eval.fingerprint import (
    _capture_probe_representations,
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


def test_interaction_influence_matrix_prefers_logits_from_embed_fast_path():
    class _FastInteractionModel(nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.forward_called = False
            self.fast_called = False

        def _fingerprint_logits_from_embed(self, embed_in):
            self.fast_called = True
            return self.proj(embed_in)

        def forward(self, _input_ids):
            self.forward_called = True
            raise RuntimeError("id-based fallback should not run")

    torch.manual_seed(0)
    model = _FastInteractionModel(vocab_size=32, hidden_dim=8)
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor([0, 2, 4], dtype=torch.long)

    combined = _interaction_influence_matrix(model, ids, positions, vocab_size=32)

    baseline = model.proj(model.embed(ids))
    perturbed = ids.expand(len(positions), -1).clone()
    row_idx = torch.arange(len(positions))
    perturbed[row_idx, positions] = (perturbed[row_idx, positions] + 1) % 32
    expected = (model.proj(model.embed(perturbed)) - baseline).abs().mean(dim=-1)

    assert model.fast_called is True
    assert model.forward_called is False
    assert torch.allclose(combined, expected, atol=1e-6, rtol=1e-6)


def test_interaction_influence_matrix_prefers_pre_logits_native_reduction(monkeypatch):
    import research.eval.fingerprint_probes as fingerprint_probes

    class _PreLogitsInteractionModel(nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.pre_called = False
            self.logits_called = False

        def _fingerprint_pre_logits_from_embed(self, embed_in):
            self.pre_called = True
            return embed_in + 0.5

        def _fingerprint_logits_from_embed(self, _embed_in):
            self.logits_called = True
            raise RuntimeError("logits fast path should not run")

    calls = {"native": 0}

    def _fake_native(delta, weight):
        calls["native"] += 1
        return torch.nn.functional.linear(delta, weight).abs().mean(dim=-1)

    monkeypatch.setattr(fingerprint_probes, "mean_abs_linear_delta", _fake_native)

    torch.manual_seed(0)
    model = _PreLogitsInteractionModel(vocab_size=32, hidden_dim=8)
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor([0, 2, 4], dtype=torch.long)

    combined = _interaction_influence_matrix(model, ids, positions, vocab_size=32)

    base = model.embed(ids) + 0.5
    perturbed = model.embed(ids.expand(len(positions), -1).clone())
    row_idx = torch.arange(len(positions))
    replacement_ids = (ids[0, positions] + 1) % 32
    perturbed[row_idx, positions] = model.embed(replacement_ids)
    expected = (
        torch.nn.functional.linear(perturbed + 0.5 - base, model.lm_head.weight)
        .abs()
        .mean(dim=-1)
    )

    assert model.pre_called is True
    assert model.logits_called is False
    assert calls["native"] == 1
    assert torch.allclose(combined, expected, atol=1e-6, rtol=1e-6)


def test_capture_probe_representations_prefers_model_fast_path():
    class _TinyRepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def _fingerprint_representations(self, input_ids):
            self.called = True
            reps = input_ids.float().unsqueeze(-1) + 2.0
            logits = reps + 3.0
            return logits, reps

        def forward(self, _input_ids):
            raise RuntimeError("fallback forward should not run")

    model = _TinyRepModel()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    captured = _capture_probe_representations(model, ids)

    assert model.called is True
    assert captured is not None
    assert torch.allclose(captured.logits, ids.float().unsqueeze(-1) + 5.0)
    assert captured.reps is not None
    assert torch.allclose(captured.reps, ids.float().unsqueeze(-1) + 2.0)


def test_capture_probe_representations_falls_back_to_embedding_output():
    class _TinyFallbackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 8)
            self.linear = nn.Linear(8, 32)

        def forward(self, input_ids):
            return self.linear(self.embed(input_ids))

    model = _TinyFallbackModel()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    captured = _capture_probe_representations(model, ids)

    assert captured is not None
    assert captured.logits.shape == (1, 3, 32)
    assert captured.reps is not None
    assert captured.reps.shape == (1, 3, 8)
    assert torch.allclose(captured.reps, model.embed(ids))
