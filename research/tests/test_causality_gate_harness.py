import torch
import torch.nn as nn
from research.eval.sandbox import safe_eval


class CausalCopyModel(nn.Module):
    """Causal model: output depends only on current token embedding."""

    def __init__(self, vocab_size=64, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        return self.proj(h)


class LookAheadModel(nn.Module):
    """Non-causal: logits at t use token at t+1 (future)."""

    def __init__(self, vocab_size=64, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # Shift left: position t sees token t+1 (look-ahead)
        shifted = torch.roll(x, shifts=-1, dims=1)
        h = self.embed(shifted)
        return self.proj(h)


def test_causality_gate_passes_causal_model():
    model = CausalCopyModel()
    result = safe_eval(
        model,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        device="cpu",
        run_stability_probe=True,
    )
    assert result.causality_passed, f"Expected causal pass, got {result.error_type}"


def test_causality_gate_blocks_lookahead_model():
    model = LookAheadModel()
    result = safe_eval(
        model,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        device="cpu",
        run_stability_probe=True,
    )
    assert not result.causality_passed
    assert result.error_type == "causality_violation"
