import pytest
import torch
import torch.nn as nn

from research.eval import fingerprint_native
from research.eval.fingerprint_runtime import analyze_interactions

pytestmark = pytest.mark.unit


def test_interaction_metrics_fallback_without_aria_core(monkeypatch):
    monkeypatch.setattr(fingerprint_native, "aria_core", None)
    influence = torch.eye(4, dtype=torch.float32)
    positions = torch.arange(4, dtype=torch.int64)

    metrics = fingerprint_native.interaction_metrics(influence, positions)

    assert metrics["locality"] == pytest.approx(1.0)
    assert metrics["sparsity"] == pytest.approx(0.5)
    assert metrics["symmetry"] == pytest.approx(0.5)
    assert metrics["hierarchy"] == pytest.approx(0.0)


def test_interaction_metrics_uses_native_when_available(monkeypatch):
    class FakeAriaCore:
        def interaction_metrics_f32(self, influence, positions):
            assert influence.device.type == "cpu"
            assert influence.is_contiguous()
            assert positions.dtype == torch.int64
            return torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    monkeypatch.setattr(fingerprint_native, "aria_core", FakeAriaCore())

    metrics = fingerprint_native.interaction_metrics(
        torch.ones(2, 3, dtype=torch.float64),
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert metrics == {
        "locality": pytest.approx(0.1),
        "sparsity": pytest.approx(0.2),
        "symmetry": pytest.approx(0.3),
        "hierarchy": pytest.approx(0.4),
    }


def test_analyze_interactions_succeeds_without_aria_core(monkeypatch):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def forward(self, input_ids):
            return self.lm_head(self.embed(input_ids))

    monkeypatch.setattr(fingerprint_native, "aria_core", None)
    model = TinyModel()

    metrics = analyze_interactions(
        model,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
        torch.device("cpu"),
        seq_len=4,
        vocab_size=8,
    )

    assert metrics["_succeeded"] is True
    assert set(metrics) == {
        "locality",
        "sparsity",
        "symmetry",
        "hierarchy",
        "_succeeded",
    }
