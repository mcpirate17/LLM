import torch

from research.eval.fingerprint import _analyze_geometry


def test_analyze_geometry_returns_valid_metrics():
    torch.manual_seed(0)
    reps = torch.randn(4, 10, 12, dtype=torch.float32)

    result = _analyze_geometry(reps)

    assert result["_succeeded"] is True
    assert result["intrinsic_dim"] >= 1.0
    assert 0.0 <= result["isotropy"] <= 1.0
    assert 0.0 <= result["rank_ratio"] <= 1.0
