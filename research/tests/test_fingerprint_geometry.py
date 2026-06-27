import pytest
import torch

# `_analyze_geometry` was renamed to public `analyze_geometry` and moved
# from `fingerprint` to `fingerprint_probes` during the module split.
from research.eval.fingerprint_runtime import analyze_geometry

pytestmark = pytest.mark.unit


def test_analyze_geometry_returns_valid_metrics():
    torch.manual_seed(0)
    reps = torch.randn(4, 10, 12, dtype=torch.float32)

    result = analyze_geometry(reps)

    # `_succeeded` is no longer part of the surface; results are returned
    # unconditionally as dict[str, float].
    assert result["intrinsic_dim"] >= 1.0
    assert 0.0 <= result["isotropy"] <= 1.0
    assert 0.0 <= result["rank_ratio"] <= 1.0
