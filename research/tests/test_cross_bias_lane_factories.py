from __future__ import annotations

import torch

from research.tools.ensemble_screening import _load_graphs_by_fingerprint
from research.tools.scaling_blimp_study import _build_lane_factory


def test_load_graphs_by_fingerprint_loads_local_ssm_diff_sources() -> None:
    specs = _load_graphs_by_fingerprint(
        (
            (
                "bb0b8d5856da1f29",  # pragma: allowlist secret
                "local_window + conv + selective_scan + diff_attention",
                0.7975,
            ),
            (
                "5c5013c79d1f0a51",  # pragma: allowlist secret
                "local_window + conv + selective_scan + diff_attention alt",
                0.4069,
            ),
        )
    )

    assert [spec[0] for spec in specs] == [
        "bb0b8d5856da1f29",  # pragma: allowlist secret
        "5c5013c79d1f0a51",  # pragma: allowlist secret
    ]
    assert [spec[3].model_dim for spec in specs] == [256, 256]
    assert [len(spec[3].nodes) for spec in specs] == [15, 15]


def test_local_ssm_diff_lane_factory_forward_cpu() -> None:
    factory = _build_lane_factory("local_ssm_diff")
    lane = factory(32)

    x = torch.randn(1, 3, 32)
    with torch.no_grad():
        y = lane(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
