from __future__ import annotations

from research.tools.data_route_ab import _ROUTE_CONDITIONS, _summarize


def _result(condition: str, val_loss: float, top1: float) -> dict:
    return {
        "condition": condition,
        "final": {"val_loss": val_loss, "top1_acc": top1},
        "curve": [
            {"step": 0, "val_loss": val_loss + 1.0},
            {"step": 10, "val_loss": val_loss},
        ],
    }


def test_route_conditions_exclude_segment_route_noops() -> None:
    assert "surprisal_split_30" not in _ROUTE_CONDITIONS
    assert "local_global_30" not in _ROUTE_CONDITIONS
    assert "fold16_vertical_alternate" in _ROUTE_CONDITIONS
    assert "doc_boundary" in _ROUTE_CONDITIONS
    assert all(spec.route == "none" for spec in _ROUTE_CONDITIONS.values())


def test_summary_uses_seed_robust_medians() -> None:
    summary = _summarize(
        [
            _result("natural", 1.0, 0.10),
            _result("natural", 2.0, 0.20),
            _result("natural", 99.0, 0.90),
            _result("fold", 0.9, 0.25),
            _result("fold", 1.5, 0.35),
            _result("fold", 90.0, 0.80),
        ]
    )

    assert summary["baseline_natural_median_val_loss"] == 2.0
    assert summary["baseline_natural_median_top1"] == 0.2
    fold = summary["by_condition"]["fold"]
    assert fold["median_final_val_loss"] == 1.5
    assert fold["median_final_top1"] == 0.35
    assert fold["delta_median_val_loss_vs_natural"] == -0.5
    assert fold["delta_median_top1_vs_natural"] == 0.15
