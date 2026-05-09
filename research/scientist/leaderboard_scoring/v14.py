"""v14 — Tier-progressive language-control probe ladder (2026-05-02).

Replaces real BLiMP/HellaSwag noise at screening tier with a calibrated
nano-scale association probe. Three difficulty tiers credit progressively
more points as the test gets harder.

Calibration data (top-30 leaderboard cohort, n=30, see
research/reports/nano_probe_audit_top30.json):
  v120/40 (S0.5): synthetic_assoc median 1.00, 67% saturate → basic floor
  v200/40 (S1.0): synthetic_assoc median 1.00, 47% saturate → real diff
  v300/40 (Inv):  synthetic_assoc median 0.94, 13% saturate → sharp

Anchors set at the cohort median per tier so half the cohort earns half
the weight. nano_blimp.order_grammaticality_acc has the richest dynamic
range (cohort std ~0.30) and is the primary nano_blimp signal.

Real BLiMP weight stays at 5pt floor (validated by champion-mode test:
12L/10K-step training moves BLiMP only 1.2pp inside the 0.013 noise band).
Real HellaSwag weight drops 15→5 (cohort spread 0.030, ρ vs composite
+0.088 — near-noise at screening eval scale).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ._config import _V14_CONFIG
from ._utils import _scurve_higher_better
from .v12 import compute_composite_v12


_CL_TIER_BD_KEYS = (
    "cl_s05_sa",
    "cl_s05_order",
    "cl_s10_sa",
    "cl_s10_order",
    "cl_s10_nb_bucket",
    "cl_investigation_sa",
    "cl_investigation_order",
    "cl_investigation_nb_bucket",
)


def _controlled_nb_bucket_fraction(value: Optional[float]) -> float:
    """Discrete bucket score for rank-order controlled NanoBind signals."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.65:
        return 0.0
    if v < 0.75:
        return 0.25
    if v < 0.85:
        return 0.50
    if v < 0.95:
        return 0.75
    return 1.0


def _score_language_control_tier(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    kw: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    """Score the language-control probe ladder.

    S0.5 remains a small learning floor. S1.0/INV keep the calibrated SA/order
    S-curves and add rank-order NanoBind buckets:
    <.65=0, .65-.75=25%, .75-.85=50%, .85-.95=75%, >=.95=100%.
    inv_failed rows zero out the tier.
    """
    bd: Dict[str, float] = {k: 0.0 for k in _CL_TIER_BD_KEYS}
    if inv_failed:
        return 0.0, bd
    pairs = (
        (
            "cl_s05_sa",
            kw.get("language_control_s05_sentence_assoc_score"),
            cfg["cl_s05_sa_anchor"],
        ),
        (
            "cl_s05_order",
            kw.get("language_control_s05_binding_order_acc"),
            cfg["cl_s05_order_anchor"],
        ),
        (
            "cl_s10_sa",
            kw.get("language_control_s10_sentence_assoc_score"),
            cfg["cl_s10_sa_anchor"],
        ),
        (
            "cl_s10_order",
            kw.get("language_control_s10_binding_order_acc"),
            cfg["cl_s10_order_anchor"],
        ),
        (
            "cl_investigation_sa",
            kw.get("language_control_investigation_sentence_assoc_score"),
            cfg["cl_investigation_sa_anchor"],
        ),
        (
            "cl_investigation_order",
            kw.get("language_control_investigation_binding_order_acc"),
            cfg["cl_investigation_order_anchor"],
        ),
    )
    total = 0.0
    for key, value, anchor in pairs:
        weight = float(cfg.get(f"w_{key}", 0.0))
        pts = weight * _scurve_higher_better(value, anchor)
        bd[key] = pts
        total += pts
    bucket_pairs = (
        ("cl_s10_nb_bucket", kw.get("language_control_s10_binding_score")),
        (
            "cl_investigation_nb_bucket",
            kw.get("language_control_investigation_binding_score"),
        ),
    )
    for key, value in bucket_pairs:
        weight = float(cfg.get(f"w_{key}", 0.0))
        pts = weight * _controlled_nb_bucket_fraction(value)
        bd[key] = pts
        total += pts
    return total, bd


def compute_composite_v14(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v14: v12 + tier-progressive language-control ladder.

    Adds 45pt of nano-scale BLiMP/HellaSwag replacement signal across
    three difficulty tiers (S0.5/S1.0/Investigation). Real BLiMP and
    HellaSwag keep 10pt floors for champion-length hard-probe reruns.
    """
    base = compute_composite_v12(decompose=True, **kw)
    score = float(base["composite_score"])
    bd = base.get("breakdown") or {}

    tier = kw.get("tier")
    inv_failed = tier in ("investigation_failed", "screened_out")
    cl_pts, cl_bd = _score_language_control_tier(
        _V14_CONFIG, inv_failed=inv_failed, kw=kw
    )
    score += cl_pts
    bd.update(cl_bd)
    bd["_v14_language_control_total"] = cl_pts

    if decompose:
        base["composite_score"] = score
        base["breakdown"] = bd
        return base
    return score
