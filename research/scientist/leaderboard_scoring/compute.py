"""Public dispatcher: ``compute_composite``, ``composite_score_ceiling``,
``get_scoring_version``.

Single canonical formula (v14). The dispatcher across v7-v14 was retired
because every rescore overwrote the version stamp, making it illusory
provenance. The replacement is a SHA256 digest of the YAML bytes — when
the config changes, the hash changes, and rescored rows pick up the new
stamp.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from .. import scoring_config as _scoring_config
from ._config import _V14_CONFIG
from .v14 import compute_composite_v14


def composite_score_ceiling(version: str | None = None) -> float:
    """Return the theoretical maximum composite score under the active formula.

    Derived from the v14 weight config so UI scale ceilings stay in sync
    with the scorer. ``version`` is accepted for backwards compatibility
    with callers from the multi-version era; it is ignored.
    """
    del version  # Single-version era; argument retained for API stability.
    cfg = _V14_CONFIG
    base_max = (
        cfg["w_perf_short"]
        + cfg["w_perf_medium"]
        + cfg["w_perf_long"]
        + cfg["w_param_eff"]
        + cfg["w_learn_eff"]
        + 50.0  # routing_savings
        + 30.0  # compression
        + 30.0  # sparsity
        + 25.0  # adaptive_computation
        + 40.0  # novelty
        + 15.0  # ncd
        + 40.0  # robustness
        + 25.0  # long_context
        + cfg["w_binding"]
        + cfg.get("w_blimp", 40.0)
        + cfg["w_tinystories"]
        + cfg["w_cross_task"]
        + cfg["w_diagnostic"]
        + cfg["w_hellaswag"]
        + cfg["w_hierarchy"]
        + 25.0  # speed
        + 10.0  # early_convergence
        + cfg["w_cap_ar"]
        + cfg.get("w_cap_ar_validation_validation", 0.0)
        + cfg.get("w_legacy_ar", 0.0)
        + cfg["w_cap_induction"]
        + cfg["w_cap_binding"]
        + cfg["w_cap_erf_density"]
        + cfg["w_cap_id_collapse"]
        + cfg["w_cap_erf_decay"]
        + cfg["w_cap_logit_margin"]
        + cfg["w_aux_erf_variance"]
        + cfg["w_aux_icld"]
        # Language-control ladder (v14 addition)
        + cfg["w_cl_s05_sa"]
        + cfg["w_cl_s05_order"]
        + cfg.get("w_cl_s05_nb_bucket", 0.0)
        + cfg["w_cl_s10_sa"]
        + cfg["w_cl_s10_order"]
        + cfg.get("w_cl_s10_nb_bucket", 0.0)
        + cfg["w_cl_investigation_sa"]
        + cfg["w_cl_investigation_order"]
        + cfg.get("w_cl_investigation_nb_bucket", 0.0)
    )
    return float(base_max)


def get_scoring_version() -> str:
    """Return the active scoring config hash (12-char SHA256 prefix).

    Replaces the multi-version era's ``v7``/``v8``/.../``v14`` strings with
    real provenance: "this row was scored under config matching hash X."
    Reload via ``scoring_config.reload_scoring_config()`` after editing the
    YAML; the hash will rotate.
    """
    return _scoring_config.get_scoring_config_hash()


def compute_composite(
    *, decompose: bool = False, **kw: Any
) -> Union[float, Dict[str, Any]]:
    """Compute the leaderboard composite score under the active formula."""
    return compute_composite_v14(decompose=decompose, **kw)
