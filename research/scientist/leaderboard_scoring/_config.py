"""Loaded scoring configs and shared tier constants.

v10/v11/v14 tunable knobs come from research/scoring_config.yaml so tuning
is a config edit, not a code patch. The three layers are kept distinct
because v11 applies a multiplicative weight rescale on top of v10 (see
v11.compute_composite_v11) — flattening would change scoring semantics.

Lives apart from version modules so champion_tiny ↔ v12 (which both need
``_V12_CHAMPION_ELIGIBILITY_CEILING``) and v10/v11/v12 (which all reach for
the tier-key tuples and the trust-ceiling floors) avoid circular imports.
"""

from __future__ import annotations

from typing import Dict

from .. import scoring_config as _scoring_config

_ = _scoring_config  # silence unused-import warning; consumed below at module scope


# Hard gates applied at the pre-investigation gate (notebook_references
# .get_investigation_eligible). Below these floors the architecture cannot
# route information and shouldn't reach investigation regardless of
# composite score. Tuned from smoke data: SSMs at init have erf_density≈0.3
# so the floor must be below that to avoid gating Mamba/RWKV-class graphs.
# Kept across the v9 retirement (2026-05-03) because notebook_references
# imports them by name.
GEMINI_HARD_GATE_ERF_DENSITY = 0.20
GEMINI_HARD_GATE_ERF_VARIANCE = 800.0


_V10_CONFIG: Dict[str, float] = _scoring_config.get_layered_configs()["v10"]
_V11_CONFIG: Dict[str, float] = _scoring_config.get_layered_configs()["v11"]
_V14_CONFIG: Dict[str, float] = _scoring_config.get_layered_configs()["v14"]


# Trust-ceiling thresholds (champion-range eligibility gates) — externalized
# to research/scoring_config.yaml::trust_ceiling.
_TRUST = _scoring_config.get_trust_ceiling()
_V11_TRUST_CEILING = float(_TRUST["ceiling"])
_V11_TRUST_PPL_FLOOR = float(_TRUST["ppl_floor"])
_V11_TRUST_HELLASWAG_FLOOR = float(_TRUST["hellaswag_floor"])
_V11_TRUST_BLIMP_FLOOR = float(_TRUST["blimp_floor"])
_V11_TRUST_INDUCTION_FLOOR = float(_TRUST["induction_floor"])
_V11_TRUST_BINDING_FLOOR = float(_TRUST["binding_floor"])


# Lives here (not in v12) because champion_tiny._apply_champion_tiny_model_hard_failure_gate
# also reads it; v12 imports champion_tiny, so the constant cannot live in v12 without
# creating a cycle.
_V12_CHAMPION_ELIGIBILITY_CEILING = 360.0


# Tier keys used by v10's CV penalty applicator, v11's breakthrough boost,
# and v12's loss-budget rebalance. Centralized to keep the three versions
# in sync.
_LOSS_TIER_BD_KEYS = (
    "perf_short",
    "perf_medium",
    "perf_long",
    "param_efficiency",
    "learning_efficiency",
    "early_convergence",
    "speed",
)
_UND_TIER_BD_KEYS = (
    "blimp",
    "tinystories",
    "cross_task",
    "diagnostic",
    "hellaswag",
    "hierarchy",
)


# Single canonical formula. Tunable knobs live in research/scoring_config.yaml;
# the dispatcher across v7-v14 was retired because every rescore overwrote
# the version stamp, making it illusory provenance. The replacement is a
# SHA256 digest of the YAML bytes — when the config changes, the hash
# changes, and rescored rows pick up the new stamp.
ACTIVE_SCORING_VERSION: str = "v14"
