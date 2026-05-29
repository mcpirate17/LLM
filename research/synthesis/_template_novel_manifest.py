"""Manifest for novel mixer templates."""

from ._templates_novel_mixers import (
    tpl_clifford_geometric_mixer_block,
    tpl_tropical_maxplus_mixer_block,
    tpl_ultrametric_hierarchical_ensemble_block,
)

# Template Registry mappings
NOVEL_MIXER_TEMPLATE_REGISTRY = {
    "clifford_geometric_mixer_block": tpl_clifford_geometric_mixer_block,
    "tropical_maxplus_mixer_block": tpl_tropical_maxplus_mixer_block,
    "ultrametric_hierarchical_ensemble_block": tpl_ultrametric_hierarchical_ensemble_block,
}

# Recommendation weights (S1 priority)
NOVEL_MIXER_TEMPLATE_DEFAULT_WEIGHTS = {
    "clifford_geometric_mixer_block": 15.0,
    "tropical_maxplus_mixer_block": 15.0,
    "ultrametric_hierarchical_ensemble_block": 15.0,
}
