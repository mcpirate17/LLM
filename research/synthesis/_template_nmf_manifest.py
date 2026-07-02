"""Manifest for NM-F operator-family templates."""

from ._templates_nmf import tpl_nmf_mixer_block, tpl_nmf_routing_block

NMF_TEMPLATE_REGISTRY = {
    "nmf_mixer_block": tpl_nmf_mixer_block,
    "nmf_routing_block": tpl_nmf_routing_block,
}

# Novel-mixer priority weight (matches the NOVEL_MIXER / COMPACTION families).
# The routing variant is additionally slot-0 eligible under routing_mandatory.
NMF_TEMPLATE_DEFAULT_WEIGHTS = {
    "nmf_mixer_block": 15.0,
    "nmf_routing_block": 15.0,
}
