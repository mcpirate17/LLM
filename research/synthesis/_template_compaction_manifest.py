"""Manifest for NM-C compaction mixer templates (Tier D program)."""

from ._templates_compaction import (
    tpl_compaction_mixer_block,
    tpl_compaction_routing_block,
)

COMPACTION_TEMPLATE_REGISTRY = {
    "compaction_mixer_block": tpl_compaction_mixer_block,
    "compaction_routing_block": tpl_compaction_routing_block,
}

# Novel-mixer priority weight (matches the NOVEL_MIXER family): two templates
# cover 11 NM-C ops, so per-op exposure is weight/pool per draw — kept high so
# the discovery loop actually exercises the compaction lanes. The routing
# variant is additionally slot-0 eligible under routing_mandatory.
COMPACTION_TEMPLATE_DEFAULT_WEIGHTS = {
    "compaction_mixer_block": 15.0,
    "compaction_routing_block": 15.0,
}
