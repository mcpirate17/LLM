"""Small registries used by ``component_fab.generator.code_generator``."""

from .registry import DispatchRule, dispatch_first
from .slots import (
    UnknownBlockSlotError,
    build_block_slot_factory,
    known_partner_kinds,
    slot_name_for_partner_kind,
)

__all__ = [
    "DispatchRule",
    "UnknownBlockSlotError",
    "build_block_slot_factory",
    "dispatch_first",
    "known_partner_kinds",
    "slot_name_for_partner_kind",
]
