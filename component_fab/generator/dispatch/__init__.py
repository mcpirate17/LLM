"""Small registries used by ``component_fab.generator.code_generator``."""

from .registry import DispatchRule, dispatch_first
from .slots import UnknownBlockSlotError, build_block_slot_factory

__all__ = [
    "DispatchRule",
    "UnknownBlockSlotError",
    "build_block_slot_factory",
    "dispatch_first",
]
