"""Policies — promotion + halting rules for the autonomous fab loop."""

from .promotion import (
    DEFAULT_PROMOTION_RULES,
    PromotionDecision,
    PromotionRules,
    decide_promotion,
    decide_promotions_for_ledger,
)

__all__ = [
    "DEFAULT_PROMOTION_RULES",
    "PromotionDecision",
    "PromotionRules",
    "decide_promotion",
    "decide_promotions_for_ledger",
]
