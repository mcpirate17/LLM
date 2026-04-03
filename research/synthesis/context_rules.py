"""
Component Context Rules — Enforcement Layer

Encodes placement constraints for ops that have been audited as
context-sensitive. These rules prevent the grammar from generating
graphs where ops appear in invalid predecessor/successor chains,
and classify ops by search-mode so niche/restricted ops are not
sprayed into default search blindly.

Sources:
  - artifacts/component_context_rules.md
  - artifacts/component_context_rules.json
  - artifacts/low_s1_root_cause_audit.md

Split into submodules for maintainability:
  _context_types.py      — SearchMode, ContextRule, context class constants
  _context_op_sets.py    — Shared frozen op sets
  _context_registry.py   — CONTEXT_RULES dict, derived sets, is_structural
  _context_motifs.py     — Motif allowlists, context class priors, motif helpers
  _context_validation.py — Graph validation and byte-safety enforcement
"""

from __future__ import annotations

# Re-export all public symbols so existing imports keep working.

from ._context_types import (
    CONTEXT_CLASS_GENERAL,
    CONTEXT_CLASS_REHAB,
    CONTEXT_CLASS_RESTRICTED,
    CONTEXT_CLASS_STRUCTURAL,
    ContextRule,
    SearchMode,
)

from ._context_registry import (
    CONTEXT_RULES,
    REQUIRES_RESIDUAL_CONTEXT,
    S1_EXEMPT_OPS,
    STRUCTURAL_OPS,
    is_structural,
)

from ._context_motifs import (
    apply_context_rule_priors,
    get_op_context_class,
    motif_allowed_in_template,
)

from ._context_validation import (
    find_byte_safety_violations,
    find_graph_context_violations,
    validate_context_rules,
)

__all__ = [
    # Types
    "SearchMode",
    "ContextRule",
    # Constants
    "CONTEXT_CLASS_GENERAL",
    "CONTEXT_CLASS_RESTRICTED",
    "CONTEXT_CLASS_STRUCTURAL",
    "CONTEXT_CLASS_REHAB",
    # Registry
    "CONTEXT_RULES",
    "STRUCTURAL_OPS",
    "S1_EXEMPT_OPS",
    "REQUIRES_RESIDUAL_CONTEXT",
    # Query helpers
    "is_structural",
    "get_op_context_class",
    "apply_context_rule_priors",
    "motif_allowed_in_template",
    # Validation
    "find_graph_context_violations",
    "validate_context_rules",
    "find_byte_safety_violations",
]
