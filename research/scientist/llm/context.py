"""Context builders compatibility facade.

Implementations live in split modules. This facade keeps the historical
``research.scientist.llm.context`` import path stable without relying on
module-wide star imports.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ._op_registry import grouped_primitive_registry, primitive_registry_size
from .context_briefing import build_briefing_context
from .context_experiment import (
    build_experiment_context,
    build_go_no_go_context,
    build_history_context,
    build_investigation_context,
    build_manual_start_fallback_context,
    build_mode_selection_context,
    build_op_reference,
    build_program_context,
    build_rich_context,
    build_validation_context,
    inject_digest_context,
)
from .context_hypothesis import (
    build_campaign_formulation_context,
    build_campaign_report_context,
    build_hypothesis_context,
    build_knowledge_extraction_context,
)

# Preserve the legacy module-level logger name exported by the old star-import
# facade. Some tests import the facade only to ensure the historical surface
# still loads cleanly.
logger = logging.getLogger(__name__)

__all__ = [
    "Dict",
    "List",
    "Optional",
    "annotations",
    "build_briefing_context",
    "build_campaign_formulation_context",
    "build_campaign_report_context",
    "build_experiment_context",
    "build_go_no_go_context",
    "build_history_context",
    "build_hypothesis_context",
    "build_investigation_context",
    "build_knowledge_extraction_context",
    "build_manual_start_fallback_context",
    "build_mode_selection_context",
    "build_op_reference",
    "build_program_context",
    "build_rich_context",
    "build_validation_context",
    "grouped_primitive_registry",
    "inject_digest_context",
    "logger",
    "logging",
    "primitive_registry_size",
    "re",
]
