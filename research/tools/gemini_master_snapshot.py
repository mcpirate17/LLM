"""Compatibility shim — the Universal Master lane has a single source of truth.

The canonical, verified implementation (causal read, differentiable soft routing,
content-addressed read) lives in ``component_fab.generator.memory_primitives``.
This module is kept only so historical importers (e.g.
``research.tools.grade_matched_corrected``) keep working; it must not host a
divergent copy of the lane again.
"""

from component_fab.generator.memory_primitives import UniversalMasterLane

__all__ = ["UniversalMasterLane"]
