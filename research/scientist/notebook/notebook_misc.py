"""Composition of misc LabNotebook mixins.

Previously a 2,953-line god file. Split into four focused mixins by
concern; this module is now the composition point so `LabNotebook`'s
import path via ``notebook/__init__.py`` is unchanged.
"""

from __future__ import annotations

from .notebook_dashboard import _DashboardNBMixin
from .notebook_entries import _EntriesMixin
from .notebook_observability import _ObservabilityMixin
from .notebook_references import _ReferencesMixin


class _MiscMixin(
    _ObservabilityMixin,
    _EntriesMixin,
    _DashboardNBMixin,
    _ReferencesMixin,
):
    """Misc operations for the Lab Notebook (composed)."""

    __slots__ = ()
