"""Experiment analytics mixin — composed from sub-modules.

All public methods remain importable from this module via _ExperimentsMixin.
"""

from __future__ import annotations

from ._exp_weights import _WeightsMixin
from ._exp_gate_structure import _GateStructureMixin
from ._exp_clustering import _ClusteringMixin
from ._exp_comparisons import _ComparisonsMixin
from ._exp_coverage import _CoverageMixin
from ._exp_insights import _InsightsMixin


class _ExperimentsMixin(
    _WeightsMixin,
    _GateStructureMixin,
    _ClusteringMixin,
    _ComparisonsMixin,
    _CoverageMixin,
    _InsightsMixin,
):
    """Experiment clustering, correlations, insights, and math coverage."""

    __slots__ = ()
