"""Behavioral-novelty and Pareto-front metadata for autonomous fab survivors."""

from __future__ import annotations

from typing import Any

from component_fab.improver.ranking import objective_vector, pareto_front_indices
from component_fab.metrics.behavior_fingerprint import (
    FRONTIER_SPECTRA,
    Normalizer,
    is_degenerate,
    operational_spectrum,
    orthogonality_radius,
    spectrum_from_metadata,
)
from component_fab.state.ledger import Ledger


def annotate_niche_metadata(survivors: list[dict[str, Any]], ledger: Ledger) -> None:
    """Attach operational-spectrum orthogonality + Pareto-front membership.

    Computed over the whole survivor set (Pareto + orthogonality are relative).
    Orthogonality is measured against BOTH the existing ledger catalog (clone
    detection) and the frontier baselines (softmax/gpt2/mamba2) — a candidate
    must be far from both to earn orthogonality credit, so a softmax-twin is
    penalized, not just an intra-population clone. The front is computed within
    this cycle's survivors. Shared by the autonomous runner and tests.
    """

    catalog = [
        spectrum_from_metadata(entry.metadata_history[-1])
        for entry in ledger.all_entries()
        if entry.metadata_history
    ]
    catalog.extend(FRONTIER_SPECTRA.values())
    spectra = [
        operational_spectrum(survivor["probe"], survivor["capability"])
        for survivor in survivors
    ]
    normalizer = Normalizer.fit(catalog + spectra)
    for survivor, spectrum in zip(survivors, spectra):
        radius = orthogonality_radius(spectrum, catalog, normalizer=normalizer)
        finite = radius != float("inf")
        survivor["metadata"]["operational_spectrum"] = spectrum
        survivor["metadata"]["orthogonality_radius"] = radius if finite else -1.0
        survivor["metadata"]["state_degenerate"] = (
            bool(is_degenerate(radius)) if finite else False
        )

    vectors = [
        objective_vector(
            survivor["probe"],
            survivor["capability"],
            orthogonality=max(0.0, survivor["metadata"]["orthogonality_radius"]),
        )
        for survivor in survivors
    ]
    front = set(pareto_front_indices(vectors))
    for index, (survivor, vector) in enumerate(zip(survivors, vectors)):
        survivor["metadata"]["pareto_objective_vector"] = dict(vector)
        survivor["metadata"]["on_pareto_front"] = index in front
