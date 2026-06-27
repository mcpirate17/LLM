"""Behavioral-novelty and Pareto-front metadata for autonomous fab survivors."""

from __future__ import annotations

from typing import Any

from component_fab.improver.ranking import objective_vector, pareto_front_indices
from component_fab.metrics.behavior_fingerprint import (
    Normalizer,
    behavior_fingerprint,
    fingerprint_from_metadata,
    is_clone,
    novelty_distance,
)
from component_fab.state.ledger import Ledger


def annotate_niche_metadata(survivors: list[dict[str, Any]], ledger: Ledger) -> None:
    """Attach behavior novelty and Pareto-front membership to survivor rows.

    This is shared by the autonomous runner and tests. Keeping it out of the
    grading loop stops the grade path from owning behavioral-novelty math.
    """

    catalog = [
        fingerprint_from_metadata(entry.metadata_history[-1])
        for entry in ledger.all_entries()
        if entry.metadata_history
    ]
    fingerprints = [
        behavior_fingerprint(survivor["probe"], survivor["capability"])
        for survivor in survivors
    ]
    normalizer = Normalizer.fit(catalog + fingerprints)
    for survivor, fingerprint in zip(survivors, fingerprints):
        distance = novelty_distance(fingerprint, catalog, normalizer=normalizer)
        finite = distance != float("inf")
        survivor["metadata"]["behavior_fingerprint"] = fingerprint
        survivor["metadata"]["novelty_distance"] = distance if finite else -1.0
        survivor["metadata"]["behavior_clone"] = (
            bool(is_clone(distance)) if finite else False
        )

    vectors = [
        objective_vector(
            survivor["probe"],
            survivor["capability"],
            novelty=max(0.0, survivor["metadata"]["novelty_distance"]),
        )
        for survivor in survivors
    ]
    front = set(pareto_front_indices(vectors))
    for index, (survivor, vector) in enumerate(zip(survivors, vectors)):
        survivor["metadata"]["pareto_objective_vector"] = dict(vector)
        survivor["metadata"]["on_pareto_front"] = index in front
