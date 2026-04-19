"""
Novelty Metrics

Information-theoretic and structural metrics for evaluating
how novel a synthesized program actually is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Novelty Metric Constants
EXPECTED_CATEGORIES = 8.0
WEIGHT_DIVERSITY = 0.50
WEIGHT_SPREAD = 0.30
WEIGHT_EVENNESS = 0.20
EXOTIC_BONUS_BASE = 1.0
EXOTIC_BONUS_PER_FLAG = 0.1

STRUCTURAL_BLEND_WEIGHT = 0.3
BEHAVIORAL_BLEND_WEIGHT = 0.7
STRUCTURAL_ONLY_WEIGHT = 0.6

CONFIDENCE_FULL = 0.9
CONFIDENCE_PARTIAL_BASE = 0.4
CONFIDENCE_PARTIAL_STEP = 0.1
CONFIDENCE_NONE = 0.3
CONFIDENCE_NO_FP = 0.2

DUPLICATE_PENALTY_MULTIPLIER = 0.1
from ..synthesis.graph import ComputationGraph
from .fingerprint_types import BehavioralFingerprint


@dataclass(slots=True)
class NoveltyMetrics:
    """Comprehensive novelty assessment."""

    # Structural novelty
    graph_fingerprint: str = ""
    n_unique_ops: int = 0
    uses_math_spaces: bool = False
    uses_frequency_domain: bool = False
    structural_novelty: float = 0.0  # 0-1

    # Behavioral novelty (from fingerprint)
    behavioral_novelty: float = 0.0  # 0-1
    max_cka_similarity: float = 0.0
    most_similar_to: str = ""

    # Combined score
    raw_novelty: float = 0.0
    overall_novelty: float = 0.0
    novelty_z_score: Optional[float] = None
    novelty_reference_version: Optional[str] = None
    novelty_valid_for_promotion: bool = False
    novelty_validity_reason: str = "missing_reference"
    novelty_confidence: float = 0.0

    # Decomposition
    op_histogram: Dict[str, int] = field(default_factory=dict)
    category_histogram: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}


def _reference_similarity_penalty(max_cka_similarity: float) -> float:
    """Penalty factor for reference-like behavior.

    Higher CKA similarity to known reference families should reduce novelty.
    Returns a multiplicative factor in [0.25, 1.0].
    """
    try:
        sim = float(max_cka_similarity)
    except (TypeError, ValueError):
        return 1.0
    sim = max(0.0, min(1.0, sim))
    return max(0.25, 1.0 - 0.75 * sim)


def novelty_score(
    graph: ComputationGraph,
    fingerprint: Optional[BehavioralFingerprint] = None,
    known_fingerprints: Optional[List[str]] = None,
    calibration: Optional[Dict[str, float]] = None,
) -> NoveltyMetrics:
    """Compute novelty metrics for a synthesized program.

    Delegates to ``batch_novelty_scores`` for the core computation to
    avoid duplicated logic between single and batch paths.
    """
    ir = graph.lower_to_ir()
    if ir.is_stale(graph):
        logger.warning(
            "stale_ir_used: graph_version=%d ir_version=%d",
            graph._ir_version,
            ir.source_version,
        )

    fps = [fingerprint] if fingerprint is not None else None
    metrics = batch_novelty_scores([graph], fps)[0]

    # Check against known fingerprints
    if known_fingerprints and metrics.graph_fingerprint in known_fingerprints:
        metrics.overall_novelty *= (
            DUPLICATE_PENALTY_MULTIPLIER  # Heavily penalize exact duplicates
        )

    if calibration:
        mean = calibration.get("noise_floor_mean")
        std = calibration.get("noise_floor_std")
        try:
            if mean is not None and std is not None and float(std) > 1e-8:
                metrics.novelty_z_score = (
                    float(metrics.raw_novelty) - float(mean)
                ) / float(std)
        except (TypeError, ValueError):
            metrics.novelty_z_score = None

    return metrics


def _opcode_category_maps():
    from ..synthesis.primitives import OPCODE_MAP, get_primitive

    opcode_to_cat = {}
    for name, code in OPCODE_MAP.items():
        if name == "input":
            continue
        try:
            opcode_to_cat[code] = get_primitive(name).category.value
        except (AttributeError, KeyError, ValueError) as exc:
            logger.debug("Novelty metric category lookup failed for %s: %s", name, exc)

    all_categories = sorted(set(opcode_to_cat.values()))
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    return opcode_to_cat, cat_to_idx


def _category_counts(
    batch_counts: np.ndarray,
    opcode_to_cat: Dict[int, str],
    cat_to_idx: Dict[str, int],
) -> np.ndarray:
    cat_counts = np.zeros((batch_counts.shape[0], len(cat_to_idx)), dtype=np.int32)
    for code, cat in opcode_to_cat.items():
        cat_counts[:, cat_to_idx[cat]] += batch_counts[:, code]
    return cat_counts


def _structural_novelty_components(
    graphs: List[ComputationGraph],
    batch_counts: np.ndarray,
):
    from ..synthesis.primitives import PRIMITIVE_REGISTRY

    opcode_to_cat, cat_to_idx = _opcode_category_maps()
    n_ops_per_graph = batch_counts.sum(axis=1)
    unique_ops_per_graph = (batch_counts > 0).sum(axis=1)
    total_available = max(len(PRIMITIVE_REGISTRY), 1)
    diversity = np.clip(unique_ops_per_graph / total_available, 0, 1.0)

    cat_counts = _category_counts(batch_counts, opcode_to_cat, cat_to_idx)
    unique_cats_per_graph = (cat_counts > 0).sum(axis=1)
    category_spread = np.clip(unique_cats_per_graph / EXPECTED_CATEGORIES, 0, 1.0)

    math_space_idx = cat_to_idx.get("math_space")
    freq_idx = cat_to_idx.get("frequency")
    uses_math = (
        cat_counts[:, math_space_idx] > 0
        if math_space_idx is not None
        else np.zeros(len(graphs), dtype=bool)
    )
    uses_freq = (
        cat_counts[:, freq_idx] > 0
        if freq_idx is not None
        else np.zeros(len(graphs), dtype=bool)
    )

    probs = batch_counts.astype(np.float32) / np.maximum(
        n_ops_per_graph[:, None], 1e-10
    )
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0)), axis=1)
    max_entropy = np.log(np.maximum(unique_ops_per_graph, 1))
    # Clamp divisor before the divide so np.where never evaluates 0/0 and
    # emits a spurious RuntimeWarning. The mask still selects the real 0
    # branch where max_entropy == 0.
    evenness = np.where(max_entropy > 0, entropy / np.maximum(max_entropy, 1e-10), 0)

    structural_novelty = (
        WEIGHT_DIVERSITY * diversity
        + WEIGHT_SPREAD * category_spread
        + WEIGHT_EVENNESS * evenness
    )
    exotic_count = uses_math.astype(np.int32) + uses_freq.astype(np.int32)
    structural_novelty = np.clip(
        structural_novelty * (EXOTIC_BONUS_BASE + EXOTIC_BONUS_PER_FLAG * exotic_count),
        0,
        1.0,
    )
    return unique_ops_per_graph, uses_math, uses_freq, structural_novelty


def _populate_histograms(metrics: NoveltyMetrics, batch_row: np.ndarray) -> None:
    from ..synthesis.primitives import REVERSE_OPCODE_MAP, get_primitive

    for code, count in enumerate(batch_row):
        if count <= 0:
            continue
        name = REVERSE_OPCODE_MAP.get(code)
        if not name:
            continue
        metrics.op_histogram[name] = int(count)
        try:
            cat = get_primitive(name).category.value
            metrics.category_histogram[cat] = metrics.category_histogram.get(
                cat, 0
            ) + int(count)
        except (AttributeError, KeyError, ValueError) as exc:
            logger.debug(
                "Novelty histogram category lookup failed for %s: %s",
                name,
                exc,
            )


def _apply_behavioral_novelty(
    metrics: NoveltyMetrics,
    fp_obj: Optional[BehavioralFingerprint],
) -> None:
    if fp_obj is None:
        metrics.behavioral_novelty = 0.0
        metrics.raw_novelty = metrics.structural_novelty * STRUCTURAL_ONLY_WEIGHT
        metrics.novelty_confidence = CONFIDENCE_NO_FP
        metrics.novelty_valid_for_promotion = False
        metrics.novelty_validity_reason = "structural_only"
        return

    metrics.behavioral_novelty = fp_obj.novelty_score
    similarities = {
        "transformer": fp_obj.cka_vs_transformer or 0.0,
        "ssm": fp_obj.cka_vs_ssm or 0.0,
        "conv": fp_obj.cka_vs_conv or 0.0,
    }
    metrics.most_similar_to = max(similarities, key=similarities.get)
    metrics.max_cka_similarity = max(similarities.values())
    metrics.raw_novelty = (
        STRUCTURAL_BLEND_WEIGHT * metrics.structural_novelty
        + BEHAVIORAL_BLEND_WEIGHT * metrics.behavioral_novelty
    )
    metrics.novelty_reference_version = fp_obj.novelty_reference_version
    metrics.novelty_valid_for_promotion = bool(
        getattr(fp_obj, "novelty_valid_for_promotion", False)
    )
    metrics.novelty_validity_reason = getattr(
        fp_obj, "novelty_validity_reason", "missing_reference"
    )
    if fp_obj.quality == "full":
        metrics.novelty_confidence = CONFIDENCE_FULL
    elif fp_obj.quality == "partial":
        metrics.novelty_confidence = CONFIDENCE_PARTIAL_BASE + (
            fp_obj.analyses_succeeded * CONFIDENCE_PARTIAL_STEP
        )
    else:
        metrics.novelty_confidence = CONFIDENCE_NONE


def _build_novelty_metric(
    graph: ComputationGraph,
    batch_row: np.ndarray,
    unique_ops: int,
    uses_math: bool,
    uses_freq: bool,
    structural_novelty: float,
    fp_obj: Optional[BehavioralFingerprint],
    seen_fps: set[str],
) -> NoveltyMetrics:
    metrics = NoveltyMetrics(
        graph_fingerprint=graph.fingerprint(),
        n_unique_ops=int(unique_ops),
        uses_math_spaces=bool(uses_math),
        uses_frequency_domain=bool(uses_freq),
        structural_novelty=float(structural_novelty),
    )
    _populate_histograms(metrics, batch_row)
    _apply_behavioral_novelty(metrics, fp_obj)

    metrics.overall_novelty = metrics.raw_novelty
    if fp_obj is not None:
        metrics.overall_novelty *= _reference_similarity_penalty(
            metrics.max_cka_similarity
        )
    if metrics.graph_fingerprint in seen_fps:
        metrics.overall_novelty *= DUPLICATE_PENALTY_MULTIPLIER
    seen_fps.add(metrics.graph_fingerprint)
    return metrics


def batch_novelty_scores(
    graphs: List[ComputationGraph],
    fingerprints: Optional[List[BehavioralFingerprint]] = None,
) -> List[NoveltyMetrics]:
    """Compute novelty scores for a batch of graphs using vectorized IR analysis.

    Also penalizes graphs that are similar to EACH OTHER
    (we want diversity in the population).
    """
    if not graphs:
        return []

    from ..synthesis.primitives import OPCODE_MAP

    n_opcodes = len(OPCODE_MAP)
    irs = [g.lower_to_ir() for g in graphs]
    from ..synthesis.graph import ComputationGraphIR

    batch_counts = ComputationGraphIR.batch_op_distribution(irs, n_opcodes)
    unique_ops_per_graph, uses_math, uses_freq, structural_novelty = (
        _structural_novelty_components(graphs, batch_counts)
    )

    results = []
    seen_fps = set()
    for i, graph in enumerate(graphs):
        fp_obj = fingerprints[i] if fingerprints and i < len(fingerprints) else None
        results.append(
            _build_novelty_metric(
                graph,
                batch_counts[i],
                int(unique_ops_per_graph[i]),
                bool(uses_math[i]),
                bool(uses_freq[i]),
                float(structural_novelty[i]),
                fp_obj,
                seen_fps,
            )
        )

    return results
