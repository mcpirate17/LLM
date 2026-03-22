"""
Novelty Metrics

Information-theoretic and structural metrics for evaluating
how novel a synthesized program actually is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

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
from .fingerprint import BehavioralFingerprint


@dataclass
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
        return self.__dict__.copy()


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
    """Compute novelty metrics for a synthesized program."""
    ir = graph.lower_to_ir()
    if ir.is_stale(graph):
        import logging

        logging.getLogger(__name__).warning(
            "stale_ir_used: graph_version=%d ir_version=%d",
            graph._ir_version,
            ir.source_version,
        )
    metrics = _novelty_score_from_ir(graph, ir, fingerprint)

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

    from ..synthesis.primitives import (
        OPCODE_MAP,
        PRIMITIVE_REGISTRY,
        get_primitive,
        REVERSE_OPCODE_MAP,
    )

    n_opcodes = len(OPCODE_MAP)

    # 1. Lower all graphs to IR once
    irs = [g.lower_to_ir() for g in graphs]

    # 2. Vectorized opcode counts
    from ..synthesis.graph import ComputationGraphIR

    batch_counts = ComputationGraphIR.batch_op_distribution(irs, n_opcodes)

    # 3. Vectorized structural metrics
    n_ops_per_graph = batch_counts.sum(axis=1)
    unique_ops_per_graph = (batch_counts > 0).sum(axis=1)

    # Op diversity
    total_available = max(len(PRIMITIVE_REGISTRY), 1)
    diversity = np.clip(unique_ops_per_graph / total_available, 0, 1.0)

    # Category spread and exotic flags
    # Pre-map opcodes to categories
    opcode_to_cat = {}
    for name, code in OPCODE_MAP.items():
        if name == "input":
            continue
        try:
            opcode_to_cat[code] = get_primitive(name).category.value
        except Exception:
            pass

    all_categories = sorted(list(set(opcode_to_cat.values())))
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    n_cats = len(all_categories)

    # Map batch_counts to category_counts: (batch, n_cats)
    cat_counts = np.zeros((len(graphs), n_cats), dtype=np.int32)
    for code, cat in opcode_to_cat.items():
        cat_counts[:, cat_to_idx[cat]] += batch_counts[:, code]

    unique_cats_per_graph = (cat_counts > 0).sum(axis=1)
    category_spread = np.clip(unique_cats_per_graph / EXPECTED_CATEGORIES, 0, 1.0)

    # Exotic flags
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

    # Op distribution entropy
    probs = batch_counts.astype(np.float32) / np.maximum(
        n_ops_per_graph[:, None], 1e-10
    )
    # Vectorized entropy calculation
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0)), axis=1)

    max_entropy = np.log(np.maximum(unique_ops_per_graph, 1))
    evenness = np.where(max_entropy > 0, entropy / max_entropy, 0)

    structural_novelty = (
        WEIGHT_DIVERSITY * diversity
        + WEIGHT_SPREAD * category_spread
        + WEIGHT_EVENNESS * evenness
    )

    # Multiplicative bonus for exotic
    exotic_count = uses_math.astype(np.int32) + uses_freq.astype(np.int32)
    structural_novelty = np.clip(
        structural_novelty * (EXOTIC_BONUS_BASE + EXOTIC_BONUS_PER_FLAG * exotic_count),
        0,
        1.0,
    )

    # 4. Assembly
    results = []
    seen_fps = set()

    for i, graph in enumerate(graphs):
        metrics = NoveltyMetrics()
        metrics.graph_fingerprint = graph.fingerprint()
        metrics.n_unique_ops = int(unique_ops_per_graph[i])
        metrics.uses_math_spaces = bool(uses_math[i])
        metrics.uses_frequency_domain = bool(uses_freq[i])
        metrics.structural_novelty = float(structural_novelty[i])

        # Populate histograms for completeness
        for code, count in enumerate(batch_counts[i]):
            if count > 0:
                name = REVERSE_OPCODE_MAP.get(code)
                if name:
                    metrics.op_histogram[name] = int(count)
                    try:
                        cat = get_primitive(name).category.value
                        metrics.category_histogram[cat] = (
                            metrics.category_histogram.get(cat, 0) + int(count)
                        )
                    except Exception:
                        pass

        # Behavioral
        fp_obj = fingerprints[i] if fingerprints and i < len(fingerprints) else None
        if fp_obj is not None:
            metrics.behavioral_novelty = fp_obj.novelty_score
            cka_t = (
                fp_obj.cka_vs_transformer
                if fp_obj.cka_vs_transformer is not None
                else 0.0
            )
            cka_s = fp_obj.cka_vs_ssm if fp_obj.cka_vs_ssm is not None else 0.0
            cka_c = fp_obj.cka_vs_conv if fp_obj.cka_vs_conv is not None else 0.0
            similarities = {
                "transformer": cka_t,
                "ssm": cka_s,
                "conv": cka_c,
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
        else:
            metrics.behavioral_novelty = 0.0
            metrics.raw_novelty = metrics.structural_novelty * STRUCTURAL_ONLY_WEIGHT
            metrics.novelty_confidence = CONFIDENCE_NO_FP
            metrics.novelty_valid_for_promotion = False
            metrics.novelty_validity_reason = "structural_only"

        metrics.overall_novelty = metrics.raw_novelty

        # Down-weight reference-like candidates (high CKA to known families)
        if fp_obj is not None:
            metrics.overall_novelty *= _reference_similarity_penalty(
                metrics.max_cka_similarity
            )

        # Internal diversity penalty
        if metrics.graph_fingerprint in seen_fps:
            metrics.overall_novelty *= DUPLICATE_PENALTY_MULTIPLIER
        seen_fps.add(metrics.graph_fingerprint)

        results.append(metrics)

    return results


def _novelty_score_from_ir(
    graph: ComputationGraph,
    ir: Any,
    fingerprint: Optional[BehavioralFingerprint] = None,
) -> NoveltyMetrics:
    """Helper that computes novelty score from pre-lowered IR."""
    metrics = NoveltyMetrics()
    metrics.graph_fingerprint = graph.fingerprint()

    # ── Structural Analysis (Vectorized via IR) ──
    op_codes = ir.op_codes
    # Exclude input node (opcode 0)
    non_input_mask = op_codes != 0
    ops_in_graph = op_codes[non_input_mask]

    if len(ops_in_graph) > 0:
        counts = np.bincount(ops_in_graph)
        active_opcodes = np.nonzero(counts)[0]

        metrics.n_unique_ops = len(active_opcodes)

        from ..synthesis.primitives import (
            REVERSE_OPCODE_MAP,
            get_primitive,
            PRIMITIVE_REGISTRY,
        )

        for opcode in active_opcodes:
            op_name = REVERSE_OPCODE_MAP.get(opcode)
            if not op_name:
                continue
            count = int(counts[opcode])
            metrics.op_histogram[op_name] = count

            try:
                op = get_primitive(op_name)
                cat = op.category.value
                metrics.category_histogram[cat] = (
                    metrics.category_histogram.get(cat, 0) + count
                )
                if cat == "math_space":
                    metrics.uses_math_spaces = True
                if cat == "frequency":
                    metrics.uses_frequency_domain = True
            except KeyError:
                pass

    # Structural novelty: combination of op diversity and distribution balance
    n_ops = len(ops_in_graph)
    if n_ops > 0:
        # Component 1: Op diversity
        try:
            total_available = max(len(PRIMITIVE_REGISTRY), 1)
        except (ImportError, AttributeError):
            total_available = 50
        diversity = min(metrics.n_unique_ops / total_available, 1.0)

        # Component 2: Category spread
        n_categories = len(metrics.category_histogram)
        max_categories = 8
        category_spread = min(n_categories / max_categories, 1.0)

        # Component 3: Evenness of op distribution
        probs = np.array(list(metrics.op_histogram.values()), dtype=np.float32) / n_ops
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = math.log(max(len(metrics.op_histogram), 1))
        evenness = entropy / max(max_entropy, 1e-10) if max_entropy > 0 else 0

        # Weighted combination
        metrics.structural_novelty = (
            0.50 * diversity + 0.30 * category_spread + 0.20 * evenness
        )

        # Multiplicative bonus for exotic ops
        n_exotic = int(metrics.uses_math_spaces) + int(metrics.uses_frequency_domain)
        if n_exotic > 0:
            metrics.structural_novelty = min(
                1.0, metrics.structural_novelty * (1 + 0.1 * n_exotic)
            )

    # ── Behavioral Novelty ──
    if fingerprint is not None:
        metrics.behavioral_novelty = fingerprint.novelty_score
        cka_t = (
            fingerprint.cka_vs_transformer
            if fingerprint.cka_vs_transformer is not None
            else 0.0
        )
        cka_s = fingerprint.cka_vs_ssm if fingerprint.cka_vs_ssm is not None else 0.0
        cka_c = fingerprint.cka_vs_conv if fingerprint.cka_vs_conv is not None else 0.0
        similarities = {
            "transformer": cka_t,
            "ssm": cka_s,
            "conv": cka_c,
        }
        metrics.most_similar_to = max(similarities, key=similarities.get)
        metrics.max_cka_similarity = max(similarities.values())

    # ── Combined Score ──
    if fingerprint is not None:
        metrics.raw_novelty = (
            0.3 * metrics.structural_novelty + 0.7 * metrics.behavioral_novelty
        )
    else:
        metrics.raw_novelty = metrics.structural_novelty * STRUCTURAL_ONLY_WEIGHT

    metrics.overall_novelty = metrics.raw_novelty

    # Down-weight reference-like candidates (high CKA to known families)
    if fingerprint is not None:
        metrics.overall_novelty *= _reference_similarity_penalty(
            metrics.max_cka_similarity
        )

    # ── Confidence Score ──
    if fingerprint is not None:
        metrics.novelty_reference_version = fingerprint.novelty_reference_version
        metrics.novelty_valid_for_promotion = bool(
            getattr(fingerprint, "novelty_valid_for_promotion", False)
        )
        metrics.novelty_validity_reason = getattr(
            fingerprint, "novelty_validity_reason", "missing_reference"
        )
        if fingerprint.quality == "full":
            metrics.novelty_confidence = CONFIDENCE_FULL
        elif fingerprint.quality == "partial":
            metrics.novelty_confidence = 0.4 + (fingerprint.analyses_succeeded * 0.1)
        else:
            metrics.novelty_confidence = CONFIDENCE_NONE
    else:
        metrics.novelty_confidence = CONFIDENCE_NO_FP
        metrics.novelty_valid_for_promotion = False
        metrics.novelty_validity_reason = "structural_only"

    return metrics
