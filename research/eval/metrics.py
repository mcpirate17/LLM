"""
Novelty Metrics

Information-theoretic and structural metrics for evaluating
how novel a synthesized program actually is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
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


def novelty_score(
    graph: ComputationGraph,
    fingerprint: Optional[BehavioralFingerprint] = None,
    known_fingerprints: Optional[List[str]] = None,
    calibration: Optional[Dict[str, float]] = None,
) -> NoveltyMetrics:
    """Compute novelty metrics for a synthesized program."""
    ir = graph.lower_to_ir()
    metrics = _novelty_score_from_ir(graph, ir, fingerprint)

    # Check against known fingerprints
    if known_fingerprints and metrics.graph_fingerprint in known_fingerprints:
        metrics.overall_novelty *= 0.1  # Heavily penalize exact duplicates

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

    # 1. Lower all graphs to IR once
    irs = [g.lower_to_ir() for g in graphs]
    
    # 2. Extract fingerprints
    fingerprints_str = [g.fingerprint() for g in graphs]
    
    results = []
    seen_fps = set()
    
    for i, (graph, ir) in enumerate(zip(graphs, irs)):
        fp_obj = fingerprints[i] if fingerprints and i < len(fingerprints) else None
        
        # Use a version of novelty_score that accepts pre-computed IR
        metrics = _novelty_score_from_ir(graph, ir, fp_obj)
        
        # Internal diversity penalty
        if metrics.graph_fingerprint in seen_fps:
            metrics.overall_novelty *= 0.1
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

        from ..synthesis.primitives import REVERSE_OPCODE_MAP, get_primitive, PRIMITIVE_REGISTRY
        for opcode in active_opcodes:
            op_name = REVERSE_OPCODE_MAP.get(opcode)
            if not op_name:
                continue
            count = int(counts[opcode])
            metrics.op_histogram[op_name] = count

            try:
                op = get_primitive(op_name)
                cat = op.category.value
                metrics.category_histogram[cat] = metrics.category_histogram.get(cat, 0) + count
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
            0.50 * diversity +
            0.30 * category_spread +
            0.20 * evenness
        )

        # Multiplicative bonus for exotic ops
        n_exotic = int(metrics.uses_math_spaces) + int(metrics.uses_frequency_domain)
        if n_exotic > 0:
            metrics.structural_novelty = min(1.0, metrics.structural_novelty * (1 + 0.1 * n_exotic))

    # ── Behavioral Novelty ──
    if fingerprint is not None:
        metrics.behavioral_novelty = fingerprint.novelty_score
        similarities = {
            "transformer": fingerprint.cka_vs_transformer,
            "ssm": fingerprint.cka_vs_ssm,
            "conv": fingerprint.cka_vs_conv,
        }
        metrics.most_similar_to = max(similarities, key=similarities.get)
        metrics.max_cka_similarity = max(similarities.values())

    # ── Combined Score ──
    if fingerprint is not None:
        metrics.raw_novelty = (
            0.3 * metrics.structural_novelty +
            0.7 * metrics.behavioral_novelty
        )
    else:
        metrics.raw_novelty = metrics.structural_novelty * 0.6

    metrics.overall_novelty = metrics.raw_novelty

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
            metrics.novelty_confidence = 0.9
        elif fingerprint.quality == "partial":
            metrics.novelty_confidence = 0.4 + (fingerprint.analyses_succeeded * 0.1)
        else:
            metrics.novelty_confidence = 0.3
    else:
        metrics.novelty_confidence = 0.2
        metrics.novelty_valid_for_promotion = False
        metrics.novelty_validity_reason = "structural_only"

    return metrics
