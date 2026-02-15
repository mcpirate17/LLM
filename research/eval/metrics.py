"""
Novelty Metrics

Information-theoretic and structural metrics for evaluating
how novel a synthesized program actually is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    overall_novelty: float = 0.0
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
) -> NoveltyMetrics:
    """Compute novelty metrics for a synthesized program."""
    metrics = NoveltyMetrics()
    metrics.graph_fingerprint = graph.fingerprint()

    # ── Structural Analysis ──
    ops_used: Set[str] = set()
    op_counts: Dict[str, int] = {}
    cat_counts: Dict[str, int] = {}

    for node in graph.nodes.values():
        if node.is_input:
            continue
        op_name = node.op_name
        ops_used.add(op_name)
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

        try:
            from ..synthesis.primitives import get_primitive
            op = get_primitive(op_name)
            cat = op.category.value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if cat == "math_space":
                metrics.uses_math_spaces = True
            if cat == "frequency":
                metrics.uses_frequency_domain = True
        except KeyError:
            pass

    metrics.n_unique_ops = len(ops_used)
    metrics.op_histogram = op_counts
    metrics.category_histogram = cat_counts

    # Structural novelty: combination of op diversity and distribution balance
    if graph.n_ops() > 0:
        # Component 1: Op diversity — unique ops used vs total available
        try:
            from ..synthesis.primitives import PRIMITIVE_REGISTRY
            total_available = max(len(PRIMITIVE_REGISTRY), 1)
        except (ImportError, AttributeError):
            total_available = 50  # reasonable estimate
        diversity = min(len(ops_used) / total_available, 1.0)

        # Component 2: Category spread — how many different categories used
        n_categories = len(cat_counts)
        max_categories = 8  # math_space, frequency, linear, activation, norm, etc.
        category_spread = min(n_categories / max_categories, 1.0)

        # Component 3: Evenness of op distribution (but capped to avoid 1.0)
        total = sum(op_counts.values())
        probs = [c / total for c in op_counts.values()]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(max(len(op_counts), 1))
        evenness = entropy / max(max_entropy, 1e-10) if max_entropy > 0 else 0

        # Weighted combination: diversity matters most
        metrics.structural_novelty = (
            0.50 * diversity +
            0.30 * category_spread +
            0.20 * evenness
        )

        # Multiplicative bonus for exotic ops (avoids inflating toward 1.0)
        n_exotic = int(metrics.uses_math_spaces) + int(metrics.uses_frequency_domain)
        if n_exotic > 0:
            metrics.structural_novelty = min(1.0, metrics.structural_novelty * (1 + 0.1 * n_exotic))

    # ── Behavioral Novelty ──
    if fingerprint is not None:
        metrics.behavioral_novelty = fingerprint.novelty_score

        # Find most similar known architecture
        similarities = {
            "transformer": fingerprint.cka_vs_transformer,
            "ssm": fingerprint.cka_vs_ssm,
            "conv": fingerprint.cka_vs_conv,
        }
        metrics.most_similar_to = max(similarities, key=similarities.get)
        metrics.max_cka_similarity = max(similarities.values())

    # ── Combined Score ──
    if fingerprint is not None:
        # Weight behavioral novelty more (it's what actually matters)
        metrics.overall_novelty = (
            0.3 * metrics.structural_novelty +
            0.7 * metrics.behavioral_novelty
        )
    else:
        # Structural-only novelty is less reliable — discount by 0.6x
        # to avoid inflated scores when no behavioral data is available
        metrics.overall_novelty = metrics.structural_novelty * 0.6

    # ── Confidence Score ──
    if fingerprint is not None:
        if fingerprint.quality == "full":
            # Cap below 1.0: CKA references are synthetic heuristics
            metrics.novelty_confidence = 0.9
        elif fingerprint.quality == "partial":
            metrics.novelty_confidence = 0.4 + (fingerprint.analyses_succeeded * 0.1)
        else:
            # quality == "none" but fingerprint object was provided
            metrics.novelty_confidence = 0.3
    else:
        # No fingerprint at all — structural-only
        metrics.novelty_confidence = 0.2

    # Check against known fingerprints
    if known_fingerprints and metrics.graph_fingerprint in known_fingerprints:
        metrics.overall_novelty *= 0.1  # Heavily penalize exact duplicates

    return metrics


def batch_novelty_scores(
    graphs: List[ComputationGraph],
    fingerprints: Optional[List[BehavioralFingerprint]] = None,
) -> List[NoveltyMetrics]:
    """Compute novelty scores for a batch of graphs.

    Also penalizes graphs that are similar to EACH OTHER
    (we want diversity in the population).
    """
    known_fps: List[str] = []
    results = []

    for i, graph in enumerate(graphs):
        fp = fingerprints[i] if fingerprints and i < len(fingerprints) else None
        metrics = novelty_score(graph, fp, known_fps)
        known_fps.append(metrics.graph_fingerprint)
        results.append(metrics)

    return results
