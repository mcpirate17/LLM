"""Topology-aware graph predictor for architecture outcome prediction.

Extracts structural features that flat op-histograms miss:
- Neighbor-aggregated profiling statistics (mean lipschitz of successors, etc.)
- Path-based features (max product of lipschitz along any path)
- Depth-weighted op profiles (ops near output matter more)
- Edge-pattern features (split→merge ratios, skip connection patterns)

These features complement the existing GBM's flat features, enabling
an ensemble that captures both "what ops" AND "how they're connected."

Inference target: <0.5ms per graph.

Usage:
    model = GraphPredictor.train(notebook_db, profiling_db)
    p_gate = model.predict_gate(graph_json_dict)
    rank = model.predict_rank(graph_json_dict)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..native.core import _try_import_rust_scheduler
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from .metrics_utils import (
    binary_classification_metrics,
    operating_point_profiles,
    safe_binary_roc_auc,
)
from .ml_corpus import (
    CorpusIntegrityError,
    build_dense_feature_matrix,
    grouped_stratified_split,
    load_screening_predictor_corpus_rows,
    rerun_confidence_weight,
)
from .predictor_artifacts import load_npz_with_metadata, save_npz_with_metadata
from .profiling_db import (
    load_op_profiles as _load_op_profiles,
    load_pair_stability_map as _load_pair_stability,
)

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[2] / "lab_notebook.db"
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)

# ── Native-fallback telemetry ───────────────────────────────────────────
# Counters track when the Python parity twin fires in place of the Rust
# extractor. Any non-zero count in prod means the Rust path returned
# None/unparseable output — either aria-scheduler was not loaded, or the
# native kernel disagreed with the graph shape. Surface via
# ``get_native_fallback_counters()``.
_NATIVE_FALLBACK_COUNTERS: Dict[str, int] = {
    "topology_features_native_hit": 0,
    "topology_features_python_fallback": 0,
    "edge_op_pairs_native_hit": 0,
    "edge_op_pairs_python_fallback": 0,
}


def get_native_fallback_counters() -> Dict[str, int]:
    """Return a snapshot of native-vs-Python dispatch counts."""
    return dict(_NATIVE_FALLBACK_COUNTERS)


def reset_native_fallback_counters() -> None:
    """Reset all native-fallback counters to zero (tests only)."""
    for key in _NATIVE_FALLBACK_COUNTERS:
        _NATIVE_FALLBACK_COUNTERS[key] = 0


_MIN_SAMPLES = 50
_FEATURE_Z_CLIP = 6.0
_POLY_FEATURE_BASES = (
    "meta_n_templates",
    "meta_n_motifs",
    "meta_template_per_op",
    "meta_motif_per_op",
    "residual_coverage",
    "topo_n_split_nodes",
    "topo_n_merge_nodes",
    "topo_depth_per_op",
    "output_parent_depth_mean",
    "depth_frac_late",
    "cat_frac_parameterized",
    "path_max_risk_accum",
    "topo_n_ops",
    "topo_edge_density",
)


@dataclass(slots=True)
class _NativeTopologyContext:
    op_profiles_json: str
    pair_stability_json: str
    op_metadata_json: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def _clip_normalized_features(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return np.clip(x, -_FEATURE_Z_CLIP, _FEATURE_Z_CLIP)


def _augment_feature_space(
    base_feature_names: List[str], X: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Add targeted polynomial terms for the most discriminative structure features."""
    if X.ndim != 2 or X.shape[1] != len(base_feature_names):
        return base_feature_names, X

    name_to_idx = {name: i for i, name in enumerate(base_feature_names)}
    selected = [name for name in _POLY_FEATURE_BASES if name in name_to_idx]
    if len(selected) < 2:
        return base_feature_names, X

    extra_cols: List[np.ndarray] = []
    extra_names: List[str] = []
    for name in selected:
        col = X[:, name_to_idx[name]]
        extra_cols.append((col * col)[:, None])
        extra_names.append(f"{name}__sq")

    for i, left in enumerate(selected):
        left_col = X[:, name_to_idx[left]]
        for right in selected[i + 1 :]:
            right_col = X[:, name_to_idx[right]]
            extra_cols.append((left_col * right_col)[:, None])
            extra_names.append(f"{left}__x__{right}")

    if not extra_cols:
        return base_feature_names, X

    X_aug = np.concatenate([X, *extra_cols], axis=1)
    return [*base_feature_names, *extra_names], X_aug


def _materialize_feature_dict(feats: Dict[str, float]) -> Dict[str, float]:
    """Compute derived polynomial features for inference from a base feature dict."""
    augmented = dict(feats)
    selected = [name for name in _POLY_FEATURE_BASES if name in feats]
    for name in selected:
        value = float(feats.get(name, 0.0))
        augmented[f"{name}__sq"] = value * value
    for i, left in enumerate(selected):
        left_value = float(feats.get(left, 0.0))
        for right in selected[i + 1 :]:
            augmented[f"{left}__x__{right}"] = left_value * float(feats.get(right, 0.0))
    return augmented


def _graph_imodel_path(path: Path) -> Path:
    return path.with_suffix(".imodel.npz")


@lru_cache(maxsize=1)
def _primitive_feature_metadata_json() -> str:
    payload: Dict[str, Dict[str, Any]] = {}
    for op_name, primitive in PRIMITIVE_REGISTRY.items():
        category = getattr(primitive, "category", "")
        payload[op_name] = {
            "category": str(getattr(category, "value", category)).lower(),
            "has_params": bool(getattr(primitive, "has_params", False)),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _native_pair_stability_payload(
    pair_stability: Dict[Tuple[str, str], float],
) -> str:
    payload = {
        f"{left}\t{right}": float(value)
        for (left, right), value in pair_stability.items()
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _make_native_topology_context(
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
) -> _NativeTopologyContext:
    return _NativeTopologyContext(
        op_profiles_json=json.dumps(op_profiles, sort_keys=True, separators=(",", ":")),
        pair_stability_json=_native_pair_stability_payload(pair_stability),
        op_metadata_json=_primitive_feature_metadata_json(),
    )


def _extract_topology_features_python(
    graph_json: Any,
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    imodel: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """Extract topology-aware features from a computation graph.

    These features capture HOW ops are connected, not just WHAT ops are present.

    Returns dict of feature_name → float, or None on parse failure.
    """
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(graph_json, dict):
        return None

    nodes = graph_json.get("nodes") or {}
    if len(nodes) < 2:
        return None

    # Build adjacency (parent → children)
    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    children: Dict[int, List[int]] = defaultdict(list)  # idx → [child_idx]
    parents: Dict[int, List[int]] = defaultdict(list)

    for nid in node_ids:
        node = nodes[nid]
        idx = id_to_idx[nid]
        for inp in node.get("input_ids") or []:
            parent_idx = id_to_idx.get(str(inp))
            if parent_idx is not None:
                children[parent_idx].append(idx)
                parents[idx].append(parent_idx)

    metadata = graph_json.get("metadata") or {}

    # Op names per node
    op_names = [nodes[nid].get("op_name", "") for nid in node_ids]

    # BFS depth
    roots = [i for i in range(n) if not parents[i]]
    if not roots:
        roots = [0]
    depth = np.full(n, -1, dtype=np.int32)
    queue = list(roots)
    for r in queue:
        depth[r] = 0
    qi = 0
    while qi < len(queue):
        idx = queue[qi]
        qi += 1
        for child in children[idx]:
            if depth[child] < 0:
                depth[child] = depth[idx] + 1
                queue.append(child)

    max_depth = max(int(depth.max()), 1)

    features: Dict[str, float] = {}

    # ── 1. Topology basics ──
    n_ops = sum(1 for op in op_names if op and op != "input")
    n_edges = sum(len(ch) for ch in children.values())
    features["topo_n_ops"] = float(n_ops)
    features["topo_depth"] = float(max_depth)
    features["topo_edge_density"] = n_edges / max(n, 1)
    features["topo_edges_per_op"] = n_edges / max(n_ops, 1)
    features["topo_depth_per_op"] = max_depth / max(n_ops, 1)

    # Fan-in / fan-out statistics
    fan_ins = [len(parents[i]) for i in range(n)]
    fan_outs = [len(children[i]) for i in range(n)]
    features["topo_max_fan_in"] = float(max(fan_ins)) if fan_ins else 0.0
    features["topo_max_fan_out"] = float(max(fan_outs)) if fan_outs else 0.0
    features["topo_mean_fan_in"] = float(np.mean(fan_ins)) if fan_ins else 0.0
    features["topo_mean_fan_out"] = float(np.mean(fan_outs)) if fan_outs else 0.0
    features["topo_n_merge_nodes"] = float(sum(1 for f in fan_ins if f > 1))
    features["topo_n_split_nodes"] = float(sum(1 for f in fan_outs if f > 1))
    features["topo_leaf_fraction"] = float(sum(1 for f in fan_outs if f == 0)) / max(
        n, 1
    )
    features["topo_root_fraction"] = float(sum(1 for f in fan_ins if f == 0)) / max(
        n, 1
    )

    # ── 2. Profiling-grounded path features ──
    # Lipschitz chain: product along paths (measures amplification risk)
    lip_values = np.ones(n, dtype=np.float64)
    grad_risks = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(op_names):
        prof = op_profiles.get(op, {})
        lip_values[i] = min(prof.get("lipschitz", 1.0), 100.0)
        grad_risks[i] = (
            prof.get("grad_vanishing", 0)
            + prof.get("grad_exploding", 0)
            + prof.get("has_nan", 0)
        )

    # Max lipschitz product along any root→leaf path (DP)
    lip_product = np.ones(n, dtype=np.float64)
    for idx in queue:  # BFS order
        if parents[idx]:
            max_parent_lip = max(lip_product[p] for p in parents[idx])
            lip_product[idx] = max_parent_lip * lip_values[idx]

    features["path_max_lip_product"] = float(np.clip(np.max(lip_product), 0, 1e6))
    features["path_mean_lip_product"] = float(np.clip(np.mean(lip_product), 0, 1e6))
    features["path_max_lip_log"] = float(np.log1p(np.max(lip_product)))

    # Gradient risk accumulation
    risk_accum = np.zeros(n, dtype=np.float64)
    for idx in queue:
        if parents[idx]:
            risk_accum[idx] = max(risk_accum[p] for p in parents[idx]) + grad_risks[idx]
    features["path_max_risk_accum"] = float(np.max(risk_accum))
    features["path_mean_risk_accum"] = float(np.mean(risk_accum))

    # ── 3. Depth-weighted op profiles ──
    # Ops near output (high depth) matter more for final loss
    depth_weights = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if depth[i] >= 0:
            depth_weights[i] = (depth[i] + 1) / (max_depth + 1)

    weighted_lip = 0.0
    weighted_std = 0.0
    weighted_grad = 0.0
    weight_sum = 0.0
    for i, op in enumerate(op_names):
        if not op or op == "input":
            continue
        prof = op_profiles.get(op, {})
        w = depth_weights[i]
        weighted_lip += w * min(prof.get("lipschitz", 1.0), 100.0)
        weighted_std += w * min(prof.get("output_std", 1.0), 10.0)
        weighted_grad += w * min(prof.get("grad_norm", 1.0), 1000.0)
        weight_sum += w

    weight_sum = max(weight_sum, 1e-8)
    features["depth_weighted_lip"] = weighted_lip / weight_sum
    features["depth_weighted_std"] = weighted_std / weight_sum
    features["depth_weighted_grad"] = float(np.log1p(weighted_grad / weight_sum))

    # ── 4. Consecutive pair stability ──
    # Use profiling pair data along actual edges
    pair_stabilities = []
    for idx in range(n):
        for child_idx in children[idx]:
            a, b = op_names[idx], op_names[child_idx]
            if a and b and a != "input" and b != "input":
                stab = pair_stability.get((a, b), 0.5)
                pair_stabilities.append(stab)

    if pair_stabilities:
        features["pair_min_stability"] = float(min(pair_stabilities))
        features["pair_mean_stability"] = float(np.mean(pair_stabilities))
        features["pair_frac_unstable"] = float(
            sum(1 for s in pair_stabilities if s < 0.5) / len(pair_stabilities)
        )
    else:
        features["pair_min_stability"] = 0.5
        features["pair_mean_stability"] = 0.5
        features["pair_frac_unstable"] = 0.5

    # ── 4b. Learned pair interaction features (from InteractionModel) ──
    # The profiling pair_stability above is static (5,979 pairs profiled once).
    # The interaction model learns from 258K+ experiment observations with temporal decay.
    if imodel is not None and hasattr(imodel, "_trained") and imodel._trained:
        imodel_stabilities = []
        imodel_losses = []
        for idx in range(n):
            for child_idx in children[idx]:
                a, b = op_names[idx], op_names[child_idx]
                if a and b and a != "input" and b != "input":
                    imodel_stabilities.append(imodel.predict_stability(a, b))
                    imodel_losses.append(imodel.predict_loss(a, b))
        if imodel_stabilities:
            features["imodel_min_stability"] = float(min(imodel_stabilities))
            features["imodel_mean_stability"] = float(np.mean(imodel_stabilities))
            features["imodel_mean_loss"] = float(np.mean(imodel_losses))
        else:
            features["imodel_min_stability"] = 0.5
            features["imodel_mean_stability"] = 0.5
            features["imodel_mean_loss"] = 0.7
    else:
        features["imodel_min_stability"] = 0.5
        features["imodel_mean_stability"] = 0.5
        features["imodel_mean_loss"] = 0.7

    # ── 5. Neighbor profile aggregation ──
    # For each node, mean lipschitz of its children (amplification risk of downstream)
    child_lip_means = []
    for idx in range(n):
        if children[idx]:
            child_lips = [lip_values[c] for c in children[idx]]
            child_lip_means.append(float(np.mean(child_lips)))
    features["neighbor_mean_child_lip"] = (
        float(np.mean(child_lip_means)) if child_lip_means else 1.0
    )
    features["neighbor_max_child_lip"] = (
        float(max(child_lip_means)) if child_lip_means else 1.0
    )

    # ── 6. Structural patterns ──
    # Residual coverage: fraction of ops with skip connections (add nodes w/ depth gap)
    n_skip = 0
    skip_spans = []
    for idx in range(n):
        if op_names[idx] == "add" and len(parents[idx]) >= 2:
            depths = [depth[p] for p in parents[idx] if depth[p] >= 0]
            if depths and max(depths) - min(depths) > 0:
                n_skip += 1
                skip_spans.append(max(depths) - min(depths))
    features["residual_coverage"] = float(n_skip) / max(n_ops, 1)
    features["residual_span_mean"] = float(np.mean(skip_spans)) if skip_spans else 0.0
    features["residual_span_max"] = float(max(skip_spans)) if skip_spans else 0.0

    # ── 7. Depth-bucket and op-category structure ──
    early = mid = late = 0
    mixing = math_space = parameterized = reduction = 0
    param_ops = 0
    math_late = 0.0
    mixing_late = 0.0
    for i, op in enumerate(op_names):
        if not op or op == "input":
            continue
        d_norm = depth_weights[i]
        if d_norm <= 0.34:
            early += 1
        elif d_norm <= 0.67:
            mid += 1
        else:
            late += 1

        prim = PRIMITIVE_REGISTRY.get(op)
        if prim is not None:
            cat_name = str(getattr(prim.category, "value", prim.category)).lower()
            if "mix" in cat_name:
                mixing += 1
                mixing_late += d_norm
            elif "math" in cat_name:
                math_space += 1
                math_late += d_norm
            elif "param" in cat_name:
                parameterized += 1
            elif "reduction" in cat_name:
                reduction += 1
            if getattr(prim, "has_params", False):
                param_ops += 1

    features["depth_frac_early"] = early / max(n_ops, 1)
    features["depth_frac_mid"] = mid / max(n_ops, 1)
    features["depth_frac_late"] = late / max(n_ops, 1)
    features["cat_frac_mixing"] = mixing / max(n_ops, 1)
    features["cat_frac_math_space"] = math_space / max(n_ops, 1)
    features["cat_frac_parameterized"] = parameterized / max(n_ops, 1)
    features["cat_frac_reduction"] = reduction / max(n_ops, 1)
    features["param_op_fraction"] = param_ops / max(n_ops, 1)
    features["late_mixing_density"] = mixing_late / max(n_ops, 1)
    features["late_math_density"] = math_late / max(n_ops, 1)

    # ── 8. Metadata structure ──
    templates_used = metadata.get("templates_used") or []
    motifs_used = metadata.get("motifs_used") or []
    features["meta_n_templates"] = float(len(templates_used))
    features["meta_n_motifs"] = float(len(motifs_used))
    features["meta_template_per_op"] = float(len(templates_used)) / max(n_ops, 1)
    features["meta_motif_per_op"] = float(len(motifs_used)) / max(n_ops, 1)

    # ── 9. Output structure ──
    # Has normalization before output
    output_id = graph_json.get("output_node_id")
    if output_id is not None and str(output_id) in nodes:
        output_node = nodes[str(output_id)]
        output_parents = [str(p) for p in (output_node.get("input_ids") or [])]
        has_norm_before_output = any(
            nodes.get(p, {}).get("op_name", "")
            in ("layernorm", "rmsnorm", "layer_norm", "rms_norm")
            for p in output_parents
        )
        features["has_norm_before_output"] = float(has_norm_before_output)
        parent_depths = [
            depth[id_to_idx[p]]
            for p in output_parents
            if p in id_to_idx and depth[id_to_idx[p]] >= 0
        ]
        features["output_parent_depth_mean"] = (
            float(np.mean(parent_depths)) if parent_depths else 0.0
        )
    else:
        features["has_norm_before_output"] = 0.0
        features["output_parent_depth_mean"] = 0.0

    return features


def _extract_edge_op_pairs_python(graph_json: Any) -> Optional[List[Tuple[str, str]]]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(graph_json, dict):
        return None
    nodes = graph_json.get("nodes") or {}
    if len(nodes) < 2:
        return None

    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    op_names = [nodes[nid].get("op_name", "") for nid in node_ids]
    pairs: List[Tuple[str, str]] = []
    for nid in node_ids:
        child_idx = id_to_idx[nid]
        child_op = op_names[child_idx]
        if not child_op or child_op == "input":
            continue
        for inp in nodes[nid].get("input_ids") or []:
            parent_idx = id_to_idx.get(str(inp))
            if parent_idx is None:
                continue
            parent_op = op_names[parent_idx]
            if not parent_op or parent_op == "input":
                continue
            pairs.append((parent_op, child_op))
    pairs.sort()
    return pairs


def _extract_edge_op_pairs_native(
    graph_payload: str,
) -> Optional[List[Tuple[str, str]]]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_edge_op_pairs_native"):
        return None
    payload = rust.extract_edge_op_pairs_native(graph_payload)
    loaded = json.loads(payload)
    if not isinstance(loaded, list):
        return None
    pairs: List[Tuple[str, str]] = []
    for item in loaded:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(part, str) for part in item)
        ):
            pairs.append((item[0], item[1]))
    pairs.sort()
    return pairs


def _serialize_graph_payload(graph_json: Any) -> Optional[str]:
    if isinstance(graph_json, str):
        return graph_json
    try:
        return json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _default_imodel_features(features: Dict[str, float]) -> Dict[str, float]:
    features["imodel_min_stability"] = 0.5
    features["imodel_mean_stability"] = 0.5
    features["imodel_mean_loss"] = 0.7
    return features


def _edge_op_pairs_with_fallback(graph_payload: str) -> Optional[List[Tuple[str, str]]]:
    edge_pairs = _extract_edge_op_pairs_native(graph_payload)
    if edge_pairs is None:
        _NATIVE_FALLBACK_COUNTERS["edge_op_pairs_python_fallback"] += 1
        return _extract_edge_op_pairs_python(graph_payload)
    _NATIVE_FALLBACK_COUNTERS["edge_op_pairs_native_hit"] += 1
    return edge_pairs


def _extract_edge_op_pairs_batch_native(
    graph_payloads: List[str],
) -> Optional[List[Optional[List[Tuple[str, str]]]]]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_edge_op_pairs_batch_native"):
        return None
    try:
        raw_result = rust.extract_edge_op_pairs_batch_native(graph_payloads)
    except Exception:
        return None
    if not isinstance(raw_result, list) or len(raw_result) != len(graph_payloads):
        return None
    decoded_batch: List[Optional[List[Tuple[str, str]]]] = []
    for payload in raw_result:
        if not isinstance(payload, str):
            decoded_batch.append(None)
            continue
        loaded = json.loads(payload)
        if not isinstance(loaded, list):
            decoded_batch.append(None)
            continue
        pairs: List[Tuple[str, str]] = []
        for item in loaded:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(part, str) for part in item)
            ):
                pairs.append((item[0], item[1]))
        pairs.sort()
        decoded_batch.append(pairs)
    return decoded_batch


def _augment_imodel_features_from_pairs(
    features: Optional[Dict[str, float]],
    edge_pairs: Optional[List[Tuple[str, str]]],
    imodel: Optional[Any],
) -> Optional[Dict[str, float]]:
    if features is None:
        return None
    if not (imodel is not None and hasattr(imodel, "_trained") and imodel._trained):
        return _default_imodel_features(features)
    if not edge_pairs:
        return _default_imodel_features(features)

    if hasattr(imodel, "predict_pair_stats"):
        pair_stats = imodel.predict_pair_stats(edge_pairs)
        if pair_stats is not None:
            min_stability, mean_stability, mean_loss = pair_stats
            features["imodel_min_stability"] = float(min_stability)
            features["imodel_mean_stability"] = float(mean_stability)
            features["imodel_mean_loss"] = float(mean_loss)
            return features

    imodel_stabilities = [
        imodel.predict_stability(left, right) for left, right in edge_pairs
    ]
    imodel_losses = [imodel.predict_loss(left, right) for left, right in edge_pairs]
    if not imodel_stabilities:
        return _default_imodel_features(features)

    features["imodel_min_stability"] = float(min(imodel_stabilities))
    features["imodel_mean_stability"] = float(np.mean(imodel_stabilities))
    features["imodel_mean_loss"] = float(np.mean(imodel_losses))
    return features


def _augment_imodel_features(
    features: Optional[Dict[str, float]],
    graph_payload: str,
    imodel: Optional[Any],
) -> Optional[Dict[str, float]]:
    edge_pairs = _edge_op_pairs_with_fallback(graph_payload)
    return _augment_imodel_features_from_pairs(features, edge_pairs, imodel)


def _decode_topology_feature_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, str):
        return None
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        return None
    return {str(key): float(value) for key, value in decoded.items()}


def _coerce_topology_feature_mapping(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    try:
        return {str(key): float(value) for key, value in payload.items()}
    except (TypeError, ValueError):
        return None


def _extract_topology_base_feature_native(
    graph_payload: str,
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    native_ctx: Optional[_NativeTopologyContext],
) -> Optional[Dict[str, float]]:
    rust = _try_import_rust_scheduler()
    if rust is None:
        return None
    ctx = native_ctx or _make_native_topology_context(op_profiles, pair_stability)
    if hasattr(rust, "extract_topology_feature_map_native_py"):
        try:
            raw_result = rust.extract_topology_feature_map_native_py(
                graph_payload,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
            )
        except Exception:
            raw_result = None
        mapped = _coerce_topology_feature_mapping(raw_result)
        if mapped is not None:
            return mapped
    if not hasattr(rust, "extract_topology_features_native"):
        return None
    try:
        raw_result = rust.extract_topology_features_native(
            graph_payload,
            ctx.op_profiles_json,
            ctx.pair_stability_json,
            ctx.op_metadata_json,
        )
    except Exception:
        return None
    return _decode_topology_feature_payload(raw_result)


def _extract_topology_feature_with_imodel_native(
    graph_payload: str,
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    imodel: Optional[Any],
    native_ctx: Optional[_NativeTopologyContext],
) -> Optional[Dict[str, float]]:
    if not (imodel is not None and hasattr(imodel, "_trained") and imodel._trained):
        return None
    required_attrs = ("op_names", "u", "v", "W_s", "W_l", "b_s", "b_l")
    if not all(hasattr(imodel, attr) for attr in required_attrs):
        return None
    rust = _try_import_rust_scheduler()
    if rust is None:
        return None
    ctx = native_ctx or _make_native_topology_context(op_profiles, pair_stability)
    if hasattr(rust, "extract_topology_feature_map_with_imodel_native_py"):
        try:
            raw_result = rust.extract_topology_feature_map_with_imodel_native_py(
                graph_payload,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
                list(imodel.op_names),
                np.ascontiguousarray(imodel.u, dtype=np.float32),
                np.ascontiguousarray(imodel.v, dtype=np.float32),
                np.ascontiguousarray(imodel.W_s, dtype=np.float32),
                np.ascontiguousarray(imodel.W_l, dtype=np.float32),
                float(imodel.b_s),
                float(imodel.b_l),
            )
        except Exception:
            raw_result = None
        mapped = _coerce_topology_feature_mapping(raw_result)
        if mapped is not None:
            return mapped
    if not hasattr(rust, "extract_topology_features_with_imodel_native"):
        return None
    try:
        raw_result = rust.extract_topology_features_with_imodel_native(
            graph_payload,
            ctx.op_profiles_json,
            ctx.pair_stability_json,
            ctx.op_metadata_json,
            list(imodel.op_names),
            np.ascontiguousarray(imodel.u, dtype=np.float32),
            np.ascontiguousarray(imodel.v, dtype=np.float32),
            np.ascontiguousarray(imodel.W_s, dtype=np.float32),
            np.ascontiguousarray(imodel.W_l, dtype=np.float32),
            float(imodel.b_s),
            float(imodel.b_l),
        )
    except Exception:
        return None
    return _decode_topology_feature_payload(raw_result)


def _extract_topology_base_features_native(
    serialized: List[str],
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    native_ctx: Optional[_NativeTopologyContext],
) -> Optional[List[Optional[Dict[str, float]]]]:
    rust = _try_import_rust_scheduler()
    if rust is None:
        return None
    ctx = native_ctx or _make_native_topology_context(op_profiles, pair_stability)
    if hasattr(rust, "extract_topology_feature_maps_batch_native_py"):
        try:
            raw_result = rust.extract_topology_feature_maps_batch_native_py(
                serialized,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
            )
        except Exception:
            raw_result = None
        if isinstance(raw_result, list):
            base_features = [
                _coerce_topology_feature_mapping(payload) for payload in raw_result
            ]
            if len(base_features) == len(serialized):
                return base_features
    if not hasattr(rust, "extract_topology_features_batch_native"):
        return None
    raw_result = rust.extract_topology_features_batch_native(
        serialized,
        ctx.op_profiles_json,
        ctx.pair_stability_json,
        ctx.op_metadata_json,
    )
    if not isinstance(raw_result, list):
        return None
    base_features: List[Optional[Dict[str, float]]] = []
    for payload in raw_result:
        base_features.append(_decode_topology_feature_payload(payload))
    if len(base_features) != len(serialized):
        return None
    return base_features


def _extract_topology_features_with_imodel_batch_native(
    serialized: List[str],
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    imodel: Optional[Any],
    native_ctx: Optional[_NativeTopologyContext],
) -> Optional[List[Optional[Dict[str, float]]]]:
    if not (imodel is not None and hasattr(imodel, "_trained") and imodel._trained):
        return None
    required_attrs = ("op_names", "u", "v", "W_s", "W_l", "b_s", "b_l")
    if not all(hasattr(imodel, attr) for attr in required_attrs):
        return None
    rust = _try_import_rust_scheduler()
    if rust is None:
        return None
    ctx = native_ctx or _make_native_topology_context(op_profiles, pair_stability)
    if hasattr(rust, "extract_topology_feature_maps_with_imodel_batch_native_py"):
        try:
            raw_result = rust.extract_topology_feature_maps_with_imodel_batch_native_py(
                serialized,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
                list(imodel.op_names),
                np.ascontiguousarray(imodel.u, dtype=np.float32),
                np.ascontiguousarray(imodel.v, dtype=np.float32),
                np.ascontiguousarray(imodel.W_s, dtype=np.float32),
                np.ascontiguousarray(imodel.W_l, dtype=np.float32),
                float(imodel.b_s),
                float(imodel.b_l),
            )
        except Exception:
            raw_result = None
        if isinstance(raw_result, list):
            decoded_features = [
                _coerce_topology_feature_mapping(payload) for payload in raw_result
            ]
            if len(decoded_features) == len(serialized):
                return decoded_features
    if not hasattr(rust, "extract_topology_features_with_imodel_batch_native"):
        return None
    try:
        raw_result = rust.extract_topology_features_with_imodel_batch_native(
            serialized,
            ctx.op_profiles_json,
            ctx.pair_stability_json,
            ctx.op_metadata_json,
            list(imodel.op_names),
            np.ascontiguousarray(imodel.u, dtype=np.float32),
            np.ascontiguousarray(imodel.v, dtype=np.float32),
            np.ascontiguousarray(imodel.W_s, dtype=np.float32),
            np.ascontiguousarray(imodel.W_l, dtype=np.float32),
            float(imodel.b_s),
            float(imodel.b_l),
        )
    except Exception:
        return None
    if not isinstance(raw_result, list):
        return None
    decoded_features: List[Optional[Dict[str, float]]] = []
    for payload in raw_result:
        decoded_features.append(_decode_topology_feature_payload(payload))
    if len(decoded_features) != len(serialized):
        return None
    return decoded_features


def _extract_topology_features_batch(
    graph_payloads: List[Any],
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    *,
    imodel: Optional[Any] = None,
    native_ctx: Optional[_NativeTopologyContext] = None,
) -> List[Optional[Dict[str, float]]]:
    if not graph_payloads:
        return []

    serialized = [_serialize_graph_payload(payload) or "" for payload in graph_payloads]

    fused_features = _extract_topology_features_with_imodel_batch_native(
        serialized, op_profiles, pair_stability, imodel, native_ctx
    )
    if fused_features is not None:
        return fused_features

    base_features = _extract_topology_base_features_native(
        serialized, op_profiles, pair_stability, native_ctx
    )
    if base_features is not None:
        edge_pairs_batch = None
        if imodel is not None and hasattr(imodel, "_trained") and imodel._trained:
            edge_pairs_batch = _extract_edge_op_pairs_batch_native(serialized)
        if edge_pairs_batch is not None:
            return [
                _augment_imodel_features_from_pairs(feats, edge_pairs, imodel)
                for feats, edge_pairs in zip(
                    base_features, edge_pairs_batch, strict=False
                )
            ]
        return [
            _augment_imodel_features(feats, graph_payload, imodel)
            for graph_payload, feats in zip(serialized, base_features, strict=False)
        ]

    return [
        extract_topology_features(
            payload,
            op_profiles,
            pair_stability,
            imodel=imodel,
            native_ctx=native_ctx,
        )
        for payload in graph_payloads
    ]


def extract_topology_features(
    graph_json: Any,
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    imodel: Optional[Any] = None,
    native_ctx: Optional[_NativeTopologyContext] = None,
) -> Optional[Dict[str, float]]:
    graph_payload = _serialize_graph_payload(graph_json)
    if graph_payload is None:
        return None

    fused_feature = _extract_topology_feature_with_imodel_native(
        graph_payload, op_profiles, pair_stability, imodel, native_ctx
    )
    if fused_feature is not None:
        _NATIVE_FALLBACK_COUNTERS["topology_features_native_hit"] += 1
        return fused_feature

    base_feature = _extract_topology_base_feature_native(
        graph_payload, op_profiles, pair_stability, native_ctx
    )

    if base_feature is None:
        _NATIVE_FALLBACK_COUNTERS["topology_features_python_fallback"] += 1
        return _extract_topology_features_python(
            graph_payload, op_profiles, pair_stability, imodel=imodel
        )
    _NATIVE_FALLBACK_COUNTERS["topology_features_native_hit"] += 1
    return _augment_imodel_features(base_feature, graph_payload, imodel)


def _train_interaction_model_for_predictor(
    notebook_db: Path, profiling_db: Path
) -> Optional[Any]:
    try:
        from .interaction_model import InteractionModel

        trained = InteractionModel.train(
            notebook_db=notebook_db, profiling_db=profiling_db, n_epochs=30
        )
        return trained if trained._trained else None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("GraphPredictor interaction-model features disabled: %s", exc)
        return None


def _empty_predictor_kwargs(
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    imodel: Optional[Any],
    native_ctx: Optional[_NativeTopologyContext],
) -> Dict[str, Any]:
    return dict(
        w_gate=np.zeros(0),
        b_gate=0.0,
        gate_threshold=0.5,
        gate_calibration_a=1.0,
        gate_calibration_b=0.0,
        w_rank=np.zeros(0),
        b_rank=5.0,
        rank_log_min=float(np.log(1.0)),
        rank_log_max=float(np.log(1e6)),
        w_loss=np.zeros(0),
        b_loss=0.7,
        w_induction=np.zeros(0),
        b_induction=0.0,
        op_profiles=op_profiles,
        pair_stability=pair_stability,
        imodel=imodel,
        _native_topology_ctx=native_ctx,
    )


def _extract_predictor_training_features(
    rows: List[Dict[str, Any]],
    op_profiles: Dict[str, Dict[str, float]],
    pair_stability: Dict[Tuple[str, str], float],
    trained_imodel: Optional[Any],
    native_ctx: Optional[_NativeTopologyContext],
) -> Dict[str, Any]:
    feat_dicts: List[Dict[str, float]] = []
    gate_labels: List[int] = []
    rank_labels: List[float] = []
    loss_labels: List[float] = []
    induction_labels: List[float] = []
    gate_sample_weights: List[float] = []
    graph_signatures: List[str] = []

    graph_payloads = [row["graph_json"] for row in rows]
    batch_features = _extract_topology_features_batch(
        graph_payloads,
        op_profiles,
        pair_stability,
        imodel=trained_imodel,
        native_ctx=native_ctx,
    )

    for row, feats in zip(rows, batch_features, strict=False):
        s1 = bool(row["stage1_any_passed"])
        ppl = row.get("wikitext_perplexity_best")
        lr = row.get("loss_ratio_best")
        induction_auc = row.get("induction_auc_500")
        s0 = bool(row.get("stage0_any_passed"))
        s05 = bool(row.get("stage05_any_passed"))
        rerun_weight = rerun_confidence_weight(int(row.get("n_rows", 1)))
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature or feats is None:
            continue
        feat_dicts.append(feats)
        graph_signatures.append(signature)
        gate_labels.append(int(s1 or 0))
        hard_neg_mult = 1.0
        if not s1:
            if s05:
                hard_neg_mult = 2.5
            elif s0:
                hard_neg_mult = 1.5
        gate_sample_weights.append(hard_neg_mult * rerun_weight)
        rank_labels.append(
            float(ppl)
            if ppl is not None and math.isfinite(float(ppl))
            else float("nan")
        )
        loss_labels.append(
            float(lr)
            if s1 and lr is not None and math.isfinite(float(lr))
            else float("nan")
        )
        induction_labels.append(
            float(induction_auc)
            if induction_auc is not None and math.isfinite(float(induction_auc))
            else float("nan")
        )
    return {
        "feat_dicts": feat_dicts,
        "graph_signatures": graph_signatures,
        "y_gate": np.array(gate_labels, dtype=np.float64),
        "y_rank": np.array(rank_labels, dtype=np.float64),
        "y_loss": np.array(loss_labels, dtype=np.float64),
        "y_induction": np.array(induction_labels, dtype=np.float64),
        "sample_weights": np.array(gate_sample_weights, dtype=np.float64),
    }


def _split_and_normalize_features(
    X: np.ndarray,
    y_gate: np.ndarray,
    sample_weights: np.ndarray,
    graph_signatures: List[str],
    n_total: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y_gate.astype(np.int32), seed=seed
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        logger.warning("GraphPredictor grouped split failed; falling back to row split")
        pos_idx = np.where(y_gate == 1)[0]
        neg_idx = np.where(y_gate == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        pos_split = int(len(pos_idx) * 0.8)
        neg_split = int(len(neg_idx) * 0.8)
        train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
        val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])
        split_stats = {
            "n_unique_graphs": n_total,
            "n_duplicate_groups": 0,
            "n_ambiguous_duplicate_groups": 0,
        }

    feat_mean = X[train_idx].mean(axis=0)
    feat_std = X[train_idx].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_norm = (X - feat_mean) / feat_std
    X_norm = _clip_normalized_features(X_norm)

    X_tr, X_va = X_norm[train_idx], X_norm[val_idx]
    y_gate_tr, y_gate_va = y_gate[train_idx], y_gate[val_idx]
    gate_w_tr = sample_weights[train_idx]
    if np.unique(y_gate_tr).size < 2 or np.unique(y_gate_va).size < 2:
        logger.info(
            "GraphPredictor: split lost class diversity (train=%d classes, val=%d classes)",
            np.unique(y_gate_tr).size,
            np.unique(y_gate_va).size,
        )
        return None
    return {
        "X_tr": X_tr,
        "X_va": X_va,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "y_gate_tr": y_gate_tr,
        "y_gate_va": y_gate_va,
        "gate_w_tr": gate_w_tr,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "split_stats": split_stats,
    }


def _train_gate_head(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_gate_tr: np.ndarray,
    y_gate_va: np.ndarray,
    gate_w_tr: np.ndarray,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    n_features = X_tr.shape[1]
    n_pos_tr = float(np.sum(y_gate_tr))
    n_neg_tr = float(len(y_gate_tr) - n_pos_tr)
    pos_weight = max(n_neg_tr / max(n_pos_tr, 1.0), 1.0)
    fit_sample_weight = np.where(y_gate_tr > 0.5, pos_weight, 1.0) * gate_w_tr
    rng = np.random.RandomState(seed)
    gate_threshold = 0.5
    val_auc = 0.0
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=float(1.0 / max(alpha, 1e-6)),
            solver="lbfgs",
            max_iter=1200,
            random_state=seed,
        )
        clf.fit(X_tr, y_gate_tr.astype(np.int32), sample_weight=fit_sample_weight)
        w_gate = clf.coef_[0].astype(np.float64)
        b_gate = float(clf.intercept_[0])
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.warning("GraphPredictor logistic fit fell back to SGD: %s", exc)
        w_gate = rng.randn(n_features).astype(np.float64) * 0.01
        b_gate = 0.0
        gate_lr = 0.01
        for _ in range(80):
            perm = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), 128):
                idx = perm[start : start + 128]
                x_b = X_tr[idx]
                y_b = y_gate_tr[idx]
                preds = _sigmoid(x_b @ w_gate + b_gate)
                grad = (preds - y_b)[:, None] * x_b
                w_gate -= gate_lr * grad.mean(axis=0) + gate_lr * alpha * 0.001 * w_gate
                b_gate -= gate_lr * float((preds - y_b).mean())

    raw_val_logits = X_va @ w_gate + b_gate
    gate_calibration_a = 1.0
    gate_calibration_b = 0.0
    try:
        from sklearn.linear_model import LogisticRegression

        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200, random_state=seed)
        cal.fit(raw_val_logits.reshape(-1, 1), y_gate_va.astype(np.int32))
        gate_calibration_a = float(cal.coef_[0][0])
        gate_calibration_b = float(cal.intercept_[0])
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.warning("GraphPredictor calibration skipped: %s", exc)

    val_preds = _sigmoid(raw_val_logits * gate_calibration_a + gate_calibration_b)
    eps = 1e-8
    val_loss = float(
        -np.mean(
            y_gate_va * np.log(val_preds + eps)
            + (1 - y_gate_va) * np.log(1 - val_preds + eps)
        )
    )
    try:
        val_auc = safe_binary_roc_auc(y_gate_va, val_preds)
        operating_points = operating_point_profiles(y_gate_va, val_preds)
        gate_threshold = float(operating_points["f1"]["threshold"])
        selected_metrics = operating_points["f1"]
        val_acc = float(selected_metrics["accuracy"])
        val_precision = float(selected_metrics["precision_ppv"])
        val_recall = float(selected_metrics["recall_tpr_sensitivity"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("GraphPredictor operating-point metrics degraded: %s", exc)
        operating_points = {}
        val_acc = float(np.mean((val_preds > gate_threshold) == y_gate_va))
        val_auc = 0.0
        val_precision = 0.0
        val_recall = 0.0
    val_gate_metrics = binary_classification_metrics(
        y_gate_va, val_preds, gate_threshold
    )
    val_gate_metrics["roc_auc"] = val_auc
    return {
        "w": w_gate,
        "b": b_gate,
        "threshold": gate_threshold,
        "cal_a": gate_calibration_a,
        "cal_b": gate_calibration_b,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_gate_metrics": val_gate_metrics,
        "operating_points": operating_points,
        "pos_weight": pos_weight,
    }


def _solve_ridge(
    X_sub: np.ndarray, y_sub: np.ndarray, n_features: int, alpha: float
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    XtX = X_sub.T @ X_sub + alpha * np.eye(n_features)
    Xty = X_sub.T @ y_sub
    try:
        w = np.linalg.solve(XtX, Xty)
        b = float(np.mean(y_sub - X_sub @ w))
        return w, b
    except np.linalg.LinAlgError as exc:
        logger.warning("GraphPredictor ridge solve failed: %s", exc)
        return None, None


def _train_ridge_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_features: int,
    alpha: float,
    default_bias: float,
    min_samples: int = 20,
) -> Dict[str, Any]:
    mask = np.isfinite(y_tr)
    w = np.zeros(n_features, dtype=np.float64)
    b = default_bias
    if mask.sum() >= min_samples:
        solved_w, solved_b = _solve_ridge(X_tr[mask], y_tr[mask], n_features, alpha)
        if solved_w is not None:
            w, b = solved_w, solved_b
    return {"w": w, "b": b}


def _train_ridge_head_with_log_bounds(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_features: int,
    alpha: float,
    default_bias: float,
    min_samples: int = 20,
) -> Dict[str, Any]:
    mask = np.isfinite(y_tr)
    w = np.zeros(n_features, dtype=np.float64)
    b = default_bias
    log_min = float(np.log(1.0))
    log_max = float(np.log(1e6))
    if mask.sum() >= min_samples:
        X_sub = X_tr[mask]
        y_log = np.log(np.maximum(y_tr[mask], 1.0))
        log_min = float(np.percentile(y_log, 0.5))
        log_max = float(np.percentile(y_log, 99.5))
        if log_max < log_min:
            log_max = log_min
        solved_w, solved_b = _solve_ridge(X_sub, y_log, n_features, alpha)
        if solved_w is not None:
            w, b = solved_w, solved_b
    return {"w": w, "b": b, "log_min": log_min, "log_max": log_max}


def _train_induction_head(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    n_features: int,
    alpha: float,
) -> Dict[str, Any]:
    tr_mask = np.isfinite(y_tr)
    va_mask = np.isfinite(y_va)
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    mae = 0.0
    spearman = 0.0
    learner_acc = 0.0
    if tr_mask.sum() >= 50:
        solved_w, solved_b = _solve_ridge(
            X_tr[tr_mask], y_tr[tr_mask], n_features, alpha
        )
        if solved_w is not None:
            w, b = solved_w, solved_b
        if va_mask.sum() >= 10:
            y_auc_val = y_va[va_mask]
            pred_auc_val = np.clip(X_va[va_mask] @ w + b, 0.0, 1.0)
            mae = float(np.mean(np.abs(y_auc_val - pred_auc_val)))
            try:
                from scipy.stats import spearmanr

                rho, _ = spearmanr(y_auc_val, pred_auc_val)
                spearman = float(rho) if np.isfinite(rho) else 0.0
            except (ImportError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "GraphPredictor induction Spearman computation skipped: %s", exc
                )
                spearman = 0.0
            y_bucket_val = (y_auc_val >= 0.02).astype(np.int32)
            pred_bucket_val = (pred_auc_val >= 0.02).astype(np.int32)
            learner_acc = float(np.mean(y_bucket_val == pred_bucket_val))
    return {
        "w": w,
        "b": b,
        "mae": mae,
        "spearman": spearman,
        "learner_acc": learner_acc,
    }


@dataclass(slots=True)
class GraphPredictor:
    """Topology-aware graph predictor using Ridge regression on structural features."""

    # Gate model (logistic regression)
    w_gate: np.ndarray  # (n_features,)
    b_gate: float
    # Rank model (linear regression on log-ppl)
    w_rank: np.ndarray  # (n_features,)
    b_rank: float
    rank_log_min: float = 0.0
    rank_log_max: float = 20.0
    # Feature metadata
    feature_names: List[str] = field(default_factory=list)
    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Loss prediction head
    w_loss: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_loss: float = 0.7
    # Induction prediction head
    w_induction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_induction: float = 0.0
    # Op data (cached for feature extraction)
    op_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pair_stability: Dict[Tuple[str, str], float] = field(default_factory=dict)
    imodel: Optional[Any] = None  # InteractionModel, cached for feature extraction
    _native_topology_ctx: Optional[_NativeTopologyContext] = field(
        default=None, repr=False
    )
    # Metadata
    gate_threshold: float = 0.5
    gate_calibration_a: float = 1.0
    gate_calibration_b: float = 0.0
    n_train: int = 0
    _trained: bool = False
    _train_metrics: Dict[str, float] = field(default_factory=dict)

    def is_fitted(self) -> bool:
        return self._trained and len(self.w_gate) > 0

    def _extract_and_normalize(self, graph_json: Any) -> Optional[np.ndarray]:
        feats = extract_topology_features(
            graph_json,
            self.op_profiles,
            self.pair_stability,
            imodel=self.imodel,
            native_ctx=self._native_topology_ctx,
        )
        if feats is None:
            return None
        feats = _materialize_feature_dict(feats)
        x = np.array([feats.get(k, 0.0) for k in self.feature_names], dtype=np.float64)
        x = (x - self.feature_mean) / self.feature_std
        return _clip_normalized_features(x)

    def predict_gate(self, graph_json: Any) -> float:
        """Predict P(pass_s1). Returns 0.5 if not fitted."""
        if not self.is_fitted():
            return 0.5
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 0.5
        logit = float(x @ self.w_gate + self.b_gate)
        calibrated = self.gate_calibration_a * logit + self.gate_calibration_b
        return float(_sigmoid(np.array([calibrated]))[0])

    def predict_rank(self, graph_json: Any) -> float:
        """Predict wikitext perplexity. Returns 1e6 if not fitted."""
        if not self.is_fitted():
            return 1e6
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 1e6
        log_ppl = float(x @ self.w_rank + self.b_rank)
        log_ppl = float(np.clip(log_ppl, self.rank_log_min, self.rank_log_max))
        return float(np.exp(log_ppl))

    def predict_loss(self, graph_json: Any) -> float:
        """Predict loss_ratio for S1-passing graphs. Returns 0.7 if not fitted."""
        if not self.is_fitted() or len(self.w_loss) == 0:
            return 0.7
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 0.7
        return float(np.clip(x @ self.w_loss + self.b_loss, 0.0, 2.0))

    def predict_induction_auc(self, graph_json: Any) -> float:
        """Predict canonical induction AUC. Returns 0.0 if not fitted."""
        if not self.is_fitted() or len(self.w_induction) == 0:
            return 0.0
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 0.0
        return float(np.clip(x @ self.w_induction + self.b_induction, 0.0, 1.0))

    @classmethod
    def train(
        cls,
        notebook_db: Path = _DEFAULT_NOTEBOOK_DB,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> "GraphPredictor":
        """Train topology-aware predictor from notebook data.

        Extracts topology features from each graph, then fits:
        - Logistic regression for gate (SGD)
        - Ridge regression for rank (wikitext ppl)
        - Ridge regression for loss (loss_ratio of S1-passing graphs)
        """
        op_profiles = _load_op_profiles(profiling_db)
        pair_stability = _load_pair_stability(profiling_db)
        native_ctx = _make_native_topology_context(op_profiles, pair_stability)
        trained_imodel = _train_interaction_model_for_predictor(
            notebook_db, profiling_db
        )
        empty_kwargs = _empty_predictor_kwargs(
            op_profiles, pair_stability, trained_imodel, native_ctx
        )

        if not notebook_db.exists():
            return cls(**empty_kwargs)

        try:
            rows = load_screening_predictor_corpus_rows(notebook_db, validate=True)
        except CorpusIntegrityError:
            raise
        except Exception as e:
            logger.warning("GraphPredictor training data query failed: %s", e)
            return cls(**empty_kwargs)

        extracted = _extract_predictor_training_features(
            rows, op_profiles, pair_stability, trained_imodel, native_ctx
        )
        n_total = len(extracted["feat_dicts"])
        if n_total < _MIN_SAMPLES:
            logger.info(
                "GraphPredictor: insufficient data (%d < %d)", n_total, _MIN_SAMPLES
            )
            return cls(**empty_kwargs)

        X, feature_names = build_dense_feature_matrix(
            extracted["feat_dicts"], dtype=np.float64
        )
        feature_names, X = _augment_feature_space(feature_names, X)
        y_gate = extracted["y_gate"]
        y_rank = extracted["y_rank"]
        y_loss = extracted["y_loss"]
        y_induction = extracted["y_induction"]
        sample_weights = extracted["sample_weights"]
        graph_signatures = extracted["graph_signatures"]

        if int(np.sum(y_gate)) < 5 or int(len(y_gate) - np.sum(y_gate)) < 5:
            logger.info("GraphPredictor: insufficient class balance")
            return cls(**empty_kwargs)

        split = _split_and_normalize_features(
            X, y_gate, sample_weights, graph_signatures, n_total, seed
        )
        if split is None:
            return cls(**empty_kwargs)

        X_tr = split["X_tr"]
        X_va = split["X_va"]
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        y_gate_tr = split["y_gate_tr"]
        y_gate_va = split["y_gate_va"]
        gate_w_tr = split["gate_w_tr"]
        n_features = X_tr.shape[1]
        logger.info(
            "GraphPredictor training: %d samples, %d topology features",
            n_total,
            n_features,
        )

        gate = _train_gate_head(
            X_tr, X_va, y_gate_tr, y_gate_va, gate_w_tr, alpha, seed
        )

        rank = _train_ridge_head_with_log_bounds(
            X_tr, y_rank[train_idx], n_features, alpha, default_bias=5.0
        )
        loss = _train_ridge_head(
            X_tr, y_loss[train_idx], n_features, alpha, default_bias=0.7
        )
        induction = _train_induction_head(
            X_tr,
            X_va,
            y_induction[train_idx],
            y_induction[val_idx],
            n_features,
            alpha,
        )

        logger.info(
            "GraphPredictor trained: val_loss=%.4f val_acc=%.3f (%d train, %d val, %d features, imodel=%s)",
            gate["val_loss"],
            gate["val_acc"],
            len(X_tr),
            len(X_va),
            n_features,
            trained_imodel is not None,
        )

        return cls(
            w_gate=gate["w"].astype(np.float32),
            b_gate=float(gate["b"]),
            gate_threshold=float(gate["threshold"]),
            gate_calibration_a=float(gate["cal_a"]),
            gate_calibration_b=float(gate["cal_b"]),
            w_rank=rank["w"].astype(np.float32),
            b_rank=float(rank["b"]),
            rank_log_min=float(rank["log_min"]),
            rank_log_max=float(rank["log_max"]),
            w_loss=loss["w"].astype(np.float32),
            b_loss=float(loss["b"]),
            w_induction=induction["w"].astype(np.float32),
            b_induction=float(induction["b"]),
            feature_names=feature_names,
            feature_mean=split["feat_mean"].astype(np.float32),
            feature_std=split["feat_std"].astype(np.float32),
            op_profiles=op_profiles,
            pair_stability=pair_stability,
            imodel=trained_imodel,
            _native_topology_ctx=native_ctx,
            n_train=len(X_tr),
            _trained=True,
            _train_metrics={
                "val_loss": gate["val_loss"],
                "val_accuracy": gate["val_acc"],
                "val_auc": gate["val_auc"],
                "val_precision": gate["val_precision"],
                "val_recall": gate["val_recall"],
                "rank_log_min": float(rank["log_min"]),
                "rank_log_max": float(rank["log_max"]),
                "gate_threshold": float(gate["threshold"]),
                "val_gate_metrics": gate["val_gate_metrics"],
                "operating_points": gate["operating_points"],
                "pos_weight": float(gate["pos_weight"]),
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "n_features": n_features,
                "n_positive": int(y_gate.sum()),
                "induction_mae": induction["mae"],
                "induction_spearman": induction["spearman"],
                "induction_learner_acc": induction["learner_acc"],
                **split["split_stats"],
            },
        )

    def save(self, path: Path) -> None:
        """Save model weights/normalization metadata to npz + JSON."""
        path = Path(path)
        imodel_path = None
        if self.imodel is not None and getattr(self.imodel, "_trained", False):
            imodel_path = _graph_imodel_path(path)
            self.imodel.save(imodel_path)
        save_npz_with_metadata(
            path,
            arrays={
                "w_gate": self.w_gate,
                "w_rank": self.w_rank,
                "w_loss": self.w_loss,
                "w_induction": self.w_induction,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
            },
            metadata={
                "b_gate": self.b_gate,
                "gate_threshold": self.gate_threshold,
                "gate_calibration_a": self.gate_calibration_a,
                "gate_calibration_b": self.gate_calibration_b,
                "b_rank": self.b_rank,
                "rank_log_min": self.rank_log_min,
                "rank_log_max": self.rank_log_max,
                "b_loss": self.b_loss,
                "b_induction": self.b_induction,
                "feature_names": self.feature_names,
                "n_train": self.n_train,
                "trained": self._trained,
                "interaction_model_path": imodel_path.name if imodel_path else None,
                "train_metrics": self._train_metrics,
            },
        )

    @classmethod
    def load(
        cls,
        path: Path,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
    ) -> "GraphPredictor":
        """Load model weights/metadata from disk and refresh profiling caches."""
        path = Path(path)
        data, meta = load_npz_with_metadata(path)
        trained_imodel = None
        imodel_relpath = meta.get("interaction_model_path")
        if isinstance(imodel_relpath, str) and imodel_relpath:
            try:
                from .interaction_model import InteractionModel

                imodel_path = path.parent / imodel_relpath
                if imodel_path.exists():
                    trained_imodel = InteractionModel.load(imodel_path)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "GraphPredictor interaction sidecar load skipped: %s", exc
                )
        return cls(
            _native_topology_ctx=_make_native_topology_context(
                _load_op_profiles(profiling_db),
                _load_pair_stability(profiling_db),
            ),
            w_gate=data["w_gate"],
            b_gate=float(meta.get("b_gate", 0.0)),
            gate_threshold=float(meta.get("gate_threshold", 0.5)),
            gate_calibration_a=float(meta.get("gate_calibration_a", 1.0)),
            gate_calibration_b=float(meta.get("gate_calibration_b", 0.0)),
            w_rank=data["w_rank"],
            b_rank=float(meta.get("b_rank", 5.0)),
            rank_log_min=float(meta.get("rank_log_min", np.log(1.0))),
            rank_log_max=float(meta.get("rank_log_max", np.log(1e6))),
            w_loss=data["w_loss"],
            w_induction=data["w_induction"] if "w_induction" in data else np.zeros(0),
            b_loss=float(meta.get("b_loss", 0.7)),
            b_induction=float(meta.get("b_induction", 0.0)),
            feature_names=list(meta.get("feature_names", [])),
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            op_profiles=_load_op_profiles(profiling_db),
            pair_stability=_load_pair_stability(profiling_db),
            imodel=trained_imodel,
            n_train=int(meta.get("n_train", 0)),
            _trained=bool(meta.get("trained", False)),
            _train_metrics=dict(meta.get("train_metrics", {})),
        )
