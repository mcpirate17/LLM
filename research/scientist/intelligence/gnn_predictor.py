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
import sqlite3
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

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[2] / "lab_notebook.db"
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)

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


def _load_op_profiles(profiling_db: Path) -> Dict[str, Dict[str, float]]:
    """Load per-op profiling stats for topology feature computation."""
    profiles: Dict[str, Dict[str, float]] = {}
    if not profiling_db.exists():
        return profiles
    try:
        conn = sqlite3.connect(str(profiling_db), timeout=5)
        rows = conn.execute(
            """SELECT op_name, output_std, grad_norm, lipschitz_estimate,
                      grad_vanishing, grad_exploding, output_has_nan, has_params
               FROM op_profiles WHERE error IS NULL"""
        ).fetchall()
        conn.close()
        for op, out_std, gn, lip, gv, ge, nan_out, has_p in rows:
            profiles[op] = {
                "output_std": float(out_std) if out_std else 1.0,
                "grad_norm": float(gn) if gn else 1.0,
                "lipschitz": float(lip) if lip else 1.0,
                "grad_vanishing": float(gv or 0),
                "grad_exploding": float(ge or 0),
                "has_nan": float(nan_out or 0),
                "has_params": float(has_p or 0),
            }
    except Exception as e:
        logger.warning("Failed to load op profiles: %s", e)
    return profiles


def _load_pair_stability(profiling_db: Path) -> Dict[Tuple[str, str], float]:
    """Load pair stability rates from profiling DB."""
    pairs: Dict[Tuple[str, str], float] = {}
    if not profiling_db.exists():
        return pairs
    try:
        conn = sqlite3.connect(str(profiling_db), timeout=5)
        rows = conn.execute(
            """SELECT op_a, op_b,
                      (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) as stable
               FROM pair_profiles WHERE error IS NULL AND composition = 'sequential'"""
        ).fetchall()
        conn.close()
        for a, b, stable in rows:
            pairs[(a, b)] = float(stable)
    except Exception as e:
        logger.warning("Failed to load pair stability: %s", e)
    return pairs


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
    return pairs


def _extract_edge_op_pairs_native(
    graph_payload: str,
) -> Optional[List[Tuple[str, str]]]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_edge_op_pairs_native"):
        return None
    try:
        payload = rust.extract_edge_op_pairs_native(graph_payload)
        loaded = json.loads(payload)
    except Exception as exc:
        logger.warning(
            "Native edge-pair extraction failed; falling back to Python: %s", exc
        )
        return None
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
    return pairs


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

    serialized: List[str] = []
    for payload in graph_payloads:
        if isinstance(payload, str):
            serialized.append(payload)
        else:
            try:
                serialized.append(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                )
            except (TypeError, ValueError):
                serialized.append("")

    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, "extract_topology_features_batch_native"):
        try:
            ctx = native_ctx or _make_native_topology_context(
                op_profiles, pair_stability
            )
            raw_result = rust.extract_topology_features_batch_native(
                serialized,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
            )
            if isinstance(raw_result, list):
                base_features: List[Optional[Dict[str, float]]] = []
                for payload in raw_result:
                    if not isinstance(payload, str):
                        base_features.append(None)
                        continue
                    decoded = json.loads(payload)
                    if isinstance(decoded, dict):
                        base_features.append(
                            {str(key): float(value) for key, value in decoded.items()}
                        )
                    else:
                        base_features.append(None)
                if len(base_features) == len(serialized):
                    if not (
                        imodel is not None
                        and hasattr(imodel, "_trained")
                        and imodel._trained
                    ):
                        return base_features
                    enriched: List[Optional[Dict[str, float]]] = []
                    for graph_payload, feats in zip(
                        serialized, base_features, strict=False
                    ):
                        if feats is None:
                            enriched.append(None)
                            continue
                        edge_pairs = _extract_edge_op_pairs_native(graph_payload)
                        if edge_pairs is None:
                            edge_pairs = _extract_edge_op_pairs_python(graph_payload)
                        if edge_pairs:
                            imodel_stabilities = [
                                imodel.predict_stability(left, right)
                                for left, right in edge_pairs
                            ]
                            imodel_losses = [
                                imodel.predict_loss(left, right)
                                for left, right in edge_pairs
                            ]
                            feats["imodel_min_stability"] = float(
                                min(imodel_stabilities)
                            )
                            feats["imodel_mean_stability"] = float(
                                np.mean(imodel_stabilities)
                            )
                            feats["imodel_mean_loss"] = float(np.mean(imodel_losses))
                        else:
                            feats["imodel_min_stability"] = 0.5
                            feats["imodel_mean_stability"] = 0.5
                            feats["imodel_mean_loss"] = 0.7
                        enriched.append(feats)
                    return enriched
        except Exception as exc:
            logger.warning(
                "Native topology batch extraction failed; falling back per-graph: %s",
                exc,
            )

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
    rust = _try_import_rust_scheduler()
    graph_payload: str
    if isinstance(graph_json, str):
        graph_payload = graph_json
    else:
        try:
            graph_payload = json.dumps(
                graph_json, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            return None

    base_features: Optional[Dict[str, float]] = None
    if rust is not None and hasattr(rust, "extract_topology_features_native"):
        try:
            ctx = native_ctx or _make_native_topology_context(
                op_profiles, pair_stability
            )
            payload = rust.extract_topology_features_native(
                graph_payload,
                ctx.op_profiles_json,
                ctx.pair_stability_json,
                ctx.op_metadata_json,
            )
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                base_features = {
                    str(key): float(value) for key, value in loaded.items()
                }
        except Exception as exc:
            logger.warning(
                "Native topology extraction failed; falling back to Python: %s", exc
            )

    if base_features is None:
        return _extract_topology_features_python(
            graph_payload, op_profiles, pair_stability, imodel=imodel
        )

    if not (imodel is not None and hasattr(imodel, "_trained") and imodel._trained):
        base_features["imodel_min_stability"] = 0.5
        base_features["imodel_mean_stability"] = 0.5
        base_features["imodel_mean_loss"] = 0.7
        return base_features

    edge_pairs = _extract_edge_op_pairs_native(graph_payload)
    if edge_pairs is None:
        edge_pairs = _extract_edge_op_pairs_python(graph_payload)
    if not edge_pairs:
        base_features["imodel_min_stability"] = 0.5
        base_features["imodel_mean_stability"] = 0.5
        base_features["imodel_mean_loss"] = 0.7
        return base_features

    imodel_stabilities = [
        imodel.predict_stability(left, right) for left, right in edge_pairs
    ]
    imodel_losses = [imodel.predict_loss(left, right) for left, right in edge_pairs]

    if imodel_stabilities:
        base_features["imodel_min_stability"] = float(min(imodel_stabilities))
        base_features["imodel_mean_stability"] = float(np.mean(imodel_stabilities))
        base_features["imodel_mean_loss"] = float(np.mean(imodel_losses))
    else:
        base_features["imodel_min_stability"] = 0.5
        base_features["imodel_mean_stability"] = 0.5
        base_features["imodel_mean_loss"] = 0.7
    return base_features


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

        # Train interaction model for learned pair features
        trained_imodel = None
        try:
            from .interaction_model import InteractionModel

            trained_imodel = InteractionModel.train(
                notebook_db=notebook_db,
                profiling_db=profiling_db,
                n_epochs=30,
            )
            if not trained_imodel._trained:
                trained_imodel = None
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "GraphPredictor interaction-model features disabled: %s", exc
            )

        _empty_kwargs = dict(
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
            imodel=trained_imodel,
            _native_topology_ctx=native_ctx,
        )

        if not notebook_db.exists():
            return cls(**_empty_kwargs)

        try:
            rows = load_screening_predictor_corpus_rows(notebook_db, validate=True)
        except CorpusIntegrityError:
            raise
        except Exception as e:
            logger.warning("GraphPredictor training data query failed: %s", e)
            return cls(**_empty_kwargs)

        # Extract features
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
            gj = row["graph_json"]
            s1 = bool(row["stage1_any_passed"])
            ppl = row.get("wikitext_perplexity_best")
            lr = row.get("loss_ratio_best")
            induction_auc = row.get("induction_auc_500")
            s0 = bool(row.get("stage0_any_passed"))
            s05 = bool(row.get("stage05_any_passed"))
            rerun_weight = rerun_confidence_weight(int(row.get("n_rows", 1)))
            signature = str(row.get("canonical_fingerprint") or "")
            if not signature:
                continue
            if feats is None:
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

        n_total = len(feat_dicts)
        if n_total < _MIN_SAMPLES:
            logger.info(
                "GraphPredictor: insufficient data (%d < %d)", n_total, _MIN_SAMPLES
            )
            return cls(
                w_gate=np.zeros(0),
                b_gate=0.0,
                gate_threshold=0.5,
                w_rank=np.zeros(0),
                b_rank=5.0,
                rank_log_min=float(np.log(1.0)),
                rank_log_max=float(np.log(1e6)),
                w_induction=np.zeros(0),
                b_induction=0.0,
                op_profiles=op_profiles,
                pair_stability=pair_stability,
                imodel=trained_imodel,
                _native_topology_ctx=native_ctx,
            )

        # Build feature matrix
        X, feature_names = build_dense_feature_matrix(feat_dicts, dtype=np.float64)
        feature_names, X = _augment_feature_space(feature_names, X)

        y_gate = np.array(gate_labels, dtype=np.float64)
        y_rank = np.array(rank_labels, dtype=np.float64)
        y_induction = np.array(induction_labels, dtype=np.float64)
        sample_weights = np.array(gate_sample_weights, dtype=np.float64)
        rng = np.random.RandomState(seed)

        n_pos_total = int(np.sum(y_gate))
        n_neg_total = int(len(y_gate) - n_pos_total)
        if n_pos_total < 5 or n_neg_total < 5:
            logger.info(
                "GraphPredictor: insufficient class balance (pos=%d, neg=%d)",
                n_pos_total,
                n_neg_total,
            )
            return cls(**_empty_kwargs)

        # Train/val split grouped by exact graph to avoid duplicate leakage.
        train_idx, val_idx, split_stats = grouped_stratified_split(
            graph_signatures, y_gate.astype(np.int32), seed=seed
        )
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning(
                "GraphPredictor grouped split failed; falling back to row split"
            )
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

        # Standardize on train only.
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
            return cls(**_empty_kwargs)

        n_features = X_tr.shape[1]
        logger.info(
            "GraphPredictor training: %d samples, %d topology features",
            n_total,
            n_features,
        )

        # ── Gate: regularized logistic regression ──
        n_pos_tr = float(np.sum(y_gate_tr))
        n_neg_tr = float(len(y_gate_tr) - n_pos_tr)
        pos_weight = max(n_neg_tr / max(n_pos_tr, 1.0), 1.0)
        fit_sample_weight = np.where(y_gate_tr > 0.5, pos_weight, 1.0) * gate_w_tr
        gate_threshold = 0.5
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
            val_preds = clf.predict_proba(X_va)[:, 1]
            val_auc = safe_binary_roc_auc(y_gate_va, val_preds)
        except (ImportError, ValueError, RuntimeError) as exc:
            # Fallback: unweighted logistic SGD if sklearn is unavailable.
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
                    logits = x_b @ w_gate + b_gate
                    preds = _sigmoid(logits)
                    grad = (preds - y_b)[:, None] * x_b
                    w_gate -= (
                        gate_lr * grad.mean(axis=0) + gate_lr * alpha * 0.001 * w_gate
                    )
                    b_gate -= gate_lr * float((preds - y_b).mean())
            val_preds = _sigmoid(X_va @ w_gate + b_gate)
            val_auc = 0.0

        # Calibrate raw logits to better probabilities while preserving ranking.
        raw_val_logits = X_va @ w_gate + b_gate
        gate_calibration_a = 1.0
        gate_calibration_b = 0.0
        try:
            from sklearn.linear_model import LogisticRegression

            cal = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                max_iter=200,
                random_state=seed,
            )
            cal.fit(raw_val_logits.reshape(-1, 1), y_gate_va.astype(np.int32))
            gate_calibration_a = float(cal.coef_[0][0])
            gate_calibration_b = float(cal.intercept_[0])
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.warning("GraphPredictor calibration skipped: %s", exc)

        # Val metrics
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

        # ── Rank: Ridge regression on log-ppl ──
        rank_mask = np.isfinite(y_rank[train_idx])
        w_rank = np.zeros(n_features, dtype=np.float64)
        b_rank = 5.0
        rank_log_min = float(np.log(1.0))
        rank_log_max = float(np.log(1e6))
        if rank_mask.sum() >= 20:
            X_rank = X_tr[rank_mask]
            y_log_ppl = np.log(np.maximum(y_rank[train_idx][rank_mask], 1.0))
            rank_log_min = float(np.percentile(y_log_ppl, 0.5))
            rank_log_max = float(np.percentile(y_log_ppl, 99.5))
            if rank_log_max < rank_log_min:
                rank_log_max = rank_log_min
            XtX = X_rank.T @ X_rank + alpha * np.eye(n_features)
            Xty = X_rank.T @ y_log_ppl
            try:
                w_rank = np.linalg.solve(XtX, Xty)
                b_rank = float(np.mean(y_log_ppl - X_rank @ w_rank))
            except np.linalg.LinAlgError as exc:
                logger.warning("GraphPredictor rank head solve failed: %s", exc)

        # ── Loss: Ridge regression on loss_ratio (S1-passing only) ──
        y_loss = np.array(loss_labels, dtype=np.float64)
        loss_mask = np.isfinite(y_loss[train_idx])
        w_loss = np.zeros(n_features, dtype=np.float64)
        b_loss = 0.7
        if loss_mask.sum() >= 20:
            X_loss = X_tr[loss_mask]
            y_lr = y_loss[train_idx][loss_mask]
            XtX_l = X_loss.T @ X_loss + alpha * np.eye(n_features)
            Xty_l = X_loss.T @ y_lr
            try:
                w_loss = np.linalg.solve(XtX_l, Xty_l)
                b_loss = float(np.mean(y_lr - X_loss @ w_loss))
            except np.linalg.LinAlgError as exc:
                logger.warning("GraphPredictor loss head solve failed: %s", exc)

        # ── Induction: Ridge regression on canonical induction AUC ──
        induction_mask = np.isfinite(y_induction[train_idx])
        induction_val_mask = np.isfinite(y_induction[val_idx])
        w_induction = np.zeros(n_features, dtype=np.float64)
        b_induction = 0.0
        induction_mae = 0.0
        induction_spearman = 0.0
        induction_learner_acc = 0.0
        if induction_mask.sum() >= 50:
            X_induction = X_tr[induction_mask]
            y_auc = y_induction[train_idx][induction_mask]
            XtX_i = X_induction.T @ X_induction + alpha * np.eye(n_features)
            Xty_i = X_induction.T @ y_auc
            try:
                w_induction = np.linalg.solve(XtX_i, Xty_i)
                b_induction = float(np.mean(y_auc - X_induction @ w_induction))
            except np.linalg.LinAlgError as exc:
                logger.warning("GraphPredictor induction head solve failed: %s", exc)
            if induction_val_mask.sum() >= 10:
                y_auc_val = y_induction[val_idx][induction_val_mask]
                pred_auc_val = np.clip(
                    X_va[induction_val_mask] @ w_induction + b_induction, 0.0, 1.0
                )
                induction_mae = float(np.mean(np.abs(y_auc_val - pred_auc_val)))
                try:
                    from scipy.stats import spearmanr

                    rho, _ = spearmanr(y_auc_val, pred_auc_val)
                    induction_spearman = float(rho) if np.isfinite(rho) else 0.0
                except (ImportError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "GraphPredictor induction Spearman computation skipped: %s", exc
                    )
                    induction_spearman = 0.0
                y_bucket_val = (y_auc_val >= 0.02).astype(np.int32)
                pred_bucket_val = (pred_auc_val >= 0.02).astype(np.int32)
                induction_learner_acc = float(np.mean(y_bucket_val == pred_bucket_val))

        logger.info(
            "GraphPredictor trained: val_loss=%.4f val_acc=%.3f (%d train, %d val, %d features, imodel=%s)",
            val_loss,
            val_acc,
            len(X_tr),
            len(X_va),
            n_features,
            trained_imodel is not None,
        )

        return cls(
            w_gate=w_gate.astype(np.float32),
            b_gate=float(b_gate),
            gate_threshold=float(gate_threshold),
            gate_calibration_a=float(gate_calibration_a),
            gate_calibration_b=float(gate_calibration_b),
            w_rank=w_rank.astype(np.float32),
            b_rank=float(b_rank),
            rank_log_min=float(rank_log_min),
            rank_log_max=float(rank_log_max),
            w_loss=w_loss.astype(np.float32),
            b_loss=float(b_loss),
            w_induction=w_induction.astype(np.float32),
            b_induction=float(b_induction),
            feature_names=feature_names,
            feature_mean=feat_mean.astype(np.float32),
            feature_std=feat_std.astype(np.float32),
            op_profiles=op_profiles,
            pair_stability=pair_stability,
            imodel=trained_imodel,
            _native_topology_ctx=native_ctx,
            n_train=len(X_tr),
            _trained=True,
            _train_metrics={
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "rank_log_min": float(rank_log_min),
                "rank_log_max": float(rank_log_max),
                "gate_threshold": float(gate_threshold),
                "val_gate_metrics": val_gate_metrics,
                "operating_points": operating_points,
                "pos_weight": float(pos_weight),
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "n_features": n_features,
                "n_positive": int(y_gate.sum()),
                "induction_mae": induction_mae,
                "induction_spearman": induction_spearman,
                "induction_learner_acc": induction_learner_acc,
                **split_stats,
            },
        )

    def save(self, path: Path) -> None:
        """Save model weights/normalization metadata to npz + JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            w_gate=self.w_gate,
            w_rank=self.w_rank,
            w_loss=self.w_loss,
            w_induction=self.w_induction,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        imodel_path = None
        if self.imodel is not None and getattr(self.imodel, "_trained", False):
            imodel_path = _graph_imodel_path(path)
            self.imodel.save(imodel_path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {
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
                f,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        path: Path,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
    ) -> "GraphPredictor":
        """Load model weights/metadata from disk and refresh profiling caches."""
        path = Path(path)
        data = np.load(str(path))
        with open(path.with_suffix(".json"), encoding="utf-8") as f:
            meta = json.load(f)
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
