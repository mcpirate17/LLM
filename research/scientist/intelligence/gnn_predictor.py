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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[2] / "lab_notebook.db"
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)

_MIN_SAMPLES = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


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


def extract_topology_features(
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

    # Fan-in / fan-out statistics
    fan_ins = [len(parents[i]) for i in range(n)]
    fan_outs = [len(children[i]) for i in range(n)]
    features["topo_max_fan_in"] = float(max(fan_ins)) if fan_ins else 0.0
    features["topo_max_fan_out"] = float(max(fan_outs)) if fan_outs else 0.0
    features["topo_n_merge_nodes"] = float(sum(1 for f in fan_ins if f > 1))
    features["topo_n_split_nodes"] = float(sum(1 for f in fan_outs if f > 1))

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
    for idx in range(n):
        if op_names[idx] == "add" and len(parents[idx]) >= 2:
            depths = [depth[p] for p in parents[idx] if depth[p] >= 0]
            if depths and max(depths) - min(depths) > 0:
                n_skip += 1
    features["residual_coverage"] = float(n_skip) / max(n_ops, 1)

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
    else:
        features["has_norm_before_output"] = 0.0

    return features


@dataclass(slots=True)
class GraphPredictor:
    """Topology-aware graph predictor using Ridge regression on structural features."""

    # Gate model (logistic regression)
    w_gate: np.ndarray  # (n_features,)
    b_gate: float
    # Rank model (linear regression on log-ppl)
    w_rank: np.ndarray  # (n_features,)
    b_rank: float
    # Feature metadata
    feature_names: List[str] = field(default_factory=list)
    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Loss prediction head
    w_loss: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_loss: float = 0.7
    # Op data (cached for feature extraction)
    op_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pair_stability: Dict[Tuple[str, str], float] = field(default_factory=dict)
    imodel: Optional[Any] = None  # InteractionModel, cached for feature extraction
    # Metadata
    n_train: int = 0
    _trained: bool = False
    _train_metrics: Dict[str, float] = field(default_factory=dict)

    def is_fitted(self) -> bool:
        return self._trained and len(self.w_gate) > 0

    def _extract_and_normalize(self, graph_json: Any) -> Optional[np.ndarray]:
        feats = extract_topology_features(
            graph_json, self.op_profiles, self.pair_stability, imodel=self.imodel
        )
        if feats is None:
            return None
        x = np.array([feats.get(k, 0.0) for k in self.feature_names], dtype=np.float64)
        x = (x - self.feature_mean) / self.feature_std
        return x

    def predict_gate(self, graph_json: Any) -> float:
        """Predict P(pass_s1). Returns 0.5 if not fitted."""
        if not self.is_fitted():
            return 0.5
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 0.5
        logit = float(x @ self.w_gate + self.b_gate)
        return float(_sigmoid(np.array([logit]))[0])

    def predict_rank(self, graph_json: Any) -> float:
        """Predict wikitext perplexity. Returns 1e6 if not fitted."""
        if not self.is_fitted():
            return 1e6
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 1e6
        log_ppl = float(x @ self.w_rank + self.b_rank)
        return float(np.exp(log_ppl))

    def predict_loss(self, graph_json: Any) -> float:
        """Predict loss_ratio for S1-passing graphs. Returns 0.7 if not fitted."""
        if not self.is_fitted() or len(self.w_loss) == 0:
            return 0.7
        x = self._extract_and_normalize(graph_json)
        if x is None:
            return 0.7
        return float(np.clip(x @ self.w_loss + self.b_loss, 0.0, 2.0))

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

        # Train interaction model for learned pair features
        trained_imodel = None
        try:
            from .interaction_model import InteractionModel

            trained_imodel = InteractionModel.train(
                notebook_db=notebook_db, profiling_db=profiling_db, n_epochs=30,
            )
            if not trained_imodel._trained:
                trained_imodel = None
        except Exception:
            pass

        _empty_kwargs = dict(
            w_gate=np.zeros(0), b_gate=0.0,
            w_rank=np.zeros(0), b_rank=5.0,
            w_loss=np.zeros(0), b_loss=0.7,
            op_profiles=op_profiles, pair_stability=pair_stability,
            imodel=trained_imodel,
        )

        if not notebook_db.exists():
            return cls(**_empty_kwargs)

        try:
            conn = sqlite3.connect(str(notebook_db), timeout=10)
            conn.execute("PRAGMA busy_timeout=10000")
            rows = conn.execute(
                """SELECT graph_json, stage1_passed, wikitext_perplexity, loss_ratio
                   FROM program_results
                   WHERE graph_json IS NOT NULL
                   ORDER BY RANDOM() LIMIT 8000"""
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("GraphPredictor training data query failed: %s", e)
            return cls(**_empty_kwargs)

        # Extract features
        feat_dicts: List[Dict[str, float]] = []
        gate_labels: List[int] = []
        rank_labels: List[float] = []
        loss_labels: List[float] = []

        for row in rows:
            gj, s1, ppl = row[0], row[1], row[2]
            lr = row[3] if len(row) > 3 else None
            feats = extract_topology_features(
                gj, op_profiles, pair_stability, imodel=trained_imodel
            )
            if feats is None:
                continue
            feat_dicts.append(feats)
            gate_labels.append(int(s1 or 0))
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

        n_total = len(feat_dicts)
        if n_total < _MIN_SAMPLES:
            logger.info(
                "GraphPredictor: insufficient data (%d < %d)", n_total, _MIN_SAMPLES
            )
            return cls(
                w_gate=np.zeros(0),
                b_gate=0.0,
                w_rank=np.zeros(0),
                b_rank=5.0,
                op_profiles=op_profiles,
                pair_stability=pair_stability,
            )

        # Build feature matrix
        feature_names = sorted(feat_dicts[0].keys())
        X = np.zeros((n_total, len(feature_names)), dtype=np.float64)
        for i, d in enumerate(feat_dicts):
            for j, k in enumerate(feature_names):
                X[i, j] = d.get(k, 0.0)

        y_gate = np.array(gate_labels, dtype=np.float64)
        y_rank = np.array(rank_labels, dtype=np.float64)

        # Standardize
        feat_mean = X.mean(axis=0)
        feat_std = X.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        X_norm = (X - feat_mean) / feat_std

        # Train/val split (stratified)
        rng = np.random.RandomState(seed)
        pos_idx = np.where(y_gate == 1)[0]
        neg_idx = np.where(y_gate == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        pos_split = int(len(pos_idx) * 0.8)
        neg_split = int(len(neg_idx) * 0.8)
        train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
        val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])

        X_tr, X_va = X_norm[train_idx], X_norm[val_idx]
        y_gate_tr, y_gate_va = y_gate[train_idx], y_gate[val_idx]

        n_features = X_tr.shape[1]
        logger.info(
            "GraphPredictor training: %d samples, %d topology features",
            n_total,
            n_features,
        )

        # ── Gate: logistic regression via SGD ──
        w_gate = rng.randn(n_features).astype(np.float64) * 0.01
        b_gate = 0.0
        lr = 0.01

        for epoch in range(80):
            perm = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), 128):
                idx = perm[start : start + 128]
                x_b = X_tr[idx]
                y_b = y_gate_tr[idx]
                logits = x_b @ w_gate + b_gate
                preds = _sigmoid(logits)
                grad = (preds - y_b)[:, None] * x_b
                w_gate -= lr * grad.mean(axis=0) + lr * alpha * 0.001 * w_gate  # L2
                b_gate -= lr * float((preds - y_b).mean())

        # Val metrics
        val_preds = _sigmoid(X_va @ w_gate + b_gate)
        val_correct = int(np.sum((val_preds > 0.5) == y_gate_va))
        val_acc = val_correct / max(len(X_va), 1)
        eps = 1e-8
        val_loss = float(
            -np.mean(
                y_gate_va * np.log(val_preds + eps)
                + (1 - y_gate_va) * np.log(1 - val_preds + eps)
            )
        )

        # ── Rank: Ridge regression on log-ppl ──
        rank_mask = np.isfinite(y_rank[train_idx])
        w_rank = np.zeros(n_features, dtype=np.float64)
        b_rank = 5.0
        if rank_mask.sum() >= 20:
            X_rank = X_tr[rank_mask]
            y_log_ppl = np.log(np.maximum(y_rank[train_idx][rank_mask], 1.0))
            XtX = X_rank.T @ X_rank + alpha * np.eye(n_features)
            Xty = X_rank.T @ y_log_ppl
            try:
                w_rank = np.linalg.solve(XtX, Xty)
                b_rank = float(np.mean(y_log_ppl - X_rank @ w_rank))
            except np.linalg.LinAlgError:
                pass

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
            except np.linalg.LinAlgError:
                pass

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
            w_rank=w_rank.astype(np.float32),
            b_rank=float(b_rank),
            w_loss=w_loss.astype(np.float32),
            b_loss=float(b_loss),
            feature_names=feature_names,
            feature_mean=feat_mean.astype(np.float32),
            feature_std=feat_std.astype(np.float32),
            op_profiles=op_profiles,
            pair_stability=pair_stability,
            imodel=trained_imodel,
            n_train=len(X_tr),
            _trained=True,
            _train_metrics={
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "n_features": n_features,
                "n_positive": int(y_gate.sum()),
            },
        )
