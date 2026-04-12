"""Factored bilinear interaction model for op pair success/loss prediction.

Predicts P(pair stable) and E[loss_ratio] for any (op_a, op_b) pair using
factored bilinear representations:
    S[i,j] = sigmoid(u_i^T W_s v_j + b_s)   # stability probability
    L[i,j] = u_i^T W_l v_j + b_l             # loss contribution

Training data combines profiling DB pair data, failure signatures, and
experiment co-occurrence data with temporal weighting.

Usage:
    model = InteractionModel.train(notebook_db, profiling_db)
    p_stable = model.predict_stability("gelu", "linear_proj")
    loss = model.predict_loss("gelu", "linear_proj")
    motif_score = model.motif_viability(["rmsnorm", "linear_proj", "gelu", "add"])
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..native.core import _try_import_rust_scheduler
from .graph_ops import extract_unique_graph_ops_batch
from .ml_corpus import (
    CorpusIntegrityError,
    load_deduped_graph_training_rows,
    rerun_confidence_weight,
)

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[2] / "lab_notebook.db"
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)

EMBED_DIM = 16
_LR = 0.005
_N_EPOCHS = 100
_BATCH_SIZE = 512
_TEMPORAL_HALF_LIFE_DAYS = 7
_PROFILING_STATIC_WEIGHT = 0.5  # weight for profiling data (no timestamp)
_HUBER_DELTA = 0.2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def _huber_loss(
    pred: np.ndarray, target: np.ndarray, delta: float = _HUBER_DELTA
) -> Tuple[float, np.ndarray]:
    """Huber loss + gradient."""
    diff = pred - target
    abs_diff = np.abs(diff)
    quadratic = abs_diff <= delta
    loss = np.where(quadratic, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
    grad = np.where(quadratic, diff, delta * np.sign(diff))
    return float(np.mean(loss)), grad


def _train_interaction_python(
    *,
    u: np.ndarray,
    v: np.ndarray,
    W_s: np.ndarray,
    W_l: np.ndarray,
    b_s: float,
    b_l: float,
    stab_idx: np.ndarray,
    stab_labels: np.ndarray,
    stab_weights: np.ndarray,
    loss_idx: np.ndarray,
    loss_labels: np.ndarray,
    loss_weights: np.ndarray,
    n_epochs: int,
    lr: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    rng = np.random.RandomState(seed)
    best_loss = float("inf")
    for _epoch in range(n_epochs):
        total_loss = 0.0
        perm = rng.permutation(len(stab_idx))
        stab_idx_s = stab_idx[perm]
        stab_labels_s = stab_labels[perm]
        stab_weights_s = stab_weights[perm]

        for start in range(0, len(stab_idx_s), _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, len(stab_idx_s))
            batch_i = stab_idx_s[start:end, 0]
            batch_j = stab_idx_s[start:end, 1]
            batch_y = stab_labels_s[start:end]
            batch_w = stab_weights_s[start:end]
            bs = end - start

            u_batch = u[batch_i]
            v_batch = v[batch_j]
            uW = u_batch @ W_s
            logits = np.sum(uW * v_batch, axis=1) + b_s
            preds = _sigmoid(logits)

            eps = 1e-8
            bce = -(
                batch_y * np.log(preds + eps) + (1 - batch_y) * np.log(1 - preds + eps)
            )
            total_loss += float(np.sum(bce * batch_w))

            d_logit = (preds - batch_y) * batch_w
            d_logit_2d = d_logit[:, None]
            dW_s = (u_batch * d_logit_2d).T @ v_batch / bs
            db_s = float(np.mean(d_logit))
            du_batch = d_logit_2d * (v_batch @ W_s.T)
            dv_batch = d_logit_2d * uW

            W_s -= lr * dW_s
            b_s -= lr * db_s
            np.add.at(u, batch_i, -lr * du_batch / bs)
            np.add.at(v, batch_j, -lr * dv_batch / bs)

        if len(loss_idx) > 0:
            perm_l = rng.permutation(len(loss_idx))
            for start in range(0, min(len(loss_idx), 2000), _BATCH_SIZE):
                end = min(start + _BATCH_SIZE, len(loss_idx))
                batch_i = loss_idx[perm_l[start:end], 0]
                batch_j = loss_idx[perm_l[start:end], 1]
                batch_y = loss_labels[perm_l[start:end]]
                batch_w = loss_weights[perm_l[start:end]]
                bs = end - start

                u_batch = u[batch_i]
                v_batch = v[batch_j]
                uW_l = u_batch @ W_l
                preds_l = np.sum(uW_l * v_batch, axis=1) + b_l

                _, grad_hl = _huber_loss(preds_l, batch_y)
                grad_hl *= batch_w
                total_loss += 0.5 * float(np.sum(np.abs(grad_hl)))

                d_2d = (grad_hl * 0.5)[:, None]
                dW_l = (u_batch * d_2d).T @ v_batch / bs
                db_l = float(np.mean(grad_hl * 0.5))

                W_l -= lr * dW_l
                b_l -= lr * db_l
                np.add.at(u, batch_i, -lr * d_2d * (v_batch @ W_l.T) / bs)
                np.add.at(v, batch_j, -lr * d_2d * uW_l / bs)

        if total_loss < best_loss:
            best_loss = total_loss

    return u, v, W_s, W_l, float(b_s), float(b_l), float(best_loss)


def _train_interaction_native(
    *,
    u: np.ndarray,
    v: np.ndarray,
    W_s: np.ndarray,
    W_l: np.ndarray,
    b_s: float,
    b_l: float,
    stab_idx: np.ndarray,
    stab_labels: np.ndarray,
    stab_weights: np.ndarray,
    loss_idx: np.ndarray,
    loss_labels: np.ndarray,
    loss_weights: np.ndarray,
    n_epochs: int,
    lr: float,
    seed: int,
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]
]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "train_interaction_model_native_py"):
        return None
    result = rust.train_interaction_model_native_py(
        np.ascontiguousarray(u, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(W_s, dtype=np.float64),
        np.ascontiguousarray(W_l, dtype=np.float64),
        float(b_s),
        float(b_l),
        np.ascontiguousarray(stab_idx, dtype=np.int32),
        np.ascontiguousarray(stab_labels, dtype=np.float64),
        np.ascontiguousarray(stab_weights, dtype=np.float64),
        np.ascontiguousarray(loss_idx, dtype=np.int32),
        np.ascontiguousarray(loss_labels, dtype=np.float64),
        np.ascontiguousarray(loss_weights, dtype=np.float64),
        int(n_epochs),
        float(lr),
        int(_BATCH_SIZE),
        int(seed),
    )
    return (
        np.asarray(result["u"], dtype=np.float64),
        np.asarray(result["v"], dtype=np.float64),
        np.asarray(result["W_s"], dtype=np.float64),
        np.asarray(result["W_l"], dtype=np.float64),
        float(result["b_s"]),
        float(result["b_l"]),
        float(result["best_loss"]),
    )


@dataclass
class InteractionModel:
    """Factored bilinear model for pairwise op interaction prediction."""

    # Op embeddings (left/right context)
    u: np.ndarray  # (n_ops, EMBED_DIM) — left context
    v: np.ndarray  # (n_ops, EMBED_DIM) — right context
    # Interaction matrices
    W_s: np.ndarray  # (EMBED_DIM, EMBED_DIM) — stability
    W_l: np.ndarray  # (EMBED_DIM, EMBED_DIM) — loss
    b_s: float = 0.0
    b_l: float = 0.5
    # Op registry
    op_names: List[str] = field(default_factory=list)
    op_to_idx: Dict[str, int] = field(default_factory=dict)
    _trained: bool = False
    _timestamp: float = 0.0
    _train_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def n_ops(self) -> int:
        return len(self.op_names)

    def predict_stability(self, op_a: str, op_b: str) -> float:
        """Predict P(pair stable) for (op_a, op_b). Returns 0.5 if unknown."""
        i = self.op_to_idx.get(op_a)
        j = self.op_to_idx.get(op_b)
        if i is None or j is None:
            return 0.5
        logit = float(self.u[i] @ self.W_s @ self.v[j]) + self.b_s
        return float(_sigmoid(np.array([logit]))[0])

    def predict_loss(self, op_a: str, op_b: str) -> float:
        """Predict loss contribution for (op_a, op_b). Returns 0.7 if unknown."""
        i = self.op_to_idx.get(op_a)
        j = self.op_to_idx.get(op_b)
        if i is None or j is None:
            return 0.7
        return float(self.u[i] @ self.W_l @ self.v[j]) + self.b_l

    def stability_matrix(self) -> np.ndarray:
        """Full NxN stability prediction matrix."""
        logits = self.u @ self.W_s @ self.v.T + self.b_s
        return _sigmoid(logits)

    def loss_matrix(self) -> np.ndarray:
        """Full NxN loss prediction matrix."""
        return self.u @ self.W_l @ self.v.T + self.b_l

    def motif_viability(self, op_sequence: List[str]) -> float:
        """Score a motif by minimum predicted stability across consecutive pairs.

        Returns float in [0, 1]. Higher = more viable.
        """
        if len(op_sequence) < 2:
            return 1.0
        scores = []
        for a, b in zip(op_sequence[:-1], op_sequence[1:]):
            scores.append(self.predict_stability(a, b))
        return float(min(scores))

    def pair_adjusted_op_weight(self, candidate: str, context_ops: List[str]) -> float:
        """Compute pair-adjusted weight for a candidate op given graph context.

        Returns multiplier in [0.1, 2.0] based on mean stability with context ops.
        """
        if not context_ops:
            return 1.0
        stabilities = []
        for op in context_ops:
            stabilities.append(self.predict_stability(candidate, op))
            stabilities.append(self.predict_stability(op, candidate))
        mean_stab = float(np.mean(stabilities))
        # Map [0, 1] → [0.1, 2.0]
        return float(np.clip(0.1 + 1.9 * mean_stab, 0.1, 2.0))

    @classmethod
    def train(
        cls,
        notebook_db: Path = _DEFAULT_NOTEBOOK_DB,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
        init_embeddings: Optional[np.ndarray] = None,
        n_epochs: int = _N_EPOCHS,
        lr: float = _LR,
        seed: int = 42,
    ) -> "InteractionModel":
        """Train the interaction model from profiling + experiment data.

        Args:
            notebook_db: Path to lab_notebook.db.
            profiling_db: Path to component_profiles.db.
            init_embeddings: Optional (n_ops, EMBED_DIM) embeddings from OpEmbeddings.
            n_epochs: Training epochs.
            lr: Learning rate.
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)

        # ── Gather all op names ──
        all_ops: set = set()
        stability_data: List[
            Tuple[str, str, bool, float]
        ] = []  # (a, b, stable, weight)
        loss_data: List[
            Tuple[str, str, float, float]
        ] = []  # (a, b, loss_ratio, weight)

        # From profiling DB
        if profiling_db.exists():
            try:
                conn = sqlite3.connect(str(profiling_db), timeout=5)
                rows = conn.execute(
                    """SELECT op_a, op_b,
                              (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) as stable,
                              lipschitz_estimate
                       FROM pair_profiles WHERE error IS NULL"""
                ).fetchall()
                conn.close()
                for op_a, op_b, stable, lip in rows:
                    all_ops.add(op_a)
                    all_ops.add(op_b)
                    stability_data.append(
                        (op_a, op_b, bool(stable), _PROFILING_STATIC_WEIGHT)
                    )
            except Exception as e:
                logger.warning("Failed to load profiling pairs: %s", e)

        # From failure signatures
        if notebook_db.exists():
            try:
                conn = sqlite3.connect(str(notebook_db), timeout=10)
                conn.execute("PRAGMA busy_timeout=10000")
                rows = conn.execute(
                    "SELECT signature, n_failures, n_successes, last_updated "
                    "FROM failure_signatures"
                ).fetchall()
                now = time.time()
                half_life = _TEMPORAL_HALF_LIFE_DAYS * 86400
                for sig, n_fail, n_succ, ts in rows:
                    parts = sig.split("->")
                    if len(parts) != 2:
                        continue
                    a, b = parts[0].strip(), parts[1].strip()
                    all_ops.add(a)
                    all_ops.add(b)
                    age = now - (ts or now)
                    w = (
                        math.exp(-math.log(2) * age / half_life)
                        if half_life > 0
                        else 1.0
                    )
                    total = n_fail + n_succ
                    if total > 0:
                        rate = n_succ / total
                        stability_data.append((a, b, rate > 0.5, w))
            except Exception as e:
                logger.warning("Failed to load failure signatures: %s", e)

            # From experiment co-occurrence
            try:
                rows = load_deduped_graph_training_rows(notebook_db, validate=True)
                now = time.time()
                graph_payloads = [row["graph_json"] for row in rows]
                extracted_ops = extract_unique_graph_ops_batch(graph_payloads)
                for row, ops in zip(rows, extracted_ops):
                    gj = row["graph_json"]
                    s1 = bool(row["stage1_any_passed"])
                    loss_ratio = row.get("loss_ratio_best")
                    ts = row.get("latest_timestamp", None)
                    rerun_weight = rerun_confidence_weight(int(row.get("n_rows", 1)))
                    if not ops:
                        continue
                    age = now - (ts or now)
                    w = (
                        math.exp(-math.log(2) * age / half_life)
                        if half_life > 0
                        else 1.0
                    )
                    w *= rerun_weight

                    for i_op, a in enumerate(ops):
                        all_ops.add(a)
                        for b in ops[i_op + 1 :]:
                            stability_data.append((a, b, bool(s1), w * 0.1))
                            if (
                                s1
                                and loss_ratio is not None
                                and math.isfinite(loss_ratio)
                            ):
                                loss_data.append((a, b, loss_ratio, w * 0.1))
            except CorpusIntegrityError:
                raise
            except Exception as e:
                logger.warning("Failed to load experiment co-occurrence: %s", e)

        # ── Build op registry ──
        op_names = sorted(all_ops)
        op_to_idx = {name: i for i, name in enumerate(op_names)}
        n = len(op_names)

        if n == 0 or len(stability_data) < 10:
            logger.info("Insufficient data for interaction model training")
            return cls._empty()

        # ── Initialize parameters ──
        if init_embeddings is not None and init_embeddings.shape[0] == n:
            u = init_embeddings.copy().astype(np.float64)
            v = init_embeddings.copy().astype(np.float64)
        else:
            u = rng.randn(n, EMBED_DIM).astype(np.float64) * 0.1
            v = rng.randn(n, EMBED_DIM).astype(np.float64) * 0.1

        W_s = np.eye(EMBED_DIM, dtype=np.float64) * 0.1
        W_l = np.eye(EMBED_DIM, dtype=np.float64) * 0.1
        b_s = 0.0
        b_l = 0.5

        # ── Convert to index arrays ──
        stab_idx = np.array(
            [
                [op_to_idx.get(a, -1), op_to_idx.get(b, -1)]
                for a, b, _, _ in stability_data
            ],
            dtype=np.int32,
        )
        stab_labels = np.array(
            [float(s) for _, _, s, _ in stability_data], dtype=np.float64
        )
        stab_weights = np.array([w for _, _, _, w in stability_data], dtype=np.float64)

        loss_idx = (
            np.array(
                [
                    [op_to_idx.get(a, -1), op_to_idx.get(b, -1)]
                    for a, b, _, _ in loss_data
                ],
                dtype=np.int32,
            )
            if loss_data
            else np.zeros((0, 2), dtype=np.int32)
        )
        loss_labels = (
            np.array([l for _, _, l, _ in loss_data], dtype=np.float64)
            if loss_data
            else np.zeros(0)
        )
        loss_weights = (
            np.array([w for _, _, _, w in loss_data], dtype=np.float64)
            if loss_data
            else np.zeros(0)
        )

        # Filter out unknown ops
        valid_stab = (stab_idx[:, 0] >= 0) & (stab_idx[:, 1] >= 0)
        stab_idx = stab_idx[valid_stab]
        stab_labels = stab_labels[valid_stab]
        stab_weights = stab_weights[valid_stab]

        if len(loss_idx) > 0:
            valid_loss = (loss_idx[:, 0] >= 0) & (loss_idx[:, 1] >= 0)
            loss_idx = loss_idx[valid_loss]
            loss_labels = loss_labels[valid_loss]
            loss_weights = loss_weights[valid_loss]

        logger.info(
            "Training interaction model: %d ops, %d stability samples, %d loss samples",
            n,
            len(stab_idx),
            len(loss_idx),
        )

        native_result = None
        try:
            native_result = _train_interaction_native(
                u=u,
                v=v,
                W_s=W_s,
                W_l=W_l,
                b_s=b_s,
                b_l=b_l,
                stab_idx=stab_idx,
                stab_labels=stab_labels,
                stab_weights=stab_weights,
                loss_idx=loss_idx,
                loss_labels=loss_labels,
                loss_weights=loss_weights,
                n_epochs=n_epochs,
                lr=lr,
                seed=seed,
            )
        except Exception as exc:
            logger.warning(
                "Native interaction training failed; falling back to Python: %s", exc
            )

        if native_result is None:
            u, v, W_s, W_l, b_s, b_l, best_loss = _train_interaction_python(
                u=u,
                v=v,
                W_s=W_s,
                W_l=W_l,
                b_s=b_s,
                b_l=b_l,
                stab_idx=stab_idx,
                stab_labels=stab_labels,
                stab_weights=stab_weights,
                loss_idx=loss_idx,
                loss_labels=loss_labels,
                loss_weights=loss_weights,
                n_epochs=n_epochs,
                lr=lr,
                seed=seed,
            )
        else:
            u, v, W_s, W_l, b_s, b_l, best_loss = native_result

        model = cls(
            u=u.astype(np.float32),
            v=v.astype(np.float32),
            W_s=W_s.astype(np.float32),
            W_l=W_l.astype(np.float32),
            b_s=float(b_s),
            b_l=float(b_l),
            op_names=op_names,
            op_to_idx=op_to_idx,
            _trained=True,
            _timestamp=time.time(),
            _train_metrics={
                "best_loss": best_loss,
                "n_stab": len(stab_idx),
                "n_loss": len(loss_idx),
            },
        )

        logger.info(
            "Interaction model trained: %d ops, best_loss=%.4f",
            n,
            best_loss,
        )
        return model

    @classmethod
    def _empty(cls) -> "InteractionModel":
        return cls(
            u=np.zeros((0, EMBED_DIM), dtype=np.float32),
            v=np.zeros((0, EMBED_DIM), dtype=np.float32),
            W_s=np.eye(EMBED_DIM, dtype=np.float32),
            W_l=np.eye(EMBED_DIM, dtype=np.float32),
            op_names=[],
            op_to_idx={},
        )

    def save(self, path: Path) -> None:
        """Save model to npz + JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), u=self.u, v=self.v, W_s=self.W_s, W_l=self.W_l)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "op_names": self.op_names,
                    "b_s": self.b_s,
                    "b_l": self.b_l,
                    "trained": self._trained,
                    "timestamp": self._timestamp,
                    "train_metrics": self._train_metrics,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "InteractionModel":
        """Load model from npz + JSON."""
        path = Path(path)
        data = np.load(str(path))
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        op_names = meta["op_names"]
        return cls(
            u=data["u"],
            v=data["v"],
            W_s=data["W_s"],
            W_l=data["W_l"],
            b_s=meta["b_s"],
            b_l=meta["b_l"],
            op_names=op_names,
            op_to_idx={n: i for i, n in enumerate(op_names)},
            _trained=meta.get("trained", False),
            _timestamp=meta.get("timestamp", 0),
            _train_metrics=meta.get("train_metrics", {}),
        )
