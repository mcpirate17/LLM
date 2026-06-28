"""Op embedding space: 16-dim dense representations for primitive ops.

Compatible ops (co-occurring in S1-passing graphs) cluster together in
embedding space. Embeddings are:
1. PCA-initialized from profiling data (meaningful even before training)
2. Fine-tuned via contrastive + auxiliary + pair stability losses
3. Re-trained periodically with temporal decay

Usage:
    embeddings = OpEmbeddings.from_profiling()           # PCA-only init
    embeddings = OpEmbeddings.train(notebook_db, profiling_db)  # full training
    score = embeddings.compatibility("gelu", ["linear_proj", "rmsnorm"])
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..native.core import _try_import_rust_scheduler
from research.defaults import RUNS_DB
from .graph_ops import extract_unique_graph_ops_batch
from .ml_corpus import load_deduped_graph_training_rows
from .predictor_artifacts import load_npz_with_metadata, save_npz_with_metadata
from .profiling_db import (
    load_op_feature_rows,
    load_pair_stability_labels,
)

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[3] / RUNS_DB
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)

EMBED_DIM = 16

# Profiling features used for PCA initialization
_PROFILING_FEATURES = [
    "output_mean",
    "output_std",
    "output_min",
    "output_max",
    "output_kurtosis",
    "grad_norm",
    "grad_max",
    "grad_min",
    "jacobian_spectral_norm",
    "jacobian_condition_num",
    "lipschitz_estimate",
    "forward_time_us",
    "backward_time_us",
]

# Training hyperparameters
_TRIPLET_MARGIN = 0.3
_AUX_WEIGHT = 0.2
_PAIR_WEIGHT = 0.3
_LEARNING_RATE = 0.01
_N_EPOCHS = 50
_BATCH_SIZE = 256
_TEMPORAL_HALF_LIFE_DAYS = 30


def _run_native_embedding_epoch(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    negative_pairs: List[Tuple[int, int]],
    pair_stability: List[Tuple[int, int, bool]],
    *,
    lr: float,
    seed: int,
) -> Optional[Tuple[np.ndarray, float, int]]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "train_op_embeddings_epoch_native_py"):
        return None
    result = rust.train_op_embeddings_epoch_native_py(
        np.ascontiguousarray(embeddings, dtype=np.float64),
        np.ascontiguousarray(np.asarray(positive_pairs, dtype=np.int32).reshape(-1, 2)),
        np.ascontiguousarray(np.asarray(negative_pairs, dtype=np.int32).reshape(-1, 2)),
        np.ascontiguousarray(
            np.asarray([(a, b) for a, b, _ in pair_stability], dtype=np.int32).reshape(
                -1, 2
            )
        ),
        np.ascontiguousarray(
            np.asarray(
                [float(stable) for _, _, stable in pair_stability], dtype=np.float64
            )
        ),
        float(lr),
        int(_BATCH_SIZE),
        float(_TRIPLET_MARGIN),
        float(_PAIR_WEIGHT),
        int(seed),
    )
    return (
        np.asarray(result["embeddings"], dtype=np.float64),
        float(result["total_loss"]),
        int(result["n_samples"]),
    )


@dataclass
class OpEmbeddings:
    """Dense op embeddings with compatibility scoring."""

    embeddings: np.ndarray  # (n_ops, EMBED_DIM)
    op_names: List[str]
    op_to_idx: Dict[str, int]
    _profiling_features: Optional[np.ndarray] = field(
        default=None, repr=False
    )  # (n_ops, n_features) raw
    _feature_mean: Optional[np.ndarray] = field(default=None, repr=False)
    _feature_std: Optional[np.ndarray] = field(default=None, repr=False)
    _trained: bool = False
    _timestamp: float = 0.0

    @property
    def n_ops(self) -> int:
        return len(self.op_names)

    def get_embedding(self, op_name: str) -> Optional[np.ndarray]:
        """Get embedding vector for an op. Returns None if unknown."""
        idx = self.op_to_idx.get(op_name)
        if idx is None:
            return None
        return self.embeddings[idx]

    def compatibility(self, candidate: str, placed_ops: List[str]) -> float:
        """Score how compatible a candidate op is with already-placed ops.

        Returns mean cosine similarity in [0, 1]. Returns 0.5 for unknown ops.
        """
        cand_emb = self.get_embedding(candidate)
        if cand_emb is None or not placed_ops:
            return 0.5

        similarities = []
        for op in placed_ops:
            emb = self.get_embedding(op)
            if emb is None:
                continue
            dot = np.dot(cand_emb, emb)
            norm_prod = np.linalg.norm(cand_emb) * np.linalg.norm(emb)
            if norm_prod < 1e-8:
                continue
            sim = dot / norm_prod
            similarities.append(float(sim))

        if not similarities:
            return 0.5
        # Map cosine similarity from [-1, 1] → [0, 1]
        return float((np.mean(similarities) + 1.0) / 2.0)

    @classmethod
    def from_profiling(
        cls,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
    ) -> "OpEmbeddings":
        """Initialize embeddings via PCA on profiling features.

        This gives a meaningful starting point even without experiment data:
        ops with similar gradient/stability profiles cluster together.
        """
        if not profiling_db.exists():
            logger.warning("Profiling DB not found: %s", profiling_db)
            return cls._empty()

        rows = load_op_feature_rows(profiling_db, _PROFILING_FEATURES)

        if not rows:
            return cls._empty()

        op_names = [r["op_name"] for r in rows]
        op_to_idx = {name: i for i, name in enumerate(op_names)}
        n = len(op_names)

        # Build feature matrix
        raw = np.zeros((n, len(_PROFILING_FEATURES)), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, feat in enumerate(_PROFILING_FEATURES):
                val = row[feat]
                raw[i, j] = (
                    float(val) if val is not None and math.isfinite(float(val)) else 0.0
                )

        # Standardize
        feat_mean = raw.mean(axis=0)
        feat_std = raw.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        X = (raw - feat_mean) / feat_std

        # PCA to EMBED_DIM
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(EMBED_DIM, Vt.shape[0])
        embeddings = U[:, :k] * S[:k]  # scaled projections

        # Pad if fewer components than EMBED_DIM
        if k < EMBED_DIM:
            pad = np.zeros((n, EMBED_DIM - k))
            embeddings = np.hstack([embeddings, pad])

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        logger.info(
            "PCA-initialized %d op embeddings (dim=%d) from profiling data",
            n,
            EMBED_DIM,
        )

        return cls(
            embeddings=embeddings.astype(np.float32),
            op_names=op_names,
            op_to_idx=op_to_idx,
            _profiling_features=raw,
            _feature_mean=feat_mean,
            _feature_std=feat_std,
            _trained=False,
            _timestamp=time.time(),
        )

    @classmethod
    def train(
        cls,
        notebook_db: Path = _DEFAULT_NOTEBOOK_DB,
        profiling_db: Path = _DEFAULT_PROFILING_DB,
        n_epochs: int = _N_EPOCHS,
        lr: float = _LEARNING_RATE,
        seed: int = 42,
        temporal_days: Optional[int] = _TEMPORAL_HALF_LIFE_DAYS,
    ) -> "OpEmbeddings":
        """Full training: PCA init + contrastive + auxiliary + pair stability.

        Args:
            notebook_db: Path to lab_notebook.db for co-occurrence data.
            profiling_db: Path to component_profiles.db for profiling features.
            n_epochs: Number of training epochs.
            lr: Learning rate.
            seed: Random seed.
            temporal_days: Only use experiment data from last N days (None=all).
        """
        # Start from PCA initialization
        obj = cls.from_profiling(profiling_db)
        if obj.n_ops == 0:
            return obj

        rng = np.random.RandomState(seed)
        embeddings = obj.embeddings.astype(np.float64).copy()

        # ── Gather training data ──
        positive_pairs, negative_pairs = _extract_cooccurrence_pairs(
            notebook_db, obj.op_to_idx, temporal_days=temporal_days
        )
        pair_stability = _load_pair_stability(profiling_db, obj.op_to_idx)

        n_pos = len(positive_pairs)
        n_neg = len(negative_pairs)
        n_stab = len(pair_stability)
        logger.info(
            "Training data: %d positive pairs, %d negative pairs, %d stability labels",
            n_pos,
            n_neg,
            n_stab,
        )

        if n_pos < 10:
            logger.info(
                "Insufficient co-occurrence data, returning PCA-only embeddings"
            )
            return obj

        # ── Training loop ──
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_samples = 0
            native_epoch = _run_native_embedding_epoch(
                embeddings,
                positive_pairs,
                negative_pairs,
                pair_stability,
                lr=lr,
                seed=seed + epoch,
            )

            if native_epoch is not None:
                embeddings, native_loss, native_samples = native_epoch
                total_loss += native_loss
                n_samples += native_samples
            else:
                rng.shuffle(positive_pairs)
                for batch_start in range(
                    0, min(len(positive_pairs), 2000), _BATCH_SIZE
                ):
                    batch = positive_pairs[batch_start : batch_start + _BATCH_SIZE]
                    for anchor_idx, pos_idx in batch:
                        if not negative_pairs:
                            continue
                        neg_idx_pair = negative_pairs[rng.randint(len(negative_pairs))]
                        neg_idx = (
                            neg_idx_pair[1]
                            if neg_idx_pair[0] == anchor_idx
                            else neg_idx_pair[0]
                        )

                        a = embeddings[anchor_idx]
                        p = embeddings[pos_idx]
                        n_vec = embeddings[neg_idx]

                        d_pos = np.sum((a - p) ** 2)
                        d_neg = np.sum((a - n_vec) ** 2)
                        margin_loss = max(0.0, d_pos - d_neg + _TRIPLET_MARGIN)

                        if margin_loss > 0:
                            grad_a = 2.0 * ((a - p) - (a - n_vec))
                            grad_p = 2.0 * (p - a)
                            grad_n = 2.0 * (a - n_vec)

                            embeddings[anchor_idx] -= lr * grad_a
                            embeddings[pos_idx] -= lr * grad_p
                            embeddings[neg_idx] += lr * grad_n

                            total_loss += margin_loss
                            n_samples += 1

            # Auxiliary loss: predict profiling features from embedding
            if obj._profiling_features is not None and obj._feature_std is not None:
                targets = (
                    obj._profiling_features - obj._feature_mean
                ) / obj._feature_std
                n_feat = targets.shape[1]
                # Simple linear decoder: W = (E^T E)^{-1} E^T T
                E = embeddings
                try:
                    W_dec = np.linalg.lstsq(E, targets, rcond=None)[
                        0
                    ]  # (EMBED_DIM, n_feat)
                    predictions = E @ W_dec
                    aux_error = predictions - targets
                    aux_loss = float(np.mean(aux_error**2))
                    # Push embeddings toward better reconstruction
                    grad_aux = (2.0 / (obj.n_ops * n_feat)) * aux_error @ W_dec.T
                    embeddings -= lr * _AUX_WEIGHT * grad_aux
                    total_loss += _AUX_WEIGHT * aux_loss
                except np.linalg.LinAlgError as exc:
                    logger.warning(
                        "OpEmbeddings auxiliary decoder solve failed: %s", exc
                    )

            # Pair stability loss stays on the Python path only when native is unavailable.
            if pair_stability and native_epoch is None:
                stab_batch = pair_stability[: min(len(pair_stability), 1000)]
                for idx_a, idx_b, stable in stab_batch:
                    # Elementwise product → linear → sigmoid
                    prod = embeddings[idx_a] * embeddings[idx_b]
                    logit = np.sum(prod)
                    pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
                    target = float(stable)
                    bce = -(
                        target * np.log(max(pred, 1e-8))
                        + (1 - target) * np.log(max(1 - pred, 1e-8))
                    )
                    # Gradient
                    d_logit = pred - target
                    grad_a = _PAIR_WEIGHT * d_logit * embeddings[idx_b]
                    grad_b = _PAIR_WEIGHT * d_logit * embeddings[idx_a]
                    embeddings[idx_a] -= lr * grad_a
                    embeddings[idx_b] -= lr * grad_b
                    total_loss += _PAIR_WEIGHT * bce

            # Re-normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                logger.info(
                    "Epoch %d/%d: total_loss=%.4f, n_samples=%d",
                    epoch + 1,
                    n_epochs,
                    total_loss,
                    n_samples,
                )

        obj.embeddings = embeddings.astype(np.float32)
        obj._trained = True
        obj._timestamp = time.time()
        logger.info("Training complete: %d ops, %d dimensions", obj.n_ops, EMBED_DIM)
        return obj

    @classmethod
    def _empty(cls) -> "OpEmbeddings":
        return cls(
            embeddings=np.zeros((0, EMBED_DIM), dtype=np.float32),
            op_names=[],
            op_to_idx={},
        )

    def save(self, path: Path) -> None:
        """Save embeddings to npz + JSON metadata."""
        save_npz_with_metadata(
            path,
            arrays={"embeddings": self.embeddings},
            metadata={
                "op_names": self.op_names,
                "embed_dim": EMBED_DIM,
                "trained": self._trained,
                "timestamp": self._timestamp,
                "n_ops": self.n_ops,
            },
        )
        logger.info("Saved embeddings to %s", path)

    @classmethod
    def load(cls, path: Path) -> "OpEmbeddings":
        """Load embeddings from npz + JSON metadata."""
        data, meta = load_npz_with_metadata(path)
        op_names = meta["op_names"]
        return cls(
            embeddings=data["embeddings"],
            op_names=op_names,
            op_to_idx={name: i for i, name in enumerate(op_names)},
            _trained=meta.get("trained", False),
            _timestamp=meta.get("timestamp", 0),
        )


def _extract_cooccurrence_pairs(
    db_path: Path,
    op_to_idx: Dict[str, int],
    temporal_days: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract positive (co-occur in S1 pass) and negative (co-occur in fail) pairs."""
    positive: List[Tuple[int, int]] = []
    negative: List[Tuple[int, int]] = []

    if not db_path.exists():
        return positive, negative

    try:
        rows = load_deduped_graph_training_rows(db_path)
    except Exception as e:
        logger.warning("Failed to load co-occurrence data: %s", e)
        return positive, negative

    cutoff = None
    if temporal_days is not None:
        cutoff = time.time() - temporal_days * 86400

    graph_payloads = [row["graph_json"] for row in rows]
    extracted_ops = extract_unique_graph_ops_batch(graph_payloads)

    for row, extracted in zip(rows, extracted_ops):
        if cutoff is not None and float(row.get("latest_timestamp", 0.0)) <= cutoff:
            continue
        s1_passed = row["stage1_any_passed"]
        op_list = [op for op in extracted if op in op_to_idx]
        target = positive if s1_passed else negative

        for i, a in enumerate(op_list):
            for b in op_list[i + 1 :]:
                target.append((op_to_idx[a], op_to_idx[b]))

    return positive, negative


def _load_pair_stability(
    profiling_db: Path,
    op_to_idx: Dict[str, int],
) -> List[Tuple[int, int, bool]]:
    """Load pair stability labels from profiling DB."""
    return load_pair_stability_labels(profiling_db, op_to_idx)
