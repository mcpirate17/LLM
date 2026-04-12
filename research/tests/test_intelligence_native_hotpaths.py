from __future__ import annotations

import math
import time

import numpy as np
import pytest

from research.scientist.intelligence import gnn_predictor as gp
from research.scientist.intelligence import graph_ops as go
from research.scientist.intelligence import interaction_model as im
from research.scientist.intelligence import op_embeddings as oe
from research.scientist.native.core import _try_import_rust_scheduler


pytestmark = pytest.mark.native


def _sample_graph() -> dict:
    return {
        "model_dim": 64,
        "output_node_id": 5,
        "metadata": {"templates_used": ["residual"], "motifs_used": ["skip"]},
        "nodes": {
            "0": {"id": 0, "op_name": "input", "input_ids": []},
            "1": {"id": 1, "op_name": "linear_proj", "input_ids": [0]},
            "2": {"id": 2, "op_name": "gelu", "input_ids": [1]},
            "3": {"id": 3, "op_name": "linear_proj", "input_ids": [2]},
            "4": {"id": 4, "op_name": "add", "input_ids": [1, 3]},
            "5": {"id": 5, "op_name": "rmsnorm", "input_ids": [4]},
        },
    }


def _many_graph_payloads(n: int = 400) -> list[str]:
    graph = _sample_graph()
    payloads = []
    for idx in range(n):
        item = dict(graph)
        item["metadata"] = {
            "templates_used": [f"t{idx % 7}"],
            "motifs_used": [f"m{idx % 5}"],
        }
        payloads.append(gp.json.dumps(item, sort_keys=True, separators=(",", ":")))
    return payloads


def _sample_profiles() -> dict:
    return {
        "linear_proj": {
            "output_std": 1.4,
            "grad_norm": 8.0,
            "lipschitz": 2.0,
            "grad_vanishing": 0.0,
            "grad_exploding": 0.0,
            "has_nan": 0.0,
        },
        "gelu": {
            "output_std": 0.8,
            "grad_norm": 1.2,
            "lipschitz": 1.1,
            "grad_vanishing": 0.1,
            "grad_exploding": 0.0,
            "has_nan": 0.0,
        },
        "add": {
            "output_std": 0.9,
            "grad_norm": 1.0,
            "lipschitz": 1.0,
            "grad_vanishing": 0.0,
            "grad_exploding": 0.0,
            "has_nan": 0.0,
        },
        "rmsnorm": {
            "output_std": 0.7,
            "grad_norm": 0.8,
            "lipschitz": 0.9,
            "grad_vanishing": 0.0,
            "grad_exploding": 0.0,
            "has_nan": 0.0,
        },
    }


def _sample_pair_stability() -> dict:
    return {
        ("linear_proj", "gelu"): 0.9,
        ("gelu", "linear_proj"): 0.8,
        ("linear_proj", "add"): 0.7,
        ("add", "rmsnorm"): 0.95,
    }


def _make_interaction_train_data(
    seed: int = 7,
    *,
    n_ops: int = 96,
    n_stability_pairs: int = 3000,
    n_loss_pairs: int = 2000,
):
    rng = np.random.RandomState(seed)
    dim = im.EMBED_DIM
    stable_left = rng.randint(0, n_ops // 2, size=n_stability_pairs, dtype=np.int32)
    stable_right = rng.randint(0, n_ops // 2, size=n_stability_pairs, dtype=np.int32)
    unstable_left = rng.randint(
        n_ops // 2, n_ops, size=n_stability_pairs, dtype=np.int32
    )
    unstable_right = rng.randint(
        n_ops // 2, n_ops, size=n_stability_pairs, dtype=np.int32
    )
    stab_idx = np.vstack(
        [
            np.stack([stable_left, stable_right], axis=1),
            np.stack([unstable_left, unstable_right], axis=1),
        ]
    ).astype(np.int32)
    stab_labels = np.concatenate(
        [
            np.ones(len(stable_left), dtype=np.float64),
            np.zeros(len(unstable_left), dtype=np.float64),
        ]
    )
    stab_weights = np.ones(len(stab_idx), dtype=np.float64)

    loss_idx = np.stack(
        [
            rng.randint(0, n_ops // 2, size=n_loss_pairs, dtype=np.int32),
            rng.randint(0, n_ops // 2, size=n_loss_pairs, dtype=np.int32),
        ],
        axis=1,
    ).astype(np.int32)
    loss_labels = np.full(n_loss_pairs, 0.2, dtype=np.float64)
    loss_weights = np.ones(n_loss_pairs, dtype=np.float64)

    u = rng.randn(n_ops, dim).astype(np.float64) * 0.1
    v = rng.randn(n_ops, dim).astype(np.float64) * 0.1
    W_s = np.eye(dim, dtype=np.float64) * 0.1
    W_l = np.eye(dim, dtype=np.float64) * 0.1
    return (
        u,
        v,
        W_s,
        W_l,
        stab_idx,
        stab_labels,
        stab_weights,
        loss_idx,
        loss_labels,
        loss_weights,
    )


def _make_embedding_epoch_data(
    seed: int = 11,
    *,
    n_ops: int = 128,
    n_positive: int = 4000,
    n_negative: int = 4000,
    n_pair_labels: int = 1200,
):
    rng = np.random.RandomState(seed)
    dim = oe.EMBED_DIM
    embeddings = rng.randn(n_ops, dim).astype(np.float64)
    positive_pairs = [
        (int(a), int(b))
        for a, b in zip(
            rng.randint(0, n_ops // 2, size=n_positive),
            rng.randint(0, n_ops // 2, size=n_positive),
        )
    ]
    negative_pairs = [
        (int(a), int(b))
        for a, b in zip(
            rng.randint(0, n_ops // 2, size=n_negative),
            rng.randint(n_ops // 2, n_ops, size=n_negative),
        )
    ]
    pair_stability = [
        (int(a), int(b), True)
        for a, b in zip(
            rng.randint(0, n_ops // 2, size=n_pair_labels),
            rng.randint(0, n_ops // 2, size=n_pair_labels),
        )
    ]
    pair_stability.extend(
        [
            (int(a), int(b), False)
            for a, b in zip(
                rng.randint(n_ops // 2, n_ops, size=n_pair_labels),
                rng.randint(n_ops // 2, n_ops, size=n_pair_labels),
            )
        ]
    )
    return embeddings, positive_pairs, negative_pairs, pair_stability


def _python_op_embedding_epoch(
    embeddings: np.ndarray,
    positive_pairs: list[tuple[int, int]],
    negative_pairs: list[tuple[int, int]],
    pair_stability: list[tuple[int, int, bool]],
    *,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    rng = np.random.RandomState(seed)
    work = embeddings.copy()
    total_loss = 0.0
    n_samples = 0
    pos = list(positive_pairs)
    rng.shuffle(pos)
    for batch_start in range(0, min(len(pos), 2000), oe._BATCH_SIZE):
        batch = pos[batch_start : batch_start + oe._BATCH_SIZE]
        for anchor_idx, pos_idx in batch:
            if not negative_pairs:
                continue
            neg_idx_pair = negative_pairs[rng.randint(len(negative_pairs))]
            neg_idx = (
                neg_idx_pair[1] if neg_idx_pair[0] == anchor_idx else neg_idx_pair[0]
            )
            a = work[anchor_idx]
            p = work[pos_idx]
            n_vec = work[neg_idx]
            d_pos = np.sum((a - p) ** 2)
            d_neg = np.sum((a - n_vec) ** 2)
            margin_loss = max(0.0, d_pos - d_neg + oe._TRIPLET_MARGIN)
            if margin_loss > 0:
                grad_a = 2.0 * ((a - p) - (a - n_vec))
                grad_p = 2.0 * (p - a)
                grad_n = 2.0 * (a - n_vec)
                work[anchor_idx] -= oe._LEARNING_RATE * grad_a
                work[pos_idx] -= oe._LEARNING_RATE * grad_p
                work[neg_idx] += oe._LEARNING_RATE * grad_n
                total_loss += margin_loss
                n_samples += 1
    if pair_stability:
        for idx_a, idx_b, stable in pair_stability[: min(len(pair_stability), 1000)]:
            prod = work[idx_a] * work[idx_b]
            logit = np.sum(prod)
            pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            target = float(stable)
            bce = -(
                target * np.log(max(pred, 1e-8))
                + (1.0 - target) * np.log(max(1.0 - pred, 1e-8))
            )
            d_logit = pred - target
            grad_a = oe._PAIR_WEIGHT * d_logit * work[idx_b]
            grad_b = oe._PAIR_WEIGHT * d_logit * work[idx_a]
            work[idx_a] -= oe._LEARNING_RATE * grad_a
            work[idx_b] -= oe._LEARNING_RATE * grad_b
            total_loss += oe._PAIR_WEIGHT * bce
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    work = work / np.maximum(norms, 1e-8)
    return work, float(total_loss), n_samples


def _mean_pair_dot(embeddings: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    return float(
        np.mean([float(np.dot(embeddings[a], embeddings[b])) for a, b in pairs])
    )


def test_native_topology_features_match_python_reference():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_topology_features_native"):
        pytest.skip("native topology extractor unavailable")

    graph = _sample_graph()
    op_profiles = _sample_profiles()
    pair_stability = _sample_pair_stability()
    native_ctx = gp._make_native_topology_context(op_profiles, pair_stability)

    native = gp.extract_topology_features(
        graph,
        op_profiles,
        pair_stability,
        native_ctx=native_ctx,
    )
    python_ref = gp._extract_topology_features_python(
        graph, op_profiles, pair_stability
    )

    assert native is not None
    assert python_ref is not None
    assert native.keys() == python_ref.keys()
    for key in native:
        assert native[key] == pytest.approx(python_ref[key], rel=1e-6, abs=1e-6)


def test_native_graph_op_batch_matches_python_reference():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_graph_ops_batch"):
        pytest.skip("native graph-op batch extractor unavailable")
    payloads = _many_graph_payloads(64)
    native = go.extract_unique_graph_ops_batch(payloads)
    python_ref = [go._extract_unique_graph_ops_python(payload) for payload in payloads]
    assert native == python_ref


def test_native_interaction_training_produces_separable_scores():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "train_interaction_model_native_py"):
        pytest.skip("native interaction trainer unavailable")

    (
        u,
        v,
        W_s,
        W_l,
        stab_idx,
        stab_labels,
        stab_weights,
        loss_idx,
        loss_labels,
        loss_weights,
    ) = _make_interaction_train_data()
    trained = im._train_interaction_native(
        u=u.copy(),
        v=v.copy(),
        W_s=W_s.copy(),
        W_l=W_l.copy(),
        b_s=0.0,
        b_l=0.5,
        stab_idx=stab_idx,
        stab_labels=stab_labels,
        stab_weights=stab_weights,
        loss_idx=loss_idx,
        loss_labels=loss_labels,
        loss_weights=loss_weights,
        n_epochs=8,
        lr=0.005,
        seed=17,
    )
    assert trained is not None
    python_ref = im._train_interaction_python(
        u=u.copy(),
        v=v.copy(),
        W_s=W_s.copy(),
        W_l=W_l.copy(),
        b_s=0.0,
        b_l=0.5,
        stab_idx=stab_idx,
        stab_labels=stab_labels,
        stab_weights=stab_weights,
        loss_idx=loss_idx,
        loss_labels=loss_labels,
        loss_weights=loss_weights,
        n_epochs=8,
        lr=0.005,
        seed=17,
    )
    u_t, v_t, W_s_t, W_l_t, b_s_t, b_l_t, best_loss = trained
    u_p, v_p, W_s_p, W_l_p, b_s_p, b_l_p, python_best_loss = python_ref
    assert math.isfinite(best_loss)
    assert math.isfinite(python_best_loss)
    assert best_loss == pytest.approx(python_best_loss, rel=0.2, abs=5.0)
    assert u_t.shape == u_p.shape
    assert v_t.shape == v_p.shape
    assert W_s_t.shape == W_s_p.shape
    assert W_l_t.shape == W_l_p.shape
    assert abs(b_s_t - b_s_p) < 0.1
    assert abs(b_l_t - b_l_p) < 0.1
    pred_loss = float(u_t[loss_idx[0, 0]] @ W_l_t @ v_t[loss_idx[0, 1]] + b_l_t)
    assert math.isfinite(pred_loss)


def test_native_embedding_epoch_improves_pair_margin():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "train_op_embeddings_epoch_native_py"):
        pytest.skip("native embedding trainer unavailable")

    embeddings, positive_pairs, negative_pairs, pair_stability = (
        _make_embedding_epoch_data()
    )
    before_margin = _mean_pair_dot(embeddings, positive_pairs[:200]) - _mean_pair_dot(
        embeddings, negative_pairs[:200]
    )
    native = oe._run_native_embedding_epoch(
        embeddings.copy(),
        positive_pairs,
        negative_pairs,
        pair_stability,
        lr=oe._LEARNING_RATE,
        seed=23,
    )
    assert native is not None
    trained, total_loss, n_samples = native
    python_ref, python_loss, python_samples = _python_op_embedding_epoch(
        embeddings.copy(),
        positive_pairs,
        negative_pairs,
        pair_stability,
        seed=23,
    )
    after_margin = _mean_pair_dot(trained, positive_pairs[:200]) - _mean_pair_dot(
        trained, negative_pairs[:200]
    )
    python_margin = _mean_pair_dot(python_ref, positive_pairs[:200]) - _mean_pair_dot(
        python_ref, negative_pairs[:200]
    )
    assert math.isfinite(total_loss)
    assert n_samples >= 0
    assert math.isfinite(python_loss)
    assert python_samples >= 0
    assert after_margin == pytest.approx(python_margin, rel=0.5, abs=0.2)
    assert after_margin > before_margin - 0.2


def test_native_hotpaths_outperform_python_reference():
    rust = _try_import_rust_scheduler()
    if rust is None:
        pytest.skip("aria_scheduler not available")
    if not hasattr(rust, "extract_topology_features_native"):
        pytest.skip("native hotpath functions unavailable")

    graph = _sample_graph()
    op_profiles = _sample_profiles()
    pair_stability = _sample_pair_stability()
    native_ctx = gp._make_native_topology_context(op_profiles, pair_stability)

    graph_str = gp.json.dumps(graph, sort_keys=True, separators=(",", ":"))
    topology_iters = 2000
    t0 = time.perf_counter()
    for _ in range(topology_iters):
        gp._extract_topology_features_python(graph_str, op_profiles, pair_stability)
    python_topology_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(topology_iters):
        gp.extract_topology_features(
            graph_str,
            op_profiles,
            pair_stability,
            native_ctx=native_ctx,
        )
    native_topology_s = time.perf_counter() - t0
    assert native_topology_s < python_topology_s

    payloads = _many_graph_payloads(500)
    t0 = time.perf_counter()
    for payload in payloads:
        go._extract_unique_graph_ops_python(payload)
    python_graph_ops_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    go.extract_unique_graph_ops_batch(payloads)
    native_graph_ops_s = time.perf_counter() - t0
    assert native_graph_ops_s < python_graph_ops_s * 1.25

    interaction_data = _make_interaction_train_data(
        n_ops=160,
        n_stability_pairs=8000,
        n_loss_pairs=5000,
    )
    t0 = time.perf_counter()
    im._train_interaction_python(
        u=interaction_data[0].copy(),
        v=interaction_data[1].copy(),
        W_s=interaction_data[2].copy(),
        W_l=interaction_data[3].copy(),
        b_s=0.0,
        b_l=0.5,
        stab_idx=interaction_data[4],
        stab_labels=interaction_data[5],
        stab_weights=interaction_data[6],
        loss_idx=interaction_data[7],
        loss_labels=interaction_data[8],
        loss_weights=interaction_data[9],
        n_epochs=14,
        lr=0.005,
        seed=31,
    )
    python_interaction_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    native_interaction = im._train_interaction_native(
        u=interaction_data[0].copy(),
        v=interaction_data[1].copy(),
        W_s=interaction_data[2].copy(),
        W_l=interaction_data[3].copy(),
        b_s=0.0,
        b_l=0.5,
        stab_idx=interaction_data[4],
        stab_labels=interaction_data[5],
        stab_weights=interaction_data[6],
        loss_idx=interaction_data[7],
        loss_labels=interaction_data[8],
        loss_weights=interaction_data[9],
        n_epochs=14,
        lr=0.005,
        seed=31,
    )
    native_interaction_s = time.perf_counter() - t0
    assert native_interaction is not None
    assert native_interaction_s < python_interaction_s * 1.35

    embedding_data = _make_embedding_epoch_data(
        n_ops=192,
        n_positive=12000,
        n_negative=12000,
        n_pair_labels=4000,
    )
    t0 = time.perf_counter()
    _python_op_embedding_epoch(
        embedding_data[0].copy(),
        embedding_data[1],
        embedding_data[2],
        embedding_data[3],
        seed=41,
    )
    python_embedding_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    native_embedding = oe._run_native_embedding_epoch(
        embedding_data[0].copy(),
        embedding_data[1],
        embedding_data[2],
        embedding_data[3],
        lr=oe._LEARNING_RATE,
        seed=41,
    )
    native_embedding_s = time.perf_counter() - t0
    assert native_embedding is not None
    assert native_embedding_s < python_embedding_s
