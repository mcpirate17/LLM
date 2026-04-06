from __future__ import annotations

import ctypes
import time

import pytest

from research.scientist.native.intelligent_router import (
    NativeSparseHybridRouter,
    _RouteMeta,
)

pytestmark = pytest.mark.native


def _build_router() -> NativeSparseHybridRouter:
    router = NativeSparseHybridRouter(vocab=64, lanes=3)
    for _ in range(12):
        router.train_token_gate(3, keep=False, strength=2.0)
        router.train_token_gate(7, keep=True, strength=2.0)
        router.train_token_gate(11, keep=True, strength=2.0)
        router.train_token_gate(15, keep=True, strength=2.0)
        router.train_span_router([7, 11, 15], lane=2, strength=2.0)
    return router


def test_native_sparse_hybrid_router_route_train_and_save_roundtrip(tmp_path):
    router = _build_router()
    try:
        result = router.route([3, 7, 11, 15])
        assert len(result.token_actions) == 4
        assert len(result.token_keep_probability) == 4
        assert all(action in (0, 1) for action in result.token_actions)
        assert result.token_keep_probability[1] > result.token_keep_probability[0]
        assert result.token_keep_probability[2] > result.token_keep_probability[0]
        assert result.token_keep_probability[3] > result.token_keep_probability[0]
        assert result.spans
        assert result.spans[0].lane == 2
        assert result.spans[0].token_indices == (0, 1, 2)
        assert 0.0 <= result.spans[0].confidence <= 1.0

        save_path = tmp_path / "router_state.bin"
        router.save(save_path)
        assert save_path.exists()
    finally:
        router.close()

    loaded = NativeSparseHybridRouter.load(save_path)
    try:
        loaded_result = loaded.route([3, 7, 11, 15])
        assert loaded_result == result
    finally:
        loaded.close()


def test_native_sparse_hybrid_router_throughput_sanity():
    router = _build_router()
    try:
        sequence = [3, 7, 11, 15, 7, 11, 15, 3]
        for _ in range(64):
            router.route(sequence)

        iterations = 2000
        t0 = time.perf_counter()
        for _ in range(iterations):
            router.route(sequence)
        dt = time.perf_counter() - t0
        assert dt < 1.0, f"native intelligent router route loop too slow: {dt:.3f}s"
    finally:
        router.close()


def test_native_sparse_hybrid_router_retries_when_span_capacity_grows():
    class FakeNativeLib:
        def __init__(self):
            self.calls = []

        def aria_irouter_route(
            self,
            handle,
            seq_buf,
            seq_len,
            token_actions_out,
            token_keep_probability_out,
            span_token_indices_out,
            span_token_indices_capacity,
            span_lanes_out,
            span_confidences_out,
            out_meta,
        ):
            self.calls.append(int(span_token_indices_capacity))
            meta = ctypes.cast(out_meta, ctypes.POINTER(_RouteMeta)).contents
            meta.span_count = 2
            meta.required_span_capacity = 6
            token_actions_out[0] = 1
            token_actions_out[1] = 0
            token_keep_probability_out[0] = 0.9
            token_keep_probability_out[1] = 0.1
            if span_token_indices_capacity < 6:
                return 7
            for idx, value in enumerate((0, 1, 2, 2, 3, 4)):
                span_token_indices_out[idx] = value
            span_lanes_out[0] = 1
            span_lanes_out[1] = 2
            span_confidences_out[0] = 0.8
            span_confidences_out[1] = 0.7
            return 0

    router = NativeSparseHybridRouter.__new__(NativeSparseHybridRouter)
    router._native_lib = FakeNativeLib()
    router._handle = 123
    router._closed = False

    result = router.route([7, 3])

    assert router._native_lib.calls == [3, 6]
    assert result.token_actions == (1, 0)
    assert result.spans[0].token_indices == (0, 1, 2)
    assert result.spans[1].token_indices == (2, 3, 4)
    assert result.spans[0].lane == 1
    assert result.spans[1].lane == 2
