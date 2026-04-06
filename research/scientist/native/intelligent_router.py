from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .abi import _try_load_native_lib


class _RouteMeta(ctypes.Structure):
    _fields_ = [
        ("span_count", ctypes.c_int32),
        ("required_span_capacity", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class HybridSparseSpan:
    token_indices: tuple[int, ...]
    lane: int
    confidence: float


@dataclass(frozen=True)
class HybridRouteResult:
    token_actions: tuple[int, ...]
    token_keep_probability: tuple[float, ...]
    spans: tuple[HybridSparseSpan, ...]


def _decode_last_error(native_lib: Any) -> str:
    fn = getattr(native_lib, "aria_irouter_last_error", None)
    if fn is None:
        return ""
    raw = fn()
    if not raw:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _configure_intelligent_router_abi(native_lib: Any) -> Any:
    if native_lib is None:
        raise RuntimeError("native intelligent router library is not available")
    if getattr(native_lib, "_intelligent_router_abi_configured", False):
        return native_lib

    native_lib.aria_irouter_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    native_lib.aria_irouter_create.restype = ctypes.c_int32

    native_lib.aria_irouter_destroy.argtypes = [ctypes.c_int64]
    native_lib.aria_irouter_destroy.restype = ctypes.c_int32

    native_lib.aria_irouter_train_token_gate.argtypes = [
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    native_lib.aria_irouter_train_token_gate.restype = ctypes.c_int32

    native_lib.aria_irouter_train_span_router.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    native_lib.aria_irouter_train_span_router.restype = ctypes.c_int32

    native_lib.aria_irouter_route.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(_RouteMeta),
    ]
    native_lib.aria_irouter_route.restype = ctypes.c_int32

    native_lib.aria_irouter_save.argtypes = [ctypes.c_int64, ctypes.c_char_p]
    native_lib.aria_irouter_save.restype = ctypes.c_int32

    native_lib.aria_irouter_load.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
    ]
    native_lib.aria_irouter_load.restype = ctypes.c_int32

    native_lib.aria_irouter_last_error.argtypes = []
    native_lib.aria_irouter_last_error.restype = ctypes.c_char_p

    native_lib._intelligent_router_abi_configured = True
    return native_lib


class NativeSparseHybridRouter:
    def __init__(self, vocab: int, lanes: int):
        self._native_lib = _configure_intelligent_router_abi(_try_load_native_lib())
        self.vocab = int(vocab)
        self.lanes = int(lanes)
        handle = ctypes.c_int64()
        status = int(
            self._native_lib.aria_irouter_create(
                self.vocab, self.lanes, ctypes.byref(handle)
            )
        )
        self._check_status(status, "create")
        self._handle = int(handle.value)
        self._closed = False

    @classmethod
    def load(cls, path: str | Path) -> "NativeSparseHybridRouter":
        native_lib = _configure_intelligent_router_abi(_try_load_native_lib())
        handle = ctypes.c_int64()
        encoded = str(Path(path)).encode("utf-8")
        status = int(
            native_lib.aria_irouter_load(ctypes.c_char_p(encoded), ctypes.byref(handle))
        )
        if status != 0:
            message = _decode_last_error(native_lib) or "unknown error"
            raise RuntimeError(
                f"intelligent router load failed: {message} (status={status})"
            )
        self = cls.__new__(cls)
        self._native_lib = native_lib
        self.vocab = 0
        self.lanes = 0
        self._handle = int(handle.value)
        self._closed = False
        return self

    def close(self) -> None:
        if self._closed:
            return
        status = int(self._native_lib.aria_irouter_destroy(self._handle))
        if status not in (0, -2):
            self._check_status(status, "destroy")
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_status(self, status: int, action: str) -> None:
        if status == 0:
            return
        message = _decode_last_error(self._native_lib) or "unknown error"
        raise RuntimeError(
            f"intelligent router {action} failed: {message} (status={status})"
        )

    def _encode_sequence(self, sequence: Sequence[int]) -> tuple[Any, int]:
        seq = tuple(int(token) for token in sequence)
        if not seq:
            raise ValueError("sequence must be non-empty")
        buf = (ctypes.c_int32 * len(seq))(*seq)
        return buf, len(seq)

    def train_token_gate(self, token: int, keep: bool, strength: float = 1.0) -> None:
        status = int(
            self._native_lib.aria_irouter_train_token_gate(
                self._handle,
                int(token),
                1 if keep else 0,
                float(strength),
            )
        )
        self._check_status(status, "train_token_gate")

    def train_span_router(
        self, sequence: Sequence[int], lane: int, strength: float = 1.0
    ) -> None:
        seq_buf, seq_len = self._encode_sequence(sequence)
        status = int(
            self._native_lib.aria_irouter_train_span_router(
                self._handle,
                seq_buf,
                seq_len,
                int(lane),
                float(strength),
            )
        )
        self._check_status(status, "train_span_router")

    def route(self, sequence: Sequence[int]) -> HybridRouteResult:
        seq_buf, seq_len = self._encode_sequence(sequence)
        token_actions = (ctypes.c_int32 * seq_len)()
        token_keep_probability = (ctypes.c_float * seq_len)()
        meta, span_token_indices, span_lanes, span_confidences = self._route_native(
            seq_buf,
            seq_len,
            token_actions,
            token_keep_probability,
        )
        spans = []
        for idx in range(int(meta.span_count)):
            base = idx * 3
            spans.append(
                HybridSparseSpan(
                    token_indices=tuple(
                        int(span_token_indices[base + j]) for j in range(3)
                    ),
                    lane=int(span_lanes[idx]),
                    confidence=float(span_confidences[idx]),
                )
            )
        return HybridRouteResult(
            token_actions=tuple(int(token_actions[i]) for i in range(seq_len)),
            token_keep_probability=tuple(
                float(token_keep_probability[i]) for i in range(seq_len)
            ),
            spans=tuple(spans),
        )

    def _route_native(
        self,
        seq_buf: Any,
        seq_len: int,
        token_actions: Any,
        token_keep_probability: Any,
    ) -> tuple[_RouteMeta, Any, Any, Any]:
        span_capacity = 3
        while True:
            span_count_capacity = max(1, span_capacity // 3)
            span_token_indices = (ctypes.c_int32 * span_capacity)()
            span_lanes = (ctypes.c_int32 * span_count_capacity)()
            span_confidences = (ctypes.c_float * span_count_capacity)()
            meta = _RouteMeta()
            status = int(
                self._native_lib.aria_irouter_route(
                    self._handle,
                    seq_buf,
                    seq_len,
                    token_actions,
                    token_keep_probability,
                    span_token_indices,
                    span_capacity,
                    span_lanes,
                    span_confidences,
                    ctypes.byref(meta),
                )
            )
            if status == 0:
                return meta, span_token_indices, span_lanes, span_confidences
            required = int(meta.required_span_capacity)
            if required > span_capacity:
                span_capacity = required
                continue
            self._check_status(status, "route")

    def save(self, path: str | Path) -> None:
        encoded = str(Path(path)).encode("utf-8")
        status = int(
            self._native_lib.aria_irouter_save(self._handle, ctypes.c_char_p(encoded))
        )
        self._check_status(status, "save")
