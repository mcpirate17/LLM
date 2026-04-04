from __future__ import annotations

import ctypes
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

from .core import NativeRunnerState, _FALLBACK_METRICS, _env_flag

logger = logging.getLogger(__name__)
_native_lib_cache: Any = False


class _NrCompileRequest(ctypes.Structure):
    _fields_ = [
        ("ir_json", ctypes.c_char_p),
        ("ir_json_len", ctypes.c_int64),
        ("vocab_size", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
    ]


class _NrCompileResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("model_handle", ctypes.c_int64),
        ("message", ctypes.c_char_p),
    ]


class _NrExecuteRequest(ctypes.Structure):
    _fields_ = [
        ("model_handle", ctypes.c_int64),
        ("token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("batch", ctypes.c_int32),
        ("seq_len", ctypes.c_int32),
    ]


class _NrExecuteResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


class _NrExecuteBatchResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("batch", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


def _try_load_native_lib() -> Any:
    """Try to load the native C kernel library. Returns ctypes CDLL or None.

    The result is cached in ``_native_lib_cache`` so subsequent calls are free.
    """
    global _native_lib_cache
    if _native_lib_cache is not False:
        return _native_lib_cache

    lib_paths = [
        Path(__file__).resolve().parents[2]
        / "runtime"
        / "native"
        / "build"
        / "libaria_native_runtime.so",
        Path(__file__).resolve().parents[3]
        / "aria_designer"
        / "runtime"
        / "lib"
        / "libaria_runtime.so",
    ]
    for p in lib_paths:
        if p.exists():
            try:
                _native_lib_cache = ctypes.CDLL(
                    str(p),
                    mode=getattr(ctypes, "RTLD_GLOBAL", 0),
                )
                logger.info("Loaded native kernel library from %s", p)
                return _native_lib_cache
            except OSError as exc:
                logger.debug("Failed to load native lib at %s: %s", p, exc)
                continue

    _native_lib_cache = None
    return None


def _reset_native_lib_cache() -> None:
    """Reset the library cache (used in tests)."""
    global _native_lib_cache
    _native_lib_cache = False


def _normalize_nr_compile_reason(
    compile_status: int, compile_message: Optional[str]
) -> str:
    msg = str(compile_message or "").strip().lower()
    if not msg:
        return f"status_{int(compile_status)}"

    known_prefixes = (
        "unsupported_graph_family_",
        "missing_",
        "invalid_",
        "strict_mode_",
        "handle_",
        "logit_",
        "add_",
        "mul_",
        "matmul_",
        "linear_",
        "softmax_",
        "rmsnorm_",
        "sub_",
        "unary_",
    )
    if msg.startswith(known_prefixes):
        return msg
    if "required_chain_missing_or_invalid" in msg:
        return "unsupported_graph_family_required_chain_missing_or_invalid"
    if "required_chain_invalid" in msg:
        return "unsupported_graph_family_required_chain_invalid"
    if "unsupported_graph_family" in msg:
        return "unsupported_graph_family_unspecified"
    if "kernel" in msg:
        return "kernel_lookup_failure"
    return msg.replace(":", "_").replace(" ", "_")


def _build_native_abi_only_model(
    abi_session: NativeRunnerAbiSession,
    vocab_size: int,
    model_dim: int = 0,
):
    """Build an inference-only torch module backed by runner ABI session."""
    import torch

    class _NativeAbiOnlyModel(torch.nn.Module):
        def __init__(self, session: NativeRunnerAbiSession, n_vocab: int, dim: int):
            super().__init__()
            self._abi_session = session
            self.vocab_size = int(n_vocab)
            self.model_dim = int(dim or 0)
            self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, input_ids):
            if input_ids is None:
                raise ValueError("input_ids is required")
            if input_ids.ndim != 2:
                raise ValueError("input_ids must be rank-2 [B, S]")
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1])
            if seq_len <= 0:
                raise ValueError("input_ids sequence length must be > 0")
            token_rows = [
                tuple(int(v) for v in row)
                for row in input_ids.detach().to(device="cpu").tolist()
            ]
            row_logits = self._abi_session.execute_token_rows(token_rows)
            out = torch.tensor(
                row_logits,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return out.view(batch_size, 1, self.vocab_size).expand(
                batch_size, seq_len, -1
            )

    return _NativeAbiOnlyModel(abi_session, vocab_size, model_dim)


class NativeRunnerAbiSession:
    """Holder for runner ABI compiled handle + token execute helper."""

    def __init__(
        self, native_lib: Any, model_handle: int, vocab_size: int, max_seq_len: int
    ):
        self._native_lib = native_lib
        self.model_handle = int(model_handle)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self._closed = False
        self._execute_cache: "OrderedDict[Tuple[int, Tuple[int, ...]], Tuple[float, ...]]" = OrderedDict()
        self._execute_cache_limit = 256
        self._has_batch_execute = hasattr(native_lib, "nr_execute_batch")

    def _cache_get(
        self, cache_key: Tuple[int, Tuple[int, ...]]
    ) -> Optional[Tuple[float, ...]]:
        cached = self._execute_cache.get(cache_key)
        if cached is None:
            return None
        self._execute_cache.move_to_end(cache_key)
        return cached

    def _cache_put(
        self, cache_key: Tuple[int, Tuple[int, ...]], logits: Sequence[float]
    ) -> Tuple[float, ...]:
        cached_logits = tuple(float(v) for v in logits)
        self._execute_cache[cache_key] = cached_logits
        self._execute_cache.move_to_end(cache_key)
        if len(self._execute_cache) > self._execute_cache_limit:
            self._execute_cache.popitem(last=False)
        return cached_logits

    def execute_tokens(self, token_ids: List[int], batch: int = 1) -> List[float]:
        if self._closed:
            raise RuntimeError("native ABI session already closed")
        flat_token_ids = tuple(int(t) for t in token_ids)
        flat_count = int(len(flat_token_ids))
        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch must be > 0")
        if flat_count <= 0:
            raise ValueError("token_ids must be non-empty")
        if flat_count % batch != 0:
            raise ValueError("token_ids length must be divisible by batch")
        seq_len = flat_count // batch
        if seq_len <= 0:
            raise ValueError("token_ids must be non-empty")
        if seq_len > self.max_seq_len:
            raise ValueError("token length exceeds compiled max_seq_len")
        cache_key = (batch, flat_token_ids)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return list(cached)

        token_buf = (ctypes.c_int32 * flat_count)(*flat_token_ids)
        req = _NrExecuteRequest(
            model_handle=self.model_handle,
            token_ids=token_buf,
            batch=batch,
            seq_len=seq_len,
        )
        resp = self._native_lib.nr_execute(ctypes.byref(req))
        if int(resp.status) != 0 or not bool(resp.logits):
            raise RuntimeError(f"runner ABI execute failed: status={int(resp.status)}")
        n_vocab = int(resp.vocab_size)
        return list(
            self._cache_put(cache_key, (float(resp.logits[i]) for i in range(n_vocab)))
        )

    def execute_token_rows(
        self, token_rows: Sequence[Sequence[int]]
    ) -> List[Tuple[float, ...]]:
        row_keys = [tuple(int(t) for t in row) for row in token_rows]
        if not row_keys:
            return []
        if self._has_batch_execute:
            uncached_rows = [
                row_key
                for row_key in dict.fromkeys(row_keys)
                if self._cache_get((1, row_key)) is None
            ]
            if uncached_rows:
                flat_tokens = [token for row_key in uncached_rows for token in row_key]
                seq_len = len(uncached_rows[0])
                if any(len(row_key) != seq_len for row_key in uncached_rows):
                    raise ValueError(
                        "all token rows must have the same sequence length"
                    )
                token_buf = (ctypes.c_int32 * len(flat_tokens))(*flat_tokens)
                req = _NrExecuteRequest(
                    model_handle=self.model_handle,
                    token_ids=token_buf,
                    batch=len(uncached_rows),
                    seq_len=seq_len,
                )
                resp = self._native_lib.nr_execute_batch(ctypes.byref(req))
                if int(resp.status) != 0 or not bool(resp.logits):
                    raise RuntimeError(
                        f"runner ABI batch execute failed: status={int(resp.status)}"
                    )
                if int(resp.batch) != len(uncached_rows):
                    raise RuntimeError(
                        f"runner ABI batch execute returned unexpected batch: {int(resp.batch)}"
                    )
                if int(resp.vocab_size) != self.vocab_size:
                    raise RuntimeError(
                        f"runner ABI batch execute returned unexpected vocab: {int(resp.vocab_size)}"
                    )
                vocab_size = self.vocab_size
                for row_idx, row_key in enumerate(uncached_rows):
                    base = row_idx * vocab_size
                    self._cache_put(
                        (1, row_key),
                        (
                            float(resp.logits[base + offset])
                            for offset in range(vocab_size)
                        ),
                    )
        deduped: Dict[Tuple[int, ...], Tuple[float, ...]] = {}
        for row_key in row_keys:
            if row_key in deduped:
                continue
            cached = self._cache_get((1, row_key))
            if cached is None:
                deduped[row_key] = tuple(self.execute_tokens(list(row_key), batch=1))
            else:
                deduped[row_key] = cached
        return [deduped[row_key] for row_key in row_keys]

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native_lib.nr_release_model(ctypes.c_int64(self.model_handle))
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)
        self._closed = True
        self._execute_cache.clear()


def _maybe_prepare_runner_abi_session(
    *,
    layer_graphs: List[Any],
    native_lib: Any,
    state: NativeRunnerState,
    vocab_size: int,
    max_seq_len: Optional[int],
) -> Dict[str, Any]:
    """Optional compile+smoke path through `runner_abi` for first-family execution."""
    report: Dict[str, Any] = {
        "requested": False,
        "attempted": False,
        "succeeded": False,
        "reason": "disabled",
        "model_handle": None,
        "session": None,
    }

    if not _env_flag("NATIVE_RUNNER_ABI_EXEC", False):
        return report
    report["requested"] = True

    if native_lib is None:
        report["reason"] = "native_lib_unavailable"
        return report
    if not layer_graphs:
        report["reason"] = "no_layer_graphs"
        return report
    if not all(
        hasattr(native_lib, name)
        for name in (
            "nr_runtime_init",
            "nr_set_strict_mode",
            "nr_compile",
            "nr_execute",
            "nr_release_model",
        )
    ):
        report["reason"] = "runner_abi_symbols_missing"
        return report

    abi_supported_unary_ops = {"relu", "gelu", "silu", "sigmoid", "tanh", "exp"}

    def _graph_is_abi_family_candidate(candidate: Any) -> bool:
        nodes = getattr(candidate, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return False
        known_node_ids = {str(node_id) for node_id in nodes.keys()}
        required_order = [
            "exp",
            "add",
            "mul",
            "matmul",
            "linear",
            "softmax",
            "rmsnorm",
            "sub",
        ]
        first_positions = {op_name: None for op_name in required_order}
        first_node_ids = {op_name: None for op_name in required_order}
        required_counts = {op_name: 0 for op_name in required_order}
        input_incoming_counts: Dict[str, Dict[str, int]] = {
            str(node_id): {} for node_id in nodes.keys()
        }
        edge_incoming_counts: Dict[str, Dict[str, int]] = {
            str(node_id): {} for node_id in nodes.keys()
        }
        input_refs_by_node: Dict[str, List[str]] = {
            str(node_id): [] for node_id in nodes.keys()
        }
        raw_declared_edges: List[Dict[str, str]] = []
        has_unary = False
        for idx, (node_id, node) in enumerate(nodes.items()):
            op_name = str(getattr(node, "op_name", "") or "").strip().lower()
            if op_name in required_counts:
                required_counts[op_name] += 1
            if op_name in first_positions and first_positions[op_name] is None:
                first_positions[op_name] = idx
                first_node_ids[op_name] = str(node_id)
            raw_inputs = getattr(node, "input_ids", None)
            if isinstance(raw_inputs, (list, tuple, set)):
                for src in raw_inputs:
                    src_id = str(src)
                    if src_id:
                        child_key = str(node_id)
                        input_refs_by_node.setdefault(child_key, []).append(src_id)
                        child_counts = input_incoming_counts.get(child_key)
                        if child_counts is not None:
                            child_counts[src_id] = int(child_counts.get(src_id, 0)) + 1
            if op_name in abi_supported_unary_ops:
                has_unary = True

        edges = getattr(candidate, "edges", None)
        has_declared_edges = edges is not None
        if isinstance(edges, (list, tuple)):
            for edge in edges:
                source = str(getattr(edge, "source", "") or "")
                target = str(getattr(edge, "target", "") or "")
                if not source or not target:
                    if isinstance(edge, dict):
                        source = str(edge.get("source", "") or "")
                        target = str(edge.get("target", "") or "")
                if source and target:
                    raw_declared_edges.append({"source": source, "target": target})
                if source and target and target in edge_incoming_counts:
                    target_counts = edge_incoming_counts.get(target)
                    if target_counts is not None:
                        target_counts[source] = int(target_counts.get(source, 0)) + 1

        if not has_unary:
            return False
        if any(first_positions[op_name] is None for op_name in required_order):
            return False
        if any(int(required_counts[op_name]) != 1 for op_name in required_order):
            return False
        if not all(
            int(first_positions[required_order[i]])
            < int(first_positions[required_order[i + 1]])
            for i in range(len(required_order) - 1)
        ):
            return False

        required_chain = [
            ("exp", "add"),
            ("add", "mul"),
            ("mul", "matmul"),
            ("matmul", "linear"),
            ("linear", "softmax"),
            ("softmax", "rmsnorm"),
            ("rmsnorm", "sub"),
        ]
        has_explicit_edges = has_declared_edges
        required_chain_node_ids = {
            str(first_node_ids[op_name])
            for op_name in required_order
            if first_node_ids[op_name] is not None
        }

        for child_node_id in required_chain_node_ids:
            # P6.R35: strict node-reference sanity for required links.
            # Reject when required-chain links reference missing node ids.
            for src_id in input_refs_by_node.get(child_node_id, []):
                if src_id not in known_node_ids:
                    return False

        if has_explicit_edges:
            for edge in raw_declared_edges:
                source = str(edge.get("source", "") or "")
                target = str(edge.get("target", "") or "")
                # P6.R35: strict reference sanity for explicit edge endpoints.
                if target in required_chain_node_ids:
                    if source not in known_node_ids or target not in known_node_ids:
                        return False

        for parent_op, child_op in required_chain:
            parent_node_id = first_node_ids[parent_op]
            child_node_id = first_node_ids[child_op]
            if parent_node_id is None or child_node_id is None:
                return False
            child_node_id_str = str(child_node_id)
            parent_node_id_str = str(parent_node_id)
            input_parent_count = int(
                input_incoming_counts.get(child_node_id_str, {}).get(
                    parent_node_id_str, 0
                )
            )
            if input_parent_count != 1:
                return False
            if has_explicit_edges:
                edge_parent_count = int(
                    edge_incoming_counts.get(child_node_id_str, {}).get(
                        parent_node_id_str, 0
                    )
                )
                if edge_parent_count != 1:
                    return False
        return True

    graph = next(
        (
            g
            for g in layer_graphs
            if hasattr(g, "nodes") and _graph_is_abi_family_candidate(g)
        ),
        None,
    )
    if graph is None:
        report["reason"] = "no_abi_family_graph"
        return report

    report["attempted"] = True
    try:
        from ...synthesis.native_ir_converter import graph_to_native_ir_json

        native_lib.nr_runtime_init.restype = ctypes.c_int32
        native_lib.nr_set_strict_mode.argtypes = [ctypes.c_int32]
        native_lib.nr_set_strict_mode.restype = ctypes.c_int32
        native_lib.nr_compile.argtypes = [ctypes.POINTER(_NrCompileRequest)]
        native_lib.nr_compile.restype = _NrCompileResponse
        native_lib.nr_execute.argtypes = [ctypes.POINTER(_NrExecuteRequest)]
        native_lib.nr_execute.restype = _NrExecuteResponse
        if hasattr(native_lib, "nr_execute_batch"):
            native_lib.nr_execute_batch.argtypes = [ctypes.POINTER(_NrExecuteRequest)]
            native_lib.nr_execute_batch.restype = _NrExecuteBatchResponse
        native_lib.nr_release_model.argtypes = [ctypes.c_int64]

        init_status = int(native_lib.nr_runtime_init())
        if init_status != 0:
            report["reason"] = f"nr_runtime_init_failed:{init_status}"
            return report
        native_lib.nr_set_strict_mode(1 if state.strict else 0)

        graph_ir = graph_to_native_ir_json(graph).encode("utf-8")
        compile_req = _NrCompileRequest(
            ir_json=graph_ir,
            ir_json_len=len(graph_ir),
            vocab_size=int(vocab_size),
            max_seq_len=int(max_seq_len or 128),
        )
        compile_resp = native_lib.nr_compile(ctypes.byref(compile_req))
        compile_status = int(compile_resp.status)
        compile_message = (
            compile_resp.message.decode("utf-8", errors="ignore")
            if getattr(compile_resp, "message", None)
            else None
        )
        if compile_status != 0:
            compile_reason = _normalize_nr_compile_reason(
                compile_status, compile_message
            )
            report["reason"] = f"nr_compile_failed:{compile_status}:{compile_reason}"
            report["compile_status"] = compile_status
            report["compile_reason"] = compile_reason
            report["compile_message"] = compile_message
            return report

        handle = int(compile_resp.model_handle)
        report["model_handle"] = handle
        session = NativeRunnerAbiSession(
            native_lib=native_lib,
            model_handle=handle,
            vocab_size=int(vocab_size),
            max_seq_len=int(max_seq_len or 128),
        )

        # Tiny deterministic execute smoke so the handle is known-good.
        logits = session.execute_tokens([1, 2, 3, 4], batch=1)
        if not logits:
            session.close()
            report["reason"] = "nr_execute_empty_logits"
            return report

        report["session"] = session
        report["succeeded"] = True
        report["reason"] = "ok"
        report["compile_message"] = compile_message
        report["compile_status"] = compile_status
        report["compile_reason"] = _normalize_nr_compile_reason(
            compile_status, compile_message
        )
        return report
    except Exception as exc:
        report["reason"] = f"runner_abi_error:{exc}"
        return report


def record_native_abi_parity_result(passed: Optional[bool]) -> None:
    """Record sampled ABI parity outcome from sandbox/runner integration."""
    if passed is None:
        return
    _FALLBACK_METRICS["parity_samples"] += 1
    if bool(passed):
        _FALLBACK_METRICS["parity_passes"] += 1
    else:
        _FALLBACK_METRICS["parity_failures"] += 1
