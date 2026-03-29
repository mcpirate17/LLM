"""Kernel dispatcher — routes calls to aria_core (pybind11) or Python/torch fallback."""

from collections import deque

import numpy as np
import torch
from typing import Optional

from research.defaults import ROPE_THETA_BASE

try:
    import aria_core

    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False


def _np_to_torch(x):
    """Convert numpy array to contiguous float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _torch_to_np(t):
    """Convert torch tensor to numpy."""
    return t.numpy()


class KernelDispatcher:
    """Dispatches calls to aria_core or Python/torch fallback."""

    __slots__ = ("use_native",)

    def __init__(self, use_native=True):
        self.use_native = use_native

    # ── Graph validation ──────────────────────────────────────────────

    def validate_graph(self, nodes, edges):
        """Validate graph topology.

        nodes: list of node IDs (indices used internally)
        edges: list of (src_node_idx, tgt_node_idx, src_port, tgt_port)
        """
        if _HAS_ARIA_CORE:
            edge_list = [[int(s), int(t), int(sp), int(tp)] for s, t, sp, tp in edges]
            return aria_core.validate_graph(len(nodes), edge_list)

        return self._validate_graph_python(nodes, edges)

    def _validate_graph_python(self, nodes, edges):
        """Pure Python fallback for graph validation (Kahn's algorithm)."""
        n = len(nodes)
        in_deg = [0] * n
        out_deg = [0] * n
        adj = [[] for _ in range(n)]
        for s, t, _, _ in edges:
            adj[s].append(t)
            in_deg[t] += 1
            out_deg[s] += 1
        queue = deque(i for i in range(n) if in_deg[i] == 0)
        topo = []
        while queue:
            node = queue.popleft()
            topo.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        if len(topo) != n:
            return {"valid": False, "error": "Cycle detected", "code": -3}
        return {
            "valid": True,
            "topo_order": topo,
            "in_degrees": [0] * n,
            "out_degrees": out_deg,
        }

    # ── Shape inference ───────────────────────────────────────────────

    def infer_shapes(self, topo_order, edges, node_rules):
        """Propagate shapes through the graph.

        node_rules: list of {rule: ShapeRule, n_inputs, n_outputs, input_shapes: [...], ...}
        """
        if _HAS_ARIA_CORE:
            edge_list = [[int(e[0]), int(e[1]), int(e[2]), int(e[3])] for e in edges]
            rules_dicts = []
            for rd in node_rules:
                d = {
                    "rule": int(rd["rule"]),
                    "n_inputs": rd.get("n_inputs", 1),
                    "n_outputs": rd.get("n_outputs", 1),
                    "split_n": rd.get("split_n", 0),
                    "out_dim": rd.get("out_dim", -1),
                    "orig_seq_len": rd.get("orig_seq_len", 0),
                }
                if "input_shapes" in rd:
                    d["input_shapes"] = rd["input_shapes"]
                rules_dicts.append(d)
            return aria_core.propagate_shapes(
                [int(x) for x in topo_order], edge_list, rules_dicts
            )

        return {"valid": False, "error": "No shape inference backend available"}

    # ── Kernel dispatch methods ───────────────────────────────────────
    # Each method tries: aria_core → Python/torch fallback

    def relu(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.relu_f32(_np_to_torch(x)))
        return torch.relu(torch.from_numpy(x)).numpy()

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.matmul_f32(_np_to_torch(a), _np_to_torch(b)))
        return (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()

    def tropical_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.tropical_add_f32(_np_to_torch(a), _np_to_torch(b))
            )
        return np.minimum(a, b)

    def tropical_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.tropical_matmul_f32(_np_to_torch(a), _np_to_torch(b))
            )
        return np.min(a[:, :, None] + b[None, :, :], axis=1).astype(np.float32)

    def tropical_center(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_center_f32(_np_to_torch(x)))
        baseline = np.min(x, axis=1, keepdims=True)
        return x - baseline

    def hyp_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.hyp_distance_f32(_np_to_torch(x), _np_to_torch(y))
            )
        return np.linalg.norm(x - y, axis=-1)

    # ── Reference Architecture Ops ──

    def embedding_lookup(
        self, table: np.ndarray, idx: np.ndarray, pe: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            pe_t = _np_to_torch(pe) if pe is not None else None
            return _torch_to_np(
                aria_core.embedding_lookup_f32(
                    _np_to_torch(table), torch.from_numpy(idx.astype(np.int32)), pe_t
                )
            )
        t = torch.from_numpy(table)
        i = torch.from_numpy(idx.astype(np.int64))
        res = t[i]
        if pe is not None:
            res += torch.from_numpy(pe)
        return res.numpy()

    def rope_rotate(
        self, x: np.ndarray, theta_base: float = ROPE_THETA_BASE
    ) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.rope_rotate_f32(_np_to_torch(x), float(theta_base))
            )
        # RoPE: rotate pairs of dimensions by position-dependent angles
        # x shape: (..., seq_len, d) where d is even
        *batch, seq_len, d = x.shape
        half_d = d // 2
        positions = np.arange(seq_len, dtype=np.float32)
        dim_indices = np.arange(half_d, dtype=np.float32)
        freqs = positions[:, None] / (theta_base ** (dim_indices[None, :] / half_d))
        cos_f = np.cos(freqs).astype(np.float32)
        sin_f = np.sin(freqs).astype(np.float32)
        x1 = x[..., :half_d]
        x2 = x[..., half_d:]
        return np.concatenate(
            [x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1
        )

    def gated_linear(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: Optional[np.ndarray],
        wg: np.ndarray,
        bg: Optional[np.ndarray],
    ) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            bt = _np_to_torch(b) if b is not None else None
            bgt = _np_to_torch(bg) if bg is not None else None
            return _torch_to_np(
                aria_core.gated_linear_f32(
                    _np_to_torch(x), _np_to_torch(w), bt, _np_to_torch(wg), bgt
                )
            )
        xt = torch.from_numpy(x)
        res = torch.nn.functional.linear(
            xt, torch.from_numpy(w), torch.from_numpy(b) if b is not None else None
        )
        gate = torch.nn.functional.linear(
            xt, torch.from_numpy(wg), torch.from_numpy(bg) if bg is not None else None
        )
        return (res * torch.sigmoid(gate)).numpy()

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.cosine_similarity_f32(_np_to_torch(a), _np_to_torch(b))
            )
        at = torch.from_numpy(a)
        bt = torch.from_numpy(b)
        return torch.nn.functional.cosine_similarity(at, bt, dim=-1).numpy()

    def rwkv_time_mixing(
        self,
        x: np.ndarray,
        decay: np.ndarray,
        bonus: np.ndarray,
        wk: np.ndarray,
        wv: np.ndarray,
        wr: np.ndarray,
    ) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.rwkv_time_mixing_f32(
                    _np_to_torch(x),
                    _np_to_torch(decay),
                    _np_to_torch(bonus),
                    _np_to_torch(wk),
                    _np_to_torch(wv),
                    _np_to_torch(wr),
                )
            )
        # RWKV time-mixing: linear attention with exponential decay
        # x: (batch, seq_len, d), decay/bonus: (d,), wk/wv/wr: (d, d)
        k = x @ wk  # (batch, seq_len, d)
        v = x @ wv
        r = 1.0 / (1.0 + np.exp(-(x @ wr)))  # receptance gate (sigmoid)

        b, seq_len, d = k.shape
        out = np.zeros_like(x, dtype=np.float32)
        state = np.zeros((b, d), dtype=np.float32)
        for t in range(seq_len):
            kt = k[:, t, :]
            vt = v[:, t, :]
            bonus_term = np.exp(bonus + kt) * vt
            out[:, t, :] = r[:, t, :] * (state + bonus_term)
            state = np.exp(decay) * state + np.exp(kt) * vt
        return out

    def causal_mask(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.causal_mask_f32(_np_to_torch(x)))
        xt = torch.from_numpy(x)
        mask = torch.triu(torch.ones(xt.size(-2), xt.size(-1)), diagonal=1).bool()
        xt.masked_fill_(mask, float("-inf"))
        return xt.numpy()

    def padic_gate(self, x: np.ndarray, p: float = 2.0) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.padic_gate_f32(_np_to_torch(x), p))
        abs_x = np.maximum(np.abs(x), 1e-10)
        valuation = -(np.log(abs_x) / np.log(p))
        valuation = np.clip(valuation, -10.0, 10.0)
        gate = 1.0 / (1.0 + np.exp(-valuation))
        return x * gate

    @staticmethod
    def _tropical_distance_weights(x: np.ndarray, temperature: float) -> np.ndarray:
        """Tropical distance matrix + softmax weights. x: (b, s, d)."""
        dist = np.min(x[:, :, None, :] + x[:, None, :, :], axis=-1)
        weights = np.exp(-dist / temperature)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return weights

    def tropical_attention(self, x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.tropical_attention_f32(_np_to_torch(x), temperature)
            )
        weights = self._tropical_distance_weights(x, temperature)
        return np.einsum("brc,bcd->brd", weights, x).astype(np.float32)

    def tropical_gate(self, x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(
                aria_core.tropical_gate_f32(_np_to_torch(x), temperature)
            )
        weights = self._tropical_distance_weights(x, temperature)
        gated = np.einsum("brc,bcd->brd", weights, x)
        return (x * (1.0 / (1.0 + np.exp(-gated)))).astype(np.float32)

    def file_loader_csv(
        self,
        file_path: str,
        _max_rows: int = 4096,
        max_cols: int = 1024,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> np.ndarray:
        skip = 1 if has_header else 0
        return np.loadtxt(
            file_path,
            dtype=np.float32,
            delimiter=delimiter,
            skiprows=skip,
            max_rows=_max_rows,
        )

    def binary_file_reader(
        self, file_path: str, max_elems: int = 1_000_000, offset_bytes: int = 0
    ) -> np.ndarray:
        with open(file_path, "rb") as f:
            if offset_bytes > 0:
                f.seek(offset_bytes)
            raw = f.read(max_elems * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def file_writer_txt(
        self, file_path: str, data: np.ndarray, overwrite: bool = False
    ) -> int:
        from pathlib import Path

        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        path = Path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output path exists: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for v in data:
                f.write(f"{float(v)}\n")
        return int(data.size)
