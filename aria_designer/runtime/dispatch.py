"""Kernel dispatcher — routes calls to aria_core (pybind11) or Python/torch fallback."""

from collections import deque

import numpy as np
import torch

from research.defaults import ROPE_THETA_BASE

try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import aria_core

    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False


# ── RWKV scan: Numba-accelerated with pure-numpy fallback ────────────


def _rwkv_scan_numpy(k, v, r, decay, bonus):
    """Pure numpy fallback for RWKV sequential scan."""
    b, seq_len, d = k.shape
    out = np.empty_like(k, dtype=np.float32)
    decay_exp = np.exp(decay)  # precompute once
    state = np.zeros((b, d), dtype=np.float32)
    for t in range(seq_len):
        kt = k[:, t, :]
        vt = v[:, t, :]
        ekt = np.exp(kt)
        out[:, t, :] = r[:, t, :] * (state + np.exp(bonus + kt) * vt)
        state = decay_exp * state + ekt * vt
    return out


if _HAS_NUMBA:

    @_njit(cache=True)
    def _rwkv_scan_numba(k, v, r, decay, bonus):
        b, seq_len, d = k.shape
        out = np.empty((b, seq_len, d), dtype=np.float32)
        for bi in range(b):
            for di in range(d):
                state = np.float32(0.0)
                decay_exp = np.exp(decay[di])
                for t in range(seq_len):
                    kt = k[bi, t, di]
                    vt = v[bi, t, di]
                    ekt = np.exp(kt)
                    bonus_term = np.exp(bonus[di] + kt) * vt
                    out[bi, t, di] = r[bi, t, di] * (state + bonus_term)
                    state = decay_exp * state + ekt * vt
        return out

    def _rwkv_scan(k, v, r, decay, bonus):
        return _rwkv_scan_numba(
            np.ascontiguousarray(k, dtype=np.float32),
            np.ascontiguousarray(v, dtype=np.float32),
            np.ascontiguousarray(r, dtype=np.float32),
            np.ascontiguousarray(decay, dtype=np.float32),
            np.ascontiguousarray(bonus, dtype=np.float32),
        )
else:
    _rwkv_scan = _rwkv_scan_numpy


def _np_to_torch(x):
    """Convert numpy array to contiguous float32 torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.float().contiguous()
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _to_output(t, *, like):
    """Return tensor in the same format as the input (numpy or torch)."""
    if isinstance(like, torch.Tensor):
        return t if isinstance(t, torch.Tensor) else torch.from_numpy(t)
    return t.numpy() if isinstance(t, torch.Tensor) else t


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

    def relu(self, x):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(aria_core.relu_f32(_np_to_torch(x)), like=x)
        return _to_output(torch.relu(_np_to_torch(x)), like=x)

    def matmul(self, a, b):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.matmul_f32(_np_to_torch(a), _np_to_torch(b)), like=a
            )
        return _to_output(_np_to_torch(a) @ _np_to_torch(b), like=a)

    def tropical_add(self, a, b):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.tropical_add_f32(_np_to_torch(a), _np_to_torch(b)), like=a
            )
        if isinstance(a, torch.Tensor):
            return torch.minimum(
                a, b if isinstance(b, torch.Tensor) else _np_to_torch(b)
            )
        return np.minimum(a, b)

    def tropical_matmul(self, a, b):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.tropical_matmul_f32(_np_to_torch(a), _np_to_torch(b)), like=a
            )
        if isinstance(a, torch.Tensor):
            bt = b if isinstance(b, torch.Tensor) else _np_to_torch(b)
            return torch.min(a[:, :, None] + bt[None, :, :], dim=1).values
        return np.min(a[:, :, None] + b[None, :, :], axis=1).astype(np.float32)

    def tropical_center(self, x):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(aria_core.tropical_center_f32(_np_to_torch(x)), like=x)
        if isinstance(x, torch.Tensor):
            return x - torch.min(x, dim=1, keepdim=True).values
        baseline = np.min(x, axis=1, keepdims=True)
        return x - baseline

    def hyp_distance(self, x, y):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.hyp_distance_f32(_np_to_torch(x), _np_to_torch(y)), like=x
            )
        if isinstance(x, torch.Tensor):
            yt = y if isinstance(y, torch.Tensor) else _np_to_torch(y)
            return torch.linalg.norm(x - yt, dim=-1)
        return np.linalg.norm(x - y, axis=-1)

    # ── Reference Architecture Ops ──

    def embedding_lookup(self, table, idx, pe=None):
        if _HAS_ARIA_CORE and self.use_native:
            pe_t = _np_to_torch(pe) if pe is not None else None
            idx_t = (
                idx
                if isinstance(idx, torch.Tensor)
                else torch.from_numpy(idx.astype(np.int32))
            )
            return _to_output(
                aria_core.embedding_lookup_f32(_np_to_torch(table), idx_t.int(), pe_t),
                like=table,
            )
        t = _np_to_torch(table)
        i = (
            idx
            if isinstance(idx, torch.Tensor)
            else torch.from_numpy(idx.astype(np.int64))
        )
        i = i.long()
        res = t[i]
        if pe is not None:
            res = res + _np_to_torch(pe)
        return _to_output(res, like=table)

    def rope_rotate(self, x, theta_base: float = ROPE_THETA_BASE):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.rope_rotate_f32(_np_to_torch(x), float(theta_base)), like=x
            )
        # RoPE: rotate pairs of dimensions by position-dependent angles
        # x shape: (..., seq_len, d) where d is even
        xt = _np_to_torch(x)
        *batch, seq_len, d = xt.shape
        half_d = d // 2
        positions = torch.arange(seq_len, dtype=torch.float32)
        dim_indices = torch.arange(half_d, dtype=torch.float32)
        freqs = positions[:, None] / (theta_base ** (dim_indices[None, :] / half_d))
        cos_f = torch.cos(freqs)
        sin_f = torch.sin(freqs)
        x1 = xt[..., :half_d]
        x2 = xt[..., half_d:]
        result = torch.cat([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1)
        return _to_output(result, like=x)

    def gated_linear(self, x, w, b, wg, bg):
        if _HAS_ARIA_CORE and self.use_native:
            bt = _np_to_torch(b) if b is not None else None
            bgt = _np_to_torch(bg) if bg is not None else None
            return _to_output(
                aria_core.gated_linear_f32(
                    _np_to_torch(x), _np_to_torch(w), bt, _np_to_torch(wg), bgt
                ),
                like=x,
            )
        xt = _np_to_torch(x)
        wt = _np_to_torch(w)
        wgt = _np_to_torch(wg)
        bt = _np_to_torch(b) if b is not None else None
        bgt = _np_to_torch(bg) if bg is not None else None
        res = torch.nn.functional.linear(xt, wt, bt)
        gate = torch.nn.functional.linear(xt, wgt, bgt)
        return _to_output(res * torch.sigmoid(gate), like=x)

    def cosine_similarity(self, a, b):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.cosine_similarity_f32(_np_to_torch(a), _np_to_torch(b)),
                like=a,
            )
        at = _np_to_torch(a)
        bt = _np_to_torch(b)
        return _to_output(torch.nn.functional.cosine_similarity(at, bt, dim=-1), like=a)

    def rwkv_time_mixing(self, x, decay, bonus, wk, wv, wr):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.rwkv_time_mixing_f32(
                    _np_to_torch(x),
                    _np_to_torch(decay),
                    _np_to_torch(bonus),
                    _np_to_torch(wk),
                    _np_to_torch(wv),
                    _np_to_torch(wr),
                ),
                like=x,
            )
        # RWKV time-mixing: linear attention with exponential decay
        # x: (batch, seq_len, d), decay/bonus: (d,), wk/wv/wr: (d, d)
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        decay_np = decay.numpy() if isinstance(decay, torch.Tensor) else decay
        bonus_np = bonus.numpy() if isinstance(bonus, torch.Tensor) else bonus
        wk_np = wk.numpy() if isinstance(wk, torch.Tensor) else wk
        wv_np = wv.numpy() if isinstance(wv, torch.Tensor) else wv
        wr_np = wr.numpy() if isinstance(wr, torch.Tensor) else wr

        k = x_np @ wk_np
        v = x_np @ wv_np
        r = 1.0 / (1.0 + np.exp(-(x_np @ wr_np)))

        out = _rwkv_scan(k, v, r, decay_np, bonus_np)
        return _to_output(torch.from_numpy(out), like=x)

    def causal_mask(self, x):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(aria_core.causal_mask_f32(_np_to_torch(x)), like=x)
        xt = _np_to_torch(x).clone()
        mask = torch.triu(torch.ones(xt.size(-2), xt.size(-1)), diagonal=1).bool()
        xt.masked_fill_(mask, float("-inf"))
        return _to_output(xt, like=x)

    def padic_gate(self, x, p: float = 2.0):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(aria_core.padic_gate_f32(_np_to_torch(x), p), like=x)
        if isinstance(x, torch.Tensor):
            abs_x = torch.clamp(x.abs(), min=1e-10)
            valuation = -(torch.log(abs_x) / np.log(p))
            valuation = torch.clamp(valuation, -10.0, 10.0)
            return x * torch.sigmoid(valuation)
        abs_x = np.maximum(np.abs(x), 1e-10)
        valuation = -(np.log(abs_x) / np.log(p))
        valuation = np.clip(valuation, -10.0, 10.0)
        gate = 1.0 / (1.0 + np.exp(-valuation))
        return x * gate

    @staticmethod
    def _tropical_distance_weights(x, temperature: float):
        """Tropical distance matrix + softmax weights. x: (b, s, d)."""
        if isinstance(x, torch.Tensor):
            dist = torch.min(x[:, :, None, :] + x[:, None, :, :], dim=-1).values
            weights = torch.exp(-dist / temperature)
            return weights / weights.sum(dim=-1, keepdim=True)
        dist = np.min(x[:, :, None, :] + x[:, None, :, :], axis=-1)
        weights = np.exp(-dist / temperature)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return weights

    def tropical_attention(self, x, temperature: float = 0.1):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.tropical_attention_f32(_np_to_torch(x), temperature), like=x
            )
        weights = self._tropical_distance_weights(x, temperature)
        if isinstance(x, torch.Tensor):
            return torch.einsum("brc,bcd->brd", weights, x)
        return np.einsum("brc,bcd->brd", weights, x).astype(np.float32)

    def tropical_gate(self, x, temperature: float = 0.1):
        if _HAS_ARIA_CORE and self.use_native:
            return _to_output(
                aria_core.tropical_gate_f32(_np_to_torch(x), temperature), like=x
            )
        weights = self._tropical_distance_weights(x, temperature)
        if isinstance(x, torch.Tensor):
            gated = torch.einsum("brc,bcd->brd", weights, x)
            return x * torch.sigmoid(gated)
        gated = np.einsum("brc,bcd->brd", weights, x)
        return (x * (1.0 / (1.0 + np.exp(-gated)))).astype(np.float32)

    def file_loader_csv(
        self,
        file_path: str,
        _max_rows: int = 4096,
        _max_cols: int = 1024,
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
        np.savetxt(file_path, data, fmt="%.6g")
        return int(data.size)
