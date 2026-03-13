"""Kernel dispatcher — routes calls to aria_core (preferred) or legacy CFFI.

Phase 4 integration (DRY_HIGH_PERF_TODO.md): replaces the fragmented CFFI-based
dispatch with the unified aria_core pybind11 library. Falls back to the legacy
CFFI bindings (bindings.py → libaria_runtime.so) when aria_core is unavailable.
"""
import numpy as np
import torch
from typing import Optional

# Try aria_core first (unified pybind11 backend)
try:
    import aria_core
    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False

# Legacy CFFI fallback
if not _HAS_ARIA_CORE:
    try:
        from .bindings import aria_lib, ffi
        _HAS_CFFI = True
    except Exception:
        aria_lib = None
        ffi = None
        _HAS_CFFI = False
else:
    aria_lib = None
    ffi = None
    _HAS_CFFI = False


def _np_to_torch(x):
    """Convert numpy array to contiguous float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _torch_to_np(t):
    """Convert torch tensor to numpy."""
    return t.numpy()


class KernelDispatcher:
    """Dispatches calls to the best available implementation for each component."""
    __slots__ = ("use_native", "lib")

    def __init__(self, use_native=True):
        self.use_native = use_native
        # Legacy fields for backward compat
        self.lib = aria_lib

    # ── Graph validation ──────────────────────────────────────────────

    def validate_graph(self, nodes, edges):
        """
        Validate graph using the C validator.
        nodes: list of node IDs (indices used internally)
        edges: list of (src_node_idx, tgt_node_idx, src_port, tgt_port)
        """
        if _HAS_ARIA_CORE:
            edge_list = [[int(s), int(t), int(sp), int(tp)] for s, t, sp, tp in edges]
            return aria_core.validate_graph(len(nodes), edge_list)

        if _HAS_CFFI:
            return self._validate_graph_cffi(nodes, edges)

        # Pure Python fallback
        return self._validate_graph_python(nodes, edges)

    def _validate_graph_cffi(self, nodes, edges):
        """Legacy CFFI graph validation."""
        graph = ffi.new("AriaGraph *")
        graph.n_nodes = len(nodes)
        graph.n_edges = len(edges)
        for i, (s, t, sp, tp) in enumerate(edges):
            graph.edges[i].source = s
            graph.edges[i].target = t
            graph.edges[i].src_port = sp
            graph.edges[i].tgt_port = tp
        result = ffi.new("AriaValidationResult *")
        rc = self.lib.aria_validate_graph(graph, result)
        if rc == 0:
            return {
                "valid": True,
                "topo_order": [result.topo_order[i] for i in range(result.topo_len)],
                "in_degrees": [result.in_degree[i] for i in range(graph.n_nodes)],
                "out_degrees": [result.out_degree[i] for i in range(graph.n_nodes)],
            }
        else:
            return {
                "valid": False,
                "error": ffi.string(result.error).decode("utf-8"),
                "code": rc
            }

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
        queue = [i for i in range(n) if in_deg[i] == 0]
        topo = []
        while queue:
            node = queue.pop(0)
            topo.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        if len(topo) != n:
            return {"valid": False, "error": "Cycle detected", "code": -3}
        return {"valid": True, "topo_order": topo, "in_degrees": [0]*n, "out_degrees": out_deg}

    # ── Shape inference ───────────────────────────────────────────────

    def infer_shapes(self, topo_order, edges, node_rules):
        """
        Propagate shapes through the graph.
        node_rules: list of {rule: ShapeRule, n_inputs, n_outputs, input_shapes: [...], ...}
        """
        if _HAS_ARIA_CORE:
            edge_list = [[int(e[0]), int(e[1]), int(e[2]), int(e[3])] for e in edges]
            rules_dicts = []
            for rd in node_rules:
                d = {"rule": int(rd["rule"]),
                     "n_inputs": rd.get("n_inputs", 1),
                     "n_outputs": rd.get("n_outputs", 1),
                     "split_n": rd.get("split_n", 0),
                     "out_dim": rd.get("out_dim", -1),
                     "orig_seq_len": rd.get("orig_seq_len", 0)}
                if "input_shapes" in rd:
                    d["input_shapes"] = rd["input_shapes"]
                rules_dicts.append(d)
            return aria_core.propagate_shapes(
                [int(x) for x in topo_order], edge_list, rules_dicts)

        if _HAS_CFFI:
            return self._infer_shapes_cffi(topo_order, edges, node_rules)

        return {"valid": False, "error": "No shape inference backend available"}

    def _infer_shapes_cffi(self, topo_order, edges, node_rules):
        """Legacy CFFI shape inference."""
        res = ffi.new("ShapeInferenceResult *")
        res.n_nodes = len(node_rules)
        for i, rule_data in enumerate(node_rules):
            node = res.nodes[i]
            node.rule = rule_data['rule']
            node.n_inputs = rule_data.get('n_inputs', 1)
            node.n_outputs = rule_data.get('n_outputs', 1)
            node.split_n = rule_data.get('split_n', 0)
            node.out_dim = rule_data.get('out_dim', -1)
            node.orig_seq_len = rule_data.get('orig_seq_len', 0)
            if 'input_shapes' in rule_data:
                for p_idx, shape in enumerate(rule_data['input_shapes']):
                    if shape:
                        node.input_shapes[p_idx].shape.ndim = len(shape)
                        for d_idx, dim in enumerate(shape):
                            node.input_shapes[p_idx].shape.dims[d_idx] = dim
                        node.input_shapes[p_idx].shape.valid = 1
        c_edges = ffi.new("int32_t[][4]", len(edges))
        for i, e in enumerate(edges):
            c_edges[i][0] = e[0]; c_edges[i][1] = e[1]
            c_edges[i][2] = e[2]; c_edges[i][3] = e[3]
        c_topo = ffi.new("int32_t[]", topo_order)
        rc = self.lib.aria_propagate_shapes(res, c_topo, len(topo_order), c_edges, len(edges))
        if rc == 0:
            output_shapes = []
            for i in range(res.n_nodes):
                node_out = []
                for p in range(res.nodes[i].n_outputs):
                    shape = res.nodes[i].output_shapes[p].shape
                    if shape.valid:
                        node_out.append([shape.dims[d] for d in range(shape.ndim)])
                    else:
                        node_out.append(None)
                output_shapes.append(node_out)
            return {"valid": True, "output_shapes": output_shapes}
        else:
            return {"valid": False, "error": ffi.string(res.error).decode("utf-8")}

    # ── Kernel dispatch methods ───────────────────────────────────────
    # Each method tries: aria_core → legacy CFFI → Python fallback

    def relu(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.relu_f32(_np_to_torch(x)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_relu_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.empty_like(x)
            self.lib.aria_relu_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), x.size)
            return y
        return torch.relu(torch.from_numpy(x)).numpy()

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.matmul_f32(_np_to_torch(a), _np_to_torch(b)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_matmul_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
            m, k = a.shape; _, n = b.shape
            c = np.empty((m, n), dtype=np.float32)
            self.lib.aria_matmul_f32(ffi.from_buffer("float *", a), ffi.from_buffer("float *", b), ffi.from_buffer("float *", c), m, k, n)
            return c
        return (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()

    def tropical_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_add_f32(_np_to_torch(a), _np_to_torch(b)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_tropical_add_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            y = np.empty_like(a)
            self.lib.aria_tropical_add_f32(ffi.from_buffer("float *", a), ffi.from_buffer("float *", b), ffi.from_buffer("float *", y), a.size)
            return y
        return np.minimum(a, b)

    def tropical_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_matmul_f32(_np_to_torch(a), _np_to_torch(b)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_tropical_matmul_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
            m, k = a.shape; _, n = b.shape
            c = np.empty((m, n), dtype=np.float32)
            self.lib.aria_tropical_matmul_f32(ffi.from_buffer("float *", a), ffi.from_buffer("float *", b), ffi.from_buffer("float *", c), m, k, n)
            return c
        m, k = a.shape; _, n = b.shape
        out = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                out[i, j] = np.min(a[i, :] + b[:, j])
        return out

    def tropical_center(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_center_f32(_np_to_torch(x)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_tropical_center_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_center_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), b, s, d)
            return y
        baseline = np.min(x, axis=1, keepdims=True)
        return x - baseline

    def hyp_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.hyp_distance_f32(_np_to_torch(x), _np_to_torch(y)))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_hyp_distance_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.ascontiguousarray(y, dtype=np.float32)
            assert x.ndim == 3 and y.ndim == 3 and x.shape == y.shape
            b, s, d = x.shape
            out = np.empty((b, s), dtype=np.float32)
            self.lib.aria_hyp_distance_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), ffi.from_buffer("float *", out), b, s, d)
            return out
        return np.linalg.norm(x - y, axis=-1)

    # ── Reference Architecture Ops ──

    def embedding_lookup(self, table: np.ndarray, idx: np.ndarray, pe: Optional[np.ndarray] = None) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            pe_t = _np_to_torch(pe) if pe is not None else None
            return _torch_to_np(aria_core.embedding_lookup_f32(_np_to_torch(table), torch.from_numpy(idx.astype(np.int32)), pe_t))
        # Fallback to PyTorch
        t = torch.from_numpy(table)
        i = torch.from_numpy(idx.astype(np.int64))
        res = t[i]
        if pe is not None:
            res += torch.from_numpy(pe)
        return res.numpy()

    def rope_rotate(self, x: np.ndarray, theta_base: float = 10000.0) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.rope_rotate_f32(_np_to_torch(x), float(theta_base)))
        # Simplified Python fallback (not a full RoPE, just for interface parity)
        return x

    def gated_linear(self, x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray], wg: np.ndarray, bg: Optional[np.ndarray]) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            bt = _np_to_torch(b) if b is not None else None
            bgt = _np_to_torch(bg) if bg is not None else None
            return _torch_to_np(aria_core.gated_linear_f32(_np_to_torch(x), _np_to_torch(w), bt, _np_to_torch(wg), bgt))
        # PyTorch fallback
        xt = torch.from_numpy(x)
        res = torch.nn.functional.linear(xt, torch.from_numpy(w), torch.from_numpy(b) if b is not None else None)
        gate = torch.nn.functional.linear(xt, torch.from_numpy(wg), torch.from_numpy(bg) if bg is not None else None)
        return (res * torch.sigmoid(gate)).numpy()

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.cosine_similarity_f32(_np_to_torch(a), _np_to_torch(b)))
        at = torch.from_numpy(a)
        bt = torch.from_numpy(b)
        return torch.nn.functional.cosine_similarity(at, bt, dim=-1).numpy()

    def rwkv_time_mixing(self, x: np.ndarray, decay: np.ndarray, bonus: np.ndarray, wk: np.ndarray, wv: np.ndarray, wr: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.rwkv_time_mixing_f32(_np_to_torch(x), _np_to_torch(decay), _np_to_torch(bonus), _np_to_torch(wk), _np_to_torch(wv), _np_to_torch(wr)))
        return x # Placeholder

    def causal_mask(self, x: np.ndarray) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.causal_mask_f32(_np_to_torch(x)))
        xt = torch.from_numpy(x)
        mask = torch.triu(torch.ones(xt.size(-2), xt.size(-1)), diagonal=1).bool()
        xt.masked_fill_(mask, float('-inf'))
        return xt.numpy()

    def padic_gate(self, x: np.ndarray, p: float = 2.0) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.padic_gate_f32(_np_to_torch(x), p))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_padic_gate_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.empty_like(x)
            self.lib.aria_padic_gate_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), x.size, float(p))
            return y
        abs_x = np.maximum(np.abs(x), 1e-10)
        valuation = -(np.log(abs_x) / np.log(p))
        valuation = np.clip(valuation, -10.0, 10.0)
        gate = 1.0 / (1.0 + np.exp(-valuation))
        return x * gate

    def tropical_attention(self, x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_attention_f32(_np_to_torch(x), temperature))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_tropical_attention_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_attention_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), b, s, d, float(temperature))
            return y
        b, s, d = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(b):
            dist = np.empty((s, s), dtype=np.float32)
            for r in range(s):
                for c in range(s):
                    dist[r, c] = np.min(x[i, r, :] + x[i, c, :])
            weights = np.exp(-dist / temperature)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            out[i] = weights @ x[i]
        return out

    def tropical_gate(self, x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        if _HAS_ARIA_CORE and self.use_native:
            return _torch_to_np(aria_core.tropical_gate_f32(_np_to_torch(x), temperature))
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_tropical_gate_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_gate_f32(ffi.from_buffer("float *", x), ffi.from_buffer("float *", y), b, s, d, float(temperature))
            return y
        b, s, d = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(b):
            dist = np.empty((s, s), dtype=np.float32)
            for r in range(s):
                for c in range(s):
                    dist[r, c] = np.min(x[i, r, :] + x[i, c, :])
            weights = np.exp(-dist / temperature)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            gated = weights @ x[i]
            out[i] = x[i] * (1.0 / (1.0 + np.exp(-gated)))
        return out

    def file_loader_csv(self, file_path: str, max_rows: int = 4096, max_cols: int = 1024,
                        delimiter: str = ",", has_header: bool = True) -> np.ndarray:
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_file_loader_csv_f32"):
            out = np.zeros((max_rows, max_cols), dtype=np.float32)
            rows = self.lib.aria_file_loader_csv_f32(
                file_path.encode("utf-8"), ffi.from_buffer("float *", out),
                int(max_rows), int(max_cols),
                (delimiter or ",")[0].encode("utf-8"), 1 if has_header else 0)
            if rows < 0:
                raise ValueError(f"Native file_loader_csv failed: {rows}")
            return out[:rows, :]
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if has_header and i == 0:
                    continue
                parts = [p.strip() for p in line.strip().split(delimiter) if p.strip() != ""]
                if not parts:
                    continue
                rows.append([float(v) for v in parts])
        return np.array(rows, dtype=np.float32)

    def binary_file_reader(self, file_path: str, max_elems: int = 1_000_000,
                           offset_bytes: int = 0) -> np.ndarray:
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_binary_file_reader_f32"):
            out = np.zeros((max_elems,), dtype=np.float32)
            n = self.lib.aria_binary_file_reader_f32(
                file_path.encode("utf-8"), ffi.from_buffer("float *", out),
                int(max_elems), int(offset_bytes))
            if n < 0:
                raise ValueError(f"Native binary_file_reader failed: {n}")
            return out[:n]
        from pathlib import Path
        with open(file_path, "rb") as f:
            if offset_bytes > 0:
                f.seek(offset_bytes)
            raw = f.read(max_elems * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def file_writer_txt(self, file_path: str, data: np.ndarray, overwrite: bool = False) -> int:
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        if _HAS_CFFI and self.use_native and hasattr(self.lib, "aria_file_writer_txt_f32"):
            rc = self.lib.aria_file_writer_txt_f32(
                file_path.encode("utf-8"), ffi.from_buffer("float *", data),
                int(data.size), 1 if overwrite else 0)
            if rc < 0:
                raise ValueError(f"Native file_writer_txt failed: {rc}")
            return int(rc)
        from pathlib import Path
        path = Path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output path exists: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for v in data:
                f.write(f"{float(v)}\n")
        return int(data.size)
