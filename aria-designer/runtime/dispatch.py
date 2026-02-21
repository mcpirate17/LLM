import numpy as np
import torch
from pathlib import Path
from .bindings import aria_lib, ffi

class KernelDispatcher:
    """Dispatches calls to the best available implementation for each component."""

    def __init__(self, use_native=True):
        self.lib = aria_lib
        self.use_native = use_native

    def validate_graph(self, nodes, edges):
        """
        Validate graph using the C validator.
        nodes: list of node IDs (indices used internally)
        edges: list of (src_node_idx, tgt_node_idx, src_port, tgt_port)
        """
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

    def infer_shapes(self, topo_order, edges, node_rules):
        """
        Propagate shapes through the graph.
        node_rules: list of {rule: ShapeRule, n_inputs, n_outputs, input_shapes: [...], ...}
        """
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

            # Set initial input shapes (for source nodes)
            if 'input_shapes' in rule_data:
                for p_idx, shape in enumerate(rule_data['input_shapes']):
                    if shape:
                        node.input_shapes[p_idx].shape.ndim = len(shape)
                        for d_idx, dim in enumerate(shape):
                            node.input_shapes[p_idx].shape.dims[d_idx] = dim
                        node.input_shapes[p_idx].shape.valid = 1

        # Prepare edges for C call: int32_t edges[][4]
        c_edges = ffi.new("int32_t[][4]", len(edges))
        for i, e in enumerate(edges):
            c_edges[i][0] = e[0] # src
            c_edges[i][1] = e[1] # tgt
            c_edges[i][2] = e[2] # src_port
            c_edges[i][3] = e[3] # tgt_port

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

    def relu(self, x: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_relu_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.empty_like(x)
            self.lib.aria_relu_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                x.size
            )
            return y
        else:
            # Python fallback
            return torch.relu(torch.from_numpy(x)).numpy()

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_matmul_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            assert a.ndim == 2 and b.ndim == 2
            assert a.shape[1] == b.shape[0]
            m, k = a.shape
            _, n = b.shape
            c = np.empty((m, n), dtype=np.float32)
            self.lib.aria_matmul_f32(
                ffi.from_buffer("float *", a),
                ffi.from_buffer("float *", b),
                ffi.from_buffer("float *", c),
                m, k, n
            )
            return c
        else:
            # Python fallback
            return (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()

    def tropical_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_tropical_add_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            y = np.empty_like(a)
            self.lib.aria_tropical_add_f32(
                ffi.from_buffer("float *", a),
                ffi.from_buffer("float *", b),
                ffi.from_buffer("float *", y),
                a.size
            )
            return y
        return np.minimum(a, b)

    def tropical_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_tropical_matmul_f32"):
            a = np.ascontiguousarray(a, dtype=np.float32)
            b = np.ascontiguousarray(b, dtype=np.float32)
            assert a.ndim == 2 and b.ndim == 2
            assert a.shape[1] == b.shape[0]
            m, k = a.shape
            _, n = b.shape
            c = np.empty((m, n), dtype=np.float32)
            self.lib.aria_tropical_matmul_f32(
                ffi.from_buffer("float *", a),
                ffi.from_buffer("float *", b),
                ffi.from_buffer("float *", c),
                m, k, n
            )
            return c
        # Python fallback: min-plus matmul
        m, k = a.shape
        _, n = b.shape
        out = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                out[i, j] = np.min(a[i, :] + b[:, j])
        return out

    def tropical_center(self, x: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_tropical_center_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_center_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                b, s, d
            )
            return y
        baseline = np.min(x, axis=1, keepdims=True)
        return x - baseline

    def hyp_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_hyp_distance_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.ascontiguousarray(y, dtype=np.float32)
            assert x.ndim == 3 and y.ndim == 3
            assert x.shape == y.shape
            b, s, d = x.shape
            out = np.empty((b, s), dtype=np.float32)
            self.lib.aria_hyp_distance_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                ffi.from_buffer("float *", out),
                b, s, d
            )
            return out
        # Python fallback: simple Euclidean norm
        return np.linalg.norm(x - y, axis=-1)

    def padic_gate(self, x: np.ndarray, p: float = 2.0) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_padic_gate_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            y = np.empty_like(x)
            self.lib.aria_padic_gate_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                x.size,
                float(p)
            )
            return y
        abs_x = np.maximum(np.abs(x), 1e-10)
        valuation = -(np.log(abs_x) / np.log(p))
        valuation = np.clip(valuation, -10.0, 10.0)
        gate = 1.0 / (1.0 + np.exp(-valuation))
        return x * gate

    def tropical_attention(self, x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        if self.use_native and hasattr(self.lib, "aria_tropical_attention_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_attention_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                b, s, d,
                float(temperature)
            )
            return y
        # Python fallback: tropical attention using softmin distances
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
        if self.use_native and hasattr(self.lib, "aria_tropical_gate_f32"):
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert x.ndim == 3
            b, s, d = x.shape
            y = np.empty_like(x)
            self.lib.aria_tropical_gate_f32(
                ffi.from_buffer("float *", x),
                ffi.from_buffer("float *", y),
                b, s, d,
                float(temperature)
            )
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
        if self.use_native and hasattr(self.lib, "aria_file_loader_csv_f32"):
            out = np.zeros((max_rows, max_cols), dtype=np.float32)
            rows = self.lib.aria_file_loader_csv_f32(
                file_path.encode("utf-8"),
                ffi.from_buffer("float *", out),
                int(max_rows),
                int(max_cols),
                (delimiter or ",")[0].encode("utf-8"),
                1 if has_header else 0,
            )
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
        if self.use_native and hasattr(self.lib, "aria_binary_file_reader_f32"):
            out = np.zeros((max_elems,), dtype=np.float32)
            n = self.lib.aria_binary_file_reader_f32(
                file_path.encode("utf-8"),
                ffi.from_buffer("float *", out),
                int(max_elems),
                int(offset_bytes),
            )
            if n < 0:
                raise ValueError(f"Native binary_file_reader failed: {n}")
            return out[:n]

        with open(file_path, "rb") as f:
            if offset_bytes > 0:
                f.seek(offset_bytes)
            raw = f.read(max_elems * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def file_writer_txt(self, file_path: str, data: np.ndarray, overwrite: bool = False) -> int:
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        if self.use_native and hasattr(self.lib, "aria_file_writer_txt_f32"):
            rc = self.lib.aria_file_writer_txt_f32(
                file_path.encode("utf-8"),
                ffi.from_buffer("float *", data),
                int(data.size),
                1 if overwrite else 0,
            )
            if rc < 0:
                raise ValueError(f"Native file_writer_txt failed: {rc}")
            return int(rc)

        path = Path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output path exists: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for v in data:
                f.write(f"{float(v)}\n")
        return int(data.size)
