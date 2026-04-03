from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from research.defaults import MODEL_DIM, VALIDATION_SEQ_LEN, VOCAB_SIZE

from .graph import ComputationGraph, ShapeInfo
from .compiled_op import CompiledOp
from .compiler_constants import MATHSPACE_OPS


class CompiledLayer(nn.Module):
    """A compiled computation graph as a PyTorch module with memory management."""

    def __init__(self, graph: ComputationGraph):
        super().__init__()
        self.graph = graph
        self.topo_order = graph.topological_order()

        counts_dict: Dict[int, int] = {}
        for nid in self.topo_order:
            node = graph.nodes[nid]
            for iid in node.input_ids:
                counts_dict[iid] = counts_dict.get(iid, 0) + 1
        max_nid = max(counts_dict.keys()) + 1 if counts_dict else 0
        self._counts_size = max_nid
        self._counts_original = [0] * max_nid
        for nid, cnt in counts_dict.items():
            self._counts_original[nid] = cnt
        self._counts_buf = list(self._counts_original)

        self.ops = nn.ModuleDict()
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input:
                continue
            input_shapes = [graph.nodes[iid].output_shape for iid in node.input_ids]
            self.ops[str(nid)] = CompiledOp(
                node.op_name,
                node.config,
                input_shapes[0] if input_shapes else ShapeInfo(),
                node.output_shape,
                graph.model_dim,
            )

        self._mathspace_boundary_nids: set = set()
        consumers_of: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
        for nid in self.topo_order:
            for iid in graph.nodes[nid].input_ids:
                consumers_of[iid].append(nid)
        output_id = graph._output_node_id
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input or node.op_name not in MATHSPACE_OPS:
                continue
            is_boundary = False
            node_consumers = consumers_of.get(nid, [])
            if not node_consumers and nid == output_id:
                is_boundary = True
            else:
                for cid in node_consumers:
                    if graph.nodes[cid].op_name not in MATHSPACE_OPS:
                        is_boundary = True
                        break
            if is_boundary:
                self._mathspace_boundary_nids.add(str(nid))

        self._fwd_plan: list = []
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input:
                self._fwd_plan.append((nid, True, None, node.input_ids, False))
            else:
                nid_str = str(nid)
                op = self.ops[nid_str]
                is_boundary = nid_str in self._mathspace_boundary_nids
                self._fwd_plan.append((nid, False, op, node.input_ids, is_boundary))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dispatcher = getattr(self, "_subgraph_dispatcher", None)
        if dispatcher is not None:
            result = dispatcher.try_dispatch(x)
            if result is not None:
                return result

        node_outputs: Dict[int, torch.Tensor] = {}
        counts = self._counts_buf
        counts[:] = self._counts_original
        output_id = self.graph._output_node_id
        if output_id is None:
            raise RuntimeError("Graph has no output node")

        is_cuda = x.is_cuda
        for nid, is_input, op, input_ids, is_boundary in self._fwd_plan:
            if is_input:
                node_outputs[nid] = x
            else:
                inputs = tuple(node_outputs[iid] for iid in input_ids)
                out = op(*inputs)
                if is_boundary:
                    out_f = out if out.dtype == torch.float32 else out.float()
                    rms = out_f.pow(2).mean(dim=-1, keepdim=True).add_(1e-6).rsqrt_()
                    out = (
                        out * rms
                        if out.dtype == torch.float32
                        else out * rms.to(out.dtype)
                    )
                node_outputs[nid] = out

            for iid in input_ids:
                counts[iid] -= 1
                if counts[iid] <= 0 and iid != output_id and iid in node_outputs:
                    out_to_del = node_outputs.pop(iid)
                    if is_cuda:
                        del out_to_del

        out = node_outputs.pop(output_id)
        node_outputs.clear()
        return out

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        for op in self.ops.values():
            op._capture_heatmap = enabled


class SynthesizedModel(nn.Module):
    """A complete language model built from synthesized layers."""

    def __init__(
        self,
        layer_graphs: List[ComputationGraph],
        vocab_size: int = VOCAB_SIZE,
        model_dim: int = MODEL_DIM,
        max_seq_len: int = VALIDATION_SEQ_LEN,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=model_dim**-0.5)
        self.layers = nn.ModuleList([CompiledLayer(g) for g in layer_graphs])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._layer_graphs = layer_graphs
        self.layer_needs_residual = [not g.has_residual_path() for g in layer_graphs]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            if self.layer_needs_residual[i]:
                out = layer(x)
                x = x + out if out.shape == x.shape else out
            else:
                x = layer(x)
        return self.lm_head(self.norm(x))

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        for layer in self.layers:
            if hasattr(layer, "set_capture_heatmap"):
                layer.set_capture_heatmap(enabled)

    @property
    def has_mathspace_ops(self) -> bool:
        return any(layer._mathspace_boundary_nids for layer in self.layers)

    @property
    def recommended_grad_clip(self) -> float:
        return 5.0 if self.has_mathspace_ops else 1.0

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        desc = [
            f"SynthesizedModel(dim={self.model_dim}, layers={len(self.layers)}, params={self.param_count():,})"
        ]
        for i, g in enumerate(self._layer_graphs):
            desc.append(
                f"\n  Layer {i}:\n"
                + "\n".join(f"    {l}" for l in g.describe().split("\n"))
            )
        return "\n".join(desc)
