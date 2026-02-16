"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import get_primitive, PrimitiveOp, OpCategory
from .graph import ComputationGraph, OpNode, ShapeInfo


class CompiledOp(nn.Module):
    """A single compiled primitive operation."""

    def __init__(self, op_name: str, config: Dict, input_shape: ShapeInfo,
                 output_shape: ShapeInfo, model_dim: int):
        super().__init__()
        self.op_name = op_name
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_dim = model_dim

        # Allocate learnable parameters
        op = get_primitive(op_name)
        if op.has_params:
            self._init_params(op, config, input_shape)

    def _init_params(self, op: PrimitiveOp, config: Dict, input_shape: ShapeInfo):
        """Initialize learnable parameters for this op."""
        D_in = input_shape.dim
        D_out = config.get("out_dim", D_in)

        if op.name in ("linear_proj", "linear_proj_down", "linear_proj_up"):
            self.weight = nn.Parameter(torch.randn(D_out, D_in) * (1.0 / math.sqrt(D_in)))
        elif op.name == "learnable_scale":
            self.scale = nn.Parameter(torch.ones(D_in))
        elif op.name == "learnable_bias":
            self.bias = nn.Parameter(torch.zeros(D_in))
        elif op.name == "selective_scan":
            # A_log, dt_proj, B_proj, C_proj — each D params
            self.A_log = nn.Parameter(torch.randn(D_in) * 0.1)
            self.dt_proj = nn.Parameter(torch.randn(D_in) * 0.1)
            self.B_proj = nn.Parameter(torch.randn(D_in) * (1.0 / math.sqrt(D_in)))
            self.C_proj = nn.Parameter(torch.randn(D_in) * (1.0 / math.sqrt(D_in)))
        elif op.name == "conv1d_seq":
            # One 3-element kernel per channel (depthwise)
            self.conv_weight = nn.Parameter(torch.randn(D_in, 1, 3) * (1.0 / math.sqrt(3)))
        elif op.name == "topk_gate":
            # Gate projection: D -> 2
            self.gate_proj = nn.Parameter(torch.randn(2, D_in) * (1.0 / math.sqrt(D_in)))
        else:
            # Math space ops or custom — check for custom init
            if hasattr(op, 'init_params'):
                op.init_params(self, D_in)
            else:
                self.weight = nn.Parameter(torch.randn(D_in, D_in) * (1.0 / math.sqrt(D_in)))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute this primitive operation."""
        return _execute_op(self, self.op_name, inputs, self.config)


def _execute_op(module: nn.Module, op_name: str, inputs: Tuple[torch.Tensor, ...],
                config: Dict) -> torch.Tensor:
    """Execute a single primitive operation."""
    x = inputs[0]

    # ── Elementwise Unary ──
    if op_name == "neg":
        return -x
    elif op_name == "abs":
        return torch.abs(x)
    elif op_name == "exp":
        return torch.exp(torch.clamp(x, -20, 20))
    elif op_name == "log":
        return torch.log(torch.clamp(x.abs(), min=1e-8))
    elif op_name == "sin":
        return torch.sin(x)
    elif op_name == "cos":
        return torch.cos(x)
    elif op_name == "tanh":
        return torch.tanh(x)
    elif op_name == "sigmoid":
        return torch.sigmoid(x)
    elif op_name == "relu":
        return F.relu(x)
    elif op_name == "gelu":
        return F.gelu(x)
    elif op_name == "silu":
        return F.silu(x)
    elif op_name == "sqrt":
        return torch.sqrt(torch.clamp(x.abs(), min=1e-8))
    elif op_name == "square":
        return x * x
    elif op_name == "sign_ste":
        # Sign with straight-through estimator
        signs = torch.sign(x)
        return x + (signs - x).detach()
    elif op_name == "reciprocal":
        return 1.0 / torch.clamp(x.abs(), min=1e-6) * torch.sign(x)

    # ── Elementwise Binary ──
    elif op_name == "add":
        return inputs[0] + inputs[1]
    elif op_name == "mul":
        return inputs[0] * inputs[1]
    elif op_name == "sub":
        return inputs[0] - inputs[1]
    elif op_name == "div_safe":
        return inputs[0] / torch.clamp(inputs[1].abs(), min=1e-6) * torch.sign(inputs[1])
    elif op_name == "maximum":
        return torch.maximum(inputs[0], inputs[1])
    elif op_name == "minimum":
        return torch.minimum(inputs[0], inputs[1])

    # ── Reductions ──
    elif op_name == "sum_last":
        return x.sum(dim=-1, keepdim=True)
    elif op_name == "mean_last":
        return x.mean(dim=-1, keepdim=True)
    elif op_name == "max_last":
        return x.max(dim=-1, keepdim=True).values
    elif op_name == "norm_last":
        return x.norm(dim=-1, keepdim=True)
    elif op_name == "sum_seq":
        return x.sum(dim=1, keepdim=True)
    elif op_name == "mean_seq":
        return x.mean(dim=1, keepdim=True)
    elif op_name == "cumsum":
        return torch.cumsum(x, dim=1)
    elif op_name == "cumprod_safe":
        return torch.cumprod(torch.clamp(x, -2, 2), dim=1)

    # ── Linear Algebra ──
    elif op_name == "matmul":
        a, b = inputs
        # (B, S, D) @ (B, S, D)^T -> (B, S, S) then back
        # Or if shapes work: (B, S, D) @ (B, D, K) -> (B, S, K)
        if a.shape[-1] == b.shape[-1]:
            # Self-attention-like: a @ b^T / sqrt(d)
            scale = math.sqrt(a.shape[-1])
            scores = torch.bmm(a, b.transpose(-2, -1)) / scale
            # Apply to b as values
            return torch.bmm(F.softmax(scores, dim=-1), b)
        else:
            return torch.bmm(a, b)
    elif op_name == "outer_product":
        a, b = inputs
        # Outer product then project back — too expensive for full outer
        # Use low-rank: element-wise multiply of projected versions
        return a * b
    elif op_name == "transpose_sd":
        # Transpose seq and dim, apply, transpose back
        return x.transpose(1, 2).contiguous().transpose(1, 2)

    # ── Structural ──
    elif op_name == "split2":
        D = x.shape[-1]
        return x[..., :D // 2]  # Return first half; second half via another split2
    elif op_name == "split3":
        D = x.shape[-1]
        return x[..., :D // 3]
    elif op_name == "concat":
        return torch.cat([inputs[0], inputs[1]], dim=-1)
    elif op_name == "roll_seq":
        return torch.roll(x, shifts=1, dims=1)
    elif op_name == "roll_neg":
        return torch.roll(x, shifts=-1, dims=1)
    elif op_name == "gather_sorted":
        data, indices = inputs
        if indices.shape[-1] == 1:
            indices = indices.expand_as(data).long()
        else:
            indices = indices[..., :1].expand_as(data).long()
        indices = indices.clamp(0, data.shape[1] - 1)
        return data.gather(1, indices)
    elif op_name == "scatter_unsort":
        data, indices = inputs
        if indices.shape[-1] == 1:
            indices = indices.expand_as(data).long()
        else:
            indices = indices[..., :1].expand_as(data).long()
        indices = indices.clamp(0, data.shape[1] - 1)
        out = torch.zeros_like(data)
        return out.scatter_(1, indices, data)
    elif op_name == "multi_head_mix":
        B, S, D = x.shape
        H = config.get("n_heads", 4)
        if D % H != 0:
            H = 1  # fallback if D not divisible
        head_dim = D // H
        # Reshape to (B, S, H, head_dim), L2 normalize per head, reshape back
        x_heads = x.view(B, S, H, head_dim)
        x_heads = F.normalize(x_heads, p=2, dim=-1)
        return x_heads.view(B, S, D)

    # ── Parameterized ──
    elif op_name in ("linear_proj", "linear_proj_down", "linear_proj_up"):
        if not hasattr(module, 'weight'):
            return x  # graceful fallback if param init failed
        return F.linear(x, module.weight)
    elif op_name == "learnable_scale":
        if not hasattr(module, 'scale'):
            return x
        return x * module.scale
    elif op_name == "learnable_bias":
        if not hasattr(module, 'bias'):
            return x
        return x + module.bias
    elif op_name == "selective_scan":
        if not hasattr(module, 'A_log'):
            return x
        B, S, D = x.shape
        # Input-dependent recurrence: h[t] = decay*h[t-1] + B(x)*x, out = C(x)*h
        A = -torch.exp(module.A_log.clamp(-10, 10))  # negative log ensures decay < 1
        dt = F.softplus(module.dt_proj)  # positive time step
        decay = torch.exp(A * dt).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        B_x = torch.sigmoid(x * module.B_proj)  # (B, S, D)
        C_x = torch.sigmoid(x * module.C_proj)  # (B, S, D)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = decay.squeeze(1) * h + B_x[:, t] * x[:, t]
            outputs.append(C_x[:, t] * h)
        return torch.stack(outputs, dim=1)
    elif op_name == "conv1d_seq":
        if not hasattr(module, 'conv_weight'):
            return x
        # Depthwise conv1d: transpose to (B,D,S), conv, transpose back
        B, S, D = x.shape
        xt = x.transpose(1, 2)  # (B, D, S)
        out = F.conv1d(xt, module.conv_weight, padding=1, groups=D)
        return out.transpose(1, 2)  # (B, S, D)
    elif op_name == "topk_gate":
        if not hasattr(module, 'gate_proj'):
            return x
        B, S, D = x.shape
        # Project to 2 gate scores
        gate_logits = F.linear(x, module.gate_proj)  # (B, S, 2)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, S, 2)
        half = D // 2
        # Weight two feature halves
        out1 = x[..., :half] * gate_weights[..., 0:1]
        out2 = x[..., half:2*half] * gate_weights[..., 1:2]
        # If D is odd, pass through the remainder unchanged
        if D > 2 * half:
            return torch.cat([out1, out2, x[..., 2*half:]], dim=-1)
        return torch.cat([out1, out2], dim=-1)

    # ── Sequence Ops ──
    elif op_name == "softmax_last":
        return F.softmax(x, dim=-1)
    elif op_name == "softmax_seq":
        return F.softmax(x, dim=1)
    elif op_name == "causal_mask":
        B, S, D = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        # Apply mask by zeroing future positions
        return x * (~mask[:S, :S]).float().unsqueeze(0).unsqueeze(-1).sum(dim=2, keepdim=False).clamp(max=1.0).unsqueeze(0)
    elif op_name == "sort_seq":
        # Sort along sequence dim using mean of features as key
        keys = x.mean(dim=-1)  # (B, S)
        indices = keys.argsort(dim=-1)
        return x.gather(1, indices.unsqueeze(-1).expand_as(x))
    elif op_name == "argsort_seq":
        keys = x.mean(dim=-1)
        indices = keys.argsort(dim=-1)
        return indices.unsqueeze(-1).expand_as(x).float()
    elif op_name == "local_window_attn":
        B, S, D = x.shape
        W = min(config.get("window_size", 32), S)
        scale = math.sqrt(D)
        # Full Q=K=V attention scores
        scores = torch.bmm(x, x.transpose(-2, -1)) / scale  # (B, S, S)
        # Causal + window mask
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)
        mask = (col_idx > row_idx) | (row_idx - col_idx >= W)  # future or outside window
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, x)
    elif op_name == "sliding_window_mask":
        B, S, D = x.shape
        W = min(config.get("window_size", 32), S)
        # Exponential distance decay along sequence dim
        row_idx = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(0)
        dist = (row_idx - col_idx).abs()
        decay = torch.exp(-dist / max(W / 4, 1.0))
        # Zero out future (causal) and beyond-window positions
        causal = (col_idx <= row_idx)
        window = (dist < W)
        mask = (causal & window).float() * decay
        # Normalize rows
        mask = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Apply as (S, S) @ (B, S, D) via bmm
        return torch.bmm(mask.unsqueeze(0).expand(B, -1, -1), x)
    elif op_name == "token_pool_restore":
        B, S, D = x.shape
        if S < 2:
            return x
        # Pool adjacent pairs: average (B, S, D) -> (B, S//2, D)
        S_half = S // 2
        pooled = (x[:, 0::2, :][:, :S_half] + x[:, 1::2, :][:, :S_half]) / 2.0
        # Restore via repeat-interleave back to S
        restored = pooled.repeat_interleave(2, dim=1)
        # Handle odd S: pad last token back
        if restored.shape[1] < S:
            restored = torch.cat([restored, x[:, -1:, :]], dim=1)
        return restored

    # ── Frequency ──
    elif op_name == "rfft_seq":
        return torch.fft.rfft(x, dim=1).real  # Take real part to keep shapes simple
    elif op_name == "irfft_seq":
        # Reconstruct — approximate since we only have real part
        B, S_freq, D = x.shape
        S_orig = (S_freq - 1) * 2
        complex_x = torch.complex(x, torch.zeros_like(x))
        return torch.fft.irfft(complex_x, n=S_orig, dim=1)

    # ── Math space ops (registered dynamically) ──
    else:
        # Check if it's a registered math space op with a custom execute
        from .primitives import PRIMITIVE_REGISTRY
        if op_name in PRIMITIVE_REGISTRY:
            prim = PRIMITIVE_REGISTRY[op_name]
            if hasattr(prim, 'execute_fn') and prim.execute_fn is not None:
                return prim.execute_fn(module, *inputs)

        raise ValueError(f"Unknown op: {op_name}")


class CompiledLayer(nn.Module):
    """A compiled computation graph as a PyTorch module."""

    def __init__(self, graph: ComputationGraph):
        super().__init__()
        self.graph = graph
        self.topo_order = graph.topological_order()

        # Create compiled ops for each non-input node
        self.ops = nn.ModuleDict()
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input:
                continue

            input_shapes = [graph.nodes[iid].output_shape for iid in node.input_ids]
            compiled = CompiledOp(
                op_name=node.op_name,
                config=node.config,
                input_shape=input_shapes[0] if input_shapes else ShapeInfo(),
                output_shape=node.output_shape,
                model_dim=graph.model_dim,
            )
            self.ops[str(nid)] = compiled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the computation graph."""
        node_outputs: Dict[int, torch.Tensor] = {}

        for nid in self.topo_order:
            node = self.graph.nodes[nid]

            if node.is_input:
                node_outputs[nid] = x
                continue

            # Gather inputs
            inputs = tuple(node_outputs[iid] for iid in node.input_ids)

            # Execute op
            compiled_op = self.ops[str(nid)]
            node_outputs[nid] = compiled_op(*inputs)

        # Return output node
        output_id = self.graph._output_node_id
        if output_id is None:
            raise RuntimeError("Graph has no output node")
        return node_outputs[output_id]


class SynthesizedModel(nn.Module):
    """A complete language model built from synthesized layers."""

    def __init__(
        self,
        layer_graphs: List[ComputationGraph],
        vocab_size: int = 32000,
        model_dim: int = 256,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.layers = nn.ModuleList([CompiledLayer(g) for g in layer_graphs])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._layer_graphs = layer_graphs

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        lines = [f"SynthesizedModel(dim={self.model_dim}, layers={len(self.layers)}, "
                 f"params={self.param_count():,})"]
        for i, g in enumerate(self._layer_graphs):
            lines.append(f"\n  Layer {i}:")
            for line in g.describe().split("\n"):
                lines.append(f"    {line}")
        return "\n".join(lines)


def compile_graph(graph: ComputationGraph) -> CompiledLayer:
    """Compile a computation graph into a PyTorch module."""
    return CompiledLayer(graph)


def compile_model(
    layer_graphs: List[ComputationGraph],
    vocab_size: int = 32000,
    max_seq_len: int = 512,
) -> SynthesizedModel:
    """Compile a list of layer graphs into a complete language model."""
    if not layer_graphs:
        raise ValueError("Need at least one layer graph")
    model_dim = layer_graphs[0].model_dim
    return SynthesizedModel(layer_graphs, vocab_size, model_dim, max_seq_len)
