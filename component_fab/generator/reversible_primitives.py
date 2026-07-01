"""Reversible novel-mechanism stack — activation-free training (VRAM reduction).

A reversible additive-coupling mixer whose coupling functions are NON-QKV causal
decayed-context mixers, plus a :class:`ReversibleSequential` that trains a DEEP
stack of them with activation memory that is **O(1) in depth** instead of
O(depth). The single block is exactly invertible; the stack exploits that: it
saves only the final activation and, in the backward pass, walks the blocks in
reverse, reconstructing each block's input from its output (via the inverse) and
recomputing just that block's local graph. No per-layer activations are stored.
This is the training-VRAM lever for the deep novel-mechanism stacks the mission
needs.

Block (channels split into halves ``x1, x2``)::

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)     ->     x2 = y2 - G(y1),  x1 = y1 - F(x2)   (exact inverse)

``F`` and ``G`` are :class:`_CausalDecayMLP` — a per-channel causal power-law
decayed context followed by a GELU MLP: stateful, strictly causal, non-QKV, and
anti-softmax-twin (no score normalization, no convex token average).

Correctness of the manual reverse-sweep backward is pinned by a
gradient-equivalence test against a plain autograd stack (exact in float64), and
the activation saving is pinned by a forward-saved-tensor-count test (~300x fewer
at depth 16) — both in ``test_reversible_primitives``.

Caveat (inherent to reversible nets): inputs are reconstructed from outputs, so
in float32 the reconstruction accumulates ~1e-3 gradient error vs a plain stack.
That is the standard activation-memory-vs-precision trade; it is exact in float64
and fine for training, where gradients are already stochastic.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from component_fab.generator._causal_scan import (
    DecayScanState,
    causal_decay_context,
    decay_scan_step,
    init_decay_scan_state,
)


class _CausalDecayMLP(nn.Module):
    """Per-channel causal power-law decayed context + GELU MLP (non-QKV coupling).

    ``c_t = Σ_{s≤t} ρ**(t-s) x_s`` with a learnable per-channel decay ``ρ``
    (SSM-like linear memory), then a GELU MLP. Strictly causal and shape
    preserving.
    """

    def __init__(self, dim: int, decay_init: float = 0.9, expansion: int = 2) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        self.dim = dim
        logit = torch.logit(torch.tensor(decay_init))
        self.log_decay = nn.Parameter(torch.full((dim,), float(logit)))
        self.w_in = nn.Linear(dim, expansion * dim)
        self.w_out = nn.Linear(expansion * dim, dim)

    def _decay(self) -> torch.Tensor:
        return torch.sigmoid(self.log_decay).clamp(1e-4, 1 - 1e-4)  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = causal_decay_context(x, self._decay())
        return self.w_out(F.gelu(self.w_in(context)))

    def stream_init(self, batch: int, *, device=None, dtype=None) -> DecayScanState:
        """A fresh O(1) streaming state for autoregressive (KV-free) decode."""
        return init_decay_scan_state(batch, self.dim, device=device, dtype=dtype)

    def stream_step(
        self, x_t: torch.Tensor, state: DecayScanState
    ) -> tuple[torch.Tensor, DecayScanState]:
        """One decode step: ``[B, D]`` in, ``[B, D]`` out, state is O(1) in length."""
        context, new_state = decay_scan_step(state, x_t, self._decay())
        return self.w_out(F.gelu(self.w_in(context))), new_state


class ReversibleCouplingMixerLane(nn.Module):
    """Exactly-invertible additive-coupling mixer over two non-QKV causal MLPs.

    A bijective sequence mixer: the map ``x -> y`` is exactly invertible (see
    :meth:`inverse`), which is what lets :class:`ReversibleSequential` train deep
    stacks without storing activations. Standalone, it is a normalizing-flow-style
    invertible lane. ``dim`` must be even (split into halves); the dispatcher
    falls back to a dense linear map otherwise.
    """

    def __init__(self, dim: int, decay_init: float = 0.9) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("ReversibleCouplingMixerLane requires an even dim")
        self.dim = dim
        self.half = dim // 2
        self.f = _CausalDecayMLP(self.half, decay_init=decay_init)
        self.g = _CausalDecayMLP(self.half, decay_init=decay_init)

    def coupling_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Additive coupling ``y1 = x1 + F(x2); y2 = x2 + G(y1)`` (grad-tracking)."""
        x1, x2 = x[..., : self.half], x[..., self.half :]
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coupling_forward(x)

    def stream_init(
        self, batch: int, *, device=None, dtype=None
    ) -> tuple[DecayScanState, DecayScanState]:
        """Fresh O(1) decode state ``(f_state, g_state)`` — no growing KV cache."""
        return (
            self.f.stream_init(batch, device=device, dtype=dtype),
            self.g.stream_init(batch, device=device, dtype=dtype),
        )

    def stream_step(
        self,
        x_t: torch.Tensor,
        state: tuple[DecayScanState, DecayScanState],
    ) -> tuple[torch.Tensor, tuple[DecayScanState, DecayScanState]]:
        """One autoregressive step: coupling applied token-wise with O(1) state.

        Reproduces :meth:`coupling_forward` exactly for the current token, keeping
        only the two ``[B, half]`` decay states — inference memory is independent
        of context length.
        """
        f_state, g_state = state
        x1, x2 = x_t[..., : self.half], x_t[..., self.half :]
        f_out, f_state = self.f.stream_step(x2, f_state)
        y1 = x1 + f_out
        g_out, g_state = self.g.stream_step(y1, g_state)
        y2 = x2 + g_out
        return torch.cat([y1, y2], dim=-1), (f_state, g_state)

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the output (exact additive-coupling inverse)."""
        y1, y2 = y[..., : self.half], y[..., self.half :]
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=-1)


class _ReversibleSequentialFn(torch.autograd.Function):
    """Deep reversible stack with O(1)-in-depth activation memory.

    ``forward`` runs every block under ``no_grad`` and saves ONLY the final
    output — no per-layer activations. ``backward`` walks the blocks in reverse:
    it reconstructs each block's input from its output via the exact inverse,
    recomputes just that block's local graph to route the incoming gradient to the
    block's input and parameters, then moves to the previous block. At most two
    activations (the current output and its reconstructed input) are live at once,
    so training memory is independent of stack depth.
    """

    @staticmethod
    def forward(ctx, blocks, param_counts, x, *params):  # noqa: ANN001
        ctx.blocks = blocks
        ctx.param_counts = param_counts
        with torch.no_grad():
            y = x
            for block in blocks:
                y = block.coupling_forward(y)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: ANN001
        (y,) = ctx.saved_tensors
        blocks = ctx.blocks
        per_block_param_grads: list[list[torch.Tensor]] = []
        for block in reversed(blocks):
            x = block.inverse(y)  # reconstruct this block's input (no activations kept)
            block_params = list(block.parameters())
            with torch.enable_grad():
                x_leaf = x.detach().requires_grad_(True)
                y_recomputed = block.coupling_forward(x_leaf)
            grads = torch.autograd.grad(
                y_recomputed,
                [x_leaf, *block_params],
                grad_outputs=dy,
                allow_unused=True,
            )
            dy = grads[0]  # gradient w.r.t. this block's input -> previous block's dy
            per_block_param_grads.append(list(grads[1:]))
            y = x  # this block's input is the previous block's output
        # per_block_param_grads is in reverse block order; flatten to forward order.
        flat_param_grads: list[torch.Tensor] = []
        for block_grads in reversed(per_block_param_grads):
            flat_param_grads.extend(block_grads)
        return (None, None, dy, *flat_param_grads)


class ReversibleSequential(nn.Module):
    """A depth-``O(1)``-activation-memory stack of reversible coupling mixers.

    Equivalent in output and gradients to applying the blocks in sequence, but
    trains without storing per-layer activations (they are recomputed from the
    final output in the backward pass). Use for deep novel-mechanism stacks whose
    activation memory would otherwise dominate training VRAM.
    """

    def __init__(self, blocks: Sequence[ReversibleCouplingMixerLane]) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("ReversibleSequential needs at least one block")
        self.blocks = nn.ModuleList(blocks)

    @classmethod
    def build(
        cls, dim: int, depth: int, decay_init: float = 0.9
    ) -> "ReversibleSequential":
        return cls([ReversibleCouplingMixerLane(dim, decay_init) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = list(self.blocks)
        param_counts = tuple(len(list(b.parameters())) for b in blocks)
        flat_params = [p for b in blocks for p in b.parameters()]
        return _ReversibleSequentialFn.apply(
            blocks, param_counts, x.contiguous(), *flat_params
        )
