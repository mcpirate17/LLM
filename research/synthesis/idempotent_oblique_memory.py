# pyright: reportPrivateImportUsage=false
"""NM-F2 - Idempotent oblique-projection memory.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer whose memory update is forced
to be an idempotent overwrite, not an additive fast-weight blend:

    S' = P S + (I - P) (v k^T),       P = I - Q Q^T,       Q^T Q = I

``Q`` is a rank-r overwrite frame generated from the current token by a small
Householder product. Because ``P^2 = P``, the selected row subspace is replaced
exactly: applying the same write twice is a no-op, and writing a new value for a
key overwrites the old subspace content instead of accumulating a blend. That is
the capability target for NM-F2: exact same-key overwrite under randomized
positions, the place additive memories usually leak old values.

This is deliberately not softmax/QKV attention. There is no pairwise token
score matrix, no probability simplex over positions, and no exponential
normalization. The only cross-token path is the causal state matrix ``S`` and
the update is a projector algebra law. The Householder controller is O(rD)
trainable parameters; the D x D key/value/query/output lifts are plumbing kept
identity/zero initialized so the module is a safe drop-in and NM-10 measurable.

The scan is a torch reference implementation. The rank-r projector update itself
is vectorized over batch and avoids materializing ``P``; a native scan is the
production path if this lane graduates to a hot runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def idempotent_oblique_core_param_count(dim: int, rank: int) -> int:
    """O(rD) Householder overwrite controller plus an O(D) sigmoid write gate."""
    _validate_dim_rank(dim, rank)
    return rank * dim + dim + 1


def idempotent_oblique_param_count(dim: int, rank: int) -> int:
    """Total trainable params: O(rD) core plus key/value/query/output D x D lifts."""
    return idempotent_oblique_core_param_count(dim, rank) + 4 * dim * dim


def _validate_dim_rank(dim: int, rank: int) -> None:
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > dim:
        raise ValueError(f"rank ({rank}) must be <= dim ({dim})")


def householder_frame(vectors: torch.Tensor, *, dim: int, rank: int) -> torch.Tensor:
    """Return orthonormal frame ``Q`` from token-generated Householder vectors.

    ``vectors`` has shape ``(..., rank, dim)``. Starting from the first ``rank``
    coordinate axes, the function applies a product of Householder reflections;
    orthonormality follows from the product of orthogonal maps, without QR/SVD.
    """
    _validate_dim_rank(dim, rank)
    if vectors.shape[-2:] != (rank, dim):
        raise ValueError(
            f"vectors must end with (rank, dim)=({rank}, {dim}), got {vectors.shape}"
        )
    eye = torch.eye(dim, rank, device=vectors.device, dtype=vectors.dtype)
    frame = eye.expand(*vectors.shape[:-2], dim, rank).clone()
    for idx in range(rank):
        v = vectors[..., idx, :]
        denom = v.square().sum(dim=-1, keepdim=True).clamp_min(_EPS)
        vtf = torch.einsum("...d,...dr->...r", v, frame)
        frame = frame - 2.0 * v.unsqueeze(-1) * (vtf / denom).unsqueeze(-2)
    return frame


def left_project(matrix: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Apply ``Q Q^T`` to the rows of ``matrix`` without forming the projector."""
    if matrix.ndim != 3:
        raise ValueError(f"matrix must be (B,D,D), got {matrix.shape}")
    if frame.ndim != 3:
        raise ValueError(f"frame must be (B,D,r), got {frame.shape}")
    if matrix.shape[0] != frame.shape[0] or matrix.shape[1] != frame.shape[1]:
        raise ValueError(
            f"matrix/frame batch or dim mismatch: {matrix.shape}, {frame.shape}"
        )
    coeff = torch.einsum("bdr,bdk->brk", frame, matrix)
    return torch.einsum("bdr,brk->bdk", frame, coeff)


def read_state(state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Read ``S q`` from the fast-weight state."""
    if state.ndim != 3 or query.ndim != 2:
        raise ValueError(
            f"state/query must be (B,D,D)/(B,D), got {state.shape}/{query.shape}"
        )
    return torch.einsum("bdk,bk->bd", state, query)


def idempotent_oblique_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frame: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Projector overwrite update ``P S + (I-P)(v k^T)``.

    With ``gate=None`` or ``gate=1``, the update is exactly idempotent for a fixed
    ``(key, value, frame)``. A gate in ``[0,1]`` gives the trainable partial-write
    path used by the sequence mixer while preserving the same overwrite direction.
    """
    if key.shape != value.shape:
        raise ValueError(
            f"key and value shapes must match, got {key.shape}/{value.shape}"
        )
    if state.shape[:2] != key.shape or state.shape[2] != key.shape[-1]:
        raise ValueError(f"state/key shape mismatch: {state.shape}/{key.shape}")
    target = value.unsqueeze(-1) * key.unsqueeze(-2)
    old_rows = left_project(state, frame)
    new_rows = left_project(target, frame)
    delta = new_rows - old_rows
    if gate is not None:
        delta = delta * gate.reshape(-1, 1, 1)
    return state + delta


class IdempotentObliqueMemory(nn.Module):
    """Causal idempotent overwrite memory with token-generated Householder frames."""

    def __init__(self, dim: int, *, rank: int = 4) -> None:
        super().__init__()
        _validate_dim_rank(dim, rank)
        self.d = int(dim)
        self.rank = int(rank)

        self.key_lift = nn.Linear(dim, dim, bias=False)
        self.value_lift = nn.Linear(dim, dim, bias=False)
        self.query_lift = nn.Linear(dim, dim, bias=False)
        self.out_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.key_lift.weight.copy_(eye)
            self.value_lift.weight.copy_(eye)
            self.query_lift.weight.copy_(eye)
        nn.init.zeros_(self.out_lift.weight)

        # O(rD) controller. Fixed seeds keep reflectors nonzero; learned gains
        # decide how strongly token content bends the overwrite subspace.
        seed = torch.zeros(rank, dim)
        for idx in range(rank):
            seed[idx, idx % dim] = 1.0
            seed[idx, (idx + 1) % dim] = 0.25
        self.register_buffer("reflector_seed", seed)
        self.reflector_gain = nn.Parameter(torch.full((rank, dim), 0.1))

        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    @property
    def num_parameters(self) -> int:
        return idempotent_oblique_param_count(self.d, self.rank)

    @property
    def core_parameters(self) -> int:
        return idempotent_oblique_core_param_count(self.d, self.rank)

    def overwrite_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Generate token-conditioned orthonormal overwrite frames ``Q``."""
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim must be {self.d}, got {x.shape[-1]}")
        rolled = torch.stack(
            [torch.roll(x, shifts=idx + 1, dims=-1) for idx in range(self.rank)],
            dim=-2,
        )
        reflectors = (
            self.reflector_seed.to(x.dtype) + torch.tanh(rolled) * self.reflector_gain
        )
        return householder_frame(reflectors, dim=self.d, rank=self.rank)

    def scan_memory(
        self, x: torch.Tensor, *, read_before_write: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the causal memory scan and return ``(reads, states)``.

        Reads default to exclusive/causal: token t reads memory written by tokens
        ``< t`` before its own write is applied.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B,L,D), got {x.shape}")
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim must be {self.d}, got {x.shape[-1]}")
        key = F.normalize(self.key_lift(x), dim=-1, eps=_EPS)
        query = F.normalize(self.query_lift(x), dim=-1, eps=_EPS)
        value = self.value_lift(x)
        frame = self.overwrite_frame(x)
        gate = torch.sigmoid(
            torch.einsum("bld,d->bl", x, self.gate_weight) + self.gate_bias
        )

        state = x.new_zeros(x.shape[0], self.d, self.d)
        reads: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for idx in range(x.shape[1]):
            if read_before_write:
                reads.append(read_state(state, query[:, idx]))
            state = idempotent_oblique_update(
                state,
                key[:, idx],
                value[:, idx],
                frame[:, idx],
                gate=gate[:, idx],
            )
            if not read_before_write:
                reads.append(read_state(state, query[:, idx]))
            states.append(state)
        return torch.stack(reads, dim=1), torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,L,D) -> (B,L,D)``: residual plus zero-init lifted memory read."""
        reads, _states = self.scan_memory(x)
        return x + self.out_lift(reads)
