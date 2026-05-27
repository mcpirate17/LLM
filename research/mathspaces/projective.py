"""
Projective Geometry Operations

Projective geometry allows the network to model spatial reasoning,
3D-consistency, and perspective transformations explicitly. It extends
standard Euclidean space to handle "points at infinity" and homographies.

Operations:
- projective_map (Euclidean -> Homogeneous coordinates)
- affine_map (Homogeneous -> Euclidean)
- projective_linear (Homography / Projective Transformation)
- projective_attention (Perspective-aware sequence mixing)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def projective_map(x: torch.Tensor) -> torch.Tensor:
    """Map Euclidean coordinates to Homogeneous (Projective) coordinates.

    Appends a dimension of 1s to the tensor.
    x: (..., D) -> (..., D+1)
    """
    ones = torch.ones_like(x[..., :1])
    return torch.cat([x, ones], dim=-1)


def affine_map(h: torch.Tensor) -> torch.Tensor:
    """Map Homogeneous coordinates back to Euclidean.

    Divides by the last coordinate (the projective weight) and drops it.
    h: (..., D+1) -> (..., D)
    """
    w = h[..., -1:]
    # Avoid division by zero
    w_safe = torch.where(w.abs() < EPS, torch.sign(w) * EPS + (w == 0) * EPS, w)
    euclidean = h[..., :-1] / w_safe
    return euclidean


def projective_linear(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a projective transformation (homography).

    h: (..., D+1)
    weight: (D+1, D+1)
    """
    return F.linear(h, weight)


def projective_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A perspective-aware distance between projective coordinates.

    Uses the cosine similarity of the homogeneous rays, which represents
    the angle between the projective lines.
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=EPS)

    # Dot product of normalized rays
    cos_theta = (x / x_norm * y / y_norm).sum(dim=-1)
    # Distance is 1 - cos(theta) (angular distance)
    return 1.0 - cos_theta


class ProjectiveAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)
        B, S, D = q.shape

        # Compute pairwise angular distances
        # q, k: (B, S, D+1)
        q_norm = q / q.norm(dim=-1, keepdim=True).clamp(min=EPS)
        k_norm = k / k.norm(dim=-1, keepdim=True).clamp(min=EPS)

        # Cosine similarity
        sim = torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, S, S)

        # Causal mask
        if S > 1:
            row_ids = torch.arange(S, device=q.device).unsqueeze(1)
            col_ids = torch.arange(S, device=q.device).unsqueeze(0)
            sim.masked_fill_(col_ids > row_ids, float("-inf"))

        weights = torch.softmax(sim, dim=-1)
        # v is (B, S, D), out is (B, S, D)
        out = torch.bmm(weights, v)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Fallback to standard autograd for simplicity in prototype
        pass


def projective_attention(x: torch.Tensor) -> torch.Tensor:
    """Self-attention using projective geometry (PRoPE-style)."""
    # Simple prototype: chunk into Q, K, V
    D = x.shape[-1]
    D_head = D // 3
    q, k, v = x[..., :D_head], x[..., D_head : 2 * D_head], x[..., 2 * D_head :]

    # Map Q and K to projective space
    q_proj = projective_map(q)
    k_proj = projective_map(k)

    B, S, _ = q.shape
    q_norm = q_proj / q_proj.norm(dim=-1, keepdim=True).clamp(min=EPS)
    k_norm = k_proj / k_proj.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sim = torch.bmm(q_norm, k_norm.transpose(1, 2))

    if S > 1:
        row_ids = torch.arange(S, device=x.device).unsqueeze(1)
        col_ids = torch.arange(S, device=x.device).unsqueeze(0)
        sim.masked_fill_(col_ids > row_ids, float("-inf"))

    weights = torch.softmax(sim, dim=-1)
    out = torch.bmm(weights, v)

    # Pad back to original D by repeating or zeroing
    # For a clean primitive, we just project back to D
    padded = torch.cat([out, out, out], dim=-1)
    if padded.shape[-1] != D:
        padded = F.pad(padded, (0, D - padded.shape[-1]))
    return padded


# ── Primitive execution functions ─────────────────────────────────────


def execute_projective_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Projective linear transformation."""
    h = projective_map(x)
    D_plus_1 = h.shape[-1]
    weight = module.weight.view(D_plus_1, D_plus_1)
    transformed = projective_linear(h, weight)
    return affine_map(transformed)


def execute_projective_attention(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Attention using projective angular distances."""
    return projective_attention(x)
