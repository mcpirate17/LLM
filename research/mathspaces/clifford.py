"""
Clifford Algebra (Geometric Algebra) Operations

Clifford algebras generalize complex numbers and quaternions.
The geometric product combines dot product and wedge product:
    ab = a·b + a∧b (scalar + bivector)

This gives rotations, reflections, and projections as algebraic
operations — no matrices needed. Potential for more parameter-efficient
geometric transformations.

We implement Cl(3,0) — 3D Clifford algebra with 8 basis elements:
{1, e1, e2, e3, e12, e13, e23, e123}

For neural nets, we work with multivectors stored as 8-channel tensors,
packing them into the feature dimension.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE
from research.mathspaces._utils import causal_mask


# Cl(3,0) has 8 basis elements
# We pack them into groups of 8 within the feature dimension
N_BASIS = 8

# Cl(3,0) metric signs: [+,+,+,+,-,-,-,-] for scalar product computation
_CL30_SIGNS = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32)


def _pack_multivector(x: torch.Tensor) -> torch.Tensor:
    """Reshape (B, S, D) into (B, S, D//8, 8) multivector format."""
    B, S, D = x.shape
    assert D % N_BASIS == 0, f"Dim {D} not divisible by {N_BASIS}"
    return x.reshape(B, S, D // N_BASIS, N_BASIS)


def _unpack_multivector(mv: torch.Tensor) -> torch.Tensor:
    """Reshape (B, S, K, 8) back to (B, S, K*8)."""
    B, S, K, _ = mv.shape
    return mv.reshape(B, S, K * N_BASIS)


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product of two multivectors in Cl(3,0).

    Combines inner product (contraction) and outer product (extension).
    This is the fundamental operation of geometric algebra.

    Input: a, b of shape (B, S, K, 8)
    Output: (B, S, K, 8)
    """
    if (
        _HAS_ARIA_CORE
        and a.is_contiguous()
        and b.is_contiguous()
        and a.device.type == "cpu"
        and not a.requires_grad
        and not b.requires_grad
    ):
        return aria_core.clifford_geometric_product_cl30_f32(a, b)

    # Basis: {1, e1, e2, e3, e12, e13, e23, e123}
    # Index: { 0,  1,  2,  3,   4,   5,   6,    7}
    #
    # Multiplication table for Cl(3,0):
    # e_i * e_i = +1 for i in {1,2,3}
    # e_i * e_j = -e_j * e_i for i != j
    # e12 = e1*e2, e13 = e1*e3, e23 = e2*e3, e123 = e1*e2*e3

    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    a12, a13, a23, a123 = a[..., 4], a[..., 5], a[..., 6], a[..., 7]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    b12, b13, b23, b123 = b[..., 4], b[..., 5], b[..., 6], b[..., 7]

    # Scalar part (grade 0)
    r0 = (
        a0 * b0
        + a1 * b1
        + a2 * b2
        + a3 * b3
        - a12 * b12
        - a13 * b13
        - a23 * b23
        - a123 * b123
    )

    # Vector parts (grade 1)
    r1 = (
        a0 * b1
        + a1 * b0
        - a2 * b12
        + a12 * b2
        - a3 * b13
        + a13 * b3
        + a23 * b123
        - a123 * b23
    )
    r2 = (
        a0 * b2
        + a1 * b12
        + a2 * b0
        - a12 * b1
        - a3 * b23
        - a13 * b123
        + a23 * b3
        + a123 * b13
    )
    r3 = (
        a0 * b3
        - a1 * b13
        + a2 * b23
        + a3 * b0
        + a12 * b123
        + a13 * b1
        - a23 * b2
        - a123 * b12
    )

    # Bivector parts (grade 2)
    r12 = (
        a0 * b12
        + a1 * b2
        - a2 * b1
        + a12 * b0
        + a3 * b123
        - a13 * b23
        + a23 * b13
        + a123 * b3
    )
    r13 = (
        a0 * b13
        + a1 * b3
        - a3 * b1
        + a13 * b0
        - a2 * b123
        + a12 * b23
        - a23 * b12
        - a123 * b2
    )
    r23 = (
        a0 * b23
        + a2 * b3
        - a3 * b2
        + a23 * b0
        + a1 * b123
        - a12 * b13
        + a13 * b12
        + a123 * b1
    )

    # Pseudoscalar (grade 3)
    r123 = (
        a0 * b123
        + a1 * b23
        - a2 * b13
        + a3 * b12
        + a12 * b3
        - a13 * b2
        + a23 * b1
        + a123 * b0
    )

    return torch.stack([r0, r1, r2, r3, r12, r13, r23, r123], dim=-1)


_GRADE_MASKS: dict[int, list[int]] = {
    0: [0],
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7],
}


def grade_select(mv: torch.Tensor, grade: int) -> torch.Tensor:
    """Select specific grade from a multivector.

    Grade 0: scalar (index 0)
    Grade 1: vectors (indices 1,2,3)
    Grade 2: bivectors (indices 4,5,6)
    Grade 3: pseudoscalar (index 7)
    """
    mask = torch.zeros(N_BASIS, device=mv.device, dtype=mv.dtype)
    mask[_GRADE_MASKS[grade]] = 1.0
    return mv * mask


def clifford_norm(mv: torch.Tensor) -> torch.Tensor:
    """Norm of a multivector: sqrt(|a * ~a|) where ~a is the reverse."""
    return (mv * mv).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()


def rotor_transform(x: torch.Tensor, rotor: torch.Tensor) -> torch.Tensor:
    """Apply a rotor transformation: R x ~R

    Rotors are the Clifford algebra analog of rotation matrices.
    Much more parameter-efficient: a rotor in Cl(3,0) uses 4 numbers
    to encode a 3D rotation (like quaternions).
    """
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and rotor.is_contiguous()
        and x.device.type == "cpu"
        and x.dtype == torch.float32
        and not x.requires_grad
        and not rotor.requires_grad
    ):
        try:
            return aria_core.clifford_rotor_transform_cl30_f32(x, rotor)
        except TypeError:
            pass  # Fall through to Python path

    # Reverse of rotor: negate bivector and pseudoscalar parts
    rotor_rev = rotor.clone()
    rotor_rev[..., 4:7] = -rotor_rev[..., 4:7]
    rotor_rev[..., 7] = -rotor_rev[..., 7]

    # R x ~R
    temp = geometric_product(rotor, x)
    return geometric_product(temp, rotor_rev)


# ── Primitive execution functions ─────────────────────────────────────


def execute_geometric_product(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Geometric product of two tensors interpreted as multivectors."""
    B, S, D = x.shape
    # Pad to multiple of 8 if needed
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x = F.pad(x, (0, pad))
        y = F.pad(y, (0, pad))

    mv_x = _pack_multivector(x)
    mv_y = _pack_multivector(y)
    result = geometric_product(mv_x, mv_y)
    result = _unpack_multivector(result)

    if pad > 0:
        result = result[..., :D]
    return result


def execute_rotor_transform(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply learned rotor transformation."""
    orig_dtype = x.dtype
    B, S, D = x.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x = F.pad(x, (0, pad))

    mv_x = _pack_multivector(x.float())

    # Learned rotor — param may be stored as 'rotor' or 'weight'
    rotor_param = getattr(module, "rotor", None)
    if rotor_param is None:
        rotor_param = getattr(module, "weight", None)
    if rotor_param is not None:
        K = D // N_BASIS if pad == 0 else (D + pad) // N_BASIS
        rotor_params = (
            rotor_param[:N_BASIS].float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        rotor = rotor_params.expand(B, S, K, -1)
        # Normalize rotor
        rotor = rotor / clifford_norm(rotor).clamp(min=1e-6)
    else:
        return _unpack_multivector(mv_x)[..., :D].to(orig_dtype)

    result = rotor_transform(mv_x, rotor)
    result = _unpack_multivector(result)

    if pad > 0:
        result = result[..., :D]
    return result.to(orig_dtype)


def execute_grade_select(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Select grade-1 (vector) components from multivector."""
    orig_dtype = x.dtype
    B, S, D = x.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x = F.pad(x, (0, pad))
    mv = _pack_multivector(x)
    selected = grade_select(mv, grade=1)
    result = _unpack_multivector(selected)
    if pad > 0:
        result = result[..., :D]
    return result.to(orig_dtype)


def execute_grade_mix(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Blend vector and bivector grades for richer geometric features."""
    orig_dtype = x.dtype
    B, S, D = x.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x = F.pad(x, (0, pad))
    mv = _pack_multivector(x)
    g1 = grade_select(mv, grade=1)
    g2 = grade_select(mv, grade=2)
    mixed = 0.7 * g1 + 0.3 * g2
    result = _unpack_multivector(mixed)
    if pad > 0:
        result = result[..., :D]
    return result.to(orig_dtype)


def clifford_reverse(mv: torch.Tensor) -> torch.Tensor:
    """Reverse of a Cl(3,0) multivector ~mv: negate bivector + pseudoscalar parts."""
    rev = mv.clone()
    rev[..., 4:7] = -rev[..., 4:7]
    rev[..., 7] = -rev[..., 7]
    return rev


def clifford_inverse(mv: torch.Tensor) -> torch.Tensor:
    """Multiplicative inverse of a Cl(3,0) multivector.

    For versors and unit multivectors mv⁻¹ = ~mv / ||mv||² where ~mv is the
    reverse and ||mv||² = (mv * ~mv)_scalar. Falls back to ~mv / max(||mv||², eps)
    for numerical stability on general multivectors.
    """
    rev = clifford_reverse(mv)
    norm_sq = (mv * mv).sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return rev / norm_sq


def versor_apply(versor: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    """Sandwich product v · mv · v⁻¹ — the canonical Clifford rotation primitive.

    For unit-versor inputs this equals the rotor sandwich `R · mv · ~R` already
    implemented by ``rotor_transform``. The general form using the inverse
    handles arbitrary versors.
    """
    v_inv = clifford_inverse(versor)
    temp = geometric_product(versor, mv)
    return geometric_product(temp, v_inv)


def execute_clifford_inverse(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply clifford_inverse to a tensor interpreted as packed multivectors."""
    orig_dtype = x.dtype
    B, S, D = x.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x = F.pad(x, (0, pad))
    mv = _pack_multivector(x.float())
    inv = clifford_inverse(mv)
    result = _unpack_multivector(inv)
    if pad > 0:
        result = result[..., :D]
    return result.to(orig_dtype)


def execute_versor_apply(
    module: nn.Module, versor_tensor: torch.Tensor, mv_tensor: torch.Tensor
) -> torch.Tensor:
    """Versor sandwich product on packed multivectors.

    Both inputs are interpreted as Cl(3,0) multivectors of dimension D. Returns
    the same dimension D.
    """
    orig_dtype = mv_tensor.dtype
    B, S, D = mv_tensor.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        versor_tensor = F.pad(versor_tensor, (0, pad))
        mv_tensor = F.pad(mv_tensor, (0, pad))
    versor = _pack_multivector(versor_tensor.float())
    mv = _pack_multivector(mv_tensor.float())
    result = versor_apply(versor, mv)
    out = _unpack_multivector(result)
    if pad > 0:
        out = out[..., :D]
    return out.to(orig_dtype)


def execute_clifford_attention(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Attention using geometric product instead of dot product."""
    orig_dtype = x.dtype
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and x.dtype == torch.float32
        and not x.requires_grad
    ):
        return aria_core.clifford_attention_f32(x)

    B, S, D = x.shape
    pad = (N_BASIS - D % N_BASIS) % N_BASIS
    if pad > 0:
        x_padded = F.pad(x, (0, pad))
    else:
        x_padded = x

    D_padded = x_padded.shape[-1]

    # QKV via learned weight or identity
    if hasattr(module, "weight"):
        # weight shape: (D*D,) — reshape to (D_padded, D_padded) for Q/K
        W = module.weight
        n = D_padded * D_padded
        if W.numel() >= n:
            Wq = W[:n].reshape(D_padded, D_padded)
            q = F.linear(x_padded, Wq.to(x_padded.dtype))
            k = x_padded
        else:
            q = k = x_padded
    else:
        q = k = x_padded

    # Pack as multivectors
    mv_q = _pack_multivector(q)  # (B, S, K, 8)
    mv_k = _pack_multivector(k)  # (B, S, K, 8)

    # Geometric product scores: sum scalar component over K
    # For each pair (i, j): gp(q_i, k_j) scalar part → attention score
    # Efficient: compute scalar part of geometric product without full product
    # Scalar = sum over basis of a_b * b_b * sign_b
    # For Cl(3,0): signs are [+,+,+,+,-,-,-,-]
    signs = _CL30_SIGNS.to(device=x.device, dtype=x.dtype)
    q_signed = mv_q * signs  # (B, S, K, 8)
    # Use mean instead of sum so score magnitude does not scale with basis width.
    q_scalar = q_signed.mean(dim=-1)  # (B, S, K)
    k_scalar = mv_k.mean(dim=-1)  # (B, S, K)
    q_scalar = F.normalize(q_scalar, dim=-1)
    k_scalar = F.normalize(k_scalar, dim=-1)

    # Attention scores via dot in the scalar-projected space
    scores = torch.bmm(q_scalar, k_scalar.transpose(1, 2))  # (B, S, S)
    scale = math.sqrt(D_padded)

    # Apply causal mask if S > 1
    if S > 1:
        scores.masked_fill_(causal_mask(S, x.device), float("-inf"))

    weights = torch.softmax((scores / scale).clamp(min=-20.0, max=20.0), dim=-1)
    out = torch.bmm(weights, x_padded)  # (B, S, D_padded)

    if pad > 0:
        out = out[..., :D]
    return out.to(orig_dtype)
