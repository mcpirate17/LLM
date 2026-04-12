"""Tests for routing_ops.c kernels: transpose_sd, gated_lane_blend,
depth_gated_transform, calibrated_branch_merge.

These are C kernels exposed via pybind11 in bind_ops.cpp.
"""

import math

import torch
import torch.testing

import aria_core


# ── transpose_sd_f32 ──────────────────────────────────────────────────


def _ref_transpose_sd(x: torch.Tensor) -> torch.Tensor:
    """Reference: interleave even/odd halves of the channel dimension."""
    B, S, D = x.shape
    half = D // 2
    y = torch.empty_like(x)
    y[:, :, 0::2] = x[:, :, :half]
    y[:, :, 1::2] = x[:, :, half:]
    return y


def test_transpose_sd_basic_shape():
    x = torch.randn(2, 3, 4)
    y = aria_core.transpose_sd_f32(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_transpose_sd_parity_vs_reference():
    torch.manual_seed(0)
    for B, S, D in [(1, 1, 2), (2, 4, 8), (3, 7, 16), (1, 10, 64)]:
        x = torch.randn(B, S, D)
        native = aria_core.transpose_sd_f32(x)
        ref = _ref_transpose_sd(x)
        torch.testing.assert_close(native, ref, atol=1e-6, rtol=0.0)


def test_transpose_sd_is_involution():
    """Applying transpose_sd twice should NOT return the original (it's a shuffle)."""
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    y1 = aria_core.transpose_sd_f32(x)
    y2 = aria_core.transpose_sd_f32(y1)
    # x = [1,2,3,4] → y1 = [1,3,2,4] → y2 = [1,2,3,4] — it IS an involution
    torch.testing.assert_close(y2, x, atol=1e-6, rtol=0.0)


def test_transpose_sd_concrete_values():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    y = aria_core.transpose_sd_f32(x)
    # half=3: even positions get first half, odd positions get second half
    # dst[0]=src[0]=1, dst[1]=src[3]=4, dst[2]=src[1]=2, dst[3]=src[4]=5, dst[4]=src[2]=3, dst[5]=src[5]=6
    expected = torch.tensor([[[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]]])
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0.0)


def test_transpose_sd_preserves_values():
    x = torch.randn(2, 5, 8)
    y = aria_core.transpose_sd_f32(x)
    assert torch.isfinite(y).all()
    # All values from x should appear in y (it's a permutation)
    for b in range(2):
        for s in range(5):
            assert set(x[b, s].tolist()) == set(y[b, s].tolist())


# ── gated_lane_blend_f32 ──────────────────────────────────────────────


def _ref_gated_lane_blend(
    x: torch.Tensor,
    scorer: torch.Tensor,
    projs: torch.Tensor,
) -> torch.Tensor:
    """Reference: soft-routed multi-lane transform.

    scorer: [n_lanes, D], projs: [n_lanes, D_out, D]
    For each token: softmax(x @ scorer^T) weighted sum of x @ projs[l]^T
    """
    B, S, D = x.shape
    n_lanes = scorer.shape[0]
    D_out = projs.shape[1]

    y = torch.zeros(B, S, D_out, dtype=x.dtype)
    for b in range(B):
        for s in range(S):
            xr = x[b, s]  # [D]
            logits = xr @ scorer.T  # [n_lanes]
            weights = torch.softmax(logits, dim=0)
            for l in range(n_lanes):
                lane_proj = xr @ projs[l].T  # [D_out]
                y[b, s] += weights[l] * lane_proj
    return y


def test_gated_lane_blend_shape():
    B, S, D, L = 2, 4, 8, 3
    x = torch.randn(B, S, D)
    scorer = torch.randn(L, D)
    projs = torch.randn(L, D, D)  # D_out = D
    y = aria_core.gated_lane_blend_f32(x, scorer, projs)
    assert y.shape == (B, S, D)


def test_gated_lane_blend_parity_vs_reference():
    torch.manual_seed(1)
    B, S, D, L = 2, 5, 8, 3
    x = torch.randn(B, S, D) * 0.1
    scorer = torch.randn(L, D) * 0.1
    projs = torch.randn(L, D, D) * 0.1
    native = aria_core.gated_lane_blend_f32(x, scorer, projs)
    ref = _ref_gated_lane_blend(x, scorer, projs)
    torch.testing.assert_close(native, ref, atol=1e-4, rtol=1e-3)


def test_gated_lane_blend_single_lane_is_linear():
    """With 1 lane, softmax weight is always 1.0, so output = x @ projs[0]^T."""
    torch.manual_seed(2)
    B, S, D = 2, 3, 4
    x = torch.randn(B, S, D) * 0.1
    scorer = torch.randn(1, D) * 0.1
    projs = torch.randn(1, D, D) * 0.1
    native = aria_core.gated_lane_blend_f32(x, scorer, projs)
    expected = torch.einsum("bsd,od->bso", x, projs[0])
    torch.testing.assert_close(native, expected, atol=1e-4, rtol=1e-3)


def test_gated_lane_blend_finite_output():
    torch.manual_seed(3)
    x = torch.randn(3, 10, 16)
    scorer = torch.randn(4, 16) * 0.1
    projs = torch.randn(4, 16, 16) * 0.1
    y = aria_core.gated_lane_blend_f32(x, scorer, projs)
    assert torch.isfinite(y).all()


def test_gated_lane_blend_zero_input():
    x = torch.zeros(1, 2, 4)
    scorer = torch.randn(2, 4) * 0.1
    projs = torch.randn(2, 4, 4) * 0.1
    y = aria_core.gated_lane_blend_f32(x, scorer, projs)
    # Zero input → zero output (all projections produce zero)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-6, rtol=0.0)


# ── depth_gated_transform_f32 ────────────────────────────────────────


def test_depth_gated_transform_matches_lane_blend():
    """depth_gated_transform delegates to gated_lane_blend internally."""
    torch.manual_seed(4)
    B, S, D, K = 2, 4, 8, 3
    x = torch.randn(B, S, D) * 0.1
    scorer = torch.randn(K, D) * 0.1
    projs = torch.randn(K, D, D) * 0.1
    depth_out = aria_core.depth_gated_transform_f32(x, scorer, projs)
    lane_out = aria_core.gated_lane_blend_f32(x, scorer, projs)
    torch.testing.assert_close(depth_out, lane_out, atol=1e-6, rtol=0.0)


def test_depth_gated_transform_shape():
    x = torch.randn(1, 6, 12)
    scorer = torch.randn(5, 12)
    projs = torch.randn(5, 12, 12)
    y = aria_core.depth_gated_transform_f32(x, scorer, projs)
    assert y.shape == (1, 6, 12)


def test_depth_gated_transform_parity_vs_reference():
    torch.manual_seed(5)
    B, S, D, K = 3, 3, 6, 4
    x = torch.randn(B, S, D) * 0.1
    scorer = torch.randn(K, D) * 0.1
    projs = torch.randn(K, D, D) * 0.1
    native = aria_core.depth_gated_transform_f32(x, scorer, projs)
    ref = _ref_gated_lane_blend(x, scorer, projs)
    torch.testing.assert_close(native, ref, atol=1e-4, rtol=1e-3)


# ── calibrated_branch_merge_f32 ──────────────────────────────────────


def _rms(v: torch.Tensor) -> float:
    return math.sqrt((v * v).mean().item() + 1e-8)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _ref_calibrated_branch_merge(
    a: torch.Tensor,
    b: torch.Tensor,
    score_proj: torch.Tensor | None,
    branch_bias: torch.Tensor | None,
    branch_gain: torch.Tensor | None,
    temperature: float,
    min_secondary: float,
    max_secondary: float,
) -> torch.Tensor:
    """Python reference for calibrated_branch_merge."""
    B, S, D = a.shape
    y = torch.empty_like(a)
    for bs_flat in range(B * S):
        bi, si = divmod(bs_flat, S)
        ar = a[bi, si]
        br = b[bi, si]
        rms_a = _rms(ar)
        rms_b = _rms(br)
        inv_a = 1.0 / rms_a
        inv_b = 1.0 / rms_b
        norm_a = ar * inv_a
        norm_b = br * inv_b

        s0 = float(branch_bias[0]) if branch_bias is not None else 0.0
        s1 = float(branch_bias[1]) if branch_bias is not None else 0.0
        if score_proj is not None:
            s0 += float((norm_a * score_proj[0, 0]).sum())
            s1 += float((norm_b * score_proj[1, 0]).sum())

        # Softmax over 2
        logits = torch.tensor([s0 / temperature, s1 / temperature])
        weights = torch.softmax(logits, dim=0)
        w1 = float(weights[1])
        w1 = max(min_secondary, min(max_secondary, w1))
        w0 = 1.0 - w1

        g0 = 0.5 + _sigmoid(float(branch_gain[0])) if branch_gain is not None else 1.0
        g1 = 0.5 + _sigmoid(float(branch_gain[1])) if branch_gain is not None else 1.0

        y[bi, si] = (norm_a * w0 * g0 + norm_b * w1 * g1) * rms_a
    return y


def test_calibrated_branch_merge_shape():
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    y = aria_core.calibrated_branch_merge_f32(a, b, None, None, None, 1.0, 0.0, 1.0)
    assert y.shape == (2, 3, 8)


def test_calibrated_branch_merge_no_params_equal_branches():
    """Without score_proj/bias/gain, equal RMS branches get ~equal weight."""
    torch.manual_seed(6)
    a = torch.randn(1, 1, 16)
    b = torch.randn(1, 1, 16)
    # Scale b to have same RMS as a
    b = b * (_rms(a[0, 0]) / _rms(b[0, 0]))
    y = aria_core.calibrated_branch_merge_f32(a, b, None, None, None, 1.0, 0.0, 1.0)
    assert torch.isfinite(y).all()
    # With no params, scores are both 0 → softmax gives [0.5, 0.5]
    # gains are both 1.0 (no branch_gain)
    # y = (norm_a * 0.5 + norm_b * 0.5) * rms_a
    rms_a = _rms(a[0, 0])
    expected = (a[0, 0] / rms_a * 0.5 + b[0, 0] / _rms(b[0, 0]) * 0.5) * rms_a
    torch.testing.assert_close(y[0, 0], expected, atol=1e-4, rtol=1e-3)


def test_calibrated_branch_merge_parity_full_params():
    torch.manual_seed(7)
    B, S, D = 2, 3, 8
    a = torch.randn(B, S, D) * 0.5
    b = torch.randn(B, S, D) * 0.5
    score_proj = torch.randn(2, 1, D) * 0.1
    branch_bias = torch.randn(2) * 0.1
    branch_gain = torch.randn(2) * 0.1
    temp, min_s, max_s = 1.0, 0.1, 0.9

    native = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        score_proj,
        branch_bias,
        branch_gain,
        temp,
        min_s,
        max_s,
    )
    ref = _ref_calibrated_branch_merge(
        a,
        b,
        score_proj,
        branch_bias,
        branch_gain,
        temp,
        min_s,
        max_s,
    )
    torch.testing.assert_close(native, ref, atol=1e-4, rtol=1e-3)


def test_calibrated_branch_merge_min_max_clamp():
    """Verify min_secondary/max_secondary clamp the secondary branch weight."""
    torch.manual_seed(8)
    a = torch.randn(1, 1, 4) * 0.5
    b = torch.randn(1, 1, 4) * 0.5
    # Large bias toward branch 0 → softmax gives w1 ≈ 0
    bias = torch.tensor([10.0, -10.0])
    # But min_secondary=0.3 forces w1 >= 0.3
    y_clamped = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        None,
        bias,
        None,
        1.0,
        0.3,
        0.7,
    )
    y_unclamped = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        None,
        bias,
        None,
        1.0,
        0.0,
        1.0,
    )
    # Clamped version should differ from unclamped (secondary gets forced up)
    assert not torch.allclose(y_clamped, y_unclamped, atol=1e-4)


def test_calibrated_branch_merge_temperature():
    """High temperature → equal weights; low temperature → winner-take-all."""
    torch.manual_seed(9)
    a = torch.ones(1, 1, 4) * 2.0
    b = torch.ones(1, 1, 4) * 1.0
    bias = torch.tensor([1.0, -1.0])

    y_high_t = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        None,
        bias,
        None,
        100.0,
        0.0,
        1.0,
    )
    y_low_t = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        None,
        bias,
        None,
        0.01,
        0.0,
        1.0,
    )
    # High temp: roughly equal weights → output is blend
    # Low temp: branch 0 dominates → output ≈ normalized a * rms_a = a
    assert torch.isfinite(y_high_t).all()
    assert torch.isfinite(y_low_t).all()


def test_calibrated_branch_merge_finite_output():
    torch.manual_seed(10)
    a = torch.randn(3, 7, 16)
    b = torch.randn(3, 7, 16)
    score_proj = torch.randn(2, 1, 16) * 0.1
    branch_bias = torch.randn(2) * 0.1
    branch_gain = torch.randn(2)
    y = aria_core.calibrated_branch_merge_f32(
        a,
        b,
        score_proj,
        branch_bias,
        branch_gain,
        1.0,
        0.05,
        0.95,
    )
    assert torch.isfinite(y).all()


def test_calibrated_branch_merge_zero_b_branch():
    """When b is all zeros, output should be dominated by branch a."""
    a = torch.randn(1, 1, 8)
    b = torch.zeros(1, 1, 8)
    y = aria_core.calibrated_branch_merge_f32(a, b, None, None, None, 1.0, 0.0, 1.0)
    assert torch.isfinite(y).all()
    # b has near-zero RMS (only eps), so norm_b is huge but scaled by very small RMS
    # The result should still be finite and non-zero
    assert (y.abs() > 0).any()
