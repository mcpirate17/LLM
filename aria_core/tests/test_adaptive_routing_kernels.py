import torch

import aria_core


def _ref_difficulty_scorer(x, w1, b1, w2, b2):
    h = torch.relu(torch.matmul(x, w1.T) + b1.view(1, 1, -1))
    w2v = w2.view(-1)
    z = torch.matmul(h, w2v.view(-1, 1)).squeeze(-1) + b2.view(-1)[0]
    return torch.sigmoid(z).unsqueeze(-1)


def _ref_lane_router_threshold(scores, lanes, thresholds):
    boundaries = (
        thresholds
        if thresholds is not None
        else torch.linspace(0, 1, lanes + 1, dtype=scores.dtype)[1:-1]
    )
    assignments = torch.bucketize(scores, boundaries)
    assignments = torch.clamp(assignments, 0, lanes - 1).to(torch.int64)
    weights = torch.zeros(scores.shape[0], scores.shape[1], lanes, dtype=scores.dtype)
    weights.scatter_(2, assignments.unsqueeze(-1), 1.0)
    return assignments, weights


def _ref_conditional_dispatch(x, assignments, lane_id):
    b, s, d = x.shape
    lane_out = torch.zeros_like(x)
    index_map = torch.full((b, s), -1, dtype=torch.int64)
    lane_counts = torch.zeros((b,), dtype=torch.int64)
    for bi in range(b):
        pos = 0
        for si in range(s):
            if int(assignments[bi, si]) == lane_id:
                index_map[bi, si] = pos
                lane_out[bi, pos] = x[bi, si]
                pos += 1
        lane_counts[bi] = pos
    return lane_out, index_map, lane_counts


def _ref_conditional_gather(lane_out, index_map, weights):
    b, s, d = lane_out.shape
    y = torch.zeros_like(lane_out)
    for bi in range(b):
        for si in range(s):
            packed = int(index_map[bi, si])
            if packed >= 0:
                y[bi, si] = weights[bi, si] * lane_out[bi, packed]
    return y


def _ref_conditional_gather_backward(grad_y, lane_out, index_map, weights):
    b, s, d = lane_out.shape
    grad_lane = torch.zeros_like(lane_out)
    grad_weights = torch.zeros((b, s), dtype=lane_out.dtype)
    for bi in range(b):
        for si in range(s):
            packed = int(index_map[bi, si])
            if packed >= 0:
                grad_lane[bi, packed] += weights[bi, si] * grad_y[bi, si]
                grad_weights[bi, si] = torch.dot(grad_y[bi, si], lane_out[bi, packed])
    return grad_lane, grad_weights


def test_difficulty_scorer_shape_and_range():
    b, s, d, h = 2, 7, 16, 4
    x = torch.randn(b, s, d)
    w1 = torch.randn(h, d) * 0.1
    b1 = torch.zeros(h)
    w2 = torch.randn(1, h) * 0.1
    b2 = torch.zeros(1)

    scores = aria_core.difficulty_scorer_f32(x, w1, b1, w2, b2)
    assert scores.shape == (b, s, 1)
    assert torch.isfinite(scores).all()
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_lane_router_threshold_assignments_and_weights():
    scores = torch.tensor([[0.1, 0.3, 0.8, 0.51]], dtype=torch.float32)
    thresholds = torch.tensor([0.25, 0.5], dtype=torch.float32)

    assignments, weights = aria_core.lane_router_threshold_f32(scores, 3, thresholds)
    assert assignments.shape == (1, 4)
    assert weights.shape == (1, 4, 3)
    assert assignments.tolist() == [[0, 1, 2, 2]]
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(1, 4), atol=1e-6, rtol=0.0
    )


def test_load_balance_loss_matches_expected_l2():
    assignments = torch.tensor([[0, 0, 1, 2]], dtype=torch.int64)
    target = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    loss_weight = 0.01

    loss, fracs = aria_core.load_balance_loss_f32(assignments, 3, loss_weight, target)
    expected_fracs = torch.tensor([0.50, 0.25, 0.25], dtype=torch.float32)
    expected_loss = loss_weight * torch.sum((expected_fracs - target) ** 2)

    torch.testing.assert_close(fracs, expected_fracs, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(loss.squeeze(0), expected_loss, atol=1e-6, rtol=0.0)


def test_difficulty_scorer_parity_vs_reference():
    torch.manual_seed(11)
    b, s, d, h = 3, 4, 10, 5
    x = torch.randn(b, s, d)
    w1 = torch.randn(h, d) * 0.1
    b1 = torch.randn(h) * 0.01
    w2 = torch.randn(1, h) * 0.1
    b2 = torch.randn(1) * 0.01

    native = aria_core.difficulty_scorer_f32(x, w1, b1, w2, b2)
    ref = _ref_difficulty_scorer(x, w1, b1, w2, b2)
    torch.testing.assert_close(native, ref, atol=1e-6, rtol=0.0)


def test_lane_router_threshold_parity_vs_reference():
    torch.manual_seed(12)
    scores = torch.rand(2, 9)
    lanes = 4
    thresholds = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)

    n_assign, n_weights = aria_core.lane_router_threshold_f32(scores, lanes, thresholds)
    r_assign, r_weights = _ref_lane_router_threshold(scores, lanes, thresholds)
    torch.testing.assert_close(n_assign, r_assign, atol=0.0, rtol=0.0)
    torch.testing.assert_close(n_weights, r_weights, atol=1e-6, rtol=0.0)


def test_conditional_dispatch_counts_and_index_map():
    x = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    assignments = torch.tensor([[0, 2, 1, 2, 0]], dtype=torch.int64)

    lane_out, index_map, lane_counts = aria_core.conditional_dispatch_f32(
        x, assignments, 2
    )

    assert lane_counts.tolist() == [2]
    assert index_map.tolist() == [[-1, 0, -1, 1, -1]]
    torch.testing.assert_close(lane_out[0, 0], x[0, 1], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(lane_out[0, 1], x[0, 3], atol=1e-6, rtol=0.0)


def test_conditional_gather_reconstructs_with_one_hot_weights():
    torch.manual_seed(0)
    b, s, d, lanes = 2, 6, 4, 3
    x = torch.randn(b, s, d)
    scores = torch.rand(b, s)
    thresholds = torch.tensor([0.33, 0.66], dtype=torch.float32)
    assignments, one_hot = aria_core.lane_router_threshold_f32(
        scores, lanes, thresholds
    )

    y = torch.zeros_like(x)
    for lane_id in range(lanes):
        lane_out, index_map, _ = aria_core.conditional_dispatch_f32(
            x, assignments, lane_id
        )
        lane_w = one_hot[:, :, lane_id].contiguous()
        y = y + aria_core.conditional_gather_f32(lane_out, index_map, lane_w)

    torch.testing.assert_close(y, x, atol=1e-6, rtol=0.0)


def test_conditional_dispatch_parity_vs_reference():
    torch.manual_seed(13)
    x = torch.randn(2, 7, 3)
    assignments = torch.randint(0, 3, (2, 7), dtype=torch.int64)
    lane_id = 1
    n_out, n_map, n_counts = aria_core.conditional_dispatch_f32(x, assignments, lane_id)
    r_out, r_map, r_counts = _ref_conditional_dispatch(x, assignments, lane_id)
    torch.testing.assert_close(n_out, r_out, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(n_map, r_map, atol=0.0, rtol=0.0)
    torch.testing.assert_close(n_counts, r_counts, atol=0.0, rtol=0.0)


def test_conditional_gather_parity_vs_reference():
    torch.manual_seed(14)
    x = torch.randn(2, 6, 4)
    assignments = torch.randint(0, 3, (2, 6), dtype=torch.int64)
    lane_out, index_map, _ = aria_core.conditional_dispatch_f32(x, assignments, 2)
    weights = torch.rand(2, 6)

    native = aria_core.conditional_gather_f32(lane_out, index_map, weights)
    ref = _ref_conditional_gather(lane_out, index_map, weights)
    torch.testing.assert_close(native, ref, atol=1e-6, rtol=0.0)


def test_conditional_backward_shapes_and_basic_values():
    x = torch.randn(1, 4, 3)
    assignments = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    lane_out, index_map, _ = aria_core.conditional_dispatch_f32(x, assignments, 1)

    grad_y = torch.ones_like(x)
    weights = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    grad_lane, grad_weights = aria_core.conditional_gather_backward_f32(
        grad_y, lane_out, index_map, weights
    )
    grad_x = aria_core.conditional_dispatch_backward_f32(grad_lane, index_map)

    assert grad_lane.shape == lane_out.shape
    assert grad_weights.shape == weights.shape
    assert grad_x.shape == x.shape
    # Tokens in lane 1 get unit gradient; others remain zero.
    torch.testing.assert_close(grad_x[0, 1], torch.ones(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_x[0, 3], torch.ones(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_x[0, 0], torch.zeros(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_x[0, 2], torch.zeros(3), atol=1e-6, rtol=0.0)


def test_conditional_gather_backward_parity_vs_reference():
    torch.manual_seed(15)
    x = torch.randn(2, 5, 3)
    assignments = torch.randint(0, 3, (2, 5), dtype=torch.int64)
    lane_out, index_map, _ = aria_core.conditional_dispatch_f32(x, assignments, 1)
    grad_y = torch.randn(2, 5, 3)
    weights = torch.rand(2, 5)

    n_grad_lane, n_grad_w = aria_core.conditional_gather_backward_f32(
        grad_y, lane_out, index_map, weights
    )
    r_grad_lane, r_grad_w = _ref_conditional_gather_backward(
        grad_y, lane_out, index_map, weights
    )
    torch.testing.assert_close(n_grad_lane, r_grad_lane, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(n_grad_w, r_grad_w, atol=1e-6, rtol=0.0)


def test_fused_adaptive_route_dispatch_matches_composed_path():
    torch.manual_seed(7)
    b, s, d, h, lanes = 2, 5, 8, 3, 3
    x = torch.randn(b, s, d)
    w1 = torch.randn(h, d) * 0.1
    b1 = torch.randn(h) * 0.01
    w2 = torch.randn(1, h) * 0.1
    b2 = torch.randn(1) * 0.01
    thresholds = torch.tensor([0.25, 0.5], dtype=torch.float32)

    f_scores, f_assign, f_weights, f_lane_out, f_index_map, f_lane_counts = (
        aria_core.adaptive_route_dispatch_f32(x, w1, b1, w2, b2, lanes, thresholds)
    )

    c_scores = aria_core.difficulty_scorer_f32(x, w1, b1, w2, b2)
    c_assign, c_weights = aria_core.lane_router_threshold_f32(
        c_scores.squeeze(-1), lanes, thresholds
    )

    lane_outs = []
    maps = []
    counts = []
    for lane in range(lanes):
        lo, im, lc = aria_core.conditional_dispatch_f32(x, c_assign, lane)
        lane_outs.append(lo)
        maps.append(im)
        counts.append(lc)
    c_lane_out = torch.stack(lane_outs, dim=0)
    c_index_map = torch.stack(maps, dim=0)
    c_lane_counts = torch.stack(counts, dim=0)

    torch.testing.assert_close(f_scores, c_scores, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(f_assign, c_assign, atol=0.0, rtol=0.0)
    torch.testing.assert_close(f_weights, c_weights, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(f_lane_out, c_lane_out, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(f_index_map, c_index_map, atol=0.0, rtol=0.0)
    torch.testing.assert_close(f_lane_counts, c_lane_counts, atol=0.0, rtol=0.0)
