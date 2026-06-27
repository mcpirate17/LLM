from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from research.training import _loss_components as loss_components
from research.training._optimizer_muon import (
    _orthogonalize_batched,
    _orthogonalize_update,
)
from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_training import (
    _allow_synthesized_training,
    _training_phase,
)
from research.training.optimizer_synthesis import (
    build_optimizer,
    MuonOptimizer,
)
from research.training.curriculum import CurriculumStrategy
from research.training._native import load_training_native
from research.training.sparse_training import RigLScheduler

pytestmark = pytest.mark.unit


class _Owner:
    __slots__ = ("_live_training_context",)

    def __init__(self, phase: str):
        self._live_training_context = {"phase": phase}


def test_training_phase_reads_runner_context():
    assert _training_phase(_Owner("synthesis")) == "synthesis"
    assert _training_phase(object()) == ""


def test_synthesized_training_is_screening_only():
    config = RunConfig(loss_type="synthesized", optimizer_type="synthesized")

    assert _allow_synthesized_training(_Owner("synthesis"), config)
    assert _allow_synthesized_training(_Owner("candidate_screening"), config)
    assert not _allow_synthesized_training(_Owner("investigation"), config)
    assert not _allow_synthesized_training(_Owner("validation"), config)


# ── build_optimizer tests ──────────────────────────────────────────────


def _make_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def test_build_optimizer_adamw_default_betas():
    model = _make_model()
    opt = build_optimizer(model.parameters(), optimizer_type="adamw", lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)


def test_build_optimizer_adamw_custom_betas():
    model = _make_model()
    opt = build_optimizer(
        model.parameters(),
        optimizer_type="adamw",
        lr=1e-3,
        betas=(0.85, 0.98),
    )
    assert opt.param_groups[0]["betas"] == (0.85, 0.98)


def test_build_optimizer_muon():
    model = _make_model()
    opt = build_optimizer(model.parameters(), optimizer_type="muon", lr=1e-3)
    assert isinstance(opt, MuonOptimizer)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_optimizer_sgd():
    model = _make_model()
    opt = build_optimizer(model.parameters(), optimizer_type="sgd", lr=0.01)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["nesterov"] is True


def test_rigl_scheduler_updates_masks():
    params = [torch.nn.Parameter(torch.randn(32, 32)) for _ in range(2)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    scheduler = RigLScheduler(params, opt, dense_allocation=0.2, T_end=10, delta=1)
    for param in params:
        param.grad = torch.randn_like(param)

    scheduler.update_topology()

    for state in scheduler._sparse_params:
        assert state.mask.dtype == torch.bool
        assert state.mask.shape == state.param.shape


def test_rigl_optimizer_state_dict_round_trips_topology():
    """state_dict must carry masks + step_count; resume must restore them."""
    from research.training.sparse_training import RigLOptimizer

    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(16, 16)) for _ in range(2)]
    opt = RigLOptimizer(params, lr=1e-3, T_end=10, delta=1)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    opt.step()

    saved = opt.state_dict()
    assert "rigl" in saved
    saved_masks = [m.clone() for m in saved["rigl"]["masks"]]

    params2 = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt2 = RigLOptimizer(params2, lr=1e-3, T_end=10, delta=1)
    opt2.load_state_dict(saved)

    assert opt2.scheduler.step_count == opt.scheduler.step_count
    for state, mask in zip(opt2.scheduler._sparse_params, saved_masks):
        assert torch.equal(state.mask.cpu(), mask)
    # Aliases must survive the rebind inside load_state_dict.
    assert opt2.state is opt2.base_optimizer.state
    assert opt2.param_groups is opt2.base_optimizer.param_groups

    legacy = opt.state_dict()
    legacy.pop("rigl")
    with pytest.raises(ValueError, match="rigl"):
        opt2.load_state_dict(legacy)


def test_rigl_mask_matches_reference_selection():
    param = torch.tensor(
        [[0.2, -0.9, 0.1], [1.4, -0.3, 0.8]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[0.6, 0.1, 1.2], [0.4, 2.0, 0.5]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    num_to_update = 1

    flat_mask = mask.reshape(-1)
    keep_k = int(flat_mask.sum().item()) - num_to_update
    expected = torch.zeros_like(flat_mask)

    weight_mag = param.abs().reshape(-1).masked_fill(~flat_mask, -1.0)
    keep_indices = torch.topk(
        weight_mag, keep_k, dim=0, largest=True, sorted=True
    ).indices
    expected[keep_indices] = True

    grad_mag = grad.abs().reshape(-1).clone()
    grad_mag.masked_fill_(expected, -1.0)
    grow_indices = torch.topk(
        grad_mag,
        num_to_update,
        dim=0,
        largest=True,
        sorted=True,
    ).indices
    expected[grow_indices] = True

    actual = (
        load_training_native()
        .rigl_compute_new_mask(param, grad, mask, num_to_update)
        .reshape(-1)
    )

    assert torch.equal(actual, expected)


def test_curriculum_native_schedule_matches_scalar_path():
    for schedule in ("fixed", "growing", "oscillating"):
        curriculum = CurriculumStrategy(
            name=f"cur_{schedule}",
            seq_len_schedule=schedule,
            initial_seq_len=16,
            max_seq_len=128,
            warmup_steps=7,
        )
        total_steps = 32
        expected = torch.tensor(
            [curriculum.get_seq_len(step, total_steps) for step in range(3, 19)],
            dtype=torch.long,
        )

        actual = curriculum.seq_len_tensor(total_steps, start=3, stop=19)

        assert torch.equal(actual, expected)


def test_build_optimizer_unknown_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        build_optimizer(model.parameters(), optimizer_type="nonexistent")


def test_entropy_reg_matches_reference_formula():
    logits = torch.randn(4, 7, dtype=torch.float32)
    targets = torch.randint(0, 7, (4,), dtype=torch.int64)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    expected = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1).mean() * 0.1
    actual = loss_components.loss_entropy_reg(logits, targets, log_probs)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_entropy_reg_reuses_log_probs_instead_of_softmax():
    assert "entropy_reg" in loss_components.LOG_PROB_COMPONENTS


def test_log_prob_mean_losses_match_reference_gradients():
    logits = torch.randn(8, 19, dtype=torch.float32, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 19, (8,), dtype=torch.int64)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    actual = loss_components.loss_label_smoothed_ce(logits, targets, log_probs)
    actual = actual + 0.3 * loss_components.loss_kl_uniform(logits, targets, log_probs)
    actual.backward()

    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    nll = -ref_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -ref_log_probs.mean(dim=-1)
    expected = ((1 - 0.1) * nll + 0.1 * smooth_loss).mean()
    expected = (
        expected
        + 0.3 * -(ref_log_probs.mean(dim=-1) + math.log(ref_logits.shape[-1])).mean()
    )
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=1e-6, atol=1e-6)


def test_rank_weighted_ce_matches_reference_formula():
    logits = torch.randn(5, 11, dtype=torch.float32)
    targets = torch.randint(0, 11, (5,), dtype=torch.int64)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_col = targets.unsqueeze(1)
    nll = -log_probs.gather(1, target_col).squeeze(1)
    target_logits = logits.gather(1, target_col)
    rank_pos = logits.gt(target_logits).sum(dim=1).to(logits.dtype)
    expected = (nll * (torch.log1p(rank_pos) + 1.0)).mean()

    actual = loss_components.loss_rank_weighted_ce(logits, targets, log_probs)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_rank_weighted_ce_preserves_autograd_and_matches_reference_gradient():
    logits = torch.randn(6, 13, dtype=torch.float32, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 13, (6,), dtype=torch.int64)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    actual = loss_components.loss_rank_weighted_ce(logits, targets, log_probs)
    actual.backward()

    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    target_col = targets.unsqueeze(1)
    nll = -ref_log_probs.gather(1, target_col).squeeze(1)
    target_logits = ref_logits.gather(1, target_col)
    rank_pos = ref_logits.gt(target_logits).sum(dim=1).to(ref_logits.dtype)
    expected = (nll * (torch.log1p(rank_pos) + 1.0)).mean()
    expected.backward()

    assert actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=1e-5, atol=1e-6)


def test_synthesized_rank_weighted_primary_backpropagates_on_cpu():
    from research.training.loss_synthesis import LossComponent, SynthesizedLoss

    logits = torch.randn(7, 17, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 17, (7,), dtype=torch.int64)
    loss = SynthesizedLoss(
        "rank_primary",
        [LossComponent("rank_weighted_ce", 1.0)],
    ).compute(logits, targets)

    loss.backward()

    assert loss.requires_grad
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_contrastive_push_native_matches_reference_gradient():
    logits = torch.randn(8, 19, dtype=torch.float32, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 19, (8,), dtype=torch.int64)

    actual = loss_components.loss_contrastive_push(logits, targets, None)
    actual.backward()

    target_logits = ref_logits.gather(1, targets.unsqueeze(1))
    topk_width = min(6, ref_logits.shape[-1])
    topk, _ = ref_logits.topk(topk_width, dim=-1)
    expected = torch.nn.functional.relu(topk[:, 1:] - target_logits + 0.5).mean()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=1e-6, atol=1e-6)


def test_gradient_penalty_uses_norm_squared_with_matching_gradient():
    logits = torch.randn(9, 23, dtype=torch.float32, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 23, (9,), dtype=torch.int64)

    actual = loss_components.loss_gradient_penalty(logits, targets, None)
    expected = ref_logits.pow(2).mean() * 0.001
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-8)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=1e-6, atol=1e-9)


# ── MuonOptimizer tests ───────────────────────────────────────────────


def test_muon_step_updates_params():
    model = _make_model()
    opt = MuonOptimizer(model.parameters(), lr=1e-2)
    before = [p.clone() for p in model.parameters()]

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    changed = any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
    assert changed, "MuonOptimizer did not update parameters"


def test_muon_state_size_is_1x_params():
    """Muon stores only momentum (1x params), not momentum+variance (2x)."""
    model = _make_model()
    opt = MuonOptimizer(model.parameters(), lr=1e-2)

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    total_param_elems = sum(p.numel() for p in model.parameters())
    total_state_elems = sum(
        v.numel()
        for p in model.parameters()
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor)
    )
    assert total_state_elems == total_param_elems, (
        f"Expected 1x params in state ({total_param_elems}), got {total_state_elems}"
    )


def test_muon_orthogonalization_matches_reference_iteration():
    matrix = torch.randn(16, 16, dtype=torch.float32)
    n_steps = 4

    a = 3.4445
    b = -4.7750
    c = 2.0315

    ref = matrix / matrix.norm()
    eye = torch.eye(ref.size(1), dtype=ref.dtype, device=ref.device)
    for _ in range(n_steps):
        gram = ref.transpose(0, 1).matmul(ref)
        ref = ref.matmul(a * eye + b * gram + c * gram.matmul(gram))

    actual = _orthogonalize_update(matrix, n_steps)

    assert torch.allclose(actual, ref, rtol=1e-5, atol=1e-5)


def test_muon_batched_orthogonalization_matches_per_matrix():
    """Shape-bucketed (bmm) Newton-Schulz must match the per-matrix path."""
    torch.manual_seed(0)
    for rows, cols in ((16, 16), (8, 24), (24, 8)):
        stack = torch.randn(5, rows, cols, dtype=torch.float32)
        batched = _orthogonalize_batched(stack.clone(), 5)
        for i in range(stack.shape[0]):
            single = _orthogonalize_update(stack[i].clone(), 5)
            assert torch.allclose(batched[i], single, rtol=1e-5, atol=1e-6), (
                f"bucketed NS diverged from per-matrix at [{i}] for "
                f"shape ({rows},{cols})"
            )


def test_muon_step_bucketing_matches_per_matrix_reference():
    """A step over many same-shape layers must equal per-matrix Muon math."""
    torch.manual_seed(1)
    n_layers, dim = 4, 12
    params = [torch.nn.Parameter(torch.randn(dim, dim)) for _ in range(n_layers)]
    grads = [torch.randn(dim, dim) for _ in range(n_layers)]
    lr, wd, momentum, ns_steps = 1e-2, 0.01, 0.95, 5

    expected = []
    for p, g in zip(params, grads):
        buf = g.clone()  # first step: buffer = 0*momentum + grad
        update = g + momentum * buf  # nesterov
        update = _orthogonalize_update(update, ns_steps)
        expected.append(p.detach() * (1.0 - lr * wd) - lr * update)

    opt = MuonOptimizer(
        params, lr=lr, weight_decay=wd, momentum=momentum, ns_steps=ns_steps
    )
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt.step()

    for i, (p, exp) in enumerate(zip(params, expected)):
        assert torch.allclose(p.detach(), exp, rtol=1e-5, atol=1e-6), (
            f"bucketed Muon step diverged from per-matrix reference at layer {i}"
        )


# ── RunConfig optimizer fields ─────────────────────────────────────────


def test_runconfig_optimizer_defaults():
    cfg = RunConfig()
    assert cfg.optimizer_type == "adamw"
    assert cfg.optimizer_betas == (0.9, 0.95)
    assert cfg.optimizer_weight_decay == 0.01
    assert cfg.screening_optimizer == ""
    assert cfg.investigation_optimizer == ""
