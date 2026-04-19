from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from research.training import _loss_components as loss_components
from research.training._optimizer_muon import _orthogonalize_update
from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_training import (
    _allow_synthesized_training,
    _training_phase,
)
from research.training.optimizer_synthesis import (
    build_optimizer,
    MuonOptimizer,
)
from research.training._loss_native import load_loss_native
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

    actual = load_loss_native().rigl_compute_new_mask(
        param, grad, mask, num_to_update
    ).reshape(-1)

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


# ── RunConfig optimizer fields ─────────────────────────────────────────


def test_runconfig_optimizer_defaults():
    cfg = RunConfig()
    assert cfg.optimizer_type == "adamw"
    assert cfg.optimizer_betas == (0.9, 0.95)
    assert cfg.optimizer_weight_decay == 0.01
    assert cfg.screening_optimizer == ""
    assert cfg.investigation_optimizer == ""
