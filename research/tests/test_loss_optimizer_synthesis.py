from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_training import (
    _allow_synthesized_training,
    _training_phase,
)
from research.training.optimizer_synthesis import (
    build_optimizer,
    MuonOptimizer,
)

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
        model.parameters(), optimizer_type="adamw",
        lr=1e-3, betas=(0.85, 0.98),
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


def test_build_optimizer_unknown_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        build_optimizer(model.parameters(), optimizer_type="nonexistent")


# ── MuonOptimizer tests ───────────────────────────────────────────────


def test_muon_step_updates_params():
    model = _make_model()
    opt = MuonOptimizer(model.parameters(), lr=1e-2)
    before = [p.clone() for p in model.parameters()]

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    changed = any(
        not torch.equal(b, a) for b, a in zip(before, model.parameters())
    )
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
        v.numel() for p in model.parameters()
        for v in opt.state[p].values() if isinstance(v, torch.Tensor)
    )
    assert total_state_elems == total_param_elems, (
        f"Expected 1x params in state ({total_param_elems}), "
        f"got {total_state_elems}"
    )


# ── RunConfig optimizer fields ─────────────────────────────────────────


def test_runconfig_optimizer_defaults():
    cfg = RunConfig()
    assert cfg.optimizer_type == "adamw"
    assert cfg.optimizer_betas == (0.9, 0.95)
    assert cfg.optimizer_weight_decay == 0.01
    assert cfg.screening_optimizer == ""
    assert cfg.investigation_optimizer == ""
