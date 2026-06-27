"""Regression: Muon must never receive degenerate (vector-shaped) matrices.

Muon's Newton-Schulz orthogonalized update is a fixed-magnitude, gradient-
independent step calibrated for full-rank hidden matrices. On a param with a
dimension of 1 (a ``Linear(dim, 1)`` gate's ``(1, dim)`` weight) it is ill-
conditioned and marches the norm up unboundedly until the forward overflows to
NaN. This killed two long runs (halt_head 2026-06-03; lane_b.write_gate
2026-06-04). ``_classify_muon_params`` must route every ``min(shape) == 1``
matrix — plus embeddings, 1D params, and the MoR halt heads — to AdamW.
"""

from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from research.tools.native_adaptive_hydra_train import (
    _cautious_step,
    _classify_muon_params,
    _gate_aux_loss,
    _gate_aux_target_dist,
    _set_native_gate_floors,
)
from research.tools._scaling_lanes import NativeAdaptiveReciprocalSlotDeltaLane


class _GateModel(nn.Module):
    """Minimal model exercising every classification branch."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)  # -> AdamW (embedding)
        self.hidden = nn.Linear(8, 8, bias=True)  # (8,8) -> Muon; bias 1D -> AdamW
        self.tall = nn.Linear(8, 4, bias=False)  # (4,8) genuine matrix -> Muon
        self.write_gate = nn.Linear(8, 1, bias=False)  # (1,8) vector -> AdamW
        self.col_gate = nn.Linear(1, 8, bias=False)  # (8,1) vector -> AdamW
        self.norm = nn.LayerNorm(8)  # 1D weight+bias -> AdamW


def test_vector_shaped_gates_routed_to_adamw() -> None:
    model = _GateModel()
    muon, adamw = _classify_muon_params(model)
    id2name = {id(p): n for n, p in model.named_parameters()}
    muon_names = {id2name[id(p)] for p in muon}
    adamw_names = {id2name[id(p)] for p in adamw}

    # No degenerate (min-dim == 1) matrix may sit on Muon — the core invariant.
    leaked = [p for p in muon if p.ndim >= 2 and min(p.shape) == 1]
    assert not leaked, (
        f"vector-shaped matrices leaked to Muon: {[id2name[id(p)] for p in leaked]}"
    )

    assert "write_gate.weight" in adamw_names
    assert "col_gate.weight" in adamw_names
    assert "embed.weight" in adamw_names
    # Genuine full-rank hidden matrices stay on Muon.
    assert "hidden.weight" in muon_names
    assert "tall.weight" in muon_names
    # Partition is total and disjoint.
    assert muon_names.isdisjoint(adamw_names)
    n_params = sum(1 for _ in model.parameters())
    assert len(muon) + len(adamw) == n_params


def test_halt_head_carveout_still_holds() -> None:
    """The by-name MoR halt-head carve-out must survive alongside the shape rule
    (it also catches the head's non-vector ``(hidden, in)`` weight)."""
    pytest.importorskip("component_fab.generator.mor_bilane")
    from component_fab.generator.mor_bilane import MoRRefineMLPLaneA

    lane = MoRRefineMLPLaneA(16, memory_dim=8, max_recursive_steps=2)
    muon, adamw = _classify_muon_params(lane)
    id2name = {id(p): n for n, p in lane.named_parameters()}
    for p in muon:
        assert "halt_head" not in id2name[id(p)], "halt_head leaked to Muon"
    assert any("halt_head" in id2name[id(p)] for p in adamw)


class _FixedDeltaOpt:
    """Stub optimizer that adds a fixed proposed update, to test the mask alone."""

    def __init__(self, p: torch.Tensor, delta: torch.Tensor) -> None:
        self.param_groups = [{"params": [p], "lr": 0.0}]
        self._p, self._delta = p, delta

    def step(self) -> None:
        self._p.data.add_(self._delta)


def test_cautious_masks_disagreeing_coords_and_renormalizes() -> None:
    p = torch.zeros(4)
    p.grad = torch.tensor([1.0, -1.0, 1.0, -1.0])
    delta = torch.tensor([-1.0, -1.0, -1.0, -1.0])  # proposed update
    # delta*grad = [-1,+1,-1,+1] -> descent-aligned (negative) at coords 0,2 only.
    _cautious_step([_FixedDeltaOpt(p, delta)], base_lrs=[[0.0]], mult=1.0)
    # kept coords scaled by numel/kept = 4/2 = 2; dropped coords stay at 0.
    assert torch.allclose(p.data, torch.tensor([-2.0, 0.0, -2.0, 0.0]))


def test_cautious_all_agree_is_plain_step() -> None:
    p = torch.zeros(3)
    p.grad = torch.tensor([1.0, 2.0, 3.0])
    delta = torch.tensor([-0.5, -0.5, -0.5])  # all descend -> mask keeps all, scale 1
    _cautious_step([_FixedDeltaOpt(p, delta)], base_lrs=[[0.0]], mult=1.0)
    assert torch.allclose(p.data, delta)


class _TinyGateAuxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(1024, 16)
        self.blocks = nn.ModuleList(
            [NativeAdaptiveReciprocalSlotDeltaLane(16) for _ in range(2)]
        )
        self.head = nn.Linear(16, 1024)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_gate_aux_targets_raw_pre_floor_gates_by_probe_and_block() -> None:
    target = _gate_aux_target_dist(
        "binding", 1, 8, device=torch.device("cpu"), dtype=torch.float32
    )
    assert target[0].item() == pytest.approx(0.05)
    assert torch.isclose(target.sum(), torch.tensor(1.0))

    args = argparse.Namespace(
        gate_aux_every=2,
        gate_aux_weight=0.01,
        gate_aux_start_step=10,
        gate_aux_probes="binding,induction,surprise",
        gate_aux_max_batches=1,
        gate_aux_batch=1,
        device="cpu",
    )
    model = _TinyGateAuxModel()

    inactive_loss, inactive_info = _gate_aux_loss(model, args, step=9)
    assert inactive_loss is None
    assert inactive_info is None

    aux_loss, info = _gate_aux_loss(model, args, step=10)
    assert aux_loss is not None
    assert info is not None
    assert torch.isfinite(aux_loss)
    assert info["probes"] == ("binding", "induction", "surprise")
    assert info["summary"]["by_probe"].keys() == {"binding", "induction", "surprise"}
    assert len(info["rows"]) == 6
    for row in info["rows"]:
        assert row["effective_gate_mean"][0] == pytest.approx(
            0.25 + 0.75 * row["raw_gate_mean"][0], abs=1e-5
        )
        assert row["target_dist"][0] in {0.05, 0.2}
        assert row["gate_entropy"] > 0.0


def test_native_gate_floors_assign_across_blocks() -> None:
    model = _TinyGateAuxModel()
    floors = _set_native_gate_floors(model, (0.05, 0.20))
    assert floors == [0.05, 0.20]

    ids = torch.tensor([[101, 102, 103]], dtype=torch.long)
    model(ids)
    first = model.blocks[0].last_gate_metrics
    second = model.blocks[1].last_gate_metrics
    assert first["effective_gate_mean"][0] >= 0.05
    assert second["effective_gate_mean"][0] >= 0.20
