from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from research.training.checkpointing import CheckpointManager


def test_save_phase_normalizes_nested_optimizer_state_to_cpu(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.tensor([0.5])
    optimizer.step()

    manager.save_phase(
        "exp-a",
        "validation",
        1,
        2,
        model_state_dict={"weight": param.detach().clone()},
        optimizer_state_dict=optimizer.state_dict(),
        step=7,
        metrics={"loss": 1.25},
    )

    state = manager.load_phase("exp-a", "validation", 1, 2)
    assert state is not None
    exp_avg = state["optimizer_state_dict"]["state"][0]["exp_avg"]
    assert exp_avg.device.type == "cpu"
    assert state["model_state_dict"]["weight"].device.type == "cpu"
    assert state["schema_version"] == 2


def test_load_phase_rejects_missing_required_fields(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    broken = tmp_path / "exp-b" / "investigation_-1_0.pt"
    broken.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": "investigation"}, broken)

    with pytest.raises(ValueError, match="missing required fields"):
        manager.load_phase("exp-b", "investigation", -1, 0)


def test_restore_phase_state_matches_uninterrupted_adamw_training(tmp_path: Path):
    torch.manual_seed(0)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    def make_model():
        torch.manual_seed(123)
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2),
        )

    def run_steps(model, optimizer, start: int, stop: int):
        losses = []
        for step in range(start, stop):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append((step + 1, float(loss.item())))
        return losses

    manager = CheckpointManager(str(tmp_path))

    baseline_model = make_model()
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3)
    baseline_losses = run_steps(baseline_model, baseline_optimizer, 0, 6)

    staged_model = make_model()
    staged_optimizer = torch.optim.AdamW(staged_model.parameters(), lr=1e-3)
    staged_losses = run_steps(staged_model, staged_optimizer, 0, 3)
    manager.save_phase(
        "exp-resume",
        "validation",
        0,
        0,
        model_state_dict=staged_model.state_dict(),
        optimizer_state_dict=staged_optimizer.state_dict(),
        step=3,
        metrics={"last_loss": staged_losses[-1][1]},
    )

    resumed_model = make_model()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    state = manager.load_phase("exp-resume", "validation", 0, 0)
    assert state is not None
    restored = manager.restore_phase_state(
        state,
        model=resumed_model,
        optimizer=resumed_optimizer,
        device="cpu",
    )
    resumed_losses = run_steps(
        resumed_model,
        resumed_optimizer,
        restored["step"],
        6,
    )

    for baseline_param, resumed_param in zip(
        baseline_model.parameters(),
        resumed_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(baseline_param, resumed_param, atol=1e-7, rtol=1e-6)

    baseline_opt_state = baseline_optimizer.state_dict()["state"]
    resumed_opt_state = resumed_optimizer.state_dict()["state"]
    assert baseline_opt_state.keys() == resumed_opt_state.keys()
    for param_id in baseline_opt_state:
        for key, value in baseline_opt_state[param_id].items():
            resumed_value = resumed_opt_state[param_id][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, resumed_value, atol=1e-7, rtol=1e-6)
            else:
                assert value == resumed_value

    assert restored["step"] == 3
    assert restored["metrics"]["last_loss"] == pytest.approx(staged_losses[-1][1])
    assert baseline_losses[3:] == pytest.approx(resumed_losses)


def test_load_phase_into_restores_live_objects(tmp_path: Path):
    manager = CheckpointManager(str(tmp_path))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)

    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    manager.save_phase(
        "exp-load-into",
        "investigation",
        2,
        1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        step=4,
        metrics={"tag": "resume"},
    )

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(), lr=0.1, momentum=0.9
    )
    restored = manager.load_phase_into(
        "exp-load-into",
        "investigation",
        2,
        1,
        model=restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )

    assert restored is not None
    assert restored["step"] == 4
    assert restored["metrics"]["tag"] == "resume"
    for expected, observed in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, observed, atol=1e-7, rtol=1e-6)


def test_phase_resume_candidate_idx_prefers_progress_metric():
    state = {"candidate_idx": -1, "metrics": {"candidate_idx": 7}}
    assert CheckpointManager.phase_resume_candidate_idx(state) == 7
