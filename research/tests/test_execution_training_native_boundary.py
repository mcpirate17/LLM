from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from research.eval.utils import language_model_loss
from research.scientist.runner.execution_training_native_boundary import (
    _build_training_step_event,
    _MicroTrainLoopProgress,
    _TrainingLoopState,
    _apply_training_aux_losses,
    _backward_loss,
    _collect_aux_modules,
    _collect_early_exit_loss,
    _collect_routing_aux_loss,
    _compute_micro_train_forward_loss,
    _maybe_extend_training_budget,
    _optimizer_step,
    _training_step_error,
)
from research.scientist.runner._curriculum_schedule import (
    precompute_curriculum_seq_lens,
)
from research.training.curriculum import CurriculumStrategy


class _RoutingModule(nn.Module):
    def __init__(self, counts: torch.Tensor):
        super().__init__()
        self.routing_telemetry = {"expert_counts": counts}


class _EarlyExitModule(nn.Module):
    def __init__(self, hidden: torch.Tensor, gate: torch.Tensor):
        super().__init__()
        self._early_exit_aux = {"hidden": hidden, "gate": gate}


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = _RoutingModule(torch.tensor([3.0, 1.0]))
        self.early = _EarlyExitModule(
            hidden=torch.tensor([[[1.0, 0.0], [0.5, -0.5], [0.2, 0.1]]]),
            gate=torch.tensor([[0.1, 0.6, 0.9]]),
        )
        self.lm_head = nn.Linear(2, 3, bias=False)
        self.norm = nn.Identity()
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.5, -0.5],
                    ]
                )
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.stack(
            (
                input_ids.float(),
                input_ids.float() * 0.5,
            ),
            dim=-1,
        )
        return self.lm_head(hidden)


def test_compute_micro_train_forward_loss_matches_cross_entropy_path():
    model = _DummyModel()
    inputs = torch.tensor([[0, 1, 2]])
    config = type("Cfg", (), {"vocab_size": 3, "loss_type": "cross_entropy"})()

    loss = _compute_micro_train_forward_loss(
        object(),
        model,
        inputs,
        config=config,
        dev=torch.device("cpu"),
        use_synthesized_training=False,
        seed=123,
    )

    expected = language_model_loss(model(inputs), inputs, 3)
    assert float(loss.item()) == pytest.approx(float(expected.item()))


def test_compute_micro_train_forward_loss_does_not_swallow_synth_loss_errors():
    model = _DummyModel()
    inputs = torch.tensor([[0, 1, 2]])
    config = type("Cfg", (), {"vocab_size": 3, "loss_type": "synthesized"})()

    class _BrokenLoss:
        @staticmethod
        def compute(*_args, **_kwargs):
            raise ValueError("bad loss")

    owner = type("Owner", (), {"_synth_loss": _BrokenLoss()})()

    with pytest.raises(ValueError, match="bad loss"):
        _compute_micro_train_forward_loss(
            owner,
            model,
            inputs,
            config=config,
            dev=torch.device("cpu"),
            use_synthesized_training=True,
            seed=123,
        )


def test_collect_aux_modules_finds_routing_and_early_exit_modules():
    model = _DummyModel()

    routing_modules, early_exit_modules, lm_head, norm = _collect_aux_modules(model)

    assert routing_modules == (model.route,)
    assert early_exit_modules == (model.early,)
    assert lm_head is model.lm_head
    assert norm is model.norm


def test_collect_routing_aux_loss_matches_expected_balance_penalty():
    result = _collect_routing_aux_loss((_RoutingModule(torch.tensor([3.0, 1.0])),))

    assert result is not None
    assert float(result.item()) == pytest.approx(0.00125)


def test_collect_early_exit_loss_matches_manual_gate_weighted_loss():
    model = _DummyModel()
    targets = torch.tensor([[0, 2, 1]])

    result = _collect_early_exit_loss(
        (model.early,), model.lm_head, model.norm, targets
    )

    assert result is not None
    assert float(result.item()) > 0.0
    assert model.early._early_exit_aux is None


def test_apply_training_aux_losses_returns_both_components():
    model = _DummyModel()
    targets = torch.tensor([[0, 2, 1]])
    base_loss = torch.tensor(1.0)

    total, routing_aux, early_aux = _apply_training_aux_losses(
        base_loss,
        routing_modules=(model.route,),
        early_exit_modules=(model.early,),
        lm_head=model.lm_head,
        norm=model.norm,
        input_ids=targets,
    )

    assert routing_aux is not None
    assert early_aux is not None
    assert float(total.item()) > 1.0


def test_backward_loss_and_optimizer_step_update_parameters():
    param = torch.tensor([[1.0, -2.0]], requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=0.1)
    loss = (param**2).sum()

    grad_norm = _backward_loss(
        loss,
        optimizer=optimizer,
        grad_clip_norm=10.0,
        model_params=(param,),
    )
    before = param.detach().clone()
    _optimizer_step(optimizer)

    assert grad_norm > 0.0
    assert not torch.equal(before, param.detach())


def test_training_loop_state_native_summary_exposes_python_shape():
    state = _TrainingLoopState(
        initial_loss=10.0,
        final_loss=5.0,
        min_loss=4.5,
        total_tokens=512,
        total_time_ms=128.0,
        step_count=4,
        step_time_sum_ms=40.0,
        grad_norm_sum=10.0,
        grad_norm_sq_sum=30.0,
        grad_norm_max=4.5,
        grad_norm_count=4,
        training_curve=[],
        collect_curve=False,
        seq_len=128,
        seed=123,
        entropy_gate_trajectory=[],
        routing_aux_loss_sum=0.0,
        routing_aux_loss_count=0,
    )

    summary = state.native_summary()

    assert summary["n_train_steps"] == 4
    assert summary["throughput"] == pytest.approx(4000.0)
    assert summary["avg_step_time_ms"] == pytest.approx(10.0)


def test_micro_train_progress_tracks_budget_extension_and_loop_state():
    progress = _MicroTrainLoopProgress()
    result = {}

    total_steps = _maybe_extend_training_budget(
        progress,
        result,
        step=250,
        loss_val=4.0,
        total_steps=512,
    )
    total_steps = _maybe_extend_training_budget(
        progress,
        result,
        step=500,
        loss_val=2.0,
        total_steps=total_steps,
    )
    progress.commit_eager_step(
        step=3,
        loss_val=2.0,
        grad_norm=0.5,
        step_time_ms=7.5,
        token_count=128,
        collect_curve=True,
    )
    loop_state = progress.to_loop_state(
        total_time_ms=9.0,
        collect_curve=True,
        seq_len=32,
        seed=7,
    )

    assert total_steps == 1000
    assert result["adaptive_budget_extension"] is True
    assert loop_state.final_loss == pytest.approx(2.0)
    assert loop_state.total_tokens == 128
    assert loop_state.training_curve[0]["step"] == 3


def test_training_step_error_reports_nonfinite_and_zero_grad():
    nonfinite = _training_step_error(step=2, loss_val=float("inf"), grad_norm=1.0)
    zero_grad = _training_step_error(step=0, loss_val=1.0, grad_norm=0.0)

    assert nonfinite == {"error": "NaN/Inf loss at step 2", "n_train_steps": 2}
    assert zero_grad is not None
    assert zero_grad["error"] == "zero_grad_precheck_failed"
    assert zero_grad["n_train_steps"] == 0


def test_build_training_step_event_includes_optional_metrics():
    event = _build_training_step_event(
        {
            "exp_id": "exp-1",
            "phase": "screening",
            "source_result_id": "rid-1",
            "candidate_index": 2,
            "total_candidates": 5,
            "training_program_index": 3,
            "total_training_programs": 4,
            "training_program_label": "tp-main",
            "training_seed": 17,
            "run_kind": "investigation",
        },
        step=20,
        total_steps=100,
        loss_val=1.2345678,
        grad_norm=0.43219,
        routing_aux_loss=0.001234,
    )

    assert event == {
        "experiment_id": "exp-1",
        "step": 20,
        "loss": 1.234568,
        "total_steps": 100,
        "phase": "screening",
        "run_kind": "investigation",
        "source_result_id": "rid-1",
        "candidate_index": 2,
        "total_candidates": 5,
        "training_program_index": 3,
        "total_training_programs": 4,
        "training_program_label": "tp-main",
        "training_seed": 17,
        "routing_aux_loss": 0.001234,
        "grad_norm": 0.4322,
    }


def test_precompute_curriculum_seq_lens_uses_native_schedule():
    curriculum = CurriculumStrategy(
        name="cur",
        seq_len_schedule="growing",
        initial_seq_len=8,
        max_seq_len=32,
        warmup_steps=4,
    )

    actual = precompute_curriculum_seq_lens(curriculum, 8)
    expected = tuple(curriculum.get_seq_len(step, 8) for step in range(8))

    assert actual == expected


def test_precompute_curriculum_seq_lens_falls_back_for_legacy_curriculum():
    class _LegacyCurriculum:
        def get_seq_len(self, step, total):
            return 8

    assert precompute_curriculum_seq_lens(_LegacyCurriculum(), 4) is None


def test_precompute_curriculum_seq_lens_rejects_broken_native_schedule():
    class _BrokenCurriculum:
        def seq_len_tensor(self, total_steps):
            return torch.tensor([8], dtype=torch.long)

    with pytest.raises(ValueError, match="invalid schedule"):
        precompute_curriculum_seq_lens(_BrokenCurriculum(), 4)
