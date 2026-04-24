from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from research.eval.training_core import make_optimizer
from research.eval._eval_native import load_eval_native
from research.eval._runner_native import load_runner_native
from research.eval.utils import clip_grad_norm, language_model_loss


pytestmark = pytest.mark.unit


def test_native_next_token_cross_entropy_matches_pytorch():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))

    got = language_model_loss(logits, targets, 7)
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, 7),
        targets[:, 1:].reshape(-1),
    )
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_native_sgd_step_matches_torch_sgd():
    torch.manual_seed(1)
    base = torch.randn(4, 3)
    grad = torch.randn(4, 3)

    native_param = base.clone().requires_grad_(True)
    native_param.grad = grad.clone()
    native_opt = make_optimizer(
        [native_param],
        optimizer_name="sgd",
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.05,
        prefer_native=True,
    )
    native_opt.step()

    ref_param = base.clone().requires_grad_(True)
    ref_param.grad = grad.clone()
    ref_opt = torch.optim.SGD(
        [ref_param],
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.05,
        nesterov=True,
    )
    ref_opt.step()

    torch.testing.assert_close(native_param, ref_param, atol=1e-6, rtol=1e-6)


def test_native_adamw_step_matches_torch_adamw():
    torch.manual_seed(2)
    base = torch.randn(5, 4)
    grad = torch.randn(5, 4)

    native_param = base.clone().requires_grad_(True)
    native_param.grad = grad.clone()
    native_opt = make_optimizer(
        [native_param],
        optimizer_name="adamw",
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        prefer_native=True,
    )
    native_opt.step()

    ref_param = base.clone().requires_grad_(True)
    ref_param.grad = grad.clone()
    ref_opt = torch.optim.AdamW(
        [ref_param],
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    ref_opt.step()

    torch.testing.assert_close(native_param, ref_param, atol=1e-6, rtol=1e-5)


def test_native_adamw_clip_step_matches_torch_clip_then_adamw():
    torch.manual_seed(22)
    bases = [torch.randn(5, 4), torch.randn(3, 2)]
    grads = [torch.randn_like(base) * 3.0 for base in bases]

    native_params = [base.clone().requires_grad_(True) for base in bases]
    for param, grad in zip(native_params, grads, strict=True):
        param.grad = grad.clone()
    native_opt = make_optimizer(
        native_params,
        optimizer_name="adamw",
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        prefer_native=True,
    )
    native_norm = native_opt.step_with_grad_clip(0.75)

    ref_params = [base.clone().requires_grad_(True) for base in bases]
    for param, grad in zip(ref_params, grads, strict=True):
        param.grad = grad.clone()
    ref_opt = torch.optim.AdamW(
        ref_params,
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    ref_norm = torch.nn.utils.clip_grad_norm_(ref_params, 0.75)
    ref_opt.step()

    assert native_norm == pytest.approx(float(ref_norm.item()), rel=1e-6, abs=1e-6)
    for native_param, ref_param in zip(native_params, ref_params, strict=True):
        torch.testing.assert_close(native_param, ref_param, atol=1e-6, rtol=1e-5)


def test_native_adamw_backward_clip_step_matches_torch_backward_clip_step():
    torch.manual_seed(23)
    bases = [torch.randn(5, 4), torch.randn(3, 2)]
    factors = [torch.randn_like(base) for base in bases]

    native_params = [base.clone().requires_grad_(True) for base in bases]
    native_loss = sum(
        (param * factor).sum()
        for param, factor in zip(native_params, factors, strict=True)
    )
    native_opt = make_optimizer(
        native_params,
        optimizer_name="adamw",
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        prefer_native=True,
    )
    native_norm = native_opt.backward_step_with_grad_clip(native_loss, 0.75)

    ref_params = [base.clone().requires_grad_(True) for base in bases]
    ref_loss = sum(
        (param * factor).sum()
        for param, factor in zip(ref_params, factors, strict=True)
    )
    ref_opt = torch.optim.AdamW(
        ref_params,
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    ref_opt.zero_grad(set_to_none=True)
    ref_loss.backward()
    ref_norm = torch.nn.utils.clip_grad_norm_(ref_params, 0.75)
    ref_opt.step()

    assert native_norm == pytest.approx(float(ref_norm.item()), rel=1e-6, abs=1e-6)
    for native_param, ref_param in zip(native_params, ref_params, strict=True):
        torch.testing.assert_close(native_param, ref_param, atol=1e-6, rtol=1e-5)


def test_native_sgd_backward_clip_step_matches_torch_backward_clip_step():
    torch.manual_seed(24)
    bases = [torch.randn(4, 3), torch.randn(2, 5)]
    factors = [torch.randn_like(base) for base in bases]

    native_params = [base.clone().requires_grad_(True) for base in bases]
    native_loss = sum(
        (param * factor).sum()
        for param, factor in zip(native_params, factors, strict=True)
    )
    native_opt = make_optimizer(
        native_params,
        optimizer_name="sgd",
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.05,
        prefer_native=True,
    )
    native_norm = native_opt.backward_step_with_grad_clip(native_loss, 0.75)

    ref_params = [base.clone().requires_grad_(True) for base in bases]
    ref_loss = sum(
        (param * factor).sum()
        for param, factor in zip(ref_params, factors, strict=True)
    )
    ref_opt = torch.optim.SGD(
        ref_params,
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.05,
        nesterov=True,
    )
    ref_opt.zero_grad(set_to_none=True)
    ref_loss.backward()
    ref_norm = torch.nn.utils.clip_grad_norm_(ref_params, 0.75)
    ref_opt.step()

    assert native_norm == pytest.approx(float(ref_norm.item()), rel=1e-6, abs=1e-6)
    for native_param, ref_param in zip(native_params, ref_params, strict=True):
        torch.testing.assert_close(native_param, ref_param, atol=1e-6, rtol=1e-6)


def test_native_clip_grad_norm_matches_torch():
    torch.manual_seed(3)
    base_one = torch.randn(4, 3)
    base_two = torch.randn(2, 5)
    grad_one = torch.randn(4, 3)
    grad_two = torch.randn(2, 5)

    native_one = base_one.clone().requires_grad_(True)
    native_two = base_two.clone().requires_grad_(True)
    native_one.grad = grad_one.clone()
    native_two.grad = grad_two.clone()
    native_norm = clip_grad_norm([native_one, native_two], 0.75)

    ref_one = base_one.clone().requires_grad_(True)
    ref_two = base_two.clone().requires_grad_(True)
    ref_one.grad = grad_one.clone()
    ref_two.grad = grad_two.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_([ref_one, ref_two], 0.75)

    torch.testing.assert_close(native_norm, ref_norm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(native_one.grad, ref_one.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(native_two.grad, ref_two.grad, atol=1e-6, rtol=1e-6)


def test_native_training_summary_matches_python():
    native = load_runner_native().summarize_training_loop(
        512,
        128.0,
        4,
        40.0,
        10.0,
        30.0,
        4.5,
        4,
    )
    assert native["n_train_steps"] == 4
    assert native["throughput"] == pytest.approx(4000.0)
    assert native["avg_step_time_ms"] == pytest.approx(10.0)
    assert native["max_grad_norm"] == pytest.approx(4.5)
    assert native["mean_grad_norm"] == pytest.approx(2.5)
    assert native["grad_norm_std"] == pytest.approx((1.25) ** 0.5)


def test_native_screening_graph_analysis_matches_python_shape():
    native = load_eval_native().screening_graph_analysis_native(
        [0, 1, 2, 3],
        ["input", "linear_proj", "moe_topk", "output"],
        [[], [0], [1], [2]],
        [True, False, False, False],
        [False, False, False, True],
        [False, True, False, False],
    )
    assert set(native["op_names"]) == {"linear_proj", "moe_topk"}
    assert tuple(native["counted_ops"]) == ("linear_proj", "moe_topk", "output")
    assert tuple(native["toxic_bigrams"]) == ("linear_proj->moe_topk",)
    assert native["has_parameterized_op"] is True


def test_native_template_summary_core_exposes_metrics():
    native = load_eval_native().summarize_template_stat_core(
        4,
        4,
        3,
        2,
        [0.5, 0.7],
        [0.6],
        [0.8],
        [0.9],
        [0.4],
        [0.08],
        [0.09],
        [0.07],
        [0.31],
        [0.4],
        1,
        2,
        3,
        2,
        1,
        1,
        [0.2],
        [0.1],
        [0.05],
    )
    assert native["n_used"] == 4
    assert native["s0_rate"] == pytest.approx(1.0)
    assert native["s05_rate"] == pytest.approx(0.75)
    assert native["s1_rate"] == pytest.approx(0.5)
    assert native["avg_loss_ratio"] == pytest.approx(0.6)
    assert native["best_loss_ratio"] == pytest.approx(0.5)
    assert native["screening_metric_coverage"]["hellaswag"] == 2
    assert native["evidence_level"] == "sparse"
