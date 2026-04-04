from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from research.eval.training_core import make_optimizer
from research.eval.utils import language_model_loss


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
