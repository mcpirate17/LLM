import copy

import pytest
import torch
import torch.nn as nn

import research.synthesis.compiler as compiler


def _reference_parallel_associative_scan(
    log_a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    S = log_a.shape[-1]
    if S <= 1:
        return b
    a = torch.exp(log_a)
    h = b
    stride = 1
    while stride < S:
        h = torch.cat(
            [
                h[..., :stride],
                a[..., stride:] * h[..., :-stride] + h[..., stride:],
            ],
            dim=-1,
        )
        a = torch.cat(
            [
                a[..., :stride],
                a[..., stride:] * a[..., :-stride],
            ],
            dim=-1,
        )
        a = a.clamp(max=1.0)
        stride *= 2
    return h


def _reference_gated_delta(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, S, D = x.shape
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)
    alpha = torch.sigmoid(module.alpha_proj(x))
    beta = torch.sigmoid(module.beta_proj(x))
    eff_decay = alpha - beta

    H = getattr(module, "_gated_delta_heads", min(8, D))
    if D % H != 0:
        H = 1
    d = D // H
    BH = B * H
    CHUNK = min(32, S)

    q_f = q.reshape(B, S, H, d).permute(0, 2, 1, 3).reshape(BH, S, d)
    k_f = k.reshape(B, S, H, d).permute(0, 2, 1, 3).reshape(BH, S, d)
    v_f = v.reshape(B, S, H, d).permute(0, 2, 1, 3).reshape(BH, S, d)
    decay_f = eff_decay.reshape(B, S, H, d).permute(0, 2, 1, 3).reshape(BH, S, d)
    beta_f = beta.reshape(B, S, H, d).permute(0, 2, 1, 3).reshape(BH, S, d)

    h = torch.zeros(BH, d, d, device=x.device, dtype=x.dtype)
    outputs = []

    for c_start in range(0, S, CHUNK):
        c_end = min(c_start + CHUNK, S)
        c_len = c_end - c_start
        q_c = q_f[:, c_start:c_end]
        k_c = k_f[:, c_start:c_end]
        v_c = v_f[:, c_start:c_end]
        decay_c = decay_f[:, c_start:c_end]
        beta_c = beta_f[:, c_start:c_end]

        bvk_c = beta_c.unsqueeze(-1) * (v_c.unsqueeze(-1) * k_c.unsqueeze(-2))
        bvk_mod = bvk_c.clone()
        bvk_mod[:, 0] = bvk_mod[:, 0] + decay_c[:, 0, :].unsqueeze(-1) * h

        a_flat = decay_c.permute(0, 2, 1).reshape(BH * d, c_len)
        b_flat = bvk_mod.permute(0, 2, 3, 1).reshape(BH * d, d, c_len)

        scan_a = torch.exp(torch.log(a_flat.clamp(min=1e-8)))
        scan_h = b_flat
        stride = 1
        while stride < c_len:
            scan_h = torch.cat(
                [
                    scan_h[..., :stride],
                    scan_a[:, None, stride:] * scan_h[..., :-stride]
                    + scan_h[..., stride:],
                ],
                dim=-1,
            )
            scan_a = torch.cat(
                [
                    scan_a[..., :stride],
                    scan_a[..., stride:] * scan_a[..., :-stride],
                ],
                dim=-1,
            )
            scan_a = scan_a.clamp(max=1.0)
            stride *= 2

        h_all = scan_h.reshape(BH, d, d, c_len).permute(0, 3, 1, 2)
        h = h_all[:, -1]
        outputs.append(torch.matmul(q_c.unsqueeze(2), h_all).squeeze(2))

    out = (
        torch.cat(outputs, dim=1)
        .reshape(B, H, S, d)
        .permute(0, 2, 1, 3)
        .reshape(B, S, D)
    )
    return module.o_proj(out)


class _GatedDeltaModule(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self._gated_delta_heads = heads


class TestTensorAllocationCleanup:
    def test_parallel_associative_scan_matches_reference_and_avoids_cat(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        torch.manual_seed(0)
        log_a = torch.randn(2, 3, 17).clamp(-8.0, -0.01).requires_grad_()
        b = torch.randn(2, 3, 17, requires_grad=True)

        ref = _reference_parallel_associative_scan(log_a, b)
        ref_log_a_grad, ref_b_grad = torch.autograd.grad(ref.sum(), (log_a, b))

        cat_calls = 0
        orig_cat = compiler.torch.cat

        def counted_cat(*args, **kwargs):
            nonlocal cat_calls
            cat_calls += 1
            return orig_cat(*args, **kwargs)

        monkeypatch.setattr(compiler.torch, "cat", counted_cat)
        out = compiler._parallel_associative_scan(log_a, b)
        out_log_a_grad, out_b_grad = torch.autograd.grad(out.sum(), (log_a, b))

        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(out_log_a_grad, ref_log_a_grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(out_b_grad, ref_b_grad, atol=1e-6, rtol=1e-6)
        assert cat_calls == 0

    def test_gated_delta_matches_reference_and_avoids_cat(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        torch.manual_seed(1)
        module = _GatedDeltaModule(d_model=16, heads=4)
        ref_module = copy.deepcopy(module)
        x = torch.randn(2, 37, 16)

        ref = _reference_gated_delta(ref_module, x)

        cat_calls = 0
        orig_cat = compiler.torch.cat

        def counted_cat(*args, **kwargs):
            nonlocal cat_calls
            cat_calls += 1
            return orig_cat(*args, **kwargs)

        monkeypatch.setattr(compiler.torch, "cat", counted_cat)
        out = compiler._op_gated_delta(module, [x], {})

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
        assert cat_calls == 0
