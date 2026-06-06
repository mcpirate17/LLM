"""Validate the isolated MoR-refine CUDA kernel (forward + backward) against a
standalone torch reference that mirrors MoRRefineMLPLaneA._scan.

Run: python research/tools/validate_mor_refine_kernel.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from component_fab.generator.mor_bilane import MoRRefineMLPLaneA

CU = Path(__file__).resolve().parents[2] / (
    "component_fab/generator/native_mor_refine_cuda.cu"
)


def refine_scan_torch(
    q, k, v, write, forget, momentum, beta, balance, W1, b1, W2, b2, R
):
    """Pure-function mirror of MoRRefineMLPLaneA._scan (read-before-write)."""
    B, L, M = v.shape
    scale = M**-0.5
    logM = math.log(M)

    def SR(memm, addr):  # [B,M,M],[B,M] -> [B,M]
        scores = memm + addr.unsqueeze(-1)
        return (torch.logsumexp(beta * scores, dim=1) - logM) / beta

    mem = q.new_zeros(B, M, M)
    sur = q.new_zeros(B, M, M)
    outs = []
    depths = []
    for t in range(L):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        w_t = write[:, t].view(B, 1, 1)
        read_q = SR(mem, q_t)
        decay = (1.0 - forget[:, t]).unsqueeze(-1)  # [B,M,1]
        mem_r, s_r = mem, sur
        rem = q.new_ones(B, 1)
        mem_acc = q.new_zeros(B, M, M)
        sur_acc = q.new_zeros(B, M, M)
        depth_acc = q.new_zeros(B, 1)
        for r in range(1, R + 1):
            err = v_t - SR(mem_r, k_t)
            delta = torch.einsum("bi,bj->bij", k_t, err) * scale
            raw = momentum * s_r + w_t * delta
            s_r = raw / (1.0 + balance * raw.abs())
            mem_r = decay * mem + s_r
            f0 = s_r.abs().mean(dim=(1, 2)).unsqueeze(-1)
            f1 = err.abs().mean(dim=-1, keepdim=True)
            f2 = q.new_full((B, 1), r / R)
            feat = torch.cat([f0, f1, f2], dim=-1)
            h = F.gelu(feat @ W1.t() + b1)
            logit = (h * W2).sum(-1, keepdim=True) + b2
            halt = torch.ones_like(logit) if r == R else torch.sigmoid(logit)
            p_r = rem * halt
            mem_acc = mem_acc + p_r.unsqueeze(-1) * mem_r
            sur_acc = sur_acc + p_r.unsqueeze(-1) * s_r
            depth_acc = depth_acc + p_r * float(r)
            rem = rem * (1.0 - halt)
        mem, sur = mem_acc, sur_acc
        outs.append(read_q)
        depths.append(depth_acc.squeeze(-1))
    return torch.stack(outs, dim=1), torch.stack(depths, dim=1)


def main() -> None:
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(0)
    import os as _os

    ext = load(
        name="mor_refine_cuda_validate"
        + ("_li" if _os.environ.get("LINEINFO") else ""),
        sources=[str(CU)],
        extra_cuda_cflags=["-O2"]
        + (["-lineinfo"] if _os.environ.get("LINEINFO") else []),
        verbose=False,
    )

    import os

    M = int(os.environ.get("MM", "8"))
    B = int(os.environ.get("BB", "2"))
    L = int(os.environ.get("LL", "5"))
    H, R, dim = 16, int(os.environ.get("RR", "4")), 32
    lane = MoRRefineMLPLaneA(
        dim,
        memory_dim=M,
        gate_bias=0.0,
        semiring_temp_init=1.0,
        recursive_balance_init=1.0,
        low_threshold=0,
        high_threshold=2,
        max_recursive_steps=R,
        router_hidden=H,
    ).to(dev)
    x = torch.randn(B, L, dim, device=dev)
    q0, k0, v0, w0, f0, mom0, beta0, bal0 = lane._scan_params(x)
    W1_0 = lane.halt_head[0].weight.detach()
    b1_0 = lane.halt_head[0].bias.detach()
    W2_0 = lane.halt_head[2].weight.detach().reshape(H)
    b2_0 = lane.halt_head[2].bias.detach()

    def leaf(t):
        return t.detach().clone().float().requires_grad_(True)

    # ---- forward parity: standalone torch vs lane._scan ----
    with torch.no_grad():
        y_lane = lane._scan(x)
        y_std, _ = refine_scan_torch(
            q0,
            k0,
            v0,
            w0,
            f0,
            float(mom0),
            float(beta0),
            float(bal0),
            W1_0,
            b1_0,
            W2_0,
            float(b2_0),
            R,
        )
    print(f"standalone vs lane fwd max|diff| = {(y_std - y_lane).abs().max():.2e}")

    # ---- torch reference grads ----
    inputs = {
        n: leaf(t)
        for n, t in [
            ("q", q0),
            ("k", k0),
            ("v", v0),
            ("w", w0),
            ("f", f0),
            ("W1", W1_0),
            ("b1", b1_0),
            ("W2", W2_0),
        ]
    }
    mom = torch.tensor(float(mom0), device=dev, requires_grad=True)
    beta = torch.tensor(float(beta0), device=dev, requires_grad=True)
    bal = torch.tensor(float(bal0), device=dev, requires_grad=True)
    b2 = torch.tensor(float(b2_0), device=dev, requires_grad=True)
    coef = torch.randn(B, L, M, device=dev)
    dcoef = torch.randn(B, L, device=dev)  # exercises the ponder/depth grad path

    y_t, depth_t = refine_scan_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["w"],
        inputs["f"],
        mom,
        beta,
        bal,
        inputs["W1"],
        inputs["b1"],
        inputs["W2"],
        b2,
        R,
    )
    ((y_t * coef).sum() + (depth_t * dcoef).sum()).backward()
    ref = {n: inputs[n].grad.clone() for n in inputs}
    ref |= {
        "mom": mom.grad.clone(),
        "beta": beta.grad.clone(),
        "bal": bal.grad.clone(),
        "b2": b2.grad.clone(),
    }

    # ---- kernel grads via autograd.Function ----
    class K(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, w, f, mom, beta, bal, W1, b1, W2, b2):
            y, depth, _hist, memp, surp = ext.mor_refine_forward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                w.contiguous(),
                f.contiguous(),
                float(mom),
                float(beta),
                float(bal),
                int(R),
                W1.contiguous(),
                b1.contiguous(),
                W2.contiguous(),
                float(b2),
            )
            ctx.save_for_backward(q, k, v, w, f, W1, b1, W2, memp, surp)
            ctx.s = (float(mom), float(beta), float(bal), float(b2))
            return y, depth

        @staticmethod
        def backward(ctx, gy, gdepth):
            q, k, v, w, f, W1, b1, W2, memp, surp = ctx.saved_tensors
            mom, beta, bal, b2 = ctx.s
            if gdepth is None:
                gdepth = torch.zeros(q.shape[0], q.shape[1], device=q.device)
            o = ext.mor_refine_backward(
                q,
                k,
                v,
                w,
                f,
                mom,
                beta,
                bal,
                int(R),
                W1,
                b1,
                W2,
                b2,
                gy.contiguous(),
                gdepth.contiguous(),
                memp,
                surp,
            )
            gq, gk, gv, gw, gf, gmom, gbeta, gbal, gW1, gb1, gW2, gb2 = o
            return (gq, gk, gv, gw, gf, gmom, gbeta, gbal, gW1, gb1, gW2, gb2)

    ki = {n: leaf(inputs[n].detach()) for n in inputs}
    kmom = torch.tensor(float(mom0), device=dev, requires_grad=True)
    kbeta = torch.tensor(float(beta0), device=dev, requires_grad=True)
    kbal = torch.tensor(float(bal0), device=dev, requires_grad=True)
    kb2 = torch.tensor(float(b2_0), device=dev, requires_grad=True)
    y_k, depth_k = K.apply(
        ki["q"],
        ki["k"],
        ki["v"],
        ki["w"],
        ki["f"],
        kmom,
        kbeta,
        kbal,
        ki["W1"],
        ki["b1"],
        ki["W2"],
        kb2,
    )
    ((y_k * coef).sum() + (depth_k * dcoef).sum()).backward()
    got = {n: ki[n].grad.clone() for n in ki}
    got |= {
        "mom": kmom.grad.clone(),
        "beta": kbeta.grad.clone(),
        "bal": kbal.grad.clone(),
        "b2": kb2.grad.clone(),
    }

    print("fwd y max rel =", f"{((y_k - y_t).abs() / (y_t.abs() + 1e-6)).max():.2e}")
    worst = 0.0
    for n in ref:
        a, bb = got[n], ref[n]
        rel = ((a - bb).abs() / (bb.abs() + 1e-4)).max().item()
        worst = max(worst, rel)
        flag = "" if rel < 2e-3 else "  <-- FAIL"
        print(f"  grad {n:4s} max rel = {rel:.2e}{flag}")
    print("BACKWARD", "PASS" if worst < 2e-3 else "FAIL", f"(worst {worst:.2e})")
    # per-timestep breakdown of grad v (shape [B,L,M]) to localize the error
    gv_t, gv_r = got["v"], ref["v"]
    per_t = ((gv_t - gv_r).abs() / (gv_r.abs() + 1e-4)).amax(dim=(0, 2))
    print("  grad v per-t max rel:", [f"{x:.1e}" for x in per_t.tolist()])
    # is k off by a structured factor? show a few worst v elements
    flat = ((gv_t - gv_r).abs() / (gv_r.abs() + 1e-4)).flatten()
    idx = flat.argmax().item()
    print(
        f"  worst v: kernel={gv_t.flatten()[idx]:.5f} torch={gv_r.flatten()[idx]:.5f}"
    )


if __name__ == "__main__":
    main()
