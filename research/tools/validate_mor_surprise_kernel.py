"""Validate the surprise-coupling term in the MoR-refine CUDA kernel.

Checks the full autograd path (_NativeMoRRefineScan) against a pure-torch
reference that mirrors the halt logit `lg = MLP(feat) - a*mean|err|`, for both
a=0 (must reproduce the original kernel) and a>0 (the new surprise floor).
Compares forward y/depth and EVERY input grad incl. the new d/da, rel<2e-5.

Run: python research/tools/validate_mor_surprise_kernel.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from component_fab.generator.mor_bilane import _NativeMoRRefineScan


def ref(q, k, v, write, forget, momentum, beta, balance, W1, b1, W2, b2, a, R):
    B, L, M = v.shape
    scale = M**-0.5
    logM = math.log(M)

    def SR(memm, addr):
        scores = memm + addr.unsqueeze(-1)
        return (torch.logsumexp(beta * scores, dim=1) - logM) / beta

    mem = q.new_zeros(B, M, M)
    sur = q.new_zeros(B, M, M)
    outs, depths = [], []
    for t in range(L):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        w_t = write[:, t].view(B, 1, 1)
        read_q = SR(mem, q_t)
        decay = (1.0 - forget[:, t]).unsqueeze(-1)
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
            logit = (h * W2).sum(-1, keepdim=True) + b2 - a * f1
            halt = torch.ones_like(logit) if r == R else torch.sigmoid(logit)
            p_r = rem * halt
            mem_acc = mem_acc + p_r.unsqueeze(-1) * mem_r
            sur_acc = sur_acc + p_r.unsqueeze(-1) * s_r
            depth_acc = depth_acc + p_r * float(r)
            rem = rem * (1.0 - halt)
        mem, sur = mem_acc, sur_acc
        outs.append(read_q)
        depths.append(depth_acc.squeeze(-1))
    return torch.stack(outs, 1), torch.stack(depths, 1)


def _mk(B, L, M, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def mk(*s):
        return torch.randn(*s, device="cuda", generator=g, requires_grad=True)

    q, k, v = mk(B, L, M), mk(B, L, M), mk(B, L, M)
    write = mk(B, L)
    forget = torch.rand(B, L, M, device="cuda", generator=g).requires_grad_(True)
    W1, b1 = mk(H, 3) * 0.3, mk(H) * 0.1
    W2, b2 = (mk(H).reshape(-1)) * 0.3, mk(())
    return q, k, v, write, forget, W1, b1, W2, b2


def run(a_val, seed=0):
    B, L, M, H, R = 2, 6, 8, 16, 4
    mom, beta, bal = 0.6, 1.5, 0.5
    q, k, v, write, forget, W1, b1, W2, b2 = _mk(B, L, M, H, seed)
    a = torch.tensor(float(a_val), device="cuda", requires_grad=True)
    inputs = [q, k, v, write, forget, W1, b1, W2, b2, a]
    # reference
    yr, dr = ref(q, k, v, write, forget, mom, beta, bal, W1, b1, W2, b2, a, R)
    gy = torch.randn_like(yr)
    gd = torch.randn_like(dr)
    gref = torch.autograd.grad(
        (yr * gy).sum() + (dr * gd).sum(), inputs, retain_graph=False
    )
    # kernel
    for x in inputs:
        x.grad = None
    yk, dk, _ = _NativeMoRRefineScan.apply(
        q,
        k,
        v,
        write,
        forget,
        torch.tensor(mom),
        torch.tensor(beta),
        torch.tensor(bal),
        W1,
        b1,
        W2.reshape(-1),
        b2,
        a,
        R,
    )
    gker = torch.autograd.grad((yk * gy).sum() + (dk * gd).sum(), inputs)

    def rel(x, y):
        return (x - y).abs().max().item() / (y.abs().max().item() + 1e-8)

    print(f"--- a = {a_val} ---")
    print(f"  fwd y    rel {rel(yk, yr):.2e}   depth rel {rel(dk, dr):.2e}")
    names = ["q", "k", "v", "write", "forget", "W1", "b1", "W2", "b2", "a"]
    worst = 0.0
    for n, gk_, gr_ in zip(names, gker, gref):
        r = rel(gk_, gr_)
        worst = max(worst, r)
        print(f"  grad {n:7s} rel {r:.2e}")
    ok = worst < 2e-5 and rel(yk, yr) < 2e-5
    print(f"  => {'PASS' if ok else 'FAIL'} (worst grad {worst:.2e})")
    return ok


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs CUDA"
    ok0 = run(0.0)
    ok1 = run(0.7)
    ok2 = run(2.0)
    print("\nALL PASS" if (ok0 and ok1 and ok2) else "\nFAILED")
