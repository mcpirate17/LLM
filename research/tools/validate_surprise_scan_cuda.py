"""Validate the CUDA surprise-memory scan against the CPU C++ reference.

Compares forward outputs AND every input gradient for both ported ops (plain,
adaptive-semiring) on identical inputs. CUDA is only trustworthy for training if
all of these match the CPU reference to tight tolerance.

Run:  python -m research.tools.validate_surprise_scan_cuda
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from component_fab.generator.native_surprise_memory import _native_ext


def _cuda_ext():
    src = Path("component_fab/generator/native_surprise_memory_cuda.cu").resolve()
    return load(
        name="native_surprise_memory_cuda_validate",
        sources=[str(src)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _cmp(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    denom = a.abs().max().item() + 1e-9
    max_abs = (a - b).abs().max().item()
    rel = max_abs / denom
    ok = rel < 2e-3
    print(f"  {name:14s} max_abs={max_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    cu = _cuda_ext()
    cpu = _native_ext()
    B, L, M = 4, 64, 32
    g = torch.Generator().manual_seed(7)
    q = torch.randn(B, L, M, generator=g) * 0.5
    k = torch.randn(B, L, M, generator=g) * 0.5
    v = torch.randn(B, L, M, generator=g) * 0.5
    write = torch.sigmoid(torch.randn(B, L, generator=g))
    forget = torch.sigmoid(torch.randn(B, L, M, generator=g))
    gy = torch.randn(B, L, M, generator=g) * 0.1
    mom = 0.5

    qc = [t.contiguous() for t in (q, k, v, write, forget)]
    qg = [t.cuda().contiguous() for t in (q, k, v, write, forget)]
    gyg = gy.cuda().contiguous()

    all_ok = True

    print("== PLAIN ==")
    y_c, mem_c, sur_c = cpu.forward(*qc, torch.tensor(mom))
    y_g, mem_g, sur_g = cu.plain_forward(*qg, mom)
    all_ok &= _cmp("y", y_c, y_g)
    gq_c, gk_c, gv_c, gw_c, gf_c, gm_c = cpu.backward(
        *qc, torch.tensor(mom), gy, mem_c, sur_c
    )
    gq_g, gk_g, gv_g, gw_g, gf_g, gm_g = cu.plain_backward(*qg, mom, gyg, mem_g, sur_g)
    for nm, ca, cb in [
        ("grad_q", gq_c, gq_g),
        ("grad_k", gk_c, gk_g),
        ("grad_v", gv_c, gv_g),
        ("grad_write", gw_c, gw_g),
        ("grad_forget", gf_c, gf_g),
        ("grad_mom", gm_c.reshape(()), gm_g.reshape(())),
    ]:
        all_ok &= _cmp(nm, ca, cb)

    print("== ADAPTIVE-SEMIRING ==")
    beta, bal, lo, hi, maxs = 2.0, 1.0, 0.01, 0.05, 4
    yc = cpu.adaptive_semiring_forward(
        *qc,
        torch.tensor(mom),
        torch.tensor(beta),
        torch.tensor(bal),
        torch.tensor(lo),
        torch.tensor(hi),
        maxs,
    )
    y_c, mem_c, sur_c, depth_c = yc
    y_g, mem_g, sur_g, depth_g = cu.adaptive_forward(*qg, mom, beta, bal, lo, hi, maxs)
    all_ok &= _cmp("y", y_c, y_g)
    all_ok &= _cmp("depth", depth_c.float(), depth_g.float())
    bc = cpu.adaptive_semiring_backward(
        *qc,
        torch.tensor(mom),
        torch.tensor(beta),
        torch.tensor(bal),
        torch.tensor(lo),
        torch.tensor(hi),
        maxs,
        gy,
        mem_c,
        sur_c,
    )
    gq_c, gk_c, gv_c, gw_c, gf_c, gm_c, gb_c, gbal_c = bc[:8]
    bg = cu.adaptive_backward(*qg, mom, beta, bal, lo, hi, maxs, gyg, mem_g, sur_g)
    gq_g, gk_g, gv_g, gw_g, gf_g, gm_g, gb_g, gbal_g = bg
    for nm, ca, cb in [
        ("grad_q", gq_c, gq_g),
        ("grad_k", gk_c, gk_g),
        ("grad_v", gv_c, gv_g),
        ("grad_write", gw_c, gw_g),
        ("grad_forget", gf_c, gf_g),
        ("grad_mom", gm_c.reshape(()), gm_g.reshape(())),
        ("grad_beta", gb_c.reshape(()), gb_g.reshape(())),
        ("grad_balance", gbal_c.reshape(()), gbal_g.reshape(())),
    ]:
        all_ok &= _cmp(nm, ca, cb)

    print("\n" + ("ALL MATCH" if all_ok else "MISMATCH -- do not use CUDA path"))


if __name__ == "__main__":
    main()
